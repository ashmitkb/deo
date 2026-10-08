"""
Text-to-speech for Deo.

Two backends behind one interface:
  * Groq/Orpheus (cloud) when it's configured and working -- one request per
    sentence, with sentence N+1's network round trip overlapping sentence N's
    playback.
  * Piper (local CLI binary) otherwise -- ONE process per reply, fed a sentence
    per line, with the raw PCM it emits piped straight to the sound device as
    it is produced, rather than synthesised to a temp .wav and played back
    afterwards.

Both shapes exist for the same reason: time-to-first-word. speak() no longer
waits for the whole reply to be synthesised before any of it is audible, and
speak_stream() doesn't even wait for the whole reply to be *written* -- main.py
feeds it sentences as the LLM streams them out, so Deo starts talking while the
model is still generating the rest of its answer.

Interruption: tts.stop() (wired to config.INTERRUPT_KEY in main.py) cuts
playback off wherever it is, kills the synthesiser, and drops what was queued.
"""

import io
import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sounddevice as sd
import soundfile as sf

from . import config
from . import providers


# ---------------------------------------------------------------------------
# Text cleanup (spoken output should never contain markup, LaTeX, or emoji)
# ---------------------------------------------------------------------------

# Compiled once at import instead of per utterance -- this runs on every turn.
_MATH_DELIMS = re.compile(r"\\\[|\\\]|\\\(|\\\)")
_DOLLARS = re.compile(r"\$\$?")
_FRAC = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
_SQRT = re.compile(r"\\sqrt\{([^{}]*)\}")
_TEXTCMD = re.compile(r"\\text\{([^{}]*)\}")
_POW_BRACED = re.compile(r"\^\{([^{}]*)\}")
_POW_BARE = re.compile(r"\^(\w+)")
_ANY_BACKSLASH_CMD = re.compile(r"\\[a-zA-Z]+")
_WHITESPACE = re.compile(r"\s+")

_LATEX_WORDS = [
    (re.compile(r"\\times"), " times "),
    (re.compile(r"\\div"), " divided by "),
    (re.compile(r"\\pm"), " plus or minus "),
    (re.compile(r"\\cdot"), " times "),
    (re.compile(r"\\leq"), " less than or equal to "),
    (re.compile(r"\\geq"), " greater than or equal to "),
    (re.compile(r"\\neq"), " not equal to "),
    (re.compile(r"\\approx"), " approximately "),
    (re.compile(r"\\rightarrow"), " leads to "),
    (re.compile(r"\\infty"), " infinity "),
]

_UNICODE_WORDS = {
    "\u2192": " leads to ",
    "\u00d7": " times ",
    "\u00f7": " divided by ",
    "\u00b1": " plus or minus ",
    "\u2264": " less than or equal to ",
    "\u2265": " greater than or equal to ",
    "\u2260": " not equal to ",
    "\u2248": " approximately ",
    "\u00b0": " degrees ",
    "\u221a": "square root of ",
    "\u03c0": " pi ",
    "\u221e": " infinity ",
}

# Piper's phonemizer reads an emoji's official Unicode name aloud instead of
# skipping it (a plain smiley becomes "smiling face with smiling eyes"), so
# they come out entirely.
_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols & pictographs, emoticons, transport, supplemental
    "\U00002600-\U000027BF"  # misc symbols & dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicator symbols (flags)
    "\U00002B00-\U00002BFF"  # misc symbols and arrows
    "\U0000FE0F"             # variation selector (emoji presentation)
    "]+"
)


# Markdown. The 9B model formats answers for reading even when told they'll be
# spoken, and Kokoro read "**Origins:**" out as "asterisk asterisk origins".
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")                 # [text](url) -> text
# Underscores only count at word edges, as in real markdown -- otherwise
# "snake_case_name" lost its underscores.
_MD_EMPHASIS = re.compile(r"(\*\*|\*|(?<!\w)__|(?<!\w)_)(?=\S)(.+?)(?<=\S)\1(?!\w)")
_MD_LINE_PREFIX = re.compile(r"^\s*(#{1,6}\s+|[-*+•]\s+|>\s*|\d+[.)]\s+)", re.M)
_MD_LEFTOVER = re.compile(r"\*{2,}|#{2,}|`+|^\s*[-*_]{3,}\s*$|\|", re.M)


def _strip_markdown(text: str) -> str:
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_LINE_PREFIX.sub("", text)
    for _ in range(2):                       # nested: ***x*** / **_x_**
        text = _MD_EMPHASIS.sub(r"\2", text)
    return _MD_LEFTOVER.sub(" ", text)


def _clean_for_speech(text: str) -> str:
    """Strip/convert markup so the synthesiser doesn't read out backslashes,
    braces, and symbols literally."""
    text = _strip_markdown(text)
    text = _MATH_DELIMS.sub("", text)
    text = _DOLLARS.sub("", text)
    text = _FRAC.sub(r"\1 over \2", text)
    text = _SQRT.sub(r"square root of \1", text)
    text = _TEXTCMD.sub(r"\1", text)

    for pattern, word in _LATEX_WORDS:
        text = pattern.sub(word, text)
    for symbol, word in _UNICODE_WORDS.items():
        text = text.replace(symbol, word)

    text = _EMOJI.sub("", text)
    text = _POW_BRACED.sub(r" to the power of \1", text)
    text = _POW_BARE.sub(r" to the power of \1", text)
    text = _ANY_BACKSLASH_CMD.sub("", text)
    text = text.replace("{", "").replace("}", "")
    return _WHITESPACE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Sentence chunking -- the unit of streaming
# ---------------------------------------------------------------------------

# Mid-stream, a terminator only counts when whitespace follows it, so a buffer
# that happens to end on "version 3." isn't mistaken for a finished sentence.
_BREAK_STREAMING = re.compile(r"[.!?\u2026]['\")\]]*(?=\s)|\n")
_BREAK_FINAL = re.compile(r"[.!?\u2026]['\")\]]*(?=\s|$)|\n")
# Anything shorter than this is held back and glued onto the next piece, which
# stops "Mr. Smith" or "3.5" from being spoken as two fragments.
_MIN_SENTENCE_CHARS = 12
# If the model writes a long clause with no terminator at all, break it anyway
# rather than sitting silent -- at a word boundary, not mid-word.
_SOFT_LIMIT = 220


def _extract_sentences(buf: str, final: bool = False):
    """Pull every complete sentence out of `buf`. Returns (sentences, leftover)."""
    pattern = _BREAK_FINAL if final else _BREAK_STREAMING
    out = []
    pos = 0

    while True:
        m = pattern.search(buf, pos)
        if not m:
            break
        candidate = buf[:m.end()].strip()
        if len(candidate) < _MIN_SENTENCE_CHARS:
            pos = m.end()  # too short to stand alone -- keep looking further out
            continue
        if candidate:
            out.append(candidate)
        buf = buf[m.end():].lstrip()
        pos = 0

    if not out and not final and len(buf) > _SOFT_LIMIT:
        cut = buf.rfind(" ", 0, _SOFT_LIMIT)
        if cut > 0:
            out.append(buf[:cut].strip())
            buf = buf[cut:].lstrip()

    if final:
        tail = buf.strip()
        if tail:
            out.append(tail)
        buf = ""

    return out, buf


# ---------------------------------------------------------------------------
# Interruption
# ---------------------------------------------------------------------------

_interrupt = threading.Event()
_active_lock = threading.Lock()
_active_speaker = None


def stop():
    """Cut off whatever Deo is currently saying, immediately. Safe to call from
    any thread, including when nothing is speaking."""
    _interrupt.set()
    sd.stop()  # covers the sd.play() path (cloud TTS, cached phrases)
    with _active_lock:
        speaker = _active_speaker
    if speaker is not None:
        speaker.abort()


def _register(speaker):
    global _active_speaker
    with _active_lock:
        _active_speaker = speaker


def _unregister(speaker):
    global _active_speaker
    with _active_lock:
        if _active_speaker is speaker:
            _active_speaker = None


# ---------------------------------------------------------------------------
# Piper
# ---------------------------------------------------------------------------

_piper_rate = None


def _piper_sample_rate() -> int:
    """--output_raw emits headerless PCM, so the rate has to come from the
    voice's own config file rather than a WAV header."""
    global _piper_rate
    if _piper_rate is not None:
        return _piper_rate

    _piper_rate = 22050  # piper's most common default, used if the json is unreadable
    try:
        with open(config.PIPER_VOICE_MODEL + ".json", encoding="utf-8") as f:
            _piper_rate = int(json.load(f)["audio"]["sample_rate"])
    except Exception as e:
        print(f"[tts] could not read the voice sample rate ({e}), assuming {_piper_rate}Hz")
    return _piper_rate


def _piper_available() -> bool:
    if not os.path.exists(config.PIPER_EXE_PATH):
        print(f"[tts] Piper executable not found at {config.PIPER_EXE_PATH} -- see README for setup.")
        return False
    if not os.path.exists(config.PIPER_VOICE_MODEL):
        print(f"[tts] Piper voice model not found at {config.PIPER_VOICE_MODEL}.")
        return False
    return True


def _piper_cmd(*extra) -> list:
    return [config.PIPER_EXE_PATH, "--model", config.PIPER_VOICE_MODEL, *extra]


# Piper loads its voice model eagerly at spawn, and that load -- not the
# synthesis -- is the bulk of its cold cost: measured on this voice, the first
# audio byte arrives 0.67s after input on a freshly spawned process, but only
# 0.14s if the process has been up for a moment already. So keep one spawned and
# idle, waiting on stdin, and hand it over when a reply starts.
#
# Each process still serves exactly ONE reply: closing its stdin at the end of a
# reply is what gives the reader thread a clean EOF to stop on, which is the
# whole reply-boundary mechanism. So the spare is replaced after each use rather
# than reused.
_spare_lock = threading.Lock()
_spare_proc = None


def _spawn_piper() -> subprocess.Popen:
    return subprocess.Popen(
        _piper_cmd("--output_raw", "--quiet"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def ensure_spare():
    """Make sure a warm Piper process is standing by. Safe to call repeatedly;
    call it from a background thread."""
    global _spare_proc
    if not _piper_available():
        return
    with _spare_lock:
        if _spare_proc is not None and _spare_proc.poll() is None:
            return
        try:
            _spare_proc = _spawn_piper()
        except Exception as e:
            print(f"[tts] could not pre-spawn Piper: {e}")
            _spare_proc = None


def _take_piper_process() -> subprocess.Popen:
    """The warm spare if there is a live one, otherwise a fresh process."""
    global _spare_proc
    with _spare_lock:
        proc, _spare_proc = _spare_proc, None
    if proc is None or proc.poll() is not None:
        proc = _spawn_piper()
    return proc


def _replace_spare_later():
    """Queue up the next warm process. Deliberately called when a reply has
    FINISHED rather than when one starts: loading the voice model is CPU-hungry
    enough that doing it alongside live synthesis measurably delayed the audio
    it was supposed to speed up. Only worth holding a spare while Piper is the
    backend actually in use."""
    if not _cloud_enabled():
        threading.Thread(target=ensure_spare, daemon=True).start()


def shutdown():
    """Kill the idle spare. Call on exit so nothing is left running."""
    global _spare_proc
    with _spare_lock:
        proc, _spare_proc = _spare_proc, None
    if proc is not None and proc.poll() is None:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


class _PiperSpeaker:
    """One Piper process for the whole reply, fed a sentence per line.

    Piper writes raw PCM to stdout as each sentence becomes available, and a
    reader thread pushes that straight into an open output stream -- so audio
    starts as soon as the first sentence is synthesised, and feed() never
    blocks the caller waiting for playback.
    """

    def __init__(self):
        self._proc = None
        self._stream = None
        self._reader = None
        self._pending = b""     # odd trailing byte between reads; int16 frames are 2 bytes
        self._aborted = False
        self.bytes_played = 0

    def _start(self):
        self._proc = _take_piper_process()   # warm if one was standing by
        self._stream = sd.RawOutputStream(
            samplerate=_piper_sample_rate(), channels=1, dtype="int16"
        )
        self._stream.start()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self):
        try:
            while not _interrupt.is_set():
                # read1() hands back whatever has arrived instead of blocking
                # for a full buffer -- that difference IS the streaming.
                chunk = self._proc.stdout.read1(4096)
                if not chunk:
                    break
                data = self._pending + chunk
                usable = len(data) - (len(data) % 2)
                self._pending = data[usable:]
                if usable:
                    self._stream.write(data[:usable])
                    self.bytes_played += usable
        except Exception as e:
            if not self._aborted:
                print(f"[tts] Piper playback stopped: {e}")

    def feed(self, sentence: str) -> bool:
        if self._aborted:
            return False
        if self._proc is None:
            self._start()
        try:
            self._proc.stdin.write((sentence + "\n").encode("utf-8"))
            self._proc.stdin.flush()
            return True
        except Exception as e:
            print(f"[tts] Piper stopped accepting input: {e}")
            self._aborted = True
            return False

    def abort(self):
        """Drop queued audio and kill the synthesiser -- used by stop()."""
        self._aborted = True
        if self._stream is not None:
            try:
                self._stream.abort()
            except Exception:
                pass
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.kill()
            except Exception:
                pass

    def close(self):
        """Wait for everything already fed to finish speaking, then tear down."""
        if self._proc is None:
            return
        try:
            if not self._aborted:
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
                if self._reader:
                    # The reader thread lives for as long as the audio plays
                    # (its writes are what pace playback), so this timeout has
                    # to exceed the length of the reply, not the synthesis --
                    # it's a stuck-process backstop, not a deadline.
                    self._reader.join(timeout=120)
            if self._stream is not None:
                try:
                    if not self._aborted:
                        # stop() drains what the device still holds; abort() would discard it.
                        self._stream.stop()
                finally:
                    self._stream.close()
            if self._proc.poll() is None:
                self._proc.kill()
            self._proc.wait(timeout=5)
        except Exception as e:
            print(f"[tts] error closing Piper: {e}")
        finally:
            _replace_spare_later()


def _render_piper_array(text: str):
    """Synthesise to an in-memory float32 array (no temp file). Used for cached
    fixed phrases and as the per-sentence fallback when cloud TTS drops out."""
    if not _piper_available():
        return None
    try:
        result = subprocess.run(
            _piper_cmd("--output_raw", "--quiet"),
            input=(text + "\n").encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        if not result.stdout:
            detail = result.stderr.decode(errors="replace").strip()
            print(f"[tts] Piper produced no audio: {detail}")
            return None
        raw = result.stdout[: len(result.stdout) - (len(result.stdout) % 2)]
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        return samples, _piper_sample_rate()
    except Exception as e:
        print(f"[tts] Piper render failed: {e}")
        return None


def _speak_piper_blocking(text: str):
    """Non-streaming fallback: whole reply to a temp .wav, then play it. Only
    reached when config.TTS_STREAMING is off, or streaming produced nothing."""
    if not _piper_available():
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name
    try:
        result = subprocess.run(
            _piper_cmd("--output_file", out_path),
            input=text,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"[tts] Piper failed: {result.stderr.strip()}")
            return
        data, samplerate = sf.read(out_path, dtype="float32")
        _play_array(data, samplerate)
    except subprocess.TimeoutExpired:
        print("[tts] Piper timed out generating speech.")
    except Exception as e:
        print(f"[tts] Error during speech playback: {e}")
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


# ---------------------------------------------------------------------------
# Groq / Orpheus (cloud)
# ---------------------------------------------------------------------------

_cloud_failures = 0
_cloud_disabled = False


def _cloud_enabled() -> bool:
    return (
        bool(config.USE_CLOUD_TTS)
        and not _cloud_disabled
        and providers.get_groq_client() is not None
    )


def _note_cloud_failure(e):
    """Orpheus's most common failure is permanent for the session (its model
    terms have not been accepted in the Groq Playground). Without this, every
    reply for the rest of the run would pay a doomed round trip before falling
    back to Piper."""
    global _cloud_failures, _cloud_disabled
    _cloud_failures += 1
    print(f"[tts] Groq Orpheus failed ({e}), using Piper")
    if _cloud_failures >= config.TTS_CLOUD_MAX_FAILURES and not _cloud_disabled:
        _cloud_disabled = True
        print(f"[tts] cloud TTS failed {_cloud_failures}x -- staying on Piper for the rest of "
              f"this session. If that was the Orpheus model terms, accept them once at "
              f"console.groq.com and restart.")


def _render_groq_array(text: str):
    """One sentence -> float32 samples, entirely in memory. Raises on failure."""
    client = providers.get_groq_client()
    if not client:
        raise RuntimeError("Groq client not configured")

    response = client.audio.speech.create(
        model=config.GROQ_TTS_MODEL,
        voice=config.GROQ_TTS_VOICE,
        input=text,
        response_format="wav",
    )
    # The SDK mirrors OpenAI's speech response. Reading the bytes keeps the
    # audio in memory; the old write_to_file() round trip through the
    # filesystem was pure latency on every reply.
    if hasattr(response, "read"):
        audio_bytes = response.read()
    elif hasattr(response, "content"):
        audio_bytes = response.content
    else:
        raise RuntimeError(f"unrecognised speech response type: {type(response)}")

    data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    return data, samplerate


class _GroqSpeaker:
    """Speaks sentence by sentence with one synthesis request in flight ahead of
    playback, so the network round trip for the next sentence overlaps the
    current one being spoken.

    Also used for Kokoro, which renders whole sentences to arrays the same way:
    `render` turns a sentence into (samples, rate), `on_error` records a failure
    before that sentence is re-spoken through Piper."""

    def __init__(self, render=None, on_error=None):
        self._render = render or _render_groq_array
        self._on_error = on_error or _note_cloud_failure
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deo-tts")
        self._queue = queue.Queue()
        self._player = threading.Thread(target=self._play_loop, daemon=True)
        self._player.start()
        self._aborted = False

    def feed(self, sentence: str) -> bool:
        if self._aborted:
            return False
        # The sentence rides along with its future so a mid-reply cloud failure
        # can still be re-synthesised locally instead of being dropped.
        self._queue.put((self._pool.submit(self._render, sentence), sentence))
        return True

    def _play_loop(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            future, sentence = item
            if _interrupt.is_set():
                future.cancel()
                continue
            try:
                data, samplerate = future.result()
            except Exception as e:
                self._on_error(e)
                rendered = _render_piper_array(sentence)
                if rendered is not None and not _interrupt.is_set():
                    _play_array(*rendered)
                continue
            if _interrupt.is_set():
                return
            _play_array(data, samplerate)

    def abort(self):
        self._aborted = True

    def close(self):
        self._queue.put(None)
        self._player.join(timeout=120)  # spans playback, see _PiperSpeaker.close
        self._pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Kokoro (local neural TTS)
# ---------------------------------------------------------------------------

_kokoro = None
_kokoro_failed = False
# Guards both loading and synthesis: the render-ahead worker and prewarm()
# can otherwise hit one onnxruntime session from two threads at once.
_kokoro_lock = threading.Lock()


def _kokoro_enabled() -> bool:
    return (
        bool(getattr(config, "USE_KOKORO_TTS", False))
        and not _kokoro_failed
        and os.path.exists(config.KOKORO_MODEL_PATH)
        and os.path.exists(config.KOKORO_VOICES_PATH)
    )


def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        import logging
        from kokoro_onnx import Kokoro
        # Kokoro's phonemizer logs "words count mismatch" on most sentences
        # with punctuation. It's cosmetic (the audio is fine) and it buried
        # every turn in the console.
        logging.getLogger("phonemizer").setLevel(logging.ERROR)
        t0 = time.monotonic()
        _kokoro = Kokoro(config.KOKORO_MODEL_PATH, config.KOKORO_VOICES_PATH)
        print(f"[tts] Kokoro loaded in {time.monotonic() - t0:.1f}s (voice {config.KOKORO_VOICE})")
    return _kokoro


def _note_kokoro_failure(e):
    """One failure is enough to stop trying: unlike a network hiccup, a local
    model that errors once (bad file, missing espeak data) will keep erroring."""
    global _kokoro_failed
    if not _kokoro_failed:
        _kokoro_failed = True
        print(f"[tts] Kokoro failed ({e}) -- using Piper for the rest of this session")


def _render_kokoro_array(text: str, voice: str = None):
    """One sentence -> (float32 samples, rate). Raises on failure."""
    with _kokoro_lock:
        samples, rate = _get_kokoro().create(
            text,
            voice=voice or config.KOKORO_VOICE,
            speed=config.KOKORO_SPEED,
            lang=config.KOKORO_LANG,
        )
    return np.asarray(samples, dtype=np.float32), rate


def _play_array(data, samplerate):
    """Blocking playback of a finished buffer. sd.stop() (and so tts.stop())
    cuts this short."""
    sd.play(data, samplerate)
    sd.wait()


# ---------------------------------------------------------------------------
# Fixed-phrase cache
# ---------------------------------------------------------------------------

# Deo says a handful of things constantly -- the wake acknowledgement above all.
# Synthesising those from scratch every time costs a process spawn (Piper) or a
# network round trip (Orpheus) at the most latency-sensitive moment there is:
# the gap between the wake word firing and the mic opening.
_CACHEABLE_MAX_CHARS = 48
_CACHE_LIMIT = 16
_phrase_cache = {}
_phrase_cache_lock = threading.Lock()


def _render_array(text: str):
    """Synthesise a short phrase through whichever backend is active."""
    if _cloud_enabled():
        try:
            return _render_groq_array(text)
        except Exception as e:
            _note_cloud_failure(e)
    if _kokoro_enabled():
        try:
            return _render_kokoro_array(text)
        except Exception as e:
            _note_kokoro_failure(e)
    return _render_piper_array(text)


def prewarm(text: str = None):
    """Synthesise a fixed phrase and cache its audio. Call from a background
    thread at startup -- it never blocks anything that matters."""
    text = _clean_for_speech(text if text is not None else config.WAKE_ACK_PHRASE)
    if not text:
        return
    with _phrase_cache_lock:
        if text in _phrase_cache:
            return
    rendered = _render_array(text)
    if rendered is not None:
        with _phrase_cache_lock:
            if len(_phrase_cache) < _CACHE_LIMIT:
                _phrase_cache[text] = rendered

    # Have a loaded Piper ready for the first real reply too -- unless Kokoro
    # is the voice, in which case rendering the phrase above already loaded it.
    if not _cloud_enabled() and not _kokoro_enabled():
        ensure_spare()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _new_speaker():
    if _cloud_enabled():
        return _GroqSpeaker()
    if _kokoro_enabled():
        return _GroqSpeaker(render=_render_kokoro_array, on_error=_note_kokoro_failure)
    if _piper_available():
        return _PiperSpeaker()
    return None


class _CodeFilter:
    """Keeps code out of speech. Everything between ``` fences is dropped and
    replaced, once per reply, by a short spoken note -- the full text, code
    included, still goes to the transcript window. Before this, a reply with a
    script in it started reading "backtick backtick backtick python" aloud."""

    NOTE = "I've put the code on screen."

    def __init__(self):
        self.in_code = False
        self.announced = False

    def feed(self, sentence: str) -> str:
        out = []
        parts = sentence.split("```")
        for i, part in enumerate(parts):
            if i > 0:
                self.in_code = not self.in_code
                if self.in_code and not self.announced:
                    out.append(self.NOTE)
                    self.announced = True
            if not self.in_code:
                out.append(part.replace("`", ""))   # inline `code` -> plain words
        return " ".join(p.strip() for p in out if p.strip())


def speak_stream(chunks) -> str:
    """Speak an iterable of text pieces, starting playback as soon as the first
    complete sentence is available rather than waiting for the last one.

    `chunks` is consumed lazily, so it can be an LLM's output stream: Deo talks
    while the model is still writing. Returns everything actually spoken.
    """
    _interrupt.clear()

    if not config.TTS_STREAMING:
        # Joined with a space, not concatenated: the chunks are usually whole
        # sentences with no trailing space of their own.
        text = _clean_for_speech(" ".join(c.strip() for c in chunks if c and c.strip()))
        if text:
            speak(text)
        return text

    speaker = _new_speaker()
    if speaker is None:
        return ""

    _register(speaker)
    spoken = []
    buf = ""
    code_filter = _CodeFilter()
    try:
        for chunk in chunks:
            if _interrupt.is_set():
                break
            # Separate the chunks. brain.ask_streaming() yields whole sentences
            # with no trailing space, and concatenating those directly produced
            # "now.It" -- which _extract_sentences deliberately refuses to split
            # (a period with no following space is far more likely a decimal or
            # an abbreviation), so the first sentence sat in the buffer waiting
            # for a later one instead of being spoken straight away.
            if buf and not buf[-1].isspace() and not chunk[:1].isspace():
                buf += " "
            buf += chunk
            sentences, buf = _extract_sentences(buf)
            for sentence in sentences:
                cleaned = _clean_for_speech(code_filter.feed(sentence))
                if cleaned and speaker.feed(cleaned):
                    spoken.append(cleaned)

        if not _interrupt.is_set():
            sentences, _ = _extract_sentences(buf, final=True)
            for sentence in sentences:
                cleaned = _clean_for_speech(code_filter.feed(sentence))
                if cleaned and speaker.feed(cleaned):
                    spoken.append(cleaned)
    finally:
        try:
            speaker.close()
        finally:
            _unregister(speaker)

    # A Piper process that was fed sentences but produced no audio at all means
    # the streaming path itself failed (bad exe, bad voice), not that the user
    # interrupted -- fall back so the reply is still spoken, once.
    if (
        isinstance(speaker, _PiperSpeaker)
        and spoken
        and speaker.bytes_played == 0
        and not _interrupt.is_set()
    ):
        print("[tts] streaming produced no audio, retrying the reply through a file")
        _speak_piper_blocking(" ".join(spoken))

    return " ".join(spoken)


def speak(text: str):
    """Speak a complete, already-known string. Blocks until it has finished."""
    if not text:
        return

    text = _clean_for_speech(text)
    if not text:
        return

    _interrupt.clear()

    # Short fixed phrases ("Yeah?") come straight from cache.
    if len(text) <= _CACHEABLE_MAX_CHARS:
        with _phrase_cache_lock:
            cached = _phrase_cache.get(text)
        if cached is not None:
            _play_array(*cached)
            return

        rendered = _render_array(text)
        if rendered is None:
            return
        with _phrase_cache_lock:
            if len(_phrase_cache) < _CACHE_LIMIT:
                _phrase_cache[text] = rendered
        _play_array(*rendered)
        return

    if config.TTS_STREAMING:
        speak_stream(iter([text]))
        return

    if _cloud_enabled():
        try:
            data, samplerate = _render_groq_array(text)
            _play_array(data, samplerate)
            return
        except Exception as e:
            _note_cloud_failure(e)

    if _kokoro_enabled():
        try:
            _play_array(*_render_kokoro_array(text))
            return
        except Exception as e:
            _note_kokoro_failure(e)

    _speak_piper_blocking(text)


def _audition():
    """Speak one sentence in each of a handful of Kokoro voices, so you can
    pick KOKORO_VOICE by ear:  python -m deo.tts --audition"""
    voices = ["am_michael", "am_adam", "am_eric", "am_liam", "am_onyx", "am_puck",
              "am_fenrir", "bm_george", "bm_lewis", "bm_daniel", "af_heart", "bf_emma"]
    for voice in voices:
        line = f"Hi, I'm Deo. This is the {voice.replace('_', ' ')} voice. Steam is open."
        print(f"  {voice}")
        try:
            _play_array(*_render_kokoro_array(line, voice=voice))
        except Exception as e:
            print(f"    failed: {e}")


if __name__ == "__main__":
    import sys
    if "--audition" in sys.argv:
        _audition()
