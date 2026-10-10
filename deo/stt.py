import os
import io
import threading
import time
import wave
from datetime import datetime

import numpy as np
import sounddevice as sd
import keyboard
from faster_whisper import WhisperModel
from . import config
from . import providers

_model = None
# Guards loading AND decoding: the wake verifier (listener thread) and normal
# transcription (voice thread) share one model, and two decodes at once on an
# 8GB GPU is how you get an out-of-memory error. RLock because _get_model()
# is called while holding it.
_model_lock = threading.RLock()

# How often the VAD loop wakes up to look at newly captured audio. This is the
# granularity of "you stopped talking", so it lands directly on turn latency.
_VAD_POLL_MS = 25


class _SpeechDetector:
    """Decides whether newly captured audio is speech.

    Prefers Silero VAD -- the same model openWakeWord ships and uses internally,
    so this costs no new dependency -- and falls back to the old fixed amplitude
    gate if it can't be loaded.

    The amplitude gate is what this replaces, and it was the weakest link in
    endpointing: one absolute threshold (config.SILENCE_THRESHOLD) had to suit
    every microphone, gain setting and room, it counted any loud noise as speech,
    and it counted a genuinely quiet voice as silence. A VAD asks "is this a
    person talking", which is the actual question.
    """

    FRAME = 480  # 30ms at 16kHz, Silero's recommended frame size

    def __init__(self):
        self._vad = None
        self._pending = np.zeros(0, dtype=np.float32)
        if getattr(config, "USE_NEURAL_VAD", False):
            try:
                from openwakeword.vad import VAD
                self._vad = VAD()
            except Exception as e:
                print(f"[stt] neural VAD unavailable ({e}), falling back to the amplitude gate")

    @property
    def kind(self) -> str:
        return "silero" if self._vad is not None else "amplitude"

    def reset(self):
        """Call before each recording -- Silero is stateful across frames."""
        self._pending = np.zeros(0, dtype=np.float32)
        if self._vad is not None:
            self._vad.reset_states()

    def is_speech(self, audio: np.ndarray):
        """True/False for the audio supplied, or None if there isn't yet enough
        to judge -- in which case the caller should keep its current state."""
        if audio is None or len(audio) == 0:
            return None

        if self._vad is None:
            return bool(np.abs(audio).mean() > config.SILENCE_THRESHOLD)

        self._pending = np.concatenate([self._pending, audio])
        usable = len(self._pending) - (len(self._pending) % self.FRAME)
        if usable == 0:
            return None

        block, self._pending = self._pending[:usable], self._pending[usable:]
        pcm = np.clip(block * 32767, -32768, 32767).astype(np.int16)
        # Any speech within the newly captured audio counts -- taking the mean
        # would let a short word be averaged away by the silence around it.
        best = 0.0
        for start in range(0, usable, self.FRAME):
            best = max(best, float(self._vad.predict(pcm[start:start + self.FRAME], self.FRAME)))
        return bool(best > config.VAD_SPEECH_THRESHOLD)


_detector = None


def _get_detector() -> _SpeechDetector:
    global _detector
    if _detector is None:
        _detector = _SpeechDetector()
        print(f"[stt] endpointing via {_detector.kind} VAD")
    return _detector


def _load_model(device: str, compute_type: str) -> WhisperModel:
    model = WhisperModel(
        config.WHISPER_MODEL_SIZE,
        device=device,
        compute_type=compute_type,
        cpu_threads=config.WHISPER_CPU_THREADS,
    )
    # Constructing a CUDA model succeeds even when the GPU kernels can't run
    # (missing cuBLAS, unsupported compute type), so run a real decode now.
    # Otherwise the failure would surface on the user's first sentence.
    segments, _ = model.transcribe(np.zeros(SAMPLE_RATE_WARMUP, dtype=np.float32),
                                   language="en", beam_size=1)
    list(segments)
    return model


SAMPLE_RATE_WARMUP = 16000  # 1s of silence is enough to exercise every kernel


def _get_model():
    """The local Whisper model, loaded once. Tries the configured device first
    and drops to CPU if that fails, so a GPU problem costs speed, not Deo."""
    global _model
    with _model_lock:
        if _model is None:
            device, compute = config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE
            print(f"Loading Whisper model ({config.WHISPER_MODEL_SIZE} on {device}/{compute})...")
            t0 = time.monotonic()
            try:
                _model = _load_model(device, compute)
            except Exception as e:
                if device == "cpu":
                    raise
                fallback = getattr(config, "WHISPER_CPU_FALLBACK_COMPUTE_TYPE", "int8")
                print(f"[stt] Whisper on {device} failed ({e}) -- using CPU/{fallback} instead")
                _model = _load_model("cpu", fallback)
            print(f"[stt] Whisper ready in {time.monotonic() - t0:.1f}s")
    return _model


def is_ready() -> bool:
    """Is Whisper loaded? Right after the laptop wakes it can take a minute
    or two (the GPU driver is still recovering), and the wake word needs it."""
    return _model is not None


def _looks_like_gpu_failure(e: Exception) -> bool:
    text = str(e).lower()
    return any(k in text for k in ("cuda", "cublas", "cudnn", "device-side", "gpu"))


def _with_gpu_recovery(run):
    """run(model) -> result, reloading Whisper once if the GPU fell over.

    A graphics driver reset (Windows logs it as LiveKernelEvent 141 -- seen
    here on waking from hibernate) leaves the loaded model pointing at a dead
    CUDA context: every call after that fails until Deo restarts. Reloading
    picks up the recovered GPU, or falls back to the CPU if it hasn't."""
    global _model
    try:
        return run(_get_model())
    except Exception as e:
        if not _looks_like_gpu_failure(e):
            raise
        print(f"[stt] Whisper hit a GPU error ({e}) -- reloading it")
        with _model_lock:
            _model = None
        return run(_get_model())


def prewarm():
    """Load (and on GPU, JIT-compile) Whisper ahead of the first utterance.
    Safe to call from a background thread; never raises."""
    try:
        _get_model()
    except Exception as e:
        print(f"[stt] Whisper prewarm failed: {e}")


def _transcribe_plain(audio: np.ndarray) -> str:
    """Transcribe a wake-word candidate, primed with the phrase itself.

    Priming sounds like cheating, but it was measured on the recordings here
    (25 real "hey Deo" clips, 40 two-second windows of other speech):
        no prompt          :  2/25 caught, 0/40 false accepts
        hotwords="Deo"     :  7/25 caught, 3/40 false accepts
        prompt "Hey Deo."  : 21/25 caught, 0/40 false accepts
    Unprimed, Whisper spells an unfamiliar name as whatever real word is
    closest ("Hey, dear", "Hey, Leo", "Radio"). Primed, it spells it "Deo"
    when the sound is there, and still writes "Can you open Brave?" when it
    isn't -- it didn't put "Deo" into a single negative."""
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak < 1e-4:
        return ""

    def run(model):
        with _model_lock:
            segments, _ = model.transcribe(
                audio * (0.5 / peak),   # the mic is quiet; Whisper does better near -6 dBFS
                language="en",
                beam_size=1,
                without_timestamps=True,
                condition_on_previous_text=False,
                initial_prompt="Hey Deo.",
            )
            return " ".join(seg.text.strip() for seg in segments).strip()
    return _with_gpu_recovery(run)


# "hey deo" as Whisper actually writes it. Built from config.WAKE_WORD_VARIANTS
# plus the near-misses Whisper writes when the audio is unclear.
_WAKE_WORD_RE = None


def _wake_word_re():
    global _WAKE_WORD_RE
    if _WAKE_WORD_RE is None:
        import re
        variants = set(v.lower() for v in config.WAKE_WORD_VARIANTS) | set(WAKE_TRANSCRIPT_VARIANTS)
        alternation = "|".join(sorted((re.escape(v).replace(r"\ ", r"[\s,.-]*") for v in variants),
                                      key=len, reverse=True))
        _WAKE_WORD_RE = re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)
    return _WAKE_WORD_RE


WAKE_TRANSCRIPT_VARIANTS = [
    "deo", "dio", "theo", "deyo", "dayo", "day o", "deyo", "diyo", "deo's",
    "dio's", "theo's", "the o", "teo", "tio", "dao", "deio", "dyo",
]


def mentions_wake_word(text: str) -> bool:
    """True if a transcript contains something that is plausibly "Deo"."""
    return bool(text) and bool(_wake_word_re().search(text))


def verify_wake(audio: np.ndarray):
    """Second opinion on a wake-word trigger. Returns (accepted, transcript).

    Fails OPEN: if Whisper can't run at all, the wake is accepted, because a
    broken verifier silently turning off the wake word would be worse than the
    false wakes it exists to stop."""
    try:
        text = _transcribe_plain(audio)
    except Exception as e:
        print(f"[wakeword] verifier unavailable ({e}), accepting wake")
        return True, ""
    return mentions_wake_word(text), text


def _save_debug_wav(audio: np.ndarray) -> str:
    os.makedirs(config.DEBUG_AUDIO_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(config.DEBUG_AUDIO_DIR, f"utterance_{timestamp}.wav")
    int16_audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.SAMPLE_RATE)
        wf.writeframes(int16_audio.tobytes())
    return path


def wait_for_push_to_talk():
    """Block until the push-to-talk key is pressed.

    Kept SEPARATE from recording on purpose. Waiting for a keypress is not using
    the microphone, but the push-to-talk loop used to hold the mic lock across
    this wait -- which meant the lock was held permanently whenever nobody was
    pressing F9, and the wake word could never acquire it. Every single wake
    detection was rejected with "already listening/speaking".
    """
    print(f"\nHold [{config.PUSH_TO_TALK_KEY.upper()}] to talk...")
    keyboard.wait(config.PUSH_TO_TALK_KEY)


def record_while_held(on_start=None) -> np.ndarray:
    """Record for as long as the push-to-talk key stays down. Call only after
    wait_for_push_to_talk(), and hold the mic lock around this part.

    on_start: optional zero-arg callback fired the instant recording actually
    begins -- lets a caller like the GUI show "Listening..." at the right
    moment instead of for the whole wait-for-key window.
    """
    if on_start:
        on_start()

    print("Listening...")
    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    with sd.InputStream(
        samplerate=config.SAMPLE_RATE, channels=1, dtype="float32", callback=callback
    ):
        while keyboard.is_pressed(config.PUSH_TO_TALK_KEY):
            sd.sleep(50)

    print("Processing...")
    if not frames:
        return np.array([], dtype=np.float32)
    return np.concatenate(frames, axis=0).flatten()


def record_until_silence(max_duration: float = None, start_timeout: float = None) -> np.ndarray:
    """Record starting immediately (no key held), stopping once speech has
    started and then gone quiet for config.SILENCE_DURATION seconds. Used
    after a wake-word detection, where there's no key to hold down.

    start_timeout: give up and return nothing if speech hasn't begun within this
    many seconds. That's what makes follow-up turns possible -- the mic can be
    left open after a reply without committing to a full max_duration wait when
    the user has actually finished talking.
    """
    max_duration = max_duration or getattr(config, "WAKE_WORD_MAX_UTTERANCE_SECONDS", 15.0)
    frames = []
    speech_started = False
    silence_start = None
    start_time = time.monotonic()

    detector = _get_detector()
    detector.reset()

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    checked = 0  # how many captured blocks have already been examined

    print("Listening...")
    with sd.InputStream(
        samplerate=config.SAMPLE_RATE, channels=1, dtype="float32", callback=callback
    ):
        while True:
            sd.sleep(_VAD_POLL_MS)
            elapsed = time.monotonic() - start_time

            if not speech_started and start_timeout and elapsed > start_timeout:
                print("[stt] nothing said, closing the mic")
                return np.array([], dtype=np.float32)

            if len(frames) > checked:
                # Every block captured since the last check, not just the newest
                # one. The callback appends faster than this loop polls, so
                # looking only at frames[-1] silently skipped blocks -- which
                # both delayed the start of speech and let a loud block go
                # unnoticed while silence was being counted, cutting people off
                # mid-sentence.
                new_blocks = frames[checked:]
                checked = len(frames)
                speech = detector.is_speech(np.concatenate(new_blocks, axis=0).flatten())

                if speech is True:
                    speech_started = True
                    silence_start = None
                elif speech is False and speech_started:
                    if silence_start is None:
                        silence_start = time.monotonic()
                    elif time.monotonic() - silence_start > config.SILENCE_DURATION:
                        break

            if elapsed > max_duration:
                print("[stt] hit max utterance duration, cutting off")
                break

    print("Processing...")
    if not frames or not speech_started:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(frames, axis=0).flatten()
    duration = len(audio) / config.SAMPLE_RATE
    if duration < config.MIN_SPEECH_DURATION:
        return np.array([], dtype=np.float32)
    return audio


def _audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    """Same int16 WAV encoding as _save_debug_wav, but in-memory -- used to
    ship the recording to Groq's transcription endpoint without touching disk."""
    buf = io.BytesIO()
    int16_audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.SAMPLE_RATE)
        wf.writeframes(int16_audio.tobytes())
    return buf.getvalue()


def _transcribe_groq(audio: np.ndarray) -> str:
    client = providers.get_groq_client()
    if not client:
        raise RuntimeError("Groq client not configured (missing key, bad install, or disabled)")

    wav_bytes = _audio_to_wav_bytes(audio)
    result = client.audio.transcriptions.create(
        file=("utterance.wav", wav_bytes),
        model=config.GROQ_STT_MODEL,
        language="en",
        prompt=config.WHISPER_INITIAL_PROMPT,  # same vocabulary bias as the local path
        response_format="text",
    )
    # response_format="text" -- SDK may hand back a plain string or an
    # object with a .text attribute depending on version; handle both.
    text = result if isinstance(result, str) else getattr(result, "text", str(result))
    return text.strip()


def _transcribe_local(audio: np.ndarray) -> str:
    return _with_gpu_recovery(lambda model: _decode_command(model, audio))


def _decode_command(model, audio: np.ndarray) -> str:
    # segments is a lazy generator: decoding happens in the join, so both
    # stay inside the lock (see _model_lock).
    with _model_lock:
        segments, _ = model.transcribe(
            audio,
            language="en",
            beam_size=config.WHISPER_BEAM_SIZE,  # greedy by default -- see config
            vad_filter=True,               # strips silence/non-speech before decoding --
                                            # this is what was likely producing phrases like
                                            # "Thank you very much" out of dead air
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,  # stops the model drifting into repeated/
                                                # hallucinated phrasing across segments
            initial_prompt=config.WHISPER_INITIAL_PROMPT,  # biases decoding toward Deo's
                                                             # own vocabulary (assistant name,
                                                             # app names) -- cuts down on
                                                             # misheard proper nouns
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip()


def transcribe(audio: np.ndarray) -> str:
    if audio.size == 0:
        return ""

    if config.DEBUG_SAVE_AUDIO:
        saved_path = _save_debug_wav(audio)
        print(f"[debug] saved audio to {saved_path}")

    if config.USE_CLOUD_STT:
        try:
            return _transcribe_groq(audio)
        except Exception as e:
            print(f"[stt] Groq transcription failed ({e}), falling back to local Whisper")

    # _get_model() is lazy -- if Groq is working, the local Whisper model
    # (a real chunk of RAM/startup time) never even loads.
    text = _transcribe_local(audio)
    if is_prompt_echo(text):
        print(f"(ignored: Whisper echoed its vocabulary hint: {text!r})")
        return ""
    return text


def is_prompt_echo(text: str) -> bool:
    """True if a transcript is nothing but words from WHISPER_INITIAL_PROMPT.

    On silence or noise, Whisper sometimes "hears" its own prompt: Deo was
    handed "DeoProjects, PyCharm, spotify, vscode." as something the user said.
    Real requests always contain words the hint list doesn't ("open", "run")."""
    import re
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    if not words:
        return False
    hint = set(re.findall(r"[a-z0-9]+", config.WHISPER_INITIAL_PROMPT.lower()))
    return all(w in hint for w in words)


def record_push_to_talk(on_start=None) -> np.ndarray:
    """Wait for the key, then record while it's held. Convenience wrapper --
    callers that share the microphone with the wake word should use
    wait_for_push_to_talk() and record_while_held() separately, so the mic lock
    only covers the recording (see the note in wait_for_push_to_talk)."""
    wait_for_push_to_talk()
    return record_while_held(on_start=on_start)


def listen_and_transcribe(on_start=None) -> str:
    audio = record_push_to_talk(on_start=on_start)
    return transcribe(audio)
