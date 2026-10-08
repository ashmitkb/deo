"""
Wake-word detection using openWakeWord -- free, fully local, no account or
API key. Runs a background thread that listens continuously and calls
`on_wake()` whenever a configured wake word crosses config.WAKE_WORD_THRESHOLD.

Setup:
    pip install openwakeword

Getting a custom "hey deo" / "deo" model:
    openWakeWord ships a synthetic-data training notebook -- it generates
    training clips with TTS and trains a small ONNX classifier from them, no
    real recordings or paid account needed. See
    https://github.com/dscripka/openWakeWord (training notebook under
    notebooks/, runs free on Google Colab). Drop the resulting .onnx file(s)
    in deo/wakeword/ and point config.WAKE_WORD_MODEL_PATHS at them.

    To sanity-check the rest of the pipeline (recording, detection loop,
    callback wiring) before training "deo", you can temporarily set
    WAKE_WORD_MODEL_PATHS = ["hey_jarvis"] -- that's one of openWakeWord's
    bundled models and downloads automatically on first use.
"""

import collections
import os
import threading
import time
import wave
from datetime import datetime

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

from . import config, stt

FRAME_SAMPLES = 1280  # openWakeWord expects 80ms chunks at 16kHz


class FrameGate:
    """Rising-edge detector that requires N consecutive frames over threshold.

    Kept as its own small class so it can be tested directly. It is also the
    single most effective defence against false wakes: streamed over real room
    audio, requiring 4 consecutive frames instead of 1 cut false wakes from
    ~587/hour to ~90/hour at the same threshold, because almost all false
    positives are isolated single-frame spikes while a real phrase holds the
    score up for a dozen frames or more.
    """

    def __init__(self, needed: int = 1):
        self.needed = max(1, needed)
        self._streak = 0
        self._armed = False

    def feed(self, above_threshold: bool) -> bool:
        """Returns True exactly once per qualifying run of frames."""
        if not above_threshold:
            self._streak = 0
            self._armed = False
            return False

        self._streak += 1
        if self._streak >= self.needed and not self._armed:
            self._armed = True
            return True
        return False

    @property
    def streak(self) -> int:
        return self._streak


def drowned_out_by_playback(transcript: str, playback: str, threshold: float = 0.6) -> bool:
    """True when most of what the verifier heard is words Deo is saying."""
    import re
    heard = re.findall(r"[a-z']+", transcript.lower())
    if not heard:
        return True          # nothing intelligible over the playback
    said = set(re.findall(r"[a-z']+", playback.lower()))
    return sum(w in said for w in heard) / len(heard) >= threshold


class WakeWordListener:
    def __init__(self, on_wake):
        """on_wake: a zero-argument callable, invoked (from the listener's own
        background thread) every time a wake word crosses threshold."""
        self._on_wake = on_wake
        self._stop = threading.Event()
        self._thread = None

        # only validate paths that look like local files -- bundled model
        # names (e.g. "hey_jarvis") aren't file paths and download themselves
        missing = [
            p for p in config.WAKE_WORD_MODEL_PATHS
            if (p.endswith(".onnx") or p.endswith(".tflite")) and not os.path.isfile(p)
        ]
        if missing:
            raise FileNotFoundError(
                "Wake-word model file(s) not found: " + ", ".join(missing) +
                " -- train one via openWakeWord's synthetic-data notebook and "
                "update config.WAKE_WORD_MODEL_PATHS, or temporarily use a "
                "bundled model name like 'hey_jarvis' to test the pipeline."
            )

        self._model = Model(wakeword_models=config.WAKE_WORD_MODEL_PATHS)
        needed = max(1, getattr(config, "WAKE_WORD_CONSECUTIVE_FRAMES", 1))
        self._gates = {name: FrameGate(needed) for name in self._model.models}

        # Rolling window of recent audio, for the Whisper verifier.
        self._verify = bool(getattr(config, "WAKE_VERIFY", False))
        # Set by main.py: returns what Deo is saying right now, or "" when it
        # isn't speaking. Used by _confirm() during barge-in.
        self.playback_text = lambda: ""
        self._pb = None   # barge-in diagnostics, see _track_playback
        keep = int(getattr(config, "WAKE_VERIFY_SECONDS", 2.0) * 16000 / FRAME_SAMPLES)
        self._recent = collections.deque(maxlen=max(1, keep))

    def _run(self):
        print(f"[wakeword] listening for: {list(self._model.models.keys())}"
              + (" (Whisper-verified)" if self._verify else ""))
        with sd.InputStream(
            samplerate=16000, channels=1, dtype="int16", blocksize=FRAME_SAMPLES
        ) as stream:
            while not self._stop.is_set():
                frame, _ = stream.read(FRAME_SAMPLES)
                frame = frame.flatten()
                self._recent.append(frame)
                predictions = self._model.predict(frame)
                self._track_playback(frame, predictions)

                for name, score in predictions.items():
                    gate = self._gates[name]
                    if not gate.feed(score > config.WAKE_WORD_THRESHOLD):
                        continue
                    print(f"[wakeword] detected '{name}' "
                          f"(score={score:.2f}, {gate.streak} frames)")

                    if self._verify and not self._confirm(stream):
                        continue
                    try:
                        self._on_wake()
                    except Exception as e:
                        print(f"[wakeword] on_wake callback raised: {e}")
                    # on_wake runs a whole conversation on this thread, and the
                    # stream kept buffering the whole time. Scoring that backlog
                    # would replay the conversation -- including Deo's own voice
                    # -- into the detector, so throw it away and start clean.
                    self._flush(stream)
                    break

    def _track_playback(self, frame, predictions):
        """Barge-in diagnostics: while Deo is talking, record the loudest
        wake score and what the mic actually hears, and print one summary line
        when Deo stops. Tells apart "the model hears you but scores low" from
        "the mic is being muted during playback" (driver echo cancellation)."""
        speaking = bool(self.playback_text())
        if speaking:
            if self._pb is None:
                self._pb = {"start": time.monotonic(), "peak_score": 0.0, "levels": [], "audio": []}
            if getattr(config, "DEBUG_SAVE_PLAYBACK_AUDIO", False) and len(self._pb["audio"]) < 1500:
                self._pb["audio"].append(frame.copy())      # <= 2 minutes
            self._pb["peak_score"] = max(self._pb["peak_score"], max(predictions.values(), default=0.0))
            self._pb["levels"].append(float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2))))
        elif self._pb is not None:
            levels = np.array(self._pb["levels"]) if self._pb["levels"] else np.zeros(1)
            db = lambda x: 20 * np.log10(max(float(x), 1e-6))
            print(f"[wakeword] while Deo spoke ({time.monotonic() - self._pb['start']:.0f}s): "
                  f"peak wake score {self._pb['peak_score']:.2f}, mic level median {db(np.median(levels)):.0f} dBFS, "
                  f"loudest {db(levels.max()):.0f} dBFS")
            if self._pb["audio"]:
                try:
                    os.makedirs(config.DEBUG_AUDIO_DIR, exist_ok=True)
                    path = os.path.join(config.DEBUG_AUDIO_DIR,
                                        datetime.now().strftime("playback_%H%M%S.wav"))
                    with wave.open(path, "wb") as wf:
                        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
                        wf.writeframes(np.concatenate(self._pb["audio"]).tobytes())
                    print(f"[wakeword] saved what the mic heard while Deo spoke: {path}")
                except Exception as e:
                    print(f"[wakeword] couldn't save playback audio: {e}")
            self._pb = None

    def _confirm(self, stream) -> bool:
        """Ask Whisper whether "Deo" was actually said. See config.WAKE_VERIFY."""
        # The gate fires a few frames before the phrase ends; let it finish.
        tail_frames = int(config.WAKE_VERIFY_TAIL_SECONDS * 16000 / FRAME_SAMPLES)
        for _ in range(tail_frames):
            frame, _ = stream.read(FRAME_SAMPLES)
            self._recent.append(frame.flatten())

        audio = np.concatenate(list(self._recent)).astype(np.float32) / 32768.0
        t0 = time.monotonic()
        accepted, text = stt.verify_wake(audio)
        took = time.monotonic() - t0
        if accepted:
            # Right words -- but whose voice? Skipped while Deo is talking: a
            # barge-in mixes your voice with Deo's, which would skew the score.
            if not self.playback_text():
                from . import speaker
                mine, score = speaker.is_owner(audio)
                if not mine:
                    print(f"[wakeword] rejected: \"hey Deo\" in someone else's voice ({score:.2f})")
                    self._flush(stream)
                    return False
            print(f"[wakeword] verified: {text!r} ({took:.2f}s)")
            return True

        # Barge-in. While Deo is talking, its own voice through the speakers
        # is louder at the mic than you are, so Whisper transcribes Deo
        # ("India is a diverse...") and misses your "hey Deo" underneath --
        # the first live barge-in attempt was rejected exactly like that. When
        # the transcript is just Deo's own reply, the verifier can't judge, so
        # trust the wake model, which DID fire.
        speaking = self.playback_text()
        if speaking and drowned_out_by_playback(text, speaking):
            print(f"[wakeword] barge-in: heard Deo's own voice over you ({text!r}), "
                  f"trusting the wake model ({took:.2f}s)")
            return True

        print(f"[wakeword] rejected false wake: {text!r} ({took:.2f}s)")
        if config.WAKE_SAVE_REJECTS:
            self._save_reject(audio)
        self._flush(stream)
        return False

    def _flush(self, stream):
        """Drop buffered audio and detector state, so the next detection
        starts from live sound."""
        try:
            backlog = stream.read_available
            if backlog:
                stream.read(backlog)
        except Exception:
            pass
        self._model.reset()
        self._recent.clear()
        for gate in self._gates.values():
            gate.feed(False)

    @staticmethod
    def _save_reject(audio: np.ndarray):
        try:
            os.makedirs(config.WAKE_REJECTS_DIR, exist_ok=True)
            name = datetime.now().strftime("false_wake_%Y%m%d_%H%M%S_%f.wav")
            pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
            with wave.open(os.path.join(config.WAKE_REJECTS_DIR, name), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm.tobytes())
        except Exception as e:
            print(f"[wakeword] couldn't save rejected clip: {e}")

    def start_in_background(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)