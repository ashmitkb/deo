"""
"Is this the owner's voice?" -- so Deo answers you, not the TV, a video, or
someone else in the room.

A WeSpeaker ResNet34 speaker-embedding model (sherpa-onnx, CPU, ~27MB) turns
a clip into a 256-number voiceprint; the clip is "you" if it's close enough
(cosine similarity) to the enrolled voiceprint in deo/speaker/enrollment.npy.
Build or rebuild that with:

    python training/enroll_voice.py

Calibrated 2026-10-09 on held-out recordings of the owner (40 "hey Deo" clips,
102 past commands) against 20 different synthetic voices, all level-matched:

    threshold 0.55 -> you 100.0%, other voices 15.0%
    threshold 0.60 -> you  99.3%, other voices 10.0%   <- default
    threshold 0.70 -> you  97.2%, other voices  7.5%

The voices that still got through were unusual synthetic ones (a breathy
whisper); real voices from TV/people should mostly score lower, but this is a
filter, not a lock. It fails OPEN: missing model, missing enrollment, or a
clip too short to judge all count as "you", so a problem here can never make
Deo deaf to its owner.
"""

import os
import threading

import numpy as np

from . import config

_extractor = None
_enrolled = None
_unavailable = False
_lock = threading.Lock()


def _load():
    global _extractor, _enrolled, _unavailable
    if _extractor is not None or _unavailable:
        return
    try:
        import sherpa_onnx as so
        if not (os.path.exists(config.SPEAKER_MODEL_PATH) and os.path.exists(config.SPEAKER_ENROLLMENT_PATH)):
            raise FileNotFoundError("speaker model or enrollment missing -- run training/enroll_voice.py")
        _extractor = so.SpeakerEmbeddingExtractor(so.SpeakerEmbeddingExtractorConfig(
            model=config.SPEAKER_MODEL_PATH, num_threads=2, provider="cpu"))
        _enrolled = np.load(config.SPEAKER_ENROLLMENT_PATH)
        print("[speaker] voice check ready")
    except Exception as e:
        _unavailable = True
        print(f"[speaker] voice check off: {e}")


def embed(audio: np.ndarray) -> np.ndarray:
    """Level-normalised voiceprint. Normalising matters: un-normalised, quiet
    recordings of the owner scored LOWER than loud synthetic voices."""
    audio = np.asarray(audio, dtype=np.float32)
    audio = audio * (0.5 / max(1e-4, float(np.abs(audio).max())))
    stream = _extractor.create_stream()
    stream.accept_waveform(16000, audio)
    stream.input_finished()
    v = np.array(_extractor.compute(stream))
    return v / np.linalg.norm(v)


def similarity(audio: np.ndarray):
    """Cosine similarity to the owner, or None if it can't be judged."""
    if not getattr(config, "SPEAKER_VERIFY", False):
        return None
    with _lock:
        _load()
        if _extractor is None or audio is None:
            return None
        if len(audio) < 16000 * getattr(config, "SPEAKER_MIN_SECONDS", 0.8):
            return None
        if float(np.abs(audio).max()) < 0.003:
            return None
        return float(embed(audio) @ _enrolled)


def is_owner(audio: np.ndarray):
    """(accepted, score). Fails open -- see the module docstring."""
    try:
        score = similarity(audio)
    except Exception as e:
        print(f"[speaker] check failed ({e}), allowing")
        return True, None
    if score is None:
        return True, None
    return score >= config.SPEAKER_THRESHOLD, score


def prewarm():
    with _lock:
        _load()
