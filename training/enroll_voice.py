"""
Build Deo's voiceprint of you, for "only answer my voice" (deo/speaker.py).

    python training/enroll_voice.py

Uses recordings that are already yours:
    training/data/voiceprint/*.wav a fresh sample (training/record_voiceprint.py)
                                   -- weighted 3x, it's the most like Deo today
    training/data/positive/*.wav   your "hey Deo" samples
    debug_audio/utterance_*.wav    past commands you gave Deo (if any)

Re-run it after re-recording your wake-word samples, or if Deo starts
ignoring you (check logs/deo.log for "[speaker] not your voice (0.xx)").
"""

import glob
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deo import config, speaker  # noqa: E402


def load(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768


def main():
    root = config.PROJECT_ROOT
    fresh = sorted(glob.glob(os.path.join(root, "training", "data", "voiceprint", "*.wav")))
    files = sorted(glob.glob(os.path.join(root, "training", "data", "positive", "*.wav")))
    files += sorted(glob.glob(os.path.join(root, "debug_audio", "utterance_*.wav")))
    usable = lambda a: len(a) >= 16000 and np.abs(a).max() > 0.01
    fresh_clips = [a for a in (load(f) for f in fresh) if usable(a)]
    clips = fresh_clips + [a for a in (load(f) for f in files) if usable(a)]
    if len(clips) < 20:
        print(f"Only {len(clips)} usable recordings -- need at least 20. Record some with:")
        print("    python training/record_samples.py")
        return 1

    if not os.path.exists(config.SPEAKER_MODEL_PATH):
        print(f"Missing {config.SPEAKER_MODEL_PATH} -- see deo/config.py (SPEAKER_MODEL_PATH).")
        return 1
    import sherpa_onnx as so
    speaker._extractor = so.SpeakerEmbeddingExtractor(so.SpeakerEmbeddingExtractorConfig(
        model=config.SPEAKER_MODEL_PATH, num_threads=4, provider="cpu"))

    prints = np.array([speaker.embed(c) for c in clips])
    weights = np.array([3.0] * len(fresh_clips) + [1.0] * (len(clips) - len(fresh_clips)))
    centroid = (prints * weights[:, None]).sum(axis=0)
    centroid /= np.linalg.norm(centroid)
    scores = prints @ centroid
    os.makedirs(os.path.dirname(config.SPEAKER_ENROLLMENT_PATH), exist_ok=True)
    np.save(config.SPEAKER_ENROLLMENT_PATH, centroid)

    if fresh_clips:
        f = scores[:len(fresh_clips)]
        print(f"Fresh sample ({len(fresh_clips)} clips) vs the voiceprint: min {f.min():.2f}, median {np.median(f):.2f}")
    below = int((scores < config.SPEAKER_THRESHOLD).sum())
    print(f"Enrolled from {len(clips)} recordings -> {config.SPEAKER_ENROLLMENT_PATH}")
    print(f"Your clips vs the voiceprint: min {scores.min():.2f}, median {np.median(scores):.2f}; "
          f"{below} of {len(scores)} below the threshold ({config.SPEAKER_THRESHOLD}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
