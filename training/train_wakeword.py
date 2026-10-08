"""
Train Deo's wake-word model locally, on your voice, with no GPU and no Colab.

    python training/train_wakeword.py                  # train from what's recorded
    python training/train_wakeword.py --synthetic 300  # + Piper-generated positives
    python training/train_wakeword.py --dry-run        # report only, don't write the model

HOW THIS WORKS, AND WHY IT CAN RUN ON A LAPTOP IN A MINUTE

openWakeWord splits detection into two halves. The expensive half -- a
melspectrogram frontend plus Google's speech-embedding network -- is generic,
shared by every wake word, and ships with the pip package as ONNX. The
wake-word-specific half is just a small classifier over a window of 16 of those
96-dim embeddings. We only have to train that second half, which is a ~100k
parameter network over ~1500 features. That is a laptop-scale problem.

So the pipeline is:

    audio clip (2.0s, 16kHz) -> embed_clips() -> (16, 96) -> our classifier -> P(wake)

and the result is exported as ONNX with input [1,16,96] and output [1,1], which
is exactly the contract openWakeWord's runtime expects. deo/wakeword.py then
loads it with no changes at all.

WHAT MAKES OR BREAKS THE RESULT

1. Augmentation, not raw sample count. Every recording is turned into many
   training examples by shifting it in time, changing its gain, and mixing real
   background over it. Time-shifting matters most: at runtime the detector slides
   over a live stream, so the phrase lands at a different offset in the window
   every time. Train on one alignment and it only fires when you get lucky.

2. Negatives that resemble your real environment. Your own debug_audio/ clips are
   ideal -- your voice, your mic, your room, not saying the wake phrase. Piper
   also synthesises "hard negatives" (video, radio, audio, stereo, demo...) which
   share the "-eo/-io" ending that a short wake word like this confuses most.

3. Honest evaluation. The split is by SOURCE recording, never by augmented
   variant -- otherwise shifted copies of the same clip land on both sides of the
   split and the reported numbers are fiction.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE_RATE = 16000
CLIP_SAMPLES = 32000        # 2.0s -> exactly the 16 embedding frames the model takes
EMBED_FRAMES = 16
EMBED_DIM = 96

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_ROOT = os.path.join(HERE, "data")
POSITIVE_DIR = os.path.join(DATA_ROOT, "positive")
NEGATIVE_DIR = os.path.join(DATA_ROOT, "negative")
# Written by the live listener whenever the Whisper verifier rejects a wake
# (config.WAKE_SAVE_REJECTS): real audio from your room that fooled the model.
FALSE_WAKE_DIR = os.path.join(DATA_ROOT, "false_wakes")
# openWakeWord's published negative set: ~10.7 hours of mixed speech, music and
# noise, already run through the same embedding network -- one row per 80ms
# frame, so any 16 consecutive rows are exactly one model input. From
# huggingface.co/datasets/davidscripka/openwakeword_features.
OWW_FEATURES = os.path.join(DATA_ROOT, "oww_features", "validation_set_features.npy")
OWW_HOLDOUT = 0.2          # last 20% of it is never trained on, only measured against
FRAME_SECONDS = 0.08
DEBUG_AUDIO_DIR = os.path.join(PROJECT_ROOT, "debug_audio")
MODEL_OUT = os.path.join(PROJECT_ROOT, "deo", "wakeword", "hey_deo.onnx")

# Words that share the wake phrase's ending or shape. A two-syllable wake word
# is easy to false-trigger, and these are the sounds that do it.
HARD_NEGATIVE_WORDS = [
    "video", "radio", "audio", "stereo", "demo", "Leo", "rodeo", "media",
    "hey there", "hey you", "okay", "hello", "idea", "theory", "day off",
    "ratio", "studio", "trio", "duo", "zero", "hey Google", "video game",
]

# Ordinary speech that is NOT the wake phrase. This is the category that matters
# most and is easiest to under-supply: without it the model has only ever seen
# the phrase and near-silence, so anything speech-shaped scores high and the
# wake word fires constantly. Deliberately includes the requests you'd actually
# make of Deo, since those are the sentences most likely to be spoken at the mic.
NEGATIVE_SENTENCES = [
    "open steam for me", "what is the weather like today",
    "play some music on spotify", "move this window to the other screen",
    "search the web for the latest news", "what time is it right now",
    "remind me to call him back later", "close all of these windows",
    "open brave and go to youtube", "turn the volume down a bit",
    "what did you just say to me", "no not that one the other one",
    "can you read that out again please", "thanks that is all for now",
    "let me think about it for a second", "open the downloads folder",
    "how long until the meeting starts", "write that down as a note",
    "actually never mind forget it", "what is on my screen right now",
    "show me the latest video from that channel", "yes go ahead and do that",
    "stop talking for a moment please", "is the model still running",
    "I need to finish this before tonight", "where did I put that file",
    "the internet seems really slow today", "check if there are any updates",
    "put this on the second monitor instead", "read the first paragraph aloud",
    "what else do you remember about me", "just give me the short version",
    "hold on let me get a pen", "that is not what I asked for",
    "try it one more time for me", "open the settings page",
    "I think something went wrong there", "give me a summary of that",
    "who was the author of that book", "set a timer for ten minutes",
]

rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# audio io
# ---------------------------------------------------------------------------

def read_wav(path: str) -> np.ndarray:
    """Any wav -> float32 mono at SAMPLE_RATE, in [-1, 1]."""
    with wave.open(path, "rb") as wf:
        channels, width, rate, frames = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        raw = wf.readframes(frames)

    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got {width * 8}-bit")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        # linear resample is plenty: everything here is already 16k in practice
        target_len = int(len(audio) * SAMPLE_RATE / rate)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_len), np.arange(len(audio)), audio
        ).astype(np.float32)
    return audio


def fit_to_clip(audio: np.ndarray, offset: int = None) -> np.ndarray:
    """Pad or crop to exactly CLIP_SAMPLES. `offset` places shorter audio inside
    the window -- that's the time-shift augmentation."""
    if len(audio) >= CLIP_SAMPLES:
        start = 0 if offset is None else max(0, min(offset, len(audio) - CLIP_SAMPLES))
        return audio[start:start + CLIP_SAMPLES]
    out = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    room = CLIP_SAMPLES - len(audio)
    start = room // 2 if offset is None else max(0, min(offset, room))
    out[start:start + len(audio)] = audio
    return out


def slice_windows(audio: np.ndarray, stride: int = SAMPLE_RATE) -> list:
    """Chop a long recording into overlapping 2s windows."""
    if len(audio) < CLIP_SAMPLES:
        return [fit_to_clip(audio)]
    return [audio[s:s + CLIP_SAMPLES]
            for s in range(0, len(audio) - CLIP_SAMPLES + 1, stride)]


# ---------------------------------------------------------------------------
# piper synthesis (bootstrap positives + hard negatives)
# ---------------------------------------------------------------------------

def piper_say(text: str, length_scale: float, noise_scale: float, noise_w: float):
    """Synthesise one utterance to float32 audio, or None if Piper isn't set up."""
    from deo import config
    if not (os.path.exists(config.PIPER_EXE_PATH) and os.path.exists(config.PIPER_VOICE_MODEL)):
        return None
    try:
        result = subprocess.run(
            [config.PIPER_EXE_PATH, "--model", config.PIPER_VOICE_MODEL,
             "--output_raw", "--quiet",
             "--length_scale", str(length_scale),
             "--noise_scale", str(noise_scale),
             "--noise_w", str(noise_w)],
            input=(text + "\n").encode("utf-8"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
    except Exception as e:
        print(f"  piper failed: {e}")
        return None
    if not result.stdout:
        return None
    raw = result.stdout[: len(result.stdout) - (len(result.stdout) % 2)]
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    # Piper's voice runs at its own rate; resample to 16k if they differ.
    try:
        with open(config.PIPER_VOICE_MODEL + ".json", encoding="utf-8") as f:
            rate = int(json.load(f)["audio"]["sample_rate"])
    except Exception:
        rate = SAMPLE_RATE
    if rate != SAMPLE_RATE:
        target_len = int(len(audio) * SAMPLE_RATE / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, target_len),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio


def synthesise(texts, per_text: int, label: str) -> list:
    """A spread of deliveries per phrase, by varying Piper's prosody knobs.

    One voice can't stand in for the speaker diversity that
    piper-sample-generator's multi-speaker checkpoint gives you -- these are a
    bootstrap and a source of hard negatives, not a substitute for your voice.
    """
    clips = []
    for text in texts:
        for _ in range(per_text):
            audio = piper_say(
                text,
                length_scale=float(rng.uniform(0.75, 1.45)),
                noise_scale=float(rng.uniform(0.45, 0.90)),
                noise_w=float(rng.uniform(0.5, 1.0)),
            )
            if audio is not None and len(audio) > SAMPLE_RATE * 0.15:
                clips.append(audio)
    print(f"  synthesised {len(clips)} {label} clip(s) from {len(texts)} phrase(s)")
    return clips


# ---------------------------------------------------------------------------
# augmentation
# ---------------------------------------------------------------------------

def augment(audio: np.ndarray, backgrounds: list, variants: int) -> list:
    """Turn one recording into `variants` training clips.

    Time shift is the important one -- see the note at the top of this file.
    """
    out = []
    trimmed = trim_silence(normalize(audio))
    for _ in range(variants):
        room = max(1, CLIP_SAMPLES - len(trimmed))

        # Place the phrase so it ENDS near the end of the window. This matters
        # more than it looks: at inference openWakeWord slides this window over
        # the live stream and the score peaks at the moment the phrase finishes,
        # so "phrase at the end" is the only alignment it ever really sees.
        # Scattering the phrase uniformly across the window instead trains for a
        # harder problem than reality poses, and costs both recall and
        # separation. A minority keep random placement for robustness.
        if rng.random() < 0.8:
            trailing = int(rng.uniform(0, 0.4 * SAMPLE_RATE))   # 0-400ms of room after
            offset = max(0, min(room - 1, room - trailing))
        else:
            offset = int(rng.integers(0, room))

        clip = fit_to_clip(trimmed, offset=offset)

        if backgrounds and rng.random() < 0.65:                  # real room noise
            bg = normalize(backgrounds[int(rng.integers(0, len(backgrounds)))])
            bg = fit_to_clip(bg, offset=int(rng.integers(0, max(1, len(bg)))))
            clip = clip + bg * float(rng.uniform(0.05, 0.45))

        if rng.random() < 0.5:                                   # mic/electrical noise
            clip = clip + rng.normal(0, float(rng.uniform(0.001, 0.02)), CLIP_SAMPLES).astype(np.float32)

        # Overall level LAST, and spanning a very wide range on a log scale.
        # This is the counterpart to normalize(): normalising alone put every
        # training clip at -8..0 dBFS while this microphone actually delivers
        # around -30, so the model was never shown audio at the level it would
        # be given at inference. Applying the gain after the noise and background
        # keeps the signal-to-noise ratio realistic as the level drops, which is
        # what quiet speech genuinely sounds like.
        clip = clip * float(np.exp(rng.uniform(np.log(0.03), np.log(1.2))))

        out.append(np.clip(clip, -1.0, 1.0).astype(np.float32))
    return out


def normalize(audio: np.ndarray, target_peak: float = 0.7) -> np.ndarray:
    """Peak-normalise a clip.

    Necessary because recording levels here vary by 30dB: the recorded samples
    came off a very quiet microphone (peaks around -30 dBFS) while Piper's
    synthesised negatives are near full scale. Left alone, loudness itself
    becomes the easiest thing for the classifier to separate on, which is not
    what we want it learning. Gain augmentation still supplies level variety
    afterwards, deliberately and symmetrically.
    """
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak < 1e-6:
        return audio
    return (audio * (target_peak / peak)).astype(np.float32)


def trim_silence(audio: np.ndarray, threshold: float = None) -> np.ndarray:
    """Drop leading/trailing dead air so the time shift controls alignment
    rather than whatever pause happened to be recorded.

    The threshold is RELATIVE to the clip's own peak by default. A fixed
    absolute threshold was cutting into the quieter parts of quietly-recorded
    words -- on the recorded set it left a median "phrase" of 0.36s, far shorter
    than "hey Deo" actually takes to say, i.e. it was training on fragments.
    """
    if len(audio) == 0:
        return audio
    if threshold is None:
        peak = float(np.abs(audio).max())
        threshold = max(0.01, 0.12 * peak)

    loud = np.abs(audio) > threshold
    if not loud.any():
        return audio
    first, last = int(np.argmax(loud)), int(len(loud) - np.argmax(loud[::-1]))
    pad = int(0.08 * SAMPLE_RATE)
    return audio[max(0, first - pad):min(len(audio), last + pad)]


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def balance(X: np.ndarray, y: np.ndarray, max_negative_ratio: float = 3.0):
    """Equalise the classes, in whichever direction needs it.

    This is where the "fires constantly without anyone speaking" bug lived. With
    100 recordings x 25 augmentations, positives came to ~2500 examples against
    only ~200 negative windows -- and the old code only rebalanced when positives
    were the MINORITY, so it did nothing. A classifier trained on 86% positives
    learns to answer yes.

    Negatives are deliberately allowed to outnumber positives (up to
    max_negative_ratio). For an always-on listener the two errors are not equally
    costly: missing a wake word is a retry, a false wake interrupts you.
    """
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return X, y

    n_pos = min(len(pos), len(neg))
    n_neg = min(len(neg), int(n_pos * max_negative_ratio))
    keep = np.concatenate([
        rng.choice(pos, n_pos, replace=False),
        rng.choice(neg, n_neg, replace=False),
    ])
    rng.shuffle(keep)
    return X[keep], y[keep]


def embed(clips: list, batch_size: int = 32) -> np.ndarray:
    """float32 clips -> (n, 16, 96) embeddings via openWakeWord's own frontend."""
    from openwakeword.utils import AudioFeatures
    af = AudioFeatures()
    pcm = np.stack([np.clip(c * 32767, -32768, 32767).astype(np.int16) for c in clips])
    features = af.embed_clips(pcm, batch_size=batch_size)
    if features.shape[1:] != (EMBED_FRAMES, EMBED_DIM):
        raise RuntimeError(
            f"expected (n, {EMBED_FRAMES}, {EMBED_DIM}) embeddings, got {features.shape}. "
            f"Clips must be exactly {CLIP_SAMPLES} samples ({CLIP_SAMPLES/SAMPLE_RATE}s)."
        )
    return features


# ---------------------------------------------------------------------------
# onnx export
# ---------------------------------------------------------------------------

def load_oww_features(path: str):
    """(train_frames, holdout_frames) from the openWakeWord feature file, or
    (None, None) if it isn't there. Split by position, not at random: the rows
    are a continuous stream, and neighbouring windows overlap by 15/16."""
    if not path or not os.path.exists(path):
        return None, None
    frames = np.load(path, mmap_mode="r")
    if frames.ndim != 2 or frames.shape[1] != EMBED_DIM:
        print(f"  ! {path}: expected (n, {EMBED_DIM}) frames, got {frames.shape} -- ignoring it")
        return None, None
    cut = int(len(frames) * (1 - OWW_HOLDOUT))
    return frames[:cut], frames[cut:]


def sample_windows(frames, count: int) -> np.ndarray:
    """`count` random 16-frame windows -> (count, 16, 96)."""
    starts = rng.integers(0, len(frames) - EMBED_FRAMES, size=count)
    return np.stack([np.asarray(frames[i:i + EMBED_FRAMES]) for i in starts]).astype(np.float32)


def background_wakes_per_hour(clf, frames, thresholds, consecutive: int) -> dict:
    """Stream the classifier over held-out background frames exactly as the live
    listener would -- one score per 80ms frame, rising edges, N consecutive
    frames -- and return false wakes per hour for each threshold."""
    windows = np.lib.stride_tricks.sliding_window_view(np.asarray(frames), (EMBED_FRAMES, EMBED_DIM))
    windows = windows.reshape(-1, EMBED_FRAMES * EMBED_DIM)
    scores = np.concatenate([
        clf.predict_proba(np.ascontiguousarray(windows[i:i + 20000]))[:, 1]
        for i in range(0, len(windows), 20000)
    ])
    hours = len(scores) * FRAME_SECONDS / 3600
    out = {}
    for t in thresholds:
        fires, streak, armed = 0, 0, False
        for above in scores > t:
            if above:
                streak += 1
                if streak >= consecutive and not armed:
                    fires += 1
                    armed = True
            else:
                streak, armed = 0, False
        out[t] = fires / hours
    return out


def streaming_report(model_path: str, pos_sources: list, neg_sources: list, background: dict = None,
                     background_hours: float = 0.0):
    """Evaluate the model the way it will actually run.

    The per-window table above scores isolated 2s windows, which turns out to be
    badly misleading: it reported ~4% false accepts for a model that fired 587
    times an hour in practice. The live detector slides 80ms frames over a
    continuous stream and fires on a RISING EDGE, so what matters is how often an
    edge occurs -- and how much requiring several consecutive frames above
    threshold (config.WAKE_WORD_CONSECUTIVE_FRAMES) suppresses them.
    """
    from openwakeword.model import Model

    oww = Model(wakeword_models=[model_path], inference_framework="onnx")
    key = list(oww.models.keys())[0]

    def score_sequence(audio):
        """Per-frame scores for one recording, as the listener would see them.

        Deliberately NOT normalised: the live listener hands openWakeWord raw
        microphone audio at whatever level it arrives. Normalising here made this
        report measure louder audio than reality provides, which flattered the
        model and hid a train/inference level mismatch entirely.
        """
        pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        oww.reset()
        return [oww.predict(pcm[i:i + 1280])[key]
                for i in range(0, len(pcm) - 1280 + 1, 1280)]

    # Score once, then evaluate every threshold/debounce combination offline.
    pos_scores = [score_sequence(a) for a in pos_sources]
    neg_scores = [score_sequence(a) for a in neg_sources]
    neg_seconds = sum(len(a) for a in neg_sources) / SAMPLE_RATE

    def edges(scores, threshold, consecutive):
        fires, streak, armed = 0, 0, False
        for score in scores:
            if score > threshold:
                streak += 1
                if streak >= consecutive and not armed:
                    fires += 1
                    armed = True
            else:
                streak, armed = 0, False
        return fires

    consecutive = max(1, getattr(__import__("deo.config", fromlist=["config"]),
                                 "WAKE_WORD_CONSECUTIVE_FRAMES", 1))
    print(f"\nStreaming check (as the live listener runs it, "
          f"{consecutive} consecutive frames required):")
    if background:
        print(f"  ('background' = {background_hours:.1f}h of held-out openWakeWord audio: "
              f"speech, music, TV, noise)")
    print(f"  {'thresh':>7} {'detected':>9} {'false wakes/hr':>15}"
          + (f" {'background/hr':>14}" if background else ""))

    best = None
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98):
        detected = sum(1 for s in pos_scores if edges(s, threshold, consecutive) > 0)
        rate = detected / max(1, len(pos_scores))
        false_per_hour = (sum(edges(s, threshold, consecutive) for s in neg_scores)
                          / max(neg_seconds / 3600, 1e-9))
        bg = background.get(threshold) if background else None
        print(f"  {threshold:7.2f} {rate:8.1%} {false_per_hour:14.0f}"
              + (f" {bg:14.1f}" if bg is not None else ""))
        # Background audio is the case that was failing, so it gets a stricter
        # bar than your own speech: it plays for hours at a time.
        if best is None and rate >= 0.8 and false_per_hour <= 6 and (bg is None or bg <= 2):
            best = threshold

    if best:
        print(f"\n  Set WAKE_WORD_THRESHOLD = {best}")
    else:
        print("\n  Nothing here reaches 80% detection at under ~6 false wakes/hour.")
        print("  Pick the row you can live with, then improve the data -- recording")
        print("  level first (see the report above), then more and more varied takes.")
    return best


def export_onnx(clf, path: str):
    """Write the sklearn MLP as ONNX shaped the way openWakeWord's runtime
    expects: input [1,16,96], output [1,1] probability."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    w1, w2 = clf.coefs_[0].astype(np.float32), clf.coefs_[1].astype(np.float32)
    b1, b2 = clf.intercepts_[0].astype(np.float32), clf.intercepts_[1].astype(np.float32)

    initializers = [
        numpy_helper.from_array(np.array([1, EMBED_FRAMES * EMBED_DIM], dtype=np.int64), "flat_shape"),
        numpy_helper.from_array(w1, "W1"), numpy_helper.from_array(b1, "B1"),
        numpy_helper.from_array(w2, "W2"), numpy_helper.from_array(b2, "B2"),
    ]
    nodes = [
        helper.make_node("Reshape", ["input", "flat_shape"], ["flat"]),
        helper.make_node("MatMul", ["flat", "W1"], ["h_lin"]),
        helper.make_node("Add", ["h_lin", "B1"], ["h_bias"]),
        helper.make_node("Relu", ["h_bias"], ["h"]),
        helper.make_node("MatMul", ["h", "W2"], ["o_lin"]),
        helper.make_node("Add", ["o_lin", "B2"], ["o_bias"]),
        helper.make_node("Sigmoid", ["o_bias"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes, "deo_wakeword",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, EMBED_FRAMES, EMBED_DIM])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8   # onnxruntime in this venv predates ir_version 9+
    onnx.checker.check_model(model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    onnx.save(model, path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Deo's wake-word model.")
    parser.add_argument("--synthetic", type=int, default=0,
                        help="Piper-synthesised variants of the wake phrase to add as positives. "
                             "Use to bootstrap before you've recorded much; your own voice is better.")
    parser.add_argument("--phrase", default="hey Deo", help="Wake phrase, for synthesis.")
    parser.add_argument("--variants", type=int, default=25,
                        help="Augmented training clips generated per source recording.")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden units in the classifier.")
    parser.add_argument("--negative-speech", type=int, default=2, metavar="N",
                        help="Piper renderings per NEGATIVE_SENTENCES entry. This is the single "
                             "best defence against false wakes: it teaches the model that "
                             "ordinary speech is not the phrase. 0 to skip.")
    parser.add_argument("--neg-ratio", type=float, default=6.0,
                        help="Max negatives kept per positive after balancing. Higher = fewer "
                             "false wakes at some cost in recall. Was effectively 3.")
    parser.add_argument("--oww-features", default=OWW_FEATURES,
                        help="openWakeWord negative feature file (.npy, one 96-dim row per 80ms). "
                             "Pass '' to skip it.")
    parser.add_argument("--oww-ratio", type=float, default=4.0,
                        help="Background windows from --oww-features added per positive example.")
    parser.add_argument("--dry-run", action="store_true", help="Report metrics, write nothing.")
    parser.add_argument("--out", default=MODEL_OUT)
    args = parser.parse_args()

    print("=== Deo wake-word training ===\n")

    # ---- gather source audio -------------------------------------------------
    print("Sources:")
    pos_files = sorted(glob.glob(os.path.join(POSITIVE_DIR, "*.wav")))
    real_positives = [read_wav(f) for f in pos_files]
    print(f"  recorded positives     : {len(real_positives)}")

    neg_files = sorted(glob.glob(os.path.join(NEGATIVE_DIR, "*.wav")))
    recorded_negatives = [read_wav(f) for f in neg_files]
    print(f"  recorded negatives     : {len(recorded_negatives)}")

    # The listener saves the 2s BEFORE the trigger plus a short tail, so the
    # moment that fooled the model is at the end. Keep the last 2s, which is
    # exactly the window the model scored.
    false_wake_files = sorted(glob.glob(os.path.join(FALSE_WAKE_DIR, "*.wav")))
    false_wakes = [read_wav(f)[-CLIP_SAMPLES:] for f in false_wake_files]
    print(f"  live false wakes       : {len(false_wakes)}  (audio that fooled the model -- the best negatives there are)")

    debug_files = sorted(glob.glob(os.path.join(DEBUG_AUDIO_DIR, "*.wav")))
    debug_negatives = [read_wav(f) for f in debug_files]
    print(f"  debug_audio negatives  : {len(debug_negatives)}  (your voice, your room, not the phrase)")

    synth_positives = []
    if args.synthetic:
        print(f"\nSynthesising with Piper:")
        synth_positives = synthesise([args.phrase], args.synthetic, "positive")
    print("\nSynthesising negatives with Piper (this is what holds false wakes down):")
    hard_negatives = synthesise(HARD_NEGATIVE_WORDS, 2, "hard-negative (similar sounds)")
    if args.negative_speech:
        hard_negatives += synthesise(NEGATIVE_SENTENCES, args.negative_speech,
                                     "negative speech (ordinary sentences)")

    if real_positives:
        peaks = np.array([float(np.abs(a).max()) for a in real_positives])
        lengths = np.array([len(trim_silence(normalize(a))) / SAMPLE_RATE for a in real_positives])
        quiet = int((peaks < 0.06).sum())
        print("")
        print(f"  recording level : median {20 * np.log10(np.median(peaks)):.0f} dBFS"
              f"  ({quiet} of {len(peaks)} below -24 dBFS)")
        print(f"  phrase length   : median {np.median(lengths):.2f}s "
              f"(a spoken \"hey Deo\" should be roughly 0.5-0.9s)")
        if quiet > len(peaks) * 0.3:
            print("  ! Most of these were recorded very quietly. Raise your microphone")
            print("    level in Windows sound settings and re-record -- no amount of")
            print("    training fixes a signal this close to the noise floor.")
        if np.median(lengths) < 0.4:
            print("  ! The phrase is coming out shorter than it should. Either it is being")
            print("    said too fast, or it is being cut off by the 2.0s window.")

    positives = real_positives + synth_positives
    if not positives:
        print("\nNo positive samples. Record some first:")
        print("    python training/record_samples.py")
        return 1
    if not real_positives:
        print("\n  ! No real recordings of your voice -- this model will only know Piper's.")
        print("    Fine for checking the pipeline runs; not fine for actual use.")

    # Background pool for augmentation, and the negative source pool.
    negative_sources = recorded_negatives + debug_negatives + hard_negatives + false_wakes
    negative_minutes = sum(len(a) for a in recorded_negatives + debug_negatives) / SAMPLE_RATE / 60
    print(f"\n  real negative audio : {negative_minutes:.1f} min")
    if negative_minutes < 20:
        print("  ! That is very little. The model has barely heard anything that ISN'T the")
        print("    wake phrase, so background audio (videos, games, music, people talking)")
        print("    is exactly what will set it off. Record some:")
        print("        python training/record_background.py --minutes 30")
    backgrounds = [c for c in (recorded_negatives + debug_negatives) if len(c) > SAMPLE_RATE]

    # ---- split by SOURCE before augmenting (see docstring) -------------------
    def split(items, val_fraction=0.2):
        idx = rng.permutation(len(items))
        cut = max(1, int(len(items) * val_fraction)) if len(items) > 4 else 0
        val = [items[i] for i in idx[:cut]]
        train = [items[i] for i in idx[cut:]]
        return train, val

    pos_train_src, pos_val_src = split(positives)
    neg_train_src, neg_val_src = split(negative_sources)

    print(f"\nSplit by source recording: "
          f"{len(pos_train_src)}/{len(pos_val_src)} positive, "
          f"{len(neg_train_src)}/{len(neg_val_src)} negative (train/val)")

    # ---- build clips ---------------------------------------------------------
    def build(pos_src, neg_src, variants):
        pos_clips, neg_clips = [], []
        for audio in pos_src:
            pos_clips += augment(audio, backgrounds, variants)
        for audio in neg_src:
            windows = slice_windows(normalize(audio))
            neg_clips += windows
            # Negatives get the SAME augmentation treatment as positives. When
            # only positives were heavily augmented, "has added gain, noise and
            # background" was itself a reliable cue for "positive" -- a shortcut
            # the classifier learned instead of learning the phrase.
            for window in windows[:3]:
                neg_clips += augment(window, backgrounds, variants)
        # silence is a negative worth teaching explicitly
        neg_clips += [rng.normal(0, 0.003, CLIP_SAMPLES).astype(np.float32)
                      for _ in range(len(pos_clips) // 20 + 5)]
        return pos_clips, neg_clips

    print("\nBuilding training clips (augmenting)...")
    pos_train, neg_train = build(pos_train_src, neg_train_src, args.variants)
    pos_val, neg_val = build(pos_val_src, neg_val_src, max(4, args.variants // 3)) if pos_val_src else ([], [])
    print(f"  train: {len(pos_train)} positive / {len(neg_train)} negative")
    print(f"  val  : {len(pos_val)} positive / {len(neg_val)} negative")

    print("\nExtracting embeddings (openWakeWord frontend, CPU)...")
    X_train = embed(pos_train + neg_train)
    y_train = np.concatenate([np.ones(len(pos_train)), np.zeros(len(neg_train))])
    print(f"  train features: {X_train.shape}")

    if pos_val and neg_val:
        X_val = embed(pos_val + neg_val)
        y_val = np.concatenate([np.ones(len(pos_val)), np.zeros(len(neg_val))])
        print(f"  val   features: {X_val.shape}")
    else:
        X_val = y_val = None

    # ---- train --------------------------------------------------------------
    from sklearn.neural_network import MLPClassifier
    print(f"\nTraining classifier ({EMBED_FRAMES * EMBED_DIM} -> {args.hidden} -> 1)...")
    clf = MLPClassifier(
        hidden_layer_sizes=(args.hidden,),
        activation="relu",
        max_iter=400,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=0,
    )
    flat_train, y_train = balance(X_train.reshape(len(X_train), -1), y_train,
                                  max_negative_ratio=args.neg_ratio)
    print(f"  balanced to {int((y_train == 1).sum())} positive / "
          f"{int((y_train == 0).sum())} negative examples")
    # Added AFTER balance() so they aren't subsampled away: these are the only
    # examples of TV, music and other people the model gets to see.
    oww_train, oww_holdout = load_oww_features(args.oww_features)
    if oww_train is not None and args.oww_ratio > 0:
        n_bg = int((y_train == 1).sum() * args.oww_ratio)
        bg = sample_windows(oww_train, n_bg).reshape(n_bg, -1)
        flat_train = np.concatenate([flat_train, bg])
        y_train = np.concatenate([y_train, np.zeros(n_bg)])
        mix = rng.permutation(len(y_train))
        flat_train, y_train = flat_train[mix], y_train[mix]
        print(f"  + {n_bg} background windows from openWakeWord's feature set "
              f"({len(oww_train) * FRAME_SECONDS / 3600:.1f}h pool)")
    elif not args.oww_features or oww_train is None:
        print("  (no openWakeWord background features -- see training/README.md)")
    clf.fit(flat_train, y_train)
    print(f"  converged in {clf.n_iter_} iterations, final loss {clf.loss_:.4f}")

    # ---- evaluate -----------------------------------------------------------
    if X_val is not None:
        scores = clf.predict_proba(X_val.reshape(len(X_val), -1))[:, 1]
        print("\nValidation (held-out source recordings):")
        print(f"  {'thresh':>7} {'recall':>8} {'false-accept':>13} {'false wakes/hr':>15}")
        best = None
        for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            tp = int(((scores >= t) & (y_val == 1)).sum())
            fp = int(((scores >= t) & (y_val == 0)).sum())
            recall = tp / max(1, int((y_val == 1).sum()))
            far = fp / max(1, int((y_val == 0).sum()))
            # Each negative window is 2s of audio, so the rate converts into
            # something you can actually feel: wakes per hour of room sound.
            # 1% sounds harmless and is 18 interruptions an hour.
            print(f"  {t:7.2f} {recall:7.1%} {far:12.2%} {far * 1800:14.0f}")
            if best is None and far <= 0.01 and recall >= 0.8:
                best = t
        if best:
            print(f"\n  Suggested WAKE_WORD_THRESHOLD: {best}")
        else:
            print("\n  No threshold hits >=80% recall at <=1% false-accepts.")
            print("  Record more positives (and more varied ones) before trusting this.")

    # ---- export -------------------------------------------------------------
    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    export_onnx(clf, args.out)
    print(f"\nWrote {args.out}")

    # Held-out sources only, streamed -- the number that actually predicts how
    # this will behave when it's listening all day.
    if pos_val_src and neg_val_src:
        background, hours = None, 0.0
        if oww_holdout is not None:
            consecutive = max(1, getattr(__import__("deo.config", fromlist=["config"]),
                                         "WAKE_WORD_CONSECUTIVE_FRAMES", 1))
            print("\nScoring held-out background audio...")
            background = background_wakes_per_hour(
                clf, oww_holdout, (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98), consecutive)
            hours = len(oww_holdout) * FRAME_SECONDS / 3600
        streaming_report(args.out, pos_val_src, neg_val_src, background, hours)

    # Prove openWakeWord itself can load and run what we just wrote.
    from openwakeword.model import Model
    oww = Model(wakeword_models=[args.out], inference_framework="onnx")
    key = list(oww.models.keys())[0]
    test = np.clip(pos_train[0] * 32767, -32768, 32767).astype(np.int16)
    for i in range(0, len(test) - 1280 + 1, 1280):
        score = oww.predict(test[i:i + 1280])[key]
    print(f"openWakeWord loaded it as '{key}'; score on a training positive: {score:.3f}")
    print("\nNext: python main.py  -- then say the phrase and watch for [wakeword] detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
