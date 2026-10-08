"""
Record wake-word training samples in your own voice.

    python training/record_samples.py                 # 100 "hey Deo" clips
    python training/record_samples.py --count 40      # top up with 40 more
    python training/record_samples.py --label negative --count 20
    python training/record_samples.py --label negative --seconds 30 --count 6

Every clip is saved as 16kHz mono 16-bit WAV -- exactly what openWakeWord's
feature extractor wants -- into training/data/<label>/.

WHY 2.0 SECONDS, EXACTLY: openWakeWord's wake-word models take a window of 16
audio embeddings, and 2.0s of 16kHz audio is what produces precisely 16 frames.
train_wakeword.py asserts this, so don't change CLIP_SECONDS without changing
the model input shape to match.

WHY THE VARYING INSTRUCTIONS: a model trained only on clips of you saying the
phrase clearly, once, at a fixed distance, learns exactly that -- and then
misses you when you're across the room with music on. The prompts below
deliberately walk you through the conditions you'll actually use it in. Boring
but it is the single biggest factor in whether the finished model feels good.
"""

import argparse
import os
import sys
import time
import wave

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CLIP_SECONDS = 2.0          # -> exactly 16 embedding frames, see module docstring
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

# Peak level gates. Clipped audio trains on distortion that won't be there at
# inference time; quiet audio has too little signal above the noise floor to
# learn a phrase from, however much of it you record.
CLIP_PEAK_CEILING = 0.99    # at/above this, samples are being flattened
# -24 dBFS. The first pass of recordings came in at a median of -30 dBFS with a
# floor of -34, which technically passed an earlier, laxer gate and then trained
# a model that couldn't tell the phrase from ordinary speech. Normalising at
# training time raises the words and the room hiss equally, so it cannot recover
# signal that was never captured -- the level has to be right here.
QUIET_PEAK_FLOOR = 0.063
GOOD_PEAK_FLOOR = 0.15      # -16 dBFS: comfortably above the noise floor

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# (through_this_many_samples, what to do differently)
POSITIVE_SCHEDULE = [
    (20, "Normal speaking voice, sitting where you usually sit."),
    (35, "Softer / quieter -- like it's late and someone's asleep."),
    (50, "From further away. Stand up, turn your head aside, face away from the mic."),
    (60, "Faster than feels natural. Run the words together."),
    (70, "Slower, with a small pause between the two words."),
    (85, "Now put music, a video or the TV on in the background and leave it playing."),
    (100, "Tired / mumbled / bored delivery. Vary your pitch between takes."),
]

NEGATIVE_PROMPTS = [
    "Say something else entirely -- a normal request you'd make of Deo.",
    "Read any sentence off a screen near you.",
    "Say words that sound close: video, radio, audio, stereo, demo, Leo.",
    "Just let the room be recorded -- background noise, no speech.",
    "Talk to someone else, or talk over background audio.",
]


def _instruction_for(index: int) -> str:
    for threshold, text in POSITIVE_SCHEDULE:
        if index <= threshold:
            return text
    return POSITIVE_SCHEDULE[-1][1]


def _peak_dbfs(peak: float) -> str:
    if peak <= 0:
        return "-inf dBFS"
    return f"{20 * np.log10(peak):.0f} dBFS"


def _bar(peak: float, width: int = 28) -> str:
    filled = min(width, int(peak * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _next_index(directory: str, prefix: str) -> int:
    existing = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".wav")]
    highest = 0
    for name in existing:
        stem = name[len(prefix):-4].lstrip("_")
        if stem.isdigit():
            highest = max(highest, int(stem))
    return highest + 1


def _save(path: str, audio: np.ndarray):
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def _record(seconds: float) -> np.ndarray:
    frames = int(SAMPLE_RATE * seconds)
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def main():
    parser = argparse.ArgumentParser(description="Record wake-word training clips.")
    parser.add_argument("--label", default="positive", choices=["positive", "negative"],
                        help="positive = you saying the wake phrase; negative = anything else.")
    parser.add_argument("--count", type=int, default=None,
                        help="How many clips to record this session (default 100 positive / 20 negative).")
    parser.add_argument("--phrase", default="hey Deo", help="The wake phrase to say.")
    parser.add_argument("--seconds", type=float, default=None,
                        help="Clip length. Positives must stay at 2.0; negatives can be longer "
                             "(they get sliced into 2.0s windows at training time).")
    args = parser.parse_args()

    is_positive = args.label == "positive"
    seconds = args.seconds or (CLIP_SECONDS if is_positive else 10.0)
    count = args.count or (100 if is_positive else 20)

    if is_positive and abs(seconds - CLIP_SECONDS) > 1e-6:
        print(f"Positive clips must be exactly {CLIP_SECONDS}s -- see the note in this file's docstring.")
        return 1

    out_dir = os.path.join(DATA_ROOT, args.label)
    os.makedirs(out_dir, exist_ok=True)
    prefix = "hey_deo" if is_positive else "negative"
    index = _next_index(out_dir, prefix)
    already = index - 1

    try:
        device = sd.query_devices(kind="input")["name"]
    except Exception:
        device = "(unknown)"

    print(f"\n=== Recording {args.label} samples ===")
    print(f"mic       : {device}")
    print(f"saving to : {out_dir}")
    print(f"clip      : {seconds}s, {SAMPLE_RATE}Hz mono")
    if already:
        print(f"existing  : {already} clip(s) already here -- continuing from #{index}")
    if is_positive:
        print(f"\nYou'll say \"{args.phrase}\" {count} times. The instruction changes every so")
        print("often -- follow it, the variety is what makes the model work in real use.")
    else:
        print(f"\n{count} clips of anything that is NOT the wake phrase.")
    print("\nENTER to record  |  s = skip  |  q = save and quit\n")

    recorded = 0
    attempted = 0
    too_quiet = 0
    last_instruction = None

    while recorded < count:
        overall = already + recorded + 1
        instruction = _instruction_for(overall) if is_positive else \
            NEGATIVE_PROMPTS[(already + recorded) % len(NEGATIVE_PROMPTS)]

        if instruction != last_instruction:
            print(f"\n--- {instruction}\n")
            last_instruction = instruction

        try:
            key = input(f"  #{overall:3d}  ({recorded}/{count} this session) > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if key == "q":
            break
        if key == "s":
            continue

        print("       recording... ", end="", flush=True)
        audio = _record(seconds)
        peak = float(np.abs(audio).max())
        attempted += 1

        if peak >= CLIP_PEAK_CEILING:
            print(f"{_bar(peak)} {_peak_dbfs(peak)}  CLIPPED -- back off the mic, not saved")
            continue
        if peak < QUIET_PEAK_FLOOR:
            too_quiet += 1
            print(f"{_bar(peak)} {_peak_dbfs(peak)}  TOO QUIET -- not saved")
            if too_quiet in (3, 10, 25):
                print()
                print("       Your microphone level is too low to train from. Fix it at the")
                print("       source rather than pushing through:")
                print("         Windows Settings > System > Sound > your mic > Input volume")
                print("       Raise it (and any 'Microphone Boost' under Device properties >")
                print("       Additional settings > Levels), or move closer to the mic.")
                print(f"       Aim for the bar to reach at least {_peak_dbfs(GOOD_PEAK_FLOOR)}.")
                print()
            continue

        path = os.path.join(out_dir, f"{prefix}_{overall:04d}.wav")
        _save(path, audio)
        recorded += 1
        quality = "saved" if peak >= GOOD_PEAK_FLOOR else "saved (a bit quiet)"
        print(f"{_bar(peak)} {_peak_dbfs(peak)}  {quality}")

    total = len([f for f in os.listdir(out_dir) if f.endswith(".wav")])
    print(f"\nDone. {recorded} saved this session ({attempted - recorded} rejected on level).")
    if too_quiet:
        print(f"{too_quiet} of those were too quiet. Raise your microphone input level "
              f"before recording more -- quiet takes cannot be rescued later.")
    print(f"{args.label} clips on disk: {total}")

    if is_positive:
        if total < 40:
            print(f"\n{total} is enough to smoke-test the pipeline but not to rely on.")
            print("Aim for 100+ before judging how well it works.")
        else:
            print("\nNext: python training/train_wakeword.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
