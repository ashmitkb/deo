"""
Record long stretches of everyday background audio as wake-word NEGATIVES.

    python training/record_background.py                  # 30 minutes
    python training/record_background.py --minutes 60
    python training/record_background.py --minutes 10 --tag youtube

WHY THIS IS THE FIX FOR "IT WAKES UP ON BACKGROUND NOISE"

The wake model is a classifier, and a classifier only learns to say "no" to the
kinds of sound it has been shown. Until now that was about five minutes of real
audio: your own commands from debug_audio/, plus some Piper sentences in one
synthetic voice. It had never heard a YouTube video, a game, music, the TV or
someone else talking in your room. So when it hears any of those, it has no
basis for telling them apart from "hey Deo".

This script just leaves the mic open while life happens, and saves 30-second
chunks into training/data/negative/. Leave it running while you:

    * watch YouTube / Netflix / anime with the sound on speakers
    * play a game with voice chat or a talky soundtrack
    * play music (with vocals -- that's the hard case)
    * talk to someone, take a call, talk to yourself
    * just sit there (fans, typing, mouse clicks)

IMPORTANT: don't say "hey Deo" (or "Deo") while this is recording -- everything
captured is labelled as NOT the wake word. If you slip, note the time printed by
the chunk counter and delete that chunk's file.

Variety beats volume: 10 minutes each of five different things is worth far more
than an hour of one podcast. Then retrain:

    python training/train_wakeword.py
"""

import argparse
import os
import sys
import time
import wave
from datetime import datetime

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
NEGATIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "negative")

# A chunk this close to digital silence is a failed capture (mic muted, wrong
# device), not "a quiet room" -- a quiet room still has a noise floor.
DEAD_PEAK = 0.001


def _save(path: str, audio: np.ndarray):
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Record background audio as wake-word negatives.")
    parser.add_argument("--minutes", type=float, default=30.0, help="How long to record in total.")
    parser.add_argument("--tag", default="background",
                        help="Filename label, e.g. youtube, game, music, talking -- handy for "
                             "deleting one session later if something went wrong.")
    args = parser.parse_args()

    os.makedirs(NEGATIVE_DIR, exist_ok=True)
    chunks = max(1, int(round(args.minutes * 60 / CHUNK_SECONDS)))
    stamp = datetime.now().strftime("%Y%m%d_%H%M")

    print(__doc__.split("IMPORTANT:")[0].strip().splitlines()[0])
    print(f"\nRecording {chunks} x {CHUNK_SECONDS}s chunks (~{chunks * CHUNK_SECONDS / 60:.0f} min) "
          f"into {NEGATIVE_DIR}")
    print("Do NOT say the wake phrase while this runs. Ctrl+C to stop early -- "
          "everything recorded so far is kept.\n")

    saved = 0
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            for i in range(1, chunks + 1):
                audio, overflowed = stream.read(SAMPLE_RATE * CHUNK_SECONDS)
                audio = audio.flatten()
                peak = float(np.abs(audio).max())
                level = f"{20 * np.log10(peak):.0f} dBFS" if peak > 0 else "-inf dBFS"

                if peak < DEAD_PEAK:
                    print(f"  [{i}/{chunks}] {time.strftime('%H:%M:%S')}  skipped: digital silence "
                          f"-- is the mic muted or the wrong device selected?")
                    continue

                name = f"{args.tag}_{stamp}_{i:03d}.wav"
                _save(os.path.join(NEGATIVE_DIR, name), audio)
                saved += 1
                note = "  (overflow: some audio dropped)" if overflowed else ""
                print(f"  [{i}/{chunks}] {time.strftime('%H:%M:%S')}  saved {name}  peak {level}{note}")
    except KeyboardInterrupt:
        print("\nStopped.")

    print(f"\nSaved {saved} chunk(s), {saved * CHUNK_SECONDS / 60:.1f} min of negatives.")
    print("Next: python training/train_wakeword.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
