"""
Live microphone level meter. Use this to set your input volume BEFORE recording
training samples, instead of finding out afterwards that 100 takes were too quiet.

    python training/check_mic_level.py

Talk normally, at the distance you'd actually use Deo from. Adjust the Windows
input volume slider while it runs and watch the numbers move.

WHAT TO AIM FOR

    peak around -12 to -6 dBFS while speaking     <- good
    peak below -24 dBFS                           <- too quiet to train from
    peak at 0 dBFS                                <- clipping, back it off

The first set of samples recorded for this project came in at a median peak of
-30 dBFS, which produced a wake-word model that could not tell the phrase from
ordinary speech. Normalising at training time cannot fix it: raising a quiet
recording raises the room hiss with it, so the signal-to-noise ratio is fixed at
the moment of capture. Hence this script.

A note on audio enhancements: if "Voice Clarity" / "Voice Focus" style processing
is enabled for the device, the level you see here will drift on its own as the
automatic gain control works -- which is itself a reason to turn it off. Whatever
you settle on, keep it the same for recording AND for running Deo; preprocessing
that differs between the two is a train/inference mismatch.
"""

import sys
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
BLOCK = 800          # 50ms
DURATION = 30.0      # seconds before it gives up on its own

GOOD_PEAK = 0.25     # -12 dBFS
OK_PEAK = 0.063      # -24 dBFS, the recorder's hard floor
CLIP_PEAK = 0.99


def dbfs(value: float) -> float:
    return 20 * np.log10(value) if value > 1e-9 else -99.0


def bar(peak: float, width: int = 40) -> str:
    """Log-scaled meter: a linear bar makes everything below -20dB invisible,
    which is precisely the range being diagnosed."""
    if peak <= 1e-9:
        return "[" + "-" * width + "]"
    fraction = max(0.0, min(1.0, (dbfs(peak) + 60) / 60))   # -60..0 dBFS
    filled = int(fraction * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def verdict(peak: float) -> str:
    if peak >= CLIP_PEAK:
        return "CLIPPING -- back off"
    if peak >= GOOD_PEAK:
        return "good"
    if peak >= OK_PEAK:
        return "usable, could be louder"
    return "TOO QUIET"


def main():
    try:
        device = sd.query_devices(kind="input")["name"]
    except Exception:
        device = "(unknown)"

    print(f"\nMic: {device}")
    print(f"Listening at {SAMPLE_RATE}Hz. Talk normally from where you'd use Deo.")
    print("Adjust the Windows input volume slider while this runs. Ctrl+C to stop.\n")
    print(f"  aim for peak between {dbfs(GOOD_PEAK):.0f} and -6 dBFS while speaking\n")

    speech_peaks = []
    start = time.monotonic()

    def callback(indata, frames, time_info, status):
        peak = float(np.abs(indata).max())
        if peak > OK_PEAK * 0.5:          # only judge blocks with something in them
            speech_peaks.append(peak)
        print(f"\r  {bar(peak)} {dbfs(peak):6.1f} dBFS  {verdict(peak):<24}", end="", flush=True)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=BLOCK, callback=callback):
            while time.monotonic() - start < DURATION:
                sd.sleep(100)
    except KeyboardInterrupt:
        pass

    print("\n")
    if not speech_peaks:
        print("Heard nothing at all. Check that the right microphone is selected as the")
        print("Windows default input device, and that it isn't muted.")
        return 1

    peaks = np.array(speech_peaks)
    loud = float(np.percentile(peaks, 90))
    print(f"While you were speaking: typical peak {dbfs(float(np.median(peaks))):.1f} dBFS, "
          f"loudest {dbfs(loud):.1f} dBFS")

    if loud >= CLIP_PEAK:
        print("=> Clipping. Lower the input volume a little.")
    elif loud >= GOOD_PEAK:
        print("=> Good. Go record: python training/record_samples.py")
    elif loud >= OK_PEAK:
        print("=> Usable but quiet. Raise the input volume (and Microphone Boost under")
        print("   Device properties > Additional settings > Levels) if you can.")
    else:
        print("=> Too quiet to train from. Raise the input volume, turn off any 'Voice")
        print("   Clarity' / 'Voice Focus' enhancement for this device, and move closer.")
        print("   Recording 100 samples at this level will not produce a working model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
