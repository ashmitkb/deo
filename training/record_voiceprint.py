"""
Record a fresh voice sample for Deo's "only answer my voice" check.

    .venv\\Scripts\\python.exe training\\record_voiceprint.py

A small window shows one sentence at a time; read it out in your normal
voice, at your normal distance from the laptop. Each sentence records for a
few seconds and the next one comes up by itself. Clips are saved to
training/data/voiceprint/ (replacing any earlier set), then the voiceprint
is rebuilt with training/enroll_voice.py.

None of the sentences say "hey Deo", so Deo won't wake up while you record.
"""

import glob
import os
import subprocess
import sys
import threading
import tkinter as tk
import wave

import numpy as np
import sounddevice as sd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "training", "data", "voiceprint")
SECONDS = 4.0

SENTENCES = [
    "Open Brave and search for the weather tomorrow.",
    "Set a timer for ten minutes, please.",
    "What's on my screen right now?",
    "Close Steam and open F1 25.",
    "Remind me to call mum at six o'clock.",
    "Can you explain what this question is asking?",
    "Move this window to the left screen.",
    "Play the latest video from my favourite channel.",
    "How long would it take to drive to the airport?",
    "Create a Python file that sorts a list of numbers.",
    "Turn it down a bit, I'm on a call.",
    "Okay, that's all for now, thank you.",
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for old in glob.glob(os.path.join(OUT, "*.wav")):
        os.remove(old)

    root = tk.Tk()
    root.title("Deo voice sample")
    root.configure(bg="#111114")
    root.geometry("760x300")
    root.attributes("-topmost", True)
    step = tk.Label(root, fg="#9a9aa5", bg="#111114", font=("Segoe UI", 12))
    step.pack(pady=(28, 6))
    line = tk.Label(root, fg="white", bg="#111114", font=("Segoe UI Semibold", 22),
                    wraplength=700, justify="center")
    line.pack(pady=10, expand=True)
    status = tk.Label(root, fg="#9a9aa5", bg="#111114", font=("Segoe UI", 12))
    status.pack(pady=(6, 28))

    def ui(fn):
        root.after(0, fn)

    def run():
        saved = 0
        i = 0
        retries = 0
        while i < len(SENTENCES):
            text = SENTENCES[i]
            ui(lambda t=text, n=i: (step.config(text=f"{n + 1} of {len(SENTENCES)}"),
                                    line.config(text=t, fg="#7a7a85"),
                                    status.config(text="Get ready...")))
            sd.sleep(1300)
            ui(lambda: (line.config(fg="white"), status.config(text="●  Recording -- read it out now")))
            audio = sd.rec(int(SECONDS * 16000), samplerate=16000, channels=1, dtype="int16")
            sd.wait()
            audio = audio.flatten()
            peak = np.abs(audio).max() / 32768
            if peak < 0.02 and retries < 2:
                retries += 1
                ui(lambda: status.config(text="Didn't hear that -- once more, a little louder"))
                sd.sleep(1500)
                continue
            retries = 0
            with wave.open(os.path.join(OUT, f"voice_{i:02d}.wav"), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                w.writeframes(audio.tobytes())
            saved += 1
            i += 1

        ui(lambda: (step.config(text="Done"), line.config(text="Building your voiceprint..."),
                    status.config(text="")))
        result = subprocess.run([sys.executable, os.path.join(ROOT, "training", "enroll_voice.py")],
                                capture_output=True, text=True, cwd=ROOT)
        print(result.stdout + result.stderr)
        summary = (result.stdout.strip().splitlines() or ["(no output)"])[-1]
        ok = result.returncode == 0
        ui(lambda: (line.config(text="All set -- Deo knows your voice." if ok else "Something went wrong.",
                                font=("Segoe UI Semibold", 20)),
                    status.config(text=summary, wraplength=700)))
        root.after(6000, root.destroy)

    threading.Thread(target=run, daemon=True).start()
    root.mainloop()


if __name__ == "__main__":
    main()
