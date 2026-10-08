# Training Deo's wake word on your voice

Fully local. No GPU, no Colab, no torch, no 1GB model download. Runs in about a
minute once you've recorded your samples.

## Why this works on a laptop

openWakeWord splits detection in two. The expensive half — a melspectrogram
frontend plus Google's speech-embedding network — is generic, shared by every
wake word, and already ships with the `openwakeword` pip package as ONNX. The
wake-word-specific half is just a small classifier over a window of 16 of those
96-dimensional embeddings.

So we only train the second half: ~100k parameters over ~1,500 features. That's
a laptop-scale problem, which is why this doesn't need the full synthetic-data
training pipeline from openWakeWord's Colab notebook.

```
audio (2.0s, 16kHz) → embed_clips() → (16, 96) → our classifier → P(wake)
```

The result is exported as ONNX with input `[1,16,96]` and output `[1,1]` — the
exact contract openWakeWord's runtime expects — so `deo/wakeword.py` loads it
with no code changes.

## The workflow

**1. Record yourself saying the phrase.**

```bash
python training/record_samples.py
```

Walks you through 100 takes, changing the instruction as you go (normal, quiet,
across the room, fast, slow, with the TV on, tired). Follow the prompts — the
variety is the single biggest factor in whether the finished model feels good.
Clips that clip or come out near-silent are rejected and not saved. Stop any
time with `q` and run it again later to top up.

**2. Record some negatives** (optional but it helps a lot):

```bash
python training/record_samples.py --label negative --count 20
```

Anything that is *not* the wake phrase, plus plain room noise. Your existing
`debug_audio/` clips are already used automatically as negatives — they're
ideal, being your voice in your room not saying the phrase.

**2b. Record background audio. This is what stops wakes from background noise.**

```bash
python training/record_background.py --minutes 30
```

Leave it running while YouTube, a game, music with vocals, the TV or a
conversation plays. Don't say the wake phrase while it runs. Until this existed,
the model had heard only about 5 minutes of real non-wake audio, and none of it
was media. That's why it woke up on videos and other people talking: it had
never been shown what those sound like. Aim for at least 30 minutes, spread
across several different kinds of sound.

**2c. Let Deo collect its own mistakes.** With `WAKE_VERIFY` on (the default),
every trigger the Whisper check rejects is saved to `training/data/false_wakes/`.
These are the most valuable negatives there are: the exact sounds that fooled
the model, recorded in your room. The trainer uses them automatically, so the
model improves each time you retrain after using Deo for a few days.

**2d. openWakeWord's background set (already downloaded).**
`training/data/oww_features/validation_set_features.npy` holds about 10.7
hours of mixed speech, music, TV and noise, already converted into the
features the wake model reads. The trainer uses the first 80% as negatives
(`--oww-ratio`, 4 per positive by default) and keeps the last 2.1 hours aside
to test against, shown in the `background/hr` column. Adding it took the old
model from **86 false wakes/hour to 1.4** on that held-out audio at threshold 0.98.
Source: huggingface.co/datasets/davidscripka/openwakeword_features.

**3. Train.**

```bash
python training/train_wakeword.py
```

Writes `deo/wakeword/hey_deo.onnx`, which is where `config.WAKE_WORD_MODEL_PATHS`
already points, then verifies openWakeWord can load and score it.

**4. Read the STREAMING table, not the per-window one.**

The script prints two evaluations. Only the second one predicts real behaviour.

The per-window table scores isolated 2-second windows. It is badly optimistic: it
reported ~4% false accepts for a model that, streamed over real room audio, fired
**587 times an hour**. The live detector slides 80ms frames over a continuous
stream and fires on a rising edge, so what matters is how often an edge occurs.

The streaming table measures exactly that, on held-out recordings, with the
consecutive-frame requirement applied. Aim for **≥80% detected at under ~6 false
wakes/hour**, then set `WAKE_WORD_THRESHOLD` to that row and live with it for a
few days.

## Recording level matters more than anything else

The first 100 samples recorded here came in at a median peak of **−30 dBFS**,
which produced a model that could not tell the phrase from ordinary speech no
matter how it was trained. Two reasons:

- there is too little signal above the room's noise floor to learn from, and
  normalising at training time raises the words and the hiss equally, so it
  cannot recover what was never captured;
- the silence trimmer had to work at a threshold comparable to the signal itself,
  so it sliced the quieter halves off words — the median "hey Deo" came out at
  0.36s, i.e. the model was trained on fragments.

If the recorder keeps saying TOO QUIET, fix it at the source:

> Windows Settings → System → Sound → your microphone → Input volume,
> and *Microphone Boost* under Device properties → Additional settings → Levels.

The recorder now rejects anything below −24 dBFS and tells you when a clip is
merely "a bit quiet". The trainer also prints a level report before training, so a
bad set is visible before you spend a run on it.

### The subtlety that cost two training runs

Peak-normalising every clip fixed trainability, but on its own it created a worse
problem: it put every training clip at −8 to 0 dBFS while the microphone actually
delivers around −30. The training distribution didn't overlap the real input level
*at all*, so the model was never shown audio resembling what it gets at inference.

Two things follow, and both are now done:

- gain augmentation is applied **last** and spans a very wide range on a log scale
  (≈ −30 dBFS up to near full scale), so quiet input is inside the training
  distribution rather than outside it;
- `streaming_report` does **not** normalise, because the live listener gets raw
  microphone audio. Normalising there measured louder audio than reality provides,
  flattered the model, and hid the mismatch completely.

The general lesson: any preprocessing applied during training has to be applied at
inference too, or else trained around. Normalisation is applied at training only,
so it has to be trained around — which is what the wide gain range does.

## Two knobs that control false wakes

- `WAKE_WORD_THRESHOLD` — the obvious one.
- `WAKE_WORD_CONSECUTIVE_FRAMES` — how many consecutive 80ms frames must clear
  the threshold. This is the effective one. Real speech holds the score up for a
  dozen frames or more; nearly every false positive is a single isolated spike.
  Going from 1 frame to 4 cut measured false wakes from ~587/hour to ~90/hour at
  the same threshold. Raise it before lowering your expectations.

## What's already here

`deo/wakeword/hey_deo_synthetic.onnx` was trained on Piper-synthesised positives
only, as a pipeline check. It loads and works, but at ~3% false-accepts on real
room audio it would wake roughly once a minute while you talk. To try it anyway,
point `WAKE_WORD_MODEL_PATHS` at it. It is not a substitute for step 1.

## Notes

- **Clips must be exactly 2.0s / 32,000 samples.** That's what produces exactly
  16 embedding frames. Both scripts assert it.
- **"hey Deo" beats "Deo".** A one-syllable wake word has very little acoustic
  signal to work with and false-fires constantly. openWakeWord's own guidance is
  3+ syllables. If you do want bare "Deo", expect to need noticeably more
  training data and a higher threshold.
- **`--synthetic N`** adds Piper-generated positives. Useful to bootstrap before
  you've recorded much, but it's one voice — it can't stand in for the speaker
  diversity that real multi-speaker synthesis gives you, and it certainly can't
  stand in for yours.
- **Hard negatives** (video, radio, audio, stereo, demo, Leo, rodeo…) are
  synthesised every run. They share the "-eo/-io" ending that short wake words
  confuse most, and they measurably cut false accepts.
- The model is **speaker-specific by design**. Trained on your voice, it will
  largely ignore other people — a feature for a personal assistant, not a bug.
