# Deo: Local → Free-Tier Hybrid Migration Plan

**Status:** Planning only, no code yet. Written so you can pick this back up
in any future session without re-deriving it.

## Why this shift makes sense

The laptop was never going to comfortably run a 4B-class reasoning model AND
Whisper AND Piper AND MediaPipe AND a webcam feed all at once — that's the
root of the slowness. The fix isn't a better local model, it's admitting the
brain and ears don't need to live on your machine. What DOES need to stay
local: anything that's always-on (mic for wake word, camera for gestures)
or privacy-sensitive by nature. Everything else is a candidate to move.

## The provider landscape (verified today, expect drift)

**Groq** is the standout pick and can plausibly cover most of the stack
under one API key:
- **No credit card required** for the free tier — this matters, it means
  there's no way to get an accidental bill.
- **LLM**: ~30 requests/min, roughly 1,000–14,400 requests/day depending on
  model. Hosts open models (GPT-OSS 120B/20B, Qwen3 variants, Llama 4) on
  custom LPU hardware — routinely 5-10x faster token generation than a CPU
  running Ollama locally. This alone should fix the "thinking is slow"
  complaint far more thoroughly than any local model swap could.
- **STT (Whisper large-v3 / large-v3-turbo, hosted)**: same model family
  you're already using locally, just fast and free. Free tier: ~2,000
  requests/day and ~28,800 audio-seconds/day — that's roughly 8 hours of
  transcription a day, which is nowhere close to a personal-use ceiling.
  Paid overage if you ever blew through it is $0.04/hour, i.e. not a real
  cost risk.
- **Vision**: some multimodal models (Llama 4 Scout/Maverick, newer Qwen3
  VL variants). Worth flagging: Groq has been actively deprecating/rotating
  vision models recently, so check console.groq.com for the current
  roster when you actually implement this rather than trusting a specific
  model name here.
- **TTS (Orpheus voices)**: exists, sounds good, fast — but it's in
  **Preview** status as of writing, one source cites a free tier as tight
  as ~100 requests/day, and Groq has already withdrawn other Preview
  models once. Treat this as "worth trying," not "worth depending on."

**Google Gemini (AI Studio)** as the backup/secondary:
- Free tier is real but has been repeatedly and unpredictably tightened
  (a ~90% cut to Flash's daily free requests was reported in Dec 2025
  alone). Numbers you'll see cited online range wildly by date.
- Strongest use case for us specifically: **vision fallback**. Gemini
  Flash's multimodal support is mature and stable in a way Groq's vision
  lineup currently isn't.
- Caveat worth knowing: free-tier prompts may be used by Google to improve
  their models. Paid tier doesn't have this clause. Worth deciding if
  that's acceptable for screen-QA content before relying on it.

**Local Piper** stays as the TTS backbone, not a fallback we hope to
retire — cloud TTS free tiers are the least mature/most volatile piece of
this whole plan (see Orpheus and ElevenLabs below). Piper is free forever,
has zero rate limit, and already works. The realistic move is: try a cloud
voice as an optional "nicer voice while the free quota lasts" layer, with
automatic fallback to Piper on any failure — never a hard dependency.

**Orpheus** (Groq's hosted TTS, Canopy Labs' model) deserves a closer look
specifically *because* it's same-provider:
- Free tier is reported at **~100 requests/day** (third-party-sourced, not
  pulled directly from Groq's own docs — verify in-console once set up).
  If one spoken reply = one request, that's actually usable as a real
  daily voice, unlike ElevenLabs' character-based quota above.
- Reuses the same `GROQ_API_KEY` already needed for LLM/STT — zero extra
  account or key to manage.
- LPU-fast (~100 chars/sec generation, near-real-time) and supports inline
  vocal-direction tags like `[cheerful]` / `[whisper]` right in the text —
  genuine delivery-style control, a fun hook for personality later (e.g.
  different tone for a 2am reply vs. a morning greeting).
- The catch: **Preview** status on Groq's own catalog, and Groq has
  already discontinued other Preview models on short notice. English +
  Arabic only, fewer/less-proven voices than ElevenLabs.

**Recommendation:** try Orpheus first -- same-provider convenience and a
free quota that plausibly covers real daily use, not just occasional
flourish. Build the Piper fallback as the *normal* path regardless, not an
edge case -- if Orpheus gets pulled or repriced, Deo should quietly drop
back to Piper without anyone noticing a break. ElevenLabs becomes the
"occasional special" / paid-upgrade-later option rather than the daily
driver, given its much tighter free economics.

**ElevenLabs** (checked at your request) is worth naming specifically:
best-in-class realism, genuinely not robotic. Free tier is 10,000
characters/month, API access included, no card required — but do the
math: that's only ~10 minutes of total audio a month, roughly 2-3 short
spoken replies a day before the quota's gone. Not viable as the everyday
voice at zero cost. Slots into the exact same "optional premium layer over
Piper" pattern as Orpheus — arguably nicer-sounding when it's available,
with an even tighter ceiling, so the fallback-to-Piper path isn't an edge
case here, it's the normal case you'll hit within a day or two of real use.

## Recommended architecture

| Component | Where it runs | Why |
|---|---|---|
| LLM brain | **Groq** (cloud), Ollama as fallback | Speed + quality jump, free tier is generous and stable |
| STT | **Groq Whisper** (cloud), faster-whisper as fallback | Same model family, just fast; 8hr/day free ceiling is a non-issue |
| TTS | **Piper** (local, primary), Orpheus tried first as premium layer, ElevenLabs as occasional/backup | Orpheus's free quota plausibly covers daily use and reuses the Groq key; ElevenLabs' tighter quota fits occasional use better |
| Screen QA vision | **Groq or Gemini vision** (cloud) | Occasional/triggered use, not continuous — cloud is a clean win here |
| Wake word | **openWakeWord** (local, unchanged) | Always-on mic listener — streaming that to any API 24/7 is both a privacy problem and an instant free-tier killer |
| Gesture recognition | **MediaPipe** (local, unchanged) | Same logic as wake word — always-on camera, lightweight enough that cloud buys nothing |
| GUI, MCP filesystem, window control | **Local (unchanged)** | Nothing here needed cloud in the first place, visual polish is a later pass |

## Status as of this session (confirmed decisions + open items)

- ✅ **Groq confirmed** as primary for LLM + STT. Ollama (local) and
  faster-whisper (local) confirmed as the offline fallback -- both already
  exist in the codebase today, so this is really just adding the Groq
  client as the new *primary* path and demoting the current local path to
  fallback, not building fallback from scratch.
- ✅ **TTS**: Piper stays primary. Try Orpheus first (same-provider, likely
  covers real daily use) with automatic fallback to Piper built in from the
  start, not as an afterthought. ElevenLabs kept in reserve for occasional
  use or a later paid upgrade, given its much tighter free quota.
- 🔧 **Wake word**: still pending, plan unchanged from above (synthetic
  bootstrap + real-voice fine-tune).
- 🔧 **Gesture recognition**: reported not working / not confirmed working
  since it was built. Needs debugging next session -- when we pick this up,
  it'll help to know specifically what happened (camera never opened?
  no console `[gesture]` logs at all, meaning MediaPipe/model init failed?
  logs show swipes detected but the window never actually moves? or no
  errors, just nothing happens?). Not urgent to answer now -- just flagging
  it so next session starts at "reproduce the specific failure" instead of
  "does this work at all."
- 🔧 **GUI**: staying local either way -- was never a cloud-migration
  candidate. Visual/UX polish is a deliberately separate, later pass.

### The resilience pattern worth building in from the start

Every cloud call should have a local fallback, in this order:

```
Groq → (rate-limited / down?) → Gemini → (also down?) → local Ollama
```

This is the same "ship working over optimal" instinct that made the
CPU-Whisper call the right one — free tiers *will* occasionally 429 you,
and Deo should degrade gracefully to "slower but working" rather than
just failing. It also means you genuinely can never get surprise-billed:
every provider in the chain has a free tier with no card on file, so the
worst case is a fallback, not a charge.

## Migration order (when we do write code)

1. **Swap the LLM first.** Highest-impact, lowest-risk change — Groq's API
   is OpenAI-compatible and very close in shape to what `ollama.Client.chat()`
   already does in `brain.py`, so this is mostly a client swap, not a
   rewrite. Directly fixes the biggest pain point (speed).
2. **Swap STT second.** `stt.py` already isolates recording from
   transcription cleanly — replacing the `transcribe()` internals to call
   Groq instead of `faster-whisper` is a contained change.
3. **Add the Groq/Gemini fallback for screen QA vision** — lower priority
   since it's already hotkey-triggered and works, just slow/heavier than
   it needs to be.
4. **Experiment with Orpheus TTS as an opt-in layer**, last, since it's
   the least load-bearing and most likely to need revisiting as that
   product matures out of Preview.
5. **Keep wake word and gesture work on their own track** — see below,
   entirely unaffected by any of the above.

## `.env` / config implications

- New keys needed: `GROQ_API_KEY`, and `GEMINI_API_KEY` if we wire in the
  vision fallback. Both go in `.env` via the existing `_load_env()` in
  `config.py` — no new mechanism needed.
- `config.py` will end up with a small provider-priority list instead of a
  single `OLLAMA_MODEL` string, so the fallback chain is a data structure,
  not a pile of if/else.

---

# hey-deo wake word: finishing the job

This entire track stays 100% local and is unaffected by the cloud
migration above.

## Where it actually stands

`openwakeword` is wired up in `wakeword.py`, pointed at
`deo/wakeword/hey_deo.onnx` — a file that doesn't exist yet. That's the
whole blocker: no trained model, not a code problem. `deo_wake_test.py`
is the openWakeWord synthetic-data notebook (via `piper-sample-generator`)
already set up to run for free on Google Colab.

## The plan: synthetic bootstrap + real-voice fine-tune

Pure synthetic (fast, decent) and pure own-voice (slow to collect, but
tuned to *you* specifically) are both individually weaker than combining
them. Recommended path:

1. **Bootstrap with synthetic data first** (what `deo_wake_test.py` already
   does) — generate a large batch of TTS-spoken "hey deo" clips across many
   synthetic voices/cadences, train the initial openWakeWord classifier
   against openWakeWord's standard negative-audio set (non-wake-word
   speech/noise, included in their training pipeline). This alone gives a
   *working* wake word today — matches the project's whole "ship working"
   philosophy.
2. **Record a modest real-voice set** — 50-100 recordings of you actually
   saying "hey deo," varied: normal tone, from across the room, with music
   or the TV on in the background, tired-and-mumbling, etc. This is the
   part that's actually "in progress" per your notes — worth just
   continuing it in parallel, it doesn't block step 1.
3. **Mix real recordings into the synthetic training set** as additional
   positive examples and retrain. This is the standard openWakeWord
   community pattern for personalizing a model without needing thousands
   of real recordings — the synthetic data provides broad coverage, your
   real voice narrows the model to actually fire reliably for *you*
   specifically and resist your particular mic/room's noise profile.
4. **Tune empirically, not theoretically.** Once the trained `.onnx` is
   dropped into `deo/wakeword/` and `config.WAKE_WORD_MODEL_PATHS` points
   at it, the real tuning happens by living with it: `WAKE_WORD_THRESHOLD`
   (already a config knob) goes up if it fires on background noise, down
   if it's missing you. No amount of pre-planning replaces a few days of
   real use here.

## Order relative to the cloud migration

Fully independent — can happen in parallel, before, or after. If you only
have bandwidth for one at a time, the cloud migration has the bigger
day-to-day quality-of-life impact (speed + accuracy), but the wake word is
the one that's already ~80% done and just needs the training data
collection finished.
