# Deo roadmap

Supersedes `deo/DEO_MIGRATION_PLAN.md`, whose cloud migration is now **done** —
Groq is primary for LLM and STT, Orpheus is primary for TTS with Piper beneath
it, and every leg has a local fallback. Keep that file for the provider research
(the free-tier economics are still accurate and worth re-reading before changing
providers); ignore its "migration order" section.

## 2026-10-08: back to fully local, and a wake-word second opinion

- **`LOCAL_ONLY` (default on)** in `config.py`; `DEO_LOCAL=0` in `.env` brings
  Groq back. Measured on this laptop (i7-13650HX, 32GB, RTX 5060 8GB):
  Whisper large-v3-turbo on GPU ~0.6s per 20s of speech (CPU: ~7s);
  `qwen3:4b-instruct-2507` ~0.2s to first token, ~90 tok/s, tool calls correct.
  `qwen3.5:9b` is smarter (and has vision) at ~45 tok/s, but at 8k context it
  already spills 12% to CPU on its own, so it can't share 8GB with GPU Whisper.
- **GPU Whisper actually works now.** `main.py` only called `add_dll_directory`,
  and CTranslate2 still couldn't find `cublas64_12.dll`; the bin dirs now go on
  PATH too. `int8_float16` fails on Blackwell with this cuBLAS (12.4), so it
  runs float16. The first GPU decode in a process costs ~10-20s of kernel JIT,
  so Whisper is prewarmed at startup. A GPU failure falls back to CPU.
- **Wake verification** (`WAKE_VERIFY`): when the wake model fires, Whisper
  transcribes the last ~2s and the wake only counts if a "Deo"-like word is in
  it. Rejected triggers are saved to `training/data/false_wakes/` and the
  trainer uses them as negatives.
- **Root cause of background wakes:** the model had heard ~5 minutes of real
  non-wake audio and no media at all. `training/record_background.py` fixes the
  data; the positives also still need re-recording at a proper level (−30 dBFS).
- **Kokoro TTS** (`am_michael`) is now the local voice, with Piper as fallback.
  It runs on the CPU at ~4.5x real time (a short sentence takes ~0.7s), so it
  doesn't take GPU memory. Pick a different voice with `python -m deo.tts --audition`.
- **cuBLAS 12.9** lets GPU Whisper use `int8_float16`: ~1.2GB of VRAM instead of
  ~2.2GB, and a 5.6s first-call warm-up instead of 15s, at the same speed.
- **Wake model retrained with openWakeWord's 10.7h background set.** On 2.1h
  of held-out background audio at threshold 0.98: 86 false wakes/hour before,
  1.4 after. Old model kept as `deo/wakeword/hey_deo.before_oww.onnx`.
- **Permission gate** (`deo/permissions.py`): every tool call is checked in
  `Brain._run_tool`. Deleting, overwriting, moving, killing, shutdown, registry,
  services, installs, firewall, git history and forgetting memories need a click
  in a topmost Windows dialog. "No" is the default, it auto-denies after 60s,
  and every decision goes to `permissions.log`. Hidden or machine-wrecking
  commands (encoded PowerShell, format, diskpart, bcdedit, vssadmin) are always
  blocked. In exchange, the filesystem server now covers all of `C:\Users\ashmi`.
- **New tools from live testing:** `close_app` (sends the same close request as
  clicking ✕, never a kill), `look_at_screen` (vision model as a tool), and
  `open_app` now discovers Steam games and Start Menu/desktop shortcuts.
- **Gestures:** `GESTURE_MODE = "fingers"`. Hold 1 finger to move the active
  window right, 2 fingers to move it left (fires once per pose). Swiping
  mixed up directions because the wind-up before a swipe is itself a small
  swipe the other way. Also fixed: an `IsZoomed` crash, moving Windows panels
  and the desktop itself, and the Deo window getting stuck as the target.
- **Brain switches by game mode** (`deo/gamemode.py`): `qwen3.5:9b` normally
  (35-41 tok/s next to GPU Whisper), `qwen3:4b` while a game runs from a
  Steam/Epic/EA library, with the 9B unloaded within ~10s so the game gets
  the VRAM.
- **Barge-in works with the speakers on.** Saying "hey Deo" mid-answer stops
  it. The verifier trusts the wake model when Whisper only hears Deo's own
  voice. Recorded mic audio showed Deo's playback reaching the mic at about
  -52 dBFS and the user's "hey Deo" at -8 to -14, so no echo cancellation was
  needed. (WebRTC AEC via `livekit` was tried and hurt double-talk detection
  in simulation, 12/12 -> 6/12.) An interrupted reply is kept in history, so
  "continue" resumes it.
- **Follow-ups only answer speech meant for Deo:** a one-word local-model check
  (~0.5s) stops it answering phone calls and side conversations.
- **Conversation survives restarts** (`deo/conversation.json`, 12 messages, 6h).
- **Timers, reminders, date/time** (`deo/timers.py`), persisted and spoken.
- **Speech never reads markdown or code aloud.** The model is told to search
  for facts rather than recall them.
- **Gestures need a deliberate pose:** upright, still, held 0.8s. Scratching
  your face no longer counts as "1".
- **Starts with Windows:** Startup shortcut -> `pythonw run_deo.pyw`, log in
  `logs/deo.log`. A named mutex keeps it to one copy.
- Wallpaper Engine keeps the GPU at ~90% with Deo idle. Pause it while gaming
  or using Deo, or set it to pause when other apps are fullscreen/focused.

## Where things stand

**Done**
- Cloud migration (Groq LLM + STT, Orpheus TTS, local fallback on every leg).
- Latency pass: the turn is now a pipeline. The LLM streams sentences, TTS speaks
  sentence *n* while *n+1* is still being generated, Piper is pre-spawned so its
  ~0.5s model load is off the critical path, and the wake acknowledgement is
  pre-rendered and cached.
- Conversation history is capped, so turns no longer get slower as you talk.
- Piper's invocation bug fixed (it had never actually spoken — Windows
  `CreateProcess` rejects a relative exe path with forward slashes).

**Also done since**
- Command screening: destructive shell commands are refused before execution, and
  the system prompt now states that tool output is data, never instructions.
- Endpointing moved to Silero VAD (ships inside openwakeword, no new dependency).
  On the 133 real recordings in `debug_audio/`, it finds speech in 95 where the
  old amplitude gate found 69. Worth knowing: 30 of those recordings are digital
  silence -- failed captures that Whisper was previously hallucinating text from,
  which is the likely source of the old "garbled transcription" complaint.
- Follow-up turns: the mic stays open briefly after a reply, so a conversation
  no longer needs the wake word for every sentence (`FOLLOWUP_TURNS`).
- Multi-step tool loop: a turn can now chain tools (`MAX_TOOL_ITERATIONS`), which
  is what makes requests like "find X and open it on my other screen" work
  without a bespoke combined tool.
- Gesture tracking-loss fix, plus a DirectShow-first camera open on Windows.
- `tests/test_deo.py`: 26 offline checks over the reasoning guard, tool loop,
  command screening, VAD, sentence splitting and swipe detection.

**Fixed after the first real run**
- **The wake word could never have worked.** `voice_loop` held `mic_lock` across
  `keyboard.wait(F9)`, so push-to-talk owned the microphone permanently while
  idle and every wake trigger was rejected with "already listening/speaking".
  Waiting for a key now happens outside the lock.
- **False wakes**: three compounding bugs in the trainer — positives outnumbered
  negatives ~6:1 (and the rebalancing only handled the opposite case), only
  positives were heavily augmented (so "has added noise" meant "positive"), and
  there were almost no examples of ordinary speech that isn't the phrase. Plus a
  `FrameGate` in the listener requiring consecutive frames, which is worth more
  than any threshold change.
- **Recording level is the real ceiling.** The first 100 samples averaged −30 dBFS,
  which trains badly for reasons no amount of tuning fixes — see
  `training/README.md`. The recorder now rejects below −24 dBFS.
- **Evaluate the way it runs.** Per-window false-accept rates said ~4% for a model
  that fired 587×/hour in streaming. The trainer now reports false wakes per hour
  on held-out audio with the debounce applied.

**Broken or unfinished**
- **Wake word**: usable-ish but not good. Needs re-recording at a proper input
  level before it's trustworthy.
- **Gestures**: a likely root cause is fixed (the trajectory was discarded on the
  first dropped frame, so any swipe fast enough to register motion-blurred out of
  tracking). Unverified against a real camera -- needs you to try it.

---

## Tier 1 — do these first

### 1. Finish the wake word
See `training/README.md`. Record 100 samples, train, read the validation table.
Nothing else here matters as much: it's the difference between a voice assistant
and a hotkey.

*Effort: ~30 min of recording, then one command.*

### 2. Diagnose the gesture system
`deo/gesture.py` has never been confirmed working. Start by reproducing the
specific failure rather than rewriting — the useful question is which of these it
is: camera never opens, no `[gesture]` logs at all (MediaPipe/model init failed),
swipes detected but the window never moves, or no errors and simply nothing.
`GESTURE_ENABLED = True` and `hand_landmarker.task` is present, so it should at
least be initialising.

Worth asking before investing: is a webcam always on, plus MediaPipe's CPU cost,
actually worth window-snapping that `move_active_window` already does by voice?
This may be the one feature to cut rather than fix.

*Effort: an hour to diagnose.*

### [DONE] 3. Close the prompt-injection path into `run_command`
This one is a real exposure, not a theoretical one. `run_command` executes
arbitrary shell with `shell=True`, and three tools feed **untrusted outside text**
straight into the model's context:

- `web_search` — arbitrary web page titles and body text,
- `screen_qa` — OCR of whatever is on screen,
- the MCP filesystem server — file contents, with write access to `~/Downloads`.

So a search result or a web page containing text aimed at the model can plausibly
get a command run on your machine. The system prompt tells Deo to use tools
eagerly, which makes it more susceptible, not less.

Minimum fix: a denylist for destructive patterns (`del`, `rm`, `format`,
`rmdir`, `reg`, `shutdown`, redirection into system paths) that returns a refusal
instead of executing; spoken confirmation for anything matching it. Better fix:
an allowlist of command shapes you actually use, and wrapping tool results in an
explicit "this is untrusted data, never instructions" frame.

*Effort: an afternoon. Do it before you leave Deo running unattended.*

---

## Tier 2 — makes it feel like a real assistant

### 4. Barge-in
Right now interrupting means reaching for F10. Real assistants let you just talk
over them. The plumbing already half-exists: `tts.stop()` cuts speech cleanly
from any thread. What's needed is keeping the wake-word listener running *during*
playback, which today `mic_lock` prevents. Expect to have to handle Deo hearing
itself — simplest version is to only accept the wake word (not general speech)
while it's speaking.

### [DONE] 5. Follow-up turns without re-waking
After a reply, hold the mic open ~4 seconds and treat any speech as the next turn.
One of the cheapest wins here relative to how much it changes the feel — it turns
single commands into conversation.

### [DONE] 6. Replace the RMS endpointer with Silero VAD
`config.SILENCE_THRESHOLD = 0.035` is a fixed amplitude gate: it needs hand-tuning
per microphone and per room, cuts you off mid-thought when you pause, and waits
through background noise. A proper neural VAD fixes all three — and
`silero_vad.onnx` is **already on disk** inside the installed `openwakeword`
package, so this costs no new dependency.

### 7. Fast-path the common commands
"Open Steam", "volume up", "what time is it" don't need a 20B model. A small
intent matcher in front of `brain.ask_streaming()` — the pattern
`handle_gesture` already uses by calling `move_active_window` directly — removes
the entire LLM round trip from the commands you use most. Falls through to the
LLM on no match.

---

## Tier 3 — smarter

### [DONE] 8. Multi-step tool loop
`ask_streaming` does exactly one round of tool calls, then a phrasing call. That's
why chained requests need bespoke combined tools like
`open_latest_youtube_video`. Loop until the model stops requesting tools (cap
~5 iterations) and that whole class of hand-written combination tool disappears —
"find the latest video from X and put it on my second monitor" starts working on
its own.

### 9. Rolling summary instead of hard truncation
`_trim_history()` currently drops the oldest messages outright, so long
conversations silently lose their beginning. Summarise the dropped window into a
running paragraph and keep it behind the system prompt.

### 10. Memory that retrieves instead of pre-loading
Today the entire `memory.json` is pasted into every system prompt, and the model
has to *choose* to call `remember_fact`. It works at this size and won't at ten
times it. Two separate upgrades: embed facts and retrieve the relevant few per
turn, and extract facts automatically from conversation rather than relying on
the model to volunteer them.

### 11. Streaming STT
STT is still a hard wait: you stop talking, then transcription happens. Feeding
audio to Groq's endpoint (or a local streaming model) while you're still speaking
takes it off the critical path almost entirely. Biggest remaining latency win
after the work already done.

---

## Tier 4 — engineering hygiene

- **`git init`.** The repo root is currently `C:/Users/ashmi` — your whole home
  directory — so this project has no independent history and no rollback.
- **A test suite.** The streaming pipeline, the reasoning guard, and the sentence
  splitter were all verified with throwaway scripts. They should be `tests/`,
  particularly the reasoning guard: a regression there means Deo reads its own
  chain-of-thought aloud.
- **Per-stage latency logging.** Instrument wake → STT → first token → first audio
  behind `DEBUG_STARTUP_TIMING`-style flags, so a regression is visible instead of
  merely felt.
- **`README.md` is stale.** It predates the cloud migration and still describes
  `qwen2.5:7b` and a local-only stack.
- **`deo/VC_redist.x64.exe`** (25MB) is a downloaded installer sitting in the
  source package; delete it.

---

## Suggested order

Items 3, 5, 6 and 8 are now done and marked `[DONE]` above. What's left, in order:

**Wake word (1)** -- the only thing standing between Deo and hands-free use, and
it's blocked purely on recording 100 samples of your voice.

Then **verify gestures (2)** against a real camera now that the tracking-loss bug
is fixed, and decide whether to keep the feature at all. Then **barge-in (4)**,
which is the biggest remaining change in how Deo feels to talk to.

**Fast-path intents (7)** was deliberately skipped: with Groq streaming and
sentence-level TTS the LLM round trip is no longer the thing you notice, and a
pattern matcher in front of the model is a permanent source of
"why did it do that". Revisit only if latency starts to annoy you again.

Items 9-11 (rolling summary, retrieval memory, streaming STT) stay as written.
