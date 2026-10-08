# Deo — Local Voice Assistant

## Audit summary: what was actually fixed

Since the prior codebase ("jarvis") was the working base, here's exactly what changed
in this overhaul, mapped to the problems you described:

1. **STT accuracy / debug mode** — migrated to `faster-whisper` large-v3-turbo
   (from a smaller model). Added `DEBUG_SAVE_AUDIO` in `config.py`: every captured
   utterance is now written to `debug_audio/utterance_<timestamp>.wav` *before*
   transcription. Listen to those files — if they sound clean but transcription is
   still wrong, it's a model/prompt issue; if they sound garbled, it's mic/VAD tuning.

2. **App opening failures** — `open_app()` in `tools.py` no longer calls
   `os.startfile(bare_name)`. It now tries, in order: (a) a protocol handler from
   `APP_PROTOCOL_REGISTRY` (e.g. `steam://open/main`), (b) a full path from
   `APP_PATH_REGISTRY`, (c) a `shutil.which()` PATH lookup. If all three fail, it
   returns a real failure with the reason instead of silently doing nothing.
   **You need to fill in real paths** in `config.py` for apps installed in
   non-default locations — the ones there are placeholders/common defaults.

3. **Assumed success → real grounding** — every function in `tools.py` now returns
   `{"success": bool, "message": str}`. `brain.py` feeds that *exact* structured
   result back to the LLM as the tool result (not a paraphrase), and the system
   prompt explicitly instructs the model to never claim success the tool didn't
   report. If `open_app` fails, Deo will now say so instead of "Opened it!"

4. **Tool-calling layer** — `brain.py` uses Ollama's native `tools=` parameter
   with the `TOOL_SCHEMAS` list from `tools.py`. The model decides which function
   to call; `Brain._run_tool()` executes it and validates the result shape before
   trusting it (if a tool returns something malformed, that itself becomes a
   reported failure, not a silent assumption).

5. **YouTube chained command** — `get_latest_youtube_video()` resolves a channel
   name → channel ID → uploads playlist → most recent video, via 3 sequential
   YouTube Data API calls. `open_latest_youtube_video()` chains that with
   `open_url_in_browser()` so "open the latest video from X on Brave" resolves
   end-to-end. Needs `YOUTUBE_API_KEY` set in a local `.env` file (see below).

## Setup

### 1. Python deps
```
pip install -r requirements.txt
```

### 2. Ollama model
```
ollama pull qwen2.5:7b-instruct-q4_K_M
```
(7B chosen specifically so it shares your 8GB VRAM comfortably with Whisper.)

### 3. Piper TTS (not a pip package — separate download)
1. Download the Windows Piper release from
   https://github.com/rhasspy/piper/releases (grab the `.zip`, e.g. `piper_windows_amd64.zip`)
2. Extract it into a `piper/` folder inside this project, so you have `piper/piper.exe`
3. Download a voice model (e.g. `en_US-amy-medium`) from
   https://github.com/rhasspy/piper/blob/master/VOICES.md — you need both the
   `.onnx` file and its matching `.onnx.json` file, both placed in `piper/`
4. Confirm `config.py`'s `PIPER_EXE_PATH` and `PIPER_VOICE_MODEL` point at the right files

### 4. YouTube API key
1. Get a key from Google Cloud Console (enable "YouTube Data API v3")
2. Copy `.env.example` to `.env` in the project root
3. Paste your key in: `YOUTUBE_API_KEY=your_actual_key`

### 5. Fill in your real app paths
Open `deo/config.py` → `APP_PATH_REGISTRY` and `APP_PROTOCOL_REGISTRY`, update paths
to match what's actually installed on your machine (check Steam/Brave/etc install
locations — right-click the shortcut → Properties → Target).

### 6. Run it
```
python main.py
```
Say "Deo" (or a close mishear like "Theo"/"Dio") followed by your request.

## Debugging audio issues
Every utterance gets saved to `debug_audio/`. After a failed/garbled transcription,
open the most recent `.wav` there and listen — confirms whether the problem is
capture quality (mic, levels, clipping) or the Whisper model itself.

## Notes / things to verify on your end
- `WHISPER_DEVICE = "cuda"` in config.py assumes your CUDA/cuDNN setup works for
  large-v3-turbo. If you hit a missing-DLL error like before, switch to `"cpu"` +
  `compute_type="int8"` temporarily while we sort the CUDA runtime.
- `APP_PATH_REGISTRY` and `APP_PROTOCOL_REGISTRY` entries are realistic defaults
  but **not verified against your actual install paths** — confirm/correct these.
- Protocol handlers (`steam://`, `discord://`) require those apps to already be
  installed and registered with Windows, which normally happens automatically
  on install.
