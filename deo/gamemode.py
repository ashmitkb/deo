"""
Pick Deo's local brain by what the GPU is doing.

Not gaming -> config.OLLAMA_MODEL (qwen3.5:9b): smarter, ~35-40 tok/s here,
              and got facts right that the 4B got wrong ("India has over 200
              million people").
Gaming     -> config.OLLAMA_MODEL_GAMING (qwen3:4b): ~2.5GB instead of ~6.3GB,
              so the game keeps the 8GB GPU. The moment a game starts, the 9B
              is unloaded from VRAM, without waiting for Deo's next turn.

"Gaming" means a process is running from a game library folder (Steam,
Epic, EA...) -- not Steam itself, launchers, or Wallpaper Engine, which lives
in steamapps too and runs all day.
"""

import os
import threading
import time

from . import config

_LIBRARY_MARKERS = ("\\steamapps\\common\\", "\\epic games\\", "\\ea games\\",
                    "\\ubisoft game launcher\\games\\", "\\gog galaxy\\games\\",
                    "\\xboxgames\\")
# Things that live in those folders but aren't games you're playing.
_NOT_GAMES = ("wallpaper_engine", "wallpaper32", "wallpaper64", "launcher", "epicgameslauncher",
              "directxredist", "redist", "steamworks shared", "crashreport", "easyanticheat",
              "unrealcefsubprocess", "ue4prereqsetup", "vc_redist", "dxsetup")

_cache = (0.0, False, "")
_lock = threading.Lock()


def _running_exes() -> list:
    import ctypes
    from ctypes import wintypes
    from .tools import _process_exe
    pids = (wintypes.DWORD * 4096)()
    needed = wintypes.DWORD()
    if not ctypes.windll.psapi.EnumProcesses(pids, ctypes.sizeof(pids), ctypes.byref(needed)):
        return []
    count = needed.value // ctypes.sizeof(wintypes.DWORD)
    return [exe for exe in (_process_exe(pids[i]) for i in range(count)) if exe]


def running_game() -> str:
    """Path of a running game, or "" if none. Cached for a few seconds."""
    global _cache
    with _lock:
        checked, _, game = _cache
        if time.monotonic() - checked < 5:
            return game
        game = ""
        extra = [g.lower() for g in getattr(config, "GAME_EXE_NAMES", [])]
        for exe in _running_exes():
            low = exe.lower()
            name = os.path.basename(low)
            if name in extra or (any(m in low for m in _LIBRARY_MARKERS)
                                 and not any(n in low for n in _NOT_GAMES)):
                game = exe
                break
        _cache = (time.monotonic(), bool(game), game)
        return game


def current_model() -> str:
    if not getattr(config, "AUTO_GAME_MODE", True):
        return config.OLLAMA_MODEL
    return config.OLLAMA_MODEL_GAMING if running_game() else config.OLLAMA_MODEL


def _unload(model: str):
    try:
        import ollama
        ollama.Client(host=config.OLLAMA_HOST).generate(model=model, prompt="", keep_alive=0)
    except Exception as e:
        print(f"[gamemode] couldn't unload {model}: {e}")


def _watch():
    """Free the GPU as soon as a game starts; drop the 4B again after."""
    last = None
    while True:
        model = current_model()
        if model != last:
            if last is not None:
                game = running_game()
                print(f"[gamemode] {'game running: ' + os.path.basename(game) if game else 'no game running'}"
                      f" -- switching brain to {model}")
                _unload(last)
            last = model
        time.sleep(10)


def start():
    threading.Thread(target=_watch, daemon=True, name="deo-gamemode").start()
