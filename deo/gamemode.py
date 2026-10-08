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
                    "\\xboxgames\\",
                    # Standalone installs like C:\Games\Red Dead Redemption 2 --
                    # missed until this was added, while RDR2 was running.
                    "\\games\\")
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


# Fullscreen apps that are NOT games.
_FULLSCREEN_OK = {"brave.exe", "chrome.exe", "msedge.exe", "firefox.exe", "opera.exe", "arc.exe",
                  "vlc.exe", "mpc-hc64.exe", "wmplayer.exe", "media.player.exe", "powerpnt.exe",
                  "code.exe", "pycharm64.exe", "explorer.exe", "applicationframehost.exe",
                  "obs64.exe", "resolve.exe", "python.exe", "pythonw.exe",
                  # system overlays that cover the screen for a moment -- the
                  # Snipping Tool was mistaken for a game the first time it ran
                  "snippingtool.exe", "screenclippinghost.exe", "magnify.exe", "lockapp.exe",
                  "searchhost.exe", "startmenuexperiencehost.exe", "shellexperiencehost.exe",
                  "textinputhost.exe", "gamebar.exe", "nvidia overlay.exe", "claude.exe"}


def _fullscreen_app() -> str:
    """Exe of the foreground window if it covers its whole monitor and isn't
    a browser/video/editor -- i.e. almost certainly a game, wherever it's
    installed."""
    try:
        import win32api
        import win32gui
        import win32process
        from .tools import _process_exe, _per_monitor_dpi
        with _per_monitor_dpi():
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd or win32gui.GetClassName(hwnd) in ("Progman", "WorkerW", "Shell_TrayWnd"):
                return ""
            rect = win32gui.GetWindowRect(hwnd)
            mon = win32api.GetMonitorInfo(win32api.MonitorFromWindow(hwnd, 2))["Monitor"]
        if rect[0] > mon[0] or rect[1] > mon[1] or rect[2] < mon[2] or rect[3] < mon[3]:
            return ""
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        exe = _process_exe(pid)
        return "" if not exe or os.path.basename(exe).lower() in _FULLSCREEN_OK else exe
    except Exception:
        return ""


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
        if not game:
            game = _fullscreen_app()
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
