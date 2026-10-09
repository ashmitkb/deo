"""
Make sure the local Ollama server is running before Deo talks to it.

Ollama starts at login from its own Startup shortcut, but if it's quit from
its tray icon, crashes, or loses the race at boot, every turn failed with
"[WinError 10061] ... actively refused it" until someone restarted it by hand.
ensure_running() is a ~1ms port check, so it's cheap enough to call before
every model request; only when nothing answers does it launch Ollama and wait.
"""

import os
import socket
import subprocess
import threading
import time
from urllib.parse import urlparse

from . import config

_lock = threading.Lock()


def _address():
    url = urlparse(config.OLLAMA_HOST)
    return url.hostname or "localhost", url.port or 11434


def is_up(timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection(_address(), timeout=timeout):
            return True
    except OSError:
        return False


def _launch():
    base = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama")
    tray_app = os.path.join(base, "ollama app.exe")
    if os.path.isfile(tray_app):
        # The tray app is what the Startup shortcut runs: it starts the server
        # and puts the icon back, exactly as if Ollama had started normally.
        subprocess.Popen([tray_app], cwd=base, close_fds=True)
        return
    exe = os.path.join(base, "ollama.exe")
    subprocess.Popen([exe if os.path.isfile(exe) else "ollama", "serve"],
                     creationflags=subprocess.CREATE_NO_WINDOW, close_fds=True)


def ensure_running(wait_s: float = 30.0) -> bool:
    """True once Ollama accepts connections (launching it if needed)."""
    if is_up():
        return True
    with _lock:                       # one launch, even if several callers race
        if is_up():
            return True
        print("[ollama] not running -- starting it")
        try:
            _launch()
        except Exception as e:
            print(f"[ollama] couldn't start Ollama: {e}")
            return False
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            time.sleep(0.5)
            if is_up():
                print("[ollama] back up")
                return True
        print(f"[ollama] still not answering after {wait_s:.0f}s")
        return False


class EnsuringClient:
    """Wraps ollama.Client so every call first makes sure the server is up."""

    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            ensure_running()
            return attr(*args, **kwargs)
        return call
