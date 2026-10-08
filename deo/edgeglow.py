"""
Siri-style screen-edge glow: a soft band of flowing colour around the edges of
the main screen while Deo listens, thinks and speaks.

Four thin layered windows (top, bottom, left, right) rather than one full-
screen one -- each strip is cheap to render, and nothing covers the middle of
the screen. Like the island they are click-through (WS_EX_TRANSPARENT), never
take focus (WS_EX_NOACTIVATE) and are excluded from screen capture, so the
island's glass never "sees" them.

Thread-safe API: post(state) with "listening" / "thinking" / "speaking" to
show, "idle" / "off" to fade out, "quit" to close.
"""

import ctypes
import math
import queue
import threading
import time

import numpy as np
from PIL import Image

from .island import Spring, _push_frame, _rgb, _user32

PALETTES = {
    "listening": ["#c6ff3d", "#34f5c5", "#9dff00", "#7dffb0", "#38bdf8"],
    "thinking":  ["#ffb020", "#ff7a1a", "#ffe14d", "#ff5fa2", "#ffb020"],
    "speaking":  ["#a78bfa", "#ff5fa2", "#38bdf8", "#c6ff3d", "#a78bfa"],
}
THICKNESS = 64          # logical px of glow at the very edge, fading inward


class EdgeGlow:
    def __init__(self):
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="deo-edgeglow")

    def start(self):
        self._thread.start()
        return self

    def post(self, state):
        self._q.put(state)

    # ---------------------------------------------------------------- thread
    def _setup(self):
        import win32api
        import win32con
        import win32gui
        _user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
        mon = win32api.MonitorFromPoint((0, 0), 1)
        info = win32api.GetMonitorInfo(mon)
        dpi = ctypes.c_uint()
        ctypes.windll.shcore.GetDpiForMonitor(int(mon), 0, ctypes.byref(dpi), ctypes.byref(ctypes.c_uint()))
        s = dpi.value / 96.0
        l, t, r, b = info["Monitor"]
        self.mw, self.mh, self.ml, self.mt = r - l, b - t, l, t
        self.T = int(THICKNESS * s)

        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = lambda h, m, w, p: win32gui.DefWindowProc(h, m, w, p)
        wc.lpszClassName = "DeoEdgeGlow"
        wc.hInstance = win32api.GetModuleHandle(None)
        try:
            atom = win32gui.RegisterClass(wc)
        except Exception:
            atom = "DeoEdgeGlow"
        ex = (win32con.WS_EX_LAYERED | win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW
              | win32con.WS_EX_TRANSPARENT | 0x08000000)
        T = self.T
        # (name, x, y, w, h)
        self.strips = {
            "top": (l, t, self.mw, T), "bottom": (l, b - T, self.mw, T),
            "left": (l, t, T, self.mh), "right": (r - T, t, T, self.mh),
        }
        self.hwnds = {}
        for name, (x, y, w, h) in self.strips.items():
            hwnd = win32gui.CreateWindowEx(ex, atom, "Deo Glow", win32con.WS_POPUP, x, y, w, h,
                                           0, 0, wc.hInstance, None)
            _user32.SetWindowDisplayAffinity(hwnd, 0x11)       # WDA_EXCLUDEFROMCAPTURE
            self.hwnds[name] = hwnd
        # Fade masks: strongest at the screen edge, eased to nothing inward.
        ramp = np.linspace(1.0, 0.0, T) ** 2.2
        self.fade_h = ramp[:, None]                            # top strip: rows fade downward
        self.fade_v = ramp[None, :]                            # left strip: columns fade rightward

    def _run(self):
        import win32gui
        self._setup()
        self.intensity = Spring(0.0, 60, 0.95)
        self.state, shown, t0, last = "off", False, time.monotonic(), time.monotonic()
        self.colours = PALETTES["listening"]
        while True:
            win32gui.PumpWaitingMessages()
            try:
                while True:
                    st = self._q.get_nowait()
                    if st == "quit":
                        for h in self.hwnds.values():
                            win32gui.DestroyWindow(h)
                        return
                    self.state = st
                    if st in PALETTES:
                        self.colours = PALETTES[st]
                        self.intensity.target = 1.0
                    else:
                        self.intensity.target = 0.0
            except queue.Empty:
                pass
            now = time.monotonic()
            self.intensity.step(min(now - last, 0.05))
            last = now
            if self.intensity.value > 0.01 or self.intensity.target > 0:
                if not shown:
                    for h in self.hwnds.values():
                        win32gui.ShowWindow(h, 4)                # SW_SHOWNOACTIVATE
                    shown = True
                try:
                    self._render(now - t0)
                except Exception as e:
                    if not getattr(self, "_logged", False):
                        print(f"[edgeglow] frame failed: {e!r}")
                        self._logged = True
                time.sleep(1 / 40)
            else:
                if shown:
                    for h in self.hwnds.values():
                        win32gui.ShowWindow(h, 0)
                    shown = False
                time.sleep(0.05)

    def _colour_line(self, n, t, offset):
        """n colours flowing along one edge."""
        cols = np.array([_rgb(c) for c in self.colours], dtype=np.float32)
        k = len(cols)
        speed = {"speaking": 0.22, "listening": 0.14, "thinking": 0.3}.get(self.state, 0.1)
        pos = (np.linspace(0, 1, n) * 1.6 + t * speed + offset) % 1.0 * k
        i = pos.astype(int)
        f = (pos - i)[:, None]
        f = f * f * (3 - 2 * f)                                  # smoothstep between stops
        return cols[i % k] * (1 - f) + cols[(i + 1) % k] * f

    def _render(self, t):
        level = self.intensity.value
        # breathing; faster and deeper while Deo is speaking
        if self.state == "speaking":
            pulse = 0.72 + 0.28 * (0.5 + 0.5 * math.sin(t * 7.0)) * (0.6 + 0.4 * math.sin(t * 2.3))
        elif self.state == "thinking":
            pulse = 0.65 + 0.2 * math.sin(t * 4.0)
        else:
            pulse = 0.8 + 0.2 * math.sin(t * 2.0)
        a_max = 210 * level * pulse
        T = self.T
        # perimeter offsets so colour flows continuously round the corners
        per = 2 * (self.mw + self.mh)
        for name, (x, y, w, h) in self.strips.items():
            horizontal = name in ("top", "bottom")
            n = max(2, (w if horizontal else h) // 8)
            start = {"top": 0, "right": self.mw, "bottom": self.mw + self.mh, "left": 2 * self.mw + self.mh}[name]
            line = self._colour_line(n, t, start / per * 1.6)
            if name in ("bottom", "left"):
                line = line[::-1]
            if horizontal:
                rgb = np.repeat(line[None, :, :], T, axis=0)                        # (T, n, 3)
                fade = self.fade_h if name == "top" else self.fade_h[::-1]
                alpha = np.repeat(fade, n, axis=1) * a_max
            else:
                rgb = np.repeat(line[:, None, :], T, axis=1)                        # (n, T, 3)
                fade = self.fade_v if name == "left" else self.fade_v[:, ::-1]
                alpha = np.repeat(fade, n, axis=0) * a_max
            small = np.dstack([rgb, alpha[..., None]]).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(small, "RGBA").resize((w, h), Image.BILINEAR)
            _push_frame(self.hwnds[name], img, x, y)
