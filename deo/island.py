"""
The "island": a Siri / Dynamic-Island style glass pill that drops from the top
of the main screen while you talk to Deo.

    listening   a small pill springs down from the top edge, teal rim
    you spoke   it stretches to show what you said
    thinking    amber rim
    speaking    violet rim; Deo's reply appears as it's spoken
    done        a few seconds later it retracts back up

Long replies are handed off to the full Deo window instead (see gui.py).

HOW IT'S DRAWN
Every frame is rendered with PIL (anti-aliased shapes, real blur for the
shadow and the flowing coloured rim) and pushed to a Win32 *layered* window
with per-pixel alpha via UpdateLayeredWindow. The window is:
    WS_EX_NOACTIVATE  -- never takes keyboard focus (safe over games)
    per-pixel alpha   -- clicks pass through every transparent pixel; only
                         the island itself can be grabbed (pull it down to
                         open the Deo window)
    WS_EX_TOPMOST / TOOLWINDOW -- on top, no taskbar button
It runs on its own thread with its own message pump, and is per-monitor DPI
aware so it's crisp on a 200% screen.

Thread-safe API: post(kind, payload) -- see Island.post.
"""

import ctypes
import ctypes.wintypes as wt
import math
import queue
import random
import threading
import time

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

# palette per state: (core, rim colours...)
PALETTES = {
    # same colours as the Deo window (deo/webui/style.css)
    "listening": ("#c6ff3d", ["#c6ff3d", "#34f5c5", "#9dff00", "#7dffb0"]),
    "thinking":  ("#ffb020", ["#ffb020", "#ff7a1a", "#ffe14d", "#ff5fa2"]),
    "speaking":  ("#a78bfa", ["#a78bfa", "#ff5fa2", "#38bdf8", "#c6ff3d"]),
    "idle":      ("#c6ff3d", ["#c6ff3d", "#7dffb0", "#38bdf8", "#a78bfa"]),
    "notice":    ("#c6ff3d", ["#c6ff3d", "#38bdf8", "#a78bfa", "#7dffb0"]),
}
LABELS = {"listening": "Listening…", "thinking": "Thinking…", "speaking": "", "idle": ""}

CANVAS_W, CANVAS_H = 760, 300          # logical px; window is this big, mostly transparent
TOP_MARGIN = 14
COMPACT = (230, 54)
BLOB_COMPACT = (78, 54)    # listening/thinking with nothing to read: just the blob

# The blob's moods -- the same numbers as the app's blob (deo/webui/app.js)
BLOB_MOODS = {
    "idle":      (0.045, 0.55, 0.10, 0.55),     # wobble, speed, spin, colour
    "listening": (0.090, 1.50, 0.35, 1.00),
    "thinking":  (0.070, 2.10, 1.30, 0.85),
    "speaking":  (0.060, 2.80, 0.50, 1.00),
    "notice":    (0.050, 0.90, 0.20, 0.80),
}
_BLOB_N = 9
_BLOB_SEEDS = [(1 + (i * 7) % 5 * 0.31, (i * 2.399) % 6.283) for i in range(_BLOB_N)]
MAX_TEXT_W = 560
MAX_REPLY_LINES = 3
PULL_OPEN = 90            # logical px you have to drag the island down to open Deo


def _smooth_lut(lo, hi):
    out = []
    for v in range(256):
        x = min(1.0, max(0.0, (v - lo) / (hi - lo)))
        out.append(int(255 * x * x * (3 - 2 * x)))
    return out


# Threshold levels on the blurred "liquid" mask (see Island._liquid_masks).
_LUT_BODY = _smooth_lut(112, 146)
_LUT_INSET = _smooth_lut(162, 196)


def _rgb(hex_):
    return tuple(int(hex_[i:i + 2], 16) for i in (1, 3, 5))


class Spring:
    """Damped spring -- underdamped a touch, which is where the 'liquid'
    wobble comes from when the pill grows or shrinks."""

    def __init__(self, value, stiffness=190.0, damping_ratio=0.62):
        self.value = self.target = float(value)
        self.velocity = 0.0
        self.k = stiffness
        self.c = 2 * math.sqrt(stiffness) * damping_ratio

    def step(self, dt):
        a = self.k * (self.target - self.value) - self.c * self.velocity
        self.velocity += a * dt
        self.value += self.velocity * dt

    def settled(self):
        return abs(self.target - self.value) < 0.3 and abs(self.velocity) < 0.5


# ---------------------------------------------------------------- win32 glue
class _BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_ubyte), ("BlendFlags", ctypes.c_ubyte),
                ("SourceConstantAlpha", ctypes.c_ubyte), ("AlphaFormat", ctypes.c_ubyte)]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG),
                ("biPlanes", wt.WORD), ("biBitCount", wt.WORD), ("biCompression", wt.DWORD),
                ("biSizeImage", wt.DWORD), ("biXPelsPerMeter", wt.LONG), ("biYPelsPerMeter", wt.LONG),
                ("biClrUsed", wt.DWORD), ("biClrImportant", wt.DWORD)]


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
# Every handle is pointer-sized. Without explicit types ctypes passes and
# returns 32-bit ints, and the first real frame died with "OverflowError: int
# too long to convert" on a 64-bit DC handle.
_user32.GetDC.argtypes, _user32.GetDC.restype = [wt.HWND], wt.HDC
_user32.ReleaseDC.argtypes, _user32.ReleaseDC.restype = [wt.HWND, wt.HDC], ctypes.c_int
_user32.UpdateLayeredWindow.argtypes = [wt.HWND, wt.HDC, ctypes.POINTER(wt.POINT), ctypes.POINTER(wt.SIZE),
                                        wt.HDC, ctypes.POINTER(wt.POINT), wt.COLORREF,
                                        ctypes.c_void_p, wt.DWORD]
_user32.UpdateLayeredWindow.restype = wt.BOOL
_gdi32.CreateCompatibleDC.argtypes, _gdi32.CreateCompatibleDC.restype = [wt.HDC], wt.HDC
_gdi32.CreateDIBSection.argtypes = [wt.HDC, ctypes.c_void_p, wt.UINT, ctypes.POINTER(ctypes.c_void_p),
                                    wt.HANDLE, wt.DWORD]
_gdi32.CreateDIBSection.restype = wt.HBITMAP
_gdi32.SelectObject.argtypes, _gdi32.SelectObject.restype = [wt.HDC, wt.HGDIOBJ], wt.HGDIOBJ
_gdi32.DeleteObject.argtypes, _gdi32.DeleteObject.restype = [wt.HGDIOBJ], wt.BOOL
_gdi32.DeleteDC.argtypes, _gdi32.DeleteDC.restype = [wt.HDC], wt.BOOL


def _push_frame(hwnd, img, x, y):
    """Hand a premultiplied-BGRA frame to the layered window."""
    import numpy as np
    w, h = img.size
    a = np.asarray(img, dtype=np.uint16)
    alpha = a[..., 3:4]
    bgra = np.empty((h, w, 4), dtype=np.uint8)
    bgra[..., 0] = (a[..., 2] * alpha[..., 0] // 255)
    bgra[..., 1] = (a[..., 1] * alpha[..., 0] // 255)
    bgra[..., 2] = (a[..., 0] * alpha[..., 0] // 255)
    bgra[..., 3] = alpha[..., 0]
    data = bgra.tobytes()

    screen_dc = _user32.GetDC(0)
    mem_dc = _gdi32.CreateCompatibleDC(screen_dc)
    bmi = _BITMAPINFOHEADER(ctypes.sizeof(_BITMAPINFOHEADER), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
    bits = ctypes.c_void_p()
    hbmp = _gdi32.CreateDIBSection(mem_dc, ctypes.byref(bmi), 0, ctypes.byref(bits), None, 0)
    ctypes.memmove(bits, data, len(data))
    old = _gdi32.SelectObject(mem_dc, hbmp)
    blend = _BLENDFUNCTION(0, 0, 255, 1)                 # AC_SRC_OVER, AC_SRC_ALPHA
    _user32.UpdateLayeredWindow(hwnd, screen_dc, ctypes.byref(wt.POINT(x, y)),
                                ctypes.byref(wt.SIZE(w, h)), mem_dc, ctypes.byref(wt.POINT(0, 0)),
                                0, ctypes.byref(blend), 2)  # ULW_ALPHA
    _gdi32.SelectObject(mem_dc, old)
    _gdi32.DeleteObject(hbmp)
    _gdi32.DeleteDC(mem_dc)
    _user32.ReleaseDC(0, screen_dc)


# ---------------------------------------------------------------- anti-aliasing
# PIL draws shapes WITHOUT anti-aliasing -- the first live island had visibly
# stair-stepped edges. Each shape is drawn at SUPERSAMPLE x size, only over its
# own bounding box, and box-filtered back down: the same smooth edges text has.
SUPERSAMPLE = 4


def _aa_mask(size, bbox, draw):
    W, H = size
    if bbox[2] - bbox[0] < 1 or bbox[3] - bbox[1] < 1:
        # An inset shape on a pill that's still tiny (mid drop-in, or the
        # spring overshooting small on retract) comes out inverted -- PIL
        # raised "x1 must be greater than or equal to x0". Nothing to draw.
        return Image.new("L", (W, H), 0)
    x1, y1 = math.floor(bbox[0]) - 2, math.floor(bbox[1]) - 2
    x2, y2 = math.ceil(bbox[2]) + 2, math.ceil(bbox[3]) + 2
    rw, rh = max(1, x2 - x1), max(1, y2 - y1)
    big = Image.new("L", (rw * SUPERSAMPLE, rh * SUPERSAMPLE), 0)
    draw(ImageDraw.Draw(big), lambda v, o: (v - o) * SUPERSAMPLE, x1, y1)
    small = big.resize((rw, rh), Image.BOX)
    mask = Image.new("L", (W, H), 0)
    mask.paste(small, (x1, y1))
    return mask


def _soft(mask, radius):
    """Gaussian blur at half resolution -- glows don't need the detail, and
    the full-size blurs were most of the frame time."""
    w, h = mask.size
    small = mask.resize((max(1, w // 2), max(1, h // 2)), Image.BILINEAR)
    return small.filter(ImageFilter.GaussianBlur(radius / 2)).resize((w, h), Image.BILINEAR)


def _aa_rrect(size, box, radius):
    def draw(d, sc, ox, oy):
        r = max(0, min(radius, (box[2] - box[0]) / 2, (box[3] - box[1]) / 2))
        d.rounded_rectangle((sc(box[0], ox), sc(box[1], oy), sc(box[2], ox), sc(box[3], oy)),
                            r * SUPERSAMPLE, fill=255)
    return _aa_mask(size, box, draw)


def _aa_ellipse(size, box):
    def draw(d, sc, ox, oy):
        d.ellipse((sc(box[0], ox), sc(box[1], oy), sc(box[2], ox), sc(box[3], oy)), fill=255)
    return _aa_mask(size, box, draw)


# ---------------------------------------------------------------- the island
class Island:
    def __init__(self, on_open=None):
        """on_open: called (on a new thread) when the island is pulled down far
        enough to open the full Deo window."""
        self.on_open = on_open
        self.dragging = False
        self.drag_raw = 0.0
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="deo-island")
        self.ready = threading.Event()

    def start(self):
        self._thread.start()
        return self

    # ---- thread-safe API ----------------------------------------------------
    def post(self, kind, payload=None):
        """kind: 'status' (listening/thinking/speaking/idle), 'user' (text),
        'reply' (text so far), 'notice' (short message), 'hide', 'quit'."""
        self._q.put((kind, payload))

    # ---- island thread --------------------------------------------------------
    def _setup(self):
        import win32api
        import win32con
        import win32gui
        _user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))   # per-monitor v2
        mon = win32api.MonitorFromPoint((0, 0), 1)                    # primary monitor
        info = win32api.GetMonitorInfo(mon)
        dpi_x, dpi_y = ctypes.c_uint(), ctypes.c_uint()
        ctypes.windll.shcore.GetDpiForMonitor(int(mon), 0, ctypes.byref(dpi_x), ctypes.byref(dpi_y))
        self.s = dpi_x.value / 96.0
        left, top, right, _ = info["Monitor"]
        self.W, self.H = int(CANVAS_W * self.s), int(CANVAS_H * self.s)
        self.win_x = left + (right - left - self.W) // 2
        self.win_y = top

        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = self._wndproc
        wc.lpszClassName = "DeoIsland"
        wc.hInstance = win32api.GetModuleHandle(None)
        try:
            atom = win32gui.RegisterClass(wc)
        except Exception:
            atom = "DeoIsland"
        # No WS_EX_TRANSPARENT any more: a layered window already lets clicks
        # through its fully transparent pixels, so only the island itself is
        # grabbable. WS_EX_NOACTIVATE keeps grabbing it from stealing focus.
        ex = (win32con.WS_EX_LAYERED | win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW
              | 0x08000000)                                       # WS_EX_NOACTIVATE
        self.hwnd = win32gui.CreateWindowEx(ex, atom, "Deo Island", win32con.WS_POPUP,
                                            self.win_x, self.win_y, self.W, self.H, 0, 0, wc.hInstance, None)
        # Invisible to screen capture, so the glass can sample what's BEHIND it
        # without seeing itself (Windows 10 2004+).
        _user32.SetWindowDisplayAffinity(self.hwnd, 0x11)        # WDA_EXCLUDEFROMCAPTURE
        self._backdrop, self._backdrop_at = None, 0.0

        import os
        # Geist: clean and modern, close in spirit to Apple's San Francisco
        # (which can't be bundled). Regular and medium weights -- not bold.
        fonts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui", "fonts")

        def f(size, weight):
            font = ImageFont.truetype(os.path.join(fonts, "Geist.ttf"), int(size * self.s))
            try:
                font.set_variation_by_axes([weight])
            except Exception:
                pass
            return font
        # Animation state lives here (not in _run) so the renderer is usable as
        # soon as the window exists -- tests drive it without the thread.
        s = self.s
        self.y = Spring(-60 * s, 210, 0.66)
        self.w = Spring(40 * s, 190, 0.58)
        self.h = Spring(30 * s, 190, 0.6)
        self.alpha = Spring(0.0, 140, 0.9)
        self.pull = Spring(0.0, 230, 0.42)              # how far it's being pulled; wobbly
        self.state, self.user_text, self.reply_text, self.notice = "idle", "", "", ""
        self.visible = False
        self.hide_at = None
        self.t0 = time.monotonic()

        try:
            self.font_text = f(15.5, 400)
            self.font_label = f(14.5, 500)
            self.font_small = f(12, 400)
        except OSError:
            self.font_text = self.font_label = self.font_small = ImageFont.load_default()

    def _run(self):
        import win32gui
        self._setup()
        threading.Thread(target=self._backdrop_loop, daemon=True, name="deo-island-glass").start()
        self.ready.set()
        last = time.monotonic()
        while True:
            win32gui.PumpWaitingMessages()
            try:
                while True:
                    kind, payload = self._q.get_nowait()
                    if kind == "quit":
                        win32gui.DestroyWindow(self.hwnd)
                        return
                    self._handle(kind, payload)
            except queue.Empty:
                pass
            now = time.monotonic()
            if self.hide_at and now >= self.hide_at and not self.dragging:
                self._retract()
            dt = min(now - last, 0.05)
            last = now
            if self.visible or not self.alpha.settled() or not self.pull.settled():
                for sp in (self.y, self.w, self.h, self.alpha):
                    sp.step(dt)
                if not self.dragging:
                    self.pull.step(dt)
                try:
                    _push_frame(self.hwnd, self._draw(now), self.win_x, self.win_y)
                except Exception as e:
                    # One bad frame must not kill the island for the session.
                    if not getattr(self, "_frame_error_logged", False):
                        print(f"[island] frame failed: {e!r}")
                        self._frame_error_logged = True
                if not self.visible and self.alpha.value < 0.02 and self.y.settled():
                    win32gui.ShowWindow(self.hwnd, 0)          # SW_HIDE
                time.sleep(1 / 60)
            else:
                time.sleep(0.03)

    # ---- events -----------------------------------------------------------------
    def _handle(self, kind, payload):
        if kind == "status":
            if payload in ("listening", "thinking", "speaking"):
                if payload == "listening" and self.state not in ("listening",):
                    if not self.visible or self.state == "idle":
                        self.user_text, self.reply_text, self.notice = "", "", ""
                self.state = payload
                self._appear()
            elif payload == "idle" and self.visible:
                self.state = "idle"
                self.hide_at = time.monotonic() + (4.0 if (self.reply_text or self.notice) else 0.8)
        elif kind == "user":
            self.user_text, self.reply_text, self.notice = payload or "", "", ""
            self._appear()
        elif kind == "reply":
            self.reply_text = payload or ""
            self._appear()
        elif kind == "notice":
            self.notice, self.user_text, self.reply_text = payload or "", "", ""
            self.state = "notice"
            self._appear()
            self.hide_at = time.monotonic() + 3.5
        elif kind == "hide":
            self._retract()

    def _appear(self):
        import win32gui
        self.hide_at = None
        if not self.visible:
            try:
                self._capture_backdrop()            # fresh glass before the first frame
            except Exception:
                pass
            self.visible = True
            self._appear_t = time.monotonic()
            s = self.s
            self.y.value, self.w.value, self.h.value = -50 * s, 44 * s, 30 * s
            win32gui.ShowWindow(self.hwnd, 4)                   # SW_SHOWNOACTIVATE
        self._retarget()

    def _retract(self):
        self.hide_at = None
        self.visible = False
        s = self.s
        self.y.target, self.w.target, self.h.target, self.alpha.target = -70 * s, 50 * s, 32 * s, 0.0

    # ---- pull to open ---------------------------------------------------------------
    def _wndproc(self, hwnd, msg, wparam, lparam):
        import win32con
        import win32gui
        if msg == 0x0021:                                   # WM_MOUSEACTIVATE
            return 3                                        # MA_NOACTIVATE: never take focus
        if msg == win32con.WM_SETCURSOR:
            win32gui.SetCursor(win32gui.LoadCursor(0, win32con.IDC_HAND))
            return 1
        if msg == win32con.WM_LBUTTONDOWN:
            self._drag_begin()
            return 0
        if msg == win32con.WM_MOUSEMOVE and self.dragging:
            self._drag_move()
            return 0
        if msg in (win32con.WM_LBUTTONUP, 0x0215) and self.dragging:   # 0x215 WM_CAPTURECHANGED
            self._drag_end()
            return 0
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    def _drag_begin(self):
        import win32api
        import win32gui
        self.dragging, self.drag_raw = True, 0.0
        self._drag_y0 = win32api.GetCursorPos()[1]
        self.hide_at = None
        self.pull.velocity = 0.0
        win32gui.SetCapture(self.hwnd)        # keep getting moves when the cursor leaves it

    def _drag_move(self):
        import win32api
        raw = max(0.0, win32api.GetCursorPos()[1] - self._drag_y0)
        self.drag_raw = raw
        # rubber band: follows the cursor at first, then resists more and more
        self.pull.value = self.pull.target = raw * 0.62 / (1 + raw / (520 * self.s))

    def _drag_end(self):
        import win32gui
        self.dragging = False                 # before ReleaseCapture: it re-enters via WM_CAPTURECHANGED
        win32gui.ReleaseCapture()
        if self.drag_raw >= PULL_OPEN * self.s and self.on_open:
            # stretch on through, then drain away while the Deo window opens
            self.pull.target = self.pull.value + 90 * self.s
            self.hide_at = time.monotonic() + 0.22
            threading.Thread(target=self.on_open, daemon=True).start()
        else:
            self.pull.target = 0.0            # spring back up, with a wobble
            if self.state in ("idle", "notice"):
                self.hide_at = time.monotonic() + 2.5

    def _liquid_masks(self, box, radius, pull, now):
        """(body, inset) masks for the island as LIQUID -- the metaball trick:
        draw the shapes, blur them together, cut back to a crisp edge, and
        they merge like water.

          dropping in : the pill hangs from the top edge of the screen on a
                        neck that thins and snaps ~0.4s after it appears
          (being pulled moves the whole island down -- see _draw)
        """
        s, W, H = self.s, self.W, self.H
        base = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(base)
        d.rounded_rectangle(box, radius, fill=255)
        cx = (box[0] + box[2]) / 2
        bw = box[2] - box[0]

        drip = max(0.0, 1 - (now - getattr(self, "_appear_t", 0.0)) / 0.42) if self.visible else 0.0
        neck = drip * min(bw * 0.34, 52 * s)
        if neck >= 3 * s:
            d.rounded_rectangle((cx - neck / 2, -60 * s, cx + neck / 2, box[1] + min(radius, 22 * s)),
                                neck / 2, fill=255)
            rw = neck * 2.6                                   # the reservoir at the screen edge
            d.ellipse((cx - rw / 2, -rw * 0.55, cx + rw / 2, rw * 0.22), fill=255)

        field = _soft(base, 7 * s)
        return field.point(_LUT_BODY), field.point(_LUT_INSET)

    # ---- layout -------------------------------------------------------------------
    def _wrap(self, text, font, width, max_lines):
        words, lines, cur = text.split(), [], ""
        for word in words:
            trial = f"{cur} {word}".strip()
            if font.getlength(trial) <= width:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = word
            if len(lines) == max_lines:
                break
        if cur and len(lines) < max_lines:
            lines.append(cur)
        if len(lines) == max_lines and " ".join(lines) != " ".join(words):
            last = lines[-1]
            while last and font.getlength(last + " …") > width:
                last = last.rsplit(" ", 1)[0] if " " in last else last[:-1]
            lines[-1] = last + " …"
        return lines

    def _content(self):
        """(label line or None, body lines) for the current state."""
        s = self.s
        width = MAX_TEXT_W * s
        if self.dragging:
            ready = self.drag_raw >= PULL_OPEN * s
            return None, ["Release to open Deo" if ready else "Pull down to open Deo"]
        if self.notice:
            return None, self._wrap(self.notice, self.font_text, width, 2)
        if self.reply_text:
            label = self.user_text and self._wrap(self.user_text, self.font_small, width, 1)[0]
            return label or None, self._wrap(self.reply_text, self.font_text, width, MAX_REPLY_LINES)
        if self.user_text:
            return None, self._wrap(self.user_text, self.font_text, width, 2)
        return None, [LABELS.get(self.state, "") or "Deo"]

    def _retarget(self):
        s = self.s
        label, lines = self._content()
        compact_only = not (self.notice or self.reply_text or self.user_text or self.dragging)
        if compact_only:
            w, h = BLOB_COMPACT[0] * s, BLOB_COMPACT[1] * s
        else:
            font = self.font_text
            tw = max([font.getlength(l) for l in lines] + [self.font_small.getlength(label) if label else 0])
            line_h = 21 * s
            h = 26 * s + line_h * len(lines) + (17 * s if label else 0)
            w = min(max(tw + 88 * s, 280 * s), (MAX_TEXT_W + 88) * s)
        self.y.target, self.w.target, self.h.target, self.alpha.target = TOP_MARGIN * s, w, h, 1.0

    # ---- rendering ------------------------------------------------------------------
    def _capture_backdrop(self):
        from PIL import ImageEnhance, ImageGrab
        import numpy as np
        W, H = self.W, self.H
        # Only the island's own strip of the primary monitor (all_screens=False:
        # coordinates are relative to the primary, which the island lives on).
        # Captured on its own thread (_backdrop_loop): grabbing both monitors
        # took 45ms a frame and dragged the animation to ~24fps.
        shot = ImageGrab.grab(bbox=(self.win_x, self.win_y, self.win_x + W, self.win_y + H)).convert("RGB")
        # Liquid glass is CLEAR, not frosted: just a touch of softening and a
        # little extra saturation. The bending happens in _draw.
        soft = shot.resize((max(1, W // 2), max(1, H // 2)), Image.BILINEAR)
        soft = soft.filter(ImageFilter.GaussianBlur(1.2)).resize((W, H), Image.BICUBIC)
        soft = ImageEnhance.Color(soft).enhance(1.25)
        self._backdrop = np.asarray(soft, dtype=np.float32)

    def _backdrop_loop(self):
        _user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
        while True:
            if self.visible:
                try:
                    self._capture_backdrop()
                except Exception:
                    pass
                time.sleep(0.12)
            else:
                time.sleep(0.2)

    def _blob(self, d, now):
        """The living blob, d px across, as an RGBA image: an outline that's a
        smooth curve through wobbling points, frosted white glass with soft
        iridescent colour swirling inside, and a glass highlight."""
        import numpy as np
        mood = BLOB_MOODS.get(self.state, BLOB_MOODS["idle"])
        cur = getattr(self, "_blob_cur", list(mood))
        dt = min(0.05, now - getattr(self, "_blob_last", now))
        self._blob_last = now
        cur = [c + (m - c) * min(1.0, dt * 3) for c, m in zip(cur, mood)]
        self._blob_cur = cur
        amp, speed, spin_rate, sat = cur
        self._blob_t = getattr(self, "_blob_t", 0.0) + dt * speed
        self._blob_spin = getattr(self, "_blob_spin", 0.0) + dt * spin_rate
        want = (0.35 + random.random() * 0.65) if self.state == "speaking" else 0.0
        self._blob_level = getattr(self, "_blob_level", 0.0)
        self._blob_level += (want - self._blob_level) * min(1.0, dt * 10)
        t, spin, level = self._blob_t, self._blob_spin, self._blob_level

        SS = 3
        size = int(d * SS)
        c = size / 2
        R = size * 0.38 * (1 + level * 0.06)
        pts = []
        for i in range(_BLOB_N):
            a = i / _BLOB_N * 2 * math.pi + spin * 0.3
            f, ph = _BLOB_SEEDS[i]
            wob = math.sin(t * f + ph) * 0.6 + math.sin(t * f * 1.7 + ph * 2.1) * 0.4
            r = R * (1 + (amp + level * 0.07) * wob)
            pts.append((c + math.cos(a) * r, c + math.sin(a) * r))
        # smooth closed outline: Catmull-Rom through the points
        outline = []
        for i in range(_BLOB_N):
            p0, p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % _BLOB_N], pts[(i + 2) % _BLOB_N]
            for k in range(10):
                u = k / 10
                u2, u3 = u * u, u * u * u
                outline.append(tuple(0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * u
                                            + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * u2
                                            + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * u3) for j in (0, 1)))
        mask_big = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask_big).polygon(outline, fill=255)
        mask = mask_big.resize((d, d), Image.BOX)

        yy, xx = np.mgrid[0:d, 0:d].astype(np.float32)
        cc, rr = d / 2, R / SS
        # frosted white base, brighter toward the top-left
        dist = np.hypot(xx - (cc - rr * 0.35), yy - (cc - rr * 0.4)) / (rr * 1.25)
        base = np.clip(1 - dist, 0, 1)[..., None]
        rgb = np.array([200, 205, 220], np.float32) * (1 - base) + 245 * base
        # iridescent colour, multiplied in
        layer = Image.new("RGB", (d, d), (255, 255, 255))
        dl = ImageDraw.Draw(layer)
        for col, off in (((165, 180, 252), 0.0), ((125, 211, 252), 2.1), ((249, 168, 212), 4.2)):
            ang = spin * 1.4 + off + math.sin(t * 0.7 + off) * 0.6
            bx, by = cc + math.cos(ang) * rr * 0.55, cc + math.sin(ang) * rr * 0.55
            rad = rr * 0.62
            mixed = tuple(int(255 - (255 - v) * sat) for v in col)
            dl.ellipse((bx - rad, by - rad, bx + rad, by + rad), fill=mixed)
        layer = np.asarray(layer.filter(ImageFilter.GaussianBlur(max(1.0, d / 9))), np.float32)
        rgb = rgb * layer / 255
        # glass highlight and a soft shade
        hi = np.clip(1 - np.hypot(xx - (cc - rr * 0.42), yy - (cc - rr * 0.5)) / (rr * 0.6), 0, 1)[..., None]
        rgb = rgb * (1 - hi * 0.85) + 255 * hi * 0.85
        lo = np.clip(1 - np.hypot(xx - (cc + rr * 0.5), yy - (cc + rr * 0.6)) / rr, 0, 1)[..., None]
        rgb = rgb * (1 - lo * 0.28) + np.array([40, 40, 60], np.float32) * lo * 0.28
        img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8), "RGB").convert("RGBA")
        img.putalpha(mask)
        # crisp white rim
        rim = ImageChops.subtract(mask, mask.filter(ImageFilter.MinFilter(3)))
        img.paste((255, 255, 255, 255), (0, 0), rim.point(lambda v: int(v * 0.55)))
        return img

    def _draw(self, now):
        """Apple-style liquid glass: clear, refracting at the edges, lit by a
        white specular rim. No colour of its own -- only what's behind it."""
        import numpy as np
        s, W, H = self.s, self.W, self.H
        t = now - self.t0
        w, h, y = max(self.w.value, 20 * s), max(self.h.value, 20 * s), self.y.value
        a = max(0.0, min(1.0, self.alpha.value))
        radius = min(h / 2, 28 * s)
        # squash & stretch: a moving drop gets taller and thinner, then squashes
        st = max(-0.16, min(0.3, self.y.velocity / (1500 * s)))
        h, w = h * (1 + st), w * (1 - st * 0.45)
        # being pulled: the whole island follows the cursor down, stretching a
        # little (a hanging droplet was tried and didn't look right)
        pull = max(0.0, self.pull.value)
        y += pull
        h *= 1 + min(0.12, pull / (800 * s))
        w *= 1 - min(0.05, pull / (1600 * s))
        x0 = (W - w) / 2
        box = (x0, y, x0 + w, y + h)
        frame = Image.new("RGBA", (W, H), (0, 0, 0, 0))

        # soft floating shadow
        sw, sh = W // 2, H // 2
        sm = Image.new("L", (sw, sh), 0)
        ImageDraw.Draw(sm).rounded_rectangle(
            [c / 2 for c in (box[0] + 6 * s, box[1] + 10 * s, box[2] - 6 * s, box[3] + 10 * s)],
            radius / 2, fill=int(110 * a))
        frame.paste((0, 0, 0, 255), (0, 0),
                    sm.filter(ImageFilter.GaussianBlur(10 * s)).resize((W, H), Image.BILINEAR))

        body, inset1 = self._liquid_masks(box, radius, pull, now)
        bb = body.getbbox() or (0, 0, 1, 1)
        rx1, ry1, rx2, ry2 = max(0, bb[0] - 2), max(0, bb[1] - 2), min(W, bb[2] + 2), min(H, bb[3] + 2)
        if rx2 - rx1 > 4 and ry2 - ry1 > 4:
            bg = self._backdrop if self._backdrop is not None else np.full((H, W, 3), 30.0, np.float32)
            # Refraction: blur the pill's outline; its gradient points inward near
            # the rim. Sampling the background a little further inward there
            # bends and magnifies it, like light through a thick lens.
            m = np.asarray(_soft(body, max(4.0, min(h, w) * 0.42)), dtype=np.float32)[ry1:ry2, rx1:rx2] / 255.0
            gy, gx = np.gradient(m)
            depth = min(h, 90 * s) * 1.6
            yy, xx = np.mgrid[ry1:ry2, rx1:rx2].astype(np.float32)
            out = np.empty((ry2 - ry1, rx2 - rx1, 3), np.float32)
            for ch, k in ((0, 1.06), (1, 1.0), (2, 0.94)):   # slight per-channel split = glassy fringe
                sx = np.clip(xx + gx * depth * k * 2.2, 0, W - 1)
                sy = np.clip(yy + gy * depth * k, 0, H - 1)
                x0i, y0i = sx.astype(np.int32), sy.astype(np.int32)
                x1i, y1i = np.minimum(x0i + 1, W - 1), np.minimum(y0i + 1, H - 1)
                fx, fy = sx - x0i, sy - y0i
                c = bg[..., ch]
                out[..., ch] = ((c[y0i, x0i] * (1 - fx) + c[y0i, x1i] * fx) * (1 - fy)
                                + (c[y1i, x0i] * (1 - fx) + c[y1i, x1i] * fx) * fy)
            # a whisper of tint, adapting to what's behind (like Apple's glass):
            # over something bright, dim more so the white text stays readable
            lum = float(out.mean())
            dim = 0.74 if lum < 120 else max(0.42, 0.74 - (lum - 120) / 135 * 0.32)
            out = out * dim + 255 * 0.05
            glass = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            glass.paste(Image.fromarray(out.clip(0, 255).astype(np.uint8), "RGB"), (rx1, ry1))
            frame.paste(glass, (0, 0), body.point(lambda v: int(v * a)))

        # specular light: bright rim at the top fading down the sides
        rim = ImageChops.subtract(body, inset1)
        vert = Image.linear_gradient("L").resize((1, 256))                       # 0 top -> 255 bottom
        vert = vert.point(lambda v: int(255 - v * 0.62)).resize((W, max(1, int(h))))
        light = Image.new("L", (W, H), 0)
        light.paste(vert, (0, int(y)))
        frame.paste((255, 255, 255, 255), (0, 0),
                    ImageChops.multiply(rim, light).point(lambda v: int(v * 0.85 * a)))
        # soft sheen across the top of the glass
        inner = _aa_rrect((W, H), (box[0] + 3 * s, box[1] + 3 * s, box[2] - 3 * s, box[3] - 3 * s),
                          max(1, radius - 3 * s))
        top = Image.linear_gradient("L").transpose(Image.FLIP_TOP_BOTTOM).resize((W, max(1, int(h * 0.42))))
        sheen = Image.new("L", (W, H), 0)
        sheen.paste(top.point(lambda v: int(v * v / 255 * 0.16)), (0, int(y)))
        frame.paste((255, 255, 255, 255), (0, 0), ImageChops.multiply(sheen, inner).point(lambda v: int(v * a)))
        # faint reflection along the inside of the bottom edge
        bot = _aa_rrect((W, H), (box[0] + radius * 0.6, box[3] - 3.2 * s, box[2] - radius * 0.6, box[3] - 2.2 * s), 1 * s)
        frame.paste((255, 255, 255, 255), (0, 0), bot.point(lambda v: int(v * 0.22 * a)))

        # content: a white glyph for the state, then white text
        if w > 50 * s and h > 40 * s and a > 0.3:
            ta = int(255 * min(1.0, (a - 0.3) / 0.5))
            compact_only = not (self.notice or self.reply_text or self.user_text or self.dragging)
            bd = int((36 if compact_only else 30) * s)
            blob = self._blob(bd, now)
            if ta < 255:
                blob.putalpha(blob.getchannel("A").point(lambda v: v * ta // 255))
            ocx = (box[0] + box[2]) / 2 if compact_only else box[0] + 28 * s
            ocy = box[1] + min(h / 2, 27 * s)
            frame.alpha_composite(blob, (int(ocx - bd / 2), int(ocy - bd / 2)))

            dr = ImageDraw.Draw(frame)

            def text(xy, txt, font, alpha):
                dr.text((xy[0], xy[1] + 1 * s), txt, font=font, fill=(0, 0, 0, int(alpha * 0.35)))
                dr.text(xy, txt, font=font, fill=(255, 255, 255, int(alpha)))

            label, lines = self._content()
            tx = box[0] + 50 * s
            if compact_only:
                pass                                  # just the blob -- no "Listening..." text
            else:
                ty = box[1] + 13 * s
                if label:
                    text((tx, ty), label, self.font_small, ta * 0.62)
                    ty += 17 * s
                for line in lines:
                    text((tx, ty), line, self.font_text, ta)
                    ty += 21 * s

        return frame
