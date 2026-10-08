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
    WS_EX_TRANSPARENT -- clicks go straight through to whatever is below
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
    "listening": ("#2dd4bf", ["#2dd4bf", "#38bdf8", "#22d3ee", "#a7f3d0"]),
    "thinking":  ("#f5b94d", ["#f5b94d", "#fb923c", "#facc15", "#f472b6"]),
    "speaking":  ("#8b7cf8", ["#8b7cf8", "#c084fc", "#60a5fa", "#f472b6"]),
    "idle":      ("#8b90a2", ["#6b7280", "#8b7cf8", "#38bdf8", "#6b7280"]),
    "notice":    ("#38bdf8", ["#38bdf8", "#8b7cf8", "#2dd4bf", "#60a5fa"]),
}
LABELS = {"listening": "Listening…", "thinking": "Thinking…", "speaking": "", "idle": ""}

CANVAS_W, CANVAS_H = 760, 300          # logical px; window is this big, mostly transparent
TOP_MARGIN = 14
COMPACT = (230, 54)
MAX_TEXT_W = 560
MAX_REPLY_LINES = 3


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
    def __init__(self):
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
        wc.lpfnWndProc = lambda h, m, w, l: win32gui.DefWindowProc(h, m, w, l)
        wc.lpszClassName = "DeoIsland"
        wc.hInstance = win32api.GetModuleHandle(None)
        try:
            atom = win32gui.RegisterClass(wc)
        except Exception:
            atom = "DeoIsland"
        ex = (win32con.WS_EX_LAYERED | win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW
              | win32con.WS_EX_TRANSPARENT | 0x08000000)          # WS_EX_NOACTIVATE
        self.hwnd = win32gui.CreateWindowEx(ex, atom, "Deo Island", win32con.WS_POPUP,
                                            self.win_x, self.win_y, self.W, self.H, 0, 0, wc.hInstance, None)

        f = lambda name, size: ImageFont.truetype(name, int(size * self.s))
        try:
            self.font_text = f("segoeui.ttf", 15)
            self.font_label = f("seguisb.ttf", 14)
            self.font_small = f("segoeui.ttf", 11.5)
        except OSError:
            self.font_text = self.font_label = self.font_small = ImageFont.load_default()

    def _run(self):
        import win32gui
        self._setup()
        s = self.s
        self.y = Spring(-60 * s, 210, 0.66)
        self.w = Spring(40 * s, 190, 0.58)
        self.h = Spring(30 * s, 190, 0.6)
        self.alpha = Spring(0.0, 140, 0.9)
        self.state, self.user_text, self.reply_text, self.notice = "idle", "", "", ""
        self.visible = False
        self.hide_at = None
        self.t0 = time.monotonic()
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
            if self.hide_at and now >= self.hide_at:
                self._retract()
            dt = min(now - last, 0.05)
            last = now
            if self.visible or not self.alpha.settled():
                for sp in (self.y, self.w, self.h, self.alpha):
                    sp.step(dt)
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
            self.visible = True
            s = self.s
            self.y.value, self.w.value, self.h.value = -50 * s, 44 * s, 30 * s
            win32gui.ShowWindow(self.hwnd, 4)                   # SW_SHOWNOACTIVATE
        self._retarget()

    def _retract(self):
        self.hide_at = None
        self.visible = False
        s = self.s
        self.y.target, self.w.target, self.h.target, self.alpha.target = -70 * s, 50 * s, 32 * s, 0.0

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
        compact_only = not (self.notice or self.reply_text or self.user_text)
        if compact_only:
            tw = self.font_label.getlength(lines[0])
            w, h = max(COMPACT[0] * s, tw + 90 * s), COMPACT[1] * s
        else:
            font = self.font_text
            tw = max([font.getlength(l) for l in lines] + [self.font_small.getlength(label) if label else 0])
            line_h = 21 * s
            h = 26 * s + line_h * len(lines) + (17 * s if label else 0)
            w = min(max(tw + 88 * s, 280 * s), (MAX_TEXT_W + 88) * s)
        self.y.target, self.w.target, self.h.target, self.alpha.target = TOP_MARGIN * s, w, h, 1.0

    # ---- rendering ------------------------------------------------------------------
    def _draw(self, now):
        s, W, H = self.s, self.W, self.H
        t = now - self.t0
        w, h, y = max(self.w.value, 20 * s), max(self.h.value, 20 * s), self.y.value
        a = max(0.0, min(1.0, self.alpha.value))
        x0 = (W - w) / 2
        box = (x0, y, x0 + w, y + h)
        radius = min(h / 2, 28 * s)
        frame = Image.new("RGBA", (W, H), (0, 0, 0, 0))

        # soft drop shadow (blurred anyway, so half resolution is plenty)
        sw, sh = W // 2, H // 2
        sm = Image.new("L", (sw, sh), 0)
        ImageDraw.Draw(sm).rounded_rectangle([c / 2 for c in (box[0], box[1] + 8 * s, box[2], box[3] + 8 * s)],
                                             radius / 2, fill=int(150 * a))
        sm = sm.filter(ImageFilter.GaussianBlur(9 * s)).resize((W, H), Image.BILINEAR)
        frame.paste((0, 0, 0, 255), (0, 0), sm)

        # the glass body -- anti-aliased edge
        body = _aa_rrect((W, H), box, radius)
        frame.paste((16, 17, 24, int(232 * a)), (0, 0), body)

        # flowing liquid rim: soft colour blobs, blurred, seen only near the edge
        core, colours = PALETTES.get(self.state, PALETTES["idle"])
        bw, bh = max(8, W // 6), max(8, H // 6)
        blobs = Image.new("RGB", (bw, bh), (0, 0, 0))
        d = ImageDraw.Draw(blobs)
        cx, cy, rw, rh = (box[0] + box[2]) / 12, (box[1] + box[3]) / 12, w / 12, h / 12
        speed = 1.6 if self.state in ("speaking", "listening") else 0.9
        # Blobs travel AROUND the edge (not across the middle), so the rim is
        # lit all the way round and the colour flows along it.
        r = max(rh * 0.9, min(rw, rh) * 1.1, 3)
        n = len(colours) * 2
        for i in range(n):
            col = colours[i % len(colours)]
            ph = t * speed * 0.55 + i * (2 * math.pi / n) + math.sin(t * 0.7 + i) * 0.25
            bx = cx + math.cos(ph) * rw * 0.92
            by = cy + math.sin(ph) * rh * 0.92
            d.ellipse([bx - r, by - r, bx + r, by + r], fill=_rgb(col))
        blobs = blobs.filter(ImageFilter.GaussianBlur(max(2, bw / 30))).resize((W, H), Image.BILINEAR)

        rim_w = 2.0 * s
        inset = _aa_rrect((W, H), (box[0] + rim_w, box[1] + rim_w, box[2] - rim_w, box[3] - rim_w),
                          max(1, radius - rim_w))
        ring = ImageChops.subtract(body, inset)                 # crisp, anti-aliased rim
        inner = _aa_rrect((W, H), (box[0] + 11 * s, box[1] + 11 * s, box[2] - 11 * s, box[3] - 11 * s),
                          max(1, radius - 11 * s))
        glow = ImageChops.subtract(body, inner).filter(ImageFilter.GaussianBlur(6 * s))
        pulse = 0.75 + 0.25 * math.sin(t * (6 if self.state == "speaking" else 2.4))
        frame.paste(blobs, (0, 0), glow.point(lambda v: int(v * 0.32 * a * pulse)))
        frame.paste(blobs, (0, 0), ring.point(lambda v: int(v * 0.95 * a)))

        # top sheen: a smooth vertical fade inside the glass, no hard edge
        grad = Image.linear_gradient("L").resize((1, 256)).transpose(Image.FLIP_TOP_BOTTOM)
        grad = grad.point(lambda v: int(v * v / 255 * 0.09 * a)).resize((W, max(1, int(h * 0.6))))
        sheen = Image.new("L", (W, H), 0)
        sheen.paste(grad, (0, int(y)))
        frame.paste((255, 255, 255, 255), (0, 0), ImageChops.multiply(sheen, inset))

        # content (only once the pill is big enough to hold it)
        if w > 120 * s and h > 40 * s and a > 0.3:
            dr = ImageDraw.Draw(frame)
            ta = int(255 * min(1.0, (a - 0.3) / 0.5))
            ocx, ocy = box[0] + 28 * s, box[1] + min(h / 2, 27 * s)
            orb_r = (8 + (2 * pulse if self.state != "idle" else 0)) * s
            halo = Image.new("L", (W, H), 0)
            ImageDraw.Draw(halo).ellipse((ocx - orb_r * 2, ocy - orb_r * 2, ocx + orb_r * 2, ocy + orb_r * 2),
                                         fill=int(120 * a))
            frame.paste(_rgb(core) + (255,), (0, 0), halo.filter(ImageFilter.GaussianBlur(5 * s)))
            orb = _aa_ellipse((W, H), (ocx - orb_r, ocy - orb_r, ocx + orb_r, ocy + orb_r))
            frame.paste(_rgb(core) + (255,), (0, 0), orb.point(lambda v: v * ta // 255))
            shine = _aa_ellipse((W, H), (ocx - orb_r * .55, ocy - orb_r * .7, ocx - orb_r * .05, ocy - orb_r * .2))
            frame.paste((255, 255, 255, 255), (0, 0), shine.point(lambda v: int(v * ta * .45 / 255)))

            label, lines = self._content()
            tx = box[0] + 50 * s
            compact_only = not (self.notice or self.reply_text or self.user_text)
            if compact_only:
                bbox = self.font_label.getbbox(lines[0])
                dr.text((tx, ocy - (bbox[1] + bbox[3]) / 2), lines[0], font=self.font_label,
                        fill=(238, 240, 246, ta))
            else:
                ty = box[1] + 13 * s
                if label:
                    dr.text((tx, ty), label, font=self.font_small, fill=(150, 156, 172, ta))
                    ty += 17 * s
                for line in lines:
                    dr.text((tx, ty), line, font=self.font_text, fill=(238, 240, 246, ta))
                    ty += 21 * s

        return frame
