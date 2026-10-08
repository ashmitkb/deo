"""
Deo's desktop window.

IMPORTANT THREADING RULE: Tkinter widgets may only be touched from the
thread running root.mainloop(). Everything else (voice loop, wake-word
listener, screen QA, timers, the tray icon) runs on background threads; they
must talk to this GUI ONLY through the thread-safe push_* / request_* methods,
which drop events onto a queue.Queue() drained by a root.after() timer.

Layout:
    header  -- animated orb (idle / listening / thinking / speaking), name,
               status, and an info line (brain model, paused, next timer)
    chat    -- rounded message bubbles on a scrollable canvas
    footer  -- pause / stop buttons, rounded input, send

Closing the window hides it (Deo keeps listening); quitting is done from the
tray icon, or the window closes for real if no tray is running.
"""

import ctypes
import math
import queue
import random
import threading
import time
import tkinter as tk
from tkinter import font as tkfont

# --- palette ------------------------------------------------------------
BG = "#0f1117"
SURFACE = "#171a23"
SURFACE_2 = "#1f2330"
BORDER = "#2a2f3d"
TEXT = "#eef0f6"
TEXT_DIM = "#8a90a2"
TEXT_FAINT = "#5c6273"
USER_BUBBLE = "#3b6cf6"
USER_TEXT = "#ffffff"
DEO_BUBBLE = "#1f2330"
DANGER = "#f0616d"

STATES = {
    #            core       glow        label
    "idle":      ("#4b5266", "#2b3040", "Idle -- say \"hey Deo\""),
    "listening": ("#2dd4bf", "#0f5f57", "Listening..."),
    "thinking":  ("#f5b94d", "#6b4d14", "Thinking..."),
    "speaking":  ("#8b7cf8", "#3b2f8c", "Speaking..."),
    "paused":    ("#3a3f4f", "#22262f", "Paused -- not listening"),
    "error":     (DANGER,    "#5c1d24", "Something went wrong"),
}

FONT_FAMILY = "Segoe UI Variable Text"


def _mix(c1: str, c2: str, t: float) -> str:
    """Blend two #rrggbb colours; t=0 -> c1, t=1 -> c2."""
    t = max(0.0, min(1.0, t))
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def _rounded_rect(canvas, x1, y1, x2, y2, r, **kw):
    r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
           x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return canvas.create_polygon(pts, smooth=True, **kw)


def make_icon_image(size=64, color="#8b7cf8"):
    """The orb as a PIL image -- shared by the window icon and the tray."""
    from PIL import Image, ImageDraw, ImageFilter
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(glow)
    c = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    d.ellipse([size * .12, size * .12, size * .88, size * .88], fill=c + (150,))
    glow = glow.filter(ImageFilter.GaussianBlur(size * .08))
    img.alpha_composite(glow)
    d = ImageDraw.Draw(img)
    d.ellipse([size * .24, size * .24, size * .76, size * .76], fill=c + (255,))
    d.ellipse([size * .34, size * .3, size * .5, size * .44], fill=(255, 255, 255, 90))
    return img


# --- the animated orb ---------------------------------------------------
class Orb(tk.Canvas):
    SIZE = 60

    def __init__(self, parent):
        super().__init__(parent, width=self.SIZE, height=self.SIZE, bg=SURFACE,
                         highlightthickness=0)
        self.state = "idle"
        self._t0 = time.monotonic()
        self._level = 0.0
        self._tick()

    def set_state(self, state):
        if state != self.state:
            self.state = state
            self._t0 = time.monotonic()

    def _tick(self):
        self.delete("all")
        core, glow, _ = STATES.get(self.state, STATES["idle"])
        t = time.monotonic() - self._t0
        cx = cy = self.SIZE / 2
        if self.state == "listening":
            for k in range(3):                       # ripples spreading out
                p = ((t * 0.9) + k / 3) % 1.0
                r = 12 + p * 17
                self.create_oval(cx - r, cy - r, cx + r, cy + r,
                                 outline=_mix(core, SURFACE, p), width=2)
            r = 12 + math.sin(t * 4) * 1.2
        elif self.state == "thinking":
            self.create_oval(cx - 22, cy - 22, cx + 22, cy + 22, outline=_mix(glow, SURFACE, .3), width=3)
            self.create_arc(cx - 22, cy - 22, cx + 22, cy + 22, start=-t * 300 % 360, extent=100,
                            style="arc", outline=core, width=3)
            r = 11
        elif self.state == "speaking":
            target = random.uniform(0.25, 1.0)        # a voice-like jitter
            self._level += (target - self._level) * 0.35
            g = 14 + self._level * 13
            self.create_oval(cx - g, cy - g, cx + g, cy + g, fill=_mix(glow, SURFACE, .45), outline="")
            r = 11 + self._level * 4
        else:                                         # idle / paused: slow breathing
            b = (math.sin(t * 1.4) + 1) / 2
            g = 16 + b * 5
            self.create_oval(cx - g, cy - g, cx + g, cy + g, fill=_mix(glow, SURFACE, .55 - b * .2), outline="")
            r = 11 + b * 1.5
        self.create_oval(cx - r, cy - r, cx + r, cy + r, fill=core, outline="")
        self.create_oval(cx - r * .55, cy - r * .7, cx - r * .05, cy - r * .2,
                         fill=_mix(core, "#ffffff", .45), outline="")
        self.after(33, self._tick)


# --- scrollable chat of bubbles -----------------------------------------
class ChatView(tk.Frame):
    MAX_MESSAGES = 200

    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self._messages = []          # (kind, who, text, stamp)
        self._width = 420
        self._body = tkfont.Font(family=FONT_FAMILY, size=11)
        self._meta = tkfont.Font(family=FONT_FAMILY, size=8)
        self._who = tkfont.Font(family=FONT_FAMILY, size=8, weight="bold")
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)

    def add(self, kind, who, text):
        if kind == "assistant" and self._messages and self._messages[-1][0] == "stream":
            self._messages[-1] = (kind, who, text, self._messages[-1][3])   # finalise the live bubble
            self._render()
            return
        if kind == "stream" and self._messages and self._messages[-1][0] == "stream":
            self._messages[-1] = (kind, who, text, self._messages[-1][3])
            self._render()
            return
        self._messages.append((kind, who, text, time.strftime("%H:%M")))
        self._messages = self._messages[-self.MAX_MESSAGES:]
        self._render()

    def _on_resize(self, event):
        if abs(event.width - self._width) > 2:
            self._width = event.width
            self._render()

    def _on_wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120) * 3, "units")

    def _render(self):
        c = self.canvas
        at_bottom = c.yview()[1] > 0.98 or not c.find_all()
        c.delete("all")
        w = max(self._width, 280)
        max_text = int(w * 0.72)
        y = 16
        for kind, who, text, stamp in self._messages:
            if kind == "system":
                t = c.create_text(w / 2, y + 12, text=text, fill=TEXT_DIM, font=self._meta,
                                  width=int(w * 0.8), justify="center")
                x1, y1, x2, y2 = c.bbox(t)
                pill = _rounded_rect(c, x1 - 12, y1 - 5, x2 + 12, y2 + 5, 12, fill=SURFACE, outline="")
                c.tag_raise(t)
                y = y2 + 18
                continue
            mine = kind == "user"
            pad_x, pad_y, margin = 14, 10, 16
            t = c.create_text(0, 0, text=text, font=self._body, width=max_text, anchor="nw",
                              fill=USER_TEXT if mine else TEXT)
            x1, y1, x2, y2 = c.bbox(t)
            bw, bh = (x2 - x1) + pad_x * 2, (y2 - y1) + pad_y * 2
            bx1 = w - margin - bw if mine else margin
            label = c.create_text(bx1 + (bw if mine else 0), y, text=f"{who}  ·  {stamp}",
                                  anchor="ne" if mine else "nw", font=self._meta, fill=TEXT_FAINT)
            ly2 = c.bbox(label)[3]
            by1 = ly2 + 4
            _rounded_rect(c, bx1, by1, bx1 + bw, by1 + bh, 16,
                          fill=USER_BUBBLE if mine else DEO_BUBBLE, outline="")
            c.coords(t, bx1 + pad_x, by1 + pad_y)
            c.tag_raise(t)
            y = by1 + bh + 14
        c.configure(scrollregion=(0, 0, w, y + 8))
        if at_bottom:
            c.yview_moveto(1.0)


# --- flat rounded button -------------------------------------------------
class PillButton(tk.Canvas):
    def __init__(self, parent, text, command, bg, fg, hover, width=None, height=36, font=None):
        self._font = font or tkfont.Font(family=FONT_FAMILY, size=10, weight="bold")
        w = width or self._font.measure(text) + 28
        super().__init__(parent, width=w, height=height, bg=parent["bg"], highlightthickness=0,
                         cursor="hand2")
        self._cmd, self._bg, self._fg, self._hover = command, bg, fg, hover
        self._text, self._bw, self._bh = text, w, height
        self._draw(bg)
        self.bind("<Enter>", lambda e: self._draw(self._hover))
        self.bind("<Leave>", lambda e: self._draw(self._bg))
        self.bind("<Button-1>", lambda e: self._cmd and self._cmd())

    def _draw(self, fill):
        self.delete("all")
        _rounded_rect(self, 1, 1, self._bw - 1, self._bh - 1, self._bh / 2, fill=fill, outline="")
        self.create_text(self._bw / 2, self._bh / 2, text=self._text, fill=self._fg, font=self._font)

    def set(self, text=None, bg=None, fg=None, hover=None):
        self._text = text or self._text
        self._bg, self._fg, self._hover = bg or self._bg, fg or self._fg, hover or self._hover
        self._draw(self._bg)


# --- island routing shared by every window implementation ----------------
class IslandRouting:
    """What goes to the island vs the window. Shared by the Tk window below
    and the web window (gui_web.py); both provide self._queue, self.island,
    self.is_gaming, self._window_shown, self._handed_off and request_show()."""

    # Replies longer than this, or with code / a list in them, open the full
    # window instead of squeezing into the island (unless a game is running).
    ISLAND_MAX_CHARS = 260

    def _to_island(self, kind, payload=None):
        if self.island is not None and not self._window_shown:
            self.island.post(kind, payload)

    @classmethod
    def _is_big(cls, text):
        return len(text) > cls.ISLAND_MAX_CHARS or "```" in text or text.count("\n") >= 3

    def push_user_message(self, text):
        self._queue.put(("user", text))
        self._handed_off = False
        self._to_island("user", text)

    def push_reply_chunk(self, text_so_far):
        """The reply as spoken so far -- streams into the window and the island."""
        self._queue.put(("stream", text_so_far))
        if self._handed_off or self.island is None or self._window_shown:
            return
        if self._is_big(text_so_far) and not self.is_gaming():
            self._handed_off = True
            self.island.post("notice", "Full answer in the Deo window")
            self.request_show(focus=False)
        else:
            self.island.post("reply", text_so_far)

    def push_assistant_message(self, text):
        self._queue.put(("assistant", text))
        if not self._handed_off:
            self._to_island("reply", text)

    def push_system_message(self, text, island=False):
        """island=True also flashes it on the island (gestures, alerts)."""
        self._queue.put(("system", text))
        if island:
            self._to_island("notice", text)

    def push_status(self, status):
        self._queue.put(("status", status))
        self._to_island("status", status)


# --- the window ----------------------------------------------------------
class DeoGUI(IslandRouting):
    def __init__(self, assistant_name="Deo", on_text_submit=None, on_stop=None,
                 on_pause_toggle=None, start_hidden=False):
        """
        on_text_submit: callable(text), run on a NEW background thread per message.
        on_stop: callable(), run on the main thread when Stop is clicked -- keep it fast.
        on_pause_toggle: callable() -> bool (new paused state), main thread.
        """
        self._queue = queue.Queue()
        self._on_text_submit = on_text_submit
        self._on_stop = on_stop
        self._on_pause_toggle = on_pause_toggle
        self._assistant_name = assistant_name
        self._paused = False
        self._status = "idle"
        self.tray_active = False      # set by the tray: closing then hides instead of quitting
        # The island (island.py) mirrors what's happening while this window is
        # closed. Set by main.py, along with is_gaming.
        self.island = None
        self.is_gaming = lambda: False
        self._window_shown = not start_hidden
        self._handed_off = False

        self.root = tk.Tk()
        if start_hidden:
            # Withdraw BEFORE the window is ever mapped. Withdrawing later (after
            # the tray started) still let Windows show it for an instant and hand
            # it keyboard focus -- stealing focus from a fullscreen game.
            self.root.withdraw()
        self.root.title(assistant_name)
        self.root.geometry("460x720")
        self.root.minsize(360, 480)
        self.root.configure(bg=BG)
        self._set_icon()
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._dark_title_bar)
        self._poll_queue()

    # ---- window chrome --------------------------------------------------
    def _set_icon(self):
        try:
            from PIL import ImageTk
            self._icon = ImageTk.PhotoImage(make_icon_image(64))
            self.root.iconphoto(True, self._icon)
        except Exception:
            pass

    def _dark_title_bar(self):
        """Windows 11 dark title bar, so the chrome matches the window."""
        try:
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            on = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(on), ctypes.sizeof(on))
            color = ctypes.c_int(int(SURFACE[5:7] + SURFACE[3:5] + SURFACE[1:3], 16))  # COLORREF is BGR
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(color), ctypes.sizeof(color))
        except Exception:
            pass

    # ---- layout -----------------------------------------------------------
    def _build(self):
        header = tk.Frame(self.root, bg=SURFACE)
        header.pack(side="top", fill="x")
        self._orb = Orb(header)
        self._orb.pack(side="left", padx=(16, 12), pady=12)
        text_col = tk.Frame(header, bg=SURFACE)
        text_col.pack(side="left", fill="x", expand=True)
        tk.Label(text_col, text=self._assistant_name, bg=SURFACE, fg=TEXT,
                 font=(FONT_FAMILY, 16, "bold"), anchor="w").pack(fill="x", pady=(12, 0))
        self._status_label = tk.Label(text_col, text=STATES["idle"][2], bg=SURFACE, fg=STATES["idle"][0],
                                      font=(FONT_FAMILY, 10), anchor="w")
        self._status_label.pack(fill="x")
        self._info_label = tk.Label(text_col, text="", bg=SURFACE, fg=TEXT_FAINT,
                                    font=(FONT_FAMILY, 8), anchor="w")
        self._info_label.pack(fill="x", pady=(0, 10))
        tk.Frame(self.root, bg=BORDER, height=1).pack(side="top", fill="x")

        footer = tk.Frame(self.root, bg=SURFACE)
        footer.pack(side="bottom", fill="x")
        tk.Frame(self.root, bg=BORDER, height=1).pack(side="bottom", fill="x")

        controls = tk.Frame(footer, bg=SURFACE)
        controls.pack(fill="x", padx=14, pady=(10, 0))
        small = tkfont.Font(family=FONT_FAMILY, size=9, weight="bold")
        self._pause_btn = PillButton(controls, "Pause listening", self._handle_pause,
                                     SURFACE_2, TEXT_DIM, BORDER, height=30, font=small)
        self._pause_btn.pack(side="left")
        PillButton(controls, "Stop talking", self._handle_stop, "#2a1a1f", "#f3a3ab", "#3a2128",
                   height=30, font=small).pack(side="left", padx=(8, 0))
        tk.Label(controls, text="1 finger → right  ·  2 fingers ← left", bg=SURFACE, fg=TEXT_FAINT,
                 font=(FONT_FAMILY, 8)).pack(side="right")

        row = tk.Frame(footer, bg=SURFACE)
        row.pack(fill="x", padx=14, pady=12)
        self._input_canvas = tk.Canvas(row, height=44, bg=SURFACE, highlightthickness=0)
        self._input_canvas.pack(side="left", fill="x", expand=True)
        self._entry = tk.Entry(self._input_canvas, bg=SURFACE_2, fg=TEXT, insertbackground=TEXT,
                               relief="flat", font=(FONT_FAMILY, 11), bd=0)
        self._entry_win = self._input_canvas.create_window(18, 22, window=self._entry, anchor="w")
        self._input_canvas.bind("<Configure>", self._layout_input)
        self._entry.bind("<Return>", self._handle_submit)
        self._placeholder(True)
        self._entry.bind("<FocusIn>", lambda e: self._placeholder(False))
        self._entry.bind("<FocusOut>", lambda e: self._placeholder(not self._entry.get()))
        PillButton(row, "➤", self._handle_submit, USER_BUBBLE, "#ffffff", "#5583ff",
                   width=44, height=44, font=tkfont.Font(family=FONT_FAMILY, size=13)).pack(side="right", padx=(10, 0))

        self._chat = ChatView(self.root)
        self._chat.pack(side="top", fill="both", expand=True)

    def _layout_input(self, event):
        c = self._input_canvas
        c.delete("pill")
        _rounded_rect(c, 1, 1, event.width - 1, 43, 21, fill=SURFACE_2, outline=BORDER, tags="pill")
        c.tag_lower("pill")
        c.itemconfigure(self._entry_win, width=event.width - 36)

    def _placeholder(self, show):
        if show and not self._entry.get():
            self._entry.insert(0, "Type a message...")
            self._entry.configure(fg=TEXT_FAINT)
            self._has_placeholder = True
        elif not show and getattr(self, "_has_placeholder", False):
            self._entry.delete(0, tk.END)
            self._entry.configure(fg=TEXT)
            self._has_placeholder = False

    # ---- event handlers (main thread) -----------------------------------
    def _handle_submit(self, event=None):
        if getattr(self, "_has_placeholder", False):
            return
        text = self._entry.get().strip()
        if not text:
            return
        self._entry.delete(0, tk.END)
        self.add_user_message(text)
        if self._on_text_submit:
            threading.Thread(target=self._on_text_submit, args=(text,), daemon=True).start()

    def _handle_stop(self):
        if self._on_stop:
            self._on_stop()

    def _handle_pause(self):
        if self._on_pause_toggle:
            self._set_paused(self._on_pause_toggle())

    def _set_paused(self, paused):
        self._paused = bool(paused)
        if self._paused:
            self._pause_btn.set("Resume listening", bg="#1d3b36", fg="#7ff0de", hover="#24504a")
        else:
            self._pause_btn.set("Pause listening", bg=SURFACE_2, fg=TEXT_DIM, hover=BORDER)
        self._render_status(self._status)

    def _on_close(self):
        if self.tray_active:
            self.root.withdraw()      # keep running in the tray
            self._window_shown = False
        else:
            self.root.destroy()

    # ---- thread-safe public API -----------------------------------------
    def push_info(self, text):
        """Small line under the status: brain model, next timer, etc."""
        self._queue.put(("info", text))

    def push_paused(self, paused):
        self._queue.put(("paused", paused))

    def request_show(self, focus=True):
        self._queue.put(("show", focus))

    def request_toggle_visible(self):
        self._queue.put(("toggle", None))

    def request_quit(self):
        self._queue.put(("quit", None))

    # ---- queue drain (main thread) ----------------------------------------
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "status":
                    self._render_status(payload)
                elif kind == "user":
                    self.add_user_message(payload)
                elif kind == "assistant":
                    self.add_assistant_message(payload)
                elif kind == "system":
                    self._chat.add("system", "", payload)
                elif kind == "info":
                    self._info_label.configure(text=payload)
                elif kind == "paused":
                    self._set_paused(payload)
                elif kind == "show":
                    self._show(focus=payload is not False)
                elif kind == "stream":
                    self._chat.add("stream", self._assistant_name, payload)
                elif kind == "toggle":
                    if self.root.winfo_viewable():
                        self.root.withdraw()
                        self._window_shown = False
                    else:
                        self._show()
                elif kind == "quit":
                    self.root.destroy()
                    return
        except queue.Empty:
            pass
        self.root.after(50, self._poll_queue)

    def _show(self, focus=True):
        self._window_shown = True
        if self.island is not None:
            self.island.post("hide")
        self.root.deiconify()
        self.root.lift()
        if focus:
            self.root.focus_force()

    def add_user_message(self, text):
        self._chat.add("user", "You", text)

    def add_assistant_message(self, text):
        self._chat.add("assistant", self._assistant_name, text)

    def _render_status(self, status):
        self._status = status
        shown = "paused" if self._paused and status == "idle" else status
        core, _, label = STATES.get(shown, STATES["idle"])
        self._orb.set_state(shown)
        self._status_label.configure(text=label, fg=core)

    def run(self):
        self.root.mainloop()
