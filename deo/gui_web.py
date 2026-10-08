"""
Deo's window, built as a web page (deo/webui/) and shown in a native window
through pywebview -- which on Windows uses Microsoft Edge WebView2, already
part of the OS. Same public API as the Tk window in gui.py, so main.py can use
either; config.GUI_STYLE picks.

Python -> page: events go on a queue that the page polls (app.js poll()).
Page -> Python: the methods of _Api, exposed as window.pywebview.api.*

Threading: pywebview must own the main thread (run() blocks on it), exactly
like Tk did. push_* / request_* are safe from any thread.
"""

import ctypes
import os
import queue
import threading

from .gui import IslandRouting, make_icon_image

WEBUI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui")
BG = "#07070a"


class _Api:
    """Everything the page can call. Kept free of public attributes --
    pywebview walks the object to expose it to JavaScript."""

    def __init__(self, gui):
        self.__gui = gui

    def poll(self):
        return self.__gui._drain()

    def submit(self, text):
        cb = self.__gui._on_text_submit
        if cb and text and text.strip():
            threading.Thread(target=cb, args=(text.strip(),), daemon=True).start()

    def stop(self):
        if self.__gui._on_stop:
            self.__gui._on_stop()

    def toggle_pause(self):
        if self.__gui._on_pause_toggle:
            return bool(self.__gui._on_pause_toggle())
        return False

    def open_link(self, url):
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            os.startfile(url)


class DeoWebGUI(IslandRouting):
    def __init__(self, assistant_name="Deo", on_text_submit=None, on_stop=None,
                 on_pause_toggle=None, start_hidden=False):
        import webview

        self._queue = queue.Queue()
        self._on_text_submit = on_text_submit
        self._on_stop = on_stop
        self._on_pause_toggle = on_pause_toggle
        self.tray_active = False
        self.island = None
        self.is_gaming = lambda: False
        self._window_shown = not start_hidden
        self._handed_off = False
        self._quitting = False

        self.window = webview.create_window(
            assistant_name,
            url=os.path.join(WEBUI_DIR, "index.html"),
            js_api=_Api(self),
            width=500, height=820, min_size=(400, 560),
            background_color=BG,
            hidden=start_hidden,
        )
        self.window.events.closing += self._on_closing
        self.window.events.shown += self._style_native_window

    # ---- page <-> python -----------------------------------------------------
    def _drain(self):
        events = []
        try:
            while len(events) < 200:
                events.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        return events

    def _on_closing(self):
        """Closing hides to the tray instead of quitting (while the tray runs)."""
        if self.tray_active and not self._quitting:
            self.window.hide()
            self._window_shown = False
            return False
        return True

    def _style_native_window(self):
        """Dark title bar matching the page, and Deo's orb as the icon."""
        try:
            hwnd = int(self.window.native.Handle.ToInt64())
            dwm = ctypes.windll.dwmapi
            on = ctypes.c_int(1)
            dwm.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(on), ctypes.sizeof(on))
            colour = ctypes.c_int(0x0a0707)                      # COLORREF (BGR) of #07070a
            dwm.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(colour), ctypes.sizeof(colour))
            ico = os.path.join(WEBUI_DIR, "deo.ico")
            if not os.path.exists(ico):
                make_icon_image(256).save(ico, sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
            user32 = ctypes.windll.user32
            user32.LoadImageW.restype = ctypes.c_void_p
            user32.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
            for which, size in ((0, 16), (1, 32)):              # ICON_SMALL, ICON_BIG
                h = user32.LoadImageW(None, ico, 1, size, size, 0x10)   # IMAGE_ICON, LR_LOADFROMFILE
                if h:
                    user32.SendMessageW(hwnd, 0x80, which, h)            # WM_SETICON
        except Exception as e:
            print(f"[gui] couldn't style the window: {e}")

    # ---- public API (thread-safe) ---------------------------------------------
    def push_info(self, text):
        self._queue.put(("info", text))

    def push_paused(self, paused):
        self._queue.put(("paused", bool(paused)))

    def request_show(self, focus=True):
        self._window_shown = True
        if self.island is not None:
            self.island.post("hide")
        if getattr(self, "edge_glow", None) is not None:
            self.edge_glow.post("off")
        self.window.show()
        if focus:
            try:
                self.window.restore()
            except Exception:
                pass

    def request_toggle_visible(self):
        if self._window_shown:
            self.window.hide()
            self._window_shown = False
        else:
            self.request_show()

    def request_quit(self):
        self._quitting = True
        self.window.destroy()

    def add_user_message(self, text):
        self._queue.put(("user", text))

    def run(self):
        import webview
        storage = os.path.join(os.path.dirname(WEBUI_DIR), "..", "logs", "webview")
        webview.start(gui="edgechromium", private_mode=False, storage_path=os.path.abspath(storage))
