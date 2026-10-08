"""
System-tray icon for Deo.

Deo starts with Windows and runs all day, so its window shouldn't have to be
open. Closing the window hides it here instead of quitting.

    left-click          show / hide the window
    Pause listening     ignore "hey Deo", gestures and F9 until resumed
    Open log            logs/deo.log in Notepad
    Quit Deo            actually exit

pystray runs its own thread; everything it does to the window goes through
the GUI's thread-safe request_* methods.
"""

import os
import subprocess

from . import config


class DeoTray:
    def __init__(self, gui, is_paused, toggle_pause, on_quit):
        """is_paused: () -> bool. toggle_pause: () -> bool (new state).
        on_quit: () -> None, called from the tray thread."""
        import pystray
        from .gui import make_icon_image

        self._gui, self._is_paused, self._toggle, self._on_quit = gui, is_paused, toggle_pause, on_quit
        self._images = {
            False: make_icon_image(64, "#8b7cf8"),   # live
            True: make_icon_image(64, "#4b5266"),    # paused: greyed out
        }
        menu = pystray.Menu(
            pystray.MenuItem("Show / hide Deo", self._show_hide, default=True),
            pystray.MenuItem(lambda item: "Resume listening" if self._is_paused() else "Pause listening",
                             self._pause),
            pystray.MenuItem("Open log", self._open_log),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit Deo", self._quit),
        )
        self.icon = pystray.Icon("deo", self._images[False], f"{config.ASSISTANT_NAME} -- listening", menu)

    def start(self):
        self.icon.run_detached()
        self._gui.tray_active = True

    def set_paused(self, paused: bool):
        self.icon.icon = self._images[bool(paused)]
        self.icon.title = f"{config.ASSISTANT_NAME} -- {'paused' if paused else 'listening'}"
        self.icon.update_menu()

    def _show_hide(self, icon=None, item=None):
        self._gui.request_toggle_visible()

    def _pause(self, icon=None, item=None):
        paused = self._toggle()
        self._gui.push_paused(paused)
        self.set_paused(paused)

    def _open_log(self, icon=None, item=None):
        path = os.path.join(config.PROJECT_ROOT, "logs", "deo.log")
        if os.path.exists(path):
            subprocess.Popen(["notepad.exe", path])

    def _quit(self, icon=None, item=None):
        self.icon.stop()
        self._on_quit()
