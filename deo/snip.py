"""
Screenshot-style area picker: "hey Deo, look at this -- what is it?"

    python -m deo.snip --out C:\\path\\pick.png
        exit 0 -> the selected area was saved to --out; prints its bounds as JSON
        exit 3 -> cancelled (Esc, right-click, or 60s with no selection)
        anything else -> it crashed (the caller reports that, not "cancelled")

Runs as its own process (tools.look_at_region starts it) so it can own a
fullscreen Tk window on its own main thread, independent of Deo's window.

The whole desktop (every monitor) is frozen as a screenshot, dimmed, with a
crosshair; dragging brightens the chosen area and shows its size. A plain
click instead of a drag picks the whole monitor under the cursor.
"""

import argparse
import ctypes
import ctypes.wintypes   # here, not inside main(): importing it in the function made
                         # "ctypes" a local name there and crashed the picker on start
import json
import sys
import tkinter as tk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    ctypes.windll.shcore.SetProcessDpiAwareness(2)       # real pixels on every monitor
    from PIL import Image, ImageEnhance, ImageGrab, ImageTk

    user32 = ctypes.windll.user32
    vx, vy = user32.GetSystemMetrics(76), user32.GetSystemMetrics(77)
    vw, vh = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
    shot = ImageGrab.grab(all_screens=True)              # covers the whole virtual desktop
    dim = ImageEnhance.Brightness(shot).enhance(0.42)

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.geometry(f"{vw}x{vh}{vx:+d}{vy:+d}")
    root.configure(cursor="crosshair", bg="black")
    cv = tk.Canvas(root, width=vw, height=vh, highlightthickness=0, bd=0, cursor="crosshair")
    cv.pack()
    dim_img = ImageTk.PhotoImage(dim)
    cv.create_image(0, 0, image=dim_img, anchor="nw")

    # instruction pill at the top of the primary monitor
    px = -vx + user32.GetSystemMetrics(0) // 2
    font = ("Segoe UI Variable Text", 13)
    label = cv.create_text(px, 46 - vy, text="Drag over what you want Deo to see   ·   Esc to cancel",
                           fill="white", font=font)
    x1, y1, x2, y2 = cv.bbox(label)
    pill = cv.create_rectangle(x1 - 18, y1 - 10, x2 + 18, y2 + 10, fill="#1c1c22", outline="#4a4a55")
    cv.tag_raise(label)

    state = {"start": None, "crop": None, "rect": None, "size": None, "result": None}

    def clear():
        for key in ("crop", "rect", "size"):
            if state[key] is not None:
                cv.delete(state[key])
                state[key] = None

    def on_down(e):
        state["start"] = (e.x, e.y)

    def on_drag(e):
        if not state["start"]:
            return
        sx, sy = state["start"]
        l, t, r, b = min(sx, e.x), min(sy, e.y), max(sx, e.x), max(sy, e.y)
        clear()
        if r - l > 2 and b - t > 2:
            state["crop_img"] = ImageTk.PhotoImage(shot.crop((l, t, r, b)))
            state["crop"] = cv.create_image(l, t, image=state["crop_img"], anchor="nw")
        state["rect"] = cv.create_rectangle(l, t, r, b, outline="white", width=2)
        state["size"] = cv.create_text(l + 4, b + 14, anchor="w", fill="white",
                                       font=("Segoe UI Variable Text", 10), text=f"{r - l} × {b - t}")
        cv.tag_raise(pill)
        cv.tag_raise(label)

    def on_up(e):
        if not state["start"]:
            return
        sx, sy = state["start"]
        l, t, r, b = min(sx, e.x), min(sy, e.y), max(sx, e.x), max(sy, e.y)
        if r - l < 8 or b - t < 8:
            # a click: take the whole monitor under the cursor
            mon = user32.MonitorFromPoint(ctypes.wintypes.POINT(e.x + vx, e.y + vy), 2)

            class MONITORINFO(ctypes.Structure):
                _fields_ = [("cbSize", ctypes.c_ulong), ("rcMonitor", ctypes.c_long * 4),
                            ("rcWork", ctypes.c_long * 4), ("dwFlags", ctypes.c_ulong)]
            mi = MONITORINFO()
            mi.cbSize = ctypes.sizeof(MONITORINFO)
            user32.GetMonitorInfoW(mon, ctypes.byref(mi))
            l, t, r, b = mi.rcMonitor[0] - vx, mi.rcMonitor[1] - vy, mi.rcMonitor[2] - vx, mi.rcMonitor[3] - vy
        shot.crop((l, t, r, b)).save(args.out)
        state["result"] = {"left": l + vx, "top": t + vy, "width": r - l, "height": b - t}
        root.destroy()

    def cancel(_=None):
        root.destroy()

    cv.bind("<ButtonPress-1>", on_down)
    cv.bind("<B1-Motion>", on_drag)
    cv.bind("<ButtonRelease-1>", on_up)
    cv.bind("<ButtonPress-3>", cancel)
    root.bind("<Escape>", cancel)
    root.after(int(args.timeout * 1000), cancel)
    root.after(50, lambda: (root.lift(), root.focus_force()))
    root.mainloop()

    if state["result"] is None:
        return 3                      # cancelled -- distinct from a crash (exit 1)
    print(json.dumps(state["result"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
