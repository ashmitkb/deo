"""
Minimal desktop GUI for Deo -- replaces the raw console as the interaction
surface so tool-call/reasoning noise doesn't leak into what the user sees.

IMPORTANT THREADING RULE: Tkinter widgets may only be touched from the
thread running root.mainloop(). Everything else (voice loop, wake-word
listener, screen QA) runs on background threads exactly as before; they
must talk to this GUI ONLY through the thread-safe push_* methods below,
which drop events onto a queue.Queue(). The GUI drains that queue on a
root.after() timer running on the main thread. Never call any other method
on DeoGUI from a background thread.

Usage (see main.py):
    gui = DeoGUI(on_text_submit=..., on_stop=...)
    threading.Thread(target=background_work, args=(gui,), daemon=True).start()
    gui.run()   # blocks on the Tk mainloop until the window is closed
"""

import queue
import threading
import tkinter as tk
from tkinter import scrolledtext, font as tkfont

# --- Colors (dark theme) ---
BG = "#1a1a1f"
PANEL_BG = "#232329"
INPUT_BG = "#2a2a31"
USER_ACCENT = "#5b8cff"
TEXT_PRIMARY = "#f2f2f5"
TEXT_MUTED = "#8b8b96"
ACCENT = "#5eead4"

STATUS_COLORS = {
    "idle": "#565660",
    "listening": "#5eead4",
    "thinking": "#f5b94d",
    "speaking": "#7c6cf7",
    "error": "#f16565",
}
STATUS_LABELS = {
    "idle": "Idle",
    "listening": "Listening...",
    "thinking": "Thinking...",
    "speaking": "Speaking...",
    "error": "Error",
}


class DeoGUI:
    def __init__(self, assistant_name="Deo", on_text_submit=None, on_stop=None):
        """
        on_text_submit: callable(text: str), called on a NEW background thread
            whenever the user types a message and hits Send/Enter. Do not do
            slow work (brain.ask, TTS) here on the caller's thread -- this
            class already spawns a thread per submission for you.
        on_stop: callable(), called on the main thread when Stop is clicked.
            Keep it fast (e.g. sd.stop()) -- it runs during a GUI event.
        """
        self._queue = queue.Queue()
        self._on_text_submit = on_text_submit
        self._on_stop = on_stop
        self._assistant_name = assistant_name

        self.root = tk.Tk()
        self.root.title(assistant_name)
        self.root.geometry("420x620")
        self.root.configure(bg=BG)
        self.root.minsize(360, 480)

        self._build_layout()
        self._poll_queue()

    # ---- layout (main thread only) ----
    def _build_layout(self):
        header = tk.Frame(self.root, bg=PANEL_BG, height=56)
        header.pack(side="top", fill="x")
        header.pack_propagate(False)

        title_font = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        tk.Label(header, text=self._assistant_name, bg=PANEL_BG, fg=TEXT_PRIMARY,
                 font=title_font).pack(side="left", padx=16)

        status_frame = tk.Frame(header, bg=PANEL_BG)
        status_frame.pack(side="right", padx=(0, 8))
        self._status_dot = tk.Canvas(status_frame, width=10, height=10, bg=PANEL_BG, highlightthickness=0)
        self._status_dot_id = self._status_dot.create_oval(1, 1, 9, 9, fill=STATUS_COLORS["idle"], outline="")
        self._status_dot.pack(side="left", padx=(0, 6))
        self._status_label = tk.Label(status_frame, text=STATUS_LABELS["idle"], bg=PANEL_BG,
                                       fg=TEXT_MUTED, font=("Segoe UI", 10))
        self._status_label.pack(side="left")

        stop_btn = tk.Button(header, text="Stop", command=self._handle_stop,
                              bg="#3a2020", fg="#f0a0a0", activebackground="#4a2828",
                              activeforeground="#f5b0b0", relief="flat", padx=10, pady=2,
                              font=("Segoe UI", 9), cursor="hand2")
        stop_btn.pack(side="right", padx=(0, 12))

        self._transcript = scrolledtext.ScrolledText(
            self.root, bg=BG, fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY,
            relief="flat", wrap="word", font=("Segoe UI", 10), padx=14, pady=12,
            state="disabled", borderwidth=0,
        )
        self._transcript.pack(side="top", fill="both", expand=True)
        self._transcript.tag_config("user", foreground=TEXT_PRIMARY, justify="right",
                                     lmargin1=50, lmargin2=50, spacing3=14)
        self._transcript.tag_config("assistant", foreground=TEXT_PRIMARY, justify="left",
                                     rmargin=50, spacing3=14)
        self._transcript.tag_config("user_label", foreground=USER_ACCENT,
                                     font=("Segoe UI", 8, "bold"), justify="right")
        self._transcript.tag_config("assistant_label", foreground=ACCENT,
                                     font=("Segoe UI", 8, "bold"), justify="left")
        self._transcript.tag_config("system", foreground=TEXT_MUTED,
                                     font=("Segoe UI", 9, "italic"), justify="center")

        input_row = tk.Frame(self.root, bg=PANEL_BG)
        input_row.pack(side="bottom", fill="x")
        self._entry = tk.Entry(input_row, bg=INPUT_BG, fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY,
                                relief="flat", font=("Segoe UI", 10))
        self._entry.pack(side="left", fill="x", expand=True, padx=(12, 8), pady=10, ipady=6)
        self._entry.bind("<Return>", self._handle_submit)
        send_btn = tk.Button(input_row, text="Send", command=self._handle_submit,
                              bg=USER_ACCENT, fg="white", activebackground="#3f6fe0",
                              relief="flat", padx=14, font=("Segoe UI", 9, "bold"), cursor="hand2")
        send_btn.pack(side="right", padx=(0, 12), pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

    # ---- event handlers (main thread only, fired by Tk) ----
    def _handle_submit(self, event=None):
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

    # ---- thread-safe public API: call these from ANY thread ----
    def push_user_message(self, text: str):
        self._queue.put(("user", text))

    def push_assistant_message(self, text: str):
        self._queue.put(("assistant", text))

    def push_system_message(self, text: str):
        self._queue.put(("system", text))

    def push_status(self, status: str):
        self._queue.put(("status", status))

    # ---- queue drain + rendering (main thread only, via root.after) ----
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
                    self._append_system(payload)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_queue)

    def add_user_message(self, text: str):
        self._append_bubble("You", text, "user_label", "user")

    def add_assistant_message(self, text: str):
        self._append_bubble(self._assistant_name, text, "assistant_label", "assistant")

    def _append_bubble(self, who, text, label_tag, body_tag):
        self._transcript.configure(state="normal")
        self._transcript.insert("end", f"{who}\n", label_tag)
        self._transcript.insert("end", f"{text}\n\n", body_tag)
        self._transcript.configure(state="disabled")
        self._transcript.see("end")

    def _append_system(self, text: str):
        self._transcript.configure(state="normal")
        self._transcript.insert("end", f"{text}\n\n", "system")
        self._transcript.configure(state="disabled")
        self._transcript.see("end")

    def _render_status(self, status: str):
        color = STATUS_COLORS.get(status, STATUS_COLORS["idle"])
        label = STATUS_LABELS.get(status, status)
        self._status_dot.itemconfig(self._status_dot_id, fill=color)
        self._status_label.configure(text=label)

    def run(self):
        """Blocks on the Tk mainloop. Call this from your program's main
        thread, after background work has been started on other threads."""
        self.root.mainloop()
