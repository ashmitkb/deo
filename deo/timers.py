"""
Timers and reminders.

    set_timer(minutes=10, label="pasta")        -> fires in 10 minutes
    set_reminder(message="call mom", at="18:30") -> fires at 6:30pm (tomorrow if
                                                    that time has passed today)
    set_reminder(message="stretch", in_minutes=45)

Saved to deo/timers.json, so a reminder for tonight survives Deo restarting in
between; anything that came due while Deo was off fires as soon as it starts.

When one fires, `on_fire(text)` is called -- main.py wires that to speech and
the Deo window. A single background thread checks once a second, rather than
one threading.Timer per item, so everything stays in one list that can be
listed, cancelled and saved.
"""

import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timedelta

from . import config

_lock = threading.Lock()
_items = []          # [{"id", "kind", "label", "due" (epoch seconds), "created"}]
_loaded = False
on_fire = lambda text: print(f"[timer] {text}")   # replaced by main.py


def _path():
    return getattr(config, "TIMERS_FILE", "")


def _load():
    global _loaded, _items
    if _loaded:
        return
    _loaded = True
    path = _path()
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                _items = json.load(f)
        except Exception as e:
            print(f"[timers] couldn't read {path}: {e}")
            _items = []


def _save():
    path = _path()
    if not path:
        return
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_items, f, indent=1)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[timers] couldn't save: {e}")


def _say_when(due: float) -> str:
    when = datetime.fromtimestamp(due)
    now = datetime.now()
    clock = when.strftime("%I:%M %p").lstrip("0")
    if when.date() == now.date():
        return f"at {clock}"
    if when.date() == (now + timedelta(days=1)).date():
        return f"tomorrow at {clock}"
    return when.strftime("on %A at ") + clock


def _say_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts = [f"{h} hour{'s' * (h != 1)}" if h else "",
             f"{m} minute{'s' * (m != 1)}" if m else "",
             f"{s} second{'s' * (s != 1)}" if s and not h else ""]
    return " ".join(p for p in parts if p) or "0 seconds"


def _add(kind: str, label: str, due: float) -> dict:
    with _lock:
        _load()
        item = {"id": uuid.uuid4().hex[:6], "kind": kind, "label": label.strip(),
                "due": due, "created": time.time()}
        _items.append(item)
        _save()
    return item


def set_timer(minutes: float = 0, seconds: float = 0, hours: float = 0, label: str = "") -> dict:
    total = float(hours or 0) * 3600 + float(minutes or 0) * 60 + float(seconds or 0)
    if total <= 0:
        return {"success": False, "message": "A timer needs a length, e.g. minutes=10."}
    if total > 7 * 24 * 3600:
        return {"success": False, "message": "That's over a week -- use set_reminder with a time instead."}
    item = _add("timer", label or "", time.time() + total)
    name = f" for {item['label']}" if item["label"] else ""
    return {"success": True,
            "message": f"Timer{name} set: {_say_duration(total)}, going off {_say_when(item['due'])}."}


def _parse_clock(at: str):
    """'18:30', '6:30 pm', '6pm', '06:30 PM' -> next datetime with that time."""
    text = at.strip().lower().replace(".", "")
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
    if not m:
        return None
    hour, minute, ampm = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ampm == "pm" and hour < 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    if not (0 <= hour < 24 and 0 <= minute < 60):
        return None
    now = datetime.now()
    when = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if when <= now:
        when += timedelta(days=1)
    return when


def set_reminder(message: str, at: str = "", in_minutes: float = 0) -> dict:
    if not message.strip():
        return {"success": False, "message": "What should the reminder say?"}
    if at:
        when = _parse_clock(at)
        if not when:
            return {"success": False, "message": f"Couldn't understand the time '{at}'. Use e.g. '18:30' or '6:30 pm'."}
        due = when.timestamp()
    elif in_minutes and float(in_minutes) > 0:
        due = time.time() + float(in_minutes) * 60
    else:
        return {"success": False, "message": "Give either a time (at='18:30') or in_minutes."}
    _add("reminder", message, due)
    return {"success": True, "message": f"Reminder set {_say_when(due)}: {message.strip()}."}


def list_timers() -> dict:
    with _lock:
        _load()
        active = sorted(_items, key=lambda i: i["due"])
    if not active:
        return {"success": True, "message": "No timers or reminders are set."}
    now = time.time()
    lines = []
    for i in active:
        what = f"timer for {i['label']}" if i["kind"] == "timer" and i["label"] else \
               "timer" if i["kind"] == "timer" else f"reminder: {i['label']}"
        lines.append(f"{what} -- {_say_duration(i['due'] - now)} left ({_say_when(i['due'])})")
    return {"success": True, "message": "; ".join(lines)}


def cancel_timer(which: str = "") -> dict:
    """Cancel by label/words in the reminder, or 'all'. Empty = the soonest."""
    with _lock:
        _load()
        if not _items:
            return {"success": False, "message": "There are no timers or reminders to cancel."}
        which = (which or "").strip().lower()
        if which in ("all", "everything", "all of them"):
            gone = list(_items)
        elif which:
            gone = [i for i in _items if which in i["label"].lower() or i["id"] == which]
        else:
            gone = [min(_items, key=lambda i: i["due"])]
        if not gone:
            return {"success": False, "message": f"No timer or reminder matches '{which}'."}
        for i in gone:
            _items.remove(i)
        _save()
    names = ", ".join(i["label"] or i["kind"] for i in gone)
    return {"success": True, "message": f"Cancelled {len(gone)}: {names}."}


def get_date_time() -> dict:
    """The model has no clock of its own -- without this, "what time is it"
    and "remind me in an hour" would be guesses."""
    now = datetime.now()
    return {"success": True,
            "message": now.strftime("It is %I:%M %p on %A, %d %B %Y.").replace(" 0", " ")}


def _announce(item: dict) -> str:
    if item["kind"] == "timer":
        return f"Your {item['label']} timer is done." if item["label"] else "Your timer is done."
    return f"Reminder: {item['label']}."


def _check_loop():
    while True:
        due_now = []
        with _lock:
            _load()
            now = time.time()
            due_now = [i for i in _items if i["due"] <= now]
            if due_now:
                for i in due_now:
                    _items.remove(i)
                _save()
        for item in sorted(due_now, key=lambda i: i["due"]):
            late = time.time() - item["due"]
            text = _announce(item)
            if late > 120:      # came due while Deo was off
                text += f" (This was due {_say_duration(late)} ago, while I was off.)"
            try:
                on_fire(text)
            except Exception as e:
                print(f"[timers] announcing failed: {e}")
        time.sleep(1)


def start():
    threading.Thread(target=_check_loop, daemon=True, name="deo-timers").start()
