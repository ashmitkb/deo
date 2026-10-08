"""
Long-term memory for Deo. Persists across sessions in a local JSON file.

Structure:
{
  "facts": {"name": "Ashmit", "prefers_browser": "Brave", ...},
  "notes": ["Pick up groceries", "F1 race this Sunday", ...],
  "apps": {"steam": "used 12 times", ...}
}

The LLM can read and write this store via tools exposed in tools.py.
On startup, the full memory is injected into the system prompt so Deo
already "knows" everything it's been told across previous sessions.
"""

import json
import os
from pathlib import Path
from . import config


def _memory_path() -> Path:
    return Path(config.MEMORY_FILE)


def load_memory() -> dict:
    path = _memory_path()
    if not path.exists():
        return {"facts": {}, "notes": [], "apps": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"facts": {}, "notes": [], "apps": {}}


def save_memory(memory: dict):
    path = _memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8")


def memory_as_prompt_block() -> str:
    """Returns a formatted string summarising memory for injection into the system prompt."""
    mem = load_memory()
    lines = []

    if mem.get("facts"):
        lines.append("What you know about the user:")
        for k, v in mem["facts"].items():
            lines.append(f"  - {k}: {v}")

    if mem.get("notes"):
        lines.append("Notes the user has asked you to remember:")
        for n in mem["notes"]:
            lines.append(f"  - {n}")

    if mem.get("apps"):
        lines.append("Frequently used apps:")
        for app, info in mem["apps"].items():
            lines.append(f"  - {app}: {info}")

    if not lines:
        return ""

    return "\n".join(["--- Your memory from previous sessions ---", *lines, "---"])


# --- Tool functions called by the LLM ---

def remember_fact(key: str, value: str) -> dict:
    """Store a personal fact about the user (e.g. name, preference)."""
    mem = load_memory()
    mem["facts"][key.strip().lower()] = value.strip()
    save_memory(mem)
    return {"success": True, "message": f"Remembered: {key} = {value}"}


def save_note(note: str) -> dict:
    """Save a note or reminder the user wants to keep."""
    mem = load_memory()
    mem["notes"].append(note.strip())
    save_memory(mem)
    return {"success": True, "message": f"Note saved: {note}"}


def forget_fact(key: str) -> dict:
    """Remove a stored fact by key."""
    mem = load_memory()
    key = key.strip().lower()
    if key in mem["facts"]:
        del mem["facts"][key]
        save_memory(mem)
        return {"success": True, "message": f"Forgot fact: {key}"}
    return {"success": False, "message": f"No fact stored with key '{key}'."}


def clear_notes() -> dict:
    """Clear all saved notes."""
    mem = load_memory()
    mem["notes"] = []
    save_memory(mem)
    return {"success": True, "message": "All notes cleared."}


def recall_all() -> dict:
    """Return everything currently stored in memory."""
    mem = load_memory()
    if not any([mem.get("facts"), mem.get("notes"), mem.get("apps")]):
        return {"success": True, "message": "I don't have anything stored in memory yet."}
    return {"success": True, "message": memory_as_prompt_block()}


def log_app_usage(app_name: str) -> dict:
    """Internally track that an app was used (called by open_app tool)."""
    mem = load_memory()
    key = app_name.strip().lower()
    current = mem["apps"].get(key, "used 0 times")
    try:
        count = int(current.split()[1]) + 1
    except Exception:
        count = 1
    mem["apps"][key] = f"used {count} times"
    save_memory(mem)
    return {"success": True, "message": f"Logged usage of {app_name}."}