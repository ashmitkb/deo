"""
Permission gate: every tool call Deo makes passes through check() first.

Three outcomes:

* BLOCK  -- never runs, no dialog. Kept to things a confirmation dialog can't
            make safe: commands that hide what they do (encoded PowerShell,
            Invoke-Expression, certutil/bitsadmin downloaders) -- you can't
            meaningfully approve what you can't read -- and the handful that
            can wreck the machine outright (formatting disks, diskpart, boot
            config, wiping shadow copies or free space).
* CONFIRM -- a Windows dialog shows exactly what is about to happen and you
            click Yes or No. "No" is the default button, and the dialog
            auto-denies after PERMISSION_TIMEOUT_SECONDS, so an unattended
            Deo never ends up doing something because nobody was there.
* ALLOW  -- everything else, as before.

Why a click and not a spoken "yes": Deo's own speakers, a video, or text on a
web page that the model read can all produce the word "yes". Only a person can
click the button. That matters here because untrusted text (web search results,
screen OCR, file contents) flows into the same model that requests actions.

Every CONFIRM and BLOCK decision is appended to permissions.log at the project
root, so there's a record of what Deo asked to do and what you answered.
"""

import ctypes
import os
import re
import threading
from datetime import datetime

from . import config

ALLOW, CONFIRM, BLOCK = "allow", "confirm", "block"


def _compile(patterns):
    return [(re.compile(p, re.IGNORECASE), why) for p, why in patterns]


# (pattern, plain-English reason shown in the dialog / refusal)
_BLOCKED_COMMANDS = _compile([
    (r"-e(nc(odedcommand)?)?\s+[A-Za-z0-9+/=]{8,}", "an encoded PowerShell command (its contents are hidden)"),
    (r"\b(Invoke-Expression|iex)\b", "Invoke-Expression, which runs text as code"),
    (r"\bcertutil\b", "certutil, a common way to download or decode hidden payloads"),
    (r"\bbitsadmin\b", "bitsadmin, a background downloader"),
    (r"\bformat(\.com)?\s+[A-Za-z]:", "formatting a drive"),
    (r"\b(Format-Volume|Clear-Disk|Initialize-Disk)\b", "erasing a disk"),
    (r"\bdiskpart\b", "diskpart (disk partitioning)"),
    (r"\bmkfs\b", "creating a filesystem over a disk"),
    (r"\bbcdedit\b", "changing the boot configuration"),
    (r"\bvssadmin\b", "deleting volume shadow copies (your restore points)"),
    (r"\bwbadmin\s+delete\b", "deleting backups"),
    (r"\bcipher\s+/w", "wiping free disk space"),
])

_CONFIRM_COMMANDS = _compile([
    # deleting
    (r"\b(del|erase)\b", "deletes files"),
    (r"\b(rmdir|rd)\b", "deletes folders"),
    (r"\brm\s+-", "deletes files/folders"),
    (r"\b(Remove-Item|ri)\b", "deletes files/folders"),
    (r"\bClear-(Content|RecycleBin)\b", "erases file contents or empties the Recycle Bin"),
    # overwriting / moving
    (r"(?<![>\d])>(?!>)\s*(?!nul\b|\$null\b|&)\S", "overwrites a file (> redirection)"),
    (r"\b(Set-Content|Out-File)\b", "overwrites a file"),
    (r"\b(move|Move-Item|mv|ren|rename|Rename-Item)\b", "moves or renames files"),
    (r"\brobocopy\b.*\s/(mir|purge)\b", "mirrors a folder (deletes extra files at the destination)"),
    # processes / power
    (r"\b(taskkill|Stop-Process|kill)\b", "force-kills a running program"),
    (r"\b(shutdown|Restart-Computer|Stop-Computer|logoff)\b", "shuts down, restarts or signs out"),
    # system configuration
    (r"\breg(\.exe)?\s+(add|delete|import|copy|restore|load|unload)\b", "changes the registry"),
    (r"\b(Set|New|Remove)-ItemProperty\b", "changes the registry"),
    (r"\bregedit\b", "the registry editor"),
    (r"\b(takeown|icacls|cacls)\b", "changes file ownership or permissions"),
    (r"\bnet\s+(user|localgroup|share)\b", "changes user accounts or shares"),
    (r"\bsc(\.exe)?\s+(create|delete|config|stop)\b", "changes Windows services"),
    (r"\b(Stop|Set|New|Remove)-Service\b", "changes Windows services"),
    (r"\bschtasks\b", "changes scheduled tasks"),
    (r"\b(Register|Unregister)-ScheduledTask\b", "changes scheduled tasks"),
    (r"\bwmic\b", "WMIC (system management)"),
    (r"\bnetsh\b", "changes network or firewall settings"),
    (r"\b(Set-MpPreference|Add-MpPreference)\b", "changes Windows Defender settings"),
    (r"\b(Set|New|Remove)-NetFirewall\w*\b", "changes firewall rules"),
    (r"\bSet-ExecutionPolicy\b", "changes the PowerShell execution policy"),
    (r"\b(runas)\b|-Verb\s+RunAs", "runs something as administrator"),
    (r"[A-Za-z]:\\Windows\b|\bSystem32\b", "touches the Windows system folder"),
    # software
    (r"\b(winget|choco|scoop)\s+(install|uninstall|remove|upgrade)\b", "installs or removes software"),
    (r"\bpip3?\s+(install|uninstall)\b", "installs or removes Python packages"),
    (r"\bnpm\s+(install|uninstall|i|rm)\s+-g\b", "installs or removes global npm packages"),
    (r"\bmsiexec\b", "installs or removes software"),
    # git history
    (r"\bgit\s+(reset\s+--hard|clean\s+-\w*f|push\s+.*--force|push\s+-f\b|branch\s+-D|checkout\s+--\s|restore\s)",
     "throws away git work"),
])


_last_request = ""


def note_request(text: str):
    """Called by Brain at the start of each turn with what the user said
    (as transcribed). Shown in every permission dialog, so a mis-hearing --
    "text sorry test.txt" for "test.txt" -- is visible before you click Yes."""
    global _last_request
    _last_request = (text or "").strip()


def describe_path(path: str) -> str:
    """Exactly what a file operation will touch, laid out so the name can't be
    missed: name on its own line, then where it is, then size/date (or, for a
    folder, how much is inside) so you can tell it's the thing you meant."""
    path = os.path.abspath(path)
    name = os.path.basename(path.rstrip("\\/")) or path
    folder = os.path.dirname(path)
    try:
        modified = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%d %b %Y, %H:%M")
    except OSError:
        modified = "unknown"
    if os.path.isdir(path):
        files = dirs = 0
        for _, dnames, fnames in os.walk(path):
            files += len(fnames)
            dirs += len(dnames)
            if files > 100000:
                break
        what = f"FOLDER:   {name}\nContains: {files} file(s) in {dirs} subfolder(s) -- ALL of it"
    else:
        size = float(os.path.getsize(path)) if os.path.exists(path) else 0.0
        unit = "bytes"
        for unit in ("bytes", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                break
            size /= 1024
        size_text = f"{size:.0f} {unit}" if unit == "bytes" else f"{size:.1f} {unit}"
        what = f"FILE:     {name}\nSize:     {size_text}"
    return f"{what}\nIn:       {folder}\nModified: {modified}"


def classify_command(command: str):
    """(ALLOW | CONFIRM | BLOCK, reason or None) for a shell command."""
    for pattern, why in _BLOCKED_COMMANDS:
        if pattern.search(command):
            return BLOCK, why
    for pattern, why in _CONFIRM_COMMANDS:
        if pattern.search(command):
            return CONFIRM, why
    return ALLOW, None


def classify(tool: str, args: dict):
    """(decision, reason, what-to-show-the-user) for any tool call."""
    args = args or {}
    if tool == "run_command":
        command = str(args.get("command", ""))
        decision, why = classify_command(command)
        return decision, why, command

    if tool == "run_code_file":
        # Scripts Deo wrote for you live in ~/DeoProjects and run freely.
        # Anything else is a program of unknown origin -- ask first.
        from .tools import _resolve_user_path
        path = str(args.get("path", ""))
        resolved = _resolve_user_path(path) or _resolve_user_path(os.path.join("DeoProjects", path))
        projects = os.path.normcase(os.path.join(os.path.expanduser("~"), "DeoProjects"))
        if resolved and os.path.normcase(os.path.abspath(resolved)).startswith(projects + os.sep):
            return ALLOW, None, resolved
        if not resolved:
            return ALLOW, None, path      # run_code_file will report "not found"
        return CONFIRM, "runs a script from outside your DeoProjects folder", describe_path(resolved)

    if tool == "delete_file":
        from .tools import _resolve_user_path
        path = str(args.get("path", ""))
        resolved = _resolve_user_path(path)
        if not resolved:
            # Nothing there to delete: let delete_file report "not found"
            # instead of asking about a file that doesn't exist. A popup for
            # "text sorry test.txt" (a mis-heard name) only teaches you to
            # click through popups without reading them.
            return ALLOW, None, path
        return CONFIRM, "moves a file or folder to the Recycle Bin", describe_path(resolved)

    # MCP filesystem server
    if tool == "write_file":
        path = str(args.get("path", ""))
        if os.path.exists(path):
            return CONFIRM, "overwrites an existing file", path
        return ALLOW, None, path
    if tool == "edit_file":
        return CONFIRM, "edits a file in place", str(args.get("path", ""))
    if tool == "move_file":
        src, dst = args.get("source", ""), args.get("destination", "")
        return CONFIRM, "moves or renames a file", f"{src}\n  ->  {dst}"

    if tool == "power":
        action = str(args.get("action", "")).lower().replace(" ", "_")
        ends_session = {"shutdown": "shuts down your PC", "shut_down": "shuts down your PC",
                        "restart": "restarts your PC", "reboot": "restarts your PC",
                        "sign_out": "signs you out of Windows", "log_off": "signs you out of Windows"}
        if action in ends_session:
            return CONFIRM, ends_session[action], f"{action.replace('_', ' ').upper()} this PC"
        return ALLOW, None, action

    # Deo's own memory -- forgetting is deleting
    if tool == "forget_fact":
        return CONFIRM, "deletes something Deo remembers about you", str(args.get("key", ""))
    if tool == "clear_notes":
        return CONFIRM, "deletes all of your saved notes", "all notes"

    return ALLOW, None, ""


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------

_MB_YESNO = 0x04
_MB_ICONWARNING = 0x30
_MB_DEFBUTTON2 = 0x100          # "No" is the default -- Enter/Space never approves
_MB_SYSTEMMODAL = 0x1000
_MB_SETFOREGROUND = 0x10000
_MB_TOPMOST = 0x40000
_IDYES = 6

_dialog_lock = threading.Lock()  # one question at a time


def _show_dialog(title: str, text: str, timeout_s: int) -> bool:
    flags = (_MB_YESNO | _MB_ICONWARNING | _MB_DEFBUTTON2 | _MB_SYSTEMMODAL
             | _MB_SETFOREGROUND | _MB_TOPMOST)
    user32 = ctypes.windll.user32
    try:
        # Undocumented but present since XP; returns 32000 on timeout.
        fn = user32.MessageBoxTimeoutW
        fn.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p,
                       ctypes.c_uint, ctypes.c_ushort, ctypes.c_uint]
        result = fn(None, text, title, flags, 0, int(timeout_s * 1000))
    except AttributeError:
        result = user32.MessageBoxW(None, text, title, flags)
    return result == _IDYES


def ask_user(tool: str, why: str, detail: str) -> bool:
    """Show the permission dialog; True only if the user clicked Yes."""
    timeout_s = getattr(config, "PERMISSION_TIMEOUT_SECONDS", 60)
    shown = detail if len(detail) <= 800 else detail[:800] + " ..."
    heard = _last_request or "(nothing -- this didn't come from something you said)"
    text = (
        f"You said:  \"{heard}\"\n\n"
        f"Deo wants to do something that {why}:\n\n"
        f"{shown}\n\n"
        f"Is this EXACTLY what you meant?\n"
        f"If the name above isn't what you asked for (Deo may have mis-heard you), click No.\n\n"
        f"(Auto-denies in {timeout_s} seconds.)"
    )
    with _dialog_lock:
        print(f"[permissions] asking: {tool} -- {why}")
        return _show_dialog(f"{config.ASSISTANT_NAME} needs your permission", text, timeout_s)


def _audit(decision: str, tool: str, why, detail: str):
    try:
        line = (f"{datetime.now():%Y-%m-%d %H:%M:%S}  {decision:<8} {tool:<12} "
                f"{why or ''} | {detail.replace(chr(10), ' ')}\n")
        with open(config.PERMISSIONS_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def check(tool: str, args: dict, ask=None):
    """None if the call may go ahead, otherwise the result dict to hand back to
    the model instead of running the tool. `ask` is injectable for tests.

    `ask` defaults to looking up ask_user at CALL time, not binding it as a
    default argument: bound at import, a test that swapped ask_user out still
    got the real dialog -- which is exactly how a test popped one up on screen."""
    ask = ask or ask_user
    decision, why, detail = classify(tool, args)
    if decision == ALLOW:
        return None

    if decision == BLOCK:
        _audit("BLOCKED", tool, why, detail)
        print(f"[permissions] BLOCKED {tool}: {why}")
        return {
            "success": False,
            "message": (
                f"Blocked: this would run {why}, which Deo never does, even with "
                f"permission. Tell the user exactly what was requested; if they really "
                f"want it, they have to do it themselves. Do not retry it another way."
            ),
        }

    approved = ask(tool, why, detail)
    _audit("APPROVED" if approved else "DENIED", tool, why, detail)
    if approved:
        print(f"[permissions] approved: {tool}")
        return None
    print(f"[permissions] denied: {tool}")
    return {
        "success": False,
        "message": (
            "The user did NOT give permission (they clicked No, or didn't answer in "
            "time), so nothing was done. Tell them it was cancelled. Do not retry it, "
            "and do not try to achieve the same thing with a different command."
        ),
    }
