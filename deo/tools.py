"""
Tool registry for Deo.

DESIGN RULE: every tool function returns a dict: {"success": bool, "message": str}.
Never return a bare string claiming success -- the brain grounds its spoken reply in
this dict, so if a tool lies about success, Deo will too. Don't lie in tools.

Add a new tool by: (1) writing the function returning {"success", "message"},
(2) adding its schema to TOOL_SCHEMAS, (3) registering it in TOOL_REGISTRY.
"""

import subprocess
import os
import re
import shutil
import glob
import sys
import time
import difflib
import webbrowser
import requests
import win32gui
import win32con
import win32api
from . import config
from . import Memory


# ---------------------------------------------------------------------------
# Command screening
#
# run_command hands the model a real shell, and the model's context is not a
# trusted place: web_search returns arbitrary web page text, screen_qa returns
# OCR of whatever happens to be on screen, and the MCP filesystem server returns
# file contents. Any of those can carry text written by someone else arguing for
# a command to be run -- and the system prompt tells Deo to reach for tools
# eagerly, which makes it more suggestible, not less.
#
# So destructive commands are refused here, before execution. The refusal comes
# back through the normal {"success": False} channel, which means the model
# reports it honestly to the user instead of pretending the work was done.
# ---------------------------------------------------------------------------

def _screen_command(command: str):
    """(allowed, reason). Only BLOCK-tier commands are refused here: anything
    that needs confirmation has already been approved by the user in
    permissions.check() before run_command is ever called. This is a backstop
    in case run_command is reached some other way."""
    from .permissions import classify_command, BLOCK
    decision, why = classify_command(command)
    return decision != BLOCK, why


# ---------------------------------------------------------------------------
# run_command
# ---------------------------------------------------------------------------

def run_command(command: str) -> dict:
    """Run a shell command on Windows and report whether it actually succeeded."""
    allowed, matched = _screen_command(command)
    if not allowed:
        print(f"[tools] BLOCKED command ({matched}): {command}")
        return {
            "success": False,
            "message": (
                f"Blocked: that command runs {matched}, which Deo never does. Tell the "
                "user exactly what was requested; they have to run it themselves if "
                "they meant it. Do not try to work around this with a different command."
            ),
        }

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=120
        )
        output = (result.stdout or "").strip()
        error = (result.stderr or "").strip()

        if result.returncode != 0:
            return {
                "success": False,
                "message": f"Command failed (exit code {result.returncode}): {error or output or 'no output'}",
            }
        return {
            "success": True,
            "message": output[:1500] if output else "Command ran successfully with no output.",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Command timed out after 2 minutes."}
    except Exception as e:
        return {"success": False, "message": f"Error running command: {e}"}


# ---------------------------------------------------------------------------
# open_app -- protocol handler first, then path registry, then PATH lookup.
# Bare os.startfile(app_name) on a name like "steam" is what was failing before --
# Windows has no idea what "steam" means unless it's a real exe on PATH.
# ---------------------------------------------------------------------------

def _resolve_registry_key(name: str, candidates) -> str:
    """Matches a spoken/transcribed name against a set of known registry
    keys, tolerating near-misses. STT mishears short proper nouns often
    enough (e.g. "stem" for "steam") that an exact-match-only lookup fails
    more than it should -- initial_prompt in stt.py cuts most of these off
    at the source, but this is the safety net for whatever still slips
    through. Falls back to the original name if nothing is close enough."""
    key = name.strip().lower()
    if key in candidates:
        return key
    close = difflib.get_close_matches(key, candidates, n=1, cutoff=0.6)
    return close[0] if close else key


def _app_key(name: str) -> str:
    """Spoken and installed names reduced to the same shape: "F1® 25" and the
    transcript "f125" both become "f125", "Assetto Corsa" -> "assettocorsa"."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


_installed_apps = None


def _steam_library_dirs() -> list:
    steam = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Steam")
    dirs = [steam]
    try:
        with open(os.path.join(steam, "steamapps", "libraryfolders.vdf"), encoding="utf-8") as f:
            for path in re.findall(r'"path"\s+"([^"]+)"', f.read()):
                dirs.append(path.replace("\\\\", "\\"))
    except OSError:
        pass
    return list(dict.fromkeys(os.path.normcase(d) for d in dirs))


def _discover_installed_apps() -> dict:
    """{normalised name: (display name, thing to os.startfile)} for everything
    installed that the hand-written registries in config.py don't cover:
    Steam games (launched through steam://rungameid/<id>) and every Start Menu
    and desktop shortcut. Built once per run -- it takes a few milliseconds."""
    apps = {}

    def add(display, target):
        # Spoken back to you, so drop ®/™ and anything that didn't decode --
        # "F1® 25" otherwise came out as "F1� 25".
        display = " ".join(re.sub(r"[^\w\s:'&.+-]", "", display).split())
        key = _app_key(display)
        if key and "uninstall" not in key and key not in apps:
            apps[key] = (display, target)

    for library in _steam_library_dirs():
        for manifest in glob.glob(os.path.join(library, "steamapps", "appmanifest_*.acf")):
            try:
                with open(manifest, encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError:
                continue
            name = re.search(r'"name"\s+"([^"]+)"', text)
            appid = re.search(r'"appid"\s+"(\d+)"', text)
            if name and appid and "redistributable" not in name.group(1).lower():
                add(name.group(1), f"steam://rungameid/{appid.group(1)}")

    roots = [
        os.path.join(os.environ.get("ProgramData", r"C:\ProgramData"), r"Microsoft\Windows\Start Menu\Programs"),
        os.path.join(os.environ.get("APPDATA", ""), r"Microsoft\Windows\Start Menu\Programs"),
        os.path.join(os.path.expanduser("~"), "Desktop"),
        os.path.join(os.environ.get("PUBLIC", r"C:\Users\Public"), "Desktop"),
    ]
    for root in roots:
        for pattern in ("*.lnk", "*.url"):
            for shortcut in glob.glob(os.path.join(root, "**", pattern), recursive=True):
                add(os.path.splitext(os.path.basename(shortcut))[0], shortcut)
    return apps


def _find_installed_app(app_name: str):
    """(display name, target) for the best installed match, or None."""
    global _installed_apps
    if _installed_apps is None:
        _installed_apps = _discover_installed_apps()
    key = _app_key(app_name)
    if not key:
        return None
    if key in _installed_apps:
        return _installed_apps[key]
    # Stricter than the registry's 0.6: there are a couple of hundred names
    # here, and opening the wrong program is worse than asking again.
    close = difflib.get_close_matches(key, list(_installed_apps), n=1, cutoff=0.8)
    if close:
        return _installed_apps[close[0]]
    # People say the start of long titles: "batman" for "Batman: Arkham Asylum
    # GOTY Edition". Only when exactly one installed name starts that way.
    if len(key) >= 4:
        starts = [k for k in _installed_apps if k.startswith(key)]
        if len(starts) == 1:
            return _installed_apps[starts[0]]
    return None


def open_app(app_name: str) -> dict:
    """Open an application, trying a protocol handler, then a known path, then
    installed games and shortcuts, then PATH lookup."""
    known_keys = set(config.APP_PROTOCOL_REGISTRY) | set(config.APP_PATH_REGISTRY)

    # Something actually installed under (nearly) that name beats the fuzzy
    # registry match below, which would otherwise happily "correct" a game's
    # name into one of the handful of apps config.py knows about.
    if app_name.strip().lower() not in known_keys:
        found = _find_installed_app(app_name)
        if found:
            display, target = found
            try:
                os.startfile(target)
                Memory.log_app_usage(display)
                return {"success": True, "message": f"Opened {display}."}
            except OSError as e:
                return {"success": False, "message": f"Found {display} but it failed to launch: {e}"}

    key = _resolve_registry_key(app_name, known_keys)
    corrected = f" (matched '{app_name}' to '{key}')" if key != app_name.strip().lower() else ""

    # 1. Protocol handler (steam://, spotify:, discord://, etc.)
    protocol = config.APP_PROTOCOL_REGISTRY.get(key)
    if protocol:
        try:
            os.startfile(protocol)
            Memory.log_app_usage(key)
            return {"success": True, "message": f"Opened {key} via protocol handler.{corrected}"}
        except OSError as e:
            # fall through to other methods instead of giving up
            last_error = f"Protocol handler failed: {e}"
        except Exception as e:
            last_error = f"Protocol handler failed: {e}"
    else:
        last_error = None

    # 2. Known full path from registry
    path_template = config.APP_PATH_REGISTRY.get(key)
    if path_template:
        resolved_path = path_template.replace("{username}", os.environ.get("USERNAME", ""))
        if os.path.exists(resolved_path):
            try:
                subprocess.Popen([resolved_path])
                Memory.log_app_usage(key)
                return {"success": True, "message": f"Opened {key} from {resolved_path}.{corrected}"}
            except Exception as e:
                last_error = f"Found path but failed to launch: {e}"
        else:
            last_error = f"Registered path does not exist: {resolved_path}"
            # Hand-written paths go stale on updates -- PyCharm's moved to a
            # "...Edition 2025.2.6" folder and "open pycharm" silently broke.
            # The Start Menu always knows where the current install is.
            found = _find_installed_app(app_name)
            if found:
                display, target = found
                try:
                    os.startfile(target)
                    Memory.log_app_usage(display)
                    return {"success": True, "message": f"Opened {display}."}
                except OSError as e:
                    last_error = f"Found {display} but it failed to launch: {e}"

    # 3. PATH lookup -- only works if the app's exe is actually on system PATH
    resolved = shutil.which(key) or shutil.which(app_name)
    if resolved:
        try:
            subprocess.Popen([resolved])
            return {"success": True, "message": f"Opened {key} (found on PATH at {resolved}).{corrected}"}
        except Exception as e:
            last_error = f"Found on PATH but failed to launch: {e}"

    # 4. Genuine failure -- report it honestly, don't pretend it worked
    return {
        "success": False,
        "message": (
            f"Could not open {app_name}{corrected}. No protocol handler, registry entry, or PATH match found. "
            f"{last_error or ''} Add it to APP_PATH_REGISTRY or APP_PROTOCOL_REGISTRY in config.py with its real path."
        ),
    }


def open_url_in_browser(url: str, browser_name: str = "") -> dict:
    """Open a URL, optionally in a specific browser (e.g. 'brave', 'chrome')."""
    if browser_name.strip():
        browser_key = _resolve_registry_key(browser_name, set(config.APP_PATH_REGISTRY))
        browser_path = config.APP_PATH_REGISTRY.get(browser_key)
        if browser_path and os.path.exists(browser_path):
            try:
                subprocess.Popen([browser_path, url])
                return {"success": True, "message": f"Opened {url} in {browser_key}."}
            except Exception as e:
                return {"success": False, "message": f"Found {browser_key} but failed to launch it: {e}"}
        else:
            return {
                "success": False,
                "message": f"Don't have a registered path for browser '{browser_name}'. Add it to APP_PATH_REGISTRY.",
            }

    try:
        webbrowser.open(url)
        return {"success": True, "message": f"Opened {url} in your default browser."}
    except Exception as e:
        return {"success": False, "message": f"Failed to open URL: {e}"}


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

def web_search(query: str) -> dict:
    """Search the web and return a short summary of top results."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return {"success": False, "message": "No results found."}
        summary = "\n".join(f"- {r['title']}: {r['body'][:200]}" for r in results)
        return {"success": True, "message": summary}
    except Exception as e:
        return {"success": False, "message": f"Search failed: {e}"}


# ---------------------------------------------------------------------------
# YouTube: find latest video from a channel, optionally open in a specific browser
# ---------------------------------------------------------------------------

def get_latest_youtube_video(channel_name: str) -> dict:
    """Look up a YouTube channel's most recent uploaded video and return its URL."""
    if not config.YOUTUBE_API_KEY:
        return {
            "success": False,
            "message": "No YouTube API key configured. Set YOUTUBE_API_KEY in your .env file.",
        }

    try:
        # Step 1: resolve channel name -> channel ID
        search_resp = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": channel_name,
                "type": "channel",
                "maxResults": 1,
                "key": config.YOUTUBE_API_KEY,
            },
            timeout=10,
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()
        items = search_data.get("items", [])
        if not items:
            return {"success": False, "message": f"Couldn't find a YouTube channel named '{channel_name}'."}
        channel_id = items[0]["snippet"]["channelId"]
        resolved_channel_title = items[0]["snippet"]["channelTitle"]

        # Step 2: get the channel's uploads playlist
        channels_resp = requests.get(
            "https://www.googleapis.com/youtube/v3/channels",
            params={"part": "contentDetails", "id": channel_id, "key": config.YOUTUBE_API_KEY},
            timeout=10,
        )
        channels_resp.raise_for_status()
        channels_data = channels_resp.json()
        channel_items = channels_data.get("items", [])
        if not channel_items:
            return {"success": False, "message": f"Found channel '{resolved_channel_title}' but couldn't read its uploads."}
        uploads_playlist_id = channel_items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

        # Step 3: get most recent video from that playlist
        playlist_resp = requests.get(
            "https://www.googleapis.com/youtube/v3/playlistItems",
            params={
                "part": "snippet",
                "playlistId": uploads_playlist_id,
                "maxResults": 1,
                "key": config.YOUTUBE_API_KEY,
            },
            timeout=10,
        )
        playlist_resp.raise_for_status()
        playlist_data = playlist_resp.json()
        playlist_items = playlist_data.get("items", [])
        if not playlist_items:
            return {"success": False, "message": f"'{resolved_channel_title}' has no videos."}

        video_id = playlist_items[0]["snippet"]["resourceId"]["videoId"]
        video_title = playlist_items[0]["snippet"]["title"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        return {
            "success": True,
            "message": f"Latest video from {resolved_channel_title}: '{video_title}'. URL: {video_url}",
            "url": video_url,
        }

    except requests.exceptions.RequestException as e:
        return {"success": False, "message": f"YouTube API request failed: {e}"}
    except Exception as e:
        return {"success": False, "message": f"Unexpected error looking up YouTube video: {e}"}


def open_latest_youtube_video(channel_name: str, browser_name: str = "") -> dict:
    """Find a channel's latest video and open it in the specified (or default) browser."""
    lookup = get_latest_youtube_video(channel_name)
    if not lookup["success"]:
        return lookup  # propagate the real failure reason instead of pretending it opened

    return open_url_in_browser(lookup["url"], browser_name)


# ---------------------------------------------------------------------------
# move_active_window -- OS-level window-to-monitor control. Used by both the
# gesture listener (direct call, no LLM round-trip needed for a swipe) and
# the LLM (via the tool schema below, e.g. "move this to the other screen").
# ---------------------------------------------------------------------------

def _monitors_sorted_by_x() -> list:
    """Monitor work areas (left, top, right, bottom) -- the screen minus the
    taskbar -- in virtual-screen coordinates, sorted left-to-right. Only
    meaningful in real pixels, i.e. inside _per_monitor_dpi()."""
    rects = [win32api.GetMonitorInfo(m[0])["Work"] for m in win32api.EnumDisplayMonitors()]
    rects.sort(key=lambda r: r[0])
    return rects


class _per_monitor_dpi:
    """Run a block in real-pixel (per-monitor DPI aware) coordinates.

    Deo's process isn't DPI aware, so Windows hands it *scaled* geometry. With
    the left screen at 200% and the right at 100%, that came back as a mix of
    real and scaled numbers -- the left screen looked like 1280x800 -- and a
    window moved there arrived at the wrong size, then rescaled itself again
    on landing until it spanned both screens. Only this thread switches, and
    only for the move, so Deo's own window keeps normal scaling."""

    def __enter__(self):
        import ctypes
        try:
            self._prev = ctypes.windll.user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
        except Exception:
            self._prev = None
        return self

    def __exit__(self, *exc):
        import ctypes
        if self._prev:
            ctypes.windll.user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(self._prev))
        return False


def _monitor_index_for_window(hwnd, monitors) -> int:
    rect = win32gui.GetWindowRect(hwnd)
    cx = (rect[0] + rect[2]) // 2
    cy = (rect[1] + rect[3]) // 2
    for i, (left, top, right, bottom) in enumerate(monitors):
        if left <= cx < right and top <= cy < bottom:
            return i
    return 0  # fallback -- window center didn't land inside any monitor rect


# Never closed by voice, whatever the model asks: the shell, Deo itself (it
# runs under python), and the app this conversation is likely happening in.
_CLOSE_PROTECTED = {"explorer", "python", "pythonw", "dwm", "csrss", "winlogon",
                    "taskmgr", "claude", "systemsettings", "searchhost", "shellhost"}


def _process_exe(pid: int) -> str:
    """Full exe path for a pid, or "" if Windows won't say (elevated/system)."""
    import ctypes
    from ctypes import wintypes
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        ok = ctypes.windll.kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        return buf.value if ok else ""
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def _windows_matching(app_name: str) -> list:
    """[(hwnd, exe stem, title)] for visible top-level windows that belong to
    the named app -- matched on the exe name ("brave" -> brave.exe) or the
    window title ("F1 25" -> "F1® 25")."""
    import win32process
    key = _app_key(app_name)
    if len(key) < 3:
        return []
    found = []

    def visit(hwnd, _):
        title = win32gui.GetWindowText(hwnd)
        if not (win32gui.IsWindowVisible(hwnd) and title):
            return True
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        exe = os.path.splitext(os.path.basename(_process_exe(pid)))[0]
        exe_key = _app_key(exe)
        if exe.lower() in _CLOSE_PROTECTED:
            return True
        if key == exe_key or key in exe_key or (len(exe_key) >= 4 and exe_key in key) \
                or key in _app_key(title):
            found.append((hwnd, exe, title))
        return True

    win32gui.EnumWindows(visit, None)
    return found


def close_app(app_name: str) -> dict:
    """Close a running app the way clicking its X would.

    WM_CLOSE, not a kill: the app gets to save state or ask about unsaved work,
    exactly as if you'd closed it yourself. That's also why this can't be used
    to take down anything that ignores a polite request -- deliberately, given
    that untrusted text (web pages, screen OCR) reaches the same model."""
    windows = _windows_matching(app_name)
    if not windows:
        return {"success": False, "message": f"No open window found for '{app_name}'."}
    for hwnd, _, _ in windows:
        try:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        except Exception:
            pass
    exes = sorted({exe for _, exe, _ in windows})
    return {"success": True,
            "message": f"Asked {len(windows)} window(s) of {', '.join(exes)} to close."}


def _resolve_user_path(path: str):
    """A spoken file reference -> an existing absolute path, or None. Bare
    names ("test.txt") are looked for in the folders people actually mean:
    DeoProjects (where Deo writes code), Downloads, Desktop, Documents, then
    the home folder."""
    path = os.path.expandvars(os.path.expanduser(path.strip().strip('"')))
    # Whisper writes file names the way they're said: "input underscore
    # output dot py". The model usually converts that, but not always.
    if not os.path.exists(path) and re.search(r"\b(underscore|dot|dash)\b", path, re.I):
        spoken = re.sub(r"\s*\bunderscore\b\s*", "_", path, flags=re.I)
        spoken = re.sub(r"\s*\bdash\b\s*", "-", spoken, flags=re.I)
        spoken = re.sub(r"\s*\bdot\b\s*", ".", spoken, flags=re.I)
        path = spoken.strip()
    if os.path.isabs(path):
        return path if os.path.exists(path) else None
    home = os.path.expanduser("~")
    for base in ("DeoProjects", "Downloads", "Desktop", "Documents", ""):
        candidate = os.path.join(home, base, path)
        if os.path.exists(candidate):
            return candidate
    return None


def delete_file(path: str) -> dict:
    """Send a file or folder to the Recycle Bin -- never a permanent delete,
    so even an approved mistake can be restored. The permission dialog has
    already been shown and accepted by the time this runs (permissions.py)."""
    from win32com.shell import shell, shellcon
    target = _resolve_user_path(path)
    if not target:
        return {"success": False,
                "message": f"Couldn't find '{path}' (looked in Downloads, Desktop, Documents and "
                           f"the home folder). Use search_files to find the exact path."}
    home = os.path.normcase(os.path.expanduser("~"))
    if not os.path.normcase(os.path.abspath(target)).startswith(home + os.sep):
        return {"success": False, "message": f"Refused: {target} is outside your user folder."}
    flags = shellcon.FOF_ALLOWUNDO | shellcon.FOF_NOCONFIRMATION | shellcon.FOF_SILENT | shellcon.FOF_NOERRORUI
    code, aborted = shell.SHFileOperation((0, shellcon.FO_DELETE, target, None, flags, None, None))
    if code != 0 or aborted or os.path.exists(target):
        return {"success": False, "message": f"Couldn't delete {target} (error {code})."}
    return {"success": True, "message": f"Moved {target} to the Recycle Bin."}


def _shortcut_target(path: str) -> str:
    """The exe behind a .lnk, so an editor can be launched WITH a file
    argument (os.startfile on the shortcut can't pass one)."""
    if not path.lower().endswith(".lnk"):
        return path
    try:
        import win32com.client
        return win32com.client.Dispatch("WScript.Shell").CreateShortcut(path).TargetPath or path
    except Exception:
        return path


def _unescape_code(code: str) -> str:
    """The local model sometimes double-escapes its newlines, sending the two
    characters backslash-n instead of a line break -- the first script Deo
    wrote came out as one long line of literal \\n's. Only undone when the code
    has NO real line breaks at all, so a genuine "\\n" inside a string in an
    otherwise multi-line program is left alone."""
    if "\n" not in code and "\\n" in code:
        code = code.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    return code


def create_code_file(filename: str, code: str, editor: str = "pycharm") -> dict:
    """Write code into a NEW file under ~/DeoProjects and open it in an editor.

    Never overwrites: an existing name gets _2, _3... appended, which is why
    this needs no permission dialog -- nothing that exists is ever touched."""
    folder = os.path.join(os.path.expanduser("~"), "DeoProjects")
    os.makedirs(folder, exist_ok=True)
    name = os.path.basename(filename.strip()) or "script.py"
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    if "." not in name:
        name += ".py"
    stem, ext = os.path.splitext(name)
    path, n = os.path.join(folder, name), 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{stem}_{n}{ext}")
        n += 1
    code = _unescape_code(code)
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(code.rstrip() + "\n")
    except OSError as e:
        return {"success": False, "message": f"Couldn't write {path}: {e}"}

    syntax_note = ""
    if path.endswith(".py"):
        try:
            compile(code, path, "exec")
        except SyntaxError as e:
            syntax_note = (f" WARNING: the code has a syntax error on line {e.lineno} ({e.msg}) -- "
                           f"tell the user, and offer to fix it.")

    opened = "but couldn't find an editor to open it in"
    found = _find_installed_app(editor) if editor.strip() else None
    if found:
        display, target = found
        exe = _shortcut_target(target)
        try:
            if exe.lower().endswith(".exe"):
                subprocess.Popen([exe, path])
            else:
                os.startfile(path)
            opened = f"and opened it in {display}"
        except Exception as e:
            opened = f"but {display} failed to open it: {e}"
    else:
        try:
            os.startfile(path)
            opened = "and opened it with the default program"
        except OSError:
            pass
    return {"success": True, "message": f"Created {path} {opened}.{syntax_note}"}


def run_code_file(path: str, input_text: str = "") -> dict:
    """Run a script in its own visible terminal window that stays open, so
    you can type into input() prompts and read the output. run_command can't
    do that: it runs hidden with no keyboard, so an input() just hangs.
    Scripts outside ~/DeoProjects need a click first (permissions.py)."""
    target = _resolve_user_path(path) or _resolve_user_path(os.path.join("DeoProjects", path))
    if not target or not os.path.isfile(target):
        return {"success": False, "message": f"Couldn't find the file '{path}'."}
    runners = {".py": [sys.executable if not sys.executable.lower().endswith("pythonw.exe") else "python"],
               ".js": ["node"], ".bat": [], ".cmd": [], ".ps1": ["powershell", "-ExecutionPolicy", "Bypass", "-File"]}
    ext = os.path.splitext(target)[1].lower()
    if ext not in runners:
        return {"success": False, "message": f"Don't know how to run {ext} files."}
    folder = os.path.dirname(target)

    if input_text:
        # You told Deo what to type ("run it and enter hello"): feed that in
        # as the keyboard input and hand back what the script printed.
        try:
            result = subprocess.run(runners[ext] + [target] if runners[ext] else ["cmd", "/c", target],
                                    cwd=folder, input=input_text.replace(";", "\n") + "\n",
                                    capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return {"success": False, "message": "The script was still running after 30 seconds -- stopped it."}
        output = (result.stdout or "").strip()
        errors = (result.stderr or "").strip()
        status = "finished" if result.returncode == 0 else f"failed (exit code {result.returncode})"
        return {"success": result.returncode == 0,
                "message": f"Ran {os.path.basename(target)} with input {input_text!r}; it {status}. "
                           f"Output:\n{output or '(nothing)'}" + (f"\nErrors:\n{errors}" if errors else "")}

    # cmd /k keeps the window open after the script finishes, so the output
    # doesn't vanish the instant it's printed.
    inner = subprocess.list2cmdline(runners[ext] + [target])
    try:
        subprocess.Popen(["cmd", "/k", inner], cwd=folder,
                         creationflags=subprocess.CREATE_NEW_CONSOLE)
    except Exception as e:
        return {"success": False, "message": f"Couldn't start it: {e}"}
    return {"success": True,
            "message": f"Running {os.path.basename(target)} in a new terminal window -- "
                       f"the user can type into it there."}


def look_at_screen(question: str = "") -> dict:
    """Imported lazily: screen_qa pulls in mss, PIL and the TTS module, none of
    which a turn that never looks at the screen should pay for."""
    from .screen_qa import describe_screen
    return describe_screen(question)


# Windows' own panels. The first live gesture test moved "Quick settings"
# to the other monitor, because that flyout was technically in front.
_SHELL_EXES = {"shellexperiencehost", "shellhost", "searchhost", "startmenuexperiencehost",
               "textinputhost", "lockapp"}
_SHELL_CLASSES = {"Shell_TrayWnd", "Shell_SecondaryTrayWnd", "Progman", "WorkerW",
                  "NotifyIconOverflowWindow", "TopLevelWindowForOverflowXamlIsland"}


def _is_app_window(hwnd) -> bool:
    if not (hwnd and win32gui.IsWindow(hwnd) and win32gui.IsWindowVisible(hwnd)):
        return False
    if not win32gui.GetWindowText(hwnd) or win32gui.GetClassName(hwnd) in _SHELL_CLASSES:
        return False
    if win32gui.GetWindowPlacement(hwnd)[1] == win32con.SW_SHOWMINIMIZED:
        return False
    import win32process
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    # Deo's own window. A move ends with SetForegroundWindow on what it moved,
    # so once Deo's window got moved even once it stayed in front, and every
    # later swipe moved Deo again instead of the app you were using.
    if pid == os.getpid():
        return False
    exe = os.path.splitext(os.path.basename(_process_exe(pid)))[0].lower()
    return exe not in _SHELL_EXES


def _frontmost_app_window():
    """The focused window if it's a real app, otherwise the next real app
    window down the z-order (EnumWindows walks top to bottom)."""
    hwnd = win32gui.GetForegroundWindow()
    if _is_app_window(hwnd):
        return hwnd
    found = []

    def visit(h, _):
        if _is_app_window(h):
            found.append(h)
            return False   # stop at the first one
        return True

    try:
        win32gui.EnumWindows(visit, None)
    except Exception:
        pass   # EnumWindows raises when the callback stops it early
    return found[0] if found else None


def move_active_window(direction: str) -> dict:
    """Move the currently focused top-level window onto a neighboring
    monitor. direction: 'left', 'right', or 'other' (whichever monitor the
    window ISN'T currently mostly on -- handy default for a 2-monitor setup)."""
    with _per_monitor_dpi():
        return _move_active_window(direction)


def _move_active_window(direction: str) -> dict:
    direction = direction.strip().lower()
    if direction not in ("left", "right", "other"):
        return {"success": False, "message": f"Unknown direction '{direction}' -- use 'left', 'right', or 'other'."}

    try:
        hwnd = _frontmost_app_window()
        if not hwnd:
            return {"success": False, "message": "Couldn't find an app window to move."}

        title = win32gui.GetWindowText(hwnd) or "(untitled window)"
        monitors = _monitors_sorted_by_x()
        if len(monitors) < 2:
            return {"success": False, "message": "Only one monitor detected -- nothing to move to."}

        current_idx = _monitor_index_for_window(hwnd, monitors)

        if direction == "left":
            target_idx = max(0, current_idx - 1)
        elif direction == "right":
            target_idx = min(len(monitors) - 1, current_idx + 1)
        else:  # "other"
            target_idx = 0 if current_idx != 0 else 1

        if target_idx == current_idx:
            return {"success": False, "message": f"'{title}' is already the closest thing to the {direction} monitor -- nothing to do."}

        left, top, right, bottom = monitors[target_idx]
        mon_w, mon_h = right - left, bottom - top

        # SetWindowPos silently ignores the new position on a maximized
        # window, so it has to be restored first or the "move" does nothing.
        # GetWindowPlacement, not win32gui.IsZoomed: this pywin32 build doesn't
        # have IsZoomed, so every move crashed with an AttributeError here.
        if win32gui.GetWindowPlacement(hwnd)[1] == win32con.SW_SHOWMAXIMIZED:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

        # Re-fit at ~85% of the target monitor, centered, rather than
        # keeping whatever size it happened to be on the old display.
        w, h = int(mon_w * 0.85), int(mon_h * 0.85)
        x = left + (mon_w - w) // 2
        y = top + (mon_h - h) // 2

        win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, x, y, w, h, win32con.SWP_SHOWWINDOW)
        # Crossing to a screen with different scaling makes the app resize
        # itself for the new DPI (WM_DPICHANGED) right after the first move,
        # which is what blew windows up to span both screens. Give it a moment
        # to do that, then put the size back to what was asked for.
        time.sleep(0.15)
        win32gui.SetWindowPos(hwnd, 0, x, y, w, h, win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE)
        win32gui.SetForegroundWindow(hwnd)

        return {"success": True, "message": f"Moved '{title}' to the {direction} monitor."}
    except Exception as e:
        if getattr(e, "winerror", None) == 5 or (e.args and e.args[0] == 5):
            return {"success": False,
                    "message": f"Windows won't let Deo move '{title}' -- it's running as "
                               f"administrator, and normal apps can't move those."}
        return {"success": False, "message": f"Failed to move window: {e}"}


# ---------------------------------------------------------------------------
# Tool schemas (what the LLM sees) and registry (what actually runs)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a Windows shell/PowerShell command and return whether it succeeded plus its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Open an installed application or game by name, e.g. 'steam', 'spotify', 'brave', 'F1 25', 'Assetto Corsa'. Knows every Steam game and Start Menu/desktop shortcut on this PC, so always try it rather than saying an app is unknown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "description": "Name of the app to open"}
                },
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url_in_browser",
            "description": "Open a specific URL, optionally in a named browser (e.g. 'brave'). Omit browser_name to use the default browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The full URL to open"},
                    "browser_name": {"type": "string", "description": "Optional browser name, e.g. 'brave', 'chrome'"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information and return a brief summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_latest_youtube_video",
            "description": "Look up the most recent video uploaded by a YouTube channel and return its title and URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_name": {"type": "string", "description": "Name of the YouTube channel"}
                },
                "required": ["channel_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_latest_youtube_video",
            "description": "Find a YouTube channel's latest video and open it directly in a specific browser (or default if none given). Use this for requests like 'open the latest video from X on Brave'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_name": {"type": "string", "description": "Name of the YouTube channel"},
                    "browser_name": {"type": "string", "description": "Optional browser to open it in, e.g. 'brave'"},
                },
                "required": ["channel_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "Store a personal fact about the user for future sessions, e.g. their name or a preference. Use when the user tells you something worth remembering long-term.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Short label for the fact, e.g. 'name' or 'favorite_browser'"},
                    "value": {"type": "string", "description": "The fact itself"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a note or reminder the user wants to keep for later, e.g. a task or something to remember.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "The note or reminder text"}
                },
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_fact",
            "description": "Remove a previously remembered fact by its key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The key of the fact to forget"}
                },
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_all",
            "description": "Return everything currently stored in long-term memory (facts, notes, app usage). Use when the user asks what you remember about them.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_code_file",
            "description": "Write code or text into a NEW file and open it in an editor (PyCharm by default, or 'vscode'). Use this whenever the user asks you to write, make or create a program, script or file. Write the complete code in the 'code' argument. Never overwrites existing files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File name, e.g. 'echo_input.py'"},
                    "code": {"type": "string", "description": "The full contents of the file"},
                    "editor": {"type": "string", "description": "'pycharm' (default) or 'visual studio code'"}
                },
                "required": ["filename", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_code_file",
            "description": "Run a script (.py, .js, .bat, .ps1) in a new visible terminal window that stays open, so the user can type into input() prompts and see the output. Use this whenever the user asks to run code or a script you created -- NOT run_command, which can't take keyboard input. Bare names are found in DeoProjects, Downloads, Desktop and Documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Script name or path, e.g. 'input_output.py'"},
                    "input_text": {"type": "string", "description": "Optional: what to type into the script's input prompts (separate several answers with ';'). When given, the script runs with that input and you get its output back to tell the user. Leave empty to open an interactive window instead."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file or folder: it goes to the Recycle Bin (restorable), and the user confirms in a popup first. Pass the file name or path straight away, e.g. 'test.txt' or '~/Downloads/old.zip' -- bare names are found in Downloads, Desktop and Documents automatically, so do NOT list folders first. Use this for every delete request instead of run_command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File/folder name or full path"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look_at_screen",
            "description": "Take a screenshot of the user's main screen and answer a question about it with a vision model. Use whenever the user asks what's on their screen, to read something shown, or about what they're looking at. Slow (several seconds) -- only when needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "What to find out from the screen, in the user's words"}
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_app",
            "description": "Close (quit) an app or game that is currently running, e.g. 'brave', 'steam', 'F1 25'. Use this for any request to close, quit or exit an app -- never open_app.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "description": "Name of the running app to close"}
                },
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_active_window",
            "description": "Move the currently focused window to a neighboring monitor. Use when the user asks to move the current window/app/tab to another screen or monitor, e.g. 'move this to the other screen', 'send it to the left monitor'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["left", "right", "other"],
                        "description": "'left'/'right' moves to the neighboring monitor in that direction. 'other' moves to whichever monitor it isn't currently mostly on -- use this on a 2-monitor setup when the user doesn't specify a side.",
                    }
                },
                "required": ["direction"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "run_command": run_command,
    "open_app": open_app,
    "open_url_in_browser": open_url_in_browser,
    "web_search": web_search,
    "get_latest_youtube_video": get_latest_youtube_video,
    "open_latest_youtube_video": open_latest_youtube_video,
    "remember_fact": Memory.remember_fact,
    "save_note": Memory.save_note,
    "forget_fact": Memory.forget_fact,
    "recall_all": Memory.recall_all,
    "close_app": close_app,
    "look_at_screen": look_at_screen,
    "delete_file": delete_file,
    "create_code_file": create_code_file,
    "run_code_file": run_code_file,
    "move_active_window": move_active_window,
}