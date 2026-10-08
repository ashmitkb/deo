"""
Start Deo with no console window -- what the Windows Startup shortcut runs.

Everything Deo prints goes to logs/deo.log instead (the previous run is kept
as logs/deo.previous.log). Run it by hand with:

    .venv\\Scripts\\pythonw.exe run_deo.pyw

To watch it live:  Get-Content logs\\deo.log -Wait     (PowerShell)
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

log_dir = os.path.join(ROOT, "logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "deo.log")
if os.path.exists(log_path):
    os.replace(log_path, os.path.join(log_dir, "deo.previous.log"))

# pythonw has no console: sys.stdout/stderr are None, and print() would go
# nowhere. Line-buffered so the log is readable while Deo runs.
log = open(log_path, "a", encoding="utf-8", buffering=1)
sys.stdout = sys.stderr = log

import main  # noqa: E402

main.main(start_hidden="--hidden" in sys.argv)
