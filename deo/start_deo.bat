@echo off
cd /d "C:\Users\ashmi\Downloads\deo"
call .venv\Scripts\activate.bat
python -W ignore::DeprecationWarning main.py

REM Keeps the window open if Deo crashes on startup, so you can read the
REM error instead of it flashing shut. Remove this line once things are
REM stable and you're ready to switch to a fully silent autostart.
pause