@echo off
rem Manual wheel (robot_dobot/manual_wheel.py): "the wheel is loaded / unloaded, my hands are clear" -> the run goes on.
rem Accepted only if the load cell is in the window. Copy to the desktop; double-click.
type nul > "%USERPROFILE%\EW_RESUME"
echo resume sent
timeout /t 2 >nul
