@echo off
rem Manual wheel: stop the run at the current wait (the robot is holding still). Copy to the desktop; double-click.
type nul > "%USERPROFILE%\EW_STOP"
echo stop sent
timeout /t 2 >nul
