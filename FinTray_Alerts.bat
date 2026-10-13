@echo off
rem Double-click to send FinTray alerts to Telegram while this window is open (market hours, every 15 min).
title FinTray Alerts
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo Could not find the .venv folder in %CD%
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
echo.
echo  ============================================================
echo   FINTRAY PHONE ALERTS. KEEP THIS WINDOW OPEN (minimise it).
echo   Every 15 minutes in market hours (9:15-15:30 IST, Mon-Fri)
echo   it checks and prints one line, e.g. "10:15 no new alerts".
echo   Open paper trades are checked every minute; a close is
echo   sent to Telegram straight away.
echo   The repeating line is normal: it is the watcher working.
echo   Closing it stops Telegram alerts.
echo  ============================================================
echo.
python scripts\alerts_runner.py
pause
