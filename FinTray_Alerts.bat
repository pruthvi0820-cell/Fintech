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
echo FinTray alerts: checking every 15 minutes, 9:15 am to 3:30 pm IST, Monday to Friday.
echo Keep this window open. Close it to stop.
python scripts\alerts_runner.py
pause
