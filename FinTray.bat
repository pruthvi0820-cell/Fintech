@echo off
rem Double-click to start FinTray. Close this window (or press Ctrl+C) to stop it.
title FinTray
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo Could not find the .venv folder in %CD%
  echo Set it up first:  py -3.12 -m venv .venv  then  pip install -e ".[dev,app]"
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
echo Starting FinTray... your browser will open at http://localhost:8501
echo Keep this window open while you use FinTray. Close it to stop.
streamlit run scripts\app.py
pause
