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
echo.
echo  ============================================================
echo   THIS WINDOW IS FINTRAY ITSELF. KEEP IT OPEN (minimise it).
echo   It shows a few lines and then waits: that is normal.
echo   Closing it switches FinTray off.
echo  ============================================================
echo.
streamlit run scripts\app.py
pause
