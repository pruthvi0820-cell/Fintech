@echo off
rem Double-click to start ATS Analyzer. Close this window (or press Ctrl+C) to stop it.
title ATS Analyzer
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo First run: setting up ATS Analyzer. This takes a minute or two, only once.
  python -m venv .venv || (
    echo Could not create the .venv folder. Is Python installed and on PATH?
    pause
    exit /b 1
  )
  call ".venv\Scripts\activate.bat"
  pip install -e ".[web]" || (
    echo Install failed. Check your internet connection and run this file again.
    pause
    exit /b 1
  )
) else (
  call ".venv\Scripts\activate.bat"
)
echo Starting ATS Analyzer... your browser will open at http://127.0.0.1:8000
echo.
echo  ============================================================
echo   THIS WINDOW IS ATS ANALYZER ITSELF. KEEP IT OPEN (minimise it).
echo   Closing it switches the app off.
echo  ============================================================
echo.
rem open the browser a few seconds after the server starts
start "" /b cmd /c "timeout /t 3 /nobreak >nul & start "" http://127.0.0.1:8000"
python -m ats_analyzer.web
pause
