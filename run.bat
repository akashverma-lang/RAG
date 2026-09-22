@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   Local RAG
echo ============================================
echo.

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: run setup.bat first.
        pause
        exit /b 1
    )
    set "PY=python"
)

echo Starting. Your browser opens by itself in a moment.
echo Leave this window open; closing it stops the app.
echo.

"%PY%" launcher.py

echo.
echo Stopped.
pause
