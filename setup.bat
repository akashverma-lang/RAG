@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   Local RAG  -  one-time setup
echo ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not on PATH.
    echo.
    echo Install Python 3.10 or newer from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM A virtual environment, so this never disturbs anything else installed on the
REM machine and can be deleted by removing one folder.
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating a private Python environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: could not create the environment.
        pause
        exit /b 1
    )
)

echo [2/3] Installing packages. The first run downloads a few hundred MB.
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: installation failed. Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo [3/3] Checking the installation...
".venv\Scripts\python.exe" rag.py doctor

echo.
echo ============================================
echo   Done. Run  run.bat  to start.
echo ============================================
echo.
echo   The first time it opens, a setup screen asks for:
echo     - the folder holding your documents
echo     - a free API key from Groq or Google Gemini
echo.
echo   Get a key at  console.groq.com/keys
echo              or aistudio.google.com/apikey
echo.
pause
