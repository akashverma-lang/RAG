@echo off
setlocal
cd /d "%~dp0"

REM The progress bar draws a block character per step. cmd.exe on its default code
REM page cannot print those, so switch the console to UTF-8 first; if that fails the
REM bar falls back to plain ASCII on its own rather than erroring.
for /f "tokens=2 delims=:" %%p in ('chcp') do set "_OLDCP=%%p"
chcp 65001 >nul 2>&1

echo ============================================
echo   Local RAG  -  reading your documents
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

"%PY%" rag.py index %*
set "_RC=%ERRORLEVEL%"

if defined _OLDCP chcp %_OLDCP% >nul 2>&1

echo.
echo Tip: "index.bat --rebuild" re-reads every file from scratch.
echo      "index.bat --quiet"   hides the progress bar.
pause
exit /b %_RC%
