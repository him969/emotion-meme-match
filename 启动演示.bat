@echo off
rem ============================================================
rem  One-click launcher for the Streamlit demo.
rem  This file is ASCII-only on purpose: cmd.exe mis-parses .bat
rem  files containing Chinese text when the console code page does
rem  not match the file encoding.
rem
rem  Python is taken from the PYTHON environment variable if set,
rem  otherwise "python" from PATH is used:
rem      set "PYTHON=<full path to python.exe>"
rem  then run this .bat file again.
rem ============================================================
title Cat Meme Demo Launcher
set "PY=%PYTHON%"
if not defined PY set "PY=python"
set "ROOT=%~dp0"
set "APP="

if exist "%PY%" goto :py_ready
where %PY% >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found: %PY%
    echo   Install Python and add it to PATH, or set the PYTHON
    echo   environment variable to the full path of python.exe.
    pause
    exit /b 1
)
:py_ready

rem Find the subfolder that contains 07_demo_app.py (its name is in
rem Chinese, so we avoid hard-coding it).
for /d %%D in ("%ROOT%*") do if exist "%%D\07_demo_app.py" set "APP=%%D\07_demo_app.py"

if not defined APP (
    echo [ERROR] 07_demo_app.py was not found in any subfolder of:
    echo   %ROOT%
    pause
    exit /b 1
)

echo ============================================================
echo   Starting the demo page...
echo   After it says "You can now view your Streamlit app", open
echo   this URL in your browser:
echo       http://localhost:8501
echo   Close this window to stop the server.
echo ============================================================
echo.

"%PY%" -m streamlit run "%APP%" --server.port 8501 --server.headless true --browser.gatherUsageStats false

echo.
echo ============================================================
echo   Server stopped. If there is an error above, screenshot it.
echo ============================================================
pause
