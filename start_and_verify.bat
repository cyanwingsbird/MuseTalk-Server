@echo off
setlocal enabledelayedexpansion

echo [INFO] Starting MuseTalk Server Verification...

:: Get project root directory (ensure it ends with backslash)
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

:: Set PYTHONPATH to include project root (.) and MuseTalk submodule (./MuseTalk)
:: This allows importing 'musetalk_server' from root and 'musetalk' from ./MuseTalk
set "PYTHONPATH=%PROJECT_ROOT%;%PROJECT_ROOT%MuseTalk"
echo [INFO] PYTHONPATH set to: %PYTHONPATH%

:: Load PORT from .env for verification
set PORT=8000
if exist ".env" (
    for /f "tokens=2 delims==" %%a in ('findstr "MUSETALK_PORT" ".env"') do set PORT=%%a
)
echo [INFO] Detected Configured Port: %PORT%

:: Change directory to MuseTalk so relative model paths (./models/...) resolve correctly
cd /d "%PROJECT_ROOT%MuseTalk"
echo [INFO] Working Directory: %CD%

:: Start the server in the background
echo [INFO] Launching server...
start /B "MuseTalk Server" "%PROJECT_ROOT%venv\Scripts\python.exe" -m musetalk_server.app > server_verify.log 2>&1
set SERVER_PID=!ERRORLEVEL!

:: Wait for server to initialize (models take time to load)
echo [INFO] Waiting 45 seconds for model loading...
ping 127.0.0.1 -n 46 >nul

:: Verify Health
echo [INFO] Checking health endpoint on port %PORT%...
curl -v http://localhost:%PORT%/health
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Server is healthy!
) else (
    echo [ERROR] Health check failed. Checking logs...
    type server_verify.log
)

:: Kill the python process (cleanup)
echo [INFO] Stopping server...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq MuseTalk Server*" >nul 2>&1
:: Force kill python processes started from this location if title match fails
taskkill /F /IM python.exe >nul 2>&1

endlocal
