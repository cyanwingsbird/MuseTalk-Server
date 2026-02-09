@echo off
setlocal enabledelayedexpansion

echo [INFO] Starting MuseTalk Server...

:: Get project root directory
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

:: Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at %PROJECT_ROOT%venv
    echo [INFO] Please run the setup/verification script first.
    pause
    exit /b 1
)

:: Set PYTHONPATH to include project root (.) and MuseTalk submodule (./MuseTalk)
set "PYTHONPATH=%PROJECT_ROOT%;%PROJECT_ROOT%MuseTalk"
echo [INFO] PYTHONPATH set to: %PYTHONPATH%

:: Change directory to MuseTalk so relative model paths resolve correctly
cd /d "%PROJECT_ROOT%MuseTalk"
echo [INFO] Working Directory: %CD%

:: Run the server using the venv python
echo [INFO] Server is starting... (This may take nearly a minute to load models)
echo [INFO] Please keep this window open.
"%PROJECT_ROOT%venv\Scripts\python.exe" -m musetalk_server.app

endlocal
pause