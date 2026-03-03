@echo off
setlocal

set "ROOT_DIR=%~dp0"
set "MUSETALK_DIR=%ROOT_DIR%MuseTalk"

if not exist "%MUSETALK_DIR%" (
    echo ERROR: MuseTalk directory not found at %MUSETALK_DIR%
    echo Ensure the MuseTalk submodule is initialized.
    exit /b 1
)

:: Set PYTHONPATH to include both MuseTalk and project root
set "PYTHONPATH=%MUSETALK_DIR%;%ROOT_DIR%;%PYTHONPATH%"

:: Must run from MuseTalk directory (upstream code uses hardcoded relative paths)
cd /d "%MUSETALK_DIR%"

echo Starting MuseTalk Server...
echo   CWD:        %CD%
echo   PYTHONPATH:  %PYTHONPATH%

python -m musetalk_server.app %*
