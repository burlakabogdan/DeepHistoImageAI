@echo off

set VENV_PATH=%~dp0.venv
set SCRIPT_PATH=%~dp0main.py


if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo Error: Virtual environment not found in %VENV_PATH%
    exit /b 1
)


call "%VENV_PATH%\Scripts\activate.bat"


if not exist "%SCRIPT_PATH%" (
    echo Error: Python script not found at %SCRIPT_PATH%
    exit /b 1
)


python "%SCRIPT_PATH%"


call "%VENV_PATH%\Scripts\deactivate.bat"
