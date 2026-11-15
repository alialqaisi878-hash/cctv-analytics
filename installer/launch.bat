@echo off
rem Starts CCTV Analytics and opens the dashboard in the browser.
rem   launch.bat              normal start: the program opens in its own window (close it to stop)
rem   launch.bat --console    start with a console window that shows the log, dashboard in the browser
rem   launch.bat --background start minimised without opening the browser (used for auto-start)
cd /d "%~dp0"
title CCTV Analytics
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

if not exist ".venv\Scripts\python.exe" (
  echo The AI libraries are not installed yet. Running setup...
  powershell -NoProfile -ExecutionPolicy Bypass -File "installer\setup.ps1" -AppDir "%~dp0."
  if not exist ".venv\Scripts\python.exe" ( pause & exit /b 1 )
)

if /i "%~1"=="--background" (
  start "CCTV Analytics" /min cmd /c ""%~f0" --serve"
  exit /b 0
)

if "%~1"=="" if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" run.py --app-window
  exit /b 0
)

set OPEN=--open-browser
if /i "%~1"=="--serve" set OPEN=

:run
".venv\Scripts\python.exe" run.py %OPEN%
rem exit code 3 = an online update was installed: restart without opening another browser tab
if errorlevel 3 if not errorlevel 4 ( echo Restarting after update... & set OPEN= & goto run )
if errorlevel 1 pause
