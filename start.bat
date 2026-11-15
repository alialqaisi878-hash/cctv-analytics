@echo off
cd /d %~dp0
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
if not exist .venv (
  py -3 -m venv .venv
  .venv\Scripts\pip install -r requirements.txt
)
:run
.venv\Scripts\python run.py %*
if errorlevel 3 if not errorlevel 4 goto run
pause
