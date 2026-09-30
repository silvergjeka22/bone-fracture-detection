@echo off
rem Windows (PowerShell or cmd):  run.bat push [--quick],  run.bat status,  run.bat get
pushd "%~dp0"
python -c "import sys" >nul 2>&1
if %errorlevel%==0 (python scripts\kaggle_run.py %*) else (py -3 scripts\kaggle_run.py %*)
set rc=%errorlevel%
popd
exit /b %rc%
