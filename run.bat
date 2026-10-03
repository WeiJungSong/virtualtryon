@echo off
rem Start Local Try-On and open it in the browser (Windows).
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto :not_installed
set "PYTHONUTF8=1"
if "%VTON_PORT%"=="" set "VTON_PORT=8765"
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 60;$i++){Start-Sleep -Milliseconds 500; try{Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:%VTON_PORT%/' -TimeoutSec 2 | Out-Null; Start-Process 'http://127.0.0.1:%VTON_PORT%/'; break}catch{}}"
".venv\Scripts\python.exe" -m server.app
pause
exit /b 0

:not_installed
echo 尚未安裝，請先雙擊 setup.bat
pause
exit /b 1
