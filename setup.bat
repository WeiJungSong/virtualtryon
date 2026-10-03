@echo off
rem Local Try-On installer for Windows. Double-click, or: setup.bat --hd
rem Everything goes inside this folder (.tools\ and .venv\); nothing is installed system-wide.
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "ROOT=%CD%"
set "TOOLS=%ROOT%\.tools"
set "UV=%TOOLS%\uv\uv.exe"
set "PYVER=3.11"
set "UV_PYTHON_INSTALL_DIR=%TOOLS%\python"
set "UV_PYTHON_BIN_DIR=%TOOLS%\python-bin"
set "UV_PYTHON_PREFERENCE=only-managed"
set "UV_CACHE_DIR=%TOOLS%\cache"
set "UV_NO_MODIFY_PATH=1"
set "PYTHONUTF8=1"
set "TARGET=x86_64-pc-windows-msvc"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "TARGET=aarch64-pc-windows-msvc"

echo === 本機虛擬試穿 安裝 (Windows) ===
echo    系統: %TARGET%

if defined VTON_UV set "UV=%VTON_UV%"
if exist "%UV%" goto :have_uv
echo [1] 下載 uv
if not exist "%TOOLS%\uv" mkdir "%TOOLS%\uv"
rem PyPI first (fast CDN), then GitHub, then the official installer
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $v=(Invoke-RestMethod 'https://pypi.org/pypi/uv/json').info.version; $r=Invoke-RestMethod ('https://pypi.org/pypi/uv/'+$v+'/json'); $tag='win_amd64'; if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { $tag='win_arm64' }; $u=($r.urls | Where-Object { $_.filename -like ('*'+$tag+'.whl') } | Select-Object -First 1).url; $w=Join-Path $env:TEMP 'uv-vton-whl.zip'; Invoke-WebRequest -UseBasicParsing -Uri $u -OutFile $w; $d=Join-Path $env:TEMP 'uv-vton-whl'; Expand-Archive -Path $w -DestinationPath $d -Force; Copy-Item (Join-Path $d ('uv-'+$v+'.data\scripts\uv.exe')) '%TOOLS%\uv\uv.exe' -Force"
if exist "%UV%" goto :have_uv
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $z=Join-Path $env:TEMP 'uv-vton.zip'; Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-%TARGET%.zip' -OutFile $z; Expand-Archive -Path $z -DestinationPath '%TOOLS%\uv' -Force"
if exist "%UV%" goto :have_uv
echo    GitHub 下載失敗，改用官方安裝程式
powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:UV_INSTALL_DIR='%TOOLS%\uv'; $env:UV_NO_MODIFY_PATH='1'; irm https://astral.sh/uv/install.ps1 | iex"
if not exist "%UV%" goto :fail_uv
:have_uv
"%UV%" --version

set "PYSEL=%PYVER%"
set "MARK=%PYVER%-%TARGET%"
if not defined VTON_PYTHON goto :py_managed
set "PYSEL=%VTON_PYTHON%"
set "UV_PYTHON_PREFERENCE=system"
set "MARK=custom-%TARGET%"
echo [2] 使用指定的 Python: %PYSEL%
goto :py_ready
:py_managed
echo [2] 準備 Python %PYVER%
"%UV%" python install %PYVER%
if errorlevel 1 goto :fail
:py_ready

set "CUR="
if exist ".venv\.vton-python" set /p CUR=<".venv\.vton-python"
if not exist ".venv" goto :make_venv
if "%CUR%"=="%MARK%" goto :make_venv
set "OLD=.venv.old-%RANDOM%%RANDOM%"
ren ".venv" "%OLD%"
echo    舊的 .venv 不相容，已改名為 %OLD%
:make_venv
if exist ".venv\Scripts\python.exe" goto :have_venv
"%UV%" venv --python "%PYSEL%" .venv
if errorlevel 1 goto :fail
:have_venv
>".venv\.vton-python" echo %MARK%

set "UV_BIN=%UV%"
".venv\Scripts\python.exe" tools\install.py %*
if errorlevel 1 goto :fail

echo.
echo 安裝完成。之後雙擊 run.bat 啟動。
pause
exit /b 0

:fail_uv
echo.
echo [錯誤] 無法下載 uv，請確認網路連線。
pause
exit /b 1

:fail
echo.
echo [錯誤] 安裝沒有完全成功，請把上面的訊息複製下來。
pause
exit /b 1
