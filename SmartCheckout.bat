@echo off
REM Paketlemeden, cift tiklayarak calistirmak icin (proje klasorunde .venv olmali).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [HATA] .venv bulunamadi. README'deki kurulum adimlarini izleyin.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" launcher.py %*
pause
