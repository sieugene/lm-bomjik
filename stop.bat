@echo off
chcp 65001 >nul
echo Stopping llama-server...
taskkill /F /IM llama-server.exe >nul 2>&1
if errorlevel 1 (echo No llama-server process found.) else (echo Done.)
ping -n 3 127.0.0.1 >nul 2>&1
