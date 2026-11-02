@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
title Qwen3.6-35B-A3B local server

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "LLAMA=%ROOT%\llama-cuda"
set "MODELS=%ROOT%\Models"
set "MODEL=%MODELS%\Qwen3.6-35B-A3B-Q4_K_M.gguf"
set "MTP=%MODELS%\mtp-Qwen3.6-35B-A3B-Q4_0.gguf"

if not exist "%LLAMA%\llama-server.exe" (
  echo [ERROR] llama-server.exe not found in %LLAMA%
  pause & exit /b 1
)
if not exist "%MODEL%" (
  echo [ERROR] model not found: %MODEL%
  echo Run download-models.bat first.
  pause & exit /b 1
)

set "CHOICE=%~1"
if "%CHOICE%"=="" goto menu
if "%CHOICE%"=="0" goto stop
if "%CHOICE%"=="1" goto launch
if "%CHOICE%"=="2" goto launch
if "%CHOICE%"=="3" goto launch
goto menu

:menu
cls
echo.
echo   ============================================================
echo     Qwen3.6-35B-A3B  ^|  RTX 3060 6GB + 32GB RAM
echo   ============================================================
echo.
echo     1  Fast 32k      MTP spec, ~43 tok/s, VRAM 5.2 GB
echo     2  Long 128k     MTP spec, ~38 tok/s, VRAM 5.9 GB
echo     3  Compat 32k    no MTP,   ~34 tok/s, VRAM 5.6 GB
echo.
echo     0  Exit (and stop any running server)
echo.
set /p CHOICE="  Select config [1-3]: "

if "%CHOICE%"=="0" goto stop
if not "%CHOICE%"=="1" if not "%CHOICE%"=="2" if not "%CHOICE%"=="3" goto menu

:launch
:: --- RAM sanity check ---
set "FREEGB="
for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)"`) do set "FREEGB=%%A"
if defined FREEGB (
  echo   Free RAM: !FREEGB! GB
  if !FREEGB! LSS 16 echo   [warn] Less than 16 GB free RAM. Close Chrome/VS Code, or the model may swap.
)

echo.
echo   Stopping old llama-server processes...
call :killserver

pushd "%LLAMA%"

if "%CHOICE%"=="1" goto cfg1
if "%CHOICE%"=="2" goto cfg2
if "%CHOICE%"=="3" goto cfg3
goto menu

:cfg1
echo   Starting: FAST 32k (MTP, n-cpu-moe 38)
echo   API: http://127.0.0.1:8080/v1  model: qwen3.6-35b-a3b
echo.
llama-server.exe -m "%MODEL%" --alias qwen3.6-35b-a3b --jinja -fa on -ngl 999 ^
  --n-cpu-moe 38 -ctk q8_0 -ctv q8_0 -np 1 --load-mode none -t 8 -ub 512 -c 32768 ^
  -md "%MTP%" --spec-type draft-mtp --spec-draft-n-max 2 ^
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --host 127.0.0.1 --port 8080 -fit off %2 %3 %4 %5 %6 %7 %8 %9
goto done

:cfg2
echo   Starting: LONG 128k (MTP, n-cpu-moe 40)
echo   API: http://127.0.0.1:8080/v1  model: qwen3.6-35b-a3b
echo.
llama-server.exe -m "%MODEL%" --alias qwen3.6-35b-a3b --jinja -fa on -ngl 999 ^
  --n-cpu-moe 40 -ctk q8_0 -ctv q8_0 -np 1 --load-mode none -t 8 -ub 512 -c 131072 ^
  -md "%MTP%" --spec-type draft-mtp --spec-draft-n-max 2 ^
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --host 127.0.0.1 --port 8080 -fit off %2 %3 %4 %5 %6 %7 %8 %9
goto done

:cfg3
echo   Starting: COMPAT 32k (no MTP, n-cpu-moe 35)
echo   API: http://127.0.0.1:8080/v1  model: qwen3.6-35b-a3b
echo.
llama-server.exe -m "%MODEL%" --alias qwen3.6-35b-a3b --jinja -fa on -ngl 999 ^
  --n-cpu-moe 35 -ctk q8_0 -ctv q8_0 -np 1 --load-mode none -t 8 -ub 512 -c 32768 ^
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --host 127.0.0.1 --port 8080 -fit off %2 %3 %4 %5 %6 %7 %8 %9
goto done

:stop
call :killserver
echo   Server stopped.
ping -n 3 127.0.0.1 >nul 2>&1
exit /b 0

:done
popd
echo.
echo   Server exited.
pause
exit /b 0

:killserver
taskkill /F /IM llama-server.exe >nul 2>&1
exit /b 0
