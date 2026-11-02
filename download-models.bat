@echo off
chcp 65001 >nul
setlocal
title Download models for Qwen3.6-35B-A3B

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "MODELS=%ROOT%\Models"
if not exist "%MODELS%" mkdir "%MODELS%"

where hf >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Hugging Face CLI "hf" not found.
  echo Install it with:  powershell -c "irm https://hf.co/cli/install.ps1 ^| iex"
  echo or:              pip install -U huggingface_hub
  pause & exit /b 1
)

echo.
echo   Downloading Qwen3.6-35B-A3B Q4_K_M (~19 GB) ...
hf download ggml-org/Qwen3.6-35B-A3B-GGUF --include "Qwen3.6-35B-A3B-Q4_K_M.gguf" --local-dir "%MODELS%"
if errorlevel 1 goto err

echo.
echo   Downloading MTP head (~1 GB) for speculative decoding ...
hf download ggml-org/Qwen3.6-35B-A3B-GGUF --include "mtp-Qwen3.6-35B-A3B-Q4_0.gguf" --local-dir "%MODELS%"
if errorlevel 1 goto err

echo.
echo   Done. Files:
dir /b "%MODELS%\*.gguf"
echo.
echo   Now run start.bat
pause
exit /b 0

:err
echo.
echo [ERROR] Download failed. Check your network / HF token.
pause
exit /b 1
