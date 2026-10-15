@echo off
chcp 65001 >nul

set HF_ENDPOINT=https://hf-mirror.com
set HF_HUB_DISABLE_XET=1
set OLLAMA_MODELS=D:\ollama_models

REM ============ 检查 Ollama 是否在跑 ============
tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | find /I "ollama.exe" >nul
if errorlevel 1 (
    echo [启动] Ollama 服务...
    start "Ollama" cmd /k "ollama serve"
    timeout /t 5 /nobreak >nul
) else (
    echo [跳过] Ollama 已在运行
)

REM ============ 启动 Streamlit，日志覆盖写入 ============
cd /d D:\ai_project\assistant
echo [启动] Streamlit...
echo [日志] D:\ai_project\assistant\streamlit.log

powershell -Command "& 'E:\miniconda\envs\myenv\python.exe' -m streamlit run app.py 2>&1 | Tee-Object -FilePath 'D:\ai_project\assistant\streamlit.log'"

pause