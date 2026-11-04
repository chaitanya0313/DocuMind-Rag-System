@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv311\Scripts\python.exe" (
    echo Python environment .venv311 was not found.
    echo Create it and install requirements before starting DocuMind.
    pause
    exit /b 1
)

curl.exe --silent --fail --max-time 1 http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    where ollama >nul 2>&1
    if errorlevel 1 (
        echo Ollama was not found. Install Ollama or start its service manually.
        pause
        exit /b 1
    )

    echo Starting Ollama server...
    start "" /min ollama serve
    echo Waiting for Ollama to become ready...
    for /L %%I in (1,1,30) do (
        curl.exe --silent --fail --max-time 1 http://localhost:11434/api/tags >nul 2>&1
        if not errorlevel 1 goto ollama_ready
        timeout /t 1 /nobreak >nul
    )
    echo Ollama did not become reachable. Check the Ollama installation and try again.
    pause
    exit /b 1
)

:ollama_ready
echo Launching DocuMind with the project Python environment...
".venv311\Scripts\python.exe" -m streamlit run app.py --server.fileWatcherType none
if errorlevel 1 echo DocuMind stopped with an error.
pause