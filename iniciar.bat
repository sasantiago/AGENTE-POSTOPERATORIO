@echo off
REM ===========================================================================
REM  Agente de voz postoperatorio - arranque de un clic (Windows)
REM
REM  Doble clic sobre este archivo. Instala dependencias, descarga la voz y
REM  levanta el servidor. Es idempotente: si algo ya esta, no lo repite, asi
REM  que la segunda ejecucion arranca en segundos.
REM
REM  Sin acentos a proposito: la consola de Windows usa codepage 850 por
REM  defecto y los mostraria como basura.
REM ===========================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo   AGENTE DE VOZ POSTOPERATORIO
echo   ============================
echo.

REM --- 1. Python 3.11 ------------------------------------------------------
REM  chroma-hnswlib (dependencia de ChromaDB) solo publica wheels hasta cp311.
REM  Con 3.12+ pip intenta compilar desde fuente y exige Visual C++ Build Tools.
set PYTHON_CMD=
py -3.11 --version >nul 2>&1
if !errorlevel! equ 0 (
    set PYTHON_CMD=py -3.11
) else (
    python --version 2>nul | findstr /C:"3.11" >nul
    if !errorlevel! equ 0 set PYTHON_CMD=python
)

if "!PYTHON_CMD!"=="" (
    echo   [ERROR] No se encontro Python 3.11.
    echo.
    echo   Este proyecto necesita 3.11 exactamente: ChromaDB depende de
    echo   chroma-hnswlib, que solo publica wheels precompilados hasta cp311.
    echo   Con 3.12 o 3.13 la instalacion intenta compilar desde fuente.
    echo.
    echo   Descargalo de https://www.python.org/downloads/release/python-3119/
    echo   y marca "Add python.exe to PATH" durante la instalacion.
    echo.
    pause
    exit /b 1
)
echo   [1/5] Python 3.11 encontrado.

REM --- 2. Entorno virtual --------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo   [2/5] Creando entorno virtual...
    !PYTHON_CMD! -m venv .venv
    if !errorlevel! neq 0 goto :error_venv
) else (
    echo   [2/5] Entorno virtual ya existe.
)
set VENV_PY=.venv\Scripts\python.exe

REM --- 3. Dependencias -----------------------------------------------------
REM  El centinela evita reinstalar en cada arranque: pip tarda minutos porque
REM  docling y sentence-transformers arrastran torch.
if not exist ".venv\.dependencias_ok" (
    echo   [3/5] Instalando dependencias. La primera vez tarda varios minutos
    echo         ^(torch pesa^): es buen momento para un cafe.
    %VENV_PY% -m pip install --upgrade pip --quiet
    %VENV_PY% -m pip install -r requirements.txt
    if !errorlevel! neq 0 goto :error_pip
    %VENV_PY% -m pip install -e . --quiet
    if !errorlevel! neq 0 goto :error_pip
    echo ok> ".venv\.dependencias_ok"
) else (
    echo   [3/5] Dependencias ya instaladas.
)

REM --- 4. Voz de Piper -----------------------------------------------------
if not exist "data\voices\es_voice.onnx" (
    echo   [4/5] Descargando la voz en espanol ^(~63 MB^)...
    if not exist "data\voices" mkdir "data\voices"
    curl -L --progress-bar -o "data\voices\es_voice.onnx" "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx"
    curl -L --silent -o "data\voices\es_voice.onnx.json" "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx.json"
    if not exist "data\voices\es_voice.onnx" goto :error_voz
) else (
    echo   [4/5] Voz de Piper ya descargada.
)

REM --- 5. Credenciales -----------------------------------------------------
REM  Lo unico que el script no puede resolver solo.
if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo.
    echo   [!] Se creo el archivo .env a partir de la plantilla.
    echo.
    echo   Falta poner las dos claves ^(ambas son de nivel gratuito^):
    echo       GROQ_API_KEY    -^> https://console.groq.com/keys
    echo       GEMINI_API_KEY  -^> https://aistudio.google.com/apikey
    echo.
    echo   Abriendo .env en el bloc de notas. Guardalo y vuelve a ejecutar
    echo   este archivo.
    echo.
    notepad .env
    pause
    exit /b 0
)

findstr /R /C:"^GROQ_API_KEY=..*" .env >nul
if !errorlevel! neq 0 (
    echo.
    echo   [ERROR] GROQ_API_KEY esta vacia en el archivo .env
    echo   Consiguela gratis en https://console.groq.com/keys
    echo.
    notepad .env
    pause
    exit /b 1
)
echo   [5/5] Credenciales configuradas.

REM --- Arranque ------------------------------------------------------------
echo.
echo   Levantando el servidor...
echo.
echo   Espera a ver "Application startup complete" ^(unos 25 segundos: se
echo   precargan la voz y el modelo de embeddings para que no los pague el
echo   primer turno de la llamada^).
echo.
echo      Interfaz de llamada    http://localhost:8000
echo      Consola de admin       http://localhost:8000/consola
echo.
echo   Ctrl+C para detenerlo.
echo.

start "" http://localhost:8000
%VENV_PY% -m uvicorn agente_postop.orchestrator.server:app --host 0.0.0.0 --port 8000
goto :fin

:error_venv
echo   [ERROR] No se pudo crear el entorno virtual.
pause
exit /b 1

:error_pip
echo.
echo   [ERROR] Fallo la instalacion de dependencias.
echo   Revisa el mensaje de arriba. Causa habitual: Python distinto de 3.11.
pause
exit /b 1

:error_voz
echo   [ERROR] No se pudo descargar la voz de Piper. Revisa tu conexion.
pause
exit /b 1

:fin
pause
