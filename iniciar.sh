#!/usr/bin/env bash
# =============================================================================
#  Agente de voz postoperatorio — arranque de un comando (Linux y macOS)
#
#      ./iniciar.sh
#
#  Instala dependencias, descarga la voz y levanta el servidor. Es idempotente:
#  lo que ya está hecho no se repite, así que la segunda ejecución arranca en
#  segundos.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

rojo=$'\033[31m'; verde=$'\033[32m'; amarillo=$'\033[33m'; negrita=$'\033[1m'; fin=$'\033[0m'
paso()  { printf '%s[%s/5]%s %s\n' "$negrita" "$1" "$fin" "$2"; }
error() { printf '%s[ERROR]%s %s\n' "$rojo" "$fin" "$1" >&2; }
aviso() { printf '%s[!]%s %s\n' "$amarillo" "$fin" "$1"; }

printf '\n  %sAGENTE DE VOZ POSTOPERATORIO%s\n  ============================\n\n' "$negrita" "$fin"

# --- 1. Python 3.11 ----------------------------------------------------------
# ChromaDB depende de chroma-hnswlib, que solo publica wheels hasta cp311.
PYTHON_CMD=""
for candidato in python3.11 python3 python; do
    if command -v "$candidato" >/dev/null 2>&1 && "$candidato" -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)' 2>/dev/null; then
        PYTHON_CMD="$candidato"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    error "No se encontró Python 3.11."
    echo
    echo "  El proyecto necesita 3.11 exactamente: ChromaDB depende de"
    echo "  chroma-hnswlib, que solo publica wheels precompilados hasta cp311."
    echo
    echo "  macOS:         brew install python@3.11"
    echo "  Debian/Ubuntu: sudo apt install python3.11 python3.11-venv"
    echo "  Fedora:        sudo dnf install python3.11"
    echo
    exit 1
fi
paso 1 "Python 3.11 encontrado ($PYTHON_CMD)."

# --- 2. Entorno virtual ------------------------------------------------------
if [ ! -x ".venv/bin/python" ]; then
    paso 2 "Creando entorno virtual..."
    "$PYTHON_CMD" -m venv .venv
else
    paso 2 "Entorno virtual ya existe."
fi
VENV_PY=".venv/bin/python"

# --- 3. Dependencias ---------------------------------------------------------
# El centinela evita reinstalar en cada arranque: docling y sentence-transformers
# arrastran torch y la instalación tarda minutos.
if [ ! -f ".venv/.dependencias_ok" ]; then
    paso 3 "Instalando dependencias. La primera vez tarda varios minutos (torch pesa)."
    "$VENV_PY" -m pip install --upgrade pip --quiet
    "$VENV_PY" -m pip install -r requirements.txt
    "$VENV_PY" -m pip install -e . --quiet
    touch ".venv/.dependencias_ok"
else
    paso 3 "Dependencias ya instaladas."
fi

# --- 4. Voz de Piper ---------------------------------------------------------
VOZ_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx"
if [ ! -f "data/voices/es_voice.onnx" ]; then
    paso 4 "Descargando la voz en español (~63 MB)..."
    mkdir -p data/voices
    curl -L --progress-bar -o "data/voices/es_voice.onnx"      "$VOZ_BASE"
    curl -L --silent       -o "data/voices/es_voice.onnx.json" "$VOZ_BASE.json"
else
    paso 4 "Voz de Piper ya descargada."
fi

# --- 5. Credenciales ---------------------------------------------------------
# Lo único que el script no puede resolver solo.
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo
    aviso "Se creó el archivo .env a partir de la plantilla."
    echo
    echo "  Faltan las dos claves (ambas de nivel gratuito):"
    echo "      GROQ_API_KEY    -> https://console.groq.com/keys"
    echo "      GEMINI_API_KEY  -> https://aistudio.google.com/apikey"
    echo
    echo "  Edítalo y vuelve a ejecutar ./iniciar.sh"
    echo
    exit 0
fi

if ! grep -qE '^GROQ_API_KEY=.+' .env; then
    error "GROQ_API_KEY está vacía en el archivo .env"
    echo "  Consíguela gratis en https://console.groq.com/keys"
    exit 1
fi
paso 5 "Credenciales configuradas."

# --- Arranque ----------------------------------------------------------------
cat <<'BANNER'

  Levantando el servidor...

  Espera a ver "Application startup complete" (unos 25 segundos: se precargan
  la voz y el modelo de embeddings para que no los pague el primer turno).

     Interfaz de llamada    http://localhost:8000
     Consola de admin       http://localhost:8000/consola

  Ctrl+C para detenerlo.

BANNER

# Abre el navegador en segundo plano cuando el puerto responda, sin bloquear el
# arranque del servidor si tarda o si no hay entorno gráfico.
(
    for _ in $(seq 1 60); do
        if curl -s -o /dev/null "http://localhost:8000" 2>/dev/null; then
            if   command -v open     >/dev/null 2>&1; then open "http://localhost:8000"
            elif command -v xdg-open >/dev/null 2>&1; then xdg-open "http://localhost:8000" >/dev/null 2>&1
            fi
            break
        fi
        sleep 2
    done
) &

exec "$VENV_PY" -m uvicorn agente_postop.orchestrator.server:app --host 0.0.0.0 --port 8000
