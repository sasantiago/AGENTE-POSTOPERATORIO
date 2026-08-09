# Agente de voz postoperatorio — Tech Sphere Challenge 2026

Agente de voz que hace seguimiento postoperatorio telefónico: conversa con el paciente en
español, entiende sus síntomas contra el corpus clínico real, detecta señales de alarma con
doble vía de decisión (reglas deterministas + LLM) y escala a personal humano cuando
corresponde. No es un chatbot reactivo: compara lo reportado contra la trayectoria de
recuperación esperada para el procedimiento y el día postoperatorio.

Contexto del reto, reglas de evaluación y dataset original: [`docs/reto-original-readme.md`](docs/reto-original-readme.md),
[`docs/rubrica-evaluacion.md`](docs/rubrica-evaluacion.md), [`docs/stack-tecnico.md`](docs/stack-tecnico.md).

## Stack

| Pieza | Herramienta | Por qué |
|---|---|---|
| LLM (razonamiento de la llamada) | Llama 3.3 70B vía Groq | Cumple G3 (familia Meta Llama, nivel gratuito); latencia mínima por LPU |
| STT | Groq Whisper Large V3 | Mismo proveedor que el LLM, menos saltos de red |
| TTS | Kokoro-82M / Piper (español) | Local, gratis, sin límite de minutos |
| RAG | ChromaDB + BGE-M3 | Local, gratis, fuerte en español médico |
| Ingesta | Docling | Un solo camino a Markdown para cualquier archivo, con OCR |
| Orquestación | FastAPI + WebSocket | Streaming bidireccional de audio y turnos |

Claude Code se usó como asistente de desarrollo (nunca como el LLM que razona en la
llamada — ver `docs/informe-final.md` una vez redactado).

## Requisitos previos

- **Python 3.11** (en Windows, `chroma-hnswlib` —dependencia de ChromaDB— solo publica
  wheels precompilados hasta cp311; con 3.12 o 3.13 el pip intenta compilar desde fuente y
  exige Visual C++ Build Tools).
- Una clave de API de [Groq](https://console.groq.com/) (nivel gratuito)

## Instalación (≤15 minutos)

```bash
git clone https://github.com/sasantiago/AGENTE-POSTOPERATORIO.git
cd AGENTE-POSTOPERATORIO
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

cp .env.example .env
# Edita .env y agrega tu GROQ_API_KEY
```

Indexar el corpus clínico (una sola vez, o cuando cambie `dataset/textos/`):

```bash
python -m agente_postop.ingestion.build_index
```

Levantar el servidor (orquestador + consola + interfaz de llamada):

```bash
uvicorn agente_postop.orchestrator.server:app --host 0.0.0.0 --port 8000
```

Abre `http://localhost:8000` para la interfaz de llamada y `http://localhost:8000/consola`
para la consola de administración.

## Estructura del repositorio

```
src/agente_postop/
├── config.py           # variables de entorno, rutas
├── clients.py           # cliente Groq
├── ingestion/            # Docling → Markdown → chunking → ChromaDB
├── rag/                  # embeddings BGE-M3, recuperación
├── voice/                # STT, TTS, fillers cacheados
├── clinical/             # arco reflejo, gemelo de trayectoria, memoria, SBAR, validador de citas
├── orchestrator/         # FastAPI + WebSocket, gestor de turno
└── console/               # consola de administración (subir/listar/eliminar)
harness/                  # evaluación contra dataset_final.xlsx
tests/adversarial/        # suite de inyección de prompt y entradas hostiles
vault/                    # carpeta vigilada (ingesta tipo Obsidian)
```

## Métricas (se completa durante el harness — §5 de la rúbrica)

- Latencia P50 / P95: _pendiente_
- Tokens in/out por turno y por llamada: _pendiente_
- Invocaciones al modelo por turno: _pendiente_
- Consultas RAG por llamada: _pendiente_
- Costo estimado por llamada: _pendiente_

## Estado

Proyecto en construcción activa. Ver historial de commits para el progreso.
