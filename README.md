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

Descargar el modelo de voz de Piper (español, ~63 MB, una sola vez):

```bash
mkdir -p data/voices
curl -L -o data/voices/es_voice.onnx "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx"
curl -L -o data/voices/es_voice.onnx.json "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx.json"
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

## Métricas (§5 de la rúbrica)

Medidas con [`harness/run_eval.py`](harness/run_eval.py) sobre una muestra estratificada
del dataset — 14 casos (los 12 `rojo` completos + 2 `amarillo`), capa `capa1_limpia`, 84
turnos de paciente. No se corrió el dataset completo (160 casos × 2 capas ≈ 1 920 turnos)
porque el tier gratuito de Groq para `llama-3.3-70b-versatile` limita a **100,000
tokens/día**, y con el consumo medido eso alcanza para ~88 llamadas al LLM, no para el
total. Detalle y matriz de confusión en [`docs/informe-final.md`](docs/informe-final.md#6-métricas-§5-de-la-rúbrica);
resultados crudos en [`harness_resultados_rojo_capa1.json`](harness_resultados_rojo_capa1.json).

Reproducir:

```bash
python -m harness.run_eval --n-rojo -1 --n-amarillo 2 --n-verde 0 --capas capa1_limpia --seed 7
```

| Métrica | Valor |
|---|---|
| Latencia P50 / P95 (orquestación, sin STT/TTS) | 5.24s / 9.91s |
| Tokens de entrada por turno (media / P50) | 1,070 / 1,084 |
| Tokens de salida por turno (media / P50) | 79 / 80 |
| Invocaciones al modelo por turno | 1 (la vía refleja no usa LLM) |
| Consultas al RAG por llamada | 1 por turno de paciente (4 chunks por consulta) |
| Costo estimado por llamada (6 turnos) | ≈ USD 0.006 |
| Falsos negativos catastróficos (rojo real → verde predicho) | 0 / 72 |
| Recall rojo / amarillo (muestra) | 8.3% / 41.7% — ver nota abajo |

La latencia reportada es la de orquestación pura (reflejo + cortex + fusión + validador),
sin sumar STT ni TTS — se midió inyectando texto directo al orquestador (`ws://.../ws/llamada`
acepta un mensaje de texto además de audio, es el mismo bypass que usa el harness).

**Sobre el recall bajo de `rojo`:** ningún turno `rojo` real fue clasificado `verde` (la
falla catastrófica que penaliza la rúbrica es 0/72), pero el cortex sub-triaja hacia
`amarillo` en la mayoría de esos turnos en vez de escalar al máximo nivel. Es una
limitación conocida y documentada, no un número escondido — ver
[`docs/informe-final.md` §6-7](docs/informe-final.md) para el detalle y el plan de
corrección (`clinical/reflex_rules.py`).

## Estado

Proyecto en construcción activa. Ver historial de commits para el progreso.
