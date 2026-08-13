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
| LLM (extracción clínica del turno) | Gemini Flash | Cumple G3 (familia Google Gemini, gama Flash, nivel gratuito). Tarea de esquema cerrado que no requiere el 70B, y su cupo es un presupuesto **separado** del de Groq: las dos llamadas del turno dejan de competir por el mismo límite diario |
| STT | Groq Whisper Large V3 | Mismo proveedor que el LLM, menos saltos de red |
| TTS | Piper (voz `es_MX`, local) | Local, gratis, sin límite de minutos. Se evaluó Kokoro-82M y se descartó: Piper sintetiza la respuesta completa en ~1s en CPU (medido), suficiente para el turno |
| RAG | ChromaDB + `multilingual-e5-base` (búsqueda híbrida: vectorial + BM25) | Local, gratis; e5-base (~1.1GB) en vez de BGE-M3 (4.3GB) para no comprometer la compuerta de 15 minutos — BM25 recupera el margen de precisión en términos exactos (dosis, fármacos) que un embedding más chico puede difuminar |
| Ingesta | Docling | Un solo camino a Markdown para cualquier archivo, con OCR |
| Orquestación | FastAPI + WebSocket | Streaming bidireccional de audio y turnos |

Claude Code se usó como asistente de desarrollo (nunca como el LLM que razona en la
llamada — ver `docs/informe-final.md` una vez redactado).

## Requisitos previos

- **Python 3.11.** No 3.12 ni 3.13: `chroma-hnswlib` —dependencia de ChromaDB— solo publica
  wheels precompilados hasta cp311, así que con versiones más nuevas pip intenta compilar
  desde fuente y exige un toolchain de C++ (Visual C++ Build Tools en Windows).
- **Dos claves de API, ambas de nivel gratuito:**
  [Groq](https://console.groq.com/keys) para la conversación y
  [Google AI Studio](https://aistudio.google.com/apikey) para la extracción clínica. Sus
  cupos diarios son presupuestos separados — ver "Continuidad del servicio".
- ~4 GB libres en disco y conexión a internet en el primer arranque.

## Instalación (≤15 minutos)

```bash
git clone https://github.com/sasantiago/AGENTE-POSTOPERATORIO.git
cd AGENTE-POSTOPERATORIO
```

### Opción A — un solo comando (recomendada)

Los scripts crean el entorno, instalan dependencias, descargan la voz, verifican las
credenciales y levantan el servidor. Son idempotentes: lo que ya está hecho no se repite,
así que a partir del segundo arranque tardan segundos.

**Windows** — doble clic sobre `iniciar.bat`, o desde una terminal:

```bat
iniciar.bat
```

**macOS y Linux:**

```bash
./iniciar.sh
```

La primera ejecución crea el archivo `.env` a partir de la plantilla y se detiene para que
pongas las dos claves. Rellénalas y vuelve a ejecutarlo: eso es todo.

### Opción B — paso a paso

Si prefieres controlar cada etapa, o si el script falla y quieres ver dónde:

```bash
# 1. Entorno virtual
python3.11 -m venv .venv

#    Windows (PowerShell o cmd):
.venv\Scriptsctivate
#    macOS / Linux:
source .venv/bin/activate

# 2. Dependencias (varios minutos: torch viene con docling y sentence-transformers)
pip install -r requirements.txt
pip install -e .

# 3. Credenciales
cp .env.example .env          # Windows: copy .env.example .env
#    Edita .env y pon GROQ_API_KEY y GEMINI_API_KEY.

# 4. Voz de Piper en español (~63 MB, una sola vez)
mkdir -p data/voices
curl -L -o data/voices/es_voice.onnx      "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx"
curl -L -o data/voices/es_voice.onnx.json "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx.json"

# 5. Arrancar
uvicorn agente_postop.orchestrator.server:app --host 0.0.0.0 --port 8000
```

En Windows, `mkdir -p` no existe en cmd: usa `mkdir dataoices` (y omite el `-p`).

### Qué esperar al arrancar

El servidor tarda **unos 25 segundos** en quedar listo. No está colgado: precarga la voz de
Piper y el modelo de embeddings para que no los pague el primer turno de la llamada. Sabrás
que terminó cuando veas estas tres líneas:

```
warm-up: voz de Piper cargada
warm-up: modelo de embeddings y ChromaDB cargados
INFO:     Application startup complete.
```

| Superficie | URL |
|---|---|
| Interfaz de llamada | http://localhost:8000 |
| Consola de administración | http://localhost:8000/consola |

Cada turno de conversación escribe una línea con su latencia desglosada, en la consola y en
`agente_postop.log`.

### Sobre el índice y los modelos

**El índice de ChromaDB ya viene construido y comiteado en `data/chroma/`** — no hace falta
indexar los 107 PDFs de `dataset/textos/` para levantar la solución (con Docling en CPU eso
toma del orden de una hora; hacerlo parte del arranque documentado habría roto la compuerta
de los 15 minutos). Lo único que se descarga en el primer arranque es el modelo de
embeddings `intfloat/multilingual-e5-base` (~1.1 GB, vía `sentence-transformers`, se cachea
en `~/.cache/huggingface`) — el mismo con el que se construyó el índice comiteado, así que
los vectores siguen siendo compatibles.

Si vas a **agregar o refrescar conocimiento** (no hace falta para levantar la demo, es para
cuando cambie el corpus):

```bash
python -m agente_postop.ingestion.build_index
```

### Si algo falla

| Síntoma | Causa y arreglo |
|---|---|
| `error: Microsoft Visual C++ 14.0 or greater is required` | Estás en Python 3.12+. Instala 3.11 y borra `.venv`. |
| `ModuleNotFoundError: No module named 'agente_postop'` | Faltó `pip install -e .`, o el entorno virtual no está activo. |
| El agente responde "Se me acabó el tiempo disponible" | Cupo diario de Groq agotado (100 000 tokens). Reinicia a medianoche UTC, o añade otra clave en `GROQ_API_KEYS_EXTRA`. |
| El servidor no arranca: `address already in use` | Ya hay una instancia en el puerto 8000. Ciérrala o usa `--port 8001`. |
| El navegador no pide permiso de micrófono | El micrófono requiere contexto seguro: usa `localhost` (no la IP de red) o sirve por HTTPS. |

## Estructura del repositorio

```
src/agente_postop/
├── config.py           # variables de entorno, rutas
├── clients.py           # cliente Groq
├── ingestion/            # Docling → Markdown → chunking → ChromaDB
├── rag/                  # embeddings e5-base, recuperación híbrida (vectorial + BM25)
├── voice/                # STT, TTS, fillers cacheados
├── clinical/             # extracción, estado de la llamada, arco reflejo, gemelo de trayectoria, memoria, SBAR, validador de citas
├── orchestrator/         # FastAPI + WebSocket, gestor de turno
└── console/               # consola de administración (subir/listar/eliminar)
harness/                  # evaluación contra dataset_final.xlsx
tests/adversarial/        # suite de inyección de prompt y entradas hostiles
vault/                    # carpeta vigilada (ingesta tipo Obsidian)
```

## Métricas (§5 de la rúbrica)

> **Nota de arquitectura:** las cifras de tokens vienen del harness corrido *antes* de
> activar la extracción de síntomas (1 llamada al LLM por turno); con la extracción activa
> suben — ver la muestra de validación en
> [`docs/informe-final.md` §6](docs/informe-final.md#6-métricas-§5-de-la-rúbrica). Las de
> latencia sí corresponden a la arquitectura actual: se midieron en vivo sobre el camino de
> voz completo, con las dos llamadas por turno corriendo en paralelo.

Medidas con [`harness/run_eval.py`](harness/run_eval.py) sobre una muestra estratificada
del dataset — 14 casos (los 12 `rojo` completos + 2 `amarillo`), capa `capa1_limpia`, 84
turnos de paciente. No se corrió el dataset completo (160 casos × 2 capas ≈ 1 920 turnos)
porque el tier gratuito de Groq para `llama-3.3-70b-versatile` limita a **100,000
tokens/día**. Tras la optimización del presupuesto de tokens
([§4.1](docs/informe-final.md#41-presupuesto-de-tokens--el-límite-real-de-una-cuenta-gratuita):
5.809 → 2.736 tokens/turno, y la extracción movida al cupo separado de Gemini) ese límite
da ~4,9 llamadas completas por día por clave, frente a 1,4 antes — suficiente para muestras
mayores, no para el dataset completo. Detalle y matriz de
confusión en [`docs/informe-final.md`](docs/informe-final.md#6-métricas-§5-de-la-rúbrica);
resultados crudos en [`harness_resultados_rojo_capa1.json`](harness_resultados_rojo_capa1.json).

Reproducir:

```bash
python -m harness.run_eval --n-rojo -1 --n-amarillo 2 --n-verde 0 --capas capa1_limpia --seed 7
```

| Métrica | Valor |
|---|---|
| **Latencia de voz — de fin de habla a primer audio** (la que pide §5) | 7.0s por turno, medido en vivo · desglose por etapa abajo |
| Latencia de orquestación (sin STT/TTS, harness) | 5.24s / 9.91s (P50 / P95) |
| Tokens de entrada por turno (media / P50) | 1,070 / 1,084 |
| Tokens de salida por turno (media / P50) | 79 / 80 |
| Invocaciones al modelo por turno | 2 (extracción + conversación, en paralelo). La vía refleja no usa LLM |
| Consultas al RAG por llamada | 1 por turno de paciente (4 chunks por consulta) |
| Costo estimado por llamada (6 turnos) | ≈ USD 0.006 (≈ USD 0.0076 con la arquitectura actual) |
| Falsos negativos catastróficos (rojo real → verde predicho) | 0 / 72 turnos · **0 / 12 casos colgaron sin escalar** |
| Recall rojo — por caso, como está etiquetado el dataset | **41.7%** (5/12) · ver nota abajo |
| Recall rojo — vía refleja sola, 160 casos, tras la calibración | **66.7%** capa1 / 58.3% capa2 |
| Falsos positivos de la vía refleja sobre casos verde | **0%** (0/123), antes 13% |

### Desglose de latencia por etapa

Cada turno escribe una línea en `agente_postop.log` con sus tiempos, así que lo de esta
tabla es contrastable contra la sesión de evaluación:

```
turno paciente=pac_42_00017 criticidad=rojo rag_ms=763 llm_conversacion_ms=697
  llm_extraccion_ms=5000 orquestacion_ms=5848 tts_ms=1076 total_ms=6968 ttfr_ms=6975
```

| Etapa | ms |
|---|---:|
| RAG (embedding e5 en CPU + Chroma + BM25) | 763 |
| LLM conversación (Llama 3.3 70B en Groq) | 697 |
| LLM extracción (Gemini Flash) | 5 000 |
| Orquestación (las dos llamadas en paralelo: cuenta el máximo) | 5 848 |
| TTS (Piper, respuesta completa) | 1 076 |
| **Total del turno** | **6 968** |

El silencio que percibe el paciente es menor que ese total: la interfaz reproduce un filler
cacheado (`voice/fillers/`) en cuanto suelta el botón, sin esperar al servidor.

Esta cifra bajó de **36.9s a 7.0s** al instrumentar el turno: la extracción en Gemini corría
sin timeout y con un techo de tokens que truncaba el JSON, forzando una segunda llamada a
Groq. Detalle en [`docs/informe-final.md` §6.3](docs/informe-final.md).

La latencia reportada es la de orquestación pura (reflejo + cortex + fusión + validador),
sin sumar STT ni TTS — se midió inyectando texto directo al orquestador (`ws://.../ws/llamada`
acepta un mensaje de texto además de audio, es el mismo bypass que usa el harness).

**Sobre el recall de `rojo`:** la entrega anterior reportaba 8.3%, medido comparando la
criticidad de **cada turno** contra `label_ground_truth`. Ese campo es una etiqueta **de
caso**, no de turno — los 160 casos del dataset tienen el mismo valor en todos sus turnos —
así que esa cifra penalizaba al agente por no gritar ROJO cuando el paciente contestaba que
había dormido mal. Agregado por caso, que es como el dataset está etiquetado, el recall
sobre los mismos resultados crudos es **41.7%**, y **ninguna** de las 12 llamadas `rojo`
colgó sin escalar. `harness/report.py` ahora imprime las dos vistas.

Con la métrica corregida, la vía refleja se recalibró contra los 160 casos —es determinista,
así que se puede evaluar sin gastar tokens— y pasó de 41.7% a 66.7% de recall con 0% de
falsos positivos sobre casos verde. Las dos causas eran concretas: la fiebre se perdía si el
paciente no decía «grados» («me la tomé y marcó como 38»), y la regla `pus` disparaba con la
negación («no le sale nada de pus»). Detalle en
[`docs/informe-final.md` §6.1-6.2](docs/informe-final.md); fijado en `tests/test_reflex_rules.py`.

## Continuidad del servicio

Un agente que llama a pacientes recién operados no puede fallar en silencio. Si algo se
rompe a mitad de una llamada, el paciente no ve un error: oye un silencio y cuelga
pensando que nadie lo está siguiendo. Por eso cada modo de falla tiene una respuesta
definida, y ninguna de ellas es "no pasa nada".

| Qué falla | Qué percibe el paciente | Qué pasa por detrás |
|---|---|---|
| Se agota el cupo diario del proveedor | El agente se despide con voz: *"Se me acabó el tiempo disponible por ahora — alguien del equipo lo va a contactar pronto"* | Antes de llegar ahí, el sistema rota automáticamente entre las claves disponibles. Solo se despide cuando todas están agotadas, y deja la llamada registrada para seguimiento humano |
| Falla el servicio de extracción clínica | Nada: la llamada continúa normal | La extracción cae al proveedor de respaldo de forma automática. Son dos proveedores distintos; que caigan los dos a la vez es el escenario que dispara la fila siguiente |
| Falla técnica inesperada | El agente se despide con voz y avisa que alguien lo llamará | La conversación se guarda igual, con todo lo que se alcanzó a recoger. Nunca se pierde el estado clínico de un paciente por un error de software |
| Se cae la conexión a internet | La llamada no puede iniciarse | El habla y el razonamiento clínico se procesan en la nube. Sin conexión no hay servicio — igual que cualquier telefonía moderna. La síntesis de voz y el buscador de conocimiento sí son locales, así que una conexión intermitente degrada la calidad, no la disponibilidad |

**Ninguna condición de error deja la criticidad de un paciente en verde.** Cuando el
sistema no pudo confirmar lo que necesitaba, la llamada se cierra como *desconocida* y
pasa a revisión del equipo médico. El verde exige evidencia positiva de que no hay alarma;
la falta de información nunca cuenta como buena noticia.

**Sobre el consumo:** el agente está diseñado para operar dentro de los niveles gratuitos
de sus proveedores. El presupuesto de tokens por llamada se auditó y se redujo a menos de
la mitad, y las dos llamadas que hace cada turno se reparten entre dos proveedores con
cupos independientes, de modo que no compiten por el mismo límite. El detalle técnico y
las cifras medidas están en
[`docs/informe-final.md` §4.1](docs/informe-final.md#41-presupuesto-de-tokens--el-límite-real-de-una-cuenta-gratuita).

## Estado

Proyecto en construcción activa. Ver historial de commits para el progreso.
