# Informe final — Agente de voz postoperatorio

Tech Sphere Challenge 2026 · Entregable 03

> Este informe se redactó durante la construcción, con Claude Code como asistente de
> desarrollo. Todo lo que declara este documento se contrasta contra el código y los logs
> del repositorio — donde hay una limitación conocida, se nombra explícitamente en vez de
> maquillarla, siguiendo el mismo principio de honestidad que se le exige al agente.

---

## 1. El problema y el objetivo

El seguimiento postoperatorio depende hoy de personal humano llamando uno a uno a cada
paciente en las primeras horas y días tras el procedimiento. Es costoso, no escala, y
queda sujeto al criterio y disponibilidad de quien hace la llamada. El paciente, mientras
tanto, describe lo que siente en lenguaje cotidiano, ambiguo y regional — no en términos
clínicos — y el conocimiento contra el que hay que contrastar esos síntomas (protocolos,
guías de recuperación, instructivos por procedimiento) cambia de versión constantemente.

El objetivo de esta solución es un agente de voz que:

1. Conversa con el paciente en español, adaptándose a lo que responde.
2. Fundamenta cada afirmación clínica en un corpus de conocimiento verificable (RAG), y
   se abstiene explícitamente cuando no tiene una fuente que respalde lo que iba a decir.
3. Compara lo reportado contra la trayectoria de recuperación esperada para ese
   procedimiento y ese día postoperatorio — un dolor de 4/10 no significa lo mismo en el
   día 1 que en el día 14.
4. Decide cuándo escalar a personal humano, con una vía de decisión determinista que
   nunca puede ser silenciada por el LLM.
5. Deja un registro estructurado de cada llamada.

## 2. Arquitectura

**Diagrama de arquitectura**: [`diagrama-arquitectura.svg`](diagrama-arquitectura.svg)
**Diagrama de flujo de decisión**: [`diagrama-flujo-decision.svg`](diagrama-flujo-decision.svg)

![Arquitectura del agente](diagrama-arquitectura.svg)

La solución expone las dos superficies exigidas por el reto sobre un único servidor
FastAPI + WebSocket (`orchestrator/server.py`): la **interfaz de llamada** (mic/altavoz en
el navegador, streaming bidireccional de audio) y la **consola de administración**
(subir/listar/eliminar documentos del corpus, más un inspector de qué respondería el RAG a
una consulta dada).

Cada turno de voz sigue el camino: audio → STT (Groq Whisper Large V3) → texto →
`orquestar_turno()` → texto de respuesta → TTS (Piper, local) → audio. `orquestar_turno()`
es el punto de integración de los cinco patrones diferenciadores del diseño: extracción
clínica estructurada, arco reflejo, gemelo de trayectoria, memoria longitudinal y
validador de citas — su funcionamiento interno está detallado en el diagrama de flujo de
decisión (§3). Cada turno hace **dos llamadas al LLM**, no una: extracción (qué dijo el
paciente, tipado) y conversación (qué se le responde) — separadas a propósito, ver §3.

### Conocimiento vivo (compuerta G5)

Subir un documento desde la consola, o dejar caer un archivo en la carpeta `vault/`
vigilada, sigue el mismo camino: Docling lo convierte a Markdown (con fallback OCR para
PDFs escaneados sin capa de texto), se trocea, y se indexa en ChromaDB con embeddings
`multilingual-e5-base`. Eliminar el documento borra sus chunks del índice — el agente lo
olvida en la siguiente consulta, sin reiniciar el servidor.

**Por qué `multilingual-e5-base` y no BGE-M3** (que era la elección inicial, sugerida en
`stack-tecnico.md`): BGE-M3 pesa 4.3GB y `sentence-transformers` lo descarga de Hugging
Face la primera vez que se usa — no solo al construir el índice, sino en **cada consulta
en vivo**, porque la pregunta del paciente también se embebe con el mismo modelo. En una
máquina fresca del jurado, esa descarga por sí sola amenaza la compuerta G2 (≤15 minutos).
`multilingual-e5-base` pesa ~1.1GB — cuatro veces menos — y para compensar la posible
pérdida de precisión frente a un embedding más grande, la búsqueda es **híbrida**: se
combina el ranking vectorial (e5) con BM25 léxico (`rank_bm25`, sin modelo que descargar)
usando Reciprocal Rank Fusion. La ventaja concreta de sumar BM25: un embedding puede
"difuminar" semánticamente un término médico exacto (una dosis, el nombre de un fármaco,
un código), mientras que BM25 lo encuentra por coincidencia literal — las dos vías se
complementan en vez de competir. El índice de ChromaDB ya construido se comitea al
repositorio (`data/chroma/`), así que el arranque real no depende de reprocesar los 107
PDFs del corpus — ver §4 para el detalle del presupuesto de tiempo.

## 3. Lógica de decisión y escalamiento

![Flujo de decisión por turno](diagrama-flujo-decision.svg)

Cada turno del paciente pasa por **extracción y decisión**, en ese orden:

- **Extracción** (`clinical/extraction.py` + `orchestrator/cortex.extraer_turno()`) —
  llamada A al LLM, separada de la conversación (temperature 0.1, sin RAG: es una tarea
  cerrada de lectura). El LLM devuelve, para ese turno únicamente, dolor/fiebre/movilidad/
  herida/apetito/sueño con un **estado epistémico** explícito (`no_preguntado` ·
  `preguntado_sin_respuesta` · `ambiguo` · `rechazado` · `no_medible` · `confirmado`), más
  seis banderas rojas como `TriEstado` (`presente`/`ausente`/`no_evaluado`, nunca `bool`),
  y quién habló (paciente/tercero/inferido). El acumulador determinista
  (`clinical/estado.py:fusionar_extraccion()`) fusiona ese delta contra el estado vivo de
  la llamada: el estado **solo escala en severidad** — si el paciente reporta dolor 8 y
  diez turnos después dice "ya estoy mejor", el máximo de la llamada sigue siendo 8, y el
  intento de retractación queda registrado (`estado.correcciones`), no obedecido. Un
  tercero tampoco puede pisar un valor que el paciente ya confirmó.
- **Reflejo** (`clinical/reflex_engine.py`): reglas deterministas y un umbral de fiebre,
  sin LLM, ~5ms. Detecta banderas rojas explícitas (p. ej. sangrado activo, dolor torácico,
  dificultad respiratoria) que no deberían depender de que el LLM las interprete bien.
- **Cortex conversacional** (`orchestrator/cortex.generar_respuesta()`) — llamada B:
  recupera 4 fragmentos relevantes del RAG híbrido y se los pasa, junto con el turno, el
  historial y las **dimensiones aún pendientes de esta llamada**, a Llama 3.3 70B con
  `response_format=json_object` forzado — el modelo nunca devuelve texto libre, siempre el
  esquema de `RespuestaEstructurada`. Inyectar `dimensiones_pendientes` es lo que hace que
  el agente pregunte lo que falta, una dimensión por turno, en vez de repetir o dejar
  dimensiones sin cubrir en llamadas largas.

La **fusión** (`clinical/fusion.py`) toma `criticidad_final = max(reflejo, cortex)`: el
reflejo puede subir la criticidad que propuso el LLM, nunca bajarla. Encima de esa fusión,
`turn_manager.py` aplica una segunda compuerta: **`estado.puede_cerrar_verde`** — un verde
propuesto por el LLM se degrada a `desconocida` si no están las 6 dimensiones confirmadas,
o si hay alguna bandera roja presente, o si algún dato confirmado viene de un tercero sin
que el paciente lo haya corroborado. Esto convierte la regla del prompt ("verde solo con
evidencia positiva, nunca por defecto") en un chequeo de código, no solo una instrucción
que el modelo podría ignorar bajo presión — es el mismo patrón que ya usa el validador de
citas, aplicado a la clasificación de criticidad.

Después de la fusión, el **validador de citas** (`clinical/citation_validator.py`) revisa
cada afirmación clínica de la respuesta: si no trae un `chunk_id` que exista en ChromaDB,
la respuesta completa se descarta y se reemplaza por un mensaje honesto que deriva al
paciente a personal médico.

El **gemelo de trayectoria** (`clinical/trajectory_twin.py`) compara los síntomas
confirmados por el paciente (`clinical/estado.a_dict_trajectory_twin()`) contra
`trayectorias_postop_silver.xlsx` (paciente conocido) o, si el paciente no está en el
dataset de referencia, contra el promedio del arquetipo `recuperacion_normal` para ese
procedimiento y día — así un dolor reportado se juzga contra lo esperado para el día
postoperatorio actual, no en el vacío. También se endureció la validación de enums
(`trajectory_twin.py`): un literal fuera de la tabla ahora lanza `ValueError` en vez de
degradarse en silencio a `0` ("normal") vía `.get(x, 0)` — la falla insegura más grave que
tenía el diseño original (§1.2b de `docs/diseno-esquema-extraccion.md`).

Cuando la criticidad final es `amarillo` o `rojo`, se construye el **SBAR de escalamiento**
(`clinical/sbar.py`) — situación / contexto / evaluación / recomendación, con los síntomas
confirmados, las desviaciones frente a la trayectoria esperada y las referencias citadas.
Va en `ResultadoTurno.sbar` y se persiste en la memoria de la llamada.

La **memoria longitudinal** (`clinical/memory.py`) persiste el `EstadoClinicoLlamada`
completo por paciente al cerrar cada llamada (no solo un resumen de texto plano); la
siguiente llamada usa las dimensiones confirmadas y las que quedaron sin evaluar para
construir el contexto de apertura, en vez de pegar la transcripción cruda de la llamada
anterior.

### Decisión de alcance tomada en esta entrega

El diseño completo de `docs/diseno-esquema-extraccion.md` se implementó con un recorte
deliberado: los campos auxiliares de más detalle por dimensión (`dolor.tendencia`,
`dolor.localizacion_cambio`, `fiebre.medida`, `fiebre.sensacion_termica`) se dejaron
afuera para no inflar más el esquema JSON que debe llenar el LLM en una sola pasada — cada
campo extra es una oportunidad más de que la extracción falle. `procedencia` y `confianza`
también se piden una vez por turno (no repetidas en cada una de las ~15 dimensiones), y se
aplican uniformemente a todo lo extraído de ese turno — mismo efecto clínico que pedía el
diseño (§6.1 punto 4), esquema bastante más chico. Estas simplificaciones están anotadas
en el código (`clinical/extraction.py`), no aplicadas en silencio.

## 4. Modelo de lenguaje — declaración explícita (compuerta G3)

**Modelo usado: Llama 3.3 70B Versatile, vía Groq Cloud (nivel gratuito), familia Meta
Llama.**

Razones:

- **Latencia.** Las LPU de Groq entregan tokens a velocidad muy por encima de un GPU
  genérico; en una conversación de voz, cada segundo de espera antes de que suene la
  respuesta es un silencio incómodo que el paciente nota. Usar el mismo proveedor para LLM
  y STT (Whisper Large V3) además evita un salto de red adicional entre servicios.
- **Tamaño y capacidad de seguir instrucciones estructuradas.** 70B es suficiente para
  sostener el formato JSON forzado con afirmaciones citadas, sin la fragilidad que se ve en
  modelos más chicos al pedirles esquemas estrictos combinados con razonamiento clínico en
  español.
- **Nivel gratuito real**, sin tarjeta de crédito, dentro de la lista de familias
  permitidas por `stack-tecnico.md`.

**Costo real, no solo teórico, de esta elección:** el tier gratuito de Groq para este
modelo tiene un límite de **100,000 tokens/día** (tokens-per-day, "on-demand"). Con el
consumo medido antes de activar la extracción de síntomas (~1,060 tokens de entrada y ~70
de salida por turno, §6), ese presupuesto alcanza para ~88 llamadas al LLM por día. Desde
que cada turno hace **dos** llamadas (extracción + conversación, §3), el presupuesto
efectivo bajó a **~44 turnos de paciente por día** — el costo real de haber activado el
gemelo de trayectoria y el SBAR. Esto se documenta con logs reales en `harness_run.log` /
`harness_run2.log` y llevó a correr el harness de evaluación (§6) sobre una muestra
estratificada en vez del dataset completo. Es información relevante para cualquiera que
quiera reproducir este proyecto con cuentas gratuitas: la arquitectura escala, el tier
gratuito no.

## 5. Diseño de la conversación

El prompt de sistema completo vive en `orchestrator/prompts.py`. Resumen de las reglas que
lo componen:

- **Respuestas de 1 a 2 frases** — nunca párrafos largos en voz.
- **Tono colombiano moderado**, con un puñado de muletillas ("listo", "un momentico", "¿sí
  me entiende?") limitadas a 2-3 por turno para que no suene forzado.
- **Indagación incremental**: dolor, fiebre, herida, movilidad, apetito y sueño, una
  dimensión por turno, no un cuestionario de un tirón.
- **Interpretación de regionalismos** ("estoy maluco", "me arde comoquien dice") en
  contexto, pidiendo aclaración solo cuando de verdad hay ambigüedad.
- **Verde nunca por defecto** — se otorga solo con evidencia positiva de ausencia de
  alarma; ante la duda, el agente indaga antes de decidir.
- **Instrucciones de seguridad explícitas contra inyección de prompt**: instrucciones para
  ignorar la misión, "ahora eres otro asistente", pedidos ajenos al seguimiento clínico —
  se redirigen con amabilidad sin abandonar el guion clínico. Cubierto también por la
  suite adversarial en `tests/adversarial/`.
- **Formato de salida único**: un objeto JSON con `respuesta_hablada`,
  `afirmaciones_clinicas` (cada una con `chunk_id` y `documento`), `criticidad_propuesta` y
  `confianza` — nunca texto libre.

## 6. Métricas (§5 de la rúbrica)

> Medidas contra muestras estratificadas del dataset (no el total), por el límite de
> tokens/día del tier gratuito descrito en §4. Metodología y comando de reproducción en el
> [README](../README.md#métricas).

**Muestra principal — arquitectura previa a la extracción de síntomas** (1 llamada al LLM
por turno): 14 casos (los 12 `rojo` completos + 2 `amarillo`), capa `capa1_limpia`, 84
turnos de paciente. Resultados crudos en
[`harness_resultados_rojo_capa1.json`](../harness_resultados_rojo_capa1.json).

| Métrica | Valor |
|---|---|
| Latencia P50 / P95 (orquestación, sin STT/TTS) | 5.24s / 9.91s |
| Tokens de entrada por turno (media / P50) | 1,070 / 1,084 |
| Tokens de salida por turno (media / P50) | 79 / 80 |
| Invocaciones al modelo por turno | 1 (la vía refleja no usa LLM) |
| Consultas al RAG por llamada | 1 por turno de paciente (N_CHUNKS_RAG=4 por consulta) |
| Costo estimado por llamada (6 turnos) | ≈ USD 0.006 (fórmula y desglose abajo) |

**Muestra de validación — arquitectura actual, con extracción activada** (2 llamadas al
LLM por turno: extracción + conversación, §3). Muestra chica (1 caso, 6 turnos) porque el
presupuesto diario de tokens ya estaba parcialmente consumido por la muestra principal —
sirve para confirmar que el cambio de arquitectura funciona y para dimensionar el impacto
real en costo/latencia, no como muestra estadísticamente representativa:

| Métrica | Valor |
|---|---|
| Latencia P50 / P95 (orquestación, sin STT/TTS) | 3.40s / 15.63s |
| Tokens de entrada por turno (media / P50) | 1,295 / 1,269 |
| Tokens de salida por turno (media / P50) | 223 / 220 |
| Invocaciones al modelo por turno | **2** (extracción + conversación) |
| Cobertura de extracción al cierre de la llamada | 100% (6/6 dimensiones confirmadas) |

La latencia P50 bajó un poco frente a la muestra principal (extracción sin RAG es rápida),
pero la P95 subió bastante — dos llamadas secuenciales al mismo proveedor multiplican el
riesgo de que alguna pegue con un pico de latencia de Groq. Con más tiempo, correrlas en
paralelo (asyncio) recortaría ese P95 — ver §7.

### Lo que dice la matriz de confusión (§4 de la rúbrica: asimetría clínica)

Sobre los 72 turnos de casos `rojo` de la muestra: **0 falsos negativos catastróficos**
(ningún turno `rojo` real fue clasificado `verde`) — el mecanismo de fusión con veto del
reflejo (§3) está cumpliendo su función central. Pero el recall de `rojo` puro es bajo
(8.3%): el sistema sub-triaje hacia `amarillo` en 49/72 turnos en vez de escalar al
máximo nivel, y declara `desconocida` — honestamente, en vez de inventar — en 17/72. El
recall de `amarillo` es 41.7% (5/12, sobre una muestra chica de solo 2 casos). El reflejo
determinista vetó (subió la criticidad del LLM) en 6 de los 84 turnos evaluados.

Lectura honesta: el diseño evita la falla más grave (rojo→verde), pero el cortex es
conservador prediciendo `amarillo` donde correspondía `rojo`, y `reflex_rules.py` no
cubre todavía los patrones que llevan a esos 49 casos — ver §7, punto 3.

### Cómo se calcula el costo estimado por llamada

La solución corre sobre APIs gratuitas (Groq) más un componente local (Piper TTS, sin
costo marginal). Para extrapolar a precios de producción:

- **LLM — Llama 3.3 70B en Groq**: USD 0.59 / millón de tokens de entrada, USD 0.79 /
  millón de tokens de salida (tarifa on-demand, agosto 2026).
- **STT — Whisper Large V3 en Groq**: USD 0.111 / hora de audio, con mínimo de facturación
  de 10s por solicitud.
- **TTS — Piper**: local, sin costo de API; el costo real es cómputo/hosting amortizado,
  no facturación por llamada.

```
costo_llamada ≈ n_turnos × [ (tokens_in/1e6 × 0.59) + (tokens_out/1e6 × 0.79) ]
              + minutos_audio_paciente × (0.111/60)
```

Con los promedios de la muestra principal (§6) — 1,070 tokens de entrada y 79 de salida
por turno, 1 llamada al LLM — y una llamada típica de 6 turnos de paciente:

- LLM: 6 × [(1070/1e6 × 0.59) + (79/1e6 × 0.79)] ≈ **USD 0.0042**
- STT: 6 turnos al mínimo de facturación de 10s ≈ 1 minuto de audio ≈ **USD 0.0019**
- **Total ≈ USD 0.006 por llamada** (seis décimas de centavo de dólar).

Con la arquitectura actual (extracción + conversación, 2 llamadas por turno) el
componente LLM aproximadamente se duplica: usando los promedios de la muestra de
validación (1,295 tokens de entrada / 223 de salida, ya cuenta las dos llamadas sumadas
por turno), 6 × [(1295/1e6 × 0.59) + (223/1e6 × 0.79)] ≈ **USD 0.0057** de LLM, más el
mismo componente de STT (≈ USD 0.0019) → **Total ≈ USD 0.0076 por llamada** — sigue siendo
menos de un centavo de dólar, pero es casi 30% más caro que antes de activar la
extracción. Es el costo explícito de que `trajectory_twin` y el SBAR corran con datos
reales en vez de código muerto.

Es una cota conservadora por abajo: asume utterances del paciente en el mínimo de
facturación de Whisper (10s); pacientes más locuaces suben el componente de STT, no el de
LLM (que depende de tokens, no de duración de audio).

## 7. Trabajo pendiente / qué haría con más tiempo

En orden de prioridad:

0. ~~Implementar la extracción real de síntomas~~ (`docs/diseno-esquema-extraccion.md`).
   **Hecho durante la construcción** — ver §3 y §8: `clinical/extraction.py` +
   `clinical/estado.py` implementan el esquema completo (`Observacion[T]`, `EstadoSlot`,
   `TriEstado`, fusión con regla "solo escala"), con el recorte de alcance documentado en
   §3. `trajectory_twin` y el SBAR corren ahora con datos reales, verificado en vivo.
1. ~~Conectar `construir_sbar()` a `turn_manager.py`~~. **Hecho** — se construye
   automáticamente cuando `criticidad_final` es `amarillo` o `rojo` (§3), verificado en
   la prueba end-to-end de §8 (`sbar` no `None` en un turno con fiebre de 39°C).
2. **Paralelizar las dos llamadas al LLM por turno** (extracción y conversación) con
   `asyncio` en vez de secuenciales — son independientes entre sí (ninguna depende de la
   salida de la otra dentro del mismo turno), así que correrlas en paralelo recortaría la
   latencia percibida casi a la mitad sin tocar el presupuesto de tokens. Es el ajuste de
   mayor impacto que falta y no se alcanzó a hacer en el tiempo disponible.
3. **Correr el harness completo (160 casos × 2 capas)** en cuanto el presupuesto de
   tokens/día lo permita — ahora más lejos todavía, porque la extracción duplicó el costo
   por turno (§4, §6).
4. **Investigar el recall bajo en `rojo`** observado en la muestra inicial — el reflejo
   determinista no está capturando todos los patrones que el LLM subestima; requiere
   revisar `clinical/reflex_rules.py` contra los casos rojo que fallaron. Sigue pendiente
   con la arquitectura nueva — sería el primer punto a re-evaluar con una muestra grande.
5. **Implementar los campos auxiliares recortados en esta entrega** (§3): tendencia del
   dolor, si el dolor migró, si la fiebre fue medida o sentida — quedaron fuera del
   esquema JSON de la llamada A para no arriesgar su validez, documentado explícitamente
   en `clinical/extraction.py`, no aplicado a medias.
6. ~~Endurecer el arranque en ≤15 minutos (compuerta G2) con una prueba de instalación
   limpia de punta a punta en una máquina sin el entorno preconfigurado.~~ **Hecho durante
   la construcción** — ver §8: la prueba de instalación limpia encontró dos riesgos reales
   contra G2 (reindexado completo del corpus, tamaño del modelo de embeddings) y ambos se
   corrigieron antes de la entrega, no quedaron como hallazgo sin resolver.
7. Medir en la práctica el tiempo de arranque en limpio con el índice ya comiteado
   (`git clone` → `uvicorn`, sin `build_index`) para tener un número real de G2, no solo la
   expectativa de diseño.

## 8. Evidencia del proceso con IA

Este proyecto se construyó con **Claude Code** como asistente de desarrollo — nunca como el
LLM que razona dentro de la llamada (ver §4). El historial de commits documenta el proceso
por etapas: andamiaje del proyecto, motor clínico + ingesta, loop de voz end-to-end,
harness de evaluación + suite adversarial, y la interfaz de llamada final.

Durante la evaluación (día 2 de construcción) se detectó y corrigió en vivo un bug real del
harness: `reproducer.py` intentaba convertir `dialogo_id` — un identificador de texto como
`dlg_caso_tray_pac_42_00027_7_0` — a entero, lo que lanzaba `ValueError` en cada caso y era
capturado en silencio por un `except` demasiado amplio en `runner.py`, dando "0 turnos
evaluados" sin ningún error visible. La corrección (`dialogo_id: str` en vez de `int`, y el
`except` ahora imprime la causa) está en el historial de commits de esta fecha.

También durante el día 2, y en gran parte gracias a haber tenido que clonar el repositorio
en una carpeta nueva por un problema de espacio en OneDrive, se ejecutó sin querer la
primera prueba real de instalación limpia (venv nuevo, sin caché, sin `data/` previo) —
justo lo que pedía el punto 4 de §7. Esa prueba encontró dos riesgos concretos contra la
compuerta G2 que no eran visibles trabajando siempre sobre el mismo entorno ya
preparado:

- **El reindexado completo de los 107 PDFs con Docling en CPU toma del orden de una
  hora** — inviable como parte del arranque documentado en el README. Corrección: el
  índice de ChromaDB ya construido se comitea al repositorio (`data/chroma/`); el README
  ya no le pide al jurado reindexar, solo cargar lo ya calculado.
- **El modelo de embeddings original (BGE-M3) pesa 4.3GB y se descarga de Hugging Face
  también en cada consulta en vivo**, no solo al indexar — un riesgo serio contra los 15
  minutos en una máquina sin el modelo cacheado. Corrección: se cambió a
  `multilingual-e5-base` (~1.1GB) y se compensó la posible pérdida de precisión sumando
  búsqueda léxica BM25 en paralelo a la vectorial (Reciprocal Rank Fusion) — ver §2.

Ninguno de los dos hallazgos se dejó como nota al margen: ambos se corrigieron en el
código antes de esta versión del informe.

Al activar la extracción de síntomas, la primera prueba en vivo contra el LLM falló con un
error de validación de Pydantic en tres campos (`medicacion.toma_analgesico`,
`contexto.acompanado`, `contexto.transporte_disponible`): el LLM devolvía `null` explícito
en vez de omitir el campo, y Pydantic solo aplica el valor por defecto cuando el campo
está *ausente*, no cuando llega `null` explícito — un `null` explícito pisa el default.
El turno no se cayó (la extracción está envuelta en `try/except` en `turn_manager.py`,
justo para que una extracción fallida no tumbe la conversación), pero la cobertura de esa
llamada se quedó en 0. Corrección: esos campos pasaron a aceptar `TriEstado | None`, con
una normalización explícita (`_o_no_evaluado()`) que trata `None` igual que
`"no_evaluado"` antes de construir el `Observacion`. Verificado con la misma prueba en
vivo repetida — ver la tabla de la muestra de validación en §6.

## Capturas del demo

_Pendiente — se completa con capturas de la interfaz de llamada y la consola en una sesión
en vivo antes de grabar el video (entregable 04)._
