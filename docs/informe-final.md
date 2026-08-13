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

**Modelos usados — dos, ambos dentro de las familias permitidas:**

| Llamada | Modelo | Familia (`stack-tecnico.md` §1) |
|---|---|---|
| B — conversación con el paciente | `llama-3.3-70b-versatile` vía Groq Cloud (nivel gratuito) | Meta Llama |
| A — extracción clínica del turno | `gemini-flash-latest` vía Google AI (nivel gratuito) | Google Gemini, gama Flash |

El alias `-latest` en Gemini es deliberado: `stack-tecnico.md` advierte que los proveedores
retiran snapshots sin aviso y por eso fija familias y no versiones. Se comprobó en la
práctica — `gemini-2.0-flash` ya responde 404. `EXTRACCION_EN_GEMINI=false` devuelve todo a
Groq sin tocar código, y ante cualquier fallo de Gemini la extracción cae a Groq
automáticamente (`orchestrator/cortex.py`), porque perder la extracción es perder la
cobertura, y sin cobertura no hay verde posible.

**Por qué la extracción no corre en el modelo grande:** es una tarea cerrada —leer un turno
y mapearlo a un vocabulario fijo de seis dimensiones y seis banderas—, no un problema de
razonamiento clínico. Lo que sí aporta separarla es que consume un cupo distinto: repartir
las dos llamadas del turno entre dos proveedores no divide el presupuesto, lo duplica.

Razones de la elección de Llama 3.3 70B para la conversación:

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

### 4.1 Presupuesto de tokens — el límite real de una cuenta gratuita

El tier gratuito de Groq da **100,000 tokens/día**. Ese número, y no la arquitectura, es lo
que fija cuántas llamadas puede atender el agente, así que se auditó turno a turno
reconstruyendo los prompts sin invocar la API (medición de costo cero).

**Punto de partida: 5.809 tokens por turno.** Una llamada de 12 turnos costaba 69.708
tokens: **1,4 llamadas por día**. El desglose mostró que solo ~58 tokens por turno (el 1%)
eran información nueva; el 99% era andamiaje re-comprado en cada turno.

| Componente | Antes | Después |
|---|---|---|
| Contexto RAG | 2.917 (50,2%) | 672 |
| `SYSTEM_PROMPT_EXTRACCION` | 1.138 | 805 |
| `SYSTEM_PROMPT` (conversación) | 695 | 614 |
| Salida generada | 470 | 230 |
| Historial (se mandaba duplicado) | 379 | 189 |
| **Total por turno** | **5.809** | **2.736** |

Las cuatro medidas, en orden de impacto:

1. **El chunker rendía 2,2× su tamaño declarado.** `TAMANO_OBJETIVO_CHARS = 1200`, pero el
   86% del índice lo excedía (mediana 2.631 chars, máximo 12.614). La causa: un párrafo más
   largo que el objetivo nunca se subdividía —el Markdown que Docling produce desde PDFs
   clínicos está lleno de ellos— y salía como un chunk único. Corregido en
   `ingestion/chunking.py`; **el índice comiteado conserva los chunks viejos** (re-indexar
   cuesta ~2,3 h de CPU) y la corrección aplica a toda ingesta nueva por el vault.
2. **Recorte al inyectar, no al indexar.** El chunk que conviene recuperar (amplio, más
   superficie de match) no es el que conviene enviar. Se manda el pasaje contiguo más
   relacionado con la consulta (~700 chars) en vez del chunk entero, conservando el
   `chunk_id`: el validador de citas verifica el id, no la longitud, así que la trazabilidad
   queda intacta. Esto rescata la mayor parte del ahorro sin re-indexar.
3. **RAG condicional.** El propio prompt de sistema dice que en un turno de pura indagación
   la lista de afirmaciones va vacía; ahí los fragmentos se pagaban para que el modelo
   tuviera prohibido usarlos. Sobre los 960 turnos de paciente del dataset, el 35% son
   reportes de normalidad sin pregunta ni señal de anomalía y no reciben fragmentos. Una
   bandera refleja siempre fuerza la recuperación.
4. **La extracción dejó de emitir nulls.** El prompt exigía las ~15 claves del esquema
   siempre presentes. Era redundante: `ExtraccionCruda` da default seguro a cada campo
   ausente y `_fusionar_dimension` descarta el delta `NO_PREGUNTADO`, de modo que
   "omitir == no preguntado" ya estaba garantizado por código determinista. Pedírselo además
   al modelo era pagar dos veces por la misma garantía — y la más frágil de las dos.

**El cambio estructural: dos presupuestos, no uno.** Mover la extracción a Gemini Flash
reparte el turno entre dos cupos independientes, así que el límite ya no es la suma sino el
mayor de los dos:

| Presupuesto | tok/turno | Llamada de 12 turnos | Llamadas/día/clave |
|---|---|---|---|
| Groq (Llama 3.3 70B) | 1.708 | 20.498 | 4,9 |
| Gemini (Flash) | 1.028 | 12.333 | 8,1 |
| **Cuello de botella** | | **20.498** | **4,9** |

De **1,4 a 4,9 llamadas por día por clave — 3,5×** — sin tocar el motor de reflejos, la
fusión determinista ni el validador de citas, que son los que sostienen la seguridad. La
suite adversarial (8 escenarios) pasa completa tras la compresión de los prompts.

Sigue siendo cierto lo esencial: **la arquitectura escala, el tier gratuito no.** Pero el
techo pasó de una llamada por día a cinco por clave, y el pool de claves con rotación
automática (`clients.py`) multiplica desde ahí.

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

### 6.1 El recall de `rojo` de 8,3 % era un artefacto de medición

El primer reporte daba **recall de `rojo` = 8,3 %** y se documentó como una limitación del
agente. Al ir a corregirla apareció que el número medía otra cosa.

`label_ground_truth` es una etiqueta **de caso, no de turno**: los 160 casos del dataset
tienen el mismo valor en todos sus turnos (comprobado: `groupby('caso_id').nunique() == 1`
para los 160). Un caso es `rojo` por el cuadro completo de la llamada, no porque cada turno
lo sea. En `caso_tray_pac_42_00017_7`, por ejemplo, la etiqueta es `rojo` y **ningún turno
aislado lo es**: es un paciente que minimiza («un poquito molesto no más», «37 y algo, nada
de escalofríos», «un poquito rojita pero nada de pus»), y la señal está en el conjunto.

El harness comparaba la criticidad *de cada turno* contra esa etiqueta *de caso*. Es decir,
penalizaba al agente por no gritar ROJO cuando el paciente contestaba que había dormido mal.
Reagregando los mismos resultados crudos por caso —¿a qué nivel llegó a escalar la llamada?—:

| Vista | Recall `rojo` | Casos `rojo` que colgaron sin escalar |
|---|---|---|
| Por turno (lo que se reportaba) | 8,3 % | — |
| **Por caso** (como está etiquetado el dataset) | **41,7 %** | **0 de 12** |

`harness/report.py` ahora imprime las dos vistas. La de turno mide reactividad turno a
turno; la de caso responde a la pregunta clínica real: al colgar, ¿esta llamada quedó
escalada? Ninguna de las dos sustituye a la otra, pero solo la segunda es comparable contra
la etiqueta del dataset.

### 6.2 Dos fallos reales de la vía refleja, y lo que valían

Con la métrica corregida, el 41,7 % restante sí era mejorable. La vía refleja se evaluó
contra los 160 casos **sin gastar un solo token** (es determinista: no hay LLM en el
camino), lo que permitió iterar sobre el dataset completo en vez de sobre una muestra de 14.

**Fallo 1 — la fiebre se perdía si el paciente no decía «grados».** `extraer_temperatura_c`
exigía la unidad (`(\d{2})\s*(?:°|grados)`), pero el paciente dice «me la tomé y marcó
como 38», «marcaba como 39 algo», «38 y algo». **5 de los 7 casos `rojo` no detectados
reportaban una temperatura ≥ 38 en palabras.** La fiebre es la bandera roja más común del
postoperatorio y se estaba perdiendo entera. Ahora se acepta la forma coloquial cuando el
turno habla de temperatura (`fiebre`, `afiebrada`, `marcó`, `escalofrío`…), acotada al rango
fisiológico 35–42,5 °C para que la intensidad del dolor («como un 5») no se lea como fiebre.

**Fallo 2 — la regla `pus` disparaba con la negación.** `pus` se buscaba como subcadena
suelta, así que «no le sale nada de líquido **ni pus**» y «no veo **pus** ni nada raro»
—turnos donde el paciente *descarta* el síntoma— forzaban ROJO, igual que «me **pus**ieron
suero». Se añadió detección de negación que mira **solo hacia atrás**, para que las reglas
que llevan el «no» dentro del patrón («no puedo respirar», «no para de sangrar») sigan
disparando intactas.

Además, el paciente no dice «secreción»: dice «un líquido, amarillo creo, saliendo de la
herida». Se añadió una regla de coocurrencia drenaje + color.

Resultado sobre los 160 casos, vía refleja sola:

| | Antes | Después |
|---|---|---|
| Recall `rojo` (capa1 / capa2) | 41,7 % / 41,7 % | **66,7 % / 58,3 %** |
| Falsos positivos sobre casos `verde` | 13,0 % (16/123) | **0,0 % (0/123)** |
| Sobre-escalamiento de casos `amarillo` a rojo | 48,0 % | 4,0 % |

Los 4 casos `rojo` que siguen sin detectarse no tienen ninguna señal dura en ningún turno:
son pacientes que evaden toda medición («no le he puesto cuidado a eso», «acalorada a
ratos»). Ahí la respuesta correcta no es un reflejo sino indagar, que es lo que ya hace
`puede_cerrar_verde` al degradar a `desconocida` en vez de cerrar en verde. Fijado en
`tests/test_reflex_rules.py` (31 casos, todos derivados del dataset).

### 6.3 Latencia: lo que se reportaba no era lo que la rúbrica mide

La tabla de arriba reporta «latencia de orquestación, sin STT/TTS». La rúbrica §5 pide otra
cosa: **desde que el paciente termina de hablar hasta que empieza a sonar el audio del
agente**. Son magnitudes distintas y la segunda siempre es mayor. Además, el camino de voz
real no estaba instrumentado —`server.py` no tenía un solo `perf_counter`— y **el logging
nunca se configuró**, así que todo `logger.info` se descartaba en silencio: no había logs
que contrastar.

Ahora cada turno emite una línea con las etapas desglosadas, a consola y a
`agente_postop.log`:

```
turno paciente=pac_42_00017 criticidad=rojo rag_ms=763 llm_conversacion_ms=697
  llm_extraccion_ms=5000 orquestacion_ms=5848 tts_ms=1076 total_ms=6968 ttfr_ms=6975
```

Lo primero que mostró esa instrumentación fue un fallo que llevaba tiempo escondido:

```
llm_conversacion_ms=5415  llm_extraccion_ms=36445  tts_ms=527  total_ms=36993
```

La extracción —la tarea *barata*, la que se movió a Gemini Flash justo por ser cerrada y
liviana— costaba **siete veces** la conversación. Dos causas: `generar_json_gemini` llamaba
a `generate_content` **sin timeout** (el cliente de Groq sí fallaba rápido, con
`TIMEOUT_S=15`), y `MAX_TOKENS_EXTRACCION=500` truncaba el JSON a mitad de cadena
(`Unterminated string starting at line 2 column 25`), lo que invalidaba la respuesta y
forzaba la caída a Groq — pagando dos llamadas donde debía pagar una, y perdiendo justo el
cupo separado que justifica usar Gemini.

Corregidos el timeout y el techo de tokens: **36,9 s → 7,0 s por turno**.

| Etapa | ms (turno medido en vivo) |
|---|---:|
| RAG (embedding e5 en CPU + Chroma + BM25) | 763 |
| LLM conversación (Llama 3.3 70B) | 697 |
| LLM extracción (Gemini Flash) | 5 000 |
| Orquestación (máximo de las dos, en paralelo) | 5 848 |
| TTS (Piper, respuesta completa) | 1 076 |
| **Total del turno** | **6 968** |

Dos consecuencias de tener por fin el desglose:

- **El TTS no era el cuello de botella.** Con 527–1 076 ms sobre ~7 000, es el 7–15 % del
  turno. Se había considerado partir la síntesis por frases para emitir la primera antes;
  con estos números, eso gana ~400 ms a cambio de tocar el protocolo del WebSocket y el
  cliente, con G4 en juego. Se descartó **por la medición**, no por falta de tiempo.
- **El primer turno pagaba la carga de los modelos.** Piper y `multilingual-e5-base` se
  cargaban de forma perezosa: ~25 s en el primer turno de la llamada, justo el que ve el
  jurado. Eso explicaba el P95 de 9,91 s. Ahora se precargan en el `lifespan` de FastAPI, y
  esos segundos se pagan en el arranque del servidor (que no cuenta contra G2: el reloj mide
  el levantamiento y esto ocurre dentro de él, sumando ~25 s a un procedimiento de minutos).

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
2. ~~**Paralelizar las dos llamadas al LLM por turno**~~ (extracción y conversación).
   **Hecho** — corren concurrentes en un `ThreadPoolExecutor` (`turn_manager.py`), así que
   el turno paga el máximo de las dos y no la suma. El cronómetro por etapa de
   `orchestrator/metrics.py` lo confirma en los logs: `llm_conversacion_ms` y
   `llm_extraccion_ms` se solapan dentro de `orquestacion_ms`.
3. **Correr el harness completo (160 casos × 2 capas)** en cuanto el presupuesto de
   tokens/día lo permita — ahora más lejos todavía, porque la extracción duplicó el costo
   por turno (§4, §6).
4. ~~**Investigar el recall bajo en `rojo`**~~. **Hecho, y el diagnóstico inicial era
   equivocado** — ver §6.1. Dos causas, ninguna de las cuales era «el LLM subestima»: la
   métrica comparaba criticidad *de turno* contra una etiqueta *de caso*, y la vía refleja
   perdía la fiebre cuando el paciente no decía «grados». Corregidas ambas, el recall de la
   vía refleja sobre los 160 casos pasa de 41,7 % a 66,7 % y los falsos positivos sobre
   casos `verde` caen de 13 % a 0 %. Queda pendiente re-correr el harness completo (punto 3)
   para medir el sistema entero, no solo la vía refleja.
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

### La primera prueba real con micrófono (compuerta G4) encontró cuatro problemas reales

Antes de esta prueba, todo se había validado por el bypass de texto del harness o con
llamadas manuales al orquestador — nunca por el flujo completo navegador → micrófono →
WebSocket. La primera prueba en vivo encontró, en orden de aparición:

1. **El modelo de voz de Piper estaba corrupto** — quedó truncado a 28.8MB de los ~63MB
   esperados, casi seguro por una descarga en background que se cortó durante una
   interrupción de sesión el día anterior. `onnxruntime` fallaba con `InvalidProtobuf` al
   cargarlo. Se volvió a descargar completo y se verificó con una síntesis de prueba antes
   de reintentar.
2. **Los nombres de procedimiento no coincidían entre la interfaz y el índice del RAG.**
   El dropdown y `reflex_rules.py` usan nombres clínicos en español ("Apendicectomía",
   "Mastectomía"...); el índice de ChromaDB usa el nombre de la carpeta de
   `dataset/textos/` (inglés: "Appendicitis", "breast_cancer"...). El filtro `where` de
   ChromaDB exige coincidencia exacta, así que **cualquier procedimiento elegido en el
   demo devolvía el RAG vacío**, sin importar lo que dijera el paciente — el validador de
   citas habría descartado entonces cada respuesta clínica. Corrección: mapeo explícito en
   `cortex.py` (`MAPEO_PROCEDIMIENTO_A_CORPUS`) que traduce el nombre clínico a la
   etiqueta indexada, solo en el punto donde se arma el filtro del RAG — no se tocó
   `reflex_rules.py` ni el dataset.
3. **Una excepción sin manejar mataba la conexión del WebSocket en silencio.** Tanto el
   Piper corrupto como, después, un `RateLimitError` de Groq (cupo diario agotado durante
   la prueba — ver más abajo) dejaban al paciente esperando una respuesta que nunca iba a
   llegar, sin ningún mensaje de error. En una sesión de evaluación en vivo esto se lee
   como "el agente se colgó", no como un límite de cuota. Corrección: `server.py` ahora
   atrapa el error, sintetiza un mensaje de cierre audible con Piper (sin depender de
   Groq) y cierra la conexión explícitamente — nunca cuelga en silencio. La memoria de la
   llamada se guarda igual, con un nuevo `motivo_cierre = error_tecnico`.
4. **La latencia real (~90s en un turno) resultó de reintentos automáticos del cliente de
   Groq** (`max_retries=2`, timeout de lectura 60s por defecto) apilados sobre dos
   llamadas secuenciales por turno, justo cuando el cupo diario ya estaba casi agotado.
   Corrección doble: se bajó el cliente a 1 reintento / 15s de timeout (falla rápido en
   vez de reintentar en silencio), y las dos llamadas del turno (extracción y
   conversación) pasaron a correr **en paralelo** con `ThreadPoolExecutor` en vez de
   secuenciales — son independientes entre sí dentro de un mismo turno. El costo de esto:
   la llamada de conversación ya no ve el delta de extracción de ese mismo turno para
   armar `dimensiones_pendientes` y la desviación de trayectoria, usa el estado de
   *inicio* del turno — una pérdida de frescura mínima, documentada en el código
   (`turn_manager.py:_desviaciones_relevantes`), no aplicada en silencio.

Ninguno de los cuatro se hubiera visto corriendo solo el harness por texto — es la
diferencia concreta entre probar el orquestador y probar la llamada real.

## Capturas del demo

_Pendiente — se completa con capturas de la interfaz de llamada y la consola en una sesión
en vivo antes de grabar el video (entregable 04)._
