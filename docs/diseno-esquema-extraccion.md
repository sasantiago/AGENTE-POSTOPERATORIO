# Diseño del esquema de extracción — variables clínicas

Documento de diseño. Define **qué variables extrae el LLM en cada turno**, cómo se acumulan
en el estado vivo de la llamada y qué se persiste en el resumen final.

Referencias de código: `src/agente_postop/clinical/models.py`, `clinical/memory.py`,
`orchestrator/turn_manager.py`, `orchestrator/server.py`, `clinical/trajectory_twin.py`.

---

## 1. Diagnóstico del esquema actual

### 1.1 El hallazgo principal: la extracción no existe

`RespuestaEstructurada` (`clinical/models.py:42`) es lo único que el LLM devuelve:

```python
respuesta_hablada: str
afirmaciones_clinicas: list[AfirmacionClinica]
criticidad_propuesta: Criticidad
confianza: str
```

**No hay ni una sola variable clínica.** El LLM decide criticidad pero nunca reporta *sobre
qué evidencia*. Consecuencias verificables hoy en el repo:

| Síntoma en el código | Ubicación | Efecto |
|---|---|---|
| `sintomas_extraidos=None` hardcodeado | `orchestrator/server.py:103` | El gemelo de trayectoria **nunca corre en producción** |
| `sintomas_extraidos=None` hardcodeado | `harness/runner.py:55` | El harness **tampoco lo evalúa** → métrica ciega |
| `sesion.sintomas_reportados` nunca se escribe | `server.py:57` (solo se lee en :131) | `ResumenLlamada.sintomas_reportados` siempre es `{}` |
| `f"Síntomas reportados: {sintomas_reportados}"` | `clinical/sbar.py:57` | El SBAR de escalamiento imprime literalmente `{}` |
| `contexto_apertura()` usa `resumen_texto` | `clinical/memory.py:64` | La memoria longitudinal arrastra transcripción cruda, no datos |

`trajectory_twin.py` —el diferenciador conceptual del proyecto— es **código muerto**. Está
escrito, testeado por su cuenta, y nunca se invoca con datos reales.

### 1.2 Fallas de diseño en los campos que sí existen

**a) `sintomas_reportados: dict[str, str]` — sin tipar, sin validar** (`memory.py:23`)

Un `dict[str, str]` no puede representar `dolor_nrs` (numérico) ni `fiebre_c` (float). Pydantic
no valida claves ni valores. Es un contrato nulo.

**b) Falla insegura hacia verde en `comparar()`** (`trajectory_twin.py:135`)

```python
valor_reportado = orden.get(reportado[dimension], 0)
```

Si el LLM devuelve `"leve"`, `"un poco roja"` o `"eritema"` en lugar del literal exacto
`"eritema_leve"`, `.get(..., 0)` lo degrada silenciosamente a **0 = normal → no empeora**.
Un error de formato del LLM se convierte en un falso verde. En un agente clínico esto es
la peor dirección posible de fallo. El esquema debe hacer imposible el valor libre (enum
cerrado, validación estricta) y el default debe ser "desconocido", nunca "normal".

**c) No hay estado epistémico — el vacío es ambiguo**

El propio system prompt (`orchestrator/prompts.py:19`) dice:

> *"Verde solo se otorga con evidencia positiva de ausencia de alarma — nunca por defecto,
> nunca por falta de información."*

Pero el esquema no puede expresar la diferencia entre:

- no le he preguntado por la herida
- le pregunté y me evadió
- le pregunté y dijo "normal"

Las tres producen la misma ausencia de clave en el dict. **La regla más importante del
prompt no es representable en el tipo de datos.** Con 25 amarillos y 12 rojos sobre 160
casos, el sesgo hacia verde es exactamente el modo de fallo que arruina el recall.

**d) No hay cobertura → el slot-filling no puede funcionar**

El prompt pide indagar seis dimensiones "una por turno" (`prompts.py:14`), pero sin estado
persistido el LLM decide la siguiente pregunta solo con `historial_turno` (últimos 6 turnos,
`server.py:61`). En una llamada larga o con capa 2 ruidosa, se le olvida qué falta y repite
o abandona dimensiones.

**e) `confianza: str` libre** (`models.py:48`)

No es enum. Nadie lo lee. Puede llegar `"alta"`, `"Alta"`, `"bastante alta"`.

**f) Sin atribución de hablante**

El dataset tiene `hablante ∈ {agente, paciente, tercero}` y la capa 2 inserta turnos de
familiares (`_c2_tercero`). Un dato dicho por el acompañante ("yo lo veo bien") no tiene el
mismo peso que el del paciente, pero el esquema no puede distinguirlo.

**g) Sin verbatim → sin trazabilidad de la extracción**

Hay trazabilidad ejemplar del RAG (`AfirmacionClinica.chunk_id` + `citation_validator`), pero
cero trazabilidad de la extracción. Si el agente escala por dolor 8/10, no queda registro de
qué dijo el paciente para producir ese 8. Auditoría clínica imposible.

**h) Todo mezclado en un objeto**

`RespuestaEstructurada` mezcla tres responsabilidades: qué decir, qué se entendió, y qué tan
grave es. El LLM las resuelve en una sola pasada, y la extracción se recalcula desde cero
cada turno sin acumularse.

---

## 2. Arquitectura propuesta: tres objetos, no uno

```
turno del paciente
        │
        ├──► ExtraccionTurno      (delta: SOLO lo nuevo de este turno)
        │            │
        │            └──► fusión con EstadoClinicoLlamada  (estado vivo, acumulado)
        │                          │
        │                          ├──► decide la siguiente pregunta (cobertura)
        │                          ├──► alimenta trajectory_twin.comparar()
        │                          └──► alimenta el SBAR
        │
        └──► RespuestaEstructurada (qué se dice + citas + criticidad)  ← ya existe

  fin de llamada ──► ResumenLlamada (snapshot congelado del estado + auditoría)
```

**Principio rector:** el LLM extrae **deltas**, no estado. Nunca se le pide que reproduzca lo
que ya sabe — eso invita a que lo altere. El acumulador es código determinista, no el modelo.

**Segundo principio:** el estado solo puede escalar en severidad dentro de una llamada, igual
que la vía refleja solo sube criticidad (`clinical/fusion.py`). Si el paciente dice dolor 8 y
diez turnos después dice "ya estoy mejor", el máximo de la llamada sigue siendo 8 y queda
registrado como tal.

---

## 3. Esquema de variables

### 3.1 Tipo base: `Observacion[T]`

Cada dimensión clínica no es un valor, es una observación con procedencia. Este es el cambio
estructural que resuelve (b), (c), (f) y (g) de un golpe.

| Campo | Tipo | Oblig. | Descripción |
|---|---|---|---|
| `valor` | `T \| None` | sí | El valor tipado. `None` si no se pudo determinar |
| `estado` | `EstadoSlot` | sí | Estado epistémico. **Default: `NO_PREGUNTADO`** |
| `verbatim` | `str \| None` | sí | Cita textual del paciente que produjo el valor. `None` si `estado ∈ {NO_PREGUNTADO}` |
| `turno_idx` | `int \| None` | sí | Turno en que se capturó — para reconstruir la línea de tiempo |
| `procedencia` | `Procedencia` | sí | Quién lo dijo |
| `confianza` | `Confianza` | sí | Certeza del LLM sobre la interpretación |

**`EstadoSlot`** (enum cerrado) — el corazón del diseño:

| Valor | Significado | ¿Habilita verde? |
|---|---|---|
| `NO_PREGUNTADO` | El agente aún no indagó esta dimensión | **No** |
| `PREGUNTADO_SIN_RESPUESTA` | Se preguntó, el paciente no contestó o cambió de tema | **No** |
| `AMBIGUO` | Respondió pero no es mapeable a un valor del enum ("maluco", "más o menos") | **No** |
| `RECHAZADO` | Se negó explícitamente a responder | **No** |
| `NO_MEDIBLE` | No tiene el instrumento ("no tengo termómetro") | **No** |
| `CONFIRMADO` | Valor extraído con confianza | **Sí** |

Solo `CONFIRMADO` con `valor` no nulo puede sustentar un verde. Todo lo demás fuerza, como
mínimo, `desconocida` en esa dimensión — que en `Criticidad.rango` ya vale lo mismo que
amarillo (`models.py:21`). La regla del prompt se vuelve estructural.

**`Procedencia`**: `PACIENTE` · `TERCERO` (familiar/acompañante) · `AGENTE_INFERIDO`.
Un valor con procedencia `TERCERO` o `AGENTE_INFERIDO` **no puede sustentar un verde por sí
solo**; requiere confirmación del paciente. Esto es lo que blinda la capa 2 ruidosa.

**`Confianza`**: `ALTA` · `MEDIA` · `BAJA` (enum, reemplaza el `str` de `models.py:48`).
`BAJA` degrada el slot a `AMBIGUO` en la fusión.

### 3.2 Las seis dimensiones clínicas

Enums **idénticos, carácter por carácter, a `dataset/trayectorias_postop_silver.xlsx`**.
Esto no es opcional: `trajectory_twin.ORDEN_*` indexa por estos literales.

| Variable | Tipo de `valor` | Valores permitidos | Columna dataset |
|---|---|---|---|
| `dolor` | `int` | `0`–`10` (dataset observa 0–9) | `dolor_nrs` |
| `fiebre` | `float` | `35.0`–`42.0` °C | `fiebre_c` |
| `movilidad` | `enum` | `normal` · `limitada_esperada` · `incapacitante_nueva` | `movilidad` |
| `herida` | `enum` | `normal` · `eritema_leve` · `secrecion_purulenta` | `herida` |
| `apetito` | `enum` | `normal` · `levemente_disminuido` · `muy_disminuido` | `apetito` |
| `sueno` | `enum` | `normal` · `levemente_alterado` · `muy_alterado` | `sueno` |

**Campos auxiliares por dimensión** (donde el valor solo no basta):

| Campo | Tipo | Por qué |
|---|---|---|
| `dolor.escala_declarada` | `bool` | ¿El paciente dio un número o el agente lo infirió de "me duele harto"? Cambia radicalmente la fiabilidad |
| `fiebre.medida` | `bool` | Termómetro real vs. "yo lo siento caliente". El umbral de 38 °C de `reflex_rules.py:18` solo aplica a medición real |
| `fiebre.sensacion_termica` | `enum \| None` | `sin_fiebre` · `escalofrios` · `sudoracion` · `siente_caliente` — la vía para el caso "no tengo termómetro" |

**Nota sobre `dolor` — escala vs. tendencia.** `dolor_nrs` absoluto tiene poco valor clínico
sin el día postoperatorio (es la premisa de `trajectory_twin`). Añadir:

| Campo | Tipo | Valores |
|---|---|---|
| `dolor.tendencia` | `enum \| None` | `mejorando` · `estable` · `empeorando` · `subito` |
| `dolor.localizacion_cambio` | `bool \| None` | ¿El dolor migró? (peritonitis post-apendicectomía) |

`subito` y `localizacion_cambio=True` son señales de alarma que un NRS de 5 nunca captura.

### 3.3 Bloque de banderas rojas (booleanos explícitos)

La vía refleja (`reflex_engine.py`) hace matching de keywords sobre texto crudo. Es rápida
(~5 ms) pero frágil: "no para de sangrar" dispara, "sigue botando sangre" no. El LLM debe
extraer **los mismos conceptos como booleanos**, para que la fusión tenga dos vías
independientes hacia la misma bandera.

| Campo | Tipo | Regla refleja equivalente |
|---|---|---|
| `sangrado_activo` | `TriEstado` | `REGLAS_COMUNES[0]` |
| `dificultad_respiratoria` | `TriEstado` | `REGLAS_COMUNES[1]` |
| `rigidez_abdominal` | `TriEstado` | `REGLAS_COMUNES[2]` |
| `secrecion_anormal` | `TriEstado` | `REGLAS_COMUNES[3]` |
| `dolor_extremo` | `TriEstado` | `REGLAS_COMUNES[4]` |
| `alteracion_conciencia` | `TriEstado` | `REGLAS_COMUNES[5]` |
| `banderas_procedimiento` | `list[str]` | Enum por procedimiento (`REGLAS_POR_PROCEDIMIENTO`) |

**`TriEstado`**: `PRESENTE` · `AUSENTE` · `NO_EVALUADO`. **Nunca `bool`.** Un `False` que
significa "no me consta" y un `False` que significa "el paciente lo negó explícitamente" no
pueden compartir representación en un sistema clínico.

Cada bandera lleva su `Observacion` completa (verbatim + procedencia), porque es lo que
alimenta el SBAR.

### 3.4 Adherencia y contexto (faltante hoy)

`memory.md` del proyecto lista "adherencia a medicación" como objetivo del agente, pero no
aparece en ningún esquema ni en el prompt.

| Campo | Tipo | Valores |
|---|---|---|
| `medicacion.toma_analgesico` | `TriEstado` | |
| `medicacion.adherencia` | `enum \| None` | `completa` · `parcial` · `abandonada` |
| `medicacion.motivo_no_adherencia` | `str \| None` | verbatim |
| `contexto.acompanado` | `TriEstado` | ¿Hay alguien con el paciente? Cambia el plan de escalamiento |
| `contexto.transporte_disponible` | `TriEstado` | Determina si "vaya a urgencias" es una recomendación viable |

`contexto.*` no afecta la criticidad pero sí la **recomendación** del SBAR
(`sbar.construir_sbar(accion_comunicada=...)`). Decirle "vaya a urgencias" a alguien solo,
sin transporte y a las 2 a.m. no es un plan.

### 3.5 Metadatos de calidad de la llamada

Necesarios para interpretar la capa 2 ruidosa y para el informe final.

| Campo | Tipo | Descripción |
|---|---|---|
| `estilo_paciente_detectado` | `enum \| None` | `colaborativo` · `evasivo` · `minimizador_sintomas` · `ansioso` · `confundido` — mismos valores que `dataset_final.estilo_paciente` |
| `intervencion_tercero` | `bool` | ¿Habló alguien más? |
| `turnos_sin_informacion` | `int` | Turnos consecutivos sin extraer nada nuevo → señal de abortar y escalar |
| `calidad_transcripcion` | `enum` | `buena` · `degradada` · `ininteligible` |

**`minimizador_sintomas` es la clase de riesgo.** Si se detecta, el umbral de escalamiento
debe bajar: un "estoy bien" de un minimizador no es evidencia positiva de ausencia de alarma.
Este campo es directamente evaluable contra el ground truth del dataset.

---

## 4. Estado vivo: `EstadoClinicoLlamada`

Vive en `SesionLlamada` (`server.py:51`), reemplazando a `sintomas_reportados: dict[str,str]`.

```
paciente_id, procedimiento, dia_postop, turno_actual
dolor, fiebre, movilidad, herida, apetito, sueno   ← Observacion[...]
banderas: BanderasRojas
medicacion, contexto, metadatos
criticidad_maxima: Criticidad
historial_criticidad: list[tuple[int, Criticidad, str]]   # (turno, nivel, motivo)
```

### 4.1 Propiedades derivadas (código, no LLM)

| Propiedad | Definición | Uso |
|---|---|---|
| `dimensiones_pendientes` | Slots con `estado == NO_PREGUNTADO` | Se inyecta al prompt → el agente sabe qué preguntar |
| `cobertura` | `n_confirmadas / 6` | Compuerta de cierre de llamada |
| `puede_cerrar_verde` | `cobertura == 1.0 ∧ ninguna bandera PRESENTE ∧ ninguna dimensión de procedencia TERCERO sin confirmar` | **Impide el verde por omisión** |
| `slots_bloqueados` | Slots con ≥2 intentos y sigue sin `CONFIRMADO` | Evita el loop infinito de reformular la misma pregunta |

`puede_cerrar_verde` es la traducción a código de la regla del prompt. Si es `False` y la
criticidad propuesta es verde, la fusión la degrada a `desconocida`.

### 4.2 Regla de fusión del delta

```
Para cada slot en ExtraccionTurno:
  1. Si estado == NO_PREGUNTADO  → ignorar (el delta no dice nada)
  2. Si confianza == BAJA        → degradar a AMBIGUO
  3. Si procedencia == TERCERO y el slot ya está CONFIRMADO por PACIENTE → descartar
  4. Si el nuevo valor es MÁS severo → sobrescribir siempre
  5. Si el nuevo valor es MENOS severo → conservar el máximo,
     registrar el nuevo en `historial_criticidad` como corrección
  6. Incrementar intentos[slot]
```

Regla 5 es deliberadamente conservadora: en la capa 2 los pacientes se retractan bajo la
sensación de estar molestando. La retractación se registra, no se obedece.

---

## 5. Resumen final: `ResumenLlamada` v2

Reemplaza `memory.py:19`. Es el snapshot congelado + lo que el agente necesita para abrir la
siguiente llamada (día 1 → 3 → 7 → 14).

| Campo | Cambio |
|---|---|
| `sintomas_reportados: dict[str,str]` | **Eliminar** → `estado_final: EstadoClinicoLlamada` |
| `resumen_texto: str` | **Reinterpretar** → deja de ser transcripción cruda; es el narrativo generado |
| `criticidad_final: str` | → `Criticidad` (enum, no `str`) |
| — | `+ cobertura: float` |
| — | `+ dimensiones_no_evaluadas: list[str]` — lo que quedó pendiente, para abrir con eso la próxima |
| — | `+ sbar: SBAR \| None` — solo si hubo escalamiento |
| — | `+ delta_vs_llamada_anterior: list[Desviacion]` — la señal longitudinal real |
| — | `+ motivo_cierre: enum` — `completada` · `paciente_colgo` · `escalada_inmediata` · `abortada_calidad` |
| — | `+ trazabilidad: list[Observacion]` — todos los verbatims con su turno |

`contexto_apertura()` (`memory.py:59`) debe reescribirse para usar `dimensiones_no_evaluadas`
y `delta_vs_llamada_anterior` en vez de concatenar los últimos 6 turnos. Esa es la diferencia
entre memoria longitudinal real y pegar transcripción.

---

## 6. Contrato de salida del LLM por turno

Se separa en **dos llamadas al modelo**, no una:

| Llamada | Salida | Modelo | Por qué separarlas |
|---|---|---|---|
| **A. Extracción** | `ExtraccionTurno` | rápido/barato | Tarea cerrada, sin creatividad. Un modelo pequeño la hace bien y en paralelo con el RAG |
| **B. Conversación** | `RespuestaEstructurada` | el principal | Necesita el contexto RAG y el estado ya actualizado por A |

Hoy las dos van juntas en un único JSON. Separarlas da tres cosas: (1) la extracción no
compite por atención con la generación de la frase hablada — el modo de fallo clásico del
prompt monolítico; (2) A puede correr en un modelo más barato y en paralelo con el retrieval,
sin costo de latencia; (3) B recibe el estado ya fusionado, así que la criticidad se decide
sobre datos estructurados, no sobre texto libre.

**`ExtraccionTurno` (salida de A) contiene únicamente el delta:**

```
turno_idx, hablante_detectado, dimensiones{...}, banderas{...},
medicacion?, contexto?, metadatos_turno{...}
```

Regla explícita en el prompt de A: **omitir toda dimensión que este turno no menciona.**
No rellenar. No repetir lo ya sabido. Un turno que solo dice "sí, ahí voy" produce un objeto
casi vacío, y eso es correcto.

### 6.1 Ajustes al prompt de extracción

Cuatro instrucciones que hoy no existen y que el esquema por sí solo no garantiza:

1. **Verbatim obligatorio.** Todo `valor` no nulo exige `verbatim` con las palabras del
   paciente. Sin verbatim → el validador rechaza el slot. Es el mismo patrón que ya usa
   `citation_validator.py` para el RAG, aplicado a la extracción.
2. **Prohibido inferir normalidad.** "El paciente no mencionó la herida" ⇒ `NO_PREGUNTADO`,
   nunca `normal`.
3. **Regionalismos → tabla de mapeo explícita en el prompt.** "Maluco", "flojito", "me arde
   comoquien dice", "ahí vamos", "más o menos" son ambiguos por diseño en este dataset. Sin
   una tabla, cada modelo los mapea distinto y la métrica se vuelve irreproducible. Los que
   no estén en la tabla → `AMBIGUO`.
4. **Atribución de hablante.** Si el turno viene de un tercero (`_c2_tercero` en el dataset),
   `procedencia = TERCERO` en todo lo extraído de ese turno.

---

## 7. Plan de implementación

Ordenado por relación impacto/esfuerzo.

| # | Cambio | Archivos | Desbloquea |
|---|---|---|---|
| 1 | Crear `clinical/extraction.py` con `Observacion`, `EstadoSlot`, `TriEstado`, `Procedencia`, `ExtraccionTurno` | nuevo | Todo lo demás |
| 2 | Prompt + llamada de extracción (llamada A) | `orchestrator/prompts.py`, `orchestrator/cortex.py` | — |
| 3 | `EstadoClinicoLlamada` + regla de fusión de deltas | `clinical/estado.py` (nuevo) | — |
| 4 | Reemplazar `sintomas_extraidos=None` por el estado real | `server.py:103`, `harness/runner.py:55` | **Activa `trajectory_twin`** |
| 5 | Inyectar `dimensiones_pendientes` en el prompt de conversación | `prompts.py:52` | Slot-filling real |
| 6 | Aplicar `puede_cerrar_verde` en la fusión | `clinical/fusion.py` | **Elimina el verde por omisión** |
| 7 | Validación estricta de enums (fallar ruidoso, no `.get(x, 0)`) | `trajectory_twin.py:135` | Cierra la falla insegura |
| 8 | `ResumenLlamada` v2 + reescribir `contexto_apertura()` | `clinical/memory.py` | Memoria longitudinal real |
| 9 | Métricas de extracción en el harness | `harness/report.py` | Ver §8 |

Los pasos 4, 6 y 7 son los que mueven la aguja en la rúbrica: hoy el proyecto tiene tres
mecanismos clínicos bien construidos (gemelo de trayectoria, memoria longitudinal, arco
reflejo) que no están conectados a datos.

---

## 8. Métricas para el harness

El esquema solo vale lo que se pueda medir. Con `dataset_final.xlsx` + `trayectorias_postop_silver.xlsx`
se puede evaluar la extracción **de forma independiente** de la clasificación:

| Métrica | Cómo | Contra qué |
|---|---|---|
| **Exactitud por dimensión** | Comparar `estado_final.<dim>.valor` al cierre vs. la fila de la trayectoria | `trayectorias_postop_silver` |
| **Cobertura** | % de casos que terminan con las 6 dimensiones `CONFIRMADO` | — |
| **Tasa de alucinación de slot** | % de slots `CONFIRMADO` cuyo `verbatim` no aparece en la transcripción | Transcripción |
| **Degradación capa 1 → capa 2** | Exactitud en `capa1_limpia` menos exactitud en `capa2_ruidosa`, mismo `caso_id` | `capa` |
| **Falso verde** | Casos con ground truth `rojo`/`amarillo` cerrados en verde | `label_ground_truth` |
| **Sesgo por estilo** | Exactitud desagregada por `estilo_paciente` | `estilo_paciente` |

La última es la más reveladora del reto: si la exactitud cae sobre `minimizador_sintomas` y
`evasivo` —y va a caer— eso es exactamente lo que el diseño de `EstadoSlot` + `Procedencia`
existe para mitigar, y se puede mostrar el antes/después en el informe final.

`turnos_sin_informacion` y `slots_bloqueados` sirven además como condición de parada del
harness: una llamada que no extrae nada en 3 turnos seguidos debe cerrarse y escalar, no
seguir consumiendo tokens.

---

## 9. Resumen de decisiones

| Decisión | Alternativa descartada | Razón |
|---|---|---|
| `Observacion[T]` en vez de valor plano | `dict[str, str]` | Sin procedencia ni verbatim no hay auditoría clínica ni defensa contra la capa 2 |
| `EstadoSlot` con 6 estados | `Optional[T]` | `None` no distingue "no pregunté" de "me evadió" — y esa distinción *es* la regla del verde |
| `TriEstado` en banderas | `bool` | Un `False` ambiguo en una bandera roja es un fallo de seguridad |
| Extracción separada de conversación | JSON monolítico | La extracción compite con la generación por atención; separadas se paralelizan y se abaratan |
| El LLM extrae deltas | El LLM devuelve el estado completo | Pedirle reproducir el estado invita a que lo altere; el acumulador debe ser determinista |
| Enums idénticos al dataset | Enums "más legibles" | `trajectory_twin.ORDEN_*` indexa por literal; cualquier divergencia degrada a normal |
| El estado solo escala en severidad | Última respuesta gana | Los pacientes se retractan por cortesía; se registra la retractación, no se obedece |
