"""Vía cortical: RAG + Llama vía Groq → RespuestaEstructurada.

Nunca texto libre: el LLM se fuerza a JSON (`response_format=json_object`) con el esquema
de `clinical.models.RespuestaEstructurada`. El validador de citas corre después, fuera de
esta función.

Las dos llamadas del turno van a proveedores distintos a propósito (ambos dentro de las
familias permitidas por `docs/stack-tecnico.md`): la conversación a Llama 3.3 70B en Groq,
donde el tamaño del modelo se paga en calidad de razonamiento clínico; la extracción a
Gemini Flash, que resuelve igual una tarea de esquema cerrado y consume un cupo distinto.
"""

from __future__ import annotations

import json
import logging
import re

from agente_postop.clients import AUSENTE, esquema_json_de, generar_json, normalizar_ausencias
from agente_postop.clinical.extraction import ExtraccionCruda
from agente_postop.clinical.models import RespuestaEstructurada
from agente_postop.clinical.reflex_rules import normalizar
from agente_postop.config import get_settings
from agente_postop.orchestrator.metrics import cronometrar
from agente_postop.orchestrator.prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_EXTRACCION,
    construir_prompt_extraccion,
    construir_prompt_usuario,
)
from agente_postop.rag.chroma_store import ETIQUETAS_CONOCIMIENTO_SUBIDO, consultar

logger = logging.getLogger("agente_postop")

N_CHUNKS_RAG = 4

# Techo de salida. Sin él, la generación queda sin límite: un JSON que se desmadra puede
# costar miles de tokens de un cupo diario que alcanza para poco más de una llamada. La
# respuesta hablada son 1-2 frases por diseño (SYSTEM_PROMPT) y la extracción es un delta
# de pocas dimensiones — estos techos no recortan nada legítimo, solo el peor caso.
MAX_TOKENS_CONVERSACION = 400
# 500 sí recortaba algo legítimo: el JSON de extracción llegaba cortado a mitad de cadena
# («Unterminated string starting at line 2 column 25»), la validación lo rechazaba y el
# turno caía a Groq — pagando DOS llamadas donde debía pagar una, y perdiendo el cupo
# separado que justifica usar Gemini. El delta trae verbatims del paciente, que son más
# largos de lo que sugería el conteo de dimensiones.
MAX_TOKENS_EXTRACCION = 900

# Extracción dirigida a un slot: la respuesta es {"valor": "..."} y nada más. El techo es
# holgado para ese contenido, y su valor real es acotar el peor caso de un modelo que se
# atasque repitiendo.
MAX_TOKENS_SLOT = 24

# Respuesta a una pregunta del paciente en el camino guiado: una frase hablada más el
# chunk_id que la respalda. Dos fragmentos en vez de cuatro porque para responder UNA
# pregunta concreta el tercero y el cuarto solo añaden tokens de prompt.
MAX_TOKENS_RESPUESTA = 160
# Tres fragmentos, no dos. Con dos, preguntas legítimas del paciente («¿es normal que la
# herida se vea rojita?») recuperaban material sobre qué es una colecistectomía y el modelo
# declaraba su límite con razón. El tercero cuesta ~200 tokens de prompt en un backend que
# responde en ~1,2 s, así que se paga sin que el paciente lo note.
#
# Que a veces siga sin encontrarlo NO es un fallo: el corpus del reto es literatura para
# profesionales, no material dirigido al paciente, y ante una pregunta que no cubre lo
# correcto es decir que no se sabe. Es la promesa del sistema, no su límite.
N_CHUNKS_RESPUESTA = 3

SIN_FRAGMENTOS = "(sin fragmentos relevantes encontrados en el conocimiento indexado)"
FALLBACK_SIN_RESPALDO = (
    "Eso no se lo puedo responder con seguridad; se lo voy a pasar a alguien del equipo médico."
)

# El procedimiento clínico (español — usado por reflex_rules.py, el dataset y el
# dropdown de la interfaz) no coincide con la etiqueta indexada en ChromaDB (nombre de
# carpeta de dataset/textos/, en inglés). Sin este mapeo, el filtro `where` del RAG no
# matchea nunca y toda consulta devuelve vacío — encontrado probando la llamada en vivo.
MAPEO_PROCEDIMIENTO_A_CORPUS: dict[str, str] = {
    "Apendicectomía": "Appendicitis",
    "Colecistectomía": "cholecystitis",
    "Colectomía": "colorectal cancer",
    "Reemplazo de cadera/rodilla": "total joint replacement",
    "Mastectomía": "breast_cancer",
}

# Lo que ofrece la interfaz de llamada. No es igual al mapeo de arriba: Rinoplastia no tiene
# corpus propio a propósito — es el procedimiento con el que se demuestra G5, porque el
# agente empieza sin saber nada de ella y solo aprende si se le sube una guía. Un
# procedimiento sin entrada en el mapeo no rompe nada: `filtro_procedimiento` usa su propio
# nombre como etiqueta, no encuentra chunks del corpus, y solo ve lo subido en caliente.
PROCEDIMIENTOS_OFRECIDOS: tuple[str, ...] = (*sorted(MAPEO_PROCEDIMIENTO_A_CORPUS), "Rinoplastia")


# El bloque de fragmentos era el más caro del prompt (~2.900 tokens/turno, la mitad del
# gasto). Pero el propio SYSTEM_PROMPT dice que en un turno de pura indagación la lista de
# afirmaciones clínicas va vacía — ahí los chunks se pagan para que el modelo tenga
# prohibido usarlos.
#
# El criterio es deliberadamente conservador: se recupera SIEMPRE salvo que el turno sea
# claramente un reporte de normalidad sin pregunta, sin señal de anomalía y sin bandera
# refleja. Medido sobre los 960 turnos de paciente del dataset, eso omite el 35% de las
# recuperaciones; el 65% restante conserva el respaldo documental intacto.
_PIDE_ORIENTACION = re.compile(
    r"[?¿]|es normal|sera normal|puedo |debo |que hago|tengo que|es grave|preocupa|angustia|"
    r"asusta|nervios|cuanto tiempo|deberia|hay que|me da miedo"
)
_SENAL_ANOMALIA = re.compile(
    r"secrecion|pus|sangr|fiebre|3[789][.,]?\d?\s*(grados|°)|4[01]\s*(grados|°)|roj|inflamad|"
    r"hinchad|no puedo|empeor|peor|mal olor|huele|duele mucho|insoportable|morad|frio|calient|"
    r"ardor|arde"
)

SIN_RESPALDO = (
    "(turno de indagación — no se recuperaron fragmentos. No hagas afirmaciones clínicas "
    "en este turno: acompaña y sigue preguntando.)"
)

# El chunk que se indexa no tiene por qué ser el chunk que se envía. Para recuperar conviene
# un chunk amplio (más contexto donde hacer match); para generar 1-2 frases habladas, no —
# ahí el excedente es ruido que se paga por token. El índice comiteado tiene una mediana de
# 2.631 chars por chunk, de los que el modelo usa una fracción.
#
# Se recorta al pasaje relevante en el momento de inyectar, conservando el chunk_id intacto:
# el validador de citas (clinical/citation_validator.py) verifica el id, no la longitud del
# texto, así que la trazabilidad de la afirmación no se toca.
#
# El puntaje es léxico y no vectorial a propósito: esto corre en cada turno de una llamada
# de voz en tiempo real, y embeber frases con e5 en CPU costaría latencia audible.
MAX_CHARS_POR_CHUNK = 700

_FIN_FRASE = re.compile(r"(?<=[.!?])\s+")
_TERMINO = re.compile(r"\w{4,}")


def _recortar_al_pasaje_relevante(texto: str, consulta: str, max_chars: int = MAX_CHARS_POR_CHUNK) -> str:
    """Devuelve la ventana CONTIGUA de frases más relacionada con la consulta. Contigua y no
    las N frases mejor puntuadas por separado: una afirmación clínica arrancada de su
    contexto ("...no requiere antibiótico") puede invertir su sentido."""
    if len(texto) <= max_chars:
        return texto

    frases = _FIN_FRASE.split(texto)
    terminos = set(_TERMINO.findall(normalizar(consulta)))
    if not terminos or not frases:
        return texto[:max_chars] + "…"

    puntajes = [len(terminos & set(_TERMINO.findall(normalizar(f)))) for f in frases]
    mejor = max(range(len(frases)), key=puntajes.__getitem__)

    if len(frases[mejor]) >= max_chars:
        return frases[mejor][:max_chars] + "…"

    inicio = fin = mejor
    total = len(frases[mejor])
    while True:
        siguiente = fin + 1 < len(frases) and total + len(frases[fin + 1]) + 1 <= max_chars
        anterior = inicio - 1 >= 0 and total + len(frases[inicio - 1]) + 1 <= max_chars
        if siguiente:
            fin += 1
            total += len(frases[fin]) + 1
        if anterior:
            inicio -= 1
            total += len(frases[inicio]) + 1
        if not siguiente and not anterior:
            break

    pasaje = " ".join(frases[inicio : fin + 1])
    return f"{'…' if inicio > 0 else ''}{pasaje}{'…' if fin < len(frases) - 1 else ''}"


def necesita_respaldo_clinico(turno_paciente: str, reflejo_disparado: bool = False) -> bool:
    """¿Este turno justifica pagar el bloque de fragmentos? Una bandera refleja siempre lo
    justifica — si el reflejo se disparó, lo que el agente diga tiene que ir respaldado."""
    if reflejo_disparado:
        return True
    normalizado = normalizar(turno_paciente)
    return bool(_PIDE_ORIENTACION.search(normalizado) or _SENAL_ANOMALIA.search(normalizado))


def filtro_procedimiento(procedimiento: str | None) -> dict | None:
    """El `where` de ChromaDB que corresponde a una llamada de este procedimiento.

    Vive aquí y se exporta porque el inspector de la consola tiene que usar EXACTAMENTE el
    mismo filtro. Si el inspector busca sin filtro (como hacía), responde "¿qué sabe el
    agente sobre X?" mirando los 1.968 chunks del corpus completo y siempre devuelve algo:
    tras borrar un documento seguía mostrando fragmentos de cadera y apendicitis para una
    consulta de rinoplastia — resultados que ninguna llamada real habría visto nunca.

    Incluye siempre el conocimiento subido en caliente (vault y consola), que no pertenece a
    ningún procedimiento del corpus original pero aplica a cualquier llamada.
    """
    etiqueta_corpus = MAPEO_PROCEDIMIENTO_A_CORPUS.get(procedimiento, procedimiento) if procedimiento else None
    if not etiqueta_corpus:
        return None
    return {"$or": [{"procedimiento": etiqueta} for etiqueta in (etiqueta_corpus, *ETIQUETAS_CONOCIMIENTO_SUBIDO)]}


def recuperar_contexto_rag(consulta: str, procedimiento: str | None = None, n_chunks: int = N_CHUNKS_RAG) -> str:
    where = filtro_procedimiento(procedimiento)
    with cronometrar("rag"):
        resultado = consultar(consulta, n_resultados=n_chunks, where=where)

    documentos = resultado.get("documents", [[]])[0]
    metadatas = resultado.get("metadatas", [[]])[0]
    ids = resultado.get("ids", [[]])[0]

    if not documentos:
        return SIN_FRAGMENTOS

    bloques = []
    for chunk_id, texto, meta in zip(ids, documentos, metadatas):
        pasaje = _recortar_al_pasaje_relevante(texto, consulta)
        bloques.append(f"[chunk_id={chunk_id} | documento={meta['documento']}]\n{pasaje}")
    return "\n\n".join(bloques)


def generar_respuesta(
    *,
    turno_paciente: str,
    procedimiento: str,
    contexto_memoria: str | None,
    desviaciones_trayectoria: str | None,
    historial_turno: str,
    dimensiones_pendientes: list[str] | None = None,
    reflejo_disparado: bool = False,
    ya_confirmado: str | None = None,
) -> RespuestaEstructurada:
    settings = get_settings()
    contexto_rag = (
        recuperar_contexto_rag(turno_paciente, procedimiento)
        if necesita_respaldo_clinico(turno_paciente, reflejo_disparado)
        else SIN_RESPALDO
    )

    prompt_usuario = construir_prompt_usuario(
        turno_paciente=turno_paciente,
        contexto_rag=contexto_rag,
        contexto_memoria=contexto_memoria,
        desviaciones_trayectoria=desviaciones_trayectoria,
        historial_turno=historial_turno,
        dimensiones_pendientes=dimensiones_pendientes,
        ya_confirmado=ya_confirmado,
    )

    with cronometrar("llm_conversacion"):
        contenido, backend = generar_json(
            instruccion_sistema=SYSTEM_PROMPT,
            prompt_usuario=prompt_usuario,
            max_tokens=MAX_TOKENS_CONVERSACION,
            temperature=0.3,
            esquema=esquema_json_de(RespuestaEstructurada),
            etiqueta="conversacion",
        )

    logger.debug("conversación resuelta por backend=%s", backend)
    datos = normalizar_ausencias(json.loads(contenido))
    return RespuestaEstructurada.model_validate(datos)


def responder_pregunta_anclada(*, pregunta_paciente: str, procedimiento: str) -> tuple[str, list]:
    """Responde a lo que preguntó el paciente, con una frase del corpus detrás.

    Versión ligera de `generar_respuesta`, para el camino guiado. La diferencia es la misma
    que en la extracción, y por el mismo motivo medido: el prompt de conversación completo
    —instrucciones largas, cuatro fragmentos y el esquema anidado de `RespuestaEstructurada`
    con su lista de afirmaciones— tardaba **52 s** en un turno real cuando el paciente
    preguntó «¿qué pasa?». Aquí se pide una cosa: una respuesta corta y el `chunk_id` que la
    respalda.

    Devuelve `(texto, afirmaciones)`. Sin fragmento que la sostenga, no se afirma nada: se
    devuelve el límite declarado, que es lo que el sistema promete y lo que impide
    tranquilizar sin respaldo.
    """
    contexto = recuperar_contexto_rag(pregunta_paciente, procedimiento, n_chunks=N_CHUNKS_RESPUESTA)
    if contexto == SIN_FRAGMENTOS:
        return FALLBACK_SIN_RESPALDO, []

    esquema = {
        "type": "object",
        "properties": {
            "respuesta": {"type": "string"},
            "chunk_id": {"type": "string"},
        },
        "required": ["respuesta", "chunk_id"],
    }
    instruccion = (
        "Eres un asistente de seguimiento postoperatorio hablando por TELÉFONO con un "
        "paciente colombiano.\n"
        # El corpus clínico está en inglés y arrastra al modelo con él: sin esta línea,
        # Gemini contestaba «it needs monitoring for spreading redness» a un paciente que
        # había preguntado en español. Hay que decirlo explícito y decirlo pronto.
        "RESPONDE SIEMPRE EN ESPAÑOL, aunque los documentos estén en inglés.\n"
        "Una sola frase corta y sencilla, sin términos médicos técnicos y sin formato: "
        "esto se va a leer en voz alta.\n"
        f'Si los fragmentos no responden la pregunta, responde exactamente: "{AUSENTE}".\n'
        "Nunca indiques medicamentos ni dosis. Nunca digas que algo es normal si el "
        "fragmento no lo dice.\n"
        "En `chunk_id` pon el identificador del fragmento que usaste.\n\n"
        f"FRAGMENTOS:\n{contexto}"
    )

    try:
      with cronometrar("llm_respuesta"):
        contenido, backend = generar_json(
            instruccion_sistema=instruccion,
            prompt_usuario=pregunta_paciente,
            max_tokens=MAX_TOKENS_RESPUESTA,
            temperature=0.2,
            esquema=esquema,
            etiqueta="respuesta_anclada",
            preferir=get_settings().llm_respuestas,
            # Sin respaldo local: ver la nota en `generar_json`. La alternativa a una
            # respuesta rápida no es una respuesta lenta, es el límite declarado.
            sin_cadena=get_settings().llm_respuestas != "ninguno",
        )

    except Exception as exc:  # noqa: BLE001 — la nube falló; se declara el límite, no se cuelga
        logger.warning("respuesta anclada no disponible (%s): %s", type(exc).__name__, exc)
        return FALLBACK_SIN_RESPALDO, []

    logger.debug("respuesta anclada resuelta por backend=%s", backend)
    try:
        datos = normalizar_ausencias(json.loads(contenido))
    except json.JSONDecodeError:
        return FALLBACK_SIN_RESPALDO, []

    texto = (datos or {}).get("respuesta")
    chunk_id = (datos or {}).get("chunk_id")
    if not texto or not chunk_id:
        return FALLBACK_SIN_RESPALDO, []

    # La cita se verifica contra ChromaDB igual que en el camino libre: un chunk_id que no
    # existe es una afirmación sin respaldo, venga por donde venga.
    from agente_postop.clinical.models import AfirmacionClinica
    from agente_postop.rag.chroma_store import chunk_id_existe

    if not chunk_id_existe(chunk_id):
        return FALLBACK_SIN_RESPALDO, []
    return texto, [AfirmacionClinica(texto=texto, chunk_id=chunk_id, documento=chunk_id)]


def extraer_slot(*, pregunta, respuesta_paciente: str) -> str | None:
    """Extrae UNA dimensión: la que el agente acaba de preguntar.

    Es la contrapartida del guion fijo. Si el agente hizo la pregunta, sabe cuál es, y
    entonces no tiene sentido pedirle al modelo que rellene las seis dimensiones más las
    banderas más la medicación más el contexto: se le pide un campo con tres valores
    posibles.

    Medido con `phi3.5:3.8b` sobre este equipo, mismo turno de paciente:

        esquema completo, prompt genérico   16,2 s   prompt 1094 tok, salida 62 tok
        dirigida a un slot                   5,8 s   prompt  115 tok, salida 16 tok

    2,8× más rápido, con un prompt 9,5× más chico. Y el modelo hace una tarea que sí sabe
    hacer: clasificar una frase corta en tres categorías descritas.

    Devuelve el valor del enum, o None si el paciente no respondió a lo que se le preguntó.
    None no es un fallo: es la respuesta correcta cuando el paciente cambia de tema, y es lo
    que hace que el motor de triaje escale por incertidumbre en vez de inventar normalidad.
    """
    if pregunta.valores is None:
        # Dolor y fiebre no pasan por el modelo: los lee `clinical/parsers.py`.
        return None

    esquema = {
        "type": "object",
        "properties": {"valor": {"type": "string", "enum": [*pregunta.valores, AUSENTE]}},
        "required": ["valor"],
    }
    # Prompt deliberadamente escueto. La versión larga —con la pregunta literal, el
    # recordatorio de no suponer y el ejemplo del pus— costaba 243 tokens de prompt y
    # 10,8 s; esta cuesta 115 y 8,1 s, con idéntico acierto sobre los mismos cinco casos.
    # El enum ya impide inventar valores y el criterio ya dice qué es cada uno: repetírselo
    # en prosa era pagar latencia por una instrucción que la gramática hace cumplir sola.
    #
    # No se le pasa el texto de la pregunta: el modelo no necesita saber cómo se formuló,
    # solo qué clasificar y con qué criterio.
    instruccion = (
        f"Clasifica {pregunta.dimension} según lo que dice el paciente.\n"
        f"{pregunta.criterio}\n"
        f'Si no habla de eso, responde "{AUSENTE}".'
    )

    with cronometrar("llm_extraccion"):
        contenido, backend = generar_json(
            instruccion_sistema=instruccion,
            prompt_usuario=respuesta_paciente,
            max_tokens=MAX_TOKENS_SLOT,
            temperature=0.0,
            esquema=esquema,
            etiqueta=f"slot:{pregunta.dimension}",
        )

    logger.debug("slot %s resuelto por backend=%s", pregunta.dimension, backend)
    try:
        valor = normalizar_ausencias(json.loads(contenido)).get("valor")
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.warning("respuesta ilegible extrayendo %s: %s", pregunta.dimension, exc)
        return None
    return valor if valor in pregunta.valores else None


def extraer_turno(*, turno_paciente: str, historial_turno: str) -> ExtraccionCruda:
    """Llamada A del diseño (§6): extracción pura, separada de la conversación.

    Es la tarea que más gana con el esquema forzado. Leer un turno y mapearlo a un
    vocabulario cerrado no requiere un modelo grande —lo que requiere es que la salida
    tenga exactamente la forma pedida—, y eso lo garantiza el decodificador, no el tamaño
    del modelo: con `format` puesto al JSON Schema, un 3B no puede omitir un campo ni
    devolver un valor fuera del enum.

    Perder la extracción es perder la cobertura, y sin cobertura completa el motor de
    triaje degrada la criticidad a `desconocida` en vez de darla por buena. Por eso el
    respaldo entre backends importa aquí tanto como en la conversación."""
    prompt_usuario = construir_prompt_extraccion(turno_paciente=turno_paciente, historial_turno=historial_turno)

    with cronometrar("llm_extraccion"):
        contenido, backend = generar_json(
            instruccion_sistema=SYSTEM_PROMPT_EXTRACCION,
            prompt_usuario=prompt_usuario,
            max_tokens=MAX_TOKENS_EXTRACCION,
            temperature=0.1,
            esquema=esquema_json_de(ExtraccionCruda),
            etiqueta="extraccion",
        )

    logger.debug("extracción resuelta por backend=%s", backend)
    return ExtraccionCruda.model_validate(normalizar_ausencias(json.loads(contenido)))
