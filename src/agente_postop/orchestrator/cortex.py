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

from agente_postop.clients import crear_completado, generar_json_gemini
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


def recuperar_contexto_rag(consulta: str, procedimiento: str | None = None) -> str:
    where = filtro_procedimiento(procedimiento)
    with cronometrar("rag"):
        resultado = consultar(consulta, n_resultados=N_CHUNKS_RAG, where=where)

    documentos = resultado.get("documents", [[]])[0]
    metadatas = resultado.get("metadatas", [[]])[0]
    ids = resultado.get("ids", [[]])[0]

    if not documentos:
        return "(sin fragmentos relevantes encontrados en el conocimiento indexado)"

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
        completado = crear_completado(
            model=settings.groq_llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_usuario},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=MAX_TOKENS_CONVERSACION,
        )

    contenido = completado.choices[0].message.content
    datos = json.loads(contenido)
    return RespuestaEstructurada.model_validate(datos)


def extraer_turno(*, turno_paciente: str, historial_turno: str) -> ExtraccionCruda:
    """Llamada A del diseño (§6): extracción pura, separada de la conversación. Corre en
    Gemini Flash, no en Llama 70B: es una tarea cerrada —leer un turno y mapearlo a un
    esquema de vocabulario fijo— que no necesita el modelo grande, y sobre todo consume un
    cupo distinto al de la conversación, con lo que deja de competir por el de Groq.

    La caída a Groq no es cortesía: si Gemini falla, el turno tiene que seguir extrayendo
    estado clínico. Perder la extracción es perder la cobertura, y sin cobertura completa
    `puede_cerrar_verde` degrada la criticidad a desconocida."""
    settings = get_settings()
    prompt_usuario = construir_prompt_extraccion(turno_paciente=turno_paciente, historial_turno=historial_turno)

    if settings.extraccion_en_gemini:
        try:
            with cronometrar("llm_extraccion"):
                contenido = generar_json_gemini(
                    instruccion_sistema=SYSTEM_PROMPT_EXTRACCION,
                    prompt_usuario=prompt_usuario,
                    max_tokens=MAX_TOKENS_EXTRACCION,
                    temperature=0.1,
                )
            return ExtraccionCruda.model_validate(json.loads(contenido))
        except Exception as exc:  # noqa: BLE001 — cupo agotado, JSON inválido, red: todo cae a Groq
            logger.warning("extracción en Gemini falló (%s: %s) — cayendo a Groq", type(exc).__name__, exc)

    completado = crear_completado(
        model=settings.groq_llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_EXTRACCION},
            {"role": "user", "content": prompt_usuario},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=MAX_TOKENS_EXTRACCION,
    )

    contenido = completado.choices[0].message.content
    datos = json.loads(contenido)
    return ExtraccionCruda.model_validate(datos)
