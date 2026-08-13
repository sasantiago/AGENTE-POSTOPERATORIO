"""Cita obligatoria o silencio.

El LLM nunca devuelve texto libre para afirmaciones clínicas: cada una viene amarrada a un
`chunk_id`. Este validador corre fuera del LLM — si una afirmación no tiene chunk_id o el
chunk_id no existe en ChromaDB, la respuesta no sale al TTS. En su lugar, un fallback
honesto. Esto hace la alucinación estructuralmente imposible de pronunciar, no solo
desincentivada por el prompt.
"""

from __future__ import annotations

import re

from agente_postop.clinical.models import RespuestaEstructurada
from agente_postop.clinical.reflex_rules import normalizar
from agente_postop.rag.chroma_store import chunk_id_existe

FALLBACK_HONESTO = (
    "Eso no se lo puedo responder con seguridad; se lo voy a pasar a alguien del equipo médico."
)

# Construcciones con las que el agente DICTAMINA qué es normal, esperable o seguro. Son
# afirmaciones clínicas aunque el modelo no las haya declarado como tales.
#
# Sin esta comprobación, la garantía de "cita o silencio" tenía un hueco por el que cabía
# justo el caso peor: el validador solo revisaba las citas presentes, así que un turno que
# afirmaba algo clínico y devolvía la lista de afirmaciones VACÍA pasaba sin verificar
# nada. Observado en vivo — con cuatro fragmentos válidos disponibles para citar, el modelo
# respondió "el enrojecimiento puede ser normal después de una cirugía" y no citó ninguno.
# Tranquilizar sin respaldo es exactamente la falla que el sistema promete impedir.
#
# El listado es deliberadamente corto: solo predicados asertivos sobre normalidad, gravedad
# o conducta. No basta con que aparezca la palabra "normal" — "me alegra que todo esté
# normal" repite al paciente, no dictamina.
#
# LÍMITE CONOCIDO, y hay que decirlo sin adornos: esto es una lista de bloqueo, y un modelo
# tiene infinitas formas de parafrasear. Observado en pruebas: bloqueada la forma "puede ser
# normal", el modelo respondió "puede presentar un eritema leve en los primeros días", que
# afirma exactamente lo mismo y no coincidía con ningún patrón. Se añadió, pero perseguir
# paráfrasis con expresiones regulares es una carrera que no se gana.
#
# Lo que esta comprobación sí garantiza es el caso más peligroso y más estereotipado: la
# tranquilización explícita sin respaldo ("no es grave", "no se preocupe", "va a mejorar").
# Verificar de verdad que lo hablado se apoya en los fragmentos es un problema de
# implicación textual y necesita una pasada de verificación aparte, no un regex.
_AFIRMACION_SIN_FIRMA = re.compile(
    r"\b("
    r"es normal|son normales|es algo normal|puede ser normal|pueden ser normales|"
    r"suele ser|suelen ser|suele presentar|es esperable|son esperables|es comun|son comunes|"
    r"es habitual|es frecuente|tiende a|puede presentar|pueden presentar|puede aparecer|"
    r"es parte del proceso|forma parte del proceso|es una reaccion normal|"
    r"no es grave|no es peligroso|no hay de que preocuparse|no se preocupe|"
    r"deberia ceder|deberia mejorar|va a mejorar|se le va a quitar|ira cediendo|"
    r"no requiere|no necesita"
    r")\b"
)


def validar_respuesta(respuesta: RespuestaEstructurada) -> tuple[bool, str]:
    """Devuelve (es_valida, texto_a_pronunciar).

    Si alguna afirmación clínica no resiste la verificación contra ChromaDB, la respuesta
    completa se descarta y se reemplaza por el fallback — no se pronuncian afirmaciones
    parcialmente validadas.
    """
    for afirmacion in respuesta.afirmaciones_clinicas:
        if not afirmacion.chunk_id or not chunk_id_existe(afirmacion.chunk_id):
            return False, FALLBACK_HONESTO

    if not respuesta.afirmaciones_clinicas and _AFIRMACION_SIN_FIRMA.search(
        normalizar(respuesta.respuesta_hablada)
    ):
        return False, FALLBACK_HONESTO

    return True, respuesta.respuesta_hablada
