"""Cita obligatoria o silencio.

El LLM nunca devuelve texto libre para afirmaciones clínicas: cada una viene amarrada a un
`chunk_id`. Este validador corre fuera del LLM — si una afirmación no tiene chunk_id o el
chunk_id no existe en ChromaDB, la respuesta no sale al TTS. En su lugar, un fallback
honesto. Esto hace la alucinación estructuralmente imposible de pronunciar, no solo
desincentivada por el prompt.
"""

from __future__ import annotations

from agente_postop.clinical.models import RespuestaEstructurada
from agente_postop.rag.chroma_store import chunk_id_existe

FALLBACK_HONESTO = (
    "Eso no se lo puedo responder con seguridad; se lo voy a pasar a alguien del equipo médico."
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

    return True, respuesta.respuesta_hablada
