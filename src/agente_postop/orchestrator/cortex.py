"""Vía cortical: RAG + Llama vía Groq → RespuestaEstructurada.

Nunca texto libre: el LLM se fuerza a JSON (`response_format=json_object`) con el esquema
de `clinical.models.RespuestaEstructurada`. El validador de citas corre después, fuera de
esta función.
"""

from __future__ import annotations

import json

from agente_postop.clients import get_groq_client
from agente_postop.clinical.models import RespuestaEstructurada
from agente_postop.config import get_settings
from agente_postop.orchestrator.prompts import SYSTEM_PROMPT, construir_prompt_usuario
from agente_postop.rag.chroma_store import consultar

N_CHUNKS_RAG = 4


def recuperar_contexto_rag(consulta: str, procedimiento: str | None = None) -> str:
    where = {"procedimiento": procedimiento} if procedimiento else None
    resultado = consultar(consulta, n_resultados=N_CHUNKS_RAG, where=where)

    documentos = resultado.get("documents", [[]])[0]
    metadatas = resultado.get("metadatas", [[]])[0]
    ids = resultado.get("ids", [[]])[0]

    if not documentos:
        return "(sin fragmentos relevantes encontrados en el conocimiento indexado)"

    bloques = []
    for chunk_id, texto, meta in zip(ids, documentos, metadatas):
        bloques.append(f"[chunk_id={chunk_id} | documento={meta['documento']}]\n{texto}")
    return "\n\n".join(bloques)


def generar_respuesta(
    *,
    turno_paciente: str,
    procedimiento: str,
    contexto_memoria: str | None,
    desviaciones_trayectoria: str | None,
    historial_turno: str,
) -> RespuestaEstructurada:
    settings = get_settings()
    contexto_rag = recuperar_contexto_rag(turno_paciente, procedimiento)

    prompt_usuario = construir_prompt_usuario(
        turno_paciente=turno_paciente,
        contexto_rag=contexto_rag,
        contexto_memoria=contexto_memoria,
        desviaciones_trayectoria=desviaciones_trayectoria,
        historial_turno=historial_turno,
    )

    cliente = get_groq_client()
    completado = cliente.chat.completions.create(
        model=settings.groq_llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_usuario},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    contenido = completado.choices[0].message.content
    datos = json.loads(contenido)
    return RespuestaEstructurada.model_validate(datos)
