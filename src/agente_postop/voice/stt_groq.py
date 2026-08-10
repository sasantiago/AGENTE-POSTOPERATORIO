"""STT vía Groq Whisper Large V3 — mismo proveedor que el LLM, menos saltos de red."""

from __future__ import annotations

import io

from agente_postop.clients import get_groq_client
from agente_postop.config import get_settings


def transcribir(audio_bytes: bytes, nombre_archivo: str = "audio.wav") -> str:
    settings = get_settings()
    cliente = get_groq_client()
    archivo = io.BytesIO(audio_bytes)
    archivo.name = nombre_archivo

    respuesta = cliente.audio.transcriptions.create(
        file=archivo,
        model=settings.groq_stt_model,
        language="es",
        response_format="text",
    )
    return str(respuesta).strip()
