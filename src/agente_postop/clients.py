"""Clientes de proveedores externos (Groq como LLM/STT primario)."""

from __future__ import annotations

from functools import lru_cache

from groq import Groq

from agente_postop.config import get_settings

# Defaults del SDK: max_retries=2, timeout de lectura 60s — en una conversación de voz en
# tiempo real, un rate-limit o un pico de latencia puede terminar reintentando 2 veces por
# llamada, hasta 60s cada intento. Con 2 llamadas por turno (extracción + conversación)
# eso es minutos de silencio para el paciente. Falla rápido: 1 reintento, 15s de timeout.
TIMEOUT_S = 15.0
MAX_REINTENTOS = 1


@lru_cache
def get_groq_client() -> Groq:
    settings = get_settings()
    return Groq(api_key=settings.groq_api_key, timeout=TIMEOUT_S, max_retries=MAX_REINTENTOS)
