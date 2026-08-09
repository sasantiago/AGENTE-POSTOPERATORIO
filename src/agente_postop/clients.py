"""Clientes de proveedores externos (Groq como LLM/STT primario)."""

from __future__ import annotations

from functools import lru_cache

from groq import Groq

from agente_postop.config import get_settings


@lru_cache
def get_groq_client() -> Groq:
    settings = get_settings()
    return Groq(api_key=settings.groq_api_key)
