"""Clientes de proveedores externos (Groq para conversación y STT, Gemini para extracción).

Pool de claves con rotación automática: cuando una clave se queda sin cupo diario
(RateLimitError), se pasa a la siguiente sin que el llamador tenga que manejarlo. Solo se
propaga el error si TODAS las claves del pool están agotadas.

El pool es de dos ejes, no uno. Rotar claves reparte el mismo cupo; repartir las dos
llamadas del turno entre DOS proveedores lo duplica, porque el cupo de Gemini es un
presupuesto aparte del de Groq. La extracción (llamada A) es una tarea cerrada —leer un
turno y mapearlo a un esquema de vocabulario fijo— que no necesita el 70B, así que es la
que se muda; la conversación se queda en Llama, que es donde el tamaño sí paga.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Any

import google.generativeai as genai
from groq import Groq, RateLimitError

from agente_postop.config import get_settings

logger = logging.getLogger("agente_postop")

# Defaults del SDK: max_retries=2, timeout de lectura 60s — en una conversación de voz en
# tiempo real, un rate-limit o un pico de latencia puede terminar reintentando 2 veces por
# llamada, hasta 60s cada intento. Falla rápido: la rotación de claves ya cubre el caso de
# cupo agotado, no hace falta que el propio cliente insista.
TIMEOUT_S = 15.0
MAX_REINTENTOS = 1

_lock = threading.Lock()
_indice_actual = 0


@lru_cache
def _pool_clientes() -> list[Groq]:
    settings = get_settings()
    return [Groq(api_key=clave, timeout=TIMEOUT_S, max_retries=MAX_REINTENTOS) for clave in settings.groq_api_keys]


def get_groq_client() -> Groq:
    """Devuelve el cliente activo del pool. Preferí `crear_completado()` para llamadas al
    LLM — esta función queda para STT (Whisper), que no rota (mismo cupo, menor consumo)."""
    pool = _pool_clientes()
    with _lock:
        return pool[_indice_actual % len(pool)]


def crear_completado(**kwargs: Any):
    """`cliente.chat.completions.create(**kwargs)` con rotación automática de clave: si la
    clave activa devuelve RateLimitError (cupo diario agotado), pasa a la siguiente del
    pool y reintenta la MISMA llamada — hasta agotar todas las claves disponibles."""
    global _indice_actual
    pool = _pool_clientes()
    n = len(pool)
    ultimo_error: RateLimitError | None = None

    for intento in range(n):
        with _lock:
            indice = _indice_actual % n
            cliente = pool[indice]
        try:
            return cliente.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            ultimo_error = exc
            with _lock:
                if _indice_actual % n == indice:  # nadie más rotó mientras tanto
                    _indice_actual += 1
            if intento < n - 1:
                logger.warning("clave Groq #%d sin cupo — rotando a la siguiente (%d/%d)", indice, intento + 2, n)

    logger.error("las %d claves de Groq del pool están sin cupo diario", n)
    raise ultimo_error


# --- Gemini (llamada A: extracción) ------------------------------------------------

_lock_gemini = threading.Lock()
_indice_gemini = 0


class SinCupoGemini(RuntimeError):
    """Todas las claves de Gemini agotadas. El llamador debe caer a Groq, no morir."""


def generar_json_gemini(*, instruccion_sistema: str, prompt_usuario: str, max_tokens: int, temperature: float) -> str:
    """Genera un JSON con Gemini Flash rotando claves ante cupo agotado.

    `genai.configure()` es estado GLOBAL del SDK, así que la configuración y la generación
    tienen que ir bajo el mismo lock: sin él, dos turnos concurrentes pueden pisarse la
    clave entre el configure y el generate_content. Serializa las llamadas a Gemini, que
    con una por turno es un costo aceptable frente a mandar la petición con la clave de
    otra sesión.
    """
    global _indice_gemini
    settings = get_settings()
    claves = settings.gemini_api_keys
    if not claves:
        raise SinCupoGemini("no hay claves de Gemini configuradas")

    n = len(claves)
    ultimo_error: Exception | None = None

    for intento in range(n):
        with _lock_gemini:
            indice = _indice_gemini % n
            try:
                genai.configure(api_key=claves[indice])
                modelo = genai.GenerativeModel(
                    settings.gemini_llm_model,
                    system_instruction=instruccion_sistema,
                    generation_config={
                        "response_mime_type": "application/json",
                        "max_output_tokens": max_tokens,
                        "temperature": temperature,
                    },
                )
                # Con timeout explícito: el cliente de Groq ya fallaba rápido (TIMEOUT_S),
                # pero este no tenía ninguno, así que una llamada colgada de Gemini colgaba
                # el turno entero. Medido en vivo antes de este arreglo: un turno con
                # `llm_extraccion_ms=36445` frente a `llm_conversacion_ms=5415` — la
                # extracción, que es la tarea barata, costaba siete veces la cara.
                return modelo.generate_content(
                    prompt_usuario, request_options={"timeout": TIMEOUT_S}
                ).text
            except Exception as exc:  # noqa: BLE001 — cualquier fallo de Gemini debe poder caer a Groq
                ultimo_error = exc
                if _indice_gemini % n == indice:
                    _indice_gemini += 1
        if intento < n - 1:
            logger.warning("clave Gemini #%d falló (%s) — rotando (%d/%d)", indice, type(ultimo_error).__name__, intento + 2, n)

    raise SinCupoGemini(f"las {n} claves de Gemini fallaron; último error: {ultimo_error}") from ultimo_error
