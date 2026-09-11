"""Síntesis de voz, con dos backends y respaldo automático.

    edge   Microsoft Edge TTS, voz `es-CO-SalomeNeural` — colombiana neuronal, sin API key
    piper  Piper local (`es_MX-claude-high`) — funciona sin conexión

El default es `edge` por el acento: Piper solo publica voces `es_MX` y `es_ES` en español, y
a un paciente colombiano recién operado le habla alguien de otro país. Salomé es de Bogotá.

Ninguno de los dos pide credenciales, así que el cambio no rompe la promesa de arrancar sin
una sola clave de API. Lo que sí cambia es que `edge` depende de un servicio de Microsoft
que nadie ha contratado: puede cortarse sin aviso, y por eso **Piper sigue instalado y se
usa solo con que el otro falle**. Sin conexión, `TTS_BACKEND=piper` deja la solución
funcionando entera en local.

Sobre la versión mínima de `edge-tts`: Microsoft rota un token de autenticación
(`Sec-MS-GEC`) y bloquea a los clientes viejos. Con 7.0.0 el servicio responde 403 y la
síntesis falla; por eso `requirements.txt` exige >=7.2.8.

**El formato no es el mismo en los dos.** Piper produce WAV y Edge produce MP3, así que
estas funciones devuelven `(audio, mime)` y quien reproduce decide qué hacer con ello. No se
convierte entre formatos a propósito: el navegador reproduce los dos sin ayuda, y meter un
transcodificador significaría arrastrar ffmpeg por una diferencia que a nadie le importa.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from functools import lru_cache

from piper import PiperVoice

from agente_postop.config import get_settings

logger = logging.getLogger("agente_postop")

MIME_WAV = "audio/wav"
MIME_MP3 = "audio/mpeg"

EXTENSION = {MIME_WAV: ".wav", MIME_MP3: ".mp3"}


@lru_cache
def _voz_piper() -> PiperVoice:
    settings = get_settings()
    return PiperVoice.load(str(settings.piper_voice_model))


def _sintetizar_piper(texto: str) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        _voz_piper().synthesize_wav(texto, wav_file)
    return buffer.getvalue(), MIME_WAV


def _sintetizar_edge(texto: str) -> tuple[bytes, str]:
    """Sintetiza contra Edge TTS. Bloqueante a propósito: se llama desde un hilo.

    `edge_tts` es asíncrono, y esta función corre dentro de `asyncio.to_thread` desde el
    servidor, es decir, en un hilo sin bucle de eventos. `asyncio.run` crea uno propio ahí,
    que es exactamente lo que hace falta; llamarlo desde la corrutina principal en cambio
    reventaría por bucle anidado.
    """
    import edge_tts

    settings = get_settings()

    async def _generar() -> bytes:
        comunicacion = edge_tts.Communicate(texto, settings.edge_voz)
        trozos = bytearray()
        async for evento in comunicacion.stream():
            if evento["type"] == "audio":
                trozos.extend(evento["data"])
        return bytes(trozos)

    audio = asyncio.run(_generar())
    if not audio:
        raise RuntimeError("Edge TTS devolvió audio vacío")
    return audio, MIME_MP3


def sintetizar(texto: str) -> tuple[bytes, str]:
    """Devuelve `(audio, mime)` por el backend configurado, cayendo a Piper si falla.

    El respaldo no es cortesía: si el servicio de Microsoft deja de responder a mitad de una
    demostración, el agente tiene que seguir hablando. Un paciente que oye silencio cuelga.
    """
    settings = get_settings()
    if settings.tts_backend == "edge":
        try:
            return _sintetizar_edge(texto)
        except Exception as exc:  # noqa: BLE001 — sin voz no hay llamada: se cae a local
            logger.warning("Edge TTS falló (%s: %s) — sintetizando con Piper", type(exc).__name__, exc)
    return _sintetizar_piper(texto)


def sintetizar_wav(texto: str) -> bytes:
    """Compatibilidad: solo el audio, por el backend configurado.

    Se conserva porque el arranque y algún camino de error la siguen usando y no les importa
    el formato. El nombre miente cuando el backend es `edge` —ahí son bytes MP3—, pero
    renombrarla obligaría a tocar llamadas que no ganan nada con el cambio.
    """
    return sintetizar(texto)[0]
