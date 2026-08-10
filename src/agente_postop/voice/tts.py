"""TTS vía Piper — local, gratis, latencia mínima (el audio empieza a sonar casi al
instante en que se genera el texto).

Requiere un modelo de voz en español descargado aparte (`.onnx` + `.onnx.json`) — ver
instrucciones de instalación en el README. La ruta se configura con `PIPER_VOICE_MODEL`.
"""

from __future__ import annotations

import io
import wave
from functools import lru_cache

from piper import PiperVoice

from agente_postop.config import get_settings


@lru_cache
def _voz() -> PiperVoice:
    settings = get_settings()
    return PiperVoice.load(str(settings.piper_voice_model))


def sintetizar_wav(texto: str) -> bytes:
    """Sintetiza texto a audio WAV en memoria (bytes), listo para enviar por WebSocket."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        _voz().synthesize_wav(texto, wav_file)
    return buffer.getvalue()
