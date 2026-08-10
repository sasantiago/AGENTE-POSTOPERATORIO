"""Genera y cachea los .wav de los fillers — se corre una sola vez (o cuando cambien las
frases). Uso: python -m agente_postop.voice.generate_fillers
"""

from __future__ import annotations

from agente_postop.voice.fillers import FRASES_FILLER, FRASES_PUENTE, ruta_filler
from agente_postop.voice.tts import sintetizar_wav


def main() -> None:
    for frase in FRASES_FILLER + FRASES_PUENTE:
        ruta = ruta_filler(frase)
        audio = sintetizar_wav(frase)
        ruta.write_bytes(audio)
        print(f"{ruta.name} <- \"{frase}\" ({len(audio)} bytes)")


if __name__ == "__main__":
    main()
