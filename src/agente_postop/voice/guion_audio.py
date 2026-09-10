"""Audio pre-sintetizado del guion — el turno deja de pagar TTS.

Las preguntas del protocolo son fijas (`clinical/guion.py`), así que su audio se sintetiza
una vez y se guarda en disco. En la llamada, el agente **lee un archivo** en vez de invocar
a Piper: el TTS pasa de ~1 s por turno a un `read_bytes()`.

Va junto a los fillers y por la misma razón, pero cubre el otro extremo del turno: los
fillers tapan el silencio *mientras* el sistema piensa; esto elimina el silencio *después*.

    python -m agente_postop.voice.guion_audio        # genera lo que falte
    python -m agente_postop.voice.guion_audio --todo # regenera todo

Se regenera solo lo ausente porque sintetizar las veinte frases tarda, y en el 99% de los
arranques ya están todas. Si se cambia el texto de una pregunta hay que pasar `--todo`, o
borrar su `.wav`: el nombre del archivo deriva del texto, así que un cambio de redacción
genera un archivo nuevo y el viejo queda huérfano, nunca desactualizado.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from agente_postop.clinical.guion import (
    APERTURA,
    CIERRE_AMARILLO,
    CIERRE_ROJO,
    CIERRE_VERDE,
    GUION,
)

DIR_GUION = Path(__file__).resolve().parent / "guion"


def _ruta(texto: str) -> Path:
    """Nombre derivado del texto, no de la dimensión.

    Deliberado: si alguien reescribe una pregunta, el hash cambia y el sistema sintetiza la
    nueva en vez de reproducir alegremente la versión antigua. Un desajuste entre lo que
    dice el guion y lo que suena por el altavoz sería invisible en una revisión de código y
    evidente para el paciente.
    """
    firma = hashlib.sha1(texto.encode("utf-8")).hexdigest()[:12]
    return DIR_GUION / f"{firma}.wav"


def frases_del_guion() -> list[tuple[str, str]]:
    """Todo lo que el agente puede llegar a decir sin ayuda del modelo: (etiqueta, texto)."""
    frases: list[tuple[str, str]] = [("apertura", APERTURA)]
    for pregunta in GUION:
        frases.append((f"{pregunta.dimension}:pregunta", pregunta.texto))
        frases.append((f"{pregunta.dimension}:reintento", pregunta.reintento))
    frases.extend([
        ("cierre:verde", CIERRE_VERDE),
        ("cierre:amarillo", CIERRE_AMARILLO),
        ("cierre:rojo", CIERRE_ROJO),
    ])
    return frases


def audio_de(texto: str) -> bytes | None:
    """El audio pre-generado de una frase del guion, o None si no está en disco.

    Devuelve None en vez de sintetizar sobre la marcha a propósito: quien llama decide si
    caer a Piper en vivo. Sintetizar en silencio escondería que el pre-generado falta, y el
    coste —un segundo por turno— solo se notaría como una demo lenta el día de la
    evaluación.
    """
    ruta = _ruta(texto)
    return ruta.read_bytes() if ruta.exists() else None


def generar(regenerar_todo: bool = False) -> int:
    """Sintetiza las frases que falten. Devuelve cuántas generó."""
    from agente_postop.voice.tts import sintetizar_wav

    DIR_GUION.mkdir(parents=True, exist_ok=True)
    generadas = 0
    for etiqueta, texto in frases_del_guion():
        ruta = _ruta(texto)
        if ruta.exists() and not regenerar_todo:
            print(f"  ya está   {etiqueta}")
            continue
        ruta.write_bytes(sintetizar_wav(texto))
        generadas += 1
        print(f"  generada  {etiqueta:<22} {ruta.name}  ({ruta.stat().st_size} bytes)")
    return generadas


def faltantes() -> list[str]:
    """Etiquetas cuyo audio no está en disco. Lo usa el arranque para avisar."""
    return [etiqueta for etiqueta, texto in frases_del_guion() if not _ruta(texto).exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--todo", action="store_true", help="regenera también las que ya existen")
    args = parser.parse_args()
    print(f"Sintetizando el guion en {DIR_GUION}")
    generadas = generar(regenerar_todo=args.todo)
    print(f"\n{generadas} frase(s) generada(s); {len(frases_del_guion())} en total.")


if __name__ == "__main__":
    main()
