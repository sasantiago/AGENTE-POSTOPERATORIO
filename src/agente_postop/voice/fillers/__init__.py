"""Fillers colombianos cacheados — máscara de latencia, capa 2.

Se reproducen al instante en cuanto se detecta fin de turno del paciente, mientras el LLM
todavía trabaja. Cero tokens, cero inferencia en el momento: son .wav pre-generados.
"""

from __future__ import annotations

import random
from pathlib import Path

DIR_FILLERS = Path(__file__).resolve().parent

FRASES_FILLER = [
    "ajá",
    "a ver...",
    "cuénteme",
    "mmm, ya",
    "listo",
    "le escucho",
    "claro que sí",
    "entiendo",
]

FRASES_PUENTE = [
    "deme un momentico que reviso eso",
    "espéreme un segundito, ya le cuento",
]


def ruta_filler(frase: str) -> Path:
    slug = frase.lower().replace(" ", "_").replace("...", "").replace(",", "").replace("¿", "").replace("?", "")
    return DIR_FILLERS / f"{slug}.wav"


def filler_aleatorio() -> Path:
    return ruta_filler(random.choice(FRASES_FILLER))


def puente_aleatorio() -> Path:
    return ruta_filler(random.choice(FRASES_PUENTE))
