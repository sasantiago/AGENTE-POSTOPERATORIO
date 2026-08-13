"""Chunking de Markdown para indexar en ChromaDB.

Chunking simple por párrafos con ventana de tamaño objetivo y solapamiento — suficiente
para guías clínicas en prosa; evita partir una afirmación clínica a la mitad.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TAMANO_OBJETIVO_CHARS = 1200
SOLAPAMIENTO_CHARS = 200

# Un párrafo puede, por sí solo, exceder el objetivo — pasa todo el tiempo con el Markdown
# que produce Docling desde PDFs clínicos: tablas volcadas a texto corrido, listas largas
# sin línea en blanco entre ítems. Acumular por párrafos sin subdividirlos dejaba salir ese
# párrafo entero como un chunk único: el 86% del índice quedaba por encima del objetivo,
# con mediana de 2.631 chars y máximos de 12.614 (~3.400 tokens en un solo chunk).
#
# El costo era doble: con 4 chunks por consulta el contexto RAG era ~2.900 tokens por turno
# (la mitad del gasto de la llamada), y la recuperación empeoraba — un chunk gigante que
# hace match por una frase arrastra todo el resto como ruido al prompt.
#
# Se recorta contra (objetivo - solapamiento) porque cada pieza puede recibir después la
# cola del chunk anterior; contra el objetivo pelado, el solapamiento volvería a pasarse.
TAMANO_MAXIMO_PIEZA = TAMANO_OBJETIVO_CHARS - SOLAPAMIENTO_CHARS

_FIN_DE_FRASE = re.compile(r"(?<=[.!?:;])\s+")


@dataclass
class Chunk:
    texto: str
    indice: int


def _subdividir(parrafo: str) -> list[str]:
    """Parte un párrafo sobredimensionado por límite de frase. Si una sola frase ya excede
    el máximo (una tabla volcada, una cadena sin puntuación), corta duro por longitud: un
    corte imperfecto es preferible a un chunk que se lleva el prompt entero."""
    if len(parrafo) <= TAMANO_MAXIMO_PIEZA:
        return [parrafo]

    piezas: list[str] = []
    actual = ""

    for frase in _FIN_DE_FRASE.split(parrafo):
        while len(frase) > TAMANO_MAXIMO_PIEZA:
            if actual:
                piezas.append(actual)
                actual = ""
            piezas.append(frase[:TAMANO_MAXIMO_PIEZA])
            frase = frase[TAMANO_MAXIMO_PIEZA:]

        if not frase:
            continue

        if actual and len(actual) + len(frase) + 1 > TAMANO_MAXIMO_PIEZA:
            piezas.append(actual)
            actual = frase
        else:
            actual = f"{actual} {frase}" if actual else frase

    if actual:
        piezas.append(actual)

    return piezas


def chunkear_markdown(markdown: str) -> list[Chunk]:
    parrafos = [p.strip() for p in markdown.split("\n\n") if p.strip()]

    chunks: list[Chunk] = []
    buffer = ""
    for parrafo in parrafos:
        for pieza in _subdividir(parrafo):
            if buffer and len(buffer) + len(pieza) + 2 > TAMANO_OBJETIVO_CHARS:
                chunks.append(Chunk(texto=buffer, indice=len(chunks)))
                cola = buffer[-SOLAPAMIENTO_CHARS:] if len(buffer) > SOLAPAMIENTO_CHARS else buffer
                buffer = f"{cola}\n\n{pieza}"
            else:
                buffer = f"{buffer}\n\n{pieza}" if buffer else pieza

    if buffer:
        chunks.append(Chunk(texto=buffer, indice=len(chunks)))

    return chunks
