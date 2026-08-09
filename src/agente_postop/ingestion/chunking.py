"""Chunking de Markdown para indexar en ChromaDB.

Chunking simple por párrafos con ventana de tamaño objetivo y solapamiento — suficiente
para guías clínicas en prosa; evita partir una afirmación clínica a la mitad.
"""

from __future__ import annotations

from dataclasses import dataclass

TAMANO_OBJETIVO_CHARS = 1200
SOLAPAMIENTO_CHARS = 200


@dataclass
class Chunk:
    texto: str
    indice: int


def chunkear_markdown(markdown: str) -> list[Chunk]:
    parrafos = [p.strip() for p in markdown.split("\n\n") if p.strip()]

    chunks: list[Chunk] = []
    buffer = ""
    for parrafo in parrafos:
        if buffer and len(buffer) + len(parrafo) + 2 > TAMANO_OBJETIVO_CHARS:
            chunks.append(Chunk(texto=buffer, indice=len(chunks)))
            cola = buffer[-SOLAPAMIENTO_CHARS:] if len(buffer) > SOLAPAMIENTO_CHARS else buffer
            buffer = f"{cola}\n\n{parrafo}"
        else:
            buffer = f"{buffer}\n\n{parrafo}" if buffer else parrafo

    if buffer:
        chunks.append(Chunk(texto=buffer, indice=len(chunks)))

    return chunks
