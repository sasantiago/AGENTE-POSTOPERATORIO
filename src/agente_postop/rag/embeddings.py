"""Embeddings BGE-M3 — multilingüe, fuerte en español médico."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODELO_BGE_M3 = "BAAI/bge-m3"


@lru_cache
def _modelo() -> SentenceTransformer:
    return SentenceTransformer(MODELO_BGE_M3)


def embeber(textos: list[str]) -> list[list[float]]:
    return _modelo().encode(textos, normalize_embeddings=True).tolist()


def embeber_uno(texto: str) -> list[float]:
    return embeber([texto])[0]
