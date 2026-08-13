"""Embeddings multilingual-e5-base — liviano (~1.1GB) para no comprometer la compuerta
de arranque en 15 minutos (BGE-M3 pesa 4.3GB y se descarga también en cada consulta en
vivo, no solo al indexar).

E5 exige prefijos distintos para lo que se indexa y lo que se busca — sin ellos la
calidad de recuperación cae notablemente (documentado en la ficha del modelo).
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODELO_EMBEDDINGS = "intfloat/multilingual-e5-base"


@lru_cache
def _modelo() -> SentenceTransformer:
    """Carga primero desde el caché en disco, y solo descarga si no está.

    Sin el intento local, sentence-transformers consulta huggingface.co para ver si hay una
    versión más nueva AUNQUE el modelo ya esté en disco. Con la red caída eso no falla
    rápido: son ~45 s de reintentos con backoff antes de rendirse y usar el caché. Medido.

    No hace falta que la red esté ausente para pagarlo — basta con que esté inestable, que
    es el caso normal del wifi de una sala de demos. Y 45 s en mitad de una llamada de voz
    no es una demora, es la llamada perdida.
    """
    try:
        return SentenceTransformer(MODELO_EMBEDDINGS, local_files_only=True)
    except Exception:  # noqa: BLE001 — primera ejecución: no está en caché, hay que bajarlo
        return SentenceTransformer(MODELO_EMBEDDINGS)


def embeber_pasajes(textos: list[str]) -> list[list[float]]:
    """Embeddings para texto que se indexa (chunks del corpus)."""
    con_prefijo = [f"passage: {t}" for t in textos]
    return _modelo().encode(con_prefijo, normalize_embeddings=True).tolist()


def embeber_consulta(texto: str) -> list[float]:
    """Embedding para una consulta de búsqueda — prefijo distinto, exigido por E5."""
    return _modelo().encode([f"query: {texto}"], normalize_embeddings=True).tolist()[0]
