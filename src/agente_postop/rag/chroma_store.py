"""Colección ChromaDB con metadata de versionado por documento (hash, timestamp).

El versionado no es cosmético: sostiene G5 (subir → el agente aprende; eliminar → el
agente olvida, sin residuo) y el inspector "¿qué sabe el agente sobre X?" de la consola.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from agente_postop.config import get_settings
from agente_postop.rag.embeddings import embeber

NOMBRE_COLECCION = "conocimiento_clinico"


@lru_cache
def get_chroma_client() -> chromadb.ClientAPI:
    settings = get_settings()
    return chromadb.PersistentClient(
        path=str(settings.chroma_persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_coleccion():
    return get_chroma_client().get_or_create_collection(NOMBRE_COLECCION)


@dataclass
class ChunkParaIndexar:
    chunk_id: str
    texto: str
    documento: str
    hash_contenido: str
    procedimiento: str
    version: int


def indexar_chunks(chunks: list[ChunkParaIndexar]) -> None:
    if not chunks:
        return
    coleccion = get_coleccion()
    embeddings = embeber([c.texto for c in chunks])
    coleccion.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=[c.texto for c in chunks],
        metadatas=[
            {
                "documento": c.documento,
                "hash_contenido": c.hash_contenido,
                "procedimiento": c.procedimiento,
                "version": c.version,
            }
            for c in chunks
        ],
    )


def eliminar_documento(documento: str) -> int:
    """Elimina todos los chunks de un documento. Devuelve cuántos se borraron."""
    coleccion = get_coleccion()
    existentes = coleccion.get(where={"documento": documento})
    ids = existentes.get("ids", [])
    if ids:
        coleccion.delete(ids=ids)
    return len(ids)


def consultar(texto: str, n_resultados: int = 5, where: dict[str, Any] | None = None):
    coleccion = get_coleccion()
    embedding = embeber([texto])[0]
    return coleccion.query(query_embeddings=[embedding], n_results=n_resultados, where=where)


def chunk_id_existe(chunk_id: str) -> bool:
    coleccion = get_coleccion()
    resultado = coleccion.get(ids=[chunk_id])
    return len(resultado.get("ids", [])) > 0


def hashes_indexados() -> set[str]:
    """Todos los hash_contenido ya presentes en la colección — para reanudar una
    indexación interrumpida sin reprocesar documentos ya hechos."""
    coleccion = get_coleccion()
    datos = coleccion.get(include=["metadatas"])
    return {meta["hash_contenido"] for meta in datos.get("metadatas", [])}


def listar_documentos() -> dict[str, dict[str, Any]]:
    """Agrupa metadata por documento — para la consola (listar / inspector)."""
    coleccion = get_coleccion()
    datos = coleccion.get()
    por_documento: dict[str, dict[str, Any]] = {}
    for meta in datos.get("metadatas", []):
        doc = meta["documento"]
        entrada = por_documento.setdefault(
            doc,
            {"documento": doc, "procedimiento": meta["procedimiento"], "n_chunks": 0, "version": meta["version"]},
        )
        entrada["n_chunks"] += 1
    return por_documento
