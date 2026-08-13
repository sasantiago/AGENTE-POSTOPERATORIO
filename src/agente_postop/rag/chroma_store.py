"""Colección ChromaDB con metadata de versionado por documento (hash, timestamp).

El versionado no es cosmético: sostiene G5 (subir → el agente aprende; eliminar → el
agente olvida, sin residuo) y el inspector "¿qué sabe el agente sobre X?" de la consola.

La búsqueda es híbrida: vectorial (e5, capta significado) + BM25 (léxica, capta
coincidencias exactas — dosis, nombres de fármacos, códigos — que un embedding a veces
difumina). Se combinan con Reciprocal Rank Fusion, no con las dos vías por separado, para
no tener que normalizar puntajes de escalas distintas (coseno vs. BM25).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from agente_postop.config import get_settings
from agente_postop.rag.embeddings import embeber_consulta, embeber_pasajes

NOMBRE_COLECCION = "conocimiento_clinico"
RRF_K = 60

# Estas dos etiquetas de `procedimiento` no nombran un procedimiento clínico: nombran el
# canal por el que entró el documento (el vault vigilado y la subida desde la consola). El
# conocimiento que se sube en caliente no pertenece a un procedimiento del corpus original,
# así que toda consulta filtrada tiene que incluirlas además de su propia etiqueta.
#
# Sin eso, G5 queda partido por la mitad: el documento se indexa y aparece en el inspector
# de la consola, pero el filtro `where` de la llamada lo excluye siempre y el agente nunca
# llega a usarlo. Subir dejaba de significar aprender.
ETIQUETA_VAULT = "vault"
ETIQUETA_CONSOLA = "general"
ETIQUETAS_CONOCIMIENTO_SUBIDO = (ETIQUETA_VAULT, ETIQUETA_CONSOLA)


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
    embeddings = embeber_pasajes([c.texto for c in chunks])
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


def _tokenizar(texto: str) -> list[str]:
    return re.findall(r"\w+", texto.lower(), flags=re.UNICODE)


def _vector_top(texto: str, n: int, where: dict[str, Any] | None) -> list[str]:
    embedding = embeber_consulta(texto)
    coleccion = get_coleccion()
    resultado = coleccion.query(query_embeddings=[embedding], n_results=n, where=where)
    return resultado.get("ids", [[]])[0]


def _bm25_top(texto: str, n: int, where: dict[str, Any] | None) -> list[str]:
    coleccion = get_coleccion()
    datos = coleccion.get(where=where, include=["documents"])
    ids = datos.get("ids", [])
    documentos = datos.get("documents", [])
    if not ids:
        return []
    bm25 = BM25Okapi([_tokenizar(d) for d in documentos])
    puntajes = bm25.get_scores(_tokenizar(texto))
    orden = sorted(range(len(ids)), key=lambda i: puntajes[i], reverse=True)[:n]
    return [ids[i] for i in orden]


def consultar(texto: str, n_resultados: int = 5, where: dict[str, Any] | None = None):
    """Búsqueda híbrida: fusiona el ranking vectorial (e5) y el léxico (BM25) con
    Reciprocal Rank Fusion — cada uno vota según su posición, no según su puntaje crudo,
    así no hace falta normalizar escalas incompatibles (coseno vs. BM25)."""
    n_candidatos = max(n_resultados * 4, 20)
    ids_vector = _vector_top(texto, n_candidatos, where)
    ids_bm25 = _bm25_top(texto, n_candidatos, where)

    puntaje_fusionado: dict[str, float] = {}
    for ranking in (ids_vector, ids_bm25):
        for rank, chunk_id in enumerate(ranking):
            puntaje_fusionado[chunk_id] = puntaje_fusionado.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

    mejores_ids = sorted(puntaje_fusionado, key=puntaje_fusionado.get, reverse=True)[:n_resultados]
    if not mejores_ids:
        return {"documents": [[]], "metadatas": [[]], "ids": [[]], "relevancias": [[]]}

    coleccion = get_coleccion()
    detalle = coleccion.get(ids=mejores_ids, include=["documents", "metadatas"])
    por_id = dict(zip(detalle["ids"], zip(detalle["documents"], detalle["metadatas"])))

    documentos_ordenados = [por_id[i][0] for i in mejores_ids if i in por_id]
    metadatas_ordenadas = [por_id[i][1] for i in mejores_ids if i in por_id]
    relevancias_ordenadas = [puntaje_fusionado[i] for i in mejores_ids if i in por_id]

    return {
        "documents": [documentos_ordenados],
        "metadatas": [metadatas_ordenadas],
        "ids": [mejores_ids],
        "relevancias": [relevancias_ordenadas],
    }


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
