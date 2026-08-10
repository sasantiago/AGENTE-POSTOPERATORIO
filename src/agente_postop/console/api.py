"""Consola de administración — contrato mínimo: subir / listar / eliminar + inspector.

Todo archivo subido entra por el mismo camino que el vault: Docling → chunking → ChromaDB
(ver ingestion/build_index.py para el flujo batch; aquí se hace un solo documento).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from agente_postop.ingestion.chunking import chunkear_markdown
from agente_postop.ingestion.docling_pipeline import convertir_a_markdown
from agente_postop.rag.chroma_store import ChunkParaIndexar, consultar, eliminar_documento, indexar_chunks, listar_documentos

router = APIRouter(prefix="/api/consola", tags=["consola"])


@router.get("/documentos")
def listar():
    return {"documentos": list(listar_documentos().values())}


@router.post("/documentos")
async def subir(archivo: UploadFile = File(...), procedimiento: str = "general"):
    sufijo = Path(archivo.filename or "documento").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=sufijo, delete=False) as tmp:
        tmp.write(await archivo.read())
        ruta_temporal = Path(tmp.name)

    try:
        doc = convertir_a_markdown(ruta_temporal)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"No se pudo procesar el documento: {exc}") from exc
    finally:
        ruta_temporal.unlink(missing_ok=True)

    chunks = chunkear_markdown(doc.markdown)
    chunks_para_indexar = [
        ChunkParaIndexar(
            chunk_id=f"{doc.hash_contenido[:12]}#p{chunk.indice}",
            texto=chunk.texto,
            documento=archivo.filename or doc.nombre_documento,
            hash_contenido=doc.hash_contenido,
            procedimiento=procedimiento,
            version=1,
        )
        for chunk in chunks
    ]
    indexar_chunks(chunks_para_indexar)

    return {
        "documento": archivo.filename,
        "n_chunks": len(chunks_para_indexar),
        "estado": "procesado · disponible",
    }


@router.delete("/documentos/{documento}")
def eliminar(documento: str):
    n_borrados = eliminar_documento(documento)
    if n_borrados == 0:
        raise HTTPException(status_code=404, detail="Documento no encontrado en el índice")
    return {"documento": documento, "chunks_eliminados": n_borrados}


@router.get("/inspector")
def inspeccionar(consulta: str, n: int = 5):
    """¿Qué sabe el agente sobre X? — muestra qué chunks responderían a una consulta dada."""
    resultado = consultar(consulta, n_resultados=n)
    documentos = resultado.get("documents", [[]])[0]
    metadatas = resultado.get("metadatas", [[]])[0]
    ids = resultado.get("ids", [[]])[0]
    distancias = resultado.get("distances", [[]])[0]

    return {
        "consulta": consulta,
        "resultados": [
            {
                "chunk_id": chunk_id,
                "documento": meta["documento"],
                "procedimiento": meta["procedimiento"],
                "texto": texto,
                "distancia": distancia,
            }
            for chunk_id, texto, meta, distancia in zip(ids, documentos, metadatas, distancias)
        ],
    }
