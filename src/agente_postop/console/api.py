"""Consola de administración — dos mitades.

Conocimiento: subir / listar / eliminar + inspector. Todo archivo subido entra por el mismo
camino que el vault: Docling → chunking → ChromaDB (ver ingestion/build_index.py para el
flujo batch; aquí se hace un solo documento).

Clínica: la bandeja de llamadas. El SBAR y el resumen ya se generaban y se persistían, pero
no existía ninguna pantalla donde el equipo médico los viera — quedaban en archivos JSON en
disco. Un traspaso clínico que nadie puede abrir no es un traspaso.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Response, UploadFile

from agente_postop.clinical.memory import cargar_historial
from agente_postop.config import get_settings
from agente_postop.ingestion.chunking import chunkear_markdown
from agente_postop.ingestion.docling_pipeline import convertir_a_markdown
from agente_postop.orchestrator.cortex import PROCEDIMIENTOS_OFRECIDOS, filtro_procedimiento
from agente_postop.rag.chroma_store import (
    ETIQUETA_CONSOLA,
    ChunkParaIndexar,
    consultar,
    eliminar_documento,
    indexar_chunks,
    listar_documentos,
)

router = APIRouter(prefix="/api/consola", tags=["consola"])


@router.get("/documentos")
def listar():
    return {"documentos": list(listar_documentos().values())}


@router.post("/documentos")
async def subir(archivo: UploadFile = File(...), procedimiento: str = ETIQUETA_CONSOLA):
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


@router.get("/procedimientos")
def listar_procedimientos():
    """La lista sale del orquestador, no de una copia en el HTML: si mañana se agrega un
    procedimiento, el inspector se entera solo."""
    return {"procedimientos": list(PROCEDIMIENTOS_OFRECIDOS)}


@router.get("/inspector")
def inspeccionar(consulta: str, n: int = 5, procedimiento: str | None = None):
    """¿Qué sabe el agente sobre X? — qué fragmentos responderían a esta consulta.

    Con `procedimiento`, aplica el MISMO filtro que usaría esa llamada, así que muestra lo
    que el agente vería de verdad. Sin él busca en todo el índice, que sirve para auditar el
    corpus pero no representa ninguna llamada: la fusión RRF siempre devuelve los N mejores
    por irrelevantes que sean, porque su puntaje mide concordancia de ranking y no parecido.
    Por eso conviene inspeccionar con procedimiento antes de una demo.
    """
    resultado = consultar(consulta, n_resultados=n, where=filtro_procedimiento(procedimiento))
    documentos = resultado.get("documents", [[]])[0]
    metadatas = resultado.get("metadatas", [[]])[0]
    ids = resultado.get("ids", [[]])[0]
    relevancias = resultado.get("relevancias", [[]])[0]

    return {
        "consulta": consulta,
        "procedimiento": procedimiento,
        "ambito": f"llamada de {procedimiento}" if procedimiento else "índice completo",
        "resultados": [
            {
                "chunk_id": chunk_id,
                "documento": meta["documento"],
                "procedimiento": meta["procedimiento"],
                "texto": texto,
                "relevancia": relevancia,
            }
            for chunk_id, texto, meta, relevancia in zip(ids, documentos, metadatas, relevancias)
        ],
    }


# --- bandeja clínica ---------------------------------------------------------------


def _directorio_memoria() -> Path:
    return get_settings().chroma_persist_dir.parent / "memoria"


def _pacientes_con_historial() -> list[str]:
    directorio = _directorio_memoria()
    if not directorio.exists():
        return []
    return sorted(p.stem for p in directorio.glob("*.json"))


@router.get("/llamadas")
def listar_llamadas():
    """Bandeja: todas las llamadas cerradas, la más reciente primero.

    Se ordenan por criticidad y no por fecha a igualdad de nada: quien abre esta pantalla
    necesita ver primero a quién hay que llamar, no qué pasó más recientemente.
    """
    orden_criticidad = {"rojo": 0, "desconocida": 1, "amarillo": 2, "verde": 3}
    llamadas = []
    for paciente_id in _pacientes_con_historial():
        for indice, resumen in enumerate(cargar_historial(paciente_id)):
            llamadas.append(
                {
                    "paciente_id": paciente_id,
                    "indice": indice,
                    "procedimiento": resumen.procedimiento,
                    "dia_postop": resumen.dia_postop,
                    "fecha": resumen.fecha,
                    "criticidad": resumen.criticidad_final.value,
                    "alerta": resumen.alerta_generada,
                    "cobertura": resumen.cobertura,
                    "tiene_sbar": resumen.sbar is not None,
                    "motivo_cierre": resumen.motivo_cierre.value,
                }
            )
    llamadas.sort(key=lambda ll: (orden_criticidad.get(ll["criticidad"], 9), ll["fecha"]), reverse=False)
    return {"llamadas": llamadas}


def _resumen_o_404(paciente_id: str, indice: int):
    historial = cargar_historial(paciente_id)
    if not 0 <= indice < len(historial):
        raise HTTPException(status_code=404, detail=f"No hay llamada {indice} para {paciente_id}")
    return historial[indice]


@router.get("/llamadas/{paciente_id}/{indice}")
def detalle_llamada(paciente_id: str, indice: int):
    resumen = _resumen_o_404(paciente_id, indice)
    return {
        "paciente_id": paciente_id,
        "indice": indice,
        "criticidad": resumen.criticidad_final.value,
        "cobertura": resumen.cobertura,
        "dimensiones_no_evaluadas": resumen.dimensiones_no_evaluadas,
        "delta_vs_llamada_anterior": resumen.delta_vs_llamada_anterior,
        "sbar": resumen.sbar.model_dump() if resumen.sbar else None,
        "resumen_markdown": resumen.resumen_texto,
    }


@router.get("/llamadas/{paciente_id}/{indice}/markdown")
def descargar_markdown(paciente_id: str, indice: int):
    """El resumen como archivo .md — para adjuntarlo a la historia clínica o mandarlo por
    correo sin tener que copiar y pegar desde una pantalla."""
    resumen = _resumen_o_404(paciente_id, indice)
    nombre = f"{paciente_id}_dia{resumen.dia_postop}.md".replace(" ", "_")
    return Response(
        content=resumen.resumen_texto,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{nombre}"'},
    )
