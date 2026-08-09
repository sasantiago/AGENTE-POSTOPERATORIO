"""Todo archivo (consola o vault) entra por el mismo camino: Docling → Markdown.

Un solo camino de ingesta, local, con OCR incluido — sin importar si el PDF trae capa de
texto o es un escaneo (caso conocido: la carpeta `colorectal cancer` del dataset).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docling.document_converter import DocumentConverter

from agente_postop.ingestion.hashing import hash_archivo

_converter = DocumentConverter()


@dataclass
class DocumentoConvertido:
    ruta_origen: Path
    nombre_documento: str
    markdown: str
    hash_contenido: str


def convertir_a_markdown(ruta: Path) -> DocumentoConvertido:
    resultado = _converter.convert(str(ruta))
    markdown = resultado.document.export_to_markdown()
    return DocumentoConvertido(
        ruta_origen=ruta,
        nombre_documento=ruta.name,
        markdown=markdown,
        hash_contenido=hash_archivo(ruta),
    )
