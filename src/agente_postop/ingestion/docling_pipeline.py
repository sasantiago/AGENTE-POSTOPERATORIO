"""Todo archivo (consola o vault) entra por el mismo camino conceptual: texto limpio con
OCR de respaldo cuando hace falta — sin importar si el PDF trae capa de texto o es un
escaneo (caso conocido: hay un PDF escaneado en el dataset, sin capa de texto).

El pipeline completo de Docling (layout + OCR con EasyOCR) es correcto pero, en CPU, es
demasiado lento para indexar un corpus de ~100 PDFs en un tiempo razonable — el modelo de
layout por sí solo puede tardar minutos por documento largo. La mayoría de los PDFs del
corpus son "born-digital" (tienen capa de texto real, no son escaneos), así que el camino
rápido es extraer esa capa de texto directamente con pypdfium2 (sin red neuronal, casi
instantáneo) y reservar el pipeline pesado de Docling solo para los documentos donde esa
extracción no arroja texto suficiente — la señal de que son un escaneo.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pypdfium2 as pdfium
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from agente_postop.ingestion.hashing import hash_archivo

UMBRAL_CHARS_POR_PAGINA = 200  # por debajo de esto, se asume escaneo y se usa Docling+OCR


@lru_cache
def _converter_docling() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = False  # TableFormer es caro y no lo necesitamos para RAG en prosa
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})


@dataclass
class DocumentoConvertido:
    ruta_origen: Path
    nombre_documento: str
    markdown: str
    hash_contenido: str
    via: str  # "texto_digital" | "docling_ocr"


def _extraer_texto_digital(ruta: Path) -> str | None:
    """Camino rápido: capa de texto ya embebida en el PDF. Devuelve None si el PDF
    parece un escaneo (muy poco texto por página) y hay que caer a Docling+OCR."""
    pdf = pdfium.PdfDocument(str(ruta))
    try:
        n_paginas = len(pdf)
        partes = []
        for pagina in pdf:
            textpage = pagina.get_textpage()
            partes.append(textpage.get_text_range())
            textpage.close()
            pagina.close()
        texto = "\n\n".join(partes)
    finally:
        pdf.close()

    if n_paginas == 0 or len(texto) / n_paginas < UMBRAL_CHARS_POR_PAGINA:
        return None
    return texto


def convertir_a_markdown(ruta: Path) -> DocumentoConvertido:
    hash_contenido = hash_archivo(ruta)

    if ruta.suffix.lower() == ".pdf":
        texto_digital = _extraer_texto_digital(ruta)
        if texto_digital is not None:
            return DocumentoConvertido(
                ruta_origen=ruta,
                nombre_documento=ruta.name,
                markdown=texto_digital,
                hash_contenido=hash_contenido,
                via="texto_digital",
            )

    resultado = _converter_docling().convert(str(ruta))
    markdown = resultado.document.export_to_markdown()
    return DocumentoConvertido(
        ruta_origen=ruta,
        nombre_documento=ruta.name,
        markdown=markdown,
        hash_contenido=hash_contenido,
        via="docling_ocr",
    )
