"""Indexa dataset/textos/ completo en ChromaDB: archivo → Docling → chunks → BGE-M3.

Uso: python -m agente_postop.ingestion.build_index

Deduplica por hash de contenido (el dataset trae documentos repetidos bajo nombres
distintos) y es inmune a carpetas con espacios en el nombre — recorre recursivamente
`dataset/textos/*/` y usa el nombre de la carpeta como `procedimiento`.
"""

from __future__ import annotations

import sys
import time

# La consola de Windows suele quedar en cp1252 — nombres de archivo con caracteres como
# el guion no separable (U+2011) rompen un print() normal ahí. UTF-8 explícito evita que
# la indexación se caiga solo por no poder mostrar un nombre de archivo.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from agente_postop.config import get_settings
from agente_postop.ingestion.chunking import chunkear_markdown
from agente_postop.ingestion.docling_pipeline import convertir_a_markdown
from agente_postop.ingestion.hashing import hash_archivo
from agente_postop.rag.chroma_store import ChunkParaIndexar, hashes_indexados, indexar_chunks

EXTENSIONES_SOPORTADAS = {".pdf", ".docx", ".md", ".txt"}


def main() -> None:
    settings = get_settings()
    directorio_textos = settings.dataset_dir / "textos"

    if not directorio_textos.exists():
        print(f"No existe {directorio_textos}", file=sys.stderr)
        sys.exit(1)

    ya_indexados = hashes_indexados()
    print(f"Ya indexados de una corrida previa: {len(ya_indexados)} documentos", flush=True)

    hashes_vistos: dict[str, str] = {}  # hash -> primer nombre de archivo que lo tuvo
    n_documentos = 0
    n_chunks = 0
    n_duplicados = 0
    n_reanudados = 0
    inicio = time.time()

    for carpeta_procedimiento in sorted(directorio_textos.iterdir()):
        if not carpeta_procedimiento.is_dir():
            continue
        procedimiento = carpeta_procedimiento.name

        for archivo in sorted(carpeta_procedimiento.iterdir()):
            if archivo.suffix.lower() not in EXTENSIONES_SOPORTADAS:
                continue

            hash_contenido = hash_archivo(archivo)
            if hash_contenido in hashes_vistos:
                n_duplicados += 1
                print(f"  [dup] {archivo.name} == {hashes_vistos[hash_contenido]}", flush=True)
                continue
            hashes_vistos[hash_contenido] = archivo.name

            if hash_contenido in ya_indexados:
                n_reanudados += 1
                print(f"  [ya-indexado] {archivo.name}", flush=True)
                continue

            print(f"[{procedimiento}] {archivo.name}", flush=True)
            try:
                doc = convertir_a_markdown(archivo)
            except Exception as exc:  # noqa: BLE001 — un PDF corrupto no debe tumbar la ingesta completa
                print(f"  [error] {exc}", file=sys.stderr, flush=True)
                continue

            chunks = chunkear_markdown(doc.markdown)
            chunks_para_indexar = [
                ChunkParaIndexar(
                    chunk_id=f"{doc.hash_contenido[:12]}#p{chunk.indice}",
                    texto=chunk.texto,
                    documento=doc.nombre_documento,
                    hash_contenido=doc.hash_contenido,
                    procedimiento=procedimiento,
                    version=1,
                )
                for chunk in chunks
            ]
            indexar_chunks(chunks_para_indexar)

            n_documentos += 1
            n_chunks += len(chunks_para_indexar)
            print(f"  -> {len(chunks_para_indexar)} chunks ({doc.via})", flush=True)

    duracion = time.time() - inicio
    print()
    print(f"Documentos indexados en esta corrida: {n_documentos}")
    print(f"Ya indexados (reanudados, sin reprocesar): {n_reanudados}")
    print(f"Duplicados omitidos: {n_duplicados}")
    print(f"Chunks nuevos: {n_chunks}")
    print(f"Tiempo: {duracion:.1f}s")


if __name__ == "__main__":
    main()
