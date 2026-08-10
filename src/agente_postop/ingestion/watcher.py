"""Watcher del vault — la carpeta que el sistema vigila e ingiere automáticamente
(equivalente al vault de Obsidian mencionado en el diseño). Un archivo nuevo se indexa
solo; un archivo borrado se olvida solo — mismo camino que la consola (Docling → chunks
→ ChromaDB), mismo contrato de G5.

Uso: python -m agente_postop.ingestion.watcher
"""

from __future__ import annotations

import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from agente_postop.config import get_settings
from agente_postop.ingestion.chunking import chunkear_markdown
from agente_postop.ingestion.docling_pipeline import convertir_a_markdown
from agente_postop.ingestion.hashing import hash_archivo
from agente_postop.rag.chroma_store import ChunkParaIndexar, eliminar_documento, indexar_chunks

EXTENSIONES_SOPORTADAS = {".pdf", ".docx", ".md", ".txt"}


def _indexar(ruta: Path) -> None:
    if ruta.suffix.lower() not in EXTENSIONES_SOPORTADAS:
        return
    print(f"[vault] nuevo/modificado: {ruta.name}")
    try:
        doc = convertir_a_markdown(ruta)
    except Exception as exc:  # noqa: BLE001
        print(f"[vault] error procesando {ruta.name}: {exc}")
        return

    chunks = chunkear_markdown(doc.markdown)
    chunks_para_indexar = [
        ChunkParaIndexar(
            chunk_id=f"{doc.hash_contenido[:12]}#p{chunk.indice}",
            texto=chunk.texto,
            documento=ruta.name,
            hash_contenido=doc.hash_contenido,
            procedimiento="vault",
            version=1,
        )
        for chunk in chunks
    ]
    indexar_chunks(chunks_para_indexar)
    print(f"[vault] {ruta.name} -> {len(chunks_para_indexar)} chunks indexados")


def _olvidar(ruta: Path) -> None:
    n = eliminar_documento(ruta.name)
    print(f"[vault] {ruta.name} eliminado -> {n} chunks olvidados")


class ManejadorVault(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            _indexar(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory:
            _indexar(Path(event.src_path))

    def on_deleted(self, event):
        if not event.is_directory:
            _olvidar(Path(event.src_path))


def main() -> None:
    settings = get_settings()
    settings.vault_dir.mkdir(parents=True, exist_ok=True)

    observador = Observer()
    observador.schedule(ManejadorVault(), str(settings.vault_dir), recursive=False)
    observador.start()
    print(f"Vigilando {settings.vault_dir} — Ctrl+C para detener")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observador.stop()
    observador.join()


if __name__ == "__main__":
    main()
