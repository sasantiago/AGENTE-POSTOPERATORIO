"""Deduplicación de documentos por hash de contenido.

El dataset trae documentos duplicados (mismo PDF bajo dos nombres) y carpetas con
espacios en el nombre. La ingesta debe ser inmune a ambos.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def hash_archivo(ruta: Path) -> str:
    hasher = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            hasher.update(bloque)
    return hasher.hexdigest()
