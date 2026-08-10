"""Deduplicación de documentos por hash de contenido.

El dataset trae documentos duplicados (mismo PDF bajo dos nombres) y carpetas con
espacios en el nombre. La ingesta debe ser inmune a ambos.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path


def ruta_larga(ruta: Path) -> str:
    """Prefijo `\\\\?\\` para rutas que exceden MAX_PATH (260 chars) en Windows — el
    dataset trae nombres de archivo largos que superan ese límite. No-op en otros SO."""
    absoluta = os.path.abspath(ruta)
    if sys.platform == "win32" and not absoluta.startswith("\\\\?\\"):
        return "\\\\?\\" + absoluta
    return absoluta


def hash_archivo(ruta: Path) -> str:
    hasher = hashlib.sha256()
    with open(ruta_larga(ruta), "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            hasher.update(bloque)
    return hasher.hexdigest()
