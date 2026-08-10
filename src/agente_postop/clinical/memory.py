"""Memoria longitudinal: cada llamada termina escribiendo su resumen estructurado;
la siguiente lo carga como contexto de apertura.

Persistencia simple en JSON por paciente — suficiente para 4 llamadas (días 1/3/7/14) y
transparente de inspeccionar durante el demo.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from agente_postop.config import get_settings


class ResumenLlamada(BaseModel):
    paciente_id: str
    dia_postop: int
    fecha: str
    sintomas_reportados: dict[str, str]
    criticidad_final: str
    alerta_generada: bool
    resumen_texto: str


def _ruta_memoria(paciente_id: str) -> Path:
    settings = get_settings()
    directorio = settings.chroma_persist_dir.parent / "memoria"
    directorio.mkdir(parents=True, exist_ok=True)
    return directorio / f"{paciente_id}.json"


def guardar_resumen(resumen: ResumenLlamada) -> None:
    ruta = _ruta_memoria(resumen.paciente_id)
    historial = cargar_historial(resumen.paciente_id)
    historial.append(resumen)
    ruta.write_text(
        json.dumps([r.model_dump() for r in historial], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def cargar_historial(paciente_id: str) -> list[ResumenLlamada]:
    ruta = _ruta_memoria(paciente_id)
    if not ruta.exists():
        return []
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    return [ResumenLlamada.model_validate(d) for d in datos]


def ultimo_resumen(paciente_id: str) -> ResumenLlamada | None:
    historial = cargar_historial(paciente_id)
    return historial[-1] if historial else None


def contexto_apertura(paciente_id: str) -> str | None:
    """Texto breve para que el agente abra la llamada recordando la anterior."""
    anterior = ultimo_resumen(paciente_id)
    if anterior is None:
        return None
    return (
        f"En la llamada del día {anterior.dia_postop} el paciente reportó: "
        f"{anterior.resumen_texto}"
    )


def timestamp_actual() -> str:
    return datetime.now().isoformat()
