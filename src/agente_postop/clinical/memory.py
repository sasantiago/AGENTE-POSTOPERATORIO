"""Memoria longitudinal: cada llamada termina escribiendo su resumen estructurado;
la siguiente lo carga como contexto de apertura.

Persistencia simple en JSON por paciente — suficiente para 4 llamadas (días 1/3/7/14) y
transparente de inspeccionar durante el demo.

v2 (§5 del diseño): `sintomas_reportados: dict[str,str]` se reemplaza por
`estado_final: EstadoClinicoLlamada` — deja de perderse la procedencia, el verbatim y el
estado epistémico de cada dimensión al cerrar la llamada.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from agente_postop.clinical.estado import EstadoClinicoLlamada
from agente_postop.clinical.models import SBAR, Criticidad
from agente_postop.config import get_settings


class MotivoCierre(StrEnum):
    COMPLETADA = "completada"
    PACIENTE_COLGO = "paciente_colgo"
    ESCALADA_INMEDIATA = "escalada_inmediata"
    ABORTADA_CALIDAD = "abortada_calidad"
    ERROR_TECNICO = "error_tecnico"


class ResumenLlamada(BaseModel):
    paciente_id: str
    dia_postop: int
    fecha: str
    estado_final: EstadoClinicoLlamada
    criticidad_final: Criticidad
    alerta_generada: bool
    resumen_texto: str
    cobertura: float = 0.0
    dimensiones_no_evaluadas: list[str] = Field(default_factory=list)
    sbar: SBAR | None = None
    delta_vs_llamada_anterior: list[dict] = Field(default_factory=list)
    motivo_cierre: MotivoCierre = MotivoCierre.COMPLETADA


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
        json.dumps([r.model_dump(mode="json") for r in historial], ensure_ascii=False, indent=2),
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
    """Texto breve para que el agente abra la llamada recordando la anterior — a partir
    del estado estructurado, no de transcripción cruda pegada (§5 del diseño)."""
    anterior = ultimo_resumen(paciente_id)
    if anterior is None:
        return None

    partes = [f"En la llamada del día {anterior.dia_postop}, criticidad final: {anterior.criticidad_final.value}."]

    confirmadas = []
    estado = anterior.estado_final
    for nombre in ("dolor", "fiebre", "movilidad", "herida", "apetito", "sueno"):
        obs = getattr(estado, nombre)
        if obs.confirmada:
            confirmadas.append(f"{nombre}={obs.valor}")
    if confirmadas:
        partes.append("Reportó: " + ", ".join(confirmadas) + ".")

    if anterior.dimensiones_no_evaluadas:
        partes.append("No se alcanzó a preguntar: " + ", ".join(anterior.dimensiones_no_evaluadas) + ".")

    return " ".join(partes)


def timestamp_actual() -> str:
    return datetime.now().isoformat()
