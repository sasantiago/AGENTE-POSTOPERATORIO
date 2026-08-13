"""Registro de pacientes — de dónde sale el procedimiento de una llamada.

El procedimiento y el día postoperatorio NO son algo que alguien elija antes de llamar:
son datos que ya existen antes de que la llamada exista. Un agente de seguimiento marca
porque a esa persona la operaron de algo, un día concreto; esa es la razón de la llamada,
no un parámetro suyo.

La interfaz los pedía en dos desplegables, lo cual funciona para una demo pero invierte la
causalidad y deja pasar un error silencioso: si alguien elige mal el procedimiento, el RAG
filtra contra el corpus equivocado y nada avisa — el agente responde con seguridad sobre
la cirugía que no fue.

Aquí la fuente es `dataset/perfiles_clinicos_pacientes_silver_contest.xlsx`, que hace las
veces de la historia clínica: se elige el paciente y todo lo demás se deduce.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from functools import lru_cache

import pandas as pd
from pydantic import BaseModel, Field

from agente_postop.config import get_settings

ARCHIVO_PERFILES = "perfiles_clinicos_pacientes_silver_contest.xlsx"

# Hitos de seguimiento del diseño. El día real puede caer en cualquier punto, así que se
# reporta tal cual y se marca a qué hito corresponde; no se redondea al hito más cercano,
# que borraría la diferencia entre el día 3 y el día 6.
HITOS_SEGUIMIENTO = (1, 3, 7, 14)


class Paciente(BaseModel):
    paciente_id: str
    procedimiento: str
    fecha_cirugia: str
    edad: int | None = None
    genero: str | None = None
    comorbilidades: list[str] = Field(default_factory=list)

    @property
    def texto_comorbilidades(self) -> str | None:
        return ", ".join(self.comorbilidades) if self.comorbilidades else None


def _parsear_comorbilidades(valor: object) -> list[str]:
    """La columna llega como texto JSON ('["hipertension","obesidad"]'). Si viniera
    malformada se devuelve vacío en vez de reventar: un perfil raro no debe impedir llamar
    al paciente, y la ausencia de comorbilidades ya se reporta como tal en el SBAR."""
    if isinstance(valor, list):
        return [str(v) for v in valor]
    if not isinstance(valor, str) or not valor.strip():
        return []
    try:
        cargado = json.loads(valor)
    except json.JSONDecodeError:
        return []
    return [str(v) for v in cargado] if isinstance(cargado, list) else []


@lru_cache
def _registro() -> dict[str, Paciente]:
    settings = get_settings()
    ruta = settings.dataset_dir / ARCHIVO_PERFILES
    if not ruta.exists():
        return {}

    df = pd.read_excel(ruta)
    registro: dict[str, Paciente] = {}
    for _, fila in df.iterrows():
        fecha = fila["fecha_cirugia"]
        registro[str(fila["paciente_id"])] = Paciente(
            paciente_id=str(fila["paciente_id"]),
            procedimiento=str(fila["procedimiento"]),
            fecha_cirugia=str(fecha.date() if hasattr(fecha, "date") else fecha),
            edad=int(fila["edad"]) if pd.notna(fila.get("edad")) else None,
            genero=str(fila["genero"]) if pd.notna(fila.get("genero")) else None,
            comorbilidades=_parsear_comorbilidades(fila.get("comorbilidades")),
        )
    return registro


def listar_pacientes() -> list[Paciente]:
    return sorted(_registro().values(), key=lambda p: p.paciente_id)


def buscar_paciente(paciente_id: str) -> Paciente | None:
    return _registro().get(paciente_id)


def dias_desde_cirugia(fecha_cirugia: str, hoy: date | None = None) -> int | None:
    try:
        operado = datetime.fromisoformat(fecha_cirugia).date()
    except (ValueError, TypeError):
        return None
    return ((hoy or date.today()) - operado).days


def hito_mas_cercano(dia: int) -> int:
    return min(HITOS_SEGUIMIENTO, key=lambda h: abs(h - dia))
