"""Reproductor: reconstruye las llamadas de dataset_final.xlsx por caso_id.

`caso_id = "caso_" + trayectoria_id`. Cada caso trae turnos de paciente/agente/tercero,
en dos capas (capa1_limpia, capa2_ruidosa) — se reproducen por separado (el delta entre
ambas es la evidencia de robustez frente al ruido conversacional, §8 del diseño).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from agente_postop.config import get_settings


@dataclass
class TurnoCaso:
    dialogo_id: str
    turno_idx: int
    hablante: str
    texto: str
    label_ground_truth: str


@dataclass
class Caso:
    caso_id: str
    paciente_id: str
    dia_postop: int
    capa: str
    turnos: list[TurnoCaso]


@lru_cache
def _cargar_dataset() -> pd.DataFrame:
    settings = get_settings()
    return pd.read_excel(settings.dataset_dir / "dataset_final.xlsx")


@lru_cache
def _cargar_procedimientos() -> dict[str, str]:
    settings = get_settings()
    perfiles = pd.read_excel(settings.dataset_dir / "perfiles_clinicos_pacientes_silver_contest.xlsx")
    return dict(zip(perfiles["paciente_id"], perfiles["procedimiento"]))


def listar_casos(capa: str | None = None) -> list[str]:
    df = _cargar_dataset()
    if capa:
        df = df[df["capa"] == capa]
    return sorted(df["caso_id"].unique().tolist())


def reconstruir_caso(caso_id: str, capa: str) -> Caso:
    df = _cargar_dataset()
    filtrado = df[(df["caso_id"] == caso_id) & (df["capa"] == capa)].sort_values("turno_idx")

    if filtrado.empty:
        raise ValueError(f"Sin turnos para caso_id={caso_id} capa={capa}")

    primera = filtrado.iloc[0]
    turnos = [
        TurnoCaso(
            dialogo_id=str(fila["dialogo_id"]),
            turno_idx=int(fila["turno_idx"]),
            hablante=fila["hablante"],
            texto=fila["texto"],
            label_ground_truth=fila["label_ground_truth"],
        )
        for _, fila in filtrado.iterrows()
    ]

    return Caso(
        caso_id=caso_id,
        paciente_id=primera["paciente_id"],
        dia_postop=int(primera["dia_postop"]),
        capa=capa,
        turnos=turnos,
    )


def procedimiento_de(paciente_id: str) -> str:
    return _cargar_procedimientos().get(paciente_id, "desconocido")
