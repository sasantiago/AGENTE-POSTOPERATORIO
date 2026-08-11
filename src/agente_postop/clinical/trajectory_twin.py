"""Gemelo digital de trayectoria: compara lo reportado contra lo esperado para ese día.

Un dolor de 4/10 no significa nada por sí solo — importa si el día postoperatorio es el 1
(normal) o el 14 (señal de alarma). La señal es la desviación, no el valor absoluto.

Fuente de la trayectoria esperada: `dataset/trayectorias_postop_silver.xlsx` (por paciente
conocido, día 1/3/7/14) con fallback al promedio del arquetipo `recuperacion_normal` por
procedimiento cuando el paciente no está en el dataset de referencia (caso real/demo).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from agente_postop.config import get_settings

ORDEN_MOVILIDAD = {"normal": 0, "limitada_esperada": 1, "incapacitante_nueva": 2}
ORDEN_HERIDA = {"normal": 0, "eritema_leve": 1, "secrecion_purulenta": 2}
ORDEN_APETITO = {"normal": 0, "levemente_disminuido": 1, "muy_disminuido": 2}
ORDEN_SUENO = {"normal": 0, "levemente_alterado": 1, "muy_alterado": 2}

DIAS_DE_REFERENCIA = (1, 3, 7, 14)


@dataclass
class CuadroEsperado:
    dolor_nrs: float
    fiebre_c: float
    movilidad: str
    herida: str
    apetito: str
    sueno: str
    fuente: str  # "paciente_conocido" | "promedio_procedimiento"


@dataclass
class Desviacion:
    dimension: str
    reportado: str
    esperado: str
    empeora: bool


@lru_cache
def _cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = get_settings()
    trayectorias = pd.read_excel(settings.dataset_dir / "trayectorias_postop_silver.xlsx")
    perfiles = pd.read_excel(
        settings.dataset_dir / "perfiles_clinicos_pacientes_silver_contest.xlsx"
    )[["paciente_id", "procedimiento"]]
    trayectorias = trayectorias.merge(perfiles, on="paciente_id", how="left")

    normal = trayectorias[trayectorias["arquetipo_trayectoria"] == "recuperacion_normal"]
    promedio_procedimiento = (
        normal.groupby(["procedimiento", "dia_postop"])
        .agg(
            dolor_nrs=("dolor_nrs", "mean"),
            fiebre_c=("fiebre_c", "mean"),
        )
        .reset_index()
    )
    return trayectorias, promedio_procedimiento


def _dia_referencia_mas_cercano(dia_postop: int) -> int:
    return min(DIAS_DE_REFERENCIA, key=lambda d: abs(d - dia_postop))


def trayectoria_esperada(paciente_id: str, procedimiento: str, dia_postop: int) -> CuadroEsperado | None:
    trayectorias, promedio_procedimiento = _cargar_datos()
    dia_ref = _dia_referencia_mas_cercano(dia_postop)

    fila_paciente = trayectorias[
        (trayectorias["paciente_id"] == paciente_id) & (trayectorias["dia_postop"] == dia_ref)
    ]
    if not fila_paciente.empty:
        f = fila_paciente.iloc[0]
        return CuadroEsperado(
            dolor_nrs=float(f["dolor_nrs"]),
            fiebre_c=float(f["fiebre_c"]),
            movilidad=f["movilidad"],
            herida=f["herida"],
            apetito=f["apetito"],
            sueno=f["sueno"],
            fuente="paciente_conocido",
        )

    fila_promedio = promedio_procedimiento[
        (promedio_procedimiento["procedimiento"] == procedimiento)
        & (promedio_procedimiento["dia_postop"] == dia_ref)
    ]
    if not fila_promedio.empty:
        f = fila_promedio.iloc[0]
        return CuadroEsperado(
            dolor_nrs=float(f["dolor_nrs"]),
            fiebre_c=float(f["fiebre_c"]),
            movilidad="normal",
            herida="normal",
            apetito="normal",
            sueno="normal",
            fuente="promedio_procedimiento",
        )

    return None


def comparar(reportado: dict, esperado: CuadroEsperado) -> list[Desviacion]:
    """Compara síntomas extraídos del turno contra el cuadro esperado. Solo reporta
    dimensiones presentes en `reportado` (no todo turno menciona los 6 síntomas)."""
    desviaciones: list[Desviacion] = []

    if "dolor_nrs" in reportado:
        empeora = reportado["dolor_nrs"] > esperado.dolor_nrs + 2
        desviaciones.append(
            Desviacion("dolor", str(reportado["dolor_nrs"]), str(esperado.dolor_nrs), empeora)
        )

    if "fiebre_c" in reportado:
        empeora = reportado["fiebre_c"] > max(esperado.fiebre_c, 37.5) + 0.5
        desviaciones.append(
            Desviacion("fiebre", str(reportado["fiebre_c"]), str(esperado.fiebre_c), empeora)
        )

    for dimension, orden in (
        ("movilidad", ORDEN_MOVILIDAD),
        ("herida", ORDEN_HERIDA),
        ("apetito", ORDEN_APETITO),
        ("sueno", ORDEN_SUENO),
    ):
        if dimension in reportado:
            # Fallar ruidoso, no silencioso: un literal que no está en la tabla no es
            # "normal" (antes: `.get(x, 0)` degradaba en silencio a 0=normal, que en un
            # agente clínico es la peor dirección de fallo posible — ver §1.2b del diseño).
            try:
                valor_reportado = orden[reportado[dimension]]
            except KeyError as exc:
                raise ValueError(
                    f"Valor no reconocido para '{dimension}': {reportado[dimension]!r} "
                    f"(válidos: {list(orden)})"
                ) from exc
            valor_esperado = orden[getattr(esperado, dimension)]
            desviaciones.append(
                Desviacion(
                    dimension,
                    reportado[dimension],
                    getattr(esperado, dimension),
                    empeora=valor_reportado > valor_esperado,
                )
            )

    return desviaciones
