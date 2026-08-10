"""Reporta matriz de confusión, recall por nivel (rojo primero), latencia P50/P95.

La métrica que importa no es accuracy general (con 77% verde, "todo verde" da ~77% y es
clínicamente inútil): es recall de rojo, luego recall de amarillo, con foco en la celda
rojo_real → verde_predicho (falla catastrófica, §4 del diseño).
"""

from __future__ import annotations

import statistics
from collections import Counter

from harness.runner import ResultadoEvaluacionTurno

NIVELES = ["verde", "amarillo", "rojo"]


def matriz_confusion(resultados: list[ResultadoEvaluacionTurno]) -> dict[str, dict[str, int]]:
    matriz = {real: {pred: 0 for pred in NIVELES + ["desconocida"]} for real in NIVELES}
    for r in resultados:
        real = r.label_ground_truth
        pred = r.criticidad_predicha
        if real in matriz and pred in matriz[real]:
            matriz[real][pred] += 1
    return matriz


def recall_por_nivel(matriz: dict[str, dict[str, int]]) -> dict[str, float]:
    recalls = {}
    for nivel in NIVELES:
        total = sum(matriz[nivel].values())
        correctos = matriz[nivel][nivel]
        recalls[nivel] = correctos / total if total else float("nan")
    return recalls


def falsos_negativos_rojo(matriz: dict[str, dict[str, int]]) -> int:
    """rojo_real -> verde_predicho — la falla catastrófica."""
    return matriz["rojo"]["verde"]


def latencias_p50_p95(resultados: list[ResultadoEvaluacionTurno]) -> tuple[float, float]:
    latencias = sorted(r.latencia_s for r in resultados)
    if not latencias:
        return float("nan"), float("nan")
    p50 = statistics.median(latencias)
    idx_p95 = int(len(latencias) * 0.95)
    p95 = latencias[min(idx_p95, len(latencias) - 1)]
    return p50, p95


def imprimir_reporte(resultados: list[ResultadoEvaluacionTurno], etiqueta: str = "") -> None:
    print(f"\n=== Reporte {etiqueta} ({len(resultados)} turnos evaluados) ===")

    matriz = matriz_confusion(resultados)
    print("\nMatriz de confusión (filas=real, columnas=predicho):")
    print(f"{'':12}" + "".join(f"{n:>12}" for n in NIVELES + ["desconocida"]))
    for real in NIVELES:
        print(f"{real:12}" + "".join(f"{matriz[real][pred]:>12}" for pred in NIVELES + ["desconocida"]))

    recalls = recall_por_nivel(matriz)
    print("\nRecall por nivel:")
    for nivel, valor in recalls.items():
        print(f"  {nivel}: {valor:.1%}" if valor == valor else f"  {nivel}: sin datos")

    fn_rojo = falsos_negativos_rojo(matriz)
    print(f"\nFalsos negativos en rojo (rojo real -> verde predicho): {fn_rojo}")

    p50, p95 = latencias_p50_p95(resultados)
    print(f"\nLatencia de orquestación (sin STT/TTS) — P50: {p50*1000:.0f}ms, P95: {p95*1000:.0f}ms")

    n_veto = sum(1 for r in resultados if r.reflejo_vetea)
    print(f"Turnos donde el reflejo vetó al cortex: {n_veto}")

    distribucion_predicha = Counter(r.criticidad_predicha for r in resultados)
    print(f"\nDistribución predicha: {dict(distribucion_predicha)}")
