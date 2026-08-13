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

# `label_ground_truth` es una etiqueta de CASO, no de turno: los 160 casos del dataset
# tienen el mismo valor en todos sus turnos. Un caso `rojo` lo es por el cuadro completo de
# la llamada, no porque cada turno lo sea — en `caso_tray_pac_42_00017_7`, por ejemplo, la
# etiqueta es `rojo` y ningún turno aislado lo es: es un paciente que minimiza, y la señal
# está en el conjunto (febrícula + herida enrojecida + inapetencia + insomnio).
#
# Por eso se reportan las dos vistas. La de turno mide la reactividad turno a turno y
# castiga al agente por no gritar ROJO cuando el paciente contesta que durmió mal; la de
# caso es la que corresponde a cómo está etiquetado el dataset, y es la que responde a la
# pregunta clínica real: al colgar, ¿esta llamada quedó escalada?
NIVEL_A_RANGO = {"verde": 0, "desconocida": 1, "amarillo": 2, "rojo": 3}
RANGO_A_NIVEL = {v: k for k, v in NIVEL_A_RANGO.items()}


def criticidad_maxima_por_caso(resultados: list[ResultadoEvaluacionTurno]) -> dict[str, tuple[str, str]]:
    """{caso::capa: (etiqueta_real, criticidad_máxima_alcanzada_en_la_llamada)}.

    El máximo y no el último turno: una llamada que detecta una bandera roja en el turno 3
    ya escaló: lo que pase después no la desescala — `sesion.criticidad_maxima` en el
    servidor acumula igual, y el SBAR queda emitido.
    """
    acumulado: dict[str, tuple[str, int]] = {}
    for r in resultados:
        clave = f"{r.caso_id}::{r.capa}"
        real, rango_previo = acumulado.get(clave, (r.label_ground_truth, -1))
        acumulado[clave] = (real, max(rango_previo, NIVEL_A_RANGO.get(r.criticidad_predicha, 0)))
    return {clave: (real, RANGO_A_NIVEL[rango]) for clave, (real, rango) in acumulado.items()}


def matriz_confusion_por_caso(resultados: list[ResultadoEvaluacionTurno]) -> dict[str, dict[str, int]]:
    matriz = {real: {pred: 0 for pred in NIVELES + ["desconocida"]} for real in NIVELES}
    for real, predicho in criticidad_maxima_por_caso(resultados).values():
        if real in matriz and predicho in matriz[real]:
            matriz[real][predicho] += 1
    return matriz


def casos_rojo_sin_escalar(resultados: list[ResultadoEvaluacionTurno]) -> int:
    """Casos `rojo` que colgaron sin haber escalado nunca por encima de `desconocida`.

    Es la falla catastrófica leída a nivel de llamada: el paciente colgó sin que nadie
    quedara avisado.
    """
    return sum(
        1
        for real, predicho in criticidad_maxima_por_caso(resultados).values()
        if real == "rojo" and NIVEL_A_RANGO[predicho] < NIVEL_A_RANGO["amarillo"]
    )


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


def cobertura_final_por_caso(resultados: list[ResultadoEvaluacionTurno]) -> dict[str, float]:
    """Cobertura de extracción (dimensiones confirmadas / 6) al último turno de cada
    caso+capa — mide si la llamada terminó sabiendo lo que necesitaba saber, no solo si
    clasificó bien (§9 del diseño de extracción)."""
    por_caso: dict[str, float] = {}
    for r in resultados:
        clave = f"{r.caso_id}::{r.capa}"
        por_caso[clave] = r.cobertura  # los turnos vienen en orden, el último gana
    return por_caso


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

    # Vista por caso — la que corresponde a cómo está etiquetado el dataset.
    matriz_caso = matriz_confusion_por_caso(resultados)
    n_casos = sum(sum(fila.values()) for fila in matriz_caso.values())
    print(f"\n--- Agregado por caso ({n_casos} llamadas): ¿a qué nivel llegó a escalar la llamada? ---")
    print(f"{'':12}" + "".join(f"{n:>12}" for n in NIVELES + ["desconocida"]))
    for real in NIVELES:
        print(f"{real:12}" + "".join(f"{matriz_caso[real][pred]:>12}" for pred in NIVELES + ["desconocida"]))

    recalls_caso = recall_por_nivel(matriz_caso)
    print("\nRecall por caso:")
    for nivel, valor in recalls_caso.items():
        print(f"  {nivel}: {valor:.1%}" if valor == valor else f"  {nivel}: sin datos")
    print(f"\nCasos rojo que colgaron sin escalar: {casos_rojo_sin_escalar(resultados)}")

    p50, p95 = latencias_p50_p95(resultados)
    print(f"\nLatencia de orquestación (sin STT/TTS) — P50: {p50*1000:.0f}ms, P95: {p95*1000:.0f}ms")

    n_veto = sum(1 for r in resultados if r.reflejo_vetea)
    print(f"Turnos donde el reflejo vetó al cortex: {n_veto}")

    coberturas = list(cobertura_final_por_caso(resultados).values())
    if coberturas:
        print(f"\nCobertura de extracción al cierre de la llamada — media: {statistics.mean(coberturas):.1%}, "
              f"casos con cobertura completa (6/6): {sum(1 for c in coberturas if c == 1.0)}/{len(coberturas)}")

    distribucion_predicha = Counter(r.criticidad_predicha for r in resultados)
    print(f"\nDistribución predicha: {dict(distribucion_predicha)}")
