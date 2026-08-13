"""CLI del harness: corre una muestra (o el total) de casos contra el orquestador real
y produce el reporte de §5 de la rúbrica (matriz de confusión, recall, latencias, tokens).

Uso:
    python -m harness.run_eval --todos
    python -m harness.run_eval --n-verde 20 --seed 7
    python -m harness.run_eval --casos caso_001,caso_002 --capas capa1_limpia
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import pandas as pd

from agente_postop.config import get_settings
from harness.report import (
    falsos_negativos_rojo,
    imprimir_reporte,
    latencias_p50_p95,
    matriz_confusion,
    recall_por_nivel,
)
from harness.reproducer import _cargar_dataset
from harness.runner import correr_casos

USOS_TOKENS: list[tuple[int, int]] = []
USOS_POR_PROVEEDOR: dict[str, list[tuple[int, int]]] = {"groq": [], "gemini": []}


def _instrumentar_tokens() -> None:
    """Envuelve las dos vías de salida del orquestador. Se parchea sobre el módulo `cortex`
    y no sobre `clients`, porque cortex importó las funciones por nombre y conserva su
    propia referencia — reemplazarlas en `clients` no lo tocaría.

    El desglose por proveedor no es cosmético: el cupo de Groq y el de Gemini son
    presupuestos separados, y el punto de mandar la extracción a Gemini es justamente que
    dejen de competir. Un total agregado escondería si el reparto está funcionando.
    """
    import agente_postop.orchestrator.cortex as cortex

    original_groq = cortex.crear_completado
    original_gemini = cortex.generar_json_gemini

    def groq_instrumentado(*args, **kwargs):
        resp = original_groq(*args, **kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            par = (usage.prompt_tokens, usage.completion_tokens)
            USOS_TOKENS.append(par)
            USOS_POR_PROVEEDOR["groq"].append(par)
        return resp

    def gemini_instrumentado(*args, **kwargs):
        # El SDK de Gemini devuelve texto plano, no un objeto con `usage`. Se estima por
        # longitud: sirve para ver el reparto entre presupuestos, no para facturar.
        texto = original_gemini(*args, **kwargs)
        entrada = len(kwargs.get("instruccion_sistema", "")) + len(kwargs.get("prompt_usuario", ""))
        par = (round(entrada / 3.7), round(len(texto) / 3.7))
        USOS_TOKENS.append(par)
        USOS_POR_PROVEEDOR["gemini"].append(par)
        return texto

    cortex.crear_completado = groq_instrumentado
    cortex.generar_json_gemini = gemini_instrumentado


def seleccionar_casos(n_rojo: int, n_amarillo: int, n_verde: int, seed: int) -> list[str]:
    df = _cargar_dataset()
    casos = df[["caso_id", "label_ground_truth"]].drop_duplicates()
    rng = random.Random(seed)

    seleccion: list[str] = []
    for nivel, n in (("rojo", n_rojo), ("amarillo", n_amarillo), ("verde", n_verde)):
        disponibles = sorted(casos[casos["label_ground_truth"] == nivel]["caso_id"].unique().tolist())
        rng.shuffle(disponibles)
        seleccion.extend(disponibles[: n if n >= 0 else len(disponibles)])
    return sorted(seleccion)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--casos", type=str, default=None, help="lista de caso_id separada por comas")
    parser.add_argument("--capas", type=str, default="capa1_limpia,capa2_ruidosa")
    parser.add_argument("--n-rojo", type=int, default=-1, help="-1 = todos")
    parser.add_argument("--n-amarillo", type=int, default=-1)
    parser.add_argument("--n-verde", type=int, default=20)
    parser.add_argument("--todos", action="store_true", help="corre los 160 casos completos")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--salida", type=str, default="harness_resultados.json")
    args = parser.parse_args()

    if args.todos:
        args.n_rojo = args.n_amarillo = args.n_verde = -1

    if args.casos:
        caso_ids = [c.strip() for c in args.casos.split(",") if c.strip()]
    else:
        caso_ids = seleccionar_casos(args.n_rojo, args.n_amarillo, args.n_verde, args.seed)

    capas = [c.strip() for c in args.capas.split(",") if c.strip()]

    print(f"Casos seleccionados: {len(caso_ids)} | capas: {capas}")
    _instrumentar_tokens()

    inicio_total = time.perf_counter()
    todos_resultados = []
    for capa in capas:
        resultados = correr_casos(caso_ids, capa)
        todos_resultados.extend(resultados)
        imprimir_reporte(resultados, etiqueta=capa)

    duracion_total = time.perf_counter() - inicio_total
    imprimir_reporte(todos_resultados, etiqueta="TOTAL")

    if USOS_TOKENS:
        prompt_tokens = [p for p, _ in USOS_TOKENS]
        completion_tokens = [c for _, c in USOS_TOKENS]
        print(f"\nTokens de entrada por turno — media: {statistics.mean(prompt_tokens):.0f}, "
              f"P50: {statistics.median(prompt_tokens):.0f}")
        print(f"Tokens de salida por turno — media: {statistics.mean(completion_tokens):.0f}, "
              f"P50: {statistics.median(completion_tokens):.0f}")
        print(f"Invocaciones al modelo (llamadas al LLM): {len(USOS_TOKENS)}")
        for proveedor, usos in USOS_POR_PROVEEDOR.items():
            if usos:
                print(
                    f"  · {proveedor:<7}: {len(usos):>4} llamadas | "
                    f"{sum(p for p, _ in usos):>8,} tok entrada | {sum(c for _, c in usos):>7,} tok salida"
                )

    print(f"\nDuración total del run: {duracion_total:.1f}s")

    matriz = matriz_confusion(todos_resultados)
    salida = {
        "n_casos": len(caso_ids),
        "capas": capas,
        "n_turnos": len(todos_resultados),
        "matriz_confusion": matriz,
        "recall_por_nivel": recall_por_nivel(matriz),
        "falsos_negativos_rojo": falsos_negativos_rojo(matriz),
        "latencia_p50_p95_s": latencias_p50_p95(todos_resultados),
        "tokens_entrada": USOS_TOKENS and [p for p, _ in USOS_TOKENS],
        "tokens_salida": USOS_TOKENS and [c for _, c in USOS_TOKENS],
        "n_invocaciones_llm": len(USOS_TOKENS),
        "tokens_por_proveedor": {
            proveedor: {
                "n_llamadas": len(usos),
                "tokens_entrada": sum(p for p, _ in usos),
                "tokens_salida": sum(c for _, c in usos),
            }
            for proveedor, usos in USOS_POR_PROVEEDOR.items()
        },
        "duracion_total_s": duracion_total,
        "resultados": [
            {
                "caso_id": r.caso_id,
                "capa": r.capa,
                "dialogo_id": r.dialogo_id,
                "turno_idx": r.turno_idx,
                "label_ground_truth": r.label_ground_truth,
                "criticidad_predicha": r.criticidad_predicha,
                "reflejo_vetea": r.reflejo_vetea,
                "latencia_s": r.latencia_s,
                "cobertura": r.cobertura,
            }
            for r in todos_resultados
        ],
    }
    Path(args.salida).write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados guardados en {args.salida}")


if __name__ == "__main__":
    main()
