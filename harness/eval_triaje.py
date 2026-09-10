"""Evaluación del motor de triaje contra los 160 casos etiquetados. Sin LLM, sin tokens.

    python -m harness.eval_triaje                      # perfil por defecto
    python -m harness.eval_triaje --perfil optimo
    python -m harness.eval_triaje --con-trayectoria    # activa la capa 6
    python -m harness.eval_triaje --calibrar           # rederiva los pesos del puntaje

**Devuelve código de salida distinto de cero si aparece un falso negativo de `rojo` o
cualquier subestimación.** No es un informe: es una prueba de regresión. Si alguien afloja
un umbral y con eso deja de escalar un caso grave, esto falla, y falla en CI.

Por qué esto existe y `run_eval.py` no basta: aquel mide el sistema completo —extracción
del LLM incluida— contra el dataset, y por eso cuesta tokens; con el cupo gratuito daba
para ~5 llamadas al día, así que la entrega anterior midió 14 casos de 160. Este mide el
**motor de decisión aislado del extractor**, alimentándolo con los slots que el propio
dataset trae como verdad. Son dos preguntas distintas y las dos hacen falta:

    eval_triaje.py  →  dado lo que el paciente tiene, ¿el sistema decide bien?   (160 casos)
    run_eval.py     →  dado lo que el paciente dice, ¿el sistema entiende bien?  (muestra)

Separarlas es lo que permite responder la primera con evidencia completa. Cuando el
extractor falla, lo que se degrada es la segunda, y eso se ve en su propia métrica en vez
de quedar mezclado en un solo número que no dice cuál de las dos partes falló.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from agente_postop.clinical.models import Criticidad
from agente_postop.clinical.trajectory_twin import Desviacion
from agente_postop.clinical.triage import (
    DOLOR_AMARILLO_NRS,
    ORDENES,
    PESO_APETITO,
    PESO_DOLOR,
    PESO_HERIDA,
    PESO_SUENO,
    PUNTAJE_AMARILLO,
    LecturaClinica,
    evaluar,
)
from agente_postop.config import get_settings

warnings.filterwarnings("ignore", message="Workbook contains no default style")

NIVELES = ("verde", "amarillo", "rojo")
DIMENSIONES_CATEGORICAS = ("movilidad", "herida", "apetito", "sueno")


@dataclass
class CasoEtiquetado:
    trayectoria_id: str
    paciente_id: str
    procedimiento: str
    dia_postop: int
    arquetipo: str
    lectura: LecturaClinica
    esperado_normal: dict  # cuadro de recuperación normal para ese procedimiento y día
    label: str


def cargar_casos() -> list[CasoEtiquetado]:
    """Cruza los slots de la trayectoria con la etiqueta de criticidad del diálogo.

    `caso_id` del dataset de diálogos es `"caso_" + trayectoria_id`, y la etiqueta es de
    **caso**, no de turno: los 160 casos tienen el mismo `label_ground_truth` en todos sus
    turnos. Medirla por turno fue lo que produjo el 8.3% de recall de la primera entrega.
    """
    settings = get_settings()
    tray = pd.read_excel(settings.dataset_dir / "trayectorias_postop_silver.xlsx")
    perfiles = pd.read_excel(settings.dataset_dir / "perfiles_clinicos_pacientes_silver_contest.xlsx")
    dialogos = pd.read_excel(settings.dataset_dir / "dataset_final.xlsx")

    tray = tray.merge(perfiles[["paciente_id", "procedimiento"]], on="paciente_id", how="left")

    etiquetas = dialogos.groupby("caso_id")["label_ground_truth"].first().reset_index()
    etiquetas["trayectoria_id"] = etiquetas["caso_id"].str.replace("^caso_", "", regex=True)
    df = tray.merge(etiquetas[["trayectoria_id", "label_ground_truth"]], on="trayectoria_id", how="inner")

    # Cuadro esperado de una recuperación normal, por procedimiento y día. Es contra esto
    # que se mide la desviación de trayectoria — no contra la fila del propio paciente,
    # que sería comparar el caso consigo mismo y daría cero desviaciones siempre.
    normal = df[df["arquetipo_trayectoria"] == "recuperacion_normal"]
    esperado = (
        normal.groupby(["procedimiento", "dia_postop"])
        .agg(dolor=("dolor_nrs", "mean"), fiebre=("fiebre_c", "mean"))
        .to_dict("index")
    )

    casos = []
    for fila in df.itertuples():
        casos.append(
            CasoEtiquetado(
                trayectoria_id=fila.trayectoria_id,
                paciente_id=fila.paciente_id,
                procedimiento=fila.procedimiento,
                dia_postop=int(fila.dia_postop),
                arquetipo=fila.arquetipo_trayectoria,
                lectura=LecturaClinica(
                    dolor=float(fila.dolor_nrs),
                    fiebre=float(fila.fiebre_c),
                    movilidad=fila.movilidad,
                    herida=fila.herida,
                    apetito=fila.apetito,
                    sueno=fila.sueno,
                ),
                esperado_normal=esperado.get((fila.procedimiento, int(fila.dia_postop)), {}),
                label=fila.label_ground_truth,
            )
        )
    return casos


def desviaciones_de(caso: CasoEtiquetado) -> list[Desviacion]:
    """Desviación del caso contra una recuperación normal del mismo procedimiento y día.

    Las categóricas se comparan contra `normal`, que es por definición el cuadro esperado
    de una recuperación sin complicaciones: cualquier grado por encima es una desviación.
    """
    if not caso.esperado_normal:
        return []
    d: list[Desviacion] = []
    esperado_dolor = caso.esperado_normal["dolor"]
    esperado_fiebre = caso.esperado_normal["fiebre"]
    d.append(
        Desviacion("dolor", str(caso.lectura.dolor), f"{esperado_dolor:.1f}",
                   empeora=(caso.lectura.dolor or 0) > esperado_dolor + 2)
    )
    d.append(
        Desviacion("fiebre", str(caso.lectura.fiebre), f"{esperado_fiebre:.1f}",
                   empeora=(caso.lectura.fiebre or 0) > max(esperado_fiebre, 37.5) + 0.5)
    )
    for dim in DIMENSIONES_CATEGORICAS:
        valor = getattr(caso.lectura, dim)
        if valor is not None:
            d.append(Desviacion(dim, valor, "normal", empeora=ORDENES[dim][valor] > 0))
    return d


def correr(casos: list[CasoEtiquetado], perfil: str, con_trayectoria: bool) -> dict:
    matriz = {(real, pred): 0 for real in NIVELES for pred in NIVELES}
    fallos: list[dict] = []
    reglas: dict[str, int] = {}

    for caso in casos:
        decision = evaluar(
            caso.lectura,
            perfil=perfil,
            desviaciones=desviaciones_de(caso) if con_trayectoria else None,
            # La cobertura no aplica: los slots vienen completos del dataset. Exigirla
            # convertiría todo verde en `desconocida` y mediría otra cosa.
            exigir_cobertura_para_verde=False,
        )
        predicho = decision.nivel.value
        matriz[(caso.label, predicho)] = matriz.get((caso.label, predicho), 0) + 1
        if decision.escalado_por:
            reglas[decision.escalado_por] = reglas.get(decision.escalado_por, 0) + 1
        if predicho != caso.label:
            fallos.append(
                {
                    "caso": caso.trayectoria_id,
                    "dia_postop": caso.dia_postop,
                    "arquetipo": caso.arquetipo,
                    "real": caso.label,
                    "predicho": predicho,
                    "subestimacion": Criticidad(predicho).rango < Criticidad(caso.label).rango,
                    "motivos": [m.detalle for m in decision.motivos],
                }
            )

    aciertos = sum(v for (real, pred), v in matriz.items() if real == pred)
    rojos = sum(v for (real, _), v in matriz.items() if real == "rojo")
    rojos_detectados = matriz[("rojo", "rojo")]
    subestimaciones = sum(1 for f in fallos if f["subestimacion"])

    return {
        "perfil": perfil,
        "capa_trayectoria": con_trayectoria,
        "total": len(casos),
        "aciertos": aciertos,
        "exactitud": round(aciertos / len(casos), 4),
        "recall_rojo": round(rojos_detectados / rojos, 4) if rojos else 0.0,
        "falsos_negativos_rojo": rojos - rojos_detectados,
        "subestimaciones": subestimaciones,
        "falsos_positivos_sobre_verde": sum(v for (real, pred), v in matriz.items() if real == "verde" and pred != "verde"),
        "matriz": {f"{real}->{pred}": v for (real, pred), v in matriz.items()},
        "reglas_que_escalaron": dict(sorted(reglas.items(), key=lambda kv: -kv[1])),
        "fallos": fallos,
    }


def imprimir(res: dict) -> None:
    print(f"\nMotor de triaje — perfil `{res['perfil']}`"
          f"{' + capa de trayectoria' if res['capa_trayectoria'] else ''}")
    print(f"{res['total']} casos etiquetados · slots del dataset (motor aislado del extractor)\n")

    ancho = max(len(n) for n in NIVELES) + 2
    sangria = 7 + ancho  # "  real " + el nombre del nivel, para que las columnas cuadren
    print(" " * (sangria + 14) + "predicho")
    print(" " * sangria + "".join(f"{n:>12}" for n in NIVELES))
    for real in NIVELES:
        fila = "".join(f"{res['matriz'][f'{real}->{pred}']:>12}" for pred in NIVELES)
        print(f"  real {real:<{ancho}}{fila}")

    print(f"\n  exactitud                    {res['aciertos']}/{res['total']} ({res['exactitud']:.1%})")
    print(f"  recall de rojo               {res['recall_rojo']:.1%}")
    print(f"  falsos negativos de rojo     {res['falsos_negativos_rojo']}")
    print(f"  subestimaciones (cualquiera) {res['subestimaciones']}")
    print(f"  falsos positivos sobre verde {res['falsos_positivos_sobre_verde']}")

    if res["reglas_que_escalaron"]:
        print("\n  regla que determinó el nivel:")
        for regla, n in res["reglas_que_escalaron"].items():
            print(f"    {regla:<28} {n}")

    sobrecalificados = [f for f in res["fallos"] if not f["subestimacion"]]
    if sobrecalificados:
        print(f"\n  {len(sobrecalificados)} caso(s) sobrecalificados (dirección segura):")
        for f in sobrecalificados[:10]:
            print(f"    {f['caso']:<24} {f['real']} → {f['predicho']}  [{f['arquetipo']}]")
            print(f"      {f['motivos'][0] if f['motivos'] else ''}")

    for f in res["fallos"]:
        if f["subestimacion"]:
            print(f"\n  !! SUBESTIMACIÓN: {f['caso']} real={f['real']} predicho={f['predicho']}")


def calibrar(casos: list[CasoEtiquetado]) -> None:
    """Rederiva los pesos del puntaje ordinal por búsqueda exhaustiva.

    Se busca sobre los casos NO rojos: los rojos ya los resuelve la regla de la capa 3 sin
    falsos positivos, así que el puntaje solo tiene que separar amarillo de verde. El
    criterio es lexicográfico y no una métrica agregada: primero recall 100% sobre
    `amarillo` (un amarillo perdido es un paciente que nadie revisa), y solo entre las
    combinaciones que lo logran se minimizan los falsos positivos.
    """
    no_rojos = [c for c in casos if c.label != "rojo"]
    resultados = []
    for w_dolor, w_herida, w_apetito, w_sueno, umbral, umbral_dolor in itertools.product(
        range(4), range(4), range(3), range(3), range(2, 6), (4.0, 5.0, 6.0)
    ):
        fn = fp = 0
        for caso in no_rojos:
            lec = caso.lectura
            puntaje = (
                w_dolor * int((lec.dolor or 0) >= umbral_dolor)
                + w_herida * ORDENES["herida"][lec.herida]
                + w_apetito * ORDENES["apetito"][lec.apetito]
                + w_sueno * ORDENES["sueno"][lec.sueno]
            )
            if caso.label == "amarillo" and puntaje < umbral:
                fn += 1
            elif caso.label == "verde" and puntaje >= umbral:
                fp += 1
        if fn == 0:
            resultados.append((fp, w_dolor, umbral_dolor, w_herida, w_apetito, w_sueno, umbral))

    resultados.sort()
    print(f"\nCalibración del puntaje ordinal sobre {len(no_rojos)} casos no rojos")
    print("Combinaciones con recall 100% sobre `amarillo`, ordenadas por falsos positivos:\n")
    print(f"  {'FP':>4}  {'dolor':>6} {'umbral':>7}  {'herida':>7} {'apetito':>8} {'sueno':>6}  {'corte':>6}")
    for fp, wd, ud, wh, wa, ws, umb in resultados[:10]:
        print(f"  {fp:>4}  {wd:>6} {ud:>7.0f}  {wh:>7} {wa:>8} {ws:>6}  {umb:>6}")

    actual = (PESO_DOLOR, DOLOR_AMARILLO_NRS, PESO_HERIDA, PESO_APETITO, PESO_SUENO, PUNTAJE_AMARILLO)
    en_uso = [r for r in resultados if r[1:] == actual]
    if en_uso:
        puesto = resultados.index(en_uso[0]) + 1
        print(f"\n  Los pesos en uso ({'/'.join(map(str, actual))}) quedan en el puesto {puesto} "
              f"de {len(resultados)}, con {en_uso[0][0]} falsos positivos.")
    else:
        print("\n  !! Los pesos en uso NO logran recall 100% sobre `amarillo`.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--perfil", choices=("conservador", "optimo"), default="conservador")
    parser.add_argument("--con-trayectoria", action="store_true",
                        help="activa la capa 6 (desviación contra recuperación normal)")
    parser.add_argument("--calibrar", action="store_true", help="rederiva los pesos y sale")
    parser.add_argument("--json", type=Path, help="guarda el resultado crudo")
    parser.add_argument("--ambos-perfiles", action="store_true", help="corre los dos perfiles")
    args = parser.parse_args()

    casos = cargar_casos()
    if len(casos) != 160:
        print(f"aviso: se cargaron {len(casos)} casos, se esperaban 160", file=sys.stderr)

    if args.calibrar:
        calibrar(casos)
        return 0

    perfiles = ("conservador", "optimo") if args.ambos_perfiles else (args.perfil,)
    resultados = [correr(casos, p, args.con_trayectoria) for p in perfiles]
    for res in resultados:
        imprimir(res)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nresultados en {args.json}")

    # Un falso negativo de rojo, o cualquier subestimación, hace fallar el comando.
    problemas = sum(r["falsos_negativos_rojo"] + r["subestimaciones"] for r in resultados)
    if problemas:
        print(f"\nFALLA: {problemas} caso(s) clasificados por debajo de su criticidad real.")
        return 1
    print("\nOK: ninguna subestimación. Ningún caso grave quedó sin escalar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
