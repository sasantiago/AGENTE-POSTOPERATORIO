"""Evaluación de los parsers deterministas de fiebre y dolor. Sin LLM, sin tokens.

    python -m harness.eval_parsers
    python -m harness.eval_parsers --capa capa2_ruidosa --mostrar-fallos

Mide las dos funciones de `clinical/parsers.py` contra los turnos reales del dataset,
usando como verdad la trayectoria del caso. Corre en segundos sobre los 3.991 turnos, que
es exactamente la ventaja de sacar estos dos valores del modelo.

## Qué se mide, y por qué la dirección importa más que la tasa

Un parser de fiebre no se juzga por su exactitud global. Se juzga por **en qué dirección se
equivoca**:

  - **detectado correcto**: el valor coincide con la trayectoria (con tolerancia).
  - **error peligroso**: lee un valor MENOR que el real, o no detecta una fiebre/dolor que
    sí cruzaba el umbral de escalamiento. Es el único que puede costar un paciente.
  - **error seguro**: lee un valor mayor que el real. Cuesta un falso positivo.
  - **no reportado**: el turno no menciona el dato. No es un fallo: la mayoría de los
    turnos de una llamada no hablan de temperatura.

El criterio de aceptación es **cero errores peligrosos sobre los turnos que cruzan el
umbral**, no un porcentaje. Por eso el comando sale con código distinto de cero si aparece
uno.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from agente_postop.clinical.parsers import extraer_dolor_nrs, extraer_temperatura_c
from agente_postop.clinical.triage import DOLOR_ROJO_NRS, FIEBRE_ROJO_C
from agente_postop.config import get_settings

warnings.filterwarnings("ignore", message="Workbook contains no default style")

TOLERANCIA_DOLOR = 1.0
TOLERANCIA_FIEBRE = 0.5  # "38 y algo" es 38.4: el paciente aproxima, el parser lee lo dicho


@dataclass
class Fallo:
    caso: str
    texto: str
    esperado: float
    obtenido: float | None
    tipo: str


def cargar_turnos(capa: str) -> pd.DataFrame:
    """Turnos de paciente con el cuadro verdadero de su caso pegado al lado."""
    settings = get_settings()
    dialogos = pd.read_excel(settings.dataset_dir / "dataset_final.xlsx")
    tray = pd.read_excel(settings.dataset_dir / "trayectorias_postop_silver.xlsx")
    tray["caso_id"] = "caso_" + tray["trayectoria_id"]

    turnos = dialogos[(dialogos["capa"] == capa) & (dialogos["hablante"] == "paciente")]
    return turnos.merge(tray[["caso_id", "dolor_nrs", "fiebre_c"]], on="caso_id", how="inner")


def evaluar(turnos: pd.DataFrame, dimension: str) -> tuple[dict, list[Fallo]]:
    """Compara lo que el parser lee en cada turno contra el valor verdadero del caso.

    Un turno que no menciona el dato no es un fallo del parser: es un turno que habla de
    otra cosa. Lo que sí es un fallo es **leer un valor equivocado**, y sobre todo leerlo
    por debajo del real.
    """
    parser = extraer_temperatura_c if dimension == "fiebre" else extraer_dolor_nrs
    columna = "fiebre_c" if dimension == "fiebre" else "dolor_nrs"
    tolerancia = TOLERANCIA_FIEBRE if dimension == "fiebre" else TOLERANCIA_DOLOR
    umbral_rojo = FIEBRE_ROJO_C if dimension == "fiebre" else DOLOR_ROJO_NRS

    conteo: Counter = Counter()
    fallos: list[Fallo] = []
    detectados_sobre_umbral = 0
    turnos_de_casos_sobre_umbral = 0
    casos_sobre_umbral_detectados: set[str] = set()
    casos_sobre_umbral: set[str] = set()

    for fila in turnos.itertuples():
        real = float(getattr(fila, columna))
        leido = parser(str(fila.texto))
        caso = fila.caso_id
        if real >= umbral_rojo:
            casos_sobre_umbral.add(caso)
            turnos_de_casos_sobre_umbral += 1

        if leido is None:
            conteo["no_reportado"] += 1
            continue

        # Lo que decide el triaje es el LADO del umbral, no el valor exacto. Ante «marcaba
        # como 38 y algo» el parser lee 38.0 cuando la trayectoria dice 38.4: son valores
        # distintos y la MISMA decisión clínica. Medir eso como fallo confunde fidelidad
        # numérica con utilidad, que es exactamente el artefacto que produjo el recall del
        # 8,3% en la primera entrega.
        if real >= umbral_rojo and leido >= umbral_rojo:
            casos_sobre_umbral_detectados.add(caso)
            detectados_sobre_umbral += 1

        if abs(leido - real) <= tolerancia:
            conteo["correcto"] += 1
        elif leido < real:
            # Peligroso solo si además cruza el umbral: leer 3 donde había 4 no cambia
            # ninguna decisión; leer 6 donde había 9 sí.
            peligroso = real >= umbral_rojo and leido < umbral_rojo
            conteo["error_peligroso" if peligroso else "error_menor"] += 1
            fallos.append(Fallo(caso, str(fila.texto)[:100], real, leido,
                                "peligroso" if peligroso else "menor"))
        else:
            conteo["error_seguro"] += 1
            fallos.append(Fallo(caso, str(fila.texto)[:100], real, leido, "seguro"))

    leidos = sum(v for k, v in conteo.items() if k != "no_reportado")
    return {
        "dimension": dimension,
        "turnos": len(turnos),
        "turnos_con_lectura": leidos,
        "correcto": conteo["correcto"],
        "error_peligroso": conteo["error_peligroso"],
        "error_menor": conteo["error_menor"],
        "error_seguro": conteo["error_seguro"],
        "no_reportado": conteo["no_reportado"],
        "precision_de_lo_leido": round(conteo["correcto"] / leidos, 3) if leidos else 0.0,
        # La métrica que de verdad importa: de los casos cuyo valor real cruzaba el umbral
        # de rojo, ¿en cuántos el parser lo vio en ALGÚN turno de la llamada? Un parser no
        # necesita leer la fiebre en los seis turnos, necesita leerla una vez.
        "casos_sobre_umbral": len(casos_sobre_umbral),
        "casos_sobre_umbral_detectados": len(casos_sobre_umbral_detectados),
        "recall_por_caso": round(len(casos_sobre_umbral_detectados) / len(casos_sobre_umbral), 3)
        if casos_sobre_umbral else 1.0,
        "turnos_de_casos_sobre_umbral": turnos_de_casos_sobre_umbral,
        "turnos_sobre_umbral_detectados": detectados_sobre_umbral,
    }, fallos


def imprimir(res: dict, fallos: list[Fallo], mostrar_fallos: bool) -> None:
    d = res["dimension"]
    umbral = FIEBRE_ROJO_C if d == "fiebre" else DOLOR_ROJO_NRS
    print(f"\n  {d.upper()} — umbral de rojo: {umbral}")
    print(f"    turnos evaluados            {res['turnos']}")
    print(f"    turnos donde leyó un valor  {res['turnos_con_lectura']}")
    print(f"    · correcto                  {res['correcto']}")
    print(f"    · error seguro (lee de más) {res['error_seguro']}")
    print(f"    · error menor (sin efecto)  {res['error_menor']}")
    print(f"    · ERROR PELIGROSO           {res['error_peligroso']}")
    print(f"    precisión de lo que leyó    {res['precision_de_lo_leido']:.1%}")
    print(f"\n    casos con {d} sobre el umbral: {res['casos_sobre_umbral']}")
    print(f"    detectados en algún turno:   {res['casos_sobre_umbral_detectados']}"
          f"  ({res['recall_por_caso']:.1%})")

    if mostrar_fallos and fallos:
        peligrosos = [f for f in fallos if f.tipo == "peligroso"]
        otros = [f for f in fallos if f.tipo != "peligroso"]
        for titulo, lista in (("PELIGROSOS", peligrosos), ("otros", otros[:8])):
            if lista:
                print(f"\n    {titulo}:")
                for f in lista[:10]:
                    print(f"      real={f.esperado} leído={f.obtenido}  «{f.texto}»")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capa", default="capa1_limpia", choices=("capa1_limpia", "capa2_ruidosa", "ambas"))
    parser.add_argument("--mostrar-fallos", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    capas = ("capa1_limpia", "capa2_ruidosa") if args.capa == "ambas" else (args.capa,)
    salida = []
    peligrosos_totales = 0

    for capa in capas:
        turnos = cargar_turnos(capa)
        print(f"\nParsers deterministas — capa `{capa}` · {len(turnos)} turnos de paciente")
        for dimension in ("fiebre", "dolor"):
            res, fallos = evaluar(turnos, dimension)
            res["capa"] = capa
            imprimir(res, fallos, args.mostrar_fallos)
            salida.append(res)
            peligrosos_totales += res["error_peligroso"]

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nresultados en {args.json}")

    if peligrosos_totales:
        print(f"\nFALLA: {peligrosos_totales} lectura(s) por debajo del umbral de escalamiento.")
        return 1
    print("\nOK: ninguna lectura cayó por debajo del umbral de escalamiento.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
