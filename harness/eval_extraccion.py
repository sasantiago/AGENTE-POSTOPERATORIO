"""Evaluación del extractor: dado lo que el paciente DICE, ¿el sistema entiende bien?

    python -m harness.eval_extraccion --n 12 --capa capa1_limpia
    python -m harness.eval_extraccion --n 12 --capa capa2_ruidosa --json harness/resultados_extraccion.json

Es la otra mitad de la pregunta. `eval_triaje.py` mide el motor de decisión alimentado con
slots correctos y sale con 100% de recall sobre `rojo`; esa garantía **se apoya entera en
que los slots sean correctos**. Si el extractor se come la fiebre, el motor decide
impecablemente sobre datos equivocados y el número de arriba no vale nada.

    eval_triaje.py      →  dado lo que el paciente TIENE, ¿decide bien?    (160 casos, 0 tokens)
    eval_extraccion.py  →  dado lo que el paciente DICE, ¿entiende bien?   (muestra, con LLM)

## Metodología

**Se compara el estado acumulado al final de la llamada, no turno a turno.** Un turno
suelto menciona una o dos dimensiones; exigirle las seis penalizaría al extractor por no
adivinar lo que nadie le preguntó. El dataset etiqueta la trayectoria del paciente ese día,
que es justo lo que la llamada entera debería haber averiguado.

**Las dos capas se miden por separado.** `capa1_limpia` es el techo del extractor con
transcripción perfecta; `capa2_ruidosa` es lo más cercano a lo que entrega Whisper en una
llamada real. El delta entre ambas es la evidencia de robustez frente al ruido.

## Las tres métricas, y por qué no basta la primera

  - **acierto**: el slot se confirmó con el valor correcto.
  - **error**: se confirmó con un valor equivocado. Es el caso grave: el motor de triaje
    decide sobre un dato falso creyéndolo bueno.
  - **omisión**: no se confirmó. No produce un falso verde —el diseño lo impide, sin
    cobertura completa la criticidad degrada a `desconocida`— pero un agente que deja
    todo en `desconocida` es seguro e inútil a la vez.

Un extractor que omite todo saca 0% de error y es basura. Por eso las tres se reportan
juntas, y por eso se reporta aparte la **dirección** de los errores: subestimar la gravedad
de una herida no es lo mismo que sobrestimarla.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from agente_postop.clinical.estado import EstadoClinicoLlamada, fusionar_extraccion
from agente_postop.clinical.extraction import DIMENSIONES, a_extraccion_turno
from agente_postop.clinical.triage import ORDENES
from agente_postop.config import get_settings
from agente_postop.orchestrator.cortex import extraer_turno
from harness.reproducer import reconstruir_caso

warnings.filterwarnings("ignore", message="Workbook contains no default style")

# Tolerancias. La trayectoria es el cuadro del paciente ese día y el paciente lo relata
# de memoria y en aproximaciones («como un 4», «38 y algo»): exigir igualdad exacta mediría
# la fidelidad del relato, no la del extractor.
TOLERANCIA_DOLOR = 1.0   # puntos de la escala 0-10
TOLERANCIA_FIEBRE = 0.3  # °C


@dataclass
class ResultadoSlot:
    caso: str
    capa: str
    dimension: str
    esperado: str
    obtenido: str | None
    veredicto: str  # acierto | error | omision
    direccion: str | None = None  # subestima | sobrestima, solo en errores ordinales


@dataclass
class ResultadoCaso:
    caso: str
    capa: str
    turnos: int
    cobertura: float
    segundos: float
    slots: list[ResultadoSlot] = field(default_factory=list)
    fallo: str | None = None


def _comparar(dimension: str, esperado, obtenido) -> tuple[str, str | None]:
    """Veredicto de un slot y, si es un error ordinal, en qué dirección."""
    if obtenido is None:
        return "omision", None

    if dimension == "dolor":
        return ("acierto", None) if abs(float(obtenido) - float(esperado)) <= TOLERANCIA_DOLOR else (
            "error", "subestima" if float(obtenido) < float(esperado) else "sobrestima"
        )
    if dimension == "fiebre":
        return ("acierto", None) if abs(float(obtenido) - float(esperado)) <= TOLERANCIA_FIEBRE else (
            "error", "subestima" if float(obtenido) < float(esperado) else "sobrestima"
        )

    if str(obtenido) == str(esperado):
        return "acierto", None
    orden = ORDENES[dimension]
    if str(obtenido) in orden and str(esperado) in orden:
        return "error", "subestima" if orden[str(obtenido)] < orden[str(esperado)] else "sobrestima"
    return "error", None


def correr_caso(caso_id: str, capa: str, esperado: dict) -> ResultadoCaso:
    """Reproduce la llamada completa pasando SOLO por el extractor.

    No se llama a `orquestar_turno`: la vía de conversación gastaría una segunda invocación
    por turno y aquí no se mide. Lo que se ejercita es exactamente el camino
    `extraer_turno` → `a_extraccion_turno` → `fusionar_extraccion` que corre en la llamada
    real, incluido el acumulador determinista.
    """
    caso = reconstruir_caso(caso_id, capa)
    estado = EstadoClinicoLlamada()
    inicio = time.perf_counter()
    historial: list[str] = []
    turnos = 0

    for turno in caso.turnos:
        if turno.hablante != "paciente":
            historial.append(f"agente: {turno.texto}")
            continue
        try:
            crudo = extraer_turno(
                turno_paciente=turno.texto,
                historial_turno="\n".join(historial[-6:]) or "(inicio de la llamada)",
            )
            fusionar_extraccion(estado, a_extraccion_turno(crudo), turnos)
        except Exception as exc:  # noqa: BLE001 — un turno perdido no invalida el caso
            return ResultadoCaso(caso_id, capa, turnos, estado.cobertura,
                                 time.perf_counter() - inicio, fallo=f"{type(exc).__name__}: {exc}")
        historial.append(f"paciente: {turno.texto}")
        turnos += 1

    resultado = ResultadoCaso(caso_id, capa, turnos, estado.cobertura, time.perf_counter() - inicio)
    for dimension in DIMENSIONES:
        obs = getattr(estado, dimension)
        obtenido = obs.valor if obs.confirmada else None
        veredicto, direccion = _comparar(dimension, esperado[dimension], obtenido)
        resultado.slots.append(
            ResultadoSlot(caso_id, capa, dimension, str(esperado[dimension]),
                          None if obtenido is None else str(obtenido), veredicto, direccion)
        )
    return resultado


def cargar_esperados() -> dict[str, dict]:
    """El cuadro verdadero de cada caso: `caso_id` → los seis slots de la trayectoria."""
    settings = get_settings()
    tray = pd.read_excel(settings.dataset_dir / "trayectorias_postop_silver.xlsx")
    return {
        f"caso_{fila.trayectoria_id}": {
            "dolor": float(fila.dolor_nrs), "fiebre": float(fila.fiebre_c),
            "movilidad": fila.movilidad, "herida": fila.herida,
            "apetito": fila.apetito, "sueno": fila.sueno,
        }
        for fila in tray.itertuples()
    }


def muestrear(esperados: dict, capa: str, n: int, semilla: int) -> list[str]:
    """Muestra estratificada por criticidad.

    Estratificada y no aleatoria porque los `rojo` son 12 de 160: una muestra uniforme de
    12 casos traería menos de uno, y son justo los casos donde una omisión de la fiebre
    cuesta más.
    """
    settings = get_settings()
    dialogos = pd.read_excel(settings.dataset_dir / "dataset_final.xlsx")
    dialogos = dialogos[dialogos["capa"] == capa]
    etiquetas = dialogos.groupby("caso_id")["label_ground_truth"].first()

    por_nivel: dict[str, list[str]] = {"rojo": [], "amarillo": [], "verde": []}
    for caso_id, nivel in etiquetas.items():
        if caso_id in esperados and nivel in por_nivel:
            por_nivel[nivel].append(caso_id)

    # Reparto a partes iguales, repartiendo el resto empezando por `rojo`. Así con n=1 se
    # evalúa un rojo —el caso donde una omisión cuesta más— y no un verde cualquiera.
    niveles = ["rojo", "amarillo", "verde"]
    reparto = {nivel: n // 3 for nivel in niveles}
    for i in range(n % 3):
        reparto[niveles[i]] += 1

    rng = random.Random(semilla)
    muestra: list[str] = []
    faltante = 0
    for nivel in niveles:
        disponibles = sorted(por_nivel[nivel])
        cuantos = min(reparto[nivel], len(disponibles))
        faltante += reparto[nivel] - cuantos
        muestra.extend(rng.sample(disponibles, cuantos))

    # Si un estrato no tenía casos suficientes (los `rojo` son 12 de 160), el hueco se
    # rellena con los demás en vez de devolver una muestra más chica de lo pedido.
    if faltante:
        resto = sorted({c for nivel in niveles for c in por_nivel[nivel]} - set(muestra))
        muestra.extend(rng.sample(resto, min(faltante, len(resto))))
    return muestra


def resumir(resultados: list[ResultadoCaso]) -> dict:
    slots = [s for r in resultados for s in r.slots]
    validos = [r for r in resultados if r.fallo is None]
    conteo = Counter(s.veredicto for s in slots)
    total = len(slots) or 1

    por_dimension = {}
    for dimension in DIMENSIONES:
        de_esta = [s for s in slots if s.dimension == dimension]
        c = Counter(s.veredicto for s in de_esta)
        por_dimension[dimension] = {
            "acierto": c["acierto"], "error": c["error"], "omision": c["omision"],
            "exactitud": round(c["acierto"] / len(de_esta), 3) if de_esta else 0.0,
        }

    errores = [s for s in slots if s.veredicto == "error"]
    return {
        "casos": len(resultados),
        "casos_con_fallo": len(resultados) - len(validos),
        "slots_evaluados": len(slots),
        "acierto": round(conteo["acierto"] / total, 3),
        "error": round(conteo["error"] / total, 3),
        "omision": round(conteo["omision"] / total, 3),
        "cobertura_media": round(sum(r.cobertura for r in validos) / len(validos), 3) if validos else 0.0,
        "llamadas_con_cobertura_completa": sum(1 for r in validos if r.cobertura == 1.0),
        "segundos_por_caso": round(sum(r.segundos for r in validos) / len(validos), 1) if validos else 0.0,
        "errores_que_subestiman": sum(1 for s in errores if s.direccion == "subestima"),
        "errores_que_sobrestiman": sum(1 for s in errores if s.direccion == "sobrestima"),
        "por_dimension": por_dimension,
    }


def imprimir(res: dict, resultados: list[ResultadoCaso], capa: str) -> None:
    settings = get_settings()
    print(f"\nExtractor — capa `{capa}` · backend `{settings.llm_backend}` "
          f"({settings.modelo_de[settings.llm_backend]})")
    print(f"{res['casos']} casos · {res['slots_evaluados']} slots · "
          f"{res['segundos_por_caso']:.0f} s por llamada\n")

    print(f"  aciertos   {res['acierto']:>6.1%}")
    print(f"  errores    {res['error']:>6.1%}   (valor confirmado pero equivocado)")
    print(f"  omisiones  {res['omision']:>6.1%}   (no se confirmó: degrada a `desconocida`)")
    print(f"\n  cobertura media de la llamada     {res['cobertura_media']:.1%}")
    print(f"  llamadas con las 6 confirmadas    {res['llamadas_con_cobertura_completa']}/{res['casos']}")
    print(f"  errores que SUBESTIMAN la gravedad {res['errores_que_subestiman']}"
          f"  ·  que la sobrestiman {res['errores_que_sobrestiman']}")

    print(f"\n  {'dimensión':<12} {'acierto':>8} {'error':>7} {'omisión':>8}")
    for dimension, d in res["por_dimension"].items():
        print(f"  {dimension:<12} {d['exactitud']:>7.0%} {d['error']:>7} {d['omision']:>8}")

    errores = [s for r in resultados for s in r.slots if s.veredicto == "error"]
    if errores:
        print(f"\n  errores concretos ({len(errores)}):")
        for s in errores[:12]:
            marca = "!!" if s.direccion == "subestima" else "  "
            print(f"  {marca} {s.caso:<28} {s.dimension:<10} esperado={s.esperado:<22} obtenido={s.obtenido}")

    for r in resultados:
        if r.fallo:
            print(f"\n  caso fallido: {r.caso} — {r.fallo}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=9, help="casos a evaluar (muestra estratificada)")
    parser.add_argument("--capa", default="capa1_limpia", choices=("capa1_limpia", "capa2_ruidosa"))
    parser.add_argument("--semilla", type=int, default=7)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    esperados = cargar_esperados()
    muestra = muestrear(esperados, args.capa, args.n, args.semilla)
    print(f"Evaluando el extractor sobre {len(muestra)} casos de {args.capa}.")
    print("Una llamada completa por caso; a varios segundos por turno, esto tarda.\n")

    resultados = []
    for i, caso_id in enumerate(muestra, 1):
        print(f"  [{i}/{len(muestra)}] {caso_id}", end="", flush=True)
        r = correr_caso(caso_id, args.capa, esperados[caso_id])
        resultados.append(r)
        aciertos = sum(1 for s in r.slots if s.veredicto == "acierto")
        print(f"  {r.turnos} turnos · {r.segundos:.0f} s · {aciertos}/6 slots"
              + (f"  FALLO: {r.fallo}" if r.fallo else ""), flush=True)

    res = resumir(resultados)
    imprimir(res, resultados, args.capa)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"resumen": res, "casos": [asdict(r) for r in resultados]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nresultados en {args.json}")

    # A diferencia de eval_triaje.py, esto NO falla el build por un umbral: no hay una
    # línea defendible que separe "extractor aceptable" de "inaceptable" sin haberlo
    # medido antes en varias configuraciones. Es un informe, y lo dice.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
