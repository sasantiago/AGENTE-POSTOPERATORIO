"""Tres llamadas completas contra el servidor vivo: una verde, una amarilla y una roja.

    python -m harness.demo_casos              # los tres, en orden
    python -m harness.demo_casos --caso rojo  # solo uno

Es el guion de la demostración: entra por el WebSocket real —el mismo que usa el
micrófono— y recorre las seis preguntas del protocolo con respuestas de paciente escritas
para que el motor de triaje llegue a un nivel concreto. No se fuerza nada: la criticidad la
decide `clinical/triage.py` con las mismas reglas de siempre, y el script **comprueba** que
salió la esperada; si no, falla.

Sirve para tres cosas:

  - ensayar la demo sin depender de que el micrófono coja bien;
  - mostrarle al jurado por qué se escaló cada caso, con la regla concreta;
  - detectar en un solo comando que un cambio rompió el escalamiento.

El bypass de texto del WebSocket es el mismo que usa `harness/runner.py`: se salta el STT
y nada más. Todo lo demás —guion, parsers, extracción, triaje, SBAR, audio— es idéntico a
una llamada hablada.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field

import websockets

URL = "ws://localhost:8000/ws/llamada"


@dataclass
class Caso:
    nivel: str
    paciente_id: str
    procedimiento: str
    dia_postop: int
    resumen: str
    # Una respuesta por pregunta del guion, en su orden:
    # dolor, fiebre, herida, movilidad, apetito, sueño.
    respuestas: list[str] = field(default_factory=list)
    regla_esperada: str | None = None


CASOS: dict[str, Caso] = {
    "verde": Caso(
        nivel="verde",
        paciente_id="pac_42_00000",
        procedimiento="Apendicectomía",
        dia_postop=7,
        resumen="Recuperación normal. Todo dentro de lo esperado, sin ninguna alarma.",
        respuestas=[
            "Muy bien doctora, el dolor casi no se siente, será un dos.",
            "Sí me la tomé esta mañana, treinta y seis y medio, normal.",
            "La herida se ve bien, cerradita, sin nada raro ni líquido.",
            "Camino normal, ya salgo hasta la tienda sin problema.",
            "Como bien, con hambre, todo me pasa bien.",
            "Duermo bien, de corrido toda la noche.",
        ],
    ),
    "amarillo": Caso(
        nivel="amarillo",
        paciente_id="pac_42_00001",
        procedimiento="Colecistectomía",
        dia_postop=3,
        resumen=(
            "Ningún síntoma cruza un umbral por sí solo, pero se acumulan: febrícula, "
            "enrojecimiento, poco apetito y mal sueño. Es el paciente que se deteriora sin "
            "disparar ninguna alarma individual."
        ),
        respuestas=[
            "Pues ahí, un cinco más o menos, molesta pero se aguanta.",
            "Me la tomé anoche y marcó treinta y siete y medio.",
            "Se ve un poquito rojita alrededor, pero no le sale nada de pus.",
            "Me muevo despacito, me cuesta un poco todavía.",
            "Como poquito, se me quitó el hambre estos días.",
            "Duermo mal, me despierto varias veces en la noche.",
        ],
        regla_esperada="febricula",
    ),
    "rojo": Caso(
        nivel="rojo",
        paciente_id="pac_42_00017",
        procedimiento="Colecistectomía",
        dia_postop=7,
        resumen=(
            "Fiebre de 38,5 °C y secreción purulenta. Escala en el segundo turno, sin "
            "esperar a terminar el cuestionario, y sin que ningún modelo intervenga en la "
            "decisión: la fiebre la lee un parser y la regla es determinista."
        ),
        respuestas=[
            "Uy doctora, el dolor está fuerte, como en un siete.",
            "Sí, me la tomé anoche y marcó treinta y ocho y medio.",
            "Le está saliendo un líquido amarillo y huele feo.",
            "Casi no me puedo mover, me cuesta mucho levantarme.",
            "No me provoca comer nada, se me quitó el hambre del todo.",
            "No he podido dormir, me la paso dando vueltas del dolor.",
        ],
        regla_esperada="fiebre_alta",
    ),
}


async def _recibir_turno(ws) -> tuple[dict, list[dict]]:
    """Lee hasta el audio del turno, recogiendo las anotaciones diferidas que lleguen."""
    anotaciones: list[dict] = []
    turno: dict | None = None
    while True:
        mensaje = await ws.recv()
        if isinstance(mensaje, bytes):
            return turno or {}, anotaciones
        datos = json.loads(mensaje)
        if datos.get("tipo") == "anotacion":
            anotaciones.append(datos)
        else:
            turno = datos


async def correr(caso: Caso, pausa: float) -> bool:
    """Reproduce la llamada. Devuelve si la criticidad final fue la esperada."""
    print(f"\n{'=' * 78}\n  CASO {caso.nivel.upper()} · {caso.paciente_id} · "
          f"{caso.procedimiento} · día {caso.dia_postop}\n  {caso.resumen}\n{'=' * 78}\n")

    async with websockets.connect(URL, max_size=None) as ws:
        await ws.send(json.dumps({
            "paciente_id": caso.paciente_id,
            "procedimiento": caso.procedimiento,
            "dia_postop": caso.dia_postop,
        }))
        turno, _ = await _recibir_turno(ws)
        print(f"AGENTE: {turno['respuesta_hablada']}\n")

        esperas: list[float] = []
        for respuesta in caso.respuestas:
            print(f"PACIENTE: {respuesta}")
            inicio = time.perf_counter()
            await ws.send(json.dumps({"texto": respuesta}))
            turno, anotaciones = await _recibir_turno(ws)
            esperas.append(time.perf_counter() - inicio)

            regla = (turno.get("decision_triaje") or {}).get("escalado_por") or "—"
            print(f"AGENTE [{esperas[-1] * 1000:.0f} ms · {turno['criticidad_final']} · "
                  f"cobertura {turno['cobertura']:.0%} · {regla}]")
            print(f"  {turno['respuesta_hablada']}")
            for anotacion in anotaciones:
                print(f"  · anotado: {anotacion['dimension']}={anotacion['valor']}")
            if turno.get("llamada_finalizada"):
                break
            # El paciente escucha antes de contestar. Sin la pausa, el guion corre más
            # rápido que una persona y las anotaciones diferidas no alcanzan a llegar.
            await asyncio.sleep(pausa)
            print()

        final = turno.get("criticidad_final")
        motivos = (turno.get("decision_triaje") or {}).get("motivos") or []
        print(f"\n  RESULTADO: {final}  (esperado: {caso.nivel})")
        if motivos:
            print("  Por qué:")
            for motivo in motivos:
                print(f"    · [{motivo['nivel']}] {motivo['detalle']}")
        if turno.get("sbar"):
            print(f"  SBAR → {turno['sbar']['situacion'][:150]}")
        print(f"  Espera mediana del paciente: {sorted(esperas)[len(esperas) // 2] * 1000:.0f} ms")

        ok = final == caso.nivel
        print(f"  {'OK' if ok else '!! NO COINCIDE'}")
        return ok


async def main_async(nombres: list[str], pausa: float) -> int:
    resultados = {}
    for nombre in nombres:
        try:
            resultados[nombre] = await correr(CASOS[nombre], pausa)
        except OSError as exc:
            print(f"\nNo se pudo conectar a {URL}: {exc}")
            print("Levanta el servidor primero.")
            return 1

    print(f"\n{'=' * 78}")
    for nombre, ok in resultados.items():
        print(f"  {nombre:<10} {'OK' if ok else 'NO COINCIDE'}")
    fallidos = [n for n, ok in resultados.items() if not ok]
    if fallidos:
        print(f"\nFALLA: {', '.join(fallidos)} no llegaron a su criticidad esperada.")
        return 1
    print("\nOK: los tres casos llegaron a la criticidad esperada.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--caso", choices=(*CASOS, "todos"), default="todos")
    parser.add_argument("--pausa", type=float, default=5.0,
                        help="segundos entre turnos, simulando que el paciente escucha")
    args = parser.parse_args()
    nombres = list(CASOS) if args.caso == "todos" else [args.caso]
    return asyncio.run(main_async(nombres, args.pausa))


if __name__ == "__main__":
    raise SystemExit(main())
