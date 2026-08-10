"""Runner: inyecta cada turno de paciente al orquestador (bypass del micrófono) y
captura la clasificación final, turno a turno, para comparar contra label_ground_truth.

El historial que ve el agente se construye con SUS PROPIAS respuestas generadas (no las
frases originales del dataset) — el agente conduce su propia conversación; el dataset solo
aporta lo que dice el paciente y la etiqueta de verdad contra la que se compara.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from agente_postop.orchestrator.turn_manager import orquestar_turno
from harness.reproducer import Caso, procedimiento_de, reconstruir_caso


@dataclass
class ResultadoEvaluacionTurno:
    caso_id: str
    capa: str
    dialogo_id: int
    turno_idx: int
    texto_paciente: str
    label_ground_truth: str
    criticidad_predicha: str
    reflejo_vetea: bool
    latencia_s: float


def correr_caso(caso: Caso) -> list[ResultadoEvaluacionTurno]:
    procedimiento = procedimiento_de(caso.paciente_id)
    resultados: list[ResultadoEvaluacionTurno] = []
    historial: list[str] = []
    es_primer_turno = True

    for turno in caso.turnos:
        if turno.hablante != "paciente":
            continue

        inicio = time.perf_counter()
        resultado = orquestar_turno(
            turno_paciente=turno.texto,
            paciente_id=caso.paciente_id,
            procedimiento=procedimiento,
            dia_postop=caso.dia_postop,
            historial_turno="\n".join(historial[-6:]) if historial else "(inicio de la llamada)",
            sintomas_extraidos=None,
            es_primer_turno_de_la_llamada=es_primer_turno,
        )
        latencia_s = time.perf_counter() - inicio
        es_primer_turno = False

        historial.append(f"paciente: {turno.texto}")
        historial.append(f"agente: {resultado.respuesta_hablada}")

        resultados.append(
            ResultadoEvaluacionTurno(
                caso_id=caso.caso_id,
                capa=caso.capa,
                dialogo_id=turno.dialogo_id,
                turno_idx=turno.turno_idx,
                texto_paciente=turno.texto,
                label_ground_truth=turno.label_ground_truth,
                criticidad_predicha=resultado.criticidad_final.value,
                reflejo_vetea=resultado.reflejo_vetea,
                latencia_s=latencia_s,
            )
        )

    return resultados


def correr_casos(caso_ids: list[str], capa: str) -> list[ResultadoEvaluacionTurno]:
    resultados: list[ResultadoEvaluacionTurno] = []
    for i, caso_id in enumerate(caso_ids, 1):
        print(f"[{i}/{len(caso_ids)}] {caso_id} ({capa})")
        try:
            caso = reconstruir_caso(caso_id, capa)
        except ValueError:
            continue
        resultados.extend(correr_caso(caso))
    return resultados
