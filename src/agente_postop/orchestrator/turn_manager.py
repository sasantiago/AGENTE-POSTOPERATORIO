"""Un turno completo del paciente: reflejo + cortex en paralelo lógico, fusión,
validación de citas, y persistencia en memoria si la llamada termina.

Este es el punto de integración de los patrones diferenciadores (arco reflejo, gemelo de
trayectoria, memoria longitudinal, validador de citas) — orquesta_turno() es lo que llama
el servidor WebSocket por cada turno de voz, y lo que el harness llama por cada fila del
dataset (bypaseando el micrófono).
"""

from __future__ import annotations

from agente_postop.clinical.citation_validator import validar_respuesta
from agente_postop.clinical.fusion import fusionar
from agente_postop.clinical.memory import contexto_apertura
from agente_postop.clinical.models import Criticidad, ResultadoTurno
from agente_postop.clinical.reflex_engine import evaluar_via_refleja
from agente_postop.clinical.trajectory_twin import comparar, trayectoria_esperada
from agente_postop.orchestrator.cortex import generar_respuesta


def orquestar_turno(
    *,
    turno_paciente: str,
    paciente_id: str,
    procedimiento: str,
    dia_postop: int,
    historial_turno: str,
    sintomas_extraidos: dict | None = None,
    es_primer_turno_de_la_llamada: bool = False,
) -> ResultadoTurno:
    bandera_reflejo = evaluar_via_refleja(turno_paciente, procedimiento)

    contexto_memoria = contexto_apertura(paciente_id) if es_primer_turno_de_la_llamada else None

    desviaciones_texto = None
    if sintomas_extraidos:
        esperado = trayectoria_esperada(paciente_id, procedimiento, dia_postop)
        if esperado is not None:
            desviaciones = comparar(sintomas_extraidos, esperado)
            relevantes = [d for d in desviaciones if d.empeora]
            if relevantes:
                desviaciones_texto = "; ".join(
                    f"{d.dimension} reportado={d.reportado} (esperado={d.esperado} para el día {dia_postop})"
                    for d in relevantes
                )

    respuesta_cortex = generar_respuesta(
        turno_paciente=turno_paciente,
        procedimiento=procedimiento,
        contexto_memoria=contexto_memoria,
        desviaciones_trayectoria=desviaciones_texto,
        historial_turno=historial_turno,
    )

    criticidad_final, reflejo_vetea = fusionar(respuesta_cortex.criticidad_propuesta, bandera_reflejo)

    es_valida, texto_final = validar_respuesta(respuesta_cortex)
    afirmaciones_finales = respuesta_cortex.afirmaciones_clinicas if es_valida else []

    return ResultadoTurno(
        respuesta_hablada=texto_final,
        criticidad_final=criticidad_final,
        criticidad_reflejo=bandera_reflejo.criticidad_forzada,
        criticidad_cortical=respuesta_cortex.criticidad_propuesta,
        reflejo_vetea=reflejo_vetea,
        afirmaciones_clinicas=afirmaciones_finales,
    )
