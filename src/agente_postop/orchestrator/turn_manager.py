"""Un turno completo del paciente: extracción (llamada A) y conversación (llamada B) en
paralelo — no dependen la una de la otra dentro del mismo turno — más reflejo, fusión,
validación de citas y SBAR si corresponde, y persistencia en memoria si la llamada termina.

Este es el punto de integración de los patrones diferenciadores (arco reflejo, gemelo de
trayectoria, memoria longitudinal, validador de citas, extracción clínica) —
orquestar_turno() es lo que llama el servidor WebSocket por cada turno de voz, y lo que
el harness llama por cada fila del dataset (bypaseando el micrófono).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from agente_postop.clinical.citation_validator import validar_respuesta
from agente_postop.clinical.estado import EstadoClinicoLlamada, a_dict_trajectory_twin, fusionar_extraccion
from agente_postop.clinical.extraction import a_extraccion_turno
from agente_postop.clinical.fusion import fusionar
from agente_postop.clinical.memory import contexto_apertura
from agente_postop.clinical.models import Criticidad, ResultadoTurno
from agente_postop.clinical.reflex_engine import evaluar_via_refleja
from agente_postop.clinical.sbar import construir_sbar
from agente_postop.clinical.trajectory_twin import Desviacion, comparar, trayectoria_esperada
from agente_postop.orchestrator.cortex import extraer_turno, generar_respuesta

logger = logging.getLogger("agente_postop")

_ACCIONES_POR_CRITICIDAD = {
    Criticidad.ROJO: "Se escala de inmediato a personal médico — permanezca disponible, alguien del equipo lo va a contactar en los próximos minutos.",
    Criticidad.AMARILLO: "Se registra su reporte para seguimiento cercano del equipo médico; si algo empeora antes de la próxima llamada, contacte a urgencias.",
    Criticidad.DESCONOCIDA: "No se pudo confirmar toda la información necesaria en esta llamada; se deja registrado para revisión del equipo.",
}


def _desviaciones_relevantes(
    paciente_id: str, procedimiento: str, dia_postop: int, sintomas: dict, turno_idx: int
) -> tuple[list[Desviacion], str | None]:
    if not sintomas:
        return [], None
    esperado = trayectoria_esperada(paciente_id, procedimiento, dia_postop)
    if esperado is None:
        return [], None
    try:
        desviaciones = comparar(sintomas, esperado)
    except ValueError as exc:
        logger.warning("comparación de trayectoria falló en turno %s: %s", turno_idx, exc)
        return [], None
    relevantes = [d for d in desviaciones if d.empeora]
    texto = (
        "; ".join(
            f"{d.dimension} reportado={d.reportado} (esperado={d.esperado} para el día {dia_postop})"
            for d in relevantes
        )
        if relevantes
        else None
    )
    return desviaciones, texto


def orquestar_turno(
    *,
    turno_paciente: str,
    paciente_id: str,
    procedimiento: str,
    dia_postop: int,
    historial_turno: str,
    estado_clinico: EstadoClinicoLlamada,
    turno_idx: int,
    es_primer_turno_de_la_llamada: bool = False,
) -> ResultadoTurno:
    bandera_reflejo = evaluar_via_refleja(turno_paciente, procedimiento)
    contexto_memoria = contexto_apertura(paciente_id) if es_primer_turno_de_la_llamada else None

    # Snapshot del estado ANTES de este turno — es lo que ve la llamada B, porque corre en
    # paralelo con la extracción de este mismo turno, no después de ella.
    dimensiones_pendientes_previas = estado_clinico.dimensiones_pendientes
    sintomas_previos = a_dict_trajectory_twin(estado_clinico)
    _, desviaciones_texto_previas = _desviaciones_relevantes(
        paciente_id, procedimiento, dia_postop, sintomas_previos, turno_idx
    )

    # Llamada A (extracción) y B (conversación) en paralelo — independientes entre sí
    # dentro del mismo turno; correrlas una detrás de la otra solo suma latencia.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futuro_extraccion = pool.submit(extraer_turno, turno_paciente=turno_paciente, historial_turno=historial_turno)
        futuro_conversacion = pool.submit(
            generar_respuesta,
            turno_paciente=turno_paciente,
            procedimiento=procedimiento,
            contexto_memoria=contexto_memoria,
            desviaciones_trayectoria=desviaciones_texto_previas,
            historial_turno=historial_turno,
            dimensiones_pendientes=dimensiones_pendientes_previas,
        )

        try:
            delta = a_extraccion_turno(futuro_extraccion.result())
            fusionar_extraccion(estado_clinico, delta, turno_idx)
        except Exception as exc:  # noqa: BLE001 — una extracción fallida no debe tumbar el turno
            logger.warning("extracción de síntomas falló en turno %s: %s", turno_idx, exc)

        respuesta_cortex = futuro_conversacion.result()  # si falla (ej. rate limit), se propaga — el llamador decide

    sintomas_extraidos = a_dict_trajectory_twin(estado_clinico)
    desviaciones, _ = _desviaciones_relevantes(paciente_id, procedimiento, dia_postop, sintomas_extraidos, turno_idx)

    criticidad_final, reflejo_vetea = fusionar(respuesta_cortex.criticidad_propuesta, bandera_reflejo)

    verde_bloqueado = False
    if criticidad_final == Criticidad.VERDE and not estado_clinico.puede_cerrar_verde:
        # "Verde solo se otorga con evidencia positiva de ausencia de alarma" (prompts.py)
        # — traducido a código: sin cobertura completa y sin banderas, no hay verde.
        criticidad_final = Criticidad.DESCONOCIDA
        verde_bloqueado = True

    es_valida, texto_final = validar_respuesta(respuesta_cortex)
    afirmaciones_finales = respuesta_cortex.afirmaciones_clinicas if es_valida else []

    sbar = None
    if criticidad_final in (Criticidad.AMARILLO, Criticidad.ROJO):
        motivo = (
            f"bandera refleja: {bandera_reflejo.regla}"
            if bandera_reflejo.disparada
            else f"criticidad propuesta por el modelo: {respuesta_cortex.criticidad_propuesta.value}"
        )
        sbar = construir_sbar(
            paciente_nombre=paciente_id,
            procedimiento=procedimiento,
            dia_postop=dia_postop,
            motivo_alerta=motivo,
            contexto_previo=contexto_memoria,
            comorbilidades=None,
            sintomas_reportados={k: str(v) for k, v in sintomas_extraidos.items()},
            criticidad_final=criticidad_final,
            bandera_reflejo=bandera_reflejo,
            desviaciones=desviaciones,
            afirmaciones_clinicas=afirmaciones_finales,
            accion_comunicada=_ACCIONES_POR_CRITICIDAD.get(criticidad_final, ""),
        )

    return ResultadoTurno(
        respuesta_hablada=texto_final,
        criticidad_final=criticidad_final,
        criticidad_reflejo=bandera_reflejo.criticidad_forzada,
        criticidad_cortical=respuesta_cortex.criticidad_propuesta,
        reflejo_vetea=reflejo_vetea,
        afirmaciones_clinicas=afirmaciones_finales,
        cobertura=estado_clinico.cobertura,
        verde_bloqueado_por_cobertura=verde_bloqueado,
        sbar=sbar,
    )
