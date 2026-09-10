"""Un turno completo del paciente: extracción (llamada A) y conversación (llamada B) en
paralelo — no dependen la una de la otra dentro del mismo turno — más reflejo, fusión,
validación de citas y SBAR si corresponde, y persistencia en memoria si la llamada termina.

Este es el punto de integración de los patrones diferenciadores (arco reflejo, gemelo de
trayectoria, memoria longitudinal, validador de citas, extracción clínica) —
orquestar_turno() es lo que llama el servidor WebSocket por cada turno de voz, y lo que
el harness llama por cada fila del dataset (bypaseando el micrófono).
"""

from __future__ import annotations

import contextvars
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from agente_postop.clinical.citation_validator import validar_respuesta
from agente_postop.clinical.estado import EstadoClinicoLlamada, a_dict_trajectory_twin, fusionar_extraccion
from agente_postop.clinical.extraction import (
    Confianza,
    EstadoSlot,
    ExtraccionTurno,
    Observacion,
    a_extraccion_turno,
)
from agente_postop.clinical.guion import (
    CIERRE_AMARILLO,
    CIERRE_ROJO,
    CIERRE_VERDE,
    siguiente_pregunta,
    texto_a_decir,
)
from agente_postop.clinical.parsers import imponer_parsers
from agente_postop.clinical.memory import contexto_apertura
from agente_postop.clinical.models import Criticidad, ResultadoTurno
from agente_postop.clinical.pacientes import buscar_paciente
from agente_postop.clinical.reflex_engine import evaluar_via_refleja
from agente_postop.clinical.sbar import construir_sbar
from agente_postop.clinical.trajectory_twin import Desviacion, comparar, trayectoria_esperada
from agente_postop.clinical.triage import desde_estado
from agente_postop.clinical.triage import evaluar as evaluar_triaje
from agente_postop.config import get_settings
from agente_postop.orchestrator.cortex import (
    extraer_slot,
    extraer_turno,
    generar_respuesta,
    necesita_respaldo_clinico,
    responder_pregunta_anclada,
)

logger = logging.getLogger("agente_postop")

_ACCIONES_POR_CRITICIDAD = {
    Criticidad.ROJO: "Se escala de inmediato a personal médico — permanezca disponible, alguien del equipo lo va a contactar en los próximos minutos.",
    Criticidad.AMARILLO: "Se registra su reporte para seguimiento cercano del equipo médico; si algo empeora antes de la próxima llamada, contacte a urgencias.",
    Criticidad.DESCONOCIDA: "No se pudo confirmar toda la información necesaria en esta llamada; se deja registrado para revisión del equipo.",
}


def _comorbilidades_de(paciente_id: str) -> str | None:
    """Del registro clínico. Un paciente que no esté en él no rompe la llamada: el SBAR
    dirá "ninguna registrada", que es exactamente lo que se sabe."""
    paciente = buscar_paciente(paciente_id)
    return paciente.texto_comorbilidades if paciente else None


def _texto_confirmado(estado: EstadoClinicoLlamada) -> str | None:
    """Lo ya confirmado, con la frase textual que lo sustenta, para que la vía cortical
    pueda replicar ante una retractación en vez de darla por buena. El verbatim importa:
    citarle al paciente sus propias palabras es lo que distingue contradecirlo de
    recordarle lo que dijo."""
    partes = []
    for nombre in estado.dimensiones_confirmadas:
        obs = getattr(estado, nombre)
        soporte = f' (dijo: "{obs.verbatim}")' if obs.verbatim else ""
        partes.append(f"{nombre}={obs.valor}{soporte}")
    return "; ".join(partes) if partes else None


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


def orquestar_turno_guiado(
    *,
    turno_paciente: str,
    paciente_id: str,
    procedimiento: str,
    dia_postop: int,
    estado_clinico: EstadoClinicoLlamada,
    pregunta_pendiente,
    turno_idx: int,
    historial_turno: str = "",
    es_primer_turno_de_la_llamada: bool = False,
    diferir_extraccion: bool = False,
    en_vuelo: set[str] | None = None,
) -> tuple[ResultadoTurno, object | None, object | None]:
    """Un turno del protocolo: el agente pregunta, el paciente responde, el sistema anota.

    Devuelve `(resultado, siguiente_pregunta, completar)`. La segunda es None cuando ya no
    queda nada por preguntar y la llamada debe cerrarse; la tercera es la clasificación
    diferida que el llamador debe ejecutar en segundo plano, o None si ya se hizo.

    Es el camino que reemplaza a `orquestar_turno` cuando el agente conduce con guion fijo.
    La diferencia no es de estilo, es de cuánto trabajo se le pide al modelo:

        camino libre    2 invocaciones/turno · esquema de 10.126 chars · redacta la pregunta
        camino guiado   0 o 1 invocación     · enum de 3 valores       · pregunta fija

    Medido en este equipo con `llama3.2:3b`: la extracción baja de 16,2 s a 8,1 s, el TTS de
    ~1.000 ms a 20 ms (audio pre-sintetizado), y el acierto de extracción sube de 1/6 slots a
    5/5. El modelo deja de hacer las tres cosas que hacía mal —redactar, decidir y rellenar
    un esquema enorme— y se queda con la única que hace bien: clasificar una frase corta.

    El modelo grande solo entra cuando el paciente **pregunta** algo, y ahí la respuesta va
    anclada al corpus y validada antes de pronunciarse.
    """
    bandera_reflejo = evaluar_via_refleja(turno_paciente, procedimiento)

    # --- Lo instantáneo: reflejo y números ----------------------------------------
    #
    # Las banderas rojas del habla y los dos valores que disparan ROJO se resuelven aquí,
    # en microsegundos y sin red. Es lo que permite responderle al paciente antes de que la
    # extracción categórica termine: nada de lo que decide un escalamiento urgente depende
    # de esa espera.
    delta = ExtraccionTurno()
    if pregunta_pendiente is not None and pregunta_pendiente.valores is None:
        # Dimensión numérica: la resuelve `imponer_parsers`. Se marca como preguntada para
        # que el contador de intentos avance aunque el parser no lea nada, que es lo que
        # habilita el reintento.
        setattr(
            delta.dimensiones,
            pregunta_pendiente.dimension,
            Observacion(estado=EstadoSlot.PREGUNTADO_SIN_RESPUESTA, verbatim=turno_paciente[:200]),
        )

    # Dolor y fiebre siempre los lee el parser, se hayan preguntado o no: el paciente puede
    # mencionar la fiebre mientras contesta por la herida, y perder ese dato por no estar
    # en el turno correcto sería tirar la señal más importante que hay.
    imponer_parsers(
        delta,
        turno_paciente,
        # Qué acaba de preguntar el agente. Sin esto el parser exigía que el propio paciente
        # dijera «dolor» o «temperatura» en su respuesta, y a «¿en cuánto estaría el dolor?»
        # la gente contesta «un ocho» a secas.
        dimension_preguntada=pregunta_pendiente.dimension if pregunta_pendiente else None,
    )
    fusionar_extraccion(estado_clinico, delta, turno_idx)

    # --- Lo lento: la clasificación categórica ------------------------------------
    #
    # Con `diferir_extraccion`, esta llamada al modelo NO bloquea la respuesta. El agente
    # hace ya la siguiente pregunta del guion —que no depende de la anterior, porque el
    # orden es fijo— y el resultado se anota cuando llegue. El paciente pasa de esperar
    # ~6 s en silencio a oír la pregunta siguiente de inmediato.
    #
    # Es seguro porque lo que se difiere es *sólo* la clasificación de apetito, sueño,
    # herida y movilidad. Si el paciente dijo algo alarmante, la vía refleja ya lo vio
    # arriba, sin esperar a nadie.
    completar = None
    if pregunta_pendiente is not None and pregunta_pendiente.valores is not None:
        def completar(_pregunta=pregunta_pendiente, _texto=turno_paciente, _idx=turno_idx) -> str:
            """Clasifica la respuesta y la anota. Devuelve la dimensión, siempre."""
            valor = extraer_slot(pregunta=_pregunta, respuesta_paciente=_texto)
            tardio = ExtraccionTurno()
            setattr(
                tardio.dimensiones,
                _pregunta.dimension,
                Observacion(
                    valor=valor,
                    # Sin valor el slot NO queda en `no_preguntado`: se preguntó y no hubo
                    # respuesta útil. La distinción es la que permite reintentar y, si
                    # vuelve a fallar, escalar por incertidumbre en vez de asumir
                    # normalidad.
                    estado=EstadoSlot.CONFIRMADO if valor else EstadoSlot.PREGUNTADO_SIN_RESPUESTA,
                    verbatim=_texto[:200],
                    confianza=Confianza.ALTA,
                ),
            )
            fusionar_extraccion(estado_clinico, tardio, _idx)
            return _pregunta.dimension

        # Qué dimensión anotará, legible sin ejecutarla: el llamador la necesita para
        # marcarla como «en vuelo» y no volver a preguntarla mientras tanto.
        completar.dimension = pregunta_pendiente.dimension

        if not diferir_extraccion:
            completar()
            completar = None
        else:
            # La dimensión que se acaba de preguntar entra en vuelo AQUÍ, no cuando el
            # llamador reciba el resultado: `siguiente_pregunta` corre unas líneas más
            # abajo y sin esto la vería sin confirmar y volvería a preguntar lo mismo.
            # Observado en la primera llamada con extracción diferida: el agente pedía dos
            # veces seguidas que le describieran la herida.
            en_vuelo = set(en_vuelo or ()) | {pregunta_pendiente.dimension}

    # --- Decidir ------------------------------------------------------------------
    decision = evaluar_triaje(
        desde_estado(estado_clinico, banderas_procedimiento=tuple(estado_clinico.banderas.banderas_procedimiento)),
        perfil=get_settings().triage_profile,
        bandera_reflejo=bandera_reflejo,
        criticidad_llm=None,  # en el camino guiado el modelo no opina sobre criticidad
        exigir_cobertura_para_verde=True,
    )

    # --- Componer lo que se dice --------------------------------------------------
    siguiente = siguiente_pregunta(estado_clinico, en_vuelo)
    partes: list[str] = []
    afirmaciones = []

    if _es_pregunta(turno_paciente):
        # El paciente preguntó algo. Se responde con material del corpus, verificado antes
        # de pronunciarse. Es la única invocación pesada que queda en el camino guiado, y
        # solo ocurre cuando el paciente realmente pregunta.
        try:
            texto_respuesta, afirmaciones = responder_pregunta_anclada(
                pregunta_paciente=turno_paciente, procedimiento=procedimiento
            )
            partes.append(texto_respuesta)
        except Exception as exc:  # noqa: BLE001 — sin respuesta al paciente, se sigue el guion
            logger.warning("respuesta anclada falló en turno %s: %s", turno_idx, exc)
            partes.append(FALLBACK_PREGUNTA)

    if siguiente is not None:
        partes.append(texto_a_decir(siguiente, estado_clinico))
    else:
        partes.append(cierre_para(decision.nivel))

    resultado = ResultadoTurno(
        respuesta_hablada=" ".join(partes),
        criticidad_final=decision.nivel,
        criticidad_reflejo=bandera_reflejo.criticidad_forzada,
        criticidad_cortical=Criticidad.VERDE,
        reflejo_vetea=bandera_reflejo.disparada,
        afirmaciones_clinicas=afirmaciones,
        cobertura=estado_clinico.cobertura,
        verde_bloqueado_por_cobertura=any(m.regla == "cobertura_incompleta" for m in decision.motivos),
        sbar=_sbar_si_corresponde(
            decision, estado_clinico, paciente_id, procedimiento, dia_postop, bandera_reflejo, afirmaciones
        ),
        decision_triaje=decision.to_dict(),
    )
    return resultado, siguiente, completar


# Una pregunta del paciente se reconoce por la interrogación o por las fórmulas con que se
# pregunta hablando, que a menudo no llevan signo en una transcripción.
_MARCA_PREGUNTA = re.compile(r"[?¿]|es normal|sera normal|es grave|que hago|debo |puedo |usted cree|sabe si")

FALLBACK_PREGUNTA = (
    "Eso prefiero que se lo confirme alguien del equipo médico; lo dejo anotado para que lo revisen."
)


def _es_pregunta(texto: str) -> bool:
    return bool(_MARCA_PREGUNTA.search(texto.lower()))


def cierre_para(nivel: Criticidad) -> str:
    return {
        Criticidad.ROJO: CIERRE_ROJO,
        Criticidad.AMARILLO: CIERRE_AMARILLO,
        Criticidad.DESCONOCIDA: CIERRE_AMARILLO,
    }.get(nivel, CIERRE_VERDE)


def _sbar_si_corresponde(
    decision, estado, paciente_id, procedimiento, dia_postop, bandera_reflejo, afirmaciones
):
    if decision.nivel not in (Criticidad.AMARILLO, Criticidad.ROJO, Criticidad.DESCONOCIDA):
        return None
    motivos = [m for m in decision.motivos if m.nivel == decision.nivel]
    return construir_sbar(
        paciente_nombre=paciente_id,
        procedimiento=procedimiento,
        dia_postop=dia_postop,
        motivo_alerta="; ".join(m.detalle for m in motivos) or decision.nivel.value,
        contexto_previo=None,
        comorbilidades=_comorbilidades_de(paciente_id),
        sintomas_reportados={k: str(v) for k, v in a_dict_trajectory_twin(estado).items()},
        criticidad_final=decision.nivel,
        bandera_reflejo=bandera_reflejo,
        desviaciones=[],
        afirmaciones_clinicas=afirmaciones,
        accion_comunicada=_ACCIONES_POR_CRITICIDAD.get(decision.nivel, ""),
    )


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
    ya_confirmado_previo = _texto_confirmado(estado_clinico)
    sintomas_previos = a_dict_trajectory_twin(estado_clinico)
    _, desviaciones_texto_previas = _desviaciones_relevantes(
        paciente_id, procedimiento, dia_postop, sintomas_previos, turno_idx
    )

    # Llamada A (extracción) y B (conversación) en paralelo — independientes entre sí
    # dentro del mismo turno; correrlas una detrás de la otra solo suma latencia.
    # `copy_context().run` para que el cronómetro del turno (un contextvar, ver
    # orchestrator/metrics.py) siga vivo dentro de los hilos: ThreadPoolExecutor no
    # propaga el contexto por su cuenta, y sin esto las etapas `rag` y `llm_*` se medirían
    # contra una medición inexistente y se perderían.
    # Una copia por hilo: un mismo objeto Context no puede entrarse dos veces a la vez.
    tarea_extraccion = partial(extraer_turno, turno_paciente=turno_paciente, historial_turno=historial_turno)
    tarea_conversacion = partial(
        generar_respuesta,
        turno_paciente=turno_paciente,
        procedimiento=procedimiento,
        contexto_memoria=contexto_memoria,
        desviaciones_trayectoria=desviaciones_texto_previas,
        historial_turno=historial_turno,
        dimensiones_pendientes=dimensiones_pendientes_previas,
        reflejo_disparado=bandera_reflejo.disparada,
        ya_confirmado=ya_confirmado_previo,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futuro_extraccion = pool.submit(contextvars.copy_context().run, tarea_extraccion)
        futuro_conversacion = pool.submit(contextvars.copy_context().run, tarea_conversacion)

        try:
            delta = a_extraccion_turno(futuro_extraccion.result())
        except Exception as exc:  # noqa: BLE001 — una extracción fallida no debe tumbar el turno
            logger.warning("extracción de síntomas falló en turno %s: %s", turno_idx, exc)
            delta = ExtraccionTurno()

        # Dolor y fiebre no los pone el modelo: los lee un parser determinista
        # (clinical/parsers.py). Son dos de las tres reglas que disparan ROJO, y dejarlas
        # en manos de un extractor que ante «no me dan ganas de comer nada» respondía
        # `levemente_disminuido` era apoyar la decisión más grave del sistema en su pieza
        # menos verificable. El parser se mide contra los 3.991 turnos del dataset sin
        # gastar un token: `python -m harness.eval_parsers`.
        imponer_parsers(delta, turno_paciente)
        fusionar_extraccion(estado_clinico, delta, turno_idx)

        respuesta_cortex = futuro_conversacion.result()  # si falla (ej. rate limit), se propaga — el llamador decide

    sintomas_extraidos = a_dict_trajectory_twin(estado_clinico)
    desviaciones, _ = _desviaciones_relevantes(paciente_id, procedimiento, dia_postop, sintomas_extraidos, turno_idx)

    # La criticidad la decide el motor determinista (clinical/triage.py), no el modelo.
    # Antes era `max(criticidad_propuesta_por_el_LLM, reflejo)`: cuando el reflejo no
    # disparaba —un tercio de los casos rojo— escalar o no quedaba en manos del modelo, y
    # esa decisión no se podía ni auditar ni evaluar sin gastar tokens. Ahora la propuesta
    # del LLM entra como capa 8, con un solo poder: subir el nivel, nunca bajarlo.
    decision = evaluar_triaje(
        desde_estado(estado_clinico, banderas_procedimiento=tuple(estado_clinico.banderas.banderas_procedimiento)),
        perfil=get_settings().triage_profile,
        bandera_reflejo=bandera_reflejo,
        # Capa de trayectoria desactivada: medida sobre los 160 casos no rescata ninguno y
        # solo añade falsos positivos (ver la tabla en clinical/triage.py). Las
        # desviaciones se siguen calculando porque alimentan el SBAR y el prompt de la
        # conversación, que es donde sí aportan.
        desviaciones=None,
        criticidad_llm=respuesta_cortex.criticidad_propuesta,
        exigir_cobertura_para_verde=True,
    )
    criticidad_final = decision.nivel

    # `reflejo_vetea` conserva su significado original —el reflejo vio algo que el modelo
    # no— porque es lo que el log de auditoría registra como desacuerdo entre vías.
    reflejo_vetea = (
        bandera_reflejo.disparada
        and bandera_reflejo.criticidad_forzada.rango > respuesta_cortex.criticidad_propuesta.rango
    )
    verde_bloqueado = any(m.regla == "cobertura_incompleta" for m in decision.motivos)

    es_valida, texto_final = validar_respuesta(respuesta_cortex)
    afirmaciones_finales = respuesta_cortex.afirmaciones_clinicas if es_valida else []

    sbar = None
    if criticidad_final in (Criticidad.AMARILLO, Criticidad.ROJO, Criticidad.DESCONOCIDA):
        # El motivo del SBAR ahora es la regla que efectivamente determinó el nivel, no una
        # reconstrucción a posteriori. Un clínico lee "temperatura 38.4 °C ≥ 38.0 °C" y
        # puede discutirla; "el modelo propuso rojo" no se puede ni confirmar ni refutar.
        motivos_del_nivel = [m for m in decision.motivos if m.nivel == criticidad_final]
        motivo = (
            "; ".join(m.detalle for m in motivos_del_nivel)
            if motivos_del_nivel
            else f"criticidad {criticidad_final.value} sin motivo registrado"
        )
        sbar = construir_sbar(
            paciente_nombre=paciente_id,
            procedimiento=procedimiento,
            dia_postop=dia_postop,
            motivo_alerta=motivo,
            contexto_previo=contexto_memoria,
            # Estaba fijo en None, así que todo SBAR decía "Comorbilidades: ninguna
            # registrada" — incluso para un paciente con obesidad e hipertensión, que es
            # justo el contexto que cambia la lectura de una alerta postoperatoria.
            comorbilidades=_comorbilidades_de(paciente_id),
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
        decision_triaje=decision.to_dict(),
    )
