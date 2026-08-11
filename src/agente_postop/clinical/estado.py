"""Estado clínico acumulado de una llamada — vive en la sesión (o en el harness), se
actualiza turno a turno fusionando el delta que extrae el LLM (`clinical/extraction.py`).

Principio rector (§2 del diseño): el LLM extrae deltas, el acumulador es código
determinista, no el modelo. Segundo principio (§4.2): el estado solo puede escalar en
severidad dentro de una llamada — un paciente que se retracta por cortesía no borra lo
que ya reportó, la retractación queda registrada, no se obedece.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agente_postop.clinical.extraction import (
    DIMENSIONES,
    BanderasRojas,
    Confianza,
    Contexto,
    DimensionesClinicas,
    EstadoSlot,
    ExtraccionTurno,
    Medicacion,
    MetadatosTurno,
    Observacion,
    Procedencia,
    TriEstado,
)
from agente_postop.clinical.trajectory_twin import ORDEN_APETITO, ORDEN_HERIDA, ORDEN_MOVILIDAD, ORDEN_SUENO

_ORDEN_ESTADO = {
    EstadoSlot.NO_PREGUNTADO: 0,
    EstadoSlot.PREGUNTADO_SIN_RESPUESTA: 1,
    EstadoSlot.AMBIGUO: 1,
    EstadoSlot.RECHAZADO: 1,
    EstadoSlot.NO_MEDIBLE: 1,
    EstadoSlot.CONFIRMADO: 2,
}

_TABLAS_ORDEN = {
    "movilidad": ORDEN_MOVILIDAD,
    "herida": ORDEN_HERIDA,
    "apetito": ORDEN_APETITO,
    "sueno": ORDEN_SUENO,
}


class CorreccionRegistrada(BaseModel):
    """Un intento de retractación — se registra, no se obedece (§4.2, regla 5)."""

    turno_idx: int
    dimension: str
    valor_conservado: str
    valor_descartado: str


class EstadoClinicoLlamada(BaseModel):
    dolor: Observacion = Field(default_factory=Observacion)
    fiebre: Observacion = Field(default_factory=Observacion)
    movilidad: Observacion = Field(default_factory=Observacion)
    herida: Observacion = Field(default_factory=Observacion)
    apetito: Observacion = Field(default_factory=Observacion)
    sueno: Observacion = Field(default_factory=Observacion)
    banderas: BanderasRojas = Field(default_factory=BanderasRojas)
    medicacion: Medicacion = Field(default_factory=Medicacion)
    contexto: Contexto = Field(default_factory=Contexto)
    metadatos: MetadatosTurno = Field(default_factory=MetadatosTurno)
    intentos: dict[str, int] = Field(default_factory=dict)
    correcciones: list[CorreccionRegistrada] = Field(default_factory=list)
    turno_actual: int = 0

    @property
    def dimensiones_pendientes(self) -> list[str]:
        return [d for d in DIMENSIONES if getattr(self, d).estado == EstadoSlot.NO_PREGUNTADO]

    @property
    def dimensiones_confirmadas(self) -> list[str]:
        return [d for d in DIMENSIONES if getattr(self, d).confirmada]

    @property
    def cobertura(self) -> float:
        return len(self.dimensiones_confirmadas) / len(DIMENSIONES)

    @property
    def alguna_dimension_de_tercero_sin_confirmar_por_paciente(self) -> bool:
        return any(
            getattr(self, d).confirmada and getattr(self, d).procedencia != Procedencia.PACIENTE
            for d in DIMENSIONES
        )

    @property
    def puede_cerrar_verde(self) -> bool:
        """Traducción a código de "verde solo con evidencia positiva, nunca por defecto"
        (prompts.py) — si es False, la fusión degrada un verde propuesto a desconocida."""
        return (
            self.cobertura == 1.0
            and not self.banderas.alguna_presente
            and not self.alguna_dimension_de_tercero_sin_confirmar_por_paciente
        )

    def slot_bloqueado(self, dimension: str) -> bool:
        return self.intentos.get(dimension, 0) >= 2 and not getattr(self, dimension).confirmada


def _severidad(dimension: str, valor: object) -> float:
    if valor is None:
        return -1.0
    if dimension in ("dolor", "fiebre"):
        return float(valor)
    return float(_TABLAS_ORDEN[dimension][valor])


def _fusionar_dimension(
    dimension: str, actual: Observacion, nueva: Observacion, turno_idx: int, correcciones: list[CorreccionRegistrada]
) -> Observacion:
    if nueva.estado == EstadoSlot.NO_PREGUNTADO:
        return actual  # el delta no dice nada de esta dimensión — regla 1 de §4.2

    candidata = nueva.model_copy(update={"turno_idx": turno_idx})

    if candidata.confianza == Confianza.BAJA and candidata.estado == EstadoSlot.CONFIRMADO:
        candidata = candidata.model_copy(update={"estado": EstadoSlot.AMBIGUO})  # regla 2

    if (
        candidata.procedencia == Procedencia.TERCERO
        and actual.estado == EstadoSlot.CONFIRMADO
        and actual.procedencia == Procedencia.PACIENTE
    ):
        return actual  # regla 3 — un tercero no pisa lo ya confirmado por el paciente

    if actual.estado != EstadoSlot.CONFIRMADO and candidata.estado != EstadoSlot.CONFIRMADO:
        return candidata if _ORDEN_ESTADO[candidata.estado] >= _ORDEN_ESTADO[actual.estado] else actual

    if actual.estado == EstadoSlot.CONFIRMADO and candidata.estado != EstadoSlot.CONFIRMADO:
        return actual  # ya sabíamos algo concreto; una respuesta ambigua después no lo borra

    if actual.estado != EstadoSlot.CONFIRMADO and candidata.estado == EstadoSlot.CONFIRMADO:
        return candidata

    # ambos confirmados con valor — regla 4/5: se queda el máximo, se registra la corrección
    severidad_actual = _severidad(dimension, actual.valor)
    severidad_nueva = _severidad(dimension, candidata.valor)
    if severidad_nueva >= severidad_actual:
        return candidata
    correcciones.append(
        CorreccionRegistrada(
            turno_idx=turno_idx,
            dimension=dimension,
            valor_conservado=str(actual.valor),
            valor_descartado=str(candidata.valor),
        )
    )
    return actual


def _fusionar_tri(actual: Observacion, nueva: Observacion, turno_idx: int) -> Observacion:
    if nueva.estado == EstadoSlot.NO_PREGUNTADO or nueva.valor is None:
        return actual
    if actual.valor == TriEstado.PRESENTE:
        return actual  # una bandera roja ya vista no se retracta
    return nueva.model_copy(update={"turno_idx": turno_idx})


def fusionar_extraccion(estado: EstadoClinicoLlamada, delta: ExtraccionTurno, turno_idx: int) -> None:
    """Fusiona el delta de este turno en el estado vivo de la llamada — muta `estado` en
    el sitio, siguiendo la regla de fusión de §4.2."""
    estado.turno_actual = turno_idx

    for nombre in DIMENSIONES:
        nueva_obs = getattr(delta.dimensiones, nombre)
        if nueva_obs.estado != EstadoSlot.NO_PREGUNTADO:
            estado.intentos[nombre] = estado.intentos.get(nombre, 0) + 1  # regla 6
        actual_obs = getattr(estado, nombre)
        setattr(estado, nombre, _fusionar_dimension(nombre, actual_obs, nueva_obs, turno_idx, estado.correcciones))

    for nombre in delta.banderas.__class__.model_fields:
        if nombre == "banderas_procedimiento":
            continue
        nueva_obs = getattr(delta.banderas, nombre)
        actual_obs = getattr(estado.banderas, nombre)
        setattr(estado.banderas, nombre, _fusionar_tri(actual_obs, nueva_obs, turno_idx))

    if delta.banderas.banderas_procedimiento:
        vistas = set(estado.banderas.banderas_procedimiento) | set(delta.banderas.banderas_procedimiento)
        estado.banderas.banderas_procedimiento = sorted(vistas)

    if delta.medicacion.toma_analgesico.estado != EstadoSlot.NO_PREGUNTADO:
        estado.medicacion.toma_analgesico = _fusionar_tri(
            estado.medicacion.toma_analgesico, delta.medicacion.toma_analgesico, turno_idx
        )
    if delta.medicacion.adherencia.estado == EstadoSlot.CONFIRMADO:
        estado.medicacion.adherencia = delta.medicacion.adherencia.model_copy(update={"turno_idx": turno_idx})
    if delta.medicacion.motivo_no_adherencia:
        estado.medicacion.motivo_no_adherencia = delta.medicacion.motivo_no_adherencia

    if delta.contexto.acompanado.estado != EstadoSlot.NO_PREGUNTADO:
        estado.contexto.acompanado = _fusionar_tri(estado.contexto.acompanado, delta.contexto.acompanado, turno_idx)
    if delta.contexto.transporte_disponible.estado != EstadoSlot.NO_PREGUNTADO:
        estado.contexto.transporte_disponible = _fusionar_tri(
            estado.contexto.transporte_disponible, delta.contexto.transporte_disponible, turno_idx
        )

    if delta.metadatos.estilo_paciente_detectado is not None:
        estado.metadatos.estilo_paciente_detectado = delta.metadatos.estilo_paciente_detectado
    if delta.metadatos.intervencion_tercero:
        estado.metadatos.intervencion_tercero = True
    estado.metadatos.calidad_transcripcion = delta.metadatos.calidad_transcripcion


def comparar_estados(anterior: EstadoClinicoLlamada, actual: EstadoClinicoLlamada) -> list[dict]:
    """Delta entre el estado final de la llamada anterior y el de esta — la señal
    longitudinal real que `contexto_apertura()` usa en la memoria (§5 del diseño), no una
    comparación contra la trayectoria esperada del procedimiento (eso es trajectory_twin)."""
    delta = []
    for nombre in DIMENSIONES:
        obs_anterior = getattr(anterior, nombre)
        obs_actual = getattr(actual, nombre)
        if not obs_anterior.confirmada or not obs_actual.confirmada:
            continue
        sev_anterior = _severidad(nombre, obs_anterior.valor)
        sev_actual = _severidad(nombre, obs_actual.valor)
        if sev_actual == sev_anterior:
            continue
        delta.append(
            {
                "dimension": nombre,
                "valor_anterior": str(obs_anterior.valor),
                "valor_actual": str(obs_actual.valor),
                "empeora": sev_actual > sev_anterior,
            }
        )
    return delta


def a_dict_trajectory_twin(estado: EstadoClinicoLlamada) -> dict:
    """Traduce el estado acumulado al `dict` que espera `trajectory_twin.comparar()` —
    solo dimensiones confirmadas, nunca un valor inventado para rellenar."""
    salida: dict[str, object] = {}
    if estado.dolor.confirmada:
        salida["dolor_nrs"] = estado.dolor.valor
    if estado.fiebre.confirmada:
        salida["fiebre_c"] = estado.fiebre.valor
    for nombre in ("movilidad", "herida", "apetito", "sueno"):
        obs = getattr(estado, nombre)
        if obs.confirmada:
            salida[nombre] = obs.valor
    return salida
