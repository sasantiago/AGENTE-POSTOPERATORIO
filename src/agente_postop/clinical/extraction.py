"""Esquema de extracción clínica — lo que el LLM extrae de cada turno del paciente.

Ver docs/diseno-esquema-extraccion.md para el diseño completo. Principio rector: el LLM
extrae DELTAS (solo lo nuevo de este turno), nunca el estado acumulado — pedirle que
reproduzca lo que ya sabe invita a que lo altere. El acumulador determinista vive en
`clinical/estado.py`.

Simplificación deliberada frente al diseño original: `procedencia` y `confianza` se piden
una vez por turno (`hablante_detectado`, `confianza_general`), no repetidas en cada una de
las ~15 dimensiones — mismo efecto clínico (design §6.1 punto 4: "si el turno viene de un
tercero, procedencia=TERCERO en todo lo extraído de ese turno"), JSON bastante más chico
para que el LLM lo llene bien en una sola pasada. Los campos auxiliares de §3.2
(dolor.tendencia, dolor.localizacion_cambio, fiebre.medida, fiebre.sensacion_termica) se
dejan fuera de esta entrega para no inflar más el esquema — quedan documentados como
trabajo pendiente, no implementados a medias.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class EstadoSlot(StrEnum):
    """Estado epistémico de una dimensión — el corazón del diseño (§3.1). Solo
    CONFIRMADO con valor no nulo puede sustentar un verde; todo lo demás degrada, como
    mínimo, a `desconocida` (que en Criticidad.rango ya pesa igual que amarillo)."""

    NO_PREGUNTADO = "no_preguntado"
    PREGUNTADO_SIN_RESPUESTA = "preguntado_sin_respuesta"
    AMBIGUO = "ambiguo"
    RECHAZADO = "rechazado"
    NO_MEDIBLE = "no_medible"
    CONFIRMADO = "confirmado"


class TriEstado(StrEnum):
    """Nunca `bool` para banderas rojas: un `False` que significa 'no me consta' y un
    `False` que significa 'el paciente lo negó explícitamente' no pueden compartir
    representación en un sistema clínico."""

    PRESENTE = "presente"
    AUSENTE = "ausente"
    NO_EVALUADO = "no_evaluado"


class Procedencia(StrEnum):
    PACIENTE = "paciente"
    TERCERO = "tercero"
    AGENTE_INFERIDO = "agente_inferido"


class Confianza(StrEnum):
    ALTA = "alta"
    MEDIA = "media"
    BAJA = "baja"


# Enums idénticos, carácter por carácter, a dataset/trayectorias_postop_silver.xlsx —
# trajectory_twin.ORDEN_* indexa por estos literales exactos (§3.2 del diseño).
class Movilidad(StrEnum):
    NORMAL = "normal"
    LIMITADA_ESPERADA = "limitada_esperada"
    INCAPACITANTE_NUEVA = "incapacitante_nueva"


class Herida(StrEnum):
    NORMAL = "normal"
    ERITEMA_LEVE = "eritema_leve"
    SECRECION_PURULENTA = "secrecion_purulenta"


class Apetito(StrEnum):
    NORMAL = "normal"
    LEVEMENTE_DISMINUIDO = "levemente_disminuido"
    MUY_DISMINUIDO = "muy_disminuido"


class Sueno(StrEnum):
    NORMAL = "normal"
    LEVEMENTE_ALTERADO = "levemente_alterado"
    MUY_ALTERADO = "muy_alterado"


class EstiloPaciente(StrEnum):
    COLABORATIVO = "colaborativo"
    EVASIVO = "evasivo"
    MINIMIZADOR_SINTOMAS = "minimizador_sintomas"
    ANSIOSO = "ansioso"
    CONFUNDIDO = "confundido"


class Adherencia(StrEnum):
    COMPLETA = "completa"
    PARCIAL = "parcial"
    ABANDONADA = "abandonada"


class CalidadTranscripcion(StrEnum):
    BUENA = "buena"
    DEGRADADA = "degradada"
    ININTELIGIBLE = "ininteligible"


DIMENSIONES = ("dolor", "fiebre", "movilidad", "herida", "apetito", "sueno")
BANDERAS = (
    "sangrado_activo",
    "dificultad_respiratoria",
    "rigidez_abdominal",
    "secrecion_anormal",
    "dolor_extremo",
    "alteracion_conciencia",
)

TIPOS_DIMENSION: dict[str, type] = {
    "dolor": int,
    "fiebre": float,
    "movilidad": Movilidad,
    "herida": Herida,
    "apetito": Apetito,
    "sueno": Sueno,
}


class Observacion(BaseModel, Generic[T]):
    """Una dimensión clínica con procedencia — nunca un valor plano (§3.1 del diseño).

    Sin este `estado` epistémico, `None` no distingue "no le pregunté" de "me evadió" —
    y esa distinción es exactamente la regla del verde del prompt de conversación.
    """

    valor: T | None = None
    estado: EstadoSlot = EstadoSlot.NO_PREGUNTADO
    verbatim: str | None = None
    turno_idx: int | None = None
    procedencia: Procedencia = Procedencia.PACIENTE
    confianza: Confianza = Confianza.MEDIA

    @property
    def confirmada(self) -> bool:
        return self.estado == EstadoSlot.CONFIRMADO and self.valor is not None


def observacion_vacia() -> Observacion:
    return Observacion()


class DimensionesClinicas(BaseModel):
    dolor: Observacion = Field(default_factory=observacion_vacia)
    fiebre: Observacion = Field(default_factory=observacion_vacia)
    movilidad: Observacion = Field(default_factory=observacion_vacia)
    herida: Observacion = Field(default_factory=observacion_vacia)
    apetito: Observacion = Field(default_factory=observacion_vacia)
    sueno: Observacion = Field(default_factory=observacion_vacia)


class BanderasRojas(BaseModel):
    sangrado_activo: Observacion = Field(default_factory=observacion_vacia)
    dificultad_respiratoria: Observacion = Field(default_factory=observacion_vacia)
    rigidez_abdominal: Observacion = Field(default_factory=observacion_vacia)
    secrecion_anormal: Observacion = Field(default_factory=observacion_vacia)
    dolor_extremo: Observacion = Field(default_factory=observacion_vacia)
    alteracion_conciencia: Observacion = Field(default_factory=observacion_vacia)
    banderas_procedimiento: list[str] = Field(default_factory=list)

    @property
    def alguna_presente(self) -> bool:
        return any(getattr(self, b).valor == TriEstado.PRESENTE for b in BANDERAS)


class Medicacion(BaseModel):
    toma_analgesico: Observacion = Field(default_factory=observacion_vacia)
    adherencia: Observacion = Field(default_factory=observacion_vacia)
    motivo_no_adherencia: str | None = None


class Contexto(BaseModel):
    acompanado: Observacion = Field(default_factory=observacion_vacia)
    transporte_disponible: Observacion = Field(default_factory=observacion_vacia)


class MetadatosTurno(BaseModel):
    estilo_paciente_detectado: EstiloPaciente | None = None
    intervencion_tercero: bool = False
    calidad_transcripcion: CalidadTranscripcion = CalidadTranscripcion.BUENA


class ExtraccionTurno(BaseModel):
    """Delta ya tipado de un turno — lo que produce `a_extraccion_turno()` a partir de lo
    que el LLM devolvió crudo. Esto es lo que `clinical/estado.py` fusiona al estado vivo
    de la llamada."""

    hablante_detectado: Procedencia = Procedencia.PACIENTE
    dimensiones: DimensionesClinicas = Field(default_factory=DimensionesClinicas)
    banderas: BanderasRojas = Field(default_factory=BanderasRojas)
    medicacion: Medicacion = Field(default_factory=Medicacion)
    contexto: Contexto = Field(default_factory=Contexto)
    metadatos: MetadatosTurno = Field(default_factory=MetadatosTurno)


# ---------------------------------------------------------------------------
# Esquema crudo que efectivamente rellena el LLM (llamada A) — más chico que
# ExtraccionTurno porque procedencia/confianza se piden una sola vez por turno.
# ---------------------------------------------------------------------------


class ValorDimensionCrudo(BaseModel):
    estado: EstadoSlot | None = EstadoSlot.NO_PREGUNTADO
    valor: str | float | int | None = None
    verbatim: str | None = None


class DimensionesCrudas(BaseModel):
    dolor: ValorDimensionCrudo = Field(default_factory=ValorDimensionCrudo)
    fiebre: ValorDimensionCrudo = Field(default_factory=ValorDimensionCrudo)
    movilidad: ValorDimensionCrudo = Field(default_factory=ValorDimensionCrudo)
    herida: ValorDimensionCrudo = Field(default_factory=ValorDimensionCrudo)
    apetito: ValorDimensionCrudo = Field(default_factory=ValorDimensionCrudo)
    sueno: ValorDimensionCrudo = Field(default_factory=ValorDimensionCrudo)


def _o_no_evaluado(v: TriEstado | None) -> TriEstado:
    """El LLM a veces manda `null` explícito en vez de omitir el campo — Pydantic solo
    aplica el default cuando el campo está AUSENTE, no cuando llega null explícito. Estos
    validadores normalizan ambos casos al mismo resultado seguro."""
    return v if v is not None else TriEstado.NO_EVALUADO


class BanderasCrudas(BaseModel):
    sangrado_activo: TriEstado | None = TriEstado.NO_EVALUADO
    dificultad_respiratoria: TriEstado | None = TriEstado.NO_EVALUADO
    rigidez_abdominal: TriEstado | None = TriEstado.NO_EVALUADO
    secrecion_anormal: TriEstado | None = TriEstado.NO_EVALUADO
    dolor_extremo: TriEstado | None = TriEstado.NO_EVALUADO
    alteracion_conciencia: TriEstado | None = TriEstado.NO_EVALUADO
    banderas_procedimiento: list[str] | None = Field(default_factory=list)


class MedicacionCruda(BaseModel):
    toma_analgesico: TriEstado | None = TriEstado.NO_EVALUADO
    adherencia: Adherencia | None = None
    motivo_no_adherencia: str | None = None


class ContextoCrudo(BaseModel):
    acompanado: TriEstado | None = TriEstado.NO_EVALUADO
    transporte_disponible: TriEstado | None = TriEstado.NO_EVALUADO


class ExtraccionCruda(BaseModel):
    """Forma exacta que el LLM debe devolver en la llamada A — ver
    `orchestrator/prompts.py:SYSTEM_PROMPT_EXTRACCION`."""

    hablante_detectado: Procedencia = Procedencia.PACIENTE
    confianza_general: Confianza = Confianza.MEDIA
    dimensiones: DimensionesCrudas = Field(default_factory=DimensionesCrudas)
    banderas: BanderasCrudas = Field(default_factory=BanderasCrudas)
    medicacion: MedicacionCruda = Field(default_factory=MedicacionCruda)
    contexto: ContextoCrudo = Field(default_factory=ContextoCrudo)
    estilo_paciente_detectado: EstiloPaciente | None = None
    calidad_transcripcion: CalidadTranscripcion | None = CalidadTranscripcion.BUENA


def _coaccionar_valor(dimension: str, valor: object) -> tuple[object | None, bool]:
    """Convierte el valor crudo del LLM al tipo de la dimensión. Si no valida, devuelve
    (None, False) — nunca degrada silenciosamente a un valor "normal" por defecto (§7 del
    plan: fallar ruidoso, no `.get(x, 0)`)."""
    if valor is None or valor == "":
        return None, False
    tipo = TIPOS_DIMENSION[dimension]
    try:
        return tipo(valor), True
    except (ValueError, TypeError):
        return None, False


def _observacion_bandera(valor: TriEstado, procedencia: Procedencia, confianza: Confianza) -> Observacion:
    estado = EstadoSlot.NO_PREGUNTADO if valor == TriEstado.NO_EVALUADO else EstadoSlot.CONFIRMADO
    return Observacion(valor=valor, estado=estado, procedencia=procedencia, confianza=confianza)


def a_extraccion_turno(cruda: ExtraccionCruda) -> ExtraccionTurno:
    """Construye el delta tipado (con Observacion completa) a partir de lo que devolvió
    el LLM. Aplica la atribución de hablante (§6.1 punto 4) a todo lo extraído del turno."""
    procedencia = cruda.hablante_detectado
    confianza = cruda.confianza_general

    dimensiones_kwargs = {}
    for nombre in DIMENSIONES:
        crudo = getattr(cruda.dimensiones, nombre)
        valor, valido = _coaccionar_valor(nombre, crudo.valor)
        estado = crudo.estado if crudo.estado is not None else EstadoSlot.NO_PREGUNTADO
        if estado == EstadoSlot.CONFIRMADO and not valido:
            # el LLM dijo "confirmado" pero el valor no es parseable — no se le cree
            estado = EstadoSlot.AMBIGUO
        dimensiones_kwargs[nombre] = Observacion(
            valor=valor,
            estado=estado,
            verbatim=crudo.verbatim,
            procedencia=procedencia,
            confianza=confianza,
        )

    banderas = BanderasRojas(
        sangrado_activo=_observacion_bandera(_o_no_evaluado(cruda.banderas.sangrado_activo), procedencia, confianza),
        dificultad_respiratoria=_observacion_bandera(
            _o_no_evaluado(cruda.banderas.dificultad_respiratoria), procedencia, confianza
        ),
        rigidez_abdominal=_observacion_bandera(_o_no_evaluado(cruda.banderas.rigidez_abdominal), procedencia, confianza),
        secrecion_anormal=_observacion_bandera(_o_no_evaluado(cruda.banderas.secrecion_anormal), procedencia, confianza),
        dolor_extremo=_observacion_bandera(_o_no_evaluado(cruda.banderas.dolor_extremo), procedencia, confianza),
        alteracion_conciencia=_observacion_bandera(
            _o_no_evaluado(cruda.banderas.alteracion_conciencia), procedencia, confianza
        ),
        banderas_procedimiento=cruda.banderas.banderas_procedimiento or [],
    )

    medicacion = Medicacion(
        toma_analgesico=_observacion_bandera(_o_no_evaluado(cruda.medicacion.toma_analgesico), procedencia, confianza),
        adherencia=Observacion(
            valor=cruda.medicacion.adherencia,
            estado=EstadoSlot.CONFIRMADO if cruda.medicacion.adherencia else EstadoSlot.NO_PREGUNTADO,
            procedencia=procedencia,
            confianza=confianza,
        ),
        motivo_no_adherencia=cruda.medicacion.motivo_no_adherencia,
    )

    contexto = Contexto(
        acompanado=_observacion_bandera(_o_no_evaluado(cruda.contexto.acompanado), procedencia, confianza),
        transporte_disponible=_observacion_bandera(
            _o_no_evaluado(cruda.contexto.transporte_disponible), procedencia, confianza
        ),
    )

    metadatos = MetadatosTurno(
        estilo_paciente_detectado=cruda.estilo_paciente_detectado,
        intervencion_tercero=procedencia != Procedencia.PACIENTE,
        calidad_transcripcion=cruda.calidad_transcripcion or CalidadTranscripcion.BUENA,
    )

    return ExtraccionTurno(
        hablante_detectado=procedencia,
        dimensiones=DimensionesClinicas(**dimensiones_kwargs),
        banderas=banderas,
        medicacion=medicacion,
        contexto=contexto,
        metadatos=metadatos,
    )
