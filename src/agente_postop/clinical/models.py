"""Esquemas de datos del motor clínico: criticidad, afirmaciones citadas, SBAR."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Criticidad(StrEnum):
    VERDE = "verde"
    AMARILLO = "amarillo"
    ROJO = "rojo"
    DESCONOCIDA = "desconocida"

    @property
    def rango(self) -> int:
        return {
            Criticidad.VERDE: 0,
            Criticidad.AMARILLO: 1,
            Criticidad.DESCONOCIDA: 1,
            Criticidad.ROJO: 2,
        }[self]


def criticidad_mas_severa(a: Criticidad, b: Criticidad) -> Criticidad:
    return a if a.rango >= b.rango else b


class AfirmacionClinica(BaseModel):
    """Una afirmación clínica del agente, amarrada a su fuente en el RAG.

    Si `chunk_id` no existe en ChromaDB, el validador bloquea la respuesta
    antes de que llegue al TTS (ver clinical/citation_validator.py).
    """

    texto: str
    chunk_id: str
    documento: str


class RespuestaEstructurada(BaseModel):
    """Salida obligatoria del LLM para cada turno — nunca texto libre sin amarrar."""

    respuesta_hablada: str
    afirmaciones_clinicas: list[AfirmacionClinica] = Field(default_factory=list)
    criticidad_propuesta: Criticidad
    confianza: str


class BanderaRefleja(BaseModel):
    """Resultado de la vía refleja determinística para un turno."""

    disparada: bool
    criticidad_forzada: Criticidad
    regla: str | None = None
    documento_sustento: str | None = None


class SBAR(BaseModel):
    situacion: str
    contexto: str
    evaluacion: str
    recomendacion: str


class ResultadoTurno(BaseModel):
    respuesta_hablada: str
    criticidad_final: Criticidad
    criticidad_reflejo: Criticidad
    criticidad_cortical: Criticidad
    reflejo_vetea: bool
    afirmaciones_clinicas: list[AfirmacionClinica]
    cobertura: float = 0.0
    verde_bloqueado_por_cobertura: bool = False
    sbar: SBAR | None = None
