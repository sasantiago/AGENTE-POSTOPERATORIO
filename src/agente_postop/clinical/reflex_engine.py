"""Vía refleja: detección determinística de banderas rojas (~5ms, sin LLM).

Corre en paralelo a la vía cortical (LLM + RAG) sobre cada turno del paciente. Solo puede
subir la criticidad, nunca bajarla — el nivel final es max(reflejo, cortical) y se calcula
en fusion.py.
"""

from __future__ import annotations

from agente_postop.clinical.models import BanderaRefleja, Criticidad
from agente_postop.clinical.reflex_rules import (
    FIEBRE_UMBRAL_C,
    REGLAS_COMUNES,
    REGLAS_POR_PROCEDIMIENTO,
    esta_negado,
    extraer_temperatura_c,
    normalizar,
)


def evaluar_via_refleja(texto_paciente: str, procedimiento: str) -> BanderaRefleja:
    reglas = REGLAS_COMUNES + REGLAS_POR_PROCEDIMIENTO.get(procedimiento, [])
    texto_normalizado = normalizar(texto_paciente)

    for regla in reglas:
        for match in regla.patron.finditer(texto_normalizado):
            # `finditer` y no `search`: si la primera aparición del término está negada
            # («no le sale nada de pus, pero ayer sí salió pus»), hay que seguir mirando
            # el resto del turno en vez de descartar la regla entera.
            if esta_negado(texto_normalizado, match.start()):
                continue
            return BanderaRefleja(
                disparada=True,
                criticidad_forzada=Criticidad.ROJO,
                regla=regla.descripcion,
                documento_sustento=regla.documento_sustento,
            )

    temperatura = extraer_temperatura_c(texto_paciente)
    if temperatura is not None and temperatura >= FIEBRE_UMBRAL_C:
        return BanderaRefleja(
            disparada=True,
            criticidad_forzada=Criticidad.ROJO,
            regla=f"Fiebre >= {FIEBRE_UMBRAL_C}°C reportada ({temperatura}°C)",
            documento_sustento=None,
        )

    return BanderaRefleja(disparada=False, criticidad_forzada=Criticidad.VERDE)
