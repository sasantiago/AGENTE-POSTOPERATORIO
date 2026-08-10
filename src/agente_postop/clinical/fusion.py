"""Fusión de las dos vías: el reflejo tiene poder de veto sobre el LLM, nunca al revés."""

from __future__ import annotations

from agente_postop.clinical.models import BanderaRefleja, Criticidad, criticidad_mas_severa


def fusionar(criticidad_cortical: Criticidad, bandera_reflejo: BanderaRefleja) -> tuple[Criticidad, bool]:
    """Devuelve (criticidad_final, reflejo_vetea).

    reflejo_vetea=True cuando el reflejo forzó un nivel más severo que el propuesto por el
    LLM — ese desacuerdo debe quedar registrado en el log como evento auditable.
    """
    final = criticidad_mas_severa(criticidad_cortical, bandera_reflejo.criticidad_forzada)
    vetea = bandera_reflejo.disparada and final != criticidad_cortical
    return final, vetea
