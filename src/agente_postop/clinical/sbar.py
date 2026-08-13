"""Genera el SBAR de escalamiento — el estándar real de traspaso clínico entre
profesionales de salud. Se produce cuando la criticidad final es amarillo o rojo."""

from __future__ import annotations

from agente_postop.clinical.models import (
    AfirmacionClinica,
    BanderaRefleja,
    Criticidad,
    SBAR,
)
from agente_postop.clinical.trajectory_twin import CuadroEsperado, Desviacion


def construir_sbar(
    *,
    paciente_nombre: str,
    procedimiento: str,
    dia_postop: int,
    motivo_alerta: str,
    contexto_previo: str | None,
    comorbilidades: str | None,
    sintomas_reportados: dict[str, str],
    criticidad_final: Criticidad,
    bandera_reflejo: BanderaRefleja,
    desviaciones: list[Desviacion],
    afirmaciones_clinicas: list[AfirmacionClinica],
    accion_comunicada: str,
) -> SBAR:
    situacion = (
        f"{paciente_nombre}, procedimiento: {procedimiento}, día postoperatorio {dia_postop}. "
        f"Motivo de la alerta: {motivo_alerta}."
    )

    background_partes = [f"Comorbilidades: {comorbilidades or 'ninguna registrada'}."]
    if contexto_previo:
        background_partes.append(f"Llamada previa: {contexto_previo}")
    background = " ".join(background_partes)

    desviaciones_txt = "; ".join(
        f"{d.dimension}: reportado={d.reportado} vs. esperado={d.esperado}"
        for d in desviaciones
        if d.empeora
    ) or "sin desviaciones relevantes frente a la trayectoria esperada"

    referencias_txt = "; ".join(
        f"{a.texto} (fuente: {a.documento})" for a in afirmaciones_clinicas
    ) or "sin referencias documentales citadas en este turno"

    bandera_txt = (
        f"Bandera refleja disparada: {bandera_reflejo.regla}."
        if bandera_reflejo.disparada
        else "Sin bandera refleja disparada; criticidad determinada por la vía cortical."
    )

    # El dict se formatea a prosa en vez de interpolarse crudo: esto lo lee personal
    # clínico en un traspaso, y `{'dolor_nrs': '3', 'herida': 'eritema_leve'}` obliga a
    # leer sintaxis de Python en mitad de una alerta médica.
    _NOMBRE_SINTOMA = {
        "dolor_nrs": "dolor (NRS)",
        "fiebre_c": "fiebre (°C)",
        "movilidad": "movilidad",
        "herida": "herida",
        "apetito": "apetito",
        "sueno": "sueño",
    }
    sintomas_txt = ", ".join(
        f"{_NOMBRE_SINTOMA.get(clave, clave)}: {valor}" for clave, valor in sintomas_reportados.items()
    ) or "ninguno confirmado durante la llamada"

    evaluacion = (
        f"Síntomas reportados — {sintomas_txt}. Criticidad asignada: {criticidad_final.value}. "
        f"Desviación vs. trayectoria esperada: {desviaciones_txt}. {bandera_txt} "
        f"Referencias: {referencias_txt}"
    )

    recomendacion = accion_comunicada

    return SBAR(
        situacion=situacion,
        contexto=background,
        evaluacion=evaluacion,
        recomendacion=recomendacion,
    )
