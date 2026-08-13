"""Memoria longitudinal: cada llamada termina escribiendo su resumen estructurado;
la siguiente lo carga como contexto de apertura.

Persistencia simple en JSON por paciente — suficiente para 4 llamadas (días 1/3/7/14) y
transparente de inspeccionar durante el demo.

v2 (§5 del diseño): `sintomas_reportados: dict[str,str]` se reemplaza por
`estado_final: EstadoClinicoLlamada` — deja de perderse la procedencia, el verbatim y el
estado epistémico de cada dimensión al cerrar la llamada.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from agente_postop.clinical.estado import EstadoClinicoLlamada
from agente_postop.clinical.extraction import BANDERAS, Procedencia, TriEstado
from agente_postop.clinical.models import SBAR, Criticidad
from agente_postop.config import get_settings


class MotivoCierre(StrEnum):
    COMPLETADA = "completada"
    PACIENTE_COLGO = "paciente_colgo"
    ESCALADA_INMEDIATA = "escalada_inmediata"
    ABORTADA_CALIDAD = "abortada_calidad"
    ERROR_TECNICO = "error_tecnico"


class ResumenLlamada(BaseModel):
    paciente_id: str
    dia_postop: int
    fecha: str
    estado_final: EstadoClinicoLlamada
    criticidad_final: Criticidad
    alerta_generada: bool
    # Resumen clínico en Markdown (`construir_resumen_markdown`). Antes este campo guardaba
    # los últimos 6 turnos crudos concatenados: en una llamada más larga se perdía el
    # principio, y nadie lo leía — `contexto_apertura()` reconstruye desde el estado
    # estructurado. Era un campo que parecía un resumen y no lo era.
    resumen_texto: str
    procedimiento: str = ""
    transcripcion: list[str] = Field(default_factory=list)
    cobertura: float = 0.0
    dimensiones_no_evaluadas: list[str] = Field(default_factory=list)
    sbar: SBAR | None = None
    delta_vs_llamada_anterior: list[dict] = Field(default_factory=list)
    motivo_cierre: MotivoCierre = MotivoCierre.COMPLETADA


_ETIQUETA_DIMENSION = {
    "dolor": "Dolor (NRS 0-10)",
    "fiebre": "Fiebre (°C)",
    "movilidad": "Movilidad",
    "herida": "Herida",
    "apetito": "Apetito",
    "sueno": "Sueño",
}

_MOTIVO_LEGIBLE = {
    MotivoCierre.COMPLETADA: "La llamada se completó con normalidad.",
    MotivoCierre.PACIENTE_COLGO: "El paciente colgó antes de terminar el seguimiento.",
    MotivoCierre.ESCALADA_INMEDIATA: "Se escaló de inmediato a personal médico.",
    MotivoCierre.ABORTADA_CALIDAD: "Se abortó por calidad de audio insuficiente.",
    MotivoCierre.ERROR_TECNICO: "La llamada terminó por una falla técnica; puede estar incompleta.",
}


def construir_resumen_markdown(
    *,
    paciente_id: str,
    procedimiento: str,
    dia_postop: int,
    fecha: str,
    estado: EstadoClinicoLlamada,
    criticidad_final: Criticidad,
    motivo_cierre: MotivoCierre,
    turnos: list[str],
    sbar: SBAR | None,
    delta_vs_anterior: list[dict],
) -> str:
    """Resumen clínico de la llamada, en Markdown, armado por código y no por el modelo.

    Es deliberado que no lo redacte un LLM. Esto es el registro que lee el equipo médico:
    si lo generara el modelo tendría que volver a confiar en que no inventa, justo en el
    documento donde una invención no la corrige nadie. Todo lo de aquí sale del estado
    acumulado, que ya pasó por la fusión determinista de `clinical/estado.py`.

    Además no cuesta tokens ni depende de que quede cupo al momento de colgar — que es
    exactamente cuando puede no quedar.
    """
    icono = {Criticidad.VERDE: "🟢", Criticidad.AMARILLO: "🟡", Criticidad.ROJO: "🔴"}.get(criticidad_final, "⚪")

    lineas = [
        f"# Seguimiento postoperatorio — {paciente_id}",
        "",
        f"**{procedimiento}** · día postoperatorio **{dia_postop}** · {fecha}",
        "",
        f"## {icono} Criticidad final: {criticidad_final.value.upper()}",
        "",
        f"Cobertura de la evaluación: **{estado.cobertura:.0%}** "
        f"({len(estado.dimensiones_confirmadas)} de {len(_ETIQUETA_DIMENSION)} dimensiones confirmadas).",
        "",
    ]

    if criticidad_final == Criticidad.DESCONOCIDA:
        lineas += [
            "> No se pudo confirmar toda la información necesaria. **El verde exige evidencia",
            "> positiva de ausencia de alarma**: la falta de datos no se interpreta como",
            "> normalidad, por eso este caso queda para revisión del equipo.",
            "",
        ]

    lineas += ["## Dimensiones clínicas", "", "| Dimensión | Estado | Valor | Textual del paciente |", "|---|---|---|---|"]
    for nombre, etiqueta in _ETIQUETA_DIMENSION.items():
        obs = getattr(estado, nombre)
        valor = "—" if obs.valor is None else str(obs.valor)
        verbatim = f"«{obs.verbatim}»" if obs.verbatim else "—"
        procedencia = "" if obs.procedencia == Procedencia.PACIENTE else f" _({obs.procedencia.value})_"
        lineas.append(f"| {etiqueta} | {obs.estado.value}{procedencia} | {valor} | {verbatim} |")
    lineas.append("")

    presentes = [b.replace("_", " ") for b in BANDERAS if getattr(estado.banderas, b).valor == TriEstado.PRESENTE]
    lineas += ["## Banderas rojas", ""]
    lineas += (
        [f"- **{b.capitalize()}**" for b in presentes] + [""]
        if presentes
        else ["Ninguna bandera roja detectada durante la llamada.", ""]
    )
    if estado.banderas.banderas_procedimiento:
        lineas += ["Específicas del procedimiento:", ""]
        lineas += [f"- {b}" for b in estado.banderas.banderas_procedimiento] + [""]

    if delta_vs_anterior:
        lineas += ["## Cambios desde la llamada anterior", ""]
        for d in delta_vs_anterior:
            flecha = "empeora ↑" if d.get("empeora") else "mejora ↓"
            lineas.append(f"- **{d['dimension']}**: {d['valor_anterior']} → {d['valor_actual']} ({flecha})")
        lineas.append("")

    if estado.correcciones:
        lineas += [
            "## Retractaciones registradas",
            "",
            "El paciente intentó rebajar un síntoma ya reportado. **Se conserva el valor más",
            "severo** y se deja constancia del intento; una retractación por cortesía no borra",
            "lo que ya se dijo.",
            "",
        ]
        for c in estado.correcciones:
            lineas.append(f"- Turno {c.turno_idx} · **{c.dimension}**: se conserva `{c.valor_conservado}`, se descarta `{c.valor_descartado}`")
        lineas.append("")

    if sbar:
        lineas += [
            "## SBAR de escalamiento",
            "",
            f"**Situación.** {sbar.situacion}",
            "",
            f"**Contexto.** {sbar.contexto}",
            "",
            f"**Evaluación.** {sbar.evaluacion}",
            "",
            f"**Recomendación.** {sbar.recomendacion}",
            "",
        ]

    if estado.dimensiones_pendientes:
        lineas += [
            "## Sin evaluar",
            "",
            "No se alcanzó a preguntar por: " + ", ".join(estado.dimensiones_pendientes) + ".",
            "",
        ]

    lineas += ["## Transcripción", ""]
    lineas += [f"> **{l.split(':', 1)[0]}:**{l.split(':', 1)[1]}" if ":" in l else f"> {l}" for l in turnos] or ["> (sin turnos)"]
    lineas += ["", "---", "", f"_Cierre: {_MOTIVO_LEGIBLE.get(motivo_cierre, motivo_cierre.value)}_", ""]

    return "\n".join(lineas)


def _ruta_memoria(paciente_id: str) -> Path:
    settings = get_settings()
    directorio = settings.chroma_persist_dir.parent / "memoria"
    directorio.mkdir(parents=True, exist_ok=True)
    return directorio / f"{paciente_id}.json"


def guardar_resumen(resumen: ResumenLlamada) -> None:
    ruta = _ruta_memoria(resumen.paciente_id)
    historial = cargar_historial(resumen.paciente_id)
    historial.append(resumen)
    ruta.write_text(
        json.dumps([r.model_dump(mode="json") for r in historial], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def cargar_historial(paciente_id: str) -> list[ResumenLlamada]:
    ruta = _ruta_memoria(paciente_id)
    if not ruta.exists():
        return []
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    return [ResumenLlamada.model_validate(d) for d in datos]


def ultimo_resumen(paciente_id: str) -> ResumenLlamada | None:
    historial = cargar_historial(paciente_id)
    return historial[-1] if historial else None


def contexto_apertura(paciente_id: str) -> str | None:
    """Texto breve para que el agente abra la llamada recordando la anterior — a partir
    del estado estructurado, no de transcripción cruda pegada (§5 del diseño)."""
    anterior = ultimo_resumen(paciente_id)
    if anterior is None:
        return None

    partes = [f"En la llamada del día {anterior.dia_postop}, criticidad final: {anterior.criticidad_final.value}."]

    confirmadas = []
    estado = anterior.estado_final
    for nombre in ("dolor", "fiebre", "movilidad", "herida", "apetito", "sueno"):
        obs = getattr(estado, nombre)
        if obs.confirmada:
            confirmadas.append(f"{nombre}={obs.valor}")
    if confirmadas:
        partes.append("Reportó: " + ", ".join(confirmadas) + ".")

    if anterior.dimensiones_no_evaluadas:
        partes.append("No se alcanzó a preguntar: " + ", ".join(anterior.dimensiones_no_evaluadas) + ".")

    return " ".join(partes)


def timestamp_actual() -> str:
    return datetime.now().isoformat()
