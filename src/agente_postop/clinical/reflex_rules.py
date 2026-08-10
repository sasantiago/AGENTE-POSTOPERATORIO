"""Banderas rojas duras por procedimiento — vía refleja.

Reglas deterministas (palabra clave + umbral numérico), no el LLM. Solo pueden subir la
criticidad, nunca bajarla. `documento_sustento` es el respaldo clínico citable de cada
regla — se completa/ajusta durante la calibración con el harness (ver §4 del diseño).

Taxonomía de procedimientos tomada de
`dataset/perfiles_clinicos_pacientes_silver_contest.xlsx` (columna `procedimiento`):
Apendicectomía, Colecistectomía, Colectomía, Reemplazo de cadera/rodilla, Mastectomía.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

FIEBRE_UMBRAL_C = 38.0


def normalizar(texto: str) -> str:
    """Quita tildes/diacríticos y pasa a minúsculas. Las transcripciones de Whisper no
    siempre son consistentes con las tildes, y el paciente tampoco las pronuncia — el
    reflejo no puede depender de que "secreción" esté bien acentuado para dispararse."""
    descompuesto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn").lower()


@dataclass(frozen=True)
class ReglaPalabraClave:
    patron: re.Pattern[str]
    descripcion: str
    documento_sustento: str | None = None


def _p(*frases: str) -> re.Pattern[str]:
    alternativas = "|".join(re.escape(normalizar(f)) for f in frases)
    return re.compile(alternativas, re.IGNORECASE)


REGLAS_COMUNES: list[ReglaPalabraClave] = [
    ReglaPalabraClave(
        _p("sangrado activo", "sigue sangrando", "no para de sangrar", "sangre que no para",
           "empapada de sangre", "empapado de sangre"),
        "Sangrado activo",
    ),
    ReglaPalabraClave(
        _p("dificultad para respirar", "no puedo respirar", "me ahogo", "falta de aire",
           "ahogado", "ahogada"),
        "Dificultad respiratoria",
    ),
    ReglaPalabraClave(
        _p("rigidez abdominal", "abdomen duro", "abdomen en tabla", "vientre duro como piedra"),
        "Rigidez / defensa abdominal",
    ),
    ReglaPalabraClave(
        _p("mal olor", "huele mal", "huele feo", "huele horrible", "olor feo", "olor fuerte",
           "pus", "secreción amarilla", "secreción verde", "secreción con mal olor"),
        "Secreción con mal olor o cambio de color",
    ),
    ReglaPalabraClave(
        _p("dolor insoportable", "el peor dolor", "no aguanto el dolor", "diez de diez",
           "10 de 10", "dolor de 9", "dolor de 10"),
        "Dolor extremo autorreportado",
    ),
    ReglaPalabraClave(
        _p("me voy a desmayar", "me desmayé", "perdí el conocimiento", "está confundido",
           "está confundida", "no reacciona", "no responde"),
        "Alteración del estado de conciencia",
    ),
]

REGLAS_POR_PROCEDIMIENTO: dict[str, list[ReglaPalabraClave]] = {
    "Apendicectomía": [
        ReglaPalabraClave(
            _p("dolor que se corrió a todo el abdomen", "el dolor se extendió", "dolor generalizado"),
            "Dolor abdominal generalizado (posible peritonitis)",
        ),
    ],
    "Colecistectomía": [
        ReglaPalabraClave(
            _p("piel amarilla", "ojos amarillos", "ictericia", "orina oscura", "heces claras",
               "heces blancas"),
            "Ictericia / signos de obstrucción biliar",
        ),
    ],
    "Colectomía": [
        ReglaPalabraClave(
            _p("no he podido evacuar", "no expulso gases", "vómito fecaloide", "vómito con olor a heces",
               "distensión abdominal severa", "el abdomen muy hinchado"),
            "Signos de obstrucción intestinal",
        ),
        ReglaPalabraClave(
            _p("la bolsa no funciona", "el estoma está morado", "estoma oscuro", "estoma negro"),
            "Compromiso isquémico del estoma",
        ),
    ],
    "Reemplazo de cadera/rodilla": [
        ReglaPalabraClave(
            _p("la pierna está fría", "la pierna morada", "hinchazón repentina de la pierna",
               "dolor y calor en la pantorrilla", "pantorrilla caliente e hinchada"),
            "Sospecha de trombosis venosa profunda",
        ),
        ReglaPalabraClave(
            _p("no puedo apoyar nada de peso", "se salió de su lugar", "se zafó", "sentí un click y no pude mover"),
            "Posible luxación de la prótesis",
        ),
    ],
    "Mastectomía": [
        ReglaPalabraClave(
            _p("el brazo muy hinchado", "el brazo no me cabe", "hinchazón del brazo"),
            "Sospecha de linfedema agudo / complicación vascular",
        ),
    ],
}


def extraer_temperatura_c(texto: str) -> float | None:
    match = re.search(r"(\d{2}(?:[.,]\d)?)\s*(?:°|grados)", texto)
    if not match:
        return None
    valor = match.group(1).replace(",", ".")
    try:
        return float(valor)
    except ValueError:
        return None
