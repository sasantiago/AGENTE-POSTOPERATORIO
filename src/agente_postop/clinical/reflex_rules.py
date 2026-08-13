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
           "secreción amarilla", "secreción verde", "secreción con mal olor"),
        "Secreción con mal olor o cambio de color",
    ),
    ReglaPalabraClave(
        # `pus` con límites de palabra: como subcadena suelta matchea «me PUSieron suero» y
        # «no le PUSe cuidado», que no son secreción purulenta.
        re.compile(r"\bpus\b|\bpurulent"),
        "Secreción purulenta",
    ),
    ReglaPalabraClave(
        # El paciente no dice «secreción»: dice «un líquido, amarillo creo, saliendo de la
        # herida» o «le sale un poquito de líquido, como amarillito». Se exige la
        # coocurrencia de drenaje + color en la misma frase, con hasta 40 caracteres de por
        # medio, porque entre las dos palabras suele haber muletillas («creo», «como», «ahí»).
        re.compile(
            r"(liquid|secrec|supura|drena|sale algo|le sale|salir)[^.;!?]{0,40}"
            r"(amarill|verdos|verde|marron|cafe)"
            r"|(amarill|verdos|verde|marron|cafe)[^.;!?]{0,40}(liquid|secrec|supura|drena|saliendo)"
        ),
        "Drenaje de aspecto purulento en la herida",
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


# Ventana que se inspecciona ANTES del término detectado para decidir si el paciente lo
# está NEGANDO. Solo hacia atrás, a propósito: reglas como «no puedo respirar» llevan la
# negación dentro del propio patrón, así que mirar hacia atrás no las desactiva.
# Dos ventanas, porque las dos formas de negar tienen alcances distintos. El negador
# escueto («no veo pus», «ni pus») solo niega lo que tiene pegado, así que se mira muy
# cerca: con una ventana amplia, «no sé, ayer me salió pus» quedaría anulado por ese «no».
# La frase negativa completa («no le sale nada de…») sí gobierna toda la enumeración que
# viene detrás, y necesita más alcance.
_VENTANA_NEGADOR_CORTO = 12
_VENTANA_FRASE_NEGATIVA = 25
_NEGADOR_CORTO = re.compile(r"\bno\b|\bni\b|\bsin\b")
_FRASE_NEGATIVA = re.compile(r"nada de|no le sale|no he visto|no veo|no tiene|no hay|no se ve|no huele")


def esta_negado(texto_normalizado: str, inicio_match: int) -> bool:
    """¿El paciente está negando el hallazgo que se acaba de detectar?

    «No le sale nada de líquido ni pus» disparaba la bandera de secreción purulenta por la
    sola presencia de la palabra. En una vía que solo puede SUBIR la criticidad, ese ruido
    se paga en credibilidad del escalamiento: el turno donde el paciente descarta el
    síntoma es justo el que no debe alertar.

    Solo se mira hacia atrás. Las reglas que llevan la negación dentro del patrón —«no
    puedo respirar», «no para de sangrar», «no reacciona»— quedan intactas por construcción.
    """
    corta = texto_normalizado[max(0, inicio_match - _VENTANA_NEGADOR_CORTO) : inicio_match]
    larga = texto_normalizado[max(0, inicio_match - _VENTANA_FRASE_NEGATIVA) : inicio_match]
    return bool(_NEGADOR_CORTO.search(corta) or _FRASE_NEGATIVA.search(larga))


# Rango fisiológicamente plausible para una temperatura corporal reportada por teléfono.
# Acota el riesgo de leer como fiebre un número que no lo es: la intensidad del dolor
# («como un 5», «un 6 tal vez») y los días postoperatorios caen todos fuera.
TEMP_MIN_PLAUSIBLE_C = 35.0
TEMP_MAX_PLAUSIBLE_C = 42.5

# El paciente casi nunca dice «grados». Dice «me la tomé y marcó 38», «me sentí afiebrada,
# como 38», «marcaba 39 algo». Exigir la unidad perdía la fiebre entera — y la fiebre es la
# bandera roja más común del postoperatorio. Medido sobre capa1_limpia del dataset: 5 de los
# 7 casos `rojo` que la vía refleja no detectaba reportaban una temperatura >= 38 en
# palabras, sin unidad.
_CONTEXTO_TERMICO = re.compile(
    r"temperatura|termometro|fiebre|afiebrad|calentura|grados|°|marc[oa]|febril|"
    r"escalofri|destemplad"
)
_NUMERO_TEMPERATURA = re.compile(r"\b(3[5-9]|4[0-2])(?:[.,](\d))?\b")


def extraer_temperatura_c(texto: str) -> float | None:
    """Temperatura en °C reportada en el turno, o None.

    Acepta la forma explícita («38.5 grados», «38°») y la coloquial («marcó como 38»,
    «38 y algo»), esta última solo cuando el turno habla de temperatura: sin esa condición,
    cualquier cifra entre 35 y 42 se leería como fiebre.
    """
    normalizado = normalizar(texto)

    explicito = re.search(r"(\d{2}(?:[.,]\d)?)\s*(?:°|grados)", normalizado)
    if explicito:
        try:
            return float(explicito.group(1).replace(",", "."))
        except ValueError:
            return None

    if not _CONTEXTO_TERMICO.search(normalizado):
        return None

    for match in _NUMERO_TEMPERATURA.finditer(normalizado):
        entero, decimal = match.group(1), match.group(2)
        valor = float(f"{entero}.{decimal}") if decimal else float(entero)
        if TEMP_MIN_PLAUSIBLE_C <= valor <= TEMP_MAX_PLAUSIBLE_C:
            return valor
    return None
