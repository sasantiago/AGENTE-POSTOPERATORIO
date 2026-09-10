"""Los dos números que disparan ROJO, extraídos sin modelo.

Fiebre ≥ 38 °C y dolor ≥ 8/10 son dos de las tres reglas que escalan a rojo en
`clinical/triage.py`. Hasta ahora esos valores los rellenaba el LLM, y eso dejaba la
decisión más grave del sistema apoyada en la pieza menos verificable.

Medido sobre `llama3.2:3b`, que es lo que corre en local: ante «no me dan ganas de comer
nada» respondía `levemente_disminuido` —subestimando—, y ante turnos que no mencionaban una
dimensión se inventaba un valor en vez de callarse. Un extractor así no puede ser la fuente
de los números que deciden si un paciente recién operado recibe una llamada del cirujano.

Un parser no razona, pero tampoco alucina, y sobre todo: **se puede medir sin gastar un
token y se puede leer**. `harness/eval_parsers.py` lo contrasta contra los 3.991 turnos del
dataset.

La regla de diseño que gobierna las dos funciones: **ante la duda, devolver None.** Un
valor ausente escala a `desconocida` por la capa de incertidumbre del motor; un valor
inventado se toma por bueno. El silencio es recuperable, la mentira no.
"""

from __future__ import annotations

import re
import unicodedata


def normalizar(texto: str) -> str:
    """Quita tildes y pasa a minúsculas.

    Las transcripciones de Whisper no son consistentes con las tildes y el paciente tampoco
    las pronuncia: el parser no puede depender de que «secreción» venga bien acentuada.
    """
    descompuesto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn").lower()


# ---------------------------------------------------------------------------
# Temperatura
# ---------------------------------------------------------------------------

# Rango fisiológicamente plausible para una temperatura reportada por teléfono. Acota el
# riesgo de leer como fiebre un número que no lo es: la intensidad del dolor («como un 5»)
# y los días postoperatorios caen todos fuera.
TEMP_MIN_PLAUSIBLE_C = 35.0
TEMP_MAX_PLAUSIBLE_C = 42.5

# El paciente casi nunca dice «grados». Dice «me la tomé y marcó 38», «me sentí afiebrada,
# como 38», «marcaba 39 algo». Exigir la unidad perdía la fiebre entera — y la fiebre es la
# bandera roja más común del postoperatorio. Medido sobre capa1_limpia: 5 de los 7 casos
# `rojo` que la vía refleja no detectaba reportaban una temperatura ≥ 38 en palabras, sin
# unidad.
_CONTEXTO_TERMICO = re.compile(
    r"temperatura|termometro|fiebre|afiebrad|calentura|grados|°|marc[oa]|febril|"
    r"escalofri|destemplad"
)
_NUMERO_TEMPERATURA = re.compile(r"\b(3[5-9]|4[0-2])(?:[.,](\d))?\b")

# La temperatura dicha en palabras. Whisper transcribe «treinta y ocho» tal cual con la
# misma frecuencia con que escribe «38», y sin esto la fiebre se perdía entera: en la
# primera prueba del caso rojo, «marcó treinta y ocho y medio» dejaba el slot vacío y el
# paciente terminaba clasificado como `desconocida` en vez de `rojo`.
#
# Solo cubre 35–42, que es el rango plausible de una temperatura corporal. No hace falta un
# parser general de números en español para leer un termómetro.
_DECENA_TEMPERATURA = {"treinta": 30, "cuarenta": 40}
_UNIDAD_TEMPERATURA = {
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
    "uno": 1, "dos": 2, "uno y": 1,
}
_TEMPERATURA_EN_PALABRAS = re.compile(
    r"\b(treinta|cuarenta)(?:\s+y\s+(cinco|seis|siete|ocho|nueve|uno|dos))?"
)


def extraer_temperatura_c(texto: str, *, se_lo_preguntaron: bool = False) -> float | None:
    """Temperatura en °C reportada en el turno, o None.

    Acepta la forma explícita («38.5 grados», «38°») y la coloquial («marcó como 38», «38 y
    algo»), esta última solo cuando el turno habla de temperatura: sin esa condición,
    cualquier cifra entre 35 y 42 se leería como fiebre.

    `se_lo_preguntaron` levanta esa exigencia, y arregla un fallo observado en la primera
    llamada real: **el contexto no siempre está en la respuesta, está en la pregunta.** Si el
    agente acaba de preguntar «¿cuánto le ha marcado?», el paciente contesta «treinta y ocho
    y medio» sin nombrar la temperatura ni una vez, y exigir la palabra tiraba el dato.
    """
    normalizado = normalizar(texto)

    explicito = re.search(r"(\d{2}(?:[.,]\d)?)\s*(?:°|grados)", normalizado)
    if explicito:
        try:
            valor = float(explicito.group(1).replace(",", "."))
        except ValueError:
            return None
        return valor if TEMP_MIN_PLAUSIBLE_C <= valor <= TEMP_MAX_PLAUSIBLE_C else None

    if not se_lo_preguntaron and not _CONTEXTO_TERMICO.search(normalizado):
        return None

    for match in _NUMERO_TEMPERATURA.finditer(normalizado):
        entero, decimal = match.group(1), match.group(2)
        valor = float(f"{entero}.{decimal}") if decimal else float(entero)
        if decimal is None:
            valor += _fraccion_hablada(normalizado, match.end())
        if TEMP_MIN_PLAUSIBLE_C <= valor <= TEMP_MAX_PLAUSIBLE_C:
            return valor

    # En palabras: «treinta y ocho y medio».
    match = _TEMPERATURA_EN_PALABRAS.search(normalizado)
    if match:
        valor = float(_DECENA_TEMPERATURA[match.group(1)] + _UNIDAD_TEMPERATURA.get(match.group(2) or "", 0))
        valor += _fraccion_hablada(normalizado, match.end())
        if TEMP_MIN_PLAUSIBLE_C <= valor <= TEMP_MAX_PLAUSIBLE_C:
            return valor
    return None


# Cómo se dicen los decimales de una temperatura hablando. «Treinta y ocho y medio» es
# 38.5, no 38: observado en la primera llamada de prueba, donde el paciente dijo «marcó 38
# y medio» y el parser leía 38.0. No cambia la decisión de triaje —los dos cruzan el
# umbral— pero sí lo que se le muestra al clínico en el SBAR, y un valor que no coincide
# con lo que el paciente dijo es un valor que nadie va a creerse.
_FRACCION_HABLADA: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"^\s*(?:y\s+)?medio\b"), 0.5),
    (re.compile(r"^\s*(?:y\s+)?punto\s+cinco\b"), 0.5),
    (re.compile(r"^\s*(?:y\s+)?algo\b"), 0.0),   # «38 y algo»: no se inventa el decimal
    (re.compile(r"^\s*(?:y\s+)?pico\b"), 0.0),
)


def _fraccion_hablada(texto: str, desde: int) -> float:
    """El decimal que sigue a un número dicho en palabras, si lo hay."""
    cola = texto[desde : desde + 16]
    for patron, valor in _FRACCION_HABLADA:
        if patron.search(cola):
            return valor
    return 0.0


# ---------------------------------------------------------------------------
# Dolor en escala numérica (NRS 0-10)
# ---------------------------------------------------------------------------

# El agente pregunta explícitamente «del 0 al 10», así que el paciente responde con un
# número la mayoría de las veces. Pero lo dice de muchas formas: «un 6», «como un cuatro»,
# «seis de diez», «un 7 más o menos», «yo diría que 8».
_PALABRA_A_NUMERO = {
    "cero": 0, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
}

# Contexto que autoriza a leer un número suelto como intensidad de dolor. Sin esta
# condición, «llevo 8 días con la herida» se leería como un dolor de 8/10 y dispararía un
# escalamiento a rojo por una frase que no habla de dolor.
_CONTEXTO_DOLOR = re.compile(
    r"dolor|duele|molest|adolorid|escala|del 0 al 10|del cero al diez|punt|"
    r"me duele|dolorcit|arde|ardor|pincha|puntada"
)

# Construcciones donde el número ES la intensidad. El orden importa: se prueban de la más
# específica a la más laxa, y la primera que casa gana.
_PATRONES_DOLOR: tuple[re.Pattern[str], ...] = (
    # «un 7 de 10», «7 sobre 10», «siete de diez»
    re.compile(r"\b(\d{1,2}|" + "|".join(_PALABRA_A_NUMERO) + r")\s*(?:de|sobre|/)\s*(?:10|diez)\b"),
    # «como un 6», «un cuatro», «en un 5», «yo diría que un 8»
    re.compile(r"\b(?:como|en|seria|es|de|diria que|estaria en|queda en)?\s*un[ao]?\s+"
               r"(\d{1,2}|" + "|".join(_PALABRA_A_NUMERO) + r")\b"),
    # «el dolor está en 6», «me duele 7»
    re.compile(r"\b(?:esta en|estaria|llega a|anda por|marca|es|duele)\s+"
               r"(\d{1,2}|" + "|".join(_PALABRA_A_NUMERO) + r")\b"),
    # Último recurso: un número suelto en una frase que habla de dolor, en cifra o en
    # palabra. La palabra suelta hace falta porque a «¿en cuánto estaría?» mucha gente
    # contesta «Ocho.» y nada más — observado en la primera llamada con micrófono, donde
    # «un ocho» sí se leía y «Ocho.» no.
    #
    # Dos exclusiones que el arnés de regresión destapó al añadirlo:
    #
    #   `(?<![\d.,])`  el dígito no puede venir detrás de otro ni de un separador decimal.
    #                  Sin esto, «me tomé la temperatura y marcó 37.6°C» leía un dolor de
    #                  6/10 —y «38.1°C», uno de 1/10— a partir del decimal de la fiebre.
    #
    #   sin `uno`/`una`  en español «uno» casi nunca es la intensidad: es el impersonal.
    #                    «un poquito molesto, uno aguanta» daba un dolor de 1 en un paciente
    #                    cuyo dolor real era 9. Un 1 se dice «un uno» o «1», y esas dos
    #                    formas las cubren los patrones de arriba.
    re.compile(
        r"(?<![\d.,])\b(\d{1,2}|"
        + "|".join(p for p in _PALABRA_A_NUMERO if p not in ("uno", "una"))
        + r")\b"
    ),
)

# Rangos: «entre 5 y 6», «un 6 o 7». Se toma el MAYOR, que es la dirección segura para un
# triaje: si el paciente duda entre 7 y 8, tratarlo como 8 escala; como 7, no.
_RANGO_DOLOR = re.compile(
    r"\b(\d{1,2}|" + "|".join(_PALABRA_A_NUMERO) + r")\s*(?:o|a|y|-)\s*"
    r"(\d{1,2}|" + "|".join(_PALABRA_A_NUMERO) + r")\b"
)

# Unidades que descartan la lectura: el número es otra cosa (días, horas, pastillas).
_UNIDAD_AJENA = re.compile(
    r"\b(dias?|horas?|semanas?|meses?|anos?|veces|pastillas?|tabletas?|miligramos?|mg|"
    r"grados?|kilos?|puntos de sutura|centimetros?|cm)\b"
)


def _a_entero(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    return _PALABRA_A_NUMERO.get(token)


def extraer_dolor_nrs(texto: str, *, se_lo_preguntaron: bool = False) -> float | None:
    """Intensidad de dolor 0-10 reportada en el turno, o None.

    Solo lee números cuando la frase habla de dolor. Es la misma condición de contexto que
    protege a la temperatura, y por el mismo motivo: en una llamada postoperatoria hay
    cifras por todas partes —días desde la cirugía, horas de sueño, número de pastillas— y
    ninguna de ellas es la escala del dolor.

    `se_lo_preguntaron` levanta esa exigencia cuando el agente **acaba de preguntar por el
    dolor**, que es justo cuando el paciente responde con el número desnudo. Encontrado en
    la primera llamada real con micrófono: a «¿en cuánto estaría el dolor hoy?» el paciente
    contestó «un ocho», el parser no vio la palabra «dolor» y devolvió None, y el agente
    repitió la misma pregunta indefinidamente.
    """
    normalizado = normalizar(texto)
    if not se_lo_preguntaron and not _CONTEXTO_DOLOR.search(normalizado):
        return None

    # Un rango se resuelve antes que un valor suelto: «un 6 o 7» debe dar 7, y el patrón
    # de valor suelto se quedaría con el 6 por ser el primero que encuentra.
    rango = _RANGO_DOLOR.search(normalizado)
    if rango and not _hay_unidad_ajena_cerca(normalizado, rango.start(), rango.end()):
        a, b = _a_entero(rango.group(1)), _a_entero(rango.group(2))
        if a is not None and b is not None and 0 <= a <= 10 and 0 <= b <= 10:
            return float(max(a, b))

    for patron in _PATRONES_DOLOR:
        for match in patron.finditer(normalizado):
            valor = _a_entero(match.group(1))
            if valor is None or not 0 <= valor <= 10:
                continue
            if _hay_unidad_ajena_cerca(normalizado, match.start(), match.end()):
                continue
            return float(valor)
    return None


# Ventana tras el número donde una unidad ajena lo descalifica. Corta a propósito: solo
# busca la unidad pegada al número («8 dias», «dos pastillas»), no en toda la frase, porque
# «me duele un 8 y llevo tres dias así» sí reporta un dolor de 8.
_VENTANA_UNIDAD = 12


def _hay_unidad_ajena_cerca(texto: str, inicio: int, fin: int) -> bool:
    return bool(_UNIDAD_AJENA.search(texto[fin : fin + _VENTANA_UNIDAD]))


# ---------------------------------------------------------------------------
# Sustitución en el delta de extracción
# ---------------------------------------------------------------------------


def imponer_parsers(delta, texto_paciente: str, dimension_preguntada: str | None = None):
    """Reemplaza lo que el modelo dijo de dolor y fiebre por lo que leyó el parser.

    Muta y devuelve el mismo `ExtraccionTurno`. Es una sustitución, no una fusión: sobre
    estas dos dimensiones el modelo deja de tener voz, incluso cuando el parser no
    encuentra nada. Si el paciente no dio el número, el slot queda **sin confirmar** y el
    motor de triaje lo trata como incertidumbre; eso es preferible a un valor que el modelo
    dedujo del tono de la frase.

    La regla nace de un caso concreto del dataset: pacientes con perfil
    `minimizador_sintomas` cuyo dolor real es 9/10 y que dicen «un poquito molesto no más,
    uno aguanta». Ahí no hay número que extraer, y un modelo que igualmente rellena el slot
    está adivinando. Lo que escala esos casos no es el valor numérico: es el puntaje
    acumulado de las otras dimensiones y la vía refleja.
    """
    from agente_postop.clinical.extraction import Confianza, EstadoSlot, Observacion, Procedencia

    lecturas = {
        "dolor": extraer_dolor_nrs(texto_paciente, se_lo_preguntaron=dimension_preguntada == "dolor"),
        "fiebre": extraer_temperatura_c(texto_paciente, se_lo_preguntaron=dimension_preguntada == "fiebre"),
    }
    for dimension, valor in lecturas.items():
        if valor is None:
            # Sin lectura no hay dato, pero tampoco se borra que se haya preguntado: poner
            # aquí una `Observacion()` vacía dejaba el slot en `no_preguntado`, y
            # `fusionar_extraccion` solo cuenta un intento cuando el estado NO es ese. El
            # contador no avanzaba nunca, el slot no llegaba a bloquearse, y el agente
            # repetía la misma pregunta para siempre — observado en la primera llamada real.
            actual = getattr(delta.dimensiones, dimension)
            setattr(
                delta.dimensiones,
                dimension,
                Observacion(estado=actual.estado, verbatim=actual.verbatim)
                if actual.estado != EstadoSlot.NO_PREGUNTADO
                else Observacion(),
            )
            continue
        setattr(
            delta.dimensiones,
            dimension,
            Observacion(
                valor=int(valor) if dimension == "dolor" else valor,
                estado=EstadoSlot.CONFIRMADO,
                verbatim=texto_paciente[:200],
                procedencia=Procedencia.PACIENTE,
                # Alta y no media: un número leído literalmente de lo que dijo el paciente
                # es la evidencia más fuerte que maneja el sistema. La confianza baja
                # degradaría el slot a AMBIGUO en la fusión (regla 2 de §4.2).
                confianza=Confianza.ALTA,
            ),
        )
    return delta
