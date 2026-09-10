"""El guion del protocolo: qué pregunta el agente, en qué orden, con qué palabras exactas.

Hasta ahora el modelo redactaba cada pregunta. Eso costaba una invocación completa por
turno —prompt de ~800 tokens, respuesta de ~50— para producir una frase que siempre dice lo
mismo: «¿cómo ha estado el dolor, del 0 al 10?». Se pagaba latencia y variabilidad a cambio
de nada, porque un protocolo postoperatorio **no debe improvisar sus preguntas**: son las
mismas seis dimensiones para todos los pacientes, y que sean idénticas llamada tras llamada
es un requisito clínico, no una limitación.

Fijarlas compra tres cosas a la vez:

  1. **Latencia.** El audio se sintetiza una sola vez y se guarda (`voice/guion/`), así que
     el turno no paga TTS. El ganador del reto reporta 7–40 ms de TTS por turno por esto.
  2. **Una tarea mucho más simple para el modelo.** Si el agente sabe qué acaba de
     preguntar, solo tiene que extraer *esa* dimensión: un enum de tres valores en vez de un
     esquema de 10.126 caracteres. Medido con `phi3.5:3.8b` en este equipo: **16,2 s → 5,8 s**,
     con un prompt 9,5× más chico.
  3. **Reproducibilidad.** El jurado oye exactamente lo mismo en cada ejecución.

Lo que el modelo sigue haciendo: extraer el valor de la respuesta, y contestar las preguntas
que hace el paciente con RAG anclado. Es decir, lo que un modelo hace bien.
"""

from __future__ import annotations

from dataclasses import dataclass

from agente_postop.clinical.extraction import DIMENSIONES


@dataclass(frozen=True)
class Pregunta:
    """Una pregunta del protocolo, con todo lo que hace falta para resolver su turno.

    `valores` es el enum que se le pasa al decodificador al extraer la respuesta, y
    `criterio` la única instrucción que necesita el modelo. Los dos juntos son el prompt
    completo: no hay nada más que decirle.
    """

    dimension: str
    texto: str
    valores: tuple[str, ...] | None  # None = numérica, la lee un parser determinista
    criterio: str
    reintento: str


# El orden importa clínicamente: se abre por dolor porque es lo que el paciente espera que
# le pregunten y rompe el hielo, y se cierra por sueño y apetito, que son las que un
# paciente minimizador contesta con más franqueza cuando ya lleva rato hablando.
GUION: tuple[Pregunta, ...] = (
    Pregunta(
        dimension="dolor",
        texto="¿Cómo ha estado el dolor desde la cirugía? Si le pongo una escala del cero al diez, "
              "donde diez es el peor dolor que se pueda imaginar, ¿en cuánto estaría hoy?",
        valores=None,
        criterio="",  # lo lee `parsers.extraer_dolor_nrs`, el modelo no interviene
        reintento="Perdóneme que insista: si tuviera que ponerle un número del cero al diez, "
                  "¿cuánto diría?",
    ),
    Pregunta(
        dimension="fiebre",
        texto="¿Se ha tomado la temperatura? ¿Cuánto le ha marcado?",
        valores=None,
        criterio="",  # lo lee `parsers.extraer_temperatura_c`
        reintento="¿Y ha sentido escalofríos o el cuerpo caliente, aunque no se la haya tomado?",
    ),
    Pregunta(
        dimension="herida",
        texto="Cuénteme cómo se ve la herida. ¿Está roja, hinchada, o le sale algún líquido?",
        valores=("normal", "eritema_leve", "secrecion_purulenta"),
        criterio="normal = sin enrojecimiento y sin secreción. "
                 "eritema_leve = enrojecimiento o hinchazón, SIN pus ni líquido. "
                 "secrecion_purulenta = pus, materia, líquido amarillo o verde, o mal olor.",
        reintento="¿Podría mirarla un momentico y decirme si ve algo distinto a los otros días?",
    ),
    Pregunta(
        dimension="movilidad",
        texto="¿Cómo se ha podido mover? ¿Camina usted solo, o le ha costado más de lo que esperaba?",
        valores=("normal", "limitada_esperada", "incapacitante_nueva"),
        criterio="normal = se mueve sin problema. "
                 "limitada_esperada = le cuesta, pero se mueve; lo normal tras una cirugía. "
                 "incapacitante_nueva = no puede moverse o apoyar, y antes sí podía.",
        reintento="¿Ha podido levantarse de la cama y caminar por la casa?",
    ),
    Pregunta(
        dimension="apetito",
        texto="¿Cómo ha estado comiendo estos días?",
        valores=("normal", "levemente_disminuido", "muy_disminuido"),
        criterio="normal = come como siempre. "
                 "levemente_disminuido = come algo menos de lo habitual. "
                 "muy_disminuido = casi no come, se le quita el hambre, o no le pasa la comida.",
        reintento="¿Ha podido comer sus comidas completas, o menos de lo normal?",
    ),
    Pregunta(
        dimension="sueno",
        texto="Y para terminar, ¿cómo ha dormido?",
        valores=("normal", "levemente_alterado", "muy_alterado"),
        criterio="normal = duerme bien. "
                 "levemente_alterado = se despierta alguna vez pero descansa. "
                 "muy_alterado = casi no duerme, se despierta muchas veces, o el dolor no lo deja.",
        reintento="¿Logra dormir de corrido, o se despierta varias veces en la noche?",
    ),
)

POR_DIMENSION: dict[str, Pregunta] = {p.dimension: p for p in GUION}

# Guardarraíl: si alguien añade una dimensión a `extraction.DIMENSIONES` y olvida su
# pregunta, el agente la dejaría sin preguntar para siempre y la llamada nunca alcanzaría
# cobertura completa — un fallo silencioso que solo se vería como criticidad `desconocida`
# en todas las llamadas. Mejor que reviente al importar.
assert set(POR_DIMENSION) == set(DIMENSIONES), (
    f"El guion no cubre todas las dimensiones: falta {set(DIMENSIONES) - set(POR_DIMENSION)}"
)

APERTURA = (
    "Buenos días, le hablo del equipo de seguimiento postoperatorio. "
    "Lo llamo para ver cómo va su recuperación. ¿Tiene un momentico?"
)

# El cierre en verde SIEMPRE enumera los signos de alarma. Tranquilizar a un paciente
# postoperatorio sin decirle qué debe vigilar es la conducta que la rúbrica penaliza de
# forma explícita, y es la más fácil de cometer sin darse cuenta: el agente suena amable y
# el paciente cuelga creyendo que cualquier cosa que pase es normal.
CIERRE_VERDE = (
    "Todo lo que me cuenta suena dentro de lo esperado para estos días. "
    "Eso sí, quiero que esté pendiente: si le sube la fiebre de treinta y ocho grados, "
    "si la herida le empieza a salir pus o mal olor, si el dolor se le vuelve insoportable, "
    "o si le cuesta respirar, no espere a la próxima llamada y vaya a urgencias. "
    "Que siga mejorando."
)

CIERRE_AMARILLO = (
    "Le agradezco que me haya contado todo esto. Voy a dejar su caso anotado para que el "
    "equipo médico lo revise de cerca. Si algo empeora antes de que lo contacten "
    "—fiebre, pus en la herida, o dolor que no se aguanta— vaya a urgencias sin esperar."
)

CIERRE_ROJO = (
    "Por lo que me cuenta, prefiero que alguien del equipo médico lo revise hoy mismo. "
    "Voy a reportar su caso de inmediato. Por favor manténgase disponible, que lo van a "
    "contactar en los próximos minutos. Si mientras tanto se siente peor, vaya a urgencias."
)


def siguiente_pregunta(estado, en_vuelo: set[str] | None = None) -> Pregunta | None:
    """La siguiente dimensión por preguntar, o None si la llamada ya puede cerrarse.

    Sigue el orden del guion y salta lo ya confirmado. Una dimensión bloqueada —preguntada
    dos veces sin respuesta útil— también se salta: insistir una tercera vez no obtiene el
    dato y sí desgasta a un paciente recién operado. Ese hueco no se da por bueno: lo
    recoge la capa de incertidumbre del motor de triaje, que escala a `desconocida`.

    `en_vuelo` son las dimensiones cuya extracción todavía está corriendo en segundo plano.
    Se saltan porque el agente ya preguntó por ellas y todavía no sabe la respuesta:
    volverlas a preguntar sería repetirse. Si la extracción termina sin confirmar el valor,
    la dimensión reaparece en esta lista y se pregunta más adelante, que es exactamente el
    reintento que se quiere.
    """
    en_vuelo = en_vuelo or set()
    for pregunta in GUION:
        if pregunta.dimension in en_vuelo:
            continue
        if getattr(estado, pregunta.dimension).confirmada:
            continue
        if estado.slot_bloqueado(pregunta.dimension):
            continue
        return pregunta
    return None


def texto_a_decir(pregunta: Pregunta, estado) -> str:
    """La formulación que toca: la primera vez la pregunta, la segunda la reformulación.

    Repetir palabra por palabra una pregunta que el paciente ya no entendió es la forma más
    rápida de que la vuelva a esquivar. La reformulación es más concreta y más fácil de
    contestar con un sí o un no.
    """
    return pregunta.reintento if estado.intentos.get(pregunta.dimension, 0) >= 1 else pregunta.texto
