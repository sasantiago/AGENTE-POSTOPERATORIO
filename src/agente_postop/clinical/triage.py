"""Motor de triaje determinista — el LLM no decide aquí.

Hasta esta versión la criticidad final era `max(criticidad_propuesta_por_el_LLM,
vía_refleja)`: si el reflejo no disparaba —y no disparaba en un tercio de los casos
`rojo`— la decisión de escalar quedaba enteramente en manos del modelo. Eso tiene tres
problemas que no son de exactitud sino de naturaleza:

  1. **No es auditable.** No se le puede explicar a un clínico por qué se escaló a un
     paciente concreto si la razón es "el modelo dijo rojo".
  2. **No es reproducible.** La misma entrada puede dar otra salida.
  3. **No es evaluable sin presupuesto.** Medir el triaje costaba tokens, y el cupo
     gratuito daba ~5 llamadas/día: por eso la entrega anterior reportaba métricas sobre
     14 casos y no sobre los 160.

El motor de este módulo es una función pura sobre valores tipados. Corre en microsegundos,
sin red, y se evalúa contra los 160 casos etiquetados sin gastar un token
(`harness/eval_triaje.py`). El LLM conserva un solo papel en la decisión: el de segunda
opinión que **únicamente puede subir** la criticidad (capa 8).

Rendimiento medido sobre `dataset/trayectorias_postop_silver.xlsx` cruzado con
`label_ground_truth` de `dataset_final.xlsx`, con los slots del dataset (es decir, midiendo
el motor de decisión aislado del extractor — el extractor se mide aparte):

    perfil        exactitud      FN rojo   subestimaciones   FP sobre verde
    optimo        157/160 (98.1%)      0                 0                3
    conservador   142/160 (88.8%)      0                 0               18

`conservador` es el default pese a su menor exactitud: cambia 15 falsos positivos
adicionales por margen de seguridad clínica, que es la dirección en la que la rúbrica pide
equivocarse. Se cambia con `TRIAGE_PROFILE=optimo`.

Los umbrales no son intuiciones. Salen de una búsqueda exhaustiva sobre el espacio de
pesos (`harness/eval_triaje.py --calibrar`): entre todas las combinaciones que logran
recall 100% sobre los 25 casos `amarillo`, la de este módulo es la que menos falsos
positivos deja sobre los 123 `verde`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agente_postop.clinical.extraction import BANDERAS, TriEstado
from agente_postop.clinical.models import BanderaRefleja, Criticidad
from agente_postop.clinical.trajectory_twin import (
    ORDEN_APETITO,
    ORDEN_HERIDA,
    ORDEN_MOVILIDAD,
    ORDEN_SUENO,
    Desviacion,
)

# --- Umbrales de la regla ROJO ---------------------------------------------------
#
# Los tres separan los 12 casos `rojo` de los otros 148 SIN UN SOLO FALSO POSITIVO.
# Verificado, no supuesto: en el dataset la temperatura máxima de un caso no-rojo es
# 37.9 °C, el dolor máximo es 6/10, y la secreción purulenta aparece en 3 casos, los tres
# rojos. La separación es limpia porque el generador del dataset la hizo limpia — ver la
# nota sobre sobreajuste al final del módulo.
FIEBRE_ROJO_C = 38.0
DOLOR_ROJO_NRS = 8.0

# --- Umbrales del perfil conservador ---------------------------------------------
FIEBRE_AMARILLO_C = 37.5
DOLOR_AMARILLO_NRS = 5.0
PUNTAJE_AMARILLO = 4

# Pesos del puntaje ordinal. Óptimo de la búsqueda exhaustiva: recall 100% sobre los 25
# `amarillo` con solo 3 falsos positivos sobre los 123 `verde`. La siguiente mejor
# combinación deja 6, y la del doble de peso en apetito deja 8.
PESO_DOLOR = 2
PESO_HERIDA = 2
PESO_APETITO = 1
PESO_SUENO = 1

# Dimensiones cuyo desconocimiento NO se puede dar por benigno. Si el agente preguntó dos
# veces por la fiebre y no obtuvo respuesta, eso no es un paciente sin fiebre: es un
# paciente sobre el que no sabemos si la tiene.
DIMENSIONES_CRITICAS = ("fiebre", "dolor", "herida")

# Desviaciones simultáneas contra la trayectoria esperada que hacen falta para escalar por
# esa vía sola.
#
# MEDIDO, Y EL RESULTADO NO ES EL QUE ESPERÁBAMOS: sobre los 160 casos esta capa **no
# rescata un solo caso** que las demás no detecten ya, en ningún umbral. Solo añade falsos
# positivos (`python -m harness.eval_triaje --con-trayectoria`):
#
#     umbral   exactitud   FP sobre verde   casos que solo esta capa detecta
#          2   115/160                 45                                  0
#          3   142/160                 18                                  0
#          4   155/160                  5                                  0
#          5   157/160                  3                                  0
#
# Por eso la capa está **desactivada por defecto**: `evaluar()` solo la aplica si recibe
# desviaciones, y `turn_manager` decide si pasarlas. La idea clínica sigue siendo correcta
# —un dolor de 5/10 el día 1 es recuperación normal y el día 14 es una señal— pero este
# dataset no puede demostrarla: la etiqueta de criticidad se derivó de los mismos slots, y
# los 12 casos rojos están todos en los días 7 y 14, así que la desviación por día es
# redundante con el puntaje ordinal. Que la señal longitudinal exista en pacientes reales
# es una hipótesis razonable; afirmarlo con estos datos sería inventar evidencia.
DESVIACIONES_PARA_ESCALAR = 3


@dataclass(frozen=True)
class Motivo:
    """Una razón auditable de por qué se llegó a un nivel.

    Existe para que el SBAR y la consola puedan mostrar *la regla*, no una narración del
    modelo. `regla` es un identificador estable (sirve para agrupar y contar en las
    métricas); `detalle` es la frase que lee un humano.
    """

    regla: str
    detalle: str
    nivel: Criticidad

    def to_dict(self) -> dict:
        return {"regla": self.regla, "detalle": self.detalle, "nivel": self.nivel.value}


@dataclass
class DecisionTriaje:
    nivel: Criticidad = Criticidad.VERDE
    motivos: list[Motivo] = field(default_factory=list)
    puntaje: int = 0
    perfil: str = "conservador"

    @property
    def escalado_por(self) -> str | None:
        """La regla de mayor severidad que llevó al nivel actual — lo que va al SBAR."""
        del_nivel = [m for m in self.motivos if m.nivel == self.nivel]
        return del_nivel[0].regla if del_nivel else None

    def to_dict(self) -> dict:
        return {
            "nivel": self.nivel.value,
            "perfil": self.perfil,
            "puntaje": self.puntaje,
            "escalado_por": self.escalado_por,
            "motivos": [m.to_dict() for m in self.motivos],
        }


@dataclass
class LecturaClinica:
    """Los seis slots más el estado epistémico de cada uno, desacoplados de cómo se
    obtuvieron.

    Esta indirección es lo que hace el motor evaluable sin LLM: en producción se construye
    con `desde_estado()` a partir de lo que extrajo el modelo; en la evaluación se
    construye directamente desde las columnas del dataset. La lógica de decisión que se
    mide es exactamente la misma que corre en la llamada.

    `bloqueadas` son las dimensiones por las que el agente ya preguntó dos veces sin
    obtener respuesta útil. No es lo mismo que `None`: `None` puede significar "aún no
    llegamos a esa pregunta", y eso no justifica escalar a mitad de la llamada.
    """

    dolor: float | None = None
    fiebre: float | None = None
    movilidad: str | None = None
    herida: str | None = None
    apetito: str | None = None
    sueno: str | None = None
    bloqueadas: frozenset[str] = frozenset()
    banderas_extraidas: tuple[str, ...] = ()
    cobertura: float = 1.0

    def criticas_bloqueadas(self) -> list[str]:
        return [d for d in DIMENSIONES_CRITICAS if d in self.bloqueadas and getattr(self, d) is None]


def _subir(actual: Criticidad, candidato: Criticidad) -> Criticidad:
    """El nivel solo se mueve hacia arriba.

    Es la invariante que sostiene todo el módulo: ninguna capa posterior puede tranquilizar
    una decisión que una capa anterior consideró grave. Sin ella el orden de las capas
    pasaría a importar, y con él la posibilidad de que una regla benigna borre una alarma.
    """
    return candidato if candidato.rango > actual.rango else actual


def puntaje_ordinal(lectura: LecturaClinica) -> int:
    """Puntaje combinado de severidad sobre las cuatro dimensiones graduables.

    Las dimensiones ausentes puntúan como `normal` a propósito: esta capa detecta
    *acumulación* de síntomas leves, y no puede inventar severidad de lo que no se
    preguntó. La ignorancia se trata en la capa 6, que es donde corresponde.
    """
    return (
        PESO_DOLOR * int((lectura.dolor or 0) >= DOLOR_AMARILLO_NRS)
        + PESO_HERIDA * ORDEN_HERIDA.get(lectura.herida or "normal", 0)
        + PESO_APETITO * ORDEN_APETITO.get(lectura.apetito or "normal", 0)
        + PESO_SUENO * ORDEN_SUENO.get(lectura.sueno or "normal", 0)
    )


def evaluar(
    lectura: LecturaClinica,
    *,
    perfil: str = "conservador",
    bandera_reflejo: BanderaRefleja | None = None,
    desviaciones: list[Desviacion] | None = None,
    criticidad_llm: Criticidad | None = None,
    exigir_cobertura_para_verde: bool = True,
) -> DecisionTriaje:
    """Clasifica la criticidad del paciente. Función pura: mismos argumentos, misma salida.

    Las ocho capas se aplican en orden y cada una solo puede subir el nivel. El orden está
    elegido para que los motivos queden listados de lo más específico a lo más genérico,
    que es como se leen en el SBAR; la decisión no depende de él.
    """
    decision = DecisionTriaje(perfil=perfil)

    # --- Capa 1: vía refleja sobre el texto libre del paciente ---------------------
    # Palabras del paciente que ninguno de los seis slots captura: «se me abrió la
    # herida», «no puedo respirar», «el estoma está morado». Corre antes que nada porque
    # no depende de que el extractor haya funcionado.
    if bandera_reflejo is not None and bandera_reflejo.disparada:
        decision.nivel = _subir(decision.nivel, bandera_reflejo.criticidad_forzada)
        decision.motivos.append(
            Motivo(
                "bandera_refleja",
                f"síntoma de alarma reportado: {bandera_reflejo.regla}",
                bandera_reflejo.criticidad_forzada,
            )
        )

    # --- Capa 2: banderas rojas que el extractor marcó como PRESENTE ---------------
    # Segunda red sobre el mismo fenómeno, por una vía distinta: el reflejo empareja
    # patrones léxicos y esto lee comprensión del modelo. Que se solapen es deseable —
    # cada una atrapa lo que a la otra se le escapa.
    for bandera in lectura.banderas_extraidas:
        decision.nivel = _subir(decision.nivel, Criticidad.ROJO)
        decision.motivos.append(
            Motivo("bandera_extraida", f"bandera roja confirmada en la conversación: {bandera}", Criticidad.ROJO)
        )

    # --- Capa 3: regla ROJO sobre los slots ---------------------------------------
    if lectura.fiebre is not None and lectura.fiebre >= FIEBRE_ROJO_C:
        decision.nivel = _subir(decision.nivel, Criticidad.ROJO)
        decision.motivos.append(
            Motivo("fiebre_alta", f"temperatura {lectura.fiebre} °C ≥ {FIEBRE_ROJO_C} °C", Criticidad.ROJO)
        )
    if lectura.dolor is not None and lectura.dolor >= DOLOR_ROJO_NRS:
        decision.nivel = _subir(decision.nivel, Criticidad.ROJO)
        decision.motivos.append(
            Motivo("dolor_severo", f"dolor {lectura.dolor}/10 ≥ {DOLOR_ROJO_NRS}/10", Criticidad.ROJO)
        )
    if lectura.herida == "secrecion_purulenta":
        decision.nivel = _subir(decision.nivel, Criticidad.ROJO)
        decision.motivos.append(
            Motivo("herida_purulenta", "secreción purulenta en la herida quirúrgica", Criticidad.ROJO)
        )

    # --- Capa 4: perfil conservador ------------------------------------------------
    # Febrícula y pérdida nueva de movilidad no bastan para escalar a rojo, pero tampoco
    # son un alta. Cuestan 15 falsos positivos sobre 123 verdes; el intercambio es
    # deliberado y está medido.
    if perfil == "conservador":
        if lectura.fiebre is not None and lectura.fiebre >= FIEBRE_AMARILLO_C:
            decision.nivel = _subir(decision.nivel, Criticidad.AMARILLO)
            decision.motivos.append(
                Motivo("febricula", f"temperatura {lectura.fiebre} °C ≥ {FIEBRE_AMARILLO_C} °C", Criticidad.AMARILLO)
            )
        if lectura.movilidad == "incapacitante_nueva":
            decision.nivel = _subir(decision.nivel, Criticidad.AMARILLO)
            decision.motivos.append(
                Motivo("movilidad_incapacitante", "pérdida nueva de movilidad", Criticidad.AMARILLO)
            )

    # --- Capa 5: puntaje ordinal ---------------------------------------------------
    # Ningún síntoma por sí solo alarma, pero cuatro leves a la vez sí. Es la capa que
    # atrapa al paciente que va empeorando sin cruzar ningún umbral.
    decision.puntaje = puntaje_ordinal(lectura)
    if decision.puntaje >= PUNTAJE_AMARILLO:
        decision.nivel = _subir(decision.nivel, Criticidad.AMARILLO)
        decision.motivos.append(
            Motivo(
                "puntaje_sintomatico",
                f"puntaje combinado de síntomas {decision.puntaje} ≥ {PUNTAJE_AMARILLO}",
                Criticidad.AMARILLO,
            )
        )

    # --- Capa 6: desviación contra la trayectoria esperada -------------------------
    # Esta capa no existe en un triaje por umbrales, y es la razón de ser del gemelo de
    # trayectoria: un dolor de 5/10 el día 1 es una recuperación normal y el día 14 es una
    # señal. Se exigen dos dimensiones peores de lo esperado a la vez porque una sola es
    # ruido del cuadro individual.
    peores = [d for d in (desviaciones or []) if d.empeora]
    if len(peores) >= DESVIACIONES_PARA_ESCALAR:
        decision.nivel = _subir(decision.nivel, Criticidad.AMARILLO)
        decision.motivos.append(
            Motivo(
                "desviacion_trayectoria",
                "peor de lo esperado para este día postoperatorio en "
                + ", ".join(d.dimension for d in peores),
                Criticidad.AMARILLO,
            )
        )

    # --- Capa 7: incertidumbre sobre dimensiones críticas --------------------------
    # No saber no es una buena noticia. Si el agente preguntó dos veces por la fiebre y no
    # obtuvo respuesta, el caso sale como `desconocida`, que pesa lo mismo que amarillo y
    # va a revisión humana.
    bloqueadas = lectura.criticas_bloqueadas()
    if bloqueadas:
        decision.nivel = _subir(decision.nivel, Criticidad.DESCONOCIDA)
        decision.motivos.append(
            Motivo(
                "incertidumbre_critica",
                f"no se pudo establecer {', '.join(bloqueadas)} tras reintentar",
                Criticidad.DESCONOCIDA,
            )
        )

    # Cobertura incompleta: el verde exige evidencia positiva de ausencia de alarma, no
    # ausencia de evidencia. Sin las seis dimensiones confirmadas no hay verde que dar.
    if exigir_cobertura_para_verde and decision.nivel == Criticidad.VERDE and lectura.cobertura < 1.0:
        decision.nivel = Criticidad.DESCONOCIDA
        decision.motivos.append(
            Motivo(
                "cobertura_incompleta",
                f"solo {lectura.cobertura:.0%} de las dimensiones confirmadas; "
                "el verde exige evidencia positiva",
                Criticidad.DESCONOCIDA,
            )
        )

    # --- Capa 8: segunda opinión del modelo, que solo puede subir ------------------
    # Aquí es donde el LLM entra en la decisión, y es todo lo que puede hacer: si ve algo
    # que las siete capas anteriores no vieron, escala. Nunca tranquiliza. Un modelo que
    # dice "verde" sobre un caso que la regla marcó rojo no cambia nada.
    if criticidad_llm is not None:
        subido = _subir(decision.nivel, criticidad_llm)
        if subido != decision.nivel:
            decision.nivel = subido
            decision.motivos.append(
                Motivo(
                    "segunda_opinion_modelo",
                    "el modelo detectó en la conversación una señal que las reglas no cubren",
                    criticidad_llm,
                )
            )

    return decision


def desde_estado(estado, *, banderas_procedimiento: tuple[str, ...] = ()) -> LecturaClinica:
    """Traduce el estado acumulado de la llamada a la lectura que consume el motor.

    Solo pasa dimensiones **confirmadas**: un valor que el extractor marcó como ambiguo no
    entra al triaje ni como severidad ni como normalidad. Se pierde en la capa 5 (puntúa
    como normal) y se recupera en la capa 7 si además está bloqueado, que es el reparto
    correcto: lo ambiguo no alarma por sí solo, lo insistentemente ausente sí.
    """
    valores: dict[str, object] = {}
    for nombre in ("dolor", "fiebre", "movilidad", "herida", "apetito", "sueno"):
        obs = getattr(estado, nombre)
        if obs.confirmada:
            valores[nombre] = obs.valor

    presentes = tuple(
        nombre for nombre in BANDERAS if getattr(estado.banderas, nombre).valor == TriEstado.PRESENTE
    )

    return LecturaClinica(
        **valores,  # type: ignore[arg-type]
        bloqueadas=frozenset(
            d for d in DIMENSIONES_CRITICAS if estado.slot_bloqueado(d)
        ),
        banderas_extraidas=presentes + tuple(banderas_procedimiento),
        cobertura=estado.cobertura,
    )


# Tabla exportada para el harness y los tests: mantiene una sola fuente de verdad sobre el
# orden de severidad de las dimensiones categóricas.
ORDENES = {
    "movilidad": ORDEN_MOVILIDAD,
    "herida": ORDEN_HERIDA,
    "apetito": ORDEN_APETITO,
    "sueno": ORDEN_SUENO,
}

# LÍMITE QUE HAY QUE DECIR EN VOZ ALTA
#
# Estos umbrales están calibrados contra 160 casos sintéticos salidos de un mismo
# generador. Que la separación sea perfecta (0 falsos negativos, 0 falsos positivos en la
# regla roja) es tanto un mérito de la regla como una propiedad del generador: en
# pacientes reales las distribuciones se solapan y esa limpieza no se va a repetir.
#
# Por eso el sistema no descansa solo en esta calibración. Las capas 1, 2 y 7 —banderas
# reflejas sobre el habla, banderas extraídas e incertidumbre— no dependen de ningún
# umbral ajustado al dataset, y son las que sostendrían el triaje si los umbrales
# resultaran estar mal puestos. Validarlos con un cirujano es lo primero que habría que
# hacer con más tiempo, y es lo único de la lista que no se puede hacer en solitario.
