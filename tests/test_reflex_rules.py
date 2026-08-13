"""Vía refleja: lo que debe disparar, lo que no, y las dos formas en que fallaba.

Estos casos salen de la calibración contra `dataset_final.xlsx` (capa1_limpia, 160 casos).
No son ejemplos inventados: cada bloque fija un fallo que el dataset expuso y que costaba
recall de `rojo` o falsos positivos sobre casos `verde`.
"""

from __future__ import annotations

import pytest

from agente_postop.clinical.models import Criticidad
from agente_postop.clinical.reflex_engine import evaluar_via_refleja
from agente_postop.clinical.reflex_rules import extraer_temperatura_c

PROC = "Colecistectomía"


# El paciente rara vez dice "grados". Exigir la unidad perdía la fiebre entera, y con ella
# 5 de los 7 casos rojo que la vía refleja no detectaba.
@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("me tomaron la temperatura y creo que marcó como 38, algo así", 38.0),
        ("me la tomé y marcaba como 39 algo, no me acuerdo bien", 39.0),
        ("me sentí como afiebrada, tenía el cuerpo caliente... creo que como 38", 38.0),
        ("la tomé y marcaba como 38 y algo, pero yo creo que es del calor de la casa", 38.0),
        ("me tomé la temperatura y marcó 38.2", 38.2),
        ("tengo 38,5 de temperatura", 38.5),
        ("la fiebre me subió a 40", 40.0),
        # Forma explícita: sigue funcionando sin contexto térmico alrededor.
        ("marcó 38.5 grados", 38.5),
        ("tenía 39°", 39.0),
    ],
)
def test_extrae_temperatura_coloquial(texto: str, esperado: float) -> None:
    assert extraer_temperatura_c(texto) == esperado


@pytest.mark.parametrize(
    "texto",
    [
        # Intensidad de dolor: el número no es una temperatura y cae fuera del rango.
        "el dolor será un 5 más o menos, pero uno aguanta",
        "creo que un 6, más o menos, ahorita me duele bastante",
        # Sin ninguna mención térmica no hay de dónde inferir que 38 sea una temperatura.
        "llevo como 38 días esperando la cita",
        # Febrícula: se extrae, pero no llega al umbral (lo comprueba el test de abajo).
        "no le he puesto cuidado a eso, no tengo termómetro en la casa",
    ],
)
def test_no_inventa_temperatura(texto: str) -> None:
    valor = extraer_temperatura_c(texto)
    assert valor is None or valor < 38.0


def test_febricula_no_dispara_el_reflejo() -> None:
    """37.x se extrae correctamente pero queda por debajo del umbral de 38."""
    assert extraer_temperatura_c("me tomé la temperatura, marcó como 37 y algo") == 37.0
    assert not evaluar_via_refleja("me tomé la temperatura, marcó como 37 y algo", PROC).disparada


def test_fiebre_dispara_rojo() -> None:
    bandera = evaluar_via_refleja("me tomaron la temperatura y marcó como 38, algo así", PROC)
    assert bandera.disparada
    assert bandera.criticidad_forzada is Criticidad.ROJO


# El paciente describe el drenaje con sus palabras — "un líquido, amarillo creo" — nunca
# con el término clínico "secreción purulenta".
@pytest.mark.parametrize(
    "texto",
    [
        "mi hija me dijo que vio como un líquido, amarillo creo, saliendo ahí de la herida",
        "la he visto como con un líquido, amarillo creo, saliendo de ahí",
        "sí le sale un poquito de líquido ahí, como amarillito, pero no es mucho",
        "le está saliendo pus",
        "la herida huele feo",
    ],
)
def test_secrecion_en_palabras_del_paciente_dispara(texto: str) -> None:
    assert evaluar_via_refleja(texto, PROC).disparada


# El turno donde el paciente DESCARTA el síntoma es justo el que no debe alertar. Estos
# textos son de casos etiquetados `verde` que la vía refleja escalaba a rojo.
@pytest.mark.parametrize(
    "texto",
    [
        "la toco suavemente y no veo pus ni nada raro",
        "no le sale nada de líquido ni pus, solo un poquito rojita",
        "se veía normal, sin nada raro, ni rojo ni con esos líquidos feos. No huele mal tampoco",
        "nada de esas cosas de pus ni nada raro, yo creo que es normal de la cicatrización",
        "no he visto que salga pus ni nada raro",
        # 'pus' como subcadena: estas dos no hablan de secreción.
        "en el hospital me pusieron suero y antibiótico",
        "no le puse mucho cuidado a la herida, la verdad",
    ],
)
def test_negacion_no_dispara(texto: str) -> None:
    assert not evaluar_via_refleja(texto, PROC).disparada


# La negación mira solo hacia atrás, así que las reglas que llevan el "no" dentro del
# patrón siguen disparando.
@pytest.mark.parametrize(
    "texto",
    [
        "doctora, no puedo respirar bien",
        "la herida no para de sangrar",
        "mi mamá no reacciona, está muy rara",
    ],
)
def test_negacion_no_desactiva_reglas_que_contienen_no(texto: str) -> None:
    assert evaluar_via_refleja(texto, PROC).disparada


def test_negacion_solo_alcanza_lo_que_tiene_pegado() -> None:
    """Un "no" lejano no debe anular un hallazgo real que viene después."""
    assert evaluar_via_refleja("no sé qué será, pero ayer me salió pus por la herida", PROC).disparada
