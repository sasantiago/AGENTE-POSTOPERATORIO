"""Motor de triaje: las invariantes que no se pueden romper sin darse cuenta.

Estos tests no comprueban la exactitud —eso lo hace `harness/eval_triaje.py` contra los
160 casos etiquetados, y falla el build si aparece una subestimación—. Comprueban las
propiedades estructurales: que ninguna capa pueda bajar el nivel, que el modelo no pueda
tranquilizar, y que la ignorancia no se lea como normalidad.

Es la distinción que importa: la exactitud depende de umbrales que un clínico puede querer
mover, y moverlos debe ser posible. Las invariantes no dependen de ningún umbral, y
romperlas es siempre un error.
"""

from __future__ import annotations

import pytest

from agente_postop.clinical.models import BanderaRefleja, Criticidad
from agente_postop.clinical.trajectory_twin import Desviacion
from agente_postop.clinical.triage import (
    DIMENSIONES_CRITICAS,
    LecturaClinica,
    evaluar,
    puntaje_ordinal,
)

SANA = dict(dolor=1.0, fiebre=36.6, movilidad="normal", herida="normal", apetito="normal", sueno="normal")


def lectura(**cambios) -> LecturaClinica:
    return LecturaClinica(**{**SANA, **cambios})


# --- La regla ROJO ---------------------------------------------------------------


@pytest.mark.parametrize(
    "cambios, regla_esperada",
    [
        ({"fiebre": 38.0}, "fiebre_alta"),
        ({"fiebre": 39.5}, "fiebre_alta"),
        ({"dolor": 8.0}, "dolor_severo"),
        ({"dolor": 10.0}, "dolor_severo"),
        ({"herida": "secrecion_purulenta"}, "herida_purulenta"),
    ],
)
def test_regla_roja_escala(cambios: dict, regla_esperada: str) -> None:
    decision = evaluar(lectura(**cambios), perfil="optimo")
    assert decision.nivel == Criticidad.ROJO
    assert decision.escalado_por == regla_esperada


@pytest.mark.parametrize("fiebre, dolor", [(37.9, 7.0), (37.5, 6.0), (36.8, 7.9)])
def test_justo_por_debajo_del_umbral_no_es_rojo(fiebre: float, dolor: float) -> None:
    """La frontera importa: 37.9 °C es la temperatura máxima de un caso no rojo en el
    dataset, así que el umbral de 38.0 se apoya en un margen de exactamente 0.1 °C."""
    assert evaluar(lectura(fiebre=fiebre, dolor=dolor), perfil="optimo").nivel != Criticidad.ROJO


# --- La invariante: el nivel solo sube -------------------------------------------


def test_el_modelo_no_puede_bajar_la_criticidad() -> None:
    """Es la razón de existir del módulo. El LLM propone verde sobre un caso con fiebre de
    39 °C; la decisión sigue siendo roja."""
    decision = evaluar(lectura(fiebre=39.0), criticidad_llm=Criticidad.VERDE)
    assert decision.nivel == Criticidad.ROJO


def test_el_modelo_si_puede_subir_la_criticidad() -> None:
    """La capa 8 existe para lo que las reglas no cubren: el paciente dice algo que ninguna
    bandera empareja y el modelo lo entiende."""
    decision = evaluar(lectura(), criticidad_llm=Criticidad.ROJO, exigir_cobertura_para_verde=False)
    assert decision.nivel == Criticidad.ROJO
    assert decision.escalado_por == "segunda_opinion_modelo"


def test_el_reflejo_no_puede_bajar_un_rojo_de_los_slots() -> None:
    reflejo_apagado = BanderaRefleja(disparada=False, criticidad_forzada=Criticidad.VERDE)
    assert evaluar(lectura(fiebre=38.5), bandera_reflejo=reflejo_apagado).nivel == Criticidad.ROJO


def test_la_bandera_refleja_escala_sin_ayuda_de_los_slots() -> None:
    """Los seis slots normales, pero el paciente dijo que se le abrió la herida. Ninguna
    regla numérica lo ve; la vía refleja sí."""
    reflejo = BanderaRefleja(
        disparada=True, criticidad_forzada=Criticidad.ROJO, regla="Dehiscencia de la herida"
    )
    decision = evaluar(lectura(), bandera_reflejo=reflejo)
    assert decision.nivel == Criticidad.ROJO
    assert "Dehiscencia" in decision.motivos[0].detalle


def test_bandera_extraida_por_el_modelo_escala() -> None:
    decision = evaluar(lectura(banderas_extraidas=("dificultad_respiratoria",)))
    assert decision.nivel == Criticidad.ROJO
    assert decision.escalado_por == "bandera_extraida"


# --- Incertidumbre: no saber nunca es una buena noticia --------------------------


@pytest.mark.parametrize("dimension", DIMENSIONES_CRITICAS)
def test_dimension_critica_bloqueada_escala_a_desconocida(dimension: str) -> None:
    """El agente preguntó dos veces por la fiebre y no obtuvo respuesta. Eso no es un
    paciente sin fiebre: es un paciente sobre el que no sabemos si la tiene."""
    decision = evaluar(
        lectura(**{dimension: None}, bloqueadas=frozenset({dimension})),
        exigir_cobertura_para_verde=False,
    )
    assert decision.nivel == Criticidad.DESCONOCIDA
    assert decision.escalado_por == "incertidumbre_critica"


def test_dimension_critica_ausente_pero_no_bloqueada_no_escala() -> None:
    """Distinción deliberada: a mitad de la llamada aún no se ha preguntado todo, y eso no
    puede disparar una alerta en cada turno. Solo escala lo insistentemente ausente."""
    decision = evaluar(lectura(fiebre=None), exigir_cobertura_para_verde=False)
    assert decision.nivel == Criticidad.VERDE


def test_cobertura_incompleta_bloquea_el_verde() -> None:
    """El verde exige evidencia positiva de ausencia de alarma, no ausencia de evidencia."""
    decision = evaluar(lectura(cobertura=0.5), exigir_cobertura_para_verde=True)
    assert decision.nivel == Criticidad.DESCONOCIDA
    assert decision.escalado_por == "cobertura_incompleta"


def test_cobertura_incompleta_no_rebaja_un_rojo() -> None:
    decision = evaluar(lectura(fiebre=38.9, cobertura=0.3), exigir_cobertura_para_verde=True)
    assert decision.nivel == Criticidad.ROJO


# --- Puntaje ordinal --------------------------------------------------------------


def test_puntaje_acumula_sintomas_leves() -> None:
    """Ningún síntoma cruza un umbral por sí solo, pero cuatro a la vez sí escalan. Es el
    paciente que va empeorando sin disparar ninguna alarma individual."""
    leve = lectura(dolor=5.0, herida="eritema_leve", apetito="muy_disminuido", sueno="muy_alterado")
    assert puntaje_ordinal(leve) >= 4
    assert evaluar(leve, perfil="optimo").nivel == Criticidad.AMARILLO


def test_dimensiones_ausentes_puntuan_como_normales() -> None:
    """El puntaje detecta acumulación; no puede inventar severidad de lo que no se
    preguntó. La ignorancia se trata en la capa 7, no aquí."""
    assert puntaje_ordinal(LecturaClinica()) == 0


# --- Perfiles ---------------------------------------------------------------------


def test_perfil_conservador_escala_la_febricula_y_el_optimo_no() -> None:
    febricula = lectura(fiebre=37.6)
    assert evaluar(febricula, perfil="conservador").nivel == Criticidad.AMARILLO
    assert evaluar(febricula, perfil="optimo").nivel == Criticidad.VERDE


def test_perfil_conservador_escala_la_movilidad_incapacitante() -> None:
    assert evaluar(lectura(movilidad="incapacitante_nueva"), perfil="conservador").nivel == Criticidad.AMARILLO


def test_ningun_perfil_cambia_la_regla_roja() -> None:
    """Los perfiles negocian falsos positivos, nunca falsos negativos."""
    for perfil in ("conservador", "optimo"):
        assert evaluar(lectura(fiebre=38.2), perfil=perfil).nivel == Criticidad.ROJO


# --- Capa de trayectoria ----------------------------------------------------------


def test_una_sola_desviacion_no_escala() -> None:
    una = [Desviacion("dolor", "5", "2", empeora=True)]
    assert evaluar(lectura(), desviaciones=una, exigir_cobertura_para_verde=False).nivel == Criticidad.VERDE


def test_desviaciones_multiples_escalan() -> None:
    varias = [Desviacion(d, "peor", "normal", empeora=True) for d in ("dolor", "movilidad", "sueno")]
    decision = evaluar(lectura(), desviaciones=varias, exigir_cobertura_para_verde=False)
    assert decision.nivel == Criticidad.AMARILLO
    assert decision.escalado_por == "desviacion_trayectoria"


def test_desviaciones_que_no_empeoran_se_ignoran() -> None:
    mejores = [Desviacion(d, "mejor", "normal", empeora=False) for d in ("dolor", "movilidad", "sueno")]
    assert evaluar(lectura(), desviaciones=mejores, exigir_cobertura_para_verde=False).nivel == Criticidad.VERDE


# --- Auditabilidad ----------------------------------------------------------------


def test_toda_decision_no_verde_trae_al_menos_un_motivo() -> None:
    for cambios in ({"fiebre": 38.5}, {"dolor": 9.0}, {"herida": "secrecion_purulenta"},
                    {"fiebre": 37.7}, {"cobertura": 0.2}):
        decision = evaluar(lectura(**cambios))
        if decision.nivel != Criticidad.VERDE:
            assert decision.motivos, f"{cambios} escaló sin dejar motivo"
            assert decision.escalado_por is not None


def test_la_decision_es_serializable_para_el_log() -> None:
    decision = evaluar(lectura(fiebre=38.4, herida="eritema_leve"))
    d = decision.to_dict()
    assert d["nivel"] == "rojo"
    assert d["escalado_por"] == "fiebre_alta"
    assert all({"regla", "detalle", "nivel"} <= set(m) for m in d["motivos"])


def test_la_funcion_es_pura() -> None:
    """Misma entrada, misma salida — es lo que un modelo no garantiza y una regla sí."""
    entrada = lectura(fiebre=37.8, dolor=6.0, herida="eritema_leve")
    assert evaluar(entrada).to_dict() == evaluar(entrada).to_dict()
