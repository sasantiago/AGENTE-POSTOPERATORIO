"""Cronometraje por etapa de un turno de voz.

La rúbrica (§5) define la latencia como el tiempo **desde que el paciente termina de hablar
hasta que empieza a sonar el audio del agente**, y exige que lo reportado sea verificable
en los logs. Eso es el `ttfa_ms` de aquí (time to first audio) — la suma de STT, la
orquestación y la síntesis de lo primero que se emite.

El harness mide otra cosa: solo la orquestación, porque inyecta texto y nunca toca el
micrófono ni el altavoz. Las dos son legítimas, pero no son la misma, y confundirlas es lo
que hace que un número del README no cuadre con el cronómetro del jurado. Por eso se
nombran distinto y se reportan por separado.

Uso:

    with medir_turno() as medicion:
        with medicion.etapa("stt"):
            texto = transcribir(audio)
        ...
        medicion.marcar_primer_audio()
    logger.info("turno medido", extra=medicion.como_dict())
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator

# El turno se orquesta en un ThreadPoolExecutor (extracción y conversación en paralelo).
# `ThreadPoolExecutor.submit` no propaga el contexto solo — hay que copiarlo explícitamente
# con `contextvars.copy_context().run(...)`, y eso lo hace `turn_manager`.
_MEDICION_ACTUAL: contextvars.ContextVar["MedicionTurno | None"] = contextvars.ContextVar(
    "medicion_turno_actual", default=None
)


@dataclass
class MedicionTurno:
    """Milisegundos por etapa de un turno. Las etapas concurrentes se suman por separado:
    `rag_ms` y `llm_ms` corren dentro de `orquestacion_ms`, no se le añaden."""

    etapas_ms: dict[str, float] = field(default_factory=dict)
    inicio: float = field(default_factory=perf_counter)
    ttfa_ms: float | None = None
    ttfr_ms: float | None = None

    @contextmanager
    def etapa(self, nombre: str) -> Iterator[None]:
        arranque = perf_counter()
        try:
            yield
        finally:
            # Acumula en vez de sobrescribir: un turno puede sintetizar varias frases y
            # cada una suma a `tts_ms`.
            transcurrido = (perf_counter() - arranque) * 1000
            self.etapas_ms[nombre] = self.etapas_ms.get(nombre, 0.0) + transcurrido

    def marcar_primer_audio(self) -> None:
        """El instante que la rúbrica cronometra: cuándo empieza a sonar algo. Solo cuenta
        la primera vez."""
        if self.ttfa_ms is None:
            self.ttfa_ms = (perf_counter() - self.inicio) * 1000

    def marcar_audio_de_respuesta(self) -> None:
        """Cuándo empieza a sonar la respuesta CON CONTENIDO.

        Se mide aparte de `ttfa_ms` a propósito. Si el turno emitió un filler de espera,
        `ttfa_ms` mide el filler —medio segundo— y publicar solo ese número sería maquillar
        la latencia: el paciente sigue esperando el resto para obtener su respuesta. Las dos
        cifras dicen cosas distintas y verdaderas: cuánto tarda en dejar de haber silencio,
        y cuánto tarda en haber respuesta.
        """
        self.marcar_primer_audio()
        if self.ttfr_ms is None:
            self.ttfr_ms = (perf_counter() - self.inicio) * 1000

    @property
    def total_ms(self) -> float:
        return (perf_counter() - self.inicio) * 1000

    def como_dict(self) -> dict[str, float]:
        datos = {f"{nombre}_ms": round(valor, 1) for nombre, valor in self.etapas_ms.items()}
        datos["total_ms"] = round(self.total_ms, 1)
        if self.ttfa_ms is not None:
            datos["ttfa_ms"] = round(self.ttfa_ms, 1)
        if self.ttfr_ms is not None:
            datos["ttfr_ms"] = round(self.ttfr_ms, 1)
        return datos

    def como_linea_log(self) -> str:
        return " ".join(f"{clave}={valor:.0f}" for clave, valor in self.como_dict().items())


@contextmanager
def medir_turno() -> Iterator[MedicionTurno]:
    medicion = MedicionTurno()
    token = _MEDICION_ACTUAL.set(medicion)
    try:
        yield medicion
    finally:
        _MEDICION_ACTUAL.reset(token)


@contextmanager
def cronometrar(nombre: str) -> Iterator[None]:
    """Cronometra una etapa contra la medición del turno en curso, si la hay.

    Fuera de un `medir_turno()` no hace nada: así el harness y los tests pueden llamar al
    cortex sin montar ninguna instrumentación.
    """
    medicion = _MEDICION_ACTUAL.get()
    if medicion is None:
        yield
        return
    with medicion.etapa(nombre):
        yield
