"""FastAPI + WebSocket: interfaz de llamada, consola de administración y el loop de voz.

Un turno completo: audio del paciente → STT (Groq Whisper) → orquestar_turno (reflejo +
cortex + fusión + validador) → TTS (Piper) → audio de vuelta. El estado de la llamada
(historial, día postoperatorio, paciente) vive en `SesionLlamada`, una por conexión.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import RateLimitError

from agente_postop.clinical.estado import EstadoClinicoLlamada, comparar_estados
from agente_postop.clinical.memory import MotivoCierre, ResumenLlamada, guardar_resumen, timestamp_actual, ultimo_resumen
from agente_postop.clinical.models import SBAR, Criticidad
from agente_postop.console.api import router as console_router
from agente_postop.orchestrator.turn_manager import orquestar_turno
from agente_postop.voice.stt_groq import transcribir
from agente_postop.voice.tts import sintetizar_wav

MENSAJE_CUPO_AGOTADO = (
    "Se me acabó el tiempo disponible por ahora — alguien del equipo lo va a contactar "
    "pronto para seguir el control. Muchas gracias por su paciencia."
)
MENSAJE_ERROR_TECNICO = (
    "Uy, tuve un problema técnico justo ahora. Alguien del equipo lo va a llamar para "
    "continuar el seguimiento. Disculpe la molestia."
)

logger = logging.getLogger("agente_postop")

app = FastAPI(title="Agente de voz postoperatorio")
app.include_router(console_router)

DIR_ESTATICOS = Path(__file__).resolve().parents[1] / "console" / "static"
app.mount("/consola/static", StaticFiles(directory=DIR_ESTATICOS), name="consola-static")

DIR_WEBUI = Path(__file__).resolve().parents[1] / "webui"
app.mount("/webui/static", StaticFiles(directory=DIR_WEBUI), name="webui-static")

DIR_FILLERS = Path(__file__).resolve().parents[1] / "voice" / "fillers"
app.mount("/fillers/static", StaticFiles(directory=DIR_FILLERS), name="fillers-static")


@app.get("/")
def interfaz_llamada():
    return FileResponse(DIR_WEBUI / "llamada.html")


@app.get("/consola")
def consola():
    return FileResponse(DIR_ESTATICOS / "index.html")


@dataclass
class SesionLlamada:
    paciente_id: str
    procedimiento: str
    dia_postop: int
    turnos: list[str] = field(default_factory=list)
    estado_clinico: EstadoClinicoLlamada = field(default_factory=EstadoClinicoLlamada)
    criticidad_maxima: Criticidad = Criticidad.VERDE
    ultimo_sbar: SBAR | None = None

    def historial_texto(self) -> str:
        return "\n".join(self.turnos[-6:]) if self.turnos else "(inicio de la llamada)"


def _guardar_memoria_sesion(sesion: SesionLlamada, motivo_cierre: MotivoCierre) -> None:
    if not sesion.turnos:
        return
    anterior = ultimo_resumen(sesion.paciente_id)
    delta_vs_anterior = comparar_estados(anterior.estado_final, sesion.estado_clinico) if anterior else []
    if motivo_cierre == MotivoCierre.COMPLETADA and sesion.criticidad_maxima == Criticidad.ROJO:
        motivo_cierre = MotivoCierre.ESCALADA_INMEDIATA
    guardar_resumen(
        ResumenLlamada(
            paciente_id=sesion.paciente_id,
            dia_postop=sesion.dia_postop,
            fecha=timestamp_actual(),
            estado_final=sesion.estado_clinico,
            criticidad_final=sesion.criticidad_maxima,
            alerta_generada=sesion.criticidad_maxima != Criticidad.VERDE,
            resumen_texto=sesion.historial_texto(),
            cobertura=sesion.estado_clinico.cobertura,
            dimensiones_no_evaluadas=sesion.estado_clinico.dimensiones_pendientes,
            sbar=sesion.ultimo_sbar,
            delta_vs_llamada_anterior=delta_vs_anterior,
            motivo_cierre=motivo_cierre,
        )
    )


async def _cerrar_con_mensaje(websocket: WebSocket, texto_paciente: str, mensaje: str) -> None:
    """Cierra la llamada con un mensaje audible en vez de matar la conexión en silencio
    — un error (cupo de Groq agotado, falla técnica) no debe dejar al paciente esperando
    sin respuesta; hay que colgar avisando, no colgar en silencio."""
    audio = sintetizar_wav(mensaje)
    await websocket.send_json(
        {
            "texto_paciente_transcrito": texto_paciente,
            "respuesta_hablada": mensaje,
            "criticidad_final": Criticidad.DESCONOCIDA.value,
            "reflejo_vetea": False,
            "afirmaciones_clinicas": [],
            "cobertura": 0.0,
            "verde_bloqueado_por_cobertura": False,
            "sbar": None,
            "llamada_finalizada": True,
        }
    )
    await websocket.send_bytes(audio)
    await websocket.close()


@app.websocket("/ws/llamada")
async def llamada(websocket: WebSocket):
    await websocket.accept()

    mensaje_inicial = await websocket.receive_json()
    sesion = SesionLlamada(
        paciente_id=mensaje_inicial["paciente_id"],
        procedimiento=mensaje_inicial["procedimiento"],
        dia_postop=int(mensaje_inicial["dia_postop"]),
    )

    try:
        while True:
            mensaje = await websocket.receive()

            if mensaje.get("type") == "websocket.disconnect":
                # receive() no siempre lanza WebSocketDisconnect al desconectar el
                # cliente (ej. al cerrar la pestaña) — a veces solo devuelve este
                # mensaje. Si no lo detectamos aquí, el except de abajo nunca corre y
                # la memoria longitudinal de la llamada no se guarda.
                raise WebSocketDisconnect

            if "bytes" in mensaje and mensaje["bytes"] is not None:
                texto_paciente = transcribir(mensaje["bytes"])
            elif "text" in mensaje and mensaje["text"] is not None:
                # Bypass de texto directo — usado por el harness de evaluación.
                texto_paciente = json.loads(mensaje["text"])["texto"]
            else:
                continue

            es_primer_turno = len(sesion.turnos) == 0
            turno_idx = len(sesion.turnos) // 2
            sesion.turnos.append(f"paciente: {texto_paciente}")

            try:
                resultado = orquestar_turno(
                    turno_paciente=texto_paciente,
                    paciente_id=sesion.paciente_id,
                    procedimiento=sesion.procedimiento,
                    dia_postop=sesion.dia_postop,
                    historial_turno=sesion.historial_texto(),
                    estado_clinico=sesion.estado_clinico,
                    turno_idx=turno_idx,
                    es_primer_turno_de_la_llamada=es_primer_turno,
                )
            except RateLimitError:
                logger.warning("cupo de Groq agotado — cerrando la llamada con aviso, paciente=%s", sesion.paciente_id)
                await _cerrar_con_mensaje(websocket, texto_paciente, MENSAJE_CUPO_AGOTADO)
                _guardar_memoria_sesion(sesion, MotivoCierre.ERROR_TECNICO)
                return
            except Exception as exc:  # noqa: BLE001 — cualquier falla del turno cuelga avisando, no en silencio
                logger.exception("error inesperado procesando el turno, paciente=%s: %s", sesion.paciente_id, exc)
                await _cerrar_con_mensaje(websocket, texto_paciente, MENSAJE_ERROR_TECNICO)
                _guardar_memoria_sesion(sesion, MotivoCierre.ERROR_TECNICO)
                return

            sesion.turnos.append(f"agente: {resultado.respuesta_hablada}")
            if resultado.criticidad_final.rango > sesion.criticidad_maxima.rango:
                sesion.criticidad_maxima = resultado.criticidad_final
            if resultado.sbar is not None:
                sesion.ultimo_sbar = resultado.sbar

            audio_respuesta = sintetizar_wav(resultado.respuesta_hablada)

            await websocket.send_json(
                {
                    "texto_paciente_transcrito": texto_paciente,
                    "respuesta_hablada": resultado.respuesta_hablada,
                    "criticidad_final": resultado.criticidad_final.value,
                    "reflejo_vetea": resultado.reflejo_vetea,
                    "afirmaciones_clinicas": [a.model_dump() for a in resultado.afirmaciones_clinicas],
                    "cobertura": resultado.cobertura,
                    "verde_bloqueado_por_cobertura": resultado.verde_bloqueado_por_cobertura,
                    "sbar": resultado.sbar.model_dump() if resultado.sbar else None,
                }
            )
            await websocket.send_bytes(audio_respuesta)

    except WebSocketDisconnect:
        _guardar_memoria_sesion(sesion, MotivoCierre.COMPLETADA)
