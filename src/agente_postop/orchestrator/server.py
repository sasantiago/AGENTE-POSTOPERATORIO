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

from agente_postop.clinical.memory import ResumenLlamada, guardar_resumen, timestamp_actual
from agente_postop.clinical.models import Criticidad
from agente_postop.console.api import router as console_router
from agente_postop.orchestrator.turn_manager import orquestar_turno
from agente_postop.voice.stt_groq import transcribir
from agente_postop.voice.tts import sintetizar_wav

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
    sintomas_reportados: dict[str, str] = field(default_factory=dict)
    criticidad_maxima: Criticidad = Criticidad.VERDE

    def historial_texto(self) -> str:
        return "\n".join(self.turnos[-6:]) if self.turnos else "(inicio de la llamada)"


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
            sesion.turnos.append(f"paciente: {texto_paciente}")

            resultado = orquestar_turno(
                turno_paciente=texto_paciente,
                paciente_id=sesion.paciente_id,
                procedimiento=sesion.procedimiento,
                dia_postop=sesion.dia_postop,
                historial_turno=sesion.historial_texto(),
                sintomas_extraidos=None,
                es_primer_turno_de_la_llamada=es_primer_turno,
            )

            sesion.turnos.append(f"agente: {resultado.respuesta_hablada}")
            if resultado.criticidad_final.rango > sesion.criticidad_maxima.rango:
                sesion.criticidad_maxima = resultado.criticidad_final

            audio_respuesta = sintetizar_wav(resultado.respuesta_hablada)

            await websocket.send_json(
                {
                    "texto_paciente_transcrito": texto_paciente,
                    "respuesta_hablada": resultado.respuesta_hablada,
                    "criticidad_final": resultado.criticidad_final.value,
                    "reflejo_vetea": resultado.reflejo_vetea,
                    "afirmaciones_clinicas": [a.model_dump() for a in resultado.afirmaciones_clinicas],
                }
            )
            await websocket.send_bytes(audio_respuesta)

    except WebSocketDisconnect:
        if sesion.turnos:
            guardar_resumen(
                ResumenLlamada(
                    paciente_id=sesion.paciente_id,
                    dia_postop=sesion.dia_postop,
                    fecha=timestamp_actual(),
                    sintomas_reportados=sesion.sintomas_reportados,
                    criticidad_final=sesion.criticidad_maxima.value,
                    alerta_generada=sesion.criticidad_maxima != Criticidad.VERDE,
                    resumen_texto=sesion.historial_texto(),
                )
            )
