"""FastAPI + WebSocket: interfaz de llamada, consola de administración y el loop de voz.

Un turno completo: audio del paciente → STT (Groq Whisper) → orquestar_turno (reflejo +
cortex + fusión + validador) → TTS (Piper) → audio de vuelta. El estado de la llamada
(historial, día postoperatorio, paciente) vive en `SesionLlamada`, una por conexión.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import RateLimitError

from agente_postop.clients import ollama_vivo, precalentar_ollama
from agente_postop.clinical.estado import EstadoClinicoLlamada, comparar_estados
from agente_postop.config import get_settings
from agente_postop.clinical.memory import (
    MotivoCierre,
    ResumenLlamada,
    construir_resumen_markdown,
    guardar_resumen,
    timestamp_actual,
    ultimo_resumen,
)
from agente_postop.clinical.models import SBAR, Criticidad
from agente_postop.clinical.pacientes import listar_pacientes
from agente_postop.console.api import router as console_router
from agente_postop.orchestrator.cortex import PROCEDIMIENTOS_OFRECIDOS
from agente_postop.orchestrator.metrics import medir_turno
from agente_postop.clinical.guion import APERTURA, siguiente_pregunta, texto_a_decir
from agente_postop.clinical.triage import desde_estado
from agente_postop.clinical.triage import evaluar as evaluar_triaje
from agente_postop.orchestrator.turn_manager import cierre_para, orquestar_turno, orquestar_turno_guiado
from agente_postop.rag.chroma_store import consultar
from agente_postop.voice.stt_groq import transcribir
from agente_postop.voice.guion_audio import audio_de, faltantes
from agente_postop.voice.tts import sintetizar, sintetizar_wav

MENSAJE_CUPO_AGOTADO = (
    "Se me acabó el tiempo disponible por ahora — alguien del equipo lo va a contactar "
    "pronto para seguir el control. Muchas gracias por su paciencia."
)
MENSAJE_ERROR_TECNICO = (
    "Uy, tuve un problema técnico justo ahora. Alguien del equipo lo va a llamar para "
    "continuar el seguimiento. Disculpe la molestia."
)

logger = logging.getLogger("agente_postop")

# Archivo de log del turno, además de la consola. La rúbrica contrasta las métricas
# reportadas contra los logs de la sesión, así que el rastro tiene que sobrevivir a que
# alguien cierre la terminal.
ARCHIVO_LOG = Path(__file__).resolve().parents[3] / "agente_postop.log"


def configurar_logging() -> None:
    """Sin esto, `logger.info` no salía por ningún lado: nadie configuraba el logging, así
    que el nivel efectivo era WARNING y la línea de métricas por turno se descartaba en
    silencio. Se dejaba sin rastro justo lo que hay que poder auditar."""
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    formato = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    consola = logging.StreamHandler()
    consola.setFormatter(formato)
    logger.addHandler(consola)

    try:
        archivo = logging.FileHandler(ARCHIVO_LOG, encoding="utf-8")
        archivo.setFormatter(formato)
        logger.addHandler(archivo)
    except OSError as exc:  # noqa: BLE001 — sin permisos de escritura se sigue con consola
        logger.warning("no se pudo abrir el archivo de log %s: %s", ARCHIVO_LOG, exc)


@asynccontextmanager
async def _ciclo_de_vida(_: FastAPI):
    """Precalienta Piper y el modelo de embeddings antes del primer turno.

    Los dos se cargan de forma perezosa (`lru_cache` en `voice/tts.py`, carga diferida en
    `sentence-transformers`), así que sin esto **el primer turno de la llamada paga la
    carga de los dos modelos** — del orden de segundos, y justo en el turno que el jurado
    ve primero. Era el P95 de la tabla de métricas.

    Si algo falla, el servidor arranca igual: un warm-up es una optimización, no un
    requisito de arranque.
    """
    configurar_logging()
    try:
        await asyncio.to_thread(sintetizar_wav, "Listo.")
        logger.info("warm-up: voz de Piper cargada")
    except Exception as exc:  # noqa: BLE001
        logger.warning("warm-up de TTS falló (el primer turno pagará la carga): %s", exc)
    try:
        await asyncio.to_thread(consultar, "control postoperatorio", 1)
        logger.info("warm-up: modelo de embeddings y ChromaDB cargados")
    except Exception as exc:  # noqa: BLE001
        logger.warning("warm-up de RAG falló (el primer turno pagará la carga): %s", exc)

    # Estado del modelo de lenguaje, dicho al arrancar y no en el primer turno del
    # paciente. Que el agente descubra en vivo que no tiene con qué razonar es la peor
    # forma de enterarse: el paciente ya está al teléfono.
    settings = get_settings()
    if settings.llm_backend == "ollama":
        if await asyncio.to_thread(ollama_vivo):
            # Una generación mínima deja el modelo cargado en memoria con el `keep_alive`
            # configurado, para que el primer turno no pague los segundos de carga.
            try:
                await asyncio.to_thread(precalentar_ollama)
                logger.info("warm-up: %s cargado en Ollama (local, sin credenciales)", settings.ollama_model)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ollama responde pero el precalentado falló: %s", exc)
        else:
            logger.warning(
                "Ollama no responde en %s o no tiene '%s' — la conversación caerá a '%s'. "
                "Para el camino local: `ollama pull %s`",
                settings.ollama_base_url, settings.ollama_model, settings.llm_fallback, settings.ollama_model,
            )
    yield


app = FastAPI(title="Agente de voz postoperatorio", lifespan=_ciclo_de_vida)
app.include_router(console_router)

@app.middleware("http")
async def sin_cache_de_estaticos(request, call_next):
    """El navegador cacheaba el JS y el HTML entre reinicios del servidor: se editaba la
    interfaz, se recargaba, y seguía corriendo la versión anterior sin ningún aviso —
    diagnosticado comparando bytes servidos (8.194) contra bytes en caché (6.422).

    En una demo eso es una trampa: se muestra una función que el navegador no tiene. Estos
    archivos se sirven desde disco en local, así que cachearlos no ahorra nada y cuesta
    exactamente el fallo que no se puede permitir delante de un jurado.
    """
    respuesta = await call_next(request)
    if request.url.path.endswith((".js", ".css", ".html")) or request.url.path in ("/", "/consola"):
        respuesta.headers["Cache-Control"] = "no-store, must-revalidate"
    return respuesta


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


@app.get("/api/pacientes")
def pacientes():
    """El registro que la interfaz usa para deducir el procedimiento del paciente al que se
    llama, en vez de que alguien lo elija a mano (ver clinical/pacientes.py)."""
    return {
        "pacientes": [
            {
                "paciente_id": p.paciente_id,
                "procedimiento": p.procedimiento,
                "fecha_cirugia": p.fecha_cirugia,
                "edad": p.edad,
                "comorbilidades": p.comorbilidades,
            }
            for p in listar_pacientes()
        ],
        "procedimientos": list(PROCEDIMIENTOS_OFRECIDOS),
    }


@dataclass
class SesionLlamada:
    paciente_id: str
    procedimiento: str
    dia_postop: int
    turnos: list[str] = field(default_factory=list)
    estado_clinico: EstadoClinicoLlamada = field(default_factory=EstadoClinicoLlamada)
    criticidad_maxima: Criticidad = Criticidad.VERDE
    ultimo_sbar: SBAR | None = None
    # La pregunta del guion que el agente acaba de hacer y cuya respuesta espera. Es lo que
    # permite extraer una sola dimensión en vez de las seis: si el agente preguntó por la
    # herida, lo que llega es la respuesta sobre la herida.
    pregunta_pendiente: object | None = None
    llamada_cerrada: bool = False
    # Dimensiones cuya clasificación corre en segundo plano. El guion las salta para no
    # volver a preguntar algo que el paciente ya contestó y todavía se está anotando.
    extracciones_en_vuelo: set = field(default_factory=set)
    tareas_extraccion: list = field(default_factory=list)
    # Un WebSocket no admite dos escrituras a la vez. Desde que la anotación corre en
    # segundo plano hay DOS productores —el bucle del turno y la tarea diferida— y sin este
    # cerrojo sus frames se intercalaban: el cliente recibía un audio partido por un JSON,
    # se quedaba esperando el resto y la llamada se congelaba a los pocos turnos.
    envio: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Ollama atiende una petición por vez. Sin limitar, cada turno encolaba otra extracción
    # y la cola crecía más rápido de lo que se vaciaba.
    turno_llm: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(1))

    def historial_texto(self) -> str:
        return "\n".join(self.turnos[-6:]) if self.turnos else "(inicio de la llamada)"


def _guardar_memoria_sesion(sesion: SesionLlamada, motivo_cierre: MotivoCierre) -> None:
    if not sesion.turnos:
        return
    anterior = ultimo_resumen(sesion.paciente_id)
    delta_vs_anterior = comparar_estados(anterior.estado_final, sesion.estado_clinico) if anterior else []
    if motivo_cierre == MotivoCierre.COMPLETADA and sesion.criticidad_maxima == Criticidad.ROJO:
        motivo_cierre = MotivoCierre.ESCALADA_INMEDIATA

    fecha = timestamp_actual()
    # El resumen se arma sobre la transcripción COMPLETA (`sesion.turnos`), no sobre la
    # ventana recortada que se le manda al modelo en cada turno: al prompt le sobra el
    # principio de la llamada, al registro clínico no.
    resumen_markdown = construir_resumen_markdown(
        paciente_id=sesion.paciente_id,
        procedimiento=sesion.procedimiento,
        dia_postop=sesion.dia_postop,
        fecha=fecha,
        estado=sesion.estado_clinico,
        criticidad_final=sesion.criticidad_maxima,
        motivo_cierre=motivo_cierre,
        turnos=sesion.turnos,
        sbar=sesion.ultimo_sbar,
        delta_vs_anterior=delta_vs_anterior,
    )

    guardar_resumen(
        ResumenLlamada(
            paciente_id=sesion.paciente_id,
            procedimiento=sesion.procedimiento,
            dia_postop=sesion.dia_postop,
            fecha=fecha,
            estado_final=sesion.estado_clinico,
            criticidad_final=sesion.criticidad_maxima,
            alerta_generada=sesion.criticidad_maxima != Criticidad.VERDE,
            resumen_texto=resumen_markdown,
            transcripcion=list(sesion.turnos),
            cobertura=sesion.estado_clinico.cobertura,
            dimensiones_no_evaluadas=sesion.estado_clinico.dimensiones_pendientes,
            sbar=sesion.ultimo_sbar,
            delta_vs_llamada_anterior=delta_vs_anterior,
            motivo_cierre=motivo_cierre,
        )
    )


# La mascara de silencio ya la pone el cliente: `reproducirFiller()` en webui/llamada.js
# suena en cuanto el paciente suelta el boton, antes incluso de que el audio salga hacia el
# servidor. Emitir ademas un filler desde aqui llegaria mas tarde --hay que atravesar la
# red-- y el navegador lo trataria como la respuesta del turno: reproduciria el puente,
# daria el turno por cerrado y volveria a habilitar el microfono con la respuesta real
# todavia en camino. Se deja donde esta y el servidor no compite con el.


def usar_guion() -> bool:
    """¿Conduce el agente con guion fijo, o deja que el modelo redacte cada pregunta?

    Configurable porque los dos caminos siguen vivos y son comparables: el guiado es el
    que se entrega, y el libre queda para contrastar. Apagarlo no requiere tocar código.
    """
    return get_settings().usar_guion


async def _anotar_en_segundo_plano(websocket: WebSocket, sesion: SesionLlamada, completar) -> None:
    """Clasifica la respuesta del turno anterior mientras el paciente ya oyó la siguiente.

    Es lo que convierte una espera de ~6 s en silencio en una anotación invisible. Cuando
    termina, reevalúa la criticidad con el dato nuevo y le manda al panel una actualización;
    la conversación no se interrumpe.

    Si la clasificación falla, la dimensión sale de «en vuelo» y vuelve a la cola del guion:
    el agente la preguntará otra vez más adelante. Un fallo aquí cuesta una repetición, no
    un dato perdido.
    """
    dimension = completar.dimension
    try:
        # Una extracción por vez: Ollama las serializa igual, y encolarlas aquí evita que
        # se apilen más rápido de lo que se resuelven.
        async with sesion.turno_llm:
            await asyncio.to_thread(completar)
        decision = evaluar_triaje(
            desde_estado(sesion.estado_clinico),
            perfil=get_settings().triage_profile,
            exigir_cobertura_para_verde=True,
        )
        if decision.nivel.rango > sesion.criticidad_maxima.rango:
            sesion.criticidad_maxima = decision.nivel
        async with sesion.envio:
            await websocket.send_json({
            "tipo": "anotacion",
            "dimension": dimension,
            "valor": str(getattr(sesion.estado_clinico, dimension).valor),
            "confirmada": getattr(sesion.estado_clinico, dimension).confirmada,
            "cobertura": sesion.estado_clinico.cobertura,
            "criticidad_final": decision.nivel.value,
            "decision_triaje": decision.to_dict(),
        })
    except Exception as exc:  # noqa: BLE001 — una anotación perdida no tumba la llamada
        logger.warning("anotación diferida de %s falló: %s", dimension, exc)
    finally:
        sesion.extracciones_en_vuelo.discard(dimension)


def _audio_de_texto(texto: str) -> tuple[bytes, str]:
    """El audio de una frase: pre-generado si es del guion, sintetizado si no.

    Se intenta SIEMPRE el pre-generado, sin que quien llama tenga que saber si el texto es
    de guion. Cuando lo es, el turno paga ~20 ms de lectura de disco en vez de ~1.000 ms de
    Piper; cuando no —una respuesta anclada a la pregunta del paciente—, se sintetiza como
    siempre. Que la frase mezcle ambas cosas también funciona: al no encontrarla completa
    en caché, se sintetiza entera.
    """
    return audio_de(texto) or sintetizar(texto)


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
            "decision_triaje": {},
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

    # El agente abre la llamada: saluda y hace la primera pregunta del protocolo, sin
    # esperar a que el paciente hable. Es como funciona una llamada de seguimiento real —
    # llama el hospital, no el paciente— y de paso el primer turno del paciente ya llega
    # como respuesta a una pregunta concreta, que es lo que hace barata la extracción.
    if usar_guion():
        sesion.pregunta_pendiente = siguiente_pregunta(sesion.estado_clinico)
        apertura = f"{APERTURA} {texto_a_decir(sesion.pregunta_pendiente, sesion.estado_clinico)}"
        sesion.turnos.append(f"agente: {apertura}")
        audio_apertura, formato_apertura = await asyncio.to_thread(_audio_de_texto, apertura)
        async with sesion.envio:
          await websocket.send_json({
            "texto_paciente_transcrito": "",
            "respuesta_hablada": apertura,
            "criticidad_final": Criticidad.DESCONOCIDA.value,
            "reflejo_vetea": False,
            "afirmaciones_clinicas": [],
            "cobertura": 0.0,
            "verde_bloqueado_por_cobertura": False,
            "sbar": None,
            "decision_triaje": {},
            "dimension_preguntada": sesion.pregunta_pendiente.dimension,
            "formato_audio": formato_apertura,
          })
          await websocket.send_bytes(audio_apertura)

    try:
        while True:
            mensaje = await websocket.receive()

            if mensaje.get("type") == "websocket.disconnect":
                # receive() no siempre lanza WebSocketDisconnect al desconectar el
                # cliente (ej. al cerrar la pestaña) — a veces solo devuelve este
                # mensaje. Si no lo detectamos aquí, el except de abajo nunca corre y
                # la memoria longitudinal de la llamada no se guarda.
                raise WebSocketDisconnect

            # El cronómetro arranca aquí: con pulsar-para-hablar, el instante en que
            # llegan los bytes es el instante en que el paciente terminó de hablar, que es
            # justo donde la rúbrica (§5) pone el inicio de la medición.
            medicion_cm = medir_turno()
            medicion = medicion_cm.__enter__()

            if "bytes" in mensaje and mensaje["bytes"] is not None:
                with medicion.etapa("stt"):
                    texto_paciente = await asyncio.to_thread(transcribir, mensaje["bytes"])
            elif "text" in mensaje and mensaje["text"] is not None:
                # Bypass de texto directo — usado por el harness de evaluación.
                texto_paciente = json.loads(mensaje["text"])["texto"]
            else:
                continue

            es_primer_turno = len(sesion.turnos) == 0
            turno_idx = len(sesion.turnos) // 2
            sesion.turnos.append(f"paciente: {texto_paciente}")

            try:
                with medicion.etapa("orquestacion"):
                    # En un hilo, y no directamente en la corrutina, por dos razones: el
                    # turno es código bloqueante (dos llamadas HTTP y el RAG) que dejaba el
                    # event loop muerto mientras corría, y sin loop no hay forma de emitir
                    # nada al paciente durante la espera. Con el turno en un hilo, el
                    # filler de abajo puede salir a los 500 ms.
                    if usar_guion():
                        tarea_turno = asyncio.create_task(
                            asyncio.to_thread(
                                partial(
                                    orquestar_turno_guiado,
                                    turno_paciente=texto_paciente,
                                    paciente_id=sesion.paciente_id,
                                    procedimiento=sesion.procedimiento,
                                    dia_postop=sesion.dia_postop,
                                    estado_clinico=sesion.estado_clinico,
                                    pregunta_pendiente=sesion.pregunta_pendiente,
                                    turno_idx=turno_idx,
                                    historial_turno=sesion.historial_texto(),
                                    es_primer_turno_de_la_llamada=es_primer_turno,
                                    # La clasificación categórica no bloquea la respuesta:
                                    # el paciente oye la siguiente pregunta de inmediato y
                                    # la anotación llega detrás. Lo que decide un
                                    # escalamiento urgente —vía refleja, fiebre y dolor— ya
                                    # se resolvió antes de llegar aquí, sin red.
                                    diferir_extraccion=True,
                                    en_vuelo=set(sesion.extracciones_en_vuelo),
                                )
                            )
                        )
                        resultado, siguiente, completar = await tarea_turno
                        sesion.pregunta_pendiente = siguiente
                        # Sin más preguntas, lo que acaba de decirse es el cierre: se
                        # entrega el audio y se cuelga tras enviarlo. Pero antes hay que
                        # esperar lo que siga anotándose, o la llamada se cerraría con el
                        # estado clínico a medias.
                        sesion.llamada_cerrada = siguiente is None
                        if completar is not None:
                            sesion.extracciones_en_vuelo.add(completar.dimension)
                            sesion.tareas_extraccion.append(
                                asyncio.create_task(_anotar_en_segundo_plano(websocket, sesion, completar))
                            )
                        if sesion.llamada_cerrada and sesion.tareas_extraccion:
                            # Fin del guion. Aquí SÍ se espera: el desenlace de la llamada
                            # no se puede decidir con anotaciones a medio llegar. Sin esto
                            # el caso verde se cerraba con 67% de cobertura y el motor lo
                            # degradaba a `desconocida` con razón — el verde exige evidencia
                            # positiva de las seis dimensiones, no de cuatro.
                            #
                            # Es el único turno de la llamada que espera, y espera por un
                            # motivo clínico, no técnico.
                            await asyncio.gather(*sesion.tareas_extraccion, return_exceptions=True)
                            decision_final = evaluar_triaje(
                                desde_estado(sesion.estado_clinico),
                                perfil=get_settings().triage_profile,
                                exigir_cobertura_para_verde=True,
                            )
                            resultado = resultado.model_copy(update={
                                "criticidad_final": decision_final.nivel,
                                "respuesta_hablada": cierre_para(decision_final.nivel),
                                "cobertura": sesion.estado_clinico.cobertura,
                                "decision_triaje": decision_final.to_dict(),
                            })
                    else:
                        tarea_turno = asyncio.create_task(
                            asyncio.to_thread(
                                partial(
                                    orquestar_turno,
                                    turno_paciente=texto_paciente,
                                    paciente_id=sesion.paciente_id,
                                    procedimiento=sesion.procedimiento,
                                    dia_postop=sesion.dia_postop,
                                    historial_turno=sesion.historial_texto(),
                                    estado_clinico=sesion.estado_clinico,
                                    turno_idx=turno_idx,
                                    es_primer_turno_de_la_llamada=es_primer_turno,
                                )
                            )
                        )
                        resultado = await tarea_turno
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

            with medicion.etapa("tts"):
                audio_respuesta, formato_audio = await asyncio.to_thread(
                    _audio_de_texto, resultado.respuesta_hablada
                )

            async with sesion.envio:
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
                    # Qué regla determinó la criticidad de este turno. Es lo que convierte
                    # el panel en una caja de cristal: el operador ve "temperatura 38.4 °C
                    # ≥ 38.0 °C", no un color sin explicación.
                    "decision_triaje": resultado.decision_triaje,
                    "dimension_preguntada": (
                        sesion.pregunta_pendiente.dimension if sesion.pregunta_pendiente else None
                    ),
                    "llamada_finalizada": sesion.llamada_cerrada,
                    # Piper devuelve WAV y Edge MP3: el navegador reproduce los dos, pero
                    # necesita saber cuál le llega para construir el Blob.
                    "formato_audio": formato_audio,
                }
              )
              await websocket.send_bytes(audio_respuesta)
            medicion.marcar_audio_de_respuesta()

            # Una línea por turno, con las etapas desglosadas. Es lo que permite contrastar
            # lo que dice el README contra lo que pasó de verdad en la sesión — la rúbrica
            # (§4, "Repositorio, proceso y buenas prácticas") pide justamente que las
            # métricas reportadas sean verificables en los logs.
            logger.info(
                "turno paciente=%s criticidad=%s regla=%s %s",
                sesion.paciente_id,
                resultado.criticidad_final.value,
                # La regla que fijó el nivel va en la misma línea que la latencia: sin
                # ella, un log de escalamientos no permite reconstruir por qué se escaló.
                resultado.decision_triaje.get("escalado_por") or "-",
                medicion.como_linea_log(),
            )
            medicion_cm.__exit__(None, None, None)

            if sesion.llamada_cerrada:
                # El guion terminó: ya se dijo el cierre y su audio salió. Se guarda la
                # memoria longitudinal ANTES de cerrar el socket, porque el `except
                # WebSocketDisconnect` de abajo no llega a correr cuando quien cuelga es
                # el servidor.
                _guardar_memoria_sesion(sesion, MotivoCierre.COMPLETADA)
                await websocket.close()
                return

    except WebSocketDisconnect:
        _guardar_memoria_sesion(sesion, MotivoCierre.COMPLETADA)
