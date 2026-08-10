// Lógica de la interfaz de llamada: captura de micrófono (push-to-talk), WebSocket con
// el orquestador, fillers para enmascarar la latencia, y el orbe reaccionando a cada fase.

import { Orb } from "/webui/static/orb.js";

const FILLERS = [
  "a_ver.wav", "ajá.wav", "claro_que_sí.wav", "cuénteme.wav",
  "entiendo.wav", "le_escucho.wav", "listo.wav", "mmm_ya.wav",
];

const $ = (id) => document.getElementById(id);

let ws = null;
let orb = null;
let mediaRecorder = null;
let audioChunks = [];
let audioCtx = null;
let micAnalyser = null;
let salidaAnalyser = null;
let micRafId = null;
let salidaRafId = null;
let esperandoRespuesta = false;

function agregarTurno(texto, esPaciente) {
  const div = document.createElement("div");
  div.className = `turno ${esPaciente ? "turno-paciente" : "turno-agente"}`;
  div.textContent = texto;
  const contenedor = $("transcripcion");
  contenedor.appendChild(div);
  contenedor.scrollTop = contenedor.scrollHeight;
}

function actualizarCriticidad(nivel) {
  const badge = $("criticidad-badge");
  const texto = $("criticidad-texto");
  badge.classList.remove("oculto", "badge-verde", "badge-amarillo", "badge-rojo");
  badge.classList.add(`badge-${nivel}`);
  texto.textContent = nivel;
}

function reproducirFiller() {
  const archivo = FILLERS[Math.floor(Math.random() * FILLERS.length)];
  const audio = new Audio(`/fillers/static/${encodeURIComponent(archivo)}`);
  audio.volume = 0.9;
  audio.play().catch(() => {});
  return audio;
}

function iniciarAnalisisAmplitud(source, onNivel) {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  source.connect(analyser);
  const datos = new Uint8Array(analyser.frequencyBinCount);

  function loop() {
    analyser.getByteFrequencyData(datos);
    const promedio = datos.reduce((a, b) => a + b, 0) / datos.length;
    onNivel(promedio / 255);
    return requestAnimationFrame(loop);
  }
  const rafId = loop();
  return { analyser, rafId };
}

async function iniciarGrabacion() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const micSource = audioCtx.createMediaStreamSource(stream);
  const resultado = iniciarAnalisisAmplitud(micSource, (nivel) => orb.setAmplitude(nivel));
  micAnalyser = resultado.analyser;
  micRafId = resultado.rafId;

  mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
  audioChunks = [];
  mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
  mediaRecorder.onstop = async () => {
    cancelAnimationFrame(micRafId);
    stream.getTracks().forEach((t) => t.stop());
    const blob = new Blob(audioChunks, { type: "audio/webm" });
    if (blob.size < 500) {
      // Grabación demasiado corta — no vale la pena enviarla.
      orb.setState("idle");
      $("estado-texto").textContent = "Toca el micrófono para hablar";
      $("boton-hablar").disabled = false;
      return;
    }
    enviarAudio(blob);
  };

  mediaRecorder.start();
  orb.setState("listening");
  $("estado-texto").textContent = "Escuchando...";
  $("boton-hablar").classList.add("grabando");
}

function detenerGrabacion() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }
  $("boton-hablar").classList.remove("grabando");
  $("boton-hablar").disabled = true;
}

async function enviarAudio(blob) {
  orb.setState("thinking");
  $("estado-texto").textContent = "Pensando...";
  esperandoRespuesta = true;
  reproducirFiller();

  const arrayBuffer = await blob.arrayBuffer();
  ws.send(arrayBuffer);
}

function reproducirRespuesta(arrayBuffer) {
  const blob = new Blob([arrayBuffer], { type: "audio/wav" });
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);

  const source = audioCtx.createMediaElementSource(audio);
  source.connect(audioCtx.destination);
  const resultado = iniciarAnalisisAmplitud(source, (nivel) => orb.setAmplitude(nivel));
  salidaAnalyser = resultado.analyser;
  salidaRafId = resultado.rafId;

  orb.setState("speaking");
  $("estado-texto").textContent = "Hablando...";
  audio.play();

  audio.onended = () => {
    cancelAnimationFrame(salidaRafId);
    URL.revokeObjectURL(url);
    orb.setState("idle");
    orb.setAmplitude(0);
    $("estado-texto").textContent = "Toca el micrófono para hablar";
    $("boton-hablar").disabled = false;
    esperandoRespuesta = false;
  };
}

function conectarWebSocket(pacienteId, procedimiento, diaPostop) {
  const protocolo = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocolo}//${location.host}/ws/llamada`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    ws.send(JSON.stringify({ paciente_id: pacienteId, procedimiento, dia_postop: diaPostop }));
    $("boton-hablar").disabled = false;
  };

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      const datos = JSON.parse(event.data);
      agregarTurno(datos.texto_paciente_transcrito, true);
      agregarTurno(datos.respuesta_hablada, false);
      actualizarCriticidad(datos.criticidad_final);
    } else {
      reproducirRespuesta(event.data);
    }
  };

  ws.onerror = () => {
    $("estado-texto").textContent = "Se perdió la conexión — recarga la página";
  };
}

$("boton-iniciar").addEventListener("click", () => {
  const pacienteId = $("input-paciente").value || "paciente";
  const procedimiento = $("input-procedimiento").value;
  const diaPostop = parseInt($("input-dia").value, 10);

  $("panel-config").classList.add("oculto");
  $("escenario").classList.remove("oculto");
  $("transcripcion").classList.remove("oculto");
  $("controles").classList.remove("oculto");

  orb = new Orb($("orbe-contenedor"));
  conectarWebSocket(pacienteId, procedimiento, diaPostop);
});

const botonHablar = $("boton-hablar");
botonHablar.addEventListener("mousedown", () => !esperandoRespuesta && iniciarGrabacion());
botonHablar.addEventListener("touchstart", (e) => { e.preventDefault(); !esperandoRespuesta && iniciarGrabacion(); });
botonHablar.addEventListener("mouseup", detenerGrabacion);
botonHablar.addEventListener("mouseleave", () => mediaRecorder?.state === "recording" && detenerGrabacion());
botonHablar.addEventListener("touchend", (e) => { e.preventDefault(); detenerGrabacion(); });
