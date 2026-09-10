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
// El agente colgó al terminar el protocolo. Distingue un cierre normal de una caída de
// conexión, que para el paciente se ven igual y significan cosas muy distintas.
let llamadaFinalizada = false;

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

// Cuánto se espera antes de soltar un filler. El filler existe para tapar un silencio
// incómodo, y desde que el turno se resuelve en ~25 ms casi nunca hay silencio que tapar:
// lanzarlo siempre hacía que el "mmm, ya" sonara ENCIMA de la siguiente pregunta. Ahora
// solo aparece si el agente de verdad se está tardando.
const MS_ANTES_DEL_FILLER = 900;

let fillerActual = null;
let fillerTimeout = null;

function programarFiller() {
  cancelarFiller();
  fillerTimeout = setTimeout(() => {
    const archivo = FILLERS[Math.floor(Math.random() * FILLERS.length)];
    fillerActual = new Audio(`/fillers/static/${encodeURIComponent(archivo)}`);
    fillerActual.volume = 0.9;
    fillerActual.play().catch(() => {});
  }, MS_ANTES_DEL_FILLER);
}

function cancelarFiller() {
  // Se corta tanto el temporizador (aún no sonó) como el audio en curso (ya empezó): sin
  // lo segundo, un filler lanzado a los 900 ms seguía sonando bajo la respuesta que llegó
  // a los 950 ms, y se oían las dos voces solapadas.
  if (fillerTimeout) { clearTimeout(fillerTimeout); fillerTimeout = null; }
  if (fillerActual) { fillerActual.pause(); fillerActual = null; }
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
  programarFiller();

  const arrayBuffer = await blob.arrayBuffer();
  ws.send(arrayBuffer);
}

function reproducirRespuesta(arrayBuffer) {
  // Lo primero: callar cualquier muletilla. El agente va a hablar.
  cancelarFiller();

  const blob = new Blob([arrayBuffer], { type: "audio/wav" });
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);

  // El AudioContext se creaba solo al grabar, así que el primer audio del agente —la
  // apertura, que suena ANTES de que el paciente hable— reventaba aquí con `audioCtx`
  // todavía en null y la llamada arrancaba muda. Ahora se crea a demanda.
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  // Algunos navegadores lo dejan suspendido hasta que hay un gesto del usuario; el clic en
  // "Iniciar llamada" cuenta, pero hay que reanudarlo explícitamente.
  if (audioCtx.state === "suspended") audioCtx.resume().catch(() => {});

  // La visualización de amplitud es un adorno: si falla, el paciente tiene que oír al
  // agente igual. Antes un error aquí se llevaba por delante la reproducción entera.
  try {
    const source = audioCtx.createMediaElementSource(audio);
    source.connect(audioCtx.destination);
    const resultado = iniciarAnalisisAmplitud(source, (nivel) => orb.setAmplitude(nivel));
    salidaAnalyser = resultado.analyser;
    salidaRafId = resultado.rafId;
  } catch (err) {
    console.warn("sin visualización de amplitud, se reproduce igual:", err);
  }

  orb.setState("speaking");
  $("estado-texto").textContent = "Hablando...";
  audio.play().catch((err) => {
    // Si el navegador bloquea la reproducción automática, el turno no puede quedarse
    // colgado esperando un `onended` que no va a llegar.
    console.warn("no se pudo reproducir el audio:", err);
    $("estado-texto").textContent = "Toca el micrófono para hablar";
    $("boton-hablar").disabled = false;
    esperandoRespuesta = false;
  });

  audio.onended = () => {
    cancelAnimationFrame(salidaRafId);
    URL.revokeObjectURL(url);
    orb.setState("idle");
    orb.setAmplitude(0);
    esperandoRespuesta = false;
    if (llamadaFinalizada) {
      $("estado-texto").textContent = "Llamada finalizada";
      $("boton-hablar").disabled = true;
      orb.setState("idle");
      return;
    }
    $("estado-texto").textContent = "Toca el micrófono para hablar";
    $("boton-hablar").disabled = false;
  };
}

function conectarWebSocket(pacienteId, procedimiento, diaPostop) {
  const protocolo = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocolo}//${location.host}/ws/llamada`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    ws.send(JSON.stringify({ paciente_id: pacienteId, procedimiento, dia_postop: diaPostop }));
    // El botón NO se habilita aquí: el agente abre la llamada hablando, igual que una
    // llamada de seguimiento real, y lo habilita el `onended` de esa apertura. Habilitarlo
    // antes dejaba al paciente interrumpir al agente en su primera frase.
    $("estado-texto").textContent = "Llamando...";
  };

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      const datos = JSON.parse(event.data);

      // Anotación tardía: la clasificación del turno ANTERIOR, que se resolvió mientras el
      // paciente ya escuchaba la pregunta siguiente. No es un turno de conversación —no
      // lleva nada hablado— así que solo refresca el estado clínico del panel.
      if (datos.tipo === "anotacion") {
        actualizarCriticidad(datos.criticidad_final);
        return;
      }

      // La apertura llega sin nada del paciente: sin esta guarda se pintaba una burbuja
      // vacía a su nombre antes de que hubiera dicho una palabra.
      if (datos.texto_paciente_transcrito) agregarTurno(datos.texto_paciente_transcrito, true);
      agregarTurno(datos.respuesta_hablada, false);
      actualizarCriticidad(datos.criticidad_final);
      if (datos.llamada_finalizada) llamadaFinalizada = true;
    } else {
      reproducirRespuesta(event.data);
    }
  };

  ws.onclose = () => {
    $("boton-hablar").disabled = true;
    $("estado-texto").textContent = llamadaFinalizada
      ? "Llamada finalizada"
      : "Se cerró la conexión";
  };

  ws.onerror = () => {
    $("estado-texto").textContent = "Se perdió la conexión — recarga la página";
  };
}

// --- registro de pacientes ---------------------------------------------------
// El procedimiento se lee de la historia clínica del paciente, no se elige. La opción de
// entrada manual existe solo para la demostración de G5 (Rinoplastia), que a propósito no
// está ni en el corpus ni en el registro.
const OPCION_MANUAL = "__manual__";
let REGISTRO = {};

async function cargarRegistro() {
  const res = await fetch("/api/pacientes");
  const { pacientes, procedimientos } = await res.json();
  REGISTRO = Object.fromEntries(pacientes.map((p) => [p.paciente_id, p]));

  $("input-paciente-registro").innerHTML =
    pacientes.map((p) => `<option value="${p.paciente_id}">${p.paciente_id} · ${p.procedimiento}</option>`).join("") +
    `<option value="${OPCION_MANUAL}">— otro procedimiento (demo) —</option>`;

  $("input-procedimiento").innerHTML = procedimientos.map((p) => `<option>${p}</option>`).join("");
  $("input-procedimiento").value = "Rinoplastia";
  mostrarFicha();
}

function mostrarFicha() {
  const id = $("input-paciente-registro").value;
  const manual = id === OPCION_MANUAL;
  $("bloque-manual").classList.toggle("oculto", !manual);

  const p = REGISTRO[id];
  if (manual) {
    $("ficha-paciente").textContent = "Entrada manual — solo para demostrar conocimiento nuevo.";
  } else if (p) {
    const comorbilidades = p.comorbilidades.length ? p.comorbilidades.join(", ") : "sin comorbilidades";
    const edad = p.edad ? ` · ${p.edad} años` : "";
    $("ficha-paciente").textContent = `${p.procedimiento} · operado el ${p.fecha_cirugia}${edad} · ${comorbilidades}`;
  } else {
    $("ficha-paciente").textContent = "";
  }
}

$("input-paciente-registro").addEventListener("change", mostrarFicha);
cargarRegistro();

$("boton-iniciar").addEventListener("click", () => {
  const seleccion = $("input-paciente-registro").value;
  const manual = seleccion === OPCION_MANUAL;
  const pacienteId = manual ? $("input-paciente").value || "paciente" : seleccion;
  const procedimiento = manual ? $("input-procedimiento").value : REGISTRO[seleccion]?.procedimiento;
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
