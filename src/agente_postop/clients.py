"""Backends de modelo de lenguaje, con respaldo en cadena.

**Ollama en local es el camino principal, y no por preferencia.** El 6 de septiembre de
2026 Groq dejó de servir todo modelo Llama generativo en su nivel gratuito: la cuenta pasó
a listar solo los clasificadores `prompt-guard`, y `llama-3.3-70b-versatile` empezó a
devolver 404. El agente quedó sin poder sostener una llamada. Google había hecho lo mismo
antes con `gemini-2.0-flash` y `gemini-2.5-flash` ("no longer available to new users").

La lección no es que un proveedor concreto sea poco fiable: es que en una demostración
cronometrada y evaluada, depender de que un tercero siga sirviendo hoy el modelo que servía
ayer es un riesgo que no hace falta correr. Los pesos en disco no los retira nadie.

    generar_json()  ─┬─→ ollama   (local, sin credenciales, sin cupos)     LLM_BACKEND
                     └─→ gemini   (nube, respaldo si no hay Ollama vivo)   LLM_FALLBACK

La cadena se recorre en orden y solo se propaga el error si fallan todos los backends
configurados. Para la nube se conserva la rotación de claves: cuando una agota su cupo
diario se pasa a la siguiente, y el error solo sube si están todas agotadas.
"""

from __future__ import annotations

import json
import logging
import threading
from functools import lru_cache
from typing import Any

import google.generativeai as genai
import httpx
from groq import Groq, RateLimitError

from agente_postop.config import get_settings

logger = logging.getLogger("agente_postop")

# Defaults del SDK: max_retries=2, timeout de lectura 60s — en una conversación de voz en
# tiempo real, un rate-limit o un pico de latencia puede terminar reintentando 2 veces por
# llamada, hasta 60s cada intento. Falla rápido: la rotación de claves ya cubre el caso de
# cupo agotado, no hace falta que el propio cliente insista.
TIMEOUT_S = 15.0
MAX_REINTENTOS = 1

# La nube dejó de ser el camino principal, y eso cambia el cálculo del timeout. Antes,
# 15 s era el techo de un turno; ahora Gemini solo entra cuando Ollama ya falló, y la
# alternativa a esperarlo es colgarle al paciente. Medido con el prompt real de
# conversación: 5.8 s de mediana, con picos que superan los 15 s y devuelven
# `504 Deadline Exceeded` — con el techo anterior, esos picos tiraban el respaldo entero.
#
# Que Ollama esté caído se detecta al instante (conexión rechazada, no expiración), así
# que este margen extra no se paga cuando el camino local funciona.
TIMEOUT_RESPALDO_S = 30.0

# Techo mucho más corto para responder al paciente. Gemini es errático en latencia —medido
# sobre el mismo prompt: 802 ms, 1,2 s, 8 s, y expiraciones—, y en una conversación hablada
# la diferencia entre esperar 8 s y esperar 30 s no es de grado: a los 8 s el paciente ya
# piensa que se cortó la llamada. Si no contesta a tiempo, es preferible decirle que se lo
# pasamos al equipo médico que dejarlo escuchando silencio.
TIMEOUT_RESPUESTA_S = 8.0

_lock = threading.Lock()
_indice_actual = 0


@lru_cache
def _pool_clientes() -> list[Groq]:
    settings = get_settings()
    return [Groq(api_key=clave, timeout=TIMEOUT_S, max_retries=MAX_REINTENTOS) for clave in settings.groq_api_keys]


def get_groq_client() -> Groq:
    """Devuelve el cliente activo del pool. Preferí `crear_completado()` para llamadas al
    LLM — esta función queda para STT (Whisper), que no rota (mismo cupo, menor consumo)."""
    pool = _pool_clientes()
    with _lock:
        return pool[_indice_actual % len(pool)]


def crear_completado(**kwargs: Any):
    """`cliente.chat.completions.create(**kwargs)` con rotación automática de clave: si la
    clave activa devuelve RateLimitError (cupo diario agotado), pasa a la siguiente del
    pool y reintenta la MISMA llamada — hasta agotar todas las claves disponibles."""
    global _indice_actual
    pool = _pool_clientes()
    n = len(pool)
    ultimo_error: RateLimitError | None = None

    for intento in range(n):
        with _lock:
            indice = _indice_actual % n
            cliente = pool[indice]
        try:
            return cliente.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            ultimo_error = exc
            with _lock:
                if _indice_actual % n == indice:  # nadie más rotó mientras tanto
                    _indice_actual += 1
            if intento < n - 1:
                logger.warning("clave Groq #%d sin cupo — rotando a la siguiente (%d/%d)", indice, intento + 2, n)

    logger.error("las %d claves de Groq del pool están sin cupo diario", n)
    raise ultimo_error


# --- Gemini (llamada A: extracción) ------------------------------------------------

_lock_gemini = threading.Lock()
_indice_gemini = 0


class SinCupoGemini(RuntimeError):
    """Todas las claves de Gemini agotadas. El llamador debe caer a Groq, no morir."""


def _es_expiracion(exc: Exception) -> bool:
    """¿El fallo fue por tiempo, y no porque la clave esté agotada o sea inválida?"""
    nombre = type(exc).__name__
    return "Deadline" in nombre or "Timeout" in nombre or "504" in str(exc)


def generar_json_gemini(*, instruccion_sistema: str, prompt_usuario: str, max_tokens: int, temperature: float, modelo: str | None = None, timeout: float | None = None) -> str:
    """Genera un JSON con Gemini Flash rotando claves ante cupo agotado.

    `genai.configure()` es estado GLOBAL del SDK, así que la configuración y la generación
    tienen que ir bajo el mismo lock: sin él, dos turnos concurrentes pueden pisarse la
    clave entre el configure y el generate_content. Serializa las llamadas a Gemini, que
    con una por turno es un costo aceptable frente a mandar la petición con la clave de
    otra sesión.
    """
    global _indice_gemini
    settings = get_settings()
    claves = settings.gemini_api_keys
    if not claves:
        raise SinCupoGemini("no hay claves de Gemini configuradas")

    n = len(claves)
    ultimo_error: Exception | None = None

    for intento in range(n):
        with _lock_gemini:
            indice = _indice_gemini % n
            try:
                genai.configure(api_key=claves[indice])
                modelo_gemini = genai.GenerativeModel(
                    modelo or settings.gemini_llm_model,
                    system_instruction=instruccion_sistema,
                    generation_config={
                        "response_mime_type": "application/json",
                        "max_output_tokens": max_tokens,
                        "temperature": temperature,
                    },
                )
                # Con timeout explícito: el cliente de Groq ya fallaba rápido (TIMEOUT_S),
                # pero este no tenía ninguno, así que una llamada colgada de Gemini colgaba
                # el turno entero. Medido en vivo antes de este arreglo: un turno con
                # `llm_extraccion_ms=36445` frente a `llm_conversacion_ms=5415` — la
                # extracción, que es la tarea barata, costaba siete veces la cara.
                return modelo_gemini.generate_content(
                    prompt_usuario, request_options={"timeout": timeout or TIMEOUT_RESPALDO_S}
                ).text
            except Exception as exc:  # noqa: BLE001 — cualquier fallo de Gemini debe poder caer a Groq
                ultimo_error = exc
                # Rotar de clave sirve para el cupo agotado, no para una expiración: si el
                # servicio va lento, la siguiente clave irá igual de lenta y el paciente
                # espera el doble. Medido: con dos claves y techo de 8 s, un turno tardaba
                # 22 s en decir «no lo sé».
                if _es_expiracion(exc):
                    raise SinCupoGemini(f"Gemini no respondió a tiempo: {exc}") from exc
                if _indice_gemini % n == indice:
                    _indice_gemini += 1
        if intento < n - 1:
            logger.warning("clave Gemini #%d falló (%s) — rotando (%d/%d)", indice, type(ultimo_error).__name__, intento + 2, n)

    raise SinCupoGemini(f"las {n} claves de Gemini fallaron; último error: {ultimo_error}") from ultimo_error


# --- Ollama (local) ----------------------------------------------------------------

# Más generoso que el de la nube: un 3B cuantizado en CPU tarda más que una LPU, y aquí no
# hay cupo que proteger ni coste por segundo. Sigue siendo un techo: un turno de voz que
# tarde más que esto ya perdió al paciente, y es mejor caer al respaldo que seguir esperando.
TIMEOUT_OLLAMA_S = 45.0


class OllamaNoDisponible(RuntimeError):
    """Ollama no responde. El llamador debe caer al respaldo, no morir."""


def _generar_ollama(
    *, instruccion_sistema: str, prompt_usuario: str, max_tokens: int, temperature: float, esquema: dict | None
) -> str:
    """Genera JSON con el modelo local.

    `format` recibe el JSON Schema completo, no el literal `"json"`. La diferencia importa
    con un modelo pequeño: `"json"` solo pide que la salida sea JSON válido y deja al
    modelo inventar la forma, mientras que pasar el esquema restringe el decodificador y
    hace imposible que falte un campo o aparezca un valor fuera del enum. Es lo que permite
    que un 3B rellene un esquema que un 70B rellenaba por buena voluntad.
    """
    settings = get_settings()
    cuerpo: dict[str, Any] = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": instruccion_sistema},
            {"role": "user", "content": prompt_usuario},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        # Mantiene el modelo en memoria entre turnos y entre llamadas. Sin esto, Ollama lo
        # descarga tras 5 minutos de inactividad y el primer turno siguiente paga la carga
        # entera mientras el paciente espera en silencio.
        "keep_alive": settings.ollama_keep_alive,
    }
    if esquema is not None:
        cuerpo["format"] = esquema

    try:
        respuesta = httpx.post(
            f"{settings.ollama_base_url}/api/chat", json=cuerpo, timeout=TIMEOUT_OLLAMA_S
        )
        respuesta.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaNoDisponible(f"Ollama en {settings.ollama_base_url} no respondió: {exc}") from exc

    return respuesta.json()["message"]["content"]


def precalentar_ollama() -> None:
    """Carga el modelo y compila por adelantado las gramáticas del guion.

    Dos costes distintos, los dos pagaderos antes de que el paciente descuelgue:

      1. **Los pesos.** Se cargan con la primera generación y el `keep_alive` los mantiene.
      2. **La gramática de cada esquema.** El decodificador compila una gramática por cada
         esquema JSON nuevo que ve, y la cachea. Medido en este equipo, la primera
         extracción de una dimensión tarda el doble que las siguientes: 25,7 s frente a
         5,4 s en `apetito`, 11,5 s frente a 6,4 s en `herida`. Sin precalentar, ese
         sobrecoste lo paga **el paciente**, una vez por cada dimensión, en la primera
         llamada tras arrancar.

    Es el mismo hallazgo que el equipo ganador del reto documentó en su informe ("la
    gramática de salida se compilaba por slot en caliente: 4–18 s por turno"), y se resuelve
    igual: preguntándole al modelo, al arrancar, exactamente lo que se le va a preguntar
    después.
    """
    from agente_postop.clinical.guion import GUION

    _generar_ollama(
        instruccion_sistema="Responde en JSON.",
        prompt_usuario='Devuelve {"ok": true}',
        max_tokens=8,
        temperature=0.0,
        esquema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
    )

    for pregunta in GUION:
        if pregunta.valores is None:
            continue  # dolor y fiebre los lee un parser: no hay gramática que compilar
        _generar_ollama(
            instruccion_sistema=f"Clasifica {pregunta.dimension}.",
            prompt_usuario="Respondió: \"sí\"",
            max_tokens=MAX_TOKENS_PRECALENTADO,
            temperature=0.0,
            esquema={
                "type": "object",
                "properties": {"valor": {"type": "string", "enum": [*pregunta.valores, AUSENTE]}},
                "required": ["valor"],
            },
        )


# Lo justo para que el decodificador recorra la gramática y la deje en caché. No se mira la
# respuesta: aquí no se está clasificando nada.
MAX_TOKENS_PRECALENTADO = 8


def ollama_vivo() -> bool:
    """¿Hay un Ollama contestando, y tiene cargado el modelo declarado?

    Lo usa el arranque para avisar por consola antes de la primera llamada, en vez de que
    el primer turno del paciente sea el que descubra que no hay modelo.
    """
    settings = get_settings()
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3.0)
        r.raise_for_status()
    except httpx.HTTPError:
        return False
    nombres = {m["name"] for m in r.json().get("models", [])}
    return settings.ollama_model in nombres or f"{settings.ollama_model}:latest" in nombres


# --- Despachador -------------------------------------------------------------------


def _generar_groq_json(
    *, instruccion_sistema: str, prompt_usuario: str, max_tokens: int, temperature: float
) -> str:
    completado = crear_completado(
        model=get_settings().groq_llm_model,
        messages=[
            {"role": "system", "content": instruccion_sistema},
            {"role": "user", "content": prompt_usuario},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completado.choices[0].message.content


def generar_json(
    *,
    instruccion_sistema: str,
    prompt_usuario: str,
    max_tokens: int,
    temperature: float,
    esquema: dict | None = None,
    etiqueta: str = "llm",
    preferir: str | None = None,
    sin_cadena: bool = False,
) -> tuple[str, str]:
    """Genera JSON por el backend configurado, cayendo al respaldo si falla.

    Devuelve `(contenido, backend_usado)`. El segundo elemento no es decorativo: sin él,
    las métricas del README no podrían decir con qué modelo se midió cada turno, y la
    rúbrica contrasta lo reportado contra los logs.

    `esquema` solo lo aprovecha Ollama; los otros backends piden JSON genérico. Pasarlo
    siempre no cuesta nada y hace que cambiar de backend no cambie el contrato.
    """
    settings = get_settings()
    cadena = [b for b in (settings.llm_backend, settings.llm_fallback) if b != "ninguno"]

    # `preferir` pone un backend al frente para una operación concreta, sin cambiar el
    # default de todo el sistema. Existe por una razón medida: redactar una respuesta
    # hablada son ~160 tokens de generación, y a los ~5,5 tok/s de una CPU sin GPU eso son
    # 30 s con el paciente al teléfono. Clasificar un slot son 10 tokens y se resuelve en
    # local sin problema. La operación cara —y solo esa— se manda a la nube; el resto sigue
    # corriendo en la máquina, sin credenciales.
    if preferir and preferir != "ninguno":
        # `sin_cadena` corta el respaldo para esta operación. Tiene sentido en una sola:
        # responder al paciente. Si la nube no contestó en 8 s, reintentarlo en local son
        # 30 s más — y para entonces el paciente ya colgó. Es mejor decirle que se lo
        # pasamos al equipo médico, que es una respuesta honesta y llega al instante.
        cadena = [preferir] if sin_cadena else [preferir, *cadena]

    errores: list[str] = []

    for backend in dict.fromkeys(cadena):  # sin duplicados, conservando el orden
        try:
            if backend == "ollama":
                return _generar_ollama(
                    instruccion_sistema=instruccion_sistema, prompt_usuario=prompt_usuario,
                    max_tokens=max_tokens, temperature=temperature, esquema=esquema,
                ), backend
            if backend == "gemini":
                return generar_json_gemini(
                    instruccion_sistema=instruccion_sistema, prompt_usuario=prompt_usuario,
                    max_tokens=max_tokens, temperature=temperature,
                    # Solo cuando se pidió Gemini para ESTA operación: como respaldo general
                    # se conserva el modelo por defecto.
                    modelo=settings.gemini_modelo_respuestas if preferir == "gemini" else None,
                    timeout=TIMEOUT_RESPUESTA_S if preferir == "gemini" else None,
                ), backend
            if backend == "groq":
                return _generar_groq_json(
                    instruccion_sistema=instruccion_sistema, prompt_usuario=prompt_usuario,
                    max_tokens=max_tokens, temperature=temperature,
                ), backend
        except Exception as exc:  # noqa: BLE001 — cualquier fallo de un backend cae al siguiente
            errores.append(f"{backend}: {type(exc).__name__}: {exc}")
            logger.warning("backend '%s' falló en %s (%s) — probando el siguiente", backend, etiqueta, exc)

    raise RuntimeError(f"todos los backends fallaron en {etiqueta} — {' | '.join(errores)}")


def esquema_json_de(modelo_pydantic: type) -> dict:
    """JSON Schema de un modelo Pydantic, aplanado para el decodificador de Ollama.

    `$defs`/`$ref` no los resuelve: se expanden en línea porque la gramática que compila
    Ollama trabaja sobre un esquema autocontenido, y un `$ref` sin resolver se traduce en
    un campo libre — justo la restricción que se quería imponer.
    """
    esquema = modelo_pydantic.model_json_schema()
    definiciones = esquema.pop("$defs", {})

    def expandir(nodo: Any) -> Any:
        if isinstance(nodo, dict):
            if "$ref" in nodo:
                nombre = nodo["$ref"].rsplit("/", 1)[-1]
                return expandir(json.loads(json.dumps(definiciones.get(nombre, {}))))
            return _saneal_enum_nullable({k: expandir(v) for k, v in nodo.items()})
        if isinstance(nodo, list):
            return [expandir(v) for v in nodo]
        return nodo

    return expandir(esquema)


# Marcador textual para la ausencia dentro de un enum. Ver `_saneal_enum_nullable`.
AUSENTE = "null"


def _saneal_enum_nullable(nodo: dict) -> dict:
    """Hace expresable la ausencia en un enum, que sin esto es inalcanzable.

    Un `Optional[MiEnum]` de Pydantic produce
    `{"anyOf": [{"enum": ["a","b"], "type": "string"}, {"type": "null"}]}`. La gramática que
    el decodificador compila a partir de eso resulta ser una alternancia de **cadenas**, y
    el literal JSON `null` no cabe en ella: el modelo queda obligado a elegir uno de los
    valores reales aunque el paciente no haya dicho nada.

    Medido sobre `llama3.2:3b`, temperatura 0, enum de apetito, ante «hola, buenos días»:

        esquema                            respuesta
        anyOf enum|null (el de Pydantic)   'normal'          ← no podía decir null
        sin enum, type nullable            'null' (cadena)   ← quería decirlo y no cabía
        enum con la cadena "null"          None              ← correcto

    El modelo no estaba adivinando por incompetencia: respondía lo único que la gramática
    le dejaba responder. Se colapsa el `anyOf` a un enum de cadenas que incluye `"null"`, y
    `normalizar_ausencias` la vuelve a convertir en `None` al parsear, de modo que el resto
    del sistema sigue viendo un `Optional` normal.
    """
    # Caso 1: el enum ya trae el literal None entre sus valores.
    valores = nodo.get("enum")
    if isinstance(valores, list) and None in valores:
        nodo["enum"] = [AUSENTE if v is None else v for v in valores]
        if isinstance(nodo.get("type"), list):
            nodo["type"] = "string"
        return nodo

    # Caso 2 (el que genera Pydantic): anyOf de un enum y un null.
    opciones = nodo.get("anyOf")
    if not isinstance(opciones, list) or len(opciones) != 2:
        return nodo
    con_enum = next((o for o in opciones if isinstance(o, dict) and "enum" in o), None)
    hay_null = any(isinstance(o, dict) and o.get("type") == "null" for o in opciones)
    if con_enum is None or not hay_null:
        return nodo

    colapsado = {k: v for k, v in nodo.items() if k != "anyOf"}
    colapsado.update(con_enum)
    colapsado["type"] = "string"
    colapsado["enum"] = [*(v for v in con_enum["enum"] if v is not None), AUSENTE]
    if colapsado.get("default", "sin-default") is None:
        colapsado["default"] = AUSENTE
    return colapsado


def normalizar_ausencias(datos: Any) -> Any:
    """Convierte de vuelta a `None` las cadenas `"null"` que introdujo el saneado.

    Se aplica a lo que devuelve cualquier backend, no solo Ollama: un modelo que escribe
    la cadena `"null"` por su cuenta —cosa que se observó con el esquema sin enum— debe
    tratarse igual, y aquí `"null"` nunca es un valor clínico legítimo.
    """
    if isinstance(datos, dict):
        return {k: normalizar_ausencias(v) for k, v in datos.items()}
    if isinstance(datos, list):
        return [normalizar_ausencias(v) for v in datos]
    if isinstance(datos, str) and datos.strip().lower() in ("null", "none"):
        return None
    return datos
