"""Configuración central del agente: carga variables de entorno y rutas del proyecto."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

# Las cuatro familias de modelo que `docs/stack-tecnico.md` permite (compuerta G3), por
# prefijo. Se valida la FAMILIA y no la versión exacta a propósito: durante la ventana de
# este reto, tres snapshots concretos dejaron de servirse —`llama-3.3-70b-versatile` en
# Groq, `gemini-2.0-flash` y `gemini-2.5-flash` en Google—, y el propio documento del reto
# contempla el sucesor vigente de la misma familia y proveedor.
FAMILIAS_PERMITIDAS: dict[str, tuple[str, ...]] = {
    "Meta Llama local, serie 3.x (1B–3B)": ("llama3.1", "llama3.2", "llama-3.1", "llama-3.2"),
    "Microsoft Phi Mini local, serie 3.5+": ("phi3.5", "phi-3.5", "phi4-mini", "phi4mini"),
    "Google Gemini gama Flash (nube)": ("gemini-flash", "gemini-1.5-flash", "gemini-2", "gemini-3"),
    "Meta Llama vía Groq (nube)": ("llama-3.3", "llama3.3", "llama-3.1-", "groq/llama"),
}


# Familias donde el reto acota además el TAMAÑO, con los tags admitidos.
#
# Solo Llama local lo necesita: el reto no permite "Llama 3.x en local" sin más, permite la
# serie 1B–3B. Sin esta comprobación, `llama3.1:8b` pasaba por el prefijo de familia y
# habría sido una infracción de G3 difícil de ver leyendo un `.env`. Phi Mini no lleva
# restricción propia porque "Mini" ya es el acotamiento: la serie entera cabe.
TAMANOS_EXIGIDOS: dict[str, tuple[str, ...]] = {
    "Meta Llama local, serie 3.x (1B–3B)": ("1b", "3b"),
}


def familia_de(modelo: str) -> str | None:
    """La familia permitida a la que pertenece un modelo, o None si no pertenece a ninguna.

    Se compara por familia y no por versión exacta porque los proveedores retiran
    snapshots —tres lo hicieron durante este reto—, y se comprueba además el tamaño en las
    familias donde el reto lo acota.

    Un modelo local sin tag de tamaño (`llama3.2` a secas) se rechaza a propósito: Ollama
    lo resolvería a 3B por defecto, pero una compuerta que depende de un default implícito
    del runtime no es una compuerta. Hay que declarar lo que se usa.
    """
    nombre, _, etiqueta = modelo.strip().lower().partition(":")
    for familia, prefijos in FAMILIAS_PERMITIDAS.items():
        if not any(nombre.startswith(p) for p in prefijos):
            continue
        exigidos = TAMANOS_EXIGIDOS.get(familia)
        if exigidos and not any(t in etiqueta or t in nombre for t in exigidos):
            return None
        return familia
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # Opcional desde que la conversación corre en local: sin esta clave el agente razona
    # igual, y lo único que deja de funcionar es el reconocimiento de voz (Whisper en
    # Groq). Antes era obligatoria y su ausencia impedía incluso importar el paquete.
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_api_keys_extra: str = Field(default="", alias="GROQ_API_KEYS_EXTRA")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_api_keys_extra: str = Field(default="", alias="GEMINI_API_KEYS_EXTRA")

    @property
    def gemini_api_keys(self) -> list[str]:
        """Mismo esquema de pool que Groq. Importa más de lo que parece: el cupo de Gemini
        es un presupuesto SEPARADO del de Groq, así que mover la extracción acá no reparte
        el mismo cupo — lo duplica."""
        extra = [k.strip() for k in self.gemini_api_keys_extra.split(",") if k.strip()]
        vistas: list[str] = []
        for clave in [self.gemini_api_key, *extra]:
            if clave and clave not in vistas:
                vistas.append(clave)
        return vistas

    @property
    def groq_api_keys(self) -> list[str]:
        """GROQ_API_KEY sigue siendo la primaria (compatibilidad); GROQ_API_KEYS_EXTRA
        es una lista opcional separada por comas de claves adicionales — se rota a la
        siguiente automáticamente cuando una se queda sin cupo diario (ver clients.py)."""
        extra = [k.strip() for k in self.groq_api_keys_extra.split(",") if k.strip()]
        vistas: list[str] = []
        for clave in [self.groq_api_key, *extra]:
            if clave not in vistas:
                vistas.append(clave)
        return vistas

    # --- Backends del LLM -----------------------------------------------------------
    #
    # `ollama` es el default y no es una preferencia estética: el 6 de septiembre de 2026
    # Groq dejó de servir todo modelo Llama generativo en el nivel gratuito
    # (`llama-3.3-70b-versatile` devuelve 404 y en la cuenta solo quedan los clasificadores
    # `prompt-guard`), y eso dejó al agente sin poder sostener una llamada. Un modelo local
    # no lo puede retirar un proveedor a mitad de una sesión evaluada.
    #
    # `llm_fallback` conserva la nube como respaldo: si Ollama no está corriendo, la
    # llamada sigue en Gemini Flash en vez de morir.
    llm_backend: str = Field(default="ollama", alias="LLM_BACKEND")
    llm_fallback: str = Field(default="gemini", alias="LLM_FALLBACK")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2:3b", alias="OLLAMA_MODEL")
    # Mantener el modelo cargado entre turnos. Sin esto Ollama lo descarga de memoria tras
    # 5 minutos de inactividad y el primer turno de la llamada siguiente paga la recarga
    # completa — decenas de segundos en los que el paciente solo oye silencio.
    ollama_keep_alive: str = Field(default="2h", alias="OLLAMA_KEEP_ALIVE")

    # Groq conserva el STT: `whisper-large-v3` sigue disponible y es lo único que se le
    # sigue pidiendo. La conversación ya no pasa por aquí.
    groq_llm_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_LLM_MODEL")
    groq_stt_model: str = Field(default="whisper-large-v3", alias="GROQ_STT_MODEL")

    @field_validator("llm_backend", "llm_fallback")
    @classmethod
    def _backend_valido(cls, v: str) -> str:
        permitidos = {"ollama", "gemini", "groq", "ninguno"}
        if v not in permitidos:
            raise ValueError(f"backend '{v}' desconocido; use uno de {sorted(permitidos)}")
        return v

    @property
    def modelo_de(self) -> dict[str, str]:
        """El modelo concreto que usa cada backend."""
        return {"ollama": self.ollama_model, "gemini": self.gemini_llm_model, "groq": self.groq_llm_model}

    def validar_compuerta_g3(self) -> None:
        """Aborta el arranque si algún backend activo declara un modelo fuera de la lista
        permitida por el reto.

        La compuerta se defiende con código y no con disciplina: es demasiado fácil poner
        un modelo cualquiera en el `.env` y descubrirlo cuando el jurado ya está mirando.
        """
        for backend in (self.llm_backend, self.llm_fallback):
            if backend == "ninguno":
                continue
            modelo = self.modelo_de[backend]
            if familia_de(modelo) is None:
                raise ValueError(
                    f"El modelo '{modelo}' (backend '{backend}') no pertenece a ninguna "
                    f"familia permitida por la compuerta G3. Permitidas: "
                    f"{', '.join(FAMILIAS_PERMITIDAS)}."
                )

    # Alias "-latest" a propósito: stack-tecnico.md advierte que los proveedores retiran
    # snapshots sin aviso y fija familias, no versiones. Fijar `gemini-2.5-flash` hoy da
    # 404 ("no longer available to new users"), igual que `gemini-2.0-flash` antes.
    #
    # `EXTRACCION_EN_GEMINI` ya no existe: qué proveedor resuelve cada llamada lo deciden
    # LLM_BACKEND y LLM_FALLBACK, que gobiernan por igual la conversación y la extracción.
    gemini_llm_model: str = Field(default="gemini-flash-latest", alias="GEMINI_LLM_MODEL")

    # Perfil del motor de triaje (clinical/triage.py). `conservador` es el default pese a
    # su menor exactitud sobre el dataset (142/160 frente a 157/160): cambia 15 falsos
    # positivos por margen de seguridad clínica, que es la dirección en la que la rúbrica
    # pide equivocarse. Los dos perfiles tienen 0 falsos negativos de `rojo`.
    triage_profile: str = Field(default="conservador", alias="TRIAGE_PROFILE")

    # El agente conduce la llamada con las preguntas fijas de `clinical/guion.py` en vez de
    # dejar que el modelo redacte cada una. Medido en este equipo: extracción de 16,2 s a
    # 8,1 s, TTS de ~1.000 ms a 20 ms, y el acierto de extracción de 1/6 slots a 5/5.
    # Ponerlo en false devuelve al camino libre, que sigue funcionando y sirve para
    # contrastar los dos.
    usar_guion: bool = Field(default=True, alias="USAR_GUION")

    # Backend para responder preguntas abiertas del paciente. Es la única operación que
    # redacta una frase entera (~160 tokens) en vez de clasificar (~10), y en una CPU sin
    # GPU eso son ~30 s medidos, con el paciente esperando. Todo lo demás —clasificación de
    # slots, triaje, parsers— sigue en local. Poner `ninguno` lo devuelve al backend por
    # defecto.
    llm_respuestas: str = Field(default="gemini", alias="LLM_RESPUESTAS")

    # Modelo de Gemini para esa operación, distinto del que se usa como respaldo general.
    # Medido sobre el prompt real de respuesta anclada, tres llamadas cada uno:
    #
    #   gemini-flash-latest        mediana 33.138 ms   y respondía en INGLÉS
    #   gemini-flash-lite-latest   mediana    802 ms   español correcto
    #
    # Cuarenta veces más rápido en la única operación que el paciente espera oyendo
    # silencio. Ambos pertenecen a la gama Flash, así que la compuerta G3 se cumple igual.
    gemini_modelo_respuestas: str = Field(
        default="gemini-flash-lite-latest", alias="GEMINI_MODELO_RESPUESTAS"
    )

    chroma_persist_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma", alias="CHROMA_PERSIST_DIR")
    vault_dir: Path = Field(default=PROJECT_ROOT / "vault", alias="VAULT_DIR")
    dataset_dir: Path = Field(default=PROJECT_ROOT / "dataset", alias="DATASET_DIR")

    piper_voice_model: Path = Field(
        default=PROJECT_ROOT / "data" / "voices" / "es_voice.onnx", alias="PIPER_VOICE_MODEL"
    )
    fillers_dir: Path = Field(default=PROJECT_ROOT / "src" / "agente_postop" / "voice" / "fillers", alias="FILLERS_DIR")

    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    def ensure_dirs(self) -> None:
        for path in (self.chroma_persist_dir, self.vault_dir, self.dataset_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    settings.validar_compuerta_g3()
    return settings
