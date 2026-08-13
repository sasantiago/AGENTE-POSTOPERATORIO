"""Configuración central del agente: carga variables de entorno y rutas del proyecto."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    groq_api_key: str = Field(alias="GROQ_API_KEY")
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

    groq_llm_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_LLM_MODEL")
    groq_stt_model: str = Field(default="whisper-large-v3", alias="GROQ_STT_MODEL")

    # Modelo de la llamada A (extracción). Alias "-latest" a propósito: stack-tecnico.md
    # advierte que los proveedores retiran snapshots sin aviso y fija familias, no versiones
    # (de hecho gemini-2.0-flash ya devuelve 404). Poner EXTRACCION_EN_GEMINI=false lo
    # devuelve todo a Groq sin tocar código.
    gemini_llm_model: str = Field(default="gemini-flash-latest", alias="GEMINI_LLM_MODEL")
    extraccion_en_gemini: bool = Field(default=True, alias="EXTRACCION_EN_GEMINI")

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
    return settings
