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
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")

    groq_llm_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_LLM_MODEL")
    groq_stt_model: str = Field(default="whisper-large-v3", alias="GROQ_STT_MODEL")

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
