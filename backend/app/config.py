"""
Application configuration via environment variables.

Uses pydantic-settings to load values from .env file with type validation
and sensible defaults for all configuration parameters.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Enterprise RAG application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Hugging Face ---
    hf_api_token: str = ""

    # --- Model Configuration ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    generation_model: str = "mistralai/Mistral-7B-Instruct-v0.3"

    # --- Chunking ---
    chunk_size: int = 600  # target tokens per chunk
    chunk_overlap: int = 120  # overlap tokens between chunks

    # --- Retrieval ---
    top_k: int = 5
    score_threshold: float = 0.3
    max_context_tokens: int = 3000

    # --- Paths ---
    data_dir: str = "./data"
    faiss_index_path: str = "./data/faiss_index"

    # --- Logging ---
    log_level: str = "INFO"

    # --- Server ---
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    @property
    def upload_dir(self) -> Path:
        """Directory for uploaded PDF files."""
        path = Path(self.data_dir) / "uploads"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cache_dir(self) -> Path:
        """Directory for embedding caches."""
        path = Path(self.data_dir) / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def index_dir(self) -> Path:
        """Directory for FAISS index files."""
        path = Path(self.faiss_index_path)
        path.mkdir(parents=True, exist_ok=True)
        return path


def get_settings() -> Settings:
    """Factory function for settings singleton."""
    return Settings()
