from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RAG_",
        extra="ignore",
    )

    app_name: str = "RAG Evaluation Project"
    environment: str = "development"

    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:1.5b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    top_k: int = Field(default=3, gt=0)

    documents_dir: Path = Path("data/documents")
    faiss_index_dir: Path = Path("data/faiss_index")


settings = Settings()
