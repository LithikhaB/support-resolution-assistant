from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "support-resolution-assistant"
    log_level: str = "INFO"

    postgres_user: str = "support"
    postgres_password: str = "support_pass"
    postgres_db: str = "support_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_connect_timeout: int = 3
    postgres_statement_timeout_ms: int = Field(default=3000, ge=1)
    retrieval_statement_timeout_ms: int = Field(default=30000, ge=1)
    retrieval_candidate_k: int = Field(default=50, ge=1, le=100)
    retrieval_rrf_constant: int = Field(default=60, ge=1)
    indexing_statement_timeout_ms: int = Field(default=600000, ge=1000)
    indexing_batch_size: int = Field(default=128, ge=1, le=1024)

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = Field(default=384, ge=1)
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    embedding_local_files_only: bool = False
    tokenizer_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    tokenizer_local_files_only: bool = False
    chunk_max_tokens: int = Field(default=256, ge=8, le=256)
    chunk_overlap_tokens: int = Field(default=32, ge=0)

    data_dir: Path = Path("data")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def evaluation_dir(self) -> Path:
        return self.data_dir / "evaluation"


@lru_cache
def get_settings() -> Settings:
    return Settings()
