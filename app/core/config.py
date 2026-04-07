"""Application configuration using pydantic-settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: str = Field(..., description="OpenAI API key for Whisper and GPT-4o")
    database_url: str = Field(
        ...,
        description="PostgreSQL connection string with asyncpg driver",
    )
    environment: str = Field(default="development", description="Environment: development, production")
    log_level: str = Field(default="INFO", description="Logging level")
    max_audio_size_mb: int = Field(default=25, description="Maximum audio file size in MB")
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model",
    )
    llm_model: str = Field(default="gpt-4o", description="OpenAI LLM model for agents")
    confidence_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for human-in-the-loop",
    )

    @property
    def max_audio_size_bytes(self) -> int:
        """Convert max audio size from MB to bytes."""
        return self.max_audio_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"


settings = Settings()
