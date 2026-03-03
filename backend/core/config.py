from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_NAME: str = "FINORA"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_REGULATORY: str = "regulatory_docs"

    # LLM — Groq
    GROQ_API_KEY: str

    # LangSmith
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_PROJECT: str = "FINORA"

    # News API
    NEWS_API_KEY: str = ""

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
