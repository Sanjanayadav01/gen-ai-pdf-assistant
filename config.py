"""Central configuration. Every value can be overridden through environment variables or a .env file."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Embeddings (runs locally, no API key needed)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "4"))
    min_score: float = float(os.getenv("MIN_SCORE", "0.20"))  # cosine similarity floor

    # Generation: "anthropic" | "openai" | "none" (extractive fallback, no API key)
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic").lower()
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "800"))

    # Persistence
    index_dir: str = os.getenv("INDEX_DIR", "storage")


settings = Settings()
