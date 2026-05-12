"""Environment-driven configuration. Single source of truth for tunables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


def _float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


@dataclass(frozen=True)
class Settings:
    llm_model: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    use_4bit: bool = _bool("USE_4BIT", True)
    llm_device: str = os.getenv("LLM_DEVICE", "auto")
    llm_max_new_tokens: int = _int("LLM_MAX_NEW_TOKENS", 384)

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

    chunk_size: int = _int("CHUNK_SIZE", 1200)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 200)
    retrieve_k: int = _int("RETRIEVE_K", 20)
    final_k: int = _int("FINAL_K", 5)
    crag_threshold: float = _float("CRAG_THRESHOLD", 0.001)
    contextual_retrieval: bool = _bool("CONTEXTUAL_RETRIEVAL", True)

    cache_dir: Path = Path(os.getenv("CACHE_DIR", ".cache"))


settings = Settings()
