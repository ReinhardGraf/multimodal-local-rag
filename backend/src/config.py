"""
Centralised application settings.

All configuration is driven by environment variables.  Values are
automatically loaded from the ``.env`` file located at the repository
root (two directories above this file), so the application works
out-of-the-box without any manual ``export`` calls.

Usage::

    from src.config import settings

    print(settings.qdrant_url)         # "http://localhost:6333"
    print(settings.reranker_backend)   # "cross-encoder"

The ``Settings`` class uses *pydantic-settings* ``BaseSettings``, which:
* reads each field from the matching environment variable (case-insensitive)
* falls back to the ``.env`` file when the variable is absent from the
  process environment
* validates and coerces types automatically (e.g. ``"1"`` → ``True``)

The singleton ``settings`` is created once at import time.  In tests you
can override individual values via environment variables or by passing
keyword arguments to ``Settings()``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Locate the .env file relative to this source file so it is found
# regardless of the current working directory.
# backend/src/config.py → parents[2] = repo root (multimodal-rag/)
_ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """Application-wide settings, one attribute per environment variable.

    All field names are the lowercase version of the matching env var.
    pydantic-settings resolves them case-insensitively, so
    ``RERANKER_BACKEND`` in the environment maps to ``reranker_backend``
    here.
    """

    model_config = SettingsConfigDict(
        env_file=str(_ROOT_ENV),
        env_file_encoding="utf-8",
        # Ignore extra keys present in the .env file that have no
        # corresponding field (e.g. n8n / Docker-only variables).
        extra="ignore",
        case_sensitive=False,
    )

    # ── Reranker ──────────────────────────────────────────────────────────
    # Backend: "cross-encoder" (local sentence-transformers) or "ollama"
    reranker_backend: str = "cross-encoder"
    # HuggingFace CrossEncoder model id
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Device for CrossEncoder: "auto" | "cuda" | "mps" | "cpu"
    reranker_device: str = "auto"
    # Batch size for CrossEncoder.predict()
    reranker_batch_size: int = 32
    # Ollama model tag used when reranker_backend == "ollama"
    ollama_reranker_model: str = "qllama/bge-reranker-v2-m3:q4_k_m"

    # ── Ollama (shared) ───────────────────────────────────────────────────
    # Local default — works when running the backend outside Docker.
    # Override with OLLAMA_URL=http://host.docker.internal:11434 in .env
    # (or as an env var) when running inside a Docker container.
    ollama_url: str = "http://localhost:11434"

    # ── Vector store / Qdrant ─────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "documents"
    # Ollama embedding model
    embedding_model: str = "qllama/multilingual-e5-large-instruct:latest"
    # Vector dimension (must match the embedding model output)
    embedding_dimension: int = 1024
    # Number of texts to embed per Ollama /api/embed request
    ollama_embed_batch_size: int = 128
    # Over-fetch multiplier for reranking: candidates = limit × factor
    rerank_overfetch_factor: int = 5

    # -- Postgres (for reconciliation) ─────────────────────────────────────────
    postgres_dsn: str = "postgresql://n8n:n8n_password@localhost:5436/n8n_rag"

    # ── HuggingFace Hub ───────────────────────────────────────────────────
    # Set to True (or export HF_HUB_OFFLINE=1) after initial model download
    # to prevent any outbound calls to the HuggingFace Hub.
    hf_hub_offline: bool = False
    transformers_offline: bool = False

    # ── Model lifecycle (preload / offload) ───────────────────────────────
    # Eagerly load all GPU-bound models at FastAPI startup.
    warmup_on_startup: bool = True
    # Seconds of inactivity after which all models are offloaded from VRAM.
    # Set to 0 to disable automatic offload.
    model_idle_timeout: int = 600


# Module-level singleton — import this directly in service modules.
settings = Settings()
