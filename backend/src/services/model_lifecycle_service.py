"""
Model lifecycle service — preload GPU-bound models into VRAM and offload
them after a configurable idle timeout.

Three GPU load points are managed here:

1. **CrossEncoder reranker** (sentence-transformers, MPS/CUDA/CPU)
   — triggered by accessing ``RerankerService.model``
2. **BM25 sparse encoder** (fastembed, CPU) — included so the idle-
   offload is coordinated and quantified in the warmup response
3. **Ollama embedding model** (GPU via Ollama process)
   — loaded by sending a dummy ``/api/embed`` request with
   ``keep_alive=-1``; offloaded by sending the same call with
   ``keep_alive=0``.

Usage
-----
The singleton ``lifecycle`` is imported by both ``main.py``
(startup / lifespan hook) and ``router.py`` (``GET /v1/warmup``).

The idle-watcher coroutine ``_idle_watcher()`` is started as an
asyncio background task in the lifespan hook and cancelled on shutdown.

Configuration
-------------
* ``WARMUP_ON_STARTUP`` — bool, default ``True``
* ``MODEL_IDLE_TIMEOUT`` — int (seconds), default ``300`` (5 min)
  Set to 0 to disable automatic offload.
* ``OLLAMA_EMBED_KEEP_ALIVE`` — str, default ``"-1"``
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


class ModelLifecycleService:
    """Singleton that manages load / offload of all GPU-bound models.

    Thread-model note: FastAPI runs handlers in asyncio.  The cross-encoder
    ``_load_model()`` is synchronous and blocks the event loop for a few
    seconds during initial load.  Because warmup is intentionally done at
    startup (not during a live request), this is acceptable.  If it ever
    becomes a concern, wrap the call in
    ``asyncio.get_event_loop().run_in_executor(None, ...)``.
    """

    def __init__(self, vec_store: Any, reranker: Any) -> None:
        self._reranker = reranker
        self._vec_store = vec_store

        # Track whether models are currently resident in VRAM / RAM
        self._cross_encoder_loaded: bool = False
        self._bm25_loaded: bool = False
        self._ollama_embed_loaded: bool = False

        # Epoch timestamp of the last recorded user activity
        self.last_activity: float = time.time()

        # Background task handle — set by the lifespan hook
        self._watcher_task: asyncio.Task | None = None

    # ── Activity tracking ────────────────────────────────────

    def record_activity(self) -> None:
        """Reset the idle clock.  Call this on every /search or /upsert."""
        self.last_activity = time.time()

    # ── Warmup ───────────────────────────────────────────────

    async def warmup(self) -> dict:
        """Idempotently load all GPU-bound models.

        Returns a status dict with per-component load times (ms) and
        ``already_loaded`` flags.
        """
        self.record_activity()
        status: dict = {}

        # ── BM25 sparse encoder (CPU, fastembed) ─────────────
        t0 = time.time()
        already = self._bm25_loaded
        if not already:
            try:
                # Accessing the property triggers lazy load
                _ = self._vec_store.sparse_encoder
                self._bm25_loaded = True
            except Exception as exc:
                logger.warning("BM25 warmup failed: %s", exc)
                status["bm25"] = {"loaded": False, "error": str(exc)}
        status["bm25"] = {
            "loaded": self._bm25_loaded,
            "already_loaded": already,
            "ms": round((time.time() - t0) * 1000),
        }

        # ── CrossEncoder (sentence-transformers, GPU) ─────────
        t0 = time.time()
        already = self._cross_encoder_loaded
        if not already:
            try:
                _ = self._reranker.model
                self._cross_encoder_loaded = True
            except Exception as exc:
                logger.warning("CrossEncoder warmup failed: %s", exc)
                status["cross_encoder"] = {"loaded": False, "error": str(exc)}
        status["cross_encoder"] = {
            "loaded": self._cross_encoder_loaded,
            "already_loaded": already,
            "ms": round((time.time() - t0) * 1000),
        }

        # ── Ollama embedding model ────────────────────────────
        t0 = time.time()
        already = self._ollama_embed_loaded
        if not already:
            try:
                await self._warmup_ollama_embed()
                self._ollama_embed_loaded = True
            except Exception as exc:
                logger.warning("Ollama embed warmup failed: %s", exc)
                status["ollama_embed"] = {"loaded": False, "error": str(exc)}
        status["ollama_embed"] = {
            "loaded": self._ollama_embed_loaded,
            "already_loaded": already,
            "ms": round((time.time() - t0) * 1000),
        }

        logger.info("Warmup complete: %s", status)
        return status

    async def _warmup_ollama_embed(self) -> None:
        """Send a minimal embed request so Ollama loads the model into VRAM."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{settings.ollama_url}/api/embed",
                json={
                    "model": settings.embedding_model,
                    "input": ["warmup"],
                    "keep_alive": -1,  # pin in VRAM until explicit offload
                },
            )
            resp.raise_for_status()
        logger.info(
            "Ollama embedding model '%s' loaded (keep_alive=-1)",
            settings.embedding_model,
        )

    # ── Offload ──────────────────────────────────────────────

    async def offload(self) -> None:
        """Evict all models from VRAM / RAM.

        * CrossEncoder: set ``_reranker._model = None`` and flush GPU cache
        * BM25: set ``_vec_store._sparse_encoder = None``
        * Ollama embedding model: POST ``keep_alive=0`` to trigger eviction
        """
        logger.info("Offloading all models (idle timeout reached) …")

        # ── CrossEncoder ─────────────────────────────────────
        if self._cross_encoder_loaded and self._reranker is not None:
            self._reranker.offload_model()
            self._cross_encoder_loaded = False
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("CUDA cache cleared")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    # PyTorch ≥ 2.1 exposes mps.empty_cache()
                    if hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
                        logger.info("MPS cache cleared")
            except ImportError:
                pass
            logger.info("CrossEncoder offloaded")

        # ── BM25 ─────────────────────────────────────────────
        if self._bm25_loaded and self._vec_store is not None:
            self._vec_store.offload_sparse_encoder()
            self._bm25_loaded = False
            logger.info("BM25 sparse encoder offloaded")

        # ── Ollama embedding model ────────────────────────────
        if self._ollama_embed_loaded:
            try:
                await self._offload_ollama_embed()
            except Exception as exc:
                logger.warning("Ollama embed offload failed: %s", exc)
            self._ollama_embed_loaded = False

    async def _offload_ollama_embed(self) -> None:
        """Ask Ollama to evict the embedding model from VRAM."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{settings.ollama_url}/api/embed",
                json={
                    "model": settings.embedding_model,
                    "input": ["."],
                    "keep_alive": 0,  # evict immediately
                },
            )
            # 200 or 404 (already evicted) are both fine
            if resp.status_code not in (200, 404):
                resp.raise_for_status()
        logger.info(
            "Ollama embedding model '%s' evicted from VRAM",
            settings.embedding_model,
        )

    # ── Idle watcher ─────────────────────────────────────────

    async def _idle_watcher(self) -> None:
        """Background coroutine: offload models after idle timeout.

        Polls every 60 seconds.  Exits cleanly when cancelled (shutdown).
        """
        timeout = settings.model_idle_timeout
        if timeout <= 0:
            logger.info("Idle-offload disabled (MODEL_IDLE_TIMEOUT=0)")
            return

        logger.info(
            "Idle-watcher started (timeout=%d s, poll=60 s)", timeout
        )
        try:
            while True:
                await asyncio.sleep(60)
                idle_for = time.time() - self.last_activity
                any_loaded = (
                    self._cross_encoder_loaded
                    or self._bm25_loaded
                    or self._ollama_embed_loaded
                )
                if any_loaded and idle_for >= timeout:
                    logger.info(
                        "Idle for %.0f s (> %d s) — offloading models …",
                        idle_for,
                        timeout,
                    )
                    await self.offload()
        except asyncio.CancelledError:
            logger.info("Idle-watcher task cancelled (shutdown)")

    def start_watcher(self) -> None:
        """Schedule the idle-watcher as a background asyncio task."""
        if self._watcher_task is None or self._watcher_task.done():
            self._watcher_task = asyncio.create_task(self._idle_watcher())

    def stop_watcher(self) -> None:
        """Cancel the idle-watcher task (called on shutdown)."""
        if self._watcher_task and not self._watcher_task.done():
            self._watcher_task.cancel()

