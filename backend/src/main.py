"""
Docling RAG Backend — FastAPI application entry-point.

Run with::

    uvicorn src.main:app --host 0.0.0.0 --port 5008
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.router import router

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: create services, warmup models on startup, clean up on shutdown."""
    from backend.src.services.model_lifecycle_service import ModelLifecycleService
    from backend.src.services.vector_store_service import VectorStoreService

    log = logging.getLogger(__name__)

    # ── Create shared service instances ──────────────────────
    vec = VectorStoreService()
    lc = ModelLifecycleService(vec_store=vec, reranker=vec.reranker)
    pg_pool = await asyncpg.create_pool(dsn=settings.postgres_dsn)

    app.state.vector_service = vec
    app.state.lifecycle = lc
    app.state.pg_pool = pg_pool

    if settings.warmup_on_startup:
        log.info("WARMUP_ON_STARTUP=True — loading models into VRAM …")
        await lc.warmup()

    # Start the idle-watcher background task
    lc.start_watcher()

    yield  # application is running

    # ── Shutdown ─────────────────────────────────────────────
    lc.stop_watcher()
    await vec.close()
    await pg_pool.close()
    log.info("FastAPI shutdown — all resources released")


app = FastAPI(title="RAG Backend", version="2.0.0", lifespan=lifespan)

# Allow the OpenWebUI origin (and localhost dev setups) to call /v1/warmup
# from browser-side JavaScript.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5008)
