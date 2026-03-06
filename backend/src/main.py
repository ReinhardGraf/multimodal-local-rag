"""
Docling RAG Backend — FastAPI application entry-point.

Run with::

    uvicorn src.main:app --host 0.0.0.0 --port 5008
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.router import router, _vector_service

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: warmup models on startup, offload on shutdown."""
    from backend.src.services.model_lifecycle_service import lifecycle

    # Wire the shared service instances into the lifecycle manager so it
    # can touch the lazy properties and null them out on offload.
    lifecycle._vec_store = _vector_service
    lifecycle._reranker = _vector_service.reranker  # lightweight constructor

    if settings.warmup_on_startup:
        logging.getLogger(__name__).info(
            "WARMUP_ON_STARTUP=True — loading models into VRAM …"
        )
        await lifecycle.warmup()

    # Start the idle-watcher background task
    lifecycle.start_watcher()

    yield  # application is running

    # ── Shutdown ─────────────────────────────────────────────
    lifecycle.stop_watcher()
    logging.getLogger(__name__).info("FastAPI shutdown — lifecycle watcher stopped")


app = FastAPI(title="RAG Backend", version="2.0.0", lifespan=lifespan)

# Allow the OpenWebUI origin (and localhost dev setups) to call /v1/warmup
# from browser-side JavaScript.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your OpenWebUI URL in production
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5008)
