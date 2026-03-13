"""
FastAPI router — all HTTP endpoints live here.

Endpoints
---------
POST /v1/chunk/hierarchical/file   → document chunking via Docling
POST /v1/vector-store/upsert       → embed + upsert into Qdrant
POST /v1/vector-store/search       → hybrid search with optional reranking
GET  /v1/warmup                     → preload GPU-bound models into VRAM
GET  /health                        → liveness probe
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

import asyncpg
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from backend.src.services.document_chunker_service import convert_and_chunk
from backend.src.services.model_lifecycle_service import ModelLifecycleService
from backend.src.services.reconciliation_service import ReconciliationService
from backend.src.services.vector_store_service import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    UpsertRequest,
    UpsertResponse,
    VectorStoreService,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Dependency helpers ───────────────────────────────────────


def get_vector_service(request: Request) -> VectorStoreService:
    return request.app.state.vector_service


def get_lifecycle(request: Request) -> ModelLifecycleService:
    return request.app.state.lifecycle


def get_pg_pool(request: Request) -> asyncpg.Pool:
    return request.app.state.pg_pool


# Annotated aliases for concise signatures
VecDep = Annotated[VectorStoreService, Depends(get_vector_service)]
LifecycleDep = Annotated[ModelLifecycleService, Depends(get_lifecycle)]
PgPoolDep = Annotated[asyncpg.Pool, Depends(get_pg_pool)]


# ── Document chunking ───────────────────────────────────────


@router.post("/v1/chunk/hierarchical/file")
async def chunk_hierarchical_file(
    files: UploadFile = File(...),
    # Chunking params (kept for API compatibility)
    chunking_include_raw_text: bool = Form(default=False),
    # Conversion params
    include_converted_doc: bool = Form(default=True),
    convert_do_table_structure: bool = Form(default=True),
    convert_do_ocr: bool = Form(default=False),
):
    """
    Accept a file upload, convert with Docling, chunk with
    HybridChunker, and return a ``ChunkDocumentResponse``-compatible
    JSON.

    Note
    ----
    HybridChunker produces structure-based chunks (respects document
    hierarchy) and does **not** enforce token limits.  ``chunking_max_tokens``,
    ``chunking_tokenizer``, and ``chunking_merge_peers`` are accepted for
    API compatibility but are **not** used by this chunker.
    """
    try:
        file_bytes = await files.read()
        filename = files.filename or "upload.pdf"
        logger.info("Received file: %s (%d bytes)", filename, len(file_bytes))

        result = convert_and_chunk(
            file_bytes,
            filename,
            chunking_include_raw_text=chunking_include_raw_text,
            include_converted_doc=include_converted_doc,
            convert_do_ocr=convert_do_ocr,
            convert_do_table_structure=convert_do_table_structure,
        )

        return JSONResponse(content=result)

    except Exception as e:
        logger.exception("Error processing %s: %s", files.filename, e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


# ── Vector-store upsert ─────────────────────────────────────


@router.post("/v1/vector-store/upsert", response_model=UpsertResponse)
async def vector_store_upsert(
    body: UpsertRequest,
    vec: VecDep,
    lc: LifecycleDep,
):
    """
    Generate dense (Ollama) + sparse (BM25) embeddings for the supplied
    documents and batch-upsert them into a Qdrant collection.

    The collection is automatically created with the correct vector
    configuration if it does not already exist.
    """
    t0 = time.time()
    collection = body.collection_name or vec.default_collection

    try:
        # Ensure the target collection has the right schema
        vec.ensure_collection(collection)

        count = await vec.upsert_documents(
            documents=body.documents,
            collection_name=collection,
        )

        lc.record_activity()

        return UpsertResponse(
            status="ok",
            upserted_count=count,
            collection_name=collection,
            processing_time=round(time.time() - t0, 3),
        )

    except Exception as e:
        logger.exception("Upsert failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "processing_time": round(time.time() - t0, 3),
            },
        )


# ── Vector-store search ────────────────────────────────────


@router.post("/v1/vector-store/search", response_model=SearchResponse)
async def vector_store_search(
    body: SearchRequest,
    vec: VecDep,
    lc: LifecycleDep,
):
    """
    Run a hybrid (dense + sparse) search with Reciprocal Rank Fusion
    against a Qdrant collection.

    Returns ranked results with ``id``, ``score``, ``text``, and
    ``metadata`` fields.
    """
    t0 = time.time()
    collection = body.collection_name or vec.default_collection

    try:
        hits, reranked = await vec.search(
            query=body.query,
            collection_name=collection,
            limit=body.limit,
            score_threshold=body.score_threshold,
            rerank=body.rerank,
            rerank_top_k=body.rerank_top_k,
            keywords=body.keywords,
        )

        lc.record_activity()

        return SearchResponse(
            results=[SearchResultItem(**h) for h in hits],
            collection_name=collection,
            processing_time=round(time.time() - t0, 3),
            reranked=reranked,
        )

    except Exception as e:
        logger.exception("Search failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "processing_time": round(time.time() - t0, 3),
            },
        )


# ── Model warmup ────────────────────────────────────────────


@router.get("/v1/warmup")
async def warmup(lc: LifecycleDep):
    """
    Preload all GPU-bound models into VRAM.

    Safe to call multiple times — already-loaded models are skipped.
    Called at startup (WARMUP_ON_STARTUP=True) and by the OpenWebUI
    browser-side JS when the user starts typing.

    Returns
    -------
    dict
        Per-component status with ``loaded``, ``already_loaded``, and
        ``ms`` (load time in milliseconds) keys.
    """
    status = await lc.warmup()
    return {"status": "ok", "components": status}


# ── Health ───────────────────────────────────────────────────


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/v1/reconciliation/file-hash")
async def file_hash_reconciliation(vec: VecDep, pg_pool: PgPoolDep):
    try:
        service = ReconciliationService(
            qdrant_client=vec.qdrant,
            pg_pool=pg_pool,
        )
        result = await service.reconcile_file_hashes()
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
            },
        )


@router.get("/v1/file-hashes/count")
async def get_file_hashes_count(pg_pool: PgPoolDep):
    """
    Return the number of files in the file_hashes PostgreSQL table.

    Returns
    -------
    dict
        A dictionary with the key "count" and the number of files as value.
    """
    try:
        async with pg_pool.acquire() as conn:
            result = await conn.fetchval("SELECT COUNT(*) FROM file_hashes")
        return {"count": result}
    except Exception as e:
        logger.exception("Error getting file hashes count: %s", e)
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
            },
        )
