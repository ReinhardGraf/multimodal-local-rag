"""
FastAPI router — all HTTP endpoints live here.

Endpoints
---------
POST /v1/chunk/hierarchical/file   → document chunking via Docling
POST /v1/vector-store/upsert       → embed + upsert into Qdrant
POST /v1/vector-store/search       → hybrid search with optional reranking
POST /v1/tables/ingest             → ingest CSV/Excel into SQLite + Qdrant
POST /v1/tables/query              → text-to-SQL on ingested tables
POST /v1/tables/delete-by-source   → remove tables by source file path
GET  /v1/warmup                     → preload GPU-bound models into VRAM
GET  /health                        → liveness probe
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Annotated

import asyncpg
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from src.services.vector_store_service import VectorStoreService
from src.services.document_chunker_service import convert_and_chunk
from src.services.model_lifecycle_service import ModelLifecycleService
from src.services.reconciliation_service import ReconciliationService
from src.services.table_store_service import TableStoreService
from src.schemas import (
    DocumentItem,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    TableDeleteRequest,
    TableQueryRequest,
    UpsertRequest,
    UpsertResponse,
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


def get_table_service(request: Request) -> TableStoreService:
    return request.app.state.table_service


# Annotated aliases for concise signatures
VecDep = Annotated[VectorStoreService, Depends(get_vector_service)]
LifecycleDep = Annotated[ModelLifecycleService, Depends(get_lifecycle)]
PgPoolDep = Annotated[asyncpg.Pool, Depends(get_pg_pool)]
TableDep = Annotated[TableStoreService, Depends(get_table_service)]


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


# ── Vector-store upsert ─────────────────────────────────────


@router.post("/v1/vector-store/upsert", response_model=UpsertResponse)
async def vector_store_upsert(
    body: UpsertRequest,
    vector_service: VecDep,
    lifecycle_service: LifecycleDep,
):
    """
    Generate dense (Ollama) + sparse (BM25) embeddings for the supplied
    documents and batch-upsert them into a Qdrant collection.

    The collection is automatically created with the correct vector
    configuration if it does not already exist.
    """
    t0 = time.time()
    collection = body.collection_name or vector_service.default_collection

    # Ensure the target collection has the right schema
    vector_service.ensure_collection(collection)

    count = await vector_service.upsert_documents(
        documents=body.documents,
        collection_name=collection,
    )

    lifecycle_service.record_activity()

    return UpsertResponse(
        status="ok",
        upserted_count=count,
        collection_name=collection,
        processing_time=round(time.time() - t0, 3),
    )


# ── Vector-store search ────────────────────────────────────


@router.post("/v1/vector-store/search", response_model=SearchResponse)
async def vector_store_search(
    body: SearchRequest,
    vector_service: VecDep,
    lifecycle_service: LifecycleDep,
):
    """
    Run a hybrid (dense + sparse) search with Reciprocal Rank Fusion
    against a Qdrant collection.

    Returns ranked results with ``id``, ``score``, ``text``, and
    ``metadata`` fields.
    """
    t0 = time.time()
    collection = body.collection_name or vector_service.default_collection

    hits, reranked = await vector_service.search(
        query=body.query,
        collection_name=collection,
        limit=body.limit,
        score_threshold=body.score_threshold,
        rerank=body.rerank,
        rerank_top_k=body.rerank_top_k,
        keywords=body.keywords,
    )

    lifecycle_service.record_activity()

    return SearchResponse(
        results=[SearchResultItem(**h) for h in hits],
        collection_name=collection,
        processing_time=round(time.time() - t0, 3),
        reranked=reranked,
    )


# ── Model warmup ────────────────────────────────────────────


@router.get("/v1/warmup")
async def warmup(lifecycle_service: LifecycleDep):
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
    status = await lifecycle_service.warmup()
    return {"status": "ok", "components": status}


# ── Health ───────────────────────────────────────────────────


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/v1/reconciliation/file-hash")
async def file_hash_reconciliation(vector_service: VecDep, pg_pool: PgPoolDep):
    service = ReconciliationService(
        qdrant_client=vector_service.qdrant,
        pg_pool=pg_pool,
    )
    result = await service.reconcile_file_hashes()
    return result


@router.get("/v1/file-hashes/count")
async def get_file_hashes_count(pg_pool: PgPoolDep):
    """
    Get the number of files in the file_hashes table.

    Returns
    -------
    dict
        A dictionary containing the count of files in the file_hashes table.
    """
    async with pg_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT COUNT(*) FROM file_hashes")
        count = row[0] if row else 0
        return {"count": count}


# ── Table RAG: ingest ────────────────────────────────────────


@router.post("/v1/tables/ingest")
async def table_ingest(
    table_service: TableDep,
    vector_service: VecDep,
    lifecycle_service: LifecycleDep,
    files: UploadFile = File(...),
    file_path: str = Form(...),
    file_hash: str = Form(...),
):
    """
    Ingest a table file (CSV / Excel).

    1. Parse the file into sheets / headers / rows.
    2. Call Ollama to generate a description and column types.
    3. Create a typed SQLite table and bulk-insert all rows.
    4. Embed the table description in the ``tables`` Qdrant collection.
    """
    t0 = time.time()
    file_bytes = await files.read()
    filename = files.filename or "upload.csv"
    logger.info("Table ingest: %s (%d bytes)", filename, len(file_bytes))

    sheets = table_service.parse_table_file(file_bytes, filename)

    results: list[dict] = []
    embedding_docs: list[DocumentItem] = []

    for sheet in sheets:
        table_id = str(uuid.uuid4())
        headers = sheet["headers"]
        rows = sheet["rows"]
        sheet_name = sheet["sheet_name"]
        sample_rows = rows[:3]

        # Ollama: description + column types
        ollama_result = await table_service.describe_table_with_ollama(
            headers=headers,
            sample_rows=sample_rows,
            file_name=filename,
            sheet_name=sheet_name,
        )

        description = ollama_result.get("description", f"Table from {filename}")
        column_types = ollama_result.get("columns", [])

        # SQLite: create table + insert
        table_name = table_service.create_and_populate_table(
            table_id=table_id,
            headers=headers,
            column_types=column_types,
            rows=rows,
            source_path=file_path,
            file_hash=file_hash,
            sheet_name=sheet_name,
            description=description,
        )

        # Build embedding text: description + header + sample rows
        sample_text = " | ".join(headers) + "\n"
        for sr in sample_rows[:2]:
            sample_text += " | ".join(str(v) if v is not None else "" for v in sr) + "\n"

        embed_text = (
            f"{description}\n\nColumns: {', '.join(headers)}\n\n"
            f"Sample data:\n{sample_text}"
        )

        embedding_docs.append(
            DocumentItem(
                id=table_id,
                text=embed_text,
                metadata={
                    "table_id": table_id,
                    "table_name": table_name,
                    "source": file_path,
                    "fileName": filename,
                    "fileHash": file_hash,
                    "sheetName": sheet_name,
                    "contentType": "table",
                    "description": description,
                    "columns": json.dumps([c["name"] for c in column_types]),
                    "rowCount": len(rows),
                },
            )
        )

        results.append(
            {
                "table_id": table_id,
                "table_name": table_name,
                "sheet_name": sheet_name,
                "description": description,
                "row_count": len(rows),
                "column_count": len(headers),
            }
        )

    # Upsert embeddings into the tables Qdrant collection
    if embedding_docs:
        from src.config import settings

        coll = settings.table_collection_name
        vector_service.ensure_collection(coll)
        await vector_service.upsert_documents(
            documents=embedding_docs, collection_name=coll
        )
        lifecycle_service.record_activity()

    return {
        "status": "ok",
        "tables": results,
        "total_tables": len(results),
        "processing_time": round(time.time() - t0, 3),
    }


# ── Table RAG: query ─────────────────────────────────────────


@router.post("/v1/tables/query")
async def table_query(
    body: TableQueryRequest,
    table_service: TableDep,
    lifecycle_service: LifecycleDep,
):
    """
    Generate a SQL query from a natural-language question and execute it
    against the ingested table data stored in SQLite.

    Includes automatic retry with error feedback if the first SQL attempt fails.
    """
    t0 = time.time()
    schemas = table_service.get_table_schemas(body.table_ids)
    if not schemas:
        return {"status": "error", "error": "No valid tables found for the given IDs"}

    # Generate SQL
    sql_result = await table_service.generate_sql(
        user_query=body.user_query, table_schemas=schemas
    )
    sql = sql_result.get("sql", "")
    explanation = sql_result.get("explanation", "")

    # Execute with up to 2 retries on SQL errors
    last_error = None
    for attempt in range(3):
        try:
            query_result = table_service.execute_sql(sql)

            citations = []
            for s in schemas:
                cite = f"[Quelle: {s['source_path']}"
                if s.get("sheet_name"):
                    cite += f", Sheet: {s['sheet_name']}"
                cite += f", Tabelle: {s['table_name']}]"
                citations.append(cite)

            lifecycle_service.record_activity()

            return {
                "status": "ok",
                "sql": sql,
                "explanation": explanation,
                "columns": query_result["columns"],
                "rows": query_result["rows"],
                "row_count": query_result["row_count"],
                "citations": citations,
                "table_schemas": [
                    {
                        "table_id": s["table_id"],
                        "source_path": s["source_path"],
                        "sheet_name": s["sheet_name"],
                        "table_name": s["table_name"],
                    }
                    for s in schemas
                ],
                "processing_time": round(time.time() - t0, 3),
            }
        except ValueError as exc:
            last_error = str(exc)
            if attempt < 2 and "SQL execution error" in last_error:
                sql_result = await table_service.generate_sql_with_error_feedback(
                    user_query=body.user_query,
                    table_schemas=schemas,
                    previous_sql=sql,
                    error=last_error,
                )
                sql = sql_result.get("sql", sql)
                explanation = sql_result.get("explanation", "")
            else:
                break

    return {"status": "error", "error": last_error}


# ── Table RAG: delete by source ──────────────────────────────


@router.post("/v1/tables/delete-by-source")
async def table_delete_by_source(
    body: TableDeleteRequest,
    table_service: TableDep,
    vector_service: VecDep,
):
    """
    Delete all tables (SQLite + Qdrant) originating from *source_path*.
    """
    from qdrant_client.http import models as qmodels
    from src.config import settings

    deleted_ids = table_service.delete_tables_by_source(body.source_path)

    # Also delete from Qdrant tables collection
    if deleted_ids:
        try:
            vector_service.qdrant.delete(
                collection_name=settings.table_collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="source",
                                match=qmodels.MatchValue(value=body.source_path),
                            )
                        ]
                    )
                ),
            )
        except Exception:
            logger.warning(
                "Failed to delete table vectors for %s (collection may not exist)",
                body.source_path,
            )

    return {"status": "ok", "deleted_table_ids": deleted_ids}
