"""
Pydantic models for the multimodal RAG application.

These models define the request and response structures for the API endpoints.
"""

from typing import Any, Optional

from pydantic import BaseModel


class DocumentItem(BaseModel):
    """A single document to embed and upsert."""

    id: Optional[str] = None
    text: str
    metadata: Optional[dict[str, Any]] = None


class UpsertRequest(BaseModel):
    """Body for ``POST /v1/vector-store/upsert``."""

    collection_name: Optional[str] = None
    documents: list[DocumentItem]


class UpsertResponse(BaseModel):
    status: str
    upserted_count: int
    collection_name: str
    processing_time: float


class SearchRequest(BaseModel):
    """Body for ``POST /v1/vector-store/search``."""

    query: str
    collection_name: Optional[str] = None
    limit: int = 5
    score_threshold: Optional[float] = None
    rerank: bool = False
    rerank_top_k: Optional[int] = None
    keywords: Optional[list[str]] = None


class SearchResultItem(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    collection_name: str
    processing_time: float
    reranked: bool = False
