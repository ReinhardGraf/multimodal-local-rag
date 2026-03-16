"""
Vector-store service — generates dense + sparse embeddings and upserts
point batches into a Qdrant collection.

Dense vectors : Ollama  ``qllama/multilingual-e5-large-instruct:latest``
Sparse vectors: fastembed ``Qdrant/bm25``  (language-agnostic BM25 that
                handles German compound words well via subword tokenisation)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Optional

from src.config import settings

import httpx
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logger = logging.getLogger(__name__)

# ── Request / response models ────────────────────────────────


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


# ── Service ──────────────────────────────────────────────────

# Embedding-batch size when calling Ollama (tune to fit your GPU VRAM).
# Sourced from settings / OLLAMA_EMBED_BATCH_SIZE env var.
_OLLAMA_EMBED_BATCH: int = settings.ollama_embed_batch_size
COLLECTION_TEXT_FIELD = "text"


# Default factor by which to over-fetch candidates before reranking.
# E.g. with limit=5 and factor=3 we retrieve 15 candidates, rerank,
# and return the best 5.  Tune via the RERANK_OVERFETCH_FACTOR env var.
_RERANK_OVERFETCH_FACTOR: int = settings.rerank_overfetch_factor


class VectorStoreService:
    """Manages embeddings and Qdrant upserts."""

    def __init__(self) -> None:
        self.qdrant_url: str = settings.qdrant_url
        self.ollama_url: str = settings.ollama_url
        self.embedding_model: str = settings.embedding_model
        self.default_collection: str = settings.qdrant_collection_name
        self.embedding_dimension: int = settings.embedding_dimension

        # Lazy-initialised heavy resources
        self._qdrant: Optional[QdrantClient] = None
        self._sparse_encoder: Any = None  # fastembed.SparseTextEmbedding
        self._reranker: Any = None  # RerankerService (lazy)
        self._http_client: Optional[httpx.AsyncClient] = None

    # ── Lazy properties ──────────────────────────────────────

    @property
    def qdrant(self) -> QdrantClient:
        if self._qdrant is None:
            self._qdrant = QdrantClient(url=self.qdrant_url)
            logger.info("QdrantClient connected to %s", self.qdrant_url)
        return self._qdrant

    @property
    def reranker(self):
        """Return (lazily create) the RerankerService."""
        if self._reranker is None:
            from src.services.reranker_service import RerankerService

            self._reranker = RerankerService()
            logger.info("RerankerService initialised")
        return self._reranker

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Return (lazily create) a reusable async HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=300.0)
            logger.info("Async HTTP client created")
        return self._http_client

    @property
    def sparse_encoder(self):
        """Return (lazily create) the BM25 sparse encoder from *fastembed*."""
        if self._sparse_encoder is None:
            from fastembed import SparseTextEmbedding

            self._sparse_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
            logger.info("Sparse encoder loaded (Qdrant/bm25)")
        return self._sparse_encoder

    # ── Resource management ──────────────────────────────────

    def offload_sparse_encoder(self) -> None:
        """Release the BM25 sparse encoder to free memory."""
        self._sparse_encoder = None

    async def close(self) -> None:
        """Shut down the async HTTP client and Qdrant connection."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        if self._qdrant is not None:
            self._qdrant.close()
            self._qdrant = None

    # ── Dense embeddings (Ollama) ────────────────────────────

    async def _get_dense_embeddings(
        self, texts: list[str]
    ) -> list[list[float]]:
        """Call Ollama ``/api/embed`` in batches and return dense vectors."""
        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), _OLLAMA_EMBED_BATCH):
            batch = texts[start : start + _OLLAMA_EMBED_BATCH]
            response = await self.http_client.post(
                f"{self.ollama_url}/api/embed",
                json={"model": self.embedding_model, "input": batch, "keep_alive": "10m"},
            )
            response.raise_for_status()
            all_embeddings.extend(response.json()["embeddings"])
            await response.aclose()
        return all_embeddings

    # ── Sparse embeddings (fastembed BM25) ───────────────────

    def _get_sparse_embeddings(self, texts: list[str]) -> list[qmodels.SparseVector]:
        """Generate BM25 sparse vectors.

        ``Qdrant/bm25`` uses subword tokenisation which handles German
        compound words well without language-specific configuration.
        """
        raw = list(self.sparse_encoder.embed(texts))
        return [
            qmodels.SparseVector(
                indices=vec.indices.tolist(),
                values=vec.values.tolist(),
            )
            for vec in raw
        ]

    # ── Collection management ────────────────────────────────

    def ensure_collection(self, collection_name: str | None = None) -> None:
        """Create the collection with named dense + sparse vectors if it
        does not already exist."""
        name = collection_name or self.default_collection
        try:
            self.qdrant.get_collection(name)
            logger.info("Collection '%s' already exists", name)
        except Exception:
            logger.info("Creating collection '%s' …", name)
            self.qdrant.create_collection(
                collection_name=name,
                vectors_config={
                    "dense": qmodels.VectorParams(
                        size=self.embedding_dimension,
                        distance=qmodels.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": qmodels.SparseVectorParams(
                        modifier=qmodels.Modifier.IDF,
                    ),
                },
            )
            logger.info("Collection '%s' created", name)

    # ── Upsert ───────────────────────────────────────────────

    async def upsert_documents(
        self,
        documents: list[DocumentItem],
        collection_name: str | None = None,
    ) -> int:
        """Embed *documents* and batch-upsert into Qdrant.

        Returns the number of points upserted.
        """
        name = collection_name or self.default_collection
        texts = [doc.text for doc in documents]

        logger.info(
            "Generating embeddings for %d documents (collection=%s) …",
            len(texts),
            name,
        )

        # Generate both embedding types
        dense_embeddings = await self._get_dense_embeddings(texts)
        sparse_embeddings = await asyncio.to_thread(
            self._get_sparse_embeddings, texts
        )

        # Build Qdrant points
        points: list[qmodels.PointStruct] = []
        for i, doc in enumerate(documents):
            point_id = doc.id or str(uuid.uuid4())
            payload: dict[str, Any] = {COLLECTION_TEXT_FIELD: doc.text}
            if doc.metadata:
                payload.update(doc.metadata)

            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_embeddings[i],
                        "sparse": sparse_embeddings[i],
                    },
                    payload=payload,
                )
            )

        # Batch upsert
        await asyncio.to_thread(
            self.qdrant.upsert, collection_name=name, points=points
        )
        logger.info("Upserted %d points into '%s'", len(points), name)

        return len(points)

    # ── Search ───────────────────────────────────────────────

    async def search(
        self,
        query: str,
        collection_name: str | None = None,
        limit: int = 5,
        score_threshold: float | None = None,
        rerank: bool = False,
        rerank_top_k: int | None = None,
        keywords: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Run a hybrid (dense + sparse) search using Reciprocal Rank Fusion,
        optionally followed by cross-encoder reranking.

        When *rerank* is ``True`` the service over-fetches candidates
        (``limit × RERANK_OVERFETCH_FACTOR``), scores them with the
        ``RerankerService``, and returns the top results.

        Parameters
        ----------
        keywords:
            Optional list of keywords.  When provided the **sparse
            (BM25) leg** of the hybrid search is built from these
            keywords instead of from *query*.  The **dense (semantic)
            leg** always uses *query*.  This gives true hybrid search:
            semantic understanding via the dense model + exact keyword
            matching via BM25.

        Returns
        -------
        tuple[list[dict], bool]
            A pair of (results, reranked).  ``reranked`` is ``True`` when
            the cross-encoder actually ran.
        """
        name = collection_name or self.default_collection

        # When reranking, pull more candidates so the reranker has
        # a larger pool to re-score.
        retrieval_limit = limit * _RERANK_OVERFETCH_FACTOR if rerank else limit

        # Dense embeddings always use the semantic query.
        dense_embs = await self._get_dense_embeddings([query])
        dense_emb = dense_embs[0]

        # Sparse (BM25) embeddings: use explicit keywords when provided,
        # otherwise fall back to the query string.
        sparse_input = " ".join(keywords) if keywords else query
        sparse_embs = await asyncio.to_thread(
            self._get_sparse_embeddings, [sparse_input]
        )
        sparse_emb = sparse_embs[0]

        prefetch = [
            qmodels.Prefetch(
                query=dense_emb,
                using="dense",
                limit=retrieval_limit * 2,
            ),
            qmodels.Prefetch(
                query=sparse_emb,
                using="sparse",
                limit=retrieval_limit * 2,
            ),
        ]

        results = await asyncio.to_thread(
            self.qdrant.query_points,
            collection_name=name,
            prefetch=prefetch,
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=retrieval_limit,
            score_threshold=score_threshold,
            with_vectors=False,
        )

        hits = [
            {
                "id": str(point.id),
                "score": point.score,
                "text": point.payload.get(COLLECTION_TEXT_FIELD, ""),
                "metadata": {k: v for k, v in point.payload.items() if k != COLLECTION_TEXT_FIELD},
            }
            for point in results.points
        ]

        # ── Optional reranking ───────────────────────────────
        reranked = False
        if rerank and hits:
            final_k = rerank_top_k if rerank_top_k is not None else limit
            hits = await asyncio.to_thread(
                self.reranker.rerank, query, hits, top_k=final_k
            )
            reranked = True

        return hits, reranked
