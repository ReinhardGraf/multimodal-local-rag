"""
Shared pytest fixtures for the backend test-suite.

The key fixture is ``qdrant_client`` — a real Qdrant client backed by an
**in-memory** store (no Docker/external service needed).
"""

from __future__ import annotations

import random
from typing import Generator

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from backend.src.services.vector_store_service import VectorStoreService

# ── Constants ────────────────────────────────────────────────

EMBEDDING_DIM = 1024
TEST_COLLECTION = "test_documents"


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture(scope="session")
def qdrant_client() -> Generator[QdrantClient, None, None]:
    """Session-scoped in-memory Qdrant client.

    Every test run starts with a completely fresh store.
    """
    client = QdrantClient(location=":memory:")
    yield client
    client.close()


@pytest.fixture()
def collection(qdrant_client: QdrantClient) -> Generator[str, None, None]:
    """Create a fresh test collection (dense + sparse) before each test and
    delete it afterwards so tests are fully isolated."""
    qdrant_client.create_collection(
        collection_name=TEST_COLLECTION,
        vectors_config={
            "dense": qmodels.VectorParams(
                size=EMBEDDING_DIM,
                distance=qmodels.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": qmodels.SparseVectorParams(
                modifier=qmodels.Modifier.IDF,
            ),
        },
    )
    yield TEST_COLLECTION
    qdrant_client.delete_collection(TEST_COLLECTION)


@pytest.fixture()
def vector_service(
    qdrant_client: QdrantClient,
) -> VectorStoreService:
    """Return a ``VectorStoreService`` wired to the in-memory Qdrant client.

    The Ollama-based dense embeddings are replaced by deterministic random
    vectors so tests don't need a live Ollama instance.  Sparse (BM25)
    embeddings use the real *fastembed* encoder.
    """
    svc = VectorStoreService()
    svc._qdrant = qdrant_client
    svc.default_collection = TEST_COLLECTION
    svc.embedding_dimension = EMBEDDING_DIM
    return svc


def _deterministic_dense_embedding(text: str) -> list[float]:
    """Produce a deterministic 1024-d vector seeded by the text hash.

    Using a deterministic seed ensures the *same* text always yields the
    *same* embedding, which makes query-relevance assertions meaningful.
    """
    rng = random.Random(hash(text))
    vec = [rng.gauss(0, 1) for _ in range(EMBEDDING_DIM)]
    # L2-normalise so cosine similarity behaves well
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec]


@pytest.fixture(autouse=True)
def _patch_dense_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the Ollama HTTP call with a local deterministic function.

    This runs automatically for every test (``autouse=True``).
    """

    async def _fake_dense(self: VectorStoreService, texts: list[str]) -> list[list[float]]:
        return [_deterministic_dense_embedding(t) for t in texts]

    monkeypatch.setattr(
        VectorStoreService, "_get_dense_embeddings", _fake_dense
    )
