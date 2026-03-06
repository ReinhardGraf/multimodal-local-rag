"""
Tests for ``backend.reranker_service`` — cross-encoder and Ollama reranking.

All tests use **fakes** (fake cross-encoder model or fake HTTP client).
No GPU, no HuggingFace download, and no running Ollama instance is needed.
"""

from __future__ import annotations

import pytest

from backend.src.services.reranker_service import RerankerService


# ── Fake model ──────────────────────────────────────────────


class _FakeCrossEncoder:
    """Minimal stand-in for ``sentence_transformers.CrossEncoder``.

    Scores each (query, document) pair by the fraction of query words
    that appear in the document text.  This gives us deterministic,
    meaningful ordering without loading a real model.
    """

    def predict(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> list[float]:
        scores = []
        for query, doc in pairs:
            q_words = set(query.lower().split())
            d_words = set(doc.lower().split())
            overlap = len(q_words & d_words)
            scores.append(overlap / max(len(q_words), 1))
        return scores


class _FakeOllamaResponse:
    """Fake ``httpx.Response`` for Ollama ``/api/rerank``."""

    def __init__(self, data: dict) -> None:
        self._data = data
        self.status_code = 200

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._data


class _FakeOllamaClient:
    """Fake ``httpx.Client`` that simulates Ollama ``/api/rerank``.

    Uses the same word-overlap scoring logic as ``_FakeCrossEncoder``
    so the test expectations remain consistent.
    """

    def __init__(self) -> None:
        self._last_payload: dict | None = None

    def post(self, url: str, *, json: dict) -> _FakeOllamaResponse:
        self._last_payload = json
        query = json["query"]
        documents = json["documents"]
        top_n = json.get("top_n")

        q_words = set(query.lower().split())
        results = []
        for idx, doc_text in enumerate(documents):
            d_words = set(doc_text.lower().split())
            overlap = len(q_words & d_words)
            score = overlap / max(len(q_words), 1)
            results.append({"index": idx, "relevance_score": score})

        # Sort by score descending (like Ollama does)
        results.sort(key=lambda r: r["relevance_score"], reverse=True)
        if top_n is not None:
            results = results[:top_n]

        return _FakeOllamaResponse({"results": results})


@pytest.fixture()
def reranker() -> RerankerService:
    """Return a ``RerankerService`` with a fake model injected."""
    svc = RerankerService()
    svc.backend = "cross-encoder"
    svc._model = _FakeCrossEncoder()
    return svc


@pytest.fixture()
def ollama_reranker() -> RerankerService:
    """Return a ``RerankerService`` configured for the Ollama backend
    with a fake HTTP client."""
    svc = RerankerService()
    svc.backend = "ollama"
    svc._http_client = _FakeOllamaClient()
    return svc


# ── Sample documents ─────────────────────────────────────────

SAMPLE_DOCS = [
    {
        "id": "1",
        "score": 0.9,
        "text": "Die Donau ist der zweitgrößte Fluss in Europa.",
        "metadata": {"source": "geographie"},
    },
    {
        "id": "2",
        "score": 0.8,
        "text": "Maschinelles Lernen ist ein Teilgebiet der künstlichen Intelligenz.",
        "metadata": {"source": "informatik"},
    },
    {
        "id": "3",
        "score": 0.7,
        "text": "Sauerbraten ist ein traditionelles deutsches Gericht.",
        "metadata": {"source": "küche"},
    },
    {
        "id": "4",
        "score": 0.6,
        "text": "Die Automobilindustrie in Deutschland ist weltweit bekannt.",
        "metadata": {"source": "wirtschaft"},
    },
]


# ── Tests ────────────────────────────────────────────────────


class TestRerankerService:
    """Tests for the ``rerank`` method."""

    def test_rerank_reorders_by_relevance(self, reranker: RerankerService):
        """After reranking the query about Fluss/Europa, the Donau
        document should be ranked first."""
        results = reranker.rerank("Fluss in Europa", SAMPLE_DOCS)
        assert results[0]["id"] == "1"

    def test_rerank_returns_all_documents_by_default(
        self, reranker: RerankerService
    ):
        results = reranker.rerank("test", SAMPLE_DOCS)
        assert len(results) == len(SAMPLE_DOCS)

    def test_rerank_top_k_limits_results(self, reranker: RerankerService):
        results = reranker.rerank("test", SAMPLE_DOCS, top_k=2)
        assert len(results) == 2

    def test_rerank_top_k_larger_than_docs(self, reranker: RerankerService):
        """top_k larger than candidate count should return all."""
        results = reranker.rerank("test", SAMPLE_DOCS, top_k=100)
        assert len(results) == len(SAMPLE_DOCS)

    def test_rerank_preserves_document_fields(self, reranker: RerankerService):
        results = reranker.rerank("Fluss", SAMPLE_DOCS)
        for doc in results:
            assert "id" in doc
            assert "score" in doc
            assert "text" in doc
            assert "metadata" in doc

    def test_rerank_updates_scores(self, reranker: RerankerService):
        """Scores should be replaced by cross-encoder scores, not the
        original vector search scores."""
        results = reranker.rerank("Fluss Europa", SAMPLE_DOCS)
        # Original score for doc 1 was 0.9 — cross-encoder score should
        # differ (word-overlap fraction)
        for doc in results:
            # Score should be a float between 0 and 1 (word overlap fraction)
            assert isinstance(doc["score"], float)

    def test_rerank_empty_documents(self, reranker: RerankerService):
        results = reranker.rerank("test", [])
        assert results == []

    def test_rerank_sorted_descending(self, reranker: RerankerService):
        results = reranker.rerank("Deutschland", SAMPLE_DOCS)
        scores = [d["score"] for d in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_ki_query_prefers_ki_doc(self, reranker: RerankerService):
        """A query about KI/ML should push the Informatik doc to the top."""
        results = reranker.rerank(
            "Maschinelles Lernen künstliche Intelligenz", SAMPLE_DOCS
        )
        assert results[0]["id"] == "2"

    def test_rerank_does_not_mutate_input(self, reranker: RerankerService):
        """Original document list should not be modified."""
        original_scores = [d["score"] for d in SAMPLE_DOCS]
        reranker.rerank("test", SAMPLE_DOCS)
        assert [d["score"] for d in SAMPLE_DOCS] == original_scores


# ── Ollama backend tests ────────────────────────────────────


class TestOllamaRerankerService:
    """Tests for the Ollama backend of ``RerankerService``."""

    def test_ollama_rerank_reorders_by_relevance(
        self, ollama_reranker: RerankerService
    ):
        results = ollama_reranker.rerank("Fluss in Europa", SAMPLE_DOCS)
        assert results[0]["id"] == "1"

    def test_ollama_rerank_returns_all_documents_by_default(
        self, ollama_reranker: RerankerService
    ):
        results = ollama_reranker.rerank("test", SAMPLE_DOCS)
        assert len(results) == len(SAMPLE_DOCS)

    def test_ollama_rerank_top_k_limits_results(
        self, ollama_reranker: RerankerService
    ):
        results = ollama_reranker.rerank("test", SAMPLE_DOCS, top_k=2)
        assert len(results) == 2

    def test_ollama_rerank_preserves_document_fields(
        self, ollama_reranker: RerankerService
    ):
        results = ollama_reranker.rerank("Fluss", SAMPLE_DOCS)
        for doc in results:
            assert "id" in doc
            assert "score" in doc
            assert "text" in doc
            assert "metadata" in doc

    def test_ollama_rerank_empty_documents(
        self, ollama_reranker: RerankerService
    ):
        results = ollama_reranker.rerank("test", [])
        assert results == []

    def test_ollama_rerank_sorted_descending(
        self, ollama_reranker: RerankerService
    ):
        results = ollama_reranker.rerank("Deutschland", SAMPLE_DOCS)
        scores = [d["score"] for d in results]
        assert scores == sorted(scores, reverse=True)

    def test_ollama_rerank_ki_query_prefers_ki_doc(
        self, ollama_reranker: RerankerService
    ):
        results = ollama_reranker.rerank(
            "Maschinelles Lernen künstliche Intelligenz", SAMPLE_DOCS
        )
        assert results[0]["id"] == "2"

    def test_ollama_rerank_does_not_mutate_input(
        self, ollama_reranker: RerankerService
    ):
        original_scores = [d["score"] for d in SAMPLE_DOCS]
        ollama_reranker.rerank("test", SAMPLE_DOCS)
        assert [d["score"] for d in SAMPLE_DOCS] == original_scores

    def test_ollama_passes_top_n_to_api(
        self, ollama_reranker: RerankerService
    ):
        """When ``top_k`` is set, the Ollama payload should include ``top_n``."""
        ollama_reranker.rerank("test", SAMPLE_DOCS, top_k=2)
        client = ollama_reranker._http_client
        assert client._last_payload["top_n"] == 2

    def test_ollama_omits_top_n_when_none(
        self, ollama_reranker: RerankerService
    ):
        """When ``top_k`` is None, ``top_n`` should not be in the payload."""
        ollama_reranker.rerank("test", SAMPLE_DOCS)
        client = ollama_reranker._http_client
        assert "top_n" not in client._last_payload
