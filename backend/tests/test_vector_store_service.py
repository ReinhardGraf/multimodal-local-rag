"""
Tests for ``backend.vector_store_service`` — upsert & search with German texts.

All tests use a **real in-memory Qdrant** instance (no mocks).  Dense embeddings
are replaced by deterministic vectors (see conftest.py) so Ollama is not
required; sparse (BM25) embeddings use the real *fastembed* encoder.
"""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient

from src.services.vector_store_service import DocumentItem, VectorStoreService

# Must match the value in conftest.py
TEST_COLLECTION = "test_documents"

# ── German sample data ───────────────────────────────────────

GERMAN_DOCUMENTS: list[dict] = [
    {
        "text": (
            "Die Bundesrepublik Deutschland ist ein demokratischer und sozialer "
            "Bundesstaat. Alle Staatsgewalt geht vom Volke aus."
        ),
        "metadata": {"source": "grundgesetz", "chapter": "Artikel 20"},
    },
    {
        "text": (
            "Die Donau ist mit einer mittleren Wasserführung von rund 6.855 m³/s "
            "der zweitgrößte Fluss in Europa. Sie durchfließt zehn Länder."
        ),
        "metadata": {"source": "geographie", "topic": "Flüsse"},
    },
    {
        "text": (
            "Die deutsche Automobilindustrie ist einer der größten Arbeitgeber "
            "des Landes. Unternehmen wie Volkswagen, BMW und Mercedes-Benz "
            "sind weltweit bekannt."
        ),
        "metadata": {"source": "wirtschaft", "topic": "Automobilindustrie"},
    },
    {
        "text": (
            "Maschinelles Lernen ist ein Teilgebiet der künstlichen Intelligenz. "
            "Dabei lernen Algorithmen aus Daten und verbessern ihre Leistung "
            "ohne explizite Programmierung."
        ),
        "metadata": {"source": "informatik", "topic": "KI"},
    },
    {
        "text": (
            "Die Straßenverkehrsordnung regelt den Verkehr auf öffentlichen "
            "Straßen in Deutschland. Geschwindigkeitsbegrenzungen dienen der "
            "Sicherheit aller Verkehrsteilnehmer."
        ),
        "metadata": {"source": "recht", "topic": "StVO"},
    },
    {
        "text": (
            "Sauerbraten ist ein traditionelles deutsches Gericht, bei dem "
            "Rindfleisch mehrere Tage in einer Marinade aus Essig und Gewürzen "
            "eingelegt wird."
        ),
        "metadata": {"source": "küche", "topic": "Rezepte"},
    },
]


def _to_doc_items(docs: list[dict], with_ids: bool = True) -> list[DocumentItem]:
    """Convert raw dicts to ``DocumentItem`` instances."""
    return [
        DocumentItem(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, d["text"])) if with_ids else None,
            text=d["text"],
            metadata=d.get("metadata"),
        )
        for d in docs
    ]


# ── Upsert tests ────────────────────────────────────────────


class TestUpsertDocuments:
    """Tests around the ``upsert_documents`` method."""

    @pytest.mark.asyncio
    async def test_upsert_returns_correct_count(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        docs = _to_doc_items(GERMAN_DOCUMENTS)
        count = await vector_service.upsert_documents(docs, collection_name=collection)
        assert count == len(GERMAN_DOCUMENTS)

    @pytest.mark.asyncio
    async def test_upserted_points_are_persisted(
        self,
        vector_service: VectorStoreService,
        qdrant_client: QdrantClient,
        collection: str,
    ):
        docs = _to_doc_items(GERMAN_DOCUMENTS)
        await vector_service.upsert_documents(docs, collection_name=collection)

        info = qdrant_client.get_collection(collection)
        assert info.points_count == len(GERMAN_DOCUMENTS)

    @pytest.mark.asyncio
    async def test_upsert_stores_payload(
        self,
        vector_service: VectorStoreService,
        qdrant_client: QdrantClient,
        collection: str,
    ):
        docs = _to_doc_items(GERMAN_DOCUMENTS[:1])
        await vector_service.upsert_documents(docs, collection_name=collection)

        # Scroll all points
        points, _ = qdrant_client.scroll(collection, limit=10, with_payload=True)
        assert len(points) == 1
        payload = points[0].payload
        assert "text" in payload
        assert payload["source"] == "grundgesetz"
        assert payload["chapter"] == "Artikel 20"

    @pytest.mark.asyncio
    async def test_upsert_generates_uuid_when_id_missing(
        self,
        vector_service: VectorStoreService,
        qdrant_client: QdrantClient,
        collection: str,
    ):
        """When ``DocumentItem.id`` is ``None`` the service must generate a UUID."""
        docs = _to_doc_items(GERMAN_DOCUMENTS[:2], with_ids=False)
        await vector_service.upsert_documents(docs, collection_name=collection)

        points, _ = qdrant_client.scroll(collection, limit=10)
        assert len(points) == 2
        for p in points:
            # Should be a valid UUID string
            uuid.UUID(p.id)

    @pytest.mark.asyncio
    async def test_upsert_idempotent_with_same_ids(
        self,
        vector_service: VectorStoreService,
        qdrant_client: QdrantClient,
        collection: str,
    ):
        """Upserting the same IDs twice must not duplicate points."""
        docs = _to_doc_items(GERMAN_DOCUMENTS[:3])
        await vector_service.upsert_documents(docs, collection_name=collection)
        await vector_service.upsert_documents(docs, collection_name=collection)

        info = qdrant_client.get_collection(collection)
        assert info.points_count == 3

    @pytest.mark.asyncio
    async def test_upsert_single_document(
        self,
        vector_service: VectorStoreService,
        qdrant_client: QdrantClient,
        collection: str,
    ):
        single = _to_doc_items(
            [{"text": "München ist die Landeshauptstadt des Freistaates Bayern."}]
        )
        count = await vector_service.upsert_documents(single, collection_name=collection)
        assert count == 1
        info = qdrant_client.get_collection(collection)
        assert info.points_count == 1

    @pytest.mark.asyncio
    async def test_upsert_vectors_have_correct_dimension(
        self,
        vector_service: VectorStoreService,
        qdrant_client: QdrantClient,
        collection: str,
    ):
        docs = _to_doc_items(GERMAN_DOCUMENTS[:1])
        await vector_service.upsert_documents(docs, collection_name=collection)

        points, _ = qdrant_client.scroll(
            collection, limit=1, with_vectors=True
        )
        dense_vec = points[0].vector["dense"]
        assert len(dense_vec) == 1024


# ── Search / query tests ────────────────────────────────────


class TestSearchDocuments:
    """Tests around the ``search`` method (hybrid dense + sparse via RRF)."""

    @pytest.fixture(autouse=True)
    async def _seed_collection(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        """Upsert all German documents before every search test."""
        docs = _to_doc_items(GERMAN_DOCUMENTS)
        await vector_service.upsert_documents(docs, collection_name=collection)

    @pytest.mark.asyncio
    async def test_search_returns_results(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        results, reranked = await vector_service.search(
            "Welche Flüsse gibt es in Europa?", collection_name=collection
        )
        assert len(results) > 0
        assert reranked is False

    @pytest.mark.asyncio
    async def test_search_respects_limit(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        results, _ = await vector_service.search(
            "Deutschland", collection_name=collection, limit=2
        )
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_search_result_structure(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        results, _ = await vector_service.search(
            "Automobilindustrie in Deutschland",
            collection_name=collection,
            limit=1,
        )
        assert len(results) >= 1
        hit = results[0]
        assert "id" in hit
        assert "score" in hit
        assert "text" in hit
        assert "metadata" in hit

    @pytest.mark.asyncio
    async def test_sparse_bm25_finds_german_compound_word(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        """BM25 should surface the document containing
        'Geschwindigkeitsbegrenzungen' when we query for it."""
        results, _ = await vector_service.search(
            "Geschwindigkeitsbegrenzungen Straßenverkehrsordnung",
            collection_name=collection,
            limit=3,
        )
        texts = [r["text"] for r in results]
        assert any("Straßenverkehrsordnung" in t for t in texts)

    @pytest.mark.asyncio
    async def test_search_german_cuisine_query(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        """A query about German cuisine should surface the Sauerbraten document."""
        results, _ = await vector_service.search(
            "traditionelles deutsches Essen Rezept",
            collection_name=collection,
            limit=3,
        )
        texts = [r["text"] for r in results]
        assert any("Sauerbraten" in t for t in texts)

    @pytest.mark.asyncio
    async def test_search_ki_query(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        """A query about AI / machine learning should return the KI document."""
        results, _ = await vector_service.search(
            "künstliche Intelligenz maschinelles Lernen",
            collection_name=collection,
            limit=3,
        )
        texts = [r["text"] for r in results]
        assert any("maschinelles Lernen" in t.lower() or "Maschinelles Lernen" in t for t in texts)

    @pytest.mark.asyncio
    async def test_search_returns_metadata(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        results, _ = await vector_service.search(
            "Donau Fluss Europa", collection_name=collection, limit=3
        )
        geo_results = [r for r in results if r["metadata"].get("topic") == "Flüsse"]
        assert len(geo_results) >= 1


# ── Collection management tests ──────────────────────────────


class TestEnsureCollection:
    """Tests for ``ensure_collection``."""

    def test_creates_collection_when_missing(
        self,
        vector_service: VectorStoreService,
        qdrant_client: QdrantClient,
    ):
        name = f"test_auto_{uuid.uuid4().hex[:8]}"
        vector_service.ensure_collection(name)

        info = qdrant_client.get_collection(name)
        assert info is not None
        # Clean up
        qdrant_client.delete_collection(name)

    def test_idempotent_when_collection_exists(
        self,
        vector_service: VectorStoreService,
        qdrant_client: QdrantClient,
        collection: str,
    ):
        """Calling ``ensure_collection`` twice must not raise."""
        vector_service.ensure_collection(collection)
        vector_service.ensure_collection(collection)
        info = qdrant_client.get_collection(collection)
        assert info is not None


# ── Reranking integration tests ──────────────────────────────

class _FakeCrossEncoderForIntegration:
    """Word-overlap scorer used to verify the reranking integration path."""

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        scores = []
        for query, doc in pairs:
            q_words = set(query.lower().split())
            d_words = set(doc.lower().split())
            overlap = len(q_words & d_words)
            scores.append(overlap / max(len(q_words), 1))
        return scores


class TestSearchWithReranking:
    """Tests that verify reranking is correctly wired into ``search``."""

    @pytest.fixture(autouse=True)
    async def _seed_collection(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        """Upsert all German documents before every search test."""
        docs = _to_doc_items(GERMAN_DOCUMENTS)
        await vector_service.upsert_documents(docs, collection_name=collection)

    @pytest.fixture(autouse=True)
    def _inject_fake_reranker(self, vector_service: VectorStoreService):
        """Inject a fake reranker so we don't need a real model."""
        from src.services.reranker_service import RerankerService
        fake_reranker = RerankerService()
        fake_reranker._model = _FakeCrossEncoderForIntegration()
        vector_service._reranker = fake_reranker

    @pytest.mark.asyncio
    async def test_search_with_rerank_returns_reranked_flag(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        results, reranked = await vector_service.search(
            "Donau Fluss Europa",
            collection_name=collection,
            rerank=True,
        )
        assert reranked is True
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_without_rerank_returns_false_flag(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        results, reranked = await vector_service.search(
            "Donau Fluss Europa",
            collection_name=collection,
            rerank=False,
        )
        assert reranked is False

    @pytest.mark.asyncio
    async def test_reranked_search_respects_limit(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        results, reranked = await vector_service.search(
            "Deutschland",
            collection_name=collection,
            limit=2,
            rerank=True,
        )
        assert reranked is True
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_reranked_search_result_structure(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        results, _ = await vector_service.search(
            "Automobilindustrie",
            collection_name=collection,
            limit=3,
            rerank=True,
        )
        for hit in results:
            assert "id" in hit
            assert "score" in hit
            assert "text" in hit
            assert "metadata" in hit

    @pytest.mark.asyncio
    async def test_rerank_top_k_overrides_limit(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        """rerank_top_k should control the final result count independently."""
        results, reranked = await vector_service.search(
            "Deutschland",
            collection_name=collection,
            limit=5,
            rerank=True,
            rerank_top_k=1,
        )
        assert reranked is True
        assert len(results) == 1


# ── Keyword search tests ─────────────────────────────────────


class TestSearchWithKeywords:
    """Tests that verify keyword-based sparse search is correctly wired."""

    @pytest.fixture(autouse=True)
    async def _seed_collection(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        """Upsert all German documents before every search test."""
        docs = _to_doc_items(GERMAN_DOCUMENTS)
        await vector_service.upsert_documents(docs, collection_name=collection)

    @pytest.mark.asyncio
    async def test_keywords_search_returns_results(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        results, _ = await vector_service.search(
            "Erzähl mir über Flüsse",
            collection_name=collection,
            keywords=["Donau", "Europa"],
        )
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_keywords_boost_bm25_relevance(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        """Providing exact keywords that appear in a document should help
        BM25 surface that document via the sparse leg of hybrid search."""
        results, _ = await vector_service.search(
            "Flüsse in Europa",
            collection_name=collection,
            keywords=["Donau", "Wasserführung", "zweitgrößte"],
            limit=3,
        )
        texts = [r["text"] for r in results]
        assert any("Donau" in t for t in texts)

    @pytest.mark.asyncio
    async def test_keywords_stvo(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        """Keywords with exact German compound words should leverage BM25."""
        results, _ = await vector_service.search(
            "Verkehrsregeln",
            collection_name=collection,
            keywords=["Straßenverkehrsordnung", "Geschwindigkeitsbegrenzungen"],
            limit=3,
        )
        texts = [r["text"] for r in results]
        assert any("Straßenverkehrsordnung" in t for t in texts)

    @pytest.mark.asyncio
    async def test_keywords_none_falls_back_to_query(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        """When keywords=None, sparse search should use the query text."""
        results_no_kw, _ = await vector_service.search(
            "Sauerbraten Rezept",
            collection_name=collection,
            keywords=None,
            limit=3,
        )
        assert len(results_no_kw) > 0

    @pytest.mark.asyncio
    async def test_keywords_empty_list_falls_back_to_query(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        """An empty keywords list should behave exactly like keywords=None."""
        results, _ = await vector_service.search(
            "Sauerbraten Rezept",
            collection_name=collection,
            keywords=[],
            limit=3,
        )
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_keywords_with_rerank(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        """Keywords + reranking should work together."""
        from src.services.reranker_service import RerankerService
        fake_reranker = RerankerService()
        fake_reranker._model = _FakeCrossEncoderForIntegration()
        vector_service._reranker = fake_reranker

        results, reranked = await vector_service.search(
            "Automobilhersteller in Deutschland",
            collection_name=collection,
            keywords=["Volkswagen", "BMW", "Mercedes-Benz", "Automobilindustrie"],
            limit=3,
            rerank=True,
        )
        assert reranked is True
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_keywords_respects_limit(
        self,
        vector_service: VectorStoreService,
        collection: str,
    ):
        results, _ = await vector_service.search(
            "Deutschland",
            collection_name=collection,
            keywords=["Deutschland"],
            limit=2,
        )
        assert len(results) <= 2
