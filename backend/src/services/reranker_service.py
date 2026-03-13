"""
Reranker service — re-ranking for hybrid search results.

Supports two backends (controlled by ``RERANKER_BACKEND``):

1. **cross-encoder** (default) — ``sentence-transformers`` ``CrossEncoder``
   with a multilingual model (default: ``BAAI/bge-reranker-v2-m3``).
   GPU acceleration (CUDA / MPS) is used automatically when available.

2. **ollama** — calls the Ollama ``/api/rerank`` endpoint using the
   GGUF-quantised model ``bge-reranker-v2-m3`` served by Ollama.
   See https://ollama.com/qllama/bge-reranker-v2-m3
   Much lower memory footprint thanks to GGUF quantisation (e.g. Q4_K_M).

The service is **lazy-initialised**: model weights are loaded only on the
first call (cross-encoder backend), so importing this module has zero cost.

Design for performance
----------------------
* Only the *top-K candidates* returned by the vector store are scored
  (typically 10–30 documents), so latency stays in the low milliseconds.
* Scoring is batched in a single call.
* The service is a singleton — the model / HTTP client is reused.

Configuration
-------------
All settings are loaded from environment variables (or the ``.env`` file
at the repository root) via :class:`src.config.Settings`.

* ``RERANKER_BACKEND``        — ``"cross-encoder"`` (default) or ``"ollama"``
* ``RERANKER_MODEL``          — HuggingFace model id (cross-encoder backend)
* ``RERANKER_DEVICE``         — ``"cuda"``, ``"mps"``, ``"cpu"``, or ``"auto"``
* ``RERANKER_BATCH_SIZE``     — batch size for ``predict()`` (default 32)
* ``OLLAMA_RERANKER_MODEL``   — Ollama model tag (default ``bge-reranker-v2-m3:q4_k_m``)
* ``OLLAMA_URL``              — Ollama base URL (default ``http://host.docker.internal:11434``)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from src.config import settings

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────
# Exposed as a module-level constant for backward compatibility.
# Sourced from the centralised Settings (reads RERANKER_BACKEND env var /
# .env file automatically).
RERANKER_BACKEND: str = settings.reranker_backend


class RerankerService:
    """Reranker that re-scores query–document pairs.

    Supports two backends selected via ``RERANKER_BACKEND`` env var:

    * ``"cross-encoder"`` — local sentence-transformers CrossEncoder
    * ``"ollama"``        — remote Ollama ``/api/rerank`` endpoint

    Parameters are read from environment variables (see module docstring).
    """

    def __init__(self) -> None:
        self.backend: str = settings.reranker_backend

        # ── Cross-encoder settings ───────────────────────────
        self.model_name: str = settings.reranker_model
        self._device: str = settings.reranker_device
        self._batch_size: int = settings.reranker_batch_size
        self._model: Any = None  # sentence_transformers.CrossEncoder

        # ── Ollama settings ──────────────────────────────────
        self.ollama_model: str = settings.ollama_reranker_model
        self.ollama_url: str = settings.ollama_url
        self._http_client: Any = None  # httpx.Client (lazy)

    # ── Lazy model / client loading ──────────────────────────

    @property
    def model(self):
        """Return (lazily load) the CrossEncoder model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @property
    def http_client(self):
        """Return (lazily create) an ``httpx.Client`` for Ollama."""
        if self._http_client is None:
            import httpx

            self._http_client = httpx.Client(timeout=120.0)
            logger.info(
                "Ollama reranker configured: model='%s', url='%s'",
                self.ollama_model,
                self.ollama_url,
            )
        return self._http_client

    def _resolve_device(self) -> str:
        """Pick the best available device when ``RERANKER_DEVICE=auto``."""
        if self._device != "auto":
            return self._device

        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        """Load the CrossEncoder model onto the resolved device."""
        from sentence_transformers import CrossEncoder

        device = self._resolve_device()
        logger.info(
            "Loading reranker model '%s' on device '%s' …",
            self.model_name,
            device,
        )
        t0 = time.time()
        model = CrossEncoder(self.model_name, device=device)
        logger.info(
            "Reranker model loaded in %.1f s (device=%s)",
            time.time() - t0,
            device,
        )
        return model

    # ── Ollama rerank via HTTP ───────────────────────────────

    def _rerank_ollama(
        self,
        query: str,
        documents: list[dict[str, Any]],
        *,
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Call Ollama ``/api/rerank`` and return scored documents."""
        t0 = time.time()

        doc_texts = [doc["text"] for doc in documents]

        payload: dict[str, Any] = {
            "model": self.ollama_model,
            "query": query,
            "documents": doc_texts,
        }
        if top_k is not None:
            payload["top_n"] = top_k

        resp = self.http_client.post(
            f"{self.ollama_url}/api/rerank",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        # Ollama returns {"results": [{"index": 0, "relevance_score": ...}, …]}
        scored_docs = []
        for item in data.get("results", []):
            idx = item["index"]
            new_doc = dict(documents[idx])
            new_doc["score"] = float(item["relevance_score"])
            scored_docs.append(new_doc)

        # Results from Ollama are already sorted, but ensure consistency
        scored_docs.sort(key=lambda d: d["score"], reverse=True)

        elapsed = time.time() - t0
        logger.info(
            "Reranked %d → %d documents via Ollama in %.3f s",
            len(documents),
            len(scored_docs),
            elapsed,
        )
        return scored_docs

    # ── Cross-encoder rerank ─────────────────────────────────

    def _rerank_cross_encoder(
        self,
        query: str,
        documents: list[dict[str, Any]],
        *,
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Score documents using a local CrossEncoder model."""
        t0 = time.time()

        # Build query–document pairs for the cross-encoder
        pairs = [(query, doc["text"]) for doc in documents]

        # Score all pairs in one batched call
        scores = self.model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        # Attach scores and sort descending
        scored_docs = []
        for doc, score in zip(documents, scores):
            new_doc = dict(doc)
            new_doc["score"] = float(score)
            scored_docs.append(new_doc)

        scored_docs.sort(key=lambda d: d["score"], reverse=True)

        if top_k is not None:
            scored_docs = scored_docs[:top_k]

        elapsed = time.time() - t0
        logger.info(
            "Reranked %d → %d documents in %.3f s",
            len(documents),
            len(scored_docs),
            elapsed,
        )
        return scored_docs

    # ── Public API ───────────────────────────────────────────

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        *,
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Re-score *documents* against *query* and return them sorted by
        relevance (highest first).

        Each document dict **must** contain a ``"text"`` key.  The returned
        list keeps the original dict structure and adds / replaces the
        ``"score"`` key with the cross-encoder score.

        Parameters
        ----------
        query:
            The user query string.
        documents:
            Candidate documents from the retrieval stage.  Each dict must
            have at least a ``"text"`` field.
        top_k:
            If given, only the *top_k* highest-scoring documents are
            returned.  Otherwise all documents are returned (re-ordered).

        Returns
        -------
        list[dict[str, Any]]
            Documents sorted by descending cross-encoder score.
        """
        if not documents:
            return []

        if self.backend == "ollama":
            return self._rerank_ollama(query, documents, top_k=top_k)
        return self._rerank_cross_encoder(query, documents, top_k=top_k)

    def offload_model(self) -> None:
        """Release the cross-encoder model and close the Ollama HTTP client."""
        self._model = None
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None
