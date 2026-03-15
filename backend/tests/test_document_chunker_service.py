"""
Tests for ``document_chunker_service.convert_and_chunk``.

PDF file paths are supplied as pytest parameters; a fixture translates each
path into ``(file_bytes, filename)`` so the actual test body stays clean.
``convert_and_chunk`` is called **once per PDF** (module scope) and the result
is shared across all tests, avoiding redundant Docling conversions.

Place test PDFs under:
    backend/tests/data/
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.services.document_chunker_service import convert_and_chunk

# ── Paths ────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "data"


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("nespresso.pdf", id="nespresso"),
    ],
)
def pdf_file(request: pytest.FixtureRequest) -> tuple[bytes, str]:
    """Resolve a parametrized PDF filename to ``(file_bytes, filename)``.

    Module-scoped so the file is read only once per param across the whole
    test module.  Add more ``pytest.param`` entries to run against additional
    documents.
    """
    filename: str = request.param
    path = FIXTURES_DIR / filename
    if not path.exists():
        pytest.skip(f"Test fixture not found: {path}")
    return path.read_bytes(), filename


@pytest.fixture(scope="module")
def chunked_result(pdf_file: tuple[bytes, str]) -> dict:
    """Run ``convert_and_chunk`` once per PDF and cache the result.

    All tests that only need the default output share this single conversion,
    so Docling's expensive pipeline is never invoked more than once per file.
    """
    file_bytes, filename = pdf_file
    return convert_and_chunk(file_bytes, filename)


@pytest.fixture(scope="module")
def chunked_result_no_doc(pdf_file: tuple[bytes, str]) -> dict:
    """Cached result with ``include_converted_doc=False``."""
    file_bytes, filename = pdf_file
    return convert_and_chunk(file_bytes, filename, include_converted_doc=False)


@pytest.fixture(scope="module")
def chunked_result_with_raw_text(pdf_file: tuple[bytes, str]) -> dict:
    """Cached result with ``chunking_include_raw_text=True``."""
    file_bytes, filename = pdf_file
    return convert_and_chunk(file_bytes, filename, chunking_include_raw_text=True)


# ── Tests ─────────────────────────────────────────────────────


class TestConvertAndChunk:
    """Integration tests for ``convert_and_chunk``."""

    def test_returns_expected_top_level_keys(self, chunked_result: dict):
        assert "chunks" in chunked_result
        assert "documents" in chunked_result
        assert "processing_time" in chunked_result

    def test_processing_time_is_positive(self, chunked_result: dict):
        assert chunked_result["processing_time"] > 0

    def test_produces_at_least_one_chunk(self, chunked_result: dict):
        assert len(chunked_result["chunks"]) > 0

    def test_chunk_structure(
        self, chunked_result: dict, pdf_file: tuple[bytes, str]
    ):
        _, filename = pdf_file
        for chunk in chunked_result["chunks"]:
            assert "filename" in chunk
            assert chunk["filename"] == filename
            assert "chunk_index" in chunk
            assert isinstance(chunk["chunk_index"], int)
            assert "text" in chunk
            assert isinstance(chunk["text"], str)
            assert len(chunk["text"].strip()) > 0
            assert "page_numbers" in chunk
            assert "headings" in chunk
            assert "doc_items" in chunk

        # for doc in chunked_result["documents"]:
        #     for pic in doc["content"]["json_content"]["pictures"]:
        #         if pic["captions"] is not None and len(pic["captions"]) != 0:
        #             print("Here")
        #             assert False
        #         if pic["captions"] is not None and len(pic["captions"]) != 0:
        #             print("Here")
        #             assert False
                

    def test_chunk_indices_are_sequential(self, chunked_result: dict):
        indices = [c["chunk_index"] for c in chunked_result["chunks"]]
        assert indices == list(range(len(indices)))

    def test_converted_document_included_by_default(
        self, chunked_result: dict, pdf_file: tuple[bytes, str]
    ):
        _, filename = pdf_file
        assert len(chunked_result["documents"]) == 1
        doc = chunked_result["documents"][0]
        assert doc["kind"] == "ExportResult"
        assert "content" in doc
        assert doc["content"]["filename"] == filename
        assert doc["content"]["json_content"] is not None

    def test_converted_document_excluded_when_requested(
        self, chunked_result_no_doc: dict
    ):
        assert chunked_result_no_doc["documents"] == []

    def test_raw_text_absent_by_default(self, chunked_result: dict):
        for chunk in chunked_result["chunks"]:
            assert chunk["raw_text"] is None

    # def test_raw_text_present_when_requested(
    #     self, chunked_result_with_raw_text: dict
    # ):
    #     for chunk in chunked_result_with_raw_text["chunks"]:
    #         assert chunk["raw_text"] == chunk["text"]

    def test_document_status_is_success(self, chunked_result: dict):
        doc = chunked_result["documents"][0]
        # Docling uses "success" for a fully converted document
        assert "success" in doc["status"].lower()
