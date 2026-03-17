"""
Document chunking service — converts documents with Docling and produces
hierarchical chunks.

Extracted from the original docling_parser.py to live behind a clean
service interface that the FastAPI router calls.
"""

from __future__ import annotations

import base64
import logging
import time
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional, cast

from src.config import settings

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.doc_chunk import DocChunk
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import (
    DoclingDocument,
    ImageRef,
)
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Align tokenizer with the embedding model used in vector_store_service
# (Ollama multilingual-e5-large-instruct → same HF tokenizer family).
# NOTE: AutoTokenizer loads only the ~5 MB SentencePiece vocab file — it is
# a pure-CPU operation and never touches GPU or runs model inference.
# The vocab files are cached locally under ~/.cache/huggingface/hub/ after the
# first download.  In production / Docker builds, download them in advance
# (see setup.sh) and set HF_HUB_OFFLINE=1 to block all outbound Hub calls.
EMBED_MODEL_ID = "intfloat/multilingual-e5-large-instruct"
MAX_TOKENS = 512  # context window of multilingual-e5-large-instruct

# ── Converter (lazy singleton — expensive to create) ─────────
_converter: Optional[DocumentConverter] = None

# ── Chunker (lazy singleton) ─────────────────────────────────
_chunker: Optional[HybridChunker] = None


def get_chunker() -> HybridChunker:
    """Return (and lazily create) a tokenizer-aware HybridChunker."""
    global _chunker
    if _chunker is None:
        # Respect HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE env vars so the
        # container never phones home after the initial model download.
        offline: bool = settings.hf_hub_offline or settings.transformers_offline
        hf_tokenizer = AutoTokenizer.from_pretrained(
            EMBED_MODEL_ID,
            local_files_only=offline,
        )
        tokenizer = HuggingFaceTokenizer(
            tokenizer=hf_tokenizer,
            max_tokens=MAX_TOKENS,
        )
        _chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
        logger.info(
            "HybridChunker initialised (model=%s, max_tokens=%d)",
            EMBED_MODEL_ID,
            MAX_TOKENS,
        )
    return _chunker


def get_converter(
    do_ocr: bool = False,
    do_table_structure: bool = True,
    images_scale: float = 2.0,
) -> DocumentConverter:
    """Return (and lazily create) a DocumentConverter configured for PDFs."""
    global _converter
    if _converter is None:
        pipeline_options = PdfPipelineOptions(
            do_ocr=do_ocr,
            ocr_options=RapidOcrOptions(),
            do_table_structure=do_table_structure,
            generate_page_images=True,
            generate_picture_images=True,
            images_scale=images_scale,
        )
        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                ),
            }
        )
        logger.info("DocumentConverter initialised (generate_picture_images=True)")
    return _converter


# ── Helpers ──────────────────────────────────────────────────


def _page_numbers_from_chunk(chunk: DocChunk) -> list[int]:
    """Extract sorted, unique page numbers from a chunk's doc_items."""
    pages: set[int] = set()
    for doc_item in chunk.meta.doc_items:
        for prov in getattr(doc_item, "prov", []):
            pages.add(prov.page_no)
    return sorted(pages) if pages else []


def _headings_from_chunk(chunk: DocChunk) -> list[str]:
    return chunk.meta.headings if chunk.meta.headings else []


def _doc_items_refs(chunk: DocChunk) -> list[str]:
    """Return self_ref strings for each doc_item in the chunk."""
    refs = []
    for item in chunk.meta.doc_items:
        if hasattr(item, "self_ref"):
            refs.append(item.self_ref)
    return refs


def _image_ref_to_embedded_uri(image_ref: ImageRef) -> str | None:
    """Convert an ImageRef to a base64 data URI string."""
    uri = str(image_ref.uri)
    if uri.startswith("data:"):
        return uri
    pil = image_ref.pil_image
    if pil is not None:
        buf = BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    return None


def _build_docling_document_json(doc: DoclingDocument) -> dict:
    """Serialise DoclingDocument, ensuring PictureItem image URIs are embedded."""
    doc_dict = doc.model_dump(mode="json", by_alias=True, exclude_none=False)

    for i, pic in enumerate(doc.pictures):
        if pic.image is not None:
            embedded = _image_ref_to_embedded_uri(pic.image)
            if embedded and i < len(doc_dict.get("pictures", [])):
                doc_dict["pictures"][i]["image"]["uri"] = embedded

    return doc_dict


# ── Public service API ───────────────────────────────────────


def convert_and_chunk(
    file_bytes: bytes,
    filename: str,
    *,
    chunking_include_raw_text: bool = False,
    include_converted_doc: bool = True,
    convert_do_ocr: bool = False,
    convert_do_table_structure: bool = True,
) -> dict:
    """
    Convert a document with Docling and produce hierarchical chunks.

    Returns a response dict with ``chunks``, ``documents``, and
    ``processing_time`` keys.

    Note
    ----
    HybridChunker is structure-based (respects document hierarchy) and
    does **not** enforce token limits.  Chunks are based on document structure
    (sections, paragraphs, tables, etc.) rather than fixed-token windows.
    """
    t0 = time.time()

    # 1. Convert with Docling ─────────────────────────────────
    converter = get_converter(
        do_ocr=convert_do_ocr,
        do_table_structure=convert_do_table_structure,
    )

    suffix = Path(filename).suffix or ".pdf"
    with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    logger.info("Converting %s …", filename)
    result = converter.convert(source=tmp_path)
    doc: DoclingDocument = result.document
    logger.info(
        "Conversion done — status=%s, pictures=%d, texts=%d, tables=%d",
        result.status,
        len(doc.pictures),
        len(doc.texts),
        len(doc.tables),
    )

    # 2. Back-fill missing picture images ─────────────────────
    for pic in doc.pictures:
        if pic.image is None:
            pil_img = pic.get_image(doc)
            if pil_img is not None:
                pic.image = ImageRef.from_pil(pil_img, dpi=144)
                logger.info("  Backfilled image for picture self_ref=%s", pic.self_ref)

    # 3. Chunk ────────────────────────────────────────────────
    chunker = get_chunker()
    raw_chunks = cast(list[DocChunk], list(chunker.chunk(dl_doc=doc)))
    logger.info("Chunking done — %d structure-based chunks produced", len(raw_chunks))

    # 4. Build response ───────────────────────────────────────
    chunk_items = []
    for idx, chunk in enumerate(raw_chunks):
        chunk_items.append(
            {
                "filename": filename,
                "chunk_index": idx,
                "text": chunker.contextualize(chunk=chunk),
                "raw_text": chunk.text if chunking_include_raw_text else None,
                "num_tokens": None,
                "headings": _headings_from_chunk(chunk) or None,
                "captions": None,
                "doc_items": _doc_items_refs(chunk),
                "page_numbers": _page_numbers_from_chunk(chunk) or None,
                "metadata": None,
            }
        )

    documents = []
    if include_converted_doc:
        doc_json = _build_docling_document_json(doc)
        documents.append(
            {
                "kind": "ExportResult",
                "content": {
                    "filename": filename,
                    "md_content": None,
                    "json_content": doc_json,
                    "html_content": None,
                    "text_content": None,
                    "doctags_content": None,
                },
                "status": result.status.value,
                "errors": [
                    {
                        "component_type": str(e.component_type),
                        "module_name": e.module_name,
                        "error_message": e.error_message,
                    }
                    for e in result.errors
                ],
                "timings": {},
            }
        )

    # Clean up temp file
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

    return {
        "chunks": chunk_items,
        "documents": documents,
        "processing_time": round(time.time() - t0, 3),
    }
