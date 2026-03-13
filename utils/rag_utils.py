"""
utils/rag_utils.py
Document ingestion, chunking, vector store management, and retrieval.
All embedding calls go through models/embeddings.py.
"""

import io
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


@dataclass
class VectorStore:
    chunks: list[Chunk] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None   # shape (N, D)

    def is_empty(self) -> bool:
        return len(self.chunks) == 0


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_text_from_file(uploaded_file) -> str:
    """
    Extract plain text from an uploaded Streamlit file object.
    Supports: .txt, .md, .pdf
    """
    filename: str = uploaded_file.name.lower()
    raw_bytes: bytes = uploaded_file.read()

    try:
        if filename.endswith((".txt", ".md")):
            return raw_bytes.decode("utf-8", errors="replace")

        elif filename.endswith(".pdf"):
            try:
                import pdfplumber  # type: ignore
                with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                    pages = [page.extract_text() or "" for page in pdf.pages]
                return "\n".join(pages)
            except ImportError:
                # Fallback: pypdf
                try:
                    from pypdf import PdfReader  # type: ignore
                    reader = PdfReader(io.BytesIO(raw_bytes))
                    pages = [p.extract_text() or "" for p in reader.pages]
                    return "\n".join(pages)
                except ImportError as exc:
                    raise ImportError(
                        "Install pdfplumber or pypdf to read PDF files."
                    ) from exc

        else:
            raise ValueError(f"Unsupported file type: '{uploaded_file.name}'. Use .txt, .md, or .pdf.")

    except Exception as exc:
        logger.error("Text extraction failed for '%s': %s", uploaded_file.name, exc)
        raise


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping word-level chunks.

    Args:
        text:       Input text.
        chunk_size: Target number of words per chunk.
        overlap:    Number of words to repeat from the previous chunk.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()

    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


# ── Vector store operations ────────────────────────────────────────────────────

def build_vector_store(
    texts: list[str],
    sources: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    chunk_size: int = 500,
    overlap: int = 50,
) -> VectorStore:
    """
    Chunk every document, embed all chunks, and return a VectorStore.

    Args:
        texts:      List of raw document texts.
        sources:    Corresponding document names / labels.
        model_name: Sentence-transformers model identifier.
        chunk_size: Words per chunk.
        overlap:    Overlap words between consecutive chunks.

    Returns:
        Populated VectorStore.
    """
    from models.embeddings import encode_texts

    all_chunks: list[Chunk] = []
    chunk_id = 0

    for text, source in zip(texts, sources):
        raw_chunks = chunk_text(text, chunk_size, overlap)
        for raw in raw_chunks:
            all_chunks.append(Chunk(text=raw, source=source, chunk_id=chunk_id))
            chunk_id += 1

    if not all_chunks:
        logger.warning("No chunks produced — documents may be empty.")
        return VectorStore()

    chunk_texts = [c.text for c in all_chunks]
    logger.info("Embedding %d chunks …", len(chunk_texts))
    embeddings = encode_texts(chunk_texts, model_name=model_name)
    logger.info("Done embedding.")

    return VectorStore(chunks=all_chunks, embeddings=embeddings)


def retrieve_relevant_chunks(
    query: str,
    store: VectorStore,
    top_k: int = 3,
    model_name: str = "all-MiniLM-L6-v2",
    min_score: float = 0.25,
) -> list[Chunk]:
    """
    Return the top-k most relevant chunks for a query.

    Args:
        query:      The user's question.
        store:      A populated VectorStore.
        top_k:      Number of chunks to return.
        model_name: Must match the model used when building the store.
        min_score:  Minimum cosine similarity threshold.

    Returns:
        List of Chunk objects, sorted by relevance (descending).
    """
    from models.embeddings import encode_texts, cosine_similarity_matrix

    if store.is_empty() or store.embeddings is None:
        return []

    try:
        query_vec = encode_texts(query, model_name=model_name)[0]
        scores = cosine_similarity_matrix(query_vec, store.embeddings)
        ranked_indices = np.argsort(scores)[::-1]

        results: list[Chunk] = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] >= min_score:
                results.append(store.chunks[idx])

        return results

    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        return []


def format_context(chunks: list[Chunk]) -> str:
    """Format retrieved chunks into a context block for the LLM prompt."""
    if not chunks:
        return ""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Source: {chunk.source} | Chunk {chunk.chunk_id}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)