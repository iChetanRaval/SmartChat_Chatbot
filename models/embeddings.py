"""
models/embeddings.py
Wraps a sentence-transformers embedding model.
Provides encode() for both documents and queries.
"""

import logging
from typing import Union
import numpy as np

logger = logging.getLogger(__name__)

_model_cache: dict = {}


def _load_model(model_name: str):
    """Lazy-load and cache the embedding model."""
    if model_name not in _model_cache:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            logger.info("Loading embedding model: %s", model_name)
            _model_cache[model_name] = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully.")
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for RAG. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        except Exception as exc:
            logger.error("Failed to load embedding model '%s': %s", model_name, exc)
            raise RuntimeError(f"Could not load embedding model: {exc}") from exc
    return _model_cache[model_name]


def encode_texts(
    texts: Union[str, list[str]],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """
    Encode one or more texts into embedding vectors.

    Args:
        texts:      A single string or list of strings.
        model_name: Name of the sentence-transformers model to use.
        batch_size: How many texts to encode in one forward pass.
        normalize:  If True, L2-normalise the vectors (recommended for cosine similarity).

    Returns:
        numpy array of shape (N, D) where N = number of texts and D = embedding dim.
    """
    if isinstance(texts, str):
        texts = [texts]

    try:
        model = _load_model(model_name)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return np.array(embeddings, dtype=np.float32)
    except Exception as exc:
        logger.error("Encoding failed: %s", exc)
        raise RuntimeError(f"Embedding encoding error: {exc}") from exc


def cosine_similarity_matrix(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarities between a query vector and a matrix of document vectors.
    Assumes inputs are already L2-normalised (i.e. dot product == cosine similarity).

    Args:
        query_vec: Shape (D,) or (1, D)
        doc_vecs:  Shape (N, D)

    Returns:
        1-D array of shape (N,) with similarity scores in [-1, 1].
    """
    query_vec = np.array(query_vec, dtype=np.float32).flatten()
    doc_vecs = np.array(doc_vecs, dtype=np.float32)
    return doc_vecs @ query_vec