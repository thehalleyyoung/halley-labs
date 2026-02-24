"""Embedding utilities."""

from typing import Optional

import numpy as np


class TextEmbedder:
    """Text embedding using random projection (deterministic fallback)."""

    def __init__(self, dim: int = 64, seed: int = 42):
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self._projection: Optional[np.ndarray] = None

    def _hash_text(self, text: str) -> np.ndarray:
        """Deterministic text-to-vector via character-level hashing."""
        rng = np.random.RandomState(hash(text) % (2**31))
        return rng.randn(self.dim)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self._hash_text(text)

    def embed_batch(self, texts: list) -> np.ndarray:
        """Embed a list of texts."""
        return np.array([self._hash_text(t) for t in texts])


def embed_texts(texts: list, dim: int = 64) -> np.ndarray:
    """Embed a list of texts to vectors."""
    embedder = TextEmbedder(dim=dim)
    return embedder.embed_batch(texts)


def project_to_sphere(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit sphere."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def reduce_dim(embeddings: np.ndarray, target_dim: int,
               method: str = "pca") -> np.ndarray:
    """Reduce embedding dimensionality."""
    if embeddings.shape[1] <= target_dim:
        return embeddings

    if method == "pca":
        # Center
        mean = np.mean(embeddings, axis=0)
        centered = embeddings - mean
        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ Vt[:target_dim].T
    elif method == "random":
        rng = np.random.RandomState(42)
        proj = rng.randn(embeddings.shape[1], target_dim) / np.sqrt(target_dim)
        return embeddings @ proj
    else:
        raise ValueError(f"Unknown method: {method}")
