"""Comprehensive diversity metrics suite."""

from typing import Dict, Optional

import numpy as np

from .kernels import Kernel, RBFKernel
from .transport import sinkhorn_divergence, cost_matrix
from .utils import log_det_safe


def cosine_diversity(embeddings: np.ndarray) -> float:
    """1 - mean pairwise cosine similarity."""
    n = embeddings.shape[0]
    if n < 2:
        return 0.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = embeddings / norms
    sim_matrix = normed @ normed.T
    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_sim = np.mean(sim_matrix[mask])
    return float(1.0 - mean_sim)


def mmd(X: np.ndarray, Y: np.ndarray, kernel: Optional[Kernel] = None) -> float:
    """Maximum Mean Discrepancy between X and Y.

    MMD^2(X,Y) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    """
    if kernel is None:
        kernel = RBFKernel(bandwidth=1.0)

    n, m = X.shape[0], Y.shape[0]

    K_xx = kernel.gram_matrix(X)
    K_yy = kernel.gram_matrix(Y)

    # Cross-kernel
    K_xy = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            K_xy[i, j] = kernel.evaluate(X[i], Y[j])

    mmd_sq = (np.sum(K_xx) / (n * n) + np.sum(K_yy) / (m * m) -
              2.0 * np.sum(K_xy) / (n * m))
    return float(max(mmd_sq, 0.0))


def sinkhorn_diversity_metric(embeddings: np.ndarray, reg: float = 0.1) -> float:
    """Sinkhorn divergence from uniform distribution over bounding box.

    Higher values indicate the points are more concentrated (less diverse).
    We return the inverse: lower Sinkhorn divergence from uniform = more diverse.
    """
    n, d = embeddings.shape
    # Generate uniform reference
    mins = np.min(embeddings, axis=0)
    maxs = np.max(embeddings, axis=0)
    rng = np.random.RandomState(42)
    uniform = rng.uniform(mins, maxs, size=(n, d))
    # We want diversity, so lower divergence from uniform = better
    div = sinkhorn_divergence(embeddings, uniform, reg=reg)
    # Return inverse: high value = high diversity
    return float(1.0 / (1.0 + div))


def log_det_diversity(embeddings: np.ndarray, kernel: Optional[Kernel] = None) -> float:
    """Log-determinant diversity: log det(K) of kernel gram matrix."""
    if kernel is None:
        kernel = RBFKernel(bandwidth=1.0)
    K = kernel.gram_matrix(embeddings)
    return log_det_safe(K)


def vendi_score(embeddings: np.ndarray, kernel: Optional[Kernel] = None) -> float:
    """Vendi score: exp(entropy of eigenvalues of kernel matrix).

    VS = exp(-sum_i lambda_i * log(lambda_i))
    where lambda_i are the normalized eigenvalues of K.
    """
    if kernel is None:
        kernel = RBFKernel(bandwidth=1.0)
    K = kernel.gram_matrix(embeddings)
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.maximum(eigvals, 0.0)
    total = np.sum(eigvals)
    if total < 1e-15:
        return 1.0
    # Normalize eigenvalues
    p = eigvals / total
    p = p[p > 1e-15]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def dispersion_metric(embeddings: np.ndarray) -> float:
    """Minimum pairwise distance."""
    n = embeddings.shape[0]
    if n < 2:
        return 0.0
    min_dist = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(embeddings[i] - embeddings[j])
            if d < min_dist:
                min_dist = d
    return float(min_dist)


def coverage_fraction(embeddings: np.ndarray, reference: np.ndarray,
                      epsilon: float) -> float:
    """Fraction of reference points within epsilon of some embedding."""
    covered = 0
    for ref in reference:
        dists = np.linalg.norm(embeddings - ref, axis=1)
        if np.min(dists) <= epsilon:
            covered += 1
    return covered / max(len(reference), 1)


def pairwise_distances(embeddings: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Pairwise distance matrix."""
    return cost_matrix(embeddings, embeddings, metric=metric)


def diversity_profile(embeddings: np.ndarray) -> Dict[str, float]:
    """Compute ALL diversity metrics at once."""
    kernel = RBFKernel(bandwidth=1.0)
    return {
        "cosine_diversity": cosine_diversity(embeddings),
        "log_det_diversity": log_det_diversity(embeddings, kernel),
        "vendi_score": vendi_score(embeddings, kernel),
        "dispersion": dispersion_metric(embeddings),
        "sinkhorn_diversity": sinkhorn_diversity_metric(embeddings),
    }
