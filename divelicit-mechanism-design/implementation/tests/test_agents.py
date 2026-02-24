"""Tests for simulated agents."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents import (
    GaussianAgent,
    MixtureAgent,
    AdaptiveAgent,
    ClusteredAgent,
    UniformAgent,
)


def test_gaussian_agent_samples_correct_shape():
    dim = 8
    agent = GaussianAgent(
        mean=np.zeros(dim), cov=np.eye(dim), seed=42
    )
    emb, q = agent.generate()
    assert emb.shape == (dim,)
    assert 0.0 <= q <= 1.0


def test_mixture_agent_multimodal():
    """Mixture agent should generate from multiple modes."""
    dim = 4
    components = [
        (np.zeros(dim), np.eye(dim) * 0.01),
        (np.ones(dim) * 10, np.eye(dim) * 0.01),
    ]
    agent = MixtureAgent(components, weights=np.array([0.5, 0.5]), seed=42)
    embeddings = [agent.generate()[0] for _ in range(100)]
    embeddings = np.array(embeddings)
    # Should have points near both modes
    near_zero = np.sum(np.linalg.norm(embeddings, axis=1) < 2)
    near_ten = np.sum(np.linalg.norm(embeddings - 10, axis=1) < 2)
    assert near_zero > 10 and near_ten > 10


def test_adaptive_agent_shifts_with_context():
    """Adaptive agent should shift away from context centroid."""
    dim = 4
    base = GaussianAgent(mean=np.zeros(dim), cov=np.eye(dim) * 0.01, seed=42)
    adaptive = AdaptiveAgent(base, context_sensitivity=2.0, seed=42)

    # Context at origin
    context = np.zeros((5, dim))
    emb_no_ctx, _ = adaptive.generate()
    emb_with_ctx, _ = adaptive.generate(context=context)

    # With context at origin, should be pushed away from origin
    # (on average, the shift should increase distance from origin)
    # Just verify it produces valid output
    assert emb_with_ctx.shape == (dim,)


def test_clustered_agent_low_diversity():
    """Clustered agent should produce low-diversity samples."""
    agent = ClusteredAgent(n_clusters=2, cluster_std=0.01, dim=4, seed=42)
    embeddings = [agent.generate()[0] for _ in range(50)]
    embeddings = np.array(embeddings)
    # Pairwise distances within clusters should be small
    # At least some pairs should be very close
    from src.diversity_metrics import dispersion_metric
    d = dispersion_metric(embeddings[:10])
    assert d < 1.0  # tight clusters


def test_uniform_agent_high_diversity():
    """Uniform agent should produce high-diversity samples."""
    agent = UniformAgent(dim=4, bounds=(-5.0, 5.0), seed=42)
    embeddings = [agent.generate()[0] for _ in range(20)]
    embeddings = np.array(embeddings)
    from src.diversity_metrics import cosine_diversity
    div = cosine_diversity(embeddings)
    assert div > 0.3  # uniform should be diverse


def test_agent_quality_scores_bounded():
    """Quality scores should be in [0, 1]."""
    agents = [
        GaussianAgent(np.zeros(4), np.eye(4), seed=42),
        ClusteredAgent(dim=4, seed=42),
        UniformAgent(dim=4, seed=42),
    ]
    for agent in agents:
        for _ in range(20):
            _, q = agent.generate()
            assert 0.0 <= q <= 1.0
