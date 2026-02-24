"""Simulated LLM agents for reproducible experiments."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class Agent(ABC):
    """Base class for simulated LLM agents."""

    @abstractmethod
    def generate(
        self, prompt: str = "", context: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """Generate a response.

        Args:
            prompt: The prompt string (unused in simulation, kept for API compat).
            context: Prior response embeddings for adaptive agents.

        Returns:
            (response_embedding, quality_score)
        """
        ...


class GaussianAgent(Agent):
    """Agent that samples from a multivariate Gaussian.

    Models an LLM with a fixed competence region.
    """

    def __init__(self, mean: np.ndarray, cov: np.ndarray, quality_mean: float = 0.7,
                 quality_std: float = 0.1, seed: Optional[int] = None):
        self.mean = mean
        self.cov = cov
        self.quality_mean = quality_mean
        self.quality_std = quality_std
        self.rng = np.random.RandomState(seed)

    def generate(
        self, prompt: str = "", context: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        embedding = self.rng.multivariate_normal(self.mean, self.cov)
        quality = np.clip(self.rng.normal(self.quality_mean, self.quality_std), 0.0, 1.0)
        return embedding, float(quality)


class MixtureAgent(Agent):
    """Gaussian mixture agent — models an LLM with multiple modes."""

    def __init__(self, components: List[Tuple[np.ndarray, np.ndarray]],
                 weights: Optional[np.ndarray] = None,
                 quality_mean: float = 0.7, quality_std: float = 0.1,
                 seed: Optional[int] = None):
        self.components = components  # list of (mean, cov)
        self.weights = weights if weights is not None else np.ones(len(components)) / len(components)
        self.quality_mean = quality_mean
        self.quality_std = quality_std
        self.rng = np.random.RandomState(seed)

    def generate(
        self, prompt: str = "", context: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        k = self.rng.choice(len(self.components), p=self.weights)
        mean, cov = self.components[k]
        embedding = self.rng.multivariate_normal(mean, cov)
        quality = np.clip(self.rng.normal(self.quality_mean, self.quality_std), 0.0, 1.0)
        return embedding, float(quality)


class AdaptiveAgent(Agent):
    """Agent that shifts its distribution based on context (prior responses).

    Models how LLMs respond to "be different" prompts.
    """

    def __init__(self, base_agent: Agent, context_sensitivity: float = 0.5,
                 seed: Optional[int] = None):
        self.base_agent = base_agent
        self.context_sensitivity = context_sensitivity
        self.rng = np.random.RandomState(seed)

    def generate(
        self, prompt: str = "", context: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        embedding, quality = self.base_agent.generate(prompt)

        if context is not None and context.ndim >= 2 and context.shape[0] > 0:
            # Repel from context centroid
            centroid = np.mean(context, axis=0)
            direction = embedding - centroid
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction = direction / norm
                embedding = embedding + self.context_sensitivity * direction

        return embedding, quality


class ClusteredAgent(Agent):
    """Agent generating from tight clusters — simulates mode collapse."""

    def __init__(self, n_clusters: int = 3, cluster_std: float = 0.1,
                 dim: int = 8, seed: Optional[int] = None):
        self.n_clusters = n_clusters
        self.cluster_std = cluster_std
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        # Generate random cluster centers
        self.centers = self.rng.randn(n_clusters, dim) * 2.0
        self.quality_per_cluster = self.rng.uniform(0.5, 0.9, n_clusters)

    def generate(
        self, prompt: str = "", context: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        k = self.rng.randint(self.n_clusters)
        embedding = self.centers[k] + self.rng.randn(self.dim) * self.cluster_std
        quality = float(np.clip(
            self.quality_per_cluster[k] + self.rng.normal(0, 0.05), 0.0, 1.0
        ))
        return embedding, quality


class UniformAgent(Agent):
    """Uniform agent over a bounded region — ideal diverse agent (upper bound)."""

    def __init__(self, dim: int = 8, bounds: Tuple[float, float] = (-1.0, 1.0),
                 quality_mean: float = 0.5, seed: Optional[int] = None):
        self.dim = dim
        self.bounds = bounds
        self.quality_mean = quality_mean
        self.rng = np.random.RandomState(seed)

    def generate(
        self, prompt: str = "", context: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        low, high = self.bounds
        embedding = self.rng.uniform(low, high, self.dim)
        quality = float(np.clip(self.rng.normal(self.quality_mean, 0.1), 0.0, 1.0))
        return embedding, quality
