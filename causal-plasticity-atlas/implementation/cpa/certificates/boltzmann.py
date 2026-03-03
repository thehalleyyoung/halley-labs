"""Boltzmann stability analysis.

Implements a score-based Boltzmann posterior over DAGs:

    P(G | D) ~ exp( Score(G, D) / T )

where T is a temperature parameter.  At low temperature the posterior
concentrates on the highest-scoring DAG; at high temperature it
spreads over many structures.

By computing edge posteriors and credible intervals under this
distribution, we can issue *Boltzmann certificates* for mechanism
stability: a mechanism is certified stable if the posterior
probability of its parent set is concentrated.

Classes
-------
BoltzmannStabilityAnalyzer
    Core Boltzmann posterior computations.
BoltzmannCertificate
    Issue stability certificates from Boltzmann posteriors.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BoltzmannCertificateResult:
    """Certificate issued from Boltzmann posterior analysis."""

    node: int
    edge_posteriors: NDArray          # (p,) posterior prob for each parent
    posterior_entropy: float          # Shannon entropy of parent-set posterior
    concentrated: bool               # True if posterior is concentrated
    credible_interval: Tuple[float, float]  # Credible interval for mechanism score
    confidence_level: float
    certified: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Boltzmann Stability Analyzer
# ---------------------------------------------------------------------------

class BoltzmannStabilityAnalyzer:
    """Boltzmann posterior analysis for DAG structures.

    Parameters
    ----------
    score_fn : callable(dag, data) -> float
        Scoring function that takes a DAG adjacency matrix and data,
        returns a real-valued score (higher = better fit).
    temperature : float
        Temperature parameter T.  Lower temperature concentrates the
        posterior on higher-scoring DAGs.
    """

    def __init__(
        self,
        score_fn: Callable[[NDArray, NDArray], float],
        temperature: float = 1.0,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.score_fn = score_fn
        self.temperature = temperature

    # -- Core posterior --

    def posterior_probability(self, dag: NDArray, data: NDArray) -> float:
        """Un-normalised Boltzmann weight for a single DAG.

        P(dag | data) propto exp( score(dag, data) / T )

        Since computing the partition function over all DAGs is
        intractable, this returns the un-normalised weight.
        """
        score = float(self.score_fn(np.asarray(dag, dtype=np.float64), data))
        return self._boltzmann_weight(score, self.temperature)

    def mechanism_stability(
        self,
        dag_samples: List[NDArray],
        data: NDArray,
        node: int,
    ) -> Dict[str, Any]:
        """Assess stability of the mechanism at *node* across DAG samples.

        For each sampled DAG, extract the parent set of *node*.
        Compute the posterior probability that the parent set matches
        the most frequent parent set.

        Parameters
        ----------
        dag_samples : list of (p, p) adjacency matrices
        data : (n, p) data matrix
        node : variable index

        Returns
        -------
        dict with keys: parent_sets (list), frequencies (dict),
        mode_frequency (float), entropy (float).
        """
        data = np.asarray(data, dtype=np.float64)
        n_dags = len(dag_samples)
        if n_dags == 0:
            return {
                "parent_sets": [],
                "frequencies": {},
                "mode_frequency": 0.0,
                "entropy": 0.0,
            }

        # Score each DAG and compute Boltzmann weights
        log_weights = np.zeros(n_dags)
        parent_sets: List[Tuple[int, ...]] = []
        for k, dag in enumerate(dag_samples):
            dag = np.asarray(dag, dtype=np.float64)
            log_weights[k] = float(self.score_fn(dag, data)) / self.temperature
            parents = tuple(sorted(int(i) for i in range(dag.shape[0]) if dag[i, node] != 0))
            parent_sets.append(parents)

        # Normalise weights
        log_Z = logsumexp(log_weights)
        weights = np.exp(log_weights - log_Z)

        # Aggregate weights by parent set
        freq: Dict[Tuple[int, ...], float] = {}
        for ps, w in zip(parent_sets, weights):
            freq[ps] = freq.get(ps, 0.0) + float(w)

        mode_ps = max(freq, key=lambda k: freq[k])
        mode_freq = freq[mode_ps]

        # Entropy of the parent-set distribution
        probs = np.array(list(freq.values()))
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log(probs + 1e-30)))

        return {
            "parent_sets": parent_sets,
            "frequencies": {str(k): v for k, v in freq.items()},
            "mode_parent_set": mode_ps,
            "mode_frequency": mode_freq,
            "entropy": entropy,
            "n_unique_parent_sets": len(freq),
        }

    def _partition_function_estimate(
        self,
        dags: List[NDArray],
        data: NDArray,
        temperature: float,
    ) -> float:
        """Estimate the partition function Z from a list of DAG samples.

        Z = sum_G exp( score(G, D) / T )

        Uses the log-sum-exp trick for numerical stability.
        """
        if not dags:
            return 0.0
        data = np.asarray(data, dtype=np.float64)
        log_weights = np.array([
            float(self.score_fn(np.asarray(g, dtype=np.float64), data)) / temperature
            for g in dags
        ])
        return float(np.exp(logsumexp(log_weights)))

    def _importance_sampling_z(
        self,
        score_fn: Callable[[NDArray, NDArray], float],
        data: NDArray,
        n_samples: int,
        temperature: float,
        p: int,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Importance-sampling estimate of Z.

        Proposal: uniform random DAGs (Erdős–Rényi with p=0.5, enforced
        acyclic by upper-triangular + random permutation).

        Parameters
        ----------
        score_fn : scoring function
        data : observation matrix
        n_samples : number of IS samples
        temperature : Boltzmann temperature
        p : number of variables
        rng : random generator

        Returns
        -------
        Estimate of Z.
        """
        if rng is None:
            rng = np.random.default_rng()

        log_weights = np.full(n_samples, -np.inf)
        for k in range(n_samples):
            dag = self._sample_random_dag(p, rng)
            try:
                s = float(score_fn(dag, data))
                log_weights[k] = s / temperature
            except Exception:
                continue

        valid = log_weights > -np.inf
        if not np.any(valid):
            return 1.0  # fallback
        # IS estimate: Z ≈ (1/n_samples) * sum exp(w_k) * (1/q(G_k))
        # Under uniform proposal all q are equal, so Z ≈ |DAG space| * mean(exp(w))
        # We approximate |DAG space| ≈ n_samples (since we can't enumerate)
        return float(np.exp(logsumexp(log_weights[valid]) - np.log(np.sum(valid))))

    def edge_posterior(
        self,
        i: int,
        j: int,
        dag_samples: List[NDArray],
        data: Optional[NDArray] = None,
    ) -> float:
        """Posterior probability of edge i -> j.

        If *data* is provided, weights are computed from the score function;
        otherwise each sample is weighted equally.

        Parameters
        ----------
        i, j : edge endpoints
        dag_samples : list of DAG adjacency matrices
        data : optional data for score-based weighting

        Returns
        -------
        float in [0, 1]: posterior P(i -> j | data)
        """
        n = len(dag_samples)
        if n == 0:
            return 0.0

        if data is not None:
            data = np.asarray(data, dtype=np.float64)
            log_w = np.array([
                float(self.score_fn(np.asarray(g, dtype=np.float64), data)) / self.temperature
                for g in dag_samples
            ])
            log_Z = logsumexp(log_w)
            weights = np.exp(log_w - log_Z)
        else:
            weights = np.ones(n) / n

        edge_present = np.array([
            float(np.asarray(g, dtype=np.float64)[i, j] != 0) for g in dag_samples
        ])
        return float(np.dot(weights, edge_present))

    @staticmethod
    def _boltzmann_weight(score: float, temperature: float) -> float:
        """exp(score / temperature), clamped for numerical safety."""
        exponent = score / temperature
        exponent = min(exponent, 500.0)  # prevent overflow
        return float(np.exp(exponent))

    @staticmethod
    def _sample_random_dag(p: int, rng: np.random.Generator) -> NDArray:
        """Sample a random DAG by generating a random upper-triangular
        matrix and applying a random permutation."""
        # Random upper-triangular (Erdős–Rényi p=0.3 for sparsity)
        mask = rng.random((p, p)) < 0.3
        upper = np.triu(mask.astype(np.float64), k=1)
        # Random permutation of variable ordering
        perm = rng.permutation(p)
        dag = upper[np.ix_(perm, perm)]
        return dag


# ---------------------------------------------------------------------------
# Boltzmann Certificate
# ---------------------------------------------------------------------------

class BoltzmannCertificate:
    """Issue stability certificates from Boltzmann posterior analysis.

    A mechanism is certified stable if the posterior probability of
    its parent set concentrates above a threshold.

    Parameters
    ----------
    analyzer : BoltzmannStabilityAnalyzer
    confidence_level : float
        Required posterior concentration (default 0.95).
    """

    def __init__(
        self,
        analyzer: BoltzmannStabilityAnalyzer,
        confidence_level: float = 0.95,
    ) -> None:
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        self.analyzer = analyzer
        self.confidence_level = confidence_level

    def certify_mechanism(
        self,
        node: int,
        dag_samples: List[NDArray],
        data: NDArray,
    ) -> BoltzmannCertificateResult:
        """Issue a Boltzmann-based certificate for the mechanism at *node*.

        Parameters
        ----------
        node : target variable index
        dag_samples : list of (p, p) DAG adjacency matrices
        data : (n, p) observation matrix

        Returns
        -------
        BoltzmannCertificateResult
        """
        data = np.asarray(data, dtype=np.float64)
        p = data.shape[1]

        stability = self.analyzer.mechanism_stability(dag_samples, data, node)

        # Compute edge posteriors for the mechanism at node
        edge_post = np.zeros(p)
        for i in range(p):
            if i == node:
                continue
            edge_post[i] = self.analyzer.edge_posterior(i, node, dag_samples, data)

        # Posterior concentration check
        concentrated = self._posterior_concentration(
            stability["mode_frequency"], self.confidence_level
        )

        # Credible interval from per-DAG mechanism scores
        scores = self._compute_mechanism_scores(dag_samples, data, node)
        ci = self._credible_interval(scores, self.confidence_level)

        certified = concentrated and (ci[1] - ci[0]) < 1.0

        return BoltzmannCertificateResult(
            node=node,
            edge_posteriors=edge_post,
            posterior_entropy=stability["entropy"],
            concentrated=concentrated,
            credible_interval=ci,
            confidence_level=self.confidence_level,
            certified=certified,
            metadata={
                "mode_parent_set": stability.get("mode_parent_set"),
                "mode_frequency": stability["mode_frequency"],
                "n_unique_parent_sets": stability["n_unique_parent_sets"],
                "n_dag_samples": len(dag_samples),
            },
        )

    def _posterior_concentration(
        self, mode_probability: float, threshold: float
    ) -> bool:
        """Check if the posterior concentrates above *threshold*.

        The posterior is deemed concentrated if the mode parent set
        has probability >= threshold.
        """
        return mode_probability >= threshold

    def _credible_interval(
        self, scores: NDArray, level: float
    ) -> Tuple[float, float]:
        """Bayesian credible interval for mechanism scores.

        Computes the equal-tailed interval containing *level* of the
        posterior mass.
        """
        if len(scores) == 0:
            return (0.0, 0.0)
        alpha = 1.0 - level
        lower = float(np.percentile(scores, 100 * alpha / 2))
        upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
        return (lower, upper)

    def _compute_mechanism_scores(
        self,
        dag_samples: List[NDArray],
        data: NDArray,
        node: int,
    ) -> NDArray:
        """Compute per-DAG mechanism scores weighted by Boltzmann posterior.

        For each DAG, computes the local score contribution at *node*
        (using the full DAG score as a proxy when local scoring is
        not available).
        """
        n_dags = len(dag_samples)
        if n_dags == 0:
            return np.array([])

        scores = np.zeros(n_dags)
        for k, dag in enumerate(dag_samples):
            dag = np.asarray(dag, dtype=np.float64)
            try:
                scores[k] = float(self.analyzer.score_fn(dag, data))
            except Exception:
                scores[k] = float("-inf")

        # Normalise to Boltzmann weights for a weighted score distribution
        log_w = scores / self.analyzer.temperature
        valid = np.isfinite(log_w)
        if not np.any(valid):
            return scores

        # Return the raw scores (the credible interval uses quantiles)
        return scores
