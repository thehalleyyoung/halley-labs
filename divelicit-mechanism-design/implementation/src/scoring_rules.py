"""Proper scoring rules with energy-augmented variant for incentive-compatible diversity.

Also provides multi-dimensional quality metrics (coherence, relevance, fluency,
factual consistency) to replace naive response-length-as-quality.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


class ScoringRule(ABC):
    """Base class for proper scoring rules.

    A scoring rule S(p, y) assigns a score to a probabilistic forecast p
    when outcome y is realized. A rule is *strictly proper* if the expected
    score E_q[S(p, Y)] is uniquely maximized when p = q.
    """

    @abstractmethod
    def score(self, p: np.ndarray, y: int) -> float:
        """Score report p when outcome y is realized."""
        ...

    @abstractmethod
    def expected_score(self, p: np.ndarray, q: np.ndarray) -> float:
        """E_q[S(p, Y)] — expected score of report p under true distribution q."""
        ...

    def properness_gap(self, p: np.ndarray, q: np.ndarray) -> float:
        """E_q[S(q,Y)] - E_q[S(p,Y)] ≥ 0 for proper rules, with equality iff p=q."""
        return self.expected_score(q, q) - self.expected_score(p, q)


class LogarithmicRule(ScoringRule):
    """S(p, y) = log p(y). Properness gap = KL(q || p)."""

    def score(self, p: np.ndarray, y: int) -> float:
        return float(np.log(np.clip(p[y], 1e-15, None)))

    def expected_score(self, p: np.ndarray, q: np.ndarray) -> float:
        # E_q[log p(Y)] = sum_y q(y) log p(y)
        return float(np.sum(q * np.log(np.clip(p, 1e-15, None))))


class BrierRule(ScoringRule):
    """S(p, y) = 2p(y) - ||p||^2. Properness gap = ||p - q||^2."""

    def score(self, p: np.ndarray, y: int) -> float:
        return float(2.0 * p[y] - np.sum(p ** 2))

    def expected_score(self, p: np.ndarray, q: np.ndarray) -> float:
        # E_q[2p(Y) - ||p||^2] = 2 sum q_i p_i - ||p||^2
        return float(2.0 * np.dot(q, p) - np.sum(p ** 2))


class SphericalRule(ScoringRule):
    """S(p, y) = p(y) / ||p||."""

    def score(self, p: np.ndarray, y: int) -> float:
        norm = np.linalg.norm(p)
        if norm < 1e-15:
            return 0.0
        return float(p[y] / norm)

    def expected_score(self, p: np.ndarray, q: np.ndarray) -> float:
        norm = np.linalg.norm(p)
        if norm < 1e-15:
            return 0.0
        return float(np.dot(q, p) / norm)


class CRPSRule(ScoringRule):
    """Continuous Ranked Probability Score (discrete version).

    CRPS(p, y) = -sum_k (F(k) - 1[k >= y])^2
    where F is the CDF of p.
    """

    def score(self, p: np.ndarray, y: int) -> float:
        cdf = np.cumsum(p)
        indicator = np.zeros_like(p)
        indicator[y:] = 1.0
        return float(-np.sum((cdf - indicator) ** 2))

    def expected_score(self, p: np.ndarray, q: np.ndarray) -> float:
        n = len(p)
        total = 0.0
        for y in range(n):
            total += q[y] * self.score(p, y)
        return total


class PowerRule(ScoringRule):
    """Power scoring rule for alpha > 1.

    S(p, y) = (alpha/(alpha-1)) * p(y)^(alpha-1) - (1/alpha) * ||p||_alpha^alpha
    where ||p||_alpha^alpha = sum p_i^alpha.
    """

    def __init__(self, alpha: float = 2.0):
        assert alpha > 1.0, "alpha must be > 1"
        self.alpha = alpha

    def score(self, p: np.ndarray, y: int) -> float:
        a = self.alpha
        term1 = (a / (a - 1.0)) * (p[y] ** (a - 1.0))
        term2 = (1.0 / a) * np.sum(p ** a)
        return float(term1 - term2)

    def expected_score(self, p: np.ndarray, q: np.ndarray) -> float:
        a = self.alpha
        term1 = (a / (a - 1.0)) * np.sum(q * (p ** (a - 1.0)))
        term2 = (1.0 / a) * np.sum(p ** a)
        return float(term1 - term2)


class EnergyAugmentedRule(ScoringRule):
    """Energy-augmented scoring rule: S_E(p, y) = S_base(p, y) + lambda * E(y, Y_history).

    KEY INSIGHT: The energy term E(y, Y_history) depends on the *outcome* y and
    the history Y_history, but NOT on the report p. Therefore:

        E_q[S_E(p, Y)] = E_q[S_base(p, Y)] + lambda * E_q[E(Y, Y_history)]

    The second term is constant w.r.t. p, so:
        argmax_p E_q[S_E(p, Y)] = argmax_p E_q[S_base(p, Y)] = q

    Thus, if S_base is strictly proper, S_E is also strictly proper. The energy
    term provides an additional incentive for diverse outcomes without breaking
    incentive compatibility, because the agent cannot manipulate it through
    their report — only through the actual outcome.
    """

    def __init__(
        self,
        base_rule: ScoringRule,
        energy_fn: Callable[[int, np.ndarray], float],
        lambda_: float = 0.1,
    ):
        self.base_rule = base_rule
        self.energy_fn = energy_fn
        self.lambda_ = lambda_
        self.history: np.ndarray = np.array([])

    def set_history(self, history: np.ndarray) -> None:
        self.history = history

    def score(self, p: np.ndarray, y: int) -> float:
        base = self.base_rule.score(p, y)
        energy = self.energy_fn(y, self.history)
        return base + self.lambda_ * energy

    def expected_score(self, p: np.ndarray, q: np.ndarray) -> float:
        base_expected = self.base_rule.expected_score(p, q)
        # Energy term is constant w.r.t. p
        energy_expected = sum(
            q[y] * self.energy_fn(y, self.history) for y in range(len(q))
        )
        return base_expected + self.lambda_ * energy_expected


def verify_properness(
    rule: ScoringRule, p: np.ndarray, q: np.ndarray, n_samples: int = 10000
) -> bool:
    """Empirically verify properness: E_q[S(q,Y)] >= E_q[S(p,Y)].

    Returns True if the rule appears proper (gap >= 0 within tolerance).
    """
    # Analytic check
    gap = rule.properness_gap(p, q)
    if gap < -1e-6:
        return False

    # Monte Carlo check
    rng = np.random.RandomState(42)
    outcomes = rng.choice(len(q), size=n_samples, p=q)
    score_q = np.mean([rule.score(q, y) for y in outcomes])
    score_p = np.mean([rule.score(p, y) for y in outcomes])
    mc_gap = score_q - score_p
    return bool(mc_gap >= -0.05)  # allow small MC noise


# ---------------------------------------------------------------------------
# Multi-dimensional Quality Metrics (replacing response-length-as-quality)
# ---------------------------------------------------------------------------

@dataclass
class QualityScore:
    """Multi-dimensional quality assessment of a response."""
    coherence: float      # intra-response sentence similarity (0-1)
    relevance: float      # query-response similarity (0-1)
    fluency: float        # inverse perplexity proxy (0-1)
    consistency: float    # factual / logical consistency (0-1)
    aggregate: float = 0.0  # weighted combination

    def __post_init__(self):
        if self.aggregate == 0.0:
            self.aggregate = self.compute_aggregate()

    def compute_aggregate(
        self,
        w_coherence: float = 0.25,
        w_relevance: float = 0.35,
        w_fluency: float = 0.20,
        w_consistency: float = 0.20,
    ) -> float:
        return (
            w_coherence * self.coherence
            + w_relevance * self.relevance
            + w_fluency * self.fluency
            + w_consistency * self.consistency
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "coherence": self.coherence,
            "relevance": self.relevance,
            "fluency": self.fluency,
            "consistency": self.consistency,
            "aggregate": self.aggregate,
        }


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def coherence_score(sentence_embeddings: np.ndarray) -> float:
    """Coherence: mean pairwise cosine similarity among sentence embeddings.

    High coherence = sentences are topically consistent.
    Expects (n_sentences, dim) array.
    """
    n = sentence_embeddings.shape[0]
    if n < 2:
        return 1.0
    sims = []
    # Adjacent sentence pairs (discourse coherence)
    for i in range(n - 1):
        sims.append(_cosine_sim(sentence_embeddings[i], sentence_embeddings[i + 1]))
    return float(np.clip(np.mean(sims), 0.0, 1.0))


def relevance_score(query_embedding: np.ndarray, response_embedding: np.ndarray) -> float:
    """Relevance: cosine similarity between query and response embeddings.

    Mapped to [0,1] via (sim+1)/2 to handle negative similarities.
    """
    sim = _cosine_sim(query_embedding, response_embedding)
    return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))


def fluency_score(token_log_probs: np.ndarray) -> float:
    """Fluency: inverse perplexity mapped to [0,1].

    Given per-token log-probabilities, computes perplexity and maps to a
    quality score via sigmoid: score = 1 / (1 + perplexity / baseline).

    If no real LM is available, provide a simulated array.
    """
    if len(token_log_probs) == 0:
        return 0.5
    mean_log_prob = float(np.mean(token_log_probs))
    perplexity = np.exp(-mean_log_prob)
    # Baseline perplexity of ~20 maps to 0.5
    baseline = 20.0
    score = 1.0 / (1.0 + perplexity / baseline)
    return float(np.clip(score, 0.0, 1.0))


def consistency_score(
    premise_embedding: np.ndarray, hypothesis_embedding: np.ndarray
) -> float:
    """Factual consistency via NLI proxy: similarity between premise and hypothesis.

    In a full system, use an NLI model. Here we use embedding similarity as a
    lightweight proxy. High similarity = hypothesis is consistent with premise.
    """
    sim = _cosine_sim(premise_embedding, hypothesis_embedding)
    return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))


def compute_quality(
    response_embedding: np.ndarray,
    query_embedding: Optional[np.ndarray] = None,
    sentence_embeddings: Optional[np.ndarray] = None,
    token_log_probs: Optional[np.ndarray] = None,
    premise_embedding: Optional[np.ndarray] = None,
) -> QualityScore:
    """Compute multi-dimensional quality score for a response.

    All inputs are optional; missing components get a default of 0.5.
    """
    # Coherence
    if sentence_embeddings is not None and sentence_embeddings.shape[0] >= 2:
        coh = coherence_score(sentence_embeddings)
    else:
        coh = 0.5

    # Relevance
    if query_embedding is not None:
        rel = relevance_score(query_embedding, response_embedding)
    else:
        rel = 0.5

    # Fluency
    if token_log_probs is not None:
        flu = fluency_score(token_log_probs)
    else:
        flu = 0.5

    # Consistency
    if premise_embedding is not None:
        con = consistency_score(premise_embedding, response_embedding)
    else:
        con = 0.5

    return QualityScore(coherence=coh, relevance=rel, fluency=flu, consistency=con)


def simulate_quality(
    response_embedding: np.ndarray,
    query_embedding: Optional[np.ndarray] = None,
    rng: Optional[np.random.RandomState] = None,
) -> QualityScore:
    """Simulate a multi-dimensional quality score for synthetic experiments.

    Uses embedding properties to generate plausible quality components rather
    than returning a single random scalar.
    """
    if rng is None:
        rng = np.random.RandomState()

    dim = len(response_embedding)
    norm = np.linalg.norm(response_embedding)

    # Coherence: based on embedding norm regularity (well-formed responses
    # have moderate norms)
    coh = float(np.clip(np.exp(-((norm - np.sqrt(dim)) ** 2) / (2 * dim)), 0.1, 1.0))

    # Relevance: cosine similarity to query if available
    if query_embedding is not None:
        rel = relevance_score(query_embedding, response_embedding)
    else:
        rel = float(np.clip(rng.beta(5, 3), 0.1, 1.0))

    # Fluency: simulated with beta distribution
    flu = float(np.clip(rng.beta(6, 2), 0.2, 1.0))

    # Consistency: simulated
    con = float(np.clip(rng.beta(5, 2), 0.2, 1.0))

    return QualityScore(coherence=coh, relevance=rel, fluency=flu, consistency=con)
