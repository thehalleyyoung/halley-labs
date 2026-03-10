"""
usability_oracle.evaluation.ordinal — Ordinal agreement validation.

Computes rank-correlation statistics (Spearman ρ, Kendall τ, concordance
rate) between model-produced orderings and human-rated orderings, along
with bootstrap confidence intervals.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

try:
    from scipy import stats as sp_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class OrdinalResult:
    """Aggregated ordinal-agreement statistics."""

    spearman_rho: float = 0.0
    kendall_tau: float = 0.0
    concordance_rate: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    n_pairs: int = 0
    p_value_spearman: float = 1.0
    p_value_kendall: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Ordinal validation (n={self.n_pairs}):\n"
            f"  Spearman ρ = {self.spearman_rho:.4f}  (p = {self.p_value_spearman:.4g})\n"
            f"  Kendall  τ = {self.kendall_tau:.4f}  (p = {self.p_value_kendall:.4g})\n"
            f"  Concordance = {self.concordance_rate:.4f}\n"
            f"  95% CI = [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        )


# ---------------------------------------------------------------------------
# OrdinalValidator
# ---------------------------------------------------------------------------

class OrdinalValidator:
    """Validate model orderings against human orderings.

    Parameters:
        n_bootstrap: Number of bootstrap resamples for CI estimation.
        alpha: Significance level for confidence intervals.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        n_bootstrap: int = 5000,
        alpha: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self._n_bootstrap = n_bootstrap
        self._alpha = alpha
        self._rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        model_orderings: list[tuple[Any, ...]] | list[float],
        human_orderings: list[tuple[Any, ...]] | list[float],
    ) -> OrdinalResult:
        """Compute ordinal agreement between two sets of orderings.

        *model_orderings* and *human_orderings* can be:
          - Lists of numeric scores / ranks.
          - Lists of ``(item_a, item_b, preference)`` tuples for pairwise
            concordance.
        """
        # Convert to flat numeric arrays for rank-correlation
        m_arr = self._to_array(model_orderings)
        h_arr = self._to_array(human_orderings)
        n = min(len(m_arr), len(h_arr))
        if n < 2:
            return OrdinalResult(n_pairs=n)

        m_arr = m_arr[:n]
        h_arr = h_arr[:n]

        rho = self._spearman_rho(m_arr, h_arr)
        tau = self._kendall_tau(m_arr, h_arr)

        # Pairwise concordance
        pairs_m = self._make_pairs(m_arr)
        pairs_h = self._make_pairs(h_arr)
        conc = self._concordance_rate(pairs_m, pairs_h)

        # p-values via scipy if available
        p_spearman, p_kendall = 1.0, 1.0
        if _HAS_SCIPY:
            _, p_spearman = sp_stats.spearmanr(m_arr, h_arr)
            _, p_kendall = sp_stats.kendalltau(m_arr, h_arr)
            p_spearman = float(p_spearman)
            p_kendall = float(p_kendall)

        ci_lo, ci_hi = self._bootstrap_ci(rho, m_arr, h_arr, self._n_bootstrap, self._alpha)

        return OrdinalResult(
            spearman_rho=rho,
            kendall_tau=tau,
            concordance_rate=conc,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            n_pairs=n,
            p_value_spearman=p_spearman,
            p_value_kendall=p_kendall,
        )

    # ------------------------------------------------------------------
    # Spearman ρ
    # ------------------------------------------------------------------

    @staticmethod
    def _spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
        """Spearman rank-correlation coefficient."""
        if _HAS_SCIPY:
            rho, _ = sp_stats.spearmanr(a, b)
            return float(rho)
        # Manual computation
        n = len(a)
        r_a = _rank_array(a)
        r_b = _rank_array(b)
        d_sq = float(np.sum((r_a - r_b) ** 2))
        return 1.0 - (6.0 * d_sq) / (n * (n ** 2 - 1))

    # ------------------------------------------------------------------
    # Kendall τ
    # ------------------------------------------------------------------

    @staticmethod
    def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
        """Kendall rank-correlation coefficient (tau-b)."""
        if _HAS_SCIPY:
            tau, _ = sp_stats.kendalltau(a, b)
            return float(tau)
        n = len(a)
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                da = a[i] - a[j]
                db = b[i] - b[j]
                prod = da * db
                if prod > 0:
                    concordant += 1
                elif prod < 0:
                    discordant += 1
        denom = concordant + discordant
        return (concordant - discordant) / denom if denom > 0 else 0.0

    # ------------------------------------------------------------------
    # Concordance rate
    # ------------------------------------------------------------------

    @staticmethod
    def _concordance_rate(pairs_m: list[int], pairs_h: list[int]) -> float:
        """Fraction of pairs where model and human agree on ordering."""
        if not pairs_m or not pairs_h:
            return 0.0
        n = min(len(pairs_m), len(pairs_h))
        agree = sum(1 for i in range(n) if pairs_m[i] == pairs_h[i])
        return agree / n

    # ------------------------------------------------------------------
    # Bootstrap CI
    # ------------------------------------------------------------------

    def _bootstrap_ci(
        self,
        statistic: float,
        a: np.ndarray,
        b: np.ndarray,
        n_bootstrap: int,
        alpha: float,
    ) -> tuple[float, float]:
        """Non-parametric bootstrap confidence interval for Spearman ρ."""
        n = len(a)
        if n < 3:
            return (statistic, statistic)
        boot_stats: list[float] = []
        for _ in range(n_bootstrap):
            idx = self._rng.choice(n, size=n, replace=True)
            rho_b = self._spearman_rho(a[idx], b[idx])
            if not math.isnan(rho_b):
                boot_stats.append(rho_b)
        if not boot_stats:
            return (statistic, statistic)
        arr = np.array(boot_stats)
        lo = float(np.percentile(arr, 100 * alpha / 2))
        hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
        return (lo, hi)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_array(data: list[Any]) -> np.ndarray:
        """Convert orderings to a flat numpy array of floats."""
        if not data:
            return np.array([])
        if isinstance(data[0], (int, float, np.integer, np.floating)):
            return np.asarray(data, dtype=float)
        # Assume tuples; extract third element as preference score
        vals = []
        for item in data:
            if isinstance(item, (tuple, list)) and len(item) >= 3:
                vals.append(float(item[2]))
            else:
                vals.append(float(item) if not isinstance(item, (tuple, list)) else 0.0)
        return np.asarray(vals, dtype=float)

    @staticmethod
    def _make_pairs(arr: np.ndarray) -> list[int]:
        """Create pairwise comparison list: +1 if arr[i] > arr[j], -1 otherwise, 0 for tie."""
        pairs: list[int] = []
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[i] - arr[j]
                if d > 0:
                    pairs.append(1)
                elif d < 0:
                    pairs.append(-1)
                else:
                    pairs.append(0)
        return pairs


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _rank_array(arr: np.ndarray) -> np.ndarray:
    """Rank values (1-based, average tie-breaking)."""
    n = len(arr)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# Weighted Kendall tau
# ---------------------------------------------------------------------------

def weighted_kendall_tau(
    x: Sequence[float],
    y: Sequence[float],
    weights: Sequence[float] | None = None,
) -> float:
    """Weighted Kendall tau-b with optional item weights.

    Pairs involving higher-weighted items count proportionally more.
    Falls back to standard tau-b when weights are uniform.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    n = len(a)
    if n < 2:
        return 0.0

    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=float)

    concordant = 0.0
    discordant = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            pair_w = w[i] * w[j]
            da = a[i] - a[j]
            db = b[i] - b[j]
            prod = da * db
            if prod > 0:
                concordant += pair_w
            elif prod < 0:
                discordant += pair_w

    denom = concordant + discordant
    if denom < 1e-12:
        return 0.0
    return float(concordant - discordant) / float(denom)


# ---------------------------------------------------------------------------
# Top-k correlation (precision at k)
# ---------------------------------------------------------------------------

def top_k_overlap(
    model_ranks: Sequence[float],
    human_ranks: Sequence[float],
    k: int = 5,
) -> float:
    """Proportion of the top-k items (by human ranking) also in model's top-k."""
    a = np.asarray(model_ranks, dtype=float)
    b = np.asarray(human_ranks, dtype=float)
    n = min(len(a), len(b))
    if n == 0 or k <= 0:
        return 0.0
    k = min(k, n)

    human_topk = set(np.argsort(b)[:k])
    model_topk = set(np.argsort(a)[:k])
    return len(human_topk & model_topk) / k


# ---------------------------------------------------------------------------
# Normalised Discounted Cumulative Gain
# ---------------------------------------------------------------------------

def ndcg(
    relevance_scores: Sequence[float],
    predicted_ranking: Sequence[int],
    k: int | None = None,
) -> float:
    """Normalised Discounted Cumulative Gain (NDCG@k).

    Parameters
    ----------
    relevance_scores : array-like
        True relevance score for each item (higher is better).
    predicted_ranking : array-like
        Indices into relevance_scores giving the predicted order.
    k : int or None
        Cutoff. If None, use all items.
    """
    rel = np.asarray(relevance_scores, dtype=float)
    order = np.asarray(predicted_ranking, dtype=int)
    n = len(order)
    if n == 0:
        return 0.0
    if k is None:
        k = n
    k = min(k, n, len(rel))

    # DCG
    dcg = 0.0
    for i in range(k):
        idx = order[i]
        if 0 <= idx < len(rel):
            dcg += (2.0 ** rel[idx] - 1.0) / np.log2(i + 2.0)

    # Ideal DCG
    ideal_order = np.argsort(-rel)
    idcg = 0.0
    for i in range(k):
        idcg += (2.0 ** rel[ideal_order[i]] - 1.0) / np.log2(i + 2.0)

    if idcg < 1e-12:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Partial Spearman correlation
# ---------------------------------------------------------------------------

def partial_spearman(
    x: Sequence[float],
    y: Sequence[float],
    z: Sequence[float],
) -> float:
    """Partial Spearman rank correlation between x and y controlling for z.

    r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz²)(1 - r_yz²))
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    c = np.asarray(z, dtype=float)
    n = min(len(a), len(b), len(c))
    if n < 3:
        return 0.0

    a, b, c = a[:n], b[:n], c[:n]
    ra = _rank_array(a)
    rb = _rank_array(b)
    rc = _rank_array(c)

    def _pearson(u: np.ndarray, v: np.ndarray) -> float:
        u_c = u - np.mean(u)
        v_c = v - np.mean(v)
        d = np.sqrt(np.sum(u_c ** 2) * np.sum(v_c ** 2))
        if d < 1e-12:
            return 0.0
        return float(np.sum(u_c * v_c) / d)

    r_xy = _pearson(ra, rb)
    r_xz = _pearson(ra, rc)
    r_yz = _pearson(rb, rc)

    denom = np.sqrt(max(0.0, (1 - r_xz ** 2) * (1 - r_yz ** 2)))
    if denom < 1e-12:
        return 0.0
    return (r_xy - r_xz * r_yz) / denom


# ---------------------------------------------------------------------------
# Mean Reciprocal Rank
# ---------------------------------------------------------------------------

def mean_reciprocal_rank(
    model_ranks: Sequence[float],
    relevant_indices: Sequence[int],
) -> float:
    """Mean Reciprocal Rank for a set of queries.

    For a single query: RR = 1 / (rank of first relevant item in model's order).
    """
    order = np.argsort(np.asarray(model_ranks, dtype=float))
    relevant = set(relevant_indices)
    for rank_pos, idx in enumerate(order, start=1):
        if idx in relevant:
            return 1.0 / rank_pos
    return 0.0
