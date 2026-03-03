"""Approximate scoring for large graphs.

Provides fast approximate local-score computation and parent-set
pruning strategies that trade a bounded error for speed.  Three
strategies are implemented:

* **ApproximateBIC** – subsample-based BIC with confidence bounds.
* **ScreeningScore** – marginal / conditional screening to prune
  unpromising parent candidates.
* **RandomizedScore** – Johnson-Lindenstrauss random projection to
  reduce dimensionality before regression scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import (
    Any,
    Callable,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np
from numpy.typing import NDArray
from scipy import linalg


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ScoreApproximation:
    """Result of an approximate score computation."""

    exact_score: Optional[float]
    approximate_score: float
    error_bound: float
    method: str


# ---------------------------------------------------------------------------
# Helper: BIC local score (exact, for a single node)
# ---------------------------------------------------------------------------

def _bic_local_score(
    data: NDArray, node: int, parents: FrozenSet[int]
) -> float:
    """Exact BIC local score for *node* given *parents*.

    BIC = n·ln(RSS/n) + k·ln(n)  where k = |parents| + 1 (intercept).
    """
    n, p = data.shape
    y = data[:, node]
    parent_list = sorted(parents)

    if len(parent_list) == 0:
        rss = float(np.sum((y - np.mean(y)) ** 2))
    else:
        X = data[:, parent_list]
        X = np.column_stack([np.ones(n), X])
        try:
            beta, rss_arr, _, _ = np.linalg.lstsq(X, y, rcond=None)
            if len(rss_arr) > 0:
                rss = float(rss_arr[0])
            else:
                residuals = y - X @ beta
                rss = float(np.dot(residuals, residuals))
        except np.linalg.LinAlgError:
            residuals = y - np.mean(y)
            rss = float(np.dot(residuals, residuals))

    rss = max(rss, 1e-15)
    k = len(parent_list) + 1
    return n * np.log(rss / n) + k * np.log(n)


# ---------------------------------------------------------------------------
# ApproximateBIC
# ---------------------------------------------------------------------------

class ApproximateBIC:
    """Approximate BIC scoring via subsampling with confidence bounds.

    Evaluates the BIC on multiple random subsamples and aggregates
    results, providing both a point estimate and a confidence bound.

    Parameters
    ----------
    data : array of shape ``(n, p)``
        Full dataset.
    sample_fraction : float
        Fraction of data used per subsample.
    n_subsamples : int
        Number of Monte-Carlo subsamples.
    rng_seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        data: NDArray,
        sample_fraction: float = 0.1,
        n_subsamples: int = 10,
        rng_seed: int = 42,
    ) -> None:
        self._data = np.asarray(data, dtype=np.float64)
        self._n, self._p = self._data.shape
        self._frac = max(0.01, min(sample_fraction, 1.0))
        self._n_sub = n_subsamples
        self._rng = np.random.RandomState(rng_seed)

    def local_score(
        self, node: int, parents: FrozenSet[int]
    ) -> ScoreApproximation:
        """Approximate BIC for *node* given *parents*."""
        sub_n = max(10, int(self._n * self._frac))
        scores: List[float] = []
        for _ in range(self._n_sub):
            idx = self._rng.choice(self._n, size=sub_n, replace=False)
            scores.append(
                self._subsample_score(self._data[idx], node, parents)
            )
        agg = self._aggregate_scores(scores)
        bound = self._confidence_bound(scores)
        return ScoreApproximation(
            exact_score=None,
            approximate_score=agg,
            error_bound=bound,
            method="subsample_bic",
        )

    @staticmethod
    def _subsample_score(
        data_subset: NDArray, node: int, parents: FrozenSet[int]
    ) -> float:
        """BIC on a single subsample."""
        return _bic_local_score(data_subset, node, parents)

    @staticmethod
    def _aggregate_scores(scores: List[float]) -> float:
        """Aggregate subsample scores via trimmed mean."""
        arr = np.array(scores)
        lo, hi = np.percentile(arr, [10, 90])
        trimmed = arr[(arr >= lo) & (arr <= hi)]
        if len(trimmed) == 0:
            return float(np.median(arr))
        return float(np.mean(trimmed))

    @staticmethod
    def _confidence_bound(scores: List[float]) -> float:
        """95% confidence half-width on the mean score."""
        arr = np.array(scores)
        if len(arr) < 2:
            return float("inf")
        se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
        return 1.96 * se


# ---------------------------------------------------------------------------
# ScreeningScore
# ---------------------------------------------------------------------------

class ScreeningScore:
    """Screen out unpromising parent candidates before scoring.

    Two-stage screening: (1) marginal correlation, (2) conditional
    partial correlation given the top-k marginal candidates.

    Parameters
    ----------
    data : array of shape ``(n, p)``
    screening_threshold : float
        Minimum absolute correlation to pass the marginal screen.
    """

    def __init__(
        self,
        data: NDArray,
        screening_threshold: float = 0.01,
    ) -> None:
        self._data = np.asarray(data, dtype=np.float64)
        self._n, self._p = self._data.shape
        self._threshold = screening_threshold
        # Pre-compute correlation matrix once
        self._corr = np.corrcoef(self._data, rowvar=False)
        np.fill_diagonal(self._corr, 0.0)

    def screen_parents(
        self, node: int, candidates: Set[int]
    ) -> List[int]:
        """Return candidates surviving marginal + conditional screens."""
        marginal = self._marginal_screen(node, candidates)
        if len(marginal) <= 3:
            return marginal
        top_k = marginal[:3]
        return self._conditional_screen(node, marginal, top_k)

    def _marginal_screen(
        self, node: int, candidates: Set[int]
    ) -> List[int]:
        """Keep candidates whose |corr(node, c)| > threshold."""
        scored: List[Tuple[float, int]] = []
        for c in candidates:
            if c == node or c >= self._p:
                continue
            r = abs(self._corr[node, c])
            if r > self._threshold:
                scored.append((r, c))
        scored.sort(reverse=True)
        return [c for _, c in scored]

    def _marginal_correlation(
        self, data: NDArray, node: int, candidate: int
    ) -> float:
        """Absolute marginal correlation between *node* and *candidate*."""
        return float(abs(self._corr[node, candidate]))

    def _conditional_screen(
        self,
        node: int,
        candidates: List[int],
        top_k: List[int],
    ) -> List[int]:
        """Conditional partial-correlation screen given *top_k*."""
        data = self._data
        n = data.shape[0]
        survivors: List[int] = list(top_k)

        for c in candidates:
            if c in top_k:
                continue
            # Partial correlation: regress both node and c on top_k
            Z = data[:, top_k]
            Z_aug = np.column_stack([np.ones(n), Z])
            try:
                res_node = data[:, node] - Z_aug @ np.linalg.lstsq(
                    Z_aug, data[:, node], rcond=None
                )[0]
                res_c = data[:, c] - Z_aug @ np.linalg.lstsq(
                    Z_aug, data[:, c], rcond=None
                )[0]
                denom = np.sqrt(
                    np.dot(res_node, res_node) * np.dot(res_c, res_c)
                )
                if denom < 1e-15:
                    continue
                pcor = abs(np.dot(res_node, res_c) / denom)
            except np.linalg.LinAlgError:
                pcor = 0.0

            if pcor > self._threshold:
                survivors.append(c)

        return survivors

    def promising_parent_sets(
        self, node: int, max_parents: int
    ) -> List[FrozenSet[int]]:
        """Enumerate promising parent sets after screening.

        Only combinations of screened candidates up to *max_parents*
        are returned.
        """
        all_candidates = set(range(self._p)) - {node}
        screened = self.screen_parents(node, all_candidates)
        result: List[FrozenSet[int]] = [frozenset()]
        for k in range(1, min(max_parents, len(screened)) + 1):
            for combo in combinations(screened, k):
                result.append(frozenset(combo))
        return result


# ---------------------------------------------------------------------------
# RandomizedScore
# ---------------------------------------------------------------------------

class RandomizedScore:
    """Randomised scoring via Johnson-Lindenstrauss sketching.

    Projects the data into a lower-dimensional space before running
    OLS regression, yielding an approximate BIC in O(n·k) where
    *k ≪ p*.

    Parameters
    ----------
    data : array of shape ``(n, p)``
    base_score : callable or None
        Optional exact scorer for calibration.
    sketch_size : int
        Target dimension after projection.
    rng_seed : int
        Random seed.
    """

    def __init__(
        self,
        data: NDArray,
        base_score: Any = None,
        sketch_size: int = 100,
        rng_seed: int = 42,
    ) -> None:
        self._data = np.asarray(data, dtype=np.float64)
        self._n, self._p = self._data.shape
        self._base_score = base_score
        self._sketch_size = min(sketch_size, self._n)
        self._rng = np.random.RandomState(rng_seed)
        self._projection = self._random_projection_matrix(
            self._n, self._sketch_size
        )
        self._sketched = self._projection @ self._data  # (k, p)

    def local_score(
        self, node: int, parents: FrozenSet[int]
    ) -> ScoreApproximation:
        """Approximate BIC score via sketched regression."""
        score = self._sketched_regression(node, parents)
        return ScoreApproximation(
            exact_score=None,
            approximate_score=score,
            error_bound=self._error_bound(node, parents),
            method="jl_sketch",
        )

    def _random_projection_matrix(
        self, n: int, k: int
    ) -> NDArray:
        """Build a ``(k, n)`` JL random projection matrix."""
        return self._rng.randn(k, n) / np.sqrt(k)

    @staticmethod
    def _random_projection(
        data: NDArray, sketch_size: int
    ) -> NDArray:
        """Apply a JL random projection to *data* ``(n, p)``."""
        n = data.shape[0]
        k = min(sketch_size, n)
        R = np.random.randn(k, n) / np.sqrt(k)
        return R @ data

    def _sketched_regression(
        self, node: int, parents: FrozenSet[int]
    ) -> float:
        """BIC-like score on the sketched data."""
        y = self._sketched[:, node]
        k = self._sketch_size
        parent_list = sorted(parents)

        if len(parent_list) == 0:
            rss = float(np.sum((y - np.mean(y)) ** 2))
        else:
            X = self._sketched[:, parent_list]
            X = np.column_stack([np.ones(k), X])
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = y - X @ beta
                rss = float(np.dot(residuals, residuals))
            except np.linalg.LinAlgError:
                rss = float(np.sum((y - np.mean(y)) ** 2))

        rss = max(rss, 1e-15)
        num_params = len(parent_list) + 1
        # Scale score to approximate full-data BIC
        scale = self._n / k
        return scale * k * np.log(rss / k) + num_params * np.log(self._n)

    def _error_bound(
        self, node: int, parents: FrozenSet[int]
    ) -> float:
        """Rough theoretical error bound from JL lemma."""
        eps = np.sqrt(8 * np.log(self._n) / self._sketch_size)
        return float(eps)


# ---------------------------------------------------------------------------
# ApproximateScorer – unified façade
# ---------------------------------------------------------------------------

class ApproximateScorer:
    """Unified approximate scorer combining subsampling and screening.

    Parameters
    ----------
    base_scorer : object
        Exact scorer used as a reference when affordable.
    max_parents : int
        Maximum parent-set size to consider.
    n_samples : int
        Number of Monte-Carlo samples for stochastic approximations.
    """

    def __init__(
        self,
        base_scorer: object,
        max_parents: int = 5,
        n_samples: int = 1000,
    ) -> None:
        self._base_scorer = base_scorer
        self._max_parents = max_parents
        self._n_samples = n_samples
        self._data: Optional[NDArray] = None
        self._approx_bic: Optional[ApproximateBIC] = None
        self._screener: Optional[ScreeningScore] = None
        self._randomised: Optional[RandomizedScore] = None

    def set_data(self, data: NDArray) -> None:
        """Attach dataset and initialise sub-scorers."""
        self._data = np.asarray(data, dtype=np.float64)
        self._approx_bic = ApproximateBIC(
            self._data, n_subsamples=min(self._n_samples, 50)
        )
        self._screener = ScreeningScore(self._data)
        self._randomised = RandomizedScore(self._data)

    def approximate_local_score(
        self, node: int, parents: FrozenSet[int]
    ) -> ScoreApproximation:
        """Return an approximate local score with error bound."""
        if self._data is None:
            if hasattr(self._base_scorer, "data"):
                self.set_data(self._base_scorer.data)
            else:
                raise ValueError("No data attached; call set_data first")
        assert self._approx_bic is not None
        return self._approx_bic.local_score(node, parents)

    def pruned_parent_sets(
        self, node: int, candidates: Set[int]
    ) -> List[FrozenSet[int]]:
        """Return only promising parent sets after screening."""
        if self._screener is None:
            if self._data is None:
                raise ValueError("No data attached; call set_data first")
            self._screener = ScreeningScore(self._data)
        return self._screener.promising_parent_sets(
            node, self._max_parents
        )

    def feature_selection_filter(
        self, node: int, data: NDArray, k: int
    ) -> List[int]:
        """Select the top-*k* candidate parents via correlation ranking."""
        data = np.asarray(data, dtype=np.float64)
        n, p = data.shape
        y = data[:, node]
        scores: List[Tuple[float, int]] = []
        for j in range(p):
            if j == node:
                continue
            r = abs(float(np.corrcoef(y, data[:, j])[0, 1]))
            scores.append((r, j))
        scores.sort(reverse=True)
        return [j for _, j in scores[:k]]
