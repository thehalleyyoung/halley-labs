"""BIC score variants for causal structure learning.

Implements the standard Bayesian Information Criterion (BIC), the
extended BIC (eBIC) with a sparsity-inducing penalty controlled by
a tuning parameter gamma, and a modified BIC with structure priors.

The BIC score for a node X_i given parents Pa_i under a linear-Gaussian
model is:

    BIC(i, Pa_i) = n * log(sigma_hat^2) + k * log(n) * penalty_weight

where sigma_hat^2 is the MLE residual variance from regressing X_i on
Pa_i, k = |Pa_i| + 1 is the number of free parameters (regression
coefficients + variance), and n is the sample size.  Higher (less
negative) scores indicate better models.  We negate the criterion so
that *larger* values are *better* (maximisation convention).
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import comb as scipy_comb


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def gaussian_log_likelihood(
    residuals: NDArray, variance: float, n: int
) -> float:
    """Compute the Gaussian log-likelihood.

    Parameters
    ----------
    residuals : NDArray
        Residual vector of length *n*.
    variance : float
        MLE variance estimate (sum of squared residuals / n).
    n : int
        Number of observations.

    Returns
    -------
    float
        The log-likelihood value.
    """
    if variance <= 0:
        variance = 1e-300
    ll = -0.5 * n * math.log(2.0 * math.pi) - 0.5 * n * math.log(variance) \
         - 0.5 * np.sum(residuals ** 2) / variance
    return float(ll)


def _ols_regression(
    y: NDArray, X: NDArray, rcond: float = 1e-12
) -> Tuple[NDArray, NDArray, float]:
    """Ordinary least-squares regression with collinearity handling.

    Parameters
    ----------
    y : NDArray, shape (n,)
        Response variable.
    X : NDArray, shape (n, p)
        Design matrix (should include intercept column if desired).
    rcond : float
        Cutoff for small singular values (passed to ``np.linalg.lstsq``).

    Returns
    -------
    coefficients : NDArray, shape (p,)
    residuals : NDArray, shape (n,)
    mle_variance : float
        ``sum(residuals**2) / n``.
    """
    n = y.shape[0]
    if X.shape[1] == 0:
        residuals = y - np.mean(y)
        var = float(np.sum(residuals ** 2) / n)
        return np.array([]), residuals, var

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=rcond)
    residuals = y - X @ coeffs
    var = float(np.sum(residuals ** 2) / n)
    return coeffs, residuals, max(var, 1e-300)


def _build_design_matrix(
    data: NDArray, parents: Sequence[int], add_intercept: bool = True
) -> NDArray:
    """Build the OLS design matrix from parent columns.

    Parameters
    ----------
    data : NDArray, shape (n, p)
    parents : Sequence[int]
        Column indices of parent variables.
    add_intercept : bool
        If *True*, prepend a column of ones.

    Returns
    -------
    NDArray, shape (n, k)
        Design matrix.
    """
    n = data.shape[0]
    parent_list = list(parents)
    if len(parent_list) == 0:
        if add_intercept:
            return np.ones((n, 1), dtype=data.dtype)
        return np.empty((n, 0), dtype=data.dtype)
    X = data[:, parent_list].copy()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if add_intercept:
        X = np.column_stack([np.ones(n, dtype=data.dtype), X])
    return X


# ---------------------------------------------------------------------------
# BICScore
# ---------------------------------------------------------------------------

class BICScore:
    """Standard BIC score for continuous Gaussian data.

    Uses the maximisation convention: larger scores indicate better fit.

    Parameters
    ----------
    data : NDArray
        Observation matrix of shape ``(n_samples, n_variables)``.
    penalty_weight : float
        Multiplier applied to the BIC penalty term (``penalty_discount``
        in the literature).  Values < 1 weaken the penalty.
    """

    def __init__(self, data: NDArray, penalty_weight: float = 1.0) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.n_samples, self.n_variables = self.data.shape
        self.penalty_weight = penalty_weight
        # Pre-compute column means for centring
        self._col_means = self.data.mean(axis=0)
        # Pre-compute marginal variances for quick sanity checks
        self._col_vars = self.data.var(axis=0, ddof=0)

    # ---- core --------------------------------------------------------

    def local_score(self, node: int, parents: Sequence[int]) -> float:
        """Return the local BIC score for *node* given *parents*.

        Score = log-likelihood - penalty  (maximisation convention).
        """
        parents = list(parents)
        self._validate_inputs(node, parents)
        ll = self.log_likelihood(node, parents)
        pen = self.penalty(node, parents)
        return ll - pen

    def score_dag(self, adj_matrix: NDArray) -> float:
        """Return the total BIC score of a DAG encoded as an adjacency matrix.

        ``adj_matrix[i, j] != 0`` means an edge from *i* to *j* (i is parent
        of j).
        """
        adj = np.asarray(adj_matrix)
        total = 0.0
        for j in range(self.n_variables):
            parents = list(np.nonzero(adj[:, j])[0])
            total += self.local_score(j, parents)
        return total

    def log_likelihood(self, node: int, parents: Sequence[int]) -> float:
        """Return the Gaussian log-likelihood of *node* given *parents*.

        Fits a linear-Gaussian model  X_node = beta^T X_parents + epsilon
        via OLS and returns the log-likelihood evaluated at the MLE.
        """
        parents = list(parents)
        y = self.data[:, node]
        X = _build_design_matrix(self.data, parents, add_intercept=True)
        _, residuals, var = _ols_regression(y, X)
        return gaussian_log_likelihood(residuals, var, self.n_samples)

    def penalty(self, node: int, parents: Sequence[int]) -> float:
        """Return the BIC penalty for the parent-set size.

        penalty = (k / 2) * log(n) * penalty_weight

        where k = |parents| + 1 (regression coefficients + variance).
        """
        k = len(list(parents)) + 1  # intercept + parents
        return self._penalty(k, self.n_samples)

    def _penalty(self, num_params: int, n: int) -> float:
        """Raw BIC penalty: (k/2) * log(n) * weight."""
        if n <= 1:
            return 0.0
        return 0.5 * num_params * math.log(n) * self.penalty_weight

    # ---- helpers -------------------------------------------------------

    def _validate_inputs(self, node: int, parents: Sequence[int]) -> None:
        """Raise on invalid node/parent indices."""
        if not 0 <= node < self.n_variables:
            raise ValueError(
                f"node {node} out of range [0, {self.n_variables})"
            )
        for p in parents:
            if not 0 <= p < self.n_variables:
                raise ValueError(
                    f"parent {p} out of range [0, {self.n_variables})"
                )
            if p == node:
                raise ValueError("A node cannot be its own parent")

    def residual_variance(self, node: int, parents: Sequence[int]) -> float:
        """Return the MLE residual variance of *node* given *parents*."""
        y = self.data[:, node]
        X = _build_design_matrix(self.data, parents, add_intercept=True)
        _, _, var = _ols_regression(y, X)
        return var

    def score_edge_addition(
        self, node: int, current_parents: Sequence[int], new_parent: int
    ) -> float:
        """Return the score change when *new_parent* is added."""
        old = self.local_score(node, current_parents)
        new_parents = list(current_parents) + [new_parent]
        new = self.local_score(node, new_parents)
        return new - old

    def score_edge_removal(
        self, node: int, current_parents: Sequence[int], removed_parent: int
    ) -> float:
        """Return the score change when *removed_parent* is dropped."""
        old = self.local_score(node, current_parents)
        new_parents = [p for p in current_parents if p != removed_parent]
        new = self.local_score(node, new_parents)
        return new - old

    def __repr__(self) -> str:
        return (
            f"BICScore(n_samples={self.n_samples}, "
            f"n_variables={self.n_variables}, "
            f"penalty_weight={self.penalty_weight})"
        )


# ---------------------------------------------------------------------------
# ExtendedBICScore  (eBIC / Chen & Chen 2008)
# ---------------------------------------------------------------------------

class ExtendedBICScore(BICScore):
    """Extended BIC (eBIC) with a sparsity-inducing gamma parameter.

    The eBIC penalty adds a combinatorial term that accounts for the
    number of possible parent sets of a given size:

        eBIC_penalty = BIC_penalty + 2 * gamma * log C(p-1, k)

    where p is the total number of variables and k is the parent-set size.

    Parameters
    ----------
    data : NDArray
        Observation matrix of shape ``(n_samples, n_variables)``.
    gamma : float
        Sparsity parameter in ``[0, 1]``.  ``gamma = 0`` gives the
        standard BIC; ``gamma = 1`` gives the strongest sparsity penalty.
    """

    def __init__(self, data: NDArray, gamma: float = 0.5) -> None:
        super().__init__(data, penalty_weight=1.0)
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        self.gamma = gamma

    def penalty(self, node: int, parents: Sequence[int]) -> float:
        """Return the eBIC penalty (standard + combinatorial term)."""
        k = len(list(parents)) + 1
        base = self._penalty(k, self.n_samples)
        return base + self.extended_penalty(node, parents)

    def extended_penalty(self, node: int, parents: Sequence[int]) -> float:
        """Return the extra combinatorial penalty for eBIC.

        extra = 2 * gamma * log C(p-1, |parents|)
        """
        num_parents = len(list(parents))
        if num_parents == 0:
            return 0.0
        p = self.n_variables
        log_comb = self._log_comb(p - 1, num_parents)
        return 2.0 * self.gamma * log_comb

    @staticmethod
    def _log_comb(n: int, k: int) -> float:
        """Compute log C(n, k) using gammaln for numerical stability."""
        from scipy.special import gammaln
        if k < 0 or k > n:
            return -math.inf
        return float(
            gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        )

    def __repr__(self) -> str:
        return (
            f"ExtendedBICScore(n_samples={self.n_samples}, "
            f"n_variables={self.n_variables}, gamma={self.gamma})"
        )


# ---------------------------------------------------------------------------
# ModifiedBIC  (structure prior variant)
# ---------------------------------------------------------------------------

class ModifiedBICScore(BICScore):
    """Modified BIC with a structural prior on parent-set size.

    Supports several priors:
    - ``'uniform'``: flat prior over all parent sets (no extra penalty).
    - ``'sparse'``:  geometric prior favouring small parent sets.
    - ``'erdos_renyi'``: ER-graph prior with given edge probability.

    Parameters
    ----------
    data : NDArray
        Observation matrix ``(n_samples, n_variables)``.
    prior : str
        One of ``'uniform'``, ``'sparse'``, ``'erdos_renyi'``.
    prior_edge_prob : float
        Edge probability for the Erdos-Renyi prior (default 0.5).
    penalty_weight : float
        BIC penalty multiplier.
    """

    VALID_PRIORS = ("uniform", "sparse", "erdos_renyi")

    def __init__(
        self,
        data: NDArray,
        prior: str = "uniform",
        prior_edge_prob: float = 0.5,
        penalty_weight: float = 1.0,
    ) -> None:
        super().__init__(data, penalty_weight=penalty_weight)
        if prior not in self.VALID_PRIORS:
            raise ValueError(
                f"prior must be one of {self.VALID_PRIORS}, got '{prior}'"
            )
        self.prior = prior
        self.prior_edge_prob = prior_edge_prob

    def penalty(self, node: int, parents: Sequence[int]) -> float:
        """Return modified BIC penalty = BIC penalty - log prior."""
        k = len(list(parents)) + 1
        base = self._penalty(k, self.n_samples)
        return base - self._log_structure_prior(len(list(parents)))

    def _log_structure_prior(self, num_parents: int) -> float:
        """Return log P(|Pa| = num_parents) under the chosen prior."""
        p = self.n_variables - 1  # candidates
        if self.prior == "uniform":
            return 0.0
        elif self.prior == "sparse":
            # geometric: P(k) propto 0.5^k
            return num_parents * math.log(0.5)
        elif self.prior == "erdos_renyi":
            # ER: P(edge) = q  =>  P(pa set of size k) = C(p,k) q^k (1-q)^(p-k)
            q = self.prior_edge_prob
            q = max(min(q, 1.0 - 1e-15), 1e-15)
            log_comb = ExtendedBICScore._log_comb(p, num_parents)
            return (
                log_comb
                + num_parents * math.log(q)
                + (p - num_parents) * math.log(1.0 - q)
            )
        return 0.0

    def __repr__(self) -> str:
        return (
            f"ModifiedBICScore(n_samples={self.n_samples}, "
            f"n_variables={self.n_variables}, prior='{self.prior}')"
        )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def compare_bic_scores(
    data: NDArray,
    node: int,
    parents: Sequence[int],
    penalty_weights: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Compare BIC scores for different penalty weights.

    Returns a dict mapping ``"pw=<weight>"`` to the corresponding local
    BIC score.
    """
    if penalty_weights is None:
        penalty_weights = [0.5, 1.0, 2.0, 4.0]
    results: Dict[str, float] = {}
    for pw in penalty_weights:
        scorer = BICScore(data, penalty_weight=pw)
        results[f"pw={pw}"] = scorer.local_score(node, parents)
    return results


def select_best_parents(
    data: NDArray,
    node: int,
    max_parents: int = 3,
    penalty_weight: float = 1.0,
) -> Tuple[List[int], float]:
    """Exhaustively select the parent set that maximises BIC.

    Parameters
    ----------
    data : NDArray
    node : int
    max_parents : int
    penalty_weight : float

    Returns
    -------
    best_parents : list of int
    best_score : float
    """
    scorer = BICScore(data, penalty_weight=penalty_weight)
    candidates = [v for v in range(scorer.n_variables) if v != node]
    best_parents: List[int] = []
    best_score = scorer.local_score(node, [])

    for size in range(1, min(max_parents, len(candidates)) + 1):
        for combo in combinations(candidates, size):
            s = scorer.local_score(node, list(combo))
            if s > best_score:
                best_score = s
                best_parents = list(combo)
    return best_parents, best_score
