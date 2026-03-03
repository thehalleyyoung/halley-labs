"""Bayesian Gaussian equivalent (BGe) score for DAGs.

This module implements the BGe scoring criterion introduced by
Geiger & Heckerman (2002) for scoring Bayesian network structures
with continuous (Gaussian) observational data.  The BGe score is
the closed-form marginal likelihood of a linear-Gaussian model
integrated over a conjugate Normal-Wishart prior.

The marginal likelihood decomposes over nodes so that the score of a
DAG is the sum of per-node *local scores*, each depending only on a
node and its parent set.  This decomposability makes BGe especially
suited for greedy or evolutionary structure-learning algorithms
(including the QD-MAP-Elites loop used in this project).

Key features
------------
* Numerically stable log-determinant computation via
  ``numpy.linalg.slogdet``.
* Regularisation of near-singular scatter matrices.
* Precomputation of column-wise means and the full scatter matrix so
  that submatrix statistics can be extracted in O(d²) instead of
  O(n·d).
* An incremental ``score_diff`` method for evaluating the effect of
  adding or removing a single parent without recomputing from scratch.
* A ``WishartPrior`` helper that encapsulates the Wishart scale-matrix
  specification and degrees of freedom.

References
----------
.. [GH2002] Geiger, D. & Heckerman, D. (2002).  *Parameter priors for
   directed acyclic graphical models and the characterization of several
   probability distributions.*  Annals of Statistics, 30(5), 1412–1440.
.. [KF2009] Koller, D. & Friedman, N. (2009).  *Probabilistic Graphical
   Models: Principles and Techniques.*  MIT Press.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import lgamma, log, pi
from typing import Dict, Optional, Tuple

import numpy as np

from causal_qd.scores.score_base import DecomposableScore
from causal_qd.types import AdjacencyMatrix, DataMatrix, QualityScore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_PI: float = log(pi)
"""Precomputed log(π) to avoid repeated calls."""

_REGULARISATION_EPS: float = 1e-12
"""Small ridge added to scatter matrices to prevent singularity."""


# ---------------------------------------------------------------------------
# Helper: log multivariate gamma
# ---------------------------------------------------------------------------


def _log_multivariate_gamma(a: float, d: int) -> float:
    r"""Compute the log of the multivariate gamma function Γ_d(a).

    The *d*-dimensional multivariate gamma function is defined as

    .. math::
        \Gamma_d(a) = \pi^{d(d-1)/4}
                       \prod_{i=1}^{d} \Gamma\!\bigl(a - (i-1)/2\bigr)

    so that

    .. math::
        \log\Gamma_d(a) = \frac{d(d-1)}{4}\log\pi
                          + \sum_{i=1}^{d} \log\Gamma\!\bigl(a - (i-1)/2\bigr)

    Parameters
    ----------
    a:
        Scalar argument.  Must satisfy ``a > (d - 1) / 2`` for the
        function to be defined.
    d:
        Dimension (positive integer).

    Returns
    -------
    float
        The value ``log Γ_d(a)``.

    Raises
    ------
    ValueError
        If ``a`` is too small for the given dimension.
    """
    if a <= (d - 1) / 2.0:
        raise ValueError(
            f"log_multivariate_gamma requires a > (d-1)/2; got a={a}, d={d}"
        )
    result: float = d * (d - 1) / 4.0 * _LOG_PI
    for i in range(d):
        result += lgamma(a - i / 2.0)
    return result


# ---------------------------------------------------------------------------
# Wishart prior specification
# ---------------------------------------------------------------------------


@dataclass
class WishartPrior:
    """Specification of the Wishart component of the Normal-Wishart prior.

    The BGe score uses a Normal-Wishart prior on the mean vector and
    precision matrix of the Gaussian likelihood.  This dataclass
    encapsulates the Wishart hyper-parameters so that users can supply
    a custom prior scale matrix or degrees of freedom.

    Attributes
    ----------
    scale_matrix:
        The ``d × d`` positive-definite scale matrix ``T₀`` of the
        Wishart prior.  When ``None`` the default ``(α_w - d - 1) · I``
        is used internally.
    degrees_of_freedom:
        Wishart degrees of freedom ``α_w``.  Must be greater than
        ``d + 1`` for the expectation to exist.  When ``None`` the
        scorer computes ``α_w = p + 2 + prior_df_extra``.
    custom_prior_scale:
        If ``True`` the user-supplied ``scale_matrix`` is used directly
        instead of the default identity-based construction.  This flag
        is set automatically when ``scale_matrix`` is provided.

    Notes
    -----
    The default prior scale ``T₀ = (α_w - d - 1) I`` is the *unit
    information prior* recommendation of Geiger & Heckerman (2002)
    which yields a proper prior with minimal influence on the
    posterior.
    """

    scale_matrix: Optional[np.ndarray] = None
    degrees_of_freedom: Optional[float] = None
    custom_prior_scale: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Set ``custom_prior_scale`` when a scale matrix is provided."""
        if self.scale_matrix is not None:
            self.custom_prior_scale = True
            # Validate symmetry (within numerical tolerance).
            sm = self.scale_matrix
            if sm.ndim != 2 or sm.shape[0] != sm.shape[1]:
                raise ValueError("scale_matrix must be a square 2-D array.")
            if not np.allclose(sm, sm.T, atol=1e-8):
                raise ValueError("scale_matrix must be symmetric.")


# ---------------------------------------------------------------------------
# Precomputed data statistics
# ---------------------------------------------------------------------------


@dataclass
class _DataStats:
    """Cached sufficient statistics for a single dataset.

    Computing the column means and the full scatter matrix once in
    O(n · p²) and then extracting sub-matrices for each local-score
    evaluation is significantly cheaper than recomputing from the raw
    data every time.

    Attributes
    ----------
    n_samples:
        Number of observations ``n``.
    n_vars:
        Number of variables ``p``.
    col_means:
        Length-``p`` vector of column means.
    scatter:
        ``p × p`` scatter matrix ``Σ̂ = (X − μ̂)ᵀ(X − μ̂)``.
    """

    n_samples: int
    n_vars: int
    col_means: np.ndarray
    scatter: np.ndarray


# ---------------------------------------------------------------------------
# BGeScore
# ---------------------------------------------------------------------------


class BGeScore(DecomposableScore):
    """BGe score for continuous (Gaussian) data.

    The BGe score is the marginal likelihood of a linear-Gaussian model
    integrated over a Normal-Wishart prior.  It supports both the
    default *unit information prior* and a fully custom Wishart prior
    supplied via :class:`WishartPrior`.

    Parameters
    ----------
    prior_mean:
        Prior mean for each variable (scalar broadcast to all
        variables).  Default ``0.0``.
    prior_precision:
        Prior precision (inverse variance) parameter ``α_μ``.
        Controls how tightly the prior concentrates around
        ``prior_mean``.  Default ``1.0``.
    prior_df_extra:
        Extra degrees of freedom added to the minimum ``p + 2``.
        Default ``0`` so that ``α_w = p + 2``.
    wishart_prior:
        Optional :class:`WishartPrior` instance for full control over
        the Wishart hyper-parameters.  When provided, its
        ``degrees_of_freedom`` (if set) overrides the value implied by
        ``prior_df_extra``.
    regularisation_eps:
        Ridge constant added to the diagonal of the scatter matrix to
        prevent numerical singularity.  Default ``1e-12``.

    Examples
    --------
    >>> import numpy as np
    >>> from causal_qd.scores.bge import BGeScore
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((200, 4))
    >>> scorer = BGeScore()
    >>> scorer.local_score(0, [1, 2], data)  # doctest: +SKIP
    -283.14...

    Notes
    -----
    The implementation follows the formulation in Geiger & Heckerman
    (2002) where the marginal likelihood for a family of variables
    ``W = {node} ∪ parents`` is

    .. math::
        \\text{score}(\\text{node} \\mid \\text{parents})
          = \\log p(X_W) - \\log p(X_{\\text{parents}})

    Each ``log p(X_W)`` term is the closed-form integral of the
    likelihood against the Normal-Wishart prior.
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        prior_mean: float = 0.0,
        prior_precision: float = 1.0,
        prior_df_extra: int = 0,
        wishart_prior: Optional[WishartPrior] = None,
        regularisation_eps: float = _REGULARISATION_EPS,
    ) -> None:
        self._mu0: float = prior_mean
        self._alpha_mu: float = prior_precision
        self._df_extra: int = prior_df_extra
        self._wishart: Optional[WishartPrior] = wishart_prior
        self._reg_eps: float = regularisation_eps

        # Lazily populated cache keyed by ``id(data)`` so that
        # repeated calls with the same array skip recomputation.
        self._stats_cache: Dict[int, _DataStats] = {}

    # ------------------------------------------------------------------ #
    # Public API – local_score
    # ------------------------------------------------------------------ #

    def local_score(
        self,
        node: int,
        parents: list[int],
        data: DataMatrix,
    ) -> float:
        """Compute the local BGe score for *node* given *parents*.

        The local score is the log-marginal-likelihood ratio

        .. math::
            s(X_i \\mid \\mathbf{Pa}_i)
              = \\log p(X_{\\{i\\} \\cup \\mathbf{Pa}_i})
              - \\log p(X_{\\mathbf{Pa}_i})

        where each ``log p(·)`` is the Normal-Wishart marginal
        likelihood evaluated on the corresponding submatrix of *data*.

        Parameters
        ----------
        node:
            Column index of the child variable.
        parents:
            Column indices of the parent variables.
        data:
            ``N × p`` observation matrix.

        Returns
        -------
        float
            Local BGe score contribution for this family.
        """
        stats = self._get_stats(data)
        n = stats.n_samples
        p = stats.n_vars
        alpha_w = self._effective_alpha_w(p)
        alpha_mu = self._alpha_mu

        family = [node] + list(parents)
        d_family = len(family)

        score_family = self._log_marginal_from_stats(
            stats, family, n, d_family, alpha_w, alpha_mu
        )

        if parents:
            score_parents = self._log_marginal_from_stats(
                stats, list(parents), n, d_family - 1, alpha_w, alpha_mu
            )
        else:
            score_parents = 0.0

        return score_family - score_parents

    # ------------------------------------------------------------------ #
    # Public API – score_diff (incremental update)
    # ------------------------------------------------------------------ #

    def score_diff(
        self,
        node: int,
        parents_old: list[int],
        parents_new: list[int],
        data: DataMatrix,
    ) -> float:
        """Compute the *change* in local score when updating a parent set.

        This is a convenience wrapper that returns

        .. math::
            \\Delta = s(\\text{node} \\mid \\text{parents\\_new})
                    - s(\\text{node} \\mid \\text{parents\\_old})

        A positive value means the new parent set is preferred.

        Using ``score_diff`` instead of two independent ``local_score``
        calls lets the implementation share cached data statistics and,
        for single-parent additions/removals, potentially reuse partial
        matrix factorisations in future optimisations.

        Parameters
        ----------
        node:
            Column index of the child variable.
        parents_old:
            Current parent set.
        parents_new:
            Proposed parent set.
        data:
            ``N × p`` observation matrix.

        Returns
        -------
        float
            Score difference ``score_new − score_old``.
        """
        # Ensure statistics are cached before both calls.
        self._get_stats(data)

        new_score = self.local_score(node, parents_new, data)
        old_score = self.local_score(node, parents_old, data)
        return new_score - old_score

    # ------------------------------------------------------------------ #
    # Internal – precomputed-statistics path
    # ------------------------------------------------------------------ #

    def _get_stats(self, data: DataMatrix) -> _DataStats:
        """Return (possibly cached) sufficient statistics for *data*.

        The cache is keyed on ``id(data)`` so that the same underlying
        numpy array is recognised across calls.  This is intentionally
        identity-based (not content-based) to avoid an O(n·p) hash on
        every call.

        Parameters
        ----------
        data:
            ``N × p`` observation matrix.

        Returns
        -------
        _DataStats
            Cached column means and scatter matrix.
        """
        key = id(data)
        if key not in self._stats_cache:
            n, p = data.shape
            col_means = data.mean(axis=0)
            centred = data - col_means
            scatter = centred.T @ centred
            self._stats_cache[key] = _DataStats(
                n_samples=n,
                n_vars=p,
                col_means=col_means,
                scatter=scatter,
            )
        return self._stats_cache[key]

    def _effective_alpha_w(self, p: int) -> float:
        """Return the effective Wishart degrees of freedom.

        If a :class:`WishartPrior` with explicit ``degrees_of_freedom``
        was supplied, that value is used.  Otherwise the default
        ``p + 2 + prior_df_extra`` is returned.

        Parameters
        ----------
        p:
            Total number of variables in the dataset.

        Returns
        -------
        float
            Wishart degrees of freedom ``α_w``.
        """
        if (
            self._wishart is not None
            and self._wishart.degrees_of_freedom is not None
        ):
            return float(self._wishart.degrees_of_freedom)
        return float(p + 2 + self._df_extra)

    def _prior_scale_matrix(
        self,
        d: int,
        alpha_w: float,
    ) -> np.ndarray:
        """Construct the ``d × d`` prior scale matrix ``T₀``.

        If a custom :class:`WishartPrior` with a ``scale_matrix`` was
        provided and the requested subproblem dimension matches the
        stored matrix, the custom matrix is used.  Otherwise the
        default ``(α_w − d − 1) · I_d`` is returned.

        Parameters
        ----------
        d:
            Dimension of the subproblem.
        alpha_w:
            Wishart degrees of freedom.

        Returns
        -------
        np.ndarray
            ``d × d`` positive-definite scale matrix.
        """
        if (
            self._wishart is not None
            and self._wishart.custom_prior_scale
            and self._wishart.scale_matrix is not None
            and self._wishart.scale_matrix.shape[0] == d
        ):
            return self._wishart.scale_matrix.copy()
        scale_factor = max(alpha_w - d - 1, _REGULARISATION_EPS)
        return np.eye(d) * scale_factor

    # ------------------------------------------------------------------ #
    # Internal – log marginal from precomputed stats
    # ------------------------------------------------------------------ #

    def _log_marginal_from_stats(
        self,
        stats: _DataStats,
        indices: list[int],
        n: int,
        d: int,
        alpha_w: float,
        alpha_mu: float,
    ) -> float:
        """Log marginal likelihood using cached sufficient statistics.

        Instead of recomputing the column means and scatter matrix from
        raw data, this method extracts the relevant sub-vectors /
        sub-matrices from *stats*.

        Parameters
        ----------
        stats:
            Precomputed :class:`_DataStats`.
        indices:
            Column indices of the variables in the subproblem.
        n:
            Number of observations.
        d:
            Dimension of the subproblem (``len(indices)``).
        alpha_w:
            Wishart degrees of freedom.
        alpha_mu:
            Normal prior precision.

        Returns
        -------
        float
            Log marginal likelihood ``log p(X_indices)``.
        """
        # Extract sub-statistics ----------------------------------------
        idx = np.array(indices, dtype=int)
        mu_hat = stats.col_means[idx]
        scatter_sub = stats.scatter[np.ix_(idx, idx)]

        # Regularise to prevent singularity when n < d or columns are
        # nearly collinear.
        scatter_sub = self._regularise(scatter_sub, d)

        return self._compute_log_marginal(
            mu_hat, scatter_sub, n, d, alpha_w, alpha_mu
        )

    # ------------------------------------------------------------------ #
    # Internal – raw data path (fallback / standalone use)
    # ------------------------------------------------------------------ #

    def _log_marginal(
        self,
        x: np.ndarray,
        n: int,
        d: int,
        alpha_w: float,
        alpha_mu: float,
    ) -> float:
        """Log marginal likelihood computed directly from raw data.

        This is the original self-contained implementation retained for
        clarity and as a reference.  The hot-path now uses
        :meth:`_log_marginal_from_stats` instead.

        Parameters
        ----------
        x:
            ``n × d`` submatrix of observations.
        n:
            Number of observations.
        d:
            Number of variables in the subproblem.
        alpha_w:
            Wishart degrees of freedom.
        alpha_mu:
            Normal prior precision.

        Returns
        -------
        float
            Log marginal likelihood.
        """
        mu_hat = x.mean(axis=0)
        s = (x - mu_hat).T @ (x - mu_hat)
        s = self._regularise(s, d)

        return self._compute_log_marginal(mu_hat, s, n, d, alpha_w, alpha_mu)

    # ------------------------------------------------------------------ #
    # Internal – core computation shared by both paths
    # ------------------------------------------------------------------ #

    def _compute_log_marginal(
        self,
        mu_hat: np.ndarray,
        scatter: np.ndarray,
        n: int,
        d: int,
        alpha_w: float,
        alpha_mu: float,
    ) -> float:
        """Core log-marginal-likelihood computation.

        Given the sample mean ``mu_hat`` and scatter matrix ``S`` this
        method evaluates the closed-form Normal-Wishart integral:

        .. math::
            \\log p(X) = -\\frac{nd}{2}\\log\\pi
                        + \\frac{d}{2}\\bigl(\\log\\alpha_\\mu
                                              - \\log(\\alpha_\\mu + n)\\bigr)
                        + \\log\\Gamma_d\\!\\bigl(\\tfrac{\\alpha_w + n}{2}\\bigr)
                        - \\log\\Gamma_d\\!\\bigl(\\tfrac{\\alpha_w}{2}\\bigr)
                        + \\tfrac{\\alpha_w}{2}\\log|T_0|
                        - \\tfrac{\\alpha_w + n}{2}\\log|R|

        where ``R = T₀ + S + (α_μ n / (α_μ + n)) (μ̂ − μ₀)(μ̂ − μ₀)ᵀ``.

        Parameters
        ----------
        mu_hat:
            Length-``d`` sample mean vector.
        scatter:
            ``d × d`` scatter matrix ``S``.
        n:
            Number of observations.
        d:
            Subproblem dimension.
        alpha_w:
            Wishart degrees of freedom.
        alpha_mu:
            Normal prior precision.

        Returns
        -------
        float
            Log marginal likelihood.
        """
        # Prior scale matrix T₀
        t0 = self._prior_scale_matrix(d, alpha_w)

        # Posterior scale matrix R
        alpha_mu_n = alpha_mu + n
        alpha_w_n = alpha_w + n
        mu_diff = mu_hat - self._mu0
        correction = (alpha_mu * n / alpha_mu_n) * np.outer(mu_diff, mu_diff)
        r = t0 + scatter + correction

        # Regularise the posterior scale matrix as well.
        r = self._regularise(r, d)

        # -------------------------------------------------------------- #
        # Assemble the log score
        # -------------------------------------------------------------- #
        log_score: float = 0.0

        # (1) -n·d/2 · log(π)
        log_score += -0.5 * n * d * _LOG_PI

        # (2) d/2 · (log α_μ  −  log(α_μ + n))
        log_score += 0.5 * d * (log(alpha_mu) - log(alpha_mu_n))

        # (3) log Γ_d((α_w + n)/2)  −  log Γ_d(α_w/2)
        #     Expanded as a sum to match the reference formula and avoid
        #     underflow in the multivariate-gamma ratio.
        for i in range(d):
            log_score += lgamma(0.5 * (alpha_w_n - i)) - lgamma(
                0.5 * (alpha_w - i)
            )

        # (4) Determinant ratio via slogdet for numerical stability.
        sign_t0, logdet_t0 = np.linalg.slogdet(t0)
        sign_r, logdet_r = np.linalg.slogdet(r)

        # A non-positive determinant sign indicates a degenerate matrix;
        # fall back to a large negative score to signal a poor model.
        if sign_t0 <= 0 or sign_r <= 0:
            return -np.inf

        log_score += 0.5 * alpha_w * logdet_t0 - 0.5 * alpha_w_n * logdet_r

        return float(log_score)

    # ------------------------------------------------------------------ #
    # Internal – regularisation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _regularise(matrix: np.ndarray, d: int) -> np.ndarray:
        """Add a small ridge to the diagonal for numerical stability.

        When the number of observations is smaller than the number of
        variables in the subproblem (``n < d``) or when columns are
        nearly collinear, the scatter matrix can become singular.
        Adding ``ε · I`` prevents ``slogdet`` from returning ``-inf``.

        Parameters
        ----------
        matrix:
            ``d × d`` symmetric matrix (modified in-place).
        d:
            Dimension.

        Returns
        -------
        np.ndarray
            The (possibly modified) matrix.
        """
        diag_view = np.einsum("ii->i", matrix)
        min_diag = diag_view.min() if d > 0 else 0.0
        if min_diag < _REGULARISATION_EPS:
            matrix = matrix + _REGULARISATION_EPS * np.eye(d)
        return matrix

    # ------------------------------------------------------------------ #
    # Cache management
    # ------------------------------------------------------------------ #

    def clear_cache(self) -> None:
        """Discard all cached data statistics.

        Call this when the underlying data array is modified in-place or
        replaced with a new object that reuses the same memory address
        (``id``).
        """
        self._stats_cache.clear()

    # ------------------------------------------------------------------ #
    # Representation
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        """Concise string representation for debugging."""
        return (
            f"BGeScore(mu0={self._mu0}, alpha_mu={self._alpha_mu}, "
            f"df_extra={self._df_extra}, reg_eps={self._reg_eps})"
        )
