"""Kernel-based conditional independence test using HSIC.

This module implements a non-parametric conditional independence (CI) test
based on the **Hilbert-Schmidt Independence Criterion (HSIC)**.  HSIC
measures statistical dependence between two random variables by computing
the squared Hilbert-Schmidt norm of the cross-covariance operator in a
reproducing kernel Hilbert space (RKHS).  The key insight is that HSIC is
zero if and only if the two variables are independent (given a
characteristic kernel such as the Gaussian RBF kernel).

The test supports:

* **Unconditional independence** (empty conditioning set) — standard HSIC.
* **Conditional independence** (non-empty conditioning set) — first regress
  X and Y on the conditioning variables via kernel ridge regression (KRR),
  then apply HSIC to the residuals.
* **Permutation-based p-values** — permute the rows/columns of one kernel
  matrix and recompute HSIC to build a null distribution.
* **Gamma approximation p-values** — fit a Gamma distribution to the HSIC
  null distribution using analytical moments of the kernel matrices, which
  is substantially faster than the permutation test for large samples.
* **Mixed data types** — discrete columns are handled with a delta (exact
  match) kernel, while continuous columns use the standard RBF kernel.

References
----------
.. [1] Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Schölkopf, B.,
       & Smola, A. J. (2007).  "A Kernel Statistical Test of
       Independence."  *NeurIPS 20*.
.. [2] Zhang, K., Peters, J., Janzing, D., & Schölkopf, B. (2011).
       "Kernel-based Conditional Independence Test and Application in
       Causal Discovery."  *UAI 2011*.
.. [3] Gretton, A. et al. (2005).  "Measuring Statistical Dependence with
       Hilbert-Schmidt Norms."  *ALT 2005*.
"""

from __future__ import annotations

from typing import FrozenSet, List, Optional, Sequence, Union

import numpy as np
from scipy import stats as sp_stats
from scipy.spatial.distance import pdist, squareform

from causal_qd.ci_tests.ci_base import CITest, CITestResult
from causal_qd.types import DataMatrix, PValue


# ---------------------------------------------------------------------------
# Helper type for column-type annotations used by the mixed-kernel API
# ---------------------------------------------------------------------------

ColumnType = str  # "continuous" or "discrete"


class KernelCITest(CITest):
    """Kernel conditional independence test based on the Hilbert-Schmidt
    Independence Criterion (HSIC).

    Uses RBF (Gaussian) kernels and either a permutation test or a Gamma
    approximation to assess conditional independence.  When the
    conditioning set is non-empty the test uses the **residual HSIC**: it
    first regresses X and Y on S using kernel ridge regression (KRR) and
    then applies HSIC to the residuals.

    Parameters
    ----------
    kernel_width : float or None
        Bandwidth (σ) of the RBF kernel.  When ``None`` (the default) the
        **median heuristic** is used: σ is set to the median of all
        pairwise squared Euclidean distances.
    n_permutations : int
        Number of random permutations used for the permutation-based
        p-value (ignored when ``use_gamma_approx=True``).
    ridge_lambda : float
        L2 regularisation parameter (λ) for kernel ridge regression when
        conditioning on a set S.
    use_gamma_approx : bool
        If ``True``, compute the p-value via a Gamma-distribution
        approximation to the HSIC null distribution instead of the
        (slower) permutation test.  The Gamma approximation uses
        analytical mean and variance of HSIC under the null hypothesis
        derived from kernel-matrix traces.
    seed : int
        Random seed for reproducibility (used for the permutation test
        and for the random number generator).

    Attributes
    ----------
    _kernel_width : float or None
        Stored bandwidth parameter.
    _n_permutations : int
        Number of permutations.
    _ridge_lambda : float
        Ridge regression regularisation strength.
    _use_gamma_approx : bool
        Whether the Gamma approximation is active.
    _seed : int
        Base random seed.

    Examples
    --------
    >>> import numpy as np
    >>> from causal_qd.ci_tests.kernel_ci import KernelCITest
    >>> rng = np.random.default_rng(42)
    >>> data = rng.standard_normal((200, 3))
    >>> ci = KernelCITest(n_permutations=200, seed=0)
    >>> result = ci.test(0, 1, frozenset(), data, alpha=0.05)
    >>> result.is_independent  # independent columns → expect True
    True
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        kernel_width: float | None = None,
        n_permutations: int = 500,
        ridge_lambda: float = 1e-3,
        use_gamma_approx: bool = False,
        seed: int = 0,
    ) -> None:
        self._kernel_width = kernel_width
        self._n_permutations = n_permutations
        self._ridge_lambda = ridge_lambda
        self._use_gamma_approx = use_gamma_approx
        self._seed = seed

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def test(
        self,
        x: int,
        y: int,
        conditioning_set: FrozenSet[int],
        data: DataMatrix,
        alpha: float = 0.05,
    ) -> CITestResult:
        """Perform the kernel-based conditional independence test.

        The test proceeds in three stages:

        1. **Conditioning** — if ``conditioning_set`` is non-empty, kernel
           ridge regression is used to regress X and Y on S and the
           residuals are extracted.
        2. **HSIC computation** — the HSIC statistic is computed from the
           centred RBF kernel matrices of X (or its residuals) and Y (or
           its residuals).
        3. **P-value** — either a permutation test or a Gamma
           approximation yields the p-value.

        Parameters
        ----------
        x : int
            Column index of the first variable in *data*.
        y : int
            Column index of the second variable in *data*.
        conditioning_set : FrozenSet[int]
            Column indices of variables to condition on.  May be empty for
            an unconditional test.
        data : DataMatrix
            Observed data matrix of shape ``(N, p)`` where *N* is the
            number of samples and *p* is the number of variables.
        alpha : float
            Significance level for the independence decision.  The null
            hypothesis of conditional independence is **not** rejected when
            ``p_value >= alpha``.

        Returns
        -------
        CITestResult
            A dataclass with fields ``statistic`` (the HSIC value),
            ``p_value``, ``is_independent`` (``True`` when the null is not
            rejected), and ``conditioning_set``.

        Notes
        -----
        The permutation test constructs the null distribution by randomly
        permuting the rows and columns of one kernel matrix while keeping
        the other fixed.  The p-value is ``(count + 1) / (B + 1)`` where
        *count* is the number of permuted HSIC values ≥ the observed one
        and *B* is ``n_permutations``.  The +1 correction avoids a p-value
        of exactly zero.

        The Gamma approximation avoids the permutation loop entirely by
        matching the first two moments of the HSIC null distribution to a
        Gamma distribution.
        """
        n = data.shape[0]
        rng = np.random.default_rng(self._seed)

        # ----- Stage 1: extract & optionally residualise ----- #
        x_vec = data[:, x].reshape(-1, 1)
        y_vec = data[:, y].reshape(-1, 1)

        if conditioning_set:
            s_cols = sorted(conditioning_set)
            s_mat = data[:, s_cols]
            x_vec = self._kernel_residuals(x_vec, s_mat)
            y_vec = self._kernel_residuals(y_vec, s_mat)

        # ----- Stage 2: kernel matrices & HSIC statistic ----- #
        kx = self._rbf_kernel(x_vec)
        ky = self._rbf_kernel(y_vec)
        observed_hsic = self._hsic(kx, ky, n)

        # ----- Stage 3: p-value computation ----- #
        if self._use_gamma_approx:
            p_value = self._gamma_pvalue(kx, ky, n, observed_hsic)
        else:
            p_value = self._permutation_pvalue(kx, ky, n, observed_hsic, rng)

        return CITestResult(
            statistic=float(observed_hsic),
            p_value=p_value,
            is_independent=(p_value >= alpha),
            conditioning_set=conditioning_set,
        )

    # ------------------------------------------------------------------ #
    #  Kernel construction                                                #
    # ------------------------------------------------------------------ #

    def _rbf_kernel(self, x: np.ndarray) -> np.ndarray:
        """Compute an RBF (Gaussian) kernel matrix for data *x*.

        The Gaussian RBF kernel is defined as:

            k(x_i, x_j) = exp(- ||x_i - x_j||² / (2σ²))

        where σ is either a user-supplied ``kernel_width`` or is chosen
        automatically via the **median heuristic**: σ is set to the median
        of the upper-triangular pairwise squared Euclidean distances.  The
        median heuristic is a simple, data-driven bandwidth selector that
        works well in practice for kernel-based independence tests.

        Parameters
        ----------
        x : np.ndarray
            Data matrix of shape ``(N, d)`` where *N* is the number of
            observations and *d* is the dimensionality.

        Returns
        -------
        np.ndarray
            Kernel matrix of shape ``(N, N)``.

        Notes
        -----
        * When the median of pairwise distances is zero (e.g. all
          observations are identical), a small guard value of ``1e-8`` is
          used to prevent division by zero.
        * The ``squareform(pdist(...))`` idiom is used for efficiency — it
          computes only the N*(N-1)/2 unique distances.
        """
        dists = squareform(pdist(x, metric="sqeuclidean"))

        if self._kernel_width is not None:
            sigma = self._kernel_width
        else:
            # Median heuristic: take the median of the upper-triangular
            # pairwise squared distances.
            triu_indices = np.triu_indices_from(dists, k=1)
            triu_dists = dists[triu_indices]
            sigma = float(np.median(triu_dists)) if triu_dists.size > 0 else 1.0
            # Guard against degenerate case where all points coincide.
            sigma = max(sigma, 1e-8)

        return np.exp(-dists / (2.0 * sigma))

    def _mixed_kernel(
        self,
        data: np.ndarray,
        col_types: Sequence[ColumnType],
    ) -> np.ndarray:
        """Compute a product kernel for mixed continuous/discrete data.

        For columns annotated as ``"continuous"`` the standard RBF
        (Gaussian) kernel is used.  For columns annotated as
        ``"discrete"`` a **delta kernel** (Kronecker delta) is used:

            k_delta(x_i, x_j) = 1  if x_i == x_j
                                 0  otherwise

        The overall kernel is the element-wise product of the per-column
        kernel matrices.  This factored construction preserves the
        characteristic-kernel property needed for HSIC consistency.

        Parameters
        ----------
        data : np.ndarray
            Data matrix of shape ``(N, d)``.  Each column is either
            continuous or discrete as specified by *col_types*.
        col_types : Sequence[ColumnType]
            A sequence of length *d* where each entry is ``"continuous"``
            or ``"discrete"`` indicating the type of the corresponding
            column.

        Returns
        -------
        np.ndarray
            Combined kernel matrix of shape ``(N, N)``.

        Raises
        ------
        ValueError
            If *col_types* length does not match the number of columns in
            *data*, or if an unrecognised column type string is
            encountered.
        """
        n, d = data.shape
        if len(col_types) != d:
            raise ValueError(
                f"col_types length ({len(col_types)}) must match the number "
                f"of columns in data ({d})."
            )

        # Start with an all-ones matrix (multiplicative identity).
        k_combined = np.ones((n, n), dtype=np.float64)

        for j, ctype in enumerate(col_types):
            col = data[:, j].reshape(-1, 1)
            if ctype == "continuous":
                k_j = self._rbf_kernel(col)
            elif ctype == "discrete":
                # Delta kernel: 1 where values match, 0 elsewhere.
                k_j = (col == col.T).astype(np.float64)
            else:
                raise ValueError(
                    f"Unrecognised column type '{ctype}' for column {j}.  "
                    f"Expected 'continuous' or 'discrete'."
                )
            k_combined *= k_j

        return k_combined

    # ------------------------------------------------------------------ #
    #  HSIC computation                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _hsic(kx: np.ndarray, ky: np.ndarray, n: int) -> float:
        """Compute the biased empirical HSIC estimator.

        The biased HSIC estimator [1]_ is given by:

            HSIC(X, Y) = trace(Kx_c · Ky_c) / (n − 1)²

        where Kx_c = H · Kx · H and Ky_c = H · Ky · H are the centred
        kernel matrices and H = I − (1/n) · 1·1ᵀ is the centering matrix.

        Parameters
        ----------
        kx : np.ndarray
            Kernel matrix for X of shape ``(n, n)``.
        ky : np.ndarray
            Kernel matrix for Y of shape ``(n, n)``.
        n : int
            Number of observations.

        Returns
        -------
        float
            The HSIC statistic value.

        References
        ----------
        .. [1] Gretton et al. (2005), "Measuring Statistical Dependence
               with Hilbert-Schmidt Norms", ALT.
        """
        h = np.eye(n) - np.ones((n, n)) / n
        kxc = h @ kx @ h
        kyc = h @ ky @ h
        return float(np.trace(kxc @ kyc) / ((n - 1) ** 2))

    # ------------------------------------------------------------------ #
    #  P-value computation                                                #
    # ------------------------------------------------------------------ #

    def _permutation_pvalue(
        self,
        kx: np.ndarray,
        ky: np.ndarray,
        n: int,
        observed_hsic: float,
        rng: np.random.Generator,
    ) -> PValue:
        """Compute the p-value via a permutation test.

        Under the null hypothesis of independence, the association between
        X and Y is spurious, so any permutation of the Y indices should
        yield an HSIC value drawn from the null distribution.  We shuffle
        both rows and columns of the Y kernel matrix to preserve its
        internal structure.

        The p-value is computed with a +1/+1 correction (Phipson &
        Smyth 2010) to avoid an exact zero p-value:

            p = (count + 1) / (B + 1)

        Parameters
        ----------
        kx : np.ndarray
            Kernel matrix for X, shape ``(n, n)``.
        ky : np.ndarray
            Kernel matrix for Y, shape ``(n, n)``.
        n : int
            Number of observations.
        observed_hsic : float
            The HSIC value computed from the original (unpermuted) data.
        rng : np.random.Generator
            Random number generator instance.

        Returns
        -------
        PValue
            Permutation p-value in the interval ``[1/(B+1), 1]``.
        """
        count = 0
        for _ in range(self._n_permutations):
            perm = rng.permutation(n)
            ky_perm = ky[np.ix_(perm, perm)]
            perm_hsic = self._hsic(kx, ky_perm, n)
            if perm_hsic >= observed_hsic:
                count += 1

        return float((count + 1) / (self._n_permutations + 1))

    @staticmethod
    def _gamma_pvalue(
        kx: np.ndarray,
        ky: np.ndarray,
        n: int,
        observed_hsic: float,
    ) -> PValue:
        """Compute the p-value via a Gamma-distribution approximation.

        Under the null hypothesis H₀: X ⊥ Y, the distribution of the
        HSIC statistic can be approximated by a Gamma distribution whose
        shape and scale parameters are determined by the mean and variance
        of HSIC under H₀.

        Following Gretton et al. (2005, §4) the null moments are computed
        from the kernel-matrix traces and element sums:

        * **Mean** (μ):
              μ = [1·Kx·1 · 1·Ky·1 / n² − tr(Kx)·tr(Ky) / n] / [n·(n−1)]

          Simplified: the expected HSIC under the null is a function of
          the overall kernel sums and traces, scaled by sample size.

        * **Variance** (σ²):
              Computed from the second moment of centred kernel matrices
              under permutation.  We use the approximation:

              σ² ≈ 2·(n−4)·(n−5) / (n·(n−1)·(n−2)·(n−3)) · S

          where S involves squared Frobenius norms of centred kernels.

        Given μ and σ², the Gamma parameters are:

            shape = μ² / σ²
            scale = σ² / μ

        The p-value is then ``1 − Gamma.cdf(observed_hsic)``.

        Parameters
        ----------
        kx : np.ndarray
            Kernel matrix for X, shape ``(n, n)``.
        ky : np.ndarray
            Kernel matrix for Y, shape ``(n, n)``.
        n : int
            Sample size.
        observed_hsic : float
            Observed HSIC statistic.

        Returns
        -------
        PValue
            Approximate p-value from the Gamma CDF.

        Notes
        -----
        For very small samples the variance estimate can become
        non-positive; in that case the function falls back to a p-value
        of 1.0 (cannot reject the null).
        """
        # Centre the kernel matrices.
        h = np.eye(n) - np.ones((n, n)) / n
        kxc = h @ kx @ h
        kyc = h @ ky @ h

        # ----- Null mean of HSIC ----- #
        # Under H₀ the expected HSIC simplifies to a function of the
        # kernel-matrix element sums and traces.
        #   E[HSIC] ≈ (1/n) * [ (1ᵀKx1)(1ᵀKy1)/n² - tr(Kx)tr(Ky)/n ]
        # We use the centred-matrix formulation for numerical stability.
        one_kx_one = float(kx.sum())  # 1ᵀ Kx 1
        one_ky_one = float(ky.sum())  # 1ᵀ Ky 1
        tr_kx = float(np.trace(kx))
        tr_ky = float(np.trace(ky))

        # Mean under null (biased estimator)
        mu = (one_kx_one * one_ky_one / (n * n) - tr_kx * tr_ky / n) / (
            n * (n - 1)
        )
        mu = max(mu, 1e-12)  # guard against non-positive mean

        # ----- Null variance of HSIC ----- #
        # Approximate variance using squared Frobenius norms of centred
        # kernel matrices.  This follows the large-sample expansion in
        # Gretton et al. (2005).
        frob_kxc_sq = float((kxc * kxc).sum())  # ||Kx_c||_F²
        frob_kyc_sq = float((kyc * kyc).sum())  # ||Ky_c||_F²

        # Leading term of the variance under H₀
        denom = n * (n - 1) * (n - 2) * (n - 3) if n > 4 else max(n, 1)
        coeff = 2.0 * max(n - 4, 1) * max(n - 5, 1)
        var = coeff / denom * frob_kxc_sq * frob_kyc_sq / ((n - 1) ** 4)

        if var <= 0 or not np.isfinite(var):
            return 1.0

        # ----- Gamma parameters ----- #
        shape = mu * mu / var
        scale = var / mu

        # Guard against degenerate Gamma parameters.
        if shape <= 0 or scale <= 0 or not (np.isfinite(shape) and np.isfinite(scale)):
            return 1.0

        p_value = float(1.0 - sp_stats.gamma.cdf(observed_hsic, a=shape, scale=scale))

        # Clamp into [0, 1] for numerical safety.
        return max(0.0, min(1.0, p_value))

    # ------------------------------------------------------------------ #
    #  Kernel ridge regression for conditioning                           #
    # ------------------------------------------------------------------ #

    def _kernel_residuals(
        self, target: np.ndarray, conditioning: np.ndarray
    ) -> np.ndarray:
        """Regress *target* on *conditioning* via kernel ridge regression
        and return the residuals.

        Kernel Ridge Regression (KRR) fits the model:

            f(s) = Kₛ · (Kₛ + λI)⁻¹ · y

        where Kₛ is the kernel matrix of the conditioning variables, λ is
        the ridge regularisation parameter, and y is the target vector.
        The residuals are simply ``target − f(s)``.

        This is used in the conditional independence test to remove the
        influence of the conditioning set S from X and Y before computing
        HSIC.

        Parameters
        ----------
        target : np.ndarray
            Column vector(s) of shape ``(N, d_target)`` to regress.
        conditioning : np.ndarray
            Data matrix of shape ``(N, d_cond)`` for the conditioning
            variables.

        Returns
        -------
        np.ndarray
            Residuals of the same shape as *target*, i.e.
            ``target − predicted``.

        Notes
        -----
        The linear system ``(Kₛ + λI)α = target`` is solved with
        ``np.linalg.solve`` rather than an explicit inverse for better
        numerical stability and efficiency.
        """
        ks = self._rbf_kernel(conditioning)
        n = ks.shape[0]
        alpha = np.linalg.solve(
            ks + self._ridge_lambda * np.eye(n), target
        )
        predicted = ks @ alpha
        return target - predicted

    # ------------------------------------------------------------------ #
    #  Utility / representation                                           #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:  # pragma: no cover
        """Return a human-readable representation of the test object."""
        return (
            f"KernelCITest(kernel_width={self._kernel_width!r}, "
            f"n_permutations={self._n_permutations}, "
            f"ridge_lambda={self._ridge_lambda}, "
            f"use_gamma_approx={self._use_gamma_approx}, "
            f"seed={self._seed})"
        )
