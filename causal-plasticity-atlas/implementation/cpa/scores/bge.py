"""BGe (Bayesian Gaussian equivalent) score.

Implements the Bayesian Gaussian equivalent score for linear-Gaussian
structural equation models with a Normal-Wishart prior.

The BGe score was introduced by Geiger & Heckerman (2002) for scoring
Bayesian network structures under a linear-Gaussian assumption.  It is
*score equivalent* (assigns the same score to Markov-equivalent DAGs).

For a node X_i with parent set Pa_i of size |Pa_i| = s, define
l = s + 1 (the family size).  The local BGe score is:

    BGe(i, Pa_i) = log c(l) + (alpha_w/2) * log|T_0[family]|
                   - ((alpha_w + n)/2) * log|T_n[family]|
                   - (alpha_w/2) * log|T_0[parents]|
                   + ((alpha_w + n)/2) * log|T_n[parents]|

where c(l) collects normalising constants involving multivariate
gamma functions, and T_0, T_n are the prior / posterior scale matrices.

References
----------
.. [1] Geiger D, Heckerman D. Parameter priors for directed acyclic
       graphical models and the characterization of several probability
       distributions.  Ann. Stat. 30(5), 2002.
.. [2] Kuipers J, Moffa G, Heckerman D. Addendum on the scoring of
       Gaussian DAG models.  Ann. Stat. 42(4), 2014.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Multivariate log-gamma helper
# ---------------------------------------------------------------------------

def _log_multivariate_gamma(a: float, p: int) -> float:
    r"""Compute the log of the multivariate gamma function.

    .. math::
        \Gamma_p(a) = \pi^{p(p-1)/4} \prod_{j=1}^{p}
                       \Gamma\!\bigl(a + (1-j)/2\bigr)

    Parameters
    ----------
    a : float
        Argument (must satisfy ``a > (p-1)/2``).
    p : int
        Dimensionality.

    Returns
    -------
    float
    """
    if p <= 0:
        return 0.0
    result = 0.25 * p * (p - 1) * math.log(math.pi)
    for j in range(1, p + 1):
        result += gammaln(a + 0.5 * (1 - j))
    return float(result)


def _log_gamma_ratio(alpha_post: float, alpha_prior: float, p: int) -> float:
    """Return log Gamma_p(alpha_post/2) - log Gamma_p(alpha_prior/2)."""
    return _log_multivariate_gamma(alpha_post / 2.0, p) \
         - _log_multivariate_gamma(alpha_prior / 2.0, p)


def _log_det_spd(M: NDArray) -> float:
    """Log-determinant of a symmetric positive-definite matrix.

    Uses Cholesky decomposition for numerical stability.  Falls back to
    ``np.linalg.slogdet`` if Cholesky fails.
    """
    try:
        L = np.linalg.cholesky(M)
        return float(2.0 * np.sum(np.log(np.diag(L))))
    except np.linalg.LinAlgError:
        sign, logdet = np.linalg.slogdet(M)
        if sign <= 0:
            return -1e30  # degenerate
        return float(logdet)


# ---------------------------------------------------------------------------
# BGeScore
# ---------------------------------------------------------------------------

class BGeScore:
    """Bayesian Gaussian equivalent (BGe) score.

    Parameters
    ----------
    data : NDArray
        Observation matrix of shape ``(n_samples, n_variables)``.
    alpha_mu : float
        Prior precision scaling for the mean (``alpha_mu > 0``).
        Controls how strongly the prior mean influences the posterior.
    alpha_w : Optional[int]
        Degrees of freedom for the Wishart prior.  Must be
        > ``n_variables + 1``.  Defaults to ``n_variables + 2``.
    prior_mean : Optional[NDArray]
        Prior mean vector of length ``n_variables``.  Defaults to the
        column-wise sample mean.
    """

    def __init__(
        self,
        data: NDArray,
        alpha_mu: float = 1.0,
        alpha_w: Optional[int] = None,
        prior_mean: Optional[NDArray] = None,
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.n_samples, self.n_variables = self.data.shape
        n, p = self.n_samples, self.n_variables

        if alpha_mu <= 0:
            raise ValueError("alpha_mu must be positive")
        self.alpha_mu = alpha_mu

        self.alpha_w = alpha_w if alpha_w is not None else p + 2
        if self.alpha_w <= p + 1:
            raise ValueError(
                f"alpha_w must be > n_variables + 1 = {p + 1}, "
                f"got {self.alpha_w}"
            )

        if prior_mean is not None:
            self.prior_mean = np.asarray(prior_mean, dtype=np.float64)
        else:
            self.prior_mean = np.zeros(p, dtype=np.float64)

        # Pre-compute the prior scale matrix T_0 (p x p) and
        # the posterior scale matrix T_n (p x p).
        self._sample_mean = self.data.mean(axis=0)
        self._T0 = self._compute_prior_scale_matrix()
        self._Tn = self._compute_posterior_scale_matrix()

    # ---- Prior / posterior matrices ------------------------------------

    def _compute_prior_scale_matrix(self) -> NDArray:
        """Compute the prior scale matrix T_0.

        T_0 = (alpha_mu * (alpha_w - p - 1)) / (alpha_mu + 1) * I_p

        This is the standard default choice ensuring that T_0 is a
        proper prior scale for the Wishart distribution.
        """
        p = self.n_variables
        t0_scale = (self.alpha_mu * (self.alpha_w - p - 1)) / (self.alpha_mu + 1.0)
        # Use identity scaled by t0_scale
        T0 = t0_scale * np.eye(p, dtype=np.float64)
        return T0

    def _compute_posterior_scale_matrix(self) -> NDArray:
        """Compute the posterior scale matrix T_n.

        T_n = T_0 + S + (alpha_mu * n) / (alpha_mu + n) *
              (sample_mean - prior_mean)(sample_mean - prior_mean)^T

        where S is the scatter matrix of the data.
        """
        n = self.n_samples
        diff = self._sample_mean - self.prior_mean
        S = self._scatter_matrix(self.data)
        correction = (self.alpha_mu * n) / (self.alpha_mu + n)
        Tn = self._T0 + S + correction * np.outer(diff, diff)
        return Tn

    @staticmethod
    def _scatter_matrix(data: NDArray) -> NDArray:
        """Return the scatter matrix sum_i (x_i - x_bar)(x_i - x_bar)^T."""
        centred = data - data.mean(axis=0)
        return centred.T @ centred

    # ---- Sub-matrix extraction -----------------------------------------

    def _submatrix(self, M: NDArray, indices: Sequence[int]) -> NDArray:
        """Extract the sub-matrix of M for the given variable indices."""
        idx = np.array(indices, dtype=int)
        return M[np.ix_(idx, idx)]

    # ---- BGe local score -----------------------------------------------

    def local_score(self, node: int, parents: Sequence[int]) -> float:
        """Return the local BGe score for *node* given *parents*.

        This is the log marginal likelihood contribution of the family
        (node, parents) under the BGe model.
        """
        parents = list(parents)
        self._validate(node, parents)
        return self.log_marginal_likelihood(node, parents)

    def log_marginal_likelihood(
        self, node: int, parents: Sequence[int]
    ) -> float:
        """Return the log marginal likelihood for *node* given *parents*.

        Implements the Geiger-Heckerman BGe formula:

        log p(D_family) = c(l, s) + (alpha_w + n - p + s) / 2 * log|T_n[Pa]|
                          - (alpha_w + n - p + l) / 2 * log|T_n[family]|
                          + (alpha_w - p + l) / 2 * log|T_0[family]|
                          - (alpha_w - p + s) / 2 * log|T_0[Pa]|

        where l = |Pa| + 1, s = |Pa|, and c(l, s) collects the gamma
        function terms.
        """
        parents = list(parents)
        s = len(parents)
        l = s + 1
        n = self.n_samples
        p = self.n_variables
        alpha_w = self.alpha_w

        family = parents + [node]

        # Degrees of freedom parameters
        alpha_w_post = alpha_w + n

        # Log-normalising constant c(l, s)
        log_c = self._log_normalising_constant(l, s)

        # Sub-matrices of T_0 and T_n
        if s > 0:
            T0_pa = self._submatrix(self._T0, parents)
            Tn_pa = self._submatrix(self._Tn, parents)
            log_det_T0_pa = _log_det_spd(T0_pa)
            log_det_Tn_pa = _log_det_spd(Tn_pa)
        else:
            log_det_T0_pa = 0.0
            log_det_Tn_pa = 0.0

        T0_fam = self._submatrix(self._T0, family)
        Tn_fam = self._submatrix(self._Tn, family)
        log_det_T0_fam = _log_det_spd(T0_fam)
        log_det_Tn_fam = _log_det_spd(Tn_fam)

        # BGe score components
        score = log_c
        score += 0.5 * (alpha_w - p + l) * log_det_T0_fam
        score -= 0.5 * (alpha_w_post - p + l) * log_det_Tn_fam
        if s > 0:
            score -= 0.5 * (alpha_w - p + s) * log_det_T0_pa
            score += 0.5 * (alpha_w_post - p + s) * log_det_Tn_pa

        return float(score)

    def _log_normalising_constant(self, l: int, s: int) -> float:
        """Compute the log normalising constant c(l, s).

        The BGe local score is  log p(D_{family}) - log p(D_{parents}).
        The normalising constant collects the gamma-function and pi terms
        from this difference:

            c(l, s) = log Gamma_l((alpha_w + n - p + l)/2)
                    - log Gamma_l((alpha_w - p + l)/2)
                    - [log Gamma_s((alpha_w + n - p + s)/2)
                       - log Gamma_s((alpha_w - p + s)/2)]   (if s > 0)
                    - (n/2) * log(pi)
                    + (1/2) * log(alpha_mu / (alpha_mu + n))

        where Gamma_d denotes the *d*-dimensional multivariate gamma.
        """
        n = self.n_samples
        p = self.n_variables
        alpha_w = self.alpha_w

        # Multivariate gamma ratio for the family (dimension l)
        log_c = _log_multivariate_gamma(0.5 * (alpha_w + n - p + l), l) \
              - _log_multivariate_gamma(0.5 * (alpha_w - p + l), l)

        # Subtract parent-only gamma ratio (dimension s) if parents exist
        if s > 0:
            log_c -= _log_multivariate_gamma(0.5 * (alpha_w + n - p + s), s) \
                   - _log_multivariate_gamma(0.5 * (alpha_w - p + s), s)

        # Data term
        log_c -= 0.5 * n * math.log(math.pi)

        # Prior precision scaling
        log_c += 0.5 * math.log(self.alpha_mu / (self.alpha_mu + n))

        return float(log_c)

    # ---- Full DAG score ------------------------------------------------

    def score_dag(self, adj_matrix: NDArray) -> float:
        """Return the total BGe score of a DAG.

        ``adj_matrix[i, j] != 0`` indicates i -> j.
        """
        adj = np.asarray(adj_matrix)
        total = 0.0
        for j in range(self.n_variables):
            parents = list(np.nonzero(adj[:, j])[0])
            total += self.local_score(j, parents)
        return total

    # ---- Posterior hyperparameters (public API) -------------------------

    def _compute_am(self, parents: Sequence[int]) -> NDArray:
        """Return the posterior precision matrix for a parent set.

        This is T_n restricted to the parent variables, inverted.
        """
        parents = list(parents)
        if len(parents) == 0:
            return np.array([[]], dtype=np.float64)
        Tn_pa = self._submatrix(self._Tn, parents)
        try:
            return np.linalg.inv(Tn_pa)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(Tn_pa)

    def posterior_parameters(
        self, node: int, parents: Sequence[int]
    ) -> Dict[str, object]:
        """Return posterior hyper-parameters for the family (node, parents).

        Returns
        -------
        dict with keys:
            'alpha_w_post': posterior degrees of freedom
            'mu_post': posterior mean vector for the family
            'T_n_family': posterior scale matrix for the family
        """
        parents = list(parents)
        family = parents + [node]
        n = self.n_samples
        alpha_w_post = self.alpha_w + n
        mu_post = (self.alpha_mu * self.prior_mean[family]
                   + n * self._sample_mean[family]) / (self.alpha_mu + n)
        T_n_fam = self._submatrix(self._Tn, family)
        return {
            "alpha_w_post": alpha_w_post,
            "mu_post": mu_post,
            "T_n_family": T_n_fam,
        }

    def _compute_t_matrix(
        self, data: NDArray, parents: Sequence[int], node: int
    ) -> NDArray:
        """Compute the T matrix (prior or posterior scale) for (node, parents).

        Provided for external use; internally the class pre-computes
        T_0 and T_n for the full variable set.
        """
        family = list(parents) + [node]
        S = self._scatter_matrix(data[:, family])
        p_fam = len(family)
        T0_fam = self._submatrix(self._T0, family)
        n = data.shape[0]
        diff = data[:, family].mean(axis=0) - self.prior_mean[family]
        correction = (self.alpha_mu * n) / (self.alpha_mu + n)
        T = T0_fam + S + correction * np.outer(diff, diff)
        return T

    # ---- Validation ----------------------------------------------------

    def _validate(self, node: int, parents: Sequence[int]) -> None:
        if not 0 <= node < self.n_variables:
            raise ValueError(f"node {node} out of range")
        for p in parents:
            if not 0 <= p < self.n_variables:
                raise ValueError(f"parent {p} out of range")
            if p == node:
                raise ValueError("node cannot be its own parent")

    # ---- Score comparison helpers --------------------------------------

    def score_edge_addition(
        self, node: int, current_parents: Sequence[int], new_parent: int
    ) -> float:
        """Score change when adding *new_parent*."""
        old = self.local_score(node, current_parents)
        new = self.local_score(node, list(current_parents) + [new_parent])
        return new - old

    def score_edge_removal(
        self, node: int, current_parents: Sequence[int], removed: int
    ) -> float:
        """Score change when removing *removed*."""
        old = self.local_score(node, current_parents)
        new = self.local_score(node, [p for p in current_parents if p != removed])
        return new - old

    def __repr__(self) -> str:
        return (
            f"BGeScore(n={self.n_samples}, p={self.n_variables}, "
            f"alpha_mu={self.alpha_mu}, alpha_w={self.alpha_w})"
        )
