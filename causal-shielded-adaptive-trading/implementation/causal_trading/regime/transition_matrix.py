"""
Bayesian transition matrix estimation and spectral analysis.

Provides Dirichlet-posterior inference over Markov chain transition
matrices, sticky priors, spectral decomposition for mixing-time
analysis, and credible sets for transition probabilities.

References
----------
- Robert, C. P. & Casella, G. (2004). Monte Carlo Statistical Methods.
- Levin, D. A., Peres, Y., & Wilmer, E. L. (2009).
  Markov Chains and Mixing Times. AMS.
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats, linalg
from scipy.special import gammaln, digamma, logsumexp


class TransitionMatrixEstimator:
    """Bayesian estimation and analysis of Markov chain transition matrices.

    Combines Dirichlet-conjugate posterior inference with spectral
    analysis tools for understanding mixing behaviour and stationarity.

    Parameters
    ----------
    n_states : int
        Number of states in the Markov chain.
    prior_alpha : float
        Symmetric Dirichlet concentration parameter for the prior
        on each row.  Values < 1 encourage sparsity; values > 1
        encourage uniformity.
    sticky_kappa : float
        Extra prior mass on the diagonal (self-transitions).
        Set to 0 for a standard symmetric Dirichlet prior.
    n_posterior_samples : int
        Number of posterior samples to draw for uncertainty quantification.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_states: int = 5,
        prior_alpha: float = 1.0,
        sticky_kappa: float = 0.0,
        n_posterior_samples: int = 2000,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_states = n_states
        self.prior_alpha = prior_alpha
        self.sticky_kappa = sticky_kappa
        self.n_posterior_samples = n_posterior_samples
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)
        self._count_matrix: Optional[NDArray] = None
        self._posterior_alpha: Optional[NDArray] = None
        self._A_map: Optional[NDArray] = None
        self._A_samples: Optional[NDArray] = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Prior construction
    # ------------------------------------------------------------------

    def _build_prior(self) -> NDArray:
        """Construct the Dirichlet prior matrix (K, K).

        Returns a matrix where entry (i, j) is the prior concentration
        for the transition i → j.  The diagonal gets an extra kappa.
        """
        K = self.n_states
        alpha_mat = np.full((K, K), self.prior_alpha)
        alpha_mat += self.sticky_kappa * np.eye(K)
        return alpha_mat

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, states: NDArray) -> "TransitionMatrixEstimator":
        """Estimate the transition matrix from an observed state sequence.

        Parameters
        ----------
        states : (T,) int array
            Observed state sequence with values in {0, ..., n_states-1}.

        Returns
        -------
        self
        """
        states = np.asarray(states, dtype=np.int64).ravel()
        K = self.n_states

        # Count transitions
        count_mat = np.zeros((K, K), dtype=np.float64)
        for t in range(len(states) - 1):
            i, j = int(states[t]), int(states[t + 1])
            if 0 <= i < K and 0 <= j < K:
                count_mat[i, j] += 1.0
        self._count_matrix = count_mat

        # Posterior = prior + counts
        prior = self._build_prior()
        self._posterior_alpha = prior + count_mat

        # MAP estimate (mode of Dirichlet)
        self._A_map = self._dirichlet_mode(self._posterior_alpha)

        # Draw posterior samples
        self._A_samples = self._sample_posterior(self.n_posterior_samples)
        self._is_fitted = True
        return self

    def fit_from_counts(self, count_matrix: NDArray) -> "TransitionMatrixEstimator":
        """Fit from a pre-computed transition count matrix.

        Parameters
        ----------
        count_matrix : (K, K) array of transition counts

        Returns
        -------
        self
        """
        count_matrix = np.asarray(count_matrix, dtype=np.float64)
        K = count_matrix.shape[0]
        if K != self.n_states:
            self.n_states = K
        self._count_matrix = count_matrix
        prior = self._build_prior()
        self._posterior_alpha = prior + count_matrix
        self._A_map = self._dirichlet_mode(self._posterior_alpha)
        self._A_samples = self._sample_posterior(self.n_posterior_samples)
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Dirichlet helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dirichlet_mode(alpha: NDArray) -> NDArray:
        """Compute the mode (MAP) of row-wise Dirichlet distributions.

        mode_j = (alpha_j - 1) / (sum(alpha) - K)  for alpha_j > 1
        Falls back to mean when any alpha_j <= 1.
        """
        K = alpha.shape[1]
        A = np.zeros_like(alpha)
        for i in range(alpha.shape[0]):
            row = alpha[i]
            if np.all(row > 1.0):
                mode = (row - 1.0) / (row.sum() - K)
            else:
                mode = row / row.sum()
            A[i] = mode
        return A

    def _sample_posterior(self, n_samples: int) -> NDArray:
        """Draw n_samples transition matrices from the Dirichlet posterior.

        Returns
        -------
        samples : (n_samples, K, K)
        """
        K = self.n_states
        samples = np.zeros((n_samples, K, K))
        for s in range(n_samples):
            for i in range(K):
                alpha_row = np.clip(self._posterior_alpha[i], 1e-300, None)
                samples[s, i] = self._rng.dirichlet(alpha_row)
        return samples

    # ------------------------------------------------------------------
    # MAP and posterior mean
    # ------------------------------------------------------------------

    def get_map_estimate(self) -> NDArray:
        """Return the MAP transition matrix."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        return self._A_map.copy()

    def get_posterior_mean(self) -> NDArray:
        """Return the posterior mean transition matrix (Dirichlet mean)."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        A = self._posterior_alpha.copy()
        row_sums = A.sum(axis=1, keepdims=True)
        return A / row_sums

    # ------------------------------------------------------------------
    # Credible intervals
    # ------------------------------------------------------------------

    def credible_intervals(
        self, level: float = 0.95
    ) -> Tuple[NDArray, NDArray]:
        """Point-wise credible intervals for each transition probability.

        Parameters
        ----------
        level : float
            Credible level (e.g. 0.95 for 95%).

        Returns
        -------
        lower : (K, K) lower bounds
        upper : (K, K) upper bounds
        """
        if self._A_samples is None:
            raise RuntimeError("Call fit() first.")
        lo = (1.0 - level) / 2.0
        hi = 1.0 - lo
        lower = np.quantile(self._A_samples, lo, axis=0)
        upper = np.quantile(self._A_samples, hi, axis=0)
        return lower, upper

    def posterior_variance(self) -> NDArray:
        """Element-wise posterior variance of the transition matrix."""
        if self._A_samples is None:
            raise RuntimeError("Call fit() first.")
        return np.var(self._A_samples, axis=0)

    def credible_set_volume(self, level: float = 0.95) -> float:
        """Approximate volume of the credible set.

        Computed as the product of interval widths across all entries.
        """
        lower, upper = self.credible_intervals(level)
        widths = upper - lower
        # Product in log-space for stability
        return float(np.exp(np.sum(np.log(widths + 1e-300))))

    # ------------------------------------------------------------------
    # Marginal likelihood
    # ------------------------------------------------------------------

    def log_marginal_likelihood(self) -> float:
        """Compute the log marginal likelihood log P(data | model).

        Uses the Dirichlet-Multinomial conjugacy:
        log P(N) = sum_i [ log B(alpha_i + N_i) - log B(alpha_i) ]
        where B is the multivariate Beta function.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        prior = self._build_prior()
        posterior = self._posterior_alpha

        def log_multi_beta(alpha: NDArray) -> float:
            return float(np.sum(gammaln(alpha)) - gammaln(np.sum(alpha)))

        ll = 0.0
        for i in range(self.n_states):
            ll += log_multi_beta(posterior[i]) - log_multi_beta(prior[i])
        return ll

    # ------------------------------------------------------------------
    # Spectral analysis
    # ------------------------------------------------------------------

    def eigendecomposition(
        self, A: Optional[NDArray] = None
    ) -> Tuple[NDArray, NDArray]:
        """Eigendecomposition of the transition matrix.

        Parameters
        ----------
        A : (K, K) transition matrix (default: MAP estimate)

        Returns
        -------
        eigenvalues : (K,) complex array sorted by magnitude (descending)
        eigenvectors : (K, K) right eigenvectors as columns
        """
        if A is None:
            A = self.get_map_estimate()
        eigvals, eigvecs = np.linalg.eig(A.T)
        order = np.argsort(-np.abs(eigvals))
        return eigvals[order], eigvecs[:, order]

    def stationary_distribution(self, A: Optional[NDArray] = None) -> NDArray:
        """Compute the stationary distribution pi such that pi A = pi.

        Uses the left eigenvector corresponding to eigenvalue 1.

        Parameters
        ----------
        A : (K, K) transition matrix (default: MAP estimate)

        Returns
        -------
        pi : (K,) stationary distribution
        """
        if A is None:
            A = self.get_map_estimate()
        K = A.shape[0]
        eigvals, eigvecs = np.linalg.eig(A.T)
        # Find eigenvalue closest to 1
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        pi = np.real(eigvecs[:, idx])
        pi = np.abs(pi)
        pi /= pi.sum()
        return pi

    def mixing_time(
        self, epsilon: float = 0.25, A: Optional[NDArray] = None
    ) -> float:
        """Estimate the mixing time of the Markov chain.

        The mixing time t_mix(epsilon) is the smallest t such that
        max_x || P^t(x, ·) - pi ||_TV <= epsilon.

        Uses the spectral gap: t_mix ≈ (1 / gap) * log(1 / epsilon).

        Parameters
        ----------
        epsilon : float
            Total-variation tolerance.
        A : (K, K) transition matrix (default: MAP estimate)

        Returns
        -------
        t_mix : float
            Estimated mixing time (continuous).
        """
        if A is None:
            A = self.get_map_estimate()
        eigvals = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
        if len(eigvals) < 2:
            return 1.0
        lambda2 = eigvals[1]
        gap = 1.0 - lambda2
        if gap <= 1e-12:
            return float("inf")
        return np.log(1.0 / epsilon) / gap

    def spectral_gap(self, A: Optional[NDArray] = None) -> float:
        """Spectral gap = 1 - |lambda_2|."""
        if A is None:
            A = self.get_map_estimate()
        eigvals = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
        if len(eigvals) < 2:
            return 1.0
        return float(1.0 - eigvals[1])

    def is_ergodic(self, A: Optional[NDArray] = None) -> bool:
        """Check whether the chain is ergodic (irreducible and aperiodic).

        A sufficient condition: A^K has all positive entries for K = n_states.
        """
        if A is None:
            A = self.get_map_estimate()
        K = A.shape[0]
        Ak = np.linalg.matrix_power(A, K)
        return bool(np.all(Ak > 1e-12))

    def is_reversible(self, A: Optional[NDArray] = None, tol: float = 1e-8) -> bool:
        """Check detailed balance: pi_i A_ij = pi_j A_ji."""
        if A is None:
            A = self.get_map_estimate()
        pi = self.stationary_distribution(A)
        K = A.shape[0]
        for i in range(K):
            for j in range(i + 1, K):
                if abs(pi[i] * A[i, j] - pi[j] * A[j, i]) > tol:
                    return False
        return True

    # ------------------------------------------------------------------
    # Posterior predictive
    # ------------------------------------------------------------------

    def posterior_predictive_stationary(self) -> Tuple[NDArray, NDArray]:
        """Posterior distribution over the stationary distribution.

        Returns mean and standard deviation of each pi_k across
        posterior samples.
        """
        if self._A_samples is None:
            raise RuntimeError("Call fit() first.")
        pis = np.zeros((self._A_samples.shape[0], self.n_states))
        for s in range(self._A_samples.shape[0]):
            pis[s] = self.stationary_distribution(self._A_samples[s])
        return pis.mean(axis=0), pis.std(axis=0)

    def posterior_mixing_times(self, epsilon: float = 0.25) -> NDArray:
        """Posterior distribution over mixing times."""
        if self._A_samples is None:
            raise RuntimeError("Call fit() first.")
        times = np.array([
            self.mixing_time(epsilon, self._A_samples[s])
            for s in range(self._A_samples.shape[0])
        ])
        return times

    # ------------------------------------------------------------------
    # Entropy and information measures
    # ------------------------------------------------------------------

    def transition_entropy(self, A: Optional[NDArray] = None) -> NDArray:
        """Shannon entropy of each row of the transition matrix.

        Returns
        -------
        H : (K,) array of per-row entropies
        """
        if A is None:
            A = self.get_map_estimate()
        return -np.sum(A * np.log(A + 1e-300), axis=1)

    def mutual_information_rate(self, A: Optional[NDArray] = None) -> float:
        """Mutual information rate I(Z_t; Z_{t+1}) of the chain.

        I = H(pi) - sum_i pi_i H(A_i)
        """
        if A is None:
            A = self.get_map_estimate()
        pi = self.stationary_distribution(A)
        H_pi = -np.sum(pi * np.log(pi + 1e-300))
        H_rows = self.transition_entropy(A)
        cond_entropy = np.sum(pi * H_rows)
        return float(H_pi - cond_entropy)

    def kl_divergence_rows(self, A: NDArray, B: NDArray) -> NDArray:
        """KL divergence from row A_i to row B_i for each state.

        Returns
        -------
        kl : (K,) array of KL(A_i || B_i)
        """
        A = np.clip(A, 1e-300, None)
        B = np.clip(B, 1e-300, None)
        return np.sum(A * (np.log(A) - np.log(B)), axis=1)

    # ------------------------------------------------------------------
    # Summary and display
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a summary dictionary."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        A = self.get_map_estimate()
        pi_mean, pi_std = self.posterior_predictive_stationary()
        return {
            "n_states": self.n_states,
            "prior_alpha": self.prior_alpha,
            "sticky_kappa": self.sticky_kappa,
            "map_estimate": A,
            "stationary_distribution": pi_mean,
            "stationary_std": pi_std,
            "spectral_gap": self.spectral_gap(A),
            "mixing_time_0.25": self.mixing_time(0.25, A),
            "is_ergodic": self.is_ergodic(A),
            "is_reversible": self.is_reversible(A),
            "log_marginal_likelihood": self.log_marginal_likelihood(),
            "transition_entropy": self.transition_entropy(A),
        }

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"TransitionMatrixEstimator(n_states={self.n_states}, "
            f"prior_alpha={self.prior_alpha}, "
            f"sticky_kappa={self.sticky_kappa}, {status})"
        )
