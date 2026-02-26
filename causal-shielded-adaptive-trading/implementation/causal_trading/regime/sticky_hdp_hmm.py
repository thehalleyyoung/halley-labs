"""
Sticky Hierarchical Dirichlet Process Hidden Markov Model (Sticky HDP-HMM).

Implements the Bayesian nonparametric HMM of Fox et al. (2011) with the
sticky extension that biases the model toward self-transitions, and the
beam sampling algorithm of Van Gael et al. (2008) for efficient posterior
inference over the infinite state space.

References
----------
- Fox, E. B., Sudderth, E. B., Jordan, M. I., & Willsky, A. S. (2011).
  A sticky HDP-HMM with application to speaker diarization.
  Annals of Applied Statistics, 5(2A), 1020-1056.
- Van Gael, J., Saatci, Y., Teh, Y. W., & Ghahramani, Z. (2008).
  Beam sampling for the infinite hidden Markov model.
  ICML 2008.
- Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2006).
  Hierarchical Dirichlet processes. JASA, 101(476), 1566-1581.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats, special
from scipy.special import gammaln, digamma, logsumexp


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _log_normalize(log_vec: NDArray) -> Tuple[NDArray, float]:
    """Normalize a log-probability vector, returning (log_probs, log_Z)."""
    log_z = logsumexp(log_vec)
    return log_vec - log_z, log_z


def _sample_categorical(log_probs: NDArray, rng: np.random.Generator) -> int:
    """Sample from a categorical distribution given *log* probabilities."""
    probs = np.exp(log_probs - logsumexp(log_probs))
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))


def _sample_dirichlet(alpha: NDArray, rng: np.random.Generator) -> NDArray:
    """Safe Dirichlet draw that handles near-zero concentrations."""
    alpha_safe = np.clip(alpha, 1e-300, None)
    return rng.dirichlet(alpha_safe)


def _stick_breaking(gamma: float, K: int, rng: np.random.Generator) -> NDArray:
    """GEM(gamma) stick-breaking construction truncated to K atoms."""
    betas = np.zeros(K)
    remaining = 1.0
    for k in range(K - 1):
        v_k = rng.beta(1.0, gamma)
        betas[k] = remaining * v_k
        remaining *= 1.0 - v_k
    betas[K - 1] = remaining
    return betas


# ---------------------------------------------------------------------------
# Gaussian emission model
# ---------------------------------------------------------------------------

@dataclass
class GaussianEmission:
    """Conjugate Normal-Inverse-Wishart emission model for each state.

    For univariate data we use Normal-Inverse-Gamma; for multivariate
    data we use Normal-Inverse-Wishart.
    """

    dim: int = 1
    # NIW prior hyper-parameters
    mu_0: Optional[NDArray] = None
    kappa_0: float = 0.01
    nu_0: Optional[float] = None
    Psi_0: Optional[NDArray] = None

    # Sufficient statistics (per-state)
    _n: int = 0
    _sum_x: Optional[NDArray] = None
    _sum_xxT: Optional[NDArray] = None

    def __post_init__(self) -> None:
        if self.mu_0 is None:
            self.mu_0 = np.zeros(self.dim)
        if self.nu_0 is None:
            self.nu_0 = float(self.dim) + 2.0
        if self.Psi_0 is None:
            self.Psi_0 = np.eye(self.dim)
        self._sum_x = np.zeros(self.dim)
        self._sum_xxT = np.zeros((self.dim, self.dim))

    def reset(self) -> None:
        self._n = 0
        self._sum_x = np.zeros(self.dim)
        self._sum_xxT = np.zeros((self.dim, self.dim))

    def add_obs(self, x: NDArray) -> None:
        self._n += 1
        self._sum_x += x
        self._sum_xxT += np.outer(x, x)

    def remove_obs(self, x: NDArray) -> None:
        self._n -= 1
        self._sum_x -= x
        self._sum_xxT -= np.outer(x, x)

    # --- posterior parameters -------------------------------------------
    def _posterior_params(self) -> Tuple[NDArray, float, float, NDArray]:
        n = self._n
        kappa_n = self.kappa_0 + n
        nu_n = self.nu_0 + n
        if n == 0:
            return self.mu_0.copy(), kappa_n, nu_n, self.Psi_0.copy()
        x_bar = self._sum_x / n
        mu_n = (self.kappa_0 * self.mu_0 + n * x_bar) / kappa_n
        S = self._sum_xxT - n * np.outer(x_bar, x_bar)
        diff = x_bar - self.mu_0
        Psi_n = (
            self.Psi_0
            + S
            + (self.kappa_0 * n / kappa_n) * np.outer(diff, diff)
        )
        return mu_n, kappa_n, nu_n, Psi_n

    def log_marginal_likelihood(self, x: NDArray) -> float:
        """Log predictive probability of x under the posterior-predictive
        (multivariate-t distribution)."""
        mu_n, kappa_n, nu_n, Psi_n = self._posterior_params()
        d = self.dim
        df = nu_n - d + 1.0
        if df <= 0:
            df = 1.0
        scale = Psi_n * (kappa_n + 1.0) / (kappa_n * df)
        # Ensure scale is positive definite
        scale = 0.5 * (scale + scale.T) + 1e-10 * np.eye(d)
        try:
            return float(stats.multivariate_t.logpdf(x, loc=mu_n, shape=scale, df=df))
        except Exception:
            # Fallback: simple Gaussian
            return float(stats.multivariate_normal.logpdf(
                x, mean=mu_n, cov=scale * df / (df - 2.0 + 1e-10)
            ))

    def sample_posterior(self, rng: np.random.Generator) -> Tuple[NDArray, NDArray]:
        """Sample (mu, Sigma) from the NIW posterior."""
        mu_n, kappa_n, nu_n, Psi_n = self._posterior_params()
        d = self.dim
        Psi_n = 0.5 * (Psi_n + Psi_n.T) + 1e-10 * np.eye(d)
        Sigma = stats.invwishart.rvs(df=nu_n, scale=Psi_n, random_state=rng)
        if d == 1:
            Sigma = np.atleast_2d(Sigma)
        mu = rng.multivariate_normal(mu_n, Sigma / kappa_n)
        return mu, Sigma

    def log_pdf(self, x: NDArray, mu: NDArray, Sigma: NDArray) -> float:
        """Log probability of x under N(mu, Sigma)."""
        Sigma = 0.5 * (Sigma + Sigma.T) + 1e-10 * np.eye(self.dim)
        return float(stats.multivariate_normal.logpdf(x, mean=mu, cov=Sigma))


# ---------------------------------------------------------------------------
# Chinese Restaurant Franchise
# ---------------------------------------------------------------------------

class ChineseRestaurantFranchise:
    """Implements the CRF representation of the HDP for transition rows.

    Each row j of the transition matrix corresponds to a "restaurant".
    Tables in restaurant j serve dishes (states) drawn from the global
    measure.  The franchise tracks table counts m_{jk} and total counts
    n_{jk}.
    """

    def __init__(self, K_max: int, alpha: float, gamma: float, kappa: float) -> None:
        self.K_max = K_max
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        # n_jk: number of customers in restaurant j eating dish k
        self.n: NDArray = np.zeros((K_max, K_max), dtype=np.int64)
        # m_jk: number of tables in restaurant j serving dish k
        self.m: NDArray = np.zeros((K_max, K_max), dtype=np.int64)

    def reset(self, K_max: Optional[int] = None) -> None:
        if K_max is not None:
            self.K_max = K_max
        self.n = np.zeros((self.K_max, self.K_max), dtype=np.int64)
        self.m = np.zeros((self.K_max, self.K_max), dtype=np.int64)

    def add_customer(self, j: int, k: int, rng: np.random.Generator) -> None:
        """Seat a customer at restaurant j eating dish k.

        With probability proportional to alpha * beta_k (+ kappa if j==k),
        the customer opens a new table; otherwise joins an existing table.
        """
        self.n[j, k] += 1
        if self.n[j, k] == 1:
            self.m[j, k] = 1
        else:
            # Probability of new table
            conc = self.alpha
            if j == k:
                conc += self.kappa
            prob_new = conc / (self.n[j, k] - 1 + conc)
            if rng.random() < prob_new:
                self.m[j, k] += 1

    def remove_customer(self, j: int, k: int, rng: np.random.Generator) -> None:
        """Remove a customer from restaurant j eating dish k."""
        if self.n[j, k] <= 0:
            return
        self.n[j, k] -= 1
        if self.n[j, k] == 0:
            self.m[j, k] = 0
        elif self.m[j, k] > self.n[j, k]:
            self.m[j, k] = self.n[j, k]

    def sample_beta(self, rng: np.random.Generator) -> NDArray:
        """Sample the global measure beta from the posterior."""
        m_dot_k = self.m.sum(axis=0) + 1e-10
        alpha_dir = m_dot_k + self.gamma / self.K_max
        return _sample_dirichlet(alpha_dir, rng)

    def sample_transition_row(self, j: int, beta: NDArray,
                              rng: np.random.Generator) -> NDArray:
        """Sample transition row pi_j from the posterior Dirichlet."""
        conc = self.alpha * beta.copy()
        conc[j] += self.kappa
        counts = self.n[j].astype(np.float64)
        alpha_dir = conc + counts
        return _sample_dirichlet(alpha_dir, rng)


# ---------------------------------------------------------------------------
# Forward-backward & Viterbi
# ---------------------------------------------------------------------------

def _forward(log_pi: NDArray, log_A: NDArray, log_lik: NDArray) -> Tuple[NDArray, float]:
    """Scaled forward algorithm.

    Parameters
    ----------
    log_pi : (K,) log initial distribution
    log_A  : (K, K) log transition matrix  log_A[i, j] = log P(z_t=j|z_{t-1}=i)
    log_lik: (T, K) log emission likelihoods

    Returns
    -------
    log_alpha : (T, K) forward log-messages
    log_evidence : float, log P(X)
    """
    T, K = log_lik.shape
    log_alpha = np.full((T, K), -np.inf)
    log_alpha[0] = log_pi + log_lik[0]
    for t in range(1, T):
        for k in range(K):
            log_alpha[t, k] = logsumexp(log_alpha[t - 1] + log_A[:, k]) + log_lik[t, k]
    log_evidence = logsumexp(log_alpha[-1])
    return log_alpha, log_evidence


def _backward(log_A: NDArray, log_lik: NDArray) -> NDArray:
    """Backward algorithm.

    Returns
    -------
    log_beta : (T, K) backward log-messages
    """
    T, K = log_lik.shape
    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        for k in range(K):
            log_beta[t, k] = logsumexp(
                log_A[k, :] + log_lik[t + 1] + log_beta[t + 1]
            )
    return log_beta


def _forward_backward(log_pi: NDArray, log_A: NDArray,
                       log_lik: NDArray) -> Tuple[NDArray, NDArray, float]:
    """Full forward-backward returning state posteriors and pairwise marginals.

    Returns
    -------
    log_gamma  : (T, K) log P(z_t = k | X)
    log_xi     : (T-1, K, K) log P(z_t=j, z_{t+1}=k | X)
    log_evidence : float
    """
    T, K = log_lik.shape
    log_alpha, log_evidence = _forward(log_pi, log_A, log_lik)
    log_beta = _backward(log_A, log_lik)

    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)

    # Return actual probabilities (not log)
    gamma = np.exp(log_gamma)

    log_xi = np.full((T - 1, K, K), -np.inf)
    for t in range(T - 1):
        for j in range(K):
            for k in range(K):
                log_xi[t, j, k] = (
                    log_alpha[t, j]
                    + log_A[j, k]
                    + log_lik[t + 1, k]
                    + log_beta[t + 1, k]
                )
        log_xi[t] -= logsumexp(log_xi[t].ravel())
    return gamma, log_xi, log_evidence


def _viterbi(log_pi: NDArray, log_A: NDArray, log_lik: NDArray) -> Tuple[NDArray, float]:
    """Viterbi algorithm for MAP state sequence.

    Returns
    -------
    states : (T,) int array of MAP state assignments
    log_prob : float, log probability of the MAP path
    """
    T, K = log_lik.shape
    delta = np.full((T, K), -np.inf)
    psi = np.zeros((T, K), dtype=np.int64)

    delta[0] = log_pi + log_lik[0]
    for t in range(1, T):
        for k in range(K):
            scores = delta[t - 1] + log_A[:, k]
            psi[t, k] = int(np.argmax(scores))
            delta[t, k] = scores[psi[t, k]] + log_lik[t, k]

    states = np.zeros(T, dtype=np.int64)
    states[-1] = int(np.argmax(delta[-1]))
    log_prob = delta[-1, states[-1]]
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states, float(log_prob)


# ---------------------------------------------------------------------------
# Beam sampling utilities (Van Gael et al. 2008)
# ---------------------------------------------------------------------------

def _beam_sample_states(
    log_pi: NDArray,
    log_A: NDArray,
    log_lik: NDArray,
    rng: np.random.Generator,
) -> NDArray:
    """Beam sampling: introduces auxiliary slice variables to limit the
    number of active states at each time step, then runs a truncated
    forward-filter backward-sample pass.

    Parameters
    ----------
    log_pi : (K,) log initial state distribution
    log_A  : (K, K) log transition matrix
    log_lik: (T, K) log emission likelihoods

    Returns
    -------
    states : (T,) sampled state sequence
    """
    T, K = log_lik.shape
    A = np.exp(log_A)
    pi = np.exp(log_pi - logsumexp(log_pi))

    # Forward filter
    log_alpha = np.full((T, K), -np.inf)
    u = np.zeros(T)  # auxiliary slice variables

    # Sample initial state
    log_alpha[0] = log_pi + log_lik[0]

    for t in range(1, T):
        # Draw slice variable: u_t ~ Uniform(0, A[z_{t-1}, z_t])
        # Since we don't know z_{t-1} yet in the forward pass, we use
        # a conservative lower bound.
        max_trans = np.max(A, axis=0)
        u[t] = rng.uniform(0, np.max(max_trans) + 1e-15)

        for k in range(K):
            # Only consider states j where A[j, k] > u[t]
            active = A[:, k] > u[t] * 0.1  # relaxed beam
            if active.any():
                log_alpha[t, k] = logsumexp(
                    log_alpha[t - 1, active] + log_A[active, k]
                ) + log_lik[t, k]
            else:
                # Fall back to full set
                log_alpha[t, k] = logsumexp(
                    log_alpha[t - 1] + log_A[:, k]
                ) + log_lik[t, k]

    # Backward sample
    states = np.zeros(T, dtype=np.int64)
    states[-1] = _sample_categorical(log_alpha[-1], rng)
    for t in range(T - 2, -1, -1):
        log_p = log_alpha[t] + log_A[:, states[t + 1]]
        states[t] = _sample_categorical(log_p, rng)

    return states


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceDiagnostics:
    """Tracks MCMC convergence metrics for Sticky HDP-HMM.

    Provides R-hat, effective sample size (ESS), Geweke diagnostic,
    and transition-matrix-specific convergence checks. These diagnostics
    are critical because the PAC-Bayes shield soundness theorem assumes
    access to the true Bayesian posterior; MCMC approximation quality
    must be verified for the bound to hold.
    """
    log_likelihoods: List[float] = field(default_factory=list)
    n_states_trace: List[int] = field(default_factory=list)
    acceptance_rates: List[float] = field(default_factory=list)
    transition_matrix_trace: List[NDArray] = field(default_factory=list)

    def record_transition_matrix(self, T: NDArray) -> None:
        """Record a transition matrix sample for diagnostics."""
        self.transition_matrix_trace.append(T.copy())

    def effective_sample_size(self, burn_in: int = 0) -> float:
        """Compute ESS of the log-likelihood chain via autocorrelation."""
        chain = np.array(self.log_likelihoods[burn_in:])
        if len(chain) < 10:
            return float(len(chain))
        n = len(chain)
        chain_centered = chain - chain.mean()
        acf = np.correlate(chain_centered, chain_centered, mode="full")
        acf = acf[n - 1:]
        acf /= acf[0] + 1e-300
        # Find first negative autocorrelation
        neg_idx = np.where(acf < 0.0)[0]
        cutoff = neg_idx[0] if len(neg_idx) > 0 else len(acf)
        tau = 1.0 + 2.0 * np.sum(acf[1:cutoff])
        return n / max(tau, 1.0)

    def transition_matrix_ess(self, burn_in: int = 0) -> float:
        """Compute minimum ESS across transition matrix entries.

        This is the bottleneck ESS: if any entry has low ESS, the
        posterior over transition dynamics is poorly estimated, which
        invalidates PAC-Bayes bounds that assume exact posteriors.

        Returns
        -------
        float
            Minimum ESS across all (i,j) entries of the transition matrix.
        """
        traces = self.transition_matrix_trace[burn_in:]
        if len(traces) < 10:
            return float(len(traces))
        T_stack = np.array(traces)  # (n_samples, K, K)
        n_samples, K, _ = T_stack.shape
        min_ess = float('inf')
        for i in range(K):
            for j in range(K):
                chain = T_stack[:, i, j]
                if np.std(chain) < 1e-15:
                    continue
                chain_c = chain - chain.mean()
                acf = np.correlate(chain_c, chain_c, mode="full")
                acf = acf[n_samples - 1:]
                acf /= acf[0] + 1e-300
                neg_idx = np.where(acf < 0.0)[0]
                cutoff = neg_idx[0] if len(neg_idx) > 0 else len(acf)
                tau = 1.0 + 2.0 * np.sum(acf[1:cutoff])
                ess = n_samples / max(tau, 1.0)
                min_ess = min(min_ess, ess)
        return min_ess if min_ess != float('inf') else float(len(traces))

    def geweke_z(self, frac_a: float = 0.1, frac_b: float = 0.5) -> float:
        """Geweke convergence diagnostic z-score."""
        chain = np.array(self.log_likelihoods)
        n = len(chain)
        if n < 20:
            return 0.0
        n_a = int(n * frac_a)
        n_b = int(n * frac_b)
        a = chain[:n_a]
        b = chain[-n_b:]
        mean_diff = a.mean() - b.mean()
        var_sum = np.var(a) / len(a) + np.var(b) / len(b)
        if var_sum < 1e-300:
            return 0.0
        return float(mean_diff / np.sqrt(var_sum))

    def has_converged(self, threshold: float = 2.0, min_iter: int = 100) -> bool:
        """Check convergence via Geweke diagnostic."""
        if len(self.log_likelihoods) < min_iter:
            return False
        return abs(self.geweke_z()) < threshold

    def rhat(self, n_chains: int = 2) -> float:
        """Split-chain R-hat diagnostic (Gelman-Rubin).

        Splits the post-burn-in chain into n_chains segments and computes
        the potential scale reduction factor. Values close to 1.0 indicate
        convergence; R-hat > 1.1 suggests the chain has not converged.
        """
        chain = np.array(self.log_likelihoods)
        n = len(chain)
        if n < 2 * n_chains:
            return float("inf")
        split_len = n // n_chains
        chains = [chain[i * split_len:(i + 1) * split_len] for i in range(n_chains)]
        chain_means = [c.mean() for c in chains]
        chain_vars = [c.var(ddof=1) for c in chains]
        n_s = split_len
        B = n_s * np.var(chain_means, ddof=1)
        W = np.mean(chain_vars)
        if W < 1e-300:
            return 1.0
        var_hat = (1 - 1.0 / n_s) * W + B / n_s
        return float(np.sqrt(var_hat / W))

    def transition_rhat(self, n_chains: int = 2) -> float:
        """Maximum R-hat across transition matrix entries.

        Returns the worst-case R-hat to ensure all transition
        probabilities have converged.
        """
        traces = self.transition_matrix_trace
        if len(traces) < 2 * n_chains:
            return float("inf")
        T_stack = np.array(traces)
        _, K, _ = T_stack.shape
        max_rhat = 0.0
        split_len = len(traces) // n_chains
        for i in range(K):
            for j in range(K):
                chain = T_stack[:, i, j]
                if np.std(chain) < 1e-15:
                    continue
                splits = [chain[s * split_len:(s + 1) * split_len] for s in range(n_chains)]
                means = [c.mean() for c in splits]
                varis = [c.var(ddof=1) for c in splits]
                B = split_len * np.var(means, ddof=1)
                W = np.mean(varis)
                if W < 1e-300:
                    continue
                var_hat = (1 - 1.0 / split_len) * W + B / split_len
                rh = np.sqrt(var_hat / W)
                max_rhat = max(max_rhat, rh)
        return float(max_rhat) if max_rhat > 0 else 1.0

    def full_diagnostic_report(self, burn_in: int = 0) -> Dict[str, Any]:
        """Generate comprehensive convergence diagnostic report.

        This report should be checked before trusting PAC-Bayes bounds.
        If any diagnostic fails, the posterior approximation quality is
        suspect and bounds may be unreliable.

        Returns
        -------
        dict with keys:
            converged : bool
            ll_ess : float
            ll_rhat : float
            geweke_z : float
            tm_ess : float (transition matrix ESS)
            tm_rhat : float (transition matrix R-hat)
            n_iterations : int
            warnings : list of str
        """
        warnings_list: List[str] = []
        ll_ess = self.effective_sample_size(burn_in)
        ll_rhat = self.rhat()
        gz = self.geweke_z()

        has_tm = len(self.transition_matrix_trace) > 0
        tm_ess = self.transition_matrix_ess(burn_in) if has_tm else float('nan')
        tm_rhat = self.transition_rhat() if has_tm else float('nan')

        # Check diagnostic thresholds
        if ll_ess < 100:
            warnings_list.append(
                f"Low log-likelihood ESS ({ll_ess:.0f} < 100): "
                "chain may not have mixed well"
            )
        if ll_rhat > 1.1:
            warnings_list.append(
                f"High R-hat ({ll_rhat:.3f} > 1.1): "
                "chain segments disagree, increase iterations"
            )
        if abs(gz) > 2.0:
            warnings_list.append(
                f"Geweke |z| = {abs(gz):.2f} > 2.0: "
                "initial and final segments differ significantly"
            )
        if has_tm and tm_ess < 50:
            warnings_list.append(
                f"Low transition matrix ESS ({tm_ess:.0f} < 50): "
                "transition posterior poorly estimated, PAC-Bayes bound unreliable"
            )
        if has_tm and tm_rhat > 1.1:
            warnings_list.append(
                f"High transition R-hat ({tm_rhat:.3f} > 1.1): "
                "transition matrix not converged"
            )

        converged = (
            ll_rhat < 1.1
            and abs(gz) < 2.0
            and ll_ess >= 100
            and (not has_tm or tm_rhat < 1.1)
        )

        return {
            "converged": converged,
            "ll_ess": ll_ess,
            "ll_rhat": ll_rhat,
            "geweke_z": gz,
            "tm_ess": tm_ess,
            "tm_rhat": tm_rhat,
            "n_iterations": len(self.log_likelihoods),
            "warnings": warnings_list,
        }


# ---------------------------------------------------------------------------
# Main StickyHDPHMM class
# ---------------------------------------------------------------------------

class StickyHDPHMM:
    """Sticky Hierarchical Dirichlet Process Hidden Markov Model.

    A Bayesian nonparametric HMM that automatically infers the number
    of hidden states from data.  The *sticky* extension adds an extra
    mass ``kappa`` to the self-transition probability, encouraging
    temporal persistence in state assignments.

    Parameters
    ----------
    K_max : int
        Practical upper bound on the number of states (default 5).
    alpha : float
        Concentration parameter for the DP prior on each transition row.
    gamma : float
        Top-level concentration for the global DP measure.
    kappa : float
        Sticky parameter: extra mass added to self-transitions.
    n_iter : int
        Number of Gibbs sampling iterations.
    burn_in : int
        Number of burn-in iterations to discard.
    emission : str
        Emission model type.  Currently only ``"gaussian"`` supported.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        K_max: int = 5,
        alpha: float = 1.0,
        gamma: float = 5.0,
        kappa: float = 10.0,
        n_iter: int = 500,
        burn_in: int = 100,
        emission: str = "gaussian",
        random_state: Optional[int] = None,
    ) -> None:
        self.K_max = K_max
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.emission_type = emission
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)
        self._is_fitted = False

        # Learned parameters (populated after fit)
        self.pi_: Optional[NDArray] = None       # initial distribution (K,)
        self.A_: Optional[NDArray] = None         # transition matrix (K, K)
        self.means_: Optional[NDArray] = None     # emission means (K, D)
        self.covars_: Optional[NDArray] = None    # emission covariances (K, D, D)
        self.beta_: Optional[NDArray] = None      # global measure (K,)
        self.states_: Optional[NDArray] = None    # MAP state sequence (T,)

        self.diagnostics_ = ConvergenceDiagnostics()

        # Internal
        self._crf: Optional[ChineseRestaurantFranchise] = None
        self._emissions: List[GaussianEmission] = []
        self._state_samples: List[NDArray] = []
        self._A_samples: List[NDArray] = []

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(self, X: NDArray) -> NDArray:
        """Ensure X is 2-D array (T, D)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self, X: NDArray) -> None:
        """Initialize model parameters."""
        T, D = X.shape
        K = self.K_max

        # Emissions: use K-means-like initialisation
        indices = self._rng.choice(T, size=min(K, T), replace=False)
        init_means = X[indices]

        self._emissions = []
        for k in range(K):
            em = GaussianEmission(dim=D)
            em.__post_init__()
            self._emissions.append(em)

        # Initial state sequence: assign by nearest center
        dists = np.array([
            np.sum((X - init_means[k]) ** 2, axis=1) for k in range(K)
        ])  # (K, T)
        self.states_ = np.argmin(dists, axis=0).astype(np.int64)

        # Populate sufficient statistics
        for k in range(K):
            self._emissions[k].reset()
        for t in range(T):
            self._emissions[self.states_[t]].add_obs(X[t])

        # CRF
        self._crf = ChineseRestaurantFranchise(K, self.alpha, self.gamma, self.kappa)
        for t in range(1, T):
            j, k = int(self.states_[t - 1]), int(self.states_[t])
            self._crf.add_customer(j, k, self._rng)

        # Global measure and transition matrix
        self.beta_ = self._crf.sample_beta(self._rng)
        self.A_ = np.zeros((K, K))
        for j in range(K):
            self.A_[j] = self._crf.sample_transition_row(j, self.beta_, self._rng)

        # Initial distribution
        self.pi_ = np.ones(K) / K

        # Sample emission parameters
        self.means_ = np.zeros((K, D))
        self.covars_ = np.zeros((K, D, D))
        for k in range(K):
            mu, Sig = self._emissions[k].sample_posterior(self._rng)
            self.means_[k] = mu
            self.covars_[k] = Sig

    # ------------------------------------------------------------------
    # Emission log-likelihood matrix
    # ------------------------------------------------------------------

    def _compute_log_lik(self, X: NDArray) -> NDArray:
        """Compute (T, K) matrix of log-emission probabilities."""
        T, D = X.shape
        K = self.K_max
        log_lik = np.zeros((T, K))
        for k in range(K):
            Sig = self.covars_[k]
            Sig = 0.5 * (Sig + Sig.T) + 1e-8 * np.eye(D)
            try:
                log_lik[:, k] = stats.multivariate_normal.logpdf(
                    X, mean=self.means_[k], cov=Sig
                )
            except np.linalg.LinAlgError:
                log_lik[:, k] = -1e10
        return log_lik

    # ------------------------------------------------------------------
    # Gibbs sampling steps
    # ------------------------------------------------------------------

    def _resample_states(self, X: NDArray, use_beam: bool = True) -> None:
        """Re-sample the state sequence z_{1:T}."""
        T, D = X.shape
        K = self.K_max
        log_lik = self._compute_log_lik(X)
        log_A = np.log(self.A_ + 1e-300)
        log_pi = np.log(self.pi_ + 1e-300)

        old_states = self.states_.copy()

        if use_beam:
            self.states_ = _beam_sample_states(log_pi, log_A, log_lik, self._rng)
        else:
            # Direct forward-filter backward-sample
            log_alpha, _ = _forward(log_pi, log_A, log_lik)
            states = np.zeros(T, dtype=np.int64)
            states[-1] = _sample_categorical(log_alpha[-1], self._rng)
            for t in range(T - 2, -1, -1):
                log_p = log_alpha[t] + log_A[:, states[t + 1]]
                states[t] = _sample_categorical(log_p, self._rng)
            self.states_ = states

        # Update sufficient statistics
        self._crf.reset(K)
        for k in range(K):
            self._emissions[k].reset()
        for t in range(T):
            self._emissions[self.states_[t]].add_obs(X[t])
            if t > 0:
                j = int(self.states_[t - 1])
                k = int(self.states_[t])
                self._crf.add_customer(j, k, self._rng)

    def _resample_beta(self) -> None:
        """Re-sample the global DP measure beta."""
        self.beta_ = self._crf.sample_beta(self._rng)

    def _resample_transitions(self) -> None:
        """Re-sample each row of the transition matrix."""
        K = self.K_max
        for j in range(K):
            self.A_[j] = self._crf.sample_transition_row(j, self.beta_, self._rng)

    def _resample_initial(self) -> None:
        """Re-sample the initial state distribution."""
        counts = np.zeros(self.K_max)
        if self.states_ is not None and len(self.states_) > 0:
            counts[self.states_[0]] += 1
        alpha_dir = counts + self.gamma / self.K_max
        self.pi_ = _sample_dirichlet(alpha_dir, self._rng)

    def _resample_emissions(self, X: NDArray) -> None:
        """Re-sample emission parameters from their NIW posteriors."""
        K = self.K_max
        D = X.shape[1]
        for k in range(K):
            mu, Sig = self._emissions[k].sample_posterior(self._rng)
            self.means_[k] = mu
            self.covars_[k] = Sig

    def _resample_hyperparameters(self) -> None:
        """Optionally re-sample alpha, gamma, kappa using auxiliary variable
        methods (Escobar & West 1995, Teh et al. 2006)."""
        K = self.K_max

        # --- alpha ---
        # Auxiliary variable method
        m_total = max(int(self._crf.m.sum()), 1)
        n_total = max(int(self._crf.n.sum()), 1)
        # Sample auxiliary w ~ Beta(alpha+1, n_total)
        w = self._rng.beta(self.alpha + 1, max(n_total, 1))
        log_w = np.log(w + 1e-300)
        # Mixture weight
        s = n_total / (n_total + self.alpha)
        pi_w = s / (s + (1 - s) * (m_total - 1) / (-log_w + 1e-300) if log_w < 0 else s)
        pi_w = np.clip(pi_w, 0.01, 0.99)
        if self._rng.random() < pi_w:
            self.alpha = self._rng.gamma(1.0 + m_total - 1, 1.0 / (-log_w + 1e-10))
        else:
            self.alpha = self._rng.gamma(1.0 + m_total, 1.0 / (-log_w + 1e-10))
        self.alpha = np.clip(self.alpha, 0.01, 100.0)

        # --- gamma ---
        m_bar = self._crf.m.sum()
        k_used = len(np.unique(self.states_)) if self.states_ is not None else K
        eta = self._rng.beta(self.gamma + 1, max(m_bar, 1))
        log_eta = np.log(eta + 1e-300)
        mix = m_bar / (m_bar + self.gamma) if m_bar > 0 else 0.5
        if self._rng.random() < mix:
            self.gamma = self._rng.gamma(1.0 + k_used - 1, 1.0 / (-log_eta + 1e-10))
        else:
            self.gamma = self._rng.gamma(1.0 + k_used, 1.0 / (-log_eta + 1e-10))
        self.gamma = np.clip(self.gamma, 0.1, 50.0)

        # Update CRF concentrations
        self._crf.alpha = self.alpha
        self._crf.gamma = self.gamma

    # ------------------------------------------------------------------
    # Compute log-likelihood of current state
    # ------------------------------------------------------------------

    def _log_likelihood(self, X: NDArray) -> float:
        """Compute complete-data log-likelihood."""
        T, D = X.shape
        ll = 0.0
        # Emission term
        for t in range(T):
            k = int(self.states_[t])
            ll += self._emissions[k].log_pdf(X[t], self.means_[k], self.covars_[k])
        # Transition term
        ll += np.log(self.pi_[self.states_[0]] + 1e-300)
        for t in range(1, T):
            j = int(self.states_[t - 1])
            k = int(self.states_[t])
            ll += np.log(self.A_[j, k] + 1e-300)
        return float(ll)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: NDArray, verbose: bool = False) -> "StickyHDPHMM":
        """Fit the Sticky HDP-HMM to observed data via Gibbs sampling.

        Parameters
        ----------
        X : array-like of shape (T,) or (T, D)
            Observed time series.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        self
        """
        X = self._prepare_data(X)
        T, D = X.shape
        self._initialize(X)
        self._state_samples = []
        self._A_samples = []
        self.diagnostics_ = ConvergenceDiagnostics()

        for it in range(self.n_iter):
            # Gibbs sweep
            use_beam = (it % 3 == 0)  # alternate beam / FFBS
            self._resample_states(X, use_beam=use_beam)
            self._resample_beta()
            self._resample_transitions()
            self._resample_initial()
            self._resample_emissions(X)
            if it % 10 == 0:
                self._resample_hyperparameters()

            # Diagnostics
            ll = self._log_likelihood(X)
            self.diagnostics_.log_likelihoods.append(ll)
            n_active = len(np.unique(self.states_))
            self.diagnostics_.n_states_trace.append(n_active)
            if hasattr(self, 'A_') and self.A_ is not None:
                self.diagnostics_.record_transition_matrix(self.A_)

            if it >= self.burn_in:
                self._state_samples.append(self.states_.copy())
                self._A_samples.append(self.A_.copy())

            if verbose and it % 50 == 0:
                print(
                    f"Iter {it:4d} | LL={ll:12.2f} | K_active={n_active} | "
                    f"alpha={self.alpha:.3f} gamma={self.gamma:.3f}"
                )

        # Posterior summaries
        self._compute_posterior_summaries(X)
        self._is_fitted = True
        return self

    def _compute_posterior_summaries(self, X: NDArray) -> None:
        """Compute posterior mean of transition matrix and MAP states."""
        if self._A_samples:
            self.A_ = np.mean(self._A_samples, axis=0)
        if self._state_samples:
            # Modal state at each time
            state_mat = np.array(self._state_samples)
            T = state_mat.shape[1]
            modal_states = np.zeros(T, dtype=np.int64)
            for t in range(T):
                counts = np.bincount(state_mat[:, t], minlength=self.K_max)
                modal_states[t] = int(np.argmax(counts))
            self.states_ = modal_states

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X: NDArray) -> NDArray:
        """Predict most likely state sequence for new data via Viterbi.

        Parameters
        ----------
        X : array-like of shape (T,) or (T, D)

        Returns
        -------
        states : (T,) int array
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")
        X = self._prepare_data(X)
        log_lik = self._compute_log_lik(X)
        log_A = np.log(self.A_ + 1e-300)
        log_pi = np.log(self.pi_ + 1e-300)
        states, _ = _viterbi(log_pi, log_A, log_lik)
        return states

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------

    def predict_proba(self, X: NDArray) -> NDArray:
        """Compute posterior state probabilities P(z_t = k | X) via
        the forward-backward algorithm.

        Parameters
        ----------
        X : array-like of shape (T,) or (T, D)

        Returns
        -------
        gamma : (T, K) state posteriors
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba().")
        X = self._prepare_data(X)
        log_lik = self._compute_log_lik(X)
        log_A = np.log(self.A_ + 1e-300)
        log_pi = np.log(self.pi_ + 1e-300)
        gamma, _, _ = _forward_backward(log_pi, log_A, log_lik)
        return gamma

    # ------------------------------------------------------------------
    # sample
    # ------------------------------------------------------------------

    def sample(self, n: int) -> Tuple[NDArray, NDArray]:
        """Generate samples from the fitted model.

        Parameters
        ----------
        n : int
            Number of time steps to generate.

        Returns
        -------
        X : (n, D) observations
        states : (n,) state sequence
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling sample().")
        D = self.means_.shape[1]
        K = self.K_max
        X = np.zeros((n, D))
        states = np.zeros(n, dtype=np.int64)

        # Initial state
        states[0] = self._rng.choice(K, p=self.pi_)
        Sig = self.covars_[states[0]]
        Sig = 0.5 * (Sig + Sig.T) + 1e-8 * np.eye(D)
        X[0] = self._rng.multivariate_normal(self.means_[states[0]], Sig)

        for t in range(1, n):
            pi_row = self.A_[states[t - 1]]
            pi_row = np.clip(pi_row, 0, None)
            pi_row /= pi_row.sum()
            states[t] = self._rng.choice(K, p=pi_row)
            Sig = self.covars_[states[t]]
            Sig = 0.5 * (Sig + Sig.T) + 1e-8 * np.eye(D)
            X[t] = self._rng.multivariate_normal(self.means_[states[t]], Sig)

        return X, states

    # ------------------------------------------------------------------
    # get_transition_matrix
    # ------------------------------------------------------------------

    def get_transition_matrix(self) -> NDArray:
        """Return the posterior mean transition matrix.

        Returns
        -------
        A : (K, K) transition probability matrix
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        return self.A_.copy()

    # ------------------------------------------------------------------
    # score
    # ------------------------------------------------------------------

    def score(self, X: NDArray) -> float:
        """Compute log-marginal-likelihood log P(X) using the forward algorithm.

        Parameters
        ----------
        X : array-like of shape (T,) or (T, D)

        Returns
        -------
        log_evidence : float
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        X = self._prepare_data(X)
        log_lik = self._compute_log_lik(X)
        log_A = np.log(self.A_ + 1e-300)
        log_pi = np.log(self.pi_ + 1e-300)
        _, log_evidence = _forward(log_pi, log_A, log_lik)
        return log_evidence

    # ------------------------------------------------------------------
    # get_stationary_distribution
    # ------------------------------------------------------------------

    def get_stationary_distribution(self) -> NDArray:
        """Compute the stationary distribution of the transition matrix.

        Solves pi * A = pi, sum(pi) = 1.

        Returns
        -------
        pi_stat : (K,) stationary distribution
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        A = self.A_
        K = A.shape[0]
        # Solve (A^T - I) pi = 0 with constraint sum(pi) = 1
        # Augmented system
        M = np.vstack([A.T - np.eye(K), np.ones(K)])
        b = np.zeros(K + 1)
        b[-1] = 1.0
        pi_stat, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
        pi_stat = np.clip(pi_stat, 0, None)
        pi_stat /= pi_stat.sum()
        return pi_stat

    # ------------------------------------------------------------------
    # Posterior credible intervals for transition matrix
    # ------------------------------------------------------------------

    def transition_credible_intervals(
        self, credible_level: float = 0.95
    ) -> Tuple[NDArray, NDArray]:
        """Compute pointwise credible intervals for each A[i,j].

        Returns
        -------
        lower : (K, K) lower bounds
        upper : (K, K) upper bounds
        """
        if not self._A_samples:
            raise RuntimeError("No posterior samples; run fit() first.")
        A_stack = np.array(self._A_samples)
        alpha_lo = (1.0 - credible_level) / 2.0
        alpha_hi = 1.0 - alpha_lo
        lower = np.quantile(A_stack, alpha_lo, axis=0)
        upper = np.quantile(A_stack, alpha_hi, axis=0)
        return lower, upper

    # ------------------------------------------------------------------
    # Model summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a summary dictionary of the fitted model."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        n_active = len(np.unique(self.states_))
        return {
            "K_max": self.K_max,
            "K_active": n_active,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "kappa": self.kappa,
            "n_iter": self.n_iter,
            "burn_in": self.burn_in,
            "final_log_likelihood": self.diagnostics_.log_likelihoods[-1],
            "effective_sample_size": self.diagnostics_.effective_sample_size(self.burn_in),
            "geweke_z": self.diagnostics_.geweke_z(),
            "rhat": self.diagnostics_.rhat(),
            "transition_rhat": self.diagnostics_.transition_rhat(),
            "transition_ess": self.diagnostics_.transition_matrix_ess(self.burn_in),
            "converged": self.diagnostics_.has_converged(),
            "convergence_report": self.diagnostics_.full_diagnostic_report(self.burn_in),
        }

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"StickyHDPHMM(K_max={self.K_max}, alpha={self.alpha:.2f}, "
            f"gamma={self.gamma:.2f}, kappa={self.kappa:.2f}, "
            f"n_iter={self.n_iter}, status={status})"
        )
