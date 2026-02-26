"""
Bayesian regime detection with online particle filtering and change-point detection.

Implements multiple complementary approaches:
- Particle-filter-based online regime tracking
- Bayesian Online Change Point Detection (Adams & MacKay, 2007)
- CUSUM-based change detection
- Full posterior regime inference

References
----------
- Adams, R. P. & MacKay, D. J. C. (2007).
  Bayesian Online Changepoint Detection. arXiv:0710.3742
- Fearnhead, P. & Liu, Z. (2007).
  On-line inference for multiple changepoint problems.
  JRSS-B, 69(4), 589-605.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats, special
from scipy.special import logsumexp, gammaln


# ---------------------------------------------------------------------------
# Particle for sequential regime inference
# ---------------------------------------------------------------------------

@dataclass
class Particle:
    """A single particle in the regime-tracking particle filter."""
    state: int
    log_weight: float
    emission_stats: Dict[str, Any] = field(default_factory=dict)
    run_length: int = 0

    def copy(self) -> "Particle":
        return Particle(
            state=self.state,
            log_weight=self.log_weight,
            emission_stats={k: v.copy() if isinstance(v, np.ndarray) else v
                            for k, v in self.emission_stats.items()},
            run_length=self.run_length,
        )


# ---------------------------------------------------------------------------
# CUSUM detector
# ---------------------------------------------------------------------------

class CUSUMDetector:
    """Cumulative Sum (CUSUM) change-point detector for mean shifts.

    Maintains two-sided CUSUM statistics and declares a change when
    either exceeds the threshold.

    Parameters
    ----------
    threshold : float
        Detection threshold (h).
    drift : float
        Allowance parameter (delta / 2 for detecting shift of size delta).
    """

    def __init__(self, threshold: float = 5.0, drift: float = 0.5) -> None:
        self.threshold = threshold
        self.drift = drift
        self.s_pos: float = 0.0
        self.s_neg: float = 0.0
        self._mu: float = 0.0
        self._n: int = 0
        self._sum: float = 0.0
        self._sum_sq: float = 0.0
        self.change_points: List[int] = []
        self._t: int = 0

    def reset(self) -> None:
        self.s_pos = 0.0
        self.s_neg = 0.0
        self._mu = 0.0
        self._n = 0
        self._sum = 0.0
        self._sum_sq = 0.0
        self.change_points = []
        self._t = 0

    def update(self, x: float) -> bool:
        """Process one observation. Returns True if change detected."""
        self._t += 1
        self._n += 1
        self._sum += x
        self._sum_sq += x * x
        self._mu = self._sum / self._n
        sigma = np.sqrt(max(self._sum_sq / self._n - self._mu ** 2, 1e-10))

        z = (x - self._mu) / sigma
        self.s_pos = max(0.0, self.s_pos + z - self.drift)
        self.s_neg = max(0.0, self.s_neg - z - self.drift)

        detected = self.s_pos > self.threshold or self.s_neg > self.threshold
        if detected:
            self.change_points.append(self._t)
            self.s_pos = 0.0
            self.s_neg = 0.0
            # Partial reset of running stats
            self._n = 1
            self._sum = x
            self._sum_sq = x * x
        return detected

    def detect_batch(self, X: NDArray) -> List[int]:
        """Run CUSUM on a full array and return detected change points."""
        self.reset()
        X_flat = np.asarray(X).ravel()
        for x in X_flat:
            self.update(x)
        return list(self.change_points)


# ---------------------------------------------------------------------------
# Bayesian Online Change Point Detection (BOCPD)
# ---------------------------------------------------------------------------

class BayesianOnlineChangePointDetector:
    """Bayesian Online Change Point Detection (Adams & MacKay 2007).

    Maintains a run-length distribution P(r_t | x_{1:t}) and computes
    the posterior probability of a change point at each time step.

    Parameters
    ----------
    hazard_rate : float
        Prior probability of a change point at each step (1 / expected_run_length).
    mu_0 : float
        Prior mean for the Gaussian conjugate model.
    kappa_0 : float
        Prior precision scaling.
    alpha_0 : float
        Prior shape for inverse-gamma variance.
    beta_0 : float
        Prior rate for inverse-gamma variance.
    """

    def __init__(
        self,
        hazard_rate: float = 1.0 / 200.0,
        mu_0: float = 0.0,
        kappa_0: float = 1.0,
        alpha_0: float = 1.0,
        beta_0: float = 1.0,
    ) -> None:
        self.hazard_rate = hazard_rate
        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        # Run-length distribution
        self._log_R: List[NDArray] = []
        self._mu: List[float] = [mu_0]
        self._kappa: List[float] = [kappa_0]
        self._alpha: List[float] = [alpha_0]
        self._beta: List[float] = [beta_0]
        self._t: int = 0
        self.change_probs: List[float] = []

    def reset(self) -> None:
        self._log_R = []
        self._mu = [self.mu_0]
        self._kappa = [self.kappa_0]
        self._alpha = [self.alpha_0]
        self._beta = [self.beta_0]
        self._t = 0
        self.change_probs = []

    def _predictive_log_prob(self, x: float, idx: int) -> float:
        """Log probability of x under the predictive t-distribution
        for run-length idx."""
        mu = self._mu[idx]
        kappa = self._kappa[idx]
        alpha = self._alpha[idx]
        beta = self._beta[idx]

        df = 2.0 * alpha
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        return float(stats.t.logpdf(x, df=df, loc=mu, scale=scale))

    def update(self, x: float) -> float:
        """Process one observation and return P(change point at t).

        Returns
        -------
        change_prob : float
            Posterior probability that a change point occurred at this step.
        """
        self._t += 1
        n_rl = len(self._mu)  # number of run-length hypotheses

        # Compute predictive probabilities for each run length
        log_pred = np.array([
            self._predictive_log_prob(x, i) for i in range(n_rl)
        ])

        # Growth probabilities
        log_h = np.log(self.hazard_rate)
        log_1mh = np.log(1.0 - self.hazard_rate)

        if self._t == 1:
            # Initialize
            log_R_prev = np.array([0.0])
        else:
            log_R_prev = self._log_R[-1]

        # Growth: extend run lengths
        log_growth = log_R_prev + log_pred + log_1mh
        # Change point: reset run length
        log_cp = logsumexp(log_R_prev + log_pred + log_h)

        # New run-length distribution
        log_R_new = np.concatenate([[log_cp], log_growth])
        # Normalize
        log_Z = logsumexp(log_R_new)
        log_R_new -= log_Z

        self._log_R.append(log_R_new)

        # Update sufficient statistics for each run length
        new_mu = [self.mu_0]
        new_kappa = [self.kappa_0]
        new_alpha = [self.alpha_0]
        new_beta = [self.beta_0]

        for i in range(n_rl):
            mu_old = self._mu[i]
            kappa_old = self._kappa[i]
            alpha_old = self._alpha[i]
            beta_old = self._beta[i]

            kappa_new = kappa_old + 1.0
            mu_new = (kappa_old * mu_old + x) / kappa_new
            alpha_new = alpha_old + 0.5
            beta_new = beta_old + 0.5 * kappa_old * (x - mu_old) ** 2 / kappa_new

            new_mu.append(mu_new)
            new_kappa.append(kappa_new)
            new_alpha.append(alpha_new)
            new_beta.append(beta_new)

        self._mu = new_mu
        self._kappa = new_kappa
        self._alpha = new_alpha
        self._beta = new_beta

        # P(change point) = P(r_t = 0)
        change_prob = float(np.exp(log_R_new[0]))
        self.change_probs.append(change_prob)
        return change_prob

    def detect_batch(self, X: NDArray, threshold: float = 0.5) -> List[int]:
        """Run BOCPD on full array. Return indices where P(cp) > threshold.

        Also detects change points where the run-length distribution suddenly
        shifts to favor short run lengths (the expected run-length drops below
        a fraction of the elapsed time).
        """
        self.reset()
        X_flat = np.asarray(X).ravel()
        cps = []
        prev_mean_rl = 0.0
        for t, x in enumerate(X_flat):
            p = self.update(x)
            if p > threshold:
                cps.append(t)
            elif t > 10:
                # Detect via run-length posterior: if mean run length drops
                # sharply, it indicates a regime change
                log_R = self._log_R[-1]
                R = np.exp(log_R)
                run_lengths = np.arange(len(R))
                mean_rl = float(np.sum(R * run_lengths))
                # Sharp drop in mean run length signals a change point
                if prev_mean_rl > 20 and mean_rl < prev_mean_rl * threshold:
                    cps.append(t)
                prev_mean_rl = mean_rl
        return cps

    def get_run_length_distribution(self) -> NDArray:
        """Return the most recent run-length posterior."""
        if not self._log_R:
            return np.array([1.0])
        return np.exp(self._log_R[-1])

    def get_expected_run_length(self) -> float:
        """Expected run length under the current posterior."""
        rl_dist = self.get_run_length_distribution()
        return float(np.sum(np.arange(len(rl_dist)) * rl_dist))


# ---------------------------------------------------------------------------
# Main BayesianRegimeDetector
# ---------------------------------------------------------------------------

class BayesianRegimeDetector:
    """Bayesian regime detector combining particle filtering with
    change-point detection.

    Uses a particle filter to maintain a posterior distribution over
    the current regime, combining HMM-style dynamics with online
    Bayesian change-point detection.

    Parameters
    ----------
    n_regimes : int
        Maximum number of regimes.
    n_particles : int
        Number of particles for the sequential Monte Carlo filter.
    hazard_rate : float
        Prior change-point probability per time step.
    resample_threshold : float
        ESS fraction below which systematic resampling is triggered.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_regimes: int = 5,
        n_particles: int = 500,
        hazard_rate: float = 1.0 / 200.0,
        resample_threshold: float = 0.5,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_regimes = n_regimes
        self.n_particles = n_particles
        self.hazard_rate = hazard_rate
        self.resample_threshold = resample_threshold
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)
        self._particles: List[Particle] = []
        self._transition_matrix = np.ones((n_regimes, n_regimes)) / n_regimes
        self._posteriors: List[NDArray] = []
        self._change_points: List[int] = []
        self._t: int = 0

        # Sub-detectors
        self._cusum = CUSUMDetector()
        self._bocpd = BayesianOnlineChangePointDetector(hazard_rate=hazard_rate)

        # Emission parameters (online estimates per regime)
        self._regime_means = np.zeros(n_regimes)
        self._regime_vars = np.ones(n_regimes)
        self._regime_counts = np.zeros(n_regimes)

        self._is_fitted = False

    def _initialize_particles(self) -> None:
        """Create initial particle set with uniform weights."""
        self._particles = []
        log_w = -np.log(self.n_particles)
        for _ in range(self.n_particles):
            state = self._rng.integers(0, self.n_regimes)
            p = Particle(
                state=state,
                log_weight=log_w,
                emission_stats={"sum": 0.0, "sum_sq": 0.0, "n": 0},
            )
            self._particles.append(p)

    def _emission_log_prob(self, x: float, state: int) -> float:
        """Log-probability of observation x under regime state."""
        mu = self._regime_means[state]
        var = max(self._regime_vars[state], 1e-6)
        return float(stats.norm.logpdf(x, loc=mu, scale=np.sqrt(var)))

    def _update_emission_stats(self, x: float, state: int) -> None:
        """Update online emission sufficient statistics for a regime."""
        self._regime_counts[state] += 1
        n = self._regime_counts[state]
        old_mu = self._regime_means[state]
        # Welford's online algorithm
        delta = x - old_mu
        self._regime_means[state] += delta / n
        delta2 = x - self._regime_means[state]
        self._regime_vars[state] += (delta * delta2 - self._regime_vars[state]) / n

    def _effective_sample_size(self) -> float:
        """Compute ESS of the particle set."""
        log_w = np.array([p.log_weight for p in self._particles])
        log_w_norm = log_w - logsumexp(log_w)
        return float(np.exp(-logsumexp(2.0 * log_w_norm)))

    def _systematic_resample(self) -> None:
        """Systematic resampling of the particle set."""
        N = self.n_particles
        log_w = np.array([p.log_weight for p in self._particles])
        log_w_norm = log_w - logsumexp(log_w)
        weights = np.exp(log_w_norm)
        weights = np.clip(weights, 0, None)
        weights /= weights.sum()

        # Systematic resampling
        cumw = np.cumsum(weights)
        u = (self._rng.random() + np.arange(N)) / N
        indices = np.searchsorted(cumw, u)
        indices = np.clip(indices, 0, N - 1)

        new_particles = []
        log_w_new = -np.log(N)
        for idx in indices:
            p = self._particles[idx].copy()
            p.log_weight = log_w_new
            new_particles.append(p)
        self._particles = new_particles

    def online_update(self, x_t: float) -> NDArray:
        """Process one observation and return the regime posterior.

        Parameters
        ----------
        x_t : float
            New observation.

        Returns
        -------
        posterior : (n_regimes,) array
            P(z_t = k | x_{1:t}) for each regime k.
        """
        self._t += 1

        if not self._particles:
            self._initialize_particles()
            # Initialize emission stats from first observation
            for k in range(self.n_regimes):
                self._regime_means[k] = x_t + self._rng.normal(0, 0.1)
                self._regime_vars[k] = 1.0

        # Propagate and weight each particle
        for p in self._particles:
            # Transition
            row = self._transition_matrix[p.state]
            row = np.clip(row, 0, None)
            row /= row.sum()
            new_state = self._rng.choice(self.n_regimes, p=row)

            # Weight update
            log_w_inc = self._emission_log_prob(x_t, new_state)
            p.log_weight += log_w_inc
            p.state = new_state
            p.run_length += 1

            # Update per-particle emission stats
            p.emission_stats["n"] += 1
            p.emission_stats["sum"] += x_t
            p.emission_stats["sum_sq"] += x_t * x_t

        # Normalize weights
        log_w = np.array([p.log_weight for p in self._particles])
        log_w -= logsumexp(log_w)
        for i, p in enumerate(self._particles):
            p.log_weight = log_w[i]

        # Resample if ESS is low
        ess = self._effective_sample_size()
        if ess < self.resample_threshold * self.n_particles:
            self._systematic_resample()

        # Compute posterior
        posterior = np.zeros(self.n_regimes)
        for p in self._particles:
            posterior[p.state] += np.exp(p.log_weight)
        posterior = np.clip(posterior, 0, None)
        total = posterior.sum()
        if total > 0:
            posterior /= total
        else:
            posterior = np.ones(self.n_regimes) / self.n_regimes
        self._posteriors.append(posterior.copy())

        # Update global emission stats
        map_state = int(np.argmax(posterior))
        self._update_emission_stats(x_t, map_state)

        # Check for change point
        self._bocpd.update(x_t)
        cusum_cp = self._cusum.update(x_t)
        bocpd_cp = self._bocpd.change_probs[-1] > 0.5 if self._bocpd.change_probs else False
        if cusum_cp or bocpd_cp:
            self._change_points.append(self._t)

        return posterior

    # ------------------------------------------------------------------
    # detect (batch)
    # ------------------------------------------------------------------

    def detect(self, X: NDArray) -> Tuple[NDArray, NDArray]:
        """Run regime detection on a full time series.

        Parameters
        ----------
        X : array-like of shape (T,) or (T, D)
            Observed time series.

        Returns
        -------
        states : (T,) int array of MAP regime assignments
        posteriors : (T, K) regime posteriors
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim > 1:
            X = X[:, 0]  # use first dimension for univariate detection
        T = len(X)

        # Fit emission model via K-means-like initialization
        sorted_x = np.sort(X)
        quantiles = np.linspace(0, 1, self.n_regimes + 2)[1:-1]
        init_means = np.quantile(sorted_x, quantiles)
        self._regime_means = init_means
        self._regime_vars = np.full(self.n_regimes, np.var(X) + 1e-6)
        self._regime_counts = np.ones(self.n_regimes)

        # Build transition matrix with sticky prior
        stick = 0.7
        off = (1.0 - stick) / max(self.n_regimes - 1, 1)
        A = np.full((self.n_regimes, self.n_regimes), off)
        np.fill_diagonal(A, stick)
        self._transition_matrix = A

        # Reset and run particle filter
        self._particles = []
        self._posteriors = []
        self._change_points = []
        self._t = 0
        self._cusum.reset()
        self._bocpd.reset()

        for t in range(T):
            self.online_update(X[t])

        posteriors = np.array(self._posteriors)
        states = np.argmax(posteriors, axis=1).astype(np.int64)
        self._is_fitted = True
        return states, posteriors

    # ------------------------------------------------------------------
    # getters
    # ------------------------------------------------------------------

    def get_posterior(self) -> NDArray:
        """Return the full sequence of regime posteriors (T, K)."""
        if not self._posteriors:
            return np.empty((0, self.n_regimes))
        return np.array(self._posteriors)

    def get_change_points(self) -> List[int]:
        """Return detected change-point indices."""
        return list(self._change_points)

    def get_regime_durations(self) -> Dict[int, List[int]]:
        """Compute duration statistics for each regime.

        Returns
        -------
        durations : dict mapping regime index to list of run lengths
        """
        if not self._posteriors:
            return {}
        states = np.argmax(np.array(self._posteriors), axis=1)
        durations: Dict[int, List[int]] = {k: [] for k in range(self.n_regimes)}
        current_state = states[0]
        run_len = 1
        for t in range(1, len(states)):
            if states[t] == current_state:
                run_len += 1
            else:
                durations[current_state].append(run_len)
                current_state = states[t]
                run_len = 1
        durations[current_state].append(run_len)
        return durations

    def get_regime_statistics(self) -> Dict[int, Dict[str, float]]:
        """Summary statistics for each detected regime.

        Returns
        -------
        stats : dict of regime → {mean_duration, frequency, ...}
        """
        durations = self.get_regime_durations()
        stats_out: Dict[int, Dict[str, float]] = {}
        total_t = sum(sum(v) for v in durations.values())
        for k in range(self.n_regimes):
            durs = durations.get(k, [])
            if durs:
                stats_out[k] = {
                    "mean_duration": float(np.mean(durs)),
                    "median_duration": float(np.median(durs)),
                    "max_duration": float(np.max(durs)),
                    "n_visits": len(durs),
                    "frequency": float(sum(durs)) / max(total_t, 1),
                    "emission_mean": float(self._regime_means[k]),
                    "emission_var": float(self._regime_vars[k]),
                }
            else:
                stats_out[k] = {
                    "mean_duration": 0.0,
                    "median_duration": 0.0,
                    "max_duration": 0.0,
                    "n_visits": 0,
                    "frequency": 0.0,
                    "emission_mean": float(self._regime_means[k]),
                    "emission_var": float(self._regime_vars[k]),
                }
        return stats_out

    def __repr__(self) -> str:
        return (
            f"BayesianRegimeDetector(n_regimes={self.n_regimes}, "
            f"n_particles={self.n_particles}, "
            f"hazard_rate={self.hazard_rate:.4f})"
        )
