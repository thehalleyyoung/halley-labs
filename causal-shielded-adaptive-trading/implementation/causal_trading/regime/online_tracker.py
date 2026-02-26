"""
Online regime tracking with sequential inference, exponential forgetting,
sliding-window estimation, and regime-change alerting.

Designed for real-time trading applications where the regime posterior
must be updated incrementally as new market data arrives.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.special import logsumexp, gammaln


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegimeChangeAlert:
    """Represents a detected regime change event."""
    time_index: int
    old_regime: int
    new_regime: int
    confidence: float
    posterior: NDArray
    description: str = ""


# ---------------------------------------------------------------------------
# Sliding window regime estimator
# ---------------------------------------------------------------------------

class SlidingWindowEstimator:
    """Estimates regime parameters within a sliding window.

    Maintains a circular buffer of observations and computes
    emission statistics (mean, variance) incrementally.

    Parameters
    ----------
    window_size : int
        Number of observations in the window.
    n_regimes : int
        Number of regimes.
    """

    def __init__(self, window_size: int = 200, n_regimes: int = 5) -> None:
        self.window_size = window_size
        self.n_regimes = n_regimes
        self._buffer: deque = deque(maxlen=window_size)
        self._state_buffer: deque = deque(maxlen=window_size)

        # Per-regime sufficient statistics within the window
        self._means = np.zeros(n_regimes)
        self._m2s = np.zeros(n_regimes)   # sum of squared deviations
        self._counts = np.zeros(n_regimes, dtype=np.int64)

    def update(self, x: float, state: int) -> None:
        """Add an observation to the window and update statistics."""
        # If the buffer is full, remove the oldest observation
        if len(self._buffer) == self.window_size:
            old_x = self._buffer[0]
            old_s = self._state_buffer[0]
            self._remove_from_stats(old_x, old_s)

        self._buffer.append(x)
        self._state_buffer.append(state)
        self._add_to_stats(x, state)

    def _add_to_stats(self, x: float, state: int) -> None:
        """Welford-style incremental update."""
        n = self._counts[state]
        self._counts[state] += 1
        n_new = self._counts[state]
        delta = x - self._means[state]
        self._means[state] += delta / n_new
        delta2 = x - self._means[state]
        self._m2s[state] += delta * delta2

    def _remove_from_stats(self, x: float, state: int) -> None:
        """Welford-style reverse update for removing an observation."""
        n = self._counts[state]
        if n <= 1:
            self._means[state] = 0.0
            self._m2s[state] = 0.0
            self._counts[state] = 0
            return
        delta = x - self._means[state]
        self._counts[state] -= 1
        n_new = self._counts[state]
        self._means[state] -= delta / n_new
        delta2 = x - self._means[state]
        self._m2s[state] -= delta * delta2

    def get_means(self) -> NDArray:
        return self._means.copy()

    def get_variances(self) -> NDArray:
        vars_ = np.zeros(self.n_regimes)
        for k in range(self.n_regimes):
            if self._counts[k] > 1:
                vars_[k] = self._m2s[k] / (self._counts[k] - 1)
            else:
                vars_[k] = 1.0
        return vars_

    def get_counts(self) -> NDArray:
        return self._counts.copy()

    def get_frequencies(self) -> NDArray:
        total = self._counts.sum()
        if total == 0:
            return np.ones(self.n_regimes) / self.n_regimes
        return self._counts.astype(np.float64) / total

    def get_window_data(self) -> NDArray:
        return np.array(self._buffer, dtype=np.float64)


# ---------------------------------------------------------------------------
# Exponential forgetting filter
# ---------------------------------------------------------------------------

class ExponentialForgettingFilter:
    """Maintains regime posteriors with exponential forgetting.

    At each step, the effective prior is a geometric mixture of
    the previous posterior and a uniform distribution, controlled
    by the forgetting factor lambda in (0, 1].

    P_eff(z_t) ∝ P(z_t | x_{1:t-1})^lambda

    Parameters
    ----------
    n_regimes : int
        Number of regimes.
    forgetting_factor : float
        Lambda in (0, 1].  Values near 1 = long memory;
        values near 0 = rapid adaptation.
    """

    def __init__(self, n_regimes: int = 5, forgetting_factor: float = 0.98) -> None:
        self.n_regimes = n_regimes
        self.forgetting_factor = forgetting_factor
        self._log_posterior = np.full(n_regimes, -np.log(n_regimes))

    def predict(self, log_A: NDArray) -> NDArray:
        """Prediction step: propagate through transition matrix.

        Parameters
        ----------
        log_A : (K, K) log transition matrix

        Returns
        -------
        log_prior : (K,) predicted log-prior for next step
        """
        K = self.n_regimes
        log_prior = np.full(K, -np.inf)
        for k in range(K):
            log_prior[k] = logsumexp(self._log_posterior + log_A[:, k])
        # Apply forgetting
        log_prior *= self.forgetting_factor
        # Add uniform component
        uniform = np.full(K, -np.log(K))
        log_prior = np.logaddexp(
            log_prior + np.log(self.forgetting_factor + 1e-300),
            uniform + np.log(1.0 - self.forgetting_factor + 1e-300),
        )
        return log_prior - logsumexp(log_prior)

    def update(self, log_likelihood: NDArray, log_A: NDArray) -> NDArray:
        """Full predict-update cycle.

        Parameters
        ----------
        log_likelihood : (K,) log-emission probability for current obs
        log_A : (K, K) log transition matrix

        Returns
        -------
        posterior : (K,) normalised posterior probabilities
        """
        log_prior = self.predict(log_A)
        log_post = log_prior + log_likelihood
        log_post -= logsumexp(log_post)
        self._log_posterior = log_post
        return np.exp(log_post)

    def get_posterior(self) -> NDArray:
        return np.exp(self._log_posterior)

    def reset(self) -> None:
        self._log_posterior = np.full(self.n_regimes, -np.log(self.n_regimes))


# ---------------------------------------------------------------------------
# Main OnlineRegimeTracker
# ---------------------------------------------------------------------------

class OnlineRegimeTracker:
    """Online regime tracker for sequential regime inference.

    Combines a forward-filtering approach with exponential forgetting
    and sliding-window emission estimation.  Produces regime posteriors
    and change-point alerts in real time.

    Parameters
    ----------
    n_regimes : int
        Maximum number of regimes.
    window_size : int
        Sliding window size for emission estimation.
    forgetting_factor : float
        Exponential forgetting factor (0, 1].
    alert_threshold : float
        Posterior probability threshold to trigger a regime-change alert.
    transition_prior : float
        Sticky diagonal prior for the transition matrix.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_regimes: int = 5,
        window_size: int = 200,
        forgetting_factor: float = 0.98,
        alert_threshold: float = 0.7,
        transition_prior: float = 0.8,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_regimes = n_regimes
        self.window_size = window_size
        self.forgetting_factor = forgetting_factor
        self.alert_threshold = alert_threshold
        self.transition_prior = transition_prior
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)
        self._filter = ExponentialForgettingFilter(n_regimes, forgetting_factor)
        self._window = SlidingWindowEstimator(window_size, n_regimes)

        # Transition matrix (sticky prior)
        off_diag = (1.0 - transition_prior) / max(n_regimes - 1, 1)
        A = np.full((n_regimes, n_regimes), off_diag)
        np.fill_diagonal(A, transition_prior)
        self._A = A
        self._log_A = np.log(A + 1e-300)

        # Emission parameters (initialised lazily)
        self._emission_means: Optional[NDArray] = None
        self._emission_vars: Optional[NDArray] = None
        self._initialised = False

        # History
        self._posteriors: List[NDArray] = []
        self._map_states: List[int] = []
        self._alerts: List[RegimeChangeAlert] = []
        self._observations: List[float] = []
        self._t: int = 0
        self._prev_regime: int = -1

        # Transition counts for online Bayesian updating
        self._trans_counts = np.zeros((n_regimes, n_regimes))
        self._trans_prior = self._build_transition_prior()

    def _build_transition_prior(self) -> NDArray:
        """Dirichlet prior for transition matrix rows."""
        K = self.n_regimes
        prior = np.ones((K, K))
        prior += self.transition_prior * np.eye(K) * 5.0
        return prior

    def _initialise_emissions(self, x: float) -> None:
        """Lazy initialisation of emission parameters around first obs."""
        K = self.n_regimes
        spread = max(abs(x) * 0.5, 1.0)
        self._emission_means = np.linspace(
            x - spread, x + spread, K
        )
        self._emission_vars = np.full(K, spread ** 2)
        self._initialised = True

    def _emission_log_prob(self, x: float) -> NDArray:
        """Log-probability of x under each regime's Gaussian emission."""
        K = self.n_regimes
        log_probs = np.zeros(K)
        for k in range(K):
            mu = self._emission_means[k]
            var = max(self._emission_vars[k], 1e-6)
            log_probs[k] = -0.5 * np.log(2 * np.pi * var) - 0.5 * (x - mu) ** 2 / var
        return log_probs

    def _update_transition_matrix(self) -> None:
        """Update transition matrix using Bayesian Dirichlet posterior."""
        alpha_post = self._trans_prior + self._trans_counts
        for i in range(self.n_regimes):
            row = alpha_post[i]
            row_sum = row.sum()
            if row_sum > 0:
                self._A[i] = row / row_sum
        self._log_A = np.log(self._A + 1e-300)

    def _update_emissions_from_window(self) -> None:
        """Refresh emission parameters from the sliding window."""
        means = self._window.get_means()
        vars_ = self._window.get_variances()
        counts = self._window.get_counts()
        for k in range(self.n_regimes):
            if counts[k] > 5:
                self._emission_means[k] = means[k]
                self._emission_vars[k] = max(vars_[k], 1e-6)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, x_t: float) -> NDArray:
        """Process one observation and return the regime posterior.

        Parameters
        ----------
        x_t : float
            New observation.

        Returns
        -------
        posterior : (n_regimes,) posterior probabilities
        """
        self._t += 1
        self._observations.append(x_t)

        if not self._initialised:
            self._initialise_emissions(x_t)

        # Compute emission log-likelihoods
        log_lik = self._emission_log_prob(x_t)

        # Forward filter step with exponential forgetting
        posterior = self._filter.update(log_lik, self._log_A)
        self._posteriors.append(posterior.copy())

        # MAP state
        map_state = int(np.argmax(posterior))
        self._map_states.append(map_state)

        # Update sliding window
        self._window.update(x_t, map_state)

        # Update transition counts
        if self._prev_regime >= 0:
            self._trans_counts[self._prev_regime, map_state] += 1

        # Check for regime change
        if self._prev_regime >= 0 and map_state != self._prev_regime:
            confidence = float(posterior[map_state])
            if confidence >= self.alert_threshold:
                alert = RegimeChangeAlert(
                    time_index=self._t,
                    old_regime=self._prev_regime,
                    new_regime=map_state,
                    confidence=confidence,
                    posterior=posterior.copy(),
                    description=(
                        f"Regime change {self._prev_regime}->{map_state} "
                        f"at t={self._t} (conf={confidence:.3f})"
                    ),
                )
                self._alerts.append(alert)

        self._prev_regime = map_state

        # Periodically refresh emissions and transition matrix
        if self._t % 50 == 0:
            self._update_emissions_from_window()
            self._update_transition_matrix()

        return posterior

    def batch_update(self, X: NDArray) -> NDArray:
        """Process a batch of observations.

        Parameters
        ----------
        X : (T,) or (T, 1) array

        Returns
        -------
        posteriors : (T, n_regimes) posterior probabilities
        """
        X = np.asarray(X, dtype=np.float64).ravel()
        posteriors = np.zeros((len(X), self.n_regimes))
        for t, x in enumerate(X):
            posteriors[t] = self.update(x)
        return posteriors

    def get_current_regime(self) -> int:
        """Return the current MAP regime assignment."""
        if not self._map_states:
            return -1
        return self._map_states[-1]

    def get_current_posterior(self) -> NDArray:
        """Return the current regime posterior."""
        return self._filter.get_posterior()

    def get_regime_history(self) -> NDArray:
        """Return the full MAP state history."""
        return np.array(self._map_states, dtype=np.int64)

    def get_posterior_history(self) -> NDArray:
        """Return the full posterior history (T, K)."""
        if not self._posteriors:
            return np.empty((0, self.n_regimes))
        return np.array(self._posteriors)

    def get_alerts(self) -> List[RegimeChangeAlert]:
        """Return all regime change alerts."""
        return list(self._alerts)

    def get_recent_alerts(self, n: int = 5) -> List[RegimeChangeAlert]:
        """Return the n most recent alerts."""
        return self._alerts[-n:]

    def get_transition_matrix(self) -> NDArray:
        """Return the current transition matrix estimate."""
        return self._A.copy()

    def get_regime_statistics(self) -> Dict[int, Dict[str, float]]:
        """Compute summary statistics for each regime."""
        states = np.array(self._map_states)
        obs = np.array(self._observations)
        result: Dict[int, Dict[str, float]] = {}
        for k in range(self.n_regimes):
            mask = states == k
            count = int(mask.sum())
            if count > 0:
                regime_obs = obs[mask]
                result[k] = {
                    "count": count,
                    "frequency": count / len(states),
                    "mean_return": float(regime_obs.mean()),
                    "volatility": float(regime_obs.std()),
                    "min": float(regime_obs.min()),
                    "max": float(regime_obs.max()),
                    "skewness": float(stats.skew(regime_obs)) if count > 2 else 0.0,
                    "kurtosis": float(stats.kurtosis(regime_obs)) if count > 3 else 0.0,
                }
            else:
                result[k] = {
                    "count": 0, "frequency": 0.0,
                    "mean_return": 0.0, "volatility": 0.0,
                    "min": 0.0, "max": 0.0,
                    "skewness": 0.0, "kurtosis": 0.0,
                }
        return result

    def regime_persistence(self) -> Dict[int, float]:
        """Compute average regime persistence (mean duration in each regime)."""
        states = np.array(self._map_states)
        if len(states) == 0:
            return {}
        durations: Dict[int, List[int]] = {k: [] for k in range(self.n_regimes)}
        current = states[0]
        run = 1
        for t in range(1, len(states)):
            if states[t] == current:
                run += 1
            else:
                durations[current].append(run)
                current = states[t]
                run = 1
        durations[current].append(run)
        return {
            k: float(np.mean(v)) if v else 0.0
            for k, v in durations.items()
        }

    def entropy_of_posterior(self) -> float:
        """Shannon entropy of the current posterior (uncertainty measure)."""
        p = self._filter.get_posterior()
        p = np.clip(p, 1e-300, None)
        return float(-np.sum(p * np.log(p)))

    def posterior_entropy_history(self) -> NDArray:
        """Entropy of the posterior at each time step."""
        if not self._posteriors:
            return np.array([])
        posteriors = np.array(self._posteriors)
        posteriors = np.clip(posteriors, 1e-300, None)
        return -np.sum(posteriors * np.log(posteriors), axis=1)

    def reset(self) -> None:
        """Reset the tracker to initial state."""
        self._filter.reset()
        self._window = SlidingWindowEstimator(self.window_size, self.n_regimes)
        self._posteriors = []
        self._map_states = []
        self._alerts = []
        self._observations = []
        self._t = 0
        self._prev_regime = -1
        self._trans_counts = np.zeros((self.n_regimes, self.n_regimes))
        self._initialised = False

    def __repr__(self) -> str:
        return (
            f"OnlineRegimeTracker(n_regimes={self.n_regimes}, "
            f"window={self.window_size}, "
            f"lambda={self.forgetting_factor}, "
            f"t={self._t})"
        )
