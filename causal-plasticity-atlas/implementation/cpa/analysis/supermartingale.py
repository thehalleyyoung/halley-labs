"""Supermartingale stopping criterion.

Provides an anytime-valid sequential test that monitors cumulative
improvement and decides when further search iterations are unlikely
to yield meaningful gains.  Based on testing-by-betting / e-process
ideas (Ville's inequality and e-values).

The wealth process is:

    W_t = W_{t-1} * (1 + lambda_t * (X_t - mu_0))

where lambda_t is the betting fraction, X_t is the observed
improvement, and mu_0 is the null hypothesis mean.  We reject
(stop) when W_t >= 1 / alpha.

The module also provides:

* :func:`confidence_sequence` – anytime-valid confidence sequence
  for the mean improvement.
* :class:`SupermartingaleStopper` – main tracker class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats


# ===================================================================
# Dataclass
# ===================================================================


@dataclass
class StoppingResult:
    """Result of a stopping-criterion evaluation.

    Attributes
    ----------
    should_stop : bool
        Whether the criterion recommends stopping.
    confidence : float
        Confidence level associated with the decision.
    expected_improvement : float
        Estimated expected improvement from continuing.
    n_iterations : int
        Number of iterations observed so far.
    wealth : float
        Current wealth of the betting process.
    wealth_history : list of float
        Full wealth trajectory.
    """

    should_stop: bool
    confidence: float
    expected_improvement: float
    n_iterations: int
    wealth: float = 1.0
    wealth_history: List[float] = field(default_factory=list)


# ===================================================================
# SupermartingaleStopper
# ===================================================================


class SupermartingaleStopper:
    """Anytime-valid stopping rule based on a supermartingale wealth process.

    The null hypothesis is *"the mean improvement is >= mu_0"*
    (i.e. the search is still making progress).  The wealth process
    bets against this null; when wealth reaches 1/alpha we reject
    the null and declare convergence (stop).

    Parameters
    ----------
    alpha : float
        Significance level (type-I error bound).
    wealth_init : float
        Initial wealth of the betting process.
    mu_0 : float
        Null-hypothesis mean improvement.  Default 0.0 means we test
        whether mean improvement is significantly > 0.
    lambda_max : float
        Maximum absolute betting fraction.
    lookback : int
        Number of past observations used to estimate the optimal
        betting fraction.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        wealth_init: float = 1.0,
        mu_0: float = 0.0,
        lambda_max: float = 0.5,
        lookback: int = 50,
    ) -> None:
        if alpha <= 0 or alpha >= 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._alpha = alpha
        self._wealth_init = wealth_init
        self._mu_0 = mu_0
        self._lambda_max = lambda_max
        self._lookback = lookback

        self._current_wealth: float = wealth_init
        self._n_iterations: int = 0
        self._history: List[float] = []
        self._wealth_history: List[float] = [wealth_init]
        self._stopped: bool = False

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def update(self, improvement: float) -> StoppingResult:
        """Observe a new improvement value and evaluate stopping.

        Parameters
        ----------
        improvement : float
            Improvement observed in this iteration (can be negative
            if quality regressed).

        Returns
        -------
        StoppingResult
        """
        self._history.append(float(improvement))
        self._n_iterations += 1

        lam = self._betting_fraction(self._history)

        centred = improvement - self._mu_0
        factor = 1.0 + lam * centred
        factor = max(factor, 1e-15)

        self._current_wealth *= factor
        self._wealth_history.append(self._current_wealth)

        threshold = 1.0 / self._alpha
        if self._current_wealth >= threshold:
            self._stopped = True

        conf = min(1.0, self._current_wealth * self._alpha)

        exp_imp = self._expected_improvement()

        return StoppingResult(
            should_stop=self._stopped,
            confidence=conf,
            expected_improvement=exp_imp,
            n_iterations=self._n_iterations,
            wealth=self._current_wealth,
            wealth_history=list(self._wealth_history),
        )

    def wealth(self) -> float:
        """Return the current wealth of the betting process."""
        return self._current_wealth

    def should_stop(self) -> bool:
        """Return whether the process should stop now."""
        return self._stopped

    def reset(self) -> None:
        """Reset the stopper to its initial state."""
        self._current_wealth = self._wealth_init
        self._n_iterations = 0
        self._history.clear()
        self._wealth_history = [self._wealth_init]
        self._stopped = False

    # -----------------------------------------------------------------
    # Betting fraction strategies
    # -----------------------------------------------------------------

    def _betting_fraction(self, history: List[float]) -> float:
        """Compute the betting fraction based on recent observations.

        Uses the *ONS (Online Newton Step)* inspired strategy:
        bet proportionally to the running z-score of the observations
        relative to mu_0, clipped to [-lambda_max, lambda_max].

        Parameters
        ----------
        history : list of float
            All observed improvements so far.

        Returns
        -------
        float
            Betting fraction in [-lambda_max, lambda_max].
        """
        n = len(history)
        if n < 2:
            return self._lambda_max * 0.1

        recent = history[-self._lookback :]
        arr = np.asarray(recent, dtype=np.float64)
        centred = arr - self._mu_0
        mean_c = np.mean(centred)
        std_c = np.std(centred, ddof=1)
        if std_c < 1e-12:
            std_c = 1e-12

        z = mean_c / std_c
        lam = np.clip(z * 0.5 / np.sqrt(n), -self._lambda_max, self._lambda_max)
        return float(lam)

    def _expected_improvement(self) -> float:
        """Estimate expected improvement from recent observations.

        Uses exponentially weighted moving average of the last
        ``lookback`` observations.

        Returns
        -------
        float
        """
        if len(self._history) == 0:
            return 0.0
        recent = self._history[-self._lookback :]
        arr = np.asarray(recent, dtype=np.float64)
        n = len(arr)
        if n == 0:
            return 0.0
        decay = 0.95
        weights = np.array([decay ** (n - 1 - i) for i in range(n)])
        weights /= np.sum(weights)
        return float(np.dot(weights, arr))

    # -----------------------------------------------------------------
    # Wealth process helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _wealth_process(
        values: Sequence[float],
        mu_0: float = 0.0,
        lam: float = 0.1,
        w0: float = 1.0,
    ) -> NDArray:
        """Compute the wealth process for constant betting fraction.

        Parameters
        ----------
        values : sequence of float
            Observed improvements.
        mu_0 : float
            Null-hypothesis mean.
        lam : float
            Constant betting fraction.
        w0 : float
            Initial wealth.

        Returns
        -------
        NDArray
            Wealth process of length ``len(values) + 1``.
        """
        arr = np.asarray(values, dtype=np.float64)
        wealth = np.empty(len(arr) + 1, dtype=np.float64)
        wealth[0] = w0
        for t, x in enumerate(arr):
            factor = 1.0 + lam * (x - mu_0)
            factor = max(factor, 1e-15)
            wealth[t + 1] = wealth[t] * factor
        return wealth

    # -----------------------------------------------------------------
    # Confidence sequence
    # -----------------------------------------------------------------

    @staticmethod
    def confidence_sequence(
        values: Sequence[float],
        alpha: float = 0.05,
        v_opt: float = 1.0,
    ) -> Tuple[NDArray, NDArray]:
        """Anytime-valid confidence sequence for the running mean.

        Uses a sub-Gaussian mixture martingale to produce a
        ``(1-alpha)``-confidence sequence that is valid at every
        sample size simultaneously.

        The half-width at time *t* is::

            c_t = sqrt( 2 * v_opt * log(log(max(t,e)) / alpha) / t )

        Parameters
        ----------
        values : sequence of float
            Observed values.
        alpha : float
            Significance level.
        v_opt : float
            Sub-Gaussian variance proxy.

        Returns
        -------
        (lower, upper) : tuple of NDArray
            Lower and upper confidence bounds at each time step
            (length ``len(values)``).
        """
        arr = np.asarray(values, dtype=np.float64)
        n = len(arr)
        if n == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        running_mean = np.cumsum(arr) / np.arange(1, n + 1)
        t_vals = np.arange(1, n + 1, dtype=np.float64)

        log_term = np.log(np.maximum(np.log(np.maximum(t_vals, np.e)), 1.0) / alpha)
        half_width = np.sqrt(2.0 * v_opt * log_term / t_vals)

        lower = running_mean - half_width
        upper = running_mean + half_width
        return lower, upper

    # -----------------------------------------------------------------
    # Mixture martingale
    # -----------------------------------------------------------------

    @staticmethod
    def _mixture_martingale(
        values: Sequence[float],
        mu_0: float = 0.0,
        sigma: float = 1.0,
        n_lambdas: int = 20,
    ) -> NDArray:
        """Mixture martingale for robust convergence detection.

        Averages the wealth process over a grid of betting fractions
        to produce a martingale that does not require tuning lambda.

        Parameters
        ----------
        values : sequence of float
            Observed values.
        mu_0 : float
            Null-hypothesis mean.
        sigma : float
            Scale parameter for the mixing distribution.
        n_lambdas : int
            Number of lambda values in the grid.

        Returns
        -------
        NDArray
            Mixture wealth process of length ``len(values) + 1``.
        """
        arr = np.asarray(values, dtype=np.float64)
        T = len(arr)
        lambdas = np.linspace(-0.5, 0.5, n_lambdas)

        individual = np.ones((n_lambdas, T + 1), dtype=np.float64)
        for li, lam in enumerate(lambdas):
            for t in range(T):
                factor = 1.0 + lam * (arr[t] - mu_0)
                factor = max(factor, 1e-15)
                individual[li, t + 1] = individual[li, t] * factor

        prior = np.exp(-0.5 * (lambdas / sigma) ** 2)
        prior /= prior.sum()

        mixture = np.zeros(T + 1, dtype=np.float64)
        for li in range(n_lambdas):
            mixture += prior[li] * individual[li]

        return mixture
