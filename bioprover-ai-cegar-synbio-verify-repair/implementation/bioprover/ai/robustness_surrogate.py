"""Robustness surrogate model for BioProver parameter repair.

Provides a Gaussian process regression surrogate predicting STL robustness
from circuit parameters, plus Bayesian optimisation routines for parameter
repair.  Implemented with NumPy / SciPy only (no GPyTorch / BoTorch).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize as scipy_minimize
from scipy.spatial.distance import cdist

from bioprover.soundness import SoundnessAnnotation, SoundnessLevel

logger = logging.getLogger(__name__)

_EPS = 1e-8
_JITTER = 1e-6


# ---------------------------------------------------------------------------
# SurrogateVerdict — prediction with uncertainty and soundness
# ---------------------------------------------------------------------------


@dataclass
class SurrogateVerdict:
    """Surrogate prediction with uncertainty quantification and soundness.

    Includes the robustness estimate, confidence interval, and a soundness
    annotation indicating that the result is APPROXIMATE.
    """
    robustness: float
    std: float
    confidence_lo: float
    confidence_hi: float
    soundness: SoundnessAnnotation = field(default_factory=lambda: SoundnessAnnotation(
        level=SoundnessLevel.APPROXIMATE,
        assumptions=["GP surrogate prediction — not formally verified"],
    ))

    @property
    def is_robust(self) -> bool:
        """Conservative check: robustness > 0 with high confidence."""
        return self.confidence_lo > 0.0

    @property
    def is_violated(self) -> bool:
        """Conservative check: robustness < 0 with high confidence."""
        return self.confidence_hi < 0.0

# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------


class RBFKernel:
    """Radial basis function (squared-exponential) kernel.

    k(x, x') = σ² · exp(-||x - x'||² / (2 l²))
    """

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0) -> None:
        self.length_scale = length_scale
        self.variance = variance

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        sq_dist = cdist(X1, X2, metric="sqeuclidean")
        return self.variance * np.exp(-sq_dist / (2.0 * self.length_scale ** 2))

    def diagonal(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.variance, dtype=np.float64)

    @property
    def hyperparameters(self) -> np.ndarray:
        return np.array([np.log(self.length_scale), np.log(self.variance)])

    @hyperparameters.setter
    def hyperparameters(self, log_hp: np.ndarray) -> None:
        self.length_scale = float(np.exp(log_hp[0]))
        self.variance = float(np.exp(log_hp[1]))

    @property
    def n_hyperparameters(self) -> int:
        return 2


class MaternKernel:
    """Matérn kernel with ν = 5/2.

    k(x, x') = σ² (1 + √5 r/l + 5 r²/(3 l²)) exp(-√5 r/l)
    where r = ||x - x'||
    """

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0) -> None:
        self.length_scale = length_scale
        self.variance = variance

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        r = cdist(X1, X2, metric="euclidean")
        sqrt5_r = math.sqrt(5.0) * r / self.length_scale
        return self.variance * (1.0 + sqrt5_r + sqrt5_r ** 2 / 3.0) * np.exp(-sqrt5_r)

    def diagonal(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.variance, dtype=np.float64)

    @property
    def hyperparameters(self) -> np.ndarray:
        return np.array([np.log(self.length_scale), np.log(self.variance)])

    @hyperparameters.setter
    def hyperparameters(self, log_hp: np.ndarray) -> None:
        self.length_scale = float(np.exp(log_hp[0]))
        self.variance = float(np.exp(log_hp[1]))

    @property
    def n_hyperparameters(self) -> int:
        return 2


# ---------------------------------------------------------------------------
# Gaussian process regression
# ---------------------------------------------------------------------------


class GaussianProcessRegressor:
    """Exact GP regression with Cholesky-based inference.

    Parameters
    ----------
    kernel : RBFKernel or MaternKernel
    noise_variance : float
        Observation noise σ²_n.
    """

    def __init__(
        self,
        kernel: Any = None,
        noise_variance: float = 0.01,
    ) -> None:
        self.kernel = kernel or RBFKernel()
        self.noise_variance = noise_variance
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._L: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._y_mean: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the GP to training data.

        Parameters
        ----------
        X : (n, d)
        y : (n,)
        """
        self._X_train = X.copy()
        self._y_mean = float(y.mean())
        y_centered = y - self._y_mean
        self._y_train = y_centered.copy()

        K = self.kernel(X, X)
        K += self.noise_variance * np.eye(K.shape[0])
        K += _JITTER * np.eye(K.shape[0])

        self._L, _ = cho_factor(K, lower=True)
        self._alpha = cho_solve((self._L, True), y_centered)

    def predict(
        self, X_new: np.ndarray, return_std: bool = False,
    ) -> Any:
        """Predict at new inputs.

        Parameters
        ----------
        X_new : (m, d)
        return_std : if True return (mean, std)

        Returns
        -------
        mean : (m,)
        std  : (m,) – only if return_std is True
        """
        if self._X_train is None:
            mean = np.full(X_new.shape[0], self._y_mean)
            if return_std:
                return mean, np.ones(X_new.shape[0])
            return mean

        K_star = self.kernel(X_new, self._X_train)  # (m, n)
        mean = K_star @ self._alpha + self._y_mean

        if not return_std:
            return mean

        v = cho_solve((self._L, True), K_star.T)  # (n, m)
        K_ss = self.kernel.diagonal(X_new)
        var = K_ss - np.sum(K_star.T * v, axis=0)
        var = np.maximum(var, 0.0)
        return mean, np.sqrt(var)

    def log_marginal_likelihood(self) -> float:
        """Compute log p(y | X, θ)."""
        if self._y_train is None:
            return -float("inf")
        n = len(self._y_train)
        # -0.5 y^T K^{-1} y  - 0.5 log|K| - n/2 log(2π)
        data_fit = -0.5 * self._y_train @ self._alpha
        log_det = np.sum(np.log(np.diag(self._L)))
        return float(data_fit - log_det - 0.5 * n * np.log(2.0 * np.pi))

    def optimize_hyperparameters(
        self, X: np.ndarray, y: np.ndarray, n_restarts: int = 3,
        rng: Optional[np.random.RandomState] = None,
    ) -> float:
        """Optimise kernel hyperparameters by maximising log marginal likelihood.

        Returns the best log marginal likelihood found.
        """
        if rng is None:
            rng = np.random.RandomState(42)

        best_lml = -float("inf")
        best_hp: Optional[np.ndarray] = None
        init_hp = self.kernel.hyperparameters.copy()

        for restart in range(n_restarts):
            if restart == 0:
                hp0 = init_hp.copy()
            else:
                hp0 = init_hp + rng.randn(len(init_hp)) * 0.5

            def neg_lml(log_hp: np.ndarray) -> float:
                self.kernel.hyperparameters = log_hp
                self.fit(X, y)
                return -self.log_marginal_likelihood()

            result = scipy_minimize(neg_lml, hp0, method="L-BFGS-B")
            if -result.fun > best_lml:
                best_lml = -result.fun
                best_hp = result.x.copy()

        if best_hp is not None:
            self.kernel.hyperparameters = best_hp
            self.fit(X, y)
        return best_lml

    def update(self, X_new: np.ndarray, y_new: np.ndarray) -> None:
        """Incrementally add observations and refit."""
        if self._X_train is None:
            self.fit(X_new, y_new)
            return
        X_all = np.vstack([self._X_train, X_new])
        y_all = np.concatenate([self._y_train + self._y_mean, y_new])
        self.fit(X_all, y_all)


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------


def expected_improvement(
    X: np.ndarray,
    gp: GaussianProcessRegressor,
    y_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected Improvement acquisition function.

    Parameters
    ----------
    X : (m, d) query points
    gp : fitted GP
    y_best : best observed value (to maximise)
    xi : exploration trade-off

    Returns
    -------
    ei : (m,)
    """
    mean, std = gp.predict(X, return_std=True)
    std = np.maximum(std, _EPS)
    z = (mean - y_best - xi) / std
    ei = std * (_standard_normal_pdf(z) + z * _standard_normal_cdf(z))
    return ei


def upper_confidence_bound(
    X: np.ndarray,
    gp: GaussianProcessRegressor,
    beta: float = 2.0,
) -> np.ndarray:
    """Upper Confidence Bound acquisition function.

    Parameters
    ----------
    X : (m, d)
    gp : fitted GP
    beta : exploration weight (≥ 0)

    Returns
    -------
    ucb : (m,)
    """
    mean, std = gp.predict(X, return_std=True)
    return mean + beta * std


def thompson_sampling(
    X: np.ndarray,
    gp: GaussianProcessRegressor,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Thompson sampling: draw a sample from the GP posterior.

    Parameters
    ----------
    X : (m, d)
    gp : fitted GP

    Returns
    -------
    sample : (m,)
    """
    if rng is None:
        rng = np.random.RandomState()
    mean, std = gp.predict(X, return_std=True)
    return mean + std * rng.randn(len(mean))


def _standard_normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf_approx(x / math.sqrt(2.0)))


def _standard_normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x ** 2) / math.sqrt(2.0 * math.pi)


def _erf_approx(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun approximation to erf."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


# ---------------------------------------------------------------------------
# Acquisition optimisation
# ---------------------------------------------------------------------------


def optimize_acquisition(
    acq_fn: Callable[[np.ndarray], np.ndarray],
    bounds: np.ndarray,
    n_restarts: int = 10,
    n_random: int = 1000,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Optimise an acquisition function over a bounded domain.

    Uses random initialisation followed by L-BFGS-B.

    Parameters
    ----------
    acq_fn : callable (X: (m,d)) -> (m,) scores (to maximise)
    bounds : (d, 2) lower and upper bounds
    n_restarts : number of L-BFGS-B restarts
    n_random : number of random candidates for initialisation

    Returns
    -------
    x_best : (d,)  the point maximising the acquisition
    """
    if rng is None:
        rng = np.random.RandomState(42)

    d = bounds.shape[0]
    lb, ub = bounds[:, 0], bounds[:, 1]

    # Random phase
    X_rand = rng.uniform(lb, ub, size=(n_random, d))
    scores = acq_fn(X_rand)
    best_idx = np.argsort(-scores)[:n_restarts]

    best_x = X_rand[best_idx[0]]
    best_val = scores[best_idx[0]]

    # L-BFGS-B refinement
    scipy_bounds = list(zip(lb, ub))
    for idx in best_idx:
        x0 = X_rand[idx].copy()

        def neg_acq(x: np.ndarray) -> float:
            return -float(acq_fn(x.reshape(1, -1))[0])

        result = scipy_minimize(neg_acq, x0, method="L-BFGS-B", bounds=scipy_bounds)
        if -result.fun > best_val:
            best_val = -result.fun
            best_x = result.x.copy()

    return best_x


# ---------------------------------------------------------------------------
# RobustnessSurrogate
# ---------------------------------------------------------------------------


class RobustnessSurrogate:
    """Surrogate model predicting STL robustness from circuit parameters,
    with Bayesian optimisation for parameter repair.

    Parameters
    ----------
    kernel : kernel instance (RBFKernel or MaternKernel)
    noise_variance : float
    acq_type : str
        Acquisition function type: ``"ei"``, ``"ucb"``, ``"ts"``.
    acq_params : dict
        Extra params for the acquisition (e.g. ``{"beta": 2.0}``).
    """

    def __init__(
        self,
        kernel: Any = None,
        noise_variance: float = 0.01,
        acq_type: str = "ei",
        acq_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.gp = GaussianProcessRegressor(
            kernel=kernel or RBFKernel(), noise_variance=noise_variance,
        )
        self.acq_type = acq_type
        self.acq_params = acq_params or {}
        self._X_observed: Optional[np.ndarray] = None
        self._y_observed: Optional[np.ndarray] = None
        self._rng = np.random.RandomState(42)
        self._accuracy_history: List[float] = []

    # -- fitting -------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit surrogate to observed (parameter, robustness) pairs."""
        self._X_observed = X.copy()
        self._y_observed = y.copy()
        self.gp.fit(X, y)

    def update(self, X_new: np.ndarray, y_new: np.ndarray) -> None:
        """Add new observations and refit."""
        if self._X_observed is None:
            self.fit(X_new, y_new)
        else:
            self._X_observed = np.vstack([self._X_observed, X_new])
            self._y_observed = np.concatenate([self._y_observed, y_new])
            self.gp.fit(self._X_observed, self._y_observed)

    def predict(
        self, X: np.ndarray, return_std: bool = False,
    ) -> Any:
        """Predict robustness at parameter points."""
        return self.gp.predict(X, return_std=return_std)

    def predict_verdict(
        self, X: np.ndarray, confidence: float = 1.96,
    ) -> List[SurrogateVerdict]:
        """Predict robustness with uncertainty quantification.

        Returns SurrogateVerdict for each input point, including
        prediction mean, std, confidence interval, and APPROXIMATE
        soundness annotation.

        Parameters
        ----------
        X : (m, d) query points
        confidence : z-score for confidence interval (default 1.96 = 95%)
        """
        mean, std = self.gp.predict(X, return_std=True)
        verdicts = []
        for i in range(len(mean)):
            m, s = float(mean[i]), float(std[i])
            lo = m - confidence * s
            hi = m + confidence * s
            soundness = SoundnessAnnotation(
                level=SoundnessLevel.APPROXIMATE,
                assumptions=["GP surrogate prediction — not formally verified"],
                approximation_error=s,
            )
            verdicts.append(SurrogateVerdict(
                robustness=m,
                std=s,
                confidence_lo=lo,
                confidence_hi=hi,
                soundness=soundness,
            ))
        return verdicts

    # -- Bayesian optimisation -----------------------------------------------

    def suggest_next(
        self, bounds: np.ndarray, n_suggestions: int = 1,
    ) -> np.ndarray:
        """Suggest next parameter(s) to evaluate via BO.

        Parameters
        ----------
        bounds : (d, 2) parameter bounds
        n_suggestions : how many points to suggest

        Returns
        -------
        np.ndarray of shape ``(n_suggestions, d)``
        """
        if self._y_observed is None or len(self._y_observed) == 0:
            d = bounds.shape[0]
            return self._rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_suggestions, d))

        y_best = float(self._y_observed.max())

        suggestions: List[np.ndarray] = []
        for _ in range(n_suggestions):
            acq_fn = self._build_acq(y_best)
            x_next = optimize_acquisition(acq_fn, bounds, rng=self._rng)
            suggestions.append(x_next)

        return np.array(suggestions)

    def _build_acq(self, y_best: float) -> Callable[[np.ndarray], np.ndarray]:
        if self.acq_type == "ei":
            xi = self.acq_params.get("xi", 0.01)
            return lambda X: expected_improvement(X, self.gp, y_best, xi=xi)
        elif self.acq_type == "ucb":
            beta = self.acq_params.get("beta", 2.0)
            return lambda X: upper_confidence_bound(X, self.gp, beta=beta)
        elif self.acq_type == "ts":
            return lambda X: thompson_sampling(X, self.gp, rng=self._rng)
        else:
            raise ValueError(f"Unknown acquisition type: {self.acq_type!r}")

    # -- surrogate-guided search ---------------------------------------------

    def parameter_repair(
        self,
        bounds: np.ndarray,
        oracle_fn: Callable[[np.ndarray], float],
        max_iterations: int = 50,
        target_robustness: float = 0.0,
        n_initial: int = 5,
    ) -> Tuple[np.ndarray, float]:
        """Run surrogate-guided parameter repair.

        Parameters
        ----------
        bounds : (d, 2) parameter bounds
        oracle_fn : callable (params: (d,)) -> robustness score
        max_iterations : budget
        target_robustness : stop when robustness >= this value
        n_initial : initial Latin Hypercube samples

        Returns
        -------
        (best_params, best_robustness)
        """
        d = bounds.shape[0]

        # Initial samples
        X_init = _latin_hypercube(n_initial, d, bounds, self._rng)
        y_init = np.array([oracle_fn(x) for x in X_init])
        self.fit(X_init, y_init)

        best_idx = int(np.argmax(y_init))
        best_x = X_init[best_idx].copy()
        best_y = float(y_init[best_idx])

        for it in range(max_iterations):
            if best_y >= target_robustness:
                logger.info("Target robustness reached at iteration %d.", it)
                break

            x_next = self.suggest_next(bounds, n_suggestions=1)[0]
            y_next = oracle_fn(x_next)
            self.update(x_next.reshape(1, -1), np.array([y_next]))

            if y_next > best_y:
                best_y = y_next
                best_x = x_next.copy()

        return best_x, best_y

    # -- active learning -----------------------------------------------------

    def query_most_uncertain(
        self, candidates: np.ndarray, n_query: int = 1,
    ) -> np.ndarray:
        """Select candidates with highest predictive uncertainty.

        Parameters
        ----------
        candidates : (m, d) parameter candidates
        n_query : number to select

        Returns
        -------
        indices of selected candidates
        """
        _, std = self.gp.predict(candidates, return_std=True)
        return np.argsort(-std)[:n_query]

    # -- accuracy monitoring -------------------------------------------------

    def evaluate_accuracy(
        self, X_test: np.ndarray, y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate surrogate accuracy on held-out data.

        Returns RMSE, MAE, R², and Spearman rank correlation.
        """
        y_pred = self.gp.predict(X_test)
        residuals = y_test - y_pred
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mae = float(np.mean(np.abs(residuals)))

        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, _EPS)

        # Spearman rank correlation
        rank_pred = np.argsort(np.argsort(y_pred)).astype(float)
        rank_true = np.argsort(np.argsort(y_test)).astype(float)
        n = len(y_test)
        if n > 1:
            d_rank = rank_pred - rank_true
            spearman = 1.0 - 6.0 * np.sum(d_rank ** 2) / (n * (n ** 2 - 1))
        else:
            spearman = 0.0

        metrics = {"rmse": rmse, "mae": mae, "r2": r2, "spearman": spearman}
        self._accuracy_history.append(rmse)
        return metrics

    @property
    def accuracy_history(self) -> List[float]:
        return self._accuracy_history

    # -- hyperparameter optimisation -----------------------------------------

    def optimize_kernel(
        self, n_restarts: int = 3,
    ) -> float:
        """Optimise GP kernel hyperparameters on current data."""
        if self._X_observed is None:
            return -float("inf")
        return self.gp.optimize_hyperparameters(
            self._X_observed, self._y_observed,
            n_restarts=n_restarts, rng=self._rng,
        )


# ---------------------------------------------------------------------------
# Latin Hypercube sampling helper
# ---------------------------------------------------------------------------


def _latin_hypercube(
    n: int, d: int, bounds: np.ndarray,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Generate *n* Latin Hypercube samples in *d* dimensions."""
    if rng is None:
        rng = np.random.RandomState(42)
    samples = np.zeros((n, d), dtype=np.float64)
    for j in range(d):
        perm = rng.permutation(n)
        intervals = np.linspace(0, 1, n + 1)
        for i in range(n):
            samples[i, j] = rng.uniform(intervals[perm[i]], intervals[perm[i] + 1])
    lb, ub = bounds[:, 0], bounds[:, 1]
    return samples * (ub - lb) + lb
