"""Hyperparameter spaces and tuning.

Defines parameter ranges, hyperparameter configurations, and search
algorithms (grid, random, Bayesian optimization via GP surrogate).

Classes
-------
ParameterRange / HyperparameterConfig
    Low-level parameter descriptors (kept for backward compatibility).
HyperparameterSpace
    Searchable parameter space supporting continuous, discrete, integer,
    and categorical dimensions.
GridSearch / RandomSearch / BayesianOptimizer
    Search algorithms over a HyperparameterSpace.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm as _norm
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Low-level descriptors (backward compatibility)
# ---------------------------------------------------------------------------

@dataclass
class ParameterRange:
    """Specification of a single hyperparameter range."""

    name: str
    low: float
    high: float
    log_scale: bool = False
    dtype: str = "float"


@dataclass
class HyperparameterConfig:
    """Configuration for a hyperparameter search."""

    parameters: List[ParameterRange] = field(default_factory=list)
    n_trials: int = 100
    method: str = "random"


# ---------------------------------------------------------------------------
# Dimension descriptors
# ---------------------------------------------------------------------------

@dataclass
class _ContinuousDim:
    name: str
    low: float
    high: float
    log_scale: bool = False


@dataclass
class _DiscreteDim:
    name: str
    values: List[Any]


@dataclass
class _IntegerDim:
    name: str
    low: int
    high: int


@dataclass
class _CategoricalDim:
    name: str
    categories: List[Any]


_Dim = _ContinuousDim | _DiscreteDim | _IntegerDim | _CategoricalDim


# ---------------------------------------------------------------------------
# Hyperparameter Space
# ---------------------------------------------------------------------------

class HyperparameterSpace:
    """Searchable hyperparameter space.

    Supports continuous, discrete, integer, and categorical dimensions.
    Also accepts a legacy ``HyperparameterConfig`` for backward compat.
    """

    def __init__(self, config: Optional[HyperparameterConfig] = None) -> None:
        self._dims: List[_Dim] = []
        self._results: List[Dict[str, Any]] = []
        if config is not None:
            self._config = config
            for p in config.parameters:
                self.add_continuous(p.name, p.low, p.high, log_scale=p.log_scale)
        else:
            self._config = HyperparameterConfig()

    # -- Building the space --

    def add_continuous(
        self, name: str, low: float, high: float, log_scale: bool = False
    ) -> None:
        """Add a continuous parameter."""
        if low >= high:
            raise ValueError(f"{name}: low ({low}) must be < high ({high})")
        if log_scale and low <= 0:
            raise ValueError(f"{name}: log_scale requires low > 0")
        self._dims.append(_ContinuousDim(name, low, high, log_scale))

    def add_discrete(self, name: str, values: List[Any]) -> None:
        """Add a discrete parameter with explicit value list."""
        if not values:
            raise ValueError(f"{name}: values must be non-empty")
        self._dims.append(_DiscreteDim(name, list(values)))

    def add_integer(self, name: str, low: int, high: int) -> None:
        """Add an integer parameter in [low, high]."""
        if low > high:
            raise ValueError(f"{name}: low ({low}) must be <= high ({high})")
        self._dims.append(_IntegerDim(name, int(low), int(high)))

    def add_categorical(self, name: str, categories: List[Any]) -> None:
        """Add a categorical parameter."""
        if not categories:
            raise ValueError(f"{name}: categories must be non-empty")
        self._dims.append(_CategoricalDim(name, list(categories)))

    # -- Sampling --

    def sample(self, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """Draw a single random sample from the space."""
        if rng is None:
            rng = np.random.default_rng()
        params: Dict[str, Any] = {}
        for dim in self._dims:
            if isinstance(dim, _ContinuousDim):
                if dim.log_scale:
                    log_val = rng.uniform(math.log(dim.low), math.log(dim.high))
                    params[dim.name] = math.exp(log_val)
                else:
                    params[dim.name] = rng.uniform(dim.low, dim.high)
            elif isinstance(dim, _DiscreteDim):
                params[dim.name] = dim.values[rng.integers(len(dim.values))]
            elif isinstance(dim, _IntegerDim):
                params[dim.name] = int(rng.integers(dim.low, dim.high + 1))
            elif isinstance(dim, _CategoricalDim):
                params[dim.name] = dim.categories[rng.integers(len(dim.categories))]
        return params

    def grid(self, n_per_dim: int = 5) -> List[Dict[str, Any]]:
        """Generate a full grid over the space."""
        per_dim_values: List[List[Any]] = []
        names: List[str] = []
        for dim in self._dims:
            names.append(dim.name)
            if isinstance(dim, _ContinuousDim):
                if dim.log_scale:
                    vals = np.exp(
                        np.linspace(math.log(dim.low), math.log(dim.high), n_per_dim)
                    ).tolist()
                else:
                    vals = np.linspace(dim.low, dim.high, n_per_dim).tolist()
                per_dim_values.append(vals)
            elif isinstance(dim, _DiscreteDim):
                per_dim_values.append(list(dim.values))
            elif isinstance(dim, _IntegerDim):
                n = min(n_per_dim, dim.high - dim.low + 1)
                vals = np.linspace(dim.low, dim.high, n).astype(int)
                per_dim_values.append(sorted(set(int(v) for v in vals)))
            elif isinstance(dim, _CategoricalDim):
                per_dim_values.append(list(dim.categories))

        grid_points: List[Dict[str, Any]] = []
        for combo in itertools.product(*per_dim_values):
            grid_points.append(dict(zip(names, combo)))
        return grid_points

    def dimensions(self) -> int:
        """Number of dimensions in the space."""
        return len(self._dims)

    # -- Backward compatibility --

    def register_parameter(
        self, name: str, low: float, high: float, log_scale: bool = False
    ) -> None:
        self.add_continuous(name, low, high, log_scale=log_scale)

    def best_params(self) -> Dict[str, Any]:
        """Return the best hyperparameters found so far (if tracked)."""
        if not self._results:
            return {}
        return max(self._results, key=lambda r: r.get("_score", float("-inf")))

    # -- Internal helpers --

    def _to_unit_vector(self, params: Dict[str, Any]) -> NDArray:
        """Map params to [0, 1]^d for continuous/integer dims."""
        vec = np.zeros(len(self._dims))
        for i, dim in enumerate(self._dims):
            v = params[dim.name]
            if isinstance(dim, _ContinuousDim):
                if dim.log_scale:
                    vec[i] = (math.log(v) - math.log(dim.low)) / (
                        math.log(dim.high) - math.log(dim.low)
                    )
                else:
                    vec[i] = (v - dim.low) / (dim.high - dim.low)
            elif isinstance(dim, _IntegerDim):
                vec[i] = (v - dim.low) / max(dim.high - dim.low, 1)
            elif isinstance(dim, (_DiscreteDim, _CategoricalDim)):
                choices = dim.values if isinstance(dim, _DiscreteDim) else dim.categories
                idx = choices.index(v) if v in choices else 0
                vec[i] = idx / max(len(choices) - 1, 1)
        return vec

    def _from_unit_vector(self, vec: NDArray) -> Dict[str, Any]:
        """Map a [0, 1]^d vector back to parameter space."""
        params: Dict[str, Any] = {}
        for i, dim in enumerate(self._dims):
            u = float(np.clip(vec[i], 0.0, 1.0))
            if isinstance(dim, _ContinuousDim):
                if dim.log_scale:
                    params[dim.name] = math.exp(
                        math.log(dim.low) + u * (math.log(dim.high) - math.log(dim.low))
                    )
                else:
                    params[dim.name] = dim.low + u * (dim.high - dim.low)
            elif isinstance(dim, _IntegerDim):
                params[dim.name] = int(round(dim.low + u * (dim.high - dim.low)))
            elif isinstance(dim, _DiscreteDim):
                idx = int(round(u * (len(dim.values) - 1)))
                params[dim.name] = dim.values[idx]
            elif isinstance(dim, _CategoricalDim):
                idx = int(round(u * (len(dim.categories) - 1)))
                params[dim.name] = dim.categories[idx]
        return params


# ---------------------------------------------------------------------------
# Grid Search
# ---------------------------------------------------------------------------

class GridSearch:
    """Exhaustive grid search over a HyperparameterSpace.

    Parameters
    ----------
    space : HyperparameterSpace
    objective_fn : callable(params) -> float (higher is better)
    """

    def __init__(
        self,
        space: HyperparameterSpace,
        objective_fn: Callable[[Dict[str, Any]], float],
    ) -> None:
        self.space = space
        self.objective_fn = objective_fn
        self._results: List[Tuple[Dict[str, Any], float]] = []

    def search(self, n_per_dim: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Run exhaustive grid search.

        Returns list of (params, score) sorted best-first.
        """
        grid_points = self.space.grid(n_per_dim)
        self._results = []
        for params in grid_points:
            score = self._evaluate_point(params)
            self._results.append((params, score))
        self._results.sort(key=lambda x: x[1], reverse=True)
        return list(self._results)

    def _evaluate_point(self, params: Dict[str, Any]) -> float:
        """Evaluate objective at *params*, catching exceptions."""
        try:
            return float(self.objective_fn(params))
        except Exception:
            return float("-inf")

    def best_params(self) -> Dict[str, Any]:
        """Return best parameters found."""
        if not self._results:
            return {}
        return self._results[0][0]


# ---------------------------------------------------------------------------
# Random Search
# ---------------------------------------------------------------------------

class RandomSearch:
    """Random search over a HyperparameterSpace.

    Parameters
    ----------
    space : HyperparameterSpace
    objective_fn : callable(params) -> float
    n_trials : int
        Number of random trials.
    """

    def __init__(
        self,
        space: HyperparameterSpace,
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int = 100,
    ) -> None:
        self.space = space
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self._results: List[Tuple[Dict[str, Any], float]] = []

    def search(self, seed: Optional[int] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Run random search; returns (params, score) list sorted best-first."""
        rng = np.random.default_rng(seed)
        self._results = []
        for _ in range(self.n_trials):
            params, score = self._sample_and_evaluate(rng)
            self._results.append((params, score))
        self._results.sort(key=lambda x: x[1], reverse=True)
        return list(self._results)

    def _sample_and_evaluate(
        self, rng: np.random.Generator
    ) -> Tuple[Dict[str, Any], float]:
        """Sample one point and evaluate."""
        params = self.space.sample(rng)
        try:
            score = float(self.objective_fn(params))
        except Exception:
            score = float("-inf")
        return params, score

    def top_k(self, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Return top-k parameter settings."""
        sorted_results = sorted(self._results, key=lambda x: x[1], reverse=True)
        return sorted_results[:k]


# ---------------------------------------------------------------------------
# Bayesian Optimizer (GP surrogate + Expected Improvement)
# ---------------------------------------------------------------------------

class BayesianOptimizer:
    """Bayesian optimization via Gaussian-process surrogate.

    Uses a simple squared-exponential kernel GP and the Expected
    Improvement acquisition function.  No external BO library is required.

    Parameters
    ----------
    space : HyperparameterSpace
    objective_fn : callable(params) -> float
    n_initial : int
        Number of initial random evaluations.
    n_iterations : int
        Number of BO iterations after the initial phase.
    """

    def __init__(
        self,
        space: HyperparameterSpace,
        objective_fn: Callable[[Dict[str, Any]], float],
        n_initial: int = 10,
        n_iterations: int = 50,
    ) -> None:
        self.space = space
        self.objective_fn = objective_fn
        self.n_initial = max(n_initial, 2)
        self.n_iterations = n_iterations
        self._X: List[NDArray] = []
        self._y: List[float] = []
        self._params_history: List[Dict[str, Any]] = []
        # GP hyper-parameters
        self._length_scale = 0.3
        self._signal_var = 1.0
        self._noise_var = 1e-6

    def optimize(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], float]:
        """Run full Bayesian optimization.

        Returns (best_params, best_score).
        """
        rng = np.random.default_rng(seed)

        # Phase 1: initial random evaluations
        for _ in range(self.n_initial):
            params = self.space.sample(rng)
            score = self._eval(params)
            self._X.append(self.space._to_unit_vector(params))
            self._y.append(score)
            self._params_history.append(params)

        # Phase 2: BO iterations
        for _ in range(self.n_iterations):
            X_obs = np.array(self._X)
            y_obs = np.array(self._y)
            next_x = self._next_point(X_obs, y_obs, rng)
            params = self.space._from_unit_vector(next_x)
            score = self._eval(params)
            self._X.append(next_x)
            self._y.append(score)
            self._params_history.append(params)

        best_idx = int(np.argmax(self._y))
        return self._params_history[best_idx], self._y[best_idx]

    # -- GP surrogate --

    def _fit_surrogate(
        self, X: NDArray, y: NDArray
    ) -> Tuple[Callable[[NDArray], Tuple[NDArray, NDArray]], NDArray]:
        """Fit GP and return (predict_fn, K_inv).

        predict_fn(X_new) -> (mean, std)
        """
        n = X.shape[0]
        K = self._kernel(X, X) + self._noise_var * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-4 * np.eye(n)
            L = np.linalg.cholesky(K)

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

        def predict(X_new: NDArray) -> Tuple[NDArray, NDArray]:
            K_s = self._kernel(X_new, X)
            K_ss = self._kernel(X_new, X_new)
            mu = K_s @ alpha
            v = np.linalg.solve(L, K_s.T)
            var = np.diag(K_ss) - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 1e-10)
            return mu, np.sqrt(var)

        return predict, alpha

    def _kernel(self, X1: NDArray, X2: NDArray) -> NDArray:
        """Squared-exponential (RBF) kernel."""
        dists = cdist(X1 / self._length_scale, X2 / self._length_scale, "sqeuclidean")
        return self._signal_var * np.exp(-0.5 * dists)

    # -- Acquisition --

    def _acquisition_function(
        self,
        X_cand: NDArray,
        gp_mean: NDArray,
        gp_std: NDArray,
        best_y: float,
    ) -> NDArray:
        """Expected Improvement acquisition."""
        return self._expected_improvement(gp_mean, gp_std, best_y)

    @staticmethod
    def _expected_improvement(
        mu: NDArray, sigma: NDArray, best_y: float
    ) -> NDArray:
        """Compute EI(x) = E[max(f(x) - f*, 0)]."""
        sigma = np.maximum(sigma, 1e-10)
        z = (mu - best_y) / sigma
        ei = (mu - best_y) * _norm.cdf(z) + sigma * _norm.pdf(z)
        return ei

    def _next_point(
        self,
        X_observed: NDArray,
        y_observed: NDArray,
        rng: np.random.Generator,
    ) -> NDArray:
        """Select the next point to evaluate via EI maximization."""
        predict_fn, _ = self._fit_surrogate(X_observed, y_observed)
        best_y = float(np.max(y_observed))

        # Generate candidates
        n_cand = max(200, 20 * self.space.dimensions())
        X_cand = rng.uniform(0, 1, size=(n_cand, self.space.dimensions()))
        mu, sigma = predict_fn(X_cand)
        ei = self._acquisition_function(X_cand, mu, sigma, best_y)

        best_idx = int(np.argmax(ei))
        return X_cand[best_idx]

    # -- Helpers --

    def _eval(self, params: Dict[str, Any]) -> float:
        try:
            return float(self.objective_fn(params))
        except Exception:
            return float("-inf")
