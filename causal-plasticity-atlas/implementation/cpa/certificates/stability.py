"""
Stability analysis utilities for the Causal-Plasticity Atlas.

Provides generic, reusable stability selection and bootstrap engines
consumed by the robustness certificate generator (ALG5) and the
plasticity descriptor confidence intervals (ALG2 Step 5).

Classes
-------
StabilitySelectionEngine
    Variable / edge selection stability via subsampling.
BootstrapEngine
    Parametric and nonparametric bootstrap with DAG re-estimation.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class StabilityResult:
    """Result from a stability selection run."""

    selection_probabilities: NDArray  # shape (n_variables, n_variables) for edges
    stable_edges: list[tuple[int, int]]  # edges with prob > upper_threshold
    unstable_edges: list[tuple[int, int]]  # edges with prob < lower_threshold
    uncertain_edges: list[tuple[int, int]]  # edges in between
    upper_threshold: float
    lower_threshold: float
    n_rounds: int
    subsample_fraction: float
    consensus_adjacency: NDArray  # binary adjacency from stable edges
    metadata: dict = field(default_factory=dict)

    def is_edge_stable(self, i: int, j: int) -> bool:
        """Check if edge (i -> j) is stably selected."""
        return (i, j) in self.stable_edges

    def is_edge_absent(self, i: int, j: int) -> bool:
        """Check if edge (i -> j) is stably absent."""
        return (i, j) in self.unstable_edges

    def edge_probability(self, i: int, j: int) -> float:
        """Return selection probability for edge (i -> j)."""
        return float(self.selection_probabilities[i, j])


@dataclass
class BootstrapResult:
    """Result from a bootstrap analysis."""

    point_estimate: NDArray  # Original parameter estimates
    bootstrap_samples: NDArray  # (B, *param_shape) bootstrap replicates
    ci_lower: NDArray  # Lower CI bound
    ci_upper: NDArray  # Upper CI bound
    ci_level: float  # Confidence level (e.g. 0.95)
    method: str  # "percentile", "bca", "normal"
    n_bootstrap: int
    se: NDArray  # Bootstrap standard errors
    bias: NDArray  # Bootstrap bias estimate
    metadata: dict = field(default_factory=dict)

    @property
    def ci_width(self) -> NDArray:
        """Width of confidence intervals."""
        return self.ci_upper - self.ci_lower

    def contains(self, value: NDArray) -> NDArray:
        """Check which CI contain a given value (element-wise)."""
        return (self.ci_lower <= value) & (value <= self.ci_upper)


# ---------------------------------------------------------------------------
# Stability Selection Engine
# ---------------------------------------------------------------------------

class StabilitySelectionEngine:
    """Generic stability selection framework.

    Implements the stability selection procedure of Meinshausen & Bühlmann
    (JRSS-B, 2010): repeatedly subsample the data, run a structure learning
    algorithm, and track which edges are consistently selected.

    Parameters
    ----------
    n_rounds : int
        Number of subsampling rounds (default 100).
    subsample_fraction : float
        Fraction of samples to use per round (default 0.5).
    upper_threshold : float
        Selection probability above which an edge is declared stable
        (default 0.6).
    lower_threshold : float
        Selection probability below which an edge is declared absent
        (default 0.4).
    random_state : int or None
        Random seed for reproducibility.
    n_jobs : int
        Number of parallel jobs (default 1, -1 for all CPUs).

    Examples
    --------
    >>> engine = StabilitySelectionEngine(n_rounds=100)
    >>> result = engine.run(data, learner_fn=my_dag_learner)
    """

    def __init__(
        self,
        n_rounds: int = 100,
        subsample_fraction: float = 0.5,
        upper_threshold: float = 0.6,
        lower_threshold: float = 0.4,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
    ):
        if not 0 < subsample_fraction < 1:
            raise ValueError("subsample_fraction must be in (0, 1).")
        if not 0 <= lower_threshold <= upper_threshold <= 1:
            raise ValueError("Thresholds must satisfy 0 <= lower <= upper <= 1.")
        if n_rounds < 1:
            raise ValueError("n_rounds must be >= 1.")

        self.n_rounds = n_rounds
        self.subsample_fraction = subsample_fraction
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.random_state = random_state
        self.n_jobs = n_jobs

    def run(
        self,
        data: NDArray,
        learner_fn: Callable[[NDArray], NDArray],
        variable_names: Optional[list[str]] = None,
    ) -> StabilityResult:
        """Run stability selection.

        Parameters
        ----------
        data : (n_samples, n_variables) data matrix
        learner_fn : function that takes data and returns adjacency matrix
            adjacency[i, j] = 1 means edge i -> j exists
        variable_names : optional variable names for reporting

        Returns
        -------
        StabilityResult
        """
        data = np.asarray(data, dtype=np.float64)
        n_samples, n_vars = data.shape
        subsample_size = max(2, int(n_samples * self.subsample_fraction))

        if subsample_size >= n_samples:
            warnings.warn(
                f"Subsample size ({subsample_size}) >= n_samples ({n_samples}). "
                "Reducing subsample_fraction.",
                stacklevel=2,
            )
            subsample_size = max(2, n_samples - 1)

        rng = np.random.default_rng(self.random_state)
        selection_counts = np.zeros((n_vars, n_vars), dtype=np.float64)

        for r in range(self.n_rounds):
            indices = rng.choice(n_samples, size=subsample_size, replace=False)
            sub_data = data[indices]

            try:
                adj = learner_fn(sub_data)
                adj = np.asarray(adj, dtype=np.float64)
                if adj.shape != (n_vars, n_vars):
                    raise ValueError(
                        f"Learner returned adjacency of shape {adj.shape}, "
                        f"expected ({n_vars}, {n_vars})."
                    )
                selection_counts += (adj != 0).astype(np.float64)
            except Exception as e:
                warnings.warn(
                    f"Stability round {r} failed: {e}. Skipping.",
                    stacklevel=2,
                )

        probs = selection_counts / self.n_rounds

        # Classify edges
        stable = []
        unstable = []
        uncertain = []
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                p = probs[i, j]
                if p >= self.upper_threshold:
                    stable.append((i, j))
                elif p <= self.lower_threshold:
                    unstable.append((i, j))
                else:
                    uncertain.append((i, j))

        consensus = (probs >= self.upper_threshold).astype(np.float64)
        np.fill_diagonal(consensus, 0.0)

        return StabilityResult(
            selection_probabilities=probs,
            stable_edges=stable,
            unstable_edges=unstable,
            uncertain_edges=uncertain,
            upper_threshold=self.upper_threshold,
            lower_threshold=self.lower_threshold,
            n_rounds=self.n_rounds,
            subsample_fraction=self.subsample_fraction,
            consensus_adjacency=consensus,
            metadata={
                "n_samples": n_samples,
                "n_variables": n_vars,
                "subsample_size": subsample_size,
                "variable_names": variable_names,
            },
        )

    def run_variable_selection(
        self,
        data: NDArray,
        target_idx: int,
        selector_fn: Callable[[NDArray, NDArray], NDArray],
    ) -> tuple[NDArray, list[int], list[int]]:
        """Run stability selection for variable (parent) selection.

        Parameters
        ----------
        data : (n_samples, n_variables) data matrix
        target_idx : index of the target variable
        selector_fn : function(X, y) -> binary selection vector of length X.shape[1]

        Returns
        -------
        (probabilities, stable_vars, unstable_vars)
            probabilities: selection probability for each predictor
            stable_vars: indices of stably selected predictors
            unstable_vars: indices of stably absent predictors
        """
        data = np.asarray(data, dtype=np.float64)
        n_samples, n_vars = data.shape

        # Separate target
        predictor_mask = np.ones(n_vars, dtype=bool)
        predictor_mask[target_idx] = False
        X_full = data[:, predictor_mask]
        y_full = data[:, target_idx]
        n_predictors = X_full.shape[1]

        subsample_size = max(2, int(n_samples * self.subsample_fraction))
        rng = np.random.default_rng(self.random_state)
        counts = np.zeros(n_predictors, dtype=np.float64)
        valid_rounds = 0

        for _ in range(self.n_rounds):
            indices = rng.choice(n_samples, size=subsample_size, replace=False)
            X_sub = X_full[indices]
            y_sub = y_full[indices]

            try:
                selected = selector_fn(X_sub, y_sub)
                selected = np.asarray(selected, dtype=np.float64).ravel()
                if len(selected) != n_predictors:
                    continue
                counts += (selected != 0).astype(np.float64)
                valid_rounds += 1
            except Exception:
                continue

        if valid_rounds == 0:
            probs = np.zeros(n_predictors)
        else:
            probs = counts / valid_rounds

        stable = [i for i in range(n_predictors) if probs[i] >= self.upper_threshold]
        unstable = [i for i in range(n_predictors) if probs[i] <= self.lower_threshold]

        return probs, stable, unstable

    def edge_selection_stability(
        self,
        data: NDArray,
        learner_fn: Callable[[NDArray], NDArray],
        edge: tuple[int, int],
    ) -> float:
        """Compute selection probability for a specific edge.

        Parameters
        ----------
        data : (n_samples, n_variables) data matrix
        learner_fn : DAG learning function
        edge : (i, j) edge to check

        Returns
        -------
        float : selection probability in [0, 1]
        """
        result = self.run(data, learner_fn)
        return result.edge_probability(edge[0], edge[1])

    def consensus_graph(
        self,
        data: NDArray,
        learner_fn: Callable[[NDArray], NDArray],
        threshold: Optional[float] = None,
    ) -> NDArray:
        """Return consensus adjacency matrix from stability selection.

        Parameters
        ----------
        data : data matrix
        learner_fn : DAG learning function
        threshold : custom threshold (default: self.upper_threshold)

        Returns
        -------
        Binary adjacency matrix
        """
        result = self.run(data, learner_fn)
        thresh = threshold if threshold is not None else self.upper_threshold
        consensus = (result.selection_probabilities >= thresh).astype(np.float64)
        np.fill_diagonal(consensus, 0.0)
        return consensus


# ---------------------------------------------------------------------------
# Bootstrap Engine
# ---------------------------------------------------------------------------

class BootstrapEngine:
    """Parametric and nonparametric bootstrap engine.

    Provides bootstrap inference for regression parameters, DAG edges,
    and general statistics. Supports percentile, BCa, and normal
    bootstrap confidence intervals.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap replicates (default 200).
    ci_level : float
        Confidence level for intervals (default 0.95).
    method : str
        CI method: "percentile", "bca", or "normal" (default "percentile").
    random_state : int or None
        Random seed.
    """

    def __init__(
        self,
        n_bootstrap: int = 200,
        ci_level: float = 0.95,
        method: str = "percentile",
        random_state: Optional[int] = None,
    ):
        if n_bootstrap < 10:
            raise ValueError("n_bootstrap must be >= 10.")
        if not 0 < ci_level < 1:
            raise ValueError("ci_level must be in (0, 1).")
        if method not in ("percentile", "bca", "normal"):
            raise ValueError(f"Unknown method: {method}. Use percentile, bca, or normal.")

        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
        self.method = method
        self.random_state = random_state

    # ---- Parametric bootstrap for linear Gaussian models ----

    def parametric_bootstrap_regression(
        self,
        X: NDArray,
        y: NDArray,
        coefficients: Optional[NDArray] = None,
        residual_var: Optional[float] = None,
    ) -> BootstrapResult:
        """Parametric bootstrap for linear regression coefficients.

        Generates bootstrap data from the fitted model:
            y* = X @ beta + epsilon*, epsilon* ~ N(0, sigma^2)
        and re-estimates coefficients on each replicate.

        Parameters
        ----------
        X : (n, p) predictor matrix
        y : (n,) response vector
        coefficients : fitted coefficients (if None, computed via OLS)
        residual_var : residual variance (if None, computed from residuals)

        Returns
        -------
        BootstrapResult with CIs for regression coefficients
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, p = X.shape

        if n < p + 1:
            warnings.warn(
                f"n={n} < p+1={p+1}: regression may be ill-conditioned.",
                stacklevel=2,
            )

        # Fit OLS if needed
        if coefficients is None:
            coefficients = self._ols_fit(X, y)
        coefficients = np.asarray(coefficients, dtype=np.float64).ravel()

        if residual_var is None:
            residuals = y - X @ coefficients
            if n > p:
                residual_var = float(np.sum(residuals ** 2) / (n - p))
            else:
                residual_var = float(np.var(residuals))

        residual_std = math.sqrt(max(residual_var, 1e-15))
        rng = np.random.default_rng(self.random_state)

        boot_coefs = np.zeros((self.n_bootstrap, p), dtype=np.float64)
        for b in range(self.n_bootstrap):
            eps = rng.normal(0, residual_std, size=n)
            y_boot = X @ coefficients + eps
            boot_coefs[b] = self._ols_fit(X, y_boot)

        return self._build_result(coefficients, boot_coefs)

    def parametric_bootstrap_variance(
        self,
        residual_var: float,
        n_samples: int,
        n_params: int = 0,
    ) -> BootstrapResult:
        """Parametric bootstrap for residual variance.

        Uses chi-squared resampling: sigma^2 * chi2(df) / df
        where df = n - p.

        Parameters
        ----------
        residual_var : point estimate of residual variance
        n_samples : number of samples
        n_params : number of estimated parameters

        Returns
        -------
        BootstrapResult with CIs for variance
        """
        df = max(n_samples - n_params, 1)
        rng = np.random.default_rng(self.random_state)

        boot_vars = np.zeros(self.n_bootstrap, dtype=np.float64)
        for b in range(self.n_bootstrap):
            chi2_sample = rng.chisquare(df)
            boot_vars[b] = residual_var * chi2_sample / df

        point = np.array([residual_var])
        return self._build_result(point, boot_vars.reshape(-1, 1))

    # ---- Nonparametric bootstrap ----

    def nonparametric_bootstrap(
        self,
        data: NDArray,
        statistic_fn: Callable[[NDArray], NDArray],
    ) -> BootstrapResult:
        """Nonparametric bootstrap with case resampling.

        Parameters
        ----------
        data : (n_samples, ...) data array
        statistic_fn : function that takes resampled data and returns
            a 1-D array of statistics

        Returns
        -------
        BootstrapResult
        """
        data = np.asarray(data, dtype=np.float64)
        n = data.shape[0]
        rng = np.random.default_rng(self.random_state)

        point = statistic_fn(data)
        point = np.asarray(point, dtype=np.float64).ravel()
        p = len(point)

        boot_stats = np.zeros((self.n_bootstrap, p), dtype=np.float64)
        for b in range(self.n_bootstrap):
            indices = rng.choice(n, size=n, replace=True)
            boot_data = data[indices]
            try:
                boot_stats[b] = statistic_fn(boot_data)
            except Exception:
                boot_stats[b] = point  # fallback

        return self._build_result(point, boot_stats)

    def nonparametric_bootstrap_dag(
        self,
        data: NDArray,
        learner_fn: Callable[[NDArray], NDArray],
    ) -> tuple[BootstrapResult, NDArray]:
        """Nonparametric bootstrap with DAG re-estimation.

        Parameters
        ----------
        data : (n_samples, n_variables) data
        learner_fn : DAG learning function returning adjacency matrix

        Returns
        -------
        (BootstrapResult, edge_probabilities)
            BootstrapResult: CIs for vectorized adjacency
            edge_probabilities: (n_vars, n_vars) edge selection probabilities
        """
        data = np.asarray(data, dtype=np.float64)
        n_samples, n_vars = data.shape
        rng = np.random.default_rng(self.random_state)

        # Original estimate
        adj_orig = np.asarray(learner_fn(data), dtype=np.float64)
        point = adj_orig.ravel()

        boot_adjs = np.zeros((self.n_bootstrap, n_vars * n_vars), dtype=np.float64)
        for b in range(self.n_bootstrap):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            boot_data = data[indices]
            try:
                adj_boot = np.asarray(learner_fn(boot_data), dtype=np.float64)
                boot_adjs[b] = adj_boot.ravel()
            except Exception:
                boot_adjs[b] = point

        result = self._build_result(point, boot_adjs)
        edge_probs = np.mean(boot_adjs.reshape(self.n_bootstrap, n_vars, n_vars) != 0, axis=0)

        return result, edge_probs

    # ---- Bootstrap aggregation (bagging) ----

    def bagging_adjacency(
        self,
        data: NDArray,
        learner_fn: Callable[[NDArray], NDArray],
        threshold: float = 0.5,
    ) -> NDArray:
        """Bootstrap aggregation for DAG estimation.

        Each bootstrap replicate produces a DAG; the bagged DAG
        includes edges present in > threshold fraction of replicates.

        Parameters
        ----------
        data : (n_samples, n_variables) data
        learner_fn : DAG learning function
        threshold : edge inclusion threshold

        Returns
        -------
        (n_variables, n_variables) bagged adjacency matrix
        """
        _, edge_probs = self.nonparametric_bootstrap_dag(data, learner_fn)
        bagged = (edge_probs >= threshold).astype(np.float64)
        np.fill_diagonal(bagged, 0.0)
        return bagged

    # ---- Confidence band computation ----

    def confidence_bands(
        self,
        x_grid: NDArray,
        data: NDArray,
        model_fn: Callable[[NDArray, NDArray], NDArray],
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Compute pointwise confidence bands for a model prediction.

        Parameters
        ----------
        x_grid : (m,) or (m, d) grid of evaluation points
        data : (n, d+1) data (last column is response)
        model_fn : function(data, x_grid) -> predictions at x_grid

        Returns
        -------
        (predictions, lower_band, upper_band)
        """
        data = np.asarray(data, dtype=np.float64)
        x_grid = np.asarray(x_grid, dtype=np.float64)
        n = data.shape[0]
        rng = np.random.default_rng(self.random_state)

        pred_orig = model_fn(data, x_grid)
        m = len(pred_orig)

        boot_preds = np.zeros((self.n_bootstrap, m), dtype=np.float64)
        for b in range(self.n_bootstrap):
            indices = rng.choice(n, size=n, replace=True)
            boot_data = data[indices]
            try:
                boot_preds[b] = model_fn(boot_data, x_grid)
            except Exception:
                boot_preds[b] = pred_orig

        alpha = 1 - self.ci_level
        lower = np.percentile(boot_preds, 100 * alpha / 2, axis=0)
        upper = np.percentile(boot_preds, 100 * (1 - alpha / 2), axis=0)

        return pred_orig, lower, upper

    # ---- Internal helpers ----

    @staticmethod
    def _ols_fit(X: NDArray, y: NDArray) -> NDArray:
        """Ordinary least squares via pseudoinverse."""
        try:
            return np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return np.linalg.pinv(X) @ y

    def _build_result(
        self,
        point: NDArray,
        boot_samples: NDArray,
    ) -> BootstrapResult:
        """Build BootstrapResult from point estimate and bootstrap samples."""
        point = np.asarray(point, dtype=np.float64).ravel()
        boot_samples = np.asarray(boot_samples, dtype=np.float64)

        if boot_samples.ndim == 1:
            boot_samples = boot_samples[:, np.newaxis]

        se = np.std(boot_samples, axis=0, ddof=1)
        bias = np.mean(boot_samples, axis=0) - point

        if self.method == "percentile":
            ci_lower, ci_upper = self._percentile_ci(boot_samples)
        elif self.method == "bca":
            ci_lower, ci_upper = self._bca_ci(point, boot_samples)
        elif self.method == "normal":
            ci_lower, ci_upper = self._normal_ci(point, se, bias)
        else:
            ci_lower, ci_upper = self._percentile_ci(boot_samples)

        return BootstrapResult(
            point_estimate=point,
            bootstrap_samples=boot_samples,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.ci_level,
            method=self.method,
            n_bootstrap=self.n_bootstrap,
            se=se,
            bias=bias,
        )

    def _percentile_ci(self, boot_samples: NDArray) -> tuple[NDArray, NDArray]:
        """Percentile bootstrap CI."""
        alpha = 1 - self.ci_level
        lower = np.percentile(boot_samples, 100 * alpha / 2, axis=0)
        upper = np.percentile(boot_samples, 100 * (1 - alpha / 2), axis=0)
        return lower, upper

    def _normal_ci(
        self,
        point: NDArray,
        se: NDArray,
        bias: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Normal approximation bootstrap CI with bias correction."""
        from scipy.stats import norm

        alpha = 1 - self.ci_level
        z = norm.ppf(1 - alpha / 2)
        corrected = point - bias
        lower = corrected - z * se
        upper = corrected + z * se
        return lower, upper

    def _bca_ci(
        self,
        point: NDArray,
        boot_samples: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """BCa (bias-corrected and accelerated) bootstrap CI.

        Implements the BCa method of Efron (1987).
        """
        from scipy.stats import norm

        alpha = 1 - self.ci_level
        B = boot_samples.shape[0]
        p = point.shape[0]

        lower = np.zeros(p)
        upper = np.zeros(p)

        for j in range(p):
            samples = boot_samples[:, j]
            theta_hat = point[j]

            # Bias correction factor z0
            prop_less = np.mean(samples < theta_hat)
            prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
            z0 = norm.ppf(prop_less)

            # Acceleration factor a (jackknife estimate)
            # Use simple estimate: a = 0 if we can't compute jackknife
            a = self._jackknife_acceleration(samples)

            # Adjusted quantiles
            z_alpha_lo = norm.ppf(alpha / 2)
            z_alpha_hi = norm.ppf(1 - alpha / 2)

            denom_lo = 1 - a * (z0 + z_alpha_lo)
            denom_hi = 1 - a * (z0 + z_alpha_hi)

            if abs(denom_lo) < 1e-10:
                denom_lo = 1e-10
            if abs(denom_hi) < 1e-10:
                denom_hi = 1e-10

            alpha_lo = norm.cdf(z0 + (z0 + z_alpha_lo) / denom_lo)
            alpha_hi = norm.cdf(z0 + (z0 + z_alpha_hi) / denom_hi)

            alpha_lo = np.clip(alpha_lo, 0.001, 0.999)
            alpha_hi = np.clip(alpha_hi, 0.001, 0.999)

            lower[j] = np.percentile(samples, 100 * alpha_lo)
            upper[j] = np.percentile(samples, 100 * alpha_hi)

        return lower, upper

    @staticmethod
    def _jackknife_acceleration(samples: NDArray) -> float:
        """Estimate acceleration factor from jackknife influence values."""
        n = len(samples)
        if n < 3:
            return 0.0

        # Jackknife pseudo-values
        theta_all = np.mean(samples)
        jack_vals = np.zeros(n)
        for i in range(n):
            jack_vals[i] = np.mean(np.delete(samples, i))

        jack_mean = np.mean(jack_vals)
        diff = jack_mean - jack_vals
        num = np.sum(diff ** 3)
        denom = 6.0 * (np.sum(diff ** 2)) ** 1.5

        if abs(denom) < 1e-15:
            return 0.0
        return num / denom
