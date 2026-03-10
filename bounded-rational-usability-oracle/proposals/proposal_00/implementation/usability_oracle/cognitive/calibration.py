"""
Parameter calibration and validation for cognitive models.

Provides least-squares fitting, goodness-of-fit statistics, cross-validation,
sensitivity analysis, bootstrap confidence intervals, and parameter
identifiability checks for cognitive timing models used in usability analysis.

Key references:
    - Motulsky, H.J. & Christopoulos, A. (2004). Fitting Models to
      Biological Data Using Linear and Nonlinear Regression. Oxford UP.
    - Efron, B. & Tibshirani, R.J. (1993). An Introduction to the Bootstrap.
      Chapman & Hall.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats
from scipy.optimize import minimize


class ParameterCalibrator:
    """Calibrate cognitive model parameters to observed human performance data.

    Supports least-squares fitting via ``scipy.optimize.minimize``,
    goodness-of-fit diagnostics, k-fold cross-validation, local sensitivity
    analysis, bootstrap confidence intervals, and parameter identifiability
    checking.
    """

    # Default optimiser settings
    _DEFAULT_METHOD: str = "L-BFGS-B"
    _DEFAULT_MAXITER: int = 5000
    _DEFAULT_TOL: float = 1e-10

    # ------------------------------------------------------------------ #
    # Core fitting
    # ------------------------------------------------------------------ #

    @staticmethod
    def residuals(
        params: Dict[str, float],
        observed: NDArray[np.floating],
        model_fn: Callable[..., NDArray[np.floating]],
    ) -> NDArray[np.floating]:
        """Compute residuals between observed data and model predictions.

        Args:
            params: Dictionary mapping parameter names to values.
            observed: 1-D array of observed response times.
            model_fn: Callable that accepts ``**params`` and returns a 1-D
                array of predicted values with the same shape as *observed*.

        Returns:
            1-D array of residuals (observed - predicted).
        """
        predicted = np.asarray(model_fn(**params), dtype=np.float64)
        return np.asarray(observed, dtype=np.float64) - predicted

    def calibrate(
        self,
        observed_times: Sequence[float],
        model_fn: Callable[..., NDArray[np.floating]],
        initial_params: Dict[str, float],
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, float]:
        """Fit model parameters to observed data via least-squares.

        Minimises the sum of squared residuals using
        ``scipy.optimize.minimize`` with the L-BFGS-B algorithm (or
        Nelder-Mead when bounds are absent).

        Args:
            observed_times: Observed response-time measurements.
            model_fn: Callable accepting keyword arguments corresponding to
                the keys of *initial_params* and returning predicted times
                as a 1-D array.
            initial_params: Starting parameter values.
            param_bounds: Optional mapping from parameter names to (lo, hi)
                bound tuples. Parameters not in the dict are unbounded.

        Returns:
            Dictionary of fitted parameter values.

        Raises:
            RuntimeError: If the optimiser fails to converge.
        """
        observed = np.asarray(observed_times, dtype=np.float64)
        param_names = list(initial_params.keys())
        x0 = np.array([initial_params[k] for k in param_names], dtype=np.float64)

        # Build bounds list aligned with x0
        bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None
        method = self._DEFAULT_METHOD
        if param_bounds:
            bounds = [
                param_bounds.get(k, (None, None)) for k in param_names
            ]
        else:
            method = "Nelder-Mead"

        def objective(x: NDArray) -> float:
            kw = {k: float(v) for k, v in zip(param_names, x)}
            res = self.residuals(kw, observed, model_fn)
            return float(np.sum(res ** 2))

        result = minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            options={"maxiter": self._DEFAULT_MAXITER, "ftol": self._DEFAULT_TOL},
        )

        if not result.success:
            warnings.warn(
                f"Optimiser did not converge: {result.message}",
                RuntimeWarning,
                stacklevel=2,
            )

        fitted = {k: float(v) for k, v in zip(param_names, result.x)}
        return fitted

    # ------------------------------------------------------------------ #
    # Goodness of fit
    # ------------------------------------------------------------------ #

    @staticmethod
    def goodness_of_fit(
        observed: Sequence[float],
        predicted: Sequence[float],
        n_params: int = 0,
    ) -> Dict[str, float]:
        """Compute goodness-of-fit statistics.

        Args:
            observed: Observed values.
            predicted: Model-predicted values (same length as *observed*).
            n_params: Number of fitted parameters (for AIC / BIC).

        Returns:
            Dictionary with keys:
                - ``r_squared``: Coefficient of determination.
                - ``rmse``: Root mean squared error.
                - ``mae``: Mean absolute error.
                - ``aic``: Akaike Information Criterion.
                - ``bic``: Bayesian Information Criterion.
        """
        obs = np.asarray(observed, dtype=np.float64)
        pred = np.asarray(predicted, dtype=np.float64)
        if obs.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}"
            )

        n = len(obs)
        residuals = obs - pred
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))

        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(ss_res / n)) if n > 0 else 0.0
        mae = float(np.mean(np.abs(residuals))) if n > 0 else 0.0

        # AIC and BIC (assuming Gaussian residuals)
        if n > 0 and ss_res > 0:
            log_likelihood = -0.5 * n * (np.log(2.0 * np.pi * ss_res / n) + 1.0)
            aic = 2.0 * n_params - 2.0 * float(log_likelihood)
            bic = n_params * np.log(n) - 2.0 * float(log_likelihood)
        else:
            aic = float("inf")
            bic = float("inf")

        return {
            "r_squared": r_squared,
            "rmse": rmse,
            "mae": mae,
            "aic": float(aic),
            "bic": float(bic),
        }

    # ------------------------------------------------------------------ #
    # Cross-validation
    # ------------------------------------------------------------------ #

    def cross_validate(
        self,
        data_x: Sequence[Any],
        data_y: Sequence[float],
        model_fn: Callable[..., NDArray[np.floating]],
        initial_params: Dict[str, float],
        k_folds: int = 5,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """Perform k-fold cross-validation.

        Splits the data into *k_folds* folds, fitting the model on each
        training partition and evaluating on the held-out partition.

        Args:
            data_x: Independent-variable data. Passed to *model_fn* as the
                first positional argument when wrapped internally.
            data_y: Dependent-variable (response time) data.
            model_fn: Callable of the form ``model_fn(data_x, **params)``
                returning predicted values for the given *data_x*.
            initial_params: Starting parameter values.
            k_folds: Number of folds (default 5).
            param_bounds: Optional parameter bounds for calibration.

        Returns:
            Dictionary with ``r_squared_mean``, ``r_squared_std``,
            ``rmse_mean``, ``rmse_std``, and ``fold_results`` list.
        """
        if k_folds < 2:
            raise ValueError(f"k_folds must be >= 2, got {k_folds}")

        x_arr = np.asarray(data_x)
        y_arr = np.asarray(data_y, dtype=np.float64)
        n = len(y_arr)
        if n < k_folds:
            raise ValueError(
                f"Not enough data ({n}) for {k_folds}-fold cross-validation."
            )

        indices = np.arange(n)
        rng = np.random.default_rng(seed=42)
        rng.shuffle(indices)
        folds = np.array_split(indices, k_folds)

        fold_results: List[Dict[str, float]] = []

        for fold_idx in range(k_folds):
            test_idx = folds[fold_idx]
            train_idx = np.concatenate(
                [folds[j] for j in range(k_folds) if j != fold_idx]
            )

            x_train, y_train = x_arr[train_idx], y_arr[train_idx]
            x_test, y_test = x_arr[test_idx], y_arr[test_idx]

            # Wrap model_fn so calibrate receives the right signature
            def _train_model(**params: float) -> NDArray:
                return np.asarray(model_fn(x_train, **params), dtype=np.float64)

            fitted = self.calibrate(y_train, _train_model, initial_params, param_bounds)

            # Predict on test set
            predicted_test = np.asarray(
                model_fn(x_test, **fitted), dtype=np.float64
            )
            gof = self.goodness_of_fit(y_test, predicted_test, len(fitted))
            fold_results.append(gof)

        r2_values = [f["r_squared"] for f in fold_results]
        rmse_values = [f["rmse"] for f in fold_results]

        return {
            "r_squared_mean": float(np.mean(r2_values)),
            "r_squared_std": float(np.std(r2_values, ddof=1)) if k_folds > 1 else 0.0,
            "rmse_mean": float(np.mean(rmse_values)),
            "rmse_std": float(np.std(rmse_values, ddof=1)) if k_folds > 1 else 0.0,
            "fold_results": fold_results,
        }

    # ------------------------------------------------------------------ #
    # Sensitivity analysis
    # ------------------------------------------------------------------ #

    @staticmethod
    def sensitivity_analysis(
        model_fn: Callable[..., float],
        base_params: Dict[str, float],
        param_ranges: Dict[str, Tuple[float, float]],
        n_steps: int = 20,
    ) -> Dict[str, Dict[str, Any]]:
        """One-at-a-time local sensitivity analysis.

        Varies each parameter across its range while holding others at
        their base values, measuring the output change.

        Args:
            model_fn: Callable accepting ``**params`` and returning a scalar.
            base_params: Nominal parameter values.
            param_ranges: Mapping from parameter names to (lo, hi) tuples.
            n_steps: Number of evaluation points per parameter (default 20).

        Returns:
            Dictionary keyed by parameter name, each containing:
                - ``values``: parameter values evaluated.
                - ``outputs``: model output at each value.
                - ``sensitivity_index``: (max_output - min_output) / base_output.
                - ``elasticity``: percentage output change per percentage
                  parameter change at the base value.
        """
        if n_steps < 2:
            raise ValueError(f"n_steps must be >= 2, got {n_steps}")

        base_output = float(model_fn(**base_params))
        results: Dict[str, Dict[str, Any]] = {}

        for param_name, (lo, hi) in param_ranges.items():
            if param_name not in base_params:
                continue

            sweep_values = np.linspace(lo, hi, n_steps)
            outputs = np.empty(n_steps, dtype=np.float64)

            for i, val in enumerate(sweep_values):
                modified = dict(base_params)
                modified[param_name] = float(val)
                outputs[i] = float(model_fn(**modified))

            output_range = float(np.max(outputs) - np.min(outputs))
            sensitivity_index = (
                output_range / abs(base_output) if base_output != 0 else float("inf")
            )

            # Elasticity: (dOutput/dParam) * (param/output) at base
            base_val = base_params[param_name]
            if abs(base_val) > 1e-12 and abs(base_output) > 1e-12:
                delta = (hi - lo) / (n_steps - 1)
                # Find index closest to base value
                idx = int(np.argmin(np.abs(sweep_values - base_val)))
                if idx < n_steps - 1:
                    d_output = outputs[idx + 1] - outputs[idx]
                    d_param = sweep_values[idx + 1] - sweep_values[idx]
                    elasticity = (d_output / d_param) * (base_val / base_output)
                else:
                    d_output = outputs[idx] - outputs[idx - 1]
                    d_param = sweep_values[idx] - sweep_values[idx - 1]
                    elasticity = (d_output / d_param) * (base_val / base_output)
                elasticity = float(elasticity)
            else:
                elasticity = 0.0

            results[param_name] = {
                "values": sweep_values.tolist(),
                "outputs": outputs.tolist(),
                "sensitivity_index": sensitivity_index,
                "elasticity": elasticity,
            }

        return results

    # ------------------------------------------------------------------ #
    # Bootstrap confidence intervals
    # ------------------------------------------------------------------ #

    def bootstrap_confidence(
        self,
        observed_times: Sequence[float],
        model_fn: Callable[..., NDArray[np.floating]],
        initial_params: Dict[str, float],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Bootstrap resampling for parameter confidence intervals.

        Resamples the observed data with replacement *n_bootstrap* times,
        refitting the model each time, and computing percentile-based
        confidence intervals for each parameter.

        Args:
            observed_times: Observed response times.
            model_fn: Model callable (see :meth:`calibrate`).
            initial_params: Starting parameter values.
            n_bootstrap: Number of bootstrap iterations (default 1000).
            confidence_level: Confidence level (default 0.95).
            param_bounds: Optional parameter bounds.

        Returns:
            Dictionary mapping each parameter name to a dict with keys
            ``mean``, ``std``, ``ci_lower``, ``ci_upper``.

        References:
            Efron, B. & Tibshirani, R.J. (1993). An Introduction to the
            Bootstrap. Chapman & Hall.
        """
        if n_bootstrap < 10:
            raise ValueError(f"n_bootstrap must be >= 10, got {n_bootstrap}")

        observed = np.asarray(observed_times, dtype=np.float64)
        n = len(observed)
        param_names = list(initial_params.keys())

        rng = np.random.default_rng(seed=0)
        bootstrap_params: Dict[str, List[float]] = {k: [] for k in param_names}

        alpha = 1.0 - confidence_level
        lo_pct = 100.0 * (alpha / 2.0)
        hi_pct = 100.0 * (1.0 - alpha / 2.0)

        for _ in range(n_bootstrap):
            sample_idx = rng.integers(0, n, size=n)
            sample = observed[sample_idx]

            try:
                fitted = self.calibrate(
                    sample, model_fn, initial_params, param_bounds
                )
                for k in param_names:
                    bootstrap_params[k].append(fitted[k])
            except Exception:
                # Skip failed fits (rare for well-posed problems)
                continue

        results: Dict[str, Dict[str, float]] = {}
        for k in param_names:
            values = np.array(bootstrap_params[k], dtype=np.float64)
            if len(values) == 0:
                results[k] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "ci_lower": float("nan"),
                    "ci_upper": float("nan"),
                }
            else:
                results[k] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "ci_lower": float(np.percentile(values, lo_pct)),
                    "ci_upper": float(np.percentile(values, hi_pct)),
                }

        return results

    # ------------------------------------------------------------------ #
    # Parameter identifiability
    # ------------------------------------------------------------------ #

    @staticmethod
    def parameter_identifiability(
        model_fn: Callable[..., float],
        params: Dict[str, float],
        param_ranges: Dict[str, Tuple[float, float]],
        n_samples: int = 500,
    ) -> Dict[str, Dict[str, Any]]:
        """Assess whether model parameters are identifiable from data.

        Uses a collinearity-based diagnostic: for each pair of parameters,
        measures the correlation between their effects on model output
        across random samples in the parameter space. High correlation
        indicates potential non-identifiability.

        Also computes the condition number of the local sensitivity matrix
        (Jacobian) as a global identifiability indicator.

        Args:
            model_fn: Callable accepting ``**params`` returning a scalar.
            params: Nominal parameter values.
            param_ranges: Parameter bounds for sampling.
            n_samples: Number of random samples (default 500).

        Returns:
            Dictionary with:
                - ``pairwise_correlations``: correlation matrix as nested dict.
                - ``condition_number``: condition number of the Jacobian.
                - ``identifiable``: dict of boolean flags per parameter.
        """
        if n_samples < 10:
            raise ValueError(f"n_samples must be >= 10, got {n_samples}")

        param_names = [k for k in params if k in param_ranges]
        n_params = len(param_names)
        if n_params == 0:
            return {
                "pairwise_correlations": {},
                "condition_number": float("nan"),
                "identifiable": {},
            }

        rng = np.random.default_rng(seed=12345)

        # Build sensitivity matrix: rows = samples, cols = parameters
        sensitivity_matrix = np.zeros((n_samples, n_params), dtype=np.float64)
        delta_frac = 0.01  # 1% perturbation for finite-difference Jacobian

        for s in range(n_samples):
            # Random point in parameter space
            sample_params = dict(params)
            for k in param_names:
                lo, hi = param_ranges[k]
                sample_params[k] = float(rng.uniform(lo, hi))

            base_output = float(model_fn(**sample_params))

            for j, k in enumerate(param_names):
                perturbed = dict(sample_params)
                delta = abs(sample_params[k]) * delta_frac
                if delta < 1e-12:
                    delta = 1e-6
                perturbed[k] = sample_params[k] + delta
                perturbed_output = float(model_fn(**perturbed))
                sensitivity_matrix[s, j] = (perturbed_output - base_output) / delta

        # Pairwise correlations
        corr_matrix = np.corrcoef(sensitivity_matrix, rowvar=False)
        # Handle NaN from constant columns
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        pairwise: Dict[str, Dict[str, float]] = {}
        for i, ki in enumerate(param_names):
            pairwise[ki] = {}
            for j, kj in enumerate(param_names):
                pairwise[ki][kj] = float(corr_matrix[i, j])

        # Condition number of the sensitivity matrix
        try:
            cond = float(np.linalg.cond(sensitivity_matrix))
        except np.linalg.LinAlgError:
            cond = float("inf")

        # Identifiability: parameter is identifiable if its column in the
        # sensitivity matrix has sufficient variance and low collinearity
        # with other parameters (|correlation| < 0.95 with all others).
        identifiable: Dict[str, bool] = {}
        col_threshold = 0.95
        for i, ki in enumerate(param_names):
            col_var = np.var(sensitivity_matrix[:, i])
            has_variance = col_var > 1e-12
            low_collinearity = all(
                abs(corr_matrix[i, j]) < col_threshold
                for j in range(n_params)
                if j != i
            )
            identifiable[ki] = has_variance and low_collinearity

        return {
            "pairwise_correlations": pairwise,
            "condition_number": cond,
            "identifiable": identifiable,
        }
