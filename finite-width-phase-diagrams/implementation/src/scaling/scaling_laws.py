"""Neural scaling laws: fitting, prediction, and optimal compute allocation."""

import numpy as np
from scipy import optimize, stats
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Callable, Any


@dataclass
class FitResult:
    """Result of a scaling law fit."""
    params: np.ndarray
    param_names: List[str]
    residuals: np.ndarray
    r_squared: float
    aic: float
    bic: float
    covariance: Optional[np.ndarray] = None
    model_fn: Optional[Callable] = None
    x_range: Optional[Tuple[float, float]] = None
    label: str = ""


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(np.asarray(x, dtype=np.float64), 1e-300))


def _compute_aic(n: int, k: int, rss: float) -> float:
    if rss <= 0 or n <= k + 1:
        return np.inf
    return n * np.log(rss / n) + 2 * k


def _compute_bic(n: int, k: int, rss: float) -> float:
    if rss <= 0 or n <= k + 1:
        return np.inf
    return n * np.log(rss / n) + k * np.log(n)


def _r_squared(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


class ScalingExponentComputer:
    """Compute scaling exponents from empirical data."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def _fit_log_linear(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
        """Fit log(y) = a + b*log(x). Returns (exponent, intercept, r2, residuals)."""
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        mask = (x > 0) & (y > 0)
        lx, ly = np.log(x[mask]), np.log(y[mask])
        slope, intercept, r_value, _, _ = stats.linregress(lx, ly)
        predicted = intercept + slope * lx
        residuals = ly - predicted
        return slope, intercept, r_value ** 2, residuals

    def loss_vs_compute(self, compute_values: np.ndarray, loss_values: np.ndarray) -> Dict[str, float]:
        """Fit L ~ C^{-alpha_C} and return exponent info."""
        slope, intercept, r2, residuals = self._fit_log_linear(compute_values, loss_values)
        alpha_C = -slope
        return {
            "alpha_C": alpha_C,
            "coefficient": np.exp(intercept),
            "r_squared": r2,
            "residual_std": np.std(residuals),
            "n_points": len(compute_values),
        }

    def loss_vs_params(self, param_counts: np.ndarray, loss_values: np.ndarray) -> Dict[str, float]:
        """Fit L ~ N^{-alpha_N}."""
        slope, intercept, r2, residuals = self._fit_log_linear(param_counts, loss_values)
        alpha_N = -slope
        return {
            "alpha_N": alpha_N,
            "coefficient": np.exp(intercept),
            "r_squared": r2,
            "residual_std": np.std(residuals),
            "n_points": len(param_counts),
        }

    def loss_vs_data(self, dataset_sizes: np.ndarray, loss_values: np.ndarray) -> Dict[str, float]:
        """Fit L ~ D^{-alpha_D}."""
        slope, intercept, r2, residuals = self._fit_log_linear(dataset_sizes, loss_values)
        alpha_D = -slope
        return {
            "alpha_D": alpha_D,
            "coefficient": np.exp(intercept),
            "r_squared": r2,
            "residual_std": np.std(residuals),
            "n_points": len(dataset_sizes),
        }

    def loss_vs_width(self, widths: np.ndarray, loss_values: np.ndarray) -> Dict[str, float]:
        """Fit L ~ n^{-alpha_n} for network width n."""
        slope, intercept, r2, residuals = self._fit_log_linear(widths, loss_values)
        alpha_n = -slope
        return {
            "alpha_n": alpha_n,
            "coefficient": np.exp(intercept),
            "r_squared": r2,
            "residual_std": np.std(residuals),
            "n_points": len(widths),
        }

    def compute_all_exponents(self, results_dict: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, float]]:
        """Extract all scaling exponents from a results dictionary.

        Expects keys like 'compute', 'params', 'data', 'width', each mapping
        to {'x': array, 'loss': array}.
        """
        exponents = {}
        dispatch = {
            "compute": self.loss_vs_compute,
            "params": self.loss_vs_params,
            "data": self.loss_vs_data,
            "width": self.loss_vs_width,
        }
        for key, fn in dispatch.items():
            if key in results_dict:
                entry = results_dict[key]
                exponents[key] = fn(np.asarray(entry["x"]), np.asarray(entry["loss"]))
        return exponents

    def exponent_uncertainties(
        self, x: np.ndarray, y: np.ndarray, n_bootstrap: int = 1000
    ) -> Dict[str, float]:
        """Bootstrap uncertainty estimates on the scaling exponent."""
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        n = len(x)
        rng = np.random.default_rng(42)
        exponents = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            slope, _, _, _ = self._fit_log_linear(x[idx], y[idx])
            exponents[i] = -slope
        return {
            "mean": np.mean(exponents),
            "std": np.std(exponents),
            "ci_lower": np.percentile(exponents, 2.5),
            "ci_upper": np.percentile(exponents, 97.5),
            "median": np.median(exponents),
        }

    def effective_exponent(
        self, x: np.ndarray, y: np.ndarray, window_size: int = 5
    ) -> Dict[str, np.ndarray]:
        """Compute local (running) exponent using a sliding window in log space."""
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        order = np.argsort(x)
        x, y = x[order], y[order]
        lx, ly = _safe_log(x), _safe_log(y)
        n = len(x)
        if window_size > n:
            window_size = n
        n_windows = n - window_size + 1
        centers = np.empty(n_windows)
        local_exponents = np.empty(n_windows)
        for i in range(n_windows):
            seg_x = lx[i : i + window_size]
            seg_y = ly[i : i + window_size]
            slope, _, _, _, _ = stats.linregress(seg_x, seg_y)
            local_exponents[i] = -slope
            centers[i] = np.exp(np.mean(seg_x))
        return {"centers": centers, "exponents": local_exponents}

    def exponent_stability(
        self, x: np.ndarray, y: np.ndarray, min_points_range: Tuple[int, int] = (5, 20)
    ) -> Dict[str, Any]:
        """Assess stability of exponent estimate as fitting range varies."""
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        order = np.argsort(x)
        x, y = x[order], y[order]
        n = len(x)
        lo, hi = min_points_range
        hi = min(hi, n)
        results = []
        for min_pts in range(lo, hi + 1):
            # Fit using last min_pts points (largest x)
            slope, _, r2, _ = self._fit_log_linear(x[-min_pts:], y[-min_pts:])
            results.append({"n_points": min_pts, "exponent": -slope, "r_squared": r2})
        exps = [r["exponent"] for r in results]
        return {
            "fits": results,
            "exponent_range": (min(exps), max(exps)),
            "exponent_std": np.std(exps),
            "is_stable": np.std(exps) < 0.05 * abs(np.mean(exps)) if np.mean(exps) != 0 else True,
        }

    def theoretical_exponent_comparison(
        self, measured: Dict[str, float], theoretical: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Compare measured exponents against theoretical predictions."""
        comparison = {}
        for key in measured:
            if key in theoretical:
                m, t = measured[key], theoretical[key]
                diff = m - t
                rel_diff = diff / abs(t) if t != 0 else float("inf")
                comparison[key] = {
                    "measured": m,
                    "theoretical": t,
                    "absolute_difference": diff,
                    "relative_difference": rel_diff,
                    "consistent": abs(rel_diff) < 0.15,
                }
        return comparison


class ScalingLawFitter:
    """Fit various scaling law functional forms to data."""

    def __init__(self, law_type: str = "power"):
        self.law_type = law_type

    def fit_power_law(self, x: np.ndarray, y: np.ndarray) -> FitResult:
        """Fit L = a * x^b + c via nonlinear least squares."""
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        # Initial guess from log-linear fit
        mask = (x > 0) & (y > 0)
        slope, intercept, _, _, _ = stats.linregress(np.log(x[mask]), np.log(y[mask]))
        a0 = np.exp(intercept)
        b0 = slope
        c0 = 0.0

        def model(xv, a, b, c):
            return a * np.power(xv, b) + c

        try:
            popt, pcov = optimize.curve_fit(
                model, x, y, p0=[a0, b0, c0], maxfev=10000,
                bounds=([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
            )
        except RuntimeError:
            popt = np.array([a0, b0, c0])
            pcov = np.full((3, 3), np.nan)

        y_pred = model(x, *popt)
        residuals = y - y_pred
        rss = np.sum(residuals ** 2)
        n, k = len(x), 3
        return FitResult(
            params=popt,
            param_names=["a", "b", "c"],
            residuals=residuals,
            r_squared=_r_squared(y, y_pred),
            aic=_compute_aic(n, k, rss),
            bic=_compute_bic(n, k, rss),
            covariance=pcov,
            model_fn=lambda xv: model(xv, *popt),
            x_range=(float(x.min()), float(x.max())),
            label="power_law",
        )

    def fit_broken_power_law(self, x: np.ndarray, y: np.ndarray) -> FitResult:
        """Fit a broken (two-regime) power law with a breakpoint.

        For x < x_break: y = a1 * x^b1
        For x >= x_break: y = a1 * x_break^(b1-b2) * x^b2   (continuity)
        """
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        order = np.argsort(x)
        x, y = x[order], y[order]

        slope, intercept, _, _ = self._log_linear(x, y)
        x_break0 = np.exp(np.mean(np.log(x[x > 0])))

        def model(xv, a1, b1, b2, x_break):
            x_break = max(x_break, xv.min() * 1.01)
            out = np.empty_like(xv)
            lo = xv < x_break
            hi = ~lo
            out[lo] = a1 * np.power(xv[lo], b1)
            out[hi] = a1 * np.power(x_break, b1 - b2) * np.power(xv[hi], b2)
            return out

        try:
            popt, pcov = optimize.curve_fit(
                model, x, y,
                p0=[np.exp(intercept), slope, slope * 0.5, x_break0],
                maxfev=20000,
            )
        except RuntimeError:
            popt = np.array([np.exp(intercept), slope, slope * 0.5, x_break0])
            pcov = np.full((4, 4), np.nan)

        y_pred = model(x, *popt)
        residuals = y - y_pred
        rss = np.sum(residuals ** 2)
        n, k = len(x), 4
        return FitResult(
            params=popt,
            param_names=["a1", "b1", "b2", "x_break"],
            residuals=residuals,
            r_squared=_r_squared(y, y_pred),
            aic=_compute_aic(n, k, rss),
            bic=_compute_bic(n, k, rss),
            covariance=pcov,
            model_fn=lambda xv: model(np.asarray(xv), *popt),
            x_range=(float(x.min()), float(x.max())),
            label="broken_power_law",
        )

    def fit_bilinear_log(self, x: np.ndarray, y: np.ndarray) -> FitResult:
        """Fit a bilinear model in log-log space with a hinge point.

        log(y) = { a1 + b1*log(x)                       if log(x) < h
                 { a1 + b1*h + b2*(log(x) - h)           if log(x) >= h
        """
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        lx, ly = _safe_log(x), _safe_log(y)
        h0 = np.median(lx)
        slope, intercept, _, _, _ = stats.linregress(lx, ly)

        def model_log(lxv, a1, b1, b2, h):
            out = np.where(lxv < h, a1 + b1 * lxv, a1 + b1 * h + b2 * (lxv - h))
            return out

        try:
            popt, pcov = optimize.curve_fit(
                model_log, lx, ly,
                p0=[intercept, slope, slope * 0.5, h0],
                maxfev=15000,
            )
        except RuntimeError:
            popt = np.array([intercept, slope, slope * 0.5, h0])
            pcov = np.full((4, 4), np.nan)

        ly_pred = model_log(lx, *popt)
        y_pred = np.exp(ly_pred)
        residuals = y - y_pred
        rss = np.sum(residuals ** 2)
        n, k = len(x), 4
        return FitResult(
            params=popt,
            param_names=["a1", "b1", "b2", "h"],
            residuals=residuals,
            r_squared=_r_squared(y, y_pred),
            aic=_compute_aic(n, k, rss),
            bic=_compute_bic(n, k, rss),
            covariance=pcov,
            model_fn=lambda xv: np.exp(model_log(_safe_log(np.asarray(xv)), *popt)),
            x_range=(float(x.min()), float(x.max())),
            label="bilinear_log",
        )

    def fit_chinchilla(self, N: np.ndarray, D: np.ndarray, L: np.ndarray) -> FitResult:
        """Fit Chinchilla scaling law: L = E + A/N^alpha + B/D^beta."""
        N, D, L = (np.asarray(v, dtype=np.float64) for v in (N, D, L))

        def model(X, E, A, alpha, B, beta):
            n, d = X
            return E + A * np.power(n, -alpha) + B * np.power(d, -beta)

        p0 = [1.69, 406.4, 0.34, 410.7, 0.28]
        bounds_lo = [0, 0, 0.01, 0, 0.01]
        bounds_hi = [10, 1e6, 2.0, 1e6, 2.0]
        try:
            popt, pcov = optimize.curve_fit(
                model, (N, D), L, p0=p0, maxfev=30000,
                bounds=(bounds_lo, bounds_hi),
            )
        except RuntimeError:
            popt = np.array(p0)
            pcov = np.full((5, 5), np.nan)

        y_pred = model((N, D), *popt)
        residuals = L - y_pred
        rss = np.sum(residuals ** 2)
        n, k = len(L), 5
        return FitResult(
            params=popt,
            param_names=["E", "A", "alpha", "B", "beta"],
            residuals=residuals,
            r_squared=_r_squared(L, y_pred),
            aic=_compute_aic(n, k, rss),
            bic=_compute_bic(n, k, rss),
            covariance=pcov,
            model_fn=lambda X: model(X, *popt),
            x_range=None,
            label="chinchilla",
        )

    def fit_openai(self, N: np.ndarray, D: np.ndarray, L: np.ndarray) -> FitResult:
        """Fit OpenAI scaling law: L = [(N_c/N)^(alpha_N/alpha_D) + (D_c/D)]^alpha_D.

        Simplified parametric form:
        L = (a * N^{-b} + c * D^{-d})^e
        """
        N, D, L = (np.asarray(v, dtype=np.float64) for v in (N, D, L))

        def model(X, a, b, c, d, e):
            n, dd = X
            inner = a * np.power(n, -b) + c * np.power(dd, -d)
            return np.power(np.maximum(inner, 1e-30), e)

        p0 = [8.8e3, 0.076, 5.4e3, 0.095, 0.35]
        try:
            popt, pcov = optimize.curve_fit(
                model, (N, D), L, p0=p0, maxfev=30000,
                bounds=([0, 0.001, 0, 0.001, 0.01], [1e8, 5, 1e8, 5, 5]),
            )
        except RuntimeError:
            popt = np.array(p0)
            pcov = np.full((5, 5), np.nan)

        y_pred = model((N, D), *popt)
        residuals = L - y_pred
        rss = np.sum(residuals ** 2)
        n_pts, k = len(L), 5
        return FitResult(
            params=popt,
            param_names=["a", "b", "c", "d", "e"],
            residuals=residuals,
            r_squared=_r_squared(L, y_pred),
            aic=_compute_aic(n_pts, k, rss),
            bic=_compute_bic(n_pts, k, rss),
            covariance=pcov,
            model_fn=lambda X: model(X, *popt),
            x_range=None,
            label="openai",
        )

    def model_selection(self, x: np.ndarray, y: np.ndarray, models: List[str]) -> Dict[str, Any]:
        """Compare models using AIC and BIC. Returns ranking and weights."""
        fit_dispatch = {
            "power": self.fit_power_law,
            "broken_power": self.fit_broken_power_law,
            "bilinear_log": self.fit_bilinear_log,
        }
        results = {}
        for name in models:
            if name in fit_dispatch:
                results[name] = fit_dispatch[name](x, y)

        aics = {name: r.aic for name, r in results.items()}
        bics = {name: r.bic for name, r in results.items()}

        min_aic = min(aics.values())
        delta_aics = {name: v - min_aic for name, v in aics.items()}
        # Akaike weights
        raw_weights = {name: np.exp(-0.5 * d) for name, d in delta_aics.items()}
        total = sum(raw_weights.values())
        akaike_weights = {name: w / total for name, w in raw_weights.items()}

        ranking = sorted(results.keys(), key=lambda n: aics[n])
        return {
            "fits": results,
            "aic": aics,
            "bic": bics,
            "akaike_weights": akaike_weights,
            "ranking": ranking,
            "best": ranking[0],
        }

    def cross_validate(
        self, x: np.ndarray, y: np.ndarray, model_fn: Callable, k_folds: int = 5
    ) -> Dict[str, float]:
        """K-fold cross validation for a model fitting function.

        model_fn(x_train, y_train) -> callable that predicts y from x.
        """
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        n = len(x)
        indices = np.arange(n)
        rng = np.random.default_rng(0)
        rng.shuffle(indices)
        fold_size = n // k_folds
        mse_folds = []
        mae_folds = []
        for k in range(k_folds):
            start = k * fold_size
            end = start + fold_size if k < k_folds - 1 else n
            val_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            predictor = model_fn(x[train_idx], y[train_idx])
            y_pred = predictor(x[val_idx])
            mse_folds.append(np.mean((y[val_idx] - y_pred) ** 2))
            mae_folds.append(np.mean(np.abs(y[val_idx] - y_pred)))
        return {
            "mean_mse": float(np.mean(mse_folds)),
            "std_mse": float(np.std(mse_folds)),
            "mean_mae": float(np.mean(mae_folds)),
            "std_mae": float(np.std(mae_folds)),
            "fold_mses": mse_folds,
        }

    def residual_analysis(self, x: np.ndarray, y: np.ndarray, fit_fn: Callable) -> Dict[str, Any]:
        """Analyze residuals for systematic deviations.

        fit_fn: callable x -> predicted y (already fitted).
        """
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        y_pred = fit_fn(x)
        residuals = y - y_pred
        std_resid = residuals / np.std(residuals) if np.std(residuals) > 0 else residuals

        # Durbin-Watson statistic for autocorrelation
        order = np.argsort(x)
        sorted_res = residuals[order]
        dw_num = np.sum(np.diff(sorted_res) ** 2)
        dw_den = np.sum(sorted_res ** 2)
        durbin_watson = dw_num / dw_den if dw_den > 0 else 2.0

        # Runs test for randomness
        signs = sorted_res > 0
        n_runs = 1 + np.sum(np.diff(signs.astype(int)) != 0)
        n_pos = np.sum(signs)
        n_neg = len(signs) - n_pos

        # Shapiro-Wilk normality test on residuals
        if len(residuals) >= 3:
            sw_stat, sw_p = stats.shapiro(residuals)
        else:
            sw_stat, sw_p = np.nan, np.nan

        # Trend in residuals (linear fit to residuals vs log(x))
        lx = _safe_log(x[order])
        trend_slope, _, _, trend_p, _ = stats.linregress(lx, sorted_res)

        return {
            "residuals": residuals,
            "standardized_residuals": std_resid,
            "durbin_watson": durbin_watson,
            "autocorrelation_flag": durbin_watson < 1.5 or durbin_watson > 2.5,
            "n_runs": int(n_runs),
            "shapiro_stat": float(sw_stat),
            "shapiro_p": float(sw_p),
            "normality_flag": sw_p < 0.05 if not np.isnan(sw_p) else False,
            "trend_slope": float(trend_slope),
            "trend_p_value": float(trend_p),
            "systematic_bias_flag": trend_p < 0.05,
        }

    def prediction_intervals(
        self, x_new: np.ndarray, fit_result: FitResult, confidence: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """Compute prediction intervals at new x values using delta method."""
        x_new = np.asarray(x_new, dtype=np.float64)
        y_pred = fit_result.model_fn(x_new)
        residual_std = np.std(fit_result.residuals)
        n = len(fit_result.residuals)
        k = len(fit_result.params)
        dof = max(n - k, 1)
        t_val = stats.t.ppf((1 + confidence) / 2, dof)
        se = residual_std * np.sqrt(1 + 1.0 / n)
        margin = t_val * se
        return {
            "predicted": y_pred,
            "lower": y_pred - margin,
            "upper": y_pred + margin,
            "margin": margin,
            "confidence": confidence,
        }

    def joint_fit(
        self, N_values: np.ndarray, D_values: np.ndarray, L_values: np.ndarray
    ) -> FitResult:
        """Joint fit of L(N, D) using Chinchilla functional form."""
        return self.fit_chinchilla(N_values, D_values, L_values)

    @staticmethod
    def _log_linear(x, y):
        mask = (x > 0) & (y > 0)
        return stats.linregress(np.log(x[mask]), np.log(y[mask]))[:4]


class ScalingLawPredictor:
    """Predict performance at new scales using a fitted scaling law."""

    def __init__(self, fitted_law: FitResult):
        self.fitted_law = fitted_law
        self._model_fn = fitted_law.model_fn

    def predict_loss(self, x: np.ndarray) -> np.ndarray:
        """Predict loss at given x values."""
        return self._model_fn(np.asarray(x, dtype=np.float64))

    def predict_at_scale(self, target_compute: float) -> float:
        """Predict loss at a target compute budget."""
        return float(self._model_fn(np.array([target_compute]))[0])

    def compute_for_target_loss(self, target_loss: float) -> float:
        """Find compute needed to reach target loss via root finding."""
        return self._invert(target_loss)

    def params_for_target_loss(self, target_loss: float) -> float:
        """Find parameter count needed to reach target loss."""
        return self._invert(target_loss)

    def data_for_target_loss(self, target_loss: float) -> float:
        """Find dataset size needed to reach target loss."""
        return self._invert(target_loss)

    def _invert(self, target_loss: float) -> float:
        """Invert the fitted law to find x for a given loss."""
        x_lo, x_hi = self.fitted_law.x_range or (1.0, 1e15)
        # Extend search range
        x_lo_search = x_lo * 0.01
        x_hi_search = x_hi * 100

        def objective(log_x):
            x_val = np.exp(log_x)
            return float(self._model_fn(np.array([x_val]))[0]) - target_loss

        # Check bracketing
        f_lo = objective(np.log(x_lo_search))
        f_hi = objective(np.log(x_hi_search))

        if f_lo * f_hi > 0:
            # No sign change; use minimization of |f|
            result = optimize.minimize_scalar(
                lambda lx: abs(objective(lx)),
                bounds=(np.log(x_lo_search), np.log(x_hi_search)),
                method="bounded",
            )
            return float(np.exp(result.x))

        sol = optimize.brentq(objective, np.log(x_lo_search), np.log(x_hi_search))
        return float(np.exp(sol))

    def extrapolation_reliability(
        self, x_target: float, x_train_range: Tuple[float, float]
    ) -> Dict[str, Any]:
        """Assess reliability of extrapolation to x_target."""
        x_lo, x_hi = x_train_range
        log_range = np.log(x_hi) - np.log(x_lo)
        if x_target > x_hi:
            log_extrap = np.log(x_target) - np.log(x_hi)
        elif x_target < x_lo:
            log_extrap = np.log(x_lo) - np.log(x_target)
        else:
            log_extrap = 0.0

        extrap_ratio = log_extrap / log_range if log_range > 0 else 0.0
        # Heuristic: reliability decays exponentially with extrapolation distance
        reliability = np.exp(-0.5 * extrap_ratio)
        return {
            "x_target": x_target,
            "train_range": x_train_range,
            "log_extrapolation_distance": float(log_extrap),
            "extrapolation_ratio": float(extrap_ratio),
            "reliability_score": float(reliability),
            "is_interpolation": x_lo <= x_target <= x_hi,
            "warning": "high" if extrap_ratio > 2 else ("moderate" if extrap_ratio > 1 else "low"),
        }

    def prediction_vs_actual(
        self, x_values: np.ndarray, actual_losses: np.ndarray
    ) -> Dict[str, Any]:
        """Compare predictions against actual observed losses."""
        x_values = np.asarray(x_values, dtype=np.float64)
        actual_losses = np.asarray(actual_losses, dtype=np.float64)
        predicted = self._model_fn(x_values)
        errors = actual_losses - predicted
        rel_errors = errors / actual_losses
        return {
            "predicted": predicted,
            "actual": actual_losses,
            "errors": errors,
            "relative_errors": rel_errors,
            "mse": float(np.mean(errors ** 2)),
            "mae": float(np.mean(np.abs(errors))),
            "mape": float(np.mean(np.abs(rel_errors)) * 100),
            "max_error": float(np.max(np.abs(errors))),
            "r_squared": _r_squared(actual_losses, predicted),
        }

    def forecast_horizon(self, fit_result: FitResult, tolerance: float = 0.1) -> Dict[str, float]:
        """Estimate maximum reliable extrapolation distance.

        Uses residual growth rate to estimate where prediction error
        exceeds tolerance fraction of predicted value.
        """
        x_lo, x_hi = fit_result.x_range or (1.0, 1e6)
        residual_std = np.std(fit_result.residuals)
        y_at_hi = float(self._model_fn(np.array([x_hi]))[0])

        if y_at_hi <= 0:
            return {"max_x": x_hi, "log_decades": 0.0, "limited_by": "zero_loss"}

        # Find x where predicted uncertainty matches tolerance * predicted value
        target_uncertainty = tolerance * y_at_hi
        if residual_std <= 0:
            return {"max_x": x_hi * 1000, "log_decades": 3.0, "limited_by": "zero_residuals"}

        # Heuristic: uncertainty grows as sqrt(extrapolation_distance)
        # residual_std * sqrt(1 + d/range) = tolerance * y_pred
        range_log = np.log(x_hi) - np.log(x_lo)
        max_d = range_log * ((target_uncertainty / residual_std) ** 2 - 1)
        max_d = max(max_d, 0)
        max_x = x_hi * np.exp(max_d)
        log_decades = max_d / np.log(10)
        return {
            "max_x": float(max_x),
            "log_decades_beyond_training": float(log_decades),
            "limited_by": "residual_growth",
            "tolerance": tolerance,
        }


class ChinchillaAllocator:
    """Chinchilla-style optimal compute allocation between parameters and data."""

    def __init__(
        self,
        alpha_N: float = 0.34,
        alpha_D: float = 0.28,
        A: float = 406.4,
        B: float = 410.7,
        E: float = 1.69,
    ):
        self.alpha_N = alpha_N
        self.alpha_D = alpha_D
        self.A = A
        self.B = B
        self.E = E

    def _loss(self, N: float, D: float) -> float:
        return self.E + self.A * N ** (-self.alpha_N) + self.B * D ** (-self.alpha_D)

    def optimal_params_for_compute(
        self, compute_budget: float, flops_per_token: float = 6.0
    ) -> float:
        """Compute optimal parameter count N*(C).

        C ≈ flops_per_token * N * D, so D = C / (flops_per_token * N).
        Minimize L(N, C/(fpt*N)) over N.
        """
        def loss_of_N(log_N):
            N = np.exp(log_N)
            D = compute_budget / (flops_per_token * N)
            if D < 1:
                return 1e30
            return self._loss(N, D)

        log_N_lo = np.log(1e3)
        log_N_hi = np.log(compute_budget / flops_per_token)
        result = optimize.minimize_scalar(loss_of_N, bounds=(log_N_lo, log_N_hi), method="bounded")
        return float(np.exp(result.x))

    def optimal_data_for_compute(
        self, compute_budget: float, flops_per_token: float = 6.0
    ) -> float:
        """Compute optimal dataset size D*(C)."""
        N_star = self.optimal_params_for_compute(compute_budget, flops_per_token)
        return compute_budget / (flops_per_token * N_star)

    def optimal_allocation(self, compute_budget: float, flops_per_token: float = 6.0) -> Dict[str, float]:
        """Return the optimal (N*, D*) pair for a given compute budget."""
        N_star = self.optimal_params_for_compute(compute_budget, flops_per_token)
        D_star = compute_budget / (flops_per_token * N_star)
        return {
            "N_star": N_star,
            "D_star": D_star,
            "compute": compute_budget,
            "loss": self._loss(N_star, D_star),
            "tokens_per_param": D_star / N_star,
        }

    def loss_at_optimal(self, compute_budget: float, flops_per_token: float = 6.0) -> float:
        """Loss achieved at optimal allocation."""
        alloc = self.optimal_allocation(compute_budget, flops_per_token)
        return alloc["loss"]

    def iso_loss_curves(
        self,
        loss_levels: List[float],
        N_range: Tuple[float, float],
        D_range: Tuple[float, float],
        n_points: int = 200,
    ) -> Dict[float, Dict[str, np.ndarray]]:
        """Compute curves of constant loss in (N, D) space.

        For each loss level, find D(N) such that L(N, D) = target.
        """
        curves = {}
        N_vals = np.geomspace(N_range[0], N_range[1], n_points)
        for target_L in loss_levels:
            D_vals = np.empty(n_points)
            valid = np.ones(n_points, dtype=bool)
            for i, N in enumerate(N_vals):
                # L = E + A/N^alpha_N + B/D^alpha_D = target_L
                # B/D^alpha_D = target_L - E - A/N^alpha_N
                remainder = target_L - self.E - self.A * N ** (-self.alpha_N)
                if remainder <= 0:
                    valid[i] = False
                    D_vals[i] = np.nan
                else:
                    D_vals[i] = (self.B / remainder) ** (1.0 / self.alpha_D)
                    if D_vals[i] < D_range[0] or D_vals[i] > D_range[1]:
                        valid[i] = False
            curves[target_L] = {
                "N": N_vals[valid],
                "D": D_vals[valid],
            }
        return curves

    def compute_efficiency(self, N: float, D: float, loss: Optional[float] = None) -> Dict[str, float]:
        """Measure how efficient an (N, D) allocation is vs Chinchilla optimal.

        Efficiency = L_optimal(C) / L_actual where C = 6*N*D.
        """
        C = 6.0 * N * D
        L_actual = loss if loss is not None else self._loss(N, D)
        L_optimal = self.loss_at_optimal(C)
        alloc = self.optimal_allocation(C)
        return {
            "compute": C,
            "actual_loss": L_actual,
            "optimal_loss": L_optimal,
            "efficiency": L_optimal / L_actual if L_actual > 0 else 0.0,
            "loss_gap": L_actual - L_optimal,
            "N_ratio": N / alloc["N_star"],
            "D_ratio": D / alloc["D_star"],
            "over_parameterized": N > alloc["N_star"],
        }

    def pareto_frontier(
        self, N_values: np.ndarray, D_values: np.ndarray, losses: np.ndarray
    ) -> Dict[str, Any]:
        """Find Pareto-optimal (N, D) pairs: no other point has both lower loss and lower compute."""
        N_values = np.asarray(N_values, dtype=np.float64)
        D_values = np.asarray(D_values, dtype=np.float64)
        losses = np.asarray(losses, dtype=np.float64)
        computes = 6.0 * N_values * D_values

        # Sort by compute
        order = np.argsort(computes)
        sorted_C = computes[order]
        sorted_L = losses[order]
        sorted_N = N_values[order]
        sorted_D = D_values[order]

        pareto_mask = np.ones(len(order), dtype=bool)
        min_loss = np.inf
        for i in range(len(order)):
            if sorted_L[i] < min_loss:
                min_loss = sorted_L[i]
            else:
                pareto_mask[i] = False

        return {
            "pareto_N": sorted_N[pareto_mask],
            "pareto_D": sorted_D[pareto_mask],
            "pareto_losses": sorted_L[pareto_mask],
            "pareto_computes": sorted_C[pareto_mask],
            "n_pareto": int(np.sum(pareto_mask)),
            "n_total": len(order),
        }

    def budget_sensitivity(
        self, compute_budget: float, perturbation: float = 0.1
    ) -> Dict[str, Any]:
        """Sensitivity of optimal allocation to perturbations in compute budget."""
        base = self.optimal_allocation(compute_budget)
        budgets = compute_budget * np.array([1 - perturbation, 1.0, 1 + perturbation])
        allocs = [self.optimal_allocation(float(c)) for c in budgets]

        dN_dC = (allocs[2]["N_star"] - allocs[0]["N_star"]) / (budgets[2] - budgets[0])
        dD_dC = (allocs[2]["D_star"] - allocs[0]["D_star"]) / (budgets[2] - budgets[0])
        dL_dC = (allocs[2]["loss"] - allocs[0]["loss"]) / (budgets[2] - budgets[0])

        # Elasticities
        elasticity_N = (dN_dC * compute_budget) / base["N_star"] if base["N_star"] > 0 else 0
        elasticity_D = (dD_dC * compute_budget) / base["D_star"] if base["D_star"] > 0 else 0
        return {
            "base_allocation": base,
            "dN_dC": float(dN_dC),
            "dD_dC": float(dD_dC),
            "dL_dC": float(dL_dC),
            "elasticity_N": float(elasticity_N),
            "elasticity_D": float(elasticity_D),
            "perturbation": perturbation,
        }

    def multi_epoch_correction(self, n_epochs: int, dataset_size: float) -> Dict[str, float]:
        """Correction factor for training with multiple epochs over the same data.

        Following Muennighoff et al., repeated tokens have diminishing returns.
        Effective data D_eff = D * (1 - exp(-n_epochs * R)) / (1 - exp(-R))
        where R is a decay rate (empirically ~0.3).
        """
        R = 0.3  # empirical decay rate
        if n_epochs <= 1:
            return {
                "effective_data": dataset_size,
                "correction_factor": 1.0,
                "n_epochs": n_epochs,
            }
        D_eff_ratio = (1 - np.exp(-n_epochs * R)) / (n_epochs * (1 - np.exp(-R)))
        effective_data = dataset_size * n_epochs * D_eff_ratio
        return {
            "effective_data": float(effective_data),
            "correction_factor": float(n_epochs * D_eff_ratio),
            "n_epochs": n_epochs,
            "unique_tokens": dataset_size,
            "total_tokens": dataset_size * n_epochs,
            "efficiency_ratio": float(D_eff_ratio),
        }


class ArchitectureScalingComparator:
    """Compare scaling behavior across different architectures."""

    def __init__(self):
        self._exponent_computer = ScalingExponentComputer()
        self._fitter = ScalingLawFitter()

    def compare_exponents(
        self, arch_results: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Any]:
        """Compare scaling exponents across architectures.

        arch_results: {arch_name: {'x': array, 'loss': array}}
        """
        exponents = {}
        for arch, data in arch_results.items():
            x, y = np.asarray(data["x"]), np.asarray(data["loss"])
            slope, _, r2, _ = self._exponent_computer._fit_log_linear(x, y)
            unc = self._exponent_computer.exponent_uncertainties(x, y, n_bootstrap=500)
            exponents[arch] = {
                "exponent": -slope,
                "r_squared": r2,
                "uncertainty": unc["std"],
                "ci": (unc["ci_lower"], unc["ci_upper"]),
            }

        # Pairwise statistical comparison
        arch_names = list(exponents.keys())
        pairwise = {}
        for i, a1 in enumerate(arch_names):
            for a2 in arch_names[i + 1:]:
                e1, e2 = exponents[a1], exponents[a2]
                diff = e1["exponent"] - e2["exponent"]
                combined_se = np.sqrt(e1["uncertainty"] ** 2 + e2["uncertainty"] ** 2)
                z = diff / combined_se if combined_se > 0 else 0
                p_val = 2 * (1 - stats.norm.cdf(abs(z)))
                pairwise[f"{a1}_vs_{a2}"] = {
                    "difference": diff,
                    "z_score": z,
                    "p_value": p_val,
                    "significant": p_val < 0.05,
                }

        return {"exponents": exponents, "pairwise_comparisons": pairwise}

    def normalized_comparison(
        self,
        arch_results: Dict[str, Dict[str, np.ndarray]],
        compute_range: Tuple[float, float],
    ) -> Dict[str, Dict[str, float]]:
        """Compare architectures at matched compute levels."""
        n_points = 20
        eval_points = np.geomspace(compute_range[0], compute_range[1], n_points)
        comparison = {}
        fits = {}
        for arch, data in arch_results.items():
            fit = self._fitter.fit_power_law(np.asarray(data["x"]), np.asarray(data["loss"]))
            fits[arch] = fit
            predicted = fit.model_fn(eval_points)
            comparison[arch] = {
                "eval_computes": eval_points,
                "predicted_losses": predicted,
                "mean_loss": float(np.mean(predicted)),
                "loss_at_min_compute": float(predicted[0]),
                "loss_at_max_compute": float(predicted[-1]),
            }

        # Rank at each compute level
        arch_names = list(comparison.keys())
        rankings = np.empty((n_points, len(arch_names)), dtype=int)
        for i in range(n_points):
            losses_at_i = [comparison[a]["predicted_losses"][i] for a in arch_names]
            rankings[i] = np.argsort(np.argsort(losses_at_i))

        for j, arch in enumerate(arch_names):
            comparison[arch]["mean_rank"] = float(np.mean(rankings[:, j]))
        return comparison

    def crossover_point(
        self, arch1_fit: FitResult, arch2_fit: FitResult
    ) -> Dict[str, Any]:
        """Find the compute/scale where arch2 overtakes arch1."""
        x_lo = min(
            arch1_fit.x_range[0] if arch1_fit.x_range else 1,
            arch2_fit.x_range[0] if arch2_fit.x_range else 1,
        )
        x_hi = max(
            arch1_fit.x_range[1] if arch1_fit.x_range else 1e12,
            arch2_fit.x_range[1] if arch2_fit.x_range else 1e12,
        ) * 10

        def diff(log_x):
            x = np.exp(log_x)
            return float(arch1_fit.model_fn(np.array([x]))[0] - arch2_fit.model_fn(np.array([x]))[0])

        # Search for sign change
        test_points = np.linspace(np.log(x_lo), np.log(x_hi), 1000)
        diffs = np.array([diff(lx) for lx in test_points])
        sign_changes = np.where(np.diff(np.sign(diffs)))[0]

        if len(sign_changes) == 0:
            return {
                "crossover_exists": False,
                "arch1_always_better": bool(diffs[0] < 0),
                "search_range": (x_lo, x_hi),
            }

        # Refine first crossover
        idx = sign_changes[0]
        x_cross = optimize.brentq(diff, test_points[idx], test_points[idx + 1])
        x_cross = np.exp(x_cross)
        loss_at_cross = float(arch1_fit.model_fn(np.array([x_cross]))[0])
        return {
            "crossover_exists": True,
            "crossover_x": float(x_cross),
            "loss_at_crossover": loss_at_cross,
            "arch1_better_below": bool(diffs[0] < 0),
            "n_crossovers": len(sign_changes),
        }

    def scaling_efficiency(
        self, arch_results: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute a scaling efficiency metric for each architecture.

        Efficiency = how much loss decreases per unit increase in log(compute).
        """
        efficiencies = {}
        for arch, data in arch_results.items():
            x, y = np.asarray(data["x"], dtype=np.float64), np.asarray(data["loss"], dtype=np.float64)
            order = np.argsort(x)
            x, y = x[order], y[order]
            lx, ly = _safe_log(x), _safe_log(y)

            # Overall efficiency (negative slope = positive efficiency)
            result = stats.linregress(lx, ly)
            slope, r2 = result.slope, result.rvalue ** 2
            total_loss_reduction = ly[0] - ly[-1]
            total_compute_increase = lx[-1] - lx[0]
            marginal_eff = -slope  # loss reduction per log-compute

            # Diminishing returns: compare first half vs second half efficiency
            mid = len(x) // 2
            slope_early = stats.linregress(lx[:mid], ly[:mid])[0] if mid > 2 else slope
            slope_late = stats.linregress(lx[mid:], ly[mid:])[0] if len(x) - mid > 2 else slope

            efficiencies[arch] = {
                "marginal_efficiency": float(marginal_eff),
                "total_loss_reduction_log": float(total_loss_reduction),
                "compute_range_log": float(total_compute_increase),
                "r_squared": float(r2),
                "early_slope": float(slope_early),
                "late_slope": float(slope_late),
                "diminishing_returns_ratio": float(slope_late / slope_early) if slope_early != 0 else 1.0,
            }
        return efficiencies

    def relative_advantage(
        self,
        arch1_fit: FitResult,
        arch2_fit: FitResult,
        compute_range: Tuple[float, float],
        n_points: int = 50,
    ) -> Dict[str, np.ndarray]:
        """Compute relative advantage ratio of arch1 over arch2 across compute range."""
        computes = np.geomspace(compute_range[0], compute_range[1], n_points)
        loss1 = arch1_fit.model_fn(computes)
        loss2 = arch2_fit.model_fn(computes)
        ratio = loss1 / np.maximum(loss2, 1e-30)
        return {
            "compute": computes,
            "loss_arch1": loss1,
            "loss_arch2": loss2,
            "ratio": ratio,
            "arch1_better_mask": ratio < 1.0,
            "mean_ratio": float(np.mean(ratio)),
            "advantage_at_min": float(ratio[0]),
            "advantage_at_max": float(ratio[-1]),
        }

    def universal_scaling_check(
        self, arch_results: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Any]:
        """Test whether scaling exponents are universal across architectures.

        Uses a statistical test: if all exponents are consistent within
        uncertainty, scaling may be universal.
        """
        exponents = []
        uncertainties = []
        arch_names = []
        for arch, data in arch_results.items():
            x, y = np.asarray(data["x"]), np.asarray(data["loss"])
            slope, _, _, _ = self._exponent_computer._fit_log_linear(x, y)
            unc = self._exponent_computer.exponent_uncertainties(x, y, n_bootstrap=500)
            exponents.append(-slope)
            uncertainties.append(unc["std"])
            arch_names.append(arch)

        exponents = np.array(exponents)
        uncertainties = np.array(uncertainties)

        # Weighted mean
        weights = 1.0 / np.maximum(uncertainties ** 2, 1e-30)
        weighted_mean = np.sum(weights * exponents) / np.sum(weights)

        # Chi-squared test for consistency
        chi2 = np.sum(weights * (exponents - weighted_mean) ** 2)
        dof = len(exponents) - 1
        p_value = 1 - stats.chi2.cdf(chi2, dof) if dof > 0 else 1.0

        return {
            "exponents": dict(zip(arch_names, exponents.tolist())),
            "uncertainties": dict(zip(arch_names, uncertainties.tolist())),
            "weighted_mean_exponent": float(weighted_mean),
            "chi_squared": float(chi2),
            "degrees_of_freedom": dof,
            "p_value": float(p_value),
            "is_universal": p_value > 0.05,
            "spread": float(np.max(exponents) - np.min(exponents)),
        }

    def architecture_ranking(
        self,
        arch_results: Dict[str, Dict[str, np.ndarray]],
        compute_budget: float,
    ) -> List[Dict[str, Any]]:
        """Rank architectures by predicted loss at a given compute budget."""
        rankings = []
        for arch, data in arch_results.items():
            x, y = np.asarray(data["x"]), np.asarray(data["loss"])
            fit = self._fitter.fit_power_law(x, y)
            predicted_loss = float(fit.model_fn(np.array([compute_budget]))[0])
            # Extrapolation distance
            x_max = float(x.max())
            extrap = np.log(compute_budget / x_max) / np.log(10) if compute_budget > x_max else 0.0
            rankings.append({
                "architecture": arch,
                "predicted_loss": predicted_loss,
                "r_squared": fit.r_squared,
                "extrapolation_decades": float(extrap),
                "fit_params": dict(zip(fit.param_names, fit.params.tolist())),
            })
        rankings.sort(key=lambda r: r["predicted_loss"])
        for i, r in enumerate(rankings):
            r["rank"] = i + 1
        return rankings

    def meta_scaling_law(
        self, arch_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fit a meta scaling law: how do scaling exponents depend on architecture parameters?

        arch_results: {arch_name: {'x': array, 'loss': array, 'arch_param': float}}
        where arch_param is some architecture characteristic (e.g., depth, width ratio).
        """
        arch_params = []
        exponents = []
        arch_names = []
        for arch, data in arch_results.items():
            x, y = np.asarray(data["x"]), np.asarray(data["loss"])
            slope, _, _, _ = self._exponent_computer._fit_log_linear(x, y)
            exponents.append(-slope)
            arch_params.append(data.get("arch_param", 0.0))
            arch_names.append(arch)

        arch_params = np.array(arch_params, dtype=np.float64)
        exponents = np.array(exponents, dtype=np.float64)

        if len(arch_params) < 3:
            return {
                "arch_params": arch_params.tolist(),
                "exponents": exponents.tolist(),
                "insufficient_data": True,
            }

        # Fit exponent = f(arch_param) as a linear model
        slope, intercept, r_value, p_value, std_err = stats.linregress(arch_params, exponents)

        # Also try log-linear
        valid = arch_params > 0
        if np.sum(valid) >= 3:
            log_slope, log_int, log_r, log_p, _ = stats.linregress(
                np.log(arch_params[valid]), exponents[valid]
            )
        else:
            log_slope, log_int, log_r, log_p = np.nan, np.nan, np.nan, np.nan

        return {
            "architectures": arch_names,
            "arch_params": arch_params.tolist(),
            "exponents": exponents.tolist(),
            "linear_fit": {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
            },
            "log_linear_fit": {
                "slope": float(log_slope),
                "intercept": float(log_int),
                "r_squared": float(log_r ** 2) if not np.isnan(log_r) else np.nan,
                "p_value": float(log_p) if not np.isnan(log_p) else np.nan,
            },
            "exponent_depends_on_architecture": p_value < 0.05,
        }
