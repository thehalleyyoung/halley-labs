"""Unit tests for usability_oracle.cognitive.calibration.ParameterCalibrator.

Tests cover least-squares parameter fitting, residual computation,
goodness-of-fit statistics, sensitivity analysis, k-fold cross-validation,
bootstrap confidence intervals, and parameter bounds enforcement.  Synthetic
Fitts' Law data is used as the primary test vehicle.

References
----------
Motulsky, H.J. & Christopoulos, A. (2004). *Fitting Models to Biological
    Data Using Linear and Nonlinear Regression*. Oxford University Press.
Efron, B. & Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*.
    Chapman & Hall.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pytest
from numpy.typing import NDArray

from usability_oracle.cognitive.calibration import ParameterCalibrator


# ------------------------------------------------------------------ #
# Helpers — synthetic Fitts' Law data
# ------------------------------------------------------------------ #

# True parameters for data generation
TRUE_A = 0.050
TRUE_B = 0.150

# Fixed distances and widths for reproducible test data
_DISTANCES = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 800], dtype=float)
_WIDTHS = np.array([20, 30, 40, 50, 60, 20, 30, 40, 50, 60], dtype=float)


def _fitts_predict(distances: NDArray, widths: NDArray, a: float, b: float) -> NDArray:
    """Shannon formulation: MT = a + b * log2(1 + D/W)."""
    return a + b * np.log2(1.0 + distances / widths)


def _make_synthetic_data(
    n: int = 10,
    noise_std: float = 0.005,
    seed: int = 42,
) -> tuple[NDArray, NDArray]:
    """Generate synthetic Fitts' Law observations with Gaussian noise.

    Returns (x_data, y_data) where x_data is a (n, 2) array of
    (distance, width) pairs and y_data is a 1-D array of noisy
    movement times.
    """
    rng = np.random.default_rng(seed)
    distances = _DISTANCES[:n]
    widths = _WIDTHS[:n]
    clean = _fitts_predict(distances, widths, TRUE_A, TRUE_B)
    noise = rng.normal(0, noise_std, size=n)
    return np.column_stack([distances, widths]), clean + noise


@pytest.fixture
def calibrator() -> ParameterCalibrator:
    """Create a fresh ParameterCalibrator instance."""
    return ParameterCalibrator()


@pytest.fixture
def synthetic_data() -> tuple[NDArray, NDArray]:
    """Provide standard synthetic Fitts' Law data (x, y)."""
    return _make_synthetic_data(n=10, noise_std=0.005)


# ------------------------------------------------------------------ #
# Residuals
# ------------------------------------------------------------------ #


class TestResiduals:
    """Tests for ParameterCalibrator.residuals()."""

    def test_perfect_fit_zero_residuals(self) -> None:
        """When params exactly match the data-generating process, residuals ≈ 0.

        Using noise-free data ensures residuals are machine-epsilon small.
        """
        distances = _DISTANCES[:5]
        widths = _WIDTHS[:5]
        observed = _fitts_predict(distances, widths, TRUE_A, TRUE_B)

        def model_fn(a: float, b: float) -> NDArray:
            return _fitts_predict(distances, widths, a, b)

        res = ParameterCalibrator.residuals(
            {"a": TRUE_A, "b": TRUE_B}, observed, model_fn
        )
        assert np.allclose(res, 0.0, atol=1e-12)

    def test_residuals_shape(self) -> None:
        """Residuals array should have the same shape as observed data."""
        n = 8
        distances = _DISTANCES[:n]
        widths = _WIDTHS[:n]
        observed = _fitts_predict(distances, widths, TRUE_A, TRUE_B)

        def model_fn(a: float, b: float) -> NDArray:
            return _fitts_predict(distances, widths, a, b)

        res = ParameterCalibrator.residuals(
            {"a": TRUE_A, "b": TRUE_B}, observed, model_fn
        )
        assert res.shape == (n,)

    def test_residual_sign(self) -> None:
        """Residuals = observed - predicted; over-prediction → negative."""
        distances = _DISTANCES[:3]
        widths = _WIDTHS[:3]
        observed = _fitts_predict(distances, widths, TRUE_A, TRUE_B)

        def model_fn(a: float, b: float) -> NDArray:
            return _fitts_predict(distances, widths, a, b)

        # Use a larger 'a' → predictions too high → residuals negative
        res = ParameterCalibrator.residuals(
            {"a": TRUE_A + 0.1, "b": TRUE_B}, observed, model_fn
        )
        assert np.all(res < 0)


# ------------------------------------------------------------------ #
# Calibrate (least-squares fitting)
# ------------------------------------------------------------------ #


class TestCalibrate:
    """Tests for ParameterCalibrator.calibrate()."""

    def test_recovers_true_params(
        self, calibrator: ParameterCalibrator, synthetic_data: tuple
    ) -> None:
        """Calibration should recover the true a and b from noisy data.

        With low noise (σ=0.005) and 10 data points, the fitted params
        should be within ~10% of the true values.
        """
        x_data, y_data = synthetic_data
        distances, widths = x_data[:, 0], x_data[:, 1]

        def model_fn(a: float, b: float) -> NDArray:
            return _fitts_predict(distances, widths, a, b)

        fitted = calibrator.calibrate(
            y_data, model_fn,
            initial_params={"a": 0.01, "b": 0.10},
            param_bounds={"a": (0.0, 1.0), "b": (0.0, 1.0)},
        )
        assert fitted["a"] == pytest.approx(TRUE_A, abs=0.02)
        assert fitted["b"] == pytest.approx(TRUE_B, abs=0.02)

    def test_returns_dict(
        self, calibrator: ParameterCalibrator, synthetic_data: tuple
    ) -> None:
        """calibrate() should return a dict with the same keys as initial_params."""
        x_data, y_data = synthetic_data
        distances, widths = x_data[:, 0], x_data[:, 1]

        def model_fn(a: float, b: float) -> NDArray:
            return _fitts_predict(distances, widths, a, b)

        result = calibrator.calibrate(
            y_data, model_fn, {"a": 0.01, "b": 0.10}
        )
        assert "a" in result and "b" in result

    def test_bounds_enforced(self, calibrator: ParameterCalibrator) -> None:
        """Fitted parameters should respect the supplied bounds.

        With tight bounds [0.04, 0.06] for 'a', the result must be within.
        """
        x_data, y_data = _make_synthetic_data(n=10, noise_std=0.001)
        distances, widths = x_data[:, 0], x_data[:, 1]

        def model_fn(a: float, b: float) -> NDArray:
            return _fitts_predict(distances, widths, a, b)

        fitted = calibrator.calibrate(
            y_data, model_fn,
            initial_params={"a": 0.05, "b": 0.15},
            param_bounds={"a": (0.04, 0.06), "b": (0.10, 0.20)},
        )
        assert 0.04 <= fitted["a"] <= 0.06
        assert 0.10 <= fitted["b"] <= 0.20


# ------------------------------------------------------------------ #
# Goodness of fit
# ------------------------------------------------------------------ #


class TestGoodnessOfFit:
    """Tests for ParameterCalibrator.goodness_of_fit()."""

    def test_perfect_fit(self) -> None:
        """Perfect predictions → R²=1.0, RMSE=0, MAE=0."""
        obs = [1.0, 2.0, 3.0, 4.0]
        gof = ParameterCalibrator.goodness_of_fit(obs, obs, n_params=0)
        assert gof["r_squared"] == pytest.approx(1.0)
        assert gof["rmse"] == pytest.approx(0.0, abs=1e-12)
        assert gof["mae"] == pytest.approx(0.0, abs=1e-12)

    def test_returns_expected_keys(self) -> None:
        """Result dict should contain r_squared, rmse, mae, aic, bic."""
        gof = ParameterCalibrator.goodness_of_fit(
            [1, 2, 3], [1.1, 2.1, 2.9], n_params=2
        )
        for key in ["r_squared", "rmse", "mae", "aic", "bic"]:
            assert key in gof, f"Missing key: {key}"

    def test_r_squared_range(self) -> None:
        """R² should be <= 1.0 for reasonable predictions."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = obs + np.array([0.1, -0.1, 0.05, -0.05, 0.0])
        gof = ParameterCalibrator.goodness_of_fit(obs, pred)
        assert gof["r_squared"] <= 1.0

    def test_rmse_positive_for_imperfect(self) -> None:
        """RMSE should be positive when predictions don't match perfectly."""
        gof = ParameterCalibrator.goodness_of_fit(
            [1, 2, 3], [1.5, 2.5, 3.5]
        )
        assert gof["rmse"] > 0.0

    def test_shape_mismatch_raises(self) -> None:
        """Different-length observed and predicted should raise ValueError."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            ParameterCalibrator.goodness_of_fit([1, 2], [1, 2, 3])


# ------------------------------------------------------------------ #
# Sensitivity analysis
# ------------------------------------------------------------------ #


class TestSensitivityAnalysis:
    """Tests for ParameterCalibrator.sensitivity_analysis()."""

    def test_returns_all_params(self) -> None:
        """Result should contain an entry for each parameter in param_ranges."""
        def model_fn(a: float, b: float) -> float:
            return a + b * math.log2(1 + 200.0 / 40.0)

        result = ParameterCalibrator.sensitivity_analysis(
            model_fn,
            base_params={"a": 0.05, "b": 0.15},
            param_ranges={"a": (0.01, 0.10), "b": (0.05, 0.30)},
            n_steps=10,
        )
        assert "a" in result and "b" in result

    def test_sensitivity_index_positive(self) -> None:
        """Sensitivity index should be positive for non-trivial models."""
        def model_fn(a: float, b: float) -> float:
            return a + b * 3.0

        result = ParameterCalibrator.sensitivity_analysis(
            model_fn,
            base_params={"a": 0.05, "b": 0.15},
            param_ranges={"a": (0.01, 0.10), "b": (0.05, 0.30)},
        )
        for param_result in result.values():
            assert param_result["sensitivity_index"] > 0.0

    def test_contains_expected_keys(self) -> None:
        """Each parameter's result should have values, outputs, sensitivity_index, elasticity."""
        def model_fn(a: float) -> float:
            return a * 10.0

        result = ParameterCalibrator.sensitivity_analysis(
            model_fn,
            base_params={"a": 1.0},
            param_ranges={"a": (0.5, 2.0)},
        )
        for key in ["values", "outputs", "sensitivity_index", "elasticity"]:
            assert key in result["a"], f"Missing key: {key}"

    def test_n_steps_minimum(self) -> None:
        """n_steps < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="n_steps must be >= 2"):
            ParameterCalibrator.sensitivity_analysis(
                lambda a: a, {"a": 1.0}, {"a": (0, 2)}, n_steps=1
            )


# ------------------------------------------------------------------ #
# Cross-validation
# ------------------------------------------------------------------ #


class TestCrossValidate:
    """Tests for ParameterCalibrator.cross_validate()."""

    def test_cv_returns_expected_keys(self, calibrator: ParameterCalibrator) -> None:
        """Result should contain mean/std for R² and RMSE, plus fold_results."""
        x_data, y_data = _make_synthetic_data(n=10, noise_std=0.005)
        distances, widths = x_data[:, 0], x_data[:, 1]

        def model_fn(x: NDArray, a: float, b: float) -> NDArray:
            return _fitts_predict(x[:, 0], x[:, 1], a, b)

        result = calibrator.cross_validate(
            x_data, y_data, model_fn,
            initial_params={"a": 0.05, "b": 0.15},
            k_folds=3,
            param_bounds={"a": (0.0, 1.0), "b": (0.0, 1.0)},
        )
        for key in ["r_squared_mean", "r_squared_std", "rmse_mean",
                     "rmse_std", "fold_results"]:
            assert key in result

    def test_cv_r_squared_reasonable(self, calibrator: ParameterCalibrator) -> None:
        """Mean R² from CV should be close to 1.0 for low-noise Fitts data."""
        x_data, y_data = _make_synthetic_data(n=10, noise_std=0.002)

        def model_fn(x: NDArray, a: float, b: float) -> NDArray:
            return _fitts_predict(x[:, 0], x[:, 1], a, b)

        result = calibrator.cross_validate(
            x_data, y_data, model_fn,
            initial_params={"a": 0.05, "b": 0.15},
            k_folds=3,
            param_bounds={"a": (0.0, 1.0), "b": (0.0, 1.0)},
        )
        assert result["r_squared_mean"] > 0.8

    def test_cv_too_few_folds_raises(self, calibrator: ParameterCalibrator) -> None:
        """k_folds < 2 should raise ValueError."""
        x, y = _make_synthetic_data(n=10)

        def model_fn(x: NDArray, a: float, b: float) -> NDArray:
            return _fitts_predict(x[:, 0], x[:, 1], a, b)

        with pytest.raises(ValueError, match="k_folds must be >= 2"):
            calibrator.cross_validate(x, y, model_fn, {"a": 0.05, "b": 0.15}, k_folds=1)

    def test_cv_fold_count(self, calibrator: ParameterCalibrator) -> None:
        """Number of fold results should equal k_folds."""
        x, y = _make_synthetic_data(n=10, noise_std=0.005)

        def model_fn(x: NDArray, a: float, b: float) -> NDArray:
            return _fitts_predict(x[:, 0], x[:, 1], a, b)

        result = calibrator.cross_validate(
            x, y, model_fn,
            initial_params={"a": 0.05, "b": 0.15},
            k_folds=5,
            param_bounds={"a": (0.0, 1.0), "b": (0.0, 1.0)},
        )
        assert len(result["fold_results"]) == 5


# ------------------------------------------------------------------ #
# Bootstrap confidence intervals
# ------------------------------------------------------------------ #


class TestBootstrapConfidence:
    """Tests for ParameterCalibrator.bootstrap_confidence()."""

    def test_returns_ci_for_all_params(self, calibrator: ParameterCalibrator) -> None:
        """Result should have confidence interval info for every parameter."""
        x_data, y_data = _make_synthetic_data(n=10, noise_std=0.005)
        distances, widths = x_data[:, 0], x_data[:, 1]

        def model_fn(a: float, b: float) -> NDArray:
            return _fitts_predict(distances, widths, a, b)

        result = calibrator.bootstrap_confidence(
            y_data, model_fn,
            initial_params={"a": 0.05, "b": 0.15},
            n_bootstrap=50,
            param_bounds={"a": (0.0, 1.0), "b": (0.0, 1.0)},
        )
        for param in ["a", "b"]:
            assert param in result
            for key in ["mean", "std", "ci_lower", "ci_upper"]:
                assert key in result[param]

    def test_ci_has_nonzero_width(self, calibrator: ParameterCalibrator) -> None:
        """Bootstrap CI should have non-zero width for each parameter.

        With noisy resampling, the fitted parameters should vary across
        bootstrap iterations, producing a CI with positive width.
        """
        x_data, y_data = _make_synthetic_data(n=10, noise_std=0.01)
        distances, widths = x_data[:, 0], x_data[:, 1]

        def model_fn(a: float, b: float) -> NDArray:
            return _fitts_predict(distances, widths, a, b)

        result = calibrator.bootstrap_confidence(
            y_data, model_fn,
            initial_params={"a": 0.05, "b": 0.15},
            n_bootstrap=50,
            confidence_level=0.95,
            param_bounds={"a": (0.0, 1.0), "b": (0.0, 1.0)},
        )
        for param in ["a", "b"]:
            ci_width = result[param]["ci_upper"] - result[param]["ci_lower"]
            assert ci_width > 0.0, f"CI for {param} should have positive width"
            assert result[param]["std"] > 0.0, f"Std for {param} should be positive"

    def test_too_few_bootstrap_raises(self, calibrator: ParameterCalibrator) -> None:
        """n_bootstrap < 10 should raise ValueError."""
        _, y = _make_synthetic_data(n=5)

        def model_fn(a: float, b: float) -> NDArray:
            return np.ones(5) * a

        with pytest.raises(ValueError, match="n_bootstrap must be >= 10"):
            calibrator.bootstrap_confidence(
                y, model_fn, {"a": 0.05, "b": 0.15}, n_bootstrap=5
            )

    def test_ci_lower_less_than_upper(self, calibrator: ParameterCalibrator) -> None:
        """CI lower bound should be ≤ upper bound for each parameter."""
        x_data, y_data = _make_synthetic_data(n=10, noise_std=0.005)
        distances, widths = x_data[:, 0], x_data[:, 1]

        def model_fn(a: float, b: float) -> NDArray:
            return _fitts_predict(distances, widths, a, b)

        result = calibrator.bootstrap_confidence(
            y_data, model_fn,
            initial_params={"a": 0.05, "b": 0.15},
            n_bootstrap=50,
            param_bounds={"a": (0.0, 1.0), "b": (0.0, 1.0)},
        )
        for param in ["a", "b"]:
            assert result[param]["ci_lower"] <= result[param]["ci_upper"]


# ------------------------------------------------------------------ #
# Synthetic Fitts' Law integration
# ------------------------------------------------------------------ #


class TestFittsLawIntegration:
    """End-to-end test using synthetic Fitts' Law data.

    This class tests the full calibration pipeline: fit parameters,
    compute goodness of fit, and verify accuracy on the known model.
    """

    def test_full_pipeline(self, calibrator: ParameterCalibrator) -> None:
        """Fit → predict → goodness-of-fit should yield R² ≈ 1.0.

        Generate noise-free data, fit, predict, and check diagnostics.
        """
        distances = _DISTANCES
        widths = _WIDTHS
        observed = _fitts_predict(distances, widths, TRUE_A, TRUE_B)

        def model_fn(a: float, b: float) -> NDArray:
            return _fitts_predict(distances, widths, a, b)

        fitted = calibrator.calibrate(
            observed, model_fn,
            initial_params={"a": 0.01, "b": 0.10},
            param_bounds={"a": (0.0, 0.5), "b": (0.0, 0.5)},
        )

        predicted = model_fn(**fitted)
        gof = ParameterCalibrator.goodness_of_fit(observed, predicted, n_params=2)
        assert gof["r_squared"] > 0.999
        assert gof["rmse"] < 0.001
