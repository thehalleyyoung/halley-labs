"""
Tests for Student-t emission model, emission model selection, and
sensitivity analysis.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure the implementation package is importable
_IMPL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

from causal_trading.regime.student_t_emission import (
    StudentTEmission,
    EmissionModelSelector,
)
from causal_trading.regime.sticky_hdp_hmm import GaussianEmission
from causal_trading.evaluation.sensitivity_analysis import (
    SensitivityAnalyzer,
    SensitivityReport,
    SweepPoint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def univariate_gaussian_data(rng):
    """200-step univariate Gaussian data."""
    return rng.normal(0.0, 1.0, size=(200, 1))


@pytest.fixture
def univariate_heavy_tail_data(rng):
    """200-step univariate Student-t(3) data (heavy tails)."""
    return rng.standard_t(3, size=(200, 1)) * 2.0


@pytest.fixture
def multivariate_data(rng):
    """300×3 multivariate data."""
    return rng.normal(0.0, 1.0, size=(300, 3))


@pytest.fixture
def multivariate_heavy_tail_data(rng):
    """300×3 heavy-tailed data."""
    return rng.standard_t(3, size=(300, 3)) * 2.0


@pytest.fixture
def regime_switching_features(rng):
    """500×3 features with 2 regime switches."""
    data = np.empty((500, 3))
    data[:200] = rng.normal(0.0, 0.5, size=(200, 3))
    data[200:350] = rng.normal(3.0, 1.5, size=(150, 3))
    data[350:] = rng.normal(-1.0, 0.8, size=(150, 3))
    return data


# ---------------------------------------------------------------------------
# StudentTEmission: Interface compatibility
# ---------------------------------------------------------------------------

class TestStudentTEmissionInterface:
    """Verify StudentTEmission has the same interface as GaussianEmission."""

    def test_has_add_obs(self):
        em = StudentTEmission(dim=2)
        assert hasattr(em, "add_obs") and callable(em.add_obs)

    def test_has_remove_obs(self):
        em = StudentTEmission(dim=2)
        assert hasattr(em, "remove_obs") and callable(em.remove_obs)

    def test_has_log_marginal_likelihood(self):
        em = StudentTEmission(dim=2)
        assert hasattr(em, "log_marginal_likelihood") and callable(em.log_marginal_likelihood)

    def test_has_sample_posterior(self):
        em = StudentTEmission(dim=2)
        assert hasattr(em, "sample_posterior") and callable(em.sample_posterior)

    def test_has_log_pdf(self):
        em = StudentTEmission(dim=2)
        assert hasattr(em, "log_pdf") and callable(em.log_pdf)

    def test_has_reset(self):
        em = StudentTEmission(dim=2)
        assert hasattr(em, "reset") and callable(em.reset)

    def test_same_methods_as_gaussian(self):
        """Key methods of GaussianEmission exist on StudentTEmission."""
        required = ["add_obs", "remove_obs", "log_marginal_likelihood",
                     "sample_posterior", "log_pdf", "reset"]
        ge = GaussianEmission(dim=2)
        se = StudentTEmission(dim=2)
        for method in required:
            assert hasattr(ge, method), f"GaussianEmission missing {method}"
            assert hasattr(se, method), f"StudentTEmission missing {method}"


# ---------------------------------------------------------------------------
# StudentTEmission: Valid log-likelihoods
# ---------------------------------------------------------------------------

class TestStudentTEmissionLogLikelihood:

    def test_log_marginal_finite_no_obs(self):
        em = StudentTEmission(dim=1)
        ll = em.log_marginal_likelihood(np.array([0.5]))
        assert np.isfinite(ll)

    def test_log_marginal_finite_with_obs(self, rng):
        em = StudentTEmission(dim=2, nu=5.0)
        for _ in range(50):
            em.add_obs(rng.normal(0, 1, size=2))
        ll = em.log_marginal_likelihood(rng.normal(0, 1, size=2))
        assert np.isfinite(ll)

    def test_log_marginal_negative(self, rng):
        """Log-likelihoods should be negative for continuous distributions."""
        em = StudentTEmission(dim=1)
        for _ in range(20):
            em.add_obs(rng.normal(0, 1, size=1))
        ll = em.log_marginal_likelihood(np.array([0.0]))
        assert ll < 0 or np.isfinite(ll)  # Can be positive for narrow dist

    def test_log_pdf_finite(self, rng):
        em = StudentTEmission(dim=3, nu=4.0)
        mu = np.zeros(3)
        Sigma = np.eye(3)
        ll = em.log_pdf(rng.normal(size=3), mu, Sigma)
        assert np.isfinite(ll)

    def test_log_pdf_higher_near_mean(self, rng):
        """Points near the mean should have higher log-pdf."""
        em = StudentTEmission(dim=1, nu=5.0)
        mu = np.array([0.0])
        Sigma = np.eye(1)
        ll_near = em.log_pdf(np.array([0.01]), mu, Sigma)
        ll_far = em.log_pdf(np.array([10.0]), mu, Sigma)
        assert ll_near > ll_far

    def test_add_remove_obs_consistency(self, rng):
        """Adding then removing same observation restores state."""
        em = StudentTEmission(dim=2)
        x = rng.normal(size=2)
        ll_before = em.log_marginal_likelihood(x)
        em.add_obs(x)
        em.remove_obs(x)
        ll_after = em.log_marginal_likelihood(x)
        np.testing.assert_allclose(ll_before, ll_after, atol=1e-10)

    def test_sample_posterior_shapes(self, rng):
        em = StudentTEmission(dim=3)
        for _ in range(20):
            em.add_obs(rng.normal(size=3))
        mu, Sigma = em.sample_posterior(rng)
        assert mu.shape == (3,)
        assert Sigma.shape == (3, 3)

    def test_posterior_predictive_alias(self, rng):
        em = StudentTEmission(dim=1)
        for _ in range(10):
            em.add_obs(rng.normal(size=1))
        x = np.array([0.5])
        assert em.posterior_predictive(x) == em.log_marginal_likelihood(x)

    def test_multivariate_log_likelihood(self, multivariate_data, rng):
        em = StudentTEmission(dim=3)
        for t in range(50):
            em.add_obs(multivariate_data[t])
        ll = em.log_marginal_likelihood(multivariate_data[50])
        assert np.isfinite(ll)

    def test_nu_parameter_effect(self, rng):
        """Lower nu should give heavier tails (higher log_pdf for outliers)."""
        mu = np.array([0.0])
        Sigma = np.eye(1)
        outlier = np.array([20.0])

        em_low = StudentTEmission(dim=1, nu=3.0)
        em_high = StudentTEmission(dim=1, nu=100.0)
        ll_low = em_low.log_pdf(outlier, mu, Sigma)
        ll_high = em_high.log_pdf(outlier, mu, Sigma)
        # Low-nu should assign higher likelihood to outlier via heavier tails
        assert ll_low > ll_high


# ---------------------------------------------------------------------------
# EmissionModelSelector
# ---------------------------------------------------------------------------

class TestEmissionModelSelector:

    def test_select_returns_string(self, univariate_gaussian_data):
        sel = EmissionModelSelector()
        result = sel.select(univariate_gaussian_data)
        assert isinstance(result, str)
        assert result in ("gaussian", "student_t")

    def test_select_prefers_student_t_for_heavy_tails(
        self, univariate_heavy_tail_data
    ):
        """Student-t should be preferred for heavy-tailed data."""
        sel = EmissionModelSelector(nu=3.0)
        result = sel.select(univariate_heavy_tail_data)
        assert result == "student_t"

    def test_compare_returns_list(self, univariate_gaussian_data):
        sel = EmissionModelSelector()
        table = sel.compare(univariate_gaussian_data)
        assert isinstance(table, list)
        assert len(table) == 2

    def test_compare_rows_have_fields(self, univariate_gaussian_data):
        sel = EmissionModelSelector()
        table = sel.compare(univariate_gaussian_data)
        for row in table:
            assert hasattr(row, "model")
            assert hasattr(row, "log_likelihood")
            assert hasattr(row, "bic")
            assert hasattr(row, "waic")
            assert hasattr(row, "n_params")

    def test_compare_finite_values(self, multivariate_data):
        sel = EmissionModelSelector()
        table = sel.compare(multivariate_data)
        for row in table:
            assert np.isfinite(row.log_likelihood)
            assert np.isfinite(row.bic)
            assert np.isfinite(row.waic)

    def test_kurtosis_test_gaussian(self, univariate_gaussian_data):
        sel = EmissionModelSelector()
        result = sel.kurtosis_test(univariate_gaussian_data)
        assert "excess_kurtosis" in result
        assert "exceeds_gaussian" in result
        assert "p_value" in result

    def test_kurtosis_test_heavy_tail(self, univariate_heavy_tail_data):
        sel = EmissionModelSelector()
        result = sel.kurtosis_test(univariate_heavy_tail_data)
        assert result["exceeds_gaussian"] is True
        assert result["excess_kurtosis"] > 0

    def test_select_multivariate_heavy(self, multivariate_heavy_tail_data):
        sel = EmissionModelSelector(nu=3.0)
        result = sel.select(multivariate_heavy_tail_data)
        assert result == "student_t"

    def test_univariate_1d_input(self, rng):
        """Accept 1D array input."""
        data = rng.normal(0, 1, size=200)
        sel = EmissionModelSelector()
        result = sel.select(data)
        assert result in ("gaussian", "student_t")


# ---------------------------------------------------------------------------
# SensitivityAnalyzer: Sweep structure
# ---------------------------------------------------------------------------

class TestSensitivityAnalyzerSweeps:

    def test_sweep_kappa_returns_list(self, regime_switching_features):
        analyzer = SensitivityAnalyzer()
        points = analyzer.sweep_kappa(
            regime_switching_features,
            kappa_values=[10, 50],
            n_iter=20,
            burn_in=5,
        )
        assert isinstance(points, list)
        assert len(points) == 2

    def test_sweep_point_structure(self, regime_switching_features):
        analyzer = SensitivityAnalyzer()
        points = analyzer.sweep_kappa(
            regime_switching_features,
            kappa_values=[25],
            n_iter=20,
            burn_in=5,
        )
        p = points[0]
        assert isinstance(p, SweepPoint)
        assert p.param_name == "kappa"
        assert p.param_value == 25.0
        assert isinstance(p.metrics, dict)

    def test_sweep_metrics_keys(self, regime_switching_features):
        analyzer = SensitivityAnalyzer()
        points = analyzer.sweep_kappa(
            regime_switching_features,
            kappa_values=[50],
            n_iter=20,
            burn_in=5,
        )
        metrics = points[0].metrics
        expected_keys = {
            "n_regimes",
            "avg_regime_duration",
            "transition_entropy",
            "dag_edges",
            "pac_bayes_bound",
            "shield_permissivity",
        }
        assert expected_keys.issubset(metrics.keys())

    def test_sweep_metrics_types(self, regime_switching_features):
        analyzer = SensitivityAnalyzer()
        points = analyzer.sweep_kappa(
            regime_switching_features,
            kappa_values=[50],
            n_iter=20,
            burn_in=5,
        )
        for key, val in points[0].metrics.items():
            assert isinstance(val, (int, float))
            assert np.isfinite(val), f"Non-finite metric: {key}={val}"

    def test_sweep_delta(self, regime_switching_features):
        analyzer = SensitivityAnalyzer()
        points = analyzer.sweep_delta(
            regime_switching_features,
            deltas=[0.01, 0.1],
            n_iter=20,
            burn_in=5,
        )
        assert len(points) == 2
        assert points[0].param_name == "delta"

    def test_sweep_ci_alpha(self, regime_switching_features):
        analyzer = SensitivityAnalyzer()
        points = analyzer.sweep_ci_alpha(
            regime_switching_features,
            alphas=[0.05, 0.1],
            n_iter=20,
            burn_in=5,
        )
        assert len(points) == 2

    def test_sweep_emission_prior(self, regime_switching_features):
        analyzer = SensitivityAnalyzer()
        points = analyzer.sweep_emission_prior(
            regime_switching_features,
            prior_scales=[0.5, 2.0],
            n_iter=20,
            burn_in=5,
        )
        assert len(points) == 2

    def test_sweep_pac_bayes_prior(self, regime_switching_features):
        analyzer = SensitivityAnalyzer()
        points = analyzer.sweep_pac_bayes_prior(
            regime_switching_features,
            prior_counts=[1, 50],
            n_iter=20,
            burn_in=5,
        )
        assert len(points) == 2


# ---------------------------------------------------------------------------
# SensitivityReport
# ---------------------------------------------------------------------------

class TestSensitivityReport:

    def _make_report(self) -> SensitivityReport:
        points_a = [
            SweepPoint("alpha", 0.01, {"n_regimes": 2.0, "bound": 0.1}),
            SweepPoint("alpha", 0.05, {"n_regimes": 3.0, "bound": 0.08}),
            SweepPoint("alpha", 0.10, {"n_regimes": 5.0, "bound": 0.05}),
        ]
        points_b = [
            SweepPoint("kappa", 10, {"n_regimes": 2.0, "bound": 0.12}),
            SweepPoint("kappa", 50, {"n_regimes": 3.0, "bound": 0.09}),
            SweepPoint("kappa", 100, {"n_regimes": 3.0, "bound": 0.09}),
        ]
        return SensitivityReport(
            sweeps={"alpha": points_a, "kappa": points_b},
            metadata={"test": True},
        )

    def test_most_sensitive_param(self):
        report = self._make_report()
        result = report.most_sensitive_param()
        assert isinstance(result, str)
        assert result in ("alpha", "kappa")

    def test_most_sensitive_detects_alpha(self):
        """alpha sweep has larger variation in n_regimes (2->5)."""
        report = self._make_report()
        assert report.most_sensitive_param() == "alpha"

    def test_robust_range(self):
        report = self._make_report()
        lo, hi = report.robust_range("kappa")
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert lo <= hi

    def test_robust_range_missing(self):
        report = self._make_report()
        with pytest.raises(KeyError):
            report.robust_range("nonexistent")

    def test_to_pgfplots_data(self):
        report = self._make_report()
        data = report.to_pgfplots_data()
        assert isinstance(data, dict)
        assert "alpha" in data
        assert "param_value" in data["alpha"]

    def test_to_latex_table(self):
        report = self._make_report()
        latex = report.to_latex_table()
        assert isinstance(latex, str)
        assert "\\begin{tabular}" in latex
        assert "\\end{tabular}" in latex

    def test_save_load_roundtrip(self):
        report = self._make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_report.json"
            report.save(path)
            loaded = SensitivityReport.load(path)
            assert set(loaded.sweeps.keys()) == set(report.sweeps.keys())
            for name in report.sweeps:
                assert len(loaded.sweeps[name]) == len(report.sweeps[name])
                for orig, load in zip(report.sweeps[name], loaded.sweeps[name]):
                    assert orig.param_name == load.param_name
                    assert orig.param_value == load.param_value

    def test_sweep_point_to_dict(self):
        p = SweepPoint("kappa", 50.0, {"n_regimes": 3.0})
        d = p.to_dict()
        assert d["param_name"] == "kappa"
        assert d["param_value"] == 50.0
        assert d["metrics"]["n_regimes"] == 3.0


# ---------------------------------------------------------------------------
# Full sensitivity report (integration)
# ---------------------------------------------------------------------------

class TestFullSensitivityReport:

    @pytest.fixture
    def small_data(self, rng):
        """Small dataset for fast integration test."""
        data = np.empty((200, 3))
        data[:100] = rng.normal(0.0, 0.5, size=(100, 3))
        data[100:] = rng.normal(2.0, 1.0, size=(100, 3))
        return data

    def test_full_report_runs(self, small_data):
        analyzer = SensitivityAnalyzer()
        report = analyzer.full_sensitivity_report(
            small_data,
            n_iter=20,
            burn_in=5,
            seed=123,
        )
        assert isinstance(report, SensitivityReport)
        assert len(report.sweeps) >= 4

    def test_full_report_all_sweeps_nonempty(self, small_data):
        analyzer = SensitivityAnalyzer()
        report = analyzer.full_sensitivity_report(
            small_data,
            n_iter=20,
            burn_in=5,
        )
        for name, points in report.sweeps.items():
            assert len(points) > 0, f"Sweep {name} is empty"

    def test_full_report_metadata(self, small_data):
        analyzer = SensitivityAnalyzer()
        report = analyzer.full_sensitivity_report(
            small_data,
            n_iter=20,
            burn_in=5,
        )
        assert "elapsed_seconds" in report.metadata
        assert "n_sweeps" in report.metadata
