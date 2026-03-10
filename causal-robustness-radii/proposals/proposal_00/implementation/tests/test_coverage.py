"""
Tests for coverage verification.

Covers Monte Carlo coverage simulation, calibration assessment,
and power analysis for the CausalCert pipeline.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from tests.conftest import _adj, random_dag


# ---------------------------------------------------------------------------
# Coverage simulation infrastructure
# ---------------------------------------------------------------------------


def _is_dag(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int).copy()
    queue = deque(i for i in range(n) if in_deg[i] == 0)
    count = 0
    while queue:
        v = queue.popleft()
        count += 1
        for c in range(n):
            if adj[v, c]:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
    return count == n


def _topo_sort(adj: np.ndarray) -> list[int]:
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int).copy()
    queue = [v for v in range(n) if in_deg[v] == 0]
    order: list[int] = []
    while queue:
        v = queue.pop(0)
        order.append(v)
        for c in range(n):
            if adj[v, c]:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
    return order


def generate_linear_gaussian_data(
    adj: np.ndarray,
    n: int,
    seed: int = 42,
    noise_std: float = 1.0,
    weight_seed: int | None = None,
) -> pd.DataFrame:
    """Generate data from a linear Gaussian SCM.

    Parameters
    ----------
    weight_seed : int | None
        If provided, use this seed for generating edge weights
        (so weights stay fixed across simulations).
        If None, derives weights from ``seed`` (legacy behavior).
    """
    weight_rng = np.random.default_rng(weight_seed if weight_seed is not None else seed)
    noise_rng = np.random.default_rng(seed)
    p = adj.shape[0]
    topo = _topo_sort(adj)
    weights = weight_rng.uniform(0.5, 1.5, size=(p, p)) * adj
    data = np.zeros((n, p))
    for v in topo:
        pa = np.where(adj[:, v] == 1)[0]
        mean = data[:, pa] @ weights[pa, v] if len(pa) else 0.0
        data[:, v] = mean + noise_std * noise_rng.standard_normal(n)
    return pd.DataFrame(data, columns=[f"X{i}" for i in range(p)])


def simple_ols_ate(
    data: pd.DataFrame,
    treatment: int,
    outcome: int,
    adjustment: list[int] | None = None,
) -> tuple[float, float]:
    """Compute OLS ATE estimate and standard error."""
    T = data.iloc[:, treatment].values
    Y = data.iloc[:, outcome].values
    X_cols = [T]
    if adjustment:
        for a in adjustment:
            X_cols.append(data.iloc[:, a].values)
    X = np.column_stack([np.ones(len(T))] + X_cols)
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    ate = beta[1]
    residuals = Y - X @ beta
    n = len(Y)
    p = X.shape[1]
    sigma2 = np.sum(residuals ** 2) / max(n - p, 1)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
    return float(ate), se


def ols_confidence_interval(
    data: pd.DataFrame,
    treatment: int,
    outcome: int,
    adjustment: list[int] | None = None,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Compute ATE and confidence interval. Returns (ate, ci_lo, ci_hi)."""
    ate, se = simple_ols_ate(data, treatment, outcome, adjustment)
    z = sp_stats.norm.ppf(1 - alpha / 2)
    return ate, ate - z * se, ate + z * se


# ---------------------------------------------------------------------------
# Monte Carlo coverage simulation
# ---------------------------------------------------------------------------


def monte_carlo_coverage(
    adj: np.ndarray,
    treatment: int,
    outcome: int,
    n_obs: int = 200,
    n_simulations: int = 500,
    alpha: float = 0.05,
    true_ate: float | None = None,
    adjustment: list[int] | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Estimate coverage of ATE confidence intervals via Monte Carlo.

    Parameters
    ----------
    adj : np.ndarray
        DAG adjacency matrix.
    treatment, outcome : int
        Treatment and outcome node indices.
    n_obs : int
        Number of observations per simulation.
    n_simulations : int
        Number of Monte Carlo replications.
    alpha : float
        Nominal significance level.
    true_ate : float | None
        True ATE if known. If None, estimated from a very large sample.
    adjustment : list[int] | None
        Adjustment set for ATE estimation.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, float]
        Dictionary with 'coverage', 'mean_width', 'mean_ate', 'se_coverage'.
    """
    rng_base = np.random.default_rng(seed)
    weight_seed = int(rng_base.integers(0, 2**31))

    # Estimate true ATE from large sample if not provided
    if true_ate is None:
        large_data = generate_linear_gaussian_data(
            adj, 50000, seed=seed + 999999, weight_seed=weight_seed,
        )
        true_ate, _ = simple_ols_ate(large_data, treatment, outcome, adjustment)

    covered = 0
    widths: list[float] = []
    ates: list[float] = []

    for sim in range(n_simulations):
        data = generate_linear_gaussian_data(
            adj, n_obs, seed=seed + sim, weight_seed=weight_seed,
        )
        ate, ci_lo, ci_hi = ols_confidence_interval(
            data, treatment, outcome, adjustment, alpha
        )
        ates.append(ate)
        widths.append(ci_hi - ci_lo)
        if ci_lo <= true_ate <= ci_hi:
            covered += 1

    cov_rate = covered / n_simulations
    se_cov = math.sqrt(cov_rate * (1 - cov_rate) / n_simulations)

    return {
        "coverage": cov_rate,
        "mean_width": float(np.mean(widths)),
        "mean_ate": float(np.mean(ates)),
        "std_ate": float(np.std(ates)),
        "se_coverage": se_cov,
        "true_ate": true_ate,
        "n_simulations": n_simulations,
    }


# ---------------------------------------------------------------------------
# Calibration assessment
# ---------------------------------------------------------------------------


def calibration_check(
    p_values: np.ndarray,
    expected_levels: np.ndarray | None = None,
) -> dict[str, float]:
    """Assess calibration of p-values.

    Under the null hypothesis, p-values should be uniformly distributed.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values.
    expected_levels : np.ndarray | None
        Significance levels to check.

    Returns
    -------
    dict[str, float]
        Dictionary with 'ks_statistic', 'ks_pvalue', 'mean_pvalue',
        and rejection rates at each level.
    """
    if expected_levels is None:
        expected_levels = np.array([0.01, 0.05, 0.10, 0.20])

    # KS test for uniformity
    ks_stat, ks_p = sp_stats.kstest(p_values, "uniform")

    result: dict[str, float] = {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "mean_pvalue": float(np.mean(p_values)),
        "median_pvalue": float(np.median(p_values)),
    }

    for level in expected_levels:
        rej_rate = float(np.mean(p_values <= level))
        result[f"rejection_rate_{level:.2f}"] = rej_rate

    return result


def generate_null_p_values(
    adj: np.ndarray,
    x: int,
    y: int,
    conditioning: list[int],
    n_obs: int = 200,
    n_simulations: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """Generate p-values under the null (when X⊥Y|Z in the DAG).

    Uses partial correlation test.
    """
    rng = np.random.default_rng(seed)
    p_values: list[float] = []

    for sim in range(n_simulations):
        data = generate_linear_gaussian_data(adj, n_obs, seed=seed + sim)

        X_vals = data.iloc[:, x].values
        Y_vals = data.iloc[:, y].values

        if conditioning:
            Z = np.column_stack([data.iloc[:, c].values for c in conditioning])
            Z_aug = np.column_stack([np.ones(n_obs), Z])
            # Residualize X and Y on Z
            beta_x = np.linalg.lstsq(Z_aug, X_vals, rcond=None)[0]
            beta_y = np.linalg.lstsq(Z_aug, Y_vals, rcond=None)[0]
            X_resid = X_vals - Z_aug @ beta_x
            Y_resid = Y_vals - Z_aug @ beta_y
        else:
            X_resid = X_vals
            Y_resid = Y_vals

        r = np.corrcoef(X_resid, Y_resid)[0, 1]
        dof = n_obs - len(conditioning) - 2
        if dof <= 0:
            p_values.append(1.0)
            continue
        t_stat = r * math.sqrt(dof / (1 - r ** 2 + 1e-12))
        p_val = 2 * (1 - sp_stats.t.cdf(abs(t_stat), dof))
        p_values.append(float(p_val))

    return np.array(p_values)


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------


def power_analysis(
    adj: np.ndarray,
    treatment: int,
    outcome: int,
    sample_sizes: list[int],
    alpha: float = 0.05,
    n_simulations: int = 200,
    seed: int = 42,
) -> dict[int, float]:
    """Estimate statistical power for different sample sizes.

    Power is defined as P(reject H0: ATE=0 | ATE ≠ 0).
    """
    results: dict[int, float] = {}
    weight_rng = np.random.default_rng(seed)
    weight_seed = int(weight_rng.integers(0, 2**31))

    for n_obs in sample_sizes:
        rejections = 0
        for sim in range(n_simulations):
            data = generate_linear_gaussian_data(
                adj, n_obs, seed=seed + sim * 1000, weight_seed=weight_seed,
            )
            ate, se = simple_ols_ate(data, treatment, outcome)
            if se > 1e-12:
                z = abs(ate / se)
                p_val = 2 * (1 - sp_stats.norm.cdf(z))
                if p_val < alpha:
                    rejections += 1
        results[n_obs] = rejections / n_simulations

    return results


def minimum_detectable_effect(
    adj: np.ndarray,
    treatment: int,
    outcome: int,
    n_obs: int = 200,
    alpha: float = 0.05,
    power_target: float = 0.80,
    n_simulations: int = 100,
    seed: int = 42,
) -> float:
    """Estimate minimum detectable effect size via simulation."""
    # Get typical SE
    ses: list[float] = []
    weight_rng = np.random.default_rng(seed)
    weight_seed = int(weight_rng.integers(0, 2**31))
    for sim in range(n_simulations):
        data = generate_linear_gaussian_data(
            adj, n_obs, seed=seed + sim, weight_seed=weight_seed,
        )
        _, se = simple_ols_ate(data, treatment, outcome)
        ses.append(se)

    mean_se = float(np.mean(ses))
    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
    z_beta = sp_stats.norm.ppf(power_target)
    mde = (z_alpha + z_beta) * mean_se
    return mde


# ===================================================================
# Tests
# ===================================================================


class TestMonteCarloCaroverage:
    def test_nominal_coverage_chain(self):
        """CI coverage should be near nominal (95%) for chain DAG."""
        adj = _adj(3, [(0, 1), (1, 2)])
        result = monte_carlo_coverage(
            adj, 0, 2, n_obs=200, n_simulations=300, alpha=0.05, seed=42
        )
        assert 0.85 <= result["coverage"] <= 1.0

    def test_coverage_with_adjustment(self):
        adj = _adj(3, [(0, 1), (0, 2), (1, 2)])
        result = monte_carlo_coverage(
            adj, 1, 2, n_obs=200, n_simulations=300, alpha=0.05,
            adjustment=[0], seed=42,
        )
        assert 0.85 <= result["coverage"] <= 1.0

    def test_width_decreases_with_n(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        r_small = monte_carlo_coverage(adj, 0, 2, n_obs=50, n_simulations=100, seed=42)
        r_large = monte_carlo_coverage(adj, 0, 2, n_obs=500, n_simulations=100, seed=42)
        assert r_large["mean_width"] < r_small["mean_width"]

    def test_mean_ate_unbiased(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        result = monte_carlo_coverage(
            adj, 0, 2, n_obs=500, n_simulations=200, seed=42
        )
        # Mean estimated ATE should be close to true ATE
        assert abs(result["mean_ate"] - result["true_ate"]) < 0.3

    def test_se_coverage_small(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        result = monte_carlo_coverage(
            adj, 0, 2, n_obs=200, n_simulations=200, seed=42
        )
        assert result["se_coverage"] < 0.1

    def test_diamond_coverage(self):
        adj = _adj(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
        result = monte_carlo_coverage(
            adj, 0, 3, n_obs=200, n_simulations=200, seed=42
        )
        assert result["coverage"] > 0.70


class TestCalibration:
    def test_uniform_p_values(self):
        """Truly uniform p-values should pass KS test."""
        rng = np.random.default_rng(42)
        p_vals = rng.uniform(0, 1, 500)
        result = calibration_check(p_vals)
        assert result["ks_pvalue"] > 0.05

    def test_anti_conservative(self):
        """P-values concentrated near 0 should fail calibration."""
        rng = np.random.default_rng(42)
        p_vals = rng.beta(0.5, 5, 500)
        result = calibration_check(p_vals)
        assert result["rejection_rate_0.05"] > 0.05

    def test_conservative(self):
        """P-values concentrated near 1 should show low rejection rates."""
        rng = np.random.default_rng(42)
        p_vals = rng.beta(5, 0.5, 500)
        result = calibration_check(p_vals)
        assert result["rejection_rate_0.05"] < 0.05

    def test_null_p_values_uniform(self):
        """Under the null, p-values from partial correlation should be ~uniform."""
        # Fork: X ← Z → Y, so X ⊥ Y | Z
        adj = _adj(3, [(0, 1), (0, 2)])
        p_vals = generate_null_p_values(
            adj, x=1, y=2, conditioning=[0],
            n_obs=200, n_simulations=200, seed=42,
        )
        result = calibration_check(p_vals)
        # Rejection rate at 0.05 should be near 0.05
        assert result["rejection_rate_0.05"] < 0.15

    def test_non_null_p_values(self):
        """Under the alternative, p-values should be small."""
        adj = _adj(3, [(0, 1), (1, 2)])  # chain: X → M → Y
        p_vals = generate_null_p_values(
            adj, x=0, y=2, conditioning=[],
            n_obs=200, n_simulations=200, seed=42,
        )
        # These are NOT null p-values (X and Y are dependent)
        assert float(np.mean(p_vals < 0.05)) > 0.5

    def test_mean_pvalue_near_half_under_null(self):
        rng = np.random.default_rng(42)
        p_vals = rng.uniform(0, 1, 1000)
        result = calibration_check(p_vals)
        assert abs(result["mean_pvalue"] - 0.5) < 0.1


class TestPowerAnalysis:
    def test_power_increases_with_n(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        result = power_analysis(
            adj, 0, 2,
            sample_sizes=[50, 200, 500],
            n_simulations=100, seed=42,
        )
        assert result[500] >= result[50] - 0.1  # power should increase

    def test_power_bounded(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        result = power_analysis(
            adj, 0, 2,
            sample_sizes=[200],
            n_simulations=100, seed=42,
        )
        assert 0 <= result[200] <= 1

    def test_high_power_large_sample(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        result = power_analysis(
            adj, 0, 2,
            sample_sizes=[1000],
            n_simulations=100, seed=42,
        )
        assert result[1000] > 0.5

    def test_no_effect_low_power(self):
        """When there's no causal path, 'power' should be near alpha."""
        adj = _adj(4, [(0, 1), (2, 3)])
        result = power_analysis(
            adj, 0, 3,
            sample_sizes=[200],
            n_simulations=100, seed=42,
        )
        assert result[200] < 0.2


class TestMinimumDetectableEffect:
    def test_positive_mde(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        mde = minimum_detectable_effect(adj, 0, 2, n_obs=200, seed=42)
        assert mde > 0

    def test_mde_decreases_with_n(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        mde_small = minimum_detectable_effect(adj, 0, 2, n_obs=50, seed=42)
        mde_large = minimum_detectable_effect(adj, 0, 2, n_obs=500, seed=42)
        assert mde_large < mde_small

    def test_mde_with_higher_power(self):
        adj = _adj(3, [(0, 1), (1, 2)])
        mde_80 = minimum_detectable_effect(adj, 0, 2, power_target=0.80, seed=42)
        mde_90 = minimum_detectable_effect(adj, 0, 2, power_target=0.90, seed=42)
        assert mde_90 > mde_80  # higher power → need bigger effect
