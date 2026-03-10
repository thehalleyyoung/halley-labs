"""
Tests for mediation analysis.

Covers NDE/NIE estimation, path-specific effects, and tests with known
mediation proportions using synthetic data.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from tests.conftest import _adj


# ---------------------------------------------------------------------------
# Mediation estimators
# ---------------------------------------------------------------------------


def estimate_nde_regression(
    data: pd.DataFrame,
    treatment_col: str,
    mediator_col: str,
    outcome_col: str,
    covariate_cols: list[str] | None = None,
) -> float:
    """Estimate Natural Direct Effect using regression-based approach.

    Assumes linear models for both mediator and outcome:
      M = alpha0 + alpha1 * T + alpha2' * C + eps_m
      Y = beta0 + beta1 * T + beta2 * M + beta3' * C + eps_y

    NDE = beta1 (coefficient of T in outcome model with M controlled)
    """
    T = data[treatment_col].values.astype(float)
    M = data[mediator_col].values.astype(float)
    Y = data[outcome_col].values.astype(float)

    # Build design matrix for outcome model
    X_cols = [T, M]
    if covariate_cols:
        for c in covariate_cols:
            X_cols.append(data[c].values.astype(float))
    X = np.column_stack([np.ones(len(T))] + X_cols)

    # OLS
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    return float(beta[1])  # coefficient on T


def estimate_nie_regression(
    data: pd.DataFrame,
    treatment_col: str,
    mediator_col: str,
    outcome_col: str,
    covariate_cols: list[str] | None = None,
) -> float:
    """Estimate Natural Indirect Effect using product-of-coefficients.

    NIE = alpha1 * beta2
    where alpha1 is T's effect on M, beta2 is M's effect on Y (controlling T).
    """
    T = data[treatment_col].values.astype(float)
    M = data[mediator_col].values.astype(float)
    Y = data[outcome_col].values.astype(float)

    # Mediator model: M ~ T + C
    X_m = [T]
    if covariate_cols:
        for c in covariate_cols:
            X_m.append(data[c].values.astype(float))
    Xm = np.column_stack([np.ones(len(T))] + X_m)
    alpha = np.linalg.lstsq(Xm, M, rcond=None)[0]
    alpha1 = alpha[1]

    # Outcome model: Y ~ T + M + C
    X_y = [T, M]
    if covariate_cols:
        for c in covariate_cols:
            X_y.append(data[c].values.astype(float))
    Xy = np.column_stack([np.ones(len(T))] + X_y)
    beta = np.linalg.lstsq(Xy, Y, rcond=None)[0]
    beta2 = beta[2]

    return float(alpha1 * beta2)


def estimate_total_effect(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: list[str] | None = None,
) -> float:
    """Estimate total effect via regression of Y on T (+ covariates)."""
    T = data[treatment_col].values.astype(float)
    Y = data[outcome_col].values.astype(float)

    X_cols = [T]
    if covariate_cols:
        for c in covariate_cols:
            X_cols.append(data[c].values.astype(float))
    X = np.column_stack([np.ones(len(T))] + X_cols)

    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    return float(beta[1])


def mediation_proportion(
    data: pd.DataFrame,
    treatment_col: str,
    mediator_col: str,
    outcome_col: str,
    covariate_cols: list[str] | None = None,
) -> float:
    """Compute proportion mediated = NIE / TE."""
    te = estimate_total_effect(data, treatment_col, outcome_col, covariate_cols)
    nie = estimate_nie_regression(
        data, treatment_col, mediator_col, outcome_col, covariate_cols
    )
    if abs(te) < 1e-12:
        return 0.0
    return nie / te


def mediation_decomposition(
    data: pd.DataFrame,
    treatment_col: str,
    mediator_col: str,
    outcome_col: str,
    covariate_cols: list[str] | None = None,
) -> dict[str, float]:
    """Full mediation decomposition: TE = NDE + NIE."""
    te = estimate_total_effect(data, treatment_col, outcome_col, covariate_cols)
    nde = estimate_nde_regression(
        data, treatment_col, mediator_col, outcome_col, covariate_cols
    )
    nie = estimate_nie_regression(
        data, treatment_col, mediator_col, outcome_col, covariate_cols
    )
    prop = nie / te if abs(te) > 1e-12 else 0.0

    return {
        "total_effect": te,
        "nde": nde,
        "nie": nie,
        "proportion_mediated": prop,
        "decomposition_check": abs(te - nde - nie),
    }


def bootstrap_mediation_ci(
    data: pd.DataFrame,
    treatment_col: str,
    mediator_col: str,
    outcome_col: str,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Bootstrap confidence intervals for mediation effects."""
    rng = np.random.default_rng(seed)
    n = len(data)
    nde_boots: list[float] = []
    nie_boots: list[float] = []
    te_boots: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_data = data.iloc[idx].reset_index(drop=True)
        try:
            nde_boots.append(estimate_nde_regression(
                boot_data, treatment_col, mediator_col, outcome_col
            ))
            nie_boots.append(estimate_nie_regression(
                boot_data, treatment_col, mediator_col, outcome_col
            ))
            te_boots.append(estimate_total_effect(
                boot_data, treatment_col, outcome_col
            ))
        except Exception:
            pass

    lo_q = (alpha / 2) * 100
    hi_q = (1 - alpha / 2) * 100
    return {
        "nde_ci": (
            float(np.percentile(nde_boots, lo_q)),
            float(np.percentile(nde_boots, hi_q)),
        ),
        "nie_ci": (
            float(np.percentile(nie_boots, lo_q)),
            float(np.percentile(nie_boots, hi_q)),
        ),
        "te_ci": (
            float(np.percentile(te_boots, lo_q)),
            float(np.percentile(te_boots, hi_q)),
        ),
    }


# ---------------------------------------------------------------------------
# Path-specific effects
# ---------------------------------------------------------------------------


def path_specific_effect(
    data: pd.DataFrame,
    treatment_col: str,
    mediators: list[str],
    outcome_col: str,
    active_mediators: list[str],
) -> float:
    """Estimate path-specific effect through active mediators only.

    The effect along the path T → active_mediators → Y, holding
    other mediators fixed.
    """
    # Full model: Y ~ T + all mediators
    T = data[treatment_col].values.astype(float)
    Y = data[outcome_col].values.astype(float)
    M_all = np.column_stack([data[m].values.astype(float) for m in mediators])

    X_full = np.column_stack([np.ones(len(T)), T, M_all])
    beta_full = np.linalg.lstsq(X_full, Y, rcond=None)[0]
    beta_m = beta_full[2:]  # mediator coefficients

    # Effect through active mediators: sum of alpha_m * beta_m for active m
    pse = 0.0
    for i, m in enumerate(mediators):
        if m in active_mediators:
            # alpha_m: coefficient of T in model M_i ~ T
            Xm = np.column_stack([np.ones(len(T)), T])
            alpha_m = np.linalg.lstsq(Xm, data[m].values.astype(float), rcond=None)[0][1]
            pse += alpha_m * beta_m[i]

    return float(pse)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _generate_simple_mediation(
    n: int = 1000,
    a: float = 1.0,  # T → M
    b: float = 1.0,  # M → Y
    c: float = 0.5,  # T → Y (direct)
    seed: int = 42,
) -> pd.DataFrame:
    """Generate T → M → Y with direct effect T → Y.

    True TE = c + a*b
    True NDE = c
    True NIE = a*b
    """
    rng = np.random.default_rng(seed)
    T = rng.standard_normal(n)
    M = a * T + 0.5 * rng.standard_normal(n)
    Y = b * M + c * T + 0.5 * rng.standard_normal(n)
    return pd.DataFrame({"T": T, "M": M, "Y": Y})


def _generate_two_mediator(
    n: int = 1000,
    a1: float = 1.0, b1: float = 0.8,
    a2: float = 0.5, b2: float = 0.6,
    c: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate T → M1 → Y, T → M2 → Y, T → Y."""
    rng = np.random.default_rng(seed)
    T = rng.standard_normal(n)
    M1 = a1 * T + 0.3 * rng.standard_normal(n)
    M2 = a2 * T + 0.3 * rng.standard_normal(n)
    Y = b1 * M1 + b2 * M2 + c * T + 0.3 * rng.standard_normal(n)
    return pd.DataFrame({"T": T, "M1": M1, "M2": M2, "Y": Y})


def _generate_confounded_mediation(
    n: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """T → M → Y with confounder C → T, C → Y."""
    rng = np.random.default_rng(seed)
    C = rng.standard_normal(n)
    T = 0.8 * C + 0.5 * rng.standard_normal(n)
    M = 1.0 * T + 0.3 * rng.standard_normal(n)
    Y = 0.8 * M + 0.5 * T + 0.6 * C + 0.3 * rng.standard_normal(n)
    return pd.DataFrame({"C": C, "T": T, "M": M, "Y": Y})


# ===================================================================
# Tests
# ===================================================================


class TestNDEEstimation:
    def test_known_nde(self):
        """NDE should approximate the true direct effect."""
        data = _generate_simple_mediation(n=2000, a=1.0, b=1.0, c=0.5, seed=42)
        nde = estimate_nde_regression(data, "T", "M", "Y")
        assert nde == pytest.approx(0.5, abs=0.15)

    def test_zero_direct_effect(self):
        """When c=0, NDE should be near zero."""
        data = _generate_simple_mediation(n=2000, a=1.0, b=1.0, c=0.0, seed=42)
        nde = estimate_nde_regression(data, "T", "M", "Y")
        assert abs(nde) < 0.15

    def test_large_direct_effect(self):
        data = _generate_simple_mediation(n=2000, a=0.5, b=0.5, c=2.0, seed=42)
        nde = estimate_nde_regression(data, "T", "M", "Y")
        assert nde == pytest.approx(2.0, abs=0.2)


class TestNIEEstimation:
    def test_known_nie(self):
        """NIE should approximate a*b."""
        data = _generate_simple_mediation(n=2000, a=1.0, b=1.0, c=0.5, seed=42)
        nie = estimate_nie_regression(data, "T", "M", "Y")
        assert nie == pytest.approx(1.0, abs=0.15)

    def test_zero_mediation(self):
        """When a=0 (no T→M effect), NIE ≈ 0."""
        data = _generate_simple_mediation(n=2000, a=0.0, b=1.0, c=1.0, seed=42)
        nie = estimate_nie_regression(data, "T", "M", "Y")
        assert abs(nie) < 0.15

    def test_full_mediation(self):
        """When c=0, all effect is mediated: NIE ≈ TE."""
        data = _generate_simple_mediation(n=2000, a=1.0, b=1.0, c=0.0, seed=42)
        nie = estimate_nie_regression(data, "T", "M", "Y")
        te = estimate_total_effect(data, "T", "Y")
        assert nie == pytest.approx(te, abs=0.2)


class TestTotalEffect:
    def test_known_te(self):
        """TE = c + a*b."""
        data = _generate_simple_mediation(n=2000, a=1.0, b=1.0, c=0.5, seed=42)
        te = estimate_total_effect(data, "T", "Y")
        assert te == pytest.approx(1.5, abs=0.15)

    def test_te_equals_sum(self):
        """TE should approximately equal NDE + NIE."""
        data = _generate_simple_mediation(n=2000, a=1.0, b=0.8, c=0.6, seed=42)
        decomp = mediation_decomposition(data, "T", "M", "Y")
        assert decomp["decomposition_check"] < 0.01


class TestMediationProportion:
    def test_full_mediation(self):
        data = _generate_simple_mediation(n=2000, a=1.0, b=1.0, c=0.0, seed=42)
        prop = mediation_proportion(data, "T", "M", "Y")
        assert prop == pytest.approx(1.0, abs=0.15)

    def test_no_mediation(self):
        data = _generate_simple_mediation(n=2000, a=0.0, b=1.0, c=1.0, seed=42)
        prop = mediation_proportion(data, "T", "M", "Y")
        assert abs(prop) < 0.15

    def test_half_mediation(self):
        """a*b=0.5, c=0.5 → proportion mediated ≈ 0.5."""
        data = _generate_simple_mediation(n=5000, a=1.0, b=0.5, c=0.5, seed=42)
        prop = mediation_proportion(data, "T", "M", "Y")
        assert prop == pytest.approx(0.5, abs=0.15)

    def test_proportion_bounded(self):
        data = _generate_simple_mediation(n=2000, seed=42)
        prop = mediation_proportion(data, "T", "M", "Y")
        assert -0.5 <= prop <= 1.5  # reasonable range


class TestMediationDecomposition:
    def test_decomposition_check(self):
        """TE should equal NDE + NIE up to estimation error."""
        for seed in [42, 43, 44]:
            data = _generate_simple_mediation(n=2000, seed=seed)
            decomp = mediation_decomposition(data, "T", "M", "Y")
            assert decomp["decomposition_check"] < 0.01

    def test_all_components_present(self):
        data = _generate_simple_mediation(n=1000, seed=42)
        decomp = mediation_decomposition(data, "T", "M", "Y")
        assert "total_effect" in decomp
        assert "nde" in decomp
        assert "nie" in decomp
        assert "proportion_mediated" in decomp


class TestBootstrapCI:
    def test_ci_covers_true_value(self):
        """CI for NIE should cover the true value a*b=1.0."""
        data = _generate_simple_mediation(n=1000, a=1.0, b=1.0, c=0.5, seed=42)
        cis = bootstrap_mediation_ci(data, "T", "M", "Y", n_bootstrap=200, seed=42)
        nie_lo, nie_hi = cis["nie_ci"]
        assert nie_lo < 1.0 < nie_hi

    def test_ci_width(self):
        """CI width should decrease with sample size."""
        data_small = _generate_simple_mediation(n=200, seed=42)
        data_large = _generate_simple_mediation(n=2000, seed=42)
        ci_small = bootstrap_mediation_ci(data_small, "T", "M", "Y", n_bootstrap=100, seed=42)
        ci_large = bootstrap_mediation_ci(data_large, "T", "M", "Y", n_bootstrap=100, seed=42)
        width_small = ci_small["nie_ci"][1] - ci_small["nie_ci"][0]
        width_large = ci_large["nie_ci"][1] - ci_large["nie_ci"][0]
        assert width_large < width_small


class TestPathSpecificEffects:
    def test_two_mediator_paths(self):
        data = _generate_two_mediator(n=2000, seed=42)
        pse_m1 = path_specific_effect(data, "T", ["M1", "M2"], "Y", ["M1"])
        pse_m2 = path_specific_effect(data, "T", ["M1", "M2"], "Y", ["M2"])
        pse_both = path_specific_effect(data, "T", ["M1", "M2"], "Y", ["M1", "M2"])
        # Individual PSEs should sum to total indirect effect
        assert pse_m1 + pse_m2 == pytest.approx(pse_both, abs=0.01)

    def test_single_mediator_matches_nie(self):
        data = _generate_simple_mediation(n=2000, a=1.0, b=1.0, c=0.5, seed=42)
        pse = path_specific_effect(data, "T", ["M"], "Y", ["M"])
        nie = estimate_nie_regression(data, "T", "M", "Y")
        assert pse == pytest.approx(nie, abs=0.01)

    def test_no_active_mediator_zero(self):
        data = _generate_simple_mediation(n=1000, seed=42)
        pse = path_specific_effect(data, "T", ["M"], "Y", [])
        assert abs(pse) < 0.01


class TestConfoundedMediation:
    def test_confounder_adjustment(self):
        data = _generate_confounded_mediation(n=2000, seed=42)
        # Without adjusting for C, NDE is biased
        nde_biased = estimate_nde_regression(data, "T", "M", "Y")
        nde_adjusted = estimate_nde_regression(data, "T", "M", "Y", ["C"])
        # The adjusted estimate should be closer to true NDE=0.5
        assert abs(nde_adjusted - 0.5) < abs(nde_biased - 0.5)

    def test_unadjusted_te_biased(self):
        data = _generate_confounded_mediation(n=2000, seed=42)
        te_unadj = estimate_total_effect(data, "T", "Y")
        te_adj = estimate_total_effect(data, "T", "Y", ["C"])
        # Unadjusted TE should be biased upward due to C→T, C→Y
        assert abs(te_unadj) > abs(te_adj) + 0.1
