"""Mediation analysis for causal effects.

Implements Natural Direct Effect (NDE), Natural Indirect Effect (NIE),
Controlled Direct Effect (CDE), sequential ignorability checking,
mediation sensitivity analysis, path-specific effects estimation,
and multiple-mediator decomposition.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats as sp_stats


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class MediationResult:
    """Output of a mediation analysis."""
    total_effect: float
    nde: float
    nie: float
    cde: float
    proportion_mediated: float
    se_nde: float
    se_nie: float
    se_cde: float
    ci_nde: Tuple[float, float] = (0.0, 0.0)
    ci_nie: Tuple[float, float] = (0.0, 0.0)
    ci_cde: Tuple[float, float] = (0.0, 0.0)
    n_bootstrap: int = 0


@dataclass
class SensitivityMediationResult:
    """Sensitivity analysis for mediation."""
    rho_range: np.ndarray
    nde_adjusted: np.ndarray
    nie_adjusted: np.ndarray
    rho_breakpoint: float = 0.0


# ===================================================================
# Helpers
# ===================================================================

def _ols_fit(Y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit OLS, return (coefficients, residuals)."""
    n = X.shape[0]
    X_aug = np.column_stack([np.ones(n), X])
    try:
        beta = np.linalg.lstsq(X_aug, Y, rcond=None)[0]
        resid = Y - X_aug @ beta
    except np.linalg.LinAlgError:
        beta = np.zeros(X_aug.shape[1])
        resid = Y.copy()
    return beta, resid


def _ols_predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    X_aug = np.column_stack([np.ones(n), X])
    return X_aug @ beta


# ===================================================================
# 1.  Natural Direct Effect (NDE) estimation
# ===================================================================

def estimate_nde(
    Y: np.ndarray,
    A: np.ndarray,
    M: np.ndarray,
    X: np.ndarray,
    *,
    treatment_value: float = 1.0,
    control_value: float = 0.0,
) -> float:
    """Estimate the Natural Direct Effect via regression.

    Under linear models:
      Y = β₀ + β₁A + β₂M + β₃X + ε
      M = γ₀ + γ₁A + γ₂X + η

    NDE = β₁(a - a*) where a* = control, a = treatment.

    Reference: VanderWeele (2015), Chapter 2.
    """
    n = Y.shape[0]
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    AMX = np.column_stack([A.reshape(-1, 1), M, X])
    beta_y, _ = _ols_fit(Y.ravel(), AMX)

    nde = beta_y[1] * (treatment_value - control_value)
    return float(nde)


# ===================================================================
# 2.  Natural Indirect Effect (NIE) estimation
# ===================================================================

def estimate_nie(
    Y: np.ndarray,
    A: np.ndarray,
    M: np.ndarray,
    X: np.ndarray,
    *,
    treatment_value: float = 1.0,
    control_value: float = 0.0,
) -> float:
    """Estimate the Natural Indirect Effect via the product method.

    NIE = β₂ × γ₁ × (a - a*).
    """
    n = Y.shape[0]
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    AMX = np.column_stack([A.reshape(-1, 1), M, X])
    beta_y, _ = _ols_fit(Y.ravel(), AMX)

    AX = np.column_stack([A.reshape(-1, 1), X])
    gamma_m, _ = _ols_fit(M.ravel(), AX)

    n_mediators = M.shape[1]
    nie = 0.0
    for k in range(n_mediators):
        beta_mk = beta_y[2 + k]
        gamma_1k = gamma_m[1]
        nie += beta_mk * gamma_1k

    nie *= (treatment_value - control_value)
    return float(nie)


# ===================================================================
# 3.  Controlled Direct Effect (CDE)
# ===================================================================

def estimate_cde(
    Y: np.ndarray,
    A: np.ndarray,
    M: np.ndarray,
    X: np.ndarray,
    *,
    mediator_level: float = 0.0,
    treatment_value: float = 1.0,
    control_value: float = 0.0,
    include_interaction: bool = False,
) -> float:
    """Estimate the Controlled Direct Effect at a given mediator level.

    CDE(m) = (β₁ + β₃ m)(a - a*)  when interaction is included.
    CDE(m) = β₁(a - a*)            without interaction.
    """
    n = Y.shape[0]
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if include_interaction:
        AM_interaction = (A.reshape(-1, 1) * M)
        design = np.column_stack([A.reshape(-1, 1), M, AM_interaction, X])
        beta, _ = _ols_fit(Y.ravel(), design)
        cde = (beta[1] + beta[2 + M.shape[1]] * mediator_level) * (treatment_value - control_value)
    else:
        design = np.column_stack([A.reshape(-1, 1), M, X])
        beta, _ = _ols_fit(Y.ravel(), design)
        cde = beta[1] * (treatment_value - control_value)

    return float(cde)


# ===================================================================
# 4.  Full mediation analysis with bootstrap
# ===================================================================

def mediation_analysis(
    Y: np.ndarray,
    A: np.ndarray,
    M: np.ndarray,
    X: np.ndarray,
    *,
    treatment_value: float = 1.0,
    control_value: float = 0.0,
    mediator_level: float = 0.0,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    rng: Optional[np.random.RandomState] = None,
) -> MediationResult:
    """Full mediation analysis: NDE, NIE, CDE with bootstrap CIs.

    Returns a :class:`MediationResult` with point estimates and
    percentile bootstrap confidence intervals.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n = Y.shape[0]
    nde = estimate_nde(Y, A, M, X, treatment_value=treatment_value, control_value=control_value)
    nie = estimate_nie(Y, A, M, X, treatment_value=treatment_value, control_value=control_value)
    cde = estimate_cde(Y, A, M, X, mediator_level=mediator_level,
                       treatment_value=treatment_value, control_value=control_value)
    te = nde + nie

    boot_nde = np.empty(n_bootstrap)
    boot_nie = np.empty(n_bootstrap)
    boot_cde = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        Y_b, A_b, M_b, X_b = Y[idx], A[idx], M[idx], X[idx]
        boot_nde[b] = estimate_nde(Y_b, A_b, M_b, X_b,
                                   treatment_value=treatment_value,
                                   control_value=control_value)
        boot_nie[b] = estimate_nie(Y_b, A_b, M_b, X_b,
                                   treatment_value=treatment_value,
                                   control_value=control_value)
        boot_cde[b] = estimate_cde(Y_b, A_b, M_b, X_b,
                                   mediator_level=mediator_level,
                                   treatment_value=treatment_value,
                                   control_value=control_value)

    se_nde = float(np.std(boot_nde))
    se_nie = float(np.std(boot_nie))
    se_cde = float(np.std(boot_cde))

    q_lo = 100 * alpha / 2
    q_hi = 100 * (1 - alpha / 2)

    proportion = nie / te if abs(te) > 1e-12 else 0.0

    return MediationResult(
        total_effect=te,
        nde=nde,
        nie=nie,
        cde=cde,
        proportion_mediated=proportion,
        se_nde=se_nde,
        se_nie=se_nie,
        se_cde=se_cde,
        ci_nde=(float(np.percentile(boot_nde, q_lo)),
                float(np.percentile(boot_nde, q_hi))),
        ci_nie=(float(np.percentile(boot_nie, q_lo)),
                float(np.percentile(boot_nie, q_hi))),
        ci_cde=(float(np.percentile(boot_cde, q_lo)),
                float(np.percentile(boot_cde, q_hi))),
        n_bootstrap=n_bootstrap,
    )


# ===================================================================
# 5.  Sequential ignorability checking
# ===================================================================

def check_sequential_ignorability(
    Y: np.ndarray,
    A: np.ndarray,
    M: np.ndarray,
    X: np.ndarray,
    *,
    n_permutations: int = 500,
    alpha: float = 0.05,
) -> Dict[str, object]:
    """Check sequential ignorability assumptions for mediation.

    Sequential ignorability requires:
      (SI-1): {Y(a',m), M(a)} ⊥ A | X    (treatment ignorable)
      (SI-2): Y(a',m) ⊥ M(a) | A, X       (mediator ignorable given treatment)

    We provide diagnostic tests for each:
      - SI-1: balance check — compare covariate distributions by treatment
      - SI-2: residual correlation between Y-residuals and M-residuals
    """
    n = Y.shape[0]
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if M.ndim == 1:
        M = M.reshape(-1, 1)

    treated = A.ravel() > 0.5
    balance_stats: Dict[str, float] = {}
    for col in range(X.shape[1]):
        x_t = X[treated, col]
        x_c = X[~treated, col]
        if len(x_t) > 1 and len(x_c) > 1:
            std_pooled = math.sqrt(
                (np.var(x_t) * (len(x_t) - 1) + np.var(x_c) * (len(x_c) - 1))
                / max(len(x_t) + len(x_c) - 2, 1)
            )
            smd = abs(np.mean(x_t) - np.mean(x_c)) / max(std_pooled, 1e-12)
        else:
            smd = 0.0
        balance_stats[f"X{col}_smd"] = smd

    si1_pass = all(v < 0.1 for v in balance_stats.values())

    AX = np.column_stack([A.reshape(-1, 1), X])
    _, res_y = _ols_fit(Y.ravel(), AX)
    _, res_m = _ols_fit(M.ravel(), AX)

    if res_m.ndim > 1:
        res_m = res_m[:, 0]

    corr = float(np.corrcoef(res_y, res_m)[0, 1])

    rng = np.random.RandomState(42)
    null_corrs = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        null_corrs[i] = float(np.corrcoef(res_y[perm], res_m)[0, 1])

    si2_pvalue = float(np.mean(np.abs(null_corrs) >= abs(corr)))
    si2_pass = si2_pvalue > alpha

    return {
        "si1_balance": balance_stats,
        "si1_pass": si1_pass,
        "si2_residual_correlation": corr,
        "si2_pvalue": si2_pvalue,
        "si2_pass": si2_pass,
        "overall_pass": si1_pass and si2_pass,
    }


# ===================================================================
# 6.  Mediation sensitivity analysis
# ===================================================================

def mediation_sensitivity(
    Y: np.ndarray,
    A: np.ndarray,
    M: np.ndarray,
    X: np.ndarray,
    *,
    rho_range: Tuple[float, float] = (-0.9, 0.9),
    n_rho: int = 50,
    treatment_value: float = 1.0,
    control_value: float = 0.0,
) -> SensitivityMediationResult:
    """Sensitivity of NDE/NIE to violations of sequential ignorability.

    Parameterises the sensitivity by ρ, the correlation between the
    error terms of the outcome and mediator models.  ρ = 0 corresponds
    to the identifying assumption.

    Reference: Imai, Keele, Yamamoto (2010).
    """
    n = Y.shape[0]
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    AX = np.column_stack([A.reshape(-1, 1), X])
    gamma, eta = _ols_fit(M.ravel(), AX)
    sigma_m = float(np.std(eta))

    AMX = np.column_stack([A.reshape(-1, 1), M, X])
    beta, epsilon = _ols_fit(Y.ravel(), AMX)
    sigma_y = float(np.std(epsilon))

    rhos = np.linspace(rho_range[0], rho_range[1], n_rho)
    nde_adj = np.empty(n_rho)
    nie_adj = np.empty(n_rho)

    beta_a = beta[1]
    beta_m = beta[2]
    gamma_a = gamma[1]

    for idx, rho in enumerate(rhos):
        bias = rho * sigma_y * sigma_m
        nde_adj[idx] = (beta_a + beta_m * bias / max(sigma_m ** 2, 1e-12)) * (treatment_value - control_value)
        nie_adj[idx] = (beta_m * gamma_a - beta_m * bias / max(sigma_m ** 2, 1e-12)) * (treatment_value - control_value)

    breakpoint_rho = 0.0
    for idx in range(len(rhos) - 1):
        if nie_adj[idx] * nie_adj[idx + 1] < 0:
            t = abs(nie_adj[idx]) / max(abs(nie_adj[idx] - nie_adj[idx + 1]), 1e-12)
            breakpoint_rho = rhos[idx] + t * (rhos[idx + 1] - rhos[idx])
            break

    return SensitivityMediationResult(
        rho_range=rhos,
        nde_adjusted=nde_adj,
        nie_adjusted=nie_adj,
        rho_breakpoint=breakpoint_rho,
    )


# ===================================================================
# 7.  Path-specific effects estimation
# ===================================================================

def path_specific_effect(
    Y: np.ndarray,
    A: np.ndarray,
    mediators: Dict[str, np.ndarray],
    X: np.ndarray,
    active_mediators: Set[str],
    *,
    treatment_value: float = 1.0,
    control_value: float = 0.0,
) -> float:
    """Estimate a path-specific effect through specified mediators.

    Decomposes the total effect into a component through the active
    mediators and the rest.  Under linear models, the path-specific
    effect through mediator set S is:
        PSE(S) = Σ_{m ∈ S} β_m γ_m (a - a*)

    Parameters
    ----------
    mediators : dict mapping mediator name → array
    active_mediators : set of mediator names for the active path
    """
    n = Y.shape[0]
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    all_m_arrays = []
    m_names = sorted(mediators.keys())
    for name in m_names:
        arr = mediators[name]
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        all_m_arrays.append(arr)

    M_all = np.column_stack(all_m_arrays) if all_m_arrays else np.empty((n, 0))

    AMX = np.column_stack([A.reshape(-1, 1), M_all, X])
    beta_y, _ = _ols_fit(Y.ravel(), AMX)

    pse = 0.0
    m_col_offset = 0
    for name in m_names:
        arr = mediators[name]
        n_cols = arr.shape[1] if arr.ndim > 1 else 1

        if name in active_mediators:
            AX = np.column_stack([A.reshape(-1, 1), X])
            m_flat = arr.ravel() if n_cols == 1 else arr[:, 0]
            gamma, _ = _ols_fit(m_flat, AX)
            gamma_a = gamma[1]

            beta_m = beta_y[2 + m_col_offset]
            pse += beta_m * gamma_a

        m_col_offset += n_cols

    pse *= (treatment_value - control_value)
    return float(pse)


# ===================================================================
# 8.  Multiple mediator decomposition
# ===================================================================

def multiple_mediator_decomposition(
    Y: np.ndarray,
    A: np.ndarray,
    mediators: Dict[str, np.ndarray],
    X: np.ndarray,
    *,
    treatment_value: float = 1.0,
    control_value: float = 0.0,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    rng: Optional[np.random.RandomState] = None,
) -> Dict[str, Dict[str, float]]:
    """Decompose the total effect into mediator-specific indirect effects.

    For each mediator M_k, computes:
      IE_k = β_{M_k} × γ_{A→M_k} × (a - a*)

    Also computes the direct effect (not through any mediator).

    Returns a dict mapping:
      mediator name → {"indirect_effect": ..., "se": ..., "proportion": ...}
      "direct" → {"effect": ..., "se": ...}
      "total" → {"effect": ..., "se": ...}
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n = Y.shape[0]
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    m_names = sorted(mediators.keys())
    all_m = []
    for name in m_names:
        arr = mediators[name]
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        all_m.append(arr)
    M_all = np.column_stack(all_m) if all_m else np.empty((n, 0))

    def _decompose(Y_s, A_s, M_s, X_s):
        AMX = np.column_stack([A_s.reshape(-1, 1), M_s, X_s])
        beta, _ = _ols_fit(Y_s.ravel(), AMX)
        direct = beta[1] * (treatment_value - control_value)

        indirect_effects = {}
        col_offset = 0
        for name in m_names:
            arr = mediators[name]
            n_cols = arr.shape[1] if arr.ndim > 1 else 1

            AX = np.column_stack([A_s.reshape(-1, 1), X_s])
            m_flat = M_s[:, col_offset] if M_s.ndim > 1 else M_s.ravel()
            gamma, _ = _ols_fit(m_flat, AX)
            ie = beta[2 + col_offset] * gamma[1] * (treatment_value - control_value)
            indirect_effects[name] = ie
            col_offset += n_cols

        total = direct + sum(indirect_effects.values())
        return direct, indirect_effects, total

    direct_pt, ie_pt, total_pt = _decompose(Y, A, M_all, X)

    boot_direct = np.empty(n_bootstrap)
    boot_ie: Dict[str, np.ndarray] = {name: np.empty(n_bootstrap) for name in m_names}
    boot_total = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        M_b = M_all[idx]
        d, ies, t = _decompose(Y[idx], A[idx], M_b, X[idx])
        boot_direct[b] = d
        boot_total[b] = t
        for name in m_names:
            boot_ie[name][b] = ies[name]

    result: Dict[str, Dict[str, float]] = {}
    for name in m_names:
        ie_val = ie_pt[name]
        se = float(np.std(boot_ie[name]))
        proportion = ie_val / total_pt if abs(total_pt) > 1e-12 else 0.0
        result[name] = {
            "indirect_effect": ie_val,
            "se": se,
            "proportion": proportion,
            "ci_lower": float(np.percentile(boot_ie[name], 100 * alpha / 2)),
            "ci_upper": float(np.percentile(boot_ie[name], 100 * (1 - alpha / 2))),
        }

    result["direct"] = {
        "effect": direct_pt,
        "se": float(np.std(boot_direct)),
        "ci_lower": float(np.percentile(boot_direct, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_direct, 100 * (1 - alpha / 2))),
    }
    result["total"] = {
        "effect": total_pt,
        "se": float(np.std(boot_total)),
        "ci_lower": float(np.percentile(boot_total, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_total, 100 * (1 - alpha / 2))),
    }

    return result
