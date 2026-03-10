"""
Partial identification and bounds for causal effects.

Implements Manski (worst-case) bounds, Balke-Pearl bounds for instrumental
variable settings, monotone treatment response bounds, Lee bounds for
sample selection, E-value sensitivity bounds, and optimization-based
tightening under shape restrictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import optimize, stats as sp_stats


# ===================================================================
# Result data structures
# ===================================================================


@dataclass(frozen=True, slots=True)
class BoundsResult:
    """Partial identification bounds for a causal effect.

    Attributes
    ----------
    lower : float
        Lower bound on the causal effect.
    upper : float
        Upper bound on the causal effect.
    method : str
        Bounding method name.
    identified : bool
        ``True`` if lower == upper (point identification).
    width : float
        Width of the identified set.
    """

    lower: float
    upper: float
    method: str = "bounds"
    identified: bool = False
    width: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", self.upper - self.lower)
        object.__setattr__(self, "identified", abs(self.width) < 1e-12)

    def contains(self, value: float) -> bool:
        """Check if a value lies within the bounds."""
        return self.lower <= value <= self.upper


@dataclass(frozen=True, slots=True)
class SensitivityBoundsResult:
    """E-value and sensitivity bounds.

    Attributes
    ----------
    e_value : float
        E-value for the point estimate.
    e_value_ci : float
        E-value for the confidence interval limit closest to null.
    bound_at_gamma : np.ndarray
        Upper/lower bounds at each sensitivity parameter value.
    gamma_values : np.ndarray
        Grid of sensitivity parameter values.
    """

    e_value: float
    e_value_ci: float
    bound_at_gamma: np.ndarray
    gamma_values: np.ndarray


# ===================================================================
# 1. Manski bounds (worst-case nonparametric)
# ===================================================================


class ManskiBounds:
    """Manski (1990) worst-case nonparametric bounds.

    Bounds the ATE under no assumptions beyond the observed data.
    For binary outcome Y ∈ {0, 1}::

        P(Y=1|A=1)·P(A=1) + 0·P(A=0) − [P(Y=1|A=0)·P(A=0) + 1·P(A=1)]
        ≤ ATE ≤
        P(Y=1|A=1)·P(A=1) + 1·P(A=0) − [P(Y=1|A=0)·P(A=0) + 0·P(A=1)]

    For bounded continuous Y ∈ [y_min, y_max], the bounds generalise
    using the support endpoints.
    """

    def __init__(self, y_min: float = 0.0, y_max: float = 1.0) -> None:
        self.y_min = y_min
        self.y_max = y_max

    def compute(
        self,
        Y: np.ndarray,
        A: np.ndarray,
    ) -> BoundsResult:
        """Compute Manski bounds.

        Parameters
        ----------
        Y : np.ndarray
            Outcome, shape ``(n,)``.
        A : np.ndarray
            Treatment (binary), shape ``(n,)``.

        Returns
        -------
        BoundsResult
        """
        Y = np.asarray(Y, dtype=np.float64).ravel()
        A = np.asarray(A, dtype=np.float64).ravel()

        mask1 = A == 1
        mask0 = A == 0
        p1 = float(np.mean(A))
        p0 = 1.0 - p1

        if not mask1.any() or not mask0.any():
            return BoundsResult(
                lower=self.y_min - self.y_max,
                upper=self.y_max - self.y_min,
                method="manski",
            )

        mu1 = float(np.mean(Y[mask1]))
        mu0 = float(np.mean(Y[mask0]))

        # Lower bound: best case for control, worst case for treated
        lower = (mu1 * p1 + self.y_min * p0) - (mu0 * p0 + self.y_max * p1)
        # Upper bound: worst case for control, best case for treated
        upper = (mu1 * p1 + self.y_max * p0) - (mu0 * p0 + self.y_min * p1)

        return BoundsResult(lower=lower, upper=upper, method="manski")

    def compute_with_mtr(
        self,
        Y: np.ndarray,
        A: np.ndarray,
    ) -> BoundsResult:
        """Manski bounds under monotone treatment response (MTR).

        Assumes E[Y(1)] ≥ E[Y(0)] (non-negative treatment effect).

        Parameters
        ----------
        Y, A : np.ndarray
            Outcome and treatment.

        Returns
        -------
        BoundsResult
        """
        base = self.compute(Y, A)
        lower_mtr = max(base.lower, 0.0)
        return BoundsResult(
            lower=lower_mtr, upper=base.upper, method="manski_mtr",
        )


# ===================================================================
# 2. Balke-Pearl bounds (instrumental variables)
# ===================================================================


class BalkePearl:
    """Balke & Pearl (1997) bounds for the ATE using an IV.

    Provides tight nonparametric bounds on the ATE when a valid
    binary instrument Z is available, for binary treatment A and
    outcome Y.
    """

    def compute(
        self,
        Y: np.ndarray,
        A: np.ndarray,
        Z: np.ndarray,
    ) -> BoundsResult:
        """Compute Balke-Pearl IV bounds.

        Parameters
        ----------
        Y : np.ndarray
            Outcome (binary), shape ``(n,)``.
        A : np.ndarray
            Treatment (binary), shape ``(n,)``.
        Z : np.ndarray
            Instrument (binary), shape ``(n,)``.

        Returns
        -------
        BoundsResult
        """
        Y = np.asarray(Y, dtype=np.float64).ravel()
        A = np.asarray(A, dtype=np.float64).ravel()
        Z = np.asarray(Z, dtype=np.float64).ravel()

        # Estimate joint probabilities P(Y=y, A=a | Z=z)
        probs = {}
        for z in (0, 1):
            mask_z = Z == z
            n_z = float(np.sum(mask_z))
            if n_z == 0:
                for y in (0, 1):
                    for a in (0, 1):
                        probs[(y, a, z)] = 0.0
                continue
            for y in (0, 1):
                for a in (0, 1):
                    probs[(y, a, z)] = float(
                        np.sum((Y[mask_z] == y) & (A[mask_z] == a))
                    ) / n_z

        p = probs
        pz1 = float(np.mean(Z))
        pz0 = 1.0 - pz1

        # Balke-Pearl linear programming bounds
        # For binary Y, A, Z the sharp bounds are given by closed-form
        # expressions derived from the LP.
        lower_candidates = [
            -1.0,
            p[(0, 0, 0)] - p[(0, 0, 1)] - p[(0, 1, 1)] - p[(1, 0, 1)],
            p[(1, 1, 0)] - p[(1, 1, 1)] - p[(1, 0, 1)] - p[(0, 1, 1)],
            p[(1, 1, 0)] + p[(0, 0, 0)] - 1.0,
            p[(0, 0, 0)] - p[(0, 0, 1)] - p[(1, 0, 1)] - p[(0, 1, 1)],
            (p[(1, 1, 0)] - p[(0, 1, 0)]
             + p[(1, 0, 1)] - p[(0, 0, 1)]
             - p[(1, 0, 0)] + p[(0, 0, 0)]
             - p[(1, 1, 1)] + p[(0, 1, 1)]) / 2.0,
        ]

        upper_candidates = [
            1.0,
            p[(1, 0, 0)] - p[(1, 0, 1)] + p[(1, 1, 1)] + p[(0, 1, 1)],
            p[(0, 1, 0)] - p[(0, 1, 1)] + p[(1, 1, 1)] + p[(0, 1, 1)],
            1.0 - p[(0, 1, 0)] - p[(1, 0, 0)],
            p[(1, 1, 1)] + p[(0, 1, 1)] + p[(1, 0, 0)] - p[(1, 0, 1)],
            (p[(0, 1, 0)] - p[(1, 1, 0)]
             + p[(0, 0, 1)] - p[(1, 0, 1)]
             + p[(0, 0, 0)] - p[(1, 0, 0)]
             + p[(0, 1, 1)] - p[(1, 1, 1)]) / 2.0 + 1.0,
        ]

        lower = max(lower_candidates)
        upper = min(upper_candidates)

        # Ensure valid interval
        if lower > upper:
            mid = (lower + upper) / 2.0
            lower, upper = mid, mid

        return BoundsResult(lower=lower, upper=upper, method="balke_pearl")

    def wald_estimate(
        self,
        Y: np.ndarray,
        A: np.ndarray,
        Z: np.ndarray,
    ) -> float:
        """Compute the Wald (IV) point estimate.

        Wald = E[Y|Z=1] − E[Y|Z=0] / (E[A|Z=1] − E[A|Z=0])

        Parameters
        ----------
        Y, A, Z : np.ndarray
            Outcome, treatment, instrument.

        Returns
        -------
        float
        """
        Y = np.asarray(Y, dtype=np.float64).ravel()
        A = np.asarray(A, dtype=np.float64).ravel()
        Z = np.asarray(Z, dtype=np.float64).ravel()

        z1 = Z == 1
        z0 = Z == 0
        num = float(np.mean(Y[z1])) - float(np.mean(Y[z0]))
        den = float(np.mean(A[z1])) - float(np.mean(A[z0]))
        if abs(den) < 1e-12:
            return float("nan")
        return num / den


# ===================================================================
# 3. Monotone treatment response bounds
# ===================================================================


def monotone_treatment_response_bounds(
    Y: np.ndarray,
    A: np.ndarray,
    *,
    direction: str = "positive",
    y_min: float | None = None,
    y_max: float | None = None,
) -> BoundsResult:
    """Bounds under monotone treatment response (MTR).

    Assumes potential outcomes satisfy Y(1) ≥ Y(0) (positive direction)
    or Y(1) ≤ Y(0) (negative direction) for all units.

    Parameters
    ----------
    Y : np.ndarray
        Outcome.
    A : np.ndarray
        Treatment (binary).
    direction : str
        ``"positive"`` assumes Y(1) ≥ Y(0), ``"negative"`` the reverse.
    y_min, y_max : float or None
        Outcome support bounds. Inferred from data if ``None``.

    Returns
    -------
    BoundsResult
    """
    Y = np.asarray(Y, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.float64).ravel()

    if y_min is None:
        y_min = float(np.min(Y))
    if y_max is None:
        y_max = float(np.max(Y))

    mask1 = A == 1
    mask0 = A == 0
    mu1 = float(np.mean(Y[mask1])) if mask1.any() else 0.0
    mu0 = float(np.mean(Y[mask0])) if mask0.any() else 0.0
    p1 = float(np.mean(A))
    p0 = 1.0 - p1

    if direction == "positive":
        # Y(1) >= Y(0) => ATE >= 0
        lower = max(0.0, mu1 - mu0 - (y_max - y_min) * (1.0 - min(p1, p0)))
        upper = mu1 * p1 + y_max * p0 - (mu0 * p0 + y_min * p1)
        upper = min(upper, y_max - y_min)
        lower = max(lower, 0.0)
    else:
        lower = mu1 * p1 + y_min * p0 - (mu0 * p0 + y_max * p1)
        lower = max(lower, y_min - y_max)
        upper = min(0.0, mu1 - mu0 + (y_max - y_min) * (1.0 - min(p1, p0)))

    return BoundsResult(lower=lower, upper=upper, method="mtr")


# ===================================================================
# 4. Lee bounds (sample selection)
# ===================================================================


def lee_bounds(
    Y: np.ndarray,
    A: np.ndarray,
    S: np.ndarray,
    *,
    alpha: float = 0.05,
) -> BoundsResult:
    """Lee (2009) bounds for sample selection.

    Provides sharp bounds on the ATE when the outcome is observed
    only for a selected subpopulation (S=1) and selection may
    depend on treatment.

    Parameters
    ----------
    Y : np.ndarray
        Outcome (observed only when S=1; can contain NaN for S=0).
    A : np.ndarray
        Treatment (binary).
    S : np.ndarray
        Selection indicator (binary).
    alpha : float
        Significance level for inference on the bounds.

    Returns
    -------
    BoundsResult
    """
    Y = np.asarray(Y, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.float64).ravel()
    S = np.asarray(S, dtype=np.float64).ravel()

    # Observed outcomes per arm
    sel1 = (A == 1) & (S == 1)
    sel0 = (A == 0) & (S == 1)
    Y1 = Y[sel1]
    Y0 = Y[sel0]

    # Selection rates
    p_s1_a1 = float(np.mean(S[A == 1])) if np.any(A == 1) else 0.0
    p_s1_a0 = float(np.mean(S[A == 0])) if np.any(A == 0) else 0.0

    if p_s1_a1 < 1e-12 or p_s1_a0 < 1e-12:
        return BoundsResult(
            lower=float("-inf"), upper=float("inf"), method="lee",
        )

    # Proportion to trim
    if p_s1_a1 >= p_s1_a0:
        # Trim from treated arm
        q = p_s1_a0 / p_s1_a1
        Y1_sorted = np.sort(Y1)
        n1 = len(Y1_sorted)
        k = max(1, int(np.floor(q * n1)))
        # Lower bound: trim top
        lower = float(np.mean(Y1_sorted[:k])) - float(np.mean(Y0))
        # Upper bound: trim bottom
        upper = float(np.mean(Y1_sorted[n1 - k:])) - float(np.mean(Y0))
    else:
        # Trim from control arm
        q = p_s1_a1 / p_s1_a0
        Y0_sorted = np.sort(Y0)
        n0 = len(Y0_sorted)
        k = max(1, int(np.floor(q * n0)))
        # Lower bound: trim top of control
        lower = float(np.mean(Y1)) - float(np.mean(Y0_sorted[n0 - k:]))
        # Upper bound: trim bottom of control
        upper = float(np.mean(Y1)) - float(np.mean(Y0_sorted[:k]))

    return BoundsResult(lower=lower, upper=upper, method="lee")


# ===================================================================
# 5. E-value sensitivity bounds
# ===================================================================


def e_value(
    rr: float,
) -> float:
    """Compute the E-value for a risk ratio.

    The E-value is the minimum strength of unmeasured confounding
    (on the risk ratio scale) needed to fully explain away the
    observed association.

    E-value = RR + sqrt(RR * (RR − 1)) for RR ≥ 1.

    Parameters
    ----------
    rr : float
        Observed risk ratio (≥ 1).

    Returns
    -------
    float
        E-value.

    References
    ----------
    VanderWeele & Ding (2017).
    """
    if rr < 1.0:
        rr = 1.0 / rr
    return rr + np.sqrt(rr * (rr - 1.0))


def e_value_from_ate(
    ate: float,
    se: float,
    *,
    alpha: float = 0.05,
    baseline_risk: float = 0.1,
) -> SensitivityBoundsResult:
    """Compute E-value from an ATE estimate (via approximate RR conversion).

    Converts ATE to an approximate risk ratio using the baseline risk,
    then computes E-values for both the point estimate and the CI limit.

    Parameters
    ----------
    ate : float
        Average treatment effect.
    se : float
        Standard error.
    alpha : float
        Significance level.
    baseline_risk : float
        Baseline outcome risk P(Y=1|A=0), used for RR conversion.

    Returns
    -------
    SensitivityBoundsResult
    """
    z = sp_stats.norm.ppf(1.0 - alpha / 2.0)
    ci_lo = ate - z * se
    ci_hi = ate + z * se

    # Approximate RR
    p0 = max(baseline_risk, 1e-6)
    rr_point = max((p0 + ate) / p0, 1e-6)
    rr_ci = max((p0 + ci_lo) / p0, 1e-6) if ci_lo > 0 else 1.0

    e_val = e_value(rr_point)
    e_val_ci = e_value(rr_ci) if ci_lo > 0 else 1.0

    # Sensitivity curve
    gammas = np.linspace(1.0, max(5.0, 2.0 * e_val), 50)
    bound_at_gamma = np.array([
        ate - (g - 1.0) / (g + 1.0) * se * z for g in gammas
    ])

    return SensitivityBoundsResult(
        e_value=e_val,
        e_value_ci=e_val_ci,
        bound_at_gamma=bound_at_gamma,
        gamma_values=gammas,
    )


# ===================================================================
# 6. Optimization-based tightening
# ===================================================================


def tighten_bounds_monotone_iv(
    Y: np.ndarray,
    A: np.ndarray,
    Z: np.ndarray,
    *,
    monotone_direction: str = "positive",
    y_min: float | None = None,
    y_max: float | None = None,
) -> BoundsResult:
    """Tighten IV bounds by combining monotone treatment response with IV.

    Uses the intersection of Manski no-assumption bounds, MTR bounds,
    and Balke-Pearl IV bounds to produce a tighter identified set.

    Parameters
    ----------
    Y : np.ndarray
        Outcome.
    A : np.ndarray
        Treatment (binary).
    Z : np.ndarray
        Instrument (binary).
    monotone_direction : str
        ``"positive"`` or ``"negative"``.
    y_min, y_max : float or None
        Outcome support bounds.

    Returns
    -------
    BoundsResult
    """
    Y = np.asarray(Y, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.float64).ravel()
    Z = np.asarray(Z, dtype=np.float64).ravel()

    if y_min is None:
        y_min = float(np.min(Y))
    if y_max is None:
        y_max = float(np.max(Y))

    # Manski
    manski = ManskiBounds(y_min=y_min, y_max=y_max).compute(Y, A)
    # MTR
    mtr = monotone_treatment_response_bounds(
        Y, A, direction=monotone_direction, y_min=y_min, y_max=y_max,
    )
    # Balke-Pearl
    bp = BalkePearl().compute(Y, A, Z)

    # Intersection of all bounds
    lower = max(manski.lower, mtr.lower, bp.lower)
    upper = min(manski.upper, mtr.upper, bp.upper)

    if lower > upper:
        mid = (lower + upper) / 2.0
        lower, upper = mid, mid

    return BoundsResult(lower=lower, upper=upper, method="tightened_iv")


def optimization_bounds(
    Y: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    *,
    y_min: float | None = None,
    y_max: float | None = None,
    n_grid: int = 50,
) -> BoundsResult:
    """Optimization-based bounds using covariate conditioning.

    Computes Manski bounds conditional on covariate strata and
    aggregates to produce tighter overall bounds via the law of
    iterated expectations.

    Parameters
    ----------
    Y : np.ndarray
        Outcome.
    A : np.ndarray
        Treatment (binary).
    X : np.ndarray
        Covariates. Uses first column for stratification.
    y_min, y_max : float or None
        Outcome support bounds.
    n_grid : int
        Number of quantile strata for X.

    Returns
    -------
    BoundsResult
    """
    Y = np.asarray(Y, dtype=np.float64).ravel()
    A = np.asarray(A, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(Y)

    if y_min is None:
        y_min = float(np.min(Y))
    if y_max is None:
        y_max = float(np.max(Y))

    # Stratify on quantiles of X[:, 0]
    x_col = X[:, 0]
    quantiles = np.linspace(0, 100, n_grid + 1)
    edges = np.percentile(x_col, quantiles)
    edges = np.unique(edges)

    total_lower = 0.0
    total_upper = 0.0
    total_weight = 0.0

    for i in range(len(edges) - 1):
        mask = (x_col >= edges[i]) & (x_col < edges[i + 1])
        if i == len(edges) - 2:
            mask = (x_col >= edges[i]) & (x_col <= edges[i + 1])
        if mask.sum() < 2:
            continue

        Y_s, A_s = Y[mask], A[mask]
        w_s = float(mask.sum()) / n

        bounds_s = ManskiBounds(y_min=y_min, y_max=y_max).compute(Y_s, A_s)
        total_lower += w_s * bounds_s.lower
        total_upper += w_s * bounds_s.upper
        total_weight += w_s

    if total_weight < 1e-12:
        return ManskiBounds(y_min=y_min, y_max=y_max).compute(Y, A)

    lower = total_lower / total_weight
    upper = total_upper / total_weight

    return BoundsResult(lower=lower, upper=upper, method="optimization")
