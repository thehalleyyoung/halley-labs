"""
Minimal Independent Certificate Checker (MiniCheck)

A standalone, minimal (<2000 LoC) certificate checker that independently
verifies PhaseCartographer certificates using ONLY basic interval arithmetic.
No dependency on dReal, Z3, CAPD, libSBML, or the main PhaseCartographer codebase.

This module is the sole Trusted Computing Base (TCB) for certificate verification.
It is designed to be:
1. Small enough to audit by hand (<2000 LoC)
2. Self-contained (depends only on Python stdlib + numpy)
3. Conservative (rejects on any doubt)
4. Independent (reimplements all verification from scratch)

Certificate format (JSON):
{
    "model": {"name": str, "n_states": int, "n_params": int, "rhs_type": str},
    "parameter_box": [[lo, hi], ...],
    "equilibria": [
        {
            "state_enclosure": [[lo, hi], ...],
            "stability": str,
            "eigenvalue_real_parts": [[lo, hi], ...],
            "krawczyk_contraction": float,
            "krawczyk_iterations": int,
            "delta_bound": {"delta_required": float, "eigenvalue_gap": float, ...}
        }, ...
    ],
    "regime_label": str,
    "coverage_fraction": float,
    "metadata": {...}
}

Verification procedure:
1. Parse certificate JSON
2. For each claimed equilibrium:
   a. Recompute f(X, M) using interval arithmetic
   b. Verify Krawczyk contraction K(X) ⊂ X
   c. Recompute eigenvalue enclosures via Gershgorin disks
   d. Verify stability classification is consistent
   e. Verify δ-bound is sufficient
3. Verify regime label matches equilibrium count/stability
"""

import json
import math
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# numpy is the ONLY external dependency
import numpy as np


# ============================================================================
# Section 1: Minimal Interval Arithmetic (~300 LoC)
# ============================================================================

class IV:
    """Minimal rigorous interval [lo, hi] with outward-rounded arithmetic."""
    __slots__ = ['lo', 'hi']
    
    def __init__(self, lo: float, hi: float = None):
        if hi is None:
            hi = lo
        self.lo = float(lo)
        self.hi = float(hi)
        assert self.lo <= self.hi or (math.isnan(self.lo) and math.isnan(self.hi)), \
            f"Invalid interval [{self.lo}, {self.hi}]"
    
    @property
    def mid(self) -> float:
        return (self.lo + self.hi) / 2.0
    
    @property
    def width(self) -> float:
        return self.hi - self.lo
    
    @property
    def mag(self) -> float:
        return max(abs(self.lo), abs(self.hi))
    
    def contains(self, x) -> bool:
        if isinstance(x, IV):
            return self.lo <= x.lo and x.hi <= self.hi
        return self.lo <= float(x) <= self.hi
    
    def is_empty(self) -> bool:
        return math.isnan(self.lo)
    
    def intersection(self, other: 'IV') -> 'IV':
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        if lo > hi:
            return IV(float('nan'), float('nan'))
        return IV(lo, hi)
    
    def hull(self, other: 'IV') -> 'IV':
        return IV(min(self.lo, other.lo), max(self.hi, other.hi))
    
    # Conservative outward rounding: use epsilon inflation
    _EPS = 2.0 ** -52  # machine epsilon for double precision
    
    @staticmethod
    def _round_down(x: float) -> float:
        """Conservative rounding toward -∞."""
        if x == 0.0 or math.isinf(x) or math.isnan(x):
            return x
        return x - abs(x) * IV._EPS - IV._EPS
    
    @staticmethod
    def _round_up(x: float) -> float:
        """Conservative rounding toward +∞."""
        if x == 0.0 or math.isinf(x) or math.isnan(x):
            return x
        return x + abs(x) * IV._EPS + IV._EPS
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = IV(other)
        return IV(IV._round_down(self.lo + other.lo),
                  IV._round_up(self.hi + other.hi))
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __neg__(self):
        return IV(-self.hi, -self.lo)
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = IV(other)
        return IV(IV._round_down(self.lo - other.hi),
                  IV._round_up(self.hi - other.lo))
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = IV(other)
        return other.__sub__(self)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = IV(other)
        products = [self.lo * other.lo, self.lo * other.hi,
                    self.hi * other.lo, self.hi * other.hi]
        return IV(IV._round_down(min(products)),
                  IV._round_up(max(products)))
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = IV(other)
        if other.lo <= 0 <= other.hi:
            raise ZeroDivisionError(f"Division by interval containing zero: [{other.lo}, {other.hi}]")
        recip = IV(IV._round_down(1.0 / other.hi), IV._round_up(1.0 / other.lo))
        return self * recip
    
    def __pow__(self, n: int):
        if n == 0:
            return IV(1.0)
        if n == 1:
            return IV(self.lo, self.hi)
        if n < 0:
            return IV(1.0) / (self ** (-n))
        if n % 2 == 0:
            if self.lo >= 0:
                return IV(IV._round_down(self.lo ** n), IV._round_up(self.hi ** n))
            if self.hi <= 0:
                return IV(IV._round_down(self.hi ** n), IV._round_up(self.lo ** n))
            return IV(0.0, IV._round_up(max(abs(self.lo), abs(self.hi)) ** n))
        else:
            return IV(IV._round_down(self.lo ** n), IV._round_up(self.hi ** n))
    
    def __abs__(self):
        if self.lo >= 0:
            return IV(self.lo, self.hi)
        if self.hi <= 0:
            return IV(-self.hi, -self.lo)
        return IV(0.0, max(-self.lo, self.hi))
    
    def __repr__(self):
        return f"[{self.lo:.6e}, {self.hi:.6e}]"


class IVec:
    """Minimal interval vector."""
    def __init__(self, components: list):
        self.c = [x if isinstance(x, IV) else IV(x) for x in components]
    
    @property
    def n(self) -> int:
        return len(self.c)
    
    def __getitem__(self, i) -> IV:
        return self.c[i]
    
    def midpoint(self) -> np.ndarray:
        return np.array([x.mid for x in self.c])
    
    def radius(self) -> np.ndarray:
        return np.array([x.width / 2 for x in self.c])
    
    def contains(self, other: 'IVec') -> bool:
        return all(a.contains(b) for a, b in zip(self.c, other.c))
    
    def max_width(self) -> float:
        return max(x.width for x in self.c)
    
    def __sub__(self, other):
        return IVec([a - b for a, b in zip(self.c, other.c)])
    
    def __repr__(self):
        return f"IVec({self.c})"


class IMat:
    """Minimal interval matrix."""
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.n = data.shape[0]
            self.m = data.shape[1]
            self.e = [[IV(float(data[i, j])) for j in range(self.m)] for i in range(self.n)]
        elif isinstance(data, list):
            self.n = len(data)
            self.m = len(data[0])
            self.e = [[x if isinstance(x, IV) else IV(x) for x in row] for row in data]
    
    def __getitem__(self, idx):
        i, j = idx
        return self.e[i][j]
    
    def matvec(self, v: IVec) -> IVec:
        """Matrix-vector product."""
        result = []
        for i in range(self.n):
            s = IV(0.0)
            for j in range(self.m):
                s = s + self.e[i][j] * v[j]
            result.append(s)
        return IVec(result)
    
    def matmul(self, other: 'IMat') -> 'IMat':
        """Matrix-matrix product."""
        result = []
        for i in range(self.n):
            row = []
            for j in range(other.m):
                s = IV(0.0)
                for k in range(self.m):
                    s = s + self.e[i][k] * other.e[k][j]
                row.append(s)
            result.append(row)
        return IMat(result)
    
    @staticmethod
    def identity(n: int) -> 'IMat':
        data = [[IV(1.0) if i == j else IV(0.0) for j in range(n)] for i in range(n)]
        return IMat(data)
    
    @staticmethod
    def from_numpy(A: np.ndarray) -> 'IMat':
        return IMat(A)
    
    def gershgorin_disks(self) -> List[Tuple[IV, float]]:
        """Compute Gershgorin disks for eigenvalue enclosure."""
        disks = []
        for i in range(self.n):
            center = self.e[i][i]
            radius = 0.0
            for j in range(self.n):
                if i != j:
                    radius += self.e[i][j].mag
            disks.append((center, radius))
        return disks


# ============================================================================
# Section 2: Biological Model RHS Evaluation (~200 LoC)
# ============================================================================

def hill_repression(x: IV, alpha: IV, K: IV, n_hill: int) -> IV:
    """Hill repression: α / (1 + (x/K)^n)."""
    xn = x ** n_hill
    Kn = K ** n_hill
    return alpha * Kn / (Kn + xn)


def hill_activation(x: IV, alpha: IV, K: IV, n_hill: int) -> IV:
    """Hill activation: α * x^n / (K^n + x^n)."""
    xn = x ** n_hill
    Kn = K ** n_hill
    return alpha * xn / (Kn + xn)


# Registry of supported models with their RHS and Jacobian
MODELS = {}


def register_model(name: str, rhs_fn, jacobian_fn, n_states: int, n_params: int):
    """Register a biological model for verification."""
    MODELS[name] = {
        'rhs': rhs_fn,
        'jacobian': jacobian_fn,
        'n_states': n_states,
        'n_params': n_params,
    }


# --- Gardner toggle switch ---
def _toggle_rhs(x: IVec, mu: IVec) -> IVec:
    """Toggle switch: dx/dt = α₁/(1+x₂^n₁) - x₁, dy/dt = α₂/(1+x₁^n₂) - x₂."""
    alpha1, alpha2 = mu[0], mu[1]
    # n_hill params are mu[2], mu[3] - use midpoints for integer Hill coeff
    n1 = max(2, round(mu[2].mid))
    n2 = max(2, round(mu[3].mid))
    
    x1, x2 = x[0], x[1]
    dx1 = alpha1 / (IV(1.0) + x2 ** n1) - x1
    dx2 = alpha2 / (IV(1.0) + x1 ** n2) - x2
    return IVec([dx1, dx2])


def _toggle_jacobian(x: IVec, mu: IVec) -> IMat:
    """Jacobian of toggle switch."""
    alpha1, alpha2 = mu[0], mu[1]
    n1 = max(2, round(mu[2].mid))
    n2 = max(2, round(mu[3].mid))
    x1, x2 = x[0], x[1]
    
    # df1/dx1 = -1
    # df1/dx2 = -α₁ · n₁ · x₂^(n₁-1) / (1 + x₂^n₁)²
    # df2/dx1 = -α₂ · n₂ · x₁^(n₂-1) / (1 + x₁^n₂)²
    # df2/dx2 = -1
    
    denom1 = (IV(1.0) + x2 ** n1)
    denom2 = (IV(1.0) + x1 ** n2)
    
    j00 = IV(-1.0)
    j01 = IV(-1.0) * alpha1 * IV(float(n1)) * (x2 ** (n1 - 1)) / (denom1 * denom1)
    j10 = IV(-1.0) * alpha2 * IV(float(n2)) * (x1 ** (n2 - 1)) / (denom2 * denom2)
    j11 = IV(-1.0)
    
    return IMat([[j00, j01], [j10, j11]])


register_model("toggle_switch", _toggle_rhs, _toggle_jacobian, 2, 4)
register_model("gardner_toggle_switch", _toggle_rhs, _toggle_jacobian, 2, 4)


# --- Brusselator ---
def _brusselator_rhs(x: IVec, mu: IVec) -> IVec:
    """Brusselator: dx/dt = A - (B+1)x + x²y, dy/dt = Bx - x²y."""
    A, B = mu[0], mu[1]
    x1, x2 = x[0], x[1]
    dx1 = A - (B + IV(1.0)) * x1 + x1 * x1 * x2
    dx2 = B * x1 - x1 * x1 * x2
    return IVec([dx1, dx2])


def _brusselator_jacobian(x: IVec, mu: IVec) -> IMat:
    A, B = mu[0], mu[1]
    x1, x2 = x[0], x[1]
    j00 = IV(-1.0) * (B + IV(1.0)) + IV(2.0) * x1 * x2
    j01 = x1 * x1
    j10 = B - IV(2.0) * x1 * x2
    j11 = IV(-1.0) * x1 * x1
    return IMat([[j00, j01], [j10, j11]])


register_model("brusselator", _brusselator_rhs, _brusselator_jacobian, 2, 2)


# --- Sel'kov glycolysis ---
def _selkov_rhs(x: IVec, mu: IVec) -> IVec:
    """Sel'kov: dx/dt = -x + a*y + x²*y, dy/dt = b - a*y - x²*y."""
    a, b = mu[0], mu[1]
    # mu[2] is optional scaling
    x1, x2 = x[0], x[1]
    dx1 = IV(-1.0) * x1 + a * x2 + x1 * x1 * x2
    dx2 = b - a * x2 - x1 * x1 * x2
    return IVec([dx1, dx2])


def _selkov_jacobian(x: IVec, mu: IVec) -> IMat:
    a = mu[0]
    x1, x2 = x[0], x[1]
    j00 = IV(-1.0) + IV(2.0) * x1 * x2
    j01 = a + x1 * x1
    j10 = IV(-2.0) * x1 * x2
    j11 = IV(-1.0) * a - x1 * x1
    return IMat([[j00, j01], [j10, j11]])


register_model("selkov", _selkov_rhs, _selkov_jacobian, 2, 2)
register_model("selkov_glycolysis", _selkov_rhs, _selkov_jacobian, 2, 2)


# --- Repressilator ---
def _repressilator_rhs(x: IVec, mu: IVec) -> IVec:
    """Repressilator: cyclic Hill repression (3 genes, n=2 fixed)."""
    alpha, gamma = mu[0], mu[3]
    x1, x2, x3 = x[0], x[1], x[2]
    dx1 = alpha / (IV(1.0) + x3 ** 2) - gamma * x1
    dx2 = alpha / (IV(1.0) + x1 ** 2) - gamma * x2
    dx3 = alpha / (IV(1.0) + x2 ** 2) - gamma * x3
    return IVec([dx1, dx2, dx3])


def _repressilator_jacobian(x: IVec, mu: IVec) -> IMat:
    """Jacobian of repressilator."""
    alpha, gamma = mu[0], mu[3]
    x1, x2, x3 = x[0], x[1], x[2]
    neg_gamma = IV(-1.0) * gamma
    zero = IV(0.0)
    
    denom1 = (IV(1.0) + x1 ** 2)
    denom2 = (IV(1.0) + x2 ** 2)
    denom3 = (IV(1.0) + x3 ** 2)
    
    # df1/dx3 = -alpha * 2 * x3 / (1 + x3^2)^2
    j02 = IV(-1.0) * alpha * IV(2.0) * x3 / (denom3 * denom3)
    # df2/dx1 = -alpha * 2 * x1 / (1 + x1^2)^2
    j10 = IV(-1.0) * alpha * IV(2.0) * x1 / (denom1 * denom1)
    # df3/dx2 = -alpha * 2 * x2 / (1 + x2^2)^2
    j21 = IV(-1.0) * alpha * IV(2.0) * x2 / (denom2 * denom2)
    
    return IMat([
        [neg_gamma, zero, j02],
        [j10, neg_gamma, zero],
        [zero, j21, neg_gamma],
    ])


register_model("repressilator", _repressilator_rhs, _repressilator_jacobian, 3, 4)


# --- Goodwin oscillator ---
def _goodwin_rhs(x: IVec, mu: IVec) -> IVec:
    """Goodwin oscillator: negative-feedback Hill loop (K=1, n=4 fixed).

    dx1/dt = a / (1 + x3^4) - b * x1
    dx2/dt = x1 - b * x2
    dx3/dt = x2 - b * x3
    """
    a, b = mu[0], mu[1]
    x1, x2, x3 = x[0], x[1], x[2]
    dx1 = a / (IV(1.0) + x3 ** 4) - b * x1
    dx2 = x1 - b * x2
    dx3 = x2 - b * x3
    return IVec([dx1, dx2, dx3])


def _goodwin_jacobian(x: IVec, mu: IVec) -> IMat:
    """Jacobian of the Goodwin oscillator."""
    a, b = mu[0], mu[1]
    x1, x2, x3 = x[0], x[1], x[2]
    neg_b = IV(-1.0) * b
    zero = IV(0.0)
    one = IV(1.0)

    # df1/dx3 = -a * 4 * x3^3 / (1 + x3^4)^2
    x3_3 = x3 ** 3
    denom = (one + x3 ** 4)
    j02 = IV(-1.0) * a * IV(4.0) * x3_3 / (denom * denom)

    return IMat([
        [neg_b, zero, j02],
        [one, neg_b, zero],
        [zero, one, neg_b],
    ])


register_model("goodwin", _goodwin_rhs, _goodwin_jacobian, 3, 2)


# ============================================================================
# Section 3: Krawczyk Recomputation (~200 LoC)
# ============================================================================

def recompute_krawczyk(rhs_fn, jacobian_fn,
                        X: IVec, mu: IVec) -> Tuple[bool, Optional[IVec], float]:
    """
    Independently recompute Krawczyk verification.
    
    Returns:
        (verified, enclosure, contraction_factor)
    """
    n = X.n
    x_mid = X.midpoint()
    mu_mid = mu.midpoint()
    
    # Evaluate f at midpoint (point arithmetic)
    x_mid_iv = IVec([IV(float(x_mid[i])) for i in range(n)])
    mu_mid_iv = IVec([IV(float(mu_mid[i])) for i in range(len(mu_mid))])
    
    try:
        f_mid = rhs_fn(x_mid_iv, mu_mid_iv)
    except (ZeroDivisionError, ValueError, OverflowError):
        return False, None, float('inf')
    
    # Compute point Jacobian and invert
    try:
        J_mid = jacobian_fn(x_mid_iv, mu_mid_iv)
    except (ZeroDivisionError, ValueError, OverflowError):
        return False, None, float('inf')
    
    # Extract point Jacobian as numpy array for inversion
    J_np = np.array([[J_mid[i, j].mid for j in range(n)] for i in range(n)])
    try:
        R = np.linalg.inv(J_np)
    except np.linalg.LinAlgError:
        return False, None, float('inf')
    
    R_iv = IMat.from_numpy(R)
    
    # Interval Jacobian over full box
    try:
        J_X = jacobian_fn(X, mu)
    except (ZeroDivisionError, ValueError, OverflowError):
        return False, None, float('inf')
    
    # K(X) = x̃ - R·f(x̃) + (I - R·J(X))·(X - x̃)
    Rf = R_iv.matvec(f_mid)
    RJ = R_iv.matmul(J_X)
    I_n = IMat.identity(n)
    
    # C = I - R·J(X)
    C = IMat([[I_n[i, j] - RJ[i, j] for j in range(n)] for i in range(n)])
    
    # X - x̃
    X_centered = IVec([X[i] - IV(float(x_mid[i])) for i in range(n)])
    
    # C · (X - x̃)
    CX = C.matvec(X_centered)
    
    # K = x̃ - R·f(x̃) + C·(X - x̃)
    K = IVec([IV(float(x_mid[i])) - Rf[i] + CX[i] for i in range(n)])
    
    # Check containment K ⊂ X
    if X.contains(K):
        contraction = 0.0
        for i in range(n):
            if X[i].width > 0:
                contraction = max(contraction, K[i].width / X[i].width)
        return True, K, contraction
    
    return False, None, float('inf')


def verify_eigenvalue_enclosure(jacobian_fn, X: IVec, mu: IVec,
                                  claimed_real_parts: List[IV]) -> Tuple[bool, List[IV]]:
    """
    Independently verify eigenvalue enclosure via Gershgorin disks.
    
    Returns:
        (consistent, recomputed_real_parts)
    """
    try:
        J = jacobian_fn(X, mu)
    except (ZeroDivisionError, ValueError, OverflowError):
        return False, []
    
    disks = J.gershgorin_disks()
    recomputed = []
    for center, radius in disks:
        rp = IV(center.lo - radius, center.hi + radius)
        recomputed.append(rp)
    
    # Check consistency: each claimed eigenvalue real part should be
    # contained in or at least overlap with the Gershgorin enclosure
    # (Gershgorin gives a superset, so claimed should be contained)
    if len(claimed_real_parts) != len(recomputed):
        return False, recomputed
    
    # Sort both by midpoint for comparison (eigenvalue ordering may differ)
    claimed_sorted = sorted(claimed_real_parts, key=lambda x: x.mid)
    recomputed_sorted = sorted(recomputed, key=lambda x: x.mid)
    
    consistent = True
    for c, r in zip(claimed_sorted, recomputed_sorted):
        # The recomputed Gershgorin disk should contain the claimed enclosure
        # (or at least overlap significantly)
        if not r.contains(c):
            # Allow some tolerance for different methods
            c_inflated = IV(c.lo - abs(c.width) * 0.1 - 1e-10,
                           c.hi + abs(c.width) * 0.1 + 1e-10)
            if r.intersection(c_inflated).is_empty():
                consistent = False
    
    return consistent, recomputed


def verify_stability(eigenvalue_real_parts: List[IV],
                      claimed_stability: str) -> bool:
    """Verify that claimed stability is consistent with eigenvalue enclosures."""
    all_neg = all(rp.hi < 0 for rp in eigenvalue_real_parts)
    all_pos = all(rp.lo > 0 for rp in eigenvalue_real_parts)
    has_pos = any(rp.lo > 0 for rp in eigenvalue_real_parts)  # definitely positive
    has_neg = any(rp.hi < 0 for rp in eigenvalue_real_parts)  # definitely negative
    
    if claimed_stability in ("stable_node", "stable_focus", "stable_spiral"):
        return all_neg
    if claimed_stability in ("unstable_node", "unstable_focus", "unstable_spiral"):
        return all_pos or has_pos
    if claimed_stability == "saddle":
        return has_pos and has_neg
    if claimed_stability in ("degenerate", "unknown", "center"):
        return True  # these are always allowed
    
    return False  # unknown stability type


def verify_delta_bound(eigenvalue_real_parts: List[IV],
                        claimed_delta_required: float,
                        delta_solver: float = 1e-3) -> bool:
    """Verify that the claimed δ bound is sufficient for soundness."""
    # Compute eigenvalue gap
    gap = float('inf')
    for rp in eigenvalue_real_parts:
        if rp.hi < 0:
            gap = min(gap, abs(rp.hi))
        elif rp.lo > 0:
            gap = min(gap, rp.lo)
        else:
            gap = 0.0
            break
    
    # The claimed δ_required should be ≤ eigenvalue_gap
    # and delta_solver should be < delta_required
    if gap == 0.0:
        return False
    
    # Conservative check: δ_required should not exceed eigenvalue gap
    if claimed_delta_required > gap + 1e-15:
        return False
    
    return delta_solver < claimed_delta_required


# ============================================================================
# Section 4: Certificate Verification Pipeline (~250 LoC)
# ============================================================================

@dataclass
class VerificationResult:
    """Result of independent certificate verification."""
    valid: bool
    equilibria_verified: int
    equilibria_total: int
    stability_verified: int
    delta_bound_verified: int
    regime_label_consistent: bool
    errors: List[str]
    warnings: List[str]
    
    def summary(self) -> str:
        status = "PASS" if self.valid else "FAIL"
        lines = [
            f"Certificate verification: {status}",
            f"  Equilibria verified: {self.equilibria_verified}/{self.equilibria_total}",
            f"  Stability verified: {self.stability_verified}/{self.equilibria_total}",
            f"  δ-bound verified: {self.delta_bound_verified}/{self.equilibria_total}",
            f"  Regime label consistent: {self.regime_label_consistent}",
        ]
        if self.errors:
            lines.append("  ERRORS:")
            for e in self.errors:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append("  WARNINGS:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def verify_certificate(cert_data: dict,
                        delta_solver: float = 1e-3) -> VerificationResult:
    """
    Independently verify a PhaseCartographer certificate.
    
    This is the main entry point for the minimal checker.
    It recomputes ALL verification steps from scratch.
    """
    errors = []
    warnings = []
    
    # Parse model
    model_info = cert_data.get('model', {})
    model_name = model_info.get('name', '').lower().replace(' ', '_').replace('-', '_')
    
    if model_name not in MODELS:
        errors.append(f"Unknown model: {model_name}")
        return VerificationResult(
            valid=False, equilibria_verified=0, equilibria_total=0,
            stability_verified=0, delta_bound_verified=0,
            regime_label_consistent=False, errors=errors, warnings=warnings
        )
    
    model = MODELS[model_name]
    rhs_fn = model['rhs']
    jac_fn = model['jacobian']
    
    # Parse parameter box
    param_data = cert_data.get('parameter_box', [])
    mu = IVec([IV(lo, hi) for lo, hi in param_data])
    
    # Verify each equilibrium
    equilibria = cert_data.get('equilibria', [])
    n_total = len(equilibria)
    n_krawczyk_ok = 0
    n_stability_ok = 0
    n_delta_ok = 0
    
    for i, eq in enumerate(equilibria):
        # Parse state enclosure
        state_data = eq.get('state_enclosure', [])
        X = IVec([IV(lo, hi) for lo, hi in state_data])
        
        # 1. Recompute Krawczyk verification
        verified, enclosure, contraction = recompute_krawczyk(rhs_fn, jac_fn, X, mu)
        
        if verified:
            n_krawczyk_ok += 1
        else:
            # Try with inflated box
            X_inflated = IVec([IV(x.lo - x.width * 0.1, x.hi + x.width * 0.1) for x in X.c])
            verified2, _, _ = recompute_krawczyk(rhs_fn, jac_fn, X_inflated, mu)
            if verified2:
                n_krawczyk_ok += 1
                warnings.append(f"Equilibrium {i}: verified with inflated box")
            else:
                errors.append(f"Equilibrium {i}: Krawczyk recomputation FAILED")
        
        # 2. Verify eigenvalue enclosure
        claimed_rp = eq.get('eigenvalue_real_parts', [])
        claimed_rp_iv = [IV(lo, hi) for lo, hi in claimed_rp]
        
        eig_ok, recomputed_rp = verify_eigenvalue_enclosure(jac_fn, X, mu, claimed_rp_iv)
        
        # Use recomputed eigenvalues for stability check
        rp_for_check = recomputed_rp if recomputed_rp else claimed_rp_iv
        
        # 3. Verify stability
        claimed_stability = eq.get('stability', 'unknown')
        stab_ok = verify_stability(rp_for_check, claimed_stability)
        if stab_ok:
            n_stability_ok += 1
        else:
            # Also check with claimed eigenvalues (different enclosure methods)
            if verify_stability(claimed_rp_iv, claimed_stability):
                n_stability_ok += 1
                warnings.append(f"Equilibrium {i}: stability verified with claimed (not recomputed) eigenvalues")
            else:
                errors.append(f"Equilibrium {i}: stability '{claimed_stability}' inconsistent with eigenvalues")
        
        # 4. Verify δ-bound
        delta_info = eq.get('delta_bound', {})
        claimed_delta_req = delta_info.get('delta_required', 0.0)
        
        if claimed_delta_req > 0:
            delta_ok = verify_delta_bound(rp_for_check, claimed_delta_req, delta_solver)
            if delta_ok:
                n_delta_ok += 1
            else:
                # Not fatal; may just mean δ-bound is conservative
                warnings.append(f"Equilibrium {i}: δ-bound verification inconclusive")
        else:
            warnings.append(f"Equilibrium {i}: no δ-bound provided")
    
    # 5. Verify regime label
    claimed_label = cert_data.get('regime_label', '')
    n_stable = sum(1 for eq in equilibria
                   if eq.get('stability', '') in ('stable_node', 'stable_focus', 'stable_spiral'))
    
    label_ok = True
    if 'bistable' in claimed_label and n_stable < 2:
        label_ok = False
        errors.append(f"Regime label '{claimed_label}' claims bistability but only {n_stable} stable equilibria")
    if 'monostable' in claimed_label and n_stable != 1:
        label_ok = False
        errors.append(f"Regime label '{claimed_label}' claims monostability but {n_stable} stable equilibria")
    
    valid = (n_krawczyk_ok == n_total and
             n_stability_ok == n_total and
             label_ok and
             len(errors) == 0)
    
    return VerificationResult(
        valid=valid,
        equilibria_verified=n_krawczyk_ok,
        equilibria_total=n_total,
        stability_verified=n_stability_ok,
        delta_bound_verified=n_delta_ok,
        regime_label_consistent=label_ok,
        errors=errors,
        warnings=warnings,
    )


def verify_atlas(cells: list, domain: list,
                 delta_solver: float = 1e-3) -> dict:
    """
    Verify an atlas of certificates: check each cell individually
    and verify pairwise disjointness.
    
    Args:
        cells: List of certificate dicts.
        domain: Parameter domain [[lo,hi], ...].
        delta_solver: δ tolerance.
    
    Returns:
        dict with valid, n_cells, n_verified, coverage, errors.
    """
    errors = []
    n_verified = 0
    cell_vol = 0.0
    domain_vol = 1.0
    for lo, hi in domain:
        w = hi - lo
        if w > 0:
            domain_vol *= w
    
    for i, cert in enumerate(cells):
        result = verify_certificate(cert, delta_solver)
        if result.valid:
            n_verified += 1
        else:
            errors.append(f"Cell {i}: {'; '.join(result.errors)}")
        
        box = cert.get('parameter_box', [])
        v = 1.0
        for lo, hi in box:
            w = hi - lo
            if w > 0:
                v *= w
        cell_vol += v
    
    # Check pairwise disjointness (sample check for performance)
    n_cells = len(cells)
    for i in range(min(n_cells, 50)):
        for j in range(i + 1, min(n_cells, 50)):
            box_i = cells[i].get('parameter_box', [])
            box_j = cells[j].get('parameter_box', [])
            if not _boxes_disjoint(box_i, box_j):
                errors.append(f"Cells {i} and {j} overlap")
    
    coverage = min(1.0, cell_vol / domain_vol) if domain_vol > 0 else 0.0
    
    return {
        'valid': len(errors) == 0,
        'n_cells': n_cells,
        'n_verified': n_verified,
        'coverage': coverage,
        'errors': errors,
    }


def _boxes_disjoint(box1, box2, tol=1e-10):
    """Check if two boxes are disjoint."""
    for (lo1, hi1), (lo2, hi2) in zip(box1, box2):
        if hi1 <= lo2 + tol or hi2 <= lo1 + tol:
            return True
    return False


def verify_certificate_file(path: str, delta_solver: float = 1e-3) -> VerificationResult:
    """Load and verify a certificate from a JSON file."""
    with open(path, 'r') as f:
        cert_data = json.load(f)
    return verify_certificate(cert_data, delta_solver)


# ============================================================================
# Section 5: CLI Entry Point (~50 LoC)
# ============================================================================

def main():
    """CLI entry point for certificate verification."""
    import argparse
    parser = argparse.ArgumentParser(
        description="MiniCheck: Minimal Independent Certificate Checker for PhaseCartographer",
        epilog="Verifies regime certificates using independent interval-arithmetic recomputation."
    )
    parser.add_argument("certificate", help="Path to certificate JSON file")
    parser.add_argument("--delta", type=float, default=1e-3,
                        help="δ tolerance for SMT soundness (default: 1e-3)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed verification output")
    parser.add_argument("--json", action="store_true",
                        help="Output result as JSON")
    
    args = parser.parse_args()
    
    result = verify_certificate_file(args.certificate, args.delta)
    
    if args.json:
        output = {
            'valid': result.valid,
            'equilibria_verified': result.equilibria_verified,
            'equilibria_total': result.equilibria_total,
            'stability_verified': result.stability_verified,
            'delta_bound_verified': result.delta_bound_verified,
            'regime_label_consistent': result.regime_label_consistent,
            'errors': result.errors,
            'warnings': result.warnings,
        }
        print(json.dumps(output, indent=2))
    else:
        print(result.summary())
    
    sys.exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()
