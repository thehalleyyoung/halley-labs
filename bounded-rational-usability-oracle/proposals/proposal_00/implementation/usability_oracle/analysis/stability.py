"""
usability_oracle.analysis.stability — Numerical stability analysis.

Analyses the numerical conditioning of the oracle's computations:
condition numbers of transition matrices, sensitivity of value iteration
to perturbations, floating-point error propagation bounds, and detection
of ill-conditioned parameter regimes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class StabilityResult:
    """Summary of numerical stability diagnostics.

    Attributes:
        stable: Whether the computation is considered numerically stable.
        condition_number: Condition number of the primary matrix.
        perturbation_sensitivity: Max relative output change per unit input change.
        error_bound: Estimated upper bound on accumulated floating-point error.
        eigenvalue_gap: Spectral gap of the transition matrix (larger = more stable).
        warnings: List of stability warnings.
    """
    stable: bool = True
    condition_number: float = 1.0
    perturbation_sensitivity: float = 0.0
    error_bound: float = 0.0
    eigenvalue_gap: float = 0.0
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "STABLE" if self.stable else "UNSTABLE"
        lines = [
            f"Stability: {status}",
            f"  Condition number: {self.condition_number:.2e}",
            f"  Perturbation sensitivity: {self.perturbation_sensitivity:.4f}",
            f"  Error bound: {self.error_bound:.2e}",
            f"  Eigenvalue gap: {self.eigenvalue_gap:.6f}",
        ]
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


@dataclass
class MatrixDiagnostics:
    """Detailed diagnostics for a single matrix."""
    shape: tuple[int, ...]
    rank: int
    condition_number: float
    spectral_radius: float
    is_stochastic: bool
    eigenvalue_gap: float
    min_nonzero: float
    max_entry: float
    sparsity: float


# ---------------------------------------------------------------------------
# Matrix analysis helpers
# ---------------------------------------------------------------------------

def _condition_number(M: np.ndarray) -> float:
    """Compute the 2-norm condition number of matrix M."""
    if M.ndim != 2 or M.shape[0] == 0 or M.shape[1] == 0:
        return 1.0
    sv = np.linalg.svd(M, compute_uv=False)
    if sv[-1] < 1e-15:
        return float("inf")
    return float(sv[0] / sv[-1])


def _spectral_gap(M: np.ndarray) -> float:
    """Compute the spectral gap of a stochastic matrix.

    The spectral gap is 1 - |lambda_2| where lambda_2 is the second-largest
    eigenvalue in absolute value.  A larger gap means faster mixing.
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return 0.0
    n = M.shape[0]
    if n <= 1:
        return 1.0

    eigenvalues = np.linalg.eigvals(M)
    mags = np.sort(np.abs(eigenvalues))[::-1]
    if len(mags) < 2:
        return 1.0
    return max(0.0, 1.0 - float(mags[1]))


def _is_stochastic(M: np.ndarray, tol: float = 1e-6) -> bool:
    """Check if a matrix is row-stochastic (rows sum to 1, non-negative)."""
    if M.ndim != 2:
        return False
    if np.any(M < -tol):
        return False
    row_sums = M.sum(axis=1)
    return bool(np.allclose(row_sums, 1.0, atol=tol))


def _matrix_diagnostics(M: np.ndarray) -> MatrixDiagnostics:
    """Compute comprehensive diagnostics for a matrix."""
    n_nonzero = np.count_nonzero(M)
    total = M.size
    sparsity = 1.0 - (n_nonzero / total) if total > 0 else 0.0
    nonzero_vals = np.abs(M[M != 0])
    min_nz = float(np.min(nonzero_vals)) if len(nonzero_vals) > 0 else 0.0

    eigenvalues = np.linalg.eigvals(M) if M.shape[0] == M.shape[1] else np.array([])
    spec_radius = float(np.max(np.abs(eigenvalues))) if len(eigenvalues) > 0 else 0.0

    return MatrixDiagnostics(
        shape=M.shape,
        rank=int(np.linalg.matrix_rank(M)),
        condition_number=_condition_number(M),
        spectral_radius=spec_radius,
        is_stochastic=_is_stochastic(M),
        eigenvalue_gap=_spectral_gap(M) if M.shape[0] == M.shape[1] else 0.0,
        min_nonzero=min_nz,
        max_entry=float(np.max(np.abs(M))) if M.size > 0 else 0.0,
        sparsity=sparsity,
    )


# ---------------------------------------------------------------------------
# Perturbation analysis
# ---------------------------------------------------------------------------

def _perturbation_sensitivity(
    fn: Callable[[np.ndarray], float],
    M: np.ndarray,
    n_perturbations: int = 50,
    epsilon: float = 1e-6,
    rng: np.random.RandomState | None = None,
) -> float:
    """Estimate perturbation sensitivity by measuring output change
    under small random perturbations of the input matrix.

    Returns the maximum relative output change per unit Frobenius-norm
    perturbation.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    base_output = fn(M)
    max_sensitivity = 0.0

    for _ in range(n_perturbations):
        # Random perturbation direction
        direction = rng.randn(*M.shape)
        direction /= max(np.linalg.norm(direction, 'fro'), 1e-15)

        perturbed = M + epsilon * direction
        # Re-normalise if stochastic
        if _is_stochastic(M):
            row_sums = perturbed.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 1e-15, row_sums, 1.0)
            perturbed = np.maximum(perturbed, 0.0) / row_sums

        try:
            perturbed_output = fn(perturbed)
            if abs(base_output) > 1e-12:
                rel_change = abs(perturbed_output - base_output) / abs(base_output)
            else:
                rel_change = abs(perturbed_output - base_output)
            sensitivity = rel_change / epsilon
            max_sensitivity = max(max_sensitivity, sensitivity)
        except Exception:
            continue

    return max_sensitivity


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

def _forward_error_bound(
    matrices: list[np.ndarray],
    machine_eps: float = np.finfo(float).eps,
) -> float:
    """Estimate the forward error bound for a sequence of matrix operations.

    Uses the rule: forward_error ≤ n * cond(A) * machine_eps for each
    matrix multiply, accumulated across the sequence.
    """
    total_bound = 0.0
    for M in matrices:
        n = max(M.shape)
        cond = _condition_number(M)
        if cond == float("inf"):
            return float("inf")
        total_bound += n * cond * machine_eps
    return total_bound


def _backward_error_estimate(
    A: np.ndarray,
    x: np.ndarray,
    b: np.ndarray,
) -> float:
    """Compute the normalised backward error for Ax = b.

    backward_error = ||Ax - b|| / (||A|| * ||x|| + ||b||)
    """
    residual = np.linalg.norm(A @ x - b)
    norm_A = np.linalg.norm(A)
    norm_x = np.linalg.norm(x)
    norm_b = np.linalg.norm(b)
    denom = norm_A * norm_x + norm_b
    if denom < 1e-15:
        return 0.0
    return float(residual / denom)


# ---------------------------------------------------------------------------
# Value iteration stability
# ---------------------------------------------------------------------------

def _value_iteration_stability(
    transition_matrix: np.ndarray,
    reward: np.ndarray,
    gamma: float = 0.99,
    n_perturbations: int = 20,
    epsilon: float = 1e-6,
    rng: np.random.RandomState | None = None,
) -> dict[str, float]:
    """Analyse stability of value iteration under reward perturbations.

    Returns:
        max_value_change: Maximum change in optimal value.
        mean_value_change: Mean change in optimal value.
        policy_change_rate: Fraction of states where greedy policy changes.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n_states = transition_matrix.shape[0]
    if n_states == 0:
        return {"max_value_change": 0.0, "mean_value_change": 0.0, "policy_change_rate": 0.0}

    # Base value iteration
    V_base = _simple_value_iteration(transition_matrix, reward, gamma)
    policy_base = _greedy_policy(transition_matrix, reward, V_base, gamma)

    max_change = 0.0
    total_change = 0.0
    policy_changes = 0

    for _ in range(n_perturbations):
        pert = reward + epsilon * rng.randn(*reward.shape)
        V_pert = _simple_value_iteration(transition_matrix, pert, gamma)
        policy_pert = _greedy_policy(transition_matrix, pert, V_pert, gamma)

        change = np.max(np.abs(V_base - V_pert))
        max_change = max(max_change, float(change))
        total_change += float(np.mean(np.abs(V_base - V_pert)))
        policy_changes += int(np.sum(policy_base != policy_pert))

    mean_change = total_change / max(n_perturbations, 1)
    policy_rate = policy_changes / max(n_perturbations * n_states, 1)

    return {
        "max_value_change": max_change,
        "mean_value_change": mean_change,
        "policy_change_rate": policy_rate,
    }


def _simple_value_iteration(
    P: np.ndarray,
    r: np.ndarray,
    gamma: float,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> np.ndarray:
    """Simple value iteration for a single-action MDP (Markov reward process)."""
    n = P.shape[0]
    V = np.zeros(n)
    for _ in range(max_iter):
        V_new = r + gamma * P @ V
        if np.max(np.abs(V_new - V)) < tol:
            return V_new
        V = V_new
    return V


def _greedy_policy(
    P: np.ndarray,
    r: np.ndarray,
    V: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Compute greedy policy (for single-action MRP, just argmax of Q)."""
    Q = r + gamma * P @ V
    return np.argmax(Q.reshape(-1, 1) if Q.ndim == 1 else Q, axis=-1)


# ---------------------------------------------------------------------------
# StabilityAnalyzer
# ---------------------------------------------------------------------------

class StabilityAnalyzer:
    """Analyse numerical stability of oracle computations.

    Parameters:
        condition_threshold: Maximum acceptable condition number.
        gap_threshold: Minimum acceptable spectral gap.
        sensitivity_threshold: Maximum acceptable perturbation sensitivity.
    """

    def __init__(
        self,
        condition_threshold: float = 1e10,
        gap_threshold: float = 0.01,
        sensitivity_threshold: float = 100.0,
    ) -> None:
        self._cond_thresh = condition_threshold
        self._gap_thresh = gap_threshold
        self._sens_thresh = sensitivity_threshold
        self._rng = np.random.RandomState(42)

    # ------------------------------------------------------------------
    # Full stability analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        transition_matrix: np.ndarray,
        reward: Optional[np.ndarray] = None,
        gamma: float = 0.99,
        output_fn: Optional[Callable[[np.ndarray], float]] = None,
    ) -> StabilityResult:
        """Run comprehensive stability analysis on an MDP transition matrix.

        Parameters:
            transition_matrix: P(s'|s, a) matrix.
            reward: Reward vector (optional, for VI stability).
            gamma: Discount factor.
            output_fn: Optional scalar function of the transition matrix
                for perturbation sensitivity analysis.
        """
        P = np.asarray(transition_matrix, dtype=float)
        warnings: list[str] = []
        details: dict[str, Any] = {}

        # Matrix diagnostics
        diag = _matrix_diagnostics(P)
        details["matrix_diagnostics"] = {
            "shape": diag.shape,
            "rank": diag.rank,
            "spectral_radius": diag.spectral_radius,
            "is_stochastic": diag.is_stochastic,
            "sparsity": diag.sparsity,
            "min_nonzero": diag.min_nonzero,
        }

        cond = diag.condition_number
        gap = diag.eigenvalue_gap

        if cond > self._cond_thresh:
            warnings.append(f"Condition number {cond:.2e} exceeds threshold {self._cond_thresh:.2e}")
        if cond == float("inf"):
            warnings.append("Matrix is singular or near-singular")

        if gap < self._gap_thresh:
            warnings.append(f"Spectral gap {gap:.6f} below threshold {self._gap_thresh}")

        if not diag.is_stochastic and P.shape[0] == P.shape[1]:
            warnings.append("Transition matrix is not row-stochastic")

        if diag.min_nonzero < 1e-10 and diag.min_nonzero > 0:
            warnings.append(f"Very small nonzero entries ({diag.min_nonzero:.2e}) may cause underflow")

        # Perturbation sensitivity
        sens = 0.0
        if output_fn is not None:
            sens = _perturbation_sensitivity(output_fn, P, rng=self._rng)
            details["perturbation_sensitivity"] = sens
            if sens > self._sens_thresh:
                warnings.append(f"Perturbation sensitivity {sens:.2f} exceeds threshold")

        # Forward error bound
        err_bound = _forward_error_bound([P])
        details["forward_error_bound"] = err_bound

        # Value iteration stability
        if reward is not None:
            r = np.asarray(reward, dtype=float)
            vi_stability = _value_iteration_stability(P, r, gamma, rng=self._rng)
            details["value_iteration"] = vi_stability
            if vi_stability["policy_change_rate"] > 0.1:
                warnings.append(
                    f"Policy changes under small perturbations ({vi_stability['policy_change_rate']:.2%})"
                )

        stable = len(warnings) == 0

        return StabilityResult(
            stable=stable,
            condition_number=cond,
            perturbation_sensitivity=sens,
            error_bound=err_bound,
            eigenvalue_gap=gap,
            warnings=warnings,
            details=details,
        )

    # ------------------------------------------------------------------
    # Matrix-specific diagnostics
    # ------------------------------------------------------------------

    def diagnose_matrix(self, M: np.ndarray) -> MatrixDiagnostics:
        """Run diagnostics on a single matrix."""
        return _matrix_diagnostics(np.asarray(M, dtype=float))

    # ------------------------------------------------------------------
    # Discount factor sensitivity
    # ------------------------------------------------------------------

    def discount_sensitivity(
        self,
        transition_matrix: np.ndarray,
        reward: np.ndarray,
        gamma_range: tuple[float, float] = (0.9, 0.999),
        n_points: int = 20,
    ) -> list[tuple[float, float, float]]:
        """Analyse how the value function changes with the discount factor.

        Returns list of (gamma, mean_value, condition_number) triples.
        """
        P = np.asarray(transition_matrix, dtype=float)
        r = np.asarray(reward, dtype=float)
        results = []

        for gamma in np.linspace(gamma_range[0], gamma_range[1], n_points):
            V = _simple_value_iteration(P, r, float(gamma))
            # Condition number of (I - gamma*P)
            n = P.shape[0]
            A = np.eye(n) - gamma * P
            cond = _condition_number(A)
            results.append((float(gamma), float(np.mean(V)), cond))

        return results

    # ------------------------------------------------------------------
    # Iterative method convergence rate
    # ------------------------------------------------------------------

    @staticmethod
    def convergence_rate(
        transition_matrix: np.ndarray,
        gamma: float = 0.99,
    ) -> float:
        """Estimate the asymptotic convergence rate of value iteration.

        The rate is gamma * |lambda_max(P)| where lambda_max is the
        largest eigenvalue magnitude of the transition matrix.
        """
        P = np.asarray(transition_matrix, dtype=float)
        if P.shape[0] == 0:
            return 0.0
        eigenvalues = np.linalg.eigvals(P)
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        return gamma * spectral_radius
