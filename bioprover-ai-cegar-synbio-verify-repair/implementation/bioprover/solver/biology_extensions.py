"""
Biology-specific solver extensions.

Exploits structural properties of biological systems (monotonicity,
positivity, sparsity, conservation laws) to improve the efficiency
and tightness of validated ODE integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import linalg as la

from bioprover.solver.interval import (
    Interval,
    IntervalMatrix,
    IntervalVector,
    _round_down,
    _round_up,
)
from bioprover.solver.ode_integrator import (
    IntegratorConfig,
    IntegrationResult,
    ODEFunc,
    ODEFuncNumpy,
    ValidatedODEIntegrator,
)


# ---------------------------------------------------------------------------
# Hill function with interval arithmetic
# ---------------------------------------------------------------------------

def hill_function_interval(
    x: Interval,
    k: Interval,
    n: Interval,
    activation: bool = True,
) -> Interval:
    """
    Evaluate Hill function with interval arithmetic.

    Activation: H(x) = x^n / (k^n + x^n)
    Repression: H(x) = k^n / (k^n + x^n)

    All computations use outward rounding for rigorous enclosure.
    """
    if x.lo < 0:
        # Clamp to non-negative (biological concentrations)
        x = Interval(max(x.lo, 0.0), x.hi)

    # Compute x^n and k^n
    # For interval exponent, use exp(n * log(x))
    if x.lo > 0:
        x_n = (n * x.log()).exp()
    else:
        # x contains 0 – evaluate at endpoints
        x_lo_n = 0.0
        x_hi_n = float(x.hi) ** float(n.hi)  # upper bound
        x_n = Interval(
            _round_down(x_lo_n),
            _round_up(x_hi_n),
        )

    k_n = (n * k.log()).exp()
    denom = k_n + x_n

    if denom.lo <= 0:
        # Should not happen for positive k, x, n, but guard anyway
        denom = Interval(max(denom.lo, 1e-300), denom.hi)

    if activation:
        return x_n / denom
    else:
        return k_n / denom


def hill_activation(x: Interval, k: float, n: float) -> Interval:
    """Convenience: Hill activation with scalar parameters."""
    return hill_function_interval(x, Interval(k), Interval(n), activation=True)


def hill_repression(x: Interval, k: float, n: float) -> Interval:
    """Convenience: Hill repression with scalar parameters."""
    return hill_function_interval(x, Interval(k), Interval(n), activation=False)


# ---------------------------------------------------------------------------
# Monotone system solver
# ---------------------------------------------------------------------------

class MonotoneSystemSolver:
    """
    Exploits monotonicity for tighter enclosures.

    For cooperative (monotone) systems where df_i/dx_j >= 0 for i != j,
    the flow preserves the partial order. This means we only need to
    evaluate f at two vertices (inf and sup of the box) instead of
    doing full interval arithmetic.
    """

    def __init__(
        self,
        f: ODEFuncNumpy,
        n: int,
        sign_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            f: ODE right-hand side
            n: state dimension
            sign_matrix: n×n matrix where entry (i,j) is +1 if df_i/dx_j >= 0,
                         -1 if <= 0, 0 if unknown. If None, estimated numerically.
        """
        self._f = f
        self._n = n
        self._sign_matrix = sign_matrix

    def estimate_sign_matrix(self, x0: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Estimate the sign structure of the Jacobian via finite differences."""
        n = self._n
        signs = np.zeros((n, n))
        eps = 1e-7
        f0 = self._f(t, x0)
        for j in range(n):
            x_pert = x0.copy()
            x_pert[j] += eps
            f_pert = self._f(t, x_pert)
            for i in range(n):
                diff = f_pert[i] - f0[i]
                if abs(diff) < 1e-14:
                    signs[i, j] = 0
                elif diff > 0:
                    signs[i, j] = 1
                else:
                    signs[i, j] = -1
        self._sign_matrix = signs
        return signs

    def is_cooperative(self) -> bool:
        """Check if the system is cooperative (all off-diagonal entries >= 0)."""
        if self._sign_matrix is None:
            return False
        n = self._n
        for i in range(n):
            for j in range(n):
                if i != j and self._sign_matrix[i, j] < 0:
                    return False
        return True

    def enclosure_from_vertices(
        self, t: float, x_box: IntervalVector, h: float
    ) -> IntervalVector:
        """
        Compute enclosure by evaluating f only at the lower-left and
        upper-right corners (valid for cooperative systems).
        """
        x_lo = x_box.lo_array()
        x_hi = x_box.hi_array()

        f_lo = self._f(t, x_lo)
        f_hi = self._f(t, x_hi)

        result = []
        for i in range(self._n):
            new_lo = _round_down(x_lo[i] + h * min(f_lo[i], f_hi[i]))
            new_hi = _round_up(x_hi[i] + h * max(f_lo[i], f_hi[i]))
            result.append(Interval(new_lo, new_hi))
        return IntervalVector(result)


# ---------------------------------------------------------------------------
# GRN Jacobian sparsity exploitation
# ---------------------------------------------------------------------------

class GRNSparseSolver:
    """
    Exploits sparsity of gene regulatory network Jacobians.

    GRNs typically have sparse interaction graphs where each gene
    is regulated by only a few others. This sparsity can be exploited
    for faster Jacobian computation and matrix operations.
    """

    def __init__(
        self,
        f: ODEFuncNumpy,
        n: int,
        interaction_graph: Optional[Dict[int, Set[int]]] = None,
    ) -> None:
        """
        Args:
            f: ODE right-hand side
            n: state dimension
            interaction_graph: dict mapping gene i -> set of genes that regulate i.
                              If None, discovered via probing.
        """
        self._f = f
        self._n = n
        self._graph = interaction_graph

    def discover_sparsity(
        self, x0: np.ndarray, t: float = 0.0, threshold: float = 1e-10
    ) -> Dict[int, Set[int]]:
        """Discover interaction graph by probing the Jacobian."""
        n = self._n
        eps = 1e-7
        f0 = self._f(t, x0)
        graph: Dict[int, Set[int]] = {i: set() for i in range(n)}

        for j in range(n):
            x_pert = x0.copy()
            x_pert[j] += eps
            f_pert = self._f(t, x_pert)
            for i in range(n):
                if abs(f_pert[i] - f0[i]) > threshold * eps:
                    graph[i].add(j)

        self._graph = graph
        return graph

    def sparse_jacobian(self, t: float, x: np.ndarray) -> np.ndarray:
        """Compute Jacobian using only non-zero entries from the graph."""
        n = self._n
        jac = np.zeros((n, n))
        if self._graph is None:
            return jac
        eps = 1e-8
        f0 = self._f(t, x)
        # Only probe columns that appear in the graph
        cols_needed = set()
        for deps in self._graph.values():
            cols_needed.update(deps)
        for j in cols_needed:
            x_pert = x.copy()
            x_pert[j] += eps
            f_pert = self._f(t, x_pert)
            for i in range(n):
                if j in self._graph.get(i, set()):
                    jac[i, j] = (f_pert[i] - f0[i]) / eps
        return jac

    def sparse_interval_evaluation(
        self, t: Interval, x: IntervalVector, f_interval: ODEFunc
    ) -> IntervalVector:
        """
        Evaluate f with interval arithmetic exploiting sparsity.
        For each component i, only the variables in graph[i] contribute
        to the dependency, others can be evaluated at midpoint.
        """
        if self._graph is None:
            return f_interval(t, x)

        n = self._n
        result = []
        for i in range(n):
            deps = self._graph.get(i, set())
            # Create a mixed point/interval evaluation
            x_mixed = IntervalVector([
                x[j] if j in deps else Interval(x[j].mid())
                for j in range(n)
            ])
            fi = f_interval(t, x_mixed)
            result.append(fi[i])
        return IntervalVector(result)

    @property
    def sparsity_ratio(self) -> float:
        """Fraction of non-zero entries in the interaction graph."""
        if self._graph is None:
            return 1.0
        total = self._n * self._n
        nnz = sum(len(deps) for deps in self._graph.values())
        return nnz / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Positivity enforcement
# ---------------------------------------------------------------------------

class PositivityEnforcer:
    """
    Enforces non-negativity of concentrations during integration.

    Biological concentrations cannot be negative. This class provides
    methods to enforce this constraint in interval enclosures.
    """

    def __init__(self, positive_indices: Optional[List[int]] = None) -> None:
        """
        Args:
            positive_indices: indices of state variables that must be >= 0.
                            If None, all variables are assumed positive.
        """
        self._indices = positive_indices

    def enforce(self, x: IntervalVector) -> IntervalVector:
        """Clamp lower bounds to 0 for positive variables."""
        n = x.dim
        indices = self._indices if self._indices is not None else list(range(n))
        intervals = [x[i] for i in range(n)]
        for i in indices:
            if intervals[i].lo < 0:
                intervals[i] = Interval(0.0, max(intervals[i].hi, 0.0))
        return IntervalVector(intervals)

    def is_satisfied(self, x: IntervalVector) -> bool:
        """Check if all positive variables are non-negative."""
        n = x.dim
        indices = self._indices if self._indices is not None else list(range(n))
        return all(x[i].lo >= 0 for i in indices)


# ---------------------------------------------------------------------------
# Conservation law exploitation
# ---------------------------------------------------------------------------

class ConservationLawReducer:
    """
    Exploits linear conservation laws to reduce system dimension.

    If C*x = c (constant) for some matrix C, we can eliminate
    some variables and integrate a reduced system.
    """

    def __init__(
        self,
        conservation_vectors: List[np.ndarray],
        conservation_values: List[float],
    ) -> None:
        """
        Args:
            conservation_vectors: list of row vectors c_i such that c_i . x = const
            conservation_values: the constant values
        """
        self._vectors = conservation_vectors
        self._values = conservation_values
        self._n_laws = len(conservation_vectors)

    @classmethod
    def from_stoichiometry(cls, S: np.ndarray) -> "ConservationLawReducer":
        """
        Discover conservation laws from stoichiometry matrix S.
        Conservation laws are in the left null space of S.
        """
        # Left null space: rows of V where S^T V^T = 0
        U, sigma, Vt = np.linalg.svd(S.T)
        rank = np.sum(sigma > 1e-10)
        null_space = Vt[rank:]  # rows of Vt beyond the rank

        vectors = []
        values = []
        for row in null_space:
            if np.max(np.abs(row)) < 1e-12:
                continue
            # Normalize
            row = row / np.max(np.abs(row))
            vectors.append(row)
            values.append(0.0)  # Will be set from initial conditions

        return cls(vectors, values)

    def set_values_from_state(self, x0: np.ndarray) -> None:
        """Set conservation values from initial state."""
        for i, vec in enumerate(self._vectors):
            self._values[i] = float(np.dot(vec, x0))

    def reduce_dim(self, x: IntervalVector) -> Tuple[IntervalVector, List[int]]:
        """
        Eliminate variables using conservation laws.
        Returns (reduced state, indices of kept variables).
        """
        n = x.dim
        eliminated: Set[int] = set()
        kept: List[int] = []

        for vec in self._vectors:
            # Find the variable with largest coefficient to eliminate
            free_indices = [j for j in range(n) if j not in eliminated]
            if not free_indices:
                break
            best_j = max(free_indices, key=lambda j: abs(vec[j]))
            if abs(vec[best_j]) < 1e-12:
                continue
            eliminated.add(best_j)

        kept = [j for j in range(n) if j not in eliminated]
        reduced = x.project(kept)
        return reduced, kept

    def reconstruct(
        self, x_reduced: IntervalVector, kept_indices: List[int], full_dim: int
    ) -> IntervalVector:
        """Reconstruct full state from reduced state using conservation laws."""
        n = full_dim
        eliminated = [j for j in range(n) if j not in kept_indices]

        # Start with reduced values in place
        intervals = [Interval(0.0)] * n
        for idx, ki in enumerate(kept_indices):
            intervals[ki] = x_reduced[idx]

        # Solve for eliminated variables using conservation laws
        for law_idx, vec in enumerate(self._vectors):
            c = self._values[law_idx]
            for ei in eliminated:
                if abs(vec[ei]) < 1e-12:
                    continue
                # x_ei = (c - sum_{j != ei} vec[j] * x[j]) / vec[ei]
                known_sum = Interval(0.0)
                for j in range(n):
                    if j != ei and abs(vec[j]) > 1e-12:
                        known_sum = known_sum + Interval(vec[j]) * intervals[j]
                intervals[ei] = (Interval(c) - known_sum) / Interval(vec[ei])

        return IntervalVector(intervals)

    @property
    def n_laws(self) -> int:
        return self._n_laws


# ---------------------------------------------------------------------------
# Steady-state detection
# ---------------------------------------------------------------------------

class SteadyStateDetector:
    """
    Detects convergence to a steady state during integration.

    A steady state is detected when the derivative magnitude drops
    below a threshold and the enclosure width is small.
    """

    def __init__(
        self,
        f: ODEFuncNumpy,
        derivative_tol: float = 1e-8,
        width_tol: float = 1e-6,
        n_confirm: int = 5,
    ) -> None:
        self._f = f
        self._deriv_tol = derivative_tol
        self._width_tol = width_tol
        self._n_confirm = n_confirm
        self._confirm_count = 0
        self._steady_state: Optional[IntervalVector] = None

    def check(self, t: float, x: IntervalVector) -> bool:
        """Check if the system has reached a steady state."""
        x_mid = x.midpoint()
        dx = self._f(t, x_mid)
        deriv_mag = np.max(np.abs(dx))
        enc_width = x.max_width()

        if deriv_mag < self._deriv_tol and enc_width < self._width_tol:
            self._confirm_count += 1
            if self._confirm_count >= self._n_confirm:
                self._steady_state = x.copy()
                return True
        else:
            self._confirm_count = 0
        return False

    @property
    def steady_state(self) -> Optional[IntervalVector]:
        return self._steady_state

    def reset(self) -> None:
        self._confirm_count = 0
        self._steady_state = None


# ---------------------------------------------------------------------------
# Contraction detection
# ---------------------------------------------------------------------------

class ContractionDetector:
    """
    Detects contraction in the dynamics via Jacobian eigenvalue analysis.

    If all eigenvalues of the Jacobian have negative real parts,
    the system is locally contracting and enclosures should shrink.
    """

    def __init__(self, f: ODEFuncNumpy, n: int) -> None:
        self._f = f
        self._n = n

    def jacobian(self, t: float, x: np.ndarray) -> np.ndarray:
        """Compute numerical Jacobian."""
        n = self._n
        jac = np.zeros((n, n))
        eps = 1e-8
        f0 = self._f(t, x)
        for j in range(n):
            x_pert = x.copy()
            x_pert[j] += eps
            f_pert = self._f(t, x_pert)
            jac[:, j] = (f_pert - f0) / eps
        return jac

    def is_contracting(self, t: float, x: np.ndarray) -> bool:
        """Check if all Jacobian eigenvalues have negative real parts."""
        jac = self.jacobian(t, x)
        eigenvalues = np.linalg.eigvals(jac)
        return bool(np.all(np.real(eigenvalues) < 0))

    def contraction_rate(self, t: float, x: np.ndarray) -> float:
        """
        Estimate the contraction rate (largest real part of eigenvalues).
        Negative means contracting.
        """
        jac = self.jacobian(t, x)
        eigenvalues = np.linalg.eigvals(jac)
        return float(np.max(np.real(eigenvalues)))

    def logarithmic_norm(self, t: float, x: np.ndarray) -> float:
        """
        Compute the logarithmic norm (matrix measure) mu(J).
        mu(J) < 0 implies contraction. Uses the infinity norm.
        """
        jac = self.jacobian(t, x)
        n = self._n
        mu = -np.inf
        for i in range(n):
            row_sum = jac[i, i] + sum(abs(jac[i, j]) for j in range(n) if j != i)
            mu = max(mu, row_sum)
        return float(mu)

    def interval_contraction_rate(
        self, t: Interval, x: IntervalVector
    ) -> Interval:
        """
        Bound the contraction rate over an interval box
        using the interval Jacobian's spectral radius bound.
        """
        # Evaluate Jacobian at several points and bound
        x_mid = x.midpoint()
        jac_mid = self.jacobian(t.mid(), x_mid)
        eigs = np.linalg.eigvals(jac_mid)
        max_real = float(np.max(np.real(eigs)))

        # Add uncertainty from box width
        eps_jac = 0.0
        n = self._n
        eps = 1e-7
        for j in range(n):
            if x[j].width() > 0:
                x_lo = x.lo_array()
                x_hi = x.hi_array()
                jac_lo = self.jacobian(t.mid(), x_lo)
                jac_hi = self.jacobian(t.mid(), x_hi)
                eps_jac = max(eps_jac, np.max(np.abs(jac_hi - jac_lo)))

        return Interval(
            _round_down(max_real - eps_jac),
            _round_up(max_real + eps_jac),
        )


# ---------------------------------------------------------------------------
# Adaptive precision controller
# ---------------------------------------------------------------------------

class AdaptivePrecisionController:
    """
    Adapts integration precision based on local dynamics.

    Near bifurcations or sensitive regions, uses tighter tolerances
    and smaller step sizes. In smooth, contracting regions, allows
    coarser integration.
    """

    def __init__(
        self,
        f: ODEFuncNumpy,
        n: int,
        base_config: Optional[IntegratorConfig] = None,
    ) -> None:
        self._f = f
        self._n = n
        self._base_config = base_config or IntegratorConfig()
        self._contraction = ContractionDetector(f, n)
        self._sensitivity_history: List[float] = []

    def adapted_config(
        self, t: float, x: IntervalVector
    ) -> IntegratorConfig:
        """Return an IntegratorConfig adapted to the local dynamics."""
        x_mid = x.midpoint()
        config = IntegratorConfig(
            method=self._base_config.method,
            taylor_order=self._base_config.taylor_order,
            initial_step=self._base_config.initial_step,
            min_step=self._base_config.min_step,
            max_step=self._base_config.max_step,
            target_width=self._base_config.target_width,
            max_steps=self._base_config.max_steps,
            adaptive=True,
            use_qr=self._base_config.use_qr,
        )

        # Check local dynamics
        rate = self._contraction.contraction_rate(t, x_mid)
        self._sensitivity_history.append(abs(rate))

        if rate > -0.01:
            # Near marginally stable / bifurcation – tighten
            config.taylor_order = max(self._base_config.taylor_order + 2, 6)
            config.target_width = self._base_config.target_width * 0.1
            config.max_step = self._base_config.max_step * 0.1
            config.initial_step = min(config.initial_step, config.max_step)
        elif rate < -10.0:
            # Strongly contracting – can be coarser
            config.max_step = min(self._base_config.max_step * 2.0, 1.0)
            config.target_width = self._base_config.target_width * 10.0
        # else: moderate contraction, use base config

        return config

    def is_near_bifurcation(self, t: float, x: np.ndarray, threshold: float = 0.01) -> bool:
        """Check if the system is near a bifurcation (eigenvalue near imaginary axis)."""
        jac = self._contraction.jacobian(t, x)
        eigenvalues = np.linalg.eigvals(jac)
        min_real_part = float(np.min(np.abs(np.real(eigenvalues))))
        return min_real_part < threshold

    @property
    def sensitivity_history(self) -> List[float]:
        return list(self._sensitivity_history)
