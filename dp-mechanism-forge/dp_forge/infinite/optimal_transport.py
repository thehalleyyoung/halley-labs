"""
Differential privacy as optimal transport.

Interprets the DP mechanism design problem through the lens of optimal
transport theory.  The key insight is that the DP constraint between two
mechanism distributions P_i and P_{i'} can be viewed as a constraint on
the cost of transporting mass from P_i to P_{i'}, where the cost encodes
the privacy requirement.

This module provides:

- Wasserstein distance computation between mechanism distributions.
- Transport plan extraction from LP dual variables.
- Cost matrix construction for the DP coupling interpretation.
- Earth mover's distance for comparing synthesised vs baseline mechanisms.

Classes
-------
- :class:`DPTransport` — Main transport computation class.
- :class:`TransportPlan` — Container for an optimal transport plan.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.optimize import linprog
from scipy.spatial.distance import cdist

from dp_forge.types import QuerySpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transport plan container
# ---------------------------------------------------------------------------


@dataclass
class TransportPlan:
    """Optimal transport plan between two distributions.

    Attributes:
        coupling: Transport matrix T[i, j] ≥ 0 giving mass moved from
            source bin i to target bin j.  Row sums = source distribution,
            column sums = target distribution.
        cost: Total transport cost Σ_{i,j} T[i,j] * C[i,j].
        source_grid: Grid points of the source distribution.
        target_grid: Grid points of the target distribution.
        cost_matrix: Cost matrix used, shape (m, n).
    """

    coupling: npt.NDArray[np.float64]
    cost: float
    source_grid: npt.NDArray[np.float64]
    target_grid: npt.NDArray[np.float64]
    cost_matrix: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        self.coupling = np.asarray(self.coupling, dtype=np.float64)
        self.cost_matrix = np.asarray(self.cost_matrix, dtype=np.float64)
        self.source_grid = np.asarray(self.source_grid, dtype=np.float64)
        self.target_grid = np.asarray(self.target_grid, dtype=np.float64)

    @property
    def source_marginal(self) -> npt.NDArray[np.float64]:
        """Source distribution (row sums of coupling)."""
        return self.coupling.sum(axis=1)

    @property
    def target_marginal(self) -> npt.NDArray[np.float64]:
        """Target distribution (column sums of coupling)."""
        return self.coupling.sum(axis=0)

    @property
    def sparsity(self) -> float:
        """Fraction of zero entries in the coupling matrix."""
        total = self.coupling.size
        if total == 0:
            return 1.0
        nnz = np.count_nonzero(self.coupling > 1e-15)
        return 1.0 - nnz / total

    def __repr__(self) -> str:
        m, n = self.coupling.shape
        return (
            f"TransportPlan(source={m}, target={n}, "
            f"cost={self.cost:.6f}, sparsity={self.sparsity:.1%})"
        )


# ---------------------------------------------------------------------------
# Cost matrix construction
# ---------------------------------------------------------------------------


def _ground_cost_matrix(
    source_grid: npt.NDArray[np.float64],
    target_grid: npt.NDArray[np.float64],
    p: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Build the ground cost matrix C[i,j] = |source[i] - target[j]|^p.

    Parameters
    ----------
    source_grid : array of shape (m,)
    target_grid : array of shape (n,)
    p : float
        Exponent for the cost.  p=1 gives Wasserstein-1, p=2 gives the
        squared cost for Wasserstein-2.

    Returns
    -------
    array of shape (m, n)
    """
    source_2d = source_grid.reshape(-1, 1)
    target_2d = target_grid.reshape(-1, 1)
    if p == 1.0:
        return cdist(source_2d, target_2d, metric="cityblock")
    elif p == 2.0:
        return cdist(source_2d, target_2d, metric="sqeuclidean")
    else:
        return cdist(source_2d, target_2d, metric="minkowski", p=p) ** p


def _dp_cost_matrix(
    grid: npt.NDArray[np.float64],
    epsilon: float,
) -> npt.NDArray[np.float64]:
    """Build a DP-aware cost matrix for the coupling interpretation.

    The cost encodes the privacy constraint: C[i,j] = max(0, log(T[i,j]) + ε)
    where the coupling T must satisfy the DP constraint.  For the transport
    interpretation, we use C[i,j] = max(0, |y_i - y_j| - ε) to penalise
    transport that moves mass farther than the privacy "budget" allows.

    Parameters
    ----------
    grid : array of shape (k,)
        Output grid points.
    epsilon : float
        Privacy parameter.

    Returns
    -------
    array of shape (k, k)
    """
    k = len(grid)
    diffs = np.abs(grid[:, np.newaxis] - grid[np.newaxis, :])
    # DP cost: movement within e^ε ratio is "free"; beyond is penalised
    cost = np.maximum(diffs - epsilon, 0.0)
    return cost


# ---------------------------------------------------------------------------
# LP-based optimal transport solver
# ---------------------------------------------------------------------------


def _solve_transport_lp(
    source: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    cost: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], float]:
    """Solve the optimal transport LP exactly.

    min  Σ_{i,j} C[i,j] T[i,j]
    s.t. Σ_j T[i,j] = source[i]   ∀ i
         Σ_i T[i,j] = target[j]   ∀ j
         T[i,j] ≥ 0               ∀ i,j

    Parameters
    ----------
    source : array of shape (m,)
        Source distribution.
    target : array of shape (n,)
        Target distribution.
    cost : array of shape (m, n)
        Cost matrix.

    Returns
    -------
    T : array of shape (m, n)
        Optimal transport coupling.
    total_cost : float
        Optimal transport cost.
    """
    m = len(source)
    n = len(target)

    # Normalise to ensure equal total mass
    s_sum = np.sum(source)
    t_sum = np.sum(target)
    if s_sum <= 0 or t_sum <= 0:
        return np.zeros((m, n)), 0.0

    source_norm = source / s_sum
    target_norm = target / t_sum

    # Flatten variables: T[i,j] → x[i*n + j]
    c_flat = cost.flatten()

    # Equality constraints
    # Row sums: for each i, Σ_j T[i,j] = source[i]
    A_eq_rows = []
    b_eq = []

    for i in range(m):
        row = np.zeros(m * n, dtype=np.float64)
        row[i * n:(i + 1) * n] = 1.0
        A_eq_rows.append(row)
        b_eq.append(source_norm[i])

    for j in range(n):
        row = np.zeros(m * n, dtype=np.float64)
        for i in range(m):
            row[i * n + j] = 1.0
        A_eq_rows.append(row)
        b_eq.append(target_norm[j])

    A_eq = np.array(A_eq_rows, dtype=np.float64)
    b_eq_arr = np.array(b_eq, dtype=np.float64)

    bounds = [(0.0, None)] * (m * n)

    try:
        result = linprog(
            c=c_flat,
            A_eq=A_eq,
            b_eq=b_eq_arr,
            bounds=bounds,
            method="highs",
            options={"maxiter": 50000},
        )
    except Exception:
        # Fallback: use Sinkhorn approximation
        return _sinkhorn_transport(source_norm, target_norm, cost)

    if not result.success:
        return _sinkhorn_transport(source_norm, target_norm, cost)

    T = result.x.reshape(m, n) * s_sum
    total_cost = float(result.fun) * s_sum
    return T, total_cost


def _sinkhorn_transport(
    source: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    cost: npt.NDArray[np.float64],
    reg: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> Tuple[npt.NDArray[np.float64], float]:
    """Sinkhorn algorithm for entropy-regularised optimal transport.

    Solves: min Σ C[i,j] T[i,j] + reg * Σ T[i,j] log(T[i,j])

    Parameters
    ----------
    source, target : arrays
        Normalised distributions.
    cost : array
        Cost matrix.
    reg : float
        Regularisation parameter.
    max_iter : int
        Maximum Sinkhorn iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    T : array
        Approximate coupling.
    cost_val : float
        Transport cost.
    """
    m = len(source)
    n = len(target)

    # Gibbs kernel
    K = np.exp(-cost / max(reg, 1e-30))
    K = np.maximum(K, 1e-300)

    u = np.ones(m, dtype=np.float64)
    v = np.ones(n, dtype=np.float64)

    for _ in range(max_iter):
        u_prev = u.copy()
        u = source / np.maximum(K @ v, 1e-300)
        v = target / np.maximum(K.T @ u, 1e-300)

        if np.max(np.abs(u - u_prev)) < tol:
            break

    T = u[:, np.newaxis] * K * v[np.newaxis, :]
    total_cost = float(np.sum(T * cost))
    return T, total_cost


# ---------------------------------------------------------------------------
# DPTransport class
# ---------------------------------------------------------------------------


class DPTransport:
    """Optimal transport tools for differential privacy mechanism analysis.

    Provides Wasserstein distance computation, transport plan extraction,
    and mechanism comparison utilities using the optimal transport framework.

    Parameters
    ----------
    p : float
        Order of the Wasserstein distance (1 or 2).
    use_sinkhorn : bool
        If True, use Sinkhorn regularisation (faster for large problems).
    sinkhorn_reg : float
        Regularisation parameter for Sinkhorn algorithm.
    """

    def __init__(
        self,
        p: float = 1.0,
        use_sinkhorn: bool = False,
        sinkhorn_reg: float = 0.01,
    ) -> None:
        if p < 1.0:
            raise ValueError(f"Wasserstein order p must be >= 1, got {p}")
        self._p = p
        self._use_sinkhorn = use_sinkhorn
        self._sinkhorn_reg = sinkhorn_reg

    @property
    def order(self) -> float:
        """Wasserstein distance order."""
        return self._p

    def wasserstein(
        self,
        source: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        source_grid: npt.NDArray[np.float64],
        target_grid: Optional[npt.NDArray[np.float64]] = None,
    ) -> float:
        """Compute the Wasserstein-p distance between two discrete distributions.

        Parameters
        ----------
        source : array of shape (m,)
            Source distribution weights.
        target : array of shape (n,)
            Target distribution weights.
        source_grid : array of shape (m,)
            Grid points for source distribution.
        target_grid : array of shape (n,), optional
            Grid points for target distribution.  If None, uses source_grid
            (assumes both distributions are on the same grid).

        Returns
        -------
        float
            Wasserstein-p distance.
        """
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        source_grid = np.asarray(source_grid, dtype=np.float64)

        if target_grid is None:
            target_grid = source_grid.copy()
        else:
            target_grid = np.asarray(target_grid, dtype=np.float64)

        # Normalise
        source = source / max(np.sum(source), 1e-30)
        target = target / max(np.sum(target), 1e-30)

        # For 1-D Wasserstein-1, use the CDF method (much faster)
        if self._p == 1.0 and np.array_equal(source_grid, target_grid):
            return self._wasserstein1_cdf(source, target, source_grid)

        cost = _ground_cost_matrix(source_grid, target_grid, p=self._p)

        if self._use_sinkhorn:
            _, total_cost = _sinkhorn_transport(
                source, target, cost,
                reg=self._sinkhorn_reg,
            )
        else:
            _, total_cost = _solve_transport_lp(source, target, cost)

        if self._p > 1.0:
            return total_cost ** (1.0 / self._p)
        return total_cost

    def _wasserstein1_cdf(
        self,
        source: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        grid: npt.NDArray[np.float64],
    ) -> float:
        """Compute W_1 using the CDF difference formula.

        W_1(P, Q) = ∫ |F_P(y) - F_Q(y)| dy
                   ≈ Σ_j |CDF_P(y_j) - CDF_Q(y_j)| * Δy_j
        """
        cdf_source = np.cumsum(source)
        cdf_target = np.cumsum(target)

        if len(grid) < 2:
            return 0.0

        bin_widths = np.diff(grid)
        # Use midpoint CDF values for trapezoidal rule
        cdf_diff = np.abs(cdf_source[:-1] - cdf_target[:-1])
        return float(np.sum(cdf_diff * bin_widths))

    def transport_plan(
        self,
        source: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        source_grid: npt.NDArray[np.float64],
        target_grid: Optional[npt.NDArray[np.float64]] = None,
    ) -> TransportPlan:
        """Compute the full optimal transport plan.

        Parameters
        ----------
        source, target : arrays
            Distribution weights.
        source_grid : array
            Source grid points.
        target_grid : array, optional
            Target grid points (defaults to source_grid).

        Returns
        -------
        TransportPlan
        """
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        source_grid = np.asarray(source_grid, dtype=np.float64)

        if target_grid is None:
            target_grid = source_grid.copy()
        else:
            target_grid = np.asarray(target_grid, dtype=np.float64)

        cost = _ground_cost_matrix(source_grid, target_grid, p=self._p)

        # Normalise
        s_norm = source / max(np.sum(source), 1e-30)
        t_norm = target / max(np.sum(target), 1e-30)

        if self._use_sinkhorn:
            T, total_cost = _sinkhorn_transport(
                s_norm, t_norm, cost, reg=self._sinkhorn_reg,
            )
        else:
            T, total_cost = _solve_transport_lp(s_norm, t_norm, cost)

        return TransportPlan(
            coupling=T,
            cost=total_cost,
            source_grid=source_grid,
            target_grid=target_grid,
            cost_matrix=cost,
        )

    def dp_transport_cost(
        self,
        mechanism: npt.NDArray[np.float64],
        y_grid: npt.NDArray[np.float64],
        spec: QuerySpec,
    ) -> npt.NDArray[np.float64]:
        """Compute DP transport cost for each adjacent pair.

        For each adjacent pair (i, i'), computes the transport cost between
        mechanism rows p[i] and p[i'] under the DP cost matrix.

        Parameters
        ----------
        mechanism : array of shape (n, k)
            Mechanism probability table.
        y_grid : array of shape (k,)
            Output grid.
        spec : QuerySpec
            Problem specification.

        Returns
        -------
        array of shape (n_edges,)
            Transport cost for each adjacent pair.
        """
        mechanism = np.asarray(mechanism, dtype=np.float64)
        y_grid = np.asarray(y_grid, dtype=np.float64)

        dp_cost = _dp_cost_matrix(y_grid, spec.epsilon)

        assert spec.edges is not None
        costs = []
        for i, ip in spec.edges.edges:
            if self._use_sinkhorn:
                _, cost_val = _sinkhorn_transport(
                    mechanism[i], mechanism[ip], dp_cost,
                    reg=self._sinkhorn_reg,
                )
            else:
                _, cost_val = _solve_transport_lp(
                    mechanism[i], mechanism[ip], dp_cost,
                )
            costs.append(cost_val)

        return np.array(costs, dtype=np.float64)

    def compare_mechanisms(
        self,
        mech_a: npt.NDArray[np.float64],
        mech_b: npt.NDArray[np.float64],
        grid_a: npt.NDArray[np.float64],
        grid_b: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """Compare two mechanisms row-by-row using Wasserstein distance.

        Computes W_p(mech_a[i], mech_b[i]) for each database i.

        Parameters
        ----------
        mech_a : array of shape (n, k_a)
            First mechanism.
        mech_b : array of shape (n, k_b)
            Second mechanism.
        grid_a : array of shape (k_a,)
            Grid for first mechanism.
        grid_b : array of shape (k_b,), optional
            Grid for second mechanism.

        Returns
        -------
        array of shape (n,)
            Per-row Wasserstein distances.
        """
        mech_a = np.asarray(mech_a, dtype=np.float64)
        mech_b = np.asarray(mech_b, dtype=np.float64)
        grid_a = np.asarray(grid_a, dtype=np.float64)
        if grid_b is not None:
            grid_b = np.asarray(grid_b, dtype=np.float64)

        if mech_a.shape[0] != mech_b.shape[0]:
            raise ValueError(
                f"Mechanisms must have same number of rows: "
                f"{mech_a.shape[0]} vs {mech_b.shape[0]}"
            )

        n = mech_a.shape[0]
        distances = np.zeros(n, dtype=np.float64)
        for i in range(n):
            distances[i] = self.wasserstein(
                mech_a[i], mech_b[i], grid_a,
                target_grid=grid_b,
            )
        return distances

    def earth_movers_distance(
        self,
        source: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        grid: npt.NDArray[np.float64],
    ) -> float:
        """Compute earth mover's distance (= W_1) between two distributions on the same grid.

        Parameters
        ----------
        source, target : arrays of shape (k,)
            Distributions on the same grid.
        grid : array of shape (k,)
            Grid points.

        Returns
        -------
        float
            Earth mover's distance.
        """
        return self.wasserstein(source, target, grid, target_grid=grid)

    def __repr__(self) -> str:
        method = "sinkhorn" if self._use_sinkhorn else "exact"
        return f"DPTransport(p={self._p}, method={method})"
