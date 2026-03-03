"""
LP construction engine for DP-Forge discrete mechanism synthesis.

This module is the computational heart of the CEGIS pipeline.  It encodes
the problem "find a discrete mechanism minimising worst-case loss subject to
(ε, δ)-differential privacy" as a sparse linear program and provides
incremental constraint management for the CEGIS loop.

Mathematical formulation
~~~~~~~~~~~~~~~~~~~~~~~~

**Decision variables.**  Given *n* databases with query outputs
``f(x_1), …, f(x_n)`` and a discretisation of the output range into *k*
bins ``y_1, …, y_k``, we introduce ``n·k`` probability variables

    ``p[i][j] = Pr[M(x_i) = y_j]``           (i ∈ [n], j ∈ [k])

plus one epigraph variable ``t`` (the minimax objective).

**Objective (minimax utility).**

    minimise  t
    s.t.  Σ_j  loss(f(x_i), y_j) · p[i][j]  ≤  t     ∀ i ∈ [n]

**Pure DP constraints (δ = 0).**  For every adjacent pair (i, i') and
every output bin j:

    p[i][j]  −  e^ε · p[i'][j]  ≤  0      (forward)
    p[i'][j] −  e^ε · p[i][j]   ≤  0      (backward)

Each row has exactly 2 non-zeros → highly sparse.

**Approximate DP constraints (δ > 0).**  Hockey-stick divergence
epigraph: for each adjacent pair (i, i') introduce *k* slack variables
``s[i,i',j] ≥ 0``:

    p[i][j]  −  e^ε · p[i'][j]  ≤  s[i,i',j]         ∀ j
    Σ_j  s[i,i',j]  ≤  δ

(and symmetrically with i ↔ i').

**Simplex constraints.**   Σ_j p[i][j] = 1   ∀ i.

**Bounds.**  p[i][j] ≥ η_min  where  η_min = exp(−ε) · solver_tol.

Implementation notes
~~~~~~~~~~~~~~~~~~~~

*  All constraint matrices are built in COO format (for efficient
   incremental construction) and converted to CSR just before the solve
   call.
*  The module never forms dense matrices: even the loss matrix is stored
   as a dense 2-D NumPy array only for coefficient lookup; it is never
   multiplied into a dense constraint block.
*  ``LPManager`` supports adding/removing constraint batches across CEGIS
   iterations and tracks constraint history for convergence analysis.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.optimize import LinearConstraint, linprog
from scipy.sparse import coo_matrix, csr_matrix

from dp_forge.exceptions import (
    ConfigurationError,
    InfeasibleSpecError,
    NumericalInstabilityError,
    SolverError,
)
from dp_forge.types import (
    LossFunction,
    LPStruct,
    NumericalConfig,
    QuerySpec,
    SolverBackend,
    SynthesisConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SOLVER_TOL: float = 1e-8
_DEFAULT_DP_TOL: float = 1e-6
_DEFAULT_MAX_COND: float = 1e12
_CONSTRAINT_SCALE_THRESHOLD: float = 1e6
_MIN_K: int = 2
_PIECEWISE_L2_DEFAULT_SEGMENTS: int = 8


# ---------------------------------------------------------------------------
# Variable layout helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariableLayout:
    """Maps semantic variable names to flat LP vector indices.

    The LP variable vector is partitioned as follows::

        [ p[0][0], p[0][1], …, p[0][k-1],   ← row 0
          p[1][0], …, p[1][k-1],              ← row 1
          …
          p[n-1][0], …, p[n-1][k-1],          ← row n-1
          t,                                    ← epigraph variable
          <auxiliary variables …> ]             ← loss / approx-DP auxiliaries

    Attributes:
        n: Number of databases (rows of the mechanism table).
        k: Number of output bins (columns of the mechanism table).
        n_aux: Number of auxiliary variables beyond p and t.
        aux_labels: Human-readable labels for auxiliary variable blocks.
    """

    n: int
    k: int
    n_aux: int = 0
    aux_labels: Tuple[str, ...] = ()

    # -- p variables --------------------------------------------------------

    def p_index(self, i: int, j: int) -> int:
        """Flat index of ``p[i][j]``."""
        return i * self.k + j

    def p_indices_row(self, i: int) -> npt.NDArray[np.intp]:
        """All flat indices for row *i* of the mechanism table."""
        start = i * self.k
        return np.arange(start, start + self.k, dtype=np.intp)

    # -- t (epigraph) variable ---------------------------------------------

    @property
    def t_index(self) -> int:
        """Flat index of the minimax epigraph variable *t*."""
        return self.n * self.k

    # -- auxiliary variables ------------------------------------------------

    @property
    def aux_start(self) -> int:
        """Start index of the auxiliary variable block."""
        return self.n * self.k + 1

    def aux_index(self, offset: int) -> int:
        """Flat index of auxiliary variable at *offset* within the aux block."""
        return self.aux_start + offset

    # -- totals -------------------------------------------------------------

    @property
    def n_vars(self) -> int:
        """Total number of LP variables."""
        return self.n * self.k + 1 + self.n_aux


# ---------------------------------------------------------------------------
# Output grid construction
# ---------------------------------------------------------------------------


def build_output_grid(
    f_values: npt.NDArray[np.float64],
    k: int,
    *,
    padding_factor: float = 0.5,
) -> npt.NDArray[np.float64]:
    """Build an evenly-spaced output discretisation grid.

    The grid spans ``[f_min − pad, f_max + pad]`` where
    ``pad = padding_factor × (f_max − f_min) / (k − 1)``.  This ensures
    that there is always at least one grid point on each side of every
    query value, which is important for bounded support.

    Parameters
    ----------
    f_values : array of shape (n,)
        Query output values ``f(x_1), …, f(x_n)``.
    k : int
        Number of grid points (bins).
    padding_factor : float
        How many bin-widths of padding to add on each side.

    Returns
    -------
    y_grid : array of shape (k,)
        Sorted grid of output values.
    """
    if k < _MIN_K:
        raise ConfigurationError(
            f"k must be >= {_MIN_K}, got {k}",
            parameter="k",
            value=k,
            constraint=f">= {_MIN_K}",
        )

    f_min, f_max = float(np.min(f_values)), float(np.max(f_values))

    if f_min == f_max:
        # Degenerate: single query value → centre the grid around it
        half_span = max(1.0, abs(f_min))
        f_min -= half_span
        f_max += half_span

    span = f_max - f_min
    pad = padding_factor * span / max(k - 1, 1)
    y_grid = np.linspace(f_min - pad, f_max + pad, k)
    return y_grid


# ---------------------------------------------------------------------------
# Loss matrix construction
# ---------------------------------------------------------------------------


def build_loss_matrix(
    f_values: npt.NDArray[np.float64],
    y_grid: npt.NDArray[np.float64],
    loss_fn: LossFunction,
    *,
    custom_loss: Optional[Callable[[float, float], float]] = None,
) -> npt.NDArray[np.float64]:
    """Compute the n × k loss matrix ``L[i][j] = loss(f[i], y_grid[j])``.

    Parameters
    ----------
    f_values : array of shape (n,)
        Query output values.
    y_grid : array of shape (k,)
        Output discretisation grid.
    loss_fn : LossFunction
        Which loss function to use.
    custom_loss : callable, optional
        Required when ``loss_fn == LossFunction.CUSTOM``.

    Returns
    -------
    L : array of shape (n, k)
        Loss coefficients (always non-negative for standard losses).
    """
    n = len(f_values)
    k = len(y_grid)

    if loss_fn == LossFunction.CUSTOM:
        if custom_loss is None:
            raise ConfigurationError(
                "custom_loss callable required for LossFunction.CUSTOM",
                parameter="custom_loss",
                value=None,
                constraint="must be a callable (float, float) -> float",
            )
        fn = custom_loss
    else:
        fn = loss_fn.fn
        assert fn is not None

    # Vectorised computation: broadcast f_values (n,1) against y_grid (1,k)
    f_col = f_values[:, np.newaxis]  # (n, 1)
    y_row = y_grid[np.newaxis, :]    # (1, k)

    if loss_fn == LossFunction.L1:
        L = np.abs(f_col - y_row)
    elif loss_fn == LossFunction.L2:
        L = (f_col - y_row) ** 2
    elif loss_fn == LossFunction.LINF:
        L = np.abs(f_col - y_row)
    else:
        # Custom: element-wise
        L = np.empty((n, k), dtype=np.float64)
        for i in range(n):
            for j in range(k):
                L[i, j] = fn(float(f_values[i]), float(y_grid[j]))

    return L


# ---------------------------------------------------------------------------
# Piecewise-linear approximation of L2 loss for LP formulation
# ---------------------------------------------------------------------------


def piecewise_linear_l2_coefficients(
    f_val: float,
    y_grid: npt.NDArray[np.float64],
    n_segments: int = _PIECEWISE_L2_DEFAULT_SEGMENTS,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute piecewise-linear upper envelope of ``(f_val − y)²``.

    The quadratic ``q(y) = (f_val − y)²`` is upper-approximated by
    *n_segments* tangent lines at evenly-spaced points.  For each
    tangent line ``a_s · y + b_s`` the LP adds a constraint
    ``z_i ≥ Σ_j (a_s · y_j + b_s) · p[i][j]`` where ``z_i`` is an
    auxiliary variable representing the linearised L2 cost for row *i*.

    Parameters
    ----------
    f_val : float
        The true query value f(x_i).
    y_grid : array of shape (k,)
        Output grid points.
    n_segments : int
        Number of tangent lines.

    Returns
    -------
    slopes : array of shape (n_segments,)
        Slopes ``a_s`` of the tangent lines.
    intercepts : array of shape (n_segments,)
        Intercepts ``b_s`` of the tangent lines.
    """
    y_min, y_max = float(y_grid[0]), float(y_grid[-1])
    touch_points = np.linspace(y_min, y_max, n_segments)

    # Tangent to q(y) = (f_val − y)² at y = y_0:
    #   q'(y_0) = −2(f_val − y_0)
    #   tangent: q(y_0) + q'(y_0)(y − y_0) = y_0² − 2·f_val·y_0 + f_val² − 2(f_val − y_0)(y − y_0)
    #          = −2(f_val − y_0)·y + (f_val − y_0)² + (f_val − y_0)·2·y_0
    #          = −2(f_val − y_0)·y + (f_val − y_0)(f_val − y_0 + 2y_0)
    #          = −2(f_val − y_0)·y + (f_val − y_0)(f_val + y_0)
    # Simplify:  slope = −2(f_val − y_0),  intercept = (f_val − y_0)(f_val + y_0)

    diffs = f_val - touch_points  # (n_segments,)
    slopes = -2.0 * diffs
    intercepts = diffs * (f_val + touch_points)

    return slopes, intercepts


# ---------------------------------------------------------------------------
# Constraint builders — Pure DP
# ---------------------------------------------------------------------------


def build_pure_dp_constraints(
    i: int,
    i_prime: int,
    k: int,
    epsilon: float,
    layout: VariableLayout,
) -> Tuple[coo_matrix, npt.NDArray[np.float64]]:
    """Build pure DP constraint rows for an adjacent pair (i, i').

    For pure (ε, 0)-DP, the constraints are (for each output bin j):

        p[i][j]    − e^ε · p[i'][j]  ≤  0    (forward)
        p[i'][j]   − e^ε · p[i][j]   ≤  0    (backward)

    This yields 2k inequality rows.  Each row has exactly 2 non-zeros.

    Parameters
    ----------
    i, i_prime : int
        Database indices forming the adjacent pair.
    k : int
        Number of output bins.
    epsilon : float
        Privacy parameter ε.
    layout : VariableLayout
        Variable index mapping.

    Returns
    -------
    A_block : coo_matrix of shape (2k, n_vars)
        Constraint matrix block.
    b_block : array of shape (2k,)
        RHS vector (all zeros).
    """
    e_eps = math.exp(epsilon)
    n_rows = 2 * k
    n_vars = layout.n_vars

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for j in range(k):
        p_ij = layout.p_index(i, j)
        p_ipj = layout.p_index(i_prime, j)

        # Forward: p[i][j] − e^ε · p[i'][j] ≤ 0
        row_fwd = 2 * j
        rows.extend([row_fwd, row_fwd])
        cols.extend([p_ij, p_ipj])
        data.extend([1.0, -e_eps])

        # Backward: p[i'][j] − e^ε · p[i][j] ≤ 0
        row_bwd = 2 * j + 1
        rows.extend([row_bwd, row_bwd])
        cols.extend([p_ipj, p_ij])
        data.extend([1.0, -e_eps])

    A_block = coo_matrix(
        (data, (rows, cols)),
        shape=(n_rows, n_vars),
        dtype=np.float64,
    )
    b_block = np.zeros(n_rows, dtype=np.float64)

    return A_block, b_block


# ---------------------------------------------------------------------------
# Constraint builders — Approximate DP (hockey-stick divergence)
# ---------------------------------------------------------------------------


class _ApproxDPSlackTracker:
    """Tracks slack variable allocation for approximate DP constraints.

    Each adjacent pair (i, i') in both directions requires *k* slack
    variables ``s[j] ≥ 0`` and one budget constraint ``Σ_j s[j] ≤ δ``.
    This tracker assigns contiguous blocks of auxiliary indices.
    """

    def __init__(self) -> None:
        self._offset: int = 0
        self._allocations: Dict[Tuple[int, int], int] = {}

    def allocate(self, pair: Tuple[int, int], k: int) -> int:
        """Allocate k slack variables for *pair*; return the start offset."""
        if pair in self._allocations:
            return self._allocations[pair]
        start = self._offset
        self._allocations[pair] = start
        self._offset += k
        return start

    @property
    def total_slacks(self) -> int:
        """Total number of slack variables allocated so far."""
        return self._offset

    @property
    def allocations(self) -> Dict[Tuple[int, int], int]:
        """Map from (i, i') pair to slack start offset."""
        return dict(self._allocations)


def build_approx_dp_constraints(
    i: int,
    i_prime: int,
    k: int,
    epsilon: float,
    delta: float,
    layout: VariableLayout,
    slack_start_offset: int,
) -> Tuple[coo_matrix, npt.NDArray[np.float64]]:
    """Build approximate DP constraints for an adjacent pair (i, i').

    For (ε, δ)-DP with δ > 0 the hockey-stick divergence epigraph is:

        p[i][j] − e^ε · p[i'][j] ≤ s_fwd[j]         ∀ j     (k rows)
        Σ_j s_fwd[j] ≤ δ                                      (1 row)
        p[i'][j] − e^ε · p[i][j] ≤ s_bwd[j]          ∀ j     (k rows)
        Σ_j s_bwd[j] ≤ δ                                      (1 row)

    Total: 2(k + 1) inequality rows.  Slack variables are non-negative
    by construction (their bounds are set in the bounds vector).

    Parameters
    ----------
    i, i_prime : int
        Adjacent pair indices.
    k : int
        Number of output bins.
    epsilon : float
        Privacy parameter ε.
    delta : float
        Privacy parameter δ > 0.
    layout : VariableLayout
        Variable index mapping.
    slack_start_offset : int
        Index within the auxiliary block where this pair's slacks begin.
        The first k slacks are for the forward direction, the next k
        for backward.

    Returns
    -------
    A_block : coo_matrix of shape (2(k+1), n_vars)
        Constraint matrix block.
    b_block : array of shape (2(k+1),)
        RHS vector.
    """
    e_eps = math.exp(epsilon)
    n_rows = 2 * (k + 1)
    n_vars = layout.n_vars

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    # -- Forward direction: p[i][j] − e^ε·p[i'][j] − s_fwd[j] ≤ 0 --------
    for j in range(k):
        p_ij = layout.p_index(i, j)
        p_ipj = layout.p_index(i_prime, j)
        s_fwd_j = layout.aux_index(slack_start_offset + j)

        row = j
        rows.extend([row, row, row])
        cols.extend([p_ij, p_ipj, s_fwd_j])
        data.extend([1.0, -e_eps, -1.0])

    # Forward budget: Σ_j s_fwd[j] ≤ δ
    row_budget_fwd = k
    for j in range(k):
        s_fwd_j = layout.aux_index(slack_start_offset + j)
        rows.append(row_budget_fwd)
        cols.append(s_fwd_j)
        data.append(1.0)

    # -- Backward direction: p[i'][j] − e^ε·p[i][j] − s_bwd[j] ≤ 0 -------
    bwd_slack_offset = slack_start_offset + k
    for j in range(k):
        p_ij = layout.p_index(i, j)
        p_ipj = layout.p_index(i_prime, j)
        s_bwd_j = layout.aux_index(bwd_slack_offset + j)

        row = k + 1 + j
        rows.extend([row, row, row])
        cols.extend([p_ipj, p_ij, s_bwd_j])
        data.extend([1.0, -e_eps, -1.0])

    # Backward budget: Σ_j s_bwd[j] ≤ δ
    row_budget_bwd = 2 * k + 1
    for j in range(k):
        s_bwd_j = layout.aux_index(bwd_slack_offset + j)
        rows.append(row_budget_bwd)
        cols.append(s_bwd_j)
        data.append(1.0)

    A_block = coo_matrix(
        (data, (rows, cols)),
        shape=(n_rows, n_vars),
        dtype=np.float64,
    )

    b_block = np.zeros(n_rows, dtype=np.float64)
    b_block[k] = delta          # forward budget
    b_block[2 * k + 1] = delta  # backward budget

    return A_block, b_block


# ---------------------------------------------------------------------------
# Constraint builders — Simplex (probability normalization)
# ---------------------------------------------------------------------------


def build_simplex_constraints(
    n: int,
    k: int,
    layout: VariableLayout,
) -> Tuple[csr_matrix, npt.NDArray[np.float64]]:
    """Build simplex equality constraints: Σ_j p[i][j] = 1 for each i.

    Parameters
    ----------
    n : int
        Number of databases.
    k : int
        Number of output bins.
    layout : VariableLayout
        Variable index mapping.

    Returns
    -------
    A_eq : csr_matrix of shape (n, n_vars)
        Equality constraint matrix.
    b_eq : array of shape (n,)
        RHS (all ones).
    """
    n_vars = layout.n_vars

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for i in range(n):
        for j in range(k):
            rows.append(i)
            cols.append(layout.p_index(i, j))
            data.append(1.0)

    A_eq = csr_matrix(
        (data, (rows, cols)),
        shape=(n, n_vars),
        dtype=np.float64,
    )
    b_eq = np.ones(n, dtype=np.float64)

    return A_eq, b_eq


# ---------------------------------------------------------------------------
# Constraint builders — Minimax objective (epigraph)
# ---------------------------------------------------------------------------


def build_minimax_objective(
    loss_matrix: npt.NDArray[np.float64],
    n: int,
    k: int,
    layout: VariableLayout,
) -> Tuple[npt.NDArray[np.float64], coo_matrix, npt.NDArray[np.float64]]:
    """Build the minimax epigraph objective.

    Objective: minimise t

    Subject to:  Σ_j L[i][j] · p[i][j] − t ≤ 0    ∀ i

    These are n inequality rows.

    Parameters
    ----------
    loss_matrix : array of shape (n, k)
        Pre-computed loss coefficients ``L[i][j] = loss(f[i], y[j])``.
    n : int
        Number of databases.
    k : int
        Number of output bins.
    layout : VariableLayout
        Variable index mapping.

    Returns
    -------
    c : array of shape (n_vars,)
        Objective vector (only the *t* component is 1.0).
    A_obj : coo_matrix of shape (n, n_vars)
        Epigraph inequality rows.
    b_obj : array of shape (n,)
        RHS (all zeros).
    """
    n_vars = layout.n_vars

    # Objective: min t
    c = np.zeros(n_vars, dtype=np.float64)
    c[layout.t_index] = 1.0

    # Epigraph constraints: Σ_j L[i][j]·p[i][j] − t ≤ 0
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for i in range(n):
        for j in range(k):
            coeff = loss_matrix[i, j]
            if coeff != 0.0:
                rows.append(i)
                cols.append(layout.p_index(i, j))
                data.append(coeff)
        # −t coefficient
        rows.append(i)
        cols.append(layout.t_index)
        data.append(-1.0)

    A_obj = coo_matrix(
        (data, (rows, cols)),
        shape=(n, n_vars),
        dtype=np.float64,
    )
    b_obj = np.zeros(n, dtype=np.float64)

    return c, A_obj, b_obj


# ---------------------------------------------------------------------------
# L1 loss — auxiliary variable formulation
# ---------------------------------------------------------------------------


def build_l1_auxiliary_constraints(
    f_values: npt.NDArray[np.float64],
    y_grid: npt.NDArray[np.float64],
    n: int,
    k: int,
    layout: VariableLayout,
    aux_offset: int,
) -> Tuple[coo_matrix, npt.NDArray[np.float64], coo_matrix, npt.NDArray[np.float64]]:
    """Build L1 absolute-error LP constraints with auxiliary variables.

    For L1 loss, we introduce auxiliary variable ``z_i`` for each row *i*
    representing the expected absolute error.  The minimax objective becomes
    ``min t s.t. z_i ≤ t ∀ i``.

    The absolute value |f_i − y_j| is already non-negative, so
    ``z_i = Σ_j |f_i − y_j| · p[i][j]`` is linear — no auxiliary
    decomposition is needed beyond the epigraph.

    For L1, the loss matrix entries ``|f_i − y_j|`` are directly used as
    coefficients of ``p[i][j]`` in the epigraph constraint.  This function
    returns the epigraph rows and the objective vector.

    Returns
    -------
    A_epi : coo_matrix of shape (n, n_vars)
        Epigraph rows: Σ_j |f_i − y_j| · p[i][j] − t ≤ 0.
    b_epi : array of shape (n,)
        RHS (all zeros).
    A_aux : coo_matrix of shape (0, n_vars)
        Empty — no auxiliary constraints needed for L1.
    b_aux : array of shape (0,)
        Empty.
    """
    loss_matrix = build_loss_matrix(f_values, y_grid, LossFunction.L1)
    _, A_epi, b_epi = build_minimax_objective(loss_matrix, n, k, layout)

    # No additional auxiliary constraints for L1
    A_aux = coo_matrix((0, layout.n_vars), dtype=np.float64)
    b_aux = np.empty(0, dtype=np.float64)

    return A_epi, b_epi, A_aux, b_aux


# ---------------------------------------------------------------------------
# L2 loss — piecewise-linear approximation
# ---------------------------------------------------------------------------


def build_l2_piecewise_constraints(
    f_values: npt.NDArray[np.float64],
    y_grid: npt.NDArray[np.float64],
    n: int,
    k: int,
    layout: VariableLayout,
    aux_offset: int,
    n_segments: int = _PIECEWISE_L2_DEFAULT_SEGMENTS,
) -> Tuple[coo_matrix, npt.NDArray[np.float64], coo_matrix, npt.NDArray[np.float64]]:
    """Build LP constraints for L2 (squared error) via piecewise linearisation.

    Since ``(f_i − y_j)²`` is convex in ``y``, its expectation under
    ``p[i][·]`` is convex.  We approximate the expected squared error
    from above using tangent lines:

    For each row *i* introduce auxiliary variable ``z_i`` (at index
    ``aux_offset + i`` in the aux block) representing the linearised
    expected squared error.  The epigraph constraint is ``z_i ≤ t``.

    For each tangent line *s* of the quadratic at row *i*:

        Σ_j (a_s · y_j + b_s) · p[i][j]  ≤  z_i

    Parameters
    ----------
    f_values : array of shape (n,)
    y_grid : array of shape (k,)
    n, k : int
    layout : VariableLayout
    aux_offset : int
        Start of z_i auxiliary variables in the aux block.
    n_segments : int
        Number of tangent segments per row.

    Returns
    -------
    A_epi : coo_matrix of shape (n, n_vars)
        Epigraph rows: z_i − t ≤ 0.
    b_epi : array of shape (n,)
    A_tang : coo_matrix of shape (n * n_segments, n_vars)
        Tangent constraints: Σ_j coeff[s,j] · p[i][j] − z_i ≤ 0.
    b_tang : array of shape (n * n_segments,)
    """
    n_vars = layout.n_vars

    # -- Epigraph: z_i − t ≤ 0 for each i ----------------------------------
    epi_rows: List[int] = []
    epi_cols: List[int] = []
    epi_data: List[float] = []

    for i in range(n):
        z_i = layout.aux_index(aux_offset + i)
        epi_rows.extend([i, i])
        epi_cols.extend([z_i, layout.t_index])
        epi_data.extend([1.0, -1.0])

    A_epi = coo_matrix(
        (epi_data, (epi_rows, epi_cols)),
        shape=(n, n_vars),
        dtype=np.float64,
    )
    b_epi = np.zeros(n, dtype=np.float64)

    # -- Tangent constraints ------------------------------------------------
    n_tang = n * n_segments
    tang_rows: List[int] = []
    tang_cols: List[int] = []
    tang_data: List[float] = []

    for i in range(n):
        f_val = float(f_values[i])
        slopes, intercepts = piecewise_linear_l2_coefficients(
            f_val, y_grid, n_segments
        )
        z_i = layout.aux_index(aux_offset + i)

        for s_idx in range(n_segments):
            row = i * n_segments + s_idx
            a_s = slopes[s_idx]
            b_s = intercepts[s_idx]

            # Σ_j (a_s · y_j + b_s) · p[i][j] − z_i ≤ 0
            for j in range(k):
                coeff = a_s * y_grid[j] + b_s
                if coeff != 0.0:
                    tang_rows.append(row)
                    tang_cols.append(layout.p_index(i, j))
                    tang_data.append(coeff)
            # −z_i
            tang_rows.append(row)
            tang_cols.append(z_i)
            tang_data.append(-1.0)

    A_tang = coo_matrix(
        (tang_data, (tang_rows, tang_cols)),
        shape=(n_tang, n_vars),
        dtype=np.float64,
    )
    b_tang = np.zeros(n_tang, dtype=np.float64)

    return A_epi, b_epi, A_tang, b_tang


# ---------------------------------------------------------------------------
# Linf loss — already in minimax form
# ---------------------------------------------------------------------------


def build_linf_objective(
    f_values: npt.NDArray[np.float64],
    y_grid: npt.NDArray[np.float64],
    n: int,
    k: int,
    layout: VariableLayout,
) -> Tuple[npt.NDArray[np.float64], coo_matrix, npt.NDArray[np.float64]]:
    """Build Linf minimax objective.

    Linf and L1 share the same loss matrix ``|f_i − y_j|``.  The minimax
    epigraph structure is identical: min t s.t. E_i[|f_i − Y|] ≤ t.
    """
    loss_matrix = build_loss_matrix(f_values, y_grid, LossFunction.LINF)
    return build_minimax_objective(loss_matrix, n, k, layout)


# ---------------------------------------------------------------------------
# Variable bounds
# ---------------------------------------------------------------------------


def build_variable_bounds(
    layout: VariableLayout,
    eta_min: float,
) -> List[Tuple[float, float]]:
    """Build per-variable (lower, upper) bounds.

    - p[i][j] ∈ [η_min, 1.0]
    - t ∈ [0, +∞)
    - Auxiliary variables (slacks, etc.) ∈ [0, +∞)

    Parameters
    ----------
    layout : VariableLayout
        Variable index mapping.
    eta_min : float
        Minimum probability floor.

    Returns
    -------
    bounds : list of (lo, hi) pairs, length n_vars.
    """
    bounds: List[Tuple[float, float]] = []

    # p variables
    for _ in range(layout.n * layout.k):
        bounds.append((eta_min, 1.0))

    # t variable
    bounds.append((0.0, None))  # type: ignore[arg-type]

    # Auxiliary variables
    for _ in range(layout.n_aux):
        bounds.append((0.0, None))  # type: ignore[arg-type]

    return bounds


# ---------------------------------------------------------------------------
# Variable map
# ---------------------------------------------------------------------------


def build_var_map(n: int, k: int, layout: VariableLayout) -> Dict[Tuple[int, int], int]:
    """Build mapping from (i, j) mechanism indices to flat variable index.

    Parameters
    ----------
    n : int
        Number of databases.
    k : int
        Number of output bins.
    layout : VariableLayout
        Variable index mapping.

    Returns
    -------
    var_map : dict mapping (i, j) → flat index.
    """
    return {(i, j): layout.p_index(i, j) for i in range(n) for j in range(k)}


# ---------------------------------------------------------------------------
# Constraint scaling
# ---------------------------------------------------------------------------


def scale_constraints(
    A: csr_matrix,
    b: npt.NDArray[np.float64],
) -> Tuple[csr_matrix, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Row-scale constraints for improved numerical conditioning.

    Each constraint row is divided by max(1, ‖row‖_∞) so that all
    coefficients are in [-1, 1].  The scaling factors are returned for
    recovering dual variables.

    Parameters
    ----------
    A : csr_matrix
        Constraint matrix.
    b : array
        RHS vector.

    Returns
    -------
    A_scaled : csr_matrix
        Row-scaled constraint matrix.
    b_scaled : array
        Row-scaled RHS.
    scale_factors : array
        Scaling factors applied to each row (multiply duals by these to undo).
    """
    if A.shape[0] == 0:
        return A, b, np.ones(0, dtype=np.float64)

    A_csr = A.tocsr()

    # Compute row-wise infinity norm
    row_norms = np.zeros(A_csr.shape[0], dtype=np.float64)
    for row_idx in range(A_csr.shape[0]):
        start = A_csr.indptr[row_idx]
        end = A_csr.indptr[row_idx + 1]
        if end > start:
            row_norms[row_idx] = np.max(np.abs(A_csr.data[start:end]))

    # Avoid division by zero
    scale_factors = np.maximum(row_norms, 1.0)

    # Scale
    A_scaled_data = A_csr.data.copy()
    b_scaled = b.copy()

    for row_idx in range(A_csr.shape[0]):
        start = A_csr.indptr[row_idx]
        end = A_csr.indptr[row_idx + 1]
        A_scaled_data[start:end] /= scale_factors[row_idx]
        b_scaled[row_idx] /= scale_factors[row_idx]

    A_scaled = csr_matrix(
        (A_scaled_data, A_csr.indices.copy(), A_csr.indptr.copy()),
        shape=A_csr.shape,
        dtype=np.float64,
    )

    return A_scaled, b_scaled, scale_factors


# ---------------------------------------------------------------------------
# Numerical diagnostics
# ---------------------------------------------------------------------------


def estimate_condition_number(A: csr_matrix, max_samples: int = 200) -> float:
    """Estimate the condition number of a sparse constraint matrix.

    Uses a randomised SVD approach for large matrices to avoid forming
    the full dense matrix.  For small matrices (< max_samples rows and
    cols), computes the exact condition number via dense SVD.

    Parameters
    ----------
    A : csr_matrix
        Constraint matrix.
    max_samples : int
        Threshold below which exact dense SVD is used.

    Returns
    -------
    cond : float
        Estimated condition number (ratio of largest to smallest
        singular value).  Returns ``inf`` if the matrix is rank-deficient.
    """
    m, n_c = A.shape
    if m == 0 or n_c == 0:
        return 1.0

    if m <= max_samples and n_c <= max_samples:
        A_dense = A.toarray()
        try:
            s = np.linalg.svd(A_dense, compute_uv=False)
            s_pos = s[s > 1e-15]
            if len(s_pos) == 0:
                return float("inf")
            return float(s_pos[0] / s_pos[-1])
        except np.linalg.LinAlgError:
            return float("inf")

    # Randomised estimate: power iteration for largest/smallest singular values
    try:
        from scipy.sparse.linalg import svds

        n_sv = min(min(m, n_c) - 1, 6)
        if n_sv < 1:
            return 1.0
        s = svds(A.astype(np.float64), k=n_sv, return_singular_vectors=False)
        s = np.sort(s)[::-1]
        s_pos = s[s > 1e-15]
        if len(s_pos) == 0:
            return float("inf")
        return float(s_pos[0] / s_pos[-1])
    except Exception:
        return float("inf")


def detect_degeneracy(
    A: csr_matrix,
    b: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    tol: float = 1e-8,
) -> Dict[str, Any]:
    """Detect LP degeneracy indicators in a solution.

    Degeneracy — many constraints active simultaneously — can cause
    cycling in the simplex method and numerical instability.

    Parameters
    ----------
    A : csr_matrix
        Inequality constraint matrix.
    b : array
        RHS vector.
    x : array
        Primal solution.
    tol : float
        Tolerance for considering a constraint "active".

    Returns
    -------
    info : dict
        ``n_active``: number of active (tight) inequality constraints.
        ``n_total``: total number of inequality constraints.
        ``degeneracy_ratio``: active / n_vars.
        ``is_degenerate``: True if ratio > 1.0 (more active than variables).
    """
    if A.shape[0] == 0:
        return {
            "n_active": 0,
            "n_total": 0,
            "degeneracy_ratio": 0.0,
            "is_degenerate": False,
        }

    slacks = b - A.dot(x)
    n_active = int(np.sum(np.abs(slacks) < tol))
    n_vars = len(x)
    ratio = n_active / max(n_vars, 1)

    return {
        "n_active": n_active,
        "n_total": A.shape[0],
        "degeneracy_ratio": ratio,
        "is_degenerate": ratio > 1.0,
    }


# ---------------------------------------------------------------------------
# Infeasibility diagnosis
# ---------------------------------------------------------------------------


def diagnose_infeasibility(
    A_ub: csr_matrix,
    b_ub: npt.NDArray[np.float64],
    A_eq: Optional[csr_matrix],
    b_eq: Optional[npt.NDArray[np.float64]],
    bounds: List[Tuple[float, float]],
    n: int,
    k: int,
    epsilon: float,
    delta: float,
    layout: VariableLayout,
    constraint_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Attempt to diagnose why an LP is infeasible.

    This performs a lightweight Irreducible Infeasible Subsystem (IIS)-like
    analysis by testing constraint groups in isolation.

    Strategy:
    1. Test simplex constraints alone → feasible? (should always be)
    2. Test simplex + bounds → feasible?
    3. For each DP pair, test simplex + bounds + that pair's constraints.
    4. Report which constraint groups cause infeasibility.

    Parameters
    ----------
    A_ub, b_ub, A_eq, b_eq, bounds : LP data
    n, k, epsilon, delta : problem parameters
    layout : VariableLayout
    constraint_labels : optional list of labels for inequality rows

    Returns
    -------
    diagnosis : dict with keys:
        ``feasible_groups``: list of constraint groups that are individually feasible.
        ``infeasible_groups``: list of constraint groups that cause infeasibility.
        ``suspected_cause``: human-readable diagnosis string.
    """
    diagnosis: Dict[str, Any] = {
        "feasible_groups": [],
        "infeasible_groups": [],
        "suspected_cause": "unknown",
    }

    n_vars = layout.n_vars

    # Test 1: simplex constraints only
    c_test = np.zeros(n_vars, dtype=np.float64)
    c_test[layout.t_index] = 1.0
    try:
        res = linprog(
            c_test,
            A_eq=A_eq.toarray() if A_eq is not None else None,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"presolve": True, "time_limit": 5.0},
        )
        if res.success:
            diagnosis["feasible_groups"].append("simplex+bounds")
        else:
            diagnosis["infeasible_groups"].append("simplex+bounds")
            diagnosis["suspected_cause"] = (
                "Basic simplex+bounds constraints are infeasible. "
                "eta_min may be too large for the given k."
            )
            return diagnosis
    except Exception:
        diagnosis["suspected_cause"] = "Could not test simplex constraints in isolation."
        return diagnosis

    # Test 2: try with all constraints to confirm infeasibility
    try:
        res = linprog(
            c_test,
            A_ub=A_ub.toarray() if A_ub is not None else None,
            b_ub=b_ub,
            A_eq=A_eq.toarray() if A_eq is not None else None,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"presolve": True, "time_limit": 10.0},
        )
        if res.success:
            diagnosis["suspected_cause"] = (
                "Full LP is actually feasible — original infeasibility may be "
                "due to solver tolerances or numerical issues."
            )
            return diagnosis
    except Exception:
        pass

    # If simplex+bounds is feasible but full LP is infeasible,
    # the DP constraints are too tight
    diagnosis["suspected_cause"] = (
        f"Simplex+bounds is feasible but DP constraints (eps={epsilon}, "
        f"delta={delta}) make the LP infeasible. Consider increasing epsilon, "
        f"delta, or k."
    )

    return diagnosis


# ---------------------------------------------------------------------------
# Solution validation
# ---------------------------------------------------------------------------


def validate_solution(
    x: npt.NDArray[np.float64],
    layout: VariableLayout,
    A_ub: csr_matrix,
    b_ub: npt.NDArray[np.float64],
    A_eq: Optional[csr_matrix],
    b_eq: Optional[npt.NDArray[np.float64]],
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """Validate an LP solution for constraint satisfaction and stochasticity.

    Parameters
    ----------
    x : array of shape (n_vars,)
        Primal solution vector.
    layout : VariableLayout
    A_ub, b_ub, A_eq, b_eq : LP constraint data.
    tol : float
        Tolerance for constraint violation.

    Returns
    -------
    result : dict with keys:
        ``valid``: bool — overall validity.
        ``max_ub_violation``: float — worst inequality violation.
        ``max_eq_violation``: float — worst equality violation.
        ``min_probability``: float — smallest p[i][j] value.
        ``max_row_sum_deviation``: float — worst |Σ_j p[i][j] − 1|.
        ``violations``: list of human-readable violation descriptions.
    """
    result: Dict[str, Any] = {
        "valid": True,
        "max_ub_violation": 0.0,
        "max_eq_violation": 0.0,
        "min_probability": float("inf"),
        "max_row_sum_deviation": 0.0,
        "violations": [],
    }

    # Check inequality constraints
    if A_ub.shape[0] > 0:
        ub_residuals = A_ub.dot(x) - b_ub
        max_ub_viol = float(np.max(ub_residuals))
        result["max_ub_violation"] = max_ub_viol
        if max_ub_viol > tol:
            worst_row = int(np.argmax(ub_residuals))
            result["valid"] = False
            result["violations"].append(
                f"Inequality constraint {worst_row} violated by {max_ub_viol:.2e}"
            )

    # Check equality constraints
    if A_eq is not None and b_eq is not None and A_eq.shape[0] > 0:
        eq_residuals = np.abs(A_eq.dot(x) - b_eq)
        max_eq_viol = float(np.max(eq_residuals))
        result["max_eq_violation"] = max_eq_viol
        if max_eq_viol > tol:
            worst_row = int(np.argmax(eq_residuals))
            result["valid"] = False
            result["violations"].append(
                f"Equality constraint {worst_row} violated by {max_eq_viol:.2e}"
            )

    # Check probability values
    n, k = layout.n, layout.k
    p_vals = x[: n * k]
    min_p = float(np.min(p_vals))
    result["min_probability"] = min_p
    if min_p < -tol:
        result["valid"] = False
        result["violations"].append(f"Negative probability: {min_p:.2e}")

    # Check row sums
    p_matrix = p_vals.reshape(n, k)
    row_sums = p_matrix.sum(axis=1)
    max_dev = float(np.max(np.abs(row_sums - 1.0)))
    result["max_row_sum_deviation"] = max_dev
    if max_dev > tol:
        result["valid"] = False
        result["violations"].append(f"Row sum deviation: {max_dev:.2e}")

    return result


# ---------------------------------------------------------------------------
# Solve statistics
# ---------------------------------------------------------------------------


@dataclass
class SolveStatistics:
    """Detailed statistics from an LP solve.

    Attributes:
        solver_name: Name of the solver backend used.
        status: Solver status string.
        iterations: Number of simplex/IPM iterations.
        solve_time: Wall-clock time in seconds.
        objective_value: Optimal objective value.
        primal_solution: Primal variable values.
        dual_solution: Dual variable values (if available).
        basis_info: Basis status for warm-starting (if available).
        duality_gap: Absolute duality gap (if available).
        n_vars: Number of decision variables.
        n_constraints: Number of constraints (ub + eq).
        condition_estimate: Estimated condition number of the constraint matrix.
    """

    solver_name: str
    status: str
    iterations: int
    solve_time: float
    objective_value: float
    primal_solution: npt.NDArray[np.float64]
    dual_solution: Optional[npt.NDArray[np.float64]] = None
    basis_info: Optional[Dict[str, Any]] = None
    duality_gap: Optional[float] = None
    n_vars: int = 0
    n_constraints: int = 0
    condition_estimate: Optional[float] = None


# ---------------------------------------------------------------------------
# Solver interface — HiGHS (via scipy)
# ---------------------------------------------------------------------------


def _solve_highs(
    lp: LPStruct,
    *,
    warm_start: Optional[Dict[str, Any]] = None,
    solver_tol: float = _DEFAULT_SOLVER_TOL,
    time_limit: float = 300.0,
    verbose: int = 0,
) -> SolveStatistics:
    """Solve an LP using HiGHS (SciPy's default LP solver).

    Parameters
    ----------
    lp : LPStruct
        The LP to solve.
    warm_start : dict, optional
        Warm-start information from a previous solve. Currently HiGHS
        in SciPy does not support full basis warm-starting, but we pass
        an initial guess via ``x0`` if available.
    solver_tol : float
        Feasibility and optimality tolerance.
    time_limit : float
        Maximum solve time in seconds.
    verbose : int
        Verbosity level.

    Returns
    -------
    stats : SolveStatistics
    """
    t_start = time.perf_counter()

    options: Dict[str, Any] = {
        "presolve": True,
        "dual_feasibility_tolerance": solver_tol,
        "primal_feasibility_tolerance": solver_tol,
        "time_limit": time_limit,
        "disp": verbose >= 2,
    }

    # Prepare matrices
    A_ub_csr = lp.A_ub.tocsr() if not isinstance(lp.A_ub, csr_matrix) else lp.A_ub
    A_eq_csr = None
    if lp.A_eq is not None:
        A_eq_csr = lp.A_eq.tocsr() if not isinstance(lp.A_eq, csr_matrix) else lp.A_eq

    try:
        result = linprog(
            c=lp.c,
            A_ub=A_ub_csr,
            b_ub=lp.b_ub,
            A_eq=A_eq_csr,
            b_eq=lp.b_eq,
            bounds=lp.bounds,
            method="highs",
            options=options,
        )
    except Exception as exc:
        raise SolverError(
            f"HiGHS solver raised an exception: {exc}",
            solver_name="HiGHS",
            solver_status="error",
            original_error=exc,
        ) from exc

    solve_time = time.perf_counter() - t_start

    if not result.success:
        status_str = str(result.status)
        message = result.message if hasattr(result, "message") else "unknown"
        if result.status == 2:  # Infeasible
            raise InfeasibleSpecError(
                f"HiGHS reports infeasible: {message}",
                solver_status=status_str,
                n_vars=lp.n_vars,
                n_constraints=lp.n_ub + lp.n_eq,
            )
        raise SolverError(
            f"HiGHS solve failed (status={result.status}): {message}",
            solver_name="HiGHS",
            solver_status=status_str,
        )

    # Extract dual information if available
    dual_sol = None
    if hasattr(result, "ineqlin") and result.ineqlin is not None:
        dual_ub = getattr(result.ineqlin, "marginals", None)
        dual_eq = None
        if hasattr(result, "eqlin") and result.eqlin is not None:
            dual_eq = getattr(result.eqlin, "marginals", None)
        if dual_ub is not None:
            if dual_eq is not None:
                dual_sol = np.concatenate([dual_ub, dual_eq])
            else:
                dual_sol = dual_ub

    n_iter = getattr(result, "nit", 0)

    return SolveStatistics(
        solver_name="HiGHS",
        status="optimal",
        iterations=n_iter,
        solve_time=solve_time,
        objective_value=float(result.fun),
        primal_solution=result.x.copy(),
        dual_solution=dual_sol,
        basis_info={"x": result.x.copy()},
        n_vars=lp.n_vars,
        n_constraints=lp.n_ub + lp.n_eq,
    )


# ---------------------------------------------------------------------------
# Solver interface — GLPK (via cvxopt, optional)
# ---------------------------------------------------------------------------


def _solve_glpk(
    lp: LPStruct,
    *,
    warm_start: Optional[Dict[str, Any]] = None,
    solver_tol: float = _DEFAULT_SOLVER_TOL,
    time_limit: float = 300.0,
    verbose: int = 0,
) -> SolveStatistics:
    """Solve an LP using GLPK via cvxopt.

    GLPK supports true basis warm-starting through the simplex method.

    Raises
    ------
    ImportError
        If cvxopt is not installed.
    SolverError
        If the solve fails.
    """
    t_start = time.perf_counter()

    try:
        from cvxopt import matrix, solvers, spmatrix
    except ImportError as exc:
        raise SolverError(
            "GLPK backend requires cvxopt. Install with: pip install cvxopt",
            solver_name="GLPK",
            solver_status="import_error",
            original_error=exc,
        ) from exc

    n_vars = lp.n_vars

    # Convert to cvxopt sparse format
    def _scipy_to_cvxopt(A_scipy: sparse.spmatrix) -> spmatrix:
        """Convert scipy sparse matrix to cvxopt spmatrix."""
        A_coo = A_scipy.tocoo()
        return spmatrix(
            list(A_coo.data.astype(float)),
            list(A_coo.row.astype(int)),
            list(A_coo.col.astype(int)),
            size=A_coo.shape,
            tc="d",
        )

    c_cvx = matrix(lp.c.astype(float), tc="d")

    # Inequality constraints: A_ub x <= b_ub → G x <= h (cvxopt convention)
    # Also add bound constraints as inequalities
    bound_rows_data: List[float] = []
    bound_rows_i: List[int] = []
    bound_rows_j: List[int] = []
    bound_rhs: List[float] = []
    row_offset = 0

    for v_idx, (lo, hi) in enumerate(lp.bounds):
        if lo is not None and lo != float("-inf"):
            # -x_v <= -lo
            bound_rows_data.append(-1.0)
            bound_rows_i.append(row_offset)
            bound_rows_j.append(v_idx)
            bound_rhs.append(-lo)
            row_offset += 1
        if hi is not None and hi != float("inf"):
            # x_v <= hi
            bound_rows_data.append(1.0)
            bound_rows_i.append(row_offset)
            bound_rows_j.append(v_idx)
            bound_rhs.append(hi)
            row_offset += 1

    n_bound_rows = row_offset

    # Stack A_ub and bound rows
    A_ub_coo = lp.A_ub.tocoo()
    all_data = list(A_ub_coo.data.astype(float)) + bound_rows_data
    all_i = list(A_ub_coo.row.astype(int)) + [
        r + lp.n_ub for r in bound_rows_i
    ]
    all_j = list(A_ub_coo.col.astype(int)) + bound_rows_j
    total_ub_rows = lp.n_ub + n_bound_rows

    if total_ub_rows > 0 and len(all_data) > 0:
        G = spmatrix(all_data, all_i, all_j, size=(total_ub_rows, n_vars), tc="d")
        h = matrix(
            list(lp.b_ub.astype(float)) + bound_rhs,
            tc="d",
        )
    else:
        G = spmatrix([], [], [], size=(0, n_vars), tc="d")
        h = matrix([], tc="d", size=(0, 1))

    # Equality constraints
    A_eq_cvx = None
    b_eq_cvx = None
    if lp.A_eq is not None and lp.b_eq is not None:
        A_eq_cvx = _scipy_to_cvxopt(lp.A_eq)
        b_eq_cvx = matrix(lp.b_eq.astype(float), tc="d")

    # Solver options
    solvers.options["show_progress"] = verbose >= 2
    solvers.options["glpk"] = {
        "msg_lev": "GLP_MSG_OFF" if verbose < 2 else "GLP_MSG_ON",
        "tm_lim": int(time_limit * 1000),
    }

    try:
        sol = solvers.lp(c_cvx, G, h, A_eq_cvx, b_eq_cvx, solver="glpk")
    except Exception as exc:
        raise SolverError(
            f"GLPK solver raised an exception: {exc}",
            solver_name="GLPK",
            solver_status="error",
            original_error=exc,
        ) from exc

    solve_time = time.perf_counter() - t_start

    if sol["status"] != "optimal":
        if sol["status"] == "primal infeasible":
            raise InfeasibleSpecError(
                f"GLPK reports infeasible",
                solver_status=sol["status"],
                n_vars=n_vars,
                n_constraints=total_ub_rows + (lp.n_eq if lp.A_eq is not None else 0),
            )
        raise SolverError(
            f"GLPK solve failed: {sol['status']}",
            solver_name="GLPK",
            solver_status=sol["status"],
        )

    x_sol = np.array(sol["x"]).flatten()
    obj_val = float(sol["primal objective"])

    dual_sol = None
    if sol.get("z") is not None:
        dual_sol = np.array(sol["z"]).flatten()

    return SolveStatistics(
        solver_name="GLPK",
        status="optimal",
        iterations=0,  # GLPK doesn't report iterations via cvxopt
        solve_time=solve_time,
        objective_value=obj_val,
        primal_solution=x_sol,
        dual_solution=dual_sol,
        basis_info={"x": x_sol.copy()},
        n_vars=n_vars,
        n_constraints=total_ub_rows + lp.n_eq,
    )


# ---------------------------------------------------------------------------
# Solver interface — SciPy (revised simplex / IPM fallback)
# ---------------------------------------------------------------------------


def _solve_scipy(
    lp: LPStruct,
    *,
    warm_start: Optional[Dict[str, Any]] = None,
    solver_tol: float = _DEFAULT_SOLVER_TOL,
    time_limit: float = 300.0,
    verbose: int = 0,
) -> SolveStatistics:
    """Solve an LP using SciPy's revised simplex or interior-point method.

    This is the most portable backend (no external dependencies) but
    typically slower than HiGHS for large instances.
    """
    t_start = time.perf_counter()

    A_ub_csr = lp.A_ub.tocsr()
    A_eq_csr = lp.A_eq.tocsr() if lp.A_eq is not None else None

    try:
        result = linprog(
            c=lp.c,
            A_ub=A_ub_csr,
            b_ub=lp.b_ub,
            A_eq=A_eq_csr,
            b_eq=lp.b_eq,
            bounds=lp.bounds,
            method="revised simplex",
            options={"tol": solver_tol, "disp": verbose >= 2},
        )
    except Exception:
        # Fall back to interior point
        result = linprog(
            c=lp.c,
            A_ub=A_ub_csr,
            b_ub=lp.b_ub,
            A_eq=A_eq_csr,
            b_eq=lp.b_eq,
            bounds=lp.bounds,
            method="interior-point",
            options={"tol": solver_tol, "disp": verbose >= 2},
        )

    solve_time = time.perf_counter() - t_start

    if not result.success:
        status_str = str(result.status)
        message = result.message if hasattr(result, "message") else "unknown"
        if result.status == 2:
            raise InfeasibleSpecError(
                f"SciPy reports infeasible: {message}",
                solver_status=status_str,
                n_vars=lp.n_vars,
                n_constraints=lp.n_ub + lp.n_eq,
            )
        raise SolverError(
            f"SciPy solve failed (status={result.status}): {message}",
            solver_name="SciPy",
            solver_status=status_str,
        )

    return SolveStatistics(
        solver_name="SciPy",
        status="optimal",
        iterations=getattr(result, "nit", 0),
        solve_time=solve_time,
        objective_value=float(result.fun),
        primal_solution=result.x.copy(),
        basis_info={"x": result.x.copy()},
        n_vars=lp.n_vars,
        n_constraints=lp.n_ub + lp.n_eq,
    )


# ---------------------------------------------------------------------------
# Unified solver dispatch
# ---------------------------------------------------------------------------


def solve_lp(
    lp: LPStruct,
    solver: SolverBackend = SolverBackend.AUTO,
    *,
    warm_start: Optional[Dict[str, Any]] = None,
    solver_tol: float = _DEFAULT_SOLVER_TOL,
    time_limit: float = 300.0,
    verbose: int = 0,
) -> SolveStatistics:
    """Solve an LP using the specified backend.

    For ``SolverBackend.AUTO``, the selection order is
    HiGHS → GLPK → SciPy.

    Parameters
    ----------
    lp : LPStruct
        The LP problem.
    solver : SolverBackend
        Which solver to use.
    warm_start : dict, optional
        Warm-start data from a previous solve.
    solver_tol : float
        Feasibility/optimality tolerance.
    time_limit : float
        Maximum solve time in seconds.
    verbose : int
        Verbosity level.

    Returns
    -------
    stats : SolveStatistics
    """
    dispatch = {
        SolverBackend.HIGHS: _solve_highs,
        SolverBackend.GLPK: _solve_glpk,
        SolverBackend.SCIPY: _solve_scipy,
    }

    if solver == SolverBackend.AUTO:
        # Try HiGHS first (most robust), then GLPK, then SciPy
        for backend in [SolverBackend.HIGHS, SolverBackend.GLPK, SolverBackend.SCIPY]:
            try:
                return dispatch[backend](
                    lp,
                    warm_start=warm_start,
                    solver_tol=solver_tol,
                    time_limit=time_limit,
                    verbose=verbose,
                )
            except (SolverError, ImportError):
                continue
        raise SolverError(
            "All solver backends failed. Ensure at least SciPy is installed.",
            solver_name="AUTO",
            solver_status="all_failed",
        )

    if solver in dispatch:
        return dispatch[solver](
            lp,
            warm_start=warm_start,
            solver_tol=solver_tol,
            time_limit=time_limit,
            verbose=verbose,
        )

    raise ConfigurationError(
        f"Unsupported solver backend: {solver}",
        parameter="solver",
        value=solver,
        constraint=f"one of {list(dispatch.keys())}",
    )


# ---------------------------------------------------------------------------
# Extract mechanism table from LP solution
# ---------------------------------------------------------------------------


def extract_mechanism_table(
    x: npt.NDArray[np.float64],
    layout: VariableLayout,
) -> npt.NDArray[np.float64]:
    """Extract the n × k mechanism probability table from a flat LP solution.

    Probabilities are clamped to [0, 1] and rows are re-normalised to
    sum to exactly 1.

    Parameters
    ----------
    x : array of shape (n_vars,)
        LP solution vector.
    layout : VariableLayout
        Variable layout.

    Returns
    -------
    P : array of shape (n, k)
        Mechanism table ``P[i][j] = Pr[M(x_i) = y_j]``.
    """
    n, k = layout.n, layout.k
    P = x[: n * k].reshape(n, k).copy()

    # Clamp to [0, 1]
    np.clip(P, 0.0, 1.0, out=P)

    # Re-normalise rows
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-15)  # avoid division by zero
    P /= row_sums

    return P


# ---------------------------------------------------------------------------
# Main LP construction: BuildPrivacyLP
# ---------------------------------------------------------------------------


def build_privacy_lp(
    spec: QuerySpec,
    edges: Optional[List[Tuple[int, int]]] = None,
    *,
    eta_min: Optional[float] = None,
    numerical_config: Optional[NumericalConfig] = None,
    l2_segments: int = _PIECEWISE_L2_DEFAULT_SEGMENTS,
) -> Tuple[LPStruct, VariableLayout, Dict[str, Any]]:
    """Construct the full privacy LP for discrete mechanism synthesis.

    This is the main entry point for LP construction.  It assembles:

    1. Output discretisation grid.
    2. Loss matrix and minimax epigraph objective.
    3. Simplex (probability normalization) equality constraints.
    4. DP inequality constraints for all specified adjacent pairs.
    5. Variable bounds with η_min probability floor.

    Parameters
    ----------
    spec : QuerySpec
        Full query specification including privacy parameters.
    edges : list of (int, int), optional
        Explicit list of adjacent pairs.  If ``None``, uses all edges from
        ``spec.edges``.
    eta_min : float, optional
        Override for the minimum probability floor.
    numerical_config : NumericalConfig, optional
        Numerical precision configuration.
    l2_segments : int
        Number of piecewise-linear segments for L2 loss approximation.

    Returns
    -------
    lp : LPStruct
        The constructed LP, ready to pass to :func:`solve_lp`.
    layout : VariableLayout
        Variable layout for interpreting solutions.
    metadata : dict
        Construction metadata (constraint counts, timings, etc.).
    """
    t_start = time.perf_counter()

    num_cfg = numerical_config or NumericalConfig()
    n = spec.n
    k = spec.k
    epsilon = spec.epsilon
    delta = spec.delta
    f_values = spec.query_values

    # Determine edges
    if edges is None:
        assert spec.edges is not None
        edge_list = list(spec.edges.edges)
        if spec.edges.symmetric:
            # Add reverse edges
            reverse = [(j, i) for (i, j) in edge_list if (j, i) not in set(edge_list)]
            edge_list = edge_list + reverse
    else:
        edge_list = list(edges)

    # Effective eta_min
    if eta_min is None:
        eff_eta_min = num_cfg.eta_min(epsilon)
    else:
        eff_eta_min = eta_min

    is_pure_dp = delta == 0.0
    is_l2 = spec.loss_fn == LossFunction.L2

    # Compute auxiliary variable count
    n_aux = 0
    slack_tracker = _ApproxDPSlackTracker()

    if not is_pure_dp:
        # Each edge pair (directed) needs 2k slacks (k forward + k backward per directed edge)
        # But build_approx_dp_constraints handles fwd+bwd for an undirected pair,
        # so we need to track unique undirected pairs
        seen_pairs: Set[Tuple[int, int]] = set()
        for i, ip in edge_list:
            canon = (min(i, ip), max(i, ip))
            if canon not in seen_pairs:
                seen_pairs.add(canon)
                slack_tracker.allocate(canon, 2 * k)
        n_aux += slack_tracker.total_slacks

    l2_aux_offset = n_aux
    if is_l2:
        n_aux += n  # z_i variables for piecewise L2

    layout = VariableLayout(n=n, k=k, n_aux=n_aux)

    # Build output grid
    y_grid = build_output_grid(f_values, k)

    # Build loss matrix
    loss_matrix = build_loss_matrix(
        f_values, y_grid, spec.loss_fn, custom_loss=spec.custom_loss
    )

    # -- Objective and epigraph constraints ---------------------------------

    if is_l2:
        # Piecewise-linear L2 approximation
        c = np.zeros(layout.n_vars, dtype=np.float64)
        c[layout.t_index] = 1.0

        A_epi, b_epi, A_tang, b_tang = build_l2_piecewise_constraints(
            f_values, y_grid, n, k, layout, l2_aux_offset, l2_segments
        )
        obj_ub_blocks = [A_epi, A_tang]
        obj_rhs_blocks = [b_epi, b_tang]
    else:
        c, A_obj, b_obj = build_minimax_objective(loss_matrix, n, k, layout)
        obj_ub_blocks = [A_obj]
        obj_rhs_blocks = [b_obj]

    # -- DP constraints -----------------------------------------------------

    dp_ub_blocks: List[coo_matrix] = []
    dp_rhs_blocks: List[npt.NDArray[np.float64]] = []

    if is_pure_dp:
        seen_pairs_pure: Set[Tuple[int, int]] = set()
        for i, ip in edge_list:
            canon = (min(i, ip), max(i, ip))
            if canon in seen_pairs_pure:
                continue
            seen_pairs_pure.add(canon)
            A_dp, b_dp = build_pure_dp_constraints(i, ip, k, epsilon, layout)
            dp_ub_blocks.append(A_dp)
            dp_rhs_blocks.append(b_dp)
    else:
        seen_pairs_approx: Set[Tuple[int, int]] = set()
        for i, ip in edge_list:
            canon = (min(i, ip), max(i, ip))
            if canon in seen_pairs_approx:
                continue
            seen_pairs_approx.add(canon)
            slack_off = slack_tracker.allocations[canon]
            A_dp, b_dp = build_approx_dp_constraints(
                canon[0], canon[1], k, epsilon, delta, layout, slack_off
            )
            dp_ub_blocks.append(A_dp)
            dp_rhs_blocks.append(b_dp)

    # -- Simplex constraints ------------------------------------------------

    A_eq, b_eq = build_simplex_constraints(n, k, layout)

    # -- Stack all inequality blocks ----------------------------------------

    all_ub_blocks = obj_ub_blocks + dp_ub_blocks
    all_rhs_blocks = obj_rhs_blocks + dp_rhs_blocks

    if len(all_ub_blocks) > 0:
        A_ub = sparse.vstack(
            [blk.tocsr() for blk in all_ub_blocks], format="csr"
        )
        b_ub = np.concatenate(all_rhs_blocks)
    else:
        A_ub = csr_matrix((0, layout.n_vars), dtype=np.float64)
        b_ub = np.empty(0, dtype=np.float64)

    # -- Variable bounds ----------------------------------------------------

    bounds = build_variable_bounds(layout, eff_eta_min)

    # -- Build var_map ------------------------------------------------------

    var_map = build_var_map(n, k, layout)

    # -- Assemble LPStruct --------------------------------------------------

    lp = LPStruct(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        var_map=var_map,
        y_grid=y_grid,
    )

    build_time = time.perf_counter() - t_start

    # Constraint counts by type
    n_obj_rows = sum(blk.shape[0] for blk in obj_ub_blocks)
    n_dp_rows = sum(blk.shape[0] for blk in dp_ub_blocks)

    metadata = {
        "n": n,
        "k": k,
        "n_vars": layout.n_vars,
        "n_ub": lp.n_ub,
        "n_eq": lp.n_eq,
        "n_obj_rows": n_obj_rows,
        "n_dp_rows": n_dp_rows,
        "n_edges": len(edge_list),
        "n_unique_pairs": len(seen_pairs_pure) if is_pure_dp else len(seen_pairs_approx),
        "is_pure_dp": is_pure_dp,
        "eta_min": eff_eta_min,
        "loss_fn": spec.loss_fn.name,
        "sparsity": lp.sparsity,
        "build_time": build_time,
    }

    logger.info(
        "Built privacy LP: %d vars, %d ub constraints, %d eq constraints, "
        "sparsity=%.2f%%, build_time=%.3fs",
        layout.n_vars,
        lp.n_ub,
        lp.n_eq,
        lp.sparsity * 100,
        build_time,
    )

    return lp, layout, metadata


# ---------------------------------------------------------------------------
# Constraint history entry
# ---------------------------------------------------------------------------


@dataclass
class ConstraintHistoryEntry:
    """Record of a constraint addition/removal in the CEGIS loop.

    Attributes:
        iteration: CEGIS iteration when this change occurred.
        action: "add" or "remove".
        pair: The (i, i') adjacent pair.
        n_rows_added: Number of inequality rows added (or removed).
        objective_before: Objective value before the change (if known).
        objective_after: Objective value after the change (if known).
    """

    iteration: int
    action: str
    pair: Tuple[int, int]
    n_rows_added: int
    objective_before: Optional[float] = None
    objective_after: Optional[float] = None


# ---------------------------------------------------------------------------
# LP Manager — incremental constraint management for CEGIS
# ---------------------------------------------------------------------------


class LPManager:
    """Manages LP state across CEGIS iterations.

    The CEGIS loop works by iteratively adding counterexample pairs to the
    LP and re-solving.  ``LPManager`` provides:

    - Incremental constraint addition (:meth:`add_constraints`) without
      rebuilding the entire LP from scratch.
    - Constraint removal (:meth:`remove_constraints`) for constraint
      management heuristics.
    - Warm-start state (:meth:`warm_start_from_previous`) to exploit dual
      simplex efficiency across iterations.
    - Constraint history tracking for convergence analysis.

    Usage::

        mgr = LPManager(spec)
        mgr.add_constraints(0, 1)   # first counterexample pair
        stats = mgr.solve()
        P = mgr.get_mechanism_table()

        mgr.add_constraints(2, 3)   # second counterexample pair
        stats = mgr.solve()         # warm-started from previous solution

    Parameters
    ----------
    spec : QuerySpec
        Query specification (defines n, k, ε, δ, loss, etc.).
    initial_edges : list of (int, int), optional
        Initial set of adjacent pairs.  If ``None``, starts with an empty
        constraint set (typical for CEGIS).
    synthesis_config : SynthesisConfig, optional
        Synthesis configuration (solver, tolerances, etc.).
    l2_segments : int
        Number of piecewise-linear segments for L2 loss approximation.
    """

    def __init__(
        self,
        spec: QuerySpec,
        initial_edges: Optional[List[Tuple[int, int]]] = None,
        synthesis_config: Optional[SynthesisConfig] = None,
        l2_segments: int = _PIECEWISE_L2_DEFAULT_SEGMENTS,
    ) -> None:
        self._spec = spec
        self._config = synthesis_config or SynthesisConfig()
        self._l2_segments = l2_segments

        self._n = spec.n
        self._k = spec.k
        self._epsilon = spec.epsilon
        self._delta = spec.delta
        self._is_pure_dp = spec.delta == 0.0

        # Effective eta_min
        self._eta_min = self._config.effective_eta_min(self._epsilon)

        # Active constraint pairs (canonical form: min, max)
        self._active_pairs: Set[Tuple[int, int]] = set()

        # Constraint history
        self._history: List[ConstraintHistoryEntry] = []
        self._iteration: int = 0

        # Warm-start state
        self._last_solution: Optional[npt.NDArray[np.float64]] = None
        self._last_basis: Optional[Dict[str, Any]] = None
        self._last_stats: Optional[SolveStatistics] = None
        self._last_objective: Optional[float] = None

        # Pre-compute the output grid (fixed across iterations)
        self._y_grid = build_output_grid(spec.query_values, spec.k)

        # Pre-compute loss matrix (fixed across iterations)
        self._loss_matrix = build_loss_matrix(
            spec.query_values,
            self._y_grid,
            spec.loss_fn,
            custom_loss=spec.custom_loss,
        )

        # Slack tracker for approximate DP
        self._slack_tracker = _ApproxDPSlackTracker()

        # Add initial edges
        if initial_edges is not None:
            for i, ip in initial_edges:
                self._register_pair(i, ip)

        # Build initial LP
        self._rebuild_lp()

    def _canonical_pair(self, i: int, i_prime: int) -> Tuple[int, int]:
        """Return canonical (min, max) form of an edge pair."""
        return (min(i, i_prime), max(i, i_prime))

    def _register_pair(self, i: int, i_prime: int) -> bool:
        """Register a pair; return True if it was new."""
        canon = self._canonical_pair(i, i_prime)
        if canon in self._active_pairs:
            return False
        self._active_pairs.add(canon)
        if not self._is_pure_dp:
            self._slack_tracker.allocate(canon, 2 * self._k)
        return True

    def _rebuild_lp(self) -> None:
        """Rebuild the LP from the current active constraint set.

        This is called after structural changes (add/remove) that change
        the number of variables (e.g., approximate DP slacks).
        """
        n, k = self._n, self._k
        is_l2 = self._spec.loss_fn == LossFunction.L2

        # Recount auxiliaries
        n_aux = self._slack_tracker.total_slacks

        self._layout = VariableLayout(n=n, k=k, n_aux=n_aux)
        layout = self._layout

        # -- Objective / epigraph -------------------------------------------
        # L2 loss E[(f_i - Y)^2] = Σ_j (f_i - y_j)^2 · p[i][j] is already
        # linear in p[i][j], so we use the direct minimax formulation for all
        # loss functions (the loss matrix entries are precomputed constants).
        c, A_obj, b_obj = build_minimax_objective(
            self._loss_matrix, n, k, layout
        )
        obj_blocks = [A_obj]
        obj_rhs = [b_obj]

        self._c = c

        # -- DP constraints -------------------------------------------------
        dp_blocks: List[coo_matrix] = []
        dp_rhs: List[npt.NDArray[np.float64]] = []

        for canon in sorted(self._active_pairs):
            i, ip = canon
            if self._is_pure_dp:
                A_dp, b_dp = build_pure_dp_constraints(i, ip, k, self._epsilon, layout)
            else:
                slack_off = self._slack_tracker.allocations[canon]
                A_dp, b_dp = build_approx_dp_constraints(
                    i, ip, k, self._epsilon, self._delta, layout, slack_off
                )
            dp_blocks.append(A_dp)
            dp_rhs.append(b_dp)

        # -- Stack ----------------------------------------------------------
        all_blocks = obj_blocks + dp_blocks
        all_rhs = obj_rhs + dp_rhs

        if all_blocks:
            self._A_ub = sparse.vstack(
                [b.tocsr() for b in all_blocks], format="csr"
            )
            self._b_ub = np.concatenate(all_rhs)
        else:
            self._A_ub = csr_matrix((0, layout.n_vars), dtype=np.float64)
            self._b_ub = np.empty(0, dtype=np.float64)

        # -- Simplex --------------------------------------------------------
        self._A_eq, self._b_eq = build_simplex_constraints(n, k, layout)

        # -- Bounds ---------------------------------------------------------
        self._bounds = build_variable_bounds(layout, self._eta_min)

        # -- Var map --------------------------------------------------------
        self._var_map = build_var_map(n, k, layout)

    def add_constraints(self, i: int, i_prime: int) -> bool:
        """Add DP constraints for the adjacent pair (i, i').

        If the pair is already in the active set, this is a no-op and
        returns ``False``.

        Parameters
        ----------
        i, i_prime : int
            Database indices of the adjacent pair.

        Returns
        -------
        added : bool
            ``True`` if the pair was newly added.
        """
        obj_before = self._last_objective
        was_new = self._register_pair(i, i_prime)

        if not was_new:
            logger.debug(
                "Pair (%d, %d) already in active set, skipping", i, i_prime
            )
            return False

        # Rebuild LP with new pair
        self._rebuild_lp()

        # Invalidate warm-start if variable count changed
        if self._last_solution is not None:
            if len(self._last_solution) != self._layout.n_vars:
                self._last_solution = None
                self._last_basis = None

        self._history.append(
            ConstraintHistoryEntry(
                iteration=self._iteration,
                action="add",
                pair=self._canonical_pair(i, i_prime),
                n_rows_added=2 * self._k if self._is_pure_dp else 2 * (self._k + 1),
                objective_before=obj_before,
            )
        )

        logger.info(
            "Added DP constraints for pair (%d, %d), "
            "active pairs: %d, total ub constraints: %d",
            i, i_prime, len(self._active_pairs), self._A_ub.shape[0],
        )

        return True

    def remove_constraints(self, i: int, i_prime: int) -> bool:
        """Remove DP constraints for the adjacent pair (i, i').

        Parameters
        ----------
        i, i_prime : int
            Database indices of the pair to remove.

        Returns
        -------
        removed : bool
            ``True`` if the pair was actually removed.
        """
        canon = self._canonical_pair(i, i_prime)
        if canon not in self._active_pairs:
            return False

        self._active_pairs.discard(canon)

        # Rebuild slack tracker without this pair
        old_tracker = self._slack_tracker
        self._slack_tracker = _ApproxDPSlackTracker()
        for p in sorted(self._active_pairs):
            if not self._is_pure_dp:
                self._slack_tracker.allocate(p, 2 * self._k)

        self._rebuild_lp()
        self._last_solution = None
        self._last_basis = None

        self._history.append(
            ConstraintHistoryEntry(
                iteration=self._iteration,
                action="remove",
                pair=canon,
                n_rows_added=-(2 * self._k if self._is_pure_dp else 2 * (self._k + 1)),
                objective_before=self._last_objective,
            )
        )

        logger.info(
            "Removed DP constraints for pair (%d, %d), "
            "active pairs: %d",
            canon[0], canon[1], len(self._active_pairs),
        )

        return True

    def get_current_lp(self) -> LPStruct:
        """Return the current LP as an ``LPStruct``.

        Returns
        -------
        lp : LPStruct
        """
        return LPStruct(
            c=self._c,
            A_ub=self._A_ub,
            b_ub=self._b_ub,
            A_eq=self._A_eq,
            b_eq=self._b_eq,
            bounds=self._bounds,
            var_map=self._var_map,
            y_grid=self._y_grid,
        )

    def solve(
        self,
        solver: Optional[SolverBackend] = None,
        *,
        time_limit: float = 300.0,
    ) -> SolveStatistics:
        """Solve the current LP.

        Uses warm-start data from the previous solve if available and
        the variable count hasn't changed.

        Parameters
        ----------
        solver : SolverBackend, optional
            Override the solver backend from the synthesis config.
        time_limit : float
            Maximum solve time.

        Returns
        -------
        stats : SolveStatistics
        """
        lp = self.get_current_lp()
        backend = solver or self._config.solver

        warm = None
        if self._last_basis is not None:
            warm = self._last_basis

        stats = solve_lp(
            lp,
            backend,
            warm_start=warm,
            solver_tol=self._config.numerical.solver_tol,
            time_limit=time_limit,
            verbose=self._config.verbose,
        )

        self._last_solution = stats.primal_solution
        self._last_basis = stats.basis_info
        self._last_stats = stats
        self._last_objective = stats.objective_value
        self._iteration += 1

        # Update history entry
        if self._history and self._history[-1].objective_after is None:
            self._history[-1].objective_after = stats.objective_value

        return stats

    def get_mechanism_table(self) -> npt.NDArray[np.float64]:
        """Extract the mechanism table from the last solve.

        Returns
        -------
        P : array of shape (n, k)
            Mechanism probability table.

        Raises
        ------
        RuntimeError
            If no solve has been performed yet.
        """
        if self._last_solution is None:
            raise RuntimeError("No LP solution available. Call solve() first.")
        return extract_mechanism_table(self._last_solution, self._layout)

    def get_last_solution(self) -> Optional[npt.NDArray[np.float64]]:
        """Return the raw LP solution vector from the last solve, or None."""
        return self._last_solution

    def get_last_objective(self) -> Optional[float]:
        """Return the objective value from the last solve, or None."""
        return self._last_objective

    def get_last_stats(self) -> Optional[SolveStatistics]:
        """Return full solve statistics from the last solve, or None."""
        return self._last_stats

    def warm_start_from_previous(
        self, solution: npt.NDArray[np.float64], basis: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set warm-start state from a previous solution.

        This is useful when the CEGIS loop wants to seed the LP manager
        with a solution from a different configuration (e.g., different k).

        Parameters
        ----------
        solution : array of shape (n_vars,)
            Primal solution vector.
        basis : dict, optional
            Solver-specific basis information.
        """
        if len(solution) != self._layout.n_vars:
            logger.warning(
                "Warm-start solution has %d variables, LP has %d. Ignoring.",
                len(solution), self._layout.n_vars,
            )
            return
        self._last_solution = solution.copy()
        self._last_basis = basis if basis is not None else {"x": solution.copy()}

    def track_constraint_history(self) -> List[ConstraintHistoryEntry]:
        """Return the full constraint addition/removal history.

        Returns
        -------
        history : list of ConstraintHistoryEntry
        """
        return list(self._history)

    @property
    def active_pairs(self) -> Set[Tuple[int, int]]:
        """Currently active adjacent pairs (canonical form)."""
        return set(self._active_pairs)

    @property
    def n_active_pairs(self) -> int:
        """Number of currently active adjacent pairs."""
        return len(self._active_pairs)

    @property
    def layout(self) -> VariableLayout:
        """Current variable layout."""
        return self._layout

    @property
    def y_grid(self) -> npt.NDArray[np.float64]:
        """Output discretisation grid."""
        return self._y_grid.copy()

    @property
    def iteration(self) -> int:
        """Current CEGIS iteration counter."""
        return self._iteration

    def has_pair(self, i: int, i_prime: int) -> bool:
        """Check if a pair is already in the active constraint set."""
        return self._canonical_pair(i, i_prime) in self._active_pairs

    def validate_last_solution(self, tol: float = 1e-6) -> Dict[str, Any]:
        """Validate the last LP solution.

        Returns
        -------
        result : dict
            Validation result from :func:`validate_solution`.

        Raises
        ------
        RuntimeError
            If no solve has been performed yet.
        """
        if self._last_solution is None:
            raise RuntimeError("No LP solution available. Call solve() first.")
        return validate_solution(
            self._last_solution,
            self._layout,
            self._A_ub,
            self._b_ub,
            self._A_eq,
            self._b_eq,
            tol=tol,
        )

    def diagnose_infeasibility(self) -> Dict[str, Any]:
        """Attempt to diagnose why the current LP is infeasible.

        Returns
        -------
        diagnosis : dict
            Diagnosis from :func:`diagnose_infeasibility`.
        """
        return diagnose_infeasibility(
            self._A_ub,
            self._b_ub,
            self._A_eq,
            self._b_eq,
            self._bounds,
            self._n,
            self._k,
            self._epsilon,
            self._delta,
            self._layout,
        )

    def estimate_condition_number(self) -> float:
        """Estimate the condition number of the current constraint matrix.

        Returns
        -------
        cond : float
        """
        if self._A_ub.shape[0] == 0:
            return 1.0
        return estimate_condition_number(self._A_ub.tocsr())

    def check_numerical_stability(self) -> Dict[str, Any]:
        """Run numerical stability checks on the current LP.

        Returns
        -------
        result : dict
            ``condition_number``: estimated condition number.
            ``is_stable``: True if below configured max_condition_number.
            ``degeneracy``: degeneracy info (if a solution is available).
        """
        cond = self.estimate_condition_number()
        max_cond = self._config.numerical.max_condition_number

        result: Dict[str, Any] = {
            "condition_number": cond,
            "max_condition_number": max_cond,
            "is_stable": cond < max_cond,
        }

        if self._last_solution is not None:
            result["degeneracy"] = detect_degeneracy(
                self._A_ub.tocsr(),
                self._b_ub,
                self._last_solution,
            )

        if not result["is_stable"]:
            logger.warning(
                "Constraint matrix condition number %.2e exceeds threshold %.2e",
                cond,
                max_cond,
            )

        return result

    def summary(self) -> str:
        """Return a human-readable summary of the current LP state."""
        lines = [
            f"LPManager Summary (iteration {self._iteration})",
            f"  Query: n={self._n}, k={self._k}",
            f"  Privacy: ε={self._epsilon}, δ={self._delta} "
            f"({'pure' if self._is_pure_dp else 'approximate'} DP)",
            f"  Loss: {self._spec.loss_fn.name}",
            f"  Active pairs: {self.n_active_pairs}",
            f"  Variables: {self._layout.n_vars}",
            f"  Inequality constraints: {self._A_ub.shape[0]}",
            f"  Equality constraints: {self._A_eq.shape[0]}",
            f"  Sparsity: {self.get_current_lp().sparsity:.2%}",
        ]
        if self._last_objective is not None:
            lines.append(f"  Last objective: {self._last_objective:.8f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"LPManager(n={self._n}, k={self._k}, "
            f"pairs={self.n_active_pairs}, iter={self._iteration})"
        )


# ---------------------------------------------------------------------------
# Convenience: build-and-solve in one call
# ---------------------------------------------------------------------------


def build_and_solve_privacy_lp(
    spec: QuerySpec,
    edges: Optional[List[Tuple[int, int]]] = None,
    *,
    solver: SolverBackend = SolverBackend.AUTO,
    eta_min: Optional[float] = None,
    numerical_config: Optional[NumericalConfig] = None,
    verbose: int = 0,
) -> Tuple[npt.NDArray[np.float64], float, SolveStatistics, Dict[str, Any]]:
    """Build and solve the privacy LP in a single call.

    This is a convenience wrapper for non-CEGIS use cases where you want
    to solve the full LP with all edges at once.

    Parameters
    ----------
    spec : QuerySpec
        Query specification.
    edges : list of (int, int), optional
        Adjacent pairs.
    solver : SolverBackend
        Solver backend.
    eta_min : float, optional
        Minimum probability floor override.
    numerical_config : NumericalConfig, optional
        Numerical configuration.
    verbose : int
        Verbosity level.

    Returns
    -------
    P : array of shape (n, k)
        Mechanism probability table.
    obj_val : float
        Optimal minimax objective value.
    stats : SolveStatistics
        Solve statistics.
    metadata : dict
        LP construction metadata.
    """
    lp, layout, metadata = build_privacy_lp(
        spec,
        edges=edges,
        eta_min=eta_min,
        numerical_config=numerical_config,
    )

    stats = solve_lp(
        lp,
        solver,
        solver_tol=(numerical_config or NumericalConfig()).solver_tol,
        verbose=verbose,
    )

    P = extract_mechanism_table(stats.primal_solution, layout)

    return P, stats.objective_value, stats, metadata


# ---------------------------------------------------------------------------
# Laplace warm-start initialisation
# ---------------------------------------------------------------------------


def build_laplace_warm_start(
    spec: QuerySpec,
    y_grid: npt.NDArray[np.float64],
    layout: VariableLayout,
) -> npt.NDArray[np.float64]:
    """Build a Laplace mechanism as a warm-start initial point.

    The Laplace mechanism with scale ``b = sensitivity / ε`` assigns
    probability proportional to ``exp(−|y − f(x_i)| / b)`` to each
    output bin.  This provides a feasible starting point for the LP that
    typically accelerates convergence.

    Parameters
    ----------
    spec : QuerySpec
        Query specification.
    y_grid : array of shape (k,)
        Output discretisation grid.
    layout : VariableLayout
        Variable layout.

    Returns
    -------
    x0 : array of shape (n_vars,)
        Initial point with Laplace probabilities and t set to the worst
        expected loss.
    """
    n, k = layout.n, layout.k
    b = spec.sensitivity / spec.epsilon

    x0 = np.zeros(layout.n_vars, dtype=np.float64)

    max_expected_loss = 0.0
    loss_fn = spec.get_loss_callable()

    for i in range(n):
        f_i = float(spec.query_values[i])

        # Un-normalised Laplace probabilities
        log_probs = -np.abs(y_grid - f_i) / b
        log_probs -= np.max(log_probs)  # shift for numerical stability
        probs = np.exp(log_probs)
        probs /= probs.sum()

        # Enforce eta_min
        eta_min = spec.eta_min
        probs = np.maximum(probs, eta_min)
        probs /= probs.sum()

        for j in range(k):
            x0[layout.p_index(i, j)] = probs[j]

        # Compute expected loss for this row
        expected_loss = sum(
            loss_fn(f_i, float(y_grid[j])) * probs[j] for j in range(k)
        )
        max_expected_loss = max(max_expected_loss, expected_loss)

    # Set t to the worst expected loss
    x0[layout.t_index] = max_expected_loss

    return x0


# ---------------------------------------------------------------------------
# Constraint scaling wrapper for LPManager
# ---------------------------------------------------------------------------


def build_scaled_privacy_lp(
    spec: QuerySpec,
    edges: Optional[List[Tuple[int, int]]] = None,
    *,
    eta_min: Optional[float] = None,
    numerical_config: Optional[NumericalConfig] = None,
    l2_segments: int = _PIECEWISE_L2_DEFAULT_SEGMENTS,
) -> Tuple[LPStruct, VariableLayout, npt.NDArray[np.float64], Dict[str, Any]]:
    """Build the privacy LP with row-scaled inequality constraints.

    Identical to :func:`build_privacy_lp` but applies row-scaling to the
    inequality constraint matrix for better numerical conditioning.
    Returns the scale factors so that dual variables can be recovered.

    Returns
    -------
    lp : LPStruct
        Row-scaled LP.
    layout : VariableLayout
        Variable layout.
    scale_factors : array
        Row scaling factors applied to A_ub and b_ub.
    metadata : dict
        Construction metadata.
    """
    lp, layout, metadata = build_privacy_lp(
        spec,
        edges=edges,
        eta_min=eta_min,
        numerical_config=numerical_config,
        l2_segments=l2_segments,
    )

    A_scaled, b_scaled, scale_factors = scale_constraints(
        lp.A_ub.tocsr(), lp.b_ub
    )

    lp_scaled = LPStruct(
        c=lp.c,
        A_ub=A_scaled,
        b_ub=b_scaled,
        A_eq=lp.A_eq,
        b_eq=lp.b_eq,
        bounds=lp.bounds,
        var_map=lp.var_map,
        y_grid=lp.y_grid,
    )

    metadata["scaled"] = True

    return lp_scaled, layout, scale_factors, metadata
