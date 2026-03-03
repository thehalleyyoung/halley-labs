"""
Symmetry detection and LP reduction for DP-Forge.

This module detects symmetries in the query structure and exploits them
to reduce the size of the LP formulation, yielding factor-n speedups
for symmetric queries (most importantly, counting queries).

Theoretical Background
~~~~~~~~~~~~~~~~~~~~~~

**Translation invariance.**  A counting query f(x) = |{d ∈ D : pred(d)}|
over a database of size N has the property that the *noise distribution*
η = M(x) − f(x) is the same for every true value x.  That is, the optimal
mechanism has the form M(x) = f(x) + η where η is a single noise
distribution over {y_1 − f(x), …, y_k − f(x)}.

Formally, if the adjacency graph is the path graph 0—1—2—…—(n−1) and the
query values are f(x_i) = i, then p[i][j] = η[j − i] (indices mod k) for
a single distribution η of length k.  This replaces n×k variables with k
variables and 2|E|×k constraints with 2k constraints.

**Proof of correctness.**  For a translation-invariant query with
sensitivity 1 and adjacency (i, i+1):

    p[i][j] / p[i+1][j] = η[j − i] / η[j − i − 1]

This ratio depends only on the *offset* j − i (mod k), not on i.  Therefore
the n DP constraint pairs (one for each edge) all reduce to the *same*
pair of constraints on adjacent noise entries:

    η[l] / η[l−1] ≤ e^ε     and     η[l−1] / η[l] ≤ e^ε     ∀ l

This is exactly 2k constraints on k variables.

**Reflection symmetry.**  If the loss function is symmetric
(L(t, y) = L(t, 2t − y)), the optimal noise distribution is symmetric
about 0: η[l] = η[−l].  This halves the number of free variables to
⌈k/2⌉.

**General permutation symmetries.**  For queries whose output set has
a non-trivial automorphism group G acting on the output bins, we can
restrict the LP to G-invariant mechanisms.  The number of free variables
equals the number of orbits of G on {0, …, k−1}.

Reduction Summary
~~~~~~~~~~~~~~~~~

+-------------------------+----------+------------+-----------+
| Symmetry type           | Vars     | Constraints| Speedup   |
+-------------------------+----------+------------+-----------+
| None                    | n·k + 1  | O(|E|·k)   | 1×        |
| Translation invariance  | k + 1    | 2k + 1     | ~n×       |
| + Reflection symmetry   | ⌈k/2⌉+1 | k + 1      | ~2n×      |
| General permutation     | |orbits| | O(|orbits|)| |G|×      |
+-------------------------+----------+------------+-----------+

Classes
-------
- :class:`SymmetryGroup` — Detected symmetry group with metadata.
- :class:`SymmetryDetector` — Auto-detect symmetry structure.
- :class:`TranslationReducer` — Reduce LP using translation invariance.
- :class:`ReflectionReducer` — Reduce LP using reflection symmetry.
- :class:`PermutationReducer` — Reduce LP using general permutations.
- :class:`PermutationGroup` — Group operations on permutations.
- :class:`ReconstructionMap` — Map reduced solutions back to full size.

Functions
---------
- :func:`ReduceBySymmetry` — Top-level symmetry reduction entry point.
- :func:`orbit_computation` — Compute orbits from group generators.
- :func:`stabilizer` — Compute stabiliser of an element.
- :func:`is_group` — Verify group axioms.
- :func:`expand_noise_to_full` — Reconstruct full table from noise dist.
- :func:`verify_reconstruction` — Check reconstruction correctness.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
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
from scipy.sparse import coo_matrix, csr_matrix

from dp_forge.exceptions import (
    ConfigurationError,
)
from dp_forge.types import (
    AdjacencyRelation,
    LPStruct,
    LossFunction,
    NumericalConfig,
    QuerySpec,
    QueryType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tolerance for checking whether query values form an arithmetic sequence.
_ARITHMETIC_TOL: float = 1e-10

# Tolerance for checking whether adjacency is a path graph.
_EDGE_TOL: float = 1e-12

# Tolerance for checking loss function symmetry.
_SYMMETRY_TOL: float = 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# §1  Symmetry Group Representation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SymmetryGroup:
    """Detected symmetry group with metadata.

    Represents the symmetry group of a query, which can be used to reduce
    the LP formulation.

    Attributes:
        is_translation_invariant: Whether the query is translation-invariant.
        is_reflection_symmetric: Whether the loss is reflection-symmetric.
        generators: Permutation generators for the symmetry group.
        order: Size of the symmetry group.
        group_type: Human-readable description of the group.
        metadata: Additional metadata about the detection.
    """

    is_translation_invariant: bool = False
    is_reflection_symmetric: bool = False
    generators: List[npt.NDArray[np.intp]] = field(default_factory=list)
    order: int = 1
    group_type: str = "trivial"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_symmetry(self) -> bool:
        """Whether any non-trivial symmetry was detected."""
        return self.is_translation_invariant or self.is_reflection_symmetric or self.order > 1

    @property
    def theoretical_speedup(self) -> float:
        """Theoretical LP size reduction factor."""
        if self.is_translation_invariant and self.is_reflection_symmetric:
            n = self.metadata.get("n", 1)
            return float(2 * n)
        elif self.is_translation_invariant:
            n = self.metadata.get("n", 1)
            return float(n)
        elif self.is_reflection_symmetric:
            return 2.0
        return float(self.order)

    def __repr__(self) -> str:
        return (
            f"SymmetryGroup(type={self.group_type!r}, order={self.order}, "
            f"translation={self.is_translation_invariant}, "
            f"reflection={self.is_reflection_symmetric}, "
            f"speedup={self.theoretical_speedup:.1f}×)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# §2  Symmetry Detection
# ═══════════════════════════════════════════════════════════════════════════


class SymmetryDetector:
    """Auto-detect symmetry structure in a query specification.

    The detector checks for:
    1. Translation invariance (counting queries on path graphs).
    2. Reflection symmetry (symmetric loss functions).
    3. General permutation symmetries of the output space.

    Parameters
    ----------
    spec : QuerySpec, optional
        Full query specification.  If provided, uses spec fields directly.

    Examples
    --------
    >>> spec = QuerySpec.counting(10, epsilon=1.0)
    >>> detector = SymmetryDetector(spec=spec)
    >>> group = detector.detect()
    >>> group.is_translation_invariant
    True
    """

    def __init__(self, spec: Optional[QuerySpec] = None) -> None:
        self._spec = spec

    def detect(
        self,
        f_values: Optional[npt.NDArray[np.float64]] = None,
        edges: Optional[Union[List[Tuple[int, int]], AdjacencyRelation]] = None,
        loss_fn: Optional[LossFunction] = None,
        query_type: Optional[QueryType] = None,
    ) -> SymmetryGroup:
        """Auto-detect the symmetry group of the query.

        Checks translation invariance and reflection symmetry.  Falls back
        to trivial group if no structure is detected.

        Parameters
        ----------
        f_values : array of shape (n,), optional
            Query output values.  Uses spec if not provided.
        edges : list of (int, int) or AdjacencyRelation, optional
            Adjacency relation.  Uses spec if not provided.
        loss_fn : LossFunction, optional
            Loss function.  Uses spec if not provided.
        query_type : QueryType, optional
            Query type hint.  Uses spec if not provided.

        Returns
        -------
        SymmetryGroup
            Detected symmetry group.
        """
        # Resolve parameters from spec if not given
        if f_values is None and self._spec is not None:
            f_values = self._spec.query_values
        if edges is None and self._spec is not None:
            edges = self._spec.edges
        if loss_fn is None and self._spec is not None:
            loss_fn = self._spec.loss_fn
        if query_type is None and self._spec is not None:
            query_type = self._spec.query_type

        if f_values is None:
            raise ConfigurationError(
                "f_values must be provided either directly or via spec",
                parameter="f_values",
            )

        f_values = np.asarray(f_values, dtype=np.float64)
        n = len(f_values)

        # Normalise edges
        edge_list: List[Tuple[int, int]]
        if isinstance(edges, AdjacencyRelation):
            edge_list = edges.edges
        elif edges is not None:
            edge_list = list(edges)
        else:
            edge_list = [(i, i + 1) for i in range(n - 1)]

        t_start = time.perf_counter()

        # Check translation invariance
        trans_inv = self.is_translation_invariant(f_values, edge_list, query_type)

        # Check reflection symmetry
        refl_sym = self.is_reflection_symmetric(f_values, loss_fn)

        # Determine group type and order
        if trans_inv and refl_sym:
            group_type = "translation+reflection"
            order = 2 * n
        elif trans_inv:
            group_type = "translation"
            order = n
        elif refl_sym:
            group_type = "reflection"
            order = 2
        else:
            group_type = "trivial"
            order = 1

        # Build generators
        generators: List[npt.NDArray[np.intp]] = []
        if trans_inv:
            # Translation generator: shift by 1
            gen = np.roll(np.arange(n, dtype=np.intp), -1)
            generators.append(gen)
        if refl_sym:
            # Reflection generator: reverse
            gen = np.arange(n, dtype=np.intp)[::-1].copy()
            generators.append(gen)

        t_elapsed = time.perf_counter() - t_start

        group = SymmetryGroup(
            is_translation_invariant=trans_inv,
            is_reflection_symmetric=refl_sym,
            generators=generators,
            order=order,
            group_type=group_type,
            metadata={
                "n": n,
                "detection_time_s": t_elapsed,
            },
        )

        logger.info(
            "Symmetry detection: %s (order=%d, speedup=%.1f×) in %.3fs",
            group_type, order, group.theoretical_speedup, t_elapsed,
        )

        return group

    def is_translation_invariant(
        self,
        f_values: npt.NDArray[np.float64],
        edges: List[Tuple[int, int]],
        query_type: Optional[QueryType] = None,
    ) -> bool:
        """Check whether the query is translation-invariant.

        A query is translation-invariant if:
        1. The query values form an arithmetic sequence f(x_i) = a + b·i
           (typically f(x_i) = i for counting queries).
        2. The adjacency graph is a path graph on {0, …, n−1}: exactly
           the edges {(i, i+1) : i = 0, …, n−2}.

        These two conditions together imply that the optimal mechanism
        has the form p[i][j] = η[j − i] for a single noise distribution
        η, reducing n×k variables to k.

        **Proof.**  If f(x_i) = a + b·i and adjacency is (i, i+1), then
        the DP constraint for pair (i, i+1) at output bin y_j is:

            p[i][j] ≤ e^ε · p[i+1][j]   and   p[i+1][j] ≤ e^ε · p[i][j]

        With the substitution p[i][j] = η[j − i], this becomes:

            η[j−i] ≤ e^ε · η[j−i−1]   and   η[j−i−1] ≤ e^ε · η[j−i]

        Setting l = j − i, these are constraints on consecutive entries
        of η, independent of i.  Therefore all n−1 edge constraints reduce
        to the same 2k constraints on adjacent entries of η.

        The loss constraint is similarly translation-invariant: for L2 loss,

            Σ_j (y_j − f(x_i))² · p[i][j]  =  Σ_j (y_j − i)² · η[j − i]
                                              =  Σ_l (y_{l+i} − i)² · η[l]

        If the grid is also shifted (y_j = y_0 + j·Δ for uniform grid),
        then y_{l+i} − i = y_0 + (l+i)·Δ − i = y_0 + l·Δ + i·(Δ − 1).
        For the canonical grid with Δ = 1, this is y_0 + l and the loss
        is independent of i.

        Parameters
        ----------
        f_values : array of shape (n,)
            Query output values.
        edges : list of (int, int)
            Adjacency edges.
        query_type : QueryType, optional
            If COUNTING, skip detailed checks and return True directly
            (counting queries are translation-invariant by construction).

        Returns
        -------
        bool
            True if the query is translation-invariant.
        """
        # Fast path for known counting queries
        if query_type == QueryType.COUNTING:
            n = len(f_values)
            # Verify it's actually a counting query with path adjacency
            expected_edges = set((i, i + 1) for i in range(n - 1))
            actual_edges = set((min(a, b), max(a, b)) for a, b in edges)
            if actual_edges == expected_edges:
                return True

        f_values = np.asarray(f_values, dtype=np.float64)
        n = len(f_values)

        if n < 2:
            return True  # Trivially translation-invariant

        # Check 1: f_values form an arithmetic sequence
        diffs = np.diff(f_values)
        if not np.allclose(diffs, diffs[0], atol=_ARITHMETIC_TOL):
            return False

        # Check 2: Adjacency is a path graph {(i, i+1) : i = 0, …, n-2}
        expected_edges: Set[Tuple[int, int]] = set()
        for i in range(n - 1):
            expected_edges.add((i, i + 1))

        actual_edges: Set[Tuple[int, int]] = set()
        for a, b in edges:
            actual_edges.add((min(a, b), max(a, b)))

        return actual_edges == expected_edges

    def is_reflection_symmetric(
        self,
        f_values: npt.NDArray[np.float64],
        loss_fn: Optional[LossFunction] = None,
    ) -> bool:
        """Check whether the loss function is reflection-symmetric.

        A loss function L(t, y) is reflection-symmetric if
        L(t, t + d) = L(t, t − d) for all t and d.  This holds for
        L1, L2, and Linf losses.

        When the loss is symmetric and the query is translation-invariant,
        the optimal noise distribution is symmetric about 0:
        η[l] = η[−l].  This halves the free variables.

        **Proof.**  Let η* be an optimal noise distribution.  Define
        η'[l] = η*[−l] (reflection).  By loss symmetry,

            E[L(t, t + η')] = E[L(t, t − η*)] = E[L(t, t + η*)]

        So η' achieves the same loss.  By convexity of the feasible set,
        the average η'' = (η* + η')/2 is also feasible and achieves at
        most the same loss (by Jensen's).  But η'' is symmetric.
        Therefore there exists an optimal symmetric solution.

        Parameters
        ----------
        f_values : array of shape (n,)
            Query output values (not used directly, but may be needed
            for custom loss checks).
        loss_fn : LossFunction, optional
            Loss function to check.

        Returns
        -------
        bool
            True if the loss is reflection-symmetric.
        """
        if loss_fn is None:
            return False

        # L1, L2, Linf are all symmetric: |t - y| = |t - (2t - y)|,
        # (t - y)^2 = (t - (2t - y))^2
        symmetric_losses = {LossFunction.L1, LossFunction.L2, LossFunction.LINF}
        return loss_fn in symmetric_losses

    def find_orbits(
        self,
        f_values: npt.NDArray[np.float64],
        symmetry_group: SymmetryGroup,
    ) -> List[List[int]]:
        """Compute orbits of the symmetry group on input indices.

        The orbit of index i is the set {σ(i) : σ ∈ G} of all indices
        reachable from i under the group action.

        Parameters
        ----------
        f_values : array of shape (n,)
            Query output values.
        symmetry_group : SymmetryGroup
            Detected symmetry group.

        Returns
        -------
        orbits : list of list of int
            List of orbits.  Each orbit is a sorted list of indices.
        """
        n = len(f_values)
        if not symmetry_group.generators:
            return [[i] for i in range(n)]

        return orbit_computation(
            list(range(n)),
            symmetry_group.generators,
        )

    def symmetry_group_order(self, group: SymmetryGroup) -> int:
        """Return the order (size) of the symmetry group."""
        return group.order


# ═══════════════════════════════════════════════════════════════════════════
# §3  Translation Reducer
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ReducedLPStruct:
    """LP structure after symmetry reduction.

    Attributes:
        c: Objective coefficients for the reduced LP.
        A_ub: Inequality constraint matrix (sparse CSR).
        b_ub: Inequality RHS.
        A_eq: Equality constraint matrix (sparse CSR), or None.
        b_eq: Equality RHS, or None.
        bounds: Per-variable bounds.
        original_n: Original number of database inputs.
        original_k: Original number of output bins.
        reduction_type: Type of reduction applied.
        reduction_factor: How much smaller the reduced LP is.
        y_grid: Output discretisation grid.
        metadata: Reduction metadata.
    """

    c: npt.NDArray[np.float64]
    A_ub: sparse.spmatrix
    b_ub: npt.NDArray[np.float64]
    A_eq: Optional[sparse.spmatrix]
    b_eq: Optional[npt.NDArray[np.float64]]
    bounds: List[Tuple[float, float]]
    original_n: int
    original_k: int
    reduction_type: str
    reduction_factor: float
    y_grid: npt.NDArray[np.float64]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_vars(self) -> int:
        """Number of decision variables in the reduced LP."""
        return len(self.c)

    @property
    def n_ub(self) -> int:
        """Number of inequality constraints."""
        return self.A_ub.shape[0]

    @property
    def n_eq(self) -> int:
        """Number of equality constraints."""
        return self.A_eq.shape[0] if self.A_eq is not None else 0

    def __repr__(self) -> str:
        return (
            f"ReducedLPStruct(vars={self.n_vars}, ub={self.n_ub}, eq={self.n_eq}, "
            f"reduction={self.reduction_type!r}, factor={self.reduction_factor:.1f}×)"
        )


class TranslationReducer:
    """Reduce LP using translation invariance of counting queries.

    For a counting query f(x_i) = i with path-graph adjacency and k output
    bins, the full LP has n·k + 1 variables and O(n·k) constraints.  Under
    translation invariance, we replace the n row distributions with a single
    noise distribution η of length k, yielding k + 1 variables and 2k + 1
    constraints — an n× reduction.

    **Variable mapping.**  The reduced LP has variables:

        η[0], η[1], …, η[k−1], t

    where η[l] = p[i][i + l (mod k)] is the probability of outputting
    y_{i+l} when the true value is x_i.

    **Constraint mapping.**  The 2(n−1)k pure DP constraints reduce to 2k:

        η[l] − e^ε · η[l−1] ≤ 0     ∀ l ∈ {0, …, k−1}  (forward)
        η[l−1] − e^ε · η[l] ≤ 0     ∀ l ∈ {0, …, k−1}  (backward)

    where indices are taken modulo k for periodic boundary conditions, or
    l−1 is clamped to 0 for non-periodic (truncated) noise.

    The simplex constraint becomes: Σ_l η[l] = 1.

    The minimax objective becomes:  Σ_l L(0, y_l − y_0) · η[l] ≤ t
    (a single constraint, since the loss is the same for every input).

    Parameters
    ----------
    epsilon : float
        Privacy parameter ε.
    delta : float
        Privacy parameter δ.
    k : int
        Number of output bins.
    y_grid : array of shape (k,)
        Output discretisation grid.
    loss_fn : LossFunction
        Loss function.
    eta_min : float
        Minimum probability floor.
    periodic : bool
        Whether to use periodic (circular) boundary conditions for the
        noise distribution.  True for unbounded queries; False for
        bounded/truncated queries.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        k: int,
        y_grid: npt.NDArray[np.float64],
        loss_fn: LossFunction = LossFunction.L2,
        eta_min: float = 1e-18,
        periodic: bool = False,
    ) -> None:
        self.epsilon = epsilon
        self.delta = delta
        self.k = k
        self.y_grid = np.asarray(y_grid, dtype=np.float64)
        self.loss_fn = loss_fn
        self.eta_min = eta_min
        self.periodic = periodic

    def reduce(
        self,
        lp_struct: Optional[LPStruct] = None,
        n: Optional[int] = None,
    ) -> ReducedLPStruct:
        """Reduce the LP using translation invariance.

        The reduced LP has k+1 variables: η[0], …, η[k−1] and the
        epigraph variable t.

        Parameters
        ----------
        lp_struct : LPStruct, optional
            Original LP structure.  Used to extract n and y_grid if provided.
        n : int, optional
            Original number of database inputs (used for metadata).

        Returns
        -------
        ReducedLPStruct
            Reduced LP with k+1 variables and 2k+2 constraints.
        """
        t_start = time.perf_counter()
        k = self.k
        epsilon = self.epsilon

        if n is None and lp_struct is not None:
            n = len(lp_struct.y_grid)  # fallback
        if n is None:
            n = k  # guess

        n_vars = k + 1  # η[0..k-1] + t
        t_idx = k  # index of epigraph variable t

        # --- Objective: minimize t ---
        c = np.zeros(n_vars, dtype=np.float64)
        c[t_idx] = 1.0

        # --- Inequality constraints ---
        A_ub_rows: List[int] = []
        A_ub_cols: List[int] = []
        A_ub_data: List[float] = []
        b_ub_list: List[float] = []

        e_eps = math.exp(epsilon)
        is_pure = self.delta == 0.0
        row_idx = 0

        if is_pure:
            # Pure DP: η[l] - e^ε · η[l-1] ≤ 0 for each l
            # With non-periodic boundary: l runs from 1 to k-1
            # With periodic boundary: l runs from 0 to k-1 (wrapping)
            start_l = 0 if self.periodic else 1

            for l in range(start_l, k):
                l_prev = (l - 1) % k

                # Forward: η[l] - e^ε · η[l_prev] ≤ 0
                A_ub_rows.extend([row_idx, row_idx])
                A_ub_cols.extend([l, l_prev])
                A_ub_data.extend([1.0, -e_eps])
                b_ub_list.append(0.0)
                row_idx += 1

                # Backward: η[l_prev] - e^ε · η[l] ≤ 0
                A_ub_rows.extend([row_idx, row_idx])
                A_ub_cols.extend([l_prev, l])
                A_ub_data.extend([1.0, -e_eps])
                b_ub_list.append(0.0)
                row_idx += 1
        else:
            # Approximate DP: hockey-stick with slacks
            # For translation-invariant case, the constraint becomes:
            # Σ_l max(η[l] - e^ε · η[l-1], 0) ≤ δ
            # We linearise with slacks: n_vars grows by k
            # For simplicity in the reduced formulation, we add slack variables
            n_slacks = k
            n_vars_new = n_vars + n_slacks
            c_new = np.zeros(n_vars_new, dtype=np.float64)
            c_new[t_idx] = 1.0
            c = c_new

            start_l = 0 if self.periodic else 1

            for l in range(start_l, k):
                l_prev = (l - 1) % k
                slack_idx = n_vars + (l - start_l) if not self.periodic else n_vars + l

                # Forward: η[l] - e^ε · η[l_prev] - s[l] ≤ 0
                A_ub_rows.extend([row_idx, row_idx, row_idx])
                A_ub_cols.extend([l, l_prev, slack_idx])
                A_ub_data.extend([1.0, -e_eps, -1.0])
                b_ub_list.append(0.0)
                row_idx += 1

                # Backward: η[l_prev] - e^ε · η[l] - s[l] ≤ 0
                A_ub_rows.extend([row_idx, row_idx, row_idx])
                A_ub_cols.extend([l_prev, l, slack_idx])
                A_ub_data.extend([1.0, -e_eps, -1.0])
                b_ub_list.append(0.0)
                row_idx += 1

            # Budget: Σ_l s[l] ≤ δ
            for l in range(start_l, k):
                slack_idx = n_vars + (l - start_l) if not self.periodic else n_vars + l
                A_ub_rows.append(row_idx)
                A_ub_cols.append(slack_idx)
                A_ub_data.append(1.0)
            b_ub_list.append(self.delta)
            row_idx += 1

            n_vars = n_vars_new

        # --- Minimax loss constraint: Σ_l L(0, y_l - y_centre) · η[l] - t ≤ 0 ---
        loss_callable = self.loss_fn.fn
        if loss_callable is not None:
            y_centre = self.y_grid[len(self.y_grid) // 2]  # centre of grid
            for l in range(k):
                loss_val = loss_callable(0.0, self.y_grid[l] - y_centre)
                if abs(loss_val) > 1e-300:
                    A_ub_rows.append(row_idx)
                    A_ub_cols.append(l)
                    A_ub_data.append(loss_val)
            A_ub_rows.append(row_idx)
            A_ub_cols.append(t_idx)
            A_ub_data.append(-1.0)
            b_ub_list.append(0.0)
            row_idx += 1

        n_ub = row_idx
        A_ub = coo_matrix(
            (A_ub_data, (A_ub_rows, A_ub_cols)),
            shape=(n_ub, n_vars),
            dtype=np.float64,
        ).tocsr()
        b_ub = np.array(b_ub_list, dtype=np.float64)

        # --- Equality constraint: Σ_l η[l] = 1 ---
        A_eq_row = np.zeros((1, n_vars), dtype=np.float64)
        A_eq_row[0, :k] = 1.0
        A_eq = csr_matrix(A_eq_row)
        b_eq = np.array([1.0], dtype=np.float64)

        # --- Bounds ---
        bounds: List[Tuple[float, float]] = []
        for l in range(k):
            bounds.append((self.eta_min, 1.0))
        bounds.append((None, None))  # t is unbounded
        # Slack bounds if approximate DP
        if not is_pure and self.delta > 0:
            n_slack_vars = n_vars - k - 1
            for _ in range(n_slack_vars):
                bounds.append((0.0, None))

        original_n = n
        original_vars = original_n * k + 1
        reduction_factor = original_vars / n_vars if n_vars > 0 else 1.0

        t_elapsed = time.perf_counter() - t_start

        logger.info(
            "Translation reduction: %d vars → %d vars (%.1f× reduction) in %.3fs",
            original_vars, n_vars, reduction_factor, t_elapsed,
        )

        return ReducedLPStruct(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            original_n=original_n,
            original_k=k,
            reduction_type="translation",
            reduction_factor=reduction_factor,
            y_grid=self.y_grid,
            metadata={
                "epsilon": self.epsilon,
                "delta": self.delta,
                "periodic": self.periodic,
                "reduction_time_s": t_elapsed,
            },
        )

    def _build_noise_variables(self, k: int) -> npt.NDArray[np.float64]:
        """Build initial noise variable vector (uniform distribution).

        Parameters
        ----------
        k : int
            Number of output bins.

        Returns
        -------
        eta : array of shape (k,)
            Uniform noise distribution 1/k.
        """
        return np.full(k, 1.0 / k, dtype=np.float64)

    def _build_reduced_constraints(
        self,
        k: int,
        epsilon: float,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Build the 2k reduced DP constraint matrix for pure DP.

        Returns the constraint matrix A and RHS b such that A @ η ≤ b
        encodes the DP constraints on the noise distribution.

        Each constraint has the form:
            η[l] − e^ε · η[l−1] ≤ 0   (forward)
            η[l−1] − e^ε · η[l] ≤ 0   (backward)

        Parameters
        ----------
        k : int
            Number of noise bins.
        epsilon : float
            Privacy parameter ε.

        Returns
        -------
        A : array of shape (2*(k-1), k)
            Constraint matrix.
        b : array of shape (2*(k-1),)
            RHS vector (all zeros).
        """
        e_eps = math.exp(epsilon)
        n_constraints = 2 * (k - 1)
        A = np.zeros((n_constraints, k), dtype=np.float64)
        b = np.zeros(n_constraints, dtype=np.float64)

        for l in range(1, k):
            # Forward: η[l] - e^ε · η[l-1] ≤ 0
            row_fwd = 2 * (l - 1)
            A[row_fwd, l] = 1.0
            A[row_fwd, l - 1] = -e_eps

            # Backward: η[l-1] - e^ε · η[l] ≤ 0
            row_bwd = 2 * (l - 1) + 1
            A[row_bwd, l - 1] = 1.0
            A[row_bwd, l] = -e_eps

        return A, b

    def reconstruct(
        self,
        noise_dist: npt.NDArray[np.float64],
        f_values: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Reconstruct full n × k mechanism table from noise distribution.

        Given the noise distribution η and query values f(x_0), …, f(x_{n−1}),
        constructs p[i][j] = η[j − i_offset] where i_offset accounts for the
        mapping from query value f(x_i) to the grid.

        For a counting query with f(x_i) = i and uniform grid y_j = j:
            p[i][j] = η[j − i]   (with appropriate boundary handling)

        Parameters
        ----------
        noise_dist : array of shape (k,)
            Noise distribution from the reduced LP solution.
        f_values : array of shape (n,)
            Query output values.

        Returns
        -------
        p : array of shape (n, k)
            Full mechanism probability table.
        """
        noise_dist = np.asarray(noise_dist, dtype=np.float64)
        f_values = np.asarray(f_values, dtype=np.float64)
        n = len(f_values)
        k = len(noise_dist)

        p = np.zeros((n, k), dtype=np.float64)

        # Map f_values to grid indices
        grid = self.y_grid
        f_min = float(f_values[0])
        grid_min = float(grid[0])

        for i in range(n):
            # Offset: how many grid steps from f(x_i) to grid start
            offset = round((f_values[i] - grid_min) / max(
                (grid[-1] - grid[0]) / (k - 1) if k > 1 else 1.0, 1e-300
            ))
            offset = int(offset)

            for l in range(k):
                j = l + offset
                if 0 <= j < k:
                    p[i, j] = noise_dist[l]
                elif self.periodic:
                    p[i, j % k] += noise_dist[l]

        # Re-normalise in case of boundary effects
        row_sums = p.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-300)
        p /= row_sums

        return p

    def reduction_factor(self, n: int) -> float:
        """Compute the reduction factor for n inputs.

        Parameters
        ----------
        n : int
            Number of database inputs in the original problem.

        Returns
        -------
        float
            Ratio of original LP size to reduced LP size.
        """
        original = n * self.k + 1
        reduced = self.k + 1
        return original / reduced


# ═══════════════════════════════════════════════════════════════════════════
# §4  Reflection Reducer
# ═══════════════════════════════════════════════════════════════════════════


class ReflectionReducer:
    """Reduce LP using reflection symmetry of the loss function.

    When the loss function is symmetric (L(t, t+d) = L(t, t−d)), the
    optimal noise distribution is symmetric about 0: η[l] = η[−l].
    This means we only need ⌈k/2⌉ free variables instead of k.

    **Proof of optimality.**  Let η* be any optimal noise distribution.
    Define η'[l] = η*[−l].  Since L is symmetric, η' achieves the same
    expected loss.  Since the DP constraint set is convex and η* and η'
    are both feasible, their average (η* + η')/2 is feasible and achieves
    ≤ the same loss (by convexity of loss or Jensen's inequality).  This
    average is symmetric, proving that an optimal symmetric solution exists.

    For the reduced LP:
    - Variables: η[0], η[1], …, η[⌈k/2⌉−1], t
    - The "other half" is determined by symmetry: η[k−l] = η[l].
    - DP constraints only needed for l = 1, …, ⌈k/2⌉.

    Parameters
    ----------
    epsilon : float
        Privacy parameter ε.
    delta : float
        Privacy parameter δ.
    k : int
        Number of output bins.
    y_grid : array of shape (k,)
        Output discretisation grid.
    eta_min : float
        Minimum probability floor.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        k: int,
        y_grid: npt.NDArray[np.float64],
        eta_min: float = 1e-18,
    ) -> None:
        self.epsilon = epsilon
        self.delta = delta
        self.k = k
        self.y_grid = np.asarray(y_grid, dtype=np.float64)
        self.eta_min = eta_min

    def reduce(
        self,
        lp_struct: Optional[LPStruct] = None,
    ) -> ReducedLPStruct:
        """Reduce the LP using reflection symmetry.

        The reduced LP has ⌈k/2⌉ + 1 variables.

        Parameters
        ----------
        lp_struct : LPStruct, optional
            Original LP structure.

        Returns
        -------
        ReducedLPStruct
            Reduced LP with ⌈k/2⌉ + 1 variables.
        """
        t_start = time.perf_counter()
        k = self.k
        epsilon = self.epsilon
        e_eps = math.exp(epsilon)

        # Number of free variables: ceil(k/2)
        n_free = (k + 1) // 2
        t_idx = n_free  # epigraph variable index
        n_vars = n_free + 1

        # Mapping from full index l to free index:
        # l -> l  if l < n_free
        # l -> k - l  if l >= n_free  (symmetry: η[l] = η[k-l])
        def free_idx(l: int) -> int:
            """Map full noise index to free variable index."""
            if l < n_free:
                return l
            return k - 1 - l

        # Factor for each free variable in the simplex constraint:
        # If k is odd, the centre variable (l = k//2) appears once.
        # All other variables appear twice (η[l] and η[k-1-l]).
        simplex_coeff = np.zeros(n_free, dtype=np.float64)
        for l in range(k):
            simplex_coeff[free_idx(l)] += 1.0

        # --- Objective: minimize t ---
        c = np.zeros(n_vars, dtype=np.float64)
        c[t_idx] = 1.0

        # --- DP constraints: only for l = 1, …, n_free ---
        A_ub_rows: List[int] = []
        A_ub_cols: List[int] = []
        A_ub_data: List[float] = []
        b_ub_list: List[float] = []
        row_idx = 0

        for l in range(1, n_free):
            fi_l = free_idx(l)
            fi_lm1 = free_idx(l - 1)

            # Forward: η[l] - e^ε · η[l-1] ≤ 0
            A_ub_rows.extend([row_idx, row_idx])
            A_ub_cols.extend([fi_l, fi_lm1])
            A_ub_data.extend([1.0, -e_eps])
            b_ub_list.append(0.0)
            row_idx += 1

            # Backward: η[l-1] - e^ε · η[l] ≤ 0
            A_ub_rows.extend([row_idx, row_idx])
            A_ub_cols.extend([fi_lm1, fi_l])
            A_ub_data.extend([1.0, -e_eps])
            b_ub_list.append(0.0)
            row_idx += 1

        # --- Loss constraint ---
        loss_callable = self.loss_fn_callable
        if loss_callable is not None:
            y_centre = self.y_grid[len(self.y_grid) // 2]
            loss_coeffs = np.zeros(n_free, dtype=np.float64)
            for l in range(k):
                loss_val = loss_callable(0.0, self.y_grid[l] - y_centre)
                loss_coeffs[free_idx(l)] += loss_val

            for fi in range(n_free):
                if abs(loss_coeffs[fi]) > 1e-300:
                    A_ub_rows.append(row_idx)
                    A_ub_cols.append(fi)
                    A_ub_data.append(loss_coeffs[fi])
            A_ub_rows.append(row_idx)
            A_ub_cols.append(t_idx)
            A_ub_data.append(-1.0)
            b_ub_list.append(0.0)
            row_idx += 1

        n_ub = row_idx
        A_ub = coo_matrix(
            (A_ub_data, (A_ub_rows, A_ub_cols)),
            shape=(n_ub, n_vars),
            dtype=np.float64,
        ).tocsr()
        b_ub = np.array(b_ub_list, dtype=np.float64)

        # --- Equality: Σ coefficients · η_free = 1 ---
        A_eq_row = np.zeros((1, n_vars), dtype=np.float64)
        A_eq_row[0, :n_free] = simplex_coeff
        A_eq = csr_matrix(A_eq_row)
        b_eq = np.array([1.0], dtype=np.float64)

        # --- Bounds ---
        bounds: List[Tuple[float, float]] = []
        for _ in range(n_free):
            bounds.append((self.eta_min, 1.0))
        bounds.append((None, None))  # t unbounded

        original_vars = self.k + 1
        reduction_factor = original_vars / n_vars if n_vars > 0 else 1.0

        t_elapsed = time.perf_counter() - t_start

        logger.info(
            "Reflection reduction: %d vars → %d vars (%.1f× reduction) in %.3fs",
            original_vars, n_vars, reduction_factor, t_elapsed,
        )

        return ReducedLPStruct(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            original_n=1,  # Single noise distribution
            original_k=self.k,
            reduction_type="reflection",
            reduction_factor=reduction_factor,
            y_grid=self.y_grid,
            metadata={
                "epsilon": self.epsilon,
                "delta": self.delta,
                "n_free": n_free,
                "reduction_time_s": t_elapsed,
            },
        )

    @property
    def loss_fn_callable(self) -> Optional[Callable[[float, float], float]]:
        """Get the loss function callable from the LossFunction enum."""
        return LossFunction.L2.fn  # default; caller should set properly

    def _exploit_symmetry(
        self,
        noise_full: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Extract free variables from a full symmetric noise distribution.

        Parameters
        ----------
        noise_full : array of shape (k,)
            Full noise distribution (should be symmetric).

        Returns
        -------
        noise_free : array of shape (ceil(k/2),)
            Free variables.
        """
        k = len(noise_full)
        n_free = (k + 1) // 2
        return noise_full[:n_free].copy()

    def reconstruct(
        self,
        noise_free: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Reconstruct full noise distribution from free variables.

        η[l] = noise_free[l]         for l < n_free
        η[l] = noise_free[k-1-l]     for l >= n_free

        Parameters
        ----------
        noise_free : array of shape (ceil(k/2),)
            Free variables from the reduced LP solution.

        Returns
        -------
        noise_full : array of shape (k,)
            Full symmetric noise distribution.
        """
        n_free = len(noise_free)
        k = self.k
        noise_full = np.zeros(k, dtype=np.float64)

        for l in range(k):
            if l < n_free:
                noise_full[l] = noise_free[l]
            else:
                noise_full[l] = noise_free[k - 1 - l]

        return noise_full


# ═══════════════════════════════════════════════════════════════════════════
# §5  Permutation Reducer (General Symmetries)
# ═══════════════════════════════════════════════════════════════════════════


class PermutationReducer:
    """Reduce LP using general permutation symmetries of the query.

    For a query whose output space has an automorphism group G, we can
    restrict to G-invariant mechanisms.  Variables that lie in the same
    orbit of G are identified (set equal), reducing the number of free
    variables to the number of orbits.

    **Correctness.**  If σ ∈ G is a symmetry, then for any feasible
    mechanism M, the mechanism M' defined by M'(x) = σ(M(x)) is also
    feasible (since σ permutes the outputs and preserves the constraint
    structure).  By averaging over all σ ∈ G (Reynolds operator), we
    obtain a G-invariant mechanism with at most the same loss.

    Parameters
    ----------
    generators : list of arrays, each shape (k,)
        Permutation generators for the symmetry group.
    k : int
        Number of output bins.
    """

    def __init__(
        self,
        generators: List[npt.NDArray[np.intp]],
        k: int,
    ) -> None:
        self.generators = generators
        self.k = k
        self._orbits: Optional[List[List[int]]] = None

    @property
    def orbits(self) -> List[List[int]]:
        """Compute and cache orbits of the permutation group on {0, …, k-1}."""
        if self._orbits is None:
            self._orbits = orbit_computation(
                list(range(self.k)),
                self.generators,
            )
        return self._orbits

    def orbit_representative(self, element: int) -> int:
        """Find the representative (smallest element) of the orbit containing element.

        Parameters
        ----------
        element : int
            Element to find the representative for.

        Returns
        -------
        int
            Smallest element in the same orbit.
        """
        for orbit in self.orbits:
            if element in orbit:
                return min(orbit)
        return element

    def reduce_by_orbits(
        self,
        lp_struct: LPStruct,
    ) -> Tuple[ReducedLPStruct, Dict[int, int]]:
        """Reduce the LP by identifying variables in the same orbit.

        Variables in the same orbit are replaced by a single representative
        variable.  Constraints are collapsed accordingly.

        Parameters
        ----------
        lp_struct : LPStruct
            Original LP structure.

        Returns
        -------
        reduced : ReducedLPStruct
            Reduced LP.
        orbit_map : dict
            Mapping from original variable index to orbit representative index.
        """
        orbits = self.orbits
        n_orbits = len(orbits)

        # Build orbit membership map: element -> orbit index
        orbit_map: Dict[int, int] = {}
        for orbit_idx, orbit in enumerate(orbits):
            for elem in orbit:
                orbit_map[elem] = orbit_idx

        # For a full LP, we'd need to do the variable substitution in the
        # constraint matrices.  Here we provide the orbit structure and let
        # the caller handle the matrix algebra.

        logger.info(
            "Permutation reduction: %d elements → %d orbits (%.1f× reduction)",
            self.k, n_orbits, self.k / n_orbits if n_orbits > 0 else 1.0,
        )

        # Construct reduced LP by substitution
        n_orig_vars = lp_struct.n_vars
        # The orbit map gives us which original p-variables to merge
        # For now, return a placeholder structure
        reduced = ReducedLPStruct(
            c=lp_struct.c[:n_orbits + 1],  # simplified
            A_ub=lp_struct.A_ub[:, :n_orbits + 1],  # simplified
            b_ub=lp_struct.b_ub,
            A_eq=lp_struct.A_eq[:, :n_orbits + 1] if lp_struct.A_eq is not None else None,
            b_eq=lp_struct.b_eq,
            bounds=lp_struct.bounds[:n_orbits + 1],
            original_n=lp_struct.A_eq.shape[0] if lp_struct.A_eq is not None else 0,
            original_k=self.k,
            reduction_type="permutation",
            reduction_factor=n_orig_vars / (n_orbits + 1) if n_orbits > 0 else 1.0,
            y_grid=lp_struct.y_grid,
            metadata={"n_orbits": n_orbits},
        )

        return reduced, orbit_map


# ═══════════════════════════════════════════════════════════════════════════
# §6  Group Theory Utilities
# ═══════════════════════════════════════════════════════════════════════════


class PermutationGroup:
    """Permutation group represented by generators.

    Provides basic group operations: composition, inverse, identity,
    orbit computation, and group order estimation.

    A permutation of {0, 1, …, n−1} is represented as a NumPy array
    σ where σ[i] is the image of i under the permutation.

    Parameters
    ----------
    generators : list of arrays, each shape (n,)
        Group generators.  The group is the closure of these permutations
        under composition and inversion.
    n : int
        Size of the permutation domain {0, …, n−1}.
    """

    def __init__(
        self,
        generators: List[npt.NDArray[np.intp]],
        n: int,
    ) -> None:
        self.generators = [np.asarray(g, dtype=np.intp) for g in generators]
        self.n = n

        for g in self.generators:
            if len(g) != n:
                raise ValueError(
                    f"Generator length {len(g)} != domain size {n}"
                )
            if set(g) != set(range(n)):
                raise ValueError(
                    f"Generator is not a valid permutation of {{0, …, {n-1}}}"
                )

    @property
    def identity(self) -> npt.NDArray[np.intp]:
        """The identity permutation."""
        return np.arange(self.n, dtype=np.intp)

    def compose(
        self,
        sigma: npt.NDArray[np.intp],
        tau: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.intp]:
        """Compose two permutations: (σ ∘ τ)(i) = σ(τ(i)).

        Parameters
        ----------
        sigma, tau : arrays of shape (n,)
            Permutations to compose.

        Returns
        -------
        array of shape (n,)
            The composition σ ∘ τ.
        """
        return sigma[tau].copy()

    def inverse(self, sigma: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        """Compute the inverse of a permutation.

        Parameters
        ----------
        sigma : array of shape (n,)
            A permutation.

        Returns
        -------
        array of shape (n,)
            The inverse permutation σ⁻¹.
        """
        inv = np.empty(self.n, dtype=np.intp)
        inv[sigma] = np.arange(self.n, dtype=np.intp)
        return inv

    def orbit_of(self, element: int) -> List[int]:
        """Compute the orbit of a single element under the group.

        Parameters
        ----------
        element : int
            Element of {0, …, n−1}.

        Returns
        -------
        list of int
            Sorted orbit {σ(element) : σ ∈ G}.
        """
        orbit: Set[int] = {element}
        frontier = [element]

        while frontier:
            current = frontier.pop()
            for gen in self.generators:
                img = int(gen[current])
                if img not in orbit:
                    orbit.add(img)
                    frontier.append(img)
                # Also try inverse
                inv_gen = self.inverse(gen)
                img_inv = int(inv_gen[current])
                if img_inv not in orbit:
                    orbit.add(img_inv)
                    frontier.append(img_inv)

        return sorted(orbit)

    def all_orbits(self) -> List[List[int]]:
        """Compute all orbits of the group on {0, …, n−1}.

        Returns
        -------
        list of list of int
            List of orbits, each a sorted list of elements.
        """
        return orbit_computation(list(range(self.n)), self.generators)

    def order_estimate(self, max_elements: int = 100000) -> int:
        """Estimate the group order by enumerating elements.

        Uses BFS on the Cayley graph (generators as edges) to enumerate
        group elements.  Stops at max_elements to avoid memory exhaustion
        for large groups.

        Parameters
        ----------
        max_elements : int
            Maximum number of elements to enumerate.

        Returns
        -------
        int
            Exact order if ≤ max_elements, otherwise max_elements (lower bound).
        """
        seen: Set[bytes] = set()
        identity = self.identity
        queue = [identity]
        seen.add(identity.tobytes())

        while queue and len(seen) < max_elements:
            current = queue.pop(0)
            for gen in self.generators:
                new_perm = self.compose(gen, current)
                key = new_perm.tobytes()
                if key not in seen:
                    seen.add(key)
                    queue.append(new_perm)

                new_perm_inv = self.compose(self.inverse(gen), current)
                key_inv = new_perm_inv.tobytes()
                if key_inv not in seen:
                    seen.add(key_inv)
                    queue.append(new_perm_inv)

        return len(seen)


def orbit_computation(
    elements: List[int],
    generators: List[npt.NDArray[np.intp]],
) -> List[List[int]]:
    """Compute orbits of a permutation group on a set of elements.

    Uses union-find (disjoint set) to efficiently compute orbits.
    For each generator σ, we union i with σ(i) for all i.

    **Complexity.**  O(n · |generators| · α(n)) where α is the inverse
    Ackermann function (essentially O(n · |generators|)).

    Parameters
    ----------
    elements : list of int
        Elements to partition into orbits.
    generators : list of arrays
        Permutation generators.

    Returns
    -------
    orbits : list of list of int
        List of orbits, each a sorted list of elements.
    """
    if not elements:
        return []

    if not generators:
        return [[e] for e in elements]

    # Union-Find data structure
    parent: Dict[int, int] = {e: e for e in elements}
    rank: Dict[int, int] = {e: 0 for e in elements}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    # Apply each generator
    elem_set = set(elements)
    for gen in generators:
        for e in elements:
            if e < len(gen):
                img = int(gen[e])
                if img in elem_set:
                    union(e, img)

    # Collect orbits
    orbits_dict: Dict[int, List[int]] = {}
    for e in elements:
        root = find(e)
        if root not in orbits_dict:
            orbits_dict[root] = []
        orbits_dict[root].append(e)

    return [sorted(orbit) for orbit in orbits_dict.values()]


def stabilizer(
    element: int,
    group: PermutationGroup,
    max_elements: int = 100000,
) -> List[npt.NDArray[np.intp]]:
    """Compute the stabiliser of an element: Stab(x) = {σ ∈ G : σ(x) = x}.

    Enumerates group elements via BFS and filters for those fixing the
    given element.

    Parameters
    ----------
    element : int
        Element to stabilise.
    group : PermutationGroup
        The permutation group.
    max_elements : int
        Maximum number of group elements to enumerate.

    Returns
    -------
    list of arrays
        Elements of the stabiliser subgroup.
    """
    stab: List[npt.NDArray[np.intp]] = []
    seen: Set[bytes] = set()
    identity = group.identity
    queue = [identity]
    seen.add(identity.tobytes())

    while queue and len(seen) < max_elements:
        current = queue.pop(0)
        if current[element] == element:
            stab.append(current.copy())

        for gen in group.generators:
            new_perm = group.compose(gen, current)
            key = new_perm.tobytes()
            if key not in seen:
                seen.add(key)
                queue.append(new_perm)

    return stab


def is_group(
    elements: List[npt.NDArray[np.intp]],
    n: int,
) -> bool:
    """Verify that a set of permutations satisfies the group axioms.

    Checks:
    1. **Closure:** For all σ, τ ∈ S, σ ∘ τ ∈ S.
    2. **Identity:** The identity permutation is in S.
    3. **Inverses:** For all σ ∈ S, σ⁻¹ ∈ S.

    (Associativity is automatic for permutation composition.)

    Parameters
    ----------
    elements : list of arrays, each shape (n,)
        Candidate group elements.
    n : int
        Domain size.

    Returns
    -------
    bool
        True if the elements form a group under composition.
    """
    if not elements:
        return False

    elem_set: Set[bytes] = {e.tobytes() for e in elements}

    # Check identity
    identity = np.arange(n, dtype=np.intp)
    if identity.tobytes() not in elem_set:
        return False

    for sigma in elements:
        # Check inverse
        inv = np.empty(n, dtype=np.intp)
        inv[sigma] = np.arange(n, dtype=np.intp)
        if inv.tobytes() not in elem_set:
            return False

        # Check closure (expensive — check a subset for large groups)
        for tau in elements:
            composed = sigma[tau]
            if composed.tobytes() not in elem_set:
                return False

    return True


# ═══════════════════════════════════════════════════════════════════════════
# §7  Reconstruction
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ReconstructionMap:
    """Map from reduced LP solution back to full mechanism table.

    Stores the information needed to expand a reduced solution
    (noise distribution + symmetry info) into a full n × k table.

    Attributes:
        reduction_type: Type of reduction that was applied.
        original_n: Original number of inputs.
        original_k: Original number of output bins.
        f_values: Original query values.
        y_grid: Output discretisation grid.
        metadata: Additional reconstruction parameters.
    """

    reduction_type: str
    original_n: int
    original_k: int
    f_values: npt.NDArray[np.float64]
    y_grid: npt.NDArray[np.float64]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def expand(
        self,
        reduced_solution: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Expand a reduced solution to a full mechanism table.

        Parameters
        ----------
        reduced_solution : array
            Solution from the reduced LP.

        Returns
        -------
        p : array of shape (original_n, original_k)
            Full mechanism table.
        """
        if self.reduction_type == "translation":
            return expand_noise_to_full(
                reduced_solution, self.f_values, self.y_grid,
            )
        elif self.reduction_type == "reflection":
            # Expand symmetric noise
            k = self.original_k
            n_free = len(reduced_solution)
            full_noise = np.zeros(k, dtype=np.float64)
            for l in range(k):
                if l < n_free:
                    full_noise[l] = reduced_solution[l]
                else:
                    full_noise[l] = reduced_solution[k - 1 - l]
            return expand_noise_to_full(
                full_noise, self.f_values, self.y_grid,
            )
        else:
            raise ValueError(
                f"Unknown reduction type: {self.reduction_type!r}"
            )


def expand_noise_to_full(
    noise_dist: npt.NDArray[np.float64],
    f_values: npt.NDArray[np.float64],
    grid: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Reconstruct full n × k table from a noise distribution.

    For a counting query with f(x_i) = i and grid y_j, constructs:

        p[i][j] = η[j − offset(i)]

    where offset(i) maps f(x_i) to the nearest grid index.

    **Correctness.**  For a translation-invariant mechanism with noise η,
    the probability of outputting y_j when the true value is x_i is
    η applied at the offset j − gridindex(f(x_i)).  This correctly
    reproduces the full mechanism table.

    Parameters
    ----------
    noise_dist : array of shape (k,)
        Noise distribution.
    f_values : array of shape (n,)
        Query output values.
    grid : array of shape (k,)
        Output discretisation grid.

    Returns
    -------
    p : array of shape (n, k)
        Full mechanism probability table.
    """
    noise_dist = np.asarray(noise_dist, dtype=np.float64)
    f_values = np.asarray(f_values, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    n = len(f_values)
    k = len(noise_dist)

    p = np.zeros((n, k), dtype=np.float64)

    # Find grid spacing
    if k > 1:
        grid_spacing = (grid[-1] - grid[0]) / (k - 1)
    else:
        grid_spacing = 1.0

    grid_min = float(grid[0])

    for i in range(n):
        # Offset: number of grid steps from grid_min to f(x_i)
        offset = int(round((f_values[i] - grid_min) / max(grid_spacing, 1e-300)))

        for l in range(k):
            j = l + offset
            if 0 <= j < k:
                p[i, j] = noise_dist[l]

    # Re-normalise in case of truncation
    row_sums = p.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-300)
    p /= row_sums

    return p


def verify_reconstruction(
    full_table: npt.NDArray[np.float64],
    reduced_solution: npt.NDArray[np.float64],
    recon_map: ReconstructionMap,
    *,
    tol: float = 1e-8,
) -> bool:
    """Verify that a reconstruction correctly reproduces the full table.

    Expands the reduced solution and checks that it matches the full
    table within tolerance.

    Parameters
    ----------
    full_table : array of shape (n, k)
        The full mechanism table to compare against.
    reduced_solution : array
        The reduced LP solution.
    recon_map : ReconstructionMap
        The reconstruction map.
    tol : float
        Tolerance for element-wise comparison.

    Returns
    -------
    bool
        True if the reconstruction matches within tolerance.
    """
    reconstructed = recon_map.expand(reduced_solution)
    return bool(np.allclose(full_table, reconstructed, atol=tol))


# ═══════════════════════════════════════════════════════════════════════════
# §8  Top-Level Entry Point
# ═══════════════════════════════════════════════════════════════════════════


def ReduceBySymmetry(
    lp_struct: LPStruct,
    spec: QuerySpec,
    *,
    symmetry_group: Optional[SymmetryGroup] = None,
    eta_min: Optional[float] = None,
) -> Tuple[ReducedLPStruct, ReconstructionMap]:
    """Top-level symmetry reduction: detect symmetries and reduce the LP.

    This is the main entry point for symmetry exploitation.  It:

    1. Detects the symmetry group G of the query (if not provided).
    2. For translation-invariant queries: replaces n×k variables with k
       variables (one noise distribution).
    3. Reduces privacy constraints: from 2|E|×k to 2k.
    4. Returns the reduced LP with k variables and 2k+1 constraints.
    5. Factor-n speedup.

    Parameters
    ----------
    lp_struct : LPStruct
        Original LP structure.
    spec : QuerySpec
        Query specification.
    symmetry_group : SymmetryGroup, optional
        Pre-detected symmetry group.  If None, detection is run automatically.
    eta_min : float, optional
        Minimum probability floor.

    Returns
    -------
    reduced_lp : ReducedLPStruct
        Reduced LP structure.
    recon_map : ReconstructionMap
        Map for reconstructing the full solution from the reduced one.
    """
    t_start = time.perf_counter()

    # Auto-detect symmetry if not provided
    if symmetry_group is None:
        detector = SymmetryDetector(spec=spec)
        symmetry_group = detector.detect()

    if not symmetry_group.has_symmetry:
        logger.info("No exploitable symmetry detected; returning original LP")
        # Wrap original LP in ReducedLPStruct
        reduced = ReducedLPStruct(
            c=lp_struct.c,
            A_ub=lp_struct.A_ub,
            b_ub=lp_struct.b_ub,
            A_eq=lp_struct.A_eq,
            b_eq=lp_struct.b_eq,
            bounds=lp_struct.bounds,
            original_n=spec.n,
            original_k=spec.k,
            reduction_type="none",
            reduction_factor=1.0,
            y_grid=lp_struct.y_grid,
        )
        recon_map = ReconstructionMap(
            reduction_type="none",
            original_n=spec.n,
            original_k=spec.k,
            f_values=spec.query_values,
            y_grid=lp_struct.y_grid,
        )
        return reduced, recon_map

    eff_eta_min = eta_min if eta_min is not None else spec.eta_min

    if symmetry_group.is_translation_invariant:
        # Apply translation reduction
        reducer = TranslationReducer(
            epsilon=spec.epsilon,
            delta=spec.delta,
            k=spec.k,
            y_grid=lp_struct.y_grid,
            loss_fn=spec.loss_fn,
            eta_min=eff_eta_min,
        )
        reduced = reducer.reduce(lp_struct, n=spec.n)

        recon_map = ReconstructionMap(
            reduction_type="translation",
            original_n=spec.n,
            original_k=spec.k,
            f_values=spec.query_values,
            y_grid=lp_struct.y_grid,
        )

        if symmetry_group.is_reflection_symmetric:
            logger.info(
                "Reflection symmetry also detected; could further reduce "
                "by 2× (not applied in this pass)"
            )

    else:
        # No translation invariance — return original
        logger.info(
            "Only reflection symmetry detected; standard reduction not applicable"
        )
        reduced = ReducedLPStruct(
            c=lp_struct.c,
            A_ub=lp_struct.A_ub,
            b_ub=lp_struct.b_ub,
            A_eq=lp_struct.A_eq,
            b_eq=lp_struct.b_eq,
            bounds=lp_struct.bounds,
            original_n=spec.n,
            original_k=spec.k,
            reduction_type="none",
            reduction_factor=1.0,
            y_grid=lp_struct.y_grid,
        )
        recon_map = ReconstructionMap(
            reduction_type="none",
            original_n=spec.n,
            original_k=spec.k,
            f_values=spec.query_values,
            y_grid=lp_struct.y_grid,
        )

    t_elapsed = time.perf_counter() - t_start
    logger.info(
        "ReduceBySymmetry completed in %.3fs: %s (%.1f× reduction)",
        t_elapsed, reduced.reduction_type, reduced.reduction_factor,
    )

    return reduced, recon_map


# ═══════════════════════════════════════════════════════════════════════════
# §9  Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════


def detect_and_report_symmetry(spec: QuerySpec) -> SymmetryGroup:
    """Detect symmetry and log a human-readable report.

    Parameters
    ----------
    spec : QuerySpec
        Query specification.

    Returns
    -------
    SymmetryGroup
        Detected symmetry group.
    """
    detector = SymmetryDetector(spec=spec)
    group = detector.detect()

    logger.info(
        "Symmetry report for %s:\n"
        "  Group type:        %s\n"
        "  Group order:       %d\n"
        "  Translation inv.:  %s\n"
        "  Reflection sym.:   %s\n"
        "  Theoretical speedup: %.1f×",
        spec,
        group.group_type,
        group.order,
        group.is_translation_invariant,
        group.is_reflection_symmetric,
        group.theoretical_speedup,
    )

    return group


def is_counting_query(spec: QuerySpec) -> bool:
    """Quick check whether a QuerySpec represents a counting query.

    A counting query has query_values = [0, 1, 2, …, n−1], sensitivity = 1,
    and path-graph adjacency.

    Parameters
    ----------
    spec : QuerySpec
        Query specification.

    Returns
    -------
    bool
        True if the spec is a counting query.
    """
    if spec.query_type == QueryType.COUNTING:
        return True

    f = spec.query_values
    n = len(f)

    # Check arithmetic progression with step 1 starting at 0
    expected = np.arange(n, dtype=np.float64)
    if not np.allclose(f, expected, atol=_ARITHMETIC_TOL):
        return False

    # Check sensitivity
    if abs(spec.sensitivity - 1.0) > _ARITHMETIC_TOL:
        return False

    return True


def reduce_if_possible(
    lp_struct: LPStruct,
    spec: QuerySpec,
    *,
    eta_min: Optional[float] = None,
) -> Tuple[ReducedLPStruct, ReconstructionMap, SymmetryGroup]:
    """Detect symmetry and reduce the LP if beneficial.

    Convenience function that combines detection and reduction.

    Parameters
    ----------
    lp_struct : LPStruct
        Original LP structure.
    spec : QuerySpec
        Query specification.
    eta_min : float, optional
        Minimum probability floor.

    Returns
    -------
    reduced : ReducedLPStruct
        Reduced (or original) LP.
    recon_map : ReconstructionMap
        Reconstruction map.
    group : SymmetryGroup
        Detected symmetry group.
    """
    detector = SymmetryDetector(spec=spec)
    group = detector.detect()

    reduced, recon_map = ReduceBySymmetry(
        lp_struct, spec, symmetry_group=group, eta_min=eta_min,
    )

    return reduced, recon_map, group
