"""
Lattice enumeration algorithms for DP mechanism synthesis.

Implements Kannan's enumeration, Fincke-Pohst, pruned enumeration,
bounded enumeration within privacy constraint polytopes, and
tree-based systematic search.
"""

from __future__ import annotations

import heapq
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import numpy.typing as npt

from dp_forge.lattice.reduction import (
    ClosestVectorProblem,
    GramSchmidt,
    LLLReduction,
)
from dp_forge.types import LatticePoint, PrivacyBudget, QuerySpec


# ---------------------------------------------------------------------------
# Pruning strategies
# ---------------------------------------------------------------------------


class PruningType(Enum):
    """Type of pruning for enumeration."""

    NONE = auto()
    LINEAR = auto()
    CYLINDRICAL = auto()
    GRADIENT = auto()


@dataclass
class PruningBounds:
    """Pruning bounds for lattice enumeration.

    Bounds R_1 ≥ R_2 ≥ ... ≥ R_n define the maximum partial norm
    at each projection level during enumeration.
    """

    radii: npt.NDArray[np.float64]
    pruning_type: PruningType = PruningType.LINEAR

    def __post_init__(self) -> None:
        self.radii = np.asarray(self.radii, dtype=np.float64)

    @property
    def dimension(self) -> int:
        return len(self.radii)


class PruningStrategy:
    """Compute pruning bounds for lattice enumeration.

    Implements several strategies for bounding the search space
    during enumeration to reduce the exponential cost.
    """

    def __init__(self, success_probability: float = 0.99) -> None:
        self._success_prob = success_probability

    def linear_pruning(
        self, radius: float, dimension: int
    ) -> PruningBounds:
        """Linear pruning: R_k = R · (k/n)^{1/2}.

        Simplest pruning; reduces enumeration cost significantly while
        maintaining reasonable success probability.
        """
        n = dimension
        radii = np.array([
            radius * math.sqrt((k + 1) / n) for k in range(n)
        ], dtype=np.float64)
        return PruningBounds(radii=radii, pruning_type=PruningType.LINEAR)

    def cylindrical_pruning(
        self, radius: float, dimension: int, tightness: float = 0.5
    ) -> PruningBounds:
        """Cylindrical pruning with configurable tightness.

        R_k = R · (1 - tightness + tightness · (k+1)/n)^{1/2}

        tightness=0 gives no pruning; tightness=1 gives linear pruning.
        """
        n = dimension
        radii = np.array([
            radius * math.sqrt(1 - tightness + tightness * (k + 1) / n)
            for k in range(n)
        ], dtype=np.float64)
        return PruningBounds(radii=radii, pruning_type=PruningType.CYLINDRICAL)

    def gradient_pruning(
        self, radius: float, gs_norms_sq: npt.NDArray[np.float64]
    ) -> PruningBounds:
        """Gradient-based pruning using Gram-Schmidt norms.

        Allocates more search budget to directions with larger
        Gram-Schmidt norms (these contribute more to the lattice volume).
        """
        n = len(gs_norms_sq)
        total = np.sum(gs_norms_sq)
        if total < 1e-15:
            return PruningBounds(
                radii=np.full(n, radius),
                pruning_type=PruningType.GRADIENT,
            )

        cumulative = np.cumsum(gs_norms_sq) / total
        radii = radius * np.sqrt(cumulative)
        return PruningBounds(radii=radii, pruning_type=PruningType.GRADIENT)

    def optimal_pruning(
        self, radius: float, dimension: int, target_nodes: int = 100000
    ) -> PruningBounds:
        """Compute pruning bounds that target a specific enumeration tree size.

        Uses the Gaussian volume heuristic to estimate the number of
        enumeration nodes for given pruning bounds, then adjusts bounds
        to hit the target.
        """
        n = dimension
        # Start with linear pruning and adjust
        tightness = 0.5
        for _ in range(20):  # binary search
            bounds = self.cylindrical_pruning(radius, n, tightness)
            est_nodes = self._estimate_nodes(bounds, radius)
            if est_nodes > target_nodes:
                tightness = min(tightness + 0.05, 0.99)
            else:
                tightness = max(tightness - 0.05, 0.01)

        return self.cylindrical_pruning(radius, n, tightness)

    def _estimate_nodes(
        self, bounds: PruningBounds, radius: float
    ) -> float:
        """Estimate enumeration tree size for given pruning bounds."""
        n = bounds.dimension
        # Volume of the pruned search region relative to full ball
        vol_ratio = 1.0
        for k in range(n):
            vol_ratio *= bounds.radii[k] / radius
        # Rough estimate: Gaussian heuristic × volume ratio
        return max(1, (2 * radius) ** n * vol_ratio)


# ---------------------------------------------------------------------------
# Lattice Enumerator
# ---------------------------------------------------------------------------


class LatticeEnumerator:
    """Enumerate lattice points within a radius bound.

    Implements the core enumeration loop used by Kannan and
    Fincke-Pohst algorithms: systematically try integer coefficient
    vectors and check norm bounds at each projection level.
    """

    def __init__(
        self,
        max_points: int = 10000,
        pruning: Optional[PruningBounds] = None,
    ) -> None:
        self._max_points = max_points
        self._pruning = pruning
        self._gs = GramSchmidt()

    def enumerate_points(
        self,
        basis: npt.NDArray[np.float64],
        radius: float,
        center: Optional[npt.NDArray[np.float64]] = None,
    ) -> List[npt.NDArray[np.float64]]:
        """Enumerate all lattice points within radius of center.

        Args:
            basis: n × d lattice basis.
            radius: Enumeration radius.
            center: Center point (default: origin).

        Returns:
            List of lattice vectors within radius of center.
        """
        B = np.array(basis, dtype=np.float64)
        n, d = B.shape

        if center is None:
            center = np.zeros(d, dtype=np.float64)
        else:
            center = np.array(center, dtype=np.float64)

        gs = self._gs.orthogonalize(B)
        mu = gs.mu_coefficients
        norms_sq = gs.norms_squared

        # Project center onto Gram-Schmidt basis
        center_proj = np.zeros(n, dtype=np.float64)
        remaining = center.copy()
        for i in range(n - 1, -1, -1):
            if norms_sq[i] > 1e-15:
                center_proj[i] = np.dot(remaining, gs.orthogonal_basis[i]) / norms_sq[i]
                remaining -= center_proj[i] * B[i]

        # Enumeration via depth-first search
        result: List[npt.NDArray[np.float64]] = []
        R_sq = radius * radius

        # Compute coefficient ranges for each level
        self._enumerate_recursive(
            B, gs, center_proj, R_sq, n - 1,
            np.zeros(n, dtype=np.float64),
            0.0, result,
        )

        return result

    def _enumerate_recursive(
        self,
        basis: npt.NDArray[np.float64],
        gs: object,
        center_proj: npt.NDArray[np.float64],
        R_sq: float,
        level: int,
        coeffs: npt.NDArray[np.float64],
        partial_sq: float,
        result: List[npt.NDArray[np.float64]],
    ) -> None:
        """Recursive depth-first enumeration at a given level."""
        if len(result) >= self._max_points:
            return

        norms_sq = gs.norms_squared
        mu = gs.mu_coefficients

        if norms_sq[level] < 1e-15:
            coeffs[level] = 0.0
            if level > 0:
                self._enumerate_recursive(
                    basis, gs, center_proj, R_sq,
                    level - 1, coeffs, partial_sq, result,
                )
            else:
                vec = coeffs.astype(np.int64) @ basis
                if np.any(vec != 0):
                    result.append(vec)
            return

        # Compute the center value for this level
        c_val = center_proj[level]
        for j in range(level + 1, len(coeffs)):
            c_val -= mu[j, level] * coeffs[j]

        # Compute range of valid integer values at this level
        remaining = R_sq - partial_sq
        if remaining < -1e-10:
            return

        # Check pruning bounds
        if self._pruning is not None and level < self._pruning.dimension:
            remaining = min(remaining, self._pruning.radii[level] ** 2 - partial_sq)
            if remaining < -1e-10:
                return

        bound = math.sqrt(max(0, remaining / norms_sq[level]))
        lo = math.ceil(c_val - bound)
        hi = math.floor(c_val + bound)

        # Schnorr-Euchner zigzag enumeration order (closest to center first)
        c_round = round(c_val)
        for v in self._zigzag_range(c_round, lo, hi):
            coeffs[level] = float(v)
            new_partial = partial_sq + norms_sq[level] * (v - c_val) ** 2

            if new_partial > R_sq + 1e-10:
                continue

            if level > 0:
                self._enumerate_recursive(
                    basis, gs, center_proj, R_sq,
                    level - 1, coeffs.copy(), new_partial, result,
                )
            else:
                vec = coeffs.astype(np.int64) @ basis
                if np.any(vec != 0):
                    result.append(vec)

            if len(result) >= self._max_points:
                return

    @staticmethod
    def _zigzag_range(center: int, lo: int, hi: int) -> Iterator[int]:
        """Generate integers from lo to hi in zigzag order around center."""
        center = max(lo, min(hi, center))
        yield center
        for delta in range(1, hi - lo + 1):
            if center + delta <= hi:
                yield center + delta
            if center - delta >= lo:
                yield center - delta

    def enumerate_shortest(
        self,
        basis: npt.NDArray[np.float64],
        num_shortest: int = 10,
    ) -> List[Tuple[npt.NDArray[np.float64], float]]:
        """Find the num_shortest shortest lattice vectors.

        Args:
            basis: n × d lattice basis.
            num_shortest: Number of shortest vectors to find.

        Returns:
            List of (vector, norm) tuples, sorted by norm.
        """
        B = LLLReduction(delta=0.99).reduce(basis.copy())
        # Use norm of first LLL-reduced vector as initial radius
        first_norm = float(np.linalg.norm(B[0]))
        radius = first_norm * 1.5

        points = self.enumerate_points(B, radius)
        if not points:
            return [(B[0].copy(), first_norm)]

        # Sort by norm
        with_norms = [(p, float(np.linalg.norm(p))) for p in points]
        with_norms.sort(key=lambda x: x[1])

        # Deduplicate (keep ±v as single entry)
        unique: List[Tuple[npt.NDArray[np.float64], float]] = []
        seen_hashes: set = set()
        for vec, norm in with_norms:
            # Canonicalize sign: first nonzero entry positive
            canon = vec.copy()
            for v in canon:
                if abs(v) > 1e-10:
                    if v < 0:
                        canon = -canon
                    break
            h = tuple(np.round(canon, 10))
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique.append((vec, norm))
                if len(unique) >= num_shortest:
                    break

        return unique


# ---------------------------------------------------------------------------
# Kannan's Enumeration
# ---------------------------------------------------------------------------


class KannanEnumeration:
    """Kannan's enumeration algorithm for the shortest vector problem.

    Uses a BKZ-reduced basis and enumerates within a ball of radius
    given by the Gaussian heuristic. More efficient than naive
    enumeration for moderate dimensions.
    """

    def __init__(
        self,
        block_size: int = 20,
        pruning_strategy: Optional[PruningStrategy] = None,
    ) -> None:
        self._block_size = block_size
        self._pruning = pruning_strategy or PruningStrategy()
        self._gs = GramSchmidt()

    def shortest_vector(
        self, basis: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Find shortest non-zero lattice vector via Kannan's algorithm.

        1. BKZ-reduce the basis
        2. Compute enumeration radius from first vector norm
        3. Apply pruning bounds
        4. Enumerate and return shortest

        Args:
            basis: n × d lattice basis.

        Returns:
            Shortest non-zero lattice vector.
        """
        from dp_forge.lattice.reduction import BKZReduction

        B = np.array(basis, dtype=np.float64)
        n = B.shape[0]

        # Step 1: BKZ preprocessing
        beta = min(self._block_size, n)
        bkz = BKZReduction(block_size=beta, max_tours=5)
        B = bkz.reduce(B)

        # Step 2: Radius from first vector
        R = float(np.linalg.norm(B[0]))

        # Step 3: Pruning bounds
        gs = self._gs.orthogonalize(B)
        bounds = self._pruning.gradient_pruning(R, gs.norms_squared)

        # Step 4: Enumerate
        enumerator = LatticeEnumerator(max_points=5000, pruning=bounds)
        points = enumerator.enumerate_points(B, R)

        if not points:
            return B[0].copy()

        # Return shortest
        best_idx = min(range(len(points)), key=lambda i: np.dot(points[i], points[i]))
        return points[best_idx]

    def successive_minima(
        self, basis: npt.NDArray[np.float64], k: int = 0
    ) -> List[Tuple[npt.NDArray[np.float64], float]]:
        """Compute approximate successive minima λ_1, ..., λ_k.

        The i-th successive minimum λ_i is the smallest radius r such
        that the ball of radius r contains i linearly independent
        lattice vectors.

        Args:
            basis: n × d lattice basis.
            k: Number of minima to compute (default: n).

        Returns:
            List of (vector, λ_i) pairs.
        """
        n = basis.shape[0]
        if k <= 0:
            k = n

        enumerator = LatticeEnumerator(max_points=10000)
        shortest = enumerator.enumerate_shortest(basis, num_shortest=k * 3)

        # Select linearly independent subset
        result: List[Tuple[npt.NDArray[np.float64], float]] = []
        selected_vecs: List[npt.NDArray[np.float64]] = []

        for vec, norm in shortest:
            if len(result) >= k:
                break
            # Check linear independence
            if not selected_vecs:
                selected_vecs.append(vec)
                result.append((vec, norm))
            else:
                mat = np.vstack(selected_vecs + [vec])
                rank = np.linalg.matrix_rank(mat, tol=1e-8)
                if rank > len(selected_vecs):
                    selected_vecs.append(vec)
                    result.append((vec, norm))

        return result


# ---------------------------------------------------------------------------
# Fincke-Pohst Enumeration
# ---------------------------------------------------------------------------


class FinckePohlstEnumeration:
    """Fincke-Pohst enumeration for lattice points in an ellipsoid.

    Enumerates all lattice vectors v such that (v - t)^T Q (v - t) ≤ R²
    where Q is a positive definite quadratic form, t is the target
    center, and R is the radius.

    This generalizes standard sphere enumeration by allowing
    ellipsoidal search regions defined by a quadratic form.
    """

    def __init__(self, max_solutions: int = 10000) -> None:
        self._max_solutions = max_solutions
        self._gs = GramSchmidt()

    def enumerate(
        self,
        basis: npt.NDArray[np.float64],
        quadratic_form: npt.NDArray[np.float64],
        radius_sq: float,
        center: Optional[npt.NDArray[np.float64]] = None,
    ) -> List[npt.NDArray[np.float64]]:
        """Enumerate lattice points in an ellipsoid.

        Args:
            basis: n × d lattice basis.
            quadratic_form: d × d positive definite matrix Q.
            radius_sq: Squared radius R².
            center: Center of the ellipsoid (default: origin).

        Returns:
            List of lattice vectors in the ellipsoid.
        """
        B = np.array(basis, dtype=np.float64)
        Q = np.array(quadratic_form, dtype=np.float64)
        n, d = B.shape

        if center is None:
            center = np.zeros(d, dtype=np.float64)

        # Compute Cholesky factorization of Q
        try:
            L = np.linalg.cholesky(Q)
        except np.linalg.LinAlgError:
            # If Q is not PD, add small diagonal
            Q_reg = Q + 1e-10 * np.eye(d)
            L = np.linalg.cholesky(Q_reg)

        # Transform problem: B' = B @ L, enumerate in B' with sphere
        B_transformed = B @ L

        # LLL reduce the transformed basis
        lll = LLLReduction(delta=0.99)
        B_trans_reduced = lll.reduce(B_transformed)

        # Transform center
        center_transformed = L.T @ center

        # Enumerate in transformed space
        radius = math.sqrt(radius_sq)
        enumerator = LatticeEnumerator(max_points=self._max_solutions)
        points_trans = enumerator.enumerate_points(
            B_trans_reduced, radius, center_transformed
        )

        # Transform back
        try:
            L_inv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            L_inv = np.eye(d)

        result = []
        for pt in points_trans:
            original = pt @ L_inv
            # Verify it's actually in the ellipsoid
            diff = original - center
            val = diff @ Q @ diff
            if val <= radius_sq * (1.0 + 1e-8):
                result.append(original)

        return result

    def count_points(
        self,
        basis: npt.NDArray[np.float64],
        quadratic_form: npt.NDArray[np.float64],
        radius_sq: float,
    ) -> int:
        """Count lattice points in an ellipsoid without storing them."""
        points = self.enumerate(basis, quadratic_form, radius_sq)
        return len(points)


# ---------------------------------------------------------------------------
# Bounded Enumeration within Privacy Constraint Polytope
# ---------------------------------------------------------------------------


class BoundedEnumeration:
    """Enumerate lattice points within a privacy constraint polytope.

    The privacy constraints form a polytope A·x ≤ b in the mechanism
    parameter space. This class enumerates all lattice points that
    satisfy both the lattice structure and the linear constraints.
    """

    def __init__(
        self,
        max_points: int = 10000,
        tol: float = 1e-8,
    ) -> None:
        self._max_points = max_points
        self._tol = tol
        self._gs = GramSchmidt()

    def enumerate(
        self,
        basis: npt.NDArray[np.float64],
        A: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        objective: Optional[npt.NDArray[np.float64]] = None,
    ) -> List[Tuple[npt.NDArray[np.float64], float]]:
        """Enumerate lattice points satisfying A·x ≤ b.

        Args:
            basis: n × d lattice basis.
            A: m × d constraint matrix.
            b: m-dimensional right-hand side.
            objective: d-dimensional objective for sorting results.

        Returns:
            List of (lattice_point, objective_value) satisfying constraints.
        """
        B = np.array(basis, dtype=np.float64)
        if B.size == 0 or B.ndim < 2 or B.shape[0] == 0 or B.shape[1] == 0:
            return []
        n, d = B.shape
        A = np.array(A, dtype=np.float64)
        b = np.array(b, dtype=np.float64)

        # Compute a bounding ellipsoid for the polytope
        center, radius = self._bounding_ball(A, b, d)

        # Enumerate lattice points in the bounding ball
        enumerator = LatticeEnumerator(max_points=self._max_points * 5)
        candidates = enumerator.enumerate_points(B, radius, center)

        # Filter by polytope constraints
        feasible: List[Tuple[npt.NDArray[np.float64], float]] = []
        for point in candidates:
            if np.all(A @ point <= b + self._tol):
                obj_val = float(objective @ point) if objective is not None else 0.0
                feasible.append((point, obj_val))

        # Also check the origin if it's feasible
        origin = np.zeros(d)
        if np.all(A @ origin <= b + self._tol):
            obj_val = 0.0 if objective is None else float(objective @ origin)
            feasible.append((origin, obj_val))

        # Sort by objective
        feasible.sort(key=lambda x: x[1])

        # Deduplicate
        unique: List[Tuple[npt.NDArray[np.float64], float]] = []
        seen: set = set()
        for pt, val in feasible:
            h = tuple(np.round(pt, 8))
            if h not in seen:
                seen.add(h)
                unique.append((pt, val))
                if len(unique) >= self._max_points:
                    break

        return unique

    def _bounding_ball(
        self,
        A: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        d: int,
    ) -> Tuple[npt.NDArray[np.float64], float]:
        """Compute a bounding ball for the polytope {x : Ax ≤ b}.

        Uses a simple Chebyshev center approximation.
        """
        from scipy.optimize import linprog

        m = A.shape[0]

        # Chebyshev center: max r s.t. A·x + ‖a_i‖·r ≤ b
        norms_A = np.linalg.norm(A, axis=1)

        c_obj = np.zeros(d + 1)
        c_obj[-1] = -1.0  # maximize r

        A_ub = np.column_stack([A, norms_A.reshape(-1, 1)])
        b_ub = b

        bounds = [(None, None)] * d + [(0, None)]

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                center = res.x[:d]
                r = res.x[-1]
                # Use a generous radius
                radius = max(r * 2, np.linalg.norm(center) + 1.0)
                return center, radius
        except Exception:
            pass

        # Fallback: use origin with heuristic radius
        center = np.zeros(d)
        radius = float(np.max(np.abs(b))) + 1.0
        return center, radius

    def enumerate_dp_feasible(
        self,
        basis: npt.NDArray[np.float64],
        epsilon: float,
        n_inputs: int,
        k_outputs: int,
    ) -> List[npt.NDArray[np.float64]]:
        """Enumerate lattice points satisfying ε-DP constraints.

        The DP constraints require that for each pair of adjacent
        inputs (i, i'), the mechanism probabilities satisfy:
        M[i,j] ≤ exp(ε) · M[i',j] for all outputs j.

        Args:
            basis: Lattice basis for mechanism parameters.
            epsilon: Privacy parameter.
            n_inputs: Number of input values.
            k_outputs: Number of output values.

        Returns:
            List of feasible lattice points.
        """
        d = n_inputs * k_outputs
        exp_eps = math.exp(epsilon)

        # Build DP constraint matrix
        constraints_A = []
        constraints_b = []

        for i in range(n_inputs):
            for ip in range(i + 1, n_inputs):
                for j in range(k_outputs):
                    # M[i,j] - exp(ε) · M[i',j] ≤ 0
                    row = np.zeros(d)
                    row[i * k_outputs + j] = 1.0
                    row[ip * k_outputs + j] = -exp_eps
                    constraints_A.append(row)
                    constraints_b.append(0.0)

                    # M[i',j] - exp(ε) · M[i,j] ≤ 0
                    row2 = np.zeros(d)
                    row2[ip * k_outputs + j] = 1.0
                    row2[i * k_outputs + j] = -exp_eps
                    constraints_A.append(row2)
                    constraints_b.append(0.0)

        # Non-negativity: -M[i,j] ≤ 0
        for idx in range(d):
            row = np.zeros(d)
            row[idx] = -1.0
            constraints_A.append(row)
            constraints_b.append(0.0)

        # Normalization: for each input i, sum_j M[i,j] ≤ 1 + tol
        # and -sum_j M[i,j] ≤ -(1 - tol)
        tol = 0.01
        for i in range(n_inputs):
            row_upper = np.zeros(d)
            row_lower = np.zeros(d)
            for j in range(k_outputs):
                row_upper[i * k_outputs + j] = 1.0
                row_lower[i * k_outputs + j] = -1.0
            constraints_A.append(row_upper)
            constraints_b.append(1.0 + tol)
            constraints_A.append(row_lower)
            constraints_b.append(-(1.0 - tol))

        A = np.array(constraints_A)
        b_vec = np.array(constraints_b)

        results = self.enumerate(basis, A, b_vec)
        return [pt for pt, _ in results]


# ---------------------------------------------------------------------------
# Enumeration Tree
# ---------------------------------------------------------------------------


@dataclass
class EnumerationNode:
    """A node in the enumeration tree."""

    node_id: int
    level: int
    coefficient: int
    partial_norm_sq: float
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    is_solution: bool = False

    def __repr__(self) -> str:
        return (
            f"EnumNode(id={self.node_id}, level={self.level}, "
            f"coeff={self.coefficient}, norm²={self.partial_norm_sq:.4f})"
        )


class EnumerationTree:
    """Tree-based systematic enumeration of lattice points.

    Builds an explicit enumeration tree where each level corresponds
    to a basis vector coefficient. Supports various traversal orders
    and pruning strategies for systematic search.
    """

    def __init__(
        self,
        max_nodes: int = 50000,
        pruning: Optional[PruningBounds] = None,
    ) -> None:
        self._max_nodes = max_nodes
        self._pruning = pruning
        self._gs = GramSchmidt()
        self._nodes: Dict[int, EnumerationNode] = {}
        self._next_id = 0
        self._solutions: List[npt.NDArray[np.float64]] = []

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_solutions(self) -> int:
        return len(self._solutions)

    @property
    def solutions(self) -> List[npt.NDArray[np.float64]]:
        return list(self._solutions)

    def build(
        self,
        basis: npt.NDArray[np.float64],
        radius: float,
        center: Optional[npt.NDArray[np.float64]] = None,
    ) -> List[npt.NDArray[np.float64]]:
        """Build the enumeration tree and return solutions.

        Args:
            basis: n × d lattice basis.
            radius: Enumeration radius.
            center: Center point.

        Returns:
            List of lattice vectors found.
        """
        B = np.array(basis, dtype=np.float64)
        n, d = B.shape

        if center is None:
            center = np.zeros(d, dtype=np.float64)

        gs = self._gs.orthogonalize(B)
        mu = gs.mu_coefficients
        norms_sq = gs.norms_squared

        # Project center
        center_proj = np.zeros(n, dtype=np.float64)
        remaining = center.copy()
        for i in range(n - 1, -1, -1):
            if norms_sq[i] > 1e-15:
                center_proj[i] = np.dot(remaining, gs.orthogonal_basis[i]) / norms_sq[i]
                remaining -= center_proj[i] * B[i]

        self._nodes.clear()
        self._solutions.clear()
        self._next_id = 0

        R_sq = radius * radius

        # Create root
        root = self._new_node(level=n, coefficient=0, partial_norm_sq=0.0)

        # BFS-style construction
        queue: deque = deque()
        queue.append((root.node_id, n - 1, 0.0, np.zeros(n, dtype=np.float64)))

        while queue and len(self._nodes) < self._max_nodes:
            parent_id, level, partial_sq, coeffs = queue.popleft()

            if level < 0:
                # Complete solution
                vec = coeffs.astype(np.int64) @ B
                if np.any(np.abs(vec) > 1e-10):
                    self._solutions.append(vec)
                    self._nodes[parent_id].is_solution = True
                continue

            if norms_sq[level] < 1e-15:
                new_coeffs = coeffs.copy()
                new_coeffs[level] = 0.0
                child = self._new_node(level, 0, partial_sq, parent_id)
                queue.append((child.node_id, level - 1, partial_sq, new_coeffs))
                continue

            # Center value at this level
            c_val = center_proj[level]
            for j in range(level + 1, n):
                c_val -= mu[j, level] * coeffs[j]

            remaining_sq = R_sq - partial_sq
            if remaining_sq < -1e-10:
                continue

            # Apply pruning
            if self._pruning is not None and level < self._pruning.dimension:
                remaining_sq = min(remaining_sq, self._pruning.radii[level] ** 2 - partial_sq)
                if remaining_sq < -1e-10:
                    continue

            bound = math.sqrt(max(0, remaining_sq / norms_sq[level]))
            lo = math.ceil(c_val - bound)
            hi = math.floor(c_val + bound)

            c_round = round(c_val)
            for v in LatticeEnumerator._zigzag_range(c_round, lo, hi):
                new_partial = partial_sq + norms_sq[level] * (v - c_val) ** 2
                if new_partial > R_sq + 1e-10:
                    continue

                child = self._new_node(level, v, new_partial, parent_id)
                new_coeffs = coeffs.copy()
                new_coeffs[level] = float(v)
                queue.append((child.node_id, level - 1, new_partial, new_coeffs))

                if len(self._nodes) >= self._max_nodes:
                    break

        return self._solutions

    def _new_node(
        self,
        level: int,
        coefficient: int,
        partial_norm_sq: float,
        parent_id: Optional[int] = None,
    ) -> EnumerationNode:
        """Create and register a new enumeration tree node."""
        node = EnumerationNode(
            node_id=self._next_id,
            level=level,
            coefficient=coefficient,
            partial_norm_sq=partial_norm_sq,
            parent_id=parent_id,
        )
        self._nodes[self._next_id] = node
        if parent_id is not None and parent_id in self._nodes:
            self._nodes[parent_id].children_ids.append(self._next_id)
        self._next_id += 1
        return node

    def statistics(self) -> Dict[str, int]:
        """Return enumeration tree statistics."""
        total = len(self._nodes)
        leaves = sum(1 for n in self._nodes.values() if not n.children_ids)
        solutions = len(self._solutions)
        max_depth = max((n.level for n in self._nodes.values()), default=0)
        return {
            "total_nodes": total,
            "leaf_nodes": leaves,
            "solutions": solutions,
            "max_depth": max_depth,
        }

    def prune(
        self,
        predicate: Callable[[EnumerationNode], bool],
    ) -> int:
        """Prune nodes satisfying a predicate. Returns count of pruned nodes."""
        to_prune = [
            nid for nid, node in self._nodes.items()
            if predicate(node)
        ]
        for nid in to_prune:
            self._remove_subtree(nid)
        return len(to_prune)

    def _remove_subtree(self, node_id: int) -> None:
        """Remove a node and all its descendants."""
        if node_id not in self._nodes:
            return
        node = self._nodes[node_id]
        for child_id in list(node.children_ids):
            self._remove_subtree(child_id)
        # Remove from parent
        if node.parent_id is not None and node.parent_id in self._nodes:
            parent = self._nodes[node.parent_id]
            if node_id in parent.children_ids:
                parent.children_ids.remove(node_id)
        del self._nodes[node_id]
