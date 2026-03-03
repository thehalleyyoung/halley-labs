"""
Mechanism-specific lattice structures for DP mechanism synthesis.

Provides lattice operations over discrete mechanism parameter spaces,
Pareto front enumeration, dominance checking, projection, and
discrete optimization routines tailored to differential privacy.
"""

from __future__ import annotations

import heapq
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import numpy.typing as npt
from scipy.optimize import linprog

from dp_forge.types import LatticePoint, PrivacyBudget, QuerySpec


# ---------------------------------------------------------------------------
# Feasibility Checker
# ---------------------------------------------------------------------------


class FeasibilityChecker:
    """Check if a mechanism (lattice point) satisfies differential privacy.

    Verifies (ε, δ)-DP constraints for a given mechanism probability
    matrix M[i,j] by checking all pairs of adjacent database rows.
    """

    def __init__(self, tol: float = 1e-8) -> None:
        self._tol = tol

    def check_pure_dp(
        self,
        mechanism: npt.NDArray[np.float64],
        epsilon: float,
        adjacency: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[bool, Optional[Tuple[int, int, int, float]]]:
        """Check if mechanism satisfies ε-differential privacy.

        For all adjacent (i, i') and all outputs j:
            M[i,j] ≤ exp(ε) · M[i',j]

        Args:
            mechanism: n × k probability matrix.
            epsilon: Privacy parameter.
            adjacency: List of adjacent pairs. Default: all consecutive.

        Returns:
            Tuple of (is_feasible, worst_violation).
            If infeasible, violation = (i, i', j, ratio).
        """
        n, k = mechanism.shape
        exp_eps = math.exp(epsilon)

        if adjacency is None:
            adjacency = [(i, i + 1) for i in range(n - 1)]

        worst_violation = None
        worst_ratio = 0.0

        for i, ip in adjacency:
            for j in range(k):
                m_ij = mechanism[i, j]
                m_ipj = mechanism[ip, j]

                # Check M[i,j] ≤ exp(ε) · M[i',j]
                if m_ipj > self._tol:
                    ratio = m_ij / m_ipj
                    if ratio > exp_eps + self._tol:
                        excess = ratio / exp_eps
                        if excess > worst_ratio:
                            worst_ratio = excess
                            worst_violation = (i, ip, j, ratio)
                elif m_ij > self._tol:
                    worst_violation = (i, ip, j, float('inf'))
                    return False, worst_violation

                # Check M[i',j] ≤ exp(ε) · M[i,j]
                if m_ij > self._tol:
                    ratio = m_ipj / m_ij
                    if ratio > exp_eps + self._tol:
                        excess = ratio / exp_eps
                        if excess > worst_ratio:
                            worst_ratio = excess
                            worst_violation = (ip, i, j, ratio)
                elif m_ipj > self._tol:
                    worst_violation = (ip, i, j, float('inf'))
                    return False, worst_violation

        if worst_violation is not None:
            return False, worst_violation
        return True, None

    def check_approximate_dp(
        self,
        mechanism: npt.NDArray[np.float64],
        epsilon: float,
        delta: float,
        adjacency: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[bool, float]:
        """Check (ε, δ)-differential privacy.

        For all adjacent (i, i') and all subsets S of outputs:
            Pr[M(i) ∈ S] ≤ exp(ε) · Pr[M(i') ∈ S] + δ

        Uses the per-output formulation:
            M[i,j] ≤ exp(ε) · M[i',j] + δ/k  (sufficient condition)

        Args:
            mechanism: n × k probability matrix.
            epsilon: Privacy parameter.
            delta: Relaxation parameter.
            adjacency: List of adjacent pairs.

        Returns:
            Tuple of (is_feasible, max_delta_needed).
        """
        n, k = mechanism.shape
        exp_eps = math.exp(epsilon)

        if adjacency is None:
            adjacency = [(i, i + 1) for i in range(n - 1)]

        max_delta = 0.0

        for i, ip in adjacency:
            for j in range(k):
                excess = mechanism[i, j] - exp_eps * mechanism[ip, j]
                if excess > self._tol:
                    max_delta = max(max_delta, excess)

                excess_rev = mechanism[ip, j] - exp_eps * mechanism[i, j]
                if excess_rev > self._tol:
                    max_delta = max(max_delta, excess_rev)

        is_feasible = max_delta <= delta + self._tol
        return is_feasible, max_delta

    def check_normalization(
        self,
        mechanism: npt.NDArray[np.float64],
    ) -> Tuple[bool, float]:
        """Check if mechanism rows sum to 1 (valid probability distribution).

        Returns:
            Tuple of (is_normalized, max_deviation_from_1).
        """
        row_sums = mechanism.sum(axis=1)
        max_dev = float(np.max(np.abs(row_sums - 1.0)))
        return max_dev < self._tol, max_dev

    def check_nonnegativity(
        self,
        mechanism: npt.NDArray[np.float64],
    ) -> Tuple[bool, float]:
        """Check if all mechanism probabilities are non-negative."""
        min_val = float(np.min(mechanism))
        return min_val >= -self._tol, min_val

    def full_check(
        self,
        mechanism: npt.NDArray[np.float64],
        epsilon: float,
        delta: float = 0.0,
        adjacency: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        """Run all feasibility checks.

        Returns:
            Dict with results of all checks.
        """
        result: Dict[str, Any] = {}

        nn_ok, nn_val = self.check_nonnegativity(mechanism)
        result["nonnegative"] = nn_ok
        result["min_probability"] = nn_val

        norm_ok, norm_dev = self.check_normalization(mechanism)
        result["normalized"] = norm_ok
        result["normalization_deviation"] = norm_dev

        if delta == 0.0:
            dp_ok, violation = self.check_pure_dp(mechanism, epsilon, adjacency)
            result["dp_feasible"] = dp_ok
            result["dp_violation"] = violation
        else:
            dp_ok, max_delta = self.check_approximate_dp(
                mechanism, epsilon, delta, adjacency
            )
            result["dp_feasible"] = dp_ok
            result["max_delta_needed"] = max_delta

        result["feasible"] = nn_ok and norm_ok and dp_ok
        return result


# ---------------------------------------------------------------------------
# Dominance Relation
# ---------------------------------------------------------------------------


class DominanceRelation:
    """Privacy-accuracy dominance checking for mechanism pairs.

    Mechanism A dominates B if A has equal or better accuracy
    AND equal or better privacy guarantee.
    """

    def __init__(self, tol: float = 1e-8) -> None:
        self._tol = tol
        self._checker = FeasibilityChecker(tol=tol)

    def dominates(
        self,
        mech_a: npt.NDArray[np.float64],
        mech_b: npt.NDArray[np.float64],
        query_values: npt.NDArray[np.float64],
        epsilon: float,
    ) -> bool:
        """Check if mechanism A dominates mechanism B.

        A dominates B iff:
        1. accuracy(A) ≥ accuracy(B) (lower expected loss)
        2. privacy(A) ≥ privacy(B) (A satisfies at least same ε)

        Args:
            mech_a: n × k mechanism A.
            mech_b: n × k mechanism B.
            query_values: Query output values.
            epsilon: Privacy parameter.

        Returns:
            True if A dominates B.
        """
        loss_a = self._expected_loss(mech_a, query_values)
        loss_b = self._expected_loss(mech_b, query_values)

        # A must have equal or better accuracy (lower loss)
        if loss_a > loss_b + self._tol:
            return False

        # A must be at least as private
        dp_a, _ = self._checker.check_pure_dp(mech_a, epsilon)
        dp_b, _ = self._checker.check_pure_dp(mech_b, epsilon)

        if dp_b and not dp_a:
            return False

        return True

    def is_dominated(
        self,
        mechanism: npt.NDArray[np.float64],
        others: List[npt.NDArray[np.float64]],
        query_values: npt.NDArray[np.float64],
        epsilon: float,
    ) -> bool:
        """Check if mechanism is dominated by any in the list."""
        for other in others:
            if self.dominates(other, mechanism, query_values, epsilon):
                return True
        return False

    def pareto_compare(
        self,
        mech_a: npt.NDArray[np.float64],
        mech_b: npt.NDArray[np.float64],
        query_values: npt.NDArray[np.float64],
        epsilon: float,
    ) -> str:
        """Compare two mechanisms on the Pareto front.

        Returns:
            "a_dominates", "b_dominates", or "incomparable".
        """
        if self.dominates(mech_a, mech_b, query_values, epsilon):
            return "a_dominates"
        elif self.dominates(mech_b, mech_a, query_values, epsilon):
            return "b_dominates"
        return "incomparable"

    def _expected_loss(
        self,
        mechanism: npt.NDArray[np.float64],
        query_values: npt.NDArray[np.float64],
    ) -> float:
        """Compute expected squared loss of a mechanism.

        E[loss] = Σ_i Σ_j M[i,j] · (q_i - o_j)²
        where o_j are uniformly spaced output values.
        """
        n, k = mechanism.shape
        q = query_values[:n]

        # Output grid
        if k > 1:
            outputs = np.linspace(q.min(), q.max(), k)
        else:
            outputs = np.array([q.mean()])

        total_loss = 0.0
        for i in range(min(n, len(q))):
            for j in range(k):
                total_loss += mechanism[i, j] * (q[i] - outputs[j]) ** 2

        return total_loss / n


# ---------------------------------------------------------------------------
# Pareto Front
# ---------------------------------------------------------------------------


class ParetoFront:
    """Pareto-optimal mechanism enumeration.

    Maintains a set of non-dominated mechanisms trading off
    between accuracy and mechanism complexity.
    """

    def __init__(self) -> None:
        self._mechanisms: List[npt.NDArray[np.float64]] = []
        self._objectives: List[float] = []
        self._complexities: List[int] = []
        self._checker = FeasibilityChecker()
        self._dominance = DominanceRelation()

    @property
    def size(self) -> int:
        return len(self._mechanisms)

    @property
    def mechanisms(self) -> List[npt.NDArray[np.float64]]:
        return list(self._mechanisms)

    @property
    def objective_values(self) -> List[float]:
        return list(self._objectives)

    @property
    def complexities(self) -> List[int]:
        return list(self._complexities)

    def add(
        self,
        mechanism: npt.NDArray[np.float64],
        objective: float,
        complexity: int,
        query_values: npt.NDArray[np.float64],
        epsilon: float,
    ) -> bool:
        """Add a mechanism to the Pareto front if non-dominated.

        Args:
            mechanism: n × k mechanism.
            objective: Objective value (lower is better).
            complexity: Mechanism complexity (e.g., support size).
            query_values: Query values for dominance checking.
            epsilon: Privacy parameter.

        Returns:
            True if mechanism was added (is Pareto-optimal).
        """
        # Check if dominated by existing front members
        for i, existing in enumerate(self._mechanisms):
            if (self._objectives[i] <= objective + 1e-10
                    and self._complexities[i] <= complexity):
                # Existing is at least as good on both criteria
                if (self._objectives[i] < objective - 1e-10
                        or self._complexities[i] < complexity):
                    return False

        # Remove dominated members
        keep = []
        for i in range(len(self._mechanisms)):
            if objective <= self._objectives[i] + 1e-10 and complexity <= self._complexities[i]:
                if objective < self._objectives[i] - 1e-10 or complexity < self._complexities[i]:
                    continue  # Remove this dominated member
            keep.append(i)

        self._mechanisms = [self._mechanisms[i] for i in keep]
        self._objectives = [self._objectives[i] for i in keep]
        self._complexities = [self._complexities[i] for i in keep]

        # Add new mechanism
        self._mechanisms.append(mechanism.copy())
        self._objectives.append(objective)
        self._complexities.append(complexity)

        return True

    def get_front(
        self,
    ) -> List[Tuple[npt.NDArray[np.float64], float, int]]:
        """Get all Pareto-optimal mechanisms sorted by objective.

        Returns:
            List of (mechanism, objective, complexity) tuples.
        """
        items = list(zip(self._mechanisms, self._objectives, self._complexities))
        items.sort(key=lambda x: x[1])
        return items

    def get_knee_point(self) -> Optional[Tuple[npt.NDArray[np.float64], float, int]]:
        """Find the 'knee' of the Pareto front.

        The knee point maximizes the distance to the line connecting
        the two extreme points of the front.
        """
        if len(self._mechanisms) < 3:
            if self._mechanisms:
                idx = min(range(len(self._objectives)), key=lambda i: self._objectives[i])
                return self._mechanisms[idx], self._objectives[idx], self._complexities[idx]
            return None

        front = self.get_front()
        objs = np.array([f[1] for f in front])
        comps = np.array([f[2] for f in front])

        # Normalize both axes
        obj_range = objs.max() - objs.min()
        comp_range = comps.max() - comps.min()
        if obj_range < 1e-10 or comp_range < 1e-10:
            return front[0]

        objs_norm = (objs - objs.min()) / obj_range
        comps_norm = (comps - comps.min()) / comp_range

        # Line from first to last point
        p1 = np.array([objs_norm[0], comps_norm[0]])
        p2 = np.array([objs_norm[-1], comps_norm[-1]])
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-10:
            return front[0]

        # Distance from each point to the line
        best_idx = 0
        best_dist = 0.0
        for i in range(len(front)):
            pt = np.array([objs_norm[i], comps_norm[i]])
            dist = abs(np.cross(line_vec, p1 - pt)) / line_len
            if dist > best_dist:
                best_dist = dist
                best_idx = i

        return front[best_idx]

    def hypervolume(
        self,
        reference: Tuple[float, int],
    ) -> float:
        """Compute the hypervolume indicator of the Pareto front.

        Args:
            reference: Reference point (obj_ref, complexity_ref).

        Returns:
            Hypervolume dominated by the front.
        """
        ref_obj, ref_comp = reference
        front = self.get_front()

        if not front:
            return 0.0

        # 2D hypervolume via sweep
        points = [(f[1], f[2]) for f in front if f[1] < ref_obj and f[2] < ref_comp]
        if not points:
            return 0.0

        points.sort(key=lambda p: p[0])

        hv = 0.0
        prev_comp = ref_comp
        for obj, comp in points:
            width = ref_obj - obj
            height = prev_comp - comp
            if height > 0:
                hv += width * height
            prev_comp = min(prev_comp, comp)

        return hv


# ---------------------------------------------------------------------------
# Lattice Projection
# ---------------------------------------------------------------------------


class LatticeProjection:
    """Project mechanism parameter space onto lower dimensions.

    Reduces the dimensionality of the mechanism lattice for
    visualization and approximate optimization.
    """

    def __init__(self) -> None:
        pass

    def pca_projection(
        self,
        mechanisms: List[npt.NDArray[np.float64]],
        target_dim: int = 2,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Project mechanisms onto principal components.

        Args:
            mechanisms: List of n × k mechanism matrices.
            target_dim: Target dimensionality.

        Returns:
            Tuple of (projected_coordinates, principal_components).
        """
        # Flatten mechanisms to vectors
        vectors = np.array([m.flatten() for m in mechanisms])
        n_samples, d = vectors.shape

        # Center data
        mean = vectors.mean(axis=0)
        centered = vectors - mean

        # SVD for PCA
        if n_samples < d:
            # Compute n×n covariance
            cov = centered @ centered.T
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort descending
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Principal components in original space
            k = min(target_dim, n_samples)
            components = np.zeros((k, d))
            for i in range(k):
                if eigenvalues[i] > 1e-10:
                    components[i] = centered.T @ eigenvectors[:, i]
                    components[i] /= np.linalg.norm(components[i])

            projected = centered @ components.T
        else:
            cov = centered.T @ centered / max(n_samples - 1, 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            k = min(target_dim, d)
            components = eigenvectors[:, idx[:k]].T
            projected = centered @ components.T

        return projected, components

    def privacy_accuracy_projection(
        self,
        mechanisms: List[npt.NDArray[np.float64]],
        query_values: npt.NDArray[np.float64],
        epsilon: float,
    ) -> npt.NDArray[np.float64]:
        """Project mechanisms to (accuracy, privacy_cost) 2D space.

        Args:
            mechanisms: List of mechanism matrices.
            query_values: Query output values.
            epsilon: Privacy parameter.

        Returns:
            N × 2 array of (accuracy_loss, privacy_cost) coordinates.
        """
        checker = FeasibilityChecker()
        dominance = DominanceRelation()
        n_mechs = len(mechanisms)
        coords = np.zeros((n_mechs, 2))

        for i, mech in enumerate(mechanisms):
            # Accuracy: expected L2 loss
            coords[i, 0] = dominance._expected_loss(mech, query_values)

            # Privacy cost: minimum ε needed
            coords[i, 1] = self._min_epsilon(mech)

        return coords

    def _min_epsilon(
        self,
        mechanism: npt.NDArray[np.float64],
    ) -> float:
        """Compute minimum ε for which mechanism satisfies pure DP."""
        n, k = mechanism.shape
        max_log_ratio = 0.0

        for i in range(n):
            for ip in range(i + 1, n):
                for j in range(k):
                    if mechanism[i, j] > 1e-15 and mechanism[ip, j] > 1e-15:
                        log_ratio = abs(math.log(mechanism[i, j] / mechanism[ip, j]))
                        max_log_ratio = max(max_log_ratio, log_ratio)
                    elif mechanism[i, j] > 1e-15 or mechanism[ip, j] > 1e-15:
                        return float('inf')

        return max_log_ratio

    def random_projection(
        self,
        mechanisms: List[npt.NDArray[np.float64]],
        target_dim: int = 2,
        seed: int = 42,
    ) -> npt.NDArray[np.float64]:
        """Johnson-Lindenstrauss random projection.

        Preserves pairwise distances approximately.

        Args:
            mechanisms: List of mechanism matrices.
            target_dim: Target dimensionality.
            seed: Random seed.

        Returns:
            N × target_dim projected coordinates.
        """
        rng = np.random.RandomState(seed)
        vectors = np.array([m.flatten() for m in mechanisms])
        d = vectors.shape[1]

        # Random projection matrix with JL guarantees
        projection = rng.randn(d, target_dim) / math.sqrt(target_dim)
        return vectors @ projection


# ---------------------------------------------------------------------------
# Mechanism Lattice
# ---------------------------------------------------------------------------


class MechanismLattice:
    """Lattice of discrete mechanisms ordered by accuracy.

    The lattice structure orders mechanism designs by refinement:
    coarser mechanisms (fewer support points) are lower in the lattice,
    and finer mechanisms (more support points) are higher.

    Lattice operations:
    - join (∨): least upper bound (finest common coarsening)
    - meet (∧): greatest lower bound (coarsest common refinement)
    """

    def __init__(
        self,
        n_inputs: int,
        epsilon: float,
        delta: float = 0.0,
        min_support: int = 2,
        max_support: int = 50,
    ) -> None:
        self._n_inputs = n_inputs
        self._epsilon = epsilon
        self._delta = delta
        self._min_support = min_support
        self._max_support = max_support
        self._checker = FeasibilityChecker()
        self._nodes: Dict[int, LatticePoint] = {}
        self._edges: Dict[int, List[int]] = {}
        self._next_id = 0

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def root(self) -> LatticePoint:
        """Return the coarsest mechanism (minimum support)."""
        k = self._min_support
        coords = np.zeros(self._n_inputs * k, dtype=np.float64)
        # Initialize with uniform mechanism
        for i in range(self._n_inputs):
            for j in range(k):
                coords[i * k + j] = 1.0 / k
        point = LatticePoint(coordinates=coords)
        self._register(point)
        return point

    def successors(self, point: LatticePoint) -> List[LatticePoint]:
        """Generate successors by refining the mechanism.

        Refinement strategies:
        1. Add a support point (increase k)
        2. Perturb probabilities to improve accuracy
        """
        successors = []
        coords = point.coordinates
        n = self._n_inputs

        # Infer current k from coordinates
        k = len(coords) // n if n > 0 else 2

        # Strategy 1: add support point
        if k < self._max_support:
            new_k = k + 1
            new_coords = np.zeros(n * new_k, dtype=np.float64)
            for i in range(n):
                old_probs = coords[i * k:(i + 1) * k]
                # Redistribute: split last bin
                new_probs = np.zeros(new_k)
                new_probs[:k] = old_probs * (k / new_k)
                new_probs[k] = 1.0 - new_probs[:k].sum()
                new_probs = np.maximum(new_probs, 0.0)
                new_probs /= new_probs.sum()
                new_coords[i * new_k:(i + 1) * new_k] = new_probs
            succ = LatticePoint(coordinates=new_coords)
            self._register(succ)
            successors.append(succ)

        # Strategy 2: perturbations along each dimension
        step = 0.05
        for dim in range(min(len(coords), 20)):
            new_coords = coords.copy()
            new_coords[dim] += step
            # Re-normalize the row this dimension belongs to
            row_idx = dim // k
            row_start = row_idx * k
            row_end = row_start + k
            row = new_coords[row_start:row_end]
            row = np.maximum(row, 0.0)
            if row.sum() > 0:
                row /= row.sum()
            new_coords[row_start:row_end] = row
            succ = LatticePoint(coordinates=new_coords)
            self._register(succ)
            successors.append(succ)

        return successors

    def predecessors(self, point: LatticePoint) -> List[LatticePoint]:
        """Generate predecessors by coarsening the mechanism."""
        predecessors = []
        coords = point.coordinates
        n = self._n_inputs

        k = len(coords) // n if n > 0 else 2

        # Coarsen: merge adjacent support points
        if k > self._min_support:
            new_k = k - 1
            new_coords = np.zeros(n * new_k, dtype=np.float64)
            for i in range(n):
                old_probs = coords[i * k:(i + 1) * k]
                # Merge last two bins
                new_probs = np.zeros(new_k)
                new_probs[:new_k - 1] = old_probs[:new_k - 1]
                new_probs[new_k - 1] = old_probs[new_k - 1:].sum()
                new_coords[i * new_k:(i + 1) * new_k] = new_probs
            pred = LatticePoint(coordinates=new_coords)
            self._register(pred)
            predecessors.append(pred)

        return predecessors

    def join(self, a: LatticePoint, b: LatticePoint) -> LatticePoint:
        """Compute join (least upper bound) of two lattice points.

        The join uses the finer grid of the two mechanisms and
        averages their probability assignments.
        """
        n = self._n_inputs
        k_a = len(a.coordinates) // n
        k_b = len(b.coordinates) // n
        k_join = max(k_a, k_b)

        coords = np.zeros(n * k_join, dtype=np.float64)
        for i in range(n):
            probs_a = self._interpolate(a.coordinates[i * k_a:(i + 1) * k_a], k_join)
            probs_b = self._interpolate(b.coordinates[i * k_b:(i + 1) * k_b], k_join)
            # Join: element-wise maximum (least upper bound)
            row = np.maximum(probs_a, probs_b)
            if row.sum() > 0:
                row /= row.sum()
            coords[i * k_join:(i + 1) * k_join] = row

        point = LatticePoint(coordinates=coords)
        self._register(point)
        return point

    def meet(self, a: LatticePoint, b: LatticePoint) -> LatticePoint:
        """Compute meet (greatest lower bound) of two lattice points.

        The meet uses the coarser grid and takes the element-wise
        minimum of probability assignments.
        """
        n = self._n_inputs
        k_a = len(a.coordinates) // n
        k_b = len(b.coordinates) // n
        k_meet = min(k_a, k_b)

        coords = np.zeros(n * k_meet, dtype=np.float64)
        for i in range(n):
            probs_a = self._interpolate(a.coordinates[i * k_a:(i + 1) * k_a], k_meet)
            probs_b = self._interpolate(b.coordinates[i * k_b:(i + 1) * k_b], k_meet)
            # Meet: element-wise minimum
            row = np.minimum(probs_a, probs_b)
            if row.sum() > 0:
                row /= row.sum()
            coords[i * k_meet:(i + 1) * k_meet] = row

        point = LatticePoint(coordinates=coords)
        self._register(point)
        return point

    def _interpolate(
        self,
        probs: npt.NDArray[np.float64],
        target_k: int,
    ) -> npt.NDArray[np.float64]:
        """Interpolate a probability vector to a different grid size."""
        k = len(probs)
        if k == target_k:
            return probs.copy()

        result = np.zeros(target_k)
        for j in range(target_k):
            # Map target bin j to source position
            src_pos = j * (k - 1) / max(target_k - 1, 1) if target_k > 1 else 0
            src_lo = int(math.floor(src_pos))
            src_hi = min(src_lo + 1, k - 1)
            frac = src_pos - src_lo
            result[j] = (1 - frac) * probs[src_lo] + frac * probs[src_hi]

        # Normalize
        total = result.sum()
        if total > 0:
            result /= total
        return result

    def is_feasible(self, point: LatticePoint) -> bool:
        """Check if a lattice point yields a DP-feasible mechanism."""
        n = self._n_inputs
        k = len(point.coordinates) // n
        mechanism = point.coordinates.reshape(n, k)

        if self._delta == 0.0:
            ok, _ = self._checker.check_pure_dp(mechanism, self._epsilon)
        else:
            ok, _ = self._checker.check_approximate_dp(
                mechanism, self._epsilon, self._delta
            )
        point.feasible = ok
        return ok

    def evaluate(
        self,
        point: LatticePoint,
        query_values: npt.NDArray[np.float64],
    ) -> float:
        """Evaluate the objective (expected loss) at a lattice point."""
        n = self._n_inputs
        k = len(point.coordinates) // n
        mechanism = point.coordinates.reshape(n, k)

        dominance = DominanceRelation()
        loss = dominance._expected_loss(mechanism, query_values)
        point.objective_value = loss
        return loss

    def _register(self, point: LatticePoint) -> int:
        """Register a lattice point and return its ID."""
        pid = self._next_id
        self._nodes[pid] = point
        self._next_id += 1
        return pid


# ---------------------------------------------------------------------------
# Discrete Optimizer
# ---------------------------------------------------------------------------


class DiscreteOptimizer:
    """Optimize over the mechanism lattice.

    Combines lattice enumeration with branch-and-bound search
    to find the optimal discrete mechanism satisfying DP constraints.
    """

    def __init__(
        self,
        max_iterations: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        self._max_iter = max_iterations
        self._tol = tol

    def optimize(
        self,
        lattice: MechanismLattice,
        query_values: npt.NDArray[np.float64],
        initial: Optional[LatticePoint] = None,
    ) -> Tuple[LatticePoint, float]:
        """Find the optimal feasible lattice point.

        Uses a best-first search over the lattice, expanding
        the most promising nodes first.

        Args:
            lattice: Mechanism lattice to search.
            query_values: Query values for loss computation.
            initial: Initial lattice point (default: root).

        Returns:
            Tuple of (best_point, best_objective).
        """
        if initial is None:
            initial = lattice.root()

        # Evaluate initial point
        obj = lattice.evaluate(initial, query_values)
        feas = lattice.is_feasible(initial)

        best_point = initial
        best_obj = obj if feas else float('inf')

        # Priority queue: (objective, point)
        heap: List[Tuple[float, int, LatticePoint]] = [(obj, 0, initial)]
        visited: Set[int] = set()
        counter = 1

        for iteration in range(self._max_iter):
            if not heap:
                break

            _, _, current = heapq.heappop(heap)
            point_hash = hash(tuple(np.round(current.coordinates, 6)))
            if point_hash in visited:
                continue
            visited.add(point_hash)

            # Evaluate current
            obj = lattice.evaluate(current, query_values)
            feas = lattice.is_feasible(current)

            if feas and obj < best_obj:
                best_obj = obj
                best_point = current

            # Expand successors
            for succ in lattice.successors(current):
                succ_hash = hash(tuple(np.round(succ.coordinates, 6)))
                if succ_hash not in visited:
                    succ_obj = lattice.evaluate(succ, query_values)
                    if succ_obj < best_obj + self._tol * 100:
                        heapq.heappush(heap, (succ_obj, counter, succ))
                        counter += 1

        return best_point, best_obj

    def local_search(
        self,
        lattice: MechanismLattice,
        query_values: npt.NDArray[np.float64],
        start: LatticePoint,
        max_no_improve: int = 20,
    ) -> Tuple[LatticePoint, float]:
        """Local search from a starting point via lattice neighborhood.

        Args:
            lattice: Mechanism lattice.
            query_values: Query values.
            start: Starting lattice point.
            max_no_improve: Max iterations without improvement before stopping.

        Returns:
            Tuple of (best_point, best_objective).
        """
        current = start
        current_obj = lattice.evaluate(current, query_values)
        if not lattice.is_feasible(current):
            current_obj = float('inf')

        best = current
        best_obj = current_obj
        no_improve = 0

        for _ in range(self._max_iter):
            improved = False
            neighbors = lattice.successors(current) + lattice.predecessors(current)

            for neighbor in neighbors:
                if not lattice.is_feasible(neighbor):
                    continue
                obj = lattice.evaluate(neighbor, query_values)
                if obj < best_obj - self._tol:
                    best = neighbor
                    best_obj = obj
                    current = neighbor
                    current_obj = obj
                    improved = True
                    break

            if not improved:
                no_improve += 1
                if no_improve >= max_no_improve:
                    break
            else:
                no_improve = 0

        return best, best_obj

    def multi_start(
        self,
        lattice: MechanismLattice,
        query_values: npt.NDArray[np.float64],
        n_starts: int = 5,
    ) -> Tuple[LatticePoint, float]:
        """Multi-start optimization with random restarts.

        Args:
            lattice: Mechanism lattice.
            query_values: Query values.
            n_starts: Number of random starting points.

        Returns:
            Tuple of (best_point, best_objective).
        """
        overall_best = None
        overall_best_obj = float('inf')

        for trial in range(n_starts):
            # Generate random starting point
            root = lattice.root()
            n = lattice.n_inputs
            k = len(root.coordinates) // n

            coords = np.random.dirichlet(np.ones(k), size=n).flatten()
            start = LatticePoint(coordinates=coords)

            point, obj = self.local_search(lattice, query_values, start)
            if obj < overall_best_obj:
                overall_best_obj = obj
                overall_best = point

        if overall_best is None:
            overall_best = lattice.root()
            overall_best_obj = lattice.evaluate(overall_best, query_values)

        return overall_best, overall_best_obj
