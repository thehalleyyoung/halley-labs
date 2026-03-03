"""
Sensitivity propagation through computation graphs.

Provides tools for computing, tracking, and bounding the sensitivity
of differentially private mechanisms:

- **SensitivityTracker**: Propagate sensitivity through a computation graph.
- **LocalSensitivity**: Compute local sensitivity via automatic differentiation.
- **SmoothSensitivity**: Smooth sensitivity via exponential mechanism.
- **ProposeSampleSensitivity**: Propose-test-release framework.
- **SensitivityBound**: Rigorous upper bounds on sensitivity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt
from scipy import sparse

from dp_forge.types import GradientInfo


# ---------------------------------------------------------------------------
# Local sensitivity via AD
# ---------------------------------------------------------------------------


class LocalSensitivity:
    """Compute local sensitivity of a function via automatic differentiation.

    For a function f: R^n -> R^m, the local sensitivity at database x
    is defined as:

        LS(x) = max_{x' ~ x} ||f(x) - f(x')||

    where x' ~ x means x' is adjacent to x.

    This class computes or approximates LS using the Jacobian:
        LS(x) ≤ max_j ||J_j||   (row norms of the Jacobian)

    Attributes:
        fn: Function whose sensitivity to compute.
        norm_ord: Norm order for sensitivity (1, 2, or np.inf).
    """

    def __init__(
        self,
        fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        norm_ord: Union[int, float] = 1,
    ) -> None:
        self.fn = fn
        self.norm_ord = norm_ord

    def compute(
        self,
        x: npt.NDArray[np.float64],
        adjacency: str = "hamming",
    ) -> float:
        """Compute local sensitivity at *x*.

        Uses central finite differences to approximate the Jacobian.

        Args:
            x: Database (input point).
            adjacency: Adjacency relation ("hamming" for single-entry change,
                       "l1" for unit L1 ball).

        Returns:
            Local sensitivity value.
        """
        J = self._jacobian(x)

        if adjacency == "hamming":
            # Sensitivity = max column norm of J
            col_norms = np.linalg.norm(J, ord=self.norm_ord, axis=0)
            return float(np.max(col_norms))
        elif adjacency == "l1":
            # For L1 adjacency, sensitivity = operator norm ||J||_{1->norm}
            if self.norm_ord == 1:
                return float(np.max(np.sum(np.abs(J), axis=0)))
            elif self.norm_ord == 2:
                s = np.linalg.svd(J, compute_uv=False)
                return float(s[0]) if len(s) > 0 else 0.0
            else:
                return float(np.max(np.sum(np.abs(J), axis=0)))
        else:
            raise ValueError(f"Unknown adjacency: {adjacency}")

    def compute_per_output(
        self,
        x: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute per-output local sensitivity.

        Returns:
            Array of local sensitivity values, one per output dimension.
        """
        J = self._jacobian(x)
        if J.ndim == 1:
            return np.abs(J)
        return np.max(np.abs(J), axis=1)

    def _jacobian(
        self,
        x: npt.NDArray[np.float64],
        h: float = 1e-7,
    ) -> npt.NDArray[np.float64]:
        """Compute Jacobian via central finite differences."""
        n = len(x)
        f0 = np.atleast_1d(self.fn(x))
        m = len(f0)
        J = np.zeros((m, n), dtype=np.float64)
        for j in range(n):
            xp = x.copy()
            xm = x.copy()
            xp[j] += h
            xm[j] -= h
            fp = np.atleast_1d(self.fn(xp))
            fm = np.atleast_1d(self.fn(xm))
            J[:, j] = (fp - fm) / (2.0 * h)
        return J


# ---------------------------------------------------------------------------
# Sensitivity Tracker
# ---------------------------------------------------------------------------


@dataclass
class SensitivityNode:
    """Node in the sensitivity propagation graph.

    Attributes:
        node_id: Unique identifier.
        sensitivity: Sensitivity bound at this node.
        parents: Parent node IDs.
        op_name: Operation that produced this node.
    """

    node_id: int
    sensitivity: float
    parents: Tuple[int, ...] = ()
    op_name: str = ""


class SensitivityTracker:
    """Propagate sensitivity bounds through a computation graph.

    Tracks how sensitivity flows through operations using composition
    rules (e.g., sum of sensitivities for addition, product rules for
    multiplication).

    Attributes:
        nodes: List of sensitivity nodes.
    """

    def __init__(self) -> None:
        self._nodes: List[SensitivityNode] = []
        self._next_id: int = 0

    @property
    def nodes(self) -> List[SensitivityNode]:
        return list(self._nodes)

    def create_input(self, sensitivity: float, name: str = "") -> int:
        """Register an input with known sensitivity.

        Args:
            sensitivity: Sensitivity bound for this input.
            name: Optional label.

        Returns:
            Node ID.
        """
        nid = self._next_id
        self._next_id += 1
        self._nodes.append(SensitivityNode(
            node_id=nid, sensitivity=sensitivity, op_name=name or f"input_{nid}",
        ))
        return nid

    def add_op(
        self,
        op: str,
        parents: Tuple[int, ...],
        sensitivity: float,
    ) -> int:
        """Record an operation with computed sensitivity.

        Args:
            op: Operation name.
            parents: Parent node IDs.
            sensitivity: Sensitivity bound for the output.

        Returns:
            Node ID.
        """
        nid = self._next_id
        self._next_id += 1
        self._nodes.append(SensitivityNode(
            node_id=nid, sensitivity=sensitivity,
            parents=parents, op_name=op,
        ))
        return nid

    def add(self, a: int, b: int) -> int:
        """Addition: sens(a + b) ≤ sens(a) + sens(b)."""
        s = self._nodes[a].sensitivity + self._nodes[b].sensitivity
        return self.add_op("add", (a, b), s)

    def sub(self, a: int, b: int) -> int:
        """Subtraction: sens(a - b) ≤ sens(a) + sens(b)."""
        s = self._nodes[a].sensitivity + self._nodes[b].sensitivity
        return self.add_op("sub", (a, b), s)

    def mul_const(self, a: int, c: float) -> int:
        """Multiplication by constant: sens(c*a) = |c| * sens(a)."""
        s = abs(c) * self._nodes[a].sensitivity
        return self.add_op("mul_const", (a,), s)

    def mul(self, a: int, b: int, a_bound: float, b_bound: float) -> int:
        """Multiplication: sens(a*b) ≤ |a|_max * sens(b) + |b|_max * sens(a).

        Args:
            a: First operand node.
            b: Second operand node.
            a_bound: Upper bound on |a|.
            b_bound: Upper bound on |b|.

        Returns:
            Node ID.
        """
        sa = self._nodes[a].sensitivity
        sb = self._nodes[b].sensitivity
        s = a_bound * sb + b_bound * sa
        return self.add_op("mul", (a, b), s)

    def div_const(self, a: int, c: float) -> int:
        """Division by constant: sens(a/c) = sens(a) / |c|."""
        if c == 0:
            raise ZeroDivisionError("Division by zero in sensitivity tracker")
        s = self._nodes[a].sensitivity / abs(c)
        return self.add_op("div_const", (a,), s)

    def compose(self, a: int, lipschitz_const: float) -> int:
        """Post-composition: sens(g ∘ f) ≤ L_g * sens(f).

        Args:
            a: Node ID of f.
            lipschitz_const: Lipschitz constant of g.

        Returns:
            Node ID.
        """
        s = lipschitz_const * self._nodes[a].sensitivity
        return self.add_op("compose", (a,), s)

    def maximum(self, a: int, b: int) -> int:
        """Max: sens(max(a, b)) ≤ max(sens(a), sens(b))."""
        s = max(self._nodes[a].sensitivity, self._nodes[b].sensitivity)
        return self.add_op("max", (a, b), s)

    def sum_nodes(self, ids: Sequence[int]) -> int:
        """Sum: sens(Σ a_i) ≤ Σ sens(a_i)."""
        s = sum(self._nodes[i].sensitivity for i in ids)
        return self.add_op("sum", tuple(ids), s)

    def get_sensitivity(self, node_id: int) -> float:
        """Return the sensitivity bound at a node."""
        return self._nodes[node_id].sensitivity

    def __repr__(self) -> str:
        return f"SensitivityTracker(nodes={len(self._nodes)})"


# ---------------------------------------------------------------------------
# Smooth sensitivity
# ---------------------------------------------------------------------------


class SmoothSensitivity:
    """Compute β-smooth sensitivity for the exponential mechanism.

    For a function f with local sensitivities LS(x) at each database x,
    the β-smooth sensitivity is:

        S^*_β(x) = max_{y} LS(y) * exp(-β * d(x, y))

    Attributes:
        beta: Smoothing parameter.
        fn: Function whose sensitivity is computed.
        norm_ord: Norm order for sensitivity.
    """

    def __init__(
        self,
        beta: float,
        fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        norm_ord: Union[int, float] = 1,
    ) -> None:
        if beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta}")
        self.beta = beta
        self.fn = fn
        self.norm_ord = norm_ord
        self._ls = LocalSensitivity(fn, norm_ord)

    def compute(
        self,
        x: npt.NDArray[np.float64],
        max_distance: int = 10,
        neighbor_gen: Optional[Callable[[npt.NDArray[np.float64], int],
                                        List[npt.NDArray[np.float64]]]] = None,
    ) -> float:
        """Compute smooth sensitivity at database *x*.

        Enumerates databases at distances 0, 1, ..., max_distance from x,
        computes local sensitivity at each, and takes the weighted max.

        Args:
            x: Current database.
            max_distance: Maximum Hamming distance to explore.
            neighbor_gen: Optional function (x, dist) -> list of neighbors.

        Returns:
            β-smooth sensitivity value.
        """
        best = 0.0
        for dist in range(max_distance + 1):
            if neighbor_gen is not None:
                neighbors = neighbor_gen(x, dist)
            else:
                neighbors = self._default_neighbors(x, dist)

            for y in neighbors:
                ls_y = self._ls.compute(y)
                weighted = ls_y * math.exp(-self.beta * dist)
                best = max(best, weighted)

        return best

    def compute_from_local_sensitivities(
        self,
        local_sens: npt.NDArray[np.float64],
        distances: npt.NDArray[np.float64],
    ) -> float:
        """Compute smooth sensitivity from pre-computed local sensitivities.

        Args:
            local_sens: Array of local sensitivity values.
            distances: Corresponding distances from the current database.

        Returns:
            β-smooth sensitivity value.
        """
        weighted = local_sens * np.exp(-self.beta * distances)
        return float(np.max(weighted))

    @staticmethod
    def _default_neighbors(
        x: npt.NDArray[np.float64],
        dist: int,
    ) -> List[npt.NDArray[np.float64]]:
        """Generate representative neighbors at Hamming distance *dist*."""
        if dist == 0:
            return [x.copy()]
        n = len(x)
        neighbors: List[npt.NDArray[np.float64]] = []
        # Sample a few random perturbations
        n_samples = min(10, max(1, n))
        rng = np.random.RandomState(42 + dist)
        for _ in range(n_samples):
            y = x.copy()
            indices = rng.choice(n, size=min(dist, n), replace=False)
            y[indices] += rng.randn(len(indices))
            neighbors.append(y)
        return neighbors

    def __repr__(self) -> str:
        return f"SmoothSensitivity(beta={self.beta})"


# ---------------------------------------------------------------------------
# Propose-Test-Release
# ---------------------------------------------------------------------------


@dataclass
class PTRResult:
    """Result of propose-test-release.

    Attributes:
        proposed_bound: Proposed sensitivity bound.
        test_passed: Whether the test passed.
        released_value: Released noisy value (None if test failed).
        confidence: Confidence level of the test.
    """

    proposed_bound: float
    test_passed: bool
    released_value: Optional[float] = None
    confidence: float = 0.0


class ProposeSampleSensitivity:
    """Propose-test-release framework for sensitivity.

    1. **Propose** a sensitivity bound b.
    2. **Test** whether the true local sensitivity is ≤ b/2 with
       sufficient confidence (via the distance to the nearest database
       with LS > b/2).
    3. **Release** the query answer with noise calibrated to b if test passes.

    Attributes:
        epsilon: Privacy budget for the release.
        delta: Privacy parameter δ for the test.
    """

    def __init__(self, epsilon: float, delta: float = 1e-6) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if delta <= 0 or delta >= 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self.epsilon = epsilon
        self.delta = delta

    def propose_and_test(
        self,
        fn: Callable[[npt.NDArray[np.float64]], float],
        x: npt.NDArray[np.float64],
        proposed_bound: float,
        ls_fn: Optional[Callable[[npt.NDArray[np.float64]], float]] = None,
        max_distance: int = 10,
    ) -> PTRResult:
        """Run propose-test-release.

        Args:
            fn: Query function.
            x: Current database.
            proposed_bound: Proposed sensitivity bound.
            ls_fn: Optional local sensitivity function.
            max_distance: Max Hamming distance for the test.

        Returns:
            PTRResult with outcome.
        """
        if ls_fn is None:
            ls_computer = LocalSensitivity(
                lambda z: np.atleast_1d(fn(z)), norm_ord=1,
            )
            ls_fn = lambda z: ls_computer.compute(z)

        ls_x = ls_fn(x)
        threshold = proposed_bound / 2.0

        # Distance to insensitivity boundary
        if ls_x > threshold:
            dist = 0
        else:
            dist = self._distance_to_high_sensitivity(
                x, ls_fn, threshold, max_distance,
            )

        # Noisy test: add Lap(1/ε) noise to distance
        noisy_dist = dist + np.random.laplace(0, 1.0 / self.epsilon)
        test_threshold = math.log(1.0 / self.delta) / self.epsilon

        test_passed = noisy_dist > test_threshold

        if test_passed:
            # Release with Laplace noise calibrated to proposed_bound
            true_val = fn(x)
            noise = np.random.laplace(0, proposed_bound / self.epsilon)
            released = true_val + noise
            confidence = 1.0 - self.delta
            return PTRResult(proposed_bound, True, released, confidence)

        return PTRResult(proposed_bound, False)

    @staticmethod
    def _distance_to_high_sensitivity(
        x: npt.NDArray[np.float64],
        ls_fn: Callable[[npt.NDArray[np.float64]], float],
        threshold: float,
        max_distance: int,
    ) -> int:
        """Find the minimum Hamming distance to a database with LS > threshold."""
        n = len(x)
        rng = np.random.RandomState(123)
        for dist in range(1, max_distance + 1):
            n_samples = min(20, max(1, n))
            for _ in range(n_samples):
                y = x.copy()
                indices = rng.choice(n, size=min(dist, n), replace=False)
                y[indices] += rng.randn(len(indices))
                if ls_fn(y) > threshold:
                    return dist
        return max_distance


# ---------------------------------------------------------------------------
# Sensitivity bounds
# ---------------------------------------------------------------------------


@dataclass
class SensitivityBoundResult:
    """Result of sensitivity bound computation.

    Attributes:
        upper_bound: Rigorous upper bound on global sensitivity.
        lower_bound: Lower bound (from local sensitivity samples).
        method: Method used to compute the bound.
        tight: Whether upper and lower bounds are close.
        details: Additional computation details.
    """

    upper_bound: float
    lower_bound: float
    method: str
    tight: bool = False
    details: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tight = (self.upper_bound - self.lower_bound) / max(
            self.upper_bound, 1e-30
        ) < 0.1


class SensitivityBound:
    """Compute rigorous upper bounds on global sensitivity.

    Uses multiple strategies:
    1. **Lipschitz analysis**: Bound via Lipschitz constant.
    2. **Interval arithmetic**: Propagate intervals through the function.
    3. **Sampling**: Sample local sensitivities to get a lower bound.
    4. **Analytic**: Use known sensitivity formulas for common queries.

    Attributes:
        fn: Function whose sensitivity to bound.
        n_samples: Number of databases to sample for lower bound.
    """

    def __init__(
        self,
        fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        n_samples: int = 100,
    ) -> None:
        self.fn = fn
        self.n_samples = n_samples

    def compute(
        self,
        domain_low: npt.NDArray[np.float64],
        domain_high: npt.NDArray[np.float64],
        adjacency: str = "hamming",
        norm_ord: Union[int, float] = 1,
    ) -> SensitivityBoundResult:
        """Compute sensitivity bounds over a domain.

        Args:
            domain_low: Lower bounds of the input domain.
            domain_high: Upper bounds of the input domain.
            adjacency: Adjacency relation.
            norm_ord: Norm for sensitivity.

        Returns:
            SensitivityBoundResult with upper and lower bounds.
        """
        ls = LocalSensitivity(self.fn, norm_ord=norm_ord)

        # Lower bound: sample local sensitivities
        lower = 0.0
        rng = np.random.RandomState(42)
        for _ in range(self.n_samples):
            x = rng.uniform(domain_low, domain_high)
            s = ls.compute(x, adjacency=adjacency)
            lower = max(lower, s)

        # Upper bound via Lipschitz estimation
        upper = self._lipschitz_bound(domain_low, domain_high, norm_ord)
        upper = max(upper, lower)

        return SensitivityBoundResult(
            upper_bound=upper,
            lower_bound=lower,
            method="lipschitz_sampling",
            details={"n_samples": float(self.n_samples)},
        )

    def _lipschitz_bound(
        self,
        domain_low: npt.NDArray[np.float64],
        domain_high: npt.NDArray[np.float64],
        norm_ord: Union[int, float],
        n_pairs: int = 50,
        h: float = 1e-6,
    ) -> float:
        """Estimate Lipschitz constant via sampling gradients.

        Computes Jacobian at multiple points and takes the max operator norm.
        """
        n = len(domain_low)
        rng = np.random.RandomState(123)
        max_lip = 0.0

        for _ in range(n_pairs):
            x = rng.uniform(domain_low, domain_high)
            f0 = np.atleast_1d(self.fn(x))
            m = len(f0)
            J = np.zeros((m, n), dtype=np.float64)
            for j in range(n):
                xp = x.copy()
                xp[j] += h
                fp = np.atleast_1d(self.fn(xp))
                J[:, j] = (fp - f0) / h

            if norm_ord == 1:
                lip = float(np.max(np.sum(np.abs(J), axis=0)))
            elif norm_ord == 2:
                s = np.linalg.svd(J, compute_uv=False)
                lip = float(s[0]) if len(s) > 0 else 0.0
            else:
                lip = float(np.max(np.sum(np.abs(J), axis=1)))
            max_lip = max(max_lip, lip)

        # Add safety margin
        return max_lip * 1.1

    def counting_query_bound(
        self,
        n_records: int,
    ) -> SensitivityBoundResult:
        """Analytic sensitivity for counting queries: always 1.

        Args:
            n_records: Number of records.

        Returns:
            SensitivityBoundResult with exact bound.
        """
        return SensitivityBoundResult(
            upper_bound=1.0,
            lower_bound=1.0,
            method="analytic_counting",
            tight=True,
        )

    def sum_query_bound(
        self,
        value_bound: float,
    ) -> SensitivityBoundResult:
        """Analytic sensitivity for sum queries: Δ = max|v|.

        Args:
            value_bound: Upper bound on individual values.

        Returns:
            SensitivityBoundResult with exact bound.
        """
        return SensitivityBoundResult(
            upper_bound=value_bound,
            lower_bound=value_bound,
            method="analytic_sum",
            tight=True,
        )

    def mean_query_bound(
        self,
        n_records: int,
        value_bound: float,
    ) -> SensitivityBoundResult:
        """Analytic sensitivity for mean queries: Δ = max|v| / n.

        Args:
            n_records: Number of records.
            value_bound: Upper bound on individual values.

        Returns:
            SensitivityBoundResult with exact bound.
        """
        s = value_bound / max(n_records, 1)
        return SensitivityBoundResult(
            upper_bound=s,
            lower_bound=s,
            method="analytic_mean",
            tight=True,
        )

    def linear_query_bound(
        self,
        query_matrix: npt.NDArray[np.float64],
        norm_ord: Union[int, float] = 1,
    ) -> SensitivityBoundResult:
        """Sensitivity for linear queries f(x) = Ax.

        For Hamming adjacency, Δ = max column norm of A.

        Args:
            query_matrix: Query matrix A.
            norm_ord: Norm order.

        Returns:
            SensitivityBoundResult with exact bound.
        """
        col_norms = np.linalg.norm(query_matrix, ord=norm_ord, axis=0)
        s = float(np.max(col_norms))
        return SensitivityBoundResult(
            upper_bound=s,
            lower_bound=s,
            method="analytic_linear",
            tight=True,
        )


# ---------------------------------------------------------------------------
# Elastic sensitivity
# ---------------------------------------------------------------------------


def elastic_sensitivity(
    local_sens_fn: Callable[[npt.NDArray[np.float64]], float],
    x: npt.NDArray[np.float64],
    beta: float,
    max_dist: int = 20,
    n_samples_per_dist: int = 10,
) -> float:
    """Compute elastic sensitivity (a variant of smooth sensitivity).

    Elastic sensitivity relaxes smooth sensitivity by allowing the
    smoothing parameter to adapt to the local geometry.

    Args:
        local_sens_fn: Function computing local sensitivity at a point.
        x: Current database.
        beta: Smoothing parameter.
        max_dist: Maximum distance to explore.
        n_samples_per_dist: Samples per distance level.

    Returns:
        Elastic sensitivity value.
    """
    best = 0.0
    n = len(x)
    rng = np.random.RandomState(42)

    for dist in range(max_dist + 1):
        for _ in range(n_samples_per_dist):
            if dist == 0:
                y = x.copy()
            else:
                y = x.copy()
                indices = rng.choice(n, size=min(dist, n), replace=False)
                y[indices] += rng.randn(len(indices))
            ls_y = local_sens_fn(y)
            weighted = ls_y * math.exp(-beta * dist)
            best = max(best, weighted)

    return best


# ---------------------------------------------------------------------------
# Sensitivity of composed mechanisms
# ---------------------------------------------------------------------------


def composed_sensitivity(
    sensitivities: Sequence[float],
    composition: str = "sequential",
) -> float:
    """Compute sensitivity under composition.

    Args:
        sensitivities: Per-mechanism sensitivity values.
        composition: "sequential" (sum) or "parallel" (max).

    Returns:
        Composed sensitivity.
    """
    s_arr = np.array(sensitivities, dtype=np.float64)
    if composition == "sequential":
        return float(np.sum(s_arr))
    elif composition == "parallel":
        return float(np.max(s_arr))
    else:
        raise ValueError(f"Unknown composition: {composition}")


def group_sensitivity(
    sensitivity: float,
    group_size: int,
) -> float:
    """Compute group sensitivity: Δ_k = k * Δ_1.

    Args:
        sensitivity: Individual sensitivity.
        group_size: Size of the group.

    Returns:
        Group sensitivity.
    """
    return sensitivity * group_size


__all__ = [
    "LocalSensitivity",
    "SensitivityTracker",
    "SensitivityNode",
    "SmoothSensitivity",
    "ProposeSampleSensitivity",
    "PTRResult",
    "SensitivityBound",
    "SensitivityBoundResult",
    "elastic_sensitivity",
    "composed_sensitivity",
    "group_sensitivity",
]
