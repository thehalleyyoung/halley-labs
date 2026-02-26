"""
Happens-before consistency constraints as linear predicates for MARACE.

Given a happens-before (HB) partial order on events in a multi-agent system,
this module derives linear constraints on the joint abstract state space.
These constraints are used to prune infeasible regions of the zonotope domain,
improving precision without sacrificing soundness.

Key idea: if event e₁ happens-before e₂, then the state reachable after e₂
must be consistent with e₁ having already occurred. This induces linear
inequalities on the joint state vector that can be intersected with the
zonotope abstract element.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
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
)

import numpy as np
from scipy.optimize import linprog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constraint types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HBConstraint:
    """A single linear constraint a^T x <= b derived from an HB edge.

    Attributes
    ----------
    normal : np.ndarray
        Coefficient vector a of shape (n,).
    bound : float
        Upper-bound scalar b.
    source_event : str
        Identifier of the source event in the HB edge.
    target_event : str
        Identifier of the target event in the HB edge.
    label : str
        Human-readable description.
    """

    normal: np.ndarray
    bound: float
    source_event: str = ""
    target_event: str = ""
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "normal",
                           np.asarray(self.normal, dtype=np.float64).ravel())

    # --- evaluation ---

    def satisfied_by(self, x: np.ndarray) -> bool:
        """Check if point *x* satisfies a^T x <= b."""
        return float(self.normal @ np.asarray(x, dtype=np.float64).ravel()) <= self.bound + 1e-12

    def margin(self, x: np.ndarray) -> float:
        """Return b - a^T x (positive means feasible)."""
        return self.bound - float(self.normal @ np.asarray(x, dtype=np.float64).ravel())

    # --- serialization ---

    def to_dict(self) -> Dict[str, Any]:
        return {
            "normal": self.normal.tolist(),
            "bound": self.bound,
            "source_event": self.source_event,
            "target_event": self.target_event,
            "label": self.label,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "HBConstraint":
        return HBConstraint(
            normal=np.array(d["normal"]),
            bound=d["bound"],
            source_event=d.get("source_event", ""),
            target_event=d.get("target_event", ""),
            label=d.get("label", ""),
        )

    def __repr__(self) -> str:
        return f"HBConstraint({self.label!r}, bound={self.bound:.4g})"


# ---------------------------------------------------------------------------
# Timing constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingConstraint:
    """Constraint on relative timing of events: t_i - t_j <= delta.

    In the joint state vector, timing dimensions are indexed by
    *dim_i* and *dim_j*. The constraint is  x[dim_i] - x[dim_j] <= delta.
    """

    dim_i: int
    dim_j: int
    delta: float
    source_event: str = ""
    target_event: str = ""

    def to_hb_constraint(self, state_dim: int) -> HBConstraint:
        """Convert to a general HBConstraint in ℝ^state_dim."""
        a = np.zeros(state_dim, dtype=np.float64)
        a[self.dim_i] = 1.0
        a[self.dim_j] = -1.0
        return HBConstraint(
            normal=a,
            bound=self.delta,
            source_event=self.source_event,
            target_event=self.target_event,
            label=f"t[{self.dim_i}] - t[{self.dim_j}] <= {self.delta}",
        )


# ---------------------------------------------------------------------------
# Ordering constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrderingConstraint:
    """Constraint on state ordering from causal dependencies.

    If agent i's state dimension *dim_a* must be no larger than agent j's
    state dimension *dim_b* plus an offset, due to a causal chain:
    x[dim_a] - x[dim_b] <= offset.
    """

    dim_a: int
    dim_b: int
    offset: float
    causal_chain: Tuple[str, ...] = ()

    def to_hb_constraint(self, state_dim: int) -> HBConstraint:
        a = np.zeros(state_dim, dtype=np.float64)
        a[self.dim_a] = 1.0
        a[self.dim_b] = -1.0
        chain_str = " -> ".join(self.causal_chain) if self.causal_chain else ""
        return HBConstraint(
            normal=a,
            bound=self.offset,
            source_event=self.causal_chain[0] if self.causal_chain else "",
            target_event=self.causal_chain[-1] if self.causal_chain else "",
            label=f"order({self.dim_a},{self.dim_b})<={self.offset} [{chain_str}]",
        )


# ---------------------------------------------------------------------------
# HBConstraintSet
# ---------------------------------------------------------------------------


class HBConstraintSet:
    """Collection of HB-derived linear constraints.

    Provides efficient batch operations for applying all constraints to
    an abstract element, checking satisfiability, and managing the
    constraint set as the analysis progresses.
    """

    def __init__(self, constraints: Optional[Sequence[HBConstraint]] = None) -> None:
        self._constraints: List[HBConstraint] = list(constraints or [])

    # --- mutation ---

    def add(self, constraint: HBConstraint) -> None:
        self._constraints.append(constraint)

    def add_timing(self, tc: TimingConstraint, state_dim: int) -> None:
        self._constraints.append(tc.to_hb_constraint(state_dim))

    def add_ordering(self, oc: OrderingConstraint, state_dim: int) -> None:
        self._constraints.append(oc.to_hb_constraint(state_dim))

    def remove_redundant(self, tol: float = 1e-8) -> int:
        """Remove constraints that are implied by others (pairwise check).

        Returns the number of constraints removed.
        """
        if len(self._constraints) <= 1:
            return 0

        keep: List[HBConstraint] = []
        removed = 0
        for i, ci in enumerate(self._constraints):
            redundant = False
            for j, cj in enumerate(self._constraints):
                if i == j:
                    continue
                # ci is redundant if cj.normal == ci.normal and cj.bound <= ci.bound
                if (np.allclose(ci.normal, cj.normal, atol=tol)
                        and cj.bound <= ci.bound - tol):
                    redundant = True
                    break
            if not redundant:
                keep.append(ci)
            else:
                removed += 1
        self._constraints = keep
        return removed

    # --- query ---

    @property
    def constraints(self) -> List[HBConstraint]:
        return list(self._constraints)

    def __len__(self) -> int:
        return len(self._constraints)

    def __iter__(self):
        return iter(self._constraints)

    def as_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (A, b) where each row of A is a constraint normal
        and b is the bound vector, so A x <= b encodes all constraints."""
        if not self._constraints:
            return np.zeros((0, 0)), np.zeros(0)
        A = np.vstack([c.normal for c in self._constraints])
        b = np.array([c.bound for c in self._constraints])
        return A, b

    def satisfied_by_point(self, x: np.ndarray) -> bool:
        """Check if all constraints are satisfied by point x."""
        x = np.asarray(x, dtype=np.float64).ravel()
        return all(c.satisfied_by(x) for c in self._constraints)

    def violated_constraints(self, x: np.ndarray) -> List[HBConstraint]:
        """Return list of constraints violated by point x."""
        x = np.asarray(x, dtype=np.float64).ravel()
        return [c for c in self._constraints if not c.satisfied_by(x)]

    def max_violation(self, x: np.ndarray) -> float:
        """Maximum violation across all constraints (0 if all satisfied)."""
        x = np.asarray(x, dtype=np.float64).ravel()
        if not self._constraints:
            return 0.0
        return max(0.0, max(-c.margin(x) for c in self._constraints))

    # --- serialization ---

    def to_dict(self) -> Dict[str, Any]:
        return {"constraints": [c.to_dict() for c in self._constraints]}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "HBConstraintSet":
        return HBConstraintSet(
            constraints=[HBConstraint.from_dict(cd) for cd in d["constraints"]]
        )

    def __repr__(self) -> str:
        return f"HBConstraintSet(n={len(self._constraints)})"


# ---------------------------------------------------------------------------
# Constraint Generator
# ---------------------------------------------------------------------------


@dataclass
class HBEdge:
    """An edge in the happens-before graph."""
    source: str
    target: str
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConstraintGenerator:
    """Generate linear constraints from an HB graph and state semantics.

    Given:
    - An HB graph (set of HBEdge)
    - A mapping from events to state dimensions
    - Semantic bounds (e.g., max state change per step)

    Produces an HBConstraintSet encoding all derivable linear constraints.
    """

    def __init__(
        self,
        state_dim: int,
        event_to_dims: Optional[Dict[str, List[int]]] = None,
        max_step_change: Optional[Dict[int, float]] = None,
    ) -> None:
        self.state_dim = state_dim
        self.event_to_dims: Dict[str, List[int]] = event_to_dims or {}
        self.max_step_change: Dict[int, float] = max_step_change or {}

    def generate(self, hb_edges: Sequence[HBEdge]) -> HBConstraintSet:
        """Generate constraints from a set of HB edges.

        For each edge (e₁ → e₂), we know e₁ completes before e₂ starts.
        This can produce:
        1. Timing constraints: t(e₂) - t(e₁) >= 0
        2. Ordering constraints: state written by e₁ must be visible to e₂
        3. Monotonicity constraints: derived from transitive closure
        """
        cs = HBConstraintSet()

        # Direct edge constraints
        for edge in hb_edges:
            new_constraints = self._constraints_from_edge(edge)
            for c in new_constraints:
                cs.add(c)

        # Transitive closure constraints
        tc_edges = self._transitive_closure(hb_edges)
        for edge in tc_edges:
            new_constraints = self._constraints_from_edge(edge)
            for c in new_constraints:
                cs.add(c)

        cs.remove_redundant()
        return cs

    def _constraints_from_edge(self, edge: HBEdge) -> List[HBConstraint]:
        """Derive constraints from a single HB edge."""
        constraints: List[HBConstraint] = []

        src_dims = self.event_to_dims.get(edge.source, [])
        tgt_dims = self.event_to_dims.get(edge.target, [])

        # For each pair of source/target dimensions, if the source writes
        # to dim_s and target reads dim_t, the ordering implies constraints.
        for ds in src_dims:
            for dt in tgt_dims:
                if ds == dt:
                    continue
                max_change = self.max_step_change.get(ds, float("inf"))
                if np.isfinite(max_change):
                    a = np.zeros(self.state_dim)
                    a[dt] = 1.0
                    a[ds] = -1.0
                    constraints.append(HBConstraint(
                        normal=a,
                        bound=max_change,
                        source_event=edge.source,
                        target_event=edge.target,
                        label=f"hb({edge.source}->{edge.target}): "
                              f"x[{dt}]-x[{ds}]<={max_change}",
                    ))

        return constraints

    def _transitive_closure(self, edges: Sequence[HBEdge]) -> List[HBEdge]:
        """Compute transitive closure of HB edges (Floyd-Warshall style)."""
        events: Set[str] = set()
        for e in edges:
            events.add(e.source)
            events.add(e.target)
        event_list = sorted(events)
        idx = {ev: i for i, ev in enumerate(event_list)}
        n = len(event_list)
        if n == 0:
            return []

        reach = np.zeros((n, n), dtype=bool)
        for e in edges:
            reach[idx[e.source], idx[e.target]] = True

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if reach[i, k] and reach[k, j]:
                        reach[i, j] = True

        tc_edges: List[HBEdge] = []
        existing = {(e.source, e.target) for e in edges}
        for i in range(n):
            for j in range(n):
                if reach[i, j] and (event_list[i], event_list[j]) not in existing:
                    tc_edges.append(HBEdge(
                        source=event_list[i],
                        target=event_list[j],
                        label="transitive",
                    ))
        return tc_edges


# ---------------------------------------------------------------------------
# Constraint Propagation
# ---------------------------------------------------------------------------


class ConstraintPropagation:
    """Propagate HB constraints through abstract transfer functions.

    When an abstract element is transformed (e.g., through a network layer),
    the HB constraints on the *input* space must be mapped to constraints
    on the *output* space. For affine transforms y = Wx + b, the constraint
    a^T x <= c becomes  a^T W^{-1} (y - b) <= c  (if W is invertible), or
    more generally, weakened via support-function computation.
    """

    @staticmethod
    def propagate_affine(
        constraints: HBConstraintSet,
        W: np.ndarray,
        b: Optional[np.ndarray] = None,
    ) -> HBConstraintSet:
        """Propagate constraints through affine map y = Wx + b.

        If W is invertible, we get exact propagation. Otherwise, we use a
        sound over-approximation: for each constraint a^T x <= c, we compute
        the weakest valid constraint on y.
        """
        W = np.atleast_2d(np.asarray(W, dtype=np.float64))
        m, n = W.shape

        new_cs = HBConstraintSet()

        # Try to compute W^{-1} for exact propagation
        if m == n:
            try:
                W_inv = np.linalg.inv(W)
                cond = np.linalg.cond(W)
                if cond < 1e10:
                    for c in constraints:
                        new_a = W_inv.T @ c.normal
                        new_b = c.bound
                        if b is not None:
                            new_b += float(new_a @ np.asarray(b, dtype=np.float64))
                        new_cs.add(HBConstraint(
                            normal=new_a,
                            bound=new_b,
                            source_event=c.source_event,
                            target_event=c.target_event,
                            label=f"prop({c.label})",
                        ))
                    return new_cs
            except np.linalg.LinAlgError:
                pass

        # Non-invertible case: use pseudo-inverse with soundness relaxation
        W_pinv = np.linalg.pinv(W)
        residual_norm = np.linalg.norm(W @ W_pinv - np.eye(m))

        for c in constraints:
            new_a = W_pinv.T @ c.normal
            relaxation = residual_norm * np.linalg.norm(c.normal)
            new_b = c.bound + relaxation
            if b is not None:
                new_b += float(new_a @ np.asarray(b, dtype=np.float64))
            new_cs.add(HBConstraint(
                normal=new_a,
                bound=new_b,
                source_event=c.source_event,
                target_event=c.target_event,
                label=f"prop_relaxed({c.label})",
            ))

        return new_cs

    @staticmethod
    def propagate_relu(
        constraints: HBConstraintSet,
        active_mask: np.ndarray,
    ) -> HBConstraintSet:
        """Propagate constraints through ReLU given known active neurons.

        For neurons known to be active (mask=1), the constraint passes
        through. For neurons known to be inactive (mask=0), the corresponding
        dimension is zeroed. For uncertain neurons (mask=0.5), the constraint
        is dropped (soundness requires this).
        """
        active_mask = np.asarray(active_mask, dtype=np.float64).ravel()
        new_cs = HBConstraintSet()

        for c in constraints:
            a = c.normal.copy()
            can_propagate = True
            for i in range(len(a)):
                if i < len(active_mask):
                    if active_mask[i] < 0.25:
                        a[i] = 0.0
                    elif active_mask[i] < 0.75:
                        # Uncertain neuron — constraint may not hold
                        if abs(a[i]) > 1e-12:
                            can_propagate = False
                            break

            if can_propagate and np.linalg.norm(a) > 1e-12:
                new_cs.add(HBConstraint(
                    normal=a,
                    bound=c.bound,
                    source_event=c.source_event,
                    target_event=c.target_event,
                    label=f"relu_prop({c.label})",
                ))

        return new_cs


# ---------------------------------------------------------------------------
# Consistency Checker
# ---------------------------------------------------------------------------


class ConsistencyChecker:
    """Check if a zonotope satisfies all HB constraints.

    Uses LP to determine if the zonotope (or any part of it) can violate
    the constraint system. If the maximum of a^T x over the zonotope
    exceeds b for any constraint, the zonotope is not fully consistent.
    """

    @staticmethod
    def check_all(
        zonotope: Any,  # Zonotope (avoid circular import)
        constraints: HBConstraintSet,
    ) -> Tuple[bool, List[HBConstraint]]:
        """Check all constraints against a zonotope.

        Returns
        -------
        (is_consistent, violated)
            is_consistent: True if all constraints are satisfied for all
            points in the zonotope.
            violated: list of constraints that can be violated.
        """
        violated: List[HBConstraint] = []
        for c in constraints:
            max_val = ConsistencyChecker._maximize_over_zonotope(
                zonotope, c.normal
            )
            if max_val > c.bound + 1e-10:
                violated.append(c)

        return len(violated) == 0, violated

    @staticmethod
    def _maximize_over_zonotope(zonotope: Any, direction: np.ndarray) -> float:
        """Maximize direction^T x over the zonotope.

        For a zonotope Z = {c + G ε : ε ∈ [-1,1]^p},
        max d^T x = d^T c + Σ_i |d^T g_i|.
        """
        d = np.asarray(direction, dtype=np.float64).ravel()
        val = float(d @ zonotope.center)
        val += float(np.sum(np.abs(zonotope.generators.T @ d)))
        return val

    @staticmethod
    def max_violation(
        zonotope: Any,
        constraints: HBConstraintSet,
    ) -> float:
        """Return the maximum constraint violation over the zonotope."""
        max_viol = 0.0
        for c in constraints:
            max_val = ConsistencyChecker._maximize_over_zonotope(
                zonotope, c.normal
            )
            viol = max_val - c.bound
            max_viol = max(max_viol, viol)
        return max(0.0, max_viol)

    @staticmethod
    def feasible_fraction_estimate(
        zonotope: Any,
        constraints: HBConstraintSet,
        num_samples: int = 1000,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Estimate the fraction of the zonotope that satisfies all constraints."""
        if rng is None:
            rng = np.random.default_rng(42)
        points = zonotope.sample(num_samples, rng=rng)
        count = sum(1 for pt in points if constraints.satisfied_by_point(pt))
        return count / num_samples


# ---------------------------------------------------------------------------
# Constraint Strengthening
# ---------------------------------------------------------------------------


class ConstraintStrengthening:
    """Adaptively strengthen constraints based on analysis results.

    During fixpoint iteration, if the analysis discovers that certain
    state regions are unreachable, we can tighten the HB constraints
    accordingly to improve precision in subsequent iterations.
    """

    def __init__(self, base_constraints: HBConstraintSet) -> None:
        self._base = base_constraints
        self._strengthened = HBConstraintSet(base_constraints.constraints)
        self._history: List[Tuple[int, float]] = []

    @property
    def current(self) -> HBConstraintSet:
        """Currently strengthened constraint set."""
        return self._strengthened

    def strengthen_from_zonotope(
        self,
        zonotope: Any,
        iteration: int,
        factor: float = 0.9,
    ) -> HBConstraintSet:
        """Strengthen constraints using the current zonotope iterate.

        For each constraint a^T x <= b, compute the actual maximum of
        a^T x over the zonotope. If max < b, we can soundly tighten
        the bound to  factor * b + (1 - factor) * max.
        """
        new_constraints: List[HBConstraint] = []

        for c in self._base:
            max_val = ConsistencyChecker._maximize_over_zonotope(zonotope, c.normal)
            if max_val < c.bound - 1e-10:
                tightened_bound = factor * c.bound + (1.0 - factor) * max_val
                tightened_bound = max(tightened_bound, max_val)
                new_constraints.append(HBConstraint(
                    normal=c.normal.copy(),
                    bound=tightened_bound,
                    source_event=c.source_event,
                    target_event=c.target_event,
                    label=f"strengthened_iter{iteration}({c.label})",
                ))
            else:
                new_constraints.append(c)

        self._strengthened = HBConstraintSet(new_constraints)
        total_tightening = sum(
            old.bound - new.bound
            for old, new in zip(self._base, new_constraints)
        )
        self._history.append((iteration, total_tightening))
        logger.debug(
            "Constraint strengthening at iteration %d: total tightening = %.6g",
            iteration, total_tightening,
        )
        return self._strengthened

    def reset(self) -> None:
        """Reset to base constraints."""
        self._strengthened = HBConstraintSet(self._base.constraints)
        self._history.clear()

    @property
    def strengthening_history(self) -> List[Tuple[int, float]]:
        """History of (iteration, total_tightening) pairs."""
        return list(self._history)
