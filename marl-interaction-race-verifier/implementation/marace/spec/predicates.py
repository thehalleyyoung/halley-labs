"""
Joint predicates over multi-agent state.

Provides an abstract ``Predicate`` interface and a rich set of concrete
predicates (linear, distance, collision, region, etc.) together with
combinators (conjunction, disjunction, negation) and evaluation over
abstract domains (zonotopes).
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Abstract predicate
# ---------------------------------------------------------------------------

class Predicate(ABC):
    """Abstract predicate over a joint multi-agent state.

    A predicate maps a state (represented as a dictionary mapping agent
    identifiers to numpy arrays, or as a flat numpy vector) to a Boolean
    value.  Predicates also expose a *quantitative* semantics that returns
    a signed distance to the satisfaction boundary (positive ⇒ satisfied).

    Attributes:
        name: Human-readable label.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    def evaluate(self, state: Dict[str, np.ndarray]) -> bool:
        """Evaluate the predicate on a joint state.

        Args:
            state: Mapping from agent id to state vector.

        Returns:
            ``True`` iff the predicate is satisfied.
        """

    def robustness(self, state: Dict[str, np.ndarray]) -> float:
        """Quantitative (signed) distance to the satisfaction boundary.

        Positive means satisfied, negative means violated.
        The default implementation returns +1 / −1.
        """
        return 1.0 if self.evaluate(state) else -1.0

    def __and__(self, other: "Predicate") -> "ConjunctivePredicate":
        return ConjunctivePredicate([self, other])

    def __or__(self, other: "Predicate") -> "DisjunctivePredicate":
        return DisjunctivePredicate([self, other])

    def __invert__(self) -> "NegationPredicate":
        return NegationPredicate(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"


# ---------------------------------------------------------------------------
# Linear predicate: a^T x <= b
# ---------------------------------------------------------------------------

class LinearPredicate(Predicate):
    """Half-space predicate  a^T x ≤ b  for a state vector x.

    The state is obtained by concatenating agent state vectors in sorted
    agent-id order.

    Attributes:
        a: Normal vector.
        b: Offset scalar.
        agent_ids: If given, only these agents' states are used (in order).
    """

    def __init__(
        self,
        a: np.ndarray,
        b: float,
        agent_ids: Optional[List[str]] = None,
        name: str = "",
    ) -> None:
        super().__init__(name=name or "linear")
        self.a = np.asarray(a, dtype=np.float64)
        self.b = float(b)
        self.agent_ids = agent_ids

    def _flat_state(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        ids = self.agent_ids or sorted(state.keys())
        return np.concatenate([state[aid].ravel() for aid in ids])

    def evaluate(self, state: Dict[str, np.ndarray]) -> bool:
        x = self._flat_state(state)
        return bool(np.dot(self.a, x) <= self.b)

    def robustness(self, state: Dict[str, np.ndarray]) -> float:
        x = self._flat_state(state)
        return float(self.b - np.dot(self.a, x))


# ---------------------------------------------------------------------------
# Combinators
# ---------------------------------------------------------------------------

class ConjunctivePredicate(Predicate):
    """Conjunction (AND) of predicates."""

    def __init__(self, predicates: List[Predicate], name: str = "") -> None:
        super().__init__(name=name or "and")
        self.predicates = list(predicates)

    def evaluate(self, state: Dict[str, np.ndarray]) -> bool:
        return all(p.evaluate(state) for p in self.predicates)

    def robustness(self, state: Dict[str, np.ndarray]) -> float:
        return min(p.robustness(state) for p in self.predicates)

    def __and__(self, other: Predicate) -> "ConjunctivePredicate":
        return ConjunctivePredicate(self.predicates + [other])


class DisjunctivePredicate(Predicate):
    """Disjunction (OR) of predicates."""

    def __init__(self, predicates: List[Predicate], name: str = "") -> None:
        super().__init__(name=name or "or")
        self.predicates = list(predicates)

    def evaluate(self, state: Dict[str, np.ndarray]) -> bool:
        return any(p.evaluate(state) for p in self.predicates)

    def robustness(self, state: Dict[str, np.ndarray]) -> float:
        return max(p.robustness(state) for p in self.predicates)

    def __or__(self, other: Predicate) -> "DisjunctivePredicate":
        return DisjunctivePredicate(self.predicates + [other])


class NegationPredicate(Predicate):
    """Negation (NOT) of a predicate."""

    def __init__(self, predicate: Predicate, name: str = "") -> None:
        super().__init__(name=name or f"not({predicate.name})")
        self.predicate = predicate

    def evaluate(self, state: Dict[str, np.ndarray]) -> bool:
        return not self.predicate.evaluate(state)

    def robustness(self, state: Dict[str, np.ndarray]) -> float:
        return -self.predicate.robustness(state)


# ---------------------------------------------------------------------------
# Distance predicate
# ---------------------------------------------------------------------------

class DistancePredicate(Predicate):
    """Check  ‖x_i − x_j‖ ≤ d  (or ≥ d) for agents *i*, *j*.

    When ``greater=True`` the semantics become ‖x_i − x_j‖ ≥ d
    (i.e. minimum-separation).

    Attributes:
        agent_i: First agent id.
        agent_j: Second agent id.
        threshold: Distance threshold *d*.
        pos_indices: Indices of position components in the state vector.
        greater: If ``True``, check ≥ instead of ≤.
    """

    def __init__(
        self,
        agent_i: str,
        agent_j: str,
        threshold: float,
        pos_indices: Sequence[int] = (0, 1),
        greater: bool = False,
        name: str = "",
    ) -> None:
        label = name or f"dist({agent_i},{agent_j})"
        super().__init__(name=label)
        self.agent_i = agent_i
        self.agent_j = agent_j
        self.threshold = threshold
        self.pos_indices = list(pos_indices)
        self.greater = greater

    def _distance(self, state: Dict[str, np.ndarray]) -> float:
        xi = state[self.agent_i][self.pos_indices]
        xj = state[self.agent_j][self.pos_indices]
        return float(np.linalg.norm(xi - xj))

    def evaluate(self, state: Dict[str, np.ndarray]) -> bool:
        d = self._distance(state)
        return d >= self.threshold if self.greater else d <= self.threshold

    def robustness(self, state: Dict[str, np.ndarray]) -> float:
        d = self._distance(state)
        if self.greater:
            return d - self.threshold
        return self.threshold - d


# ---------------------------------------------------------------------------
# Collision predicate
# ---------------------------------------------------------------------------

class CollisionPredicate(Predicate):
    """Check overlap of axis-aligned bounding boxes.

    Each agent's bounding box is centred at its position (first two state
    components) with half-widths given by ``extents``.

    Attributes:
        agent_i: First agent id.
        agent_j: Second agent id.
        extents_i: ``(half_length, half_width)`` of agent i.
        extents_j: ``(half_length, half_width)`` of agent j.
    """

    def __init__(
        self,
        agent_i: str,
        agent_j: str,
        extents_i: Tuple[float, float] = (2.25, 1.0),
        extents_j: Tuple[float, float] = (2.25, 1.0),
        name: str = "",
    ) -> None:
        super().__init__(name=name or f"collision({agent_i},{agent_j})")
        self.agent_i = agent_i
        self.agent_j = agent_j
        self.extents_i = extents_i
        self.extents_j = extents_j

    def evaluate(self, state: Dict[str, np.ndarray]) -> bool:
        pi = state[self.agent_i][:2]
        pj = state[self.agent_j][:2]
        dx = abs(pi[0] - pj[0])
        dy = abs(pi[1] - pj[1])
        overlap_x = dx < (self.extents_i[0] + self.extents_j[0])
        overlap_y = dy < (self.extents_i[1] + self.extents_j[1])
        return bool(overlap_x and overlap_y)

    def robustness(self, state: Dict[str, np.ndarray]) -> float:
        pi = state[self.agent_i][:2]
        pj = state[self.agent_j][:2]
        dx = abs(pi[0] - pj[0])
        dy = abs(pi[1] - pj[1])
        margin_x = dx - (self.extents_i[0] + self.extents_j[0])
        margin_y = dy - (self.extents_i[1] + self.extents_j[1])
        # Negative robustness = collision
        return -min(margin_x, margin_y)


# ---------------------------------------------------------------------------
# Region predicate
# ---------------------------------------------------------------------------

class RegionPredicate(Predicate):
    """Check whether agent *i* is inside an axis-aligned rectangular region.

    Attributes:
        agent_id: Agent to check.
        low: Lower corner ``(x_min, y_min)``.
        high: Upper corner ``(x_max, y_max)``.
        pos_indices: Indices of position in state vector.
    """

    def __init__(
        self,
        agent_id: str,
        low: Sequence[float],
        high: Sequence[float],
        pos_indices: Sequence[int] = (0, 1),
        name: str = "",
    ) -> None:
        super().__init__(name=name or f"region({agent_id})")
        self.agent_id = agent_id
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.pos_indices = list(pos_indices)

    def evaluate(self, state: Dict[str, np.ndarray]) -> bool:
        pos = state[self.agent_id][self.pos_indices]
        return bool(np.all(pos >= self.low) and np.all(pos <= self.high))

    def robustness(self, state: Dict[str, np.ndarray]) -> float:
        pos = state[self.agent_id][self.pos_indices]
        margins = np.minimum(pos - self.low, self.high - pos)
        return float(np.min(margins))


# ---------------------------------------------------------------------------
# Relative velocity predicate
# ---------------------------------------------------------------------------

class RelativeVelocityPredicate(Predicate):
    """Check  ‖v_i − v_j‖ ≤ threshold  for agents *i*, *j*.

    Attributes:
        agent_i: First agent.
        agent_j: Second agent.
        threshold: Relative speed threshold.
        vel_indices: Indices of velocity components in state vector.
    """

    def __init__(
        self,
        agent_i: str,
        agent_j: str,
        threshold: float,
        vel_indices: Sequence[int] = (2, 3),
        name: str = "",
    ) -> None:
        super().__init__(name=name or f"relvel({agent_i},{agent_j})")
        self.agent_i = agent_i
        self.agent_j = agent_j
        self.threshold = threshold
        self.vel_indices = list(vel_indices)

    def evaluate(self, state: Dict[str, np.ndarray]) -> bool:
        vi = state[self.agent_i][self.vel_indices]
        vj = state[self.agent_j][self.vel_indices]
        return bool(np.linalg.norm(vi - vj) <= self.threshold)

    def robustness(self, state: Dict[str, np.ndarray]) -> float:
        vi = state[self.agent_i][self.vel_indices]
        vj = state[self.agent_j][self.vel_indices]
        return self.threshold - float(np.linalg.norm(vi - vj))


# ---------------------------------------------------------------------------
# Custom predicate
# ---------------------------------------------------------------------------

class CustomPredicate(Predicate):
    """User-defined predicate backed by an arbitrary callable.

    The callable must accept ``state: Dict[str, np.ndarray]`` and return
    a ``bool`` (or a ``float`` for quantitative semantics).
    """

    def __init__(
        self,
        fn: Callable[[Dict[str, np.ndarray]], Union[bool, float]],
        name: str = "custom",
    ) -> None:
        super().__init__(name=name)
        self._fn = fn

    def evaluate(self, state: Dict[str, np.ndarray]) -> bool:
        result = self._fn(state)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
        return float(result) >= 0.0

    def robustness(self, state: Dict[str, np.ndarray]) -> float:
        result = self._fn(state)
        if isinstance(result, (bool, np.bool_)):
            return 1.0 if result else -1.0
        return float(result)


# ---------------------------------------------------------------------------
# Predicate evaluator (abstract domains)
# ---------------------------------------------------------------------------

@dataclass
class Zonotope:
    """Axis-aligned zonotope represented as center + generator matrix.

    Attributes:
        center: Centre point (n,).
        generators: Generator matrix (n × m).
    """
    center: np.ndarray
    generators: np.ndarray

    @property
    def dim(self) -> int:
        return len(self.center)

    @property
    def num_generators(self) -> int:
        return self.generators.shape[1] if self.generators.ndim == 2 else 0

    def interval_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute axis-aligned bounding box."""
        delta = np.sum(np.abs(self.generators), axis=1)
        return self.center - delta, self.center + delta

    @staticmethod
    def from_interval(low: np.ndarray, high: np.ndarray) -> "Zonotope":
        center = (low + high) / 2
        radius = (high - low) / 2
        generators = np.diag(radius)
        return Zonotope(center=center, generators=generators)


class PredicateEvaluator:
    """Evaluate predicates over abstract domains (zonotopes).

    For each predicate type the evaluator returns a tri-valued result:
    ``True`` (definitely satisfied), ``False`` (definitely violated),
    or ``None`` (inconclusive).
    """

    def evaluate_linear(
        self,
        pred: LinearPredicate,
        zonotope: Zonotope,
        agent_ids: Optional[List[str]] = None,
    ) -> Optional[bool]:
        """Evaluate a linear predicate over a zonotope.

        Uses interval arithmetic on the zonotope bounding box.
        """
        low, high = zonotope.interval_bounds()
        # Upper bound of a^T x
        a = pred.a
        upper = float(np.sum(np.where(a >= 0, a * high, a * low)))
        lower = float(np.sum(np.where(a >= 0, a * low, a * high)))
        if upper <= pred.b:
            return True
        if lower > pred.b:
            return False
        return None

    def evaluate_distance(
        self,
        pred: DistancePredicate,
        zonotope_i: Zonotope,
        zonotope_j: Zonotope,
    ) -> Optional[bool]:
        """Evaluate a distance predicate over two zonotopes."""
        low_i, high_i = zonotope_i.interval_bounds()
        low_j, high_j = zonotope_j.interval_bounds()
        idx = pred.pos_indices
        # Max possible distance
        diff_max = np.maximum(np.abs(high_i[idx] - low_j[idx]),
                              np.abs(high_j[idx] - low_i[idx]))
        max_dist = float(np.linalg.norm(diff_max))
        # Min possible distance
        diff_min = np.maximum(0.0, np.maximum(low_i[idx] - high_j[idx],
                                              low_j[idx] - high_i[idx]))
        min_dist = float(np.linalg.norm(diff_min))

        if pred.greater:
            if min_dist >= pred.threshold:
                return True
            if max_dist < pred.threshold:
                return False
        else:
            if max_dist <= pred.threshold:
                return True
            if min_dist > pred.threshold:
                return False
        return None


# ---------------------------------------------------------------------------
# Predicate library
# ---------------------------------------------------------------------------

class PredicateLibrary:
    """Standard safety predicates for driving, warehouse, etc."""

    @staticmethod
    def pairwise_min_distance(
        agent_ids: List[str],
        threshold: float,
        pos_indices: Sequence[int] = (0, 1),
    ) -> ConjunctivePredicate:
        """All pairs maintain minimum distance."""
        preds: List[Predicate] = []
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                preds.append(DistancePredicate(
                    agent_ids[i], agent_ids[j],
                    threshold=threshold,
                    pos_indices=pos_indices,
                    greater=True,
                    name=f"min_dist({agent_ids[i]},{agent_ids[j]})",
                ))
        return ConjunctivePredicate(preds, name="pairwise_min_distance")

    @staticmethod
    def no_collision(
        agent_ids: List[str],
        extents: Tuple[float, float] = (2.25, 1.0),
    ) -> ConjunctivePredicate:
        """No pair of agents collides (AABB overlap)."""
        preds: List[Predicate] = []
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                preds.append(NegationPredicate(
                    CollisionPredicate(agent_ids[i], agent_ids[j],
                                       extents_i=extents, extents_j=extents),
                    name=f"no_collision({agent_ids[i]},{agent_ids[j]})",
                ))
        return ConjunctivePredicate(preds, name="collision_freedom")

    @staticmethod
    def all_in_region(
        agent_ids: List[str],
        low: Sequence[float],
        high: Sequence[float],
    ) -> ConjunctivePredicate:
        """All agents are inside a rectangular region."""
        preds = [
            RegionPredicate(aid, low=low, high=high)
            for aid in agent_ids
        ]
        return ConjunctivePredicate(preds, name="all_in_region")

    @staticmethod
    def max_relative_speed(
        agent_ids: List[str],
        threshold: float,
        vel_indices: Sequence[int] = (2, 3),
    ) -> ConjunctivePredicate:
        """All pairs have relative speed ≤ threshold."""
        preds: List[Predicate] = []
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                preds.append(RelativeVelocityPredicate(
                    agent_ids[i], agent_ids[j],
                    threshold=threshold,
                    vel_indices=vel_indices,
                ))
        return ConjunctivePredicate(preds, name="max_relative_speed")
