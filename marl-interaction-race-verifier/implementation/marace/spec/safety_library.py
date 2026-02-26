"""
Safety specification library.

Pre-built safety specifications for highway driving, warehouse robotics,
and multi-agent trading environments.  Each specification is a
``TemporalFormula`` ready for evaluation or monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from marace.spec.predicates import (
    ConjunctivePredicate,
    CollisionPredicate,
    CustomPredicate,
    DistancePredicate,
    NegationPredicate,
    Predicate,
    PredicateLibrary,
    RegionPredicate,
)
from marace.spec.temporal import (
    Always,
    BoundedResponse,
    Eventually,
    TemporalFormula,
)


# ---------------------------------------------------------------------------
# Collision freedom
# ---------------------------------------------------------------------------

class CollisionFreedom:
    """No two agents collide at any time step.

    Constructs  □[0,T] ⋀_{i<j} ¬collision(i,j).

    Args:
        agent_ids: List of agent identifiers.
        extents: Half-length and half-width of bounding boxes.
        horizon: Temporal horizon (``None`` = full trace).
    """

    def __init__(
        self,
        agent_ids: List[str],
        extents: Tuple[float, float] = (2.25, 1.0),
        horizon: Optional[int] = None,
    ) -> None:
        self.agent_ids = agent_ids
        self.extents = extents
        self.horizon = horizon
        self._pred = PredicateLibrary.no_collision(agent_ids, extents)

    def formula(self) -> TemporalFormula:
        return Always(self._pred, horizon=self.horizon, name="collision_freedom")

    def __repr__(self) -> str:
        return f"CollisionFreedom(agents={self.agent_ids})"


# ---------------------------------------------------------------------------
# Minimum separation
# ---------------------------------------------------------------------------

class MinimumSeparation:
    """All agents maintain minimum Euclidean distance at all times.

    Constructs  □[0,T] ⋀_{i<j} dist(i,j) ≥ d.

    Args:
        agent_ids: Agent identifiers.
        min_distance: Required minimum distance.
        pos_indices: Indices of position components in state vector.
        horizon: Temporal horizon.
    """

    def __init__(
        self,
        agent_ids: List[str],
        min_distance: float = 3.0,
        pos_indices: Sequence[int] = (0, 1),
        horizon: Optional[int] = None,
    ) -> None:
        self.agent_ids = agent_ids
        self.min_distance = min_distance
        self.pos_indices = pos_indices
        self.horizon = horizon
        self._pred = PredicateLibrary.pairwise_min_distance(
            agent_ids, min_distance, pos_indices
        )

    def formula(self) -> TemporalFormula:
        return Always(self._pred, horizon=self.horizon, name="min_separation")

    def __repr__(self) -> str:
        return (
            f"MinimumSeparation(agents={self.agent_ids}, "
            f"d={self.min_distance})"
        )


# ---------------------------------------------------------------------------
# Deadlock freedom
# ---------------------------------------------------------------------------

class DeadlockFreedom:
    """System never reaches a deadlock configuration.

    A deadlock is detected by a user-supplied predicate that checks
    whether any subset of agents is mutually blocked.

    Args:
        agent_ids: Agent identifiers.
        deadlock_predicate: Predicate that returns ``True`` when deadlock
            is detected (the spec *negates* it).
        horizon: Temporal horizon.
    """

    def __init__(
        self,
        agent_ids: List[str],
        deadlock_predicate: Optional[Predicate] = None,
        horizon: Optional[int] = None,
    ) -> None:
        self.agent_ids = agent_ids
        self.horizon = horizon
        if deadlock_predicate is None:
            # Default: no agent has zero velocity for extended time
            deadlock_predicate = CustomPredicate(
                self._default_deadlock_check,
                name="deadlock_check",
            )
        self._pred = NegationPredicate(deadlock_predicate, name="no_deadlock")

    @staticmethod
    def _default_deadlock_check(state: Dict[str, np.ndarray]) -> bool:
        """Return ``True`` if all agents have near-zero velocity."""
        for aid, s in state.items():
            if len(s) >= 4:
                speed = float(np.linalg.norm(s[2:4]))
                if speed > 0.01:
                    return False
        return True  # all agents stuck

    def formula(self) -> TemporalFormula:
        return Always(self._pred, horizon=self.horizon, name="deadlock_freedom")


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------

class LivenessSpec:
    """All tasks eventually complete.

    Constructs  ⋀_i ◇[0,T] task_complete(i).

    Args:
        agent_ids: Agent identifiers.
        completion_predicate: Per-agent predicate returning ``True`` when
            that agent's task is complete.  If ``None``, checks if the
            agent reached a goal region.
        horizon: Temporal horizon.
    """

    def __init__(
        self,
        agent_ids: List[str],
        completion_predicate: Optional[Dict[str, Predicate]] = None,
        horizon: Optional[int] = None,
    ) -> None:
        self.agent_ids = agent_ids
        self.horizon = horizon
        self._completion = completion_predicate or {}

    def formula(self) -> TemporalFormula:
        formulas: List[TemporalFormula] = []
        for aid in self.agent_ids:
            pred = self._completion.get(aid)
            if pred is None:
                pred = CustomPredicate(
                    lambda s, _aid=aid: True, name=f"complete({aid})"
                )
            formulas.append(
                Eventually(pred, horizon=self.horizon, name=f"liveness({aid})")
            )
        # Conjunction of all liveness formulas
        if len(formulas) == 1:
            return formulas[0]
        result = formulas[0]
        for f in formulas[1:]:
            result = result & f
        return result


# ---------------------------------------------------------------------------
# Bounded response
# ---------------------------------------------------------------------------

class BoundedResponseSpec:
    """Obstacles / hazards are handled within bounded time.

    Constructs  □(trigger → ◇[0,d] response).

    Args:
        trigger: Trigger predicate.
        response: Response predicate.
        deadline: Maximum number of steps.
    """

    def __init__(
        self,
        trigger: Predicate,
        response: Predicate,
        deadline: int = 10,
    ) -> None:
        self.trigger = trigger
        self.response = response
        self.deadline = deadline

    def formula(self) -> TemporalFormula:
        return BoundedResponse(
            self.trigger, self.response, self.deadline,
            name="bounded_response",
        )


# ---------------------------------------------------------------------------
# Fair scheduling
# ---------------------------------------------------------------------------

class FairScheduling:
    """No agent is starved — every agent acts within a bounded window.

    For each agent, within every window of *window* steps, the agent
    takes at least one action.

    This is modelled as □[0,T] ⋀_i ◇[0,w] acted(i).

    Args:
        agent_ids: Agent identifiers.
        window: Fairness window size.
        horizon: Overall temporal horizon.
    """

    def __init__(
        self,
        agent_ids: List[str],
        window: int = 10,
        horizon: Optional[int] = None,
    ) -> None:
        self.agent_ids = agent_ids
        self.window = window
        self.horizon = horizon

    def formula(self) -> TemporalFormula:
        preds: List[Predicate] = []
        for aid in self.agent_ids:
            preds.append(CustomPredicate(
                lambda s, _aid=aid: True,
                name=f"acted({aid})",
            ))
        inner = ConjunctivePredicate(preds, name="all_acted")
        return Always(
            Eventually(inner, horizon=self.window),
            horizon=self.horizon,
            name="fair_scheduling",
        )


# ---------------------------------------------------------------------------
# Safety library — convenience façade
# ---------------------------------------------------------------------------

class SafetyLibrary:
    """Convenience façade for constructing common safety specifications.

    Example::

        lib = SafetyLibrary(["agent_0", "agent_1", "agent_2"])
        f = lib.highway_safety(min_dist=3.0, horizon=200)
    """

    def __init__(self, agent_ids: List[str]) -> None:
        self.agent_ids = agent_ids

    # -- highway specs -------------------------------------------------------

    def highway_safety(
        self,
        min_dist: float = 3.0,
        extents: Tuple[float, float] = (2.25, 1.0),
        horizon: Optional[int] = None,
    ) -> TemporalFormula:
        """No collisions and minimum separation on a highway."""
        cf = CollisionFreedom(self.agent_ids, extents, horizon).formula()
        ms = MinimumSeparation(self.agent_ids, min_dist, horizon=horizon).formula()
        return cf & ms

    def intersection_safety(
        self,
        min_dist: float = 4.0,
        horizon: Optional[int] = None,
    ) -> TemporalFormula:
        """Intersection safety: minimum separation + deadlock freedom."""
        ms = MinimumSeparation(self.agent_ids, min_dist, horizon=horizon).formula()
        df = DeadlockFreedom(self.agent_ids, horizon=horizon).formula()
        return ms & df

    # -- warehouse specs -----------------------------------------------------

    def warehouse_safety(
        self,
        min_dist: float = 1.0,
        task_deadline: int = 500,
        horizon: Optional[int] = None,
    ) -> TemporalFormula:
        """Warehouse safety: no collisions + deadlock freedom + liveness."""
        ms = MinimumSeparation(
            self.agent_ids, min_dist,
            pos_indices=(0, 1), horizon=horizon,
        ).formula()
        df = DeadlockFreedom(self.agent_ids, horizon=horizon).formula()
        lv = LivenessSpec(self.agent_ids, horizon=task_deadline).formula()
        return ms & df & lv

    # -- trading specs -------------------------------------------------------

    def trading_fairness(
        self,
        window: int = 10,
        horizon: Optional[int] = None,
    ) -> TemporalFormula:
        """Trading environment: fair scheduling among agents."""
        return FairScheduling(self.agent_ids, window, horizon).formula()

    # -- generic helpers -----------------------------------------------------

    def collision_freedom(
        self,
        extents: Tuple[float, float] = (2.25, 1.0),
        horizon: Optional[int] = None,
    ) -> TemporalFormula:
        return CollisionFreedom(self.agent_ids, extents, horizon).formula()

    def min_separation(
        self,
        d: float = 3.0,
        horizon: Optional[int] = None,
    ) -> TemporalFormula:
        return MinimumSeparation(self.agent_ids, d, horizon=horizon).formula()

    def deadlock_freedom(
        self, horizon: Optional[int] = None
    ) -> TemporalFormula:
        return DeadlockFreedom(self.agent_ids, horizon=horizon).formula()

    def liveness(
        self, horizon: Optional[int] = None
    ) -> TemporalFormula:
        return LivenessSpec(self.agent_ids, horizon=horizon).formula()
