"""
Adversarial replay trace generation for MARACE.

Given an adversarial schedule discovered by MCTS (or other search),
this module generates, verifies, minimises, visualises, and serialises
concrete replay traces that demonstrate multi-agent interaction races.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from marace.abstract.zonotope import Zonotope
from marace.hb.hb_graph import HBGraph
from marace.hb.vector_clock import VectorClock

logger = logging.getLogger(__name__)

# ======================================================================
# Numpy helpers
# ======================================================================


def _ndarray_to_list(arr: np.ndarray) -> List[float]:
    """Convert a numpy array to a JSON-serialisable list."""
    return arr.tolist()


def _list_to_ndarray(lst: List[float]) -> np.ndarray:
    """Convert a list back to a numpy array."""
    return np.asarray(lst, dtype=np.float64)


# ======================================================================
# ReplayStep
# ======================================================================


@dataclass
class ReplayStep:
    """A single step in an adversarial replay trace.

    Attributes:
        step_index:   Position of this step in the overall trace.
        agent_id:     Identifier of the acting agent.
        action:       Action vector taken by the agent.
        state_before: Joint state immediately before this step.
        state_after:  Joint state immediately after this step.
        timing_offset: Simulated wall-clock offset (seconds) applied to
                       model timing non-determinism in the schedule.
        vector_clock: Snapshot of the agent's vector clock *after* this
                      step, encoded as ``{agent_id: logical_time}``.
    """

    step_index: int
    agent_id: str
    action: np.ndarray
    state_before: np.ndarray
    state_after: np.ndarray
    timing_offset: float = 0.0
    vector_clock: Dict[str, int] = field(default_factory=dict)

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "agent_id": self.agent_id,
            "action": _ndarray_to_list(self.action),
            "state_before": _ndarray_to_list(self.state_before),
            "state_after": _ndarray_to_list(self.state_after),
            "timing_offset": self.timing_offset,
            "vector_clock": dict(self.vector_clock),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReplayStep":
        return cls(
            step_index=int(data["step_index"]),
            agent_id=str(data["agent_id"]),
            action=_list_to_ndarray(data["action"]),
            state_before=_list_to_ndarray(data["state_before"]),
            state_after=_list_to_ndarray(data["state_after"]),
            timing_offset=float(data.get("timing_offset", 0.0)),
            vector_clock={str(k): int(v) for k, v in data.get("vector_clock", {}).items()},
        )


# ======================================================================
# ReplayTrace
# ======================================================================


@dataclass
class ReplayTrace:
    """A full adversarial execution trace.

    Attributes:
        steps:         Ordered sequence of replay steps.
        safety_margin: Minimum observed safety margin across the trace.
                       Negative values indicate a safety violation.
        is_race:       Whether the trace demonstrates a confirmed race.
        metadata:      Arbitrary metadata (schedule hash, creation time,
                       search statistics, …).
    """

    steps: List[ReplayStep]
    safety_margin: float = 0.0
    is_race: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- derived properties ----------------------------------------------

    @property
    def duration(self) -> float:
        """Total simulated duration (sum of timing offsets)."""
        if not self.steps:
            return 0.0
        return sum(s.timing_offset for s in self.steps)

    @property
    def involved_agents(self) -> List[str]:
        """Sorted list of unique agent identifiers in the trace."""
        return sorted({s.agent_id for s in self.steps})

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "safety_margin": self.safety_margin,
            "is_race": self.is_race,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReplayTrace":
        return cls(
            steps=[ReplayStep.from_dict(s) for s in data["steps"]],
            safety_margin=float(data.get("safety_margin", 0.0)),
            is_race=bool(data.get("is_race", False)),
            metadata=data.get("metadata", {}),
        )


# ======================================================================
# ReplayGenerator
# ======================================================================


class ReplayGenerator:
    """Generate a concrete replay trace from an adversarial schedule.

    Parameters:
        transition_fn: ``(state, agent_id, action) -> new_state``
        policy_fn:     ``(state, agent_id) -> action``
    """

    def __init__(
        self,
        transition_fn: Callable[[np.ndarray, str, np.ndarray], np.ndarray],
        policy_fn: Callable[[np.ndarray, str], np.ndarray],
    ) -> None:
        self._transition_fn = transition_fn
        self._policy_fn = policy_fn

    def generate(
        self,
        initial_state: np.ndarray,
        schedule: Sequence[Dict[str, Any]],
        agent_ids: List[str],
    ) -> ReplayTrace:
        """Execute *schedule* from *initial_state* and record every step.

        Each element of *schedule* is a dict with at least ``agent_id``
        and an optional ``timing_offset``.  The policy function is queried
        to obtain the action for the scheduled agent, and the transition
        function advances the state.

        Args:
            initial_state: Starting joint state vector.
            schedule:      Ordered list of scheduling decisions.
            agent_ids:     Complete list of agent identifiers.

        Returns:
            A :class:`ReplayTrace` capturing the full execution.
        """
        state = np.array(initial_state, dtype=np.float64)
        clocks: Dict[str, VectorClock] = {aid: VectorClock() for aid in agent_ids}
        steps: List[ReplayStep] = []

        for idx, sched_entry in enumerate(schedule):
            # Accept both dict-like and attribute-based schedule entries
            # (e.g. ScheduleAction dataclasses).
            if hasattr(sched_entry, "agent_id"):
                agent_id = str(sched_entry.agent_id)
                timing_offset = float(getattr(sched_entry, "timing_offset", 0.0))
            else:
                agent_id = str(sched_entry["agent_id"])
                timing_offset = float(sched_entry.get("timing_offset", 0.0))

            action = self._policy_fn(state, agent_id)
            action = np.asarray(action, dtype=np.float64)

            state_before = state.copy()
            state_after = self._transition_fn(state, agent_id, action)
            state_after = np.asarray(state_after, dtype=np.float64)

            # Advance the acting agent's vector clock.
            clocks[agent_id].increment(agent_id)

            step = ReplayStep(
                step_index=idx,
                agent_id=agent_id,
                action=action,
                state_before=state_before,
                state_after=state_after,
                timing_offset=timing_offset,
                vector_clock=clocks[agent_id].to_dict(),
            )
            steps.append(step)
            state = state_after

        schedule_hash = hashlib.sha256(
            json.dumps([s.to_dict() for s in steps], sort_keys=True).encode()
        ).hexdigest()[:16]

        trace = ReplayTrace(
            steps=steps,
            metadata={
                "schedule_hash": schedule_hash,
                "creation_time": time.time(),
                "num_agents": len(agent_ids),
                "agent_ids": agent_ids,
            },
        )
        logger.debug(
            "Generated replay trace with %d steps (hash=%s)",
            len(steps),
            schedule_hash,
        )
        return trace


# ======================================================================
# ReplayVerifier
# ======================================================================


class ReplayVerifier:
    """Verify that a replay trace actually demonstrates the claimed race.

    The caller provides a *safety_predicate* ``(state) -> bool`` that
    returns ``True`` when the state is safe and ``False`` otherwise.
    """

    @staticmethod
    def verify(
        trace: ReplayTrace,
        safety_predicate: Callable[[np.ndarray], bool],
    ) -> bool:
        """Return ``True`` iff *trace* contains at least one safety violation.

        Also updates ``trace.is_race`` and ``trace.safety_margin`` as a
        side-effect.
        """
        found_violation = False
        min_margin: Optional[float] = None

        for step in trace.steps:
            safe = safety_predicate(step.state_after)
            if not safe:
                found_violation = True
            # Track margin as +1 (safe) / -1 (unsafe) for simple predicates.
            margin = 1.0 if safe else -1.0
            if min_margin is None or margin < min_margin:
                min_margin = margin

        trace.is_race = found_violation
        trace.safety_margin = min_margin if min_margin is not None else 0.0
        return found_violation

    @staticmethod
    def find_violation_step(
        trace: ReplayTrace,
        safety_predicate: Callable[[np.ndarray], bool],
    ) -> Optional[int]:
        """Return the index of the first step whose ``state_after`` violates
        the safety predicate, or ``None`` if the trace is safe."""
        for step in trace.steps:
            if not safety_predicate(step.state_after):
                return step.step_index
        return None


# ======================================================================
# MinimalReplayExtractor
# ======================================================================


class MinimalReplayExtractor:
    """Find a minimal sub-schedule that still triggers the race.

    Uses iterative single-step removal with a binary-search narrowing
    pass to find a (locally) minimal subset of the original schedule
    that still leads to a safety violation.
    """

    @staticmethod
    def extract(
        trace: ReplayTrace,
        generator: ReplayGenerator,
        safety_predicate: Callable[[np.ndarray], bool],
        initial_state: np.ndarray,
    ) -> ReplayTrace:
        """Return a minimal :class:`ReplayTrace` that still violates safety.

        If the original trace does not violate safety, it is returned
        unchanged.
        """
        if not ReplayVerifier.verify(copy.deepcopy(trace), safety_predicate):
            logger.info("Original trace is safe; nothing to minimise.")
            return trace

        agent_ids = trace.involved_agents
        schedule = [
            {"agent_id": s.agent_id, "timing_offset": s.timing_offset}
            for s in trace.steps
        ]

        current_schedule = list(schedule)

        # -- binary-search pass: try removing large blocks ---------------
        window = len(current_schedule) // 2
        while window >= 1:
            i = 0
            while i + window <= len(current_schedule):
                candidate = current_schedule[:i] + current_schedule[i + window:]
                if not candidate:
                    i += 1
                    continue
                candidate_trace = generator.generate(
                    initial_state, candidate, agent_ids,
                )
                if ReplayVerifier.verify(candidate_trace, safety_predicate):
                    current_schedule = candidate
                    logger.debug(
                        "Removed block at %d (window=%d), %d steps remain",
                        i, window, len(current_schedule),
                    )
                else:
                    i += 1
            window //= 2

        # -- single-step removal pass ------------------------------------
        changed = True
        while changed:
            changed = False
            for i in range(len(current_schedule)):
                candidate = current_schedule[:i] + current_schedule[i + 1:]
                if not candidate:
                    continue
                candidate_trace = generator.generate(
                    initial_state, candidate, agent_ids,
                )
                if ReplayVerifier.verify(candidate_trace, safety_predicate):
                    current_schedule = candidate
                    changed = True
                    logger.debug(
                        "Removed step %d, %d steps remain",
                        i, len(current_schedule),
                    )
                    break  # restart from beginning

        minimal_trace = generator.generate(initial_state, current_schedule, agent_ids)
        ReplayVerifier.verify(minimal_trace, safety_predicate)
        minimal_trace.metadata["minimised"] = True
        minimal_trace.metadata["original_length"] = len(trace.steps)
        logger.info(
            "Minimised trace from %d to %d steps.",
            len(trace.steps),
            len(minimal_trace.steps),
        )
        return minimal_trace


# ======================================================================
# ReplayVisualization
# ======================================================================


class ReplayVisualization:
    """Human-readable descriptions of replay traces."""

    @staticmethod
    def format_step(step: ReplayStep) -> str:
        """Format a single replay step."""
        lines = [
            f"Step {step.step_index}: agent={step.agent_id}",
            f"  action      = {np.array2string(step.action, precision=4)}",
            f"  state_before= {np.array2string(step.state_before, precision=4)}",
            f"  state_after = {np.array2string(step.state_after, precision=4)}",
            f"  timing_off  = {step.timing_offset:.4f}",
            f"  vclock      = {step.vector_clock}",
        ]
        return "\n".join(lines)

    @classmethod
    def format_trace(cls, trace: ReplayTrace) -> str:
        """Return a multi-line human-readable representation of *trace*."""
        header_parts = [
            f"ReplayTrace  race={trace.is_race}  "
            f"margin={trace.safety_margin:.4f}  "
            f"steps={len(trace.steps)}  "
            f"agents={trace.involved_agents}",
        ]
        if trace.metadata.get("schedule_hash"):
            header_parts.append(f"  hash={trace.metadata['schedule_hash']}")
        header = "".join(header_parts)
        sep = "-" * max(60, len(header))
        body = "\n".join(cls.format_step(s) for s in trace.steps)
        return f"{sep}\n{header}\n{sep}\n{body}\n{sep}"

    @classmethod
    def format_race_summary(cls, trace: ReplayTrace) -> str:
        """Short summary highlighting the race (if any)."""
        if not trace.is_race:
            return "No race detected in trace."

        violation_idx: Optional[int] = None
        for step in trace.steps:
            if step.step_index is not None:
                # We can't re-check the predicate here, so report metadata.
                pass
        # Best-effort: report first & last agents involved
        agents = trace.involved_agents
        lines = [
            f"RACE DETECTED  (safety_margin={trace.safety_margin:.4f})",
            f"  Agents involved : {agents}",
            f"  Schedule length : {len(trace.steps)}",
            f"  Total duration  : {trace.duration:.4f}",
        ]
        if trace.metadata.get("schedule_hash"):
            lines.append(f"  Schedule hash   : {trace.metadata['schedule_hash']}")
        if trace.metadata.get("minimised"):
            lines.append(
                f"  Minimised from  : {trace.metadata.get('original_length')} steps"
            )
        return "\n".join(lines)


# ======================================================================
# ReplaySerializer
# ======================================================================


class ReplaySerializer:
    """Persist replay traces as JSON files."""

    @staticmethod
    def save(trace: ReplayTrace, filepath: str) -> None:
        """Write *trace* to *filepath* as JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(trace.to_dict(), fh, indent=2)
        logger.debug("Saved trace to %s", filepath)

    @staticmethod
    def load(filepath: str) -> ReplayTrace:
        """Read a :class:`ReplayTrace` from *filepath*."""
        with open(filepath, "r") as fh:
            data = json.load(fh)
        return ReplayTrace.from_dict(data)

    @staticmethod
    def save_batch(traces: List[ReplayTrace], filepath: str) -> None:
        """Write a list of traces to *filepath* as a JSON array."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump([t.to_dict() for t in traces], fh, indent=2)
        logger.debug("Saved %d traces to %s", len(traces), filepath)

    @staticmethod
    def load_batch(filepath: str) -> List[ReplayTrace]:
        """Read a list of :class:`ReplayTrace` from *filepath*."""
        with open(filepath, "r") as fh:
            data = json.load(fh)
        return [ReplayTrace.from_dict(d) for d in data]


# ======================================================================
# CounterfactualReplay
# ======================================================================


class CounterfactualReplay:
    """Show what happens under alternative schedules.

    Useful for demonstrating that the race is *schedule-sensitive*:
    small perturbations to the ordering can make or break safety.
    """

    @staticmethod
    def compare(
        trace: ReplayTrace,
        alt_schedule: Sequence[Dict[str, Any]],
        generator: ReplayGenerator,
        initial_state: np.ndarray,
        safety_predicate: Callable[[np.ndarray], bool],
    ) -> Dict[str, Any]:
        """Compare the original trace against an alternative schedule.

        Returns a dict with keys:

        * ``original_trace`` – the input trace.
        * ``alternative_trace`` – trace generated from *alt_schedule*.
        * ``original_safe`` – whether the original is safe.
        * ``alternative_safe`` – whether the alternative is safe.
        """
        agent_ids = trace.involved_agents
        alt_trace = generator.generate(initial_state, alt_schedule, agent_ids)

        original_safe = not ReplayVerifier.verify(
            copy.deepcopy(trace), safety_predicate,
        )
        alternative_safe = not ReplayVerifier.verify(alt_trace, safety_predicate)

        return {
            "original_trace": trace,
            "alternative_trace": alt_trace,
            "original_safe": original_safe,
            "alternative_safe": alternative_safe,
        }

    @staticmethod
    def demonstrate_hb_sensitivity(
        trace: ReplayTrace,
        generator: ReplayGenerator,
        initial_state: np.ndarray,
        safety_predicate: Callable[[np.ndarray], bool],
        num_permutations: int = 20,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """Generate *num_permutations* random valid permutations of the
        schedule and report which ones violate safety.

        A "valid permutation" preserves per-agent ordering (i.e. the
        relative order of steps for each agent is maintained) while
        allowing interleaving to change.

        Returns a dict with:

        * ``total`` – number of permutations tested.
        * ``violations`` – number that triggered a safety violation.
        * ``safe`` – number that were safe.
        * ``traces`` – list of ``(permuted_trace, is_violation)`` pairs.
        """
        if rng is None:
            rng = np.random.default_rng()

        agent_ids = trace.involved_agents

        # Group original schedule entries by agent, preserving per-agent order.
        per_agent: Dict[str, List[Dict[str, Any]]] = {aid: [] for aid in agent_ids}
        for step in trace.steps:
            per_agent[step.agent_id].append(
                {"agent_id": step.agent_id, "timing_offset": step.timing_offset}
            )

        results: List[Tuple[ReplayTrace, bool]] = []
        violations = 0

        for _ in range(num_permutations):
            permuted = _random_interleaving(per_agent, rng)
            perm_trace = generator.generate(initial_state, permuted, agent_ids)
            is_violation = ReplayVerifier.verify(perm_trace, safety_predicate)
            if is_violation:
                violations += 1
            results.append((perm_trace, is_violation))

        safe_count = num_permutations - violations
        logger.info(
            "HB sensitivity: %d/%d permutations violated safety.",
            violations,
            num_permutations,
        )
        return {
            "total": num_permutations,
            "violations": violations,
            "safe": safe_count,
            "traces": results,
        }


# ======================================================================
# Internal helpers
# ======================================================================


def _random_interleaving(
    per_agent: Dict[str, List[Dict[str, Any]]],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Produce a random interleaving that preserves per-agent order.

    Each agent's steps appear in their original relative order, but the
    interleaving across agents is uniformly random (generated by
    repeatedly picking a non-empty agent queue at random).
    """
    queues: Dict[str, List[Dict[str, Any]]] = {
        aid: list(entries) for aid, entries in per_agent.items() if entries
    }
    result: List[Dict[str, Any]] = []
    while queues:
        eligible = list(queues.keys())
        chosen = eligible[int(rng.integers(len(eligible)))]
        result.append(queues[chosen].pop(0))
        if not queues[chosen]:
            del queues[chosen]
    return result
