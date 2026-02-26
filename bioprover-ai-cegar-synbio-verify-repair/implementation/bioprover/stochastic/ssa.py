"""
Stochastic Simulation Algorithm (SSA) implementations.

Provides Gillespie's direct method and Gibson-Bruck next reaction method
with indexed priority queue, dependency graph, and trajectory recording.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Reaction:
    """A chemical reaction with stoichiometry and rate constant.

    Attributes:
        name: Human-readable name.
        reactants: Dict mapping species index to stoichiometric coefficient consumed.
        products: Dict mapping species index to stoichiometric coefficient produced.
        rate_constant: Stochastic rate constant (units depend on reaction order).
        propensity_func: Optional custom propensity function f(state) -> float.
    """

    name: str
    reactants: Dict[int, int]
    products: Dict[int, int]
    rate_constant: float
    propensity_func: Optional[Callable[[np.ndarray], float]] = None

    @property
    def state_change(self) -> Dict[int, int]:
        """Net state-change vector as sparse dict."""
        change: Dict[int, int] = {}
        for sp, coeff in self.reactants.items():
            change[sp] = change.get(sp, 0) - coeff
        for sp, coeff in self.products.items():
            change[sp] = change.get(sp, 0) + coeff
        return {k: v for k, v in change.items() if v != 0}

    def propensity(self, state: np.ndarray) -> float:
        """Compute propensity for mass-action kinetics or custom function."""
        if self.propensity_func is not None:
            return self.propensity_func(state)
        a = self.rate_constant
        for sp, coeff in self.reactants.items():
            x = int(state[sp])
            if x < coeff:
                return 0.0
            # Combinatorial factor: x choose coeff * coeff!
            for i in range(coeff):
                a *= (x - i)
        return a

    def affected_species(self) -> set:
        """Species whose copy number changes when this reaction fires."""
        return set(self.state_change.keys())

    def depends_on_species(self) -> set:
        """Species that appear in the propensity calculation."""
        return set(self.reactants.keys())


@dataclass
class StochasticState:
    """Immutable snapshot of simulation state."""

    time: float
    copy_numbers: np.ndarray
    fired_reaction: Optional[int] = None

    def copy(self) -> "StochasticState":
        return StochasticState(
            time=self.time,
            copy_numbers=self.copy_numbers.copy(),
            fired_reaction=self.fired_reaction,
        )


class TrajectoryRecorder:
    """Records simulation trajectories with full or sampled recording.

    Parameters:
        num_species: Number of molecular species.
        record_mode: 'full' records every event, 'sampled' records at fixed intervals.
        sample_interval: Time interval for sampled recording.
        max_events: Maximum events to store (prevents memory blowup).
    """

    def __init__(
        self,
        num_species: int,
        record_mode: str = "full",
        sample_interval: float = 1.0,
        max_events: int = 10_000_000,
    ):
        self.num_species = num_species
        self.record_mode = record_mode
        self.sample_interval = sample_interval
        self.max_events = max_events
        self._times: List[float] = []
        self._states: List[np.ndarray] = []
        self._reactions: List[int] = []
        self._next_sample_time = 0.0
        self._last_state: Optional[np.ndarray] = None
        self._event_count = 0

    def reset(self):
        self._times.clear()
        self._states.clear()
        self._reactions.clear()
        self._next_sample_time = 0.0
        self._last_state = None
        self._event_count = 0

    def record(self, time: float, state: np.ndarray, reaction_idx: int = -1):
        """Record a simulation event."""
        self._last_state = state.copy()
        self._event_count += 1
        if self.record_mode == "full":
            if len(self._times) < self.max_events:
                self._times.append(time)
                self._states.append(state.copy())
                self._reactions.append(reaction_idx)
        elif self.record_mode == "sampled":
            while self._next_sample_time <= time:
                if len(self._times) < self.max_events:
                    self._times.append(self._next_sample_time)
                    self._states.append(state.copy())
                    self._reactions.append(reaction_idx)
                self._next_sample_time += self.sample_interval

    def record_initial(self, time: float, state: np.ndarray):
        self._times.append(time)
        self._states.append(state.copy())
        self._reactions.append(-1)
        self._last_state = state.copy()
        if self.record_mode == "sampled":
            self._next_sample_time = time + self.sample_interval

    def get_times(self) -> np.ndarray:
        return np.array(self._times)

    def get_states(self) -> np.ndarray:
        if not self._states:
            return np.empty((0, self.num_species))
        return np.array(self._states)

    def get_reactions(self) -> np.ndarray:
        return np.array(self._reactions, dtype=int)

    @property
    def event_count(self) -> int:
        return self._event_count

    def interpolate_at(self, query_times: np.ndarray) -> np.ndarray:
        """Interpolate trajectory at given query times (piecewise constant)."""
        times = self.get_times()
        states = self.get_states()
        if len(times) == 0:
            raise ValueError("No trajectory data recorded")
        indices = np.searchsorted(times, query_times, side="right") - 1
        indices = np.clip(indices, 0, len(times) - 1)
        return states[indices]


class DependencyGraph:
    """Tracks which reaction propensities need recomputation after a reaction fires.

    For each reaction j, stores the set of reactions whose propensities depend on
    species affected by j.
    """

    def __init__(self, reactions: List[Reaction], num_species: int):
        self.num_reactions = len(reactions)
        self.num_species = num_species
        # species_to_reactions[s] = set of reaction indices whose propensity depends on s
        species_to_reactions: Dict[int, set] = {i: set() for i in range(num_species)}
        for j, rxn in enumerate(reactions):
            for sp in rxn.depends_on_species():
                if sp < num_species:
                    species_to_reactions[sp].add(j)
        # dependency_graph[j] = reactions that must be updated when reaction j fires
        self.graph: List[set] = []
        for j, rxn in enumerate(reactions):
            deps = set()
            for sp in rxn.affected_species():
                if sp < num_species:
                    deps.update(species_to_reactions[sp])
            self.graph.append(deps)

    def get_dependents(self, reaction_idx: int) -> set:
        return self.graph[reaction_idx]


class DirectMethod:
    """Gillespie's Direct Method for exact stochastic simulation.

    Parameters:
        reactions: List of Reaction objects.
        num_species: Number of molecular species.
        seed: Random seed for reproducibility.
        use_binary_search: Use binary search for reaction selection.
    """

    def __init__(
        self,
        reactions: List[Reaction],
        num_species: int,
        seed: Optional[int] = None,
        use_binary_search: bool = False,
    ):
        self.reactions = reactions
        self.num_species = num_species
        self.num_reactions = len(reactions)
        self.rng = np.random.default_rng(seed)
        self.use_binary_search = use_binary_search
        # Precompute dense state-change vectors
        self._stoich_matrix = np.zeros(
            (self.num_reactions, self.num_species), dtype=int
        )
        for j, rxn in enumerate(reactions):
            for sp, delta in rxn.state_change.items():
                if sp < self.num_species:
                    self._stoich_matrix[j, sp] = delta

    def _compute_propensities(self, state: np.ndarray) -> np.ndarray:
        """Compute all propensities for the current state."""
        props = np.empty(self.num_reactions)
        for j, rxn in enumerate(self.reactions):
            props[j] = rxn.propensity(state)
        return props

    def _select_reaction_linear(
        self, propensities: np.ndarray, a0: float
    ) -> int:
        """Linear search for reaction selection."""
        r = self.rng.random() * a0
        cumsum = 0.0
        for j in range(self.num_reactions):
            cumsum += propensities[j]
            if cumsum >= r:
                return j
        return self.num_reactions - 1

    def _select_reaction_binary(
        self, propensities: np.ndarray, a0: float
    ) -> int:
        """Binary search for reaction selection using cumulative sums."""
        cumsum = np.cumsum(propensities)
        r = self.rng.random() * a0
        idx = np.searchsorted(cumsum, r, side="left")
        return min(idx, self.num_reactions - 1)

    def _select_reaction(self, propensities: np.ndarray, a0: float) -> int:
        if self.use_binary_search:
            return self._select_reaction_binary(propensities, a0)
        return self._select_reaction_linear(propensities, a0)

    def _generate_waiting_time(self, a0: float) -> float:
        """Generate exponential waiting time."""
        u = self.rng.random()
        while u == 0.0:
            u = self.rng.random()
        return -math.log(u) / a0

    def simulate(
        self,
        initial_state: np.ndarray,
        t_end: float,
        t_start: float = 0.0,
        max_steps: int = 10_000_000,
        recorder: Optional[TrajectoryRecorder] = None,
    ) -> StochasticState:
        """Run a single SSA trajectory.

        Returns:
            Final StochasticState at t_end or when all propensities reach zero.
        """
        state = initial_state.astype(np.int64).copy()
        t = t_start
        if recorder is not None:
            recorder.record_initial(t, state)

        for step in range(max_steps):
            propensities = self._compute_propensities(state)
            a0 = propensities.sum()
            if a0 <= 0.0:
                break
            dt = self._generate_waiting_time(a0)
            if t + dt > t_end:
                break
            t += dt
            j = self._select_reaction(propensities, a0)
            state += self._stoich_matrix[j]
            if recorder is not None:
                recorder.record(t, state, j)

        return StochasticState(time=t, copy_numbers=state.copy())

    def simulate_trajectory(
        self,
        initial_state: np.ndarray,
        t_end: float,
        t_start: float = 0.0,
        record_mode: str = "full",
        sample_interval: float = 1.0,
        max_steps: int = 10_000_000,
    ) -> TrajectoryRecorder:
        """Convenience: simulate and return the recorder."""
        recorder = TrajectoryRecorder(
            self.num_species,
            record_mode=record_mode,
            sample_interval=sample_interval,
        )
        self.simulate(initial_state, t_end, t_start, max_steps, recorder)
        return recorder


class IndexedPriorityQueue:
    """Min-heap with O(log n) update for the Next Reaction Method.

    Stores (putative_time, reaction_index) pairs with efficient update
    of individual entries.
    """

    def __init__(self, size: int):
        self.size = size
        self._times = np.full(size, np.inf)
        self._heap: List[Tuple[float, int]] = []
        self._valid = np.zeros(size, dtype=bool)
        self._generation = np.zeros(size, dtype=int)

    def initialize(self, times: np.ndarray):
        """Initialize with an array of putative times."""
        self._times[:] = times
        self._generation[:] = 0
        self._valid[:] = True
        self._heap = [(times[i], i) for i in range(self.size)]
        heapq.heapify(self._heap)

    def update(self, idx: int, new_time: float):
        """Update the putative time for reaction idx."""
        self._times[idx] = new_time
        self._generation[idx] += 1
        heapq.heappush(self._heap, (new_time, idx))

    def top(self) -> Tuple[float, int]:
        """Return (time, reaction_index) of the minimum without removing."""
        while self._heap:
            t, idx = self._heap[0]
            if t == self._times[idx]:
                return t, idx
            heapq.heappop(self._heap)
        return np.inf, -1

    def pop(self) -> Tuple[float, int]:
        """Remove and return the minimum (time, reaction_index)."""
        while self._heap:
            t, idx = heapq.heappop(self._heap)
            if t == self._times[idx]:
                return t, idx
        return np.inf, -1


class NextReactionMethod:
    """Gibson-Bruck Next Reaction Method with indexed priority queue.

    Uses a dependency graph for efficient propensity updates and an indexed
    priority queue for O(log M) reaction selection where M = num reactions.

    Parameters:
        reactions: List of Reaction objects.
        num_species: Number of molecular species.
        seed: Random seed.
    """

    def __init__(
        self,
        reactions: List[Reaction],
        num_species: int,
        seed: Optional[int] = None,
    ):
        self.reactions = reactions
        self.num_species = num_species
        self.num_reactions = len(reactions)
        self.rng = np.random.default_rng(seed)
        self.dep_graph = DependencyGraph(reactions, num_species)
        self._stoich_matrix = np.zeros(
            (self.num_reactions, self.num_species), dtype=int
        )
        for j, rxn in enumerate(reactions):
            for sp, delta in rxn.state_change.items():
                if sp < self.num_species:
                    self._stoich_matrix[j, sp] = delta

    def _compute_propensity(self, j: int, state: np.ndarray) -> float:
        return self.reactions[j].propensity(state)

    def simulate(
        self,
        initial_state: np.ndarray,
        t_end: float,
        t_start: float = 0.0,
        max_steps: int = 10_000_000,
        recorder: Optional[TrajectoryRecorder] = None,
    ) -> StochasticState:
        """Run NRM simulation."""
        state = initial_state.astype(np.int64).copy()
        t = t_start

        if recorder is not None:
            recorder.record_initial(t, state)

        # Initialize propensities and putative times
        propensities = np.array(
            [self._compute_propensity(j, state) for j in range(self.num_reactions)]
        )
        # Internal clocks (unit-rate Poisson process times)
        internal_times = np.zeros(self.num_reactions)
        # Generate initial putative times
        putative_times = np.full(self.num_reactions, np.inf)
        for j in range(self.num_reactions):
            if propensities[j] > 0:
                dt = -math.log(self.rng.random()) / propensities[j]
                putative_times[j] = t + dt
            else:
                putative_times[j] = np.inf

        pq = IndexedPriorityQueue(self.num_reactions)
        pq.initialize(putative_times)

        for step in range(max_steps):
            tau, mu = pq.top()
            if tau > t_end or mu < 0:
                break
            t = tau
            # Update state
            state += self._stoich_matrix[mu]
            if recorder is not None:
                recorder.record(t, state, mu)

            # Generate new internal time for fired reaction
            old_a_mu = propensities[mu]

            # Recompute propensities for affected reactions
            dependents = self.dep_graph.get_dependents(mu)
            for j in dependents:
                old_a = propensities[j]
                new_a = self._compute_propensity(j, state)
                propensities[j] = new_a
                if j == mu:
                    # Fired reaction: generate new putative time from scratch
                    if new_a > 0:
                        dt = -math.log(self.rng.random()) / new_a
                        new_tau = t + dt
                    else:
                        new_tau = np.inf
                else:
                    # Rescale putative time
                    if new_a > 0:
                        old_tau = pq._times[j]
                        if old_a > 0:
                            new_tau = t + (old_a / new_a) * (old_tau - t)
                        else:
                            dt = -math.log(self.rng.random()) / new_a
                            new_tau = t + dt
                    else:
                        new_tau = np.inf
                pq.update(j, new_tau)

        return StochasticState(time=t, copy_numbers=state.copy())

    def simulate_trajectory(
        self,
        initial_state: np.ndarray,
        t_end: float,
        t_start: float = 0.0,
        record_mode: str = "full",
        sample_interval: float = 1.0,
        max_steps: int = 10_000_000,
    ) -> TrajectoryRecorder:
        recorder = TrajectoryRecorder(
            self.num_species,
            record_mode=record_mode,
            sample_interval=sample_interval,
        )
        self.simulate(initial_state, t_end, t_start, max_steps, recorder)
        return recorder


def run_ensemble_ssa(
    reactions: List[Reaction],
    num_species: int,
    initial_state: np.ndarray,
    t_end: float,
    num_runs: int = 100,
    method: str = "direct",
    seed: Optional[int] = None,
    sample_times: Optional[np.ndarray] = None,
    sample_interval: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Run multiple SSA trajectories and compute statistics.

    Parameters:
        reactions: Reaction list.
        num_species: Number of species.
        initial_state: Initial copy numbers.
        t_end: End time.
        num_runs: Number of independent trajectories.
        method: 'direct' or 'next_reaction'.
        seed: Base random seed. Each run uses seed + i.
        sample_times: Times at which to sample. If None, uses sample_interval.
        sample_interval: Interval for uniform sampling.

    Returns:
        Dict with keys 'times', 'mean', 'variance', 'trajectories'.
    """
    if sample_times is None:
        sample_times = np.arange(0, t_end + sample_interval, sample_interval)

    all_trajectories = np.zeros((num_runs, len(sample_times), num_species))

    for i in range(num_runs):
        run_seed = (seed + i) if seed is not None else None
        if method == "next_reaction":
            sim = NextReactionMethod(reactions, num_species, seed=run_seed)
        else:
            sim = DirectMethod(reactions, num_species, seed=run_seed)

        rec = sim.simulate_trajectory(
            initial_state,
            t_end,
            record_mode="sampled",
            sample_interval=sample_interval,
        )
        all_trajectories[i] = rec.interpolate_at(sample_times)

    mean = np.mean(all_trajectories, axis=0)
    variance = np.var(all_trajectories, axis=0)

    return {
        "times": sample_times,
        "mean": mean,
        "variance": variance,
        "trajectories": all_trajectories,
    }
