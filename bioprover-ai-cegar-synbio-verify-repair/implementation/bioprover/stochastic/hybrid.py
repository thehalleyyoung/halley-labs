"""
Hybrid SSA/ODE simulation for biochemical networks.

Implements the Haseltine-Rawlings partitioned approach: species with high
copy numbers are simulated deterministically (ODE) while species with low
copy numbers use exact SSA. Partitions are updated dynamically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .ssa import Reaction, StochasticState, TrajectoryRecorder


@dataclass
class SpeciesPartition:
    """Partitioning of species into stochastic and deterministic subsets.

    Attributes:
        stochastic: Set of species indices simulated stochastically.
        deterministic: Set of species indices simulated deterministically.
        threshold_low: Below this copy number, species moves to stochastic.
        threshold_high: Above this, species moves to deterministic.
    """

    stochastic: Set[int]
    deterministic: Set[int]
    threshold_low: int = 100
    threshold_high: int = 200

    @staticmethod
    def from_state(
        state: np.ndarray,
        threshold_low: int = 100,
        threshold_high: int = 200,
    ) -> "SpeciesPartition":
        """Create partition based on current copy numbers."""
        stoch = set()
        det = set()
        for i, x in enumerate(state):
            if x < threshold_low:
                stoch.add(i)
            else:
                det.add(i)
        return SpeciesPartition(stoch, det, threshold_low, threshold_high)

    def should_repartition(self, state: np.ndarray) -> bool:
        """Check if any species should switch partition."""
        for i in self.stochastic:
            if state[i] >= self.threshold_high:
                return True
        for i in self.deterministic:
            if state[i] < self.threshold_low:
                return True
        return False

    def repartition(self, state: np.ndarray) -> "SpeciesPartition":
        """Create updated partition based on current state."""
        return SpeciesPartition.from_state(
            state, self.threshold_low, self.threshold_high
        )


class DynamicRepartitioner:
    """Manages dynamic repartitioning with hysteresis and rate limiting.

    Prevents rapid switching by enforcing minimum intervals between
    repartitioning events and using hysteresis thresholds.
    """

    def __init__(
        self,
        threshold_low: int = 100,
        threshold_high: int = 200,
        min_interval: float = 0.1,
        max_switches_per_species: int = 10,
    ):
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.min_interval = min_interval
        self.max_switches_per_species = max_switches_per_species
        self._last_repartition_time = -np.inf
        self._switch_counts: Dict[int, int] = {}

    def check_and_repartition(
        self,
        partition: SpeciesPartition,
        state: np.ndarray,
        current_time: float,
    ) -> Tuple[SpeciesPartition, bool]:
        """Check if repartitioning is needed and perform if so.

        Returns (new_partition, did_repartition).
        """
        if current_time - self._last_repartition_time < self.min_interval:
            return partition, False

        if not partition.should_repartition(state):
            return partition, False

        new_partition = SpeciesPartition(
            set(), set(), self.threshold_low, self.threshold_high
        )
        for i in range(len(state)):
            count = self._switch_counts.get(i, 0)
            currently_stoch = i in partition.stochastic

            if state[i] < self.threshold_low:
                want_stoch = True
            elif state[i] >= self.threshold_high:
                want_stoch = False
            else:
                want_stoch = currently_stoch

            if want_stoch != currently_stoch:
                if count >= self.max_switches_per_species:
                    want_stoch = currently_stoch
                else:
                    self._switch_counts[i] = count + 1

            if want_stoch:
                new_partition.stochastic.add(i)
            else:
                new_partition.deterministic.add(i)

        self._last_repartition_time = current_time
        changed = (
            new_partition.stochastic != partition.stochastic
            or new_partition.deterministic != partition.deterministic
        )
        return new_partition, changed


@dataclass
class HybridTrajectory:
    """Recorded hybrid simulation trajectory.

    Stores times, full state, and partition membership at each point.
    """

    times: List[float] = field(default_factory=list)
    states: List[np.ndarray] = field(default_factory=list)
    partitions: List[SpeciesPartition] = field(default_factory=list)
    events: List[str] = field(default_factory=list)

    def record(
        self,
        time: float,
        state: np.ndarray,
        partition: SpeciesPartition,
        event: str = "",
    ):
        self.times.append(time)
        self.states.append(state.copy())
        self.partitions.append(partition)
        self.events.append(event)

    def get_times(self) -> np.ndarray:
        return np.array(self.times)

    def get_states(self) -> np.ndarray:
        if not self.states:
            return np.empty((0,))
        return np.array(self.states)

    def interpolate_at(self, query_times: np.ndarray) -> np.ndarray:
        """Piecewise-constant interpolation at query times."""
        times = self.get_times()
        states = self.get_states()
        indices = np.searchsorted(times, query_times, side="right") - 1
        indices = np.clip(indices, 0, len(times) - 1)
        return states[indices]


class _ReactionClassifier:
    """Classifies reactions based on species partition.

    A reaction is:
    - 'stochastic' if it depends on any stochastic species
    - 'deterministic' if all reactants are deterministic
    - 'coupling' if it involves both partitions
    """

    def __init__(
        self,
        reactions: List[Reaction],
        partition: SpeciesPartition,
    ):
        self.reactions = reactions
        self.partition = partition
        self.stochastic_rxns: List[int] = []
        self.deterministic_rxns: List[int] = []
        self.coupling_rxns: List[int] = []
        self._classify()

    def _classify(self):
        self.stochastic_rxns.clear()
        self.deterministic_rxns.clear()
        self.coupling_rxns.clear()
        for j, rxn in enumerate(self.reactions):
            deps = rxn.depends_on_species()
            affects = rxn.affected_species()
            all_species = deps | affects
            has_stoch = bool(all_species & self.partition.stochastic)
            has_det = bool(all_species & self.partition.deterministic)

            if has_stoch and has_det:
                self.coupling_rxns.append(j)
            elif has_stoch:
                self.stochastic_rxns.append(j)
            else:
                self.deterministic_rxns.append(j)

    def update(self, partition: SpeciesPartition):
        self.partition = partition
        self._classify()


class HaseltineRawlingsHybrid:
    """Haseltine-Rawlings hybrid SSA/ODE simulation.

    Species are partitioned into stochastic (low copy) and deterministic
    (high copy) subsets. Stochastic species use SSA with propensities
    that depend on the current deterministic state. Deterministic species
    evolve via ODE between stochastic events.

    Parameters:
        reactions: List of Reaction objects.
        num_species: Number of species.
        threshold_low: Copy number below which species is stochastic.
        threshold_high: Copy number above which species is deterministic.
        seed: Random seed.
        ode_method: Integration method for deterministic species.
        dynamic_partition: Whether to allow repartitioning.
    """

    def __init__(
        self,
        reactions: List[Reaction],
        num_species: int,
        threshold_low: int = 100,
        threshold_high: int = 200,
        seed: Optional[int] = None,
        ode_method: str = "RK45",
        dynamic_partition: bool = True,
    ):
        self.reactions = reactions
        self.num_species = num_species
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.rng = np.random.default_rng(seed)
        self.ode_method = ode_method
        self.dynamic_partition = dynamic_partition
        self._stoich = np.zeros((len(reactions), num_species), dtype=int)
        for j, rxn in enumerate(reactions):
            for sp, delta in rxn.state_change.items():
                if sp < num_species:
                    self._stoich[j, sp] = delta

        if dynamic_partition:
            self.repartitioner = DynamicRepartitioner(
                threshold_low, threshold_high
            )
        else:
            self.repartitioner = None

    def _compute_stochastic_propensities(
        self,
        state: np.ndarray,
        stoch_rxn_indices: List[int],
    ) -> np.ndarray:
        """Compute propensities for stochastic reactions only."""
        props = np.empty(len(stoch_rxn_indices))
        for i, j in enumerate(stoch_rxn_indices):
            props[i] = self.reactions[j].propensity(state)
        return props

    def _ode_rhs(
        self,
        state: np.ndarray,
        det_species: List[int],
        det_rxn_indices: List[int],
        stoch_state_snapshot: np.ndarray,
    ) -> Callable:
        """Create ODE right-hand-side function for deterministic species.

        The stochastic species are held constant at their current values.
        """
        def rhs(t: float, y_det: np.ndarray) -> np.ndarray:
            # Build full state
            full_state = stoch_state_snapshot.copy().astype(float)
            for i_local, i_global in enumerate(det_species):
                full_state[i_global] = y_det[i_local]
            # Compute rates for deterministic reactions
            dy = np.zeros(len(det_species))
            for j in det_rxn_indices:
                rate = self.reactions[j].propensity(
                    np.maximum(full_state, 0)
                )
                for i_local, i_global in enumerate(det_species):
                    dy[i_local] += self._stoich[j, i_global] * rate
            return dy

        return rhs

    def _ssa_step(
        self,
        state: np.ndarray,
        stoch_rxn_indices: List[int],
        coupling_rxn_indices: List[int],
    ) -> Tuple[Optional[float], Optional[int]]:
        """Perform one SSA step for stochastic + coupling reactions.

        Returns (waiting_time, reaction_index_in_full_list) or (None, None).
        """
        all_rxn_indices = stoch_rxn_indices + coupling_rxn_indices
        if not all_rxn_indices:
            return None, None

        props = np.array(
            [self.reactions[j].propensity(state) for j in all_rxn_indices]
        )
        a0 = props.sum()
        if a0 <= 0:
            return None, None

        u1 = self.rng.random()
        while u1 == 0.0:
            u1 = self.rng.random()
        dt = -math.log(u1) / a0

        u2 = self.rng.random() * a0
        cumsum = 0.0
        selected_local = 0
        for k in range(len(all_rxn_indices)):
            cumsum += props[k]
            if cumsum >= u2:
                selected_local = k
                break

        return dt, all_rxn_indices[selected_local]

    def simulate(
        self,
        initial_state: np.ndarray,
        t_end: float,
        t_start: float = 0.0,
        max_steps: int = 10_000_000,
        partition: Optional[SpeciesPartition] = None,
        trajectory: Optional[HybridTrajectory] = None,
    ) -> StochasticState:
        """Run hybrid simulation.

        Parameters:
            initial_state: Initial copy numbers.
            t_end: End time.
            partition: Initial partition. If None, auto-determined from state.
            trajectory: Optional trajectory recorder.

        Returns:
            Final StochasticState.
        """
        state = initial_state.astype(np.float64).copy()
        t = t_start

        if partition is None:
            partition = SpeciesPartition.from_state(
                state, self.threshold_low, self.threshold_high
            )

        classifier = _ReactionClassifier(self.reactions, partition)

        if trajectory is not None:
            trajectory.record(t, state, partition, "init")

        for step in range(max_steps):
            if t >= t_end:
                break

            # Dynamic repartitioning
            if self.dynamic_partition and self.repartitioner is not None:
                new_partition, changed = self.repartitioner.check_and_repartition(
                    partition, state, t
                )
                if changed:
                    partition = new_partition
                    classifier.update(partition)
                    if trajectory is not None:
                        trajectory.record(t, state, partition, "repartition")

            det_species = sorted(partition.deterministic)
            stoch_species = sorted(partition.stochastic)

            # If all species are deterministic, just integrate ODE
            if not stoch_species and not classifier.coupling_rxns:
                if det_species:
                    y0 = np.array([state[i] for i in det_species])
                    rhs = self._ode_rhs(
                        state, det_species,
                        classifier.deterministic_rxns,
                        state,
                    )
                    sol = solve_ivp(
                        rhs, (t, t_end), y0,
                        method=self.ode_method,
                        rtol=1e-6, atol=1e-8,
                    )
                    for i_local, i_global in enumerate(det_species):
                        state[i_global] = max(sol.y[i_local, -1], 0)
                t = t_end
                if trajectory is not None:
                    trajectory.record(t, state, partition, "ode_final")
                break

            # If all species are stochastic, just do SSA
            if not det_species:
                dt, j = self._ssa_step(
                    state,
                    classifier.stochastic_rxns,
                    classifier.coupling_rxns,
                )
                if dt is None or t + dt > t_end:
                    t = t_end
                    break
                t += dt
                state += self._stoich[j]
                state = np.maximum(state, 0)
                if trajectory is not None:
                    trajectory.record(t, state, partition, f"ssa_rxn_{j}")
                continue

            # Hybrid: compute next SSA time
            dt_ssa, j_ssa = self._ssa_step(
                state,
                classifier.stochastic_rxns,
                classifier.coupling_rxns,
            )
            if dt_ssa is None:
                dt_ssa = t_end - t + 1.0
                j_ssa = None

            ode_end = min(t + dt_ssa, t_end)

            # Integrate deterministic species from t to ode_end
            if det_species and ode_end > t:
                y0 = np.array([state[i] for i in det_species])
                rhs = self._ode_rhs(
                    state, det_species,
                    classifier.deterministic_rxns + classifier.coupling_rxns,
                    state,
                )
                sol = solve_ivp(
                    rhs, (t, ode_end), y0,
                    method=self.ode_method,
                    rtol=1e-6, atol=1e-8,
                )
                for i_local, i_global in enumerate(det_species):
                    state[i_global] = max(sol.y[i_local, -1], 0)

            t = ode_end

            # Fire stochastic reaction if within t_end
            if j_ssa is not None and t < t_end:
                state += self._stoich[j_ssa]
                state = np.maximum(state, 0)
                if trajectory is not None:
                    trajectory.record(t, state, partition, f"hybrid_rxn_{j_ssa}")

        # Record final state
        if trajectory is not None and (
            not trajectory.times or trajectory.times[-1] < t
        ):
            trajectory.record(t, state, partition, "final")

        return StochasticState(
            time=t,
            copy_numbers=np.round(state).astype(np.int64),
        )

    def simulate_trajectory(
        self,
        initial_state: np.ndarray,
        t_end: float,
        t_start: float = 0.0,
        max_steps: int = 10_000_000,
    ) -> HybridTrajectory:
        """Convenience: simulate and return trajectory."""
        traj = HybridTrajectory()
        self.simulate(initial_state, t_end, t_start, max_steps, trajectory=traj)
        return traj
