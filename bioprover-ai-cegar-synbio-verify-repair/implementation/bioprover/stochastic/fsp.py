"""
Finite State Projection (FSP) for solving the Chemical Master Equation.

Constructs a truncated state space, builds a sparse transition rate matrix,
and computes probability distributions via matrix exponential. Supports
adaptive state space expansion when truncation error exceeds tolerance.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


@dataclass(frozen=True)
class DiscreteState:
    """Hashable representation of a discrete molecular state."""

    values: Tuple[int, ...]

    def __getitem__(self, idx: int) -> int:
        return self.values[idx]

    def __len__(self) -> int:
        return len(self.values)

    def to_array(self) -> np.ndarray:
        return np.array(self.values, dtype=np.int64)

    @staticmethod
    def from_array(arr: np.ndarray) -> "DiscreteState":
        return DiscreteState(tuple(int(x) for x in arr))


class StateSpace:
    """Manages the truncated state space for FSP.

    Maintains a mapping between DiscreteStates and integer indices,
    supporting dynamic expansion.
    """

    def __init__(self, num_species: int):
        self.num_species = num_species
        self._state_to_idx: Dict[DiscreteState, int] = {}
        self._idx_to_state: List[DiscreteState] = []

    @property
    def size(self) -> int:
        return len(self._idx_to_state)

    def add_state(self, state: DiscreteState) -> int:
        """Add a state, returning its index. No-op if already present."""
        if state in self._state_to_idx:
            return self._state_to_idx[state]
        idx = len(self._idx_to_state)
        self._state_to_idx[state] = idx
        self._idx_to_state.append(state)
        return idx

    def get_index(self, state: DiscreteState) -> Optional[int]:
        return self._state_to_idx.get(state)

    def get_state(self, idx: int) -> DiscreteState:
        return self._idx_to_state[idx]

    def contains(self, state: DiscreteState) -> bool:
        return state in self._state_to_idx

    def all_states(self) -> List[DiscreteState]:
        return list(self._idx_to_state)

    @staticmethod
    def enumerate_hypercube(
        num_species: int, bounds: List[int]
    ) -> "StateSpace":
        """Create a state space as a hypercube [0, bounds[i]) for each species."""
        space = StateSpace(num_species)
        ranges = [range(b) for b in bounds]
        for combo in itertools.product(*ranges):
            space.add_state(DiscreteState(combo))
        return space

    @staticmethod
    def enumerate_simplex(
        num_species: int, max_total: int
    ) -> "StateSpace":
        """Create states where sum of copy numbers <= max_total."""
        space = StateSpace(num_species)

        def _recurse(depth: int, remaining: int, current: list):
            if depth == num_species:
                space.add_state(DiscreteState(tuple(current)))
                return
            for i in range(remaining + 1):
                current.append(i)
                _recurse(depth + 1, remaining - i, current)
                current.pop()

        _recurse(0, max_total, [])
        return space


@dataclass
class FSPReaction:
    """Reaction for FSP with stoichiometry and rate constant."""

    reactants: Dict[int, int]
    products: Dict[int, int]
    rate_constant: float

    @property
    def state_change(self) -> Tuple[int, ...]:
        """Not cached; compute on demand."""
        return tuple()

    def propensity(self, state: DiscreteState) -> float:
        a = self.rate_constant
        for sp, coeff in self.reactants.items():
            x = state[sp]
            if x < coeff:
                return 0.0
            for i in range(coeff):
                a *= (x - i)
        return a

    def apply(self, state: DiscreteState, num_species: int) -> DiscreteState:
        vals = list(state.values)
        for sp, coeff in self.reactants.items():
            vals[sp] -= coeff
        for sp, coeff in self.products.items():
            vals[sp] += coeff
        return DiscreteState(tuple(vals))


class SparseTransitionMatrix:
    """Builds the sparse transition rate matrix for FSP.

    The generator matrix Q has entries:
        Q[i,j] = rate of transitioning from state i to state j (i != j)
        Q[i,i] = -(sum of off-diagonal entries in row i) - sink_rate[i]

    A sink state absorbs probability that leaves the truncated space.
    """

    def __init__(
        self,
        state_space: StateSpace,
        reactions: List[FSPReaction],
        use_sink: bool = True,
    ):
        self.state_space = state_space
        self.reactions = reactions
        self.use_sink = use_sink
        self._matrix: Optional[sparse.csc_matrix] = None
        self._sink_rates: Optional[np.ndarray] = None

    def build(self) -> sparse.csc_matrix:
        """Construct the sparse generator matrix Q."""
        n = self.state_space.size
        num_species = self.state_space.num_species
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        sink_rates = np.zeros(n)

        for i in range(n):
            state_i = self.state_space.get_state(i)
            diag_sum = 0.0
            for rxn in self.reactions:
                rate = rxn.propensity(state_i)
                if rate <= 0:
                    continue
                new_state = rxn.apply(state_i, num_species)
                j = self.state_space.get_index(new_state)
                if j is not None:
                    rows.append(j)
                    cols.append(i)
                    data.append(rate)
                    diag_sum += rate
                else:
                    # Transition leaves truncated space -> sink
                    sink_rates[i] += rate
                    diag_sum += rate
            # Diagonal: negative sum ensures columns sum to zero (or <=0 with sink)
            rows.append(i)
            cols.append(i)
            data.append(-diag_sum)

        self._matrix = sparse.csc_matrix(
            (data, (rows, cols)), shape=(n, n)
        )
        self._sink_rates = sink_rates
        return self._matrix

    @property
    def matrix(self) -> sparse.csc_matrix:
        if self._matrix is None:
            self.build()
        return self._matrix

    @property
    def sink_rates(self) -> np.ndarray:
        if self._sink_rates is None:
            self.build()
        return self._sink_rates

    def rebuild(self):
        """Force rebuild after state space expansion."""
        self._matrix = None
        self._sink_rates = None
        self.build()


class MarginalDistribution:
    """Extracts marginal distributions from the joint probability vector."""

    def __init__(self, state_space: StateSpace, probability: np.ndarray):
        self.state_space = state_space
        self.probability = probability

    def marginal(self, species_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute marginal distribution for a single species.

        Returns (values, probabilities) sorted by value.
        """
        value_probs: Dict[int, float] = {}
        for i in range(self.state_space.size):
            state = self.state_space.get_state(i)
            val = state[species_idx]
            value_probs[val] = value_probs.get(val, 0.0) + self.probability[i]
        values = sorted(value_probs.keys())
        probs = np.array([value_probs[v] for v in values])
        return np.array(values), probs

    def joint_marginal(
        self, species_indices: List[int]
    ) -> Tuple[List[Tuple[int, ...]], np.ndarray]:
        """Compute joint marginal over a subset of species."""
        combo_probs: Dict[Tuple[int, ...], float] = {}
        for i in range(self.state_space.size):
            state = self.state_space.get_state(i)
            key = tuple(state[s] for s in species_indices)
            combo_probs[key] = combo_probs.get(key, 0.0) + self.probability[i]
        combos = sorted(combo_probs.keys())
        probs = np.array([combo_probs[c] for c in combos])
        return combos, probs

    def mean(self, species_idx: int) -> float:
        vals, probs = self.marginal(species_idx)
        return float(np.dot(vals, probs))

    def variance(self, species_idx: int) -> float:
        vals, probs = self.marginal(species_idx)
        mu = np.dot(vals, probs)
        return float(np.dot(vals**2, probs) - mu**2)

    def covariance(self, sp_i: int, sp_j: int) -> float:
        mu_i = self.mean(sp_i)
        mu_j = self.mean(sp_j)
        cov = 0.0
        for k in range(self.state_space.size):
            state = self.state_space.get_state(k)
            cov += (state[sp_i] - mu_i) * (state[sp_j] - mu_j) * self.probability[k]
        return cov


class FSPSolver:
    """Finite State Projection solver for the Chemical Master Equation.

    Parameters:
        reactions: List of FSPReaction objects.
        num_species: Number of molecular species.
        initial_state: Initial copy numbers as tuple or array.
        state_bounds: Upper bounds for initial state space enumeration.
        fsp_tol: Tolerance for truncation error (probability lost to sink).
        max_expansions: Maximum number of state space expansions.
        expansion_factor: Factor by which to expand bounds on each expansion.
    """

    def __init__(
        self,
        reactions: List[FSPReaction],
        num_species: int,
        initial_state: Tuple[int, ...],
        state_bounds: Optional[List[int]] = None,
        fsp_tol: float = 1e-4,
        max_expansions: int = 10,
        expansion_factor: float = 1.5,
    ):
        self.reactions = reactions
        self.num_species = num_species
        self.initial_state = DiscreteState(initial_state)
        self.fsp_tol = fsp_tol
        self.max_expansions = max_expansions
        self.expansion_factor = expansion_factor

        if state_bounds is None:
            state_bounds = [max(initial_state) + 10] * num_species
        self.state_bounds = list(state_bounds)

        self.state_space = StateSpace.enumerate_hypercube(num_species, state_bounds)
        self.trans_matrix = SparseTransitionMatrix(
            self.state_space, reactions, use_sink=True
        )

    def _initial_probability(self) -> np.ndarray:
        """Create initial probability vector (delta at initial_state)."""
        p = np.zeros(self.state_space.size)
        idx = self.state_space.get_index(self.initial_state)
        if idx is None:
            raise ValueError(
                f"Initial state {self.initial_state} not in state space"
            )
        p[idx] = 1.0
        return p

    def _expand_state_space(self):
        """Expand state space bounds by expansion_factor."""
        self.state_bounds = [
            int(b * self.expansion_factor) + 1 for b in self.state_bounds
        ]
        self.state_space = StateSpace.enumerate_hypercube(
            self.num_species, self.state_bounds
        )
        self.trans_matrix = SparseTransitionMatrix(
            self.state_space, self.reactions, use_sink=True
        )

    def _embed_probability(
        self, old_space: StateSpace, old_prob: np.ndarray
    ) -> np.ndarray:
        """Map old probability vector into the expanded state space."""
        new_prob = np.zeros(self.state_space.size)
        for i in range(old_space.size):
            state = old_space.get_state(i)
            new_idx = self.state_space.get_index(state)
            if new_idx is not None:
                new_prob[new_idx] = old_prob[i]
        return new_prob

    def solve(self, t_final: float) -> Tuple[np.ndarray, float]:
        """Solve the CME up to t_final with adaptive state space expansion.

        Returns:
            (probability_vector, truncation_error)
        """
        Q = self.trans_matrix.matrix
        p0 = self._initial_probability()

        for expansion in range(self.max_expansions + 1):
            Q = self.trans_matrix.matrix
            p = expm_multiply(Q.T, p0, start=0, stop=t_final, num=2)[-1]
            p = np.maximum(p, 0.0)
            total_prob = p.sum()
            truncation_error = 1.0 - total_prob

            if truncation_error <= self.fsp_tol:
                return p, truncation_error

            # Expand
            old_space = self.state_space
            self._expand_state_space()
            p0 = self._embed_probability(old_space, p)

        return p, truncation_error

    def solve_at_times(
        self, times: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the CME at multiple time points.

        Returns:
            (probability_matrix [len(times) x num_states], errors)
        """
        Q = self.trans_matrix.matrix
        p0 = self._initial_probability()

        all_probs = np.zeros((len(times), self.state_space.size))
        errors = np.zeros(len(times))

        # Use expm_multiply for the entire time span
        results = expm_multiply(
            Q.T, p0, start=0, stop=times[-1], num=len(times) + 1
        )
        # results[0] is at t=0, results[-1] is at t=times[-1]
        for i in range(len(times)):
            p = np.maximum(results[i + 1], 0.0)
            all_probs[i] = p
            errors[i] = 1.0 - p.sum()

        return all_probs, errors

    def steady_state(
        self, t_long: float = 1000.0, tol: float = 1e-8
    ) -> np.ndarray:
        """Approximate steady-state by propagating to large time.

        Alternatively, solve Q^T * p_ss = 0 with normalization.
        """
        Q = self.trans_matrix.matrix
        n = self.state_space.size
        # Try null-space approach first
        try:
            Qt = Q.T.toarray()
            # Add normalization constraint
            A = np.vstack([Qt, np.ones(n)])
            b = np.zeros(n + 1)
            b[-1] = 1.0
            p_ss, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            p_ss = np.maximum(p_ss, 0.0)
            p_ss /= p_ss.sum()
            if np.allclose(Qt @ p_ss, 0, atol=tol):
                return p_ss
        except Exception:
            pass
        # Fallback: long-time propagation
        p0 = self._initial_probability()
        p_ss = expm_multiply(Q.T, p0, start=0, stop=t_long, num=2)[-1]
        p_ss = np.maximum(p_ss, 0.0)
        p_ss /= p_ss.sum() if p_ss.sum() > 0 else 1.0
        return p_ss

    def get_distribution(self, probability: np.ndarray) -> MarginalDistribution:
        """Wrap probability vector as MarginalDistribution."""
        return MarginalDistribution(self.state_space, probability)

    def error_bound(self, probability: np.ndarray) -> float:
        """Compute truncation error: 1 - sum(probability)."""
        return 1.0 - np.sum(probability)
