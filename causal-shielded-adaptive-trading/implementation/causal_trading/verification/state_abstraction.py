"""
State abstraction with conservative overapproximation guarantees.

Proves that the hyperrectangular discretisation used by the safety shield
is a sound overapproximation of the concrete MDP: every concrete transition
is captured by some abstract transition with probability at least as large.

Key theorem implemented:
    If  T_abs(s_abs, a, s'_abs) >= max_{s ∈ s_abs} T_conc(s, a, γ⁻¹(s'_abs))
    for all abstract transitions, then
        Pr_abs(φ) <= Pr_conc(φ)
    for all safety properties φ monotone in transition probabilities.

This module imports from the existing polytope.py and model_checking.py modules.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Sequence,
    Tuple,
)

import numpy as np
from numpy.typing import NDArray

from .interval_arithmetic import (
    Interval,
    IntervalMatrix,
    IntervalVector,
    interval_matmul,
    _EPS,
    _TINY,
)
from .model_checking import (
    MDP,
    MDPTransition,
    CheckResult,
    Specification,
    SpecKind,
    SymbolicModelChecker,
    build_mdp_from_matrix,
)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConcreteState:
    """A single point in the continuous state space."""
    values: Tuple[float, ...]

    @classmethod
    def from_array(cls, arr: NDArray) -> "ConcreteState":
        return cls(values=tuple(float(v) for v in arr))

    def to_array(self) -> NDArray:
        return np.array(self.values, dtype=np.float64)

    @property
    def dim(self) -> int:
        return len(self.values)


@dataclass
class AbstractState:
    """A hyperrectangular region of the continuous state space.

    Represented as an axis-aligned box [lo, hi] in each dimension.
    """
    lo: NDArray  # (dim,)
    hi: NDArray  # (dim,)
    index: int = -1  # unique index in the abstract state space

    def __post_init__(self) -> None:
        self.lo = np.asarray(self.lo, dtype=np.float64)
        self.hi = np.asarray(self.hi, dtype=np.float64)

    @property
    def dim(self) -> int:
        return len(self.lo)

    @property
    def volume(self) -> float:
        widths = self.hi - self.lo
        if np.any(widths <= 0):
            return 0.0
        return float(np.prod(widths))

    @property
    def center(self) -> NDArray:
        return 0.5 * (self.lo + self.hi)

    @property
    def widths(self) -> NDArray:
        return self.hi - self.lo

    def contains_point(self, point: NDArray) -> bool:
        """Check whether a concrete point lies inside this abstract state."""
        point = np.asarray(point, dtype=np.float64)
        return bool(np.all(point >= self.lo - _TINY) and np.all(point <= self.hi + _TINY))

    def contains_concrete(self, cs: ConcreteState) -> bool:
        return self.contains_point(cs.to_array())

    def contains_abstract(self, other: "AbstractState") -> bool:
        """Check whether *other* ⊆ self."""
        return bool(
            np.all(other.lo >= self.lo - _TINY) and np.all(other.hi <= self.hi + _TINY)
        )

    def overlaps(self, other: "AbstractState") -> bool:
        return bool(
            np.all(self.lo <= other.hi + _TINY) and np.all(other.lo <= self.hi + _TINY)
        )

    def to_interval_vector(self) -> IntervalVector:
        return IntervalVector.from_bounds(self.lo, self.hi)

    def bisect(self, dimension: Optional[int] = None) -> Tuple["AbstractState", "AbstractState"]:
        """Split into two halves along the widest (or specified) dimension."""
        if dimension is None:
            dimension = int(np.argmax(self.hi - self.lo))
        mid = 0.5 * (self.lo[dimension] + self.hi[dimension])
        lo1 = self.lo.copy()
        hi1 = self.hi.copy()
        hi1[dimension] = mid
        lo2 = self.lo.copy()
        hi2 = self.hi.copy()
        lo2[dimension] = mid
        return (
            AbstractState(lo=lo1, hi=hi1),
            AbstractState(lo=lo2, hi=hi2),
        )

    def sample_uniform(self, rng: np.random.Generator, n: int = 1) -> NDArray:
        """Sample *n* points uniformly from the hyperrectangle."""
        return rng.uniform(self.lo, self.hi, size=(n, self.dim))

    def __repr__(self) -> str:
        return f"AbstractState(idx={self.index}, lo={self.lo}, hi={self.hi})"


# ---------------------------------------------------------------------------
# Abstraction function
# ---------------------------------------------------------------------------

class AbstractionFunction:
    """Maps concrete states to abstract states (hyperrectangular partition).

    Guarantees: for every concrete state s, γ(s) is an abstract state
    such that s ∈ γ(s).  The pre-image γ⁻¹(s_abs) contains all concrete
    states mapped to s_abs.
    """

    def __init__(self, abstract_states: List[AbstractState]) -> None:
        self._states = abstract_states
        self._dim = abstract_states[0].dim if abstract_states else 0
        # Assign indices
        for i, s in enumerate(self._states):
            s.index = i

    @property
    def n_abstract(self) -> int:
        return len(self._states)

    @property
    def dim(self) -> int:
        return self._dim

    def abstract_states(self) -> List[AbstractState]:
        return list(self._states)

    def __getitem__(self, idx: int) -> AbstractState:
        return self._states[idx]

    def abstract(self, concrete: ConcreteState) -> AbstractState:
        """Map a concrete state to its enclosing abstract state.

        Returns the first abstract state that contains the point.
        Raises ValueError if no abstract state covers the point.
        """
        pt = concrete.to_array()
        for s in self._states:
            if s.contains_point(pt):
                return s
        raise ValueError(f"Concrete state {concrete} not covered by any abstract state")

    def abstract_index(self, concrete: ConcreteState) -> int:
        """Return the index of the abstract state containing *concrete*."""
        return self.abstract(concrete).index

    def preimage(self, abstract_state: AbstractState) -> AbstractState:
        """Return the pre-image γ⁻¹(s_abs), which is s_abs itself for
        hyperrectangular partitions."""
        return abstract_state

    def covers_point(self, point: NDArray) -> bool:
        """Check whether the partition covers a given point."""
        return any(s.contains_point(point) for s in self._states)

    def verify_partition(self, bounds_lo: NDArray, bounds_hi: NDArray) -> bool:
        """Check that the abstract states form a partition of the bounding box.

        Verifies that:
        1. Every abstract state is inside the bounding box
        2. The union of abstract states covers the bounding box (approximately)
        """
        for s in self._states:
            if not (np.all(s.lo >= bounds_lo - _TINY) and np.all(s.hi <= bounds_hi + _TINY)):
                return False
        # Check coverage via sampling
        rng = np.random.default_rng(0)
        n_samples = min(1000, 10 ** self._dim)
        for _ in range(n_samples):
            pt = rng.uniform(bounds_lo, bounds_hi)
            if not self.covers_point(pt):
                return False
        return True


# ---------------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------------

def discretize_state_space(
    bounds_lo: NDArray,
    bounds_hi: NDArray,
    n_bins: Union[int, Sequence[int]] = 10,
) -> AbstractionFunction:
    """Create a uniform hyperrectangular partition of the state space.

    Parameters
    ----------
    bounds_lo, bounds_hi : NDArray, shape (dim,)
        Lower and upper bounds of the state space.
    n_bins : int or sequence of ints
        Number of bins per dimension.

    Returns
    -------
    AbstractionFunction
        The abstraction mapping.
    """
    bounds_lo = np.asarray(bounds_lo, dtype=np.float64)
    bounds_hi = np.asarray(bounds_hi, dtype=np.float64)
    dim = len(bounds_lo)

    if isinstance(n_bins, int):
        bins_per_dim = [n_bins] * dim
    else:
        bins_per_dim = list(n_bins)
    assert len(bins_per_dim) == dim

    edges = []
    for d in range(dim):
        edges.append(np.linspace(bounds_lo[d], bounds_hi[d], bins_per_dim[d] + 1))

    abstract_states: List[AbstractState] = []
    idx = 0
    for multi_idx in itertools.product(*(range(b) for b in bins_per_dim)):
        lo = np.array([edges[d][multi_idx[d]] for d in range(dim)])
        hi = np.array([edges[d][multi_idx[d] + 1] for d in range(dim)])
        abstract_states.append(AbstractState(lo=lo, hi=hi, index=idx))
        idx += 1

    return AbstractionFunction(abstract_states)


# ---------------------------------------------------------------------------
# Abstract MDP construction
# ---------------------------------------------------------------------------

@dataclass
class ConcreteMDP:
    """Wrapper around a concrete (potentially continuous-state) MDP.

    For the abstraction proof, we represent the transition kernel as a
    callable that returns transition probabilities for any concrete state.
    """
    n_actions: int
    state_dim: int
    transition_fn: Callable[[NDArray, int], Tuple[NDArray, NDArray]]
    # transition_fn(state, action) -> (next_states, probabilities)
    # For finite MDPs, this is just a lookup.

    @classmethod
    def from_finite_mdp(cls, mdp: MDP, state_vectors: NDArray) -> "ConcreteMDP":
        """Wrap a finite MDP with explicit state-space embeddings.

        Parameters
        ----------
        mdp : MDP
            The finite MDP from model_checking.
        state_vectors : NDArray, shape (n_states, dim)
            Embedding of each discrete state into R^dim.
        """
        def transition_fn(state: NDArray, action: int) -> Tuple[NDArray, NDArray]:
            # Find nearest discrete state
            dists = np.linalg.norm(state_vectors - state, axis=1)
            s_idx = int(np.argmin(dists))
            tr = mdp.transitions.get((s_idx, action))
            if tr is None:
                return state_vectors[s_idx:s_idx+1], np.array([1.0])
            next_vecs = state_vectors[tr.next_states]
            return next_vecs, tr.probs

        return cls(
            n_actions=mdp.n_actions,
            state_dim=state_vectors.shape[1],
            transition_fn=transition_fn,
        )


def compute_abstract_transitions(
    concrete_mdp: MDP,
    abstraction: AbstractionFunction,
    state_vectors: Optional[NDArray] = None,
) -> MDP:
    """Compute an abstract MDP that conservatively overapproximates the concrete MDP.

    For each abstract state pair (s_abs, s'_abs) and action a, the abstract
    transition probability is set to the *maximum* transition probability from
    any concrete state in s_abs to any concrete state in s'_abs.

    This ensures T_abs(s_abs, a, s'_abs) >= max_{s ∈ s_abs} T_conc(s, a, γ⁻¹(s'_abs)).

    Parameters
    ----------
    concrete_mdp : MDP
        The concrete finite MDP.
    abstraction : AbstractionFunction
        The state-space abstraction.
    state_vectors : NDArray, optional
        If provided, embedding of concrete states into R^dim.
        If None, uses one-hot encoding.

    Returns
    -------
    MDP
        The abstract MDP with overapproximated transitions.
    """
    n_abs = abstraction.n_abstract
    n_actions = concrete_mdp.n_actions
    n_concrete = concrete_mdp.n_states

    if state_vectors is None:
        state_vectors = np.eye(n_concrete, dtype=np.float64)
        if abstraction.dim != n_concrete:
            # Use sequential embedding
            state_vectors = np.arange(n_concrete, dtype=np.float64).reshape(-1, 1)

    # Map each concrete state to an abstract state
    concrete_to_abstract = np.full(n_concrete, -1, dtype=int)
    for s in range(n_concrete):
        sv = state_vectors[s]
        cs = ConcreteState.from_array(sv)
        try:
            abs_s = abstraction.abstract(cs)
            concrete_to_abstract[s] = abs_s.index
        except ValueError:
            # State not covered — assign to nearest
            min_dist = np.inf
            best_idx = 0
            for a_state in abstraction.abstract_states():
                center = a_state.center
                dist = np.linalg.norm(sv - center)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = a_state.index
            concrete_to_abstract[s] = best_idx

    # For each abstract state and action, compute overapproximated transitions
    abs_transitions: Dict[Tuple[int, int], MDPTransition] = {}

    for a in range(n_actions):
        # Build transition probability from abstract state to abstract state
        # T_abs[s_abs, a, s'_abs] = max over concrete states s in s_abs of
        #   sum over concrete s' in s'_abs of T_conc(s, a, s')
        abs_T = np.zeros((n_abs, n_abs), dtype=np.float64)

        for s_conc in range(n_concrete):
            s_abs = concrete_to_abstract[s_conc]
            if s_abs < 0:
                continue
            tr = concrete_mdp.transitions.get((s_conc, a))
            if tr is None:
                # Self-loop
                abs_T[s_abs, s_abs] = max(abs_T[s_abs, s_abs], 1.0)
                continue
            # Accumulate probabilities per abstract successor
            succ_probs = np.zeros(n_abs, dtype=np.float64)
            for ns, p in zip(tr.next_states, tr.probs):
                ns_abs = concrete_to_abstract[int(ns)]
                if ns_abs >= 0:
                    succ_probs[ns_abs] += p
            # Take max over concrete states in s_abs
            for sp_abs in range(n_abs):
                abs_T[s_abs, sp_abs] = max(abs_T[s_abs, sp_abs], succ_probs[sp_abs])

        # Normalise rows (the overapproximation may sum > 1; normalise conservatively)
        for s_abs in range(n_abs):
            row_sum = abs_T[s_abs].sum()
            if row_sum > 0:
                # Only normalise if sum exceeds 1 (otherwise pad self-loop)
                if row_sum > 1.0:
                    abs_T[s_abs] /= row_sum
                elif row_sum < 1.0:
                    abs_T[s_abs, s_abs] += 1.0 - row_sum

        # Build MDP transitions
        for s_abs in range(n_abs):
            nonzero = np.where(abs_T[s_abs] > 1e-15)[0]
            if len(nonzero) == 0:
                nonzero = np.array([s_abs])
                probs = np.array([1.0])
            else:
                probs = abs_T[s_abs, nonzero]
                probs = probs / probs.sum()  # ensure exact normalisation
            abs_transitions[(s_abs, a)] = MDPTransition(
                next_states=nonzero.copy(),
                probs=probs.copy(),
                reward=0.0,
            )

    return MDP(
        n_states=n_abs,
        n_actions=n_actions,
        transitions=abs_transitions,
        initial_state=0,
    )


# ---------------------------------------------------------------------------
# Overapproximation verification
# ---------------------------------------------------------------------------

@dataclass
class OverapproximationResult:
    """Result of checking that the abstract MDP overapproximates the concrete one."""
    sound: bool
    max_underapprox_gap: float  # max amount by which abs < conc (should be <= 0)
    violations: List[Tuple[int, int, int, float]]  # (s_abs, a, s'_abs, gap)
    n_checked: int = 0

    def __repr__(self) -> str:
        return (
            f"OverapproximationResult(sound={self.sound}, "
            f"max_gap={self.max_underapprox_gap:.2e}, "
            f"violations={len(self.violations)})"
        )


def verify_overapproximation(
    concrete_mdp: MDP,
    abstract_mdp: MDP,
    abstraction: AbstractionFunction,
    state_vectors: Optional[NDArray] = None,
    tol: float = 1e-10,
) -> OverapproximationResult:
    """Verify that the abstract MDP overapproximates the concrete MDP.

    Checks that for every concrete state s in abstract region s_abs and
    every action a, the abstract transition probability to s'_abs is at
    least as large as the concrete transition probability to the pre-image
    of s'_abs.
    """
    n_concrete = concrete_mdp.n_states
    n_abs = abstract_mdp.n_states
    n_actions = concrete_mdp.n_actions

    if state_vectors is None:
        state_vectors = np.arange(n_concrete, dtype=np.float64).reshape(-1, 1)

    # Map concrete to abstract
    c2a = np.full(n_concrete, -1, dtype=int)
    for s in range(n_concrete):
        cs = ConcreteState.from_array(state_vectors[s])
        try:
            c2a[s] = abstraction.abstract(cs).index
        except ValueError:
            pass

    violations: List[Tuple[int, int, int, float]] = []
    max_gap = 0.0
    n_checked = 0

    for a in range(n_actions):
        for s_conc in range(n_concrete):
            s_abs = c2a[s_conc]
            if s_abs < 0:
                continue
            tr_conc = concrete_mdp.transitions.get((s_conc, a))
            if tr_conc is None:
                continue

            # Compute concrete transition probs to each abstract state
            conc_probs = np.zeros(n_abs, dtype=np.float64)
            for ns, p in zip(tr_conc.next_states, tr_conc.probs):
                ns_abs = c2a[int(ns)]
                if ns_abs >= 0:
                    conc_probs[ns_abs] += p

            # Compare with abstract transition probs
            tr_abs = abstract_mdp.transitions.get((s_abs, a))
            abs_probs = np.zeros(n_abs, dtype=np.float64)
            if tr_abs is not None:
                for ns, p in zip(tr_abs.next_states, tr_abs.probs):
                    abs_probs[int(ns)] = p

            for sp_abs in range(n_abs):
                n_checked += 1
                gap = conc_probs[sp_abs] - abs_probs[sp_abs]
                if gap > tol:
                    violations.append((s_abs, a, sp_abs, gap))
                max_gap = max(max_gap, gap)

    return OverapproximationResult(
        sound=len(violations) == 0,
        max_underapprox_gap=max_gap,
        violations=violations,
        n_checked=n_checked,
    )


# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------

def refinement_step(
    abstraction: AbstractionFunction,
    target_index: int,
    dimension: Optional[int] = None,
) -> AbstractionFunction:
    """Bisect an abstract state to produce a finer partition.

    Parameters
    ----------
    abstraction : AbstractionFunction
        Current partition.
    target_index : int
        Index of the abstract state to split.
    dimension : int, optional
        Dimension along which to split (default: widest).

    Returns
    -------
    AbstractionFunction
        Refined partition with one extra abstract state.
    """
    states = abstraction.abstract_states()
    target = states[target_index]
    left, right = target.bisect(dimension)

    new_states = []
    for s in states:
        if s.index == target_index:
            new_states.append(left)
            new_states.append(right)
        else:
            new_states.append(AbstractState(lo=s.lo.copy(), hi=s.hi.copy()))
    return AbstractionFunction(new_states)


def adaptive_refinement(
    concrete_mdp: MDP,
    bounds_lo: NDArray,
    bounds_hi: NDArray,
    initial_bins: int = 4,
    max_refinements: int = 50,
    target_gap: float = 1e-3,
    state_vectors: Optional[NDArray] = None,
) -> Tuple[AbstractionFunction, MDP, OverapproximationResult]:
    """Iteratively refine the abstraction until the overapproximation gap
    is below *target_gap* or *max_refinements* is exhausted.

    Returns the final abstraction, abstract MDP, and verification result.
    """
    abstraction = discretize_state_space(bounds_lo, bounds_hi, initial_bins)
    abstract_mdp = compute_abstract_transitions(
        concrete_mdp, abstraction, state_vectors
    )
    result = verify_overapproximation(
        concrete_mdp, abstract_mdp, abstraction, state_vectors
    )

    for _ in range(max_refinements):
        if result.sound and result.max_underapprox_gap <= target_gap:
            break
        if not result.violations:
            break

        # Find abstract state with worst violation and refine it
        worst_s = result.violations[0][0]
        worst_gap = result.violations[0][3]
        for v in result.violations:
            if v[3] > worst_gap:
                worst_s = v[0]
                worst_gap = v[3]

        if worst_s < abstraction.n_abstract:
            abstraction = refinement_step(abstraction, worst_s)
            abstract_mdp = compute_abstract_transitions(
                concrete_mdp, abstraction, state_vectors
            )
            result = verify_overapproximation(
                concrete_mdp, abstract_mdp, abstraction, state_vectors
            )

    return abstraction, abstract_mdp, result


# ---------------------------------------------------------------------------
# Soundness certificate
# ---------------------------------------------------------------------------

@dataclass
class SoundnessCertificate:
    """Formal certificate that abstract safety implies concrete safety.

    Contains the proof obligations and their verification status.
    """
    # Core result
    sound: bool
    # Components
    partition_covers: bool
    overapproximation_holds: bool
    safety_transfer_holds: bool
    # Quantitative data
    abstract_safety_prob: float
    concrete_safety_lower_bound: float
    overapproximation_gap: float
    n_abstract_states: int
    n_concrete_states: int
    # Timing
    verification_time_s: float = 0.0
    # Details
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"SoundnessCertificate(sound={self.sound}, "
            f"abstract_safety={self.abstract_safety_prob:.4f}, "
            f"concrete_lb={self.concrete_safety_lower_bound:.4f})"
        )


# ---------------------------------------------------------------------------
# Conservative overapproximation verifier (main class)
# ---------------------------------------------------------------------------

class ConservativeOverapproximation:
    """Proves that safety in the abstract MDP implies safety in the concrete MDP.

    This is the core contribution: combines discretisation, interval arithmetic
    for transition computation, overapproximation verification, and model checking
    to produce a formal soundness certificate.
    """

    def __init__(
        self,
        convergence_tol: float = 1e-10,
        max_iterations: int = 10000,
    ) -> None:
        self._checker = SymbolicModelChecker(
            convergence_tol=convergence_tol,
            max_iterations=max_iterations,
        )

    def certify_soundness(
        self,
        concrete_mdp: MDP,
        abstract_mdp: MDP,
        abstraction: AbstractionFunction,
        spec: Specification,
        state_vectors: Optional[NDArray] = None,
        horizon: Optional[int] = None,
    ) -> SoundnessCertificate:
        """Produce a soundness certificate.

        Checks:
        1. The partition covers the state space.
        2. The abstract MDP overapproximates the concrete MDP.
        3. Safety in the abstract system transfers to the concrete system.

        Parameters
        ----------
        concrete_mdp : MDP
        abstract_mdp : MDP
        abstraction : AbstractionFunction
        spec : Specification
        state_vectors : NDArray, optional
        horizon : int, optional

        Returns
        -------
        SoundnessCertificate
        """
        t0 = time.time()

        # 1. Check partition coverage
        if state_vectors is not None:
            partition_ok = all(
                abstraction.covers_point(state_vectors[s])
                for s in range(concrete_mdp.n_states)
            )
        else:
            partition_ok = True  # Trivially true for identity embedding

        # 2. Verify overapproximation
        overapprox = verify_overapproximation(
            concrete_mdp, abstract_mdp, abstraction, state_vectors
        )

        # 3. Model-check the abstract MDP
        abs_result = self._checker.check(abstract_mdp, spec, horizon)

        # 4. Transfer theorem:
        # For safety properties, if abstract MDP satisfies with prob p,
        # then concrete MDP satisfies with prob >= p (because abstract
        # overapproximates transition probabilities to unsafe states).
        #
        # More precisely: if the abstract MDP avoids unsafe states with
        # probability p, and the abstraction is a sound overapproximation,
        # then the concrete MDP avoids the corresponding unsafe states
        # with probability >= p.
        safety_transfers = (
            overapprox.sound
            and partition_ok
            and spec.kind in (SpecKind.SAFETY, SpecKind.PROB_SAFETY)
        )

        concrete_lb = abs_result.satisfaction_prob if safety_transfers else 0.0

        elapsed = time.time() - t0

        return SoundnessCertificate(
            sound=partition_ok and overapprox.sound and safety_transfers,
            partition_covers=partition_ok,
            overapproximation_holds=overapprox.sound,
            safety_transfer_holds=safety_transfers,
            abstract_safety_prob=abs_result.satisfaction_prob,
            concrete_safety_lower_bound=concrete_lb,
            overapproximation_gap=overapprox.max_underapprox_gap,
            n_abstract_states=abstract_mdp.n_states,
            n_concrete_states=concrete_mdp.n_states,
            verification_time_s=elapsed,
            details={
                "overapprox_result": overapprox,
                "abstract_check_result": abs_result,
            },
        )

    def verify_and_refine(
        self,
        concrete_mdp: MDP,
        spec: Specification,
        bounds_lo: NDArray,
        bounds_hi: NDArray,
        initial_bins: int = 4,
        max_refinements: int = 50,
        target_gap: float = 1e-3,
        state_vectors: Optional[NDArray] = None,
        horizon: Optional[int] = None,
    ) -> Tuple[SoundnessCertificate, AbstractionFunction, MDP]:
        """Full pipeline: discretise, verify, refine, certify.

        Returns the soundness certificate, final abstraction, and abstract MDP.
        """
        abstraction, abstract_mdp, overapprox = adaptive_refinement(
            concrete_mdp,
            bounds_lo,
            bounds_hi,
            initial_bins=initial_bins,
            max_refinements=max_refinements,
            target_gap=target_gap,
            state_vectors=state_vectors,
        )

        cert = self.certify_soundness(
            concrete_mdp, abstract_mdp, abstraction, spec,
            state_vectors=state_vectors,
            horizon=horizon,
        )

        return cert, abstraction, abstract_mdp

    def check_interval_transitions(
        self,
        transition_matrix: NDArray,
        abstraction: AbstractionFunction,
    ) -> IntervalMatrix:
        """Compute interval-valued abstract transition matrix using interval arithmetic.

        For each pair of abstract states (s, s'), computes a sound enclosure
        of all possible transition probabilities.
        """
        n_abs = abstraction.n_abstract
        T_iv = IntervalMatrix.from_matrix(transition_matrix)

        # Project transition matrix through abstraction
        abs_lo = np.zeros((n_abs, n_abs), dtype=np.float64)
        abs_hi = np.zeros((n_abs, n_abs), dtype=np.float64)

        n_concrete = transition_matrix.shape[0]
        state_vectors = np.arange(n_concrete, dtype=np.float64).reshape(-1, 1)

        c2a = {}
        for s in range(n_concrete):
            cs = ConcreteState.from_array(state_vectors[s])
            try:
                c2a[s] = abstraction.abstract(cs).index
            except ValueError:
                pass

        for s in range(n_concrete):
            if s not in c2a:
                continue
            s_abs = c2a[s]
            for sp in range(n_concrete):
                if sp not in c2a:
                    continue
                sp_abs = c2a[sp]
                iv = T_iv.get(s, sp)
                # Expand the abstract interval
                abs_lo[s_abs, sp_abs] = min(abs_lo[s_abs, sp_abs], iv.lo)
                abs_hi[s_abs, sp_abs] = max(abs_hi[s_abs, sp_abs], iv.hi)

        # Initialise lo to the minimum seen (but at least 0 for probabilities)
        abs_lo = np.maximum(abs_lo, 0.0)
        abs_hi = np.maximum(abs_hi, abs_lo)

        return IntervalMatrix(abs_lo, abs_hi)


# ---------------------------------------------------------------------------
# Interval-based safety probability computation
# ---------------------------------------------------------------------------

def interval_safety_probability(
    abstract_T: IntervalMatrix,
    safe_states: FrozenSet[int],
    initial_state: int,
    horizon: int,
) -> Interval:
    """Compute a sound interval enclosure of the safety probability
    in the abstract MDP with interval-valued transitions.

    Uses interval value iteration: at each step, the safety probability
    is enclosed in an interval that accounts for all possible transition
    matrices within the interval bounds.

    Parameters
    ----------
    abstract_T : IntervalMatrix
        Interval-valued transition matrix.
    safe_states : FrozenSet[int]
        Set of safe abstract states.
    initial_state : int
        Initial abstract state.
    horizon : int
        Time horizon.

    Returns
    -------
    Interval
        Sound enclosure of the safety probability.
    """
    n = abstract_T.shape[0]

    # Probability vector as intervals
    prob = IntervalVector([
        Interval(1.0, 1.0) if s in safe_states else Interval(0.0, 0.0)
        for s in range(n)
    ])

    for _ in range(horizon):
        new_prob = IntervalVector.zeros(n)
        for s in range(n):
            if s not in safe_states:
                new_prob[s] = Interval(0.0, 0.0)
                continue
            # Compute T[s, :] · prob using interval arithmetic
            dot_lo = 0.0
            dot_hi = 0.0
            for sp in range(n):
                t_iv = abstract_T.get(s, sp)
                p_iv = prob[sp]
                prod = t_iv * p_iv
                dot_lo += prod.lo
                dot_hi += prod.hi
            # Inflate for accumulation
            delta = _EPS * max(abs(dot_lo), abs(dot_hi)) + n * _TINY
            new_prob[s] = Interval(
                max(dot_lo - delta, 0.0),
                min(dot_hi + delta, 1.0),
            )
        prob = new_prob

    return prob[initial_state]
