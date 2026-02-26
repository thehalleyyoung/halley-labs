"""
Equivalence oracle for L*-style coalgebraic learning.

Checks whether a hypothesis coalgebra is behaviourally equivalent to the
concrete system using bounded conformance testing.  When a difference is
found the oracle returns a *counterexample*: an action sequence on which
the hypothesis and the concrete system disagree.

Testing strategies
------------------
* **Systematic (breadth-first)**: enumerate all action sequences up to
  length *k* and compare F-behaviours.
* **Random walk**: sample random sequences up to configurable length.
* **W-method**: use a characterisation set *W* (from the hypothesis) to
  construct a more thorough test suite.
* **Adaptive depth**: start with small *k* and increase when no
  counterexample is found.
"""

from __future__ import annotations

import itertools
import logging
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .observation_table import AccessSequence, Observation, Suffix
from .membership_oracle import MembershipOracle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Counterexample descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Counterexample:
    """A distinguishing sequence where hypothesis ≠ concrete system."""

    sequence: Tuple[str, ...]
    hypothesis_observation: Optional[Observation]
    concrete_observation: Optional[Observation]
    discovery_method: str = "unknown"

    @property
    def length(self) -> int:
        return len(self.sequence)

    def __repr__(self) -> str:
        seq_str = ".".join(self.sequence) if self.sequence else "ε"
        return f"Counterexample({seq_str}, method={self.discovery_method})"


# ---------------------------------------------------------------------------
# Equivalence oracle statistics
# ---------------------------------------------------------------------------

@dataclass
class EquivalenceStats:
    total_rounds: int = 0
    total_tests: int = 0
    counterexamples_found: int = 0
    max_depth_tested: int = 0
    total_time_seconds: float = 0.0
    systematic_tests: int = 0
    random_tests: int = 0
    w_method_tests: int = 0

    def summary(self) -> str:
        return (
            f"EQ stats: {self.total_rounds} rounds, "
            f"{self.total_tests} tests, "
            f"{self.counterexamples_found} counterexamples, "
            f"max depth {self.max_depth_tested}, "
            f"{self.total_time_seconds:.2f}s"
        )


# ---------------------------------------------------------------------------
# Hypothesis interface (protocol)
# ---------------------------------------------------------------------------

class HypothesisInterface:
    """Protocol for hypothesis coalgebras used by the equivalence oracle."""

    def initial_state(self) -> str:
        raise NotImplementedError

    def transition(self, state: str, action: str) -> Optional[str]:
        raise NotImplementedError

    def observation_at(self, state: str) -> Observation:
        raise NotImplementedError

    def states(self) -> Set[str]:
        raise NotImplementedError

    def actions(self) -> Set[str]:
        raise NotImplementedError

    def state_reached(self, sequence: Tuple[str, ...]) -> Optional[str]:
        """Return the state reached by executing *sequence* from the
        initial state, or None if some action is undefined."""
        s = self.initial_state()
        for act in sequence:
            s = self.transition(s, act)
            if s is None:
                return None
        return s


# ---------------------------------------------------------------------------
# Equivalence oracle
# ---------------------------------------------------------------------------

class EquivalenceOracle:
    """Bounded conformance testing for equivalence queries.

    Parameters
    ----------
    membership_oracle : MembershipOracle
        Used to query the concrete system.
    actions : set of str
        The action alphabet.
    initial_depth : int
        Starting depth for conformance testing.
    max_depth : int
        Maximum depth for systematic testing.
    random_walks : int
        Number of random walks per round.
    max_random_length : int
        Maximum length of each random walk.
    adaptive : bool
        If True, increase depth when no counterexample is found.
    depth_increment : int
        How much to increase depth in adaptive mode.
    timeout : float
        Maximum seconds per equivalence round.
    seed : int or None
        Random seed for reproducibility.
    confidence : float
        Target statistical confidence for random testing (0–1).
    """

    def __init__(
        self,
        membership_oracle: MembershipOracle,
        actions: Set[str],
        *,
        initial_depth: int = 3,
        max_depth: int = 12,
        random_walks: int = 200,
        max_random_length: int = 20,
        adaptive: bool = True,
        depth_increment: int = 1,
        timeout: float = 60.0,
        seed: Optional[int] = None,
        confidence: float = 0.95,
    ) -> None:
        self._mq = membership_oracle
        self._actions = sorted(actions)
        self._action_list = list(self._actions)
        self._current_depth = initial_depth
        self._max_depth = max_depth
        self._random_walks = random_walks
        self._max_random_length = max_random_length
        self._adaptive = adaptive
        self._depth_increment = depth_increment
        self._timeout = timeout
        self._confidence = confidence
        self._rng = random.Random(seed)
        self._stats = EquivalenceStats()

    # -- public interface ---------------------------------------------------

    def check_equivalence(
        self,
        hypothesis: HypothesisInterface,
    ) -> Optional[Counterexample]:
        """Pose an equivalence query.

        Returns a counterexample if one is found, or ``None`` if the
        hypothesis passes all tests at the current depth.
        """
        self._stats.total_rounds += 1
        t0 = time.monotonic()

        # Phase 1: systematic BFS up to current depth
        cex = self._systematic_test(hypothesis, t0)
        if cex is not None:
            self._stats.counterexamples_found += 1
            self._stats.total_time_seconds += time.monotonic() - t0
            return cex

        # Phase 2: random walk testing
        cex = self._random_walk_test(hypothesis, t0)
        if cex is not None:
            self._stats.counterexamples_found += 1
            self._stats.total_time_seconds += time.monotonic() - t0
            return cex

        # Phase 3: W-method testing
        cex = self._w_method_test(hypothesis, t0)
        if cex is not None:
            self._stats.counterexamples_found += 1
            self._stats.total_time_seconds += time.monotonic() - t0
            return cex

        # No counterexample found — adapt depth
        if self._adaptive and self._current_depth < self._max_depth:
            self._current_depth = min(
                self._current_depth + self._depth_increment,
                self._max_depth,
            )
            logger.info(
                "No counterexample found; increasing depth to %d",
                self._current_depth,
            )

        self._stats.total_time_seconds += time.monotonic() - t0
        return None

    # -- systematic (BFS) testing -------------------------------------------

    def _systematic_test(
        self,
        hypothesis: HypothesisInterface,
        t0: float,
    ) -> Optional[Counterexample]:
        """Enumerate all sequences up to ``_current_depth`` and compare."""
        depth = self._current_depth
        self._stats.max_depth_tested = max(
            self._stats.max_depth_tested, depth
        )

        queue: Deque[Tuple[str, ...]] = deque()
        queue.append(())  # start with empty sequence

        while queue:
            if time.monotonic() - t0 > self._timeout:
                logger.warning("Systematic test timed out at depth %d", depth)
                break

            seq = queue.popleft()
            cex = self._compare_single(hypothesis, seq, "systematic")
            self._stats.systematic_tests += 1
            self._stats.total_tests += 1
            if cex is not None:
                return cex

            if len(seq) < depth:
                for act in self._actions:
                    queue.append(seq + (act,))

        return None

    # -- random walk testing ------------------------------------------------

    def _random_walk_test(
        self,
        hypothesis: HypothesisInterface,
        t0: float,
    ) -> Optional[Counterexample]:
        """Perform random walks and check each prefix."""
        walks = self._compute_random_walk_count()

        for _ in range(walks):
            if time.monotonic() - t0 > self._timeout:
                logger.warning("Random walk test timed out")
                break

            length = self._rng.randint(1, self._max_random_length)
            seq = tuple(
                self._rng.choice(self._action_list) for _ in range(length)
            )

            # Check every prefix of the walk
            for prefix_len in range(len(seq) + 1):
                prefix = seq[:prefix_len]
                cex = self._compare_single(hypothesis, prefix, "random_walk")
                self._stats.random_tests += 1
                self._stats.total_tests += 1
                if cex is not None:
                    return cex

        return None

    def _compute_random_walk_count(self) -> int:
        """Compute number of walks needed for target confidence."""
        if self._confidence <= 0:
            return self._random_walks
        # Simple bound: n ≥ ln(1/(1-c)) / p_min
        # Use the configured number as a baseline, scale with confidence
        base = self._random_walks
        scale = -math.log(1.0 - self._confidence) if self._confidence < 1.0 else 5.0
        return max(base, int(base * scale / 3.0))

    # -- W-method testing ---------------------------------------------------

    def _w_method_test(
        self,
        hypothesis: HypothesisInterface,
        t0: float,
    ) -> Optional[Counterexample]:
        """W-method based conformance testing.

        Constructs a characterisation set W from the hypothesis and tests
        S · Σ^{≤m-n+1} · W where m is an upper bound on the number of
        concrete states and n = |hypothesis states|.
        """
        w_set = self._compute_characterisation_set(hypothesis)
        if not w_set:
            return None

        n = len(hypothesis.states())
        # Assume concrete system has at most 2n states
        m = 2 * n
        extra_depth = max(m - n + 1, 1)
        extra_depth = min(extra_depth, 4)  # bound for practicality

        # State cover: one access sequence per hypothesis state
        state_cover = self._compute_state_cover(hypothesis)

        for access_seq in state_cover:
            if time.monotonic() - t0 > self._timeout:
                break
            # Generate middle parts Σ^{≤extra_depth}
            for mid_len in range(extra_depth + 1):
                if time.monotonic() - t0 > self._timeout:
                    break
                for mid in itertools.product(self._actions, repeat=mid_len):
                    if time.monotonic() - t0 > self._timeout:
                        break
                    for w in w_set:
                        full_seq = access_seq + mid + w
                        cex = self._compare_single(
                            hypothesis, full_seq, "w_method"
                        )
                        self._stats.w_method_tests += 1
                        self._stats.total_tests += 1
                        if cex is not None:
                            return cex

        return None

    def _compute_characterisation_set(
        self,
        hypothesis: HypothesisInterface,
    ) -> List[Tuple[str, ...]]:
        """Compute a characterisation set W for the hypothesis.

        W is a set of suffixes that distinguishes every pair of states.
        """
        states = sorted(hypothesis.states())
        if len(states) <= 1:
            return [()]

        w_set: List[Tuple[str, ...]] = [()]  # always include ε
        w_set_frozen: Set[Tuple[str, ...]] = {()}

        for i, s1 in enumerate(states):
            for s2 in states[i + 1:]:
                # Check if already distinguished by current W
                distinguished = False
                for w in w_set:
                    obs1 = self._hyp_observation(hypothesis, s1, w)
                    obs2 = self._hyp_observation(hypothesis, s2, w)
                    if obs1 != obs2:
                        distinguished = True
                        break
                if distinguished:
                    continue

                # BFS for a distinguishing suffix
                suffix = self._find_distinguishing_suffix(
                    hypothesis, s1, s2, max_depth=self._current_depth
                )
                if suffix is not None and suffix not in w_set_frozen:
                    w_set.append(suffix)
                    w_set_frozen.add(suffix)

        return w_set

    def _find_distinguishing_suffix(
        self,
        hypothesis: HypothesisInterface,
        s1: str,
        s2: str,
        max_depth: int,
    ) -> Optional[Tuple[str, ...]]:
        """BFS for the shortest suffix distinguishing s1 from s2."""
        queue: Deque[Tuple[str, str, Tuple[str, ...]]] = deque()
        queue.append((s1, s2, ()))
        visited: Set[Tuple[str, str]] = set()

        while queue:
            q1, q2, suffix = queue.popleft()
            if (q1, q2) in visited:
                continue
            visited.add((q1, q2))

            obs1 = hypothesis.observation_at(q1)
            obs2 = hypothesis.observation_at(q2)
            if obs1 != obs2:
                return suffix

            if len(suffix) >= max_depth:
                continue

            for act in self._actions:
                t1 = hypothesis.transition(q1, act)
                t2 = hypothesis.transition(q2, act)
                if t1 is not None and t2 is not None and (t1, t2) not in visited:
                    queue.append((t1, t2, suffix + (act,)))

        return None

    def _compute_state_cover(
        self,
        hypothesis: HypothesisInterface,
    ) -> List[Tuple[str, ...]]:
        """Compute an access-sequence state cover for the hypothesis."""
        init = hypothesis.initial_state()
        cover: Dict[str, Tuple[str, ...]] = {init: ()}
        queue: Deque[str] = deque([init])

        while queue:
            state = queue.popleft()
            for act in self._actions:
                t = hypothesis.transition(state, act)
                if t is not None and t not in cover:
                    cover[t] = cover[state] + (act,)
                    queue.append(t)

        return list(cover.values())

    def _hyp_observation(
        self,
        hypothesis: HypothesisInterface,
        state: str,
        suffix: Tuple[str, ...],
    ) -> Optional[Observation]:
        """Get the hypothesis observation after applying *suffix* from *state*."""
        s = state
        for act in suffix:
            s = hypothesis.transition(s, act)
            if s is None:
                return None
        return hypothesis.observation_at(s)

    # -- comparison helper --------------------------------------------------

    def _compare_single(
        self,
        hypothesis: HypothesisInterface,
        sequence: Tuple[str, ...],
        method: str,
    ) -> Optional[Counterexample]:
        """Compare hypothesis and concrete on a single sequence."""
        # Hypothesis side
        hyp_state = hypothesis.state_reached(sequence)
        if hyp_state is None:
            # Hypothesis has no transition for this sequence — treat as
            # potential mismatch if concrete can execute it.
            concrete_result = self._mq.query(sequence)
            if concrete_result.success and concrete_result.observation:
                return Counterexample(
                    sequence=sequence,
                    hypothesis_observation=None,
                    concrete_observation=concrete_result.observation,
                    discovery_method=method,
                )
            return None

        hyp_obs = hypothesis.observation_at(hyp_state)

        # Concrete side
        concrete_result = self._mq.query(sequence)
        if not concrete_result.success:
            return None  # can't compare on timeout/error

        concrete_obs = concrete_result.observation
        if hyp_obs != concrete_obs:
            logger.info(
                "Counterexample found via %s: %s (len=%d)",
                method,
                ".".join(sequence) if sequence else "ε",
                len(sequence),
            )
            return Counterexample(
                sequence=sequence,
                hypothesis_observation=hyp_obs,
                concrete_observation=concrete_obs,
                discovery_method=method,
            )
        return None

    # -- configuration ------------------------------------------------------

    @property
    def current_depth(self) -> int:
        return self._current_depth

    @current_depth.setter
    def current_depth(self, value: int) -> None:
        self._current_depth = min(value, self._max_depth)

    @property
    def stats(self) -> EquivalenceStats:
        return self._stats

    def reset_stats(self) -> None:
        self._stats = EquivalenceStats()

    def reset_depth(self) -> None:
        self._current_depth = 3

    def __repr__(self) -> str:
        return (
            f"EquivalenceOracle(depth={self._current_depth}, "
            f"tests={self._stats.total_tests})"
        )
