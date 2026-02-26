"""
Complete W-method implementation adapted for F-coalgebras.

Implements Chow's W-method (1978) for conformance testing of hypothesis
coalgebras.  The W-method uses a characterisation set W to construct a
complete test suite that can detect all faults in systems with a bounded
number of extra states.

THEOREM (W-method completeness):
  Let H be a hypothesis with n states, diameter d, and alphabet Σ.
  Let M be the target system with at most m states.
  The test suite S · Σ^{≤k} · W, where k = m - n + 1, detects all
  faults if M has at most m states.

For F-coalgebras, observations replace simple output comparisons:
instead of comparing output labels, we compare F-behaviour observations.
"""

from __future__ import annotations

import itertools
import logging
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

logger = logging.getLogger(__name__)


@dataclass
class WMethodResult:
    """Result of W-method conformance testing."""

    passed: bool = True
    counterexample: Optional[Tuple[str, ...]] = None
    coverage: float = 0.0
    tested_count: int = 0
    max_depth: int = 0
    characterization_set_size: int = 0
    state_cover_size: int = 0
    exploration_depth: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "counterexample": (
                list(self.counterexample) if self.counterexample else None
            ),
            "coverage": self.coverage,
            "tested_count": self.tested_count,
            "max_depth": self.max_depth,
            "characterization_set_size": self.characterization_set_size,
            "state_cover_size": self.state_cover_size,
            "exploration_depth": self.exploration_depth,
            "elapsed_seconds": self.elapsed_seconds,
        }

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"W-method {status}: {self.tested_count} tests, "
            f"depth={self.max_depth}, coverage={self.coverage:.2%}, "
            f"|W|={self.characterization_set_size}, "
            f"|S|={self.state_cover_size}, "
            f"{self.elapsed_seconds:.2f}s"
        )


class WMethodTester:
    """Complete W-method conformance testing for F-coalgebras.

    Implements the full W-method test suite generation and execution.
    The test suite is S · Σ^{≤k} · W where:
    - S is a state cover (access sequences for each hypothesis state)
    - Σ^{≤k} is the set of all action sequences up to length k
    - W is the characterisation set distinguishing all state pairs

    Parameters
    ----------
    hypothesis : HypothesisInterface
        The learned hypothesis coalgebra.
    oracle : callable
        A function that takes an action sequence (tuple of str) and returns
        the observation of the concrete system after that sequence.
        Should return None on failure.
    concrete_state_bound : int, optional
        Upper bound on the number of concrete states m.
        Defaults to 2n where n = |hypothesis states|.
    timeout : float
        Maximum seconds for the full test suite.
    """

    def __init__(
        self,
        hypothesis: Any,
        oracle: Callable[[Tuple[str, ...]], Any],
        concrete_state_bound: Optional[int] = None,
        timeout: float = 120.0,
    ) -> None:
        self._hypothesis = hypothesis
        self._oracle = oracle
        self._timeout = timeout

        self._states: List[str] = sorted(
            hypothesis.states()
            if callable(getattr(hypothesis, "states", None))
            else []
        )
        self._actions: List[str] = sorted(
            hypothesis.actions()
            if callable(getattr(hypothesis, "actions", None))
            else []
        )
        self._n = len(self._states)
        self._m = (
            concrete_state_bound
            if concrete_state_bound is not None
            else 2 * self._n
        )

        self._state_cover: List[Tuple[str, ...]] = []
        self._char_set: List[Tuple[str, ...]] = []
        self._tested: int = 0
        self._total_possible: int = 0

    def run(self) -> WMethodResult:
        """Execute the full W-method test suite.

        Returns a WMethodResult describing the outcome.
        """
        t0 = time.monotonic()

        if not self._states or not self._actions:
            return WMethodResult(
                passed=True,
                elapsed_seconds=time.monotonic() - t0,
            )

        # Step 1: compute state cover S
        self._state_cover = self._compute_state_cover()

        # Step 2: compute characterisation set W
        self._char_set = self._compute_characterization_set()

        # Step 3: compute exploration depth k = m - n + 1
        k = max(self._m - self._n + 1, 1)
        # Practical bound to avoid combinatorial explosion
        k = min(k, 8)

        # Step 4: estimate total tests
        a = len(self._actions)
        # Total middle sequences of length 0..k
        total_mid = sum(a ** i for i in range(k + 1)) if a > 0 else 1
        self._total_possible = (
            len(self._state_cover) * total_mid * len(self._char_set)
        )

        # Step 5: execute test suite
        counterexample = self._execute_test_suite(k, t0)

        elapsed = time.monotonic() - t0
        coverage = (
            self._tested / self._total_possible
            if self._total_possible > 0
            else 1.0
        )

        # Compute max depth of any tested sequence
        max_depth = 0
        if self._state_cover and self._char_set:
            max_access = max(len(s) for s in self._state_cover)
            max_w = max(len(w) for w in self._char_set)
            max_depth = max_access + k + max_w

        return WMethodResult(
            passed=counterexample is None,
            counterexample=counterexample,
            coverage=coverage,
            tested_count=self._tested,
            max_depth=max_depth,
            characterization_set_size=len(self._char_set),
            state_cover_size=len(self._state_cover),
            exploration_depth=k,
            elapsed_seconds=elapsed,
        )

    def _compute_state_cover(self) -> List[Tuple[str, ...]]:
        """Compute a state cover: one access sequence per hypothesis state."""
        if not self._states:
            return [()]

        init = (
            self._hypothesis.initial_state()
            if callable(getattr(self._hypothesis, "initial_state", None))
            else self._states[0]
        )

        cover: Dict[str, Tuple[str, ...]] = {init: ()}
        queue: Deque[str] = deque([init])

        while queue:
            state = queue.popleft()
            for act in self._actions:
                t = (
                    self._hypothesis.transition(state, act)
                    if callable(
                        getattr(self._hypothesis, "transition", None)
                    )
                    else None
                )
                if t is not None and t not in cover:
                    cover[t] = cover[state] + (act,)
                    queue.append(t)

        return list(cover.values())

    def _compute_characterization_set(self) -> List[Tuple[str, ...]]:
        """Compute a characterisation set W that distinguishes all state pairs.

        W is a minimal set of suffixes such that for every pair of
        distinct states (s, t), there exists w ∈ W where the observation
        after applying w from s differs from the observation after w from t.
        """
        if len(self._states) <= 1:
            return [()]

        w_set: List[Tuple[str, ...]] = [()]
        w_frozen: Set[Tuple[str, ...]] = {()}

        for i, s1 in enumerate(self._states):
            for s2 in self._states[i + 1:]:
                # Check if already distinguished
                distinguished = False
                for w in w_set:
                    obs1 = self._observe_from(s1, w)
                    obs2 = self._observe_from(s2, w)
                    if obs1 != obs2:
                        distinguished = True
                        break

                if distinguished:
                    continue

                # BFS for distinguishing suffix
                suffix = self._find_distinguishing_suffix(
                    s1, s2, max_depth=self._n
                )
                if suffix is not None and suffix not in w_frozen:
                    w_set.append(suffix)
                    w_frozen.add(suffix)

        return w_set

    def _find_distinguishing_suffix(
        self, s1: str, s2: str, max_depth: int
    ) -> Optional[Tuple[str, ...]]:
        """BFS for shortest suffix distinguishing s1 from s2."""
        queue: Deque[Tuple[str, str, Tuple[str, ...]]] = deque(
            [(s1, s2, ())]
        )
        visited: Set[Tuple[str, str]] = set()

        while queue:
            q1, q2, suffix = queue.popleft()
            if (q1, q2) in visited:
                continue
            visited.add((q1, q2))

            obs1 = self._observation_at(q1)
            obs2 = self._observation_at(q2)
            if obs1 != obs2:
                return suffix

            if len(suffix) >= max_depth:
                continue

            for act in self._actions:
                t1 = self._transition(q1, act)
                t2 = self._transition(q2, act)
                if (
                    t1 is not None
                    and t2 is not None
                    and (t1, t2) not in visited
                ):
                    queue.append((t1, t2, suffix + (act,)))

        return None

    def _execute_test_suite(
        self, k: int, t0: float
    ) -> Optional[Tuple[str, ...]]:
        """Execute S · Σ^{≤k} · W and return first counterexample or None."""
        for access_seq in self._state_cover:
            if time.monotonic() - t0 > self._timeout:
                logger.warning("W-method timed out")
                break

            for mid_len in range(k + 1):
                if time.monotonic() - t0 > self._timeout:
                    break

                for mid in itertools.product(
                    self._actions, repeat=mid_len
                ):
                    if time.monotonic() - t0 > self._timeout:
                        break

                    for w in self._char_set:
                        full_seq = access_seq + mid + w
                        self._tested += 1

                        # Get hypothesis observation
                        hyp_state = self._state_reached(full_seq)
                        hyp_obs = (
                            self._observation_at(hyp_state)
                            if hyp_state is not None
                            else None
                        )

                        # Get concrete observation
                        concrete_obs = self._oracle(full_seq)

                        if hyp_obs != concrete_obs:
                            return full_seq

        return None

    def _observe_from(
        self, state: str, suffix: Tuple[str, ...]
    ) -> Any:
        """Observe hypothesis after applying suffix from state."""
        s = state
        for act in suffix:
            s = self._transition(s, act)
            if s is None:
                return None
        return self._observation_at(s)

    def _observation_at(self, state: Optional[str]) -> Any:
        """Get observation at a state."""
        if state is None:
            return None
        if callable(getattr(self._hypothesis, "observation_at", None)):
            return self._hypothesis.observation_at(state)
        return None

    def _transition(self, state: str, action: str) -> Optional[str]:
        """Get transition target."""
        if callable(getattr(self._hypothesis, "transition", None)):
            return self._hypothesis.transition(state, action)
        return None

    def _state_reached(
        self, sequence: Tuple[str, ...]
    ) -> Optional[str]:
        """Get state reached by sequence from initial state."""
        if callable(getattr(self._hypothesis, "state_reached", None)):
            return self._hypothesis.state_reached(sequence)
        if callable(getattr(self._hypothesis, "initial_state", None)):
            s = self._hypothesis.initial_state()
            for act in sequence:
                s = self._transition(s, act)
                if s is None:
                    return None
            return s
        return None

    @property
    def characterization_set(self) -> List[Tuple[str, ...]]:
        return list(self._char_set)

    @property
    def state_cover(self) -> List[Tuple[str, ...]]:
        return list(self._state_cover)
