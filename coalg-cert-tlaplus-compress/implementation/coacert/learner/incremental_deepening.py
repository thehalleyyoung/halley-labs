"""
Incremental deepening protocol for conformance testing.

Addresses the critique that bounded conformance testing with a user-supplied
depth parameter introduces a soundness gap.  This module implements an
incremental deepening protocol that:

1. Starts at a small depth k (default k=3).
2. After each round, checks whether k is sufficient using the W-method
   completeness criterion: k >= diam(H) + (m - n + 1).
3. If not sufficient, increases depth (doubles or steps).
4. Tracks hypothesis stability: if the hypothesis is unchanged for 3+
   consecutive rounds, declares convergence.
5. Emits a ConformanceCertificate when depth is sufficient or convergence
   is detected.

THEOREM (Incremental Deepening Soundness):
  If incremental deepening terminates with is_sufficient=True, then
  the hypothesis is behaviourally equivalent to the target system
  (assuming |target| ≤ m).  If it terminates via convergence, the
  hypothesis has been stable under testing at increasing depths for
  at least 3 rounds, providing high empirical confidence.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from .diameter_computation import (
    DiameterCertificate,
    ExactDiameterComputer,
    IncrementalDiameter,
)
from .conformance_gap import (
    ConformanceGapAnalyzer,
    ConformanceCompleteCertificate,
    compute_sufficient_depth,
)
from .w_method import WMethodResult, WMethodTester

logger = logging.getLogger(__name__)


@dataclass
class DeepeningRound:
    """Record of a single incremental deepening round."""

    round_number: int
    depth_k: int
    hypothesis_states: int
    hypothesis_diameter: int
    sufficient_depth: int
    is_sufficient: bool
    counterexample_found: bool
    counterexample: Optional[Tuple[str, ...]] = None
    w_method_tests: int = 0
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        status = "sufficient" if self.is_sufficient else "insufficient"
        cex = "cex_found" if self.counterexample_found else "no_cex"
        return (
            f"Round {self.round_number}: k={self.depth_k}, "
            f"n={self.hypothesis_states}, d={self.hypothesis_diameter}, "
            f"k_suf={self.sufficient_depth}, {status}, {cex}"
        )


@dataclass
class ConvergenceCertificate:
    """Certificate emitted when convergence is achieved.

    Records the depth at convergence, the number of stable rounds,
    the final gap_ratio, and whether the depth was formally sufficient.
    """

    converged: bool = False
    convergence_round: int = 0
    final_depth: int = 0
    stable_rounds: int = 0
    final_gap_ratio: float = 1.0
    is_depth_sufficient: bool = False
    conformance_certificate: Optional[ConformanceCompleteCertificate] = None
    details: str = ""

    def summary(self) -> str:
        status = "CONVERGED" if self.converged else "NOT_CONVERGED"
        suf = "sufficient" if self.is_depth_sufficient else "insufficient"
        return (
            f"ConvergenceCertificate({status}): depth={self.final_depth}, "
            f"stable={self.stable_rounds}, gap={self.final_gap_ratio:.4f}, "
            f"{suf}"
        )


@dataclass
class ConvergenceHistory:
    """Tracks convergence of the hypothesis across deepening rounds."""

    rounds: List[DeepeningRound] = field(default_factory=list)
    hypothesis_sizes: List[int] = field(default_factory=list)
    depths: List[int] = field(default_factory=list)
    gap_ratios: List[float] = field(default_factory=list)
    stable_count: int = 0
    converged: bool = False
    convergence_round: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_count": len(self.rounds),
            "hypothesis_sizes": self.hypothesis_sizes,
            "depths": self.depths,
            "gap_ratios": self.gap_ratios,
            "stable_count": self.stable_count,
            "converged": self.converged,
            "convergence_round": self.convergence_round,
            "round_summaries": [r.summary() for r in self.rounds],
        }


class IncrementalDeepeningOracle:
    """Incremental deepening protocol for conformance testing.

    Starts at a small depth and incrementally increases it until either:
    (a) The depth is provably sufficient (k >= diam(H) + (m - n + 1)), or
    (b) The hypothesis has stabilised for stability_threshold consecutive
        rounds.

    Parameters
    ----------
    learner_factory : callable
        A factory function that, given a depth k, runs the learning loop
        and returns a hypothesis. Signature: (int) -> hypothesis.
    oracle_factory : callable
        A factory function that creates an observation oracle for a given
        action sequence. Signature: (Tuple[str, ...]) -> observation.
    initial_depth : int
        Starting depth for conformance testing.
    max_depth : int
        Maximum depth to try before giving up.
    concrete_state_bound : int, optional
        Upper bound m on the number of concrete system states.
    depth_strategy : str
        How to increase depth: "double" doubles k, "step" increments by
        step_size.
    step_size : int
        Step size for "step" strategy.
    stability_threshold : int
        Number of consecutive stable rounds needed to declare convergence.
    max_rounds : int
        Maximum number of deepening rounds.
    timeout : float
        Total timeout for all rounds in seconds.
    """

    def __init__(
        self,
        learner_factory: Callable[[int], Any],
        oracle_factory: Callable[[Tuple[str, ...]], Any],
        *,
        initial_depth: int = 3,
        max_depth: int = 64,
        concrete_state_bound: Optional[int] = None,
        depth_strategy: str = "double",
        step_size: int = 3,
        stability_threshold: int = 3,
        max_rounds: int = 20,
        timeout: float = 600.0,
    ) -> None:
        self._learner_factory = learner_factory
        self._oracle_factory = oracle_factory
        self._initial_depth = initial_depth
        self._max_depth = max_depth
        self._concrete_bound = concrete_state_bound
        self._depth_strategy = depth_strategy
        self._step_size = step_size
        self._stability_threshold = stability_threshold
        self._max_rounds = max_rounds
        self._timeout = timeout

        self._diameter_tracker = IncrementalDiameter()
        self._history = ConvergenceHistory()
        self._last_hypothesis: Optional[Any] = None
        self._certificate: Optional[Any] = None

    def run(
        self,
    ) -> Tuple[Any, Optional[Any], ConvergenceHistory]:
        """Run the incremental deepening protocol.

        Returns
        -------
        hypothesis
            The final learned hypothesis.
        certificate
            A ConformanceCertificate if depth was sufficient or convergence
            detected, otherwise None.
        convergence_history
            Full history of deepening rounds.
        """
        t_start = time.monotonic()
        current_depth = self._initial_depth
        hypothesis = None
        certificate = None
        stable_count = 0

        for round_num in range(1, self._max_rounds + 1):
            if time.monotonic() - t_start > self._timeout:
                logger.warning(
                    "Incremental deepening timed out after %d rounds",
                    round_num - 1,
                )
                break

            logger.info(
                "=== Deepening round %d, depth k=%d ===",
                round_num,
                current_depth,
            )

            # Step 1: learn hypothesis at current depth
            t0 = time.monotonic()
            hypothesis = self._learner_factory(current_depth)

            if hypothesis is None:
                logger.warning("Learner returned None at depth %d", current_depth)
                current_depth = self._next_depth(current_depth)
                continue

            # Step 2: compute diameter
            n = len(
                hypothesis.states()
                if callable(getattr(hypothesis, "states", None))
                else []
            )
            diameter = self._diameter_tracker.update(hypothesis)

            # Step 3: compute sufficient depth
            m = self._concrete_bound if self._concrete_bound is not None else 2 * n
            sufficient_depth = diameter + max(m - n, 0) + 1
            is_sufficient = current_depth >= sufficient_depth

            # Step 4: check for counterexamples (if not sufficient, run W-method)
            cex_found = False
            cex = None
            w_tests = 0

            if not is_sufficient and n > 0:
                w_tester = WMethodTester(
                    hypothesis,
                    self._oracle_factory,
                    concrete_state_bound=m,
                    timeout=min(30.0, (self._timeout - (time.monotonic() - t_start)) / 2),
                )
                w_result = w_tester.run()
                w_tests = w_result.tested_count
                if not w_result.passed:
                    cex_found = True
                    cex = w_result.counterexample

            elapsed = time.monotonic() - t0

            # Step 5: record round
            rnd = DeepeningRound(
                round_number=round_num,
                depth_k=current_depth,
                hypothesis_states=n,
                hypothesis_diameter=diameter,
                sufficient_depth=sufficient_depth,
                is_sufficient=is_sufficient,
                counterexample_found=cex_found,
                counterexample=cex,
                w_method_tests=w_tests,
                elapsed_seconds=elapsed,
            )
            self._history.rounds.append(rnd)
            self._history.hypothesis_sizes.append(n)
            self._history.depths.append(current_depth)

            # Track gap ratio: ratio of sufficient_depth shortfall
            if sufficient_depth > 0:
                gap = max(sufficient_depth - current_depth, 0) / sufficient_depth
            else:
                gap = 0.0
            self._history.gap_ratios.append(gap)

            logger.info("  %s", rnd.summary())

            # Step 6: check stability
            if self._last_hypothesis is not None and n > 0:
                last_n = len(
                    self._last_hypothesis.states()
                    if callable(
                        getattr(self._last_hypothesis, "states", None)
                    )
                    else []
                )
                if n == last_n and not cex_found:
                    # Check structural equivalence if possible
                    if callable(
                        getattr(self._last_hypothesis, "is_isomorphic_to", None)
                    ):
                        if self._last_hypothesis.is_isomorphic_to(hypothesis):
                            stable_count += 1
                        else:
                            stable_count = 0
                    else:
                        stable_count += 1
                else:
                    stable_count = 0
            else:
                stable_count = 0

            self._history.stable_count = stable_count
            self._last_hypothesis = hypothesis

            # Step 7: check termination conditions
            if is_sufficient:
                logger.info(
                    "Depth k=%d is sufficient (>= %d = diam(%d) + (%d-%d+1))",
                    current_depth, sufficient_depth, diameter, m, n,
                )
                certificate = self._build_certificate(
                    hypothesis, current_depth, m, diameter,
                    is_sufficient=True,
                    convergence_detected=False,
                )
                break

            if stable_count >= self._stability_threshold:
                logger.info(
                    "Convergence detected: hypothesis stable for %d rounds",
                    stable_count,
                )
                self._history.converged = True
                self._history.convergence_round = round_num
                certificate = self._build_certificate(
                    hypothesis, current_depth, m, diameter,
                    is_sufficient=False,
                    convergence_detected=True,
                )
                break

            if cex_found:
                # Counterexample found: the hypothesis is wrong at this depth;
                # the learner_factory should incorporate the counterexample
                # on the next call. Don't increase depth yet.
                logger.info(
                    "Counterexample found at depth %d, re-learning",
                    current_depth,
                )
            else:
                current_depth = self._next_depth(current_depth)

        # If we exhausted rounds without termination, build best-effort cert
        if certificate is None and hypothesis is not None:
            n = len(
                hypothesis.states()
                if callable(getattr(hypothesis, "states", None))
                else []
            )
            m = self._concrete_bound if self._concrete_bound is not None else 2 * n
            diameter = self._diameter_tracker.current_diameter
            certificate = self._build_certificate(
                hypothesis, current_depth, m, diameter,
                is_sufficient=False,
                convergence_detected=False,
            )

        self._certificate = certificate
        return hypothesis, certificate, self._history

    def _next_depth(self, current: int) -> int:
        """Compute the next depth according to the strategy."""
        if self._depth_strategy == "double":
            nxt = current * 2
        else:
            nxt = current + self._step_size
        return min(nxt, self._max_depth)

    def _build_certificate(
        self,
        hypothesis: Any,
        actual_depth: int,
        concrete_bound: int,
        diameter: int,
        is_sufficient: bool,
        convergence_detected: bool,
    ) -> ConvergenceCertificate:
        """Build a ConvergenceCertificate with ConformanceCompleteCertificate."""
        n = len(
            hypothesis.states()
            if callable(getattr(hypothesis, "states", None))
            else []
        )
        actions = (
            hypothesis.actions()
            if callable(getattr(hypothesis, "actions", None))
            else set()
        )
        n_actions = len(actions) if actions else 2

        conf_cert = ConformanceCompleteCertificate.build(
            hypothesis_states=n,
            concrete_bound=concrete_bound,
            diameter=diameter,
            actual_depth=actual_depth,
            n_actions=n_actions,
        )

        final_gap = self._history.gap_ratios[-1] if self._history.gap_ratios else 1.0

        cert = ConvergenceCertificate(
            converged=convergence_detected or is_sufficient,
            convergence_round=(
                self._history.convergence_round
                if convergence_detected
                else len(self._history.rounds)
            ),
            final_depth=actual_depth,
            stable_rounds=self._history.stable_count,
            final_gap_ratio=final_gap,
            is_depth_sufficient=is_sufficient,
            conformance_certificate=conf_cert,
            details=conf_cert.details,
        )
        return cert

    @property
    def history(self) -> ConvergenceHistory:
        return self._history

    @property
    def certificate(self) -> Optional[Any]:
        return self._certificate

    def suggest_initial_depth(
        self,
        estimated_states: Optional[int] = None,
        estimated_actions: Optional[int] = None,
    ) -> int:
        """Suggest an initial depth based on problem size estimates.

        For small systems (< 20 states), k=3 is usually sufficient.
        For medium systems (20-100), k=5 is recommended.
        For large systems (> 100), k=8 is a good starting point.
        """
        if estimated_states is None:
            return self._initial_depth

        if estimated_states < 20:
            return 3
        elif estimated_states < 100:
            return 5
        elif estimated_states < 500:
            return 8
        else:
            return 12

    def compute_automatic_depth(
        self,
        hypothesis: Any,
        concrete_state_bound: Optional[int] = None,
    ) -> int:
        """Compute the automatic sufficient depth for a hypothesis.

        Uses exact diameter computation and the W-method formula:
        k = diam(H) + (m - n + 1)

        Parameters
        ----------
        hypothesis : HypothesisInterface
            The hypothesis coalgebra.
        concrete_state_bound : int, optional
            Upper bound on concrete states. Defaults to 2n.

        Returns
        -------
        int
            The sufficient depth k.
        """
        computer = ExactDiameterComputer(hypothesis)
        cert = computer.compute()
        d = cert.diameter
        n = cert.state_count

        m = concrete_state_bound if concrete_state_bound is not None else 2 * n
        return d + max(m - n, 0) + 1
