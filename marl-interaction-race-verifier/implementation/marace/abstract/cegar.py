"""
Counter-Example Guided Abstraction Refinement (CEGAR) for MARACE.

Implements a CEGAR loop that closes the gap between abstract verification
and concrete system behaviour.  When the zonotope fixpoint engine reports
a potential interaction race (abstract states overlap the unsafe region),
the CEGAR loop decides whether the alarm is *real* (there exists a concrete
schedule that triggers the race) or *spurious* (an over-approximation
artifact).  Spurious counterexamples drive abstraction refinement — the
zonotope is split along the counterexample dimension and verification is
repeated on both halves.  The process terminates when:

  (a)  a real counterexample is found   → UNSAFE,
  (b)  all refined sub-problems are safe → SAFE, or
  (c)  a resource budget is exhausted    → UNKNOWN.

Soundness argument
------------------
The zonotope fixpoint is a *sound over-approximation* of the reachable
state set.  If the fixpoint does not intersect the unsafe region, the
concrete system is safe — no CEGAR step is needed.  When the fixpoint
*does* intersect, CEGAR refines the abstraction by partitioning the
initial zonotope and recomputing the fixpoint on each partition.  The
union of the partitions still covers the original zonotope, so if every
partition yields a safe fixpoint, the system is safe by soundness of the
individual fixpoints plus the covering argument.  A concrete
counterexample is validated by forward simulation of the neural-network
policy, ruling out false alarms.

Integration points
------------------
* :class:`~marace.abstract.zonotope.Zonotope` — abstract domain
* :class:`~marace.abstract.fixpoint.FixpointEngine` / ``FixpointResult``
* :class:`~marace.policy.onnx_loader.NetworkArchitecture`, ``LayerInfo``,
  ``ActivationType``
* :class:`~marace.policy.abstract_policy.AbstractPolicyEvaluator`,
  ``AbstractOutput``
* :class:`~marace.race.definition.InteractionRace`, ``RaceCondition``
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from marace.abstract.zonotope import Zonotope
from marace.abstract.fixpoint import (
    FixpointEngine,
    FixpointResult,
    WideningStrategy,
)
from marace.policy.onnx_loader import (
    ActivationType,
    LayerInfo,
    NetworkArchitecture,
)
from marace.policy.abstract_policy import AbstractPolicyEvaluator, AbstractOutput
from marace.race.definition import InteractionRace, RaceCondition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verdict enum
# ---------------------------------------------------------------------------


class Verdict(Enum):
    """Outcome of CEGAR verification."""

    SAFE = auto()
    UNSAFE = auto()
    UNKNOWN = auto()


# ---------------------------------------------------------------------------
# Refinement record
# ---------------------------------------------------------------------------


@dataclass
class RefinementRecord:
    """Record of a single abstraction-refinement step.

    Attributes
    ----------
    iteration : int
        Zero-based refinement iteration index.
    split_dimension : int
        State-space dimension along which the zonotope was split.
    split_point : float
        Value at which the split was performed.
    pre_split_volume : float
        Bounding-box volume of the zonotope before the split.
    post_split_volumes : Tuple[float, float]
        Bounding-box volumes of the two child zonotopes after the split.
    precision_improvement : float
        Relative reduction in volume:
        ``1 - sum(post_split_volumes) / pre_split_volume``.
    spurious_point : Optional[np.ndarray]
        The concrete point that was tested and found to be spurious
        (i.e. safe under concrete evaluation).
    wall_time_s : float
        Wall-clock seconds consumed by this refinement step.
    """

    iteration: int
    split_dimension: int
    split_point: float
    pre_split_volume: float
    post_split_volumes: Tuple[float, float]
    precision_improvement: float
    spurious_point: Optional[np.ndarray] = None
    wall_time_s: float = 0.0


# ---------------------------------------------------------------------------
# CEGAR result
# ---------------------------------------------------------------------------


@dataclass
class CEGARResult:
    """Result of a CEGAR verification run.

    Attributes
    ----------
    verdict : Verdict
        Final verification outcome — ``SAFE``, ``UNSAFE``, or ``UNKNOWN``.
    counterexample : Optional[np.ndarray]
        A concrete initial state that leads to the unsafe region when the
        system is ``UNSAFE``.  ``None`` when ``SAFE`` or ``UNKNOWN``.
    refinement_iterations : int
        Number of refinement iterations performed.
    refinement_history : List[RefinementRecord]
        Per-iteration refinement statistics.
    total_time_s : float
        Total wall-clock seconds.
    fixpoint_results : List[FixpointResult]
        Fixpoint results from each sub-problem (for diagnostics).
    """

    verdict: Verdict
    counterexample: Optional[np.ndarray] = None
    refinement_iterations: int = 0
    refinement_history: List[RefinementRecord] = field(default_factory=list)
    total_time_s: float = 0.0
    fixpoint_results: List[FixpointResult] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return self.verdict == Verdict.SAFE

    @property
    def is_unsafe(self) -> bool:
        return self.verdict == Verdict.UNSAFE

    @property
    def total_precision_improvement(self) -> float:
        """Cumulative precision improvement across all refinements."""
        if not self.refinement_history:
            return 0.0
        remaining = 1.0
        for r in self.refinement_history:
            remaining *= (1.0 - r.precision_improvement)
        return 1.0 - remaining

    def summary(self) -> str:
        lines = [
            f"CEGAR verdict: {self.verdict.name}",
            f"  Refinement iterations: {self.refinement_iterations}",
            f"  Total time: {self.total_time_s:.3f}s",
        ]
        if self.counterexample is not None:
            lines.append(f"  Counterexample: {self.counterexample}")
        if self.refinement_history:
            dims = [r.split_dimension for r in self.refinement_history]
            lines.append(f"  Split dimensions: {dims}")
            lines.append(
                f"  Total precision improvement: "
                f"{self.total_precision_improvement:.4f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Spuriousness checker
# ---------------------------------------------------------------------------


class SpuriousnessChecker:
    """Decide whether an abstract counterexample is real or spurious.

    Given a candidate point *x* in the fixpoint zonotope that also lies
    in the unsafe region according to the abstraction, we perform a
    *concrete* forward simulation of every agent's neural-network policy
    and check whether the resulting concrete joint state actually violates
    the safety property.

    Parameters
    ----------
    concrete_evaluator : callable
        ``concrete_evaluator(x) -> np.ndarray`` — evaluates the concrete
        (non-abstract) multi-agent policy at state *x* and returns the
        successor state.
    safety_predicate : callable
        ``safety_predicate(x) -> bool`` — returns ``True`` when *x* is
        in the *unsafe* region.
    num_samples : int
        Number of concrete samples to draw from the abstract
        counterexample zonotope when checking spuriousness.
    rng : Optional[np.random.Generator]
        Random-number generator for reproducibility.
    """

    def __init__(
        self,
        concrete_evaluator: Callable[[np.ndarray], np.ndarray],
        safety_predicate: Callable[[np.ndarray], bool],
        num_samples: int = 64,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._eval = concrete_evaluator
        self._pred = safety_predicate
        self._num_samples = num_samples
        self._rng = rng or np.random.default_rng(42)

    def check_point(self, point: np.ndarray) -> bool:
        """Return ``True`` if *point* is a **real** counterexample.

        The point is evaluated concretely through the policy; if the
        successor state violates the safety predicate, the counterexample
        is real.
        """
        successor = self._eval(point)
        return self._pred(successor)

    def check_zonotope(
        self, zonotope: Zonotope
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Sample the zonotope and check for real counterexamples.

        Returns
        -------
        (is_real, witness)
            ``is_real`` is ``True`` if a concrete counterexample was found.
            ``witness`` is the offending concrete point (or ``None``).
        """
        # Always check the center first (cheapest, most representative).
        if self.check_point(zonotope.center):
            return True, zonotope.center.copy()

        samples = zonotope.sample(self._num_samples, rng=self._rng)
        for s in samples:
            if self.check_point(s):
                return True, s.copy()

        return False, None

    def check_along_direction(
        self,
        zonotope: Zonotope,
        direction: np.ndarray,
        num_points: int = 16,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Check points along *direction* inside the zonotope.

        Useful for gradient-guided refinement: the direction typically
        points towards the unsafe region.

        Returns
        -------
        (is_real, witness)
        """
        _, extreme = zonotope.maximize(direction)
        _, opposite = zonotope.minimize(direction)

        for alpha in np.linspace(0.0, 1.0, num_points):
            point = (1.0 - alpha) * zonotope.center + alpha * extreme
            if zonotope.contains_point(point) and self.check_point(point):
                return True, point.copy()

        for alpha in np.linspace(0.0, 1.0, num_points):
            point = (1.0 - alpha) * zonotope.center + alpha * opposite
            if zonotope.contains_point(point) and self.check_point(point):
                return True, point.copy()

        return False, None


# ---------------------------------------------------------------------------
# Abstraction refinement strategies
# ---------------------------------------------------------------------------


class RefinementStrategy(Enum):
    """Strategy for choosing how to refine a zonotope."""

    DIMENSION = auto()
    COUNTEREXAMPLE = auto()
    GRADIENT = auto()


class AbstractionRefinement:
    """Strategies for refining a zonotope abstraction.

    Three splitting strategies are supported:

    1. **Dimension-based** — split along the dimension with the largest
       interval width in the bounding box.  Simple and reliable.
    2. **Counterexample-guided** — split along the dimension of the
       spurious counterexample that contributes most to the
       over-approximation.
    3. **Gradient-based** — split along the direction of steepest ascent
       of the safety-violation measure.  Requires a differentiable
       safety predicate (or a finite-difference approximation).

    Parameters
    ----------
    strategy : RefinementStrategy
        Default splitting strategy.
    safety_normal : Optional[np.ndarray]
        Normal vector of the unsafe half-space.  Used by the gradient
        strategy as the direction of steepest violation.
    """

    def __init__(
        self,
        strategy: RefinementStrategy = RefinementStrategy.COUNTEREXAMPLE,
        safety_normal: Optional[np.ndarray] = None,
    ) -> None:
        self._strategy = strategy
        self._safety_normal = safety_normal

    # -- Public API ---------------------------------------------------------

    def refine(
        self,
        zonotope: Zonotope,
        spurious_point: Optional[np.ndarray] = None,
    ) -> Tuple[Zonotope, Zonotope, int, float]:
        """Split *zonotope* according to the configured strategy.

        Returns
        -------
        (left, right, split_dim, split_value)
            Two child zonotopes whose union covers the original, the
            dimension index, and the split value.
        """
        if self._strategy == RefinementStrategy.COUNTEREXAMPLE and spurious_point is not None:
            return self.split_counterexample(zonotope, spurious_point)
        elif self._strategy == RefinementStrategy.GRADIENT and self._safety_normal is not None:
            return self.split_gradient(zonotope, self._safety_normal)
        else:
            return self.split_widest_dimension(zonotope)

    # -- Dimension-based splitting ------------------------------------------

    def split_widest_dimension(
        self, zonotope: Zonotope
    ) -> Tuple[Zonotope, Zonotope, int, float]:
        """Split along the dimension with the widest bounding-box interval.

        Soundness: the two children cover the original zonotope because
        ``split_halfspace`` partitions the space into two complementary
        half-spaces whose union is ℝⁿ, and the zonotope is a subset of ℝⁿ.

        Returns
        -------
        (left, right, dim, midpoint)
        """
        bbox = zonotope.bounding_box()  # shape (dim, 2)
        widths = bbox[:, 1] - bbox[:, 0]
        dim = int(np.argmax(widths))
        midpoint = float(zonotope.center[dim])

        normal = np.zeros(zonotope.dimension, dtype=np.float64)
        normal[dim] = 1.0

        left, right = zonotope.split_halfspace(normal, midpoint)
        return left, right, dim, midpoint

    # -- Counterexample-guided splitting ------------------------------------

    def split_counterexample(
        self,
        zonotope: Zonotope,
        spurious_point: np.ndarray,
    ) -> Tuple[Zonotope, Zonotope, int, float]:
        """Split along the dimension where the spurious point deviates most.

        The idea is that the spurious counterexample reveals which
        dimension contributes most to the over-approximation error.
        Splitting there maximally reduces the volume of the child that
        *does not* contain the spurious point.

        Soundness: same covering argument as dimension-based splitting.

        Parameters
        ----------
        zonotope : Zonotope
            Zonotope to refine.
        spurious_point : np.ndarray
            A point that the abstract analysis considers unsafe but
            concrete evaluation shows is safe.

        Returns
        -------
        (left, right, dim, split_value)
        """
        deviation = np.abs(spurious_point - zonotope.center)
        bbox = zonotope.bounding_box()
        widths = bbox[:, 1] - bbox[:, 0]
        # Normalise deviation by width to find the dimension with the
        # largest *relative* deviation — the one most likely responsible
        # for the spuriousness.
        safe_widths = np.where(widths > 1e-12, widths, 1.0)
        normalised = deviation / safe_widths
        dim = int(np.argmax(normalised))

        # Split at the midpoint between the center and the spurious point.
        split_value = float(
            0.5 * (zonotope.center[dim] + spurious_point[dim])
        )

        normal = np.zeros(zonotope.dimension, dtype=np.float64)
        normal[dim] = 1.0

        left, right = zonotope.split_halfspace(normal, split_value)
        return left, right, dim, split_value

    # -- Gradient-based splitting -------------------------------------------

    def split_gradient(
        self,
        zonotope: Zonotope,
        direction: np.ndarray,
    ) -> Tuple[Zonotope, Zonotope, int, float]:
        """Split along the direction of steepest safety-violation gradient.

        Given a direction *d* (typically the normal of the unsafe
        half-space), we project *d* onto the canonical axes and split
        along the axis with the largest projected component, weighted
        by the zonotope's bounding-box width.

        Soundness: same covering argument as dimension-based splitting.

        Parameters
        ----------
        zonotope : Zonotope
            Zonotope to refine.
        direction : np.ndarray
            Gradient or normal direction pointing toward the unsafe
            region.

        Returns
        -------
        (left, right, dim, split_value)
        """
        bbox = zonotope.bounding_box()
        widths = bbox[:, 1] - bbox[:, 0]
        safe_widths = np.where(widths > 1e-12, widths, 1e-12)

        # Weight each axis by |direction component| × interval width.
        scores = np.abs(direction) * safe_widths
        dim = int(np.argmax(scores))
        split_value = float(zonotope.center[dim])

        normal = np.zeros(zonotope.dimension, dtype=np.float64)
        normal[dim] = 1.0

        left, right = zonotope.split_halfspace(normal, split_value)
        return left, right, dim, split_value


# ---------------------------------------------------------------------------
# CEGAR Verifier
# ---------------------------------------------------------------------------


class CEGARVerifier:
    """Counter-Example Guided Abstraction Refinement verifier.

    Orchestrates the CEGAR loop:

    1. Run the fixpoint engine on the current abstraction.
    2. Check whether the fixpoint intersects the unsafe region.
    3. If not, return SAFE.
    4. If yes, extract a candidate counterexample and check it concretely.
    5. If the counterexample is real, return UNSAFE.
    6. Otherwise, refine the abstraction and go to 1.

    Soundness
    ---------
    * If the method returns ``SAFE``, all sub-problems into which the
      initial zonotope was partitioned have safe fixpoints.  Since the
      partitions cover the original zonotope, and each fixpoint is a
      sound over-approximation, the concrete reachable set is safe.
    * If the method returns ``UNSAFE``, a concrete witness schedule was
      found that violates the safety property.
    * ``UNKNOWN`` means the budget was exhausted without a conclusive
      answer; the system may or may not be safe.

    Parameters
    ----------
    transfer_fn : callable
        Transfer function ``Zonotope -> Zonotope`` for the fixpoint
        engine.
    spuriousness_checker : SpuriousnessChecker
        Checker that concretely evaluates candidate counterexamples.
    refinement : AbstractionRefinement
        Strategy for splitting zonotopes on spurious counterexamples.
    unsafe_halfspace : Tuple[np.ndarray, float]
        ``(a, b)`` defining the unsafe region ``{x : aᵀx ≥ b}``, i.e.
        the negation of the safety constraint ``aᵀx < b``.
    max_refinements : int
        Maximum number of refinement iterations (total across all
        sub-problems in the worklist).
    max_splits : int
        Maximum number of zonotope partitions in the worklist at any
        time.
    timeout_s : float
        Wall-clock timeout in seconds.
    fixpoint_kwargs : dict
        Extra keyword arguments forwarded to ``FixpointEngine``.
    """

    def __init__(
        self,
        transfer_fn: Callable[[Zonotope], Zonotope],
        spuriousness_checker: SpuriousnessChecker,
        refinement: AbstractionRefinement,
        unsafe_halfspace: Tuple[np.ndarray, float],
        max_refinements: int = 20,
        max_splits: int = 64,
        timeout_s: float = 300.0,
        fixpoint_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._transfer_fn = transfer_fn
        self._checker = spuriousness_checker
        self._refinement = refinement
        self._unsafe_normal, self._unsafe_bound = unsafe_halfspace
        self._max_refinements = max_refinements
        self._max_splits = max_splits
        self._timeout_s = timeout_s
        self._fixpoint_kwargs = fixpoint_kwargs or {}

    # -- Main entry point ---------------------------------------------------

    def verify(self, initial: Zonotope) -> CEGARResult:
        """Run the CEGAR loop on *initial*.

        Returns a ``CEGARResult`` with the verification verdict, any
        counterexample found, and refinement statistics.
        """
        start_time = time.monotonic()
        worklist: List[Zonotope] = [initial]
        refinement_count = 0
        history: List[RefinementRecord] = []
        all_fp_results: List[FixpointResult] = []

        while worklist:
            if self._budget_exhausted(start_time, refinement_count, len(worklist)):
                logger.warning(
                    "CEGAR budget exhausted: refinements=%d, worklist=%d",
                    refinement_count, len(worklist),
                )
                return CEGARResult(
                    verdict=Verdict.UNKNOWN,
                    refinement_iterations=refinement_count,
                    refinement_history=history,
                    total_time_s=time.monotonic() - start_time,
                    fixpoint_results=all_fp_results,
                )

            current = worklist.pop(0)

            # Step 1: compute fixpoint
            fp_result = self._run_fixpoint(current)
            all_fp_results.append(fp_result)

            # Step 2: check safety
            if not self._intersects_unsafe(fp_result.element):
                # This partition is safe — continue with remaining worklist.
                logger.debug("Partition safe (no intersection with unsafe region)")
                continue

            # Step 3: extract candidate counterexample
            candidate = self._extract_counterexample(fp_result.element)

            # Step 4: check spuriousness
            is_real, witness = self._check_spurious(fp_result.element, candidate)

            if is_real:
                assert witness is not None
                logger.info("Real counterexample found: %s", witness)
                return CEGARResult(
                    verdict=Verdict.UNSAFE,
                    counterexample=witness,
                    refinement_iterations=refinement_count,
                    refinement_history=history,
                    total_time_s=time.monotonic() - start_time,
                    fixpoint_results=all_fp_results,
                )

            # Step 5: refine
            ref_start = time.monotonic()
            left, right, dim, split_val = self._refine_abstraction(
                current, candidate
            )
            ref_time = time.monotonic() - ref_start

            pre_vol = current.volume_bound
            post_vols = (left.volume_bound, right.volume_bound)
            vol_sum = post_vols[0] + post_vols[1]
            improvement = max(0.0, 1.0 - vol_sum / max(pre_vol, 1e-30))

            record = RefinementRecord(
                iteration=refinement_count,
                split_dimension=dim,
                split_point=split_val,
                pre_split_volume=pre_vol,
                post_split_volumes=post_vols,
                precision_improvement=improvement,
                spurious_point=candidate,
                wall_time_s=ref_time,
            )
            history.append(record)
            refinement_count += 1

            logger.info(
                "CEGAR refinement %d: split dim=%d at %.4f, "
                "improvement=%.4f",
                refinement_count, dim, split_val, improvement,
            )

            worklist.append(left)
            worklist.append(right)

        # All partitions verified safe.
        return CEGARResult(
            verdict=Verdict.SAFE,
            refinement_iterations=refinement_count,
            refinement_history=history,
            total_time_s=time.monotonic() - start_time,
            fixpoint_results=all_fp_results,
        )

    # -- Internal helpers ---------------------------------------------------

    def _run_fixpoint(self, initial: Zonotope) -> FixpointResult:
        """Compute the abstract fixpoint on a (possibly refined) initial set."""
        engine = FixpointEngine(
            transfer_fn=self._transfer_fn,
            **self._fixpoint_kwargs,
        )
        return engine.compute(initial)

    def _intersects_unsafe(self, zonotope: Zonotope) -> bool:
        """Check whether *zonotope* intersects the unsafe region {aᵀx ≥ b}.

        We compute ``max_{x ∈ Z} aᵀx`` via the support function.  If
        the maximum is at least *b*, the zonotope might contain unsafe
        points.

        Soundness: if the support function is less than *b*, no point
        in the zonotope can satisfy ``aᵀx ≥ b``, so the region is safe.
        """
        sup = zonotope.support_function(self._unsafe_normal)
        return sup >= self._unsafe_bound

    def _extract_counterexample(self, zonotope: Zonotope) -> np.ndarray:
        """Extract a concrete candidate counterexample from the fixpoint.

        We maximise the unsafe direction ``aᵀx`` over the zonotope;
        the maximiser is the most likely point to be a real
        counterexample.

        Returns
        -------
        np.ndarray
            Candidate counterexample point.
        """
        _, point = zonotope.maximize(self._unsafe_normal)
        return point

    def _check_spurious(
        self,
        zonotope: Zonotope,
        candidate: np.ndarray,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Check whether *candidate* is a real or spurious counterexample.

        First checks the candidate point itself, then samples the
        intersection of the zonotope with the unsafe region.

        Returns
        -------
        (is_real, witness)
            ``is_real`` is ``True`` if a concrete unsafe point was found.
        """
        # Direct check on the candidate point.
        if self._checker.check_point(candidate):
            return True, candidate.copy()

        # Sample the intersection of the zonotope with the unsafe region
        # to look for other potentially real counterexamples.
        unsafe_meet = zonotope.meet_halfspace(
            self._unsafe_normal, self._unsafe_bound
        )
        return self._checker.check_zonotope(unsafe_meet)

    def _refine_abstraction(
        self,
        initial: Zonotope,
        spurious_point: np.ndarray,
    ) -> Tuple[Zonotope, Zonotope, int, float]:
        """Refine the initial zonotope by splitting on the spurious point.

        Returns
        -------
        (left, right, dim, split_value)
        """
        return self._refinement.refine(initial, spurious_point)

    def _budget_exhausted(
        self,
        start_time: float,
        refinements: int,
        worklist_size: int,
    ) -> bool:
        """Return ``True`` if any resource budget has been exceeded."""
        if refinements >= self._max_refinements:
            return True
        if worklist_size > self._max_splits:
            return True
        if time.monotonic() - start_time > self._timeout_s:
            return True
        return False


# ---------------------------------------------------------------------------
# Compositional CEGAR for multi-agent groups
# ---------------------------------------------------------------------------


class CompositionalCEGARVerifier:
    """CEGAR over independent multi-agent interaction groups.

    For a system decomposed into *k* interaction groups, each group is
    verified independently.  If any group reports ``UNSAFE``, the whole
    system is ``UNSAFE``.  If all groups report ``SAFE``, the system is
    ``SAFE``.

    Compositional soundness follows from the fact that each group's
    reachable set is over-approximated independently: if no individual
    group intersects the unsafe region, the joint reachable set (a
    subset of the Cartesian product of the individual reachable sets)
    does not either.

    Parameters
    ----------
    group_verifiers : dict
        Mapping from group identifier to ``CEGARVerifier``.
    """

    def __init__(
        self,
        group_verifiers: Dict[str, CEGARVerifier],
    ) -> None:
        self._verifiers = group_verifiers

    def verify_all(
        self,
        group_initials: Dict[str, Zonotope],
    ) -> Dict[str, CEGARResult]:
        """Verify each group independently.

        Parameters
        ----------
        group_initials : dict
            Mapping from group identifier to the initial zonotope for
            that group.

        Returns
        -------
        dict mapping group_id -> CEGARResult
        """
        results: Dict[str, CEGARResult] = {}
        for gid, verifier in self._verifiers.items():
            if gid not in group_initials:
                logger.warning("No initial zonotope for group %s — skipping", gid)
                continue
            logger.info("CEGAR: verifying group %s", gid)
            results[gid] = verifier.verify(group_initials[gid])
        return results

    def combined_verdict(
        self, results: Dict[str, CEGARResult]
    ) -> Verdict:
        """Combine per-group verdicts into a system-level verdict.

        Returns
        -------
        Verdict
            ``SAFE`` iff all groups are ``SAFE``.
            ``UNSAFE`` if any group is ``UNSAFE``.
            ``UNKNOWN`` otherwise.
        """
        if any(r.verdict == Verdict.UNSAFE for r in results.values()):
            return Verdict.UNSAFE
        if all(r.verdict == Verdict.SAFE for r in results.values()):
            return Verdict.SAFE
        return Verdict.UNKNOWN

    def combined_result(
        self, results: Dict[str, CEGARResult]
    ) -> CEGARResult:
        """Merge per-group results into a single ``CEGARResult``.

        The counterexample (if any) is taken from the first ``UNSAFE``
        group.  Statistics are aggregated.
        """
        verdict = self.combined_verdict(results)
        counterexample: Optional[np.ndarray] = None
        total_refinements = 0
        all_history: List[RefinementRecord] = []
        total_time = 0.0
        all_fp: List[FixpointResult] = []

        for r in results.values():
            total_refinements += r.refinement_iterations
            all_history.extend(r.refinement_history)
            total_time += r.total_time_s
            all_fp.extend(r.fixpoint_results)
            if r.verdict == Verdict.UNSAFE and counterexample is None:
                counterexample = r.counterexample

        return CEGARResult(
            verdict=verdict,
            counterexample=counterexample,
            refinement_iterations=total_refinements,
            refinement_history=all_history,
            total_time_s=total_time,
            fixpoint_results=all_fp,
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def make_cegar_verifier(
    transfer_fn: Callable[[Zonotope], Zonotope],
    concrete_evaluator: Callable[[np.ndarray], np.ndarray],
    safety_predicate: Callable[[np.ndarray], bool],
    unsafe_halfspace: Tuple[np.ndarray, float],
    *,
    strategy: RefinementStrategy = RefinementStrategy.COUNTEREXAMPLE,
    max_refinements: int = 20,
    max_splits: int = 64,
    timeout_s: float = 300.0,
    num_samples: int = 64,
    fixpoint_kwargs: Optional[Dict[str, Any]] = None,
) -> CEGARVerifier:
    """Build a ``CEGARVerifier`` with sensible defaults.

    Parameters
    ----------
    transfer_fn : callable
        ``Zonotope -> Zonotope`` transfer function.
    concrete_evaluator : callable
        ``np.ndarray -> np.ndarray`` concrete policy evaluation.
    safety_predicate : callable
        ``np.ndarray -> bool`` — ``True`` if state is *unsafe*.
    unsafe_halfspace : (np.ndarray, float)
        ``(a, b)`` defining unsafe region ``{x : aᵀx ≥ b}``.
    strategy : RefinementStrategy
        Splitting strategy.
    max_refinements : int
        Maximum number of refinement iterations.
    max_splits : int
        Maximum worklist size.
    timeout_s : float
        Timeout in seconds.
    num_samples : int
        Number of samples for spuriousness checking.
    fixpoint_kwargs : dict
        Forwarded to ``FixpointEngine``.

    Returns
    -------
    CEGARVerifier
    """
    checker = SpuriousnessChecker(
        concrete_evaluator=concrete_evaluator,
        safety_predicate=safety_predicate,
        num_samples=num_samples,
    )
    refinement = AbstractionRefinement(
        strategy=strategy,
        safety_normal=unsafe_halfspace[0],
    )
    return CEGARVerifier(
        transfer_fn=transfer_fn,
        spuriousness_checker=checker,
        refinement=refinement,
        unsafe_halfspace=unsafe_halfspace,
        max_refinements=max_refinements,
        max_splits=max_splits,
        timeout_s=timeout_s,
        fixpoint_kwargs=fixpoint_kwargs,
    )
