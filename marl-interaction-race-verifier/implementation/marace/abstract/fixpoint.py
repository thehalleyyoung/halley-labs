"""
Fixpoint computation engine for MARACE.

Computes abstract fixpoints for iterative systems (e.g., multi-step MARL
interaction) using widening to ensure convergence. Supports parallel
fixpoint computation for independent interaction groups and provides
sound over-approximations even when the fixpoint has not converged.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from marace.abstract.zonotope import Zonotope
from marace.abstract.hb_constraints import (
    ConsistencyChecker,
    ConstraintStrengthening,
    HBConstraintSet,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Widening strategy
# ---------------------------------------------------------------------------


class WideningStrategy(Enum):
    """Strategy for widening during fixpoint iteration."""
    STANDARD = auto()     # Widen at every iteration
    DELAYED = auto()      # Wait N iterations before widening
    THRESHOLD = auto()    # Widen only when growth exceeds threshold


# ---------------------------------------------------------------------------
# Iteration state
# ---------------------------------------------------------------------------


@dataclass
class IterationState:
    """State of a single fixpoint iteration."""

    iteration: int
    element: Zonotope
    volume_bound: float
    hausdorff_from_prev: float
    converged: bool
    wall_time_s: float
    hb_consistent: bool = True
    num_generators: int = 0

    def __post_init__(self) -> None:
        self.num_generators = self.element.num_generators


# ---------------------------------------------------------------------------
# Convergence checker
# ---------------------------------------------------------------------------


class ConvergenceChecker:
    """Detect when fixpoint is reached.

    Uses the Hausdorff-distance upper bound between successive iterates.
    Convergence is declared when the distance falls below *threshold* for
    *patience* consecutive iterations.
    """

    def __init__(self, threshold: float = 1e-6, patience: int = 2) -> None:
        self.threshold = threshold
        self.patience = patience
        self._consecutive_below = 0
        self._distances: List[float] = []

    def check(self, prev: Zonotope, curr: Zonotope) -> Tuple[bool, float]:
        """Check convergence between two successive iterates.

        Returns
        -------
        (converged, distance)
        """
        dist = prev.hausdorff_upper_bound(curr)
        self._distances.append(dist)

        if dist <= self.threshold:
            self._consecutive_below += 1
        else:
            self._consecutive_below = 0

        converged = self._consecutive_below >= self.patience
        return converged, dist

    def reset(self) -> None:
        self._consecutive_below = 0
        self._distances.clear()

    @property
    def distance_history(self) -> List[float]:
        return list(self._distances)


# ---------------------------------------------------------------------------
# Bounded ascending chain
# ---------------------------------------------------------------------------


class BoundedAscendingChain:
    """For ReLU networks, track the ascending chain and bound its length.

    In a ReLU network with n neurons, the number of distinct activation
    patterns is at most 2^n, bounding the ascending chain length in the
    zonotope lattice (modulo widening).  This tracker monitors the chain
    and provides termination guarantees.
    """

    def __init__(self, max_length: int, num_relu_neurons: int = 0) -> None:
        self.max_length = max_length
        self.num_relu_neurons = num_relu_neurons
        self._chain: List[Zonotope] = []
        self._activation_patterns: List[Optional[np.ndarray]] = []

    def add(self, element: Zonotope, activation_pattern: Optional[np.ndarray] = None) -> None:
        self._chain.append(element)
        self._activation_patterns.append(activation_pattern)

    @property
    def length(self) -> int:
        return len(self._chain)

    @property
    def terminated(self) -> bool:
        """True if the chain has reached its theoretical bound."""
        return self.length >= self.max_length

    def unique_activation_patterns(self) -> int:
        """Count distinct activation patterns seen so far."""
        patterns = [p for p in self._activation_patterns if p is not None]
        if not patterns:
            return 0
        seen: set = set()
        for p in patterns:
            key = tuple(p.tolist())
            seen.add(key)
        return len(seen)

    def is_monotone(self) -> bool:
        """Check if the chain is monotonically increasing (by bounding box)."""
        for i in range(1, len(self._chain)):
            lo_prev, hi_prev = self._chain[i - 1].bounding_box()[:, 0], self._chain[i - 1].bounding_box()[:, 1]
            lo_curr, hi_curr = self._chain[i].bounding_box()[:, 0], self._chain[i].bounding_box()[:, 1]
            if np.any(lo_curr > lo_prev + 1e-10) or np.any(hi_curr < hi_prev - 1e-10):
                return False
        return True


# ---------------------------------------------------------------------------
# Fixpoint result
# ---------------------------------------------------------------------------


@dataclass
class FixpointResult:
    """Result of a fixpoint computation."""

    element: Zonotope
    converged: bool
    iterations: int
    iteration_history: List[IterationState]
    wall_time_s: float
    hb_consistent: bool = True
    final_hausdorff: float = 0.0

    @property
    def per_iteration_bounds(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Bounding box at each iteration."""
        return [s.element.bounding_box() for s in self.iteration_history]

    @property
    def volume_history(self) -> List[float]:
        return [s.volume_bound for s in self.iteration_history]

    def summary(self) -> str:
        status = "CONVERGED" if self.converged else "NOT CONVERGED"
        lines = [
            f"Fixpoint {status} after {self.iterations} iterations "
            f"({self.wall_time_s:.3f}s)",
            f"  Final element: {self.element}",
            f"  HB-consistent: {self.hb_consistent}",
            f"  Final Hausdorff: {self.final_hausdorff:.6g}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sound over-approximation
# ---------------------------------------------------------------------------


class SoundOverApproximation:
    """Even without fixpoint convergence, provide sound per-iteration bounds.

    At each iteration k, the element Z_k over-approximates all reachable
    states up to k steps. The join of all Z_k up to the current iteration
    is a sound over-approximation of all reachable states.
    """

    def __init__(self) -> None:
        self._iterates: List[Zonotope] = []

    def add(self, z: Zonotope) -> None:
        self._iterates.append(z)

    @property
    def num_iterates(self) -> int:
        return len(self._iterates)

    def cumulative_bound(self) -> Optional[Zonotope]:
        """Return the join of all iterates (sound over-approximation)."""
        if not self._iterates:
            return None
        result = self._iterates[0].copy()
        for z in self._iterates[1:]:
            result = result.join(z)
        return result

    def per_iteration_bounds(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Bounding box at each iteration."""
        return [z.bounding_box() for z in self._iterates]

    def is_monotone_inclusion(self) -> bool:
        """Check if Z_k ⊆ Z_{k+1} for all k (ascending chain)."""
        for i in range(1, len(self._iterates)):
            lo_prev, hi_prev = self._iterates[i - 1].bounding_box()[:, 0], self._iterates[i - 1].bounding_box()[:, 1]
            lo_curr, hi_curr = self._iterates[i].bounding_box()[:, 0], self._iterates[i].bounding_box()[:, 1]
            if np.any(lo_curr > lo_prev + 1e-10) or np.any(hi_curr < hi_prev - 1e-10):
                return False
        return True


# ---------------------------------------------------------------------------
# Fixpoint engine
# ---------------------------------------------------------------------------


class FixpointEngine:
    """Main fixpoint computation engine.

    Computes the abstract fixpoint of a transfer function F starting from
    an initial abstract element Z₀:

        Z_{k+1} = Z_k ∇ F(Z_k)

    where ∇ is the widening operator.
    """

    def __init__(
        self,
        transfer_fn: Callable[[Zonotope], Zonotope],
        strategy: WideningStrategy = WideningStrategy.DELAYED,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        convergence_patience: int = 2,
        delay_widening: int = 3,
        widening_threshold: float = 1.05,
        max_generators: Optional[int] = None,
        hb_constraints: Optional[HBConstraintSet] = None,
        strengthen_constraints: bool = False,
    ) -> None:
        self._transfer_fn = transfer_fn
        self._strategy = strategy
        self._max_iterations = max_iterations
        self._convergence = ConvergenceChecker(
            threshold=convergence_threshold,
            patience=convergence_patience,
        )
        self._delay = delay_widening
        self._widening_threshold = widening_threshold
        self._max_generators = max_generators
        self._hb_constraints = hb_constraints or HBConstraintSet()
        self._strengthen = strengthen_constraints
        self._strengthener: Optional[ConstraintStrengthening] = None
        if strengthen_constraints and len(self._hb_constraints) > 0:
            self._strengthener = ConstraintStrengthening(self._hb_constraints)

    def compute(self, initial: Zonotope) -> FixpointResult:
        """Run fixpoint iteration starting from *initial*.

        Returns a FixpointResult with the final abstract element and
        convergence information.
        """
        start_time = time.monotonic()
        self._convergence.reset()

        current = initial.copy()
        history: List[IterationState] = []
        sound_approx = SoundOverApproximation()

        converged = False
        final_dist = float("inf")

        for k in range(self._max_iterations):
            iter_start = time.monotonic()

            # Apply transfer function
            next_elem = self._transfer_fn(current)

            # Apply HB constraints
            active_constraints = self._hb_constraints
            if self._strengthener is not None:
                active_constraints = self._strengthener.strengthen_from_zonotope(
                    next_elem, k
                )

            for c in active_constraints:
                if c.normal.shape[0] == next_elem.dimension:
                    next_elem = next_elem.meet_halfspace(c.normal, c.bound)

            # Widening decision
            should_widen = self._should_widen(k, current, next_elem)

            if should_widen:
                widened = current.widening(next_elem, threshold=self._widening_threshold)
                next_elem = widened

            # Generator reduction
            if (self._max_generators is not None
                    and next_elem.num_generators > self._max_generators):
                next_elem = next_elem.reduce_generators(self._max_generators)

            # Check convergence
            converged, dist = self._convergence.check(current, next_elem)
            final_dist = dist

            # HB consistency check
            hb_ok = True
            if len(self._hb_constraints) > 0:
                hb_ok, _ = ConsistencyChecker.check_all(
                    next_elem, self._hb_constraints
                )

            iter_time = time.monotonic() - iter_start
            state = IterationState(
                iteration=k,
                element=next_elem.copy(),
                volume_bound=next_elem.volume_bound,
                hausdorff_from_prev=dist,
                converged=converged,
                wall_time_s=iter_time,
                hb_consistent=hb_ok,
            )
            history.append(state)
            sound_approx.add(next_elem.copy())

            logger.debug(
                "Fixpoint iter %d: dist=%.6g, vol=%.6g, gens=%d, hb=%s",
                k, dist, next_elem.volume_bound, next_elem.num_generators,
                hb_ok,
            )

            if converged:
                logger.info("Fixpoint converged at iteration %d", k)
                break

            current = next_elem

        total_time = time.monotonic() - start_time

        # If not converged, use sound over-approximation
        if not converged:
            final_elem = sound_approx.cumulative_bound()
            if final_elem is None:
                final_elem = current
            logger.warning(
                "Fixpoint did not converge in %d iterations; "
                "returning sound over-approximation",
                self._max_iterations,
            )
        else:
            final_elem = current if not history else history[-1].element

        hb_final = True
        if len(self._hb_constraints) > 0:
            hb_final, _ = ConsistencyChecker.check_all(
                final_elem, self._hb_constraints
            )

        return FixpointResult(
            element=final_elem,
            converged=converged,
            iterations=len(history),
            iteration_history=history,
            wall_time_s=total_time,
            hb_consistent=hb_final,
            final_hausdorff=final_dist,
        )

    def _should_widen(self, k: int, prev: Zonotope, curr: Zonotope) -> bool:
        """Decide whether to apply widening at iteration k."""
        if self._strategy == WideningStrategy.STANDARD:
            return True

        if self._strategy == WideningStrategy.DELAYED:
            return k >= self._delay

        if self._strategy == WideningStrategy.THRESHOLD:
            lo_p, hi_p = prev.bounding_box()[:, 0], prev.bounding_box()[:, 1]
            lo_c, hi_c = curr.bounding_box()[:, 0], curr.bounding_box()[:, 1]
            growth = np.max(np.maximum(lo_p - lo_c, hi_c - hi_p))
            return growth > self._widening_threshold

        return True


# ---------------------------------------------------------------------------
# Parallel fixpoint
# ---------------------------------------------------------------------------


class ParallelFixpoint:
    """Run fixpoints for multiple interaction groups in parallel.

    Each interaction group has its own transfer function and initial
    element, but they may share HB constraints. Results are combined
    into a joint fixpoint result.
    """

    def __init__(
        self,
        max_workers: int = 4,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._max_workers = max_workers
        self._engine_kwargs = engine_kwargs or {}

    def compute_parallel(
        self,
        groups: Dict[str, Tuple[Callable[[Zonotope], Zonotope], Zonotope]],
        hb_constraints: Optional[Dict[str, HBConstraintSet]] = None,
    ) -> Dict[str, FixpointResult]:
        """Run fixpoint for each group in parallel.

        Parameters
        ----------
        groups : dict mapping group_id -> (transfer_fn, initial_element)
        hb_constraints : dict mapping group_id -> HBConstraintSet

        Returns
        -------
        dict mapping group_id -> FixpointResult
        """
        hb_constraints = hb_constraints or {}
        results: Dict[str, FixpointResult] = {}

        def _run_group(group_id: str) -> Tuple[str, FixpointResult]:
            transfer_fn, initial = groups[group_id]
            kwargs = dict(self._engine_kwargs)
            if group_id in hb_constraints:
                kwargs["hb_constraints"] = hb_constraints[group_id]
            engine = FixpointEngine(transfer_fn=transfer_fn, **kwargs)
            result = engine.compute(initial)
            return group_id, result

        if self._max_workers <= 1 or len(groups) <= 1:
            # Sequential fallback
            for gid in groups:
                gid_out, res = _run_group(gid)
                results[gid_out] = res
        else:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    executor.submit(_run_group, gid): gid for gid in groups
                }
                for future in as_completed(futures):
                    gid, res = future.result()
                    results[gid] = res

        return results

    @staticmethod
    def combine_results(
        results: Dict[str, FixpointResult],
        group_dims: Dict[str, List[int]],
        total_dim: int,
    ) -> Zonotope:
        """Combine per-group fixpoint results into a joint zonotope.

        Parameters
        ----------
        results : per-group fixpoint results
        group_dims : mapping from group_id to dimension indices in the joint space
        total_dim : total dimension of the joint state
        """
        center = np.zeros(total_dim)
        gen_blocks: List[np.ndarray] = []

        for gid, result in results.items():
            dims = group_dims[gid]
            z = result.element
            for i, d in enumerate(dims):
                center[d] = z.center[i]

            # Embed generators in full space
            block = np.zeros((total_dim, z.num_generators))
            for i, d in enumerate(dims):
                block[d, :] = z.generators[i, :]
            gen_blocks.append(block)

        if gen_blocks:
            generators = np.hstack(gen_blocks)
        else:
            generators = np.zeros((total_dim, 0))

        return Zonotope(center=center, generators=generators)


# ---------------------------------------------------------------------------
# Acceleration: extrapolation-based widening
# ---------------------------------------------------------------------------


def extrapolation_widening(
    z_prev: Zonotope,
    z_curr: Zonotope,
    z_next: Zonotope,
    acceleration_factor: float = 2.0,
) -> Zonotope:
    """Extrapolation-based widening for faster convergence.

    Given three successive iterates Z_{k-1}, Z_k, Z_{k+1}, extrapolate
    the trend to jump ahead:

        Z_accel = Z_{k+1} + α * (Z_{k+1} - Z_k)

    where α is the acceleration factor. The result is then joined with
    Z_{k+1} to maintain soundness.

    This can significantly reduce the number of iterations needed to
    reach a fixpoint, at the cost of a larger over-approximation.
    """
    if (z_prev.dimension != z_curr.dimension
            or z_curr.dimension != z_next.dimension):
        raise ValueError("All zonotopes must have the same dimension")

    # Compute trend from bounding boxes
    lo_curr, hi_curr = z_curr.bounding_box()[:, 0], z_curr.bounding_box()[:, 1]
    lo_next, hi_next = z_next.bounding_box()[:, 0], z_next.bounding_box()[:, 1]

    delta_lo = lo_next - lo_curr
    delta_hi = hi_next - hi_curr

    # Extrapolate
    extrap_lo = lo_next + acceleration_factor * np.minimum(delta_lo, 0.0)
    extrap_hi = hi_next + acceleration_factor * np.maximum(delta_hi, 0.0)

    # Ensure the extrapolation contains z_next
    extrap_lo = np.minimum(extrap_lo, lo_next)
    extrap_hi = np.maximum(extrap_hi, hi_next)

    z_extrap = Zonotope.from_interval(extrap_lo, extrap_hi)
    return z_next.join(z_extrap)
