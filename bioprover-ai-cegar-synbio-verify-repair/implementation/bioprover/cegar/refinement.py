"""
Biology-aware refinement strategies for CEGAR.

Implements structural (Hill threshold, nullcline, eigenspace),
monotonicity-based, time-scale, interpolation, and simulation-guided
refinement strategies, plus a combinator for composing them.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from bioprover.encoding.expression import (
    Add,
    And,
    Const,
    Div,
    ExprNode,
    Ge,
    Gt,
    Interval,
    Le,
    Lt,
    Mul,
    Neg,
    Not,
    Or,
    Var,
    HillAct,
    HillRep,
    Eq,
    collect_nodes,
)
from bioprover.cegar.abstraction import (
    AbstractState,
    AbstractionDomain,
    IntervalBox,
)
from bioprover.cegar.counterexample import (
    AbstractCounterexample,
    SpuriousnessResult,
)
from bioprover.cegar.predicate_manager import (
    Predicate,
    PredicateOrigin,
    PredicateSet,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Refinement result
# ---------------------------------------------------------------------------


@dataclass
class RefinementResult:
    """Outcome of a refinement step."""

    new_predicates: List[Predicate]
    states_refined: List[int]
    strategy_name: str = ""
    time_elapsed: float = 0.0
    success: bool = True
    message: str = ""

    @property
    def predicate_count(self) -> int:
        return len(self.new_predicates)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "new_predicates": self.predicate_count,
            "states_refined": len(self.states_refined),
            "time_s": round(self.time_elapsed, 3),
            "success": self.success,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Base strategy
# ---------------------------------------------------------------------------


class RefinementStrategy(ABC):
    """Base class for CEGAR refinement strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def refine(
        self,
        cex: AbstractCounterexample,
        analysis: SpuriousnessResult,
        domain: AbstractionDomain,
        iteration: int,
        **kwargs: Any,
    ) -> RefinementResult:
        """Produce predicates / refinement actions for a spurious cex."""
        ...

    def applicable(
        self,
        cex: AbstractCounterexample,
        analysis: SpuriousnessResult,
    ) -> bool:
        """Check whether this strategy is applicable to the given cex."""
        return analysis.is_spurious


# ---------------------------------------------------------------------------
# 1. Structural refinement (Hill threshold, nullcline, eigenspace)
# ---------------------------------------------------------------------------


class StructuralRefinement(RefinementStrategy):
    """Add predicates derived from the biological model structure.

    * Hill threshold predicates:  ``x > K`` for each Hill function H(x,K,n)
    * Nullcline predicates:  ``f_i(x) = 0`` for each species i
    * Eigenspace predicates from the Jacobian at steady states
    """

    def __init__(
        self,
        rhs: Dict[str, ExprNode],
        hill_params: Optional[List[Dict[str, Any]]] = None,
        steady_states: Optional[List[Dict[str, float]]] = None,
        jacobian: Optional[Callable[[Dict[str, float]], List[List[float]]]] = None,
    ) -> None:
        self._rhs = rhs
        self._hill_params = hill_params or []
        self._steady_states = steady_states or []
        self._jacobian = jacobian
        self._used_thresholds: Set[Tuple[str, float]] = set()

    @property
    def name(self) -> str:
        return "structural"

    def refine(
        self,
        cex: AbstractCounterexample,
        analysis: SpuriousnessResult,
        domain: AbstractionDomain,
        iteration: int,
        **kwargs: Any,
    ) -> RefinementResult:
        t0 = time.monotonic()
        preds: List[Predicate] = []

        # Identify failure region
        failure_idx = analysis.failure_index
        if failure_idx is None:
            failure_idx = cex.length // 2
        failure_state = cex.path[min(failure_idx, cex.length - 1)]

        # 1. Hill threshold predicates
        preds.extend(self._hill_threshold_predicates(failure_state, iteration))

        # 2. Nullcline predicates
        preds.extend(self._nullcline_predicates(failure_state, iteration))

        # 3. Eigenspace predicates
        preds.extend(self._eigenspace_predicates(failure_state, iteration))

        states_refined: List[int] = []
        if preds:
            new_states = domain.refine(failure_state, preds)
            states_refined = [s.state_id for s in new_states]

        elapsed = time.monotonic() - t0
        return RefinementResult(
            new_predicates=preds,
            states_refined=states_refined,
            strategy_name=self.name,
            time_elapsed=elapsed,
            success=len(preds) > 0,
            message=f"Generated {len(preds)} structural predicates",
        )

    # -- Hill thresholds ----------------------------------------------------

    def _hill_threshold_predicates(
        self,
        state: AbstractState,
        iteration: int,
    ) -> List[Predicate]:
        preds: List[Predicate] = []
        var_names = state.box.variable_names

        for hf in self._hill_params:
            species = hf.get("species", "")
            k_val = hf.get("K", 1.0)
            if species not in var_names:
                continue

            key = (species, k_val)
            if key in self._used_thresholds:
                continue

            iv = state.box.dimensions.get(species)
            if iv is None:
                continue

            # Only add if threshold K falls within the state's interval
            if iv.lo <= k_val <= iv.hi:
                self._used_thresholds.add(key)
                preds.append(
                    Predicate(
                        expr=Gt(Var(species), Const(k_val)),
                        name=f"{species}>K({k_val})",
                        origin=PredicateOrigin.HILL_THRESHOLD,
                        iteration_added=iteration,
                        info={"K": k_val, "species": species},
                    )
                )

        # Also scan the RHS expressions for Hill functions
        for sp, rhs_expr in self._rhs.items():
            hill_nodes = collect_nodes(
                rhs_expr, lambda n: isinstance(n, (HillAct, HillRep))
            )
            for hn in hill_nodes:
                if isinstance(hn, (HillAct, HillRep)):
                    # Extract K from the Hill node children
                    # HillAct/HillRep: children = (activator/repressor, K, n)
                    if len(hn.children) >= 2:
                        k_node = hn.children[1]
                        if isinstance(k_node, Const):
                            k_val = k_node.value
                            # Find the activator/repressor variable
                            act_node = hn.children[0]
                            if isinstance(act_node, Var):
                                key = (act_node.name, k_val)
                                if key not in self._used_thresholds:
                                    iv = state.box.dimensions.get(act_node.name)
                                    if iv and iv.lo <= k_val <= iv.hi:
                                        self._used_thresholds.add(key)
                                        preds.append(
                                            Predicate(
                                                expr=Gt(Var(act_node.name), Const(k_val)),
                                                name=f"{act_node.name}>K({k_val})",
                                                origin=PredicateOrigin.HILL_THRESHOLD,
                                                iteration_added=iteration,
                                                info={"K": k_val, "species": act_node.name},
                                            )
                                        )

        return preds

    # -- Nullcline predicates -----------------------------------------------

    def _nullcline_predicates(
        self,
        state: AbstractState,
        iteration: int,
    ) -> List[Predicate]:
        preds: List[Predicate] = []

        for sp, rhs_expr in self._rhs.items():
            if sp not in state.box.variable_names:
                continue

            # f_i(x) > 0  at the midpoint?  Add nullcline predicate.
            midpoint = state.box.midpoint()
            try:
                from bioprover.cegar.counterexample import _eval_expr_float

                val = _eval_expr_float(rhs_expr, midpoint)
                # If sign changes across the box, add nullcline predicate
                vertices = state.box.vertices()
                signs = set()
                for v in vertices[:8]:  # limit evaluation
                    try:
                        signs.add(_eval_expr_float(rhs_expr, v) > 0)
                    except Exception:
                        pass
                if len(signs) > 1:
                    preds.append(
                        Predicate(
                            expr=Gt(rhs_expr, Const(0.0)),
                            name=f"d{sp}/dt>0",
                            origin=PredicateOrigin.NULLCLINE,
                            iteration_added=iteration,
                            info={"species": sp},
                        )
                    )
            except Exception:
                pass

        return preds

    # -- Eigenspace predicates ----------------------------------------------

    def _eigenspace_predicates(
        self,
        state: AbstractState,
        iteration: int,
    ) -> List[Predicate]:
        if self._jacobian is None or not self._steady_states:
            return []

        preds: List[Predicate] = []
        var_names = state.box.variable_names

        for ss in self._steady_states:
            # Check if steady state is near this abstract state
            if not state.box.contains_point(ss):
                continue

            try:
                jac = self._jacobian(ss)
                # Compute eigenvalues/eigenvectors
                import numpy as np

                jac_arr = np.array(jac)
                eigenvalues, eigenvectors = np.linalg.eig(jac_arr)

                # For each real eigenvector, create a hyperplane predicate
                for i, (eigval, eigvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
                    if not np.isreal(eigval):
                        continue
                    eigvec_real = np.real(eigvec)
                    # Predicate: eigvec · (x - ss) > 0
                    terms: List[ExprNode] = []
                    for j, var in enumerate(var_names):
                        if j < len(eigvec_real):
                            coeff = float(eigvec_real[j])
                            ss_val = ss.get(var, 0.0)
                            terms.append(
                                Mul(Const(coeff), Add(Var(var), Neg(Const(ss_val))))
                            )

                    if terms:
                        sum_expr = terms[0]
                        for t in terms[1:]:
                            sum_expr = Add(sum_expr, t)
                        preds.append(
                            Predicate(
                                expr=Gt(sum_expr, Const(0.0)),
                                name=f"eigvec_{i}·(x-ss)>0",
                                origin=PredicateOrigin.EIGENSPACE,
                                iteration_added=iteration,
                                info={"eigenvalue": float(np.real(eigval))},
                            )
                        )
            except Exception as exc:
                logger.debug("Eigenspace predicate computation failed: %s", exc)

        return preds


# ---------------------------------------------------------------------------
# 2. Monotonicity refinement
# ---------------------------------------------------------------------------


class MonotonicityRefinement(RefinementStrategy):
    """Exploit monotonicity in gene regulatory networks.

    For monotone subsystems, only check vertices of interval boxes.
    Uses binary search refinement on each monotone dimension
    independently, reducing refinement from exponential to logarithmic
    in precision.
    """

    def __init__(
        self,
        rhs: Dict[str, ExprNode],
        monotone_info: Optional[Dict[str, Dict[str, int]]] = None,
        max_splits_per_dim: int = 10,
    ) -> None:
        """
        Args:
            rhs: ODE right-hand sides.
            monotone_info: Map species → {dep_species: +1/-1} indicating
                monotone dependency direction (+1 = increasing, -1 = decreasing).
            max_splits_per_dim: Maximum binary search depth per dimension.
        """
        self._rhs = rhs
        self._monotone_info = monotone_info or {}
        self._max_splits = max_splits_per_dim

    @property
    def name(self) -> str:
        return "monotonicity"

    def applicable(
        self,
        cex: AbstractCounterexample,
        analysis: SpuriousnessResult,
    ) -> bool:
        return analysis.is_spurious and len(self._monotone_info) > 0

    def refine(
        self,
        cex: AbstractCounterexample,
        analysis: SpuriousnessResult,
        domain: AbstractionDomain,
        iteration: int,
        **kwargs: Any,
    ) -> RefinementResult:
        t0 = time.monotonic()
        preds: List[Predicate] = []

        failure_idx = analysis.failure_index or (cex.length // 2)
        failure_state = cex.path[min(failure_idx, cex.length - 1)]

        # For each monotone dimension, do binary search for the
        # tightest threshold that separates genuine from spurious
        for species, deps in self._monotone_info.items():
            iv = failure_state.box.dimensions.get(species)
            if iv is None:
                continue

            threshold = self._binary_search_threshold(
                species, iv, failure_state, deps, iteration
            )
            if threshold is not None:
                preds.append(
                    Predicate(
                        expr=Gt(Var(species), Const(threshold)),
                        name=f"mono_{species}>{threshold:.4g}",
                        origin=PredicateOrigin.MONOTONICITY,
                        iteration_added=iteration,
                        info={
                            "species": species,
                            "threshold": threshold,
                            "monotone_deps": deps,
                        },
                    )
                )

        states_refined: List[int] = []
        if preds:
            new_states = domain.refine(failure_state, preds)
            states_refined = [s.state_id for s in new_states]

        elapsed = time.monotonic() - t0
        return RefinementResult(
            new_predicates=preds,
            states_refined=states_refined,
            strategy_name=self.name,
            time_elapsed=elapsed,
            success=len(preds) > 0,
            message=f"Monotonicity: {len(preds)} threshold predicates",
        )

    def _binary_search_threshold(
        self,
        species: str,
        interval: Interval,
        state: AbstractState,
        deps: Dict[str, int],
        iteration: int,
    ) -> Optional[float]:
        """Binary search for threshold separating behaviour regions.

        For a monotone system, the sign of the RHS can only change once
        as we move along a monotone dimension → binary search works.
        """
        rhs_expr = self._rhs.get(species)
        if rhs_expr is None:
            return None

        from bioprover.cegar.counterexample import _eval_expr_float

        lo, hi = interval.lo, interval.hi
        midpoint_base = state.box.midpoint()

        # Evaluate at endpoints
        pt_lo = dict(midpoint_base)
        pt_lo[species] = lo
        pt_hi = dict(midpoint_base)
        pt_hi[species] = hi

        try:
            val_lo = _eval_expr_float(rhs_expr, pt_lo)
            val_hi = _eval_expr_float(rhs_expr, pt_hi)
        except Exception:
            return None

        # No sign change → no useful threshold
        if (val_lo > 0) == (val_hi > 0):
            return None

        # Binary search for zero crossing
        for _ in range(self._max_splits):
            mid = (lo + hi) / 2.0
            pt_mid = dict(midpoint_base)
            pt_mid[species] = mid
            try:
                val_mid = _eval_expr_float(rhs_expr, pt_mid)
            except Exception:
                return mid

            if (val_mid > 0) == (val_lo > 0):
                lo = mid
            else:
                hi = mid

        return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# 3. Time-scale refinement
# ---------------------------------------------------------------------------


class TimeScaleRefinement(RefinementStrategy):
    """Temporal splitting guided by time-scale separation.

    Instead of splitting state-space regions, split the time axis.
    Useful when dynamics have fast and slow modes (common in GRNs
    with mRNA–protein cascades).
    """

    def __init__(
        self,
        rhs: Dict[str, ExprNode],
        fast_species: Optional[Set[str]] = None,
        slow_species: Optional[Set[str]] = None,
        time_scale_ratio: float = 10.0,
    ) -> None:
        self._rhs = rhs
        self._fast = fast_species or set()
        self._slow = slow_species or set()
        self._ratio = time_scale_ratio

    @property
    def name(self) -> str:
        return "time_scale"

    def refine(
        self,
        cex: AbstractCounterexample,
        analysis: SpuriousnessResult,
        domain: AbstractionDomain,
        iteration: int,
        **kwargs: Any,
    ) -> RefinementResult:
        t0 = time.monotonic()
        preds: List[Predicate] = []

        failure_idx = analysis.failure_index or (cex.length // 2)

        # Classify species by time scale if not provided
        if not self._fast and not self._slow:
            self._classify_time_scales(cex.path[0].box.midpoint())

        # For fast species: add fine-grained temporal predicates
        for sp in self._fast:
            if sp not in cex.path[0].box.variable_names:
                continue
            # Add predicates that capture the quasi-steady-state approximation
            rhs_expr = self._rhs.get(sp)
            if rhs_expr is not None:
                preds.append(
                    Predicate(
                        expr=Lt(
                            _abs_expr(rhs_expr),
                            Const(1e-2),
                        ),
                        name=f"qssa_{sp}",
                        origin=PredicateOrigin.TIME_SCALE,
                        iteration_added=iteration,
                        info={
                            "species": sp,
                            "type": "quasi_steady_state",
                        },
                    )
                )

        # For slow species at the failure point: split spatially
        failure_state = cex.path[min(failure_idx, cex.length - 1)]
        for sp in self._slow:
            iv = failure_state.box.dimensions.get(sp)
            if iv is None or iv.width() < 1e-6:
                continue
            mid = iv.midpoint()
            preds.append(
                Predicate(
                    expr=Gt(Var(sp), Const(mid)),
                    name=f"slow_{sp}>{mid:.4g}",
                    origin=PredicateOrigin.TIME_SCALE,
                    iteration_added=iteration,
                    info={"species": sp, "threshold": mid},
                )
            )

        states_refined: List[int] = []
        if preds:
            new_states = domain.refine(failure_state, preds)
            states_refined = [s.state_id for s in new_states]

        elapsed = time.monotonic() - t0
        return RefinementResult(
            new_predicates=preds,
            states_refined=states_refined,
            strategy_name=self.name,
            time_elapsed=elapsed,
            success=len(preds) > 0,
            message=f"Time-scale: {len(preds)} predicates "
            f"(fast={len(self._fast)}, slow={len(self._slow)})",
        )

    def _classify_time_scales(self, point: Dict[str, float]) -> None:
        """Auto-classify species as fast or slow based on RHS magnitude."""
        from bioprover.cegar.counterexample import _eval_expr_float

        magnitudes: Dict[str, float] = {}
        for sp, rhs_expr in self._rhs.items():
            try:
                magnitudes[sp] = abs(_eval_expr_float(rhs_expr, point))
            except Exception:
                magnitudes[sp] = 0.0

        if not magnitudes:
            return

        median_mag = sorted(magnitudes.values())[len(magnitudes) // 2]
        threshold = self._ratio * median_mag if median_mag > 0 else 1.0

        for sp, mag in magnitudes.items():
            if mag > threshold:
                self._fast.add(sp)
            else:
                self._slow.add(sp)

        logger.debug(
            "Time-scale classification: fast=%s, slow=%s",
            self._fast,
            self._slow,
        )


# ---------------------------------------------------------------------------
# 4. Interpolation refinement
# ---------------------------------------------------------------------------


class InterpolationRefinement(RefinementStrategy):
    """Use Craig interpolants from UNSAT proofs to derive predicates.

    When the SMT solver proves a path infeasible, extract an
    interpolant at the failure point and convert it to a predicate.
    """

    def __init__(
        self,
        solver: Optional[Any] = None,
    ) -> None:
        self._solver = solver

    @property
    def name(self) -> str:
        return "interpolation"

    def applicable(
        self,
        cex: AbstractCounterexample,
        analysis: SpuriousnessResult,
    ) -> bool:
        return analysis.is_spurious and self._solver is not None

    def refine(
        self,
        cex: AbstractCounterexample,
        analysis: SpuriousnessResult,
        domain: AbstractionDomain,
        iteration: int,
        **kwargs: Any,
    ) -> RefinementResult:
        t0 = time.monotonic()
        preds: List[Predicate] = []

        failure_idx = analysis.failure_index
        if failure_idx is None:
            failure_idx = cex.length // 2

        # Build interpolation query at the failure point
        # Partition the path into A (prefix) and B (suffix)
        interpolant = self._compute_interpolant(cex, failure_idx)

        if interpolant is not None:
            preds.append(
                Predicate(
                    expr=interpolant,
                    name=f"itp@{failure_idx}",
                    origin=PredicateOrigin.INTERPOLATION,
                    iteration_added=iteration,
                    info={"failure_index": failure_idx},
                )
            )

        failure_state = cex.path[min(failure_idx, cex.length - 1)]
        states_refined: List[int] = []
        if preds:
            new_states = domain.refine(failure_state, preds)
            states_refined = [s.state_id for s in new_states]

        elapsed = time.monotonic() - t0
        return RefinementResult(
            new_predicates=preds,
            states_refined=states_refined,
            strategy_name=self.name,
            time_elapsed=elapsed,
            success=len(preds) > 0,
            message=f"Interpolation: {'computed' if preds else 'failed'}",
        )

    def _compute_interpolant(
        self,
        cex: AbstractCounterexample,
        failure_idx: int,
    ) -> Optional[ExprNode]:
        """Compute a Craig interpolant at the failure point.

        Encodes prefix constraints as A and suffix constraints as B.
        If A ∧ B is UNSAT, the interpolant I satisfies:
          A ⊨ I  and  I ∧ B is UNSAT
        using only variables shared between A and B.
        """
        if self._solver is None:
            return None

        try:
            # Check if the solver supports interpolation
            if not hasattr(self._solver, "get_interpolant"):
                # Fall back to extracting predicate from UNSAT core
                return self._interpolant_from_unsat_core(cex, failure_idx)

            var_names = cex.path[0].box.variable_names

            # Encode prefix (A)
            prefix_constraints: List[ExprNode] = []
            for step in range(failure_idx + 1):
                state = cex.path[step]
                for var in var_names:
                    iv = state.box.dimensions.get(var)
                    if iv:
                        x = Var(f"{var}_{step}")
                        prefix_constraints.append(Ge(x, Const(iv.lo)))
                        prefix_constraints.append(Le(x, Const(iv.hi)))

            # Encode suffix (B)
            suffix_constraints: List[ExprNode] = []
            for step in range(failure_idx, cex.length):
                state = cex.path[step]
                for var in var_names:
                    iv = state.box.dimensions.get(var)
                    if iv:
                        x = Var(f"{var}_{step}")
                        suffix_constraints.append(Ge(x, Const(iv.lo)))
                        suffix_constraints.append(Le(x, Const(iv.hi)))

            a_formula = And(*prefix_constraints) if prefix_constraints else Const(1.0)
            b_formula = And(*suffix_constraints) if suffix_constraints else Const(1.0)

            interpolant = self._solver.get_interpolant(a_formula, b_formula)
            return interpolant

        except Exception as exc:
            logger.debug("Interpolant computation failed: %s", exc)
            return None

    def _interpolant_from_unsat_core(
        self,
        cex: AbstractCounterexample,
        failure_idx: int,
    ) -> Optional[ExprNode]:
        """Approximate interpolant from UNSAT core.

        When true interpolation is unavailable, extract the constraints
        involved in the UNSAT core and form a conjunction.
        """
        try:
            self._solver.push()
            var_names = cex.path[0].box.variable_names

            # Assert all path constraints
            for step, state in enumerate(cex.path):
                for var in var_names:
                    iv = state.box.dimensions.get(var)
                    if iv:
                        x = Var(f"{var}_{step}")
                        self._solver.assert_formula(Ge(x, Const(iv.lo)))
                        self._solver.assert_formula(Le(x, Const(iv.hi)))

            result = self._solver.check_sat()
            if result.is_unsat:
                core = self._solver.get_unsat_core()
                self._solver.pop()
                if core:
                    return And(*core) if len(core) > 1 else core[0]
            else:
                self._solver.pop()
        except Exception:
            try:
                self._solver.pop()
            except Exception:
                pass
        return None


# ---------------------------------------------------------------------------
# 5. Simulation-guided refinement
# ---------------------------------------------------------------------------


class SimulationGuidedRefinement(RefinementStrategy):
    """Use ODE simulation to guide refinement.

    Run a concrete simulation from the counterexample's initial state
    and identify where the abstract path diverges from the concrete
    trajectory.  Add predicates that separate the concrete trajectory
    from the spurious abstract path.
    """

    def __init__(
        self,
        rhs: Dict[str, ExprNode],
        step_size: float = 0.01,
        num_samples: int = 5,
    ) -> None:
        self._rhs = rhs
        self._step_size = step_size
        self._num_samples = num_samples

    @property
    def name(self) -> str:
        return "simulation_guided"

    def refine(
        self,
        cex: AbstractCounterexample,
        analysis: SpuriousnessResult,
        domain: AbstractionDomain,
        iteration: int,
        **kwargs: Any,
    ) -> RefinementResult:
        t0 = time.monotonic()
        preds: List[Predicate] = []

        # Sample initial points from the initial abstract state
        init_state = cex.initial_state
        sample_points = self._sample_initial_points(init_state)

        var_names = init_state.box.variable_names
        divergence_points: List[Tuple[int, Dict[str, float], Dict[str, float]]] = []

        for pt in sample_points:
            trajectory = self._simulate(pt, var_names, cex.length)
            # Find where trajectory diverges from abstract path
            for step in range(1, min(len(trajectory), cex.length)):
                if not cex.path[step].box.contains_point(trajectory[step]):
                    divergence_points.append(
                        (step, trajectory[step - 1], trajectory[step])
                    )
                    break

        # Generate predicates that separate concrete from spurious
        for div_step, pt_before, pt_after in divergence_points:
            cex_state = cex.path[div_step]
            for var in var_names:
                concrete_val = pt_after.get(var, 0.0)
                iv = cex_state.box.dimensions.get(var)
                if iv is None:
                    continue

                # If concrete value is outside the abstract box,
                # add a predicate at the boundary
                if concrete_val < iv.lo:
                    threshold = (concrete_val + iv.lo) / 2.0
                    preds.append(
                        Predicate(
                            expr=Gt(Var(var), Const(threshold)),
                            name=f"sim_{var}>{threshold:.4g}@{div_step}",
                            origin=PredicateOrigin.SIMULATION_GUIDED,
                            iteration_added=iteration,
                            info={
                                "species": var,
                                "step": div_step,
                                "threshold": threshold,
                                "direction": "below",
                            },
                        )
                    )
                elif concrete_val > iv.hi:
                    threshold = (concrete_val + iv.hi) / 2.0
                    preds.append(
                        Predicate(
                            expr=Gt(Var(var), Const(threshold)),
                            name=f"sim_{var}>{threshold:.4g}@{div_step}",
                            origin=PredicateOrigin.SIMULATION_GUIDED,
                            iteration_added=iteration,
                            info={
                                "species": var,
                                "step": div_step,
                                "threshold": threshold,
                                "direction": "above",
                            },
                        )
                    )

        # Deduplicate
        seen: Set[str] = set()
        unique_preds: List[Predicate] = []
        for p in preds:
            if p.digest not in seen:
                seen.add(p.digest)
                unique_preds.append(p)
        preds = unique_preds

        failure_idx = analysis.failure_index or (cex.length // 2)
        failure_state = cex.path[min(failure_idx, cex.length - 1)]
        states_refined: List[int] = []
        if preds:
            new_states = domain.refine(failure_state, preds)
            states_refined = [s.state_id for s in new_states]

        elapsed = time.monotonic() - t0
        return RefinementResult(
            new_predicates=preds,
            states_refined=states_refined,
            strategy_name=self.name,
            time_elapsed=elapsed,
            success=len(preds) > 0,
            message=f"Simulation: {len(preds)} predicates from "
            f"{len(divergence_points)} divergence points",
        )

    def _sample_initial_points(
        self,
        state: AbstractState,
    ) -> List[Dict[str, float]]:
        """Sample points from an abstract state for simulation."""
        points: List[Dict[str, float]] = [state.box.midpoint()]

        if self._num_samples <= 1:
            return points

        import random

        for _ in range(self._num_samples - 1):
            pt: Dict[str, float] = {}
            for var, iv in state.box.dimensions.items():
                pt[var] = random.uniform(iv.lo, iv.hi)
            points.append(pt)

        return points

    def _simulate(
        self,
        initial: Dict[str, float],
        var_names: List[str],
        num_steps: int,
    ) -> List[Dict[str, float]]:
        """Simple Euler simulation of the ODE system."""
        from bioprover.cegar.counterexample import _eval_expr_float

        trajectory: List[Dict[str, float]] = [dict(initial)]
        current = dict(initial)

        for _ in range(num_steps - 1):
            new_state: Dict[str, float] = {}
            for var in var_names:
                rhs_expr = self._rhs.get(var)
                if rhs_expr is None:
                    new_state[var] = current.get(var, 0.0)
                    continue
                try:
                    deriv = _eval_expr_float(rhs_expr, current)
                    new_state[var] = current[var] + self._step_size * deriv
                except Exception:
                    new_state[var] = current.get(var, 0.0)
            trajectory.append(new_state)
            current = new_state

        return trajectory


# ---------------------------------------------------------------------------
# 6. Refinement combinator
# ---------------------------------------------------------------------------


class RefinementCombinator(RefinementStrategy):
    """Compose multiple refinement strategies with priority and fallback.

    Tries strategies in order; uses the first successful one.
    Can optionally combine results from multiple strategies.
    """

    def __init__(
        self,
        strategies: List[RefinementStrategy],
        combine_results: bool = False,
        max_predicates_per_round: int = 20,
    ) -> None:
        self._strategies = strategies
        self._combine = combine_results
        self._max_preds = max_predicates_per_round
        self._usage_counts: Dict[str, int] = {}
        self._success_counts: Dict[str, int] = {}

    @property
    def name(self) -> str:
        return "combinator"

    def refine(
        self,
        cex: AbstractCounterexample,
        analysis: SpuriousnessResult,
        domain: AbstractionDomain,
        iteration: int,
        **kwargs: Any,
    ) -> RefinementResult:
        t0 = time.monotonic()
        all_preds: List[Predicate] = []
        all_refined: List[int] = []
        messages: List[str] = []

        for strategy in self._strategies:
            if not strategy.applicable(cex, analysis):
                continue

            self._usage_counts[strategy.name] = (
                self._usage_counts.get(strategy.name, 0) + 1
            )

            result = strategy.refine(cex, analysis, domain, iteration, **kwargs)

            if result.success and result.new_predicates:
                self._success_counts[strategy.name] = (
                    self._success_counts.get(strategy.name, 0) + 1
                )
                all_preds.extend(result.new_predicates)
                all_refined.extend(result.states_refined)
                messages.append(f"{strategy.name}: {result.predicate_count} preds")

                if not self._combine:
                    break  # Use first successful strategy

            if len(all_preds) >= self._max_preds:
                break

        # Truncate if needed
        all_preds = all_preds[: self._max_preds]

        elapsed = time.monotonic() - t0
        return RefinementResult(
            new_predicates=all_preds,
            states_refined=all_refined,
            strategy_name=self.name,
            time_elapsed=elapsed,
            success=len(all_preds) > 0,
            message="; ".join(messages) if messages else "No strategy produced predicates",
        )

    def strategy_statistics(self) -> Dict[str, Dict[str, int]]:
        """Per-strategy usage and success counts."""
        stats: Dict[str, Dict[str, int]] = {}
        for s in self._strategies:
            stats[s.name] = {
                "usage": self._usage_counts.get(s.name, 0),
                "success": self._success_counts.get(s.name, 0),
            }
        return stats

    def reorder_by_success_rate(self) -> None:
        """Reorder strategies: most successful first."""
        def success_rate(s: RefinementStrategy) -> float:
            usage = self._usage_counts.get(s.name, 0)
            success = self._success_counts.get(s.name, 0)
            return success / usage if usage > 0 else 0.5

        self._strategies.sort(key=success_rate, reverse=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _abs_expr(e: ExprNode) -> ExprNode:
    """Build an expression for |e| using if-then-else logic.

    Approximation: max(e, -e).
    """
    from bioprover.encoding.expression import Max

    return Max(e, Neg(e))


def build_default_combinator(
    rhs: Dict[str, ExprNode],
    hill_params: Optional[List[Dict[str, Any]]] = None,
    monotone_info: Optional[Dict[str, Dict[str, int]]] = None,
    solver: Optional[Any] = None,
    steady_states: Optional[List[Dict[str, float]]] = None,
    jacobian: Optional[Callable[..., Any]] = None,
) -> RefinementCombinator:
    """Build a default refinement combinator with all strategies.

    Priority order:
    1. Structural (cheapest, most informative for bio models)
    2. Monotonicity (logarithmic refinement)
    3. Time-scale (temporal splitting)
    4. Simulation-guided (concrete-guided)
    5. Interpolation (most expensive, last resort)
    """
    strategies: List[RefinementStrategy] = [
        StructuralRefinement(
            rhs=rhs,
            hill_params=hill_params,
            steady_states=steady_states,
            jacobian=jacobian,
        ),
        MonotonicityRefinement(rhs=rhs, monotone_info=monotone_info),
        TimeScaleRefinement(rhs=rhs),
        SimulationGuidedRefinement(rhs=rhs),
        InterpolationRefinement(solver=solver),
    ]
    return RefinementCombinator(strategies=strategies)


# ---------------------------------------------------------------------------
# 7. AI-guided refinement with quality monitoring
# ---------------------------------------------------------------------------


class AIGuidedRefinement(RefinementStrategy):
    """AI-guided refinement with quality monitoring and fallback.

    Uses :class:`~bioprover.ai.predicate_predictor.PredicatePredictor` to rank
    candidate predicates.  Falls back to a structural strategy when the
    :class:`~bioprover.ai.predicate_predictor.PredictionQualityMonitor`
    disables the AI component due to low precision.

    Tracks which AI-predicted predicates actually eliminated spurious
    counterexamples and feeds this data back to the quality monitor.
    Logs the AI acceleration factor (speedup vs baseline refinement).
    """

    def __init__(
        self,
        predictor: Any,
        fallback: RefinementStrategy,
        monitor: Any,
        graph_embedding: Optional[Any] = None,
        species_names: Optional[List[str]] = None,
        quality_threshold: float = 0.3,
    ) -> None:
        from bioprover.ai.predicate_predictor import (
            PredictionQualityMonitor,
            PredicatePredictor,
            generate_candidate_predicates,
        )

        self._predictor: PredicatePredictor = predictor
        self._fallback = fallback
        self._monitor: PredictionQualityMonitor = monitor
        self._graph_embedding = graph_embedding
        self._species_names = species_names or []
        self._quality_threshold = quality_threshold
        self._ai_times: List[float] = []
        self._fallback_times: List[float] = []
        self._ai_successes: int = 0
        self._ai_attempts: int = 0

    @property
    def name(self) -> str:
        return "ai_guided"

    @property
    def acceleration_factor(self) -> float:
        """Speedup from AI-guided vs fallback refinement."""
        if not self._ai_times or not self._fallback_times:
            return 1.0
        import numpy as np
        mean_ai = float(np.mean(self._ai_times)) if self._ai_times else 1.0
        mean_fb = float(np.mean(self._fallback_times)) if self._fallback_times else 1.0
        return mean_fb / max(mean_ai, 1e-8)

    def refine(
        self,
        cex: AbstractCounterexample,
        analysis: SpuriousnessResult,
        domain: AbstractionDomain,
        iteration: int,
        **kwargs: Any,
    ) -> RefinementResult:
        # Fall back when AI is disabled or quality drops below threshold
        if not self._monitor.is_enabled or self._monitor.current_f1 < self._quality_threshold:
            logger.debug(
                "AI predictor disabled or quality below threshold (F1=%.3f); "
                "using fallback strategy",
                self._monitor.current_f1,
            )
            t0_fb = time.monotonic()
            result = self._fallback.refine(cex, analysis, domain, iteration, **kwargs)
            self._fallback_times.append(time.monotonic() - t0_fb)
            return result

        t0 = time.monotonic()
        self._ai_attempts += 1

        from bioprover.ai.predicate_predictor import generate_candidate_predicates
        import numpy as np

        # Build feature vectors from cex
        failure_idx = analysis.failure_index
        if failure_idx is None:
            failure_idx = cex.length // 2
        failure_state = cex.path[min(failure_idx, cex.length - 1)]

        midpoint = failure_state.box.midpoint()
        var_names = list(midpoint.keys())
        cex_values = np.array([midpoint.get(v, 0.0) for v in var_names])

        # Graph embedding (use zeros if not provided)
        graph_embed = (
            self._graph_embedding
            if self._graph_embedding is not None
            else np.zeros(self._predictor.graph_embed_dim)
        )

        # Build cex features
        cex_features = np.zeros(self._predictor.cex_feat_dim)
        cex_features[: min(len(cex_values), len(cex_features))] = cex_values[
            : self._predictor.cex_feat_dim
        ]

        # Abstraction features
        abstraction_features = np.array([
            float(domain.state_count()),
            float(domain.transition_count()),
            float(len(var_names)),
        ] + [0.0] * max(0, self._predictor.abstraction_feat_dim - 3))
        abstraction_features = abstraction_features[: self._predictor.abstraction_feat_dim]

        # CEGAR features
        cegar_features = np.array([
            float(iteration), 0.0, 0.0, 0.0,
        ])[: self._predictor.cegar_feat_dim]

        # Generate candidates and rank them
        species = self._species_names or var_names
        candidates = generate_candidate_predicates(
            species, cex_values.reshape(1, -1) if len(cex_values) > 0 else None,
        )

        ranked = self._predictor.predict_top_k(
            graph_embed, cex_features, abstraction_features, cegar_features,
            candidates, k=5,
        )

        # Convert top candidate to a Predicate and try refinement
        preds: List[Predicate] = []
        for cand in ranked:
            species_name = cand.species
            threshold = cand.threshold
            if cand.template == ">=":
                expr = Ge(Var(species_name), Const(threshold))
            elif cand.template == "<=":
                expr = Le(Var(species_name), Const(threshold))
            elif cand.template == "d/dt>=0":
                expr = Ge(Var(species_name), Const(0.0))
            else:
                expr = Ge(Var(species_name), Const(threshold))

            preds.append(Predicate(
                expr=expr,
                name=f"ai_{species_name}_{cand.template}{threshold:.4g}@{iteration}",
                origin=PredicateOrigin.AI_HEURISTIC,
                iteration_added=iteration,
                info={"ai_score": cand.score, "type": cand.predicate_type},
            ))

        # Try refinement with AI predicates
        states_refined: List[int] = []
        if preds:
            new_states = domain.refine(failure_state, preds)
            states_refined = [s.state_id for s in new_states]
            ai_success = len(states_refined) > 0
        else:
            ai_success = False

        elapsed = time.monotonic() - t0

        # Determine circuit family from kwargs for per-family tracking
        circuit_family = kwargs.get("circuit_family", None)

        # Feed back to monitor: did AI-predicted predicates actually help?
        self._monitor.record(
            predicted_useful=True,
            actually_useful=ai_success,
            circuit_family=circuit_family,
        )

        if ai_success:
            self._ai_successes += 1
            self._ai_times.append(elapsed)
            logger.debug(
                "AI-guided refinement succeeded: %d predicates, "
                "acceleration=%.2fx, F1=%.3f",
                len(preds), self.acceleration_factor, self._monitor.current_f1,
            )
        else:
            # If AI failed, fall back
            logger.debug("AI predicates ineffective; falling back")
            t0_fb = time.monotonic()
            fallback_result = self._fallback.refine(
                cex, analysis, domain, iteration, **kwargs,
            )
            self._fallback_times.append(time.monotonic() - t0_fb)
            return fallback_result

        return RefinementResult(
            new_predicates=preds,
            states_refined=states_refined,
            strategy_name=self.name,
            time_elapsed=elapsed,
            success=True,
            message=(
                f"AI-guided: {len(preds)} predicates "
                f"(precision={self._monitor.current_precision:.3f}, "
                f"F1={self._monitor.current_f1:.3f}, "
                f"accel={self.acceleration_factor:.2f}x)"
            ),
        )

    def quality_metrics(self) -> dict:
        """Return quality metrics from the monitor plus acceleration data."""
        report = self._monitor.get_report()
        report["acceleration_factor"] = self.acceleration_factor
        report["ai_attempts"] = self._ai_attempts
        report["ai_successes"] = self._ai_successes
        report["ai_success_rate"] = (
            self._ai_successes / max(self._ai_attempts, 1)
        )
        return report
