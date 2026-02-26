"""Parameter synthesis orchestrator.

Combines CEGIS, CMA-ES robustness optimization, realizability checking,
and Pareto exploration into a unified synthesis workflow with multiple
operating modes: feasibility, robustness, minimal perturbation, and
multi-objective.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from bioprover.models.parameters import ParameterSet
from bioprover.repair.cegis import (
    CEGISConfig,
    CEGISLoop,
    CEGISResult,
    CEGISStatus,
    Counterexample,
    OptimizationProposalStrategy,
    ProposalStrategy,
    SurrogateProposalStrategy,
    VerifierProtocol,
)
from bioprover.repair.design_space import DesignPoint, DesignSpace, ParetoFrontier
from bioprover.repair.realizability import RealizabilityChecker, RealizabilityReport
from bioprover.repair.repair_report import RepairReport, RepairResult
from bioprover.repair.robustness_optimization import (
    CMAESConfig,
    OptimizationResult,
    RobustnessOptimizer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthesis modes and configuration
# ---------------------------------------------------------------------------

class SynthesisMode(Enum):
    """Operating mode for parameter synthesis."""

    FEASIBILITY = auto()       # Find any satisfying assignment
    ROBUSTNESS = auto()        # Maximise robustness margin
    MINIMAL_PERTURBATION = auto()  # Closest to original parameters
    MULTI_OBJECTIVE = auto()   # Pareto-optimal (robustness vs perturbation)


@dataclass
class SynthesisConfig:
    """Configuration for the synthesis orchestrator."""

    mode: SynthesisMode = SynthesisMode.ROBUSTNESS
    max_outer_iterations: int = 10
    timeout: float = 600.0
    cmaes_config: Optional[CMAESConfig] = None
    cegis_config: Optional[CEGISConfig] = None
    n_cmaes_restarts: int = 3
    perturbation_weight: float = 0.1  # trade-off in minimal-perturbation
    pareto_samples: int = 500
    check_realizability: bool = True
    warm_start_params: Optional[np.ndarray] = None
    verbose: bool = True


# ---------------------------------------------------------------------------
# Synthesis result
# ---------------------------------------------------------------------------

@dataclass
class SynthesisResult:
    """Result of a parameter synthesis run."""

    success: bool = False
    mode: SynthesisMode = SynthesisMode.FEASIBILITY
    repair_result: Optional[RepairResult] = None
    pareto_frontier: Optional[ParetoFrontier] = None
    realizability_report: Optional[RealizabilityReport] = None
    cegis_result: Optional[CEGISResult] = None
    optimization_result: Optional[OptimizationResult] = None
    outer_iterations: int = 0
    total_time: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Synthesis Result: {'SUCCESS' if self.success else 'FAILURE'}",
            f"  Mode: {self.mode.name}",
            f"  Outer iterations: {self.outer_iterations}",
            f"  Total time: {self.total_time:.2f}s",
        ]
        if self.repair_result:
            lines.append(f"  Robustness: {self.repair_result.robustness_after:.6g}")
            lines.append(f"  Perturbation L2: {self.repair_result.perturbation_l2:.6g}")
            lines.append(f"  Verified: {self.repair_result.verified}")
        if self.realizability_report:
            lines.append(
                f"  Realizable: {self.realizability_report.feasible} "
                f"(errors={self.realizability_report.error_count})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parameter synthesizer
# ---------------------------------------------------------------------------

class ParameterSynthesizer:
    """Orchestrator combining CEGIS + optimization for parameter synthesis.

    The overall workflow is:

    1. Run fast optimisation (CMA-ES) to find a candidate.
    2. Verify the candidate via the CEGAR verifier.
    3. If verification fails, run CEGIS with the counterexample.
    4. Iterate until success or timeout.

    Parameters
    ----------
    param_set : ParameterSet
        Parameters with bounds.
    verifier : VerifierProtocol
        Formal verifier returning ``(ok, counterexample)``.
    robustness_fn : callable
        ``params_vector -> float`` computing STL robustness.
    original_params : array, optional
        Original (pre-repair) parameter values.
    realizability_checker : RealizabilityChecker, optional
        Biological constraint checker.
    config : SynthesisConfig, optional
    """

    def __init__(
        self,
        param_set: ParameterSet,
        verifier: VerifierProtocol,
        robustness_fn: Callable[[np.ndarray], float],
        original_params: Optional[np.ndarray] = None,
        realizability_checker: Optional[RealizabilityChecker] = None,
        config: Optional[SynthesisConfig] = None,
    ) -> None:
        self._param_set = param_set
        self._verifier = verifier
        self._robustness_fn = robustness_fn
        self._config = config or SynthesisConfig()
        self._realizability = realizability_checker

        params = list(param_set)
        self._names = [p.name for p in params]
        self._bounds = [(p.lower_bound, p.upper_bound) for p in params]
        self._dim = len(params)

        if original_params is not None:
            self._original = original_params.copy()
        else:
            self._original = np.array([p.value for p in params])

        self._best_params: Optional[np.ndarray] = None
        self._best_robustness = float("-inf")
        self._history: List[Dict[str, Any]] = []

    # -- objective functions ------------------------------------------------

    def _robustness_objective(self, x: np.ndarray) -> float:
        """Pure robustness (for CMA-ES which *minimises*)."""
        return self._robustness_fn(x)

    def _perturbation_objective(self, x: np.ndarray) -> float:
        """Perturbation from original parameters."""
        return -float(np.linalg.norm(x - self._original))

    def _combined_objective(self, x: np.ndarray) -> float:
        """Robustness minus weighted perturbation (for minimisation by CMA-ES)."""
        rob = self._robustness_fn(x)
        pert = float(np.linalg.norm(x - self._original))
        return rob - self._config.perturbation_weight * pert

    # -- constraint function ------------------------------------------------

    def _constraint_fn(self, x: np.ndarray) -> bool:
        if self._realizability is None:
            return True
        params = dict(zip(self._names, x.tolist()))
        return self._realizability.is_feasible(params)

    # -- main entry ---------------------------------------------------------

    def synthesize(self) -> SynthesisResult:
        """Run the full synthesis pipeline."""
        mode = self._config.mode
        logger.info("Parameter synthesis starting: mode=%s", mode.name)
        t0 = time.time()

        if mode == SynthesisMode.FEASIBILITY:
            result = self._synthesize_feasibility()
        elif mode == SynthesisMode.ROBUSTNESS:
            result = self._synthesize_robustness()
        elif mode == SynthesisMode.MINIMAL_PERTURBATION:
            result = self._synthesize_minimal_perturbation()
        elif mode == SynthesisMode.MULTI_OBJECTIVE:
            result = self._synthesize_multi_objective()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        result.total_time = time.time() - t0
        result.history = list(self._history)
        logger.info("Synthesis complete: %s", result.summary())
        return result

    # -- feasibility mode ---------------------------------------------------

    def _synthesize_feasibility(self) -> SynthesisResult:
        """Find any parameter assignment satisfying the specification."""
        result = SynthesisResult(mode=SynthesisMode.FEASIBILITY)

        for outer in range(1, self._config.max_outer_iterations + 1):
            result.outer_iterations = outer
            if self._time_exceeded():
                break

            # Phase 1: CMA-ES
            opt_result = self._run_cmaes(self._robustness_objective)
            result.optimization_result = opt_result
            candidate = opt_result.best_params

            # Phase 2: Verify
            verified, cex = self._verify(candidate)
            self._record_iteration(outer, candidate, opt_result.best_robustness, verified)

            if verified:
                result.success = True
                result.repair_result = self._make_repair_result(candidate, verified=True)
                result.realizability_report = self._check_realizability(candidate)
                return result

            # Phase 3: CEGIS with counterexample
            if cex is not None:
                cegis_result = self._run_cegis(initial_cex=cex)
                result.cegis_result = cegis_result
                if cegis_result.success and cegis_result.parameters is not None:
                    verified2, _ = self._verify(cegis_result.parameters)
                    if verified2:
                        result.success = True
                        result.repair_result = self._make_repair_result(
                            cegis_result.parameters, verified=True
                        )
                        result.realizability_report = self._check_realizability(
                            cegis_result.parameters
                        )
                        return result

        # Return best-effort
        if self._best_params is not None:
            result.repair_result = self._make_repair_result(self._best_params, verified=False)
            result.realizability_report = self._check_realizability(self._best_params)
        return result

    # -- robustness mode ----------------------------------------------------

    def _synthesize_robustness(self) -> SynthesisResult:
        """Maximise robustness margin while satisfying the specification."""
        result = SynthesisResult(mode=SynthesisMode.ROBUSTNESS)

        for outer in range(1, self._config.max_outer_iterations + 1):
            result.outer_iterations = outer
            if self._time_exceeded():
                break

            opt_result = self._run_cmaes(self._robustness_objective)
            result.optimization_result = opt_result
            candidate = opt_result.best_params

            verified, cex = self._verify(candidate)
            self._record_iteration(outer, candidate, opt_result.best_robustness, verified)

            if verified:
                # Continue optimising to improve robustness
                if self._best_params is None or opt_result.best_robustness > self._best_robustness:
                    self._best_params = candidate.copy()
                    self._best_robustness = opt_result.best_robustness

                # Try to improve further with L-BFGS-B
                optimizer = RobustnessOptimizer(
                    robustness_fn=self._robustness_fn,
                    bounds=self._bounds,
                    constraint_fn=self._constraint_fn if self._realizability else None,
                )
                local_result = optimizer.optimize_lbfgsb_only(candidate)
                verified2, _ = self._verify(local_result.best_params)
                if verified2 and local_result.best_robustness > self._best_robustness:
                    self._best_params = local_result.best_params.copy()
                    self._best_robustness = local_result.best_robustness

                result.success = True
                result.repair_result = self._make_repair_result(
                    self._best_params, verified=True
                )
                result.realizability_report = self._check_realizability(self._best_params)
                return result

            if cex is not None:
                cegis_result = self._run_cegis(initial_cex=cex)
                result.cegis_result = cegis_result
                if cegis_result.success and cegis_result.parameters is not None:
                    self._best_params = cegis_result.parameters.copy()
                    self._best_robustness = self._robustness_fn(cegis_result.parameters)

        if self._best_params is not None:
            result.success = True
            result.repair_result = self._make_repair_result(self._best_params, verified=False)
            result.realizability_report = self._check_realizability(self._best_params)
        return result

    # -- minimal perturbation mode ------------------------------------------

    def _synthesize_minimal_perturbation(self) -> SynthesisResult:
        """Find parameters closest to original that satisfy the spec."""
        result = SynthesisResult(mode=SynthesisMode.MINIMAL_PERTURBATION)

        for outer in range(1, self._config.max_outer_iterations + 1):
            result.outer_iterations = outer
            if self._time_exceeded():
                break

            opt_result = self._run_cmaes(
                self._combined_objective, x0=self._original
            )
            result.optimization_result = opt_result
            candidate = opt_result.best_params

            verified, cex = self._verify(candidate)
            pert = float(np.linalg.norm(candidate - self._original))
            self._record_iteration(outer, candidate, opt_result.best_robustness, verified)

            if verified:
                if self._best_params is None or pert < float(
                    np.linalg.norm(self._best_params - self._original)
                ):
                    self._best_params = candidate.copy()
                    self._best_robustness = self._robustness_fn(candidate)

                result.success = True
                result.repair_result = self._make_repair_result(
                    self._best_params, verified=True
                )
                result.realizability_report = self._check_realizability(self._best_params)
                return result

            if cex is not None:
                cegis_result = self._run_cegis(initial_cex=cex)
                result.cegis_result = cegis_result

        if self._best_params is not None:
            result.repair_result = self._make_repair_result(self._best_params, verified=False)
            result.realizability_report = self._check_realizability(self._best_params)
        return result

    # -- multi-objective mode -----------------------------------------------

    def _synthesize_multi_objective(self) -> SynthesisResult:
        """Compute Pareto frontier of robustness vs perturbation."""
        result = SynthesisResult(mode=SynthesisMode.MULTI_OBJECTIVE)

        ds = DesignSpace(self._bounds, self._names)

        def obj_rob(x: np.ndarray) -> float:
            return self._robustness_fn(x)

        def obj_pert(x: np.ndarray) -> float:
            return -float(np.linalg.norm(x - self._original))

        frontier = ds.pareto_explore(
            objectives=[obj_rob, obj_pert],
            n_samples=self._config.pareto_samples,
        )
        result.pareto_frontier = frontier

        # Verify points on the Pareto front
        verified_results: List[RepairResult] = []
        for pt in frontier.front:
            v, _ = self._verify(pt.parameters)
            if v:
                rr = self._make_repair_result(pt.parameters, verified=True)
                verified_results.append(rr)

        if verified_results:
            result.success = True
            # Primary: best robustness among verified
            verified_results.sort(key=lambda r: r.robustness_after, reverse=True)
            result.repair_result = verified_results[0]
            if len(verified_results) > 1:
                # Also build report with alternatives
                pass
            result.realizability_report = self._check_realizability(
                verified_results[0].repaired
            )

        result.outer_iterations = 1
        return result

    # -- helpers ------------------------------------------------------------

    def _run_cmaes(
        self,
        objective: Callable[[np.ndarray], float],
        x0: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """Run CMA-ES optimisation."""
        start = x0 if x0 is not None else self._config.warm_start_params
        optimizer = RobustnessOptimizer(
            robustness_fn=objective,
            bounds=self._bounds,
            constraint_fn=self._constraint_fn if self._realizability else None,
            cmaes_config=self._config.cmaes_config or CMAESConfig(),
            n_restarts=self._config.n_cmaes_restarts,
        )
        remaining = self._remaining_time()
        return optimizer.optimize(x0=start, timeout=min(remaining, 120.0))

    def _run_cegis(
        self,
        initial_cex: Optional[Counterexample] = None,
    ) -> CEGISResult:
        """Run CEGIS inner loop."""
        surrogate = SurrogateProposalStrategy()
        if self._best_params is not None:
            rob = self._robustness_fn(self._best_params)
            surrogate.add_observation(self._best_params, rob)

        cegis = CEGISLoop(
            param_set=self._param_set,
            verifier=self._verifier,
            strategy=surrogate,
            config=self._config.cegis_config or CEGISConfig(
                max_iterations=50,
                timeout=min(self._remaining_time(), 120.0),
            ),
            objective_fn=lambda x: -self._robustness_fn(x),
        )
        if initial_cex is not None:
            cegis.seed_counterexamples([initial_cex])

        return cegis.run()

    def _verify(
        self, params: np.ndarray
    ) -> Tuple[bool, Optional[Counterexample]]:
        """Invoke the formal verifier."""
        param_dict = dict(zip(self._names, params.tolist()))
        return self._verifier.verify(param_dict)

    def _check_realizability(
        self, params: np.ndarray
    ) -> Optional[RealizabilityReport]:
        if self._realizability is None:
            return None
        return self._realizability.check_vector(params, self._names)

    def _make_repair_result(
        self, params: np.ndarray, verified: bool
    ) -> RepairResult:
        rob = self._robustness_fn(params)
        rob_before = self._robustness_fn(self._original)
        return RepairResult(
            original=self._original.copy(),
            repaired=params.copy(),
            parameter_names=list(self._names),
            robustness_before=rob_before,
            robustness_after=rob,
            verified=verified,
            method=self._config.mode.name,
        )

    def _record_iteration(
        self,
        outer: int,
        candidate: np.ndarray,
        robustness: float,
        verified: bool,
    ) -> None:
        self._history.append({
            "outer_iteration": outer,
            "robustness": robustness,
            "verified": verified,
            "perturbation_l2": float(np.linalg.norm(candidate - self._original)),
            "time": time.time(),
        })

    def _time_exceeded(self) -> bool:
        return self._remaining_time() <= 0

    def _remaining_time(self) -> float:
        if not self._history:
            return self._config.timeout
        elapsed = time.time() - self._history[0].get("time", time.time())
        return max(self._config.timeout - elapsed, 0.0)

    # -- warm start ---------------------------------------------------------

    def warm_start(self, params: np.ndarray, robustness: float) -> None:
        """Seed the synthesizer with a known good starting point."""
        self._best_params = params.copy()
        self._best_robustness = robustness
        self._config.warm_start_params = params.copy()

    # -- report generation --------------------------------------------------

    def generate_report(self, result: SynthesisResult) -> RepairReport:
        """Build a comprehensive repair report from synthesis results."""
        if result.repair_result is None:
            # Create a dummy result
            rr = RepairResult(
                original=self._original,
                repaired=self._original,
                parameter_names=list(self._names),
            )
        else:
            rr = result.repair_result

        alternatives: List[RepairResult] = []
        if result.pareto_frontier is not None:
            for pt in result.pareto_frontier.front:
                alt = self._make_repair_result(pt.parameters, verified=False)
                alternatives.append(alt)

        confidence = 1.0 if rr.verified else 0.5
        notes = ""
        if result.realizability_report and not result.realizability_report.feasible:
            notes = (
                f"Design has {result.realizability_report.error_count} "
                f"realizability errors."
            )

        return RepairReport(
            primary=rr,
            alternatives=alternatives,
            confidence=confidence,
            notes=notes,
        )
