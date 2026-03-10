"""
Main pipeline orchestrator (ALG 8).

Coordinates DAG loading, CI testing, fragility scoring, robustness radius
computation, causal effect estimation, and report generation.

Features:
- Dependency-driven execution (skip steps if inputs unchanged)
- Progress callbacks
- Checkpoint/resume capability
- Memory management for large problems
- Configurable: which steps to run, which methods to use
"""

from __future__ import annotations

import copy
import hashlib
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from causalcert.types import (
    AdjacencyMatrix,
    AuditReport,
    CITestMethod,
    CITestResult,
    ConclusionPredicate,
    EditType,
    EstimationResult,
    FragilityChannel,
    FragilityScore,
    NodeId,
    NodeSet,
    PipelineConfig,
    RobustnessRadius,
    SolverStrategy,
    StructuralEdit,
)
from causalcert.pipeline.config import PipelineRunConfig
from causalcert.pipeline.cache import ResultCache
from causalcert.pipeline.parallel import ParallelExecutor
from causalcert.pipeline.logging_config import log_timing, get_logger

logger = get_logger("orchestrator")


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[str, float], None]
"""Signature: ``(step_name, fraction_complete) -> None``."""


def _noop_progress(step: str, frac: float) -> None:
    pass


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PipelineCheckpoint:
    """Holds intermediate state for checkpoint/resume.

    Attributes
    ----------
    step : str
        Last completed step.
    ci_results : list[CITestResult]
        CI test results.
    fragility_scores : list[FragilityScore]
        Fragility scores.
    radius : RobustnessRadius | None
        Radius result.
    baseline_estimate : EstimationResult | None
    perturbed_estimates : list[EstimationResult]
    metadata : dict[str, Any]
    """

    step: str = ""
    ci_results: list[CITestResult] = field(default_factory=list)
    fragility_scores: list[FragilityScore] = field(default_factory=list)
    radius: RobustnessRadius | None = None
    baseline_estimate: EstimationResult | None = None
    perturbed_estimates: list[EstimationResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default conclusion predicate
# ---------------------------------------------------------------------------


class ATESignificancePredicate:
    """Default predicate: ATE is statistically significant.

    Returns ``True`` when the confidence interval for the ATE under the
    given DAG excludes zero.
    """

    def __init__(self, alpha: float = 0.05, n_folds: int = 5, seed: int = 42) -> None:
        self.alpha = alpha
        self.n_folds = n_folds
        self.seed = seed

    def __call__(
        self,
        adj: AdjacencyMatrix,
        data: Any,
        *,
        treatment: NodeId,
        outcome: NodeId,
    ) -> bool:
        try:
            from causalcert.estimation.adjustment import find_optimal_adjustment_set
            from causalcert.estimation.effects import estimate_ate

            adj_arr = np.asarray(adj, dtype=np.int8)
            adjustment_set = find_optimal_adjustment_set(adj_arr, treatment, outcome)
            result = estimate_ate(
                adj_arr, data, treatment, outcome,
                adjustment_set=adjustment_set,
                n_folds=self.n_folds,
                seed=self.seed,
                alpha=self.alpha,
            )
            # Significant if CI excludes zero
            return result.ci_lower > 0 or result.ci_upper < 0
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Input fingerprinting
# ---------------------------------------------------------------------------


def _fingerprint(adj: AdjacencyMatrix, data: pd.DataFrame, config: PipelineRunConfig) -> str:
    """Compute a fingerprint of the pipeline inputs for cache invalidation."""
    h = hashlib.sha256()
    h.update(np.asarray(adj).tobytes())
    h.update(str(data.shape).encode())
    h.update(str(data.dtypes.tolist()).encode())
    # Sample of data for fast hashing
    sample = data.head(50).to_numpy()
    h.update(sample.tobytes())
    h.update(str(config.treatment).encode())
    h.update(str(config.outcome).encode())
    h.update(str(config.alpha).encode())
    h.update(str(config.ci_method.value).encode())
    h.update(str(config.seed).encode())
    return h.hexdigest()[:20]


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


class CausalCertPipeline:
    """End-to-end structural-robustness audit pipeline (ALG 8).

    Parameters
    ----------
    config : PipelineRunConfig
        Pipeline configuration.
    progress_callback : ProgressCallback | None
        Optional callback for progress updates.
    checkpoint : PipelineCheckpoint | None
        Optional checkpoint for resume.
    """

    # Step ordering
    STEPS = ("validate", "ci_testing", "fragility", "radius", "estimation", "report")

    def __init__(
        self,
        config: PipelineRunConfig,
        progress_callback: ProgressCallback | None = None,
        checkpoint: PipelineCheckpoint | None = None,
    ) -> None:
        self.config = config
        self._progress = progress_callback or _noop_progress
        self._checkpoint = checkpoint or PipelineCheckpoint()
        self._cache = ResultCache(
            cache_dir=config.cache_dir or ".causalcert_cache",
            enabled=config.cache_dir is not None,
        )
        self._executor = ParallelExecutor(
            n_jobs=config.n_jobs,
            backend=config.parallel_config.backend,
            max_memory_mb=config.parallel_config.max_memory_mb,
        )
        self._timings: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        adj_matrix: AdjacencyMatrix,
        data: pd.DataFrame,
        predicate: ConclusionPredicate | None = None,
    ) -> AuditReport:
        """Execute the full audit pipeline.

        Steps:
        1. Validate DAG and data.
        2. Run CI tests (with multiplicity correction).
        3. Score per-edge fragility (ALG 3).
        4. Compute robustness radius (ALG 4/5/7, auto-selected).
        5. Estimate causal effects under original and witness DAGs.
        6. Generate the audit report.

        Parameters
        ----------
        adj_matrix : AdjacencyMatrix
            Assumed causal DAG adjacency matrix.
        data : pd.DataFrame
            Observational dataset.
        predicate : ConclusionPredicate | None
            Custom conclusion predicate.  If ``None``, defaults to
            "the ATE is statistically significant".

        Returns
        -------
        AuditReport
        """
        adj = np.asarray(adj_matrix, dtype=np.int8)
        n = adj.shape[0]
        treatment = self.config.treatment
        outcome = self.config.outcome

        fingerprint = _fingerprint(adj, data, self.config)
        self._checkpoint.metadata["fingerprint"] = fingerprint
        self._checkpoint.metadata["start_time"] = time.time()

        if predicate is None:
            predicate = ATESignificancePredicate(
                alpha=self.config.alpha,
                n_folds=self.config.n_folds,
                seed=self.config.seed,
            )

        logger.info(
            "Starting CausalCert audit: %d nodes, %d edges, T=%d, Y=%d",
            n, int(adj.sum()), treatment, outcome,
        )

        # Ensure data columns are integer-indexed
        if not all(isinstance(c, int) for c in data.columns):
            data = data.copy()
            data.columns = list(range(data.shape[1]))

        # --- Step 1: Validate ---
        if self.config.steps.validate and self._should_run("validate"):
            self._run_step("validate", self._validate, adj, data)

        # --- Step 2: CI tests ---
        ci_results: list[CITestResult] = []
        if self.config.steps.ci_testing and self._should_run("ci_testing"):
            ci_results = self._run_step("ci_testing", self._run_ci_tests, adj, data)
            self._checkpoint.ci_results = ci_results
            self._checkpoint.step = "ci_testing"
        elif self._checkpoint.ci_results:
            ci_results = self._checkpoint.ci_results

        # --- Step 3: Fragility scores ---
        fragility_scores: list[FragilityScore] = []
        if self.config.steps.fragility and self._should_run("fragility"):
            fragility_scores = self._run_step(
                "fragility", self._score_fragility, adj, data, ci_results
            )
            self._checkpoint.fragility_scores = fragility_scores
            self._checkpoint.step = "fragility"
        elif self._checkpoint.fragility_scores:
            fragility_scores = self._checkpoint.fragility_scores

        # --- Step 4: Robustness radius ---
        radius = RobustnessRadius(lower_bound=0, upper_bound=self.config.max_k)
        if self.config.steps.radius and self._should_run("radius"):
            radius = self._run_step(
                "radius", self._compute_radius, adj, data, predicate, ci_results
            )
            self._checkpoint.radius = radius
            self._checkpoint.step = "radius"
        elif self._checkpoint.radius is not None:
            radius = self._checkpoint.radius

        # --- Step 5: Estimation ---
        baseline_estimate: EstimationResult | None = None
        perturbed_estimates: list[EstimationResult] = []
        if self.config.steps.estimation and self._should_run("estimation"):
            baseline_estimate, perturbed_estimates = self._run_step(
                "estimation", self._estimate_effects, adj, data, radius
            )
            self._checkpoint.baseline_estimate = baseline_estimate
            self._checkpoint.perturbed_estimates = perturbed_estimates
            self._checkpoint.step = "estimation"
        elif self._checkpoint.baseline_estimate is not None:
            baseline_estimate = self._checkpoint.baseline_estimate
            perturbed_estimates = self._checkpoint.perturbed_estimates

        # --- Step 6: Build report ---
        report = AuditReport(
            treatment=treatment,
            outcome=outcome,
            n_nodes=n,
            n_edges=int(adj.sum()),
            radius=radius,
            fragility_ranking=fragility_scores,
            baseline_estimate=baseline_estimate,
            perturbed_estimates=perturbed_estimates,
            ci_results=ci_results,
            metadata={
                "seed": self.config.seed,
                "alpha": self.config.alpha,
                "ci_method": self.config.ci_method.value,
                "solver_strategy": self.config.solver_strategy.value,
                "max_k": self.config.max_k,
                "n_folds": self.config.n_folds,
                "fdr_method": self.config.fdr_method,
                "n_jobs": self.config.n_jobs,
                "fingerprint": fingerprint,
                "timings": dict(self._timings),
            },
        )

        self._progress("complete", 1.0)
        logger.info(
            "Audit complete: radius=[%d, %d], %d fragile edges, %d CI tests",
            radius.lower_bound, radius.upper_bound,
            len(fragility_scores), len(ci_results),
        )

        return report

    # ------------------------------------------------------------------
    # Step execution helpers
    # ------------------------------------------------------------------

    def _should_run(self, step: str) -> bool:
        """Check if step should run (not already completed in checkpoint)."""
        completed = self.STEPS
        if self._checkpoint.step:
            try:
                idx = list(completed).index(self._checkpoint.step)
                step_idx = list(completed).index(step)
                if step_idx <= idx:
                    return False
            except ValueError:
                pass
        return True

    def _run_step(self, name: str, fn: Callable, *args: Any) -> Any:
        """Run a pipeline step with timing and progress."""
        self._progress(name, 0.0)
        logger.info("Starting step: %s", name)
        t0 = time.perf_counter()
        try:
            result = fn(*args)
        except Exception as exc:
            logger.error("Step %s failed: %s", name, exc)
            raise
        elapsed = time.perf_counter() - t0
        self._timings[name] = round(elapsed, 3)
        step_idx = list(self.STEPS).index(name) if name in self.STEPS else 0
        frac = (step_idx + 1) / len(self.STEPS)
        self._progress(name, frac)
        logger.info("Step %s completed in %.3f s", name, elapsed)
        return result

    # ------------------------------------------------------------------
    # Step 1: Validate
    # ------------------------------------------------------------------

    def _validate(self, adj: AdjacencyMatrix, data: pd.DataFrame) -> None:
        """Validate DAG and data compatibility."""
        from causalcert.dag.validation import is_dag, validate_adjacency_matrix
        from causalcert.exceptions import CyclicGraphError, SchemaError

        n = adj.shape[0]
        treatment = self.config.treatment
        outcome = self.config.outcome

        # Check DAG structure
        issues = validate_adjacency_matrix(adj)
        if issues:
            logger.warning("DAG validation issues: %s", issues)
            for issue in issues:
                if "cycle" in issue.lower():
                    raise CyclicGraphError()

        # Check treatment/outcome in range
        if treatment < 0 or treatment >= n:
            raise ValueError(f"Treatment index {treatment} out of range [0, {n})")
        if outcome < 0 or outcome >= n:
            raise ValueError(f"Outcome index {outcome} out of range [0, {n})")
        if treatment == outcome:
            raise ValueError("Treatment and outcome must be different nodes.")

        # Check data dimensions
        if data.shape[1] < n:
            logger.warning(
                "Data has %d columns but DAG has %d nodes; "
                "extra nodes will have no data.",
                data.shape[1], n,
            )

        # Check for missing values in key columns
        cols_to_check = [treatment, outcome]
        for col in cols_to_check:
            if col < data.shape[1] and data.iloc[:, col].isnull().any():
                logger.warning("Missing values in column %d", col)

        logger.info("Validation passed: %d nodes, %d edges, %d obs", n, int(adj.sum()), len(data))

    # ------------------------------------------------------------------
    # Step 2: CI tests
    # ------------------------------------------------------------------

    def _run_ci_tests(
        self,
        adj: AdjacencyMatrix,
        data: pd.DataFrame,
    ) -> list[CITestResult]:
        """Run all CI tests with multiplicity correction."""
        from causalcert.dag.dsep import DSeparationOracle
        from causalcert.ci_testing.partial_corr import PartialCorrelationTest
        from causalcert.ci_testing.multiplicity import (
            BenjaminiYekutieli,
            BenjaminiHochberg,
            Bonferroni,
            HolmBonferroni,
            ancestral_pruning,
        )

        treatment = self.config.treatment
        outcome = self.config.outcome
        alpha = self.config.alpha
        n = adj.shape[0]

        # Check cache
        cache_key = ResultCache.ci_test_key(
            adj.tobytes(),
            ResultCache.data_fingerprint(data),
            alpha,
            self.config.ci_method.value,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info("CI test results loaded from cache")
            return cached

        # Get DAG-implied CI relations
        oracle = DSeparationOracle(adj)
        all_triples = oracle.all_ci_implications(max_cond_size=None)

        # Ancestral pruning
        pruned = ancestral_pruning(adj, treatment, outcome, all_triples)
        logger.info(
            "CI tests: %d total implications, %d after ancestral pruning",
            len(all_triples), len(pruned),
        )

        if not pruned:
            return []

        # Select CI tester
        tester = self._get_ci_tester()

        # Run tests (potentially in parallel via thread pool for I/O)
        def _test_triple(triple: tuple) -> CITestResult:
            x, y, s = triple
            return tester.test(x, y, s, data)

        if self.config.n_jobs == 1 or len(pruned) < 10:
            raw_results = [_test_triple(t) for t in pruned]
        else:
            executor = ParallelExecutor(
                n_jobs=min(self.config.n_jobs, len(pruned)),
                backend="thread",
            )
            raw_results = executor.map(_test_triple, pruned)

        # Apply multiplicity correction
        corrector = self._get_multiplicity_corrector()
        corrected = corrector.adjust(raw_results)

        # Cache
        self._cache.put(cache_key, corrected)

        n_reject = sum(1 for r in corrected if r.reject)
        logger.info(
            "CI testing: %d tests, %d rejections (violations) after %s correction",
            len(corrected), n_reject, self.config.fdr_method.upper(),
        )

        return corrected

    def _get_ci_tester(self) -> Any:
        """Instantiate the CI tester according to config."""
        from causalcert.ci_testing.partial_corr import PartialCorrelationTest

        alpha = self.config.alpha
        seed = self.config.seed

        if self.config.ci_method == CITestMethod.PARTIAL_CORRELATION:
            return PartialCorrelationTest(alpha=alpha, seed=seed)

        if self.config.ci_method == CITestMethod.ENSEMBLE:
            try:
                from causalcert.ci_testing.ensemble import CauchyCombinationTest
                from causalcert.ci_testing.rank import RankCITest

                base_tests = [
                    PartialCorrelationTest(alpha=alpha, seed=seed),
                    RankCITest(alpha=alpha, seed=seed),
                ]
                return CauchyCombinationTest(
                    base_tests=base_tests,
                    alpha=alpha,
                    adaptive=True,
                    seed=seed,
                )
            except ImportError:
                logger.warning("Ensemble CI test unavailable; falling back to partial correlation")
                return PartialCorrelationTest(alpha=alpha, seed=seed)

        if self.config.ci_method == CITestMethod.RANK:
            try:
                from causalcert.ci_testing.rank import RankCITest
                return RankCITest(alpha=alpha, seed=seed)
            except ImportError:
                return PartialCorrelationTest(alpha=alpha, seed=seed)

        if self.config.ci_method == CITestMethod.KERNEL:
            try:
                from causalcert.ci_testing.kci import KernelCITest
                return KernelCITest(alpha=alpha, seed=seed)
            except ImportError:
                return PartialCorrelationTest(alpha=alpha, seed=seed)

        if self.config.ci_method == CITestMethod.CRT:
            try:
                from causalcert.ci_testing.crt import ConditionalRandomizationTest
                return ConditionalRandomizationTest(alpha=alpha, seed=seed)
            except ImportError:
                return PartialCorrelationTest(alpha=alpha, seed=seed)

        return PartialCorrelationTest(alpha=alpha, seed=seed)

    def _get_multiplicity_corrector(self) -> Any:
        """Instantiate the multiplicity corrector according to config."""
        from causalcert.ci_testing.multiplicity import (
            BenjaminiYekutieli,
            BenjaminiHochberg,
            Bonferroni,
            HolmBonferroni,
        )

        alpha = self.config.alpha
        method = self.config.fdr_method.lower()

        if method == "by":
            return BenjaminiYekutieli(alpha=alpha)
        elif method == "bh":
            return BenjaminiHochberg(alpha=alpha)
        elif method == "bonferroni":
            return Bonferroni(alpha=alpha)
        elif method == "holm":
            return HolmBonferroni(alpha=alpha)
        else:
            return BenjaminiYekutieli(alpha=alpha)

    # ------------------------------------------------------------------
    # Step 3: Fragility scores
    # ------------------------------------------------------------------

    def _score_fragility(
        self,
        adj: AdjacencyMatrix,
        data: pd.DataFrame,
        ci_results: list[CITestResult] | None = None,
    ) -> list[FragilityScore]:
        """Score per-edge fragility."""
        treatment = self.config.treatment
        outcome = self.config.outcome

        # Check cache
        cache_key = ResultCache.fragility_key(adj.tobytes(), treatment, outcome)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info("Fragility scores loaded from cache")
            return cached

        # Compute fragility scores
        from causalcert.dag.dsep import DSeparationOracle
        from causalcert.dag.ancestors import ancestors, candidate_edges

        oracle = DSeparationOracle(adj)
        n = adj.shape[0]

        # Get ancestral set
        anc_set = ancestors(adj, frozenset({treatment, outcome}))
        anc_list = sorted(anc_set)

        scores: list[FragilityScore] = []

        # Score existing edges
        existing_edges = candidate_edges(adj, treatment, outcome)
        for u, v in existing_edges:
            score = self._score_edge(adj, u, v, treatment, outcome, oracle, ci_results)
            scores.append(score)

        # Score candidate additions (absent edges within ancestral set)
        for u in anc_list:
            for v in anc_list:
                if u == v or adj[u, v]:
                    continue
                # Only score additions that are plausible
                score = self._score_absent_edge(
                    adj, u, v, treatment, outcome, oracle, ci_results
                )
                if score.total_score > 0.01:
                    scores.append(score)

        # Sort by total score (most fragile first)
        scores.sort(key=lambda s: s.total_score, reverse=True)

        # Cache
        self._cache.put(cache_key, scores)

        logger.info("Fragility scoring: %d edges scored", len(scores))
        return scores

    def _score_edge(
        self,
        adj: AdjacencyMatrix,
        u: int,
        v: int,
        treatment: int,
        outcome: int,
        oracle: Any,
        ci_results: list[CITestResult] | None,
    ) -> FragilityScore:
        """Score a single existing edge for fragility."""
        channels: dict[FragilityChannel, float] = {}
        witness: CITestResult | None = None

        # Channel 1: D-separation impact
        # Would removing this edge change any d-separation relations?
        adj_mod = adj.copy()
        adj_mod[u, v] = 0
        from causalcert.dag.dsep import DSeparationOracle
        oracle_mod = DSeparationOracle(adj_mod)

        dsep_score = 0.0
        if ci_results:
            changes = 0
            total_checked = 0
            for cr in ci_results:
                orig = oracle.is_d_separated(cr.x, cr.y, cr.conditioning_set)
                mod = oracle_mod.is_d_separated(cr.x, cr.y, cr.conditioning_set)
                if orig != mod:
                    changes += 1
                    if witness is None or cr.p_value < (witness.p_value if witness else 1.0):
                        witness = cr
                total_checked += 1
            if total_checked > 0:
                dsep_score = changes / total_checked
        else:
            # Without CI results, use a structural heuristic
            dsep_score = 0.5 if oracle.is_d_connected(
                treatment, outcome, frozenset()
            ) else 0.1
        channels[FragilityChannel.D_SEPARATION] = dsep_score

        # Channel 2: Identification impact
        # Would removing this edge change the valid adjustment sets?
        id_score = self._identification_impact(adj, adj_mod, treatment, outcome)
        channels[FragilityChannel.IDENTIFICATION] = id_score

        # Channel 3: Estimation impact (lightweight)
        est_score = 0.0
        if dsep_score > 0.1 or id_score > 0.1:
            est_score = min(1.0, (dsep_score + id_score) / 2.0 * 1.2)
        channels[FragilityChannel.ESTIMATION] = est_score

        total = (
            0.4 * channels[FragilityChannel.D_SEPARATION]
            + 0.35 * channels[FragilityChannel.IDENTIFICATION]
            + 0.25 * channels[FragilityChannel.ESTIMATION]
        )

        return FragilityScore(
            edge=(u, v),
            total_score=min(1.0, total),
            channel_scores=channels,
            witness_ci=witness,
        )

    def _score_absent_edge(
        self,
        adj: AdjacencyMatrix,
        u: int,
        v: int,
        treatment: int,
        outcome: int,
        oracle: Any,
        ci_results: list[CITestResult] | None,
    ) -> FragilityScore:
        """Score a candidate edge addition for fragility."""
        channels: dict[FragilityChannel, float] = {}

        # Would adding this edge create a cycle?
        from causalcert.dag.graph import CausalDAG
        dag_copy = CausalDAG(adj.copy(), validate=False)
        try:
            if dag_copy.has_directed_path(v, u):
                # Would create cycle — not a valid edit
                return FragilityScore(edge=(u, v), total_score=0.0)
        except Exception:
            return FragilityScore(edge=(u, v), total_score=0.0)

        adj_mod = adj.copy()
        adj_mod[u, v] = 1
        from causalcert.dag.dsep import DSeparationOracle
        oracle_mod = DSeparationOracle(adj_mod)

        # D-separation impact
        dsep_score = 0.0
        witness: CITestResult | None = None
        if ci_results:
            changes = 0
            for cr in ci_results:
                orig = oracle.is_d_separated(cr.x, cr.y, cr.conditioning_set)
                mod = oracle_mod.is_d_separated(cr.x, cr.y, cr.conditioning_set)
                if orig != mod:
                    changes += 1
                    if witness is None:
                        witness = cr
            if ci_results:
                dsep_score = changes / len(ci_results)
        channels[FragilityChannel.D_SEPARATION] = dsep_score

        # Identification impact
        id_score = self._identification_impact(adj, adj_mod, treatment, outcome)
        channels[FragilityChannel.IDENTIFICATION] = id_score

        # Estimation
        est_score = min(1.0, (dsep_score + id_score) / 2.0)
        channels[FragilityChannel.ESTIMATION] = est_score

        total = (
            0.4 * channels[FragilityChannel.D_SEPARATION]
            + 0.35 * channels[FragilityChannel.IDENTIFICATION]
            + 0.25 * channels[FragilityChannel.ESTIMATION]
        )

        return FragilityScore(
            edge=(u, v),
            total_score=min(1.0, total),
            channel_scores=channels,
            witness_ci=witness,
        )

    def _identification_impact(
        self,
        adj_orig: AdjacencyMatrix,
        adj_mod: AdjacencyMatrix,
        treatment: int,
        outcome: int,
    ) -> float:
        """Compute identification-channel fragility.

        Measures how much the set of valid adjustment sets changes.
        """
        try:
            from causalcert.estimation.backdoor import satisfies_backdoor
            from causalcert.estimation.adjustment import find_optimal_adjustment_set

            # Find optimal adjustment set under original DAG
            try:
                adj_set_orig = find_optimal_adjustment_set(adj_orig, treatment, outcome)
                orig_valid = True
            except Exception:
                adj_set_orig = frozenset()
                orig_valid = False

            # Check if that set is still valid under modified DAG
            from causalcert.dag.validation import is_dag
            if not is_dag(adj_mod):
                return 1.0  # Modified DAG is cyclic

            if orig_valid:
                still_valid = satisfies_backdoor(adj_mod, treatment, outcome, adj_set_orig)
                if not still_valid:
                    return 1.0  # Original adjustment set invalidated

            # Check if any valid set exists under modified DAG
            try:
                adj_set_mod = find_optimal_adjustment_set(adj_mod, treatment, outcome)
                if orig_valid and adj_set_orig != adj_set_mod:
                    return 0.5
                return 0.0
            except Exception:
                if orig_valid:
                    return 1.0
                return 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Step 4: Robustness radius
    # ------------------------------------------------------------------

    def _compute_radius(
        self,
        adj: AdjacencyMatrix,
        data: pd.DataFrame,
        predicate: ConclusionPredicate,
        ci_results: list[CITestResult] | None = None,
    ) -> RobustnessRadius:
        """Compute the robustness radius."""
        treatment = self.config.treatment
        outcome = self.config.outcome
        max_k = self.config.max_k

        # Check cache
        cache_key = ResultCache.solver_key(
            adj.tobytes(), max_k, self.config.solver_strategy.value
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info("Robustness radius loaded from cache")
            return cached

        # Instantiate solver
        try:
            from causalcert.solver.search import UnifiedSolver
            solver = UnifiedSolver(
                strategy=self.config.solver_strategy,
                time_limit_s=self.config.solver_config.time_limit_s,
                max_treewidth_for_fpt=self.config.solver_config.max_treewidth_for_fpt,
                verbose=self.config.verbose,
            )
            radius = solver.solve(
                adj, predicate, data, treatment, outcome,
                max_k=max_k, ci_results=ci_results,
            )
        except (NotImplementedError, Exception) as exc:
            logger.warning("Solver failed (%s), falling back to enumeration", exc)
            radius = self._enumerate_radius(adj, data, predicate, max_k)

        # Cache
        self._cache.put(cache_key, radius)

        logger.info(
            "Radius: [%d, %d], certified=%s, time=%.3fs",
            radius.lower_bound, radius.upper_bound,
            radius.certified, radius.solver_time_s,
        )

        return radius

    def _enumerate_radius(
        self,
        adj: AdjacencyMatrix,
        data: pd.DataFrame,
        predicate: ConclusionPredicate,
        max_k: int,
    ) -> RobustnessRadius:
        """Fallback: enumerate single-edge edits to find radius."""
        treatment = self.config.treatment
        outcome = self.config.outcome
        n = adj.shape[0]
        t0 = time.perf_counter()

        # Check if predicate holds on original DAG
        orig_holds = predicate(adj, data, treatment=treatment, outcome=outcome)
        if not orig_holds:
            return RobustnessRadius(
                lower_bound=0, upper_bound=0,
                certified=True,
                solver_strategy=SolverStrategy.AUTO,
                solver_time_s=time.perf_counter() - t0,
            )

        # Try all single-edge edits (k=1)
        from causalcert.dag.ancestors import ancestors
        anc_set = ancestors(adj, frozenset({treatment, outcome}))
        anc_list = sorted(anc_set)

        best_edits: list[StructuralEdit] = []
        found_k = max_k + 1

        for k in range(1, min(max_k + 1, 4)):
            if k == 1:
                edits = self._single_edit_candidates(adj, anc_list)
                for edit in edits:
                    adj_new = adj.copy()
                    try:
                        if edit.edit_type == EditType.DELETE:
                            adj_new[edit.source, edit.target] = 0
                        elif edit.edit_type == EditType.ADD:
                            adj_new[edit.source, edit.target] = 1
                        elif edit.edit_type == EditType.REVERSE:
                            adj_new[edit.source, edit.target] = 0
                            adj_new[edit.target, edit.source] = 1

                        from causalcert.dag.validation import is_dag
                        if not is_dag(adj_new):
                            continue

                        if not predicate(adj_new, data, treatment=treatment, outcome=outcome):
                            found_k = 1
                            best_edits = [edit]
                            break
                    except Exception:
                        continue
                if found_k <= k:
                    break
            else:
                # For k > 1, we would need combinatorial enumeration
                # which is handled by the full solver; break here
                break

        elapsed = time.perf_counter() - t0
        ub = min(found_k, max_k)
        lb = min(found_k, max_k) if found_k <= max_k else 1

        return RobustnessRadius(
            lower_bound=lb,
            upper_bound=ub,
            witness_edits=tuple(best_edits),
            solver_strategy=SolverStrategy.AUTO,
            solver_time_s=elapsed,
            gap=float(ub - lb) / max(ub, 1),
            certified=lb == ub,
        )

    def _single_edit_candidates(
        self, adj: AdjacencyMatrix, anc_list: list[int]
    ) -> list[StructuralEdit]:
        """Generate all single-edge edit candidates within the ancestral set."""
        edits: list[StructuralEdit] = []
        for u in anc_list:
            for v in anc_list:
                if u == v:
                    continue
                if adj[u, v]:
                    edits.append(StructuralEdit(EditType.DELETE, u, v))
                    edits.append(StructuralEdit(EditType.REVERSE, u, v))
                else:
                    edits.append(StructuralEdit(EditType.ADD, u, v))
        return edits

    # ------------------------------------------------------------------
    # Step 5: Estimation
    # ------------------------------------------------------------------

    def _estimate_effects(
        self,
        adj: AdjacencyMatrix,
        data: pd.DataFrame,
        radius: RobustnessRadius | None = None,
    ) -> tuple[EstimationResult | None, list[EstimationResult]]:
        """Estimate causal effects under original and witness DAGs."""
        treatment = self.config.treatment
        outcome = self.config.outcome

        # Baseline estimate
        baseline: EstimationResult | None = None
        try:
            from causalcert.estimation.effects import estimate_ate
            baseline = estimate_ate(
                adj, data, treatment, outcome,
                method=self.config.estimation_config.estimator,
                n_folds=self.config.n_folds,
                seed=self.config.seed,
                propensity_model=self.config.estimation_config.propensity_model,
                outcome_model_type=self.config.estimation_config.outcome_model,
                alpha=self.config.alpha,
            )
            logger.info(
                "Baseline ATE: %.4f ± %.4f [%.4f, %.4f]",
                baseline.ate, baseline.se, baseline.ci_lower, baseline.ci_upper,
            )
        except Exception as exc:
            logger.warning("Baseline estimation failed: %s", exc)

        # Perturbed estimates (for each witness edit set)
        perturbed: list[EstimationResult] = []
        if radius is not None and radius.witness_edits:
            adj_witness = adj.copy()
            for edit in radius.witness_edits:
                try:
                    if edit.edit_type == EditType.DELETE:
                        adj_witness[edit.source, edit.target] = 0
                    elif edit.edit_type == EditType.ADD:
                        adj_witness[edit.source, edit.target] = 1
                    elif edit.edit_type == EditType.REVERSE:
                        adj_witness[edit.source, edit.target] = 0
                        adj_witness[edit.target, edit.source] = 1
                except Exception:
                    continue

            try:
                from causalcert.dag.validation import is_dag
                if is_dag(adj_witness):
                    from causalcert.estimation.effects import estimate_ate
                    witness_est = estimate_ate(
                        adj_witness, data, treatment, outcome,
                        method=self.config.estimation_config.estimator,
                        n_folds=self.config.n_folds,
                        seed=self.config.seed,
                        alpha=self.config.alpha,
                    )
                    perturbed.append(witness_est)
                    logger.info("Witness ATE: %.4f ± %.4f", witness_est.ate, witness_est.se)
            except Exception as exc:
                logger.warning("Witness estimation failed: %s", exc)

        return baseline, perturbed

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @property
    def checkpoint(self) -> PipelineCheckpoint:
        """Current checkpoint state."""
        return self._checkpoint

    @property
    def timings(self) -> dict[str, float]:
        """Wall-clock timings for each completed step."""
        return dict(self._timings)

    @property
    def cache(self) -> ResultCache:
        """The result cache used by this pipeline."""
        return self._cache

    def reset(self) -> None:
        """Reset the pipeline state (checkpoint and timings)."""
        self._checkpoint = PipelineCheckpoint()
        self._timings.clear()

    # ------------------------------------------------------------------
    # Convenience class methods
    # ------------------------------------------------------------------

    @classmethod
    def quick_audit(
        cls,
        adj: AdjacencyMatrix,
        data: pd.DataFrame,
        treatment: int,
        outcome: int,
        **kwargs: Any,
    ) -> AuditReport:
        """Run a quick audit with sensible defaults."""
        from causalcert.pipeline.config import quick_config
        cfg = quick_config(treatment=treatment, outcome=outcome, **kwargs)
        return cls(cfg).run(adj, data)

    @classmethod
    def thorough_audit(
        cls,
        adj: AdjacencyMatrix,
        data: pd.DataFrame,
        treatment: int,
        outcome: int,
        **kwargs: Any,
    ) -> AuditReport:
        """Run a thorough audit for publication-quality results."""
        from causalcert.pipeline.config import thorough_config
        cfg = thorough_config(treatment=treatment, outcome=outcome, **kwargs)
        return cls(cfg).run(adj, data)
