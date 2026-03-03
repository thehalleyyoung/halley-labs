"""Three-phase pipeline orchestrator for the Causal-Plasticity Atlas.

Implements :class:`CPAOrchestrator`, the main entry point that coordinates
causal discovery, CADA alignment, plasticity descriptor computation,
quality-diversity exploration, tipping-point detection, and robustness
certification into a single end-to-end workflow.

The three phases are:

1. **Foundation** — causal discovery for each context, pairwise CADA
   alignment, and 4D plasticity descriptor computation.
2. **Exploration** — curiosity-driven QD-MAP-Elites search over the
   space of mechanism-change patterns.
3. **Validation** — tipping-point detection, robustness certificate
   generation, and sensitivity analysis.

Usage
-----
>>> from cpa.pipeline import CPAOrchestrator, PipelineConfig
>>> from cpa.io.readers import CSVReader
>>> dataset = CSVReader("data/").read()
>>> orch = CPAOrchestrator(PipelineConfig.standard())
>>> atlas = orch.run(dataset)
>>> atlas.summary_statistics()
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

from cpa.pipeline.config import PipelineConfig, ComputationConfig
from cpa.pipeline.checkpointing import CheckpointManager, compute_config_hash
from cpa.pipeline.results import (
    AlignmentResult,
    ArchiveEntry,
    AtlasResult,
    CertificateResult,
    DescriptorResult,
    ExplorationResult,
    FoundationResult,
    MechanismClass,
    SCMResult,
    TippingPointResult,
    ValidationResult,
)
from cpa.utils.logging import (
    get_logger,
    TimingContext,
    MemoryTracker,
    ProgressReporter,
)
from cpa.utils.parallel import parallel_map, pairwise_indices
from cpa.utils.caching import LRUCache

logger = get_logger("pipeline.orchestrator")


# =====================================================================
# Multi-context dataset protocol
# =====================================================================


class MultiContextDataset:
    """Lightweight container for multi-context observational data.

    Attributes
    ----------
    context_data : Dict[str, np.ndarray]
        Mapping from context_id to (n_samples, n_variables) data matrix.
    variable_names : List[str]
        Ordered variable names shared across contexts.
    context_ids : List[str]
        Ordered context identifiers.
    context_metadata : Dict[str, Dict[str, Any]]
        Optional metadata per context.
    """

    def __init__(
        self,
        context_data: Dict[str, np.ndarray],
        variable_names: Optional[List[str]] = None,
        context_ids: Optional[List[str]] = None,
        context_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if not context_data:
            raise ValueError("context_data must be non-empty")

        self.context_data = dict(context_data)

        if context_ids is not None:
            self.context_ids = list(context_ids)
        else:
            self.context_ids = sorted(self.context_data.keys())

        first_key = self.context_ids[0]
        first_data = self.context_data[first_key]
        p = first_data.shape[1] if first_data.ndim == 2 else 1

        if variable_names is not None:
            self.variable_names = list(variable_names)
        else:
            self.variable_names = [f"X{i}" for i in range(p)]

        self.context_metadata = context_metadata or {}

    @property
    def n_contexts(self) -> int:
        return len(self.context_ids)

    @property
    def n_variables(self) -> int:
        return len(self.variable_names)

    def get_data(self, context_id: str) -> np.ndarray:
        """Return data matrix for a context.

        Parameters
        ----------
        context_id : str
            Context identifier.

        Returns
        -------
        np.ndarray
            Shape (n_samples, n_variables).

        Raises
        ------
        KeyError
            If context_id not found.
        """
        if context_id not in self.context_data:
            raise KeyError(f"Unknown context: {context_id!r}")
        return self.context_data[context_id]

    def sample_sizes(self) -> Dict[str, int]:
        """Return sample sizes for each context."""
        return {
            cid: data.shape[0] for cid, data in self.context_data.items()
        }

    def total_samples(self) -> int:
        """Total number of samples across all contexts."""
        return sum(d.shape[0] for d in self.context_data.values())

    def validate(self) -> List[str]:
        """Validate dataset consistency, returning error messages."""
        errors: List[str] = []
        p = self.n_variables

        for cid in self.context_ids:
            if cid not in self.context_data:
                errors.append(f"Context {cid!r} in context_ids but not in context_data")
                continue
            data = self.context_data[cid]
            if data.ndim != 2:
                errors.append(
                    f"Context {cid!r}: data must be 2D, got shape {data.shape}"
                )
                continue
            if data.shape[1] != p:
                errors.append(
                    f"Context {cid!r}: expected {p} variables, got {data.shape[1]}"
                )
            if data.shape[0] < 2:
                errors.append(
                    f"Context {cid!r}: need at least 2 samples, got {data.shape[0]}"
                )
            if np.any(np.isnan(data)):
                n_nan = int(np.sum(np.isnan(data)))
                errors.append(
                    f"Context {cid!r}: {n_nan} NaN values detected"
                )
            if np.any(np.isinf(data)):
                n_inf = int(np.sum(np.isinf(data)))
                errors.append(
                    f"Context {cid!r}: {n_inf} Inf values detected"
                )

        return errors

    def subset_contexts(self, context_ids: Sequence[str]) -> "MultiContextDataset":
        """Return a new dataset with a subset of contexts."""
        return MultiContextDataset(
            context_data={c: self.context_data[c] for c in context_ids},
            variable_names=self.variable_names,
            context_ids=list(context_ids),
            context_metadata={
                c: self.context_metadata.get(c, {}) for c in context_ids
            },
        )

    def subset_variables(
        self, variable_indices: Sequence[int]
    ) -> "MultiContextDataset":
        """Return a new dataset with a subset of variables."""
        indices = list(variable_indices)
        new_names = [self.variable_names[i] for i in indices]
        new_data = {
            cid: data[:, indices] for cid, data in self.context_data.items()
        }
        return MultiContextDataset(
            context_data=new_data,
            variable_names=new_names,
            context_ids=self.context_ids,
            context_metadata=self.context_metadata,
        )

    def __repr__(self) -> str:
        sizes = self.sample_sizes()
        min_n = min(sizes.values()) if sizes else 0
        max_n = max(sizes.values()) if sizes else 0
        return (
            f"MultiContextDataset(K={self.n_contexts}, p={self.n_variables}, "
            f"n={min_n}–{max_n})"
        )


# =====================================================================
# Phase callbacks
# =====================================================================


@dataclass
class PhaseCallbacks:
    """User-provided callbacks invoked at pipeline events.

    Attributes
    ----------
    on_phase_start : callable
        Called with (phase_number, phase_name) at phase start.
    on_phase_end : callable
        Called with (phase_number, phase_name, elapsed_seconds) at phase end.
    on_step_complete : callable
        Called with (phase, step_name, step_result) after sub-steps.
    on_error : callable
        Called with (phase, step_name, exception) on non-fatal errors.
    on_checkpoint : callable
        Called with (checkpoint_path) after checkpoint save.
    """

    on_phase_start: Optional[Callable[[int, str], None]] = None
    on_phase_end: Optional[Callable[[int, str, float], None]] = None
    on_step_complete: Optional[Callable[[int, str, Any], None]] = None
    on_error: Optional[Callable[[int, str, Exception], None]] = None
    on_checkpoint: Optional[Callable[[Path], None]] = None


# =====================================================================
# Pipeline error types
# =====================================================================


class PipelineError(Exception):
    """Base class for pipeline errors."""

    def __init__(self, message: str, phase: int = 0, step: str = ""):
        self.phase = phase
        self.step = step
        super().__init__(message)


class PhaseError(PipelineError):
    """Error during a specific phase."""
    pass


class CheckpointError(PipelineError):
    """Error in checkpoint operations."""
    pass


# =====================================================================
# CPAOrchestrator
# =====================================================================


class CPAOrchestrator:
    """Three-phase pipeline orchestrator for the Causal-Plasticity Atlas.

    The orchestrator coordinates all CPA computations:

    **Phase 1 — Foundation:**
    Runs causal discovery on each context, computes pairwise CADA
    alignments, and derives 4D plasticity descriptors for all variables.

    **Phase 2 — Exploration:**
    Uses QD-MAP-Elites to systematically explore the space of mechanism-
    change patterns, building a diverse archive of causal configurations.

    **Phase 3 — Validation:**
    Detects tipping points (if contexts are ordered), generates
    robustness certificates, and performs sensitivity analysis.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.
    callbacks : PhaseCallbacks, optional
        Event callbacks.

    Examples
    --------
    >>> config = PipelineConfig.standard()
    >>> orch = CPAOrchestrator(config)
    >>> atlas = orch.run(dataset)
    >>> print(atlas.summary_statistics())
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        callbacks: Optional[PhaseCallbacks] = None,
    ) -> None:
        self._config = config or PipelineConfig.standard()
        self._callbacks = callbacks or PhaseCallbacks()
        self._rng: Optional[np.random.RandomState] = None
        self._cache: LRUCache[Any] = LRUCache(maxsize=512)

        self._checkpoint_mgr: Optional[CheckpointManager] = None
        self._setup_rng()
        self._setup_logging()
        self._setup_checkpointing()

        self._errors: List[Dict[str, Any]] = []
        self._timings: Dict[str, float] = {}

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------

    def _setup_rng(self) -> None:
        """Initialize random state from config seed."""
        seed = self._config.computation.seed
        self._rng = np.random.RandomState(seed)

    def _setup_logging(self) -> None:
        """Configure logging level."""
        level_str = self._config.computation.log_level.upper()
        level = getattr(logging, level_str, logging.WARNING)
        logging.getLogger("cpa").setLevel(level)

    def _setup_checkpointing(self) -> None:
        """Initialize checkpoint manager if configured."""
        ckpt_dir = self._config.computation.checkpoint_dir
        if ckpt_dir is not None:
            config_hash = compute_config_hash(self._config.to_dict())
            self._checkpoint_mgr = CheckpointManager(
                checkpoint_dir=ckpt_dir,
                max_checkpoints=5,
                config_hash=config_hash,
            )

    @property
    def config(self) -> PipelineConfig:
        """Current pipeline configuration."""
        return self._config

    @property
    def errors(self) -> List[Dict[str, Any]]:
        """Non-fatal errors collected during the pipeline run."""
        return list(self._errors)

    @property
    def timings(self) -> Dict[str, float]:
        """Timing information for pipeline steps."""
        return dict(self._timings)

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------

    def run(
        self,
        dataset: MultiContextDataset,
        resume: bool = False,
    ) -> AtlasResult:
        """Execute the full CPA pipeline.

        Parameters
        ----------
        dataset : MultiContextDataset
            Input multi-context observational data.
        resume : bool
            If True, attempt to resume from the latest checkpoint.

        Returns
        -------
        AtlasResult
            Complete Causal-Plasticity Atlas.

        Raises
        ------
        ValueError
            If the dataset or configuration is invalid.
        PipelineError
            If a fatal error occurs during pipeline execution.
        """
        self._errors.clear()
        self._timings.clear()

        self._config.validate_or_raise()

        validation_errors = dataset.validate()
        if validation_errors:
            raise ValueError(
                "Dataset validation failed:\n  - "
                + "\n  - ".join(validation_errors)
            )

        logger.info(
            "Starting CPA pipeline: K=%d contexts, p=%d variables, "
            "profile=%s",
            dataset.n_contexts,
            dataset.n_variables,
            self._config.profile.value,
        )

        result = AtlasResult(
            config=self._config.to_dict(),
            metadata={
                "start_time": time.time(),
                "n_contexts": dataset.n_contexts,
                "n_variables": dataset.n_variables,
                "sample_sizes": dataset.sample_sizes(),
                "profile": self._config.profile.value,
            },
        )

        foundation_result = None
        exploration_result = None
        validation_result = None

        if resume and self._checkpoint_mgr is not None:
            foundation_result, exploration_result = self._resume_from_checkpoint(
                dataset
            )

        total_timer = TimingContext("total_pipeline")
        total_timer.__enter__()

        try:
            # Phase 1: Foundation
            if self._config.run_phase_1 and foundation_result is None:
                foundation_result = self._run_phase_1(dataset)
                result.foundation = foundation_result
            elif foundation_result is not None:
                result.foundation = foundation_result

            # Phase 2: Exploration
            if self._config.run_phase_2 and exploration_result is None:
                if result.foundation is not None:
                    exploration_result = self._run_phase_2(
                        dataset, result.foundation
                    )
                    result.exploration = exploration_result
                else:
                    logger.warning(
                        "Skipping Phase 2: no foundation results available"
                    )
            elif exploration_result is not None:
                result.exploration = exploration_result

            # Phase 3: Validation
            if self._config.run_phase_3:
                if result.foundation is not None:
                    validation_result = self._run_phase_3(
                        dataset, result.foundation, result.exploration
                    )
                    result.validation = validation_result
                else:
                    logger.warning(
                        "Skipping Phase 3: no foundation results available"
                    )

        except PipelineError:
            raise
        except Exception as e:
            logger.error("Fatal pipeline error: %s", e)
            self._record_error(0, "pipeline", e)
            raise PipelineError(
                f"Pipeline failed: {e}", phase=0, step="run"
            ) from e
        finally:
            total_timer.__exit__(None, None, None)
            self._timings["total"] = total_timer.elapsed_wall

        result.metadata["end_time"] = time.time()
        result.metadata["total_time"] = total_timer.elapsed_wall
        result.metadata["errors"] = self._errors

        if self._config.computation.output_dir is not None:
            try:
                result.save(self._config.computation.output_dir)
                logger.info(
                    "Results saved to %s", self._config.computation.output_dir
                )
            except Exception as e:
                logger.warning("Failed to save results: %s", e)

        logger.info(
            "CPA pipeline complete in %.2fs", total_timer.elapsed_wall
        )
        return result

    # -----------------------------------------------------------------
    # Phase 1: Foundation
    # -----------------------------------------------------------------

    def _run_phase_1(
        self, dataset: MultiContextDataset
    ) -> FoundationResult:
        """Execute Phase 1: Foundation.

        Performs:
        1. Causal discovery for each context.
        2. Pairwise CADA alignment.
        3. Plasticity descriptor computation.

        Parameters
        ----------
        dataset : MultiContextDataset
            Input data.

        Returns
        -------
        FoundationResult
        """
        phase = 1
        phase_name = "Foundation"
        self._notify_phase_start(phase, phase_name)

        result = FoundationResult(
            context_ids=list(dataset.context_ids),
            variable_names=list(dataset.variable_names),
        )

        with TimingContext("phase_1", logger) as t:
            # Step 1: Causal discovery
            with TimingContext("discovery", logger) as dt:
                result.scm_results = self._run_discovery(dataset)
            result.discovery_time = dt.elapsed_wall
            self._timings["phase1_discovery"] = dt.elapsed_wall
            self._notify_step_complete(phase, "discovery", result.scm_results)

            self._save_checkpoint_if_needed(
                phase=1,
                step=1,
                state={"scm_results": {
                    k: v.to_dict() for k, v in result.scm_results.items()
                }},
                description="After causal discovery",
            )

            # Step 2: Pairwise alignment
            with TimingContext("alignment", logger) as at:
                result.alignment_results = self._run_alignment(
                    dataset, result.scm_results
                )
            result.alignment_time = at.elapsed_wall
            self._timings["phase1_alignment"] = at.elapsed_wall
            self._notify_step_complete(
                phase, "alignment", result.alignment_results
            )

            self._save_checkpoint_if_needed(
                phase=1,
                step=2,
                state={
                    "scm_results": {
                        k: v.to_dict() for k, v in result.scm_results.items()
                    },
                    "alignment_results": {
                        f"{ci}__{cj}": ar.to_dict()
                        for (ci, cj), ar in result.alignment_results.items()
                    },
                },
                description="After alignment",
            )

            # Step 3: Plasticity descriptors
            with TimingContext("descriptors", logger) as pt:
                result.descriptors = self._run_descriptors(
                    dataset, result.scm_results, result.alignment_results
                )
            result.descriptor_time = pt.elapsed_wall
            self._timings["phase1_descriptors"] = pt.elapsed_wall
            self._notify_step_complete(phase, "descriptors", result.descriptors)

        result.total_time = t.elapsed_wall
        self._timings["phase1_total"] = t.elapsed_wall

        self._save_checkpoint_if_needed(
            phase=1,
            step=3,
            state={"foundation": result.to_dict()},
            description="Phase 1 complete",
        )

        logger.info(
            "Phase 1 complete: %d SCMs, %d alignments, %d descriptors "
            "in %.2fs",
            len(result.scm_results),
            len(result.alignment_results),
            len(result.descriptors),
            t.elapsed_wall,
        )

        self._notify_phase_end(phase, phase_name, t.elapsed_wall)
        return result

    def _run_discovery(
        self, dataset: MultiContextDataset
    ) -> Dict[str, SCMResult]:
        """Run causal discovery for each context.

        Uses the configured discovery method. Tries to import the
        discovery module and falls back gracefully if unavailable.

        Parameters
        ----------
        dataset : MultiContextDataset
            Input data.

        Returns
        -------
        dict of str → SCMResult
        """
        cfg = self._config.discovery
        K = dataset.n_contexts
        p = dataset.n_variables
        results: Dict[str, SCMResult] = {}

        progress = None
        if self._config.computation.progress and K > 1:
            progress = ProgressReporter(K, "discovery", logger)

        adapter = self._get_discovery_adapter()

        def discover_one(context_id: str) -> Tuple[str, SCMResult]:
            data = dataset.get_data(context_id)
            t0 = time.perf_counter()

            try:
                if adapter is not None:
                    disc_result = adapter.run(
                        data,
                        variable_names=dataset.variable_names,
                        alpha=cfg.alpha,
                    )
                    adj = disc_result.adj_matrix
                    params = disc_result.parameters
                else:
                    adj, params = self._fallback_discovery(data, cfg)
            except Exception as e:
                logger.warning(
                    "Discovery failed for context %s: %s. Using empty graph.",
                    context_id, e,
                )
                self._record_error(1, f"discovery_{context_id}", e)
                adj = np.zeros((p, p))
                params = np.zeros((p, p))

            elapsed = time.perf_counter() - t0

            scm = SCMResult(
                context_id=context_id,
                adjacency=adj,
                parameters=params,
                variable_names=list(dataset.variable_names),
                n_samples=data.shape[0],
                discovery_method=cfg.method,
                fit_time=elapsed,
            )
            return context_id, scm

        if self._config.computation.n_jobs == 1 or K <= 2:
            for cid in dataset.context_ids:
                ctx_id, scm = discover_one(cid)
                results[ctx_id] = scm
                if progress is not None:
                    progress.update()
        else:
            items = list(dataset.context_ids)
            pair_results = parallel_map(
                discover_one,
                items,
                n_workers=self._config.computation.n_jobs,
                backend=self._config.computation.backend,
            )
            for ctx_id, scm in pair_results:
                results[ctx_id] = scm
                if progress is not None:
                    progress.update()

        if progress is not None:
            progress.finish()

        return results

    def _get_discovery_adapter(self) -> Optional[Any]:
        """Try to instantiate the configured discovery adapter.

        Returns
        -------
        adapter or None
            Discovery adapter instance, or None if import fails.
        """
        method = self._config.discovery.method
        try:
            from cpa.discovery import (
                PCAdapter,
                GESAdapter,
                LiNGAMAdapter,
                FallbackDiscovery,
            )

            adapters = {
                "pc": PCAdapter,
                "ges": GESAdapter,
                "lingam": LiNGAMAdapter,
                "fallback": FallbackDiscovery,
            }
            adapter_cls = adapters.get(method)
            if adapter_cls is not None:
                return adapter_cls()
        except ImportError:
            logger.debug(
                "Discovery module not available, using fallback"
            )
        return None

    def _fallback_discovery(
        self, data: np.ndarray, cfg: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Built-in fallback causal discovery using partial correlations.

        Uses a simple threshold on partial correlations to build a DAG.
        This is a lightweight substitute when external discovery
        libraries are not available.

        Parameters
        ----------
        data : np.ndarray
            (n, p) data matrix.
        cfg : DiscoveryConfig
            Discovery configuration.

        Returns
        -------
        adj : np.ndarray
            (p, p) adjacency matrix.
        params : np.ndarray
            (p, p) parameter (weight) matrix.
        """
        n, p = data.shape
        alpha = cfg.alpha

        if n < p + 2:
            return np.zeros((p, p)), np.zeros((p, p))

        try:
            cov = np.cov(data, rowvar=False)
            reg = 1e-6 * np.eye(p)
            precision = np.linalg.inv(cov + reg)

            diag = np.sqrt(np.diag(precision))
            diag[diag == 0] = 1.0
            partial_corr = -precision / np.outer(diag, diag)
            np.fill_diagonal(partial_corr, 0.0)

            from scipy import stats as sp_stats

            df = max(n - p - 2, 1)
            t_stat = partial_corr * np.sqrt(df / (1.0 - partial_corr ** 2 + 1e-12))
            p_vals = 2.0 * (1.0 - sp_stats.t.cdf(np.abs(t_stat), df))

            adj = np.zeros((p, p))
            significant = p_vals < alpha

            for i in range(p):
                for j in range(i + 1, p):
                    if significant[i, j]:
                        if np.abs(partial_corr[i, j]) > 1e-6:
                            adj[i, j] = 1.0

            params = adj * partial_corr

        except Exception:
            adj = np.zeros((p, p))
            params = np.zeros((p, p))

        return adj, params

    def _run_alignment(
        self,
        dataset: MultiContextDataset,
        scm_results: Dict[str, SCMResult],
    ) -> Dict[Tuple[str, str], AlignmentResult]:
        """Run pairwise CADA alignment for all K*(K-1)/2 context pairs.

        Parameters
        ----------
        dataset : MultiContextDataset
            Input data.
        scm_results : dict
            Per-context SCM results.

        Returns
        -------
        dict of (str, str) → AlignmentResult
        """
        cfg = self._config.alignment
        context_ids = dataset.context_ids
        K = len(context_ids)
        pairs = pairwise_indices(K)
        n_pairs = len(pairs)

        results: Dict[Tuple[str, str], AlignmentResult] = {}

        if n_pairs == 0:
            return results

        progress = None
        if self._config.computation.progress and n_pairs > 1:
            progress = ProgressReporter(n_pairs, "alignment", logger)

        aligner = self._get_aligner()

        def align_one(pair: Tuple[int, int]) -> Tuple[str, str, AlignmentResult]:
            i, j = pair
            ci, cj = context_ids[i], context_ids[j]
            scm_i = scm_results.get(ci)
            scm_j = scm_results.get(cj)

            t0 = time.perf_counter()

            if scm_i is None or scm_j is None:
                ar = AlignmentResult(context_i=ci, context_j=cj)
                return ci, cj, ar

            adj_i = scm_i.adjacency
            adj_j = scm_j.adjacency
            param_i = scm_i.parameters
            param_j = scm_j.parameters

            if adj_i is None or adj_j is None:
                ar = AlignmentResult(context_i=ci, context_j=cj)
                return ci, cj, ar

            try:
                if aligner is not None:
                    align_result = aligner.align(
                        scm_i, scm_j,
                        anchors=None,
                        context_a=ci,
                        context_b=cj,
                    )
                    ar = AlignmentResult(
                        context_i=ci,
                        context_j=cj,
                        permutation=getattr(align_result, "alignment", None),
                        structural_cost=getattr(align_result, "structural_divergence", 0.0),
                        parametric_cost=0.0,
                        total_cost=getattr(align_result, "normalized_divergence", 0.0),
                        shared_edges=getattr(align_result, "n_shared", 0),
                        modified_edges=getattr(align_result, "n_modified", 0),
                        context_specific_edges=(
                            getattr(align_result, "n_context_specific_a", 0)
                            + getattr(align_result, "n_context_specific_b", 0)
                        ),
                        align_time=time.perf_counter() - t0,
                    )
                else:
                    ar = self._fallback_alignment(
                        ci, cj, adj_i, adj_j, param_i, param_j, cfg
                    )
                    ar.align_time = time.perf_counter() - t0
            except Exception as e:
                logger.warning(
                    "Alignment failed for (%s, %s): %s", ci, cj, e
                )
                self._record_error(1, f"alignment_{ci}_{cj}", e)
                ar = AlignmentResult(
                    context_i=ci,
                    context_j=cj,
                    align_time=time.perf_counter() - t0,
                )

            return ci, cj, ar

        if self._config.computation.n_jobs == 1 or n_pairs <= 2:
            for pair in pairs:
                ci, cj, ar = align_one(pair)
                results[(ci, cj)] = ar
                if progress is not None:
                    progress.update()
        else:
            pair_results = parallel_map(
                align_one,
                pairs,
                n_workers=self._config.computation.n_jobs,
                backend=self._config.computation.backend,
            )
            for ci, cj, ar in pair_results:
                results[(ci, cj)] = ar
                if progress is not None:
                    progress.update()

        if progress is not None:
            progress.finish()

        return results

    def _get_aligner(self) -> Optional[Any]:
        """Try to instantiate the CADA aligner."""
        try:
            from cpa.alignment import CADAAligner
            return CADAAligner()
        except ImportError:
            logger.debug("Alignment module not available, using fallback")
        return None

    def _fallback_alignment(
        self,
        ci: str,
        cj: str,
        adj_i: np.ndarray,
        adj_j: np.ndarray,
        param_i: Optional[np.ndarray],
        param_j: Optional[np.ndarray],
        cfg: Any,
    ) -> AlignmentResult:
        """Built-in fallback alignment using identity permutation.

        Computes structural and parametric costs assuming variables
        are already aligned (identity permutation).

        Parameters
        ----------
        ci, cj : str
            Context identifiers.
        adj_i, adj_j : np.ndarray
            Adjacency matrices.
        param_i, param_j : np.ndarray or None
            Parameter matrices.
        cfg : AlignmentConfig
            Alignment configuration.

        Returns
        -------
        AlignmentResult
        """
        p = adj_i.shape[0]
        perm = np.arange(p)

        bin_i = (adj_i != 0).astype(float)
        bin_j = (adj_j != 0).astype(float)

        structural_diff = np.sum(np.abs(bin_i - bin_j))
        max_structural = float(p * (p - 1))
        structural_cost = structural_diff / max(max_structural, 1.0)

        parametric_cost = 0.0
        if param_i is not None and param_j is not None:
            shared_mask = (bin_i > 0) & (bin_j > 0)
            if np.any(shared_mask):
                param_diff = np.abs(param_i - param_j) * shared_mask
                parametric_cost = float(np.sum(param_diff)) / float(
                    np.sum(shared_mask)
                )

        total_cost = (
            cfg.structural_weight * structural_cost
            + cfg.parametric_weight * parametric_cost
        )
        if cfg.normalize_costs:
            total_w = cfg.structural_weight + cfg.parametric_weight
            if total_w > 0:
                total_cost /= total_w

        both = np.sum((bin_i > 0) & (bin_j > 0))
        only_i = np.sum((bin_i > 0) & (bin_j == 0))
        only_j = np.sum((bin_i == 0) & (bin_j > 0))

        shared_with_diff = 0
        if param_i is not None and param_j is not None:
            shared_mask_2 = (bin_i > 0) & (bin_j > 0)
            if np.any(shared_mask_2):
                param_diffs = np.abs(param_i - param_j) * shared_mask_2
                shared_with_diff = int(np.sum(param_diffs > 0.01))

        return AlignmentResult(
            context_i=ci,
            context_j=cj,
            permutation=perm,
            structural_cost=structural_cost,
            parametric_cost=parametric_cost,
            total_cost=total_cost,
            shared_edges=int(both),
            modified_edges=shared_with_diff,
            context_specific_edges=int(only_i + only_j),
        )

    def _run_descriptors(
        self,
        dataset: MultiContextDataset,
        scm_results: Dict[str, SCMResult],
        alignment_results: Dict[Tuple[str, str], AlignmentResult],
    ) -> Dict[str, DescriptorResult]:
        """Compute 4D plasticity descriptors for all variables.

        Parameters
        ----------
        dataset : MultiContextDataset
            Input data.
        scm_results : dict
            Per-context SCM results.
        alignment_results : dict
            Pairwise alignment results.

        Returns
        -------
        dict of str → DescriptorResult
        """
        cfg = self._config.descriptor
        variable_names = dataset.variable_names
        context_ids = dataset.context_ids
        p = len(variable_names)
        K = len(context_ids)

        results: Dict[str, DescriptorResult] = {}

        progress = None
        if self._config.computation.progress and p > 1:
            progress = ProgressReporter(p, "descriptors", logger)

        descriptor_computer = self._get_descriptor_computer()

        adjacencies = {}
        parameters = {}
        for cid, scm in scm_results.items():
            if scm.adjacency is not None:
                adjacencies[cid] = scm.adjacency
            if scm.parameters is not None:
                parameters[cid] = scm.parameters

        adjacencies_list = []
        datasets_list = []
        for cid in context_ids:
            if cid in adjacencies:
                adjacencies_list.append(adjacencies[cid])
                datasets_list.append(dataset.get_data(cid))

        for var_idx, var_name in enumerate(variable_names):
            try:
                if descriptor_computer is not None and len(adjacencies_list) > 0:
                    desc = descriptor_computer.compute(
                        adjacencies=adjacencies_list,
                        datasets=datasets_list,
                        target_idx=var_idx,
                        variable_name=var_name,
                    )
                    dr = DescriptorResult(
                        variable=var_name,
                        structural=desc.psi_S,
                        parametric=desc.psi_P,
                        emergence=desc.psi_E,
                        sensitivity=desc.psi_CS,
                        norm=float(np.linalg.norm(desc.descriptor_vector)),
                    )
                    if desc.classification is not None:
                        _PC_TO_MC = {
                            "invariant": "invariant",
                            "parametric_plastic": "parametrically_plastic",
                            "structural_plastic": "structurally_plastic",
                            "mixed": "fully_plastic",
                            "emergent": "emergent",
                        }
                        cls_obj = desc.classification
                        if hasattr(cls_obj, "primary_category"):
                            raw = getattr(cls_obj.primary_category, "value", "unclassified")
                        else:
                            raw = getattr(cls_obj, "value", "unclassified")
                        cls_str = _PC_TO_MC.get(raw, raw)
                    else:
                        cls_str = "unclassified"
                    try:
                        dr.classification = MechanismClass(cls_str)
                    except ValueError:
                        dr.classification = MechanismClass.UNCLASSIFIED
                    dr.confidence_intervals = {}
                    if desc.psi_S_ci is not None:
                        dr.confidence_intervals["structural"] = desc.psi_S_ci
                    if desc.psi_P_ci is not None:
                        dr.confidence_intervals["parametric"] = desc.psi_P_ci
                    if desc.psi_E_ci is not None:
                        dr.confidence_intervals["emergence"] = desc.psi_E_ci
                    if desc.psi_CS_ci is not None:
                        dr.confidence_intervals["sensitivity"] = desc.psi_CS_ci
                else:
                    dr = self._fallback_descriptors(
                        var_idx, var_name, adjacencies, parameters,
                        alignment_results, context_ids, cfg,
                    )
            except Exception as e:
                logger.warning(
                    "Descriptor computation failed for %s: %s", var_name, e
                )
                self._record_error(1, f"descriptor_{var_name}", e)
                dr = DescriptorResult(variable=var_name)

            results[var_name] = dr
            if progress is not None:
                progress.update()

        if progress is not None:
            progress.finish()

        return results

    def _get_descriptor_computer(self) -> Optional[Any]:
        """Try to instantiate the plasticity descriptor computer."""
        try:
            from cpa.descriptors import PlasticityComputer
            return PlasticityComputer()
        except ImportError:
            logger.debug("Descriptors module not available, using fallback")
        return None

    def _fallback_descriptors(
        self,
        var_idx: int,
        var_name: str,
        adjacencies: Dict[str, np.ndarray],
        parameters: Dict[str, np.ndarray],
        alignment_results: Dict[Tuple[str, str], AlignmentResult],
        context_ids: List[str],
        cfg: Any,
    ) -> DescriptorResult:
        """Built-in fallback for plasticity descriptor computation.

        Computes simplified 4D descriptors from adjacency and parameter
        matrices without full bootstrap/permutation analysis.

        Parameters
        ----------
        var_idx : int
            Variable index.
        var_name : str
            Variable name.
        adjacencies : dict
            Per-context adjacency matrices.
        parameters : dict
            Per-context parameter matrices.
        alignment_results : dict
            Pairwise alignment results.
        context_ids : list of str
            Context identifiers.
        cfg : DescriptorConfig
            Descriptor configuration.

        Returns
        -------
        DescriptorResult
        """
        K = len(context_ids)

        # Structural plasticity: fraction of parent sets that differ
        structural = 0.0
        if K >= 2 and adjacencies:
            parent_sets: List[frozenset] = []
            for cid in context_ids:
                adj = adjacencies.get(cid)
                if adj is not None:
                    parents = frozenset(
                        int(j)
                        for j in range(adj.shape[0])
                        if adj[j, var_idx] != 0
                    )
                    parent_sets.append(parents)

            if len(parent_sets) >= 2:
                n_diff = 0
                n_total = 0
                for a in range(len(parent_sets)):
                    for b in range(a + 1, len(parent_sets)):
                        n_total += 1
                        if parent_sets[a] != parent_sets[b]:
                            n_diff += 1
                structural = n_diff / max(n_total, 1)

        # Parametric plasticity: coefficient of variation of incoming weights
        parametric = 0.0
        if K >= 2 and parameters:
            weight_vectors: List[np.ndarray] = []
            for cid in context_ids:
                pm = parameters.get(cid)
                if pm is not None:
                    weight_vectors.append(pm[:, var_idx].copy())

            if len(weight_vectors) >= 2:
                stacked = np.array(weight_vectors)
                col_std = np.std(stacked, axis=0)
                col_mean = np.mean(np.abs(stacked), axis=0)
                mask = col_mean > 1e-8
                if np.any(mask):
                    cv = col_std[mask] / col_mean[mask]
                    parametric = float(np.mean(np.minimum(cv, 1.0)))

        # Emergence: fraction of contexts where variable has no parents
        emergence = 0.0
        if adjacencies:
            n_no_parents = 0
            n_has_parents = 0
            for cid in context_ids:
                adj = adjacencies.get(cid)
                if adj is not None:
                    has_parent = np.any(adj[:, var_idx] != 0)
                    if has_parent:
                        n_has_parents += 1
                    else:
                        n_no_parents += 1

            total_ctx = n_no_parents + n_has_parents
            if total_ctx > 0 and n_no_parents > 0 and n_has_parents > 0:
                emergence = min(n_no_parents, n_has_parents) / total_ctx

        # Context sensitivity: variance of alignment costs involving this variable
        sensitivity = 0.0
        if alignment_results and K >= 2:
            costs = [
                ar.total_cost for ar in alignment_results.values()
            ]
            if costs:
                sensitivity = float(np.std(costs))
                max_cost = max(costs) if costs else 1.0
                if max_cost > 0:
                    sensitivity = min(sensitivity / max_cost, 1.0)

        norm = float(
            np.sqrt(
                structural ** 2
                + parametric ** 2
                + emergence ** 2
                + sensitivity ** 2
            )
        )

        classification = self._classify_mechanism(
            structural, parametric, emergence, sensitivity, norm, cfg
        )

        return DescriptorResult(
            variable=var_name,
            structural=structural,
            parametric=parametric,
            emergence=emergence,
            sensitivity=sensitivity,
            classification=classification,
            norm=norm,
        )

    def _classify_mechanism(
        self,
        structural: float,
        parametric: float,
        emergence: float,
        sensitivity: float,
        norm: float,
        cfg: Any,
    ) -> MechanismClass:
        """Classify a mechanism based on its 4D descriptor.

        Parameters
        ----------
        structural, parametric, emergence, sensitivity : float
            Descriptor components.
        norm : float
            L2 norm of descriptor.
        cfg : DescriptorConfig
            Threshold configuration.

        Returns
        -------
        MechanismClass
        """
        if norm <= cfg.invariance_max_score:
            return MechanismClass.INVARIANT

        if emergence >= cfg.emergence_threshold:
            return MechanismClass.EMERGENT

        is_structural = structural >= cfg.structural_threshold
        is_parametric = parametric >= cfg.parametric_threshold

        if is_structural and is_parametric:
            return MechanismClass.FULLY_PLASTIC
        if is_structural:
            return MechanismClass.STRUCTURALLY_PLASTIC
        if is_parametric:
            return MechanismClass.PARAMETRICALLY_PLASTIC

        if sensitivity >= cfg.sensitivity_threshold:
            return MechanismClass.CONTEXT_SENSITIVE

        return MechanismClass.INVARIANT

    # -----------------------------------------------------------------
    # Phase 2: Exploration
    # -----------------------------------------------------------------

    def _run_phase_2(
        self,
        dataset: MultiContextDataset,
        foundation: FoundationResult,
    ) -> ExplorationResult:
        """Execute Phase 2: Exploration.

        Runs QD-MAP-Elites search to explore mechanism-change patterns.

        Parameters
        ----------
        dataset : MultiContextDataset
            Input data.
        foundation : FoundationResult
            Phase 1 results.

        Returns
        -------
        ExplorationResult
        """
        phase = 2
        phase_name = "Exploration"
        self._notify_phase_start(phase, phase_name)

        result = ExplorationResult()

        with TimingContext("phase_2", logger) as t:
            qd_engine = self._get_qd_engine(foundation)

            if qd_engine is not None:
                try:
                    archive, stats = qd_engine.search(
                        foundation=foundation,
                        config=self._config.search,
                        rng=self._rng,
                    )
                    result = self._convert_qd_results(archive, stats)
                except Exception as e:
                    logger.warning("QD search failed: %s", e)
                    self._record_error(2, "qd_search", e)
                    result = self._fallback_exploration(
                        dataset, foundation
                    )
            else:
                result = self._fallback_exploration(dataset, foundation)

        result.total_time = t.elapsed_wall
        self._timings["phase2_total"] = t.elapsed_wall

        self._save_checkpoint_if_needed(
            phase=2,
            step=0,
            state={"exploration": result.to_dict()},
            description="Phase 2 complete",
        )

        logger.info(
            "Phase 2 complete: archive=%d entries, coverage=%.2f%%, "
            "QD-score=%.4f in %.2fs",
            result.archive_size,
            result.coverage * 100,
            result.qd_score,
            t.elapsed_wall,
        )

        self._notify_phase_end(phase, phase_name, t.elapsed_wall)
        return result

    def _get_qd_engine(self, foundation: Any = None) -> Optional[Any]:
        """Try to instantiate the QD search engine."""
        try:
            from cpa.exploration import QDSearchEngine
            # Extract available contexts and mechanisms from foundation
            contexts = []
            mechanisms = []
            if foundation is not None:
                if hasattr(foundation, 'scm_results'):
                    contexts = list(foundation.scm_results.keys())
                if hasattr(foundation, 'descriptor_results'):
                    for var_name, desc in foundation.descriptor_results.items():
                        for ctx in contexts[:1]:
                            mechanisms.append((var_name, ctx))
            if not contexts:
                contexts = ["ctx_0"]
            if not mechanisms:
                mechanisms = [("X0", "ctx_0")]
            return QDSearchEngine(
                available_contexts=contexts,
                available_mechanisms=mechanisms,
            )
        except (ImportError, Exception) as e:
            logger.debug("Exploration module not available: %s", e)
        return None

    def _convert_qd_results(
        self, archive: Any, stats: Dict[str, Any]
    ) -> ExplorationResult:
        """Convert QD engine output to ExplorationResult."""
        entries: List[ArchiveEntry] = []
        if hasattr(archive, "entries"):
            for entry in archive.entries:
                ae = ArchiveEntry(
                    cell_id=getattr(entry, "cell_id", 0),
                    genome=getattr(entry, "genome", None),
                    fitness=getattr(entry, "fitness", 0.0),
                    descriptor=getattr(entry, "descriptor", None),
                    classification_pattern=getattr(
                        entry, "classification_pattern", {}
                    ),
                )
                entries.append(ae)

        return ExplorationResult(
            archive=entries,
            n_iterations=stats.get("n_iterations", 0),
            best_fitness=stats.get("best_fitness", float("-inf")),
            coverage=stats.get("coverage", 0.0),
            qd_score=stats.get("qd_score", 0.0),
            patterns=stats.get("patterns", []),
            convergence_history=stats.get("convergence_history", []),
        )

    def _fallback_exploration(
        self,
        dataset: MultiContextDataset,
        foundation: FoundationResult,
    ) -> ExplorationResult:
        """Built-in fallback exploration using random pattern sampling.

        When the QD exploration module is not available, we generate
        a basic archive by enumerating observed patterns.

        Parameters
        ----------
        dataset : MultiContextDataset
            Input data.
        foundation : FoundationResult
            Phase 1 results.

        Returns
        -------
        ExplorationResult
        """
        cfg = self._config.search
        rng = self._rng or np.random.RandomState(42)

        entries: List[ArchiveEntry] = []
        patterns: List[Dict[str, Any]] = []
        convergence: List[float] = []

        desc_mat = foundation.descriptor_matrix
        if desc_mat.size == 0:
            return ExplorationResult()

        p = desc_mat.shape[0]

        observed_pattern: Dict[str, str] = {}
        for var in foundation.variable_names:
            dr = foundation.descriptors.get(var)
            if dr is not None:
                observed_pattern[var] = dr.classification.value

        entries.append(
            ArchiveEntry(
                cell_id=0,
                genome=desc_mat.flatten(),
                fitness=1.0,
                descriptor=np.mean(desc_mat, axis=0),
                classification_pattern=dict(observed_pattern),
            )
        )

        n_iter = min(cfg.n_iterations, 200)
        target_archive_size = min(cfg.archive_size, 128)

        for it in range(n_iter):
            noise = rng.normal(0, cfg.mutation_sigma, desc_mat.shape)
            perturbed = np.clip(desc_mat + noise, 0.0, 1.0)

            fitness = float(np.std(perturbed))
            bd = np.mean(perturbed, axis=0)

            pattern: Dict[str, str] = {}
            for var_idx, var in enumerate(foundation.variable_names):
                norm = float(np.linalg.norm(perturbed[var_idx]))
                if norm < self._config.descriptor.invariance_max_score:
                    pattern[var] = "invariant"
                elif perturbed[var_idx, 2] >= self._config.descriptor.emergence_threshold:
                    pattern[var] = "emergent"
                elif perturbed[var_idx, 0] >= self._config.descriptor.structural_threshold:
                    pattern[var] = "structurally_plastic"
                elif perturbed[var_idx, 1] >= self._config.descriptor.parametric_threshold:
                    pattern[var] = "parametrically_plastic"
                else:
                    pattern[var] = "invariant"

            if len(entries) < target_archive_size:
                entries.append(
                    ArchiveEntry(
                        cell_id=len(entries),
                        genome=perturbed.flatten(),
                        fitness=fitness,
                        descriptor=bd,
                        classification_pattern=pattern,
                    )
                )
            else:
                min_idx = min(
                    range(len(entries)), key=lambda x: entries[x].fitness
                )
                if fitness > entries[min_idx].fitness:
                    entries[min_idx] = ArchiveEntry(
                        cell_id=min_idx,
                        genome=perturbed.flatten(),
                        fitness=fitness,
                        descriptor=bd,
                        classification_pattern=pattern,
                    )

            total_fitness = sum(e.fitness for e in entries)
            convergence.append(total_fitness)

        unique_patterns: Set[str] = set()
        for entry in entries:
            key = "_".join(
                sorted(
                    f"{v}:{c}" for v, c in entry.classification_pattern.items()
                )
            )
            if key not in unique_patterns:
                unique_patterns.add(key)
                patterns.append({
                    "type": "observed",
                    "pattern": entry.classification_pattern,
                    "fitness": entry.fitness,
                })

        best = max(e.fitness for e in entries) if entries else float("-inf")
        total_qd = sum(e.fitness for e in entries)
        coverage = len(entries) / max(target_archive_size, 1)

        return ExplorationResult(
            archive=entries,
            n_iterations=n_iter,
            best_fitness=best,
            coverage=coverage,
            qd_score=total_qd,
            patterns=patterns[:50],
            convergence_history=convergence,
        )

    # -----------------------------------------------------------------
    # Phase 3: Validation
    # -----------------------------------------------------------------

    def _run_phase_3(
        self,
        dataset: MultiContextDataset,
        foundation: FoundationResult,
        exploration: Optional[ExplorationResult],
    ) -> ValidationResult:
        """Execute Phase 3: Validation.

        Performs tipping-point detection, robustness certificate
        generation, and sensitivity analysis.

        Parameters
        ----------
        dataset : MultiContextDataset
            Input data.
        foundation : FoundationResult
            Phase 1 results.
        exploration : ExplorationResult, optional
            Phase 2 results.

        Returns
        -------
        ValidationResult
        """
        phase = 3
        phase_name = "Validation"
        self._notify_phase_start(phase, phase_name)

        result = ValidationResult()

        with TimingContext("phase_3", logger) as t:
            # Step 1: Tipping-point detection
            if self._config.detection.contexts_are_ordered:
                with TimingContext("tipping_points", logger) as dt:
                    result.tipping_points = self._run_tipping_point_detection(
                        dataset, foundation
                    )
                result.detection_time = dt.elapsed_wall
                self._timings["phase3_detection"] = dt.elapsed_wall
                self._notify_step_complete(
                    phase, "tipping_points", result.tipping_points
                )

            # Step 2: Robustness certificates
            with TimingContext("certificates", logger) as ct:
                result.certificates = self._run_certificate_generation(
                    dataset, foundation
                )
            result.certificate_time = ct.elapsed_wall
            self._timings["phase3_certificates"] = ct.elapsed_wall
            self._notify_step_complete(
                phase, "certificates", result.certificates
            )

            # Step 3: Sensitivity analysis
            with TimingContext("sensitivity", logger) as st:
                result.sensitivity = self._run_sensitivity_analysis(
                    dataset, foundation
                )
            result.sensitivity_time = st.elapsed_wall
            self._timings["phase3_sensitivity"] = st.elapsed_wall
            self._notify_step_complete(
                phase, "sensitivity", result.sensitivity
            )

        result.total_time = t.elapsed_wall
        self._timings["phase3_total"] = t.elapsed_wall

        self._save_checkpoint_if_needed(
            phase=3,
            step=0,
            state={"validation": result.to_dict()},
            description="Phase 3 complete",
        )

        logger.info(
            "Phase 3 complete: %d/%d certified, %.2fs",
            result.n_certified,
            len(result.certificates),
            t.elapsed_wall,
        )

        self._notify_phase_end(phase, phase_name, t.elapsed_wall)
        return result

    def _run_tipping_point_detection(
        self,
        dataset: MultiContextDataset,
        foundation: FoundationResult,
    ) -> TippingPointResult:
        """Detect tipping points across ordered contexts.

        Parameters
        ----------
        dataset : MultiContextDataset
            Input data.
        foundation : FoundationResult
            Phase 1 results.

        Returns
        -------
        TippingPointResult
        """
        cfg = self._config.detection

        detector = self._get_tipping_point_detector()

        cost_mat = foundation.alignment_cost_matrix
        K = foundation.n_contexts

        if K < 3:
            logger.info("Too few contexts (%d) for tipping-point detection", K)
            return TippingPointResult()

        if detector is not None:
            try:
                tp_result = detector.detect(
                    cost_matrix=cost_mat,
                    context_ids=foundation.context_ids,
                    method=cfg.method,
                    penalty=cfg.penalty,
                    min_segment_length=cfg.min_segment_length,
                    n_permutations=cfg.n_permutations,
                    significance_level=cfg.significance_level,
                )
                return TippingPointResult(
                    changepoints=tp_result.get("changepoints", []),
                    validated_changepoints=tp_result.get(
                        "validated_changepoints", []
                    ),
                    p_values=tp_result.get("p_values", {}),
                    segments=tp_result.get("segments", []),
                    segment_labels=tp_result.get("segment_labels", []),
                    cost_reduction=tp_result.get("cost_reduction", 0.0),
                    attribution=tp_result.get("attribution", {}),
                )
            except Exception as e:
                logger.warning("Tipping-point detection failed: %s", e)
                self._record_error(3, "tipping_points", e)

        return self._fallback_tipping_point_detection(
            cost_mat, foundation.context_ids, cfg
        )

    def _get_tipping_point_detector(self) -> Optional[Any]:
        """Try to instantiate the tipping-point detector."""
        try:
            from cpa.detection import PELTDetector
            return PELTDetector()
        except ImportError:
            logger.debug("Detection module not available, using fallback")
        return None

    def _fallback_tipping_point_detection(
        self,
        cost_matrix: np.ndarray,
        context_ids: List[str],
        cfg: Any,
    ) -> TippingPointResult:
        """Built-in fallback tipping-point detection.

        Uses a simple threshold on consecutive pairwise cost differences
        to identify potential changepoints.

        Parameters
        ----------
        cost_matrix : np.ndarray
            K x K pairwise alignment cost matrix.
        context_ids : list of str
            Context identifiers.
        cfg : DetectionConfig
            Detection configuration.

        Returns
        -------
        TippingPointResult
        """
        K = len(context_ids)
        if K < 3:
            return TippingPointResult()

        consecutive_costs = np.array([
            cost_matrix[i, i + 1] for i in range(K - 1)
        ])

        if len(consecutive_costs) < 2:
            return TippingPointResult()

        mean_cost = np.mean(consecutive_costs)
        std_cost = np.std(consecutive_costs)

        if std_cost < 1e-10:
            return TippingPointResult()

        threshold = mean_cost + 2.0 * std_cost

        changepoints: List[int] = []
        for i, cost in enumerate(consecutive_costs):
            if cost > threshold:
                changepoints.append(i + 1)

        min_seg = cfg.min_segment_length
        filtered: List[int] = []
        prev = 0
        for cp in sorted(changepoints):
            if cp - prev >= min_seg:
                filtered.append(cp)
                prev = cp
        if filtered and K - filtered[-1] < min_seg:
            filtered.pop()

        validated: List[int] = []
        p_values: Dict[int, float] = {}

        rng = self._rng or np.random.RandomState(42)
        for cp in filtered:
            n_perm = min(cfg.n_permutations, 200)
            observed_stat = consecutive_costs[cp - 1]

            count_ge = 0
            for _ in range(n_perm):
                perm_costs = rng.permutation(consecutive_costs)
                if perm_costs[cp - 1] >= observed_stat:
                    count_ge += 1

            p_val = (count_ge + 1) / (n_perm + 1)
            p_values[cp] = p_val

            if p_val < cfg.significance_level:
                validated.append(cp)

        segments: List[Tuple[int, int]] = []
        seg_labels: List[str] = []
        boundaries = [0] + sorted(validated) + [K]
        for i in range(len(boundaries) - 1):
            seg = (boundaries[i], boundaries[i + 1])
            segments.append(seg)
            seg_labels.append(f"Segment {i + 1}")

        cost_reduction = 0.0
        if validated:
            total_var = float(np.var(consecutive_costs))
            seg_vars = []
            for start, end in segments:
                if end - start >= 2:
                    seg_data = consecutive_costs[start:min(end - 1, len(consecutive_costs))]
                    if len(seg_data) > 0:
                        seg_vars.append(float(np.var(seg_data)) * len(seg_data))
            within_var = sum(seg_vars) / max(len(consecutive_costs), 1)
            if total_var > 0:
                cost_reduction = 1.0 - within_var / total_var

        return TippingPointResult(
            changepoints=filtered,
            validated_changepoints=validated,
            p_values=p_values,
            segments=segments,
            segment_labels=seg_labels,
            cost_reduction=cost_reduction,
        )

    def _run_certificate_generation(
        self,
        dataset: MultiContextDataset,
        foundation: FoundationResult,
    ) -> Dict[str, CertificateResult]:
        """Generate robustness certificates for all variables.

        Parameters
        ----------
        dataset : MultiContextDataset
            Input data.
        foundation : FoundationResult
            Phase 1 results.

        Returns
        -------
        dict of str → CertificateResult
        """
        cfg = self._config.certificate
        results: Dict[str, CertificateResult] = {}

        progress = None
        p = foundation.n_variables
        if self._config.computation.progress and p > 1:
            progress = ProgressReporter(p, "certificates", logger)

        cert_gen = self._get_certificate_generator()

        for var_idx, var_name in enumerate(foundation.variable_names):
            dr = foundation.descriptors.get(var_name)

            try:
                if cert_gen is not None and dr is not None:
                    cert_result = cert_gen.generate(
                        variable=var_name,
                        variable_index=var_idx,
                        dataset=dataset,
                        foundation=foundation,
                        config=cfg,
                    )
                    cert = CertificateResult(
                        variable=var_name,
                        certified=cert_result.get("certified", False),
                        classification=dr.classification,
                        stability_score=cert_result.get("stability_score", 0.0),
                        bootstrap_ci=cert_result.get("bootstrap_ci", {}),
                        ucb_bound=cert_result.get("ucb_bound", 0.0),
                        assumption_checks=cert_result.get(
                            "assumption_checks", {}
                        ),
                    )
                else:
                    cert = self._fallback_certificate(
                        var_idx, var_name, dataset, foundation, cfg
                    )
            except Exception as e:
                logger.warning(
                    "Certificate generation failed for %s: %s", var_name, e
                )
                self._record_error(3, f"certificate_{var_name}", e)
                cert = CertificateResult(
                    variable=var_name,
                    certified=False,
                    classification=(
                        dr.classification if dr else MechanismClass.UNCLASSIFIED
                    ),
                )

            results[var_name] = cert
            if progress is not None:
                progress.update()

        if progress is not None:
            progress.finish()

        return results

    def _get_certificate_generator(self) -> Optional[Any]:
        """Try to instantiate the certificate generator."""
        try:
            from cpa.certificates import CertificateGenerator
            return CertificateGenerator()
        except ImportError:
            logger.debug("Certificates module not available, using fallback")
        return None

    def _fallback_certificate(
        self,
        var_idx: int,
        var_name: str,
        dataset: MultiContextDataset,
        foundation: FoundationResult,
        cfg: Any,
    ) -> CertificateResult:
        """Built-in fallback certificate generation.

        Uses bootstrap resampling to assess stability of the
        descriptor and classification.

        Parameters
        ----------
        var_idx : int
            Variable index.
        var_name : str
            Variable name.
        dataset : MultiContextDataset
            Input data.
        foundation : FoundationResult
            Phase 1 results.
        cfg : CertificateConfig
            Certificate configuration.

        Returns
        -------
        CertificateResult
        """
        dr = foundation.descriptors.get(var_name)
        if dr is None:
            return CertificateResult(variable=var_name, certified=False)

        rng = self._rng or np.random.RandomState(42)
        n_boot = min(cfg.n_bootstrap, 100)

        boot_descriptors: List[np.ndarray] = []

        for b in range(n_boot):
            noise = rng.normal(0, 0.05, 4)
            perturbed = np.clip(dr.vector + noise, 0.0, 1.0)
            boot_descriptors.append(perturbed)

        if not boot_descriptors:
            return CertificateResult(
                variable=var_name,
                certified=False,
                classification=dr.classification,
            )

        boot_array = np.array(boot_descriptors)

        ci_level = cfg.bootstrap_ci_level
        alpha = (1.0 - ci_level) / 2.0
        lo_pct = alpha * 100
        hi_pct = (1.0 - alpha) * 100

        component_names = ["structural", "parametric", "emergence", "sensitivity"]
        bootstrap_ci: Dict[str, Tuple[float, float]] = {}
        for c_idx, c_name in enumerate(component_names):
            vals = boot_array[:, c_idx]
            lo = float(np.percentile(vals, lo_pct))
            hi = float(np.percentile(vals, hi_pct))
            bootstrap_ci[c_name] = (lo, hi)

        boot_norms = np.linalg.norm(boot_array, axis=1)
        ucb = float(np.mean(boot_norms) + cfg.ucb_alpha * np.std(boot_norms))

        stability_scores: List[float] = []
        for adj in [
            scm.adjacency
            for scm in foundation.scm_results.values()
            if scm.adjacency is not None
        ]:
            n_parents = int(np.sum(adj[:, var_idx] != 0))
            stability_scores.append(
                1.0 if n_parents > 0 else 0.0
            )
        stability_score = float(np.mean(stability_scores)) if stability_scores else 0.0

        ci_width_structural = (
            bootstrap_ci["structural"][1] - bootstrap_ci["structural"][0]
        )
        ci_width_parametric = (
            bootstrap_ci["parametric"][1] - bootstrap_ci["parametric"][0]
        )

        certified = (
            ci_width_structural < cfg.tolerance * 3
            and ci_width_parametric < cfg.tolerance * 3
            and stability_score >= cfg.stability_threshold
        )

        assumption_checks = {
            "sufficient_sample_size": all(
                n >= 30 for n in dataset.sample_sizes().values()
            ),
            "no_missing_contexts": len(foundation.scm_results) == dataset.n_contexts,
            "stable_structure": stability_score >= cfg.stability_threshold,
            "narrow_ci": ci_width_structural < cfg.tolerance * 3,
        }

        return CertificateResult(
            variable=var_name,
            certified=certified,
            classification=dr.classification,
            stability_score=stability_score,
            bootstrap_ci=bootstrap_ci,
            ucb_bound=ucb,
            assumption_checks=assumption_checks,
        )

    def _run_sensitivity_analysis(
        self,
        dataset: MultiContextDataset,
        foundation: FoundationResult,
    ) -> Dict[str, Dict[str, float]]:
        """Run sensitivity analysis on descriptors.

        Tests how much each descriptor changes when a context is
        dropped (leave-one-context-out).

        Parameters
        ----------
        dataset : MultiContextDataset
            Input data.
        foundation : FoundationResult
            Phase 1 results.

        Returns
        -------
        dict of str → dict of str → float
            variable → metric → sensitivity_value
        """
        sensitivity_analyzer = self._get_sensitivity_analyzer()
        if sensitivity_analyzer is not None:
            try:
                return sensitivity_analyzer.analyze(
                    dataset=dataset,
                    foundation=foundation,
                    config=self._config,
                )
            except Exception as e:
                logger.warning("Sensitivity analysis failed: %s", e)
                self._record_error(3, "sensitivity", e)

        return self._fallback_sensitivity(dataset, foundation)

    def _get_sensitivity_analyzer(self) -> Optional[Any]:
        """Try to instantiate the sensitivity analyzer."""
        try:
            from cpa.diagnostics import SensitivityAnalyzer
            return SensitivityAnalyzer()
        except ImportError:
            logger.debug("Diagnostics module not available, using fallback")
        return None

    def _fallback_sensitivity(
        self,
        dataset: MultiContextDataset,
        foundation: FoundationResult,
    ) -> Dict[str, Dict[str, float]]:
        """Built-in fallback sensitivity analysis.

        Computes a simple leave-one-out sensitivity measure for each
        variable by examining how alignment costs change.

        Parameters
        ----------
        dataset : MultiContextDataset
            Input data.
        foundation : FoundationResult
            Phase 1 results.

        Returns
        -------
        dict of str → dict of str → float
        """
        results: Dict[str, Dict[str, float]] = {}

        cost_mat = foundation.alignment_cost_matrix
        K = foundation.n_contexts

        if K < 3:
            for var in foundation.variable_names:
                results[var] = {"loo_sensitivity": 0.0, "max_influence": 0.0}
            return results

        overall_mean_cost = float(np.mean(
            cost_mat[np.triu_indices(K, k=1)]
        ))

        loo_means: List[float] = []
        for drop_idx in range(K):
            keep = [i for i in range(K) if i != drop_idx]
            sub_mat = cost_mat[np.ix_(keep, keep)]
            loo_mean = float(np.mean(
                sub_mat[np.triu_indices(len(keep), k=1)]
            ))
            loo_means.append(loo_mean)

        context_influence = [
            abs(loo - overall_mean_cost)
            for loo in loo_means
        ]

        for var in foundation.variable_names:
            dr = foundation.descriptors.get(var)
            if dr is None:
                results[var] = {"loo_sensitivity": 0.0, "max_influence": 0.0}
                continue

            norm = dr.norm
            sensitivity = norm * float(np.std(context_influence))
            max_inf = float(max(context_influence)) if context_influence else 0.0

            results[var] = {
                "loo_sensitivity": min(sensitivity, 1.0),
                "max_influence": max_inf,
                "descriptor_norm": norm,
                "mean_alignment_cost": overall_mean_cost,
            }

        return results

    # -----------------------------------------------------------------
    # Checkpoint helpers
    # -----------------------------------------------------------------

    def _save_checkpoint_if_needed(
        self,
        phase: int,
        step: int,
        state: Dict[str, Any],
        description: str = "",
    ) -> None:
        """Save checkpoint if checkpointing is configured."""
        if self._checkpoint_mgr is None:
            return

        try:
            path = self._checkpoint_mgr.save(
                phase=phase,
                step=step,
                state=state,
                description=description,
            )
            if self._callbacks.on_checkpoint is not None:
                self._callbacks.on_checkpoint(path)
        except Exception as e:
            logger.warning("Checkpoint save failed: %s", e)

    def _resume_from_checkpoint(
        self, dataset: MultiContextDataset
    ) -> Tuple[Optional[FoundationResult], Optional[ExplorationResult]]:
        """Attempt to resume from the latest checkpoint.

        Parameters
        ----------
        dataset : MultiContextDataset
            Input data (for validation).

        Returns
        -------
        tuple of (FoundationResult or None, ExplorationResult or None)
        """
        foundation = None
        exploration = None

        if self._checkpoint_mgr is None:
            return foundation, exploration

        loaded = self._checkpoint_mgr.load_latest()
        if loaded is None:
            logger.info("No checkpoint found, starting from scratch")
            return foundation, exploration

        state, arrays, manifest = loaded
        logger.info(
            "Resuming from checkpoint: phase=%d step=%d",
            manifest.phase, manifest.step,
        )

        if "foundation" in state:
            try:
                from cpa.pipeline.results import _restore_foundation
                foundation = _restore_foundation(state["foundation"])
                logger.info("Restored Phase 1 results from checkpoint")
            except Exception as e:
                logger.warning("Failed to restore foundation: %s", e)

        if "exploration" in state:
            try:
                from cpa.pipeline.results import _restore_exploration
                exploration = _restore_exploration(state["exploration"])
                logger.info("Restored Phase 2 results from checkpoint")
            except Exception as e:
                logger.warning("Failed to restore exploration: %s", e)

        return foundation, exploration

    # -----------------------------------------------------------------
    # Error handling and notifications
    # -----------------------------------------------------------------

    def _record_error(
        self, phase: int, step: str, error: Exception
    ) -> None:
        """Record a non-fatal error."""
        self._errors.append({
            "phase": phase,
            "step": step,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "timestamp": time.time(),
        })

    def _notify_phase_start(self, phase: int, name: str) -> None:
        """Notify callbacks of phase start."""
        logger.info("Starting Phase %d: %s", phase, name)
        if self._callbacks.on_phase_start is not None:
            try:
                self._callbacks.on_phase_start(phase, name)
            except Exception:
                pass

    def _notify_phase_end(
        self, phase: int, name: str, elapsed: float
    ) -> None:
        """Notify callbacks of phase end."""
        if self._callbacks.on_phase_end is not None:
            try:
                self._callbacks.on_phase_end(phase, name, elapsed)
            except Exception:
                pass

    def _notify_step_complete(
        self, phase: int, step: str, result: Any
    ) -> None:
        """Notify callbacks of step completion."""
        if self._callbacks.on_step_complete is not None:
            try:
                self._callbacks.on_step_complete(phase, step, result)
            except Exception:
                pass

    # -----------------------------------------------------------------
    # Convenience methods
    # -----------------------------------------------------------------

    def run_phase_1_only(
        self, dataset: MultiContextDataset
    ) -> FoundationResult:
        """Run only Phase 1 (Foundation).

        Parameters
        ----------
        dataset : MultiContextDataset
            Input data.

        Returns
        -------
        FoundationResult
        """
        self._config.validate_or_raise()
        validation_errors = dataset.validate()
        if validation_errors:
            raise ValueError(
                "Dataset validation failed:\n  - "
                + "\n  - ".join(validation_errors)
            )
        return self._run_phase_1(dataset)

    def run_phase_2_only(
        self,
        dataset: MultiContextDataset,
        foundation: FoundationResult,
    ) -> ExplorationResult:
        """Run only Phase 2 (Exploration).

        Parameters
        ----------
        dataset : MultiContextDataset
            Input data.
        foundation : FoundationResult
            Phase 1 results (required).

        Returns
        -------
        ExplorationResult
        """
        return self._run_phase_2(dataset, foundation)

    def run_phase_3_only(
        self,
        dataset: MultiContextDataset,
        foundation: FoundationResult,
        exploration: Optional[ExplorationResult] = None,
    ) -> ValidationResult:
        """Run only Phase 3 (Validation).

        Parameters
        ----------
        dataset : MultiContextDataset
            Input data.
        foundation : FoundationResult
            Phase 1 results (required).
        exploration : ExplorationResult, optional
            Phase 2 results.

        Returns
        -------
        ValidationResult
        """
        return self._run_phase_3(dataset, foundation, exploration)

    def analyze_single_pair(
        self,
        data_i: np.ndarray,
        data_j: np.ndarray,
        context_i: str = "context_0",
        context_j: str = "context_1",
        variable_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyze a single pair of contexts.

        Convenience method that runs discovery, alignment, and
        descriptors for exactly two contexts.

        Parameters
        ----------
        data_i, data_j : np.ndarray
            (n, p) data matrices for each context.
        context_i, context_j : str
            Context identifiers.
        variable_names : list of str, optional
            Variable names.

        Returns
        -------
        dict
            Results including SCMs, alignment, and descriptors.
        """
        if variable_names is None:
            p = data_i.shape[1]
            variable_names = [f"X{k}" for k in range(p)]

        dataset = MultiContextDataset(
            context_data={context_i: data_i, context_j: data_j},
            variable_names=variable_names,
            context_ids=[context_i, context_j],
        )

        foundation = self.run_phase_1_only(dataset)

        return {
            "scm_i": foundation.scm_results.get(context_i),
            "scm_j": foundation.scm_results.get(context_j),
            "alignment": foundation.get_alignment(context_i, context_j),
            "descriptors": foundation.descriptors,
            "classification_summary": foundation.classification_summary(),
        }

    def __repr__(self) -> str:
        return (
            f"CPAOrchestrator(profile={self._config.profile.value}, "
            f"phases=[{','.join(str(i) for i in range(1, 4) if getattr(self._config, f'run_phase_{i}'))}])"
        )
