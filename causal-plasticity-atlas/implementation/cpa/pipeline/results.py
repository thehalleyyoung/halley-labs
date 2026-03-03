"""Result containers for the CPA pipeline.

Provides structured dataclasses for holding pipeline outputs at each
phase (Foundation, Exploration, Validation) and the aggregate
:class:`AtlasResult` for the complete Causal-Plasticity Atlas.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np


# =====================================================================
# Classification enum
# =====================================================================


class MechanismClass(Enum):
    """Mechanism plasticity classification."""

    INVARIANT = "invariant"
    STRUCTURALLY_PLASTIC = "structurally_plastic"
    PARAMETRICALLY_PLASTIC = "parametrically_plastic"
    FULLY_PLASTIC = "fully_plastic"
    EMERGENT = "emergent"
    CONTEXT_SENSITIVE = "context_sensitive"
    UNCLASSIFIED = "unclassified"


# =====================================================================
# Lightweight result records
# =====================================================================


@dataclass
class SCMResult:
    """Result of causal discovery for a single context.

    Attributes
    ----------
    context_id : str
        Context identifier.
    adjacency : np.ndarray
        Adjacency matrix (p x p).
    parameters : Optional[np.ndarray]
        Weight matrix (p x p) if estimated.
    variable_names : List[str]
        Ordered variable names.
    n_samples : int
        Number of samples used.
    discovery_method : str
        Algorithm used.
    fit_time : float
        Wall-clock seconds for discovery.
    metadata : Dict[str, Any]
        Additional info from the discovery adapter.
    """

    context_id: str = ""
    adjacency: Optional[np.ndarray] = None
    parameters: Optional[np.ndarray] = None
    variable_names: List[str] = field(default_factory=list)
    n_samples: int = 0
    discovery_method: str = ""
    fit_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_variables(self) -> int:
        """Number of variables."""
        if self.adjacency is not None:
            return self.adjacency.shape[0]
        return len(self.variable_names)

    @property
    def n_vars(self) -> int:
        """Alias for n_variables (compatibility with CADA alignment)."""
        return self.n_variables

    @property
    def adjacency_matrix(self) -> Optional[np.ndarray]:
        """Alias for adjacency (compatibility with CADA alignment)."""
        return self.adjacency

    @property
    def residual_variances(self) -> np.ndarray:
        """Residual variances (compatibility with CADA alignment)."""
        if isinstance(self.parameters, dict) and "residual_variances" in self.parameters:
            return np.asarray(self.parameters["residual_variances"])
        p = self.n_variables
        return np.ones(p)

    @property
    def regression_coefficients(self) -> np.ndarray:
        """Regression coefficients (compatibility with CADA alignment)."""
        if isinstance(self.parameters, dict) and "coefficients" in self.parameters:
            return np.asarray(self.parameters["coefficients"])
        if self.adjacency is not None:
            return self.adjacency.copy()
        return np.zeros((self.n_variables, self.n_variables))

    @property
    def n_edges(self) -> int:
        """Number of directed edges."""
        if self.adjacency is not None:
            return int(np.sum(self.adjacency != 0))
        return 0

    @property
    def density(self) -> float:
        """Edge density p(edge)."""
        p = self.n_variables
        if p < 2:
            return 0.0
        return self.n_edges / (p * (p - 1))

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "context_id": self.context_id,
            "variable_names": self.variable_names,
            "n_samples": self.n_samples,
            "discovery_method": self.discovery_method,
            "fit_time": self.fit_time,
            "n_variables": self.n_variables,
            "n_edges": self.n_edges,
            "density": self.density,
            "metadata": self.metadata,
        }
        if self.adjacency is not None:
            d["adjacency"] = self.adjacency.tolist()
        if self.parameters is not None:
            if hasattr(self.parameters, 'tolist'):
                d["parameters"] = self.parameters.tolist()
            else:
                d["parameters"] = self.parameters
        return d


@dataclass
class AlignmentResult:
    """Result of pairwise CADA alignment.

    Attributes
    ----------
    context_i : str
        First context identifier.
    context_j : str
        Second context identifier.
    permutation : Optional[np.ndarray]
        Variable mapping (permutation of range(p)).
    structural_cost : float
        Structural component of alignment cost.
    parametric_cost : float
        Parametric component of alignment cost.
    total_cost : float
        Total CADA alignment cost.
    shared_edges : int
        Number of edges present in both contexts (after alignment).
    modified_edges : int
        Edges with changed parameters.
    context_specific_edges : int
        Edges present in only one context.
    align_time : float
        Wall-clock seconds for alignment.
    """

    context_i: str = ""
    context_j: str = ""
    permutation: Optional[np.ndarray] = None
    structural_cost: float = 0.0
    parametric_cost: float = 0.0
    total_cost: float = 0.0
    shared_edges: int = 0
    modified_edges: int = 0
    context_specific_edges: int = 0
    align_time: float = 0.0

    @property
    def pair_key(self) -> Tuple[str, str]:
        return (self.context_i, self.context_j)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "context_i": self.context_i,
            "context_j": self.context_j,
            "structural_cost": self.structural_cost,
            "parametric_cost": self.parametric_cost,
            "total_cost": self.total_cost,
            "shared_edges": self.shared_edges,
            "modified_edges": self.modified_edges,
            "context_specific_edges": self.context_specific_edges,
            "align_time": self.align_time,
        }
        if self.permutation is not None:
            d["permutation"] = (
                self.permutation.tolist()
                if hasattr(self.permutation, "tolist")
                else list(self.permutation) if not isinstance(self.permutation, dict) else self.permutation
            )
        return d


@dataclass
class DescriptorResult:
    """Plasticity descriptor for a single variable.

    Attributes
    ----------
    variable : str
        Variable name.
    structural : float
        Structural plasticity score in [0, 1].
    parametric : float
        Parametric plasticity score in [0, 1].
    emergence : float
        Emergence score in [0, 1].
    sensitivity : float
        Context-sensitivity score in [0, 1].
    classification : MechanismClass
        Mechanism classification.
    confidence_intervals : Dict[str, Tuple[float, float]]
        Bootstrap CIs for each descriptor component.
    norm : float
        L2 norm of the 4D descriptor vector.
    """

    variable: str = ""
    structural: float = 0.0
    parametric: float = 0.0
    emergence: float = 0.0
    sensitivity: float = 0.0
    classification: MechanismClass = MechanismClass.UNCLASSIFIED
    confidence_intervals: Dict[str, Tuple[float, float]] = field(
        default_factory=dict
    )
    norm: float = 0.0

    @property
    def vector(self) -> np.ndarray:
        """4D descriptor vector."""
        return np.array(
            [self.structural, self.parametric, self.emergence, self.sensitivity]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "structural": self.structural,
            "parametric": self.parametric,
            "emergence": self.emergence,
            "sensitivity": self.sensitivity,
            "classification": self.classification.value,
            "confidence_intervals": {
                k: list(v) for k, v in self.confidence_intervals.items()
            },
            "norm": self.norm,
        }


@dataclass
class TippingPointResult:
    """Result of tipping-point detection.

    Attributes
    ----------
    changepoints : List[int]
        Indices of detected changepoints in context ordering.
    validated_changepoints : List[int]
        Changepoints that passed permutation significance test.
    p_values : Dict[int, float]
        P-value for each detected changepoint.
    segments : List[Tuple[int, int]]
        (start, end) ranges of homogeneous segments.
    segment_labels : List[str]
        Human-readable label for each segment.
    cost_reduction : float
        Total cost reduction achieved by segmentation.
    attribution : Dict[int, Dict[str, float]]
        Per-changepoint attribution to individual mechanisms.
    """

    changepoints: List[int] = field(default_factory=list)
    validated_changepoints: List[int] = field(default_factory=list)
    p_values: Dict[int, float] = field(default_factory=dict)
    segments: List[Tuple[int, int]] = field(default_factory=list)
    segment_labels: List[str] = field(default_factory=list)
    cost_reduction: float = 0.0
    attribution: Dict[int, Dict[str, float]] = field(default_factory=dict)

    @property
    def n_changepoints(self) -> int:
        return len(self.validated_changepoints)

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "changepoints": self.changepoints,
            "validated_changepoints": self.validated_changepoints,
            "p_values": {str(k): v for k, v in self.p_values.items()},
            "segments": [list(s) for s in self.segments],
            "segment_labels": self.segment_labels,
            "cost_reduction": self.cost_reduction,
            "n_changepoints": self.n_changepoints,
            "attribution": {
                str(k): v for k, v in self.attribution.items()
            },
        }


@dataclass
class CertificateResult:
    """Robustness certificate for a single variable.

    Attributes
    ----------
    variable : str
        Variable name.
    certified : bool
        Whether the classification is certified robust.
    classification : MechanismClass
        Certified mechanism classification.
    stability_score : float
        Edge stability selection score.
    bootstrap_ci : Dict[str, Tuple[float, float]]
        Bootstrap CIs for descriptor under perturbation.
    ucb_bound : float
        Upper confidence bound on descriptor norm.
    assumption_checks : Dict[str, bool]
        Results of assumption validation tests.
    """

    variable: str = ""
    certified: bool = False
    classification: MechanismClass = MechanismClass.UNCLASSIFIED
    stability_score: float = 0.0
    bootstrap_ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    ucb_bound: float = 0.0
    assumption_checks: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "certified": self.certified,
            "classification": self.classification.value,
            "stability_score": self.stability_score,
            "bootstrap_ci": {
                k: list(v) for k, v in self.bootstrap_ci.items()
            },
            "ucb_bound": self.ucb_bound,
            "assumption_checks": self.assumption_checks,
        }


@dataclass
class ArchiveEntry:
    """Single entry in the QD archive.

    Attributes
    ----------
    cell_id : int
        CVT cell index.
    genome : np.ndarray
        Mechanism-configuration genome.
    fitness : float
        Fitness value.
    descriptor : np.ndarray
        Behaviour descriptor (4D).
    classification_pattern : Dict[str, str]
        Per-variable classification in this configuration.
    metadata : Dict[str, Any]
        Auxiliary metadata.
    """

    cell_id: int = 0
    genome: Optional[np.ndarray] = None
    fitness: float = 0.0
    descriptor: Optional[np.ndarray] = None
    classification_pattern: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "cell_id": self.cell_id,
            "fitness": self.fitness,
            "classification_pattern": self.classification_pattern,
            "metadata": self.metadata,
        }
        if self.genome is not None:
            d["genome"] = self.genome.tolist()
        if self.descriptor is not None:
            d["descriptor"] = self.descriptor.tolist()
        return d


# =====================================================================
# Phase-level results
# =====================================================================


@dataclass
class FoundationResult:
    """Phase 1 (Foundation) outputs.

    Attributes
    ----------
    scm_results : Dict[str, SCMResult]
        Per-context causal discovery results, keyed by context_id.
    alignment_results : Dict[Tuple[str, str], AlignmentResult]
        Pairwise alignment results, keyed by (context_i, context_j).
    descriptors : Dict[str, DescriptorResult]
        Per-variable plasticity descriptors, keyed by variable name.
    context_ids : List[str]
        Ordered list of context identifiers.
    variable_names : List[str]
        Ordered list of variable names.
    total_time : float
        Total wall-clock time for Phase 1.
    discovery_time : float
        Time spent on causal discovery.
    alignment_time : float
        Time spent on alignment.
    descriptor_time : float
        Time spent on descriptor computation.
    """

    scm_results: Dict[str, SCMResult] = field(default_factory=dict)
    alignment_results: Dict[Tuple[str, str], AlignmentResult] = field(
        default_factory=dict
    )
    descriptors: Dict[str, DescriptorResult] = field(default_factory=dict)
    context_ids: List[str] = field(default_factory=list)
    variable_names: List[str] = field(default_factory=list)
    total_time: float = 0.0
    discovery_time: float = 0.0
    alignment_time: float = 0.0
    descriptor_time: float = 0.0

    @property
    def n_contexts(self) -> int:
        return len(self.context_ids)

    @property
    def n_variables(self) -> int:
        return len(self.variable_names)

    @property
    def n_pairs(self) -> int:
        return len(self.alignment_results)

    @property
    def alignment_cost_matrix(self) -> np.ndarray:
        """K x K matrix of pairwise alignment costs."""
        k = self.n_contexts
        mat = np.zeros((k, k))
        cid_to_idx = {c: i for i, c in enumerate(self.context_ids)}
        for (ci, cj), ar in self.alignment_results.items():
            i, j = cid_to_idx.get(ci), cid_to_idx.get(cj)
            if i is not None and j is not None:
                mat[i, j] = ar.total_cost
                mat[j, i] = ar.total_cost
        return mat

    @property
    def descriptor_matrix(self) -> np.ndarray:
        """p x 4 matrix of descriptor vectors."""
        if not self.descriptors:
            return np.empty((0, 4))
        vecs = []
        for v in self.variable_names:
            if v in self.descriptors:
                vecs.append(self.descriptors[v].vector)
            else:
                vecs.append(np.zeros(4))
        return np.array(vecs)

    def classification_summary(self) -> Dict[str, int]:
        """Count variables per classification category."""
        counts: Dict[str, int] = {}
        for dr in self.descriptors.values():
            cls_name = dr.classification.value
            counts[cls_name] = counts.get(cls_name, 0) + 1
        return counts

    def variables_by_class(
        self, cls: MechanismClass
    ) -> List[str]:
        """Return variable names with the given classification."""
        return [
            v
            for v, dr in self.descriptors.items()
            if dr.classification == cls
        ]

    def get_alignment(self, ci: str, cj: str) -> Optional[AlignmentResult]:
        """Look up alignment result for a context pair (order-agnostic)."""
        if (ci, cj) in self.alignment_results:
            return self.alignment_results[(ci, cj)]
        if (cj, ci) in self.alignment_results:
            return self.alignment_results[(cj, ci)]
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_ids": self.context_ids,
            "variable_names": self.variable_names,
            "n_contexts": self.n_contexts,
            "n_variables": self.n_variables,
            "n_pairs": self.n_pairs,
            "total_time": self.total_time,
            "discovery_time": self.discovery_time,
            "alignment_time": self.alignment_time,
            "descriptor_time": self.descriptor_time,
            "scm_results": {
                k: v.to_dict() for k, v in self.scm_results.items()
            },
            "alignment_results": {
                f"{ci}__{cj}": ar.to_dict()
                for (ci, cj), ar in self.alignment_results.items()
            },
            "descriptors": {
                k: v.to_dict() for k, v in self.descriptors.items()
            },
            "classification_summary": self.classification_summary(),
        }


@dataclass
class ExplorationResult:
    """Phase 2 (Exploration) outputs.

    Attributes
    ----------
    archive : List[ArchiveEntry]
        QD archive entries.
    n_iterations : int
        Number of search iterations completed.
    best_fitness : float
        Best fitness found.
    coverage : float
        Archive coverage (fraction of cells filled).
    qd_score : float
        Sum of fitnesses in the archive.
    patterns : List[Dict[str, Any]]
        Extracted mechanism-change patterns.
    convergence_history : List[float]
        QD-score per iteration.
    total_time : float
        Wall-clock time for Phase 2.
    """

    archive: List[ArchiveEntry] = field(default_factory=list)
    n_iterations: int = 0
    best_fitness: float = float("-inf")
    coverage: float = 0.0
    qd_score: float = 0.0
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    convergence_history: List[float] = field(default_factory=list)
    total_time: float = 0.0

    @property
    def archive_size(self) -> int:
        return len(self.archive)

    def top_entries(self, n: int = 10) -> List[ArchiveEntry]:
        """Return top-*n* archive entries by fitness."""
        return sorted(self.archive, key=lambda e: e.fitness, reverse=True)[:n]

    def pattern_summary(self) -> Dict[str, int]:
        """Count occurrences of each pattern type."""
        counts: Dict[str, int] = {}
        for pat in self.patterns:
            ptype = pat.get("type", "unknown")
            counts[ptype] = counts.get(ptype, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "archive_size": self.archive_size,
            "n_iterations": self.n_iterations,
            "best_fitness": self.best_fitness,
            "coverage": self.coverage,
            "qd_score": self.qd_score,
            "patterns": self.patterns,
            "convergence_history": self.convergence_history,
            "total_time": self.total_time,
            "archive": [e.to_dict() for e in self.archive],
        }


@dataclass
class ValidationResult:
    """Phase 3 (Validation) outputs.

    Attributes
    ----------
    tipping_points : Optional[TippingPointResult]
        Tipping-point detection results (None if contexts not ordered).
    certificates : Dict[str, CertificateResult]
        Per-variable robustness certificates.
    sensitivity : Dict[str, Dict[str, float]]
        Sensitivity analysis results (variable → {metric: value}).
    diagnostics : Dict[str, Any]
        Additional diagnostic information.
    total_time : float
        Wall-clock time for Phase 3.
    detection_time : float
        Time for tipping-point detection.
    certificate_time : float
        Time for certificate generation.
    sensitivity_time : float
        Time for sensitivity analysis.
    """

    tipping_points: Optional[TippingPointResult] = None
    certificates: Dict[str, CertificateResult] = field(default_factory=dict)
    sensitivity: Dict[str, Dict[str, float]] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    total_time: float = 0.0
    detection_time: float = 0.0
    certificate_time: float = 0.0
    sensitivity_time: float = 0.0

    @property
    def n_certified(self) -> int:
        """Number of variables with certified classifications."""
        return sum(1 for c in self.certificates.values() if c.certified)

    @property
    def certification_rate(self) -> float:
        """Fraction of variables certified robust."""
        total = len(self.certificates)
        if total == 0:
            return 0.0
        return self.n_certified / total

    def certified_variables(self) -> List[str]:
        """Return names of certified variables."""
        return [v for v, c in self.certificates.items() if c.certified]

    def uncertified_variables(self) -> List[str]:
        """Return names of uncertified variables."""
        return [v for v, c in self.certificates.items() if not c.certified]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "n_certified": self.n_certified,
            "certification_rate": self.certification_rate,
            "total_time": self.total_time,
            "detection_time": self.detection_time,
            "certificate_time": self.certificate_time,
            "sensitivity_time": self.sensitivity_time,
            "certificates": {
                k: v.to_dict() for k, v in self.certificates.items()
            },
            "sensitivity": self.sensitivity,
            "diagnostics": self.diagnostics,
        }
        if self.tipping_points is not None:
            d["tipping_points"] = self.tipping_points.to_dict()
        return d


# =====================================================================
# Master AtlasResult
# =====================================================================


@dataclass
class AtlasResult:
    """Complete Causal-Plasticity Atlas result.

    Aggregates outputs from all three phases into a single queryable
    structure.  Provides accessor methods for common queries, summary
    statistics, and serialisation.

    Attributes
    ----------
    foundation : Optional[FoundationResult]
        Phase 1 outputs.
    exploration : Optional[ExplorationResult]
        Phase 2 outputs.
    validation : Optional[ValidationResult]
        Phase 3 outputs.
    config : Dict[str, Any]
        Configuration used for this run.
    metadata : Dict[str, Any]
        Run metadata (timestamps, version, etc.).
    """

    foundation: Optional[FoundationResult] = None
    exploration: Optional[ExplorationResult] = None
    validation: Optional[ValidationResult] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------------------
    # Convenience accessors
    # -----------------------------------------------------------------

    @property
    def context_ids(self) -> List[str]:
        """Ordered context identifiers."""
        if self.foundation is not None:
            return self.foundation.context_ids
        return []

    @property
    def variable_names(self) -> List[str]:
        """Ordered variable names."""
        if self.foundation is not None:
            return self.foundation.variable_names
        return []

    @property
    def n_contexts(self) -> int:
        return len(self.context_ids)

    @property
    def n_variables(self) -> int:
        return len(self.variable_names)

    @property
    def total_time(self) -> float:
        """Total wall-clock time for all phases."""
        t = 0.0
        if self.foundation is not None:
            t += self.foundation.total_time
        if self.exploration is not None:
            t += self.exploration.total_time
        if self.validation is not None:
            t += self.validation.total_time
        return t

    # -----------------------------------------------------------------
    # Descriptor queries
    # -----------------------------------------------------------------

    def get_descriptor(self, variable: str) -> Optional[DescriptorResult]:
        """Return the plasticity descriptor for *variable*."""
        if self.foundation is not None:
            return self.foundation.descriptors.get(variable)
        return None

    def get_classification(self, variable: str) -> MechanismClass:
        """Return the classification for *variable*."""
        dr = self.get_descriptor(variable)
        if dr is not None:
            return dr.classification
        return MechanismClass.UNCLASSIFIED

    def classification_summary(self) -> Dict[str, int]:
        """Count variables per classification category."""
        if self.foundation is not None:
            return self.foundation.classification_summary()
        return {}

    def variables_by_class(self, cls: MechanismClass) -> List[str]:
        """Return variables with the given classification."""
        if self.foundation is not None:
            return self.foundation.variables_by_class(cls)
        return []

    def filter_variables(
        self,
        classification: Optional[MechanismClass] = None,
        min_structural: Optional[float] = None,
        max_structural: Optional[float] = None,
        min_parametric: Optional[float] = None,
        max_parametric: Optional[float] = None,
        min_emergence: Optional[float] = None,
        max_emergence: Optional[float] = None,
        certified_only: bool = False,
    ) -> List[str]:
        """Filter variables by descriptor criteria.

        Parameters
        ----------
        classification : MechanismClass, optional
            Filter to this classification only.
        min_structural, max_structural : float, optional
            Range for structural score.
        min_parametric, max_parametric : float, optional
            Range for parametric score.
        min_emergence, max_emergence : float, optional
            Range for emergence score.
        certified_only : bool
            If True, only include certified variables.

        Returns
        -------
        list of str
            Variable names matching all criteria.
        """
        result: List[str] = []
        if self.foundation is None:
            return result

        for var in self.variable_names:
            dr = self.foundation.descriptors.get(var)
            if dr is None:
                continue

            if classification is not None and dr.classification != classification:
                continue
            if min_structural is not None and dr.structural < min_structural:
                continue
            if max_structural is not None and dr.structural > max_structural:
                continue
            if min_parametric is not None and dr.parametric < min_parametric:
                continue
            if max_parametric is not None and dr.parametric > max_parametric:
                continue
            if min_emergence is not None and dr.emergence < min_emergence:
                continue
            if max_emergence is not None and dr.emergence > max_emergence:
                continue

            if certified_only and self.validation is not None:
                cert = self.validation.certificates.get(var)
                if cert is None or not cert.certified:
                    continue

            result.append(var)
        return result

    # -----------------------------------------------------------------
    # Alignment queries
    # -----------------------------------------------------------------

    def get_alignment(self, ci: str, cj: str) -> Optional[AlignmentResult]:
        """Look up alignment result for a context pair."""
        if self.foundation is not None:
            return self.foundation.get_alignment(ci, cj)
        return None

    def alignment_cost_matrix(self) -> np.ndarray:
        """K x K matrix of pairwise alignment costs."""
        if self.foundation is not None:
            return self.foundation.alignment_cost_matrix
        return np.empty((0, 0))

    def most_similar_contexts(
        self, n: int = 5
    ) -> List[Tuple[str, str, float]]:
        """Return the *n* most similar context pairs by alignment cost."""
        if self.foundation is None:
            return []
        pairs: List[Tuple[str, str, float]] = []
        for (ci, cj), ar in self.foundation.alignment_results.items():
            pairs.append((ci, cj, ar.total_cost))
        pairs.sort(key=lambda x: x[2])
        return pairs[:n]

    def most_different_contexts(
        self, n: int = 5
    ) -> List[Tuple[str, str, float]]:
        """Return the *n* most dissimilar context pairs by alignment cost."""
        if self.foundation is None:
            return []
        pairs: List[Tuple[str, str, float]] = []
        for (ci, cj), ar in self.foundation.alignment_results.items():
            pairs.append((ci, cj, ar.total_cost))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:n]

    # -----------------------------------------------------------------
    # Certificate queries
    # -----------------------------------------------------------------

    def is_certified(self, variable: str) -> bool:
        """Check if *variable* has a robustness certificate."""
        if self.validation is None:
            return False
        cert = self.validation.certificates.get(variable)
        return cert is not None and cert.certified

    def certification_rate(self) -> float:
        """Fraction of variables with robustness certificates."""
        if self.validation is not None:
            return self.validation.certification_rate
        return 0.0

    # -----------------------------------------------------------------
    # Tipping-point queries
    # -----------------------------------------------------------------

    def has_tipping_points(self) -> bool:
        """Whether tipping points were detected."""
        if self.validation is None or self.validation.tipping_points is None:
            return False
        return self.validation.tipping_points.n_changepoints > 0

    def tipping_point_locations(self) -> List[int]:
        """Return validated tipping-point indices."""
        if self.validation is not None and self.validation.tipping_points is not None:
            return list(self.validation.tipping_points.validated_changepoints)
        return []

    # -----------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------

    def summary_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics across all phases.

        Returns
        -------
        dict
            Nested dictionary of summary statistics.
        """
        stats: Dict[str, Any] = {
            "n_contexts": self.n_contexts,
            "n_variables": self.n_variables,
            "total_time_seconds": self.total_time,
        }

        if self.foundation is not None:
            desc_mat = self.foundation.descriptor_matrix
            if desc_mat.size > 0:
                stats["descriptors"] = {
                    "mean_structural": float(np.mean(desc_mat[:, 0])),
                    "mean_parametric": float(np.mean(desc_mat[:, 1])),
                    "mean_emergence": float(np.mean(desc_mat[:, 2])),
                    "mean_sensitivity": float(np.mean(desc_mat[:, 3])),
                    "std_structural": float(np.std(desc_mat[:, 0])),
                    "std_parametric": float(np.std(desc_mat[:, 1])),
                    "std_emergence": float(np.std(desc_mat[:, 2])),
                    "std_sensitivity": float(np.std(desc_mat[:, 3])),
                }
            stats["classification_summary"] = self.classification_summary()

            cost_mat = self.foundation.alignment_cost_matrix
            if cost_mat.size > 0:
                upper = cost_mat[np.triu_indices_from(cost_mat, k=1)]
                if upper.size > 0:
                    stats["alignment"] = {
                        "mean_cost": float(np.mean(upper)),
                        "median_cost": float(np.median(upper)),
                        "max_cost": float(np.max(upper)),
                        "min_cost": float(np.min(upper)),
                    }

            scm_edges = [
                scm.n_edges for scm in self.foundation.scm_results.values()
            ]
            if scm_edges:
                stats["graphs"] = {
                    "mean_edges": float(np.mean(scm_edges)),
                    "min_edges": int(min(scm_edges)),
                    "max_edges": int(max(scm_edges)),
                    "mean_density": float(
                        np.mean(
                            [
                                scm.density
                                for scm in self.foundation.scm_results.values()
                            ]
                        )
                    ),
                }

        if self.exploration is not None:
            stats["exploration"] = {
                "archive_size": self.exploration.archive_size,
                "coverage": self.exploration.coverage,
                "qd_score": self.exploration.qd_score,
                "best_fitness": self.exploration.best_fitness,
                "n_patterns": len(self.exploration.patterns),
            }

        if self.validation is not None:
            stats["validation"] = {
                "n_certified": self.validation.n_certified,
                "certification_rate": self.validation.certification_rate,
            }
            if self.validation.tipping_points is not None:
                stats["validation"]["n_tipping_points"] = (
                    self.validation.tipping_points.n_changepoints
                )

        return stats

    # -----------------------------------------------------------------
    # Plasticity heatmap data
    # -----------------------------------------------------------------

    def plasticity_heatmap(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """Return data for a plasticity heatmap.

        Returns
        -------
        matrix : np.ndarray
            Shape (n_variables, 4).  Columns: structural, parametric,
            emergence, sensitivity.
        variable_names : list of str
        component_names : list of str
        """
        if self.foundation is None:
            return np.empty((0, 4)), [], []
        mat = self.foundation.descriptor_matrix
        components = ["structural", "parametric", "emergence", "sensitivity"]
        return mat, list(self.variable_names), components

    # -----------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert the full atlas to a nested dictionary."""
        d: Dict[str, Any] = {
            "metadata": self.metadata,
            "config": self.config,
            "summary": self.summary_statistics(),
        }
        if self.foundation is not None:
            d["foundation"] = self.foundation.to_dict()
        if self.exploration is not None:
            d["exploration"] = self.exploration.to_dict()
        if self.validation is not None:
            d["validation"] = self.validation.to_dict()
        return d

    def to_json(self, path: Optional[Union[str, Path]] = None) -> str:
        """Serialize to JSON string.

        Parameters
        ----------
        path : str or Path, optional
            If given, write to this file.

        Returns
        -------
        str
            JSON representation.
        """
        d = self.to_dict()
        text = json.dumps(d, indent=2, default=_json_default)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text)
        return text

    @classmethod
    def from_json(cls, source: Union[str, Path]) -> "AtlasResult":
        """Load from a JSON file or string.

        Reconstructs the AtlasResult with basic types (numpy arrays
        as lists, enums as strings).  For full-fidelity restoration,
        use :func:`cpa.io.serialization.deserialize_atlas`.

        Parameters
        ----------
        source : str or Path
            File path or JSON string.

        Returns
        -------
        AtlasResult
        """
        p = Path(source)
        if p.exists():
            text = p.read_text()
        else:
            text = str(source)
        data = json.loads(text)
        result = cls()
        result.metadata = data.get("metadata", {})
        result.config = data.get("config", {})
        if "foundation" in data:
            result.foundation = _restore_foundation(data["foundation"])
        if "exploration" in data:
            result.exploration = _restore_exploration(data["exploration"])
        if "validation" in data:
            result.validation = _restore_validation(data["validation"])
        return result

    def save(self, directory: Union[str, Path]) -> Path:
        """Save the atlas to a directory structure.

        Creates ``directory/`` with:
        - ``atlas.json`` — full serialized atlas
        - ``descriptors.csv`` — plasticity descriptors table
        - ``summary.json`` — summary statistics

        Parameters
        ----------
        directory : str or Path
            Output directory (created if needed).

        Returns
        -------
        Path
            Path to the output directory.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        self.to_json(out / "atlas.json")

        stats = self.summary_statistics()
        with open(out / "summary.json", "w") as f:
            json.dump(stats, f, indent=2, default=_json_default)

        if self.foundation is not None and self.foundation.descriptors:
            lines = [
                "variable,structural,parametric,emergence,sensitivity,classification"
            ]
            for v in self.variable_names:
                dr = self.foundation.descriptors.get(v)
                if dr is not None:
                    lines.append(
                        f"{v},{dr.structural:.6f},{dr.parametric:.6f},"
                        f"{dr.emergence:.6f},{dr.sensitivity:.6f},"
                        f"{dr.classification.value}"
                    )
            (out / "descriptors.csv").write_text("\n".join(lines))

        return out

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "AtlasResult":
        """Load an atlas from a directory created by :meth:`save`.

        Parameters
        ----------
        directory : str or Path
            Path to atlas directory.

        Returns
        -------
        AtlasResult
        """
        d = Path(directory)
        atlas_path = d / "atlas.json"
        if atlas_path.exists():
            return cls.from_json(atlas_path)
        raise FileNotFoundError(f"No atlas.json found in {d}")

    def __repr__(self) -> str:
        parts = [f"AtlasResult(n_ctx={self.n_contexts}, n_var={self.n_variables}"]
        if self.foundation is not None:
            parts.append(f"foundation=✓")
        if self.exploration is not None:
            parts.append(f"exploration=✓")
        if self.validation is not None:
            parts.append(f"validation=✓")
        return ", ".join(parts) + ")"


# =====================================================================
# Helpers
# =====================================================================


def _json_default(obj: Any) -> Any:
    """JSON default serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _restore_foundation(data: Dict[str, Any]) -> FoundationResult:
    """Reconstruct FoundationResult from a dictionary."""
    fr = FoundationResult()
    fr.context_ids = data.get("context_ids", [])
    fr.variable_names = data.get("variable_names", [])
    fr.total_time = data.get("total_time", 0.0)
    fr.discovery_time = data.get("discovery_time", 0.0)
    fr.alignment_time = data.get("alignment_time", 0.0)
    fr.descriptor_time = data.get("descriptor_time", 0.0)

    for cid, sd in data.get("scm_results", {}).items():
        scm = SCMResult(
            context_id=cid,
            variable_names=sd.get("variable_names", []),
            n_samples=sd.get("n_samples", 0),
            discovery_method=sd.get("discovery_method", ""),
            fit_time=sd.get("fit_time", 0.0),
            metadata=sd.get("metadata", {}),
        )
        if "adjacency" in sd:
            scm.adjacency = np.array(sd["adjacency"])
        if "parameters" in sd:
            scm.parameters = np.array(sd["parameters"])
        fr.scm_results[cid] = scm

    for key, ad in data.get("alignment_results", {}).items():
        parts = key.split("__", 1)
        if len(parts) == 2:
            ci, cj = parts
        else:
            ci, cj = key, ""
        ar = AlignmentResult(
            context_i=ci,
            context_j=cj,
            structural_cost=ad.get("structural_cost", 0.0),
            parametric_cost=ad.get("parametric_cost", 0.0),
            total_cost=ad.get("total_cost", 0.0),
            shared_edges=ad.get("shared_edges", 0),
            modified_edges=ad.get("modified_edges", 0),
            context_specific_edges=ad.get("context_specific_edges", 0),
            align_time=ad.get("align_time", 0.0),
        )
        if "permutation" in ad:
            ar.permutation = np.array(ad["permutation"])
        fr.alignment_results[(ci, cj)] = ar

    for var, dd in data.get("descriptors", {}).items():
        dr = DescriptorResult(
            variable=var,
            structural=dd.get("structural", 0.0),
            parametric=dd.get("parametric", 0.0),
            emergence=dd.get("emergence", 0.0),
            sensitivity=dd.get("sensitivity", 0.0),
            norm=dd.get("norm", 0.0),
        )
        cls_str = dd.get("classification", "unclassified")
        try:
            dr.classification = MechanismClass(cls_str)
        except ValueError:
            dr.classification = MechanismClass.UNCLASSIFIED
        ci_data = dd.get("confidence_intervals", {})
        dr.confidence_intervals = {
            k: tuple(v) for k, v in ci_data.items()
        }
        fr.descriptors[var] = dr

    return fr


def _restore_exploration(data: Dict[str, Any]) -> ExplorationResult:
    """Reconstruct ExplorationResult from a dictionary."""
    er = ExplorationResult(
        n_iterations=data.get("n_iterations", 0),
        best_fitness=data.get("best_fitness", float("-inf")),
        coverage=data.get("coverage", 0.0),
        qd_score=data.get("qd_score", 0.0),
        patterns=data.get("patterns", []),
        convergence_history=data.get("convergence_history", []),
        total_time=data.get("total_time", 0.0),
    )

    for entry_d in data.get("archive", []):
        entry = ArchiveEntry(
            cell_id=entry_d.get("cell_id", 0),
            fitness=entry_d.get("fitness", 0.0),
            classification_pattern=entry_d.get("classification_pattern", {}),
            metadata=entry_d.get("metadata", {}),
        )
        if "genome" in entry_d:
            entry.genome = np.array(entry_d["genome"])
        if "descriptor" in entry_d:
            entry.descriptor = np.array(entry_d["descriptor"])
        er.archive.append(entry)

    return er


def _restore_validation(data: Dict[str, Any]) -> ValidationResult:
    """Reconstruct ValidationResult from a dictionary."""
    vr = ValidationResult(
        total_time=data.get("total_time", 0.0),
        detection_time=data.get("detection_time", 0.0),
        certificate_time=data.get("certificate_time", 0.0),
        sensitivity_time=data.get("sensitivity_time", 0.0),
        sensitivity=data.get("sensitivity", {}),
        diagnostics=data.get("diagnostics", {}),
    )

    if "tipping_points" in data:
        tp = data["tipping_points"]
        vr.tipping_points = TippingPointResult(
            changepoints=tp.get("changepoints", []),
            validated_changepoints=tp.get("validated_changepoints", []),
            p_values={int(k): v for k, v in tp.get("p_values", {}).items()},
            segments=[tuple(s) for s in tp.get("segments", [])],
            segment_labels=tp.get("segment_labels", []),
            cost_reduction=tp.get("cost_reduction", 0.0),
            attribution={
                int(k): v for k, v in tp.get("attribution", {}).items()
            },
        )

    for var, cd in data.get("certificates", {}).items():
        cert = CertificateResult(
            variable=var,
            certified=cd.get("certified", False),
            stability_score=cd.get("stability_score", 0.0),
            ucb_bound=cd.get("ucb_bound", 0.0),
            assumption_checks=cd.get("assumption_checks", {}),
        )
        cls_str = cd.get("classification", "unclassified")
        try:
            cert.classification = MechanismClass(cls_str)
        except ValueError:
            cert.classification = MechanismClass.UNCLASSIFIED
        ci_data = cd.get("bootstrap_ci", {})
        cert.bootstrap_ci = {k: tuple(v) for k, v in ci_data.items()}
        vr.certificates[var] = cert

    return vr
