"""Pipeline configuration for the Causal-Plasticity Atlas.

Provides :class:`PipelineConfig` — the master configuration dataclass — and
sub-configurations for each pipeline phase.  Also defines named profiles
(fast / standard / thorough) and YAML/JSON loaders.
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union


# =====================================================================
# Enumerations
# =====================================================================


class ConfigProfile(Enum):
    """Named configuration profiles."""

    FAST = "fast"
    STANDARD = "standard"
    THOROUGH = "thorough"
    CUSTOM = "custom"


class DiscoveryMethod(Enum):
    """Supported causal discovery algorithms."""

    PC = "pc"
    GES = "ges"
    LINGAM = "lingam"
    FALLBACK = "fallback"


class CITestType(Enum):
    """Conditional independence test types."""

    FISHER_Z = "fisher_z"
    PARTIAL_CORRELATION = "partial_correlation"
    KCI = "kci"
    CHI_SQUARE = "chi_square"


class AlignmentStrategy(Enum):
    """CADA alignment strategies."""

    GREEDY = "greedy"
    EXACT = "exact"
    SPECTRAL = "spectral"


class SearchStrategy(Enum):
    """QD search strategies."""

    MAP_ELITES = "map_elites"
    CURIOSITY_DRIVEN = "curiosity_driven"
    RANDOM = "random"


class ChangePointMethod(Enum):
    """Tipping-point detection algorithms."""

    PELT = "pelt"
    BINARY_SEGMENTATION = "binary_segmentation"
    CUSUM = "cusum"


# =====================================================================
# Sub-configuration dataclasses
# =====================================================================


@dataclass
class DiscoveryConfig:
    """Configuration for Phase 1: Causal discovery.

    Parameters
    ----------
    method : str
        Discovery algorithm name ('pc', 'ges', 'lingam', 'fallback').
    ci_test : str
        Conditional independence test ('fisher_z', 'partial_correlation', etc.).
    alpha : float
        Significance level for CI tests.
    max_cond_set_size : int
        Maximum conditioning set size (-1 for unlimited).
    score_function : str
        Score function for score-based methods ('bic', 'aic').
    estimate_parameters : bool
        Whether to estimate SCM parameters after structure learning.
    max_iterations : int
        Maximum iterations for iterative algorithms.
    stable : bool
        Use order-independent variant of PC algorithm.
    use_fallback : bool
        Fall back to built-in implementation if external library unavailable.
    """

    method: str = "pc"
    ci_test: str = "fisher_z"
    alpha: float = 0.05
    max_cond_set_size: int = -1
    score_function: str = "bic"
    estimate_parameters: bool = True
    max_iterations: int = 1000
    stable: bool = True
    use_fallback: bool = True

    def validate(self) -> List[str]:
        """Validate this configuration, returning a list of error messages."""
        errors: List[str] = []
        valid_methods = {e.value for e in DiscoveryMethod}
        if self.method not in valid_methods:
            errors.append(
                f"discovery.method must be one of {valid_methods}, got {self.method!r}"
            )
        valid_tests = {e.value for e in CITestType}
        if self.ci_test not in valid_tests:
            errors.append(
                f"discovery.ci_test must be one of {valid_tests}, got {self.ci_test!r}"
            )
        if not 0.0 < self.alpha < 1.0:
            errors.append(f"discovery.alpha must be in (0, 1), got {self.alpha}")
        if self.max_cond_set_size < -1:
            errors.append(
                f"discovery.max_cond_set_size must be >= -1, got {self.max_cond_set_size}"
            )
        if self.max_iterations < 1:
            errors.append(
                f"discovery.max_iterations must be >= 1, got {self.max_iterations}"
            )
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiscoveryConfig":
        """Create from dictionary, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class AlignmentConfig:
    """Configuration for CADA pairwise alignment.

    Parameters
    ----------
    strategy : str
        Alignment strategy ('greedy', 'exact', 'spectral').
    structural_weight : float
        Weight for structural similarity in CADA cost.
    parametric_weight : float
        Weight for parametric similarity in CADA cost.
    lambda_s : float
        Edge-existence mismatch penalty.
    lambda_p : float
        Parameter divergence penalty.
    max_permutation_size : int
        Maximum variable count for exact alignment (falls back to greedy above).
    use_warm_start : bool
        Initialize greedy search with spectral matching.
    convergence_tol : float
        Convergence tolerance for iterative alignment.
    max_alignment_iterations : int
        Maximum iterations for iterative alignment.
    normalize_costs : bool
        Normalize alignment costs to [0, 1].
    """

    strategy: str = "greedy"
    structural_weight: float = 0.5
    parametric_weight: float = 0.5
    lambda_s: float = 1.0
    lambda_p: float = 1.0
    max_permutation_size: int = 8
    use_warm_start: bool = True
    convergence_tol: float = 1e-6
    max_alignment_iterations: int = 100
    normalize_costs: bool = True

    def validate(self) -> List[str]:
        """Validate this configuration."""
        errors: List[str] = []
        valid_strats = {e.value for e in AlignmentStrategy}
        if self.strategy not in valid_strats:
            errors.append(
                f"alignment.strategy must be one of {valid_strats}, "
                f"got {self.strategy!r}"
            )
        if self.structural_weight < 0:
            errors.append(
                f"alignment.structural_weight must be >= 0, "
                f"got {self.structural_weight}"
            )
        if self.parametric_weight < 0:
            errors.append(
                f"alignment.parametric_weight must be >= 0, "
                f"got {self.parametric_weight}"
            )
        total = self.structural_weight + self.parametric_weight
        if total <= 0:
            errors.append("Sum of alignment weights must be positive")
        if self.lambda_s < 0:
            errors.append(f"alignment.lambda_s must be >= 0, got {self.lambda_s}")
        if self.lambda_p < 0:
            errors.append(f"alignment.lambda_p must be >= 0, got {self.lambda_p}")
        if self.max_permutation_size < 2:
            errors.append(
                f"alignment.max_permutation_size must be >= 2, "
                f"got {self.max_permutation_size}"
            )
        if self.convergence_tol <= 0:
            errors.append(
                f"alignment.convergence_tol must be > 0, got {self.convergence_tol}"
            )
        if self.max_alignment_iterations < 1:
            errors.append(
                f"alignment.max_alignment_iterations must be >= 1, "
                f"got {self.max_alignment_iterations}"
            )
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AlignmentConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class DescriptorConfig:
    """Configuration for plasticity descriptor computation (ALG2).

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap samples for confidence intervals.
    bootstrap_ci_level : float
        Confidence interval level (e.g. 0.95).
    n_permutations : int
        Number of permutations for null distribution calibration.
    structural_threshold : float
        Threshold for structural plasticity classification.
    parametric_threshold : float
        Threshold for parametric plasticity classification.
    emergence_threshold : float
        Threshold for emergence classification.
    sensitivity_threshold : float
        Threshold for context-sensitivity classification.
    invariance_max_score : float
        Maximum 4D-norm for invariant classification.
    auto_threshold : bool
        Automatically calibrate thresholds via permutation test.
    use_stability_selection : bool
        Apply stability selection for structural descriptor.
    stability_n_subsamples : int
        Number of subsamples for stability selection.
    stability_fraction : float
        Subsample fraction for stability selection.
    """

    n_bootstrap: int = 200
    bootstrap_ci_level: float = 0.95
    n_permutations: int = 500
    structural_threshold: float = 0.3
    parametric_threshold: float = 0.3
    emergence_threshold: float = 0.2
    sensitivity_threshold: float = 0.3
    invariance_max_score: float = 0.1
    auto_threshold: bool = False
    use_stability_selection: bool = False
    stability_n_subsamples: int = 100
    stability_fraction: float = 0.7

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.n_bootstrap < 10:
            errors.append(
                f"descriptor.n_bootstrap must be >= 10, got {self.n_bootstrap}"
            )
        if not 0.5 <= self.bootstrap_ci_level < 1.0:
            errors.append(
                f"descriptor.bootstrap_ci_level must be in [0.5, 1.0), "
                f"got {self.bootstrap_ci_level}"
            )
        if self.n_permutations < 10:
            errors.append(
                f"descriptor.n_permutations must be >= 10, got {self.n_permutations}"
            )
        for name in [
            "structural_threshold",
            "parametric_threshold",
            "emergence_threshold",
            "sensitivity_threshold",
        ]:
            val = getattr(self, name)
            if not 0.0 <= val <= 1.0:
                errors.append(f"descriptor.{name} must be in [0, 1], got {val}")
        if self.invariance_max_score < 0:
            errors.append(
                f"descriptor.invariance_max_score must be >= 0, "
                f"got {self.invariance_max_score}"
            )
        if self.stability_n_subsamples < 10:
            errors.append(
                f"descriptor.stability_n_subsamples must be >= 10, "
                f"got {self.stability_n_subsamples}"
            )
        if not 0.1 <= self.stability_fraction <= 0.9:
            errors.append(
                f"descriptor.stability_fraction must be in [0.1, 0.9], "
                f"got {self.stability_fraction}"
            )
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DescriptorConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SearchConfig:
    """Configuration for Phase 2: QD-MAP-Elites search (ALG3).

    Parameters
    ----------
    strategy : str
        Search strategy ('map_elites', 'curiosity_driven', 'random').
    n_iterations : int
        Number of search iterations.
    archive_size : int
        Target number of CVT cells in the archive.
    n_cvt_samples : int
        Number of samples for CVT initialization.
    batch_size : int
        Number of genomes evaluated per iteration.
    mutation_rate : float
        Probability of mutating each gene.
    mutation_sigma : float
        Standard deviation of Gaussian mutation.
    crossover_rate : float
        Probability of crossover between parents.
    curiosity_weight : float
        Weight for curiosity bonus in fitness.
    novelty_k : int
        Number of nearest neighbours for novelty computation.
    surprise_decay : float
        Exponential decay factor for surprise signal.
    elite_fraction : float
        Fraction of archive to preserve as elites.
    min_fitness : float
        Minimum fitness threshold for archive admission.
    descriptor_bounds : Optional[List[Tuple[float, float]]]
        Bounds for behaviour descriptor dimensions. None = auto-detect.
    """

    strategy: str = "curiosity_driven"
    n_iterations: int = 500
    archive_size: int = 256
    n_cvt_samples: int = 25000
    batch_size: int = 32
    mutation_rate: float = 0.2
    mutation_sigma: float = 0.1
    crossover_rate: float = 0.3
    curiosity_weight: float = 0.5
    novelty_k: int = 15
    surprise_decay: float = 0.95
    elite_fraction: float = 0.1
    min_fitness: float = -float("inf")
    descriptor_bounds: Optional[List[Tuple[float, float]]] = None

    def validate(self) -> List[str]:
        errors: List[str] = []
        valid = {e.value for e in SearchStrategy}
        if self.strategy not in valid:
            errors.append(
                f"search.strategy must be one of {valid}, got {self.strategy!r}"
            )
        if self.n_iterations < 1:
            errors.append(
                f"search.n_iterations must be >= 1, got {self.n_iterations}"
            )
        if self.archive_size < 4:
            errors.append(
                f"search.archive_size must be >= 4, got {self.archive_size}"
            )
        if self.batch_size < 1:
            errors.append(f"search.batch_size must be >= 1, got {self.batch_size}")
        if not 0.0 <= self.mutation_rate <= 1.0:
            errors.append(
                f"search.mutation_rate must be in [0, 1], got {self.mutation_rate}"
            )
        if self.mutation_sigma <= 0:
            errors.append(
                f"search.mutation_sigma must be > 0, got {self.mutation_sigma}"
            )
        if not 0.0 <= self.crossover_rate <= 1.0:
            errors.append(
                f"search.crossover_rate must be in [0, 1], got {self.crossover_rate}"
            )
        if self.curiosity_weight < 0:
            errors.append(
                f"search.curiosity_weight must be >= 0, got {self.curiosity_weight}"
            )
        if self.novelty_k < 1:
            errors.append(f"search.novelty_k must be >= 1, got {self.novelty_k}")
        if not 0.0 < self.surprise_decay <= 1.0:
            errors.append(
                f"search.surprise_decay must be in (0, 1], got {self.surprise_decay}"
            )
        if not 0.0 <= self.elite_fraction <= 1.0:
            errors.append(
                f"search.elite_fraction must be in [0, 1], got {self.elite_fraction}"
            )
        if self.descriptor_bounds is not None:
            for i, (lo, hi) in enumerate(self.descriptor_bounds):
                if lo >= hi:
                    errors.append(
                        f"search.descriptor_bounds[{i}]: lo ({lo}) >= hi ({hi})"
                    )
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SearchConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class DetectionConfig:
    """Configuration for Phase 3: Tipping-point detection (ALG4).

    Parameters
    ----------
    method : str
        Changepoint detection algorithm ('pelt', 'binary_segmentation', 'cusum').
    penalty : str
        PELT penalty function ('bic', 'aic', 'hannan_quinn', 'manual').
    manual_penalty_value : float
        Value when penalty='manual'.
    min_segment_length : int
        Minimum segment length (contexts).
    max_changepoints : int
        Maximum number of changepoints (-1 for unlimited).
    n_permutations : int
        Number of permutations for significance testing.
    significance_level : float
        Significance threshold for validated changepoints.
    cost_function : str
        Cost function for PELT ('l2', 'rbf', 'rank').
    contexts_are_ordered : bool
        Whether contexts have a natural ordering.
    """

    method: str = "pelt"
    penalty: str = "bic"
    manual_penalty_value: float = 1.0
    min_segment_length: int = 2
    max_changepoints: int = -1
    n_permutations: int = 200
    significance_level: float = 0.05
    cost_function: str = "l2"
    contexts_are_ordered: bool = False

    def validate(self) -> List[str]:
        errors: List[str] = []
        valid_methods = {e.value for e in ChangePointMethod}
        if self.method not in valid_methods:
            errors.append(
                f"detection.method must be one of {valid_methods}, "
                f"got {self.method!r}"
            )
        valid_penalties = {"bic", "aic", "hannan_quinn", "manual"}
        if self.penalty not in valid_penalties:
            errors.append(
                f"detection.penalty must be one of {valid_penalties}, "
                f"got {self.penalty!r}"
            )
        if self.min_segment_length < 1:
            errors.append(
                f"detection.min_segment_length must be >= 1, "
                f"got {self.min_segment_length}"
            )
        if self.max_changepoints < -1:
            errors.append(
                f"detection.max_changepoints must be >= -1, "
                f"got {self.max_changepoints}"
            )
        if self.n_permutations < 10:
            errors.append(
                f"detection.n_permutations must be >= 10, "
                f"got {self.n_permutations}"
            )
        if not 0.0 < self.significance_level < 1.0:
            errors.append(
                f"detection.significance_level must be in (0, 1), "
                f"got {self.significance_level}"
            )
        valid_costs = {"l2", "rbf", "rank"}
        if self.cost_function not in valid_costs:
            errors.append(
                f"detection.cost_function must be one of {valid_costs}, "
                f"got {self.cost_function!r}"
            )
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DetectionConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class CertificateConfig:
    """Configuration for Phase 3: Robustness certificate generation (ALG5).

    Parameters
    ----------
    n_stability_rounds : int
        Number of stability selection rounds.
    stability_fraction : float
        Subsample fraction per round.
    stability_threshold : float
        Minimum selection probability for stable edges.
    n_bootstrap : int
        Number of bootstrap samples for parametric certificates.
    bootstrap_ci_level : float
        Bootstrap confidence level.
    tolerance : float
        Tolerance for certificate verification.
    ucb_alpha : float
        UCB confidence parameter for certificate bounds.
    max_certificate_order : int
        Maximum order (conditioning set size) for certificates.
    validate_assumptions : bool
        Whether to run assumption validation checks.
    """

    n_stability_rounds: int = 100
    stability_fraction: float = 0.7
    stability_threshold: float = 0.6
    n_bootstrap: int = 200
    bootstrap_ci_level: float = 0.95
    tolerance: float = 0.05
    ucb_alpha: float = 2.0
    max_certificate_order: int = 3
    validate_assumptions: bool = True

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.n_stability_rounds < 10:
            errors.append(
                f"certificate.n_stability_rounds must be >= 10, "
                f"got {self.n_stability_rounds}"
            )
        if not 0.1 <= self.stability_fraction <= 0.9:
            errors.append(
                f"certificate.stability_fraction must be in [0.1, 0.9], "
                f"got {self.stability_fraction}"
            )
        if not 0.0 < self.stability_threshold <= 1.0:
            errors.append(
                f"certificate.stability_threshold must be in (0, 1], "
                f"got {self.stability_threshold}"
            )
        if self.n_bootstrap < 10:
            errors.append(
                f"certificate.n_bootstrap must be >= 10, got {self.n_bootstrap}"
            )
        if not 0.5 <= self.bootstrap_ci_level < 1.0:
            errors.append(
                f"certificate.bootstrap_ci_level must be in [0.5, 1.0), "
                f"got {self.bootstrap_ci_level}"
            )
        if self.tolerance <= 0:
            errors.append(
                f"certificate.tolerance must be > 0, got {self.tolerance}"
            )
        if self.ucb_alpha <= 0:
            errors.append(
                f"certificate.ucb_alpha must be > 0, got {self.ucb_alpha}"
            )
        if self.max_certificate_order < 0:
            errors.append(
                f"certificate.max_certificate_order must be >= 0, "
                f"got {self.max_certificate_order}"
            )
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CertificateConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ComputationConfig:
    """Global computation settings.

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers (-1 = all CPUs, 1 = sequential).
    backend : str
        Parallelism backend ('thread' or 'process').
    memory_limit_mb : int
        Approximate memory budget in MB (0 = unlimited).
    seed : Optional[int]
        Global random seed for reproducibility.
    cache_dir : Optional[str]
        Directory for disk cache (None = no disk cache).
    cache_max_entries : int
        Maximum cache entries.
    log_level : str
        Logging verbosity ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    progress : bool
        Whether to display progress bars/reports.
    checkpoint_dir : Optional[str]
        Directory for pipeline checkpoints.
    checkpoint_interval : int
        Save checkpoint every N sub-steps within a phase.
    output_dir : Optional[str]
        Default output directory for results.
    """

    n_jobs: int = 1
    backend: str = "thread"
    memory_limit_mb: int = 0
    seed: Optional[int] = None
    cache_dir: Optional[str] = None
    cache_max_entries: int = 1024
    log_level: str = "WARNING"
    progress: bool = True
    checkpoint_dir: Optional[str] = None
    checkpoint_interval: int = 1
    output_dir: Optional[str] = None

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.n_jobs == 0 or self.n_jobs < -1:
            errors.append(
                f"computation.n_jobs must be >= 1 or -1, got {self.n_jobs}"
            )
        if self.backend not in ("thread", "process"):
            errors.append(
                f"computation.backend must be 'thread' or 'process', "
                f"got {self.backend!r}"
            )
        if self.memory_limit_mb < 0:
            errors.append(
                f"computation.memory_limit_mb must be >= 0, "
                f"got {self.memory_limit_mb}"
            )
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            errors.append(
                f"computation.log_level must be one of {valid_levels}, "
                f"got {self.log_level!r}"
            )
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ComputationConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# =====================================================================
# Master PipelineConfig
# =====================================================================


@dataclass
class PipelineConfig:
    """Master pipeline configuration aggregating all sub-configs.

    The pipeline has three phases:
    1. Foundation  — discovery, alignment, descriptors
    2. Exploration — QD-MAP-Elites search
    3. Validation  — tipping-points, certificates, diagnostics

    Parameters
    ----------
    discovery : DiscoveryConfig
        Causal discovery settings.
    alignment : AlignmentConfig
        CADA alignment settings.
    descriptor : DescriptorConfig
        Plasticity descriptor settings.
    search : SearchConfig
        QD search settings.
    detection : DetectionConfig
        Tipping-point detection settings.
    certificate : CertificateConfig
        Robustness certificate settings.
    computation : ComputationConfig
        Global computation settings.
    run_phase_1 : bool
        Whether to execute Phase 1 (Foundation).
    run_phase_2 : bool
        Whether to execute Phase 2 (Exploration).
    run_phase_3 : bool
        Whether to execute Phase 3 (Validation).
    profile : ConfigProfile
        Named profile this config was derived from.
    """

    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    descriptor: DescriptorConfig = field(default_factory=DescriptorConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    certificate: CertificateConfig = field(default_factory=CertificateConfig)
    computation: ComputationConfig = field(default_factory=ComputationConfig)
    run_phase_1: bool = True
    run_phase_2: bool = True
    run_phase_3: bool = True
    profile: ConfigProfile = ConfigProfile.STANDARD

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate the full configuration tree.

        Returns
        -------
        list of str
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []
        errors.extend(self.discovery.validate())
        errors.extend(self.alignment.validate())
        errors.extend(self.descriptor.validate())
        errors.extend(self.search.validate())
        errors.extend(self.detection.validate())
        errors.extend(self.certificate.validate())
        errors.extend(self.computation.validate())
        if not (self.run_phase_1 or self.run_phase_2 or self.run_phase_3):
            errors.append("At least one phase must be enabled")
        if self.run_phase_2 and not self.run_phase_1:
            errors.append("Phase 2 requires Phase 1 (or a checkpoint)")
        if self.run_phase_3 and not self.run_phase_1:
            errors.append("Phase 3 requires Phase 1 (or a checkpoint)")
        return errors

    def validate_or_raise(self) -> None:
        """Validate and raise :class:`ValueError` if invalid."""
        errors = self.validate()
        if errors:
            msg = "Configuration validation failed:\n  - " + "\n  - ".join(errors)
            raise ValueError(msg)

    # -----------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert the full config tree to a nested dictionary."""
        return {
            "discovery": self.discovery.to_dict(),
            "alignment": self.alignment.to_dict(),
            "descriptor": self.descriptor.to_dict(),
            "search": self.search.to_dict(),
            "detection": self.detection.to_dict(),
            "certificate": self.certificate.to_dict(),
            "computation": self.computation.to_dict(),
            "run_phase_1": self.run_phase_1,
            "run_phase_2": self.run_phase_2,
            "run_phase_3": self.run_phase_3,
            "profile": self.profile.value,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        """Construct a PipelineConfig from a nested dictionary.

        Unknown keys are silently ignored so that forward-compatible
        config files can be loaded by older code versions.

        Parameters
        ----------
        d : dict
            Configuration dictionary.

        Returns
        -------
        PipelineConfig
        """
        cfg = cls()
        if "discovery" in d:
            cfg.discovery = DiscoveryConfig.from_dict(d["discovery"])
        if "alignment" in d:
            cfg.alignment = AlignmentConfig.from_dict(d["alignment"])
        if "descriptor" in d:
            cfg.descriptor = DescriptorConfig.from_dict(d["descriptor"])
        if "search" in d:
            cfg.search = SearchConfig.from_dict(d["search"])
        if "detection" in d:
            cfg.detection = DetectionConfig.from_dict(d["detection"])
        if "certificate" in d:
            cfg.certificate = CertificateConfig.from_dict(d["certificate"])
        if "computation" in d:
            cfg.computation = ComputationConfig.from_dict(d["computation"])
        cfg.run_phase_1 = d.get("run_phase_1", True)
        cfg.run_phase_2 = d.get("run_phase_2", True)
        cfg.run_phase_3 = d.get("run_phase_3", True)
        if "profile" in d:
            try:
                cfg.profile = ConfigProfile(d["profile"])
            except ValueError:
                cfg.profile = ConfigProfile.CUSTOM
        return cfg

    def to_json(self, path: Optional[Union[str, Path]] = None) -> str:
        """Serialize to JSON string.  Optionally write to *path*.

        Parameters
        ----------
        path : str or Path, optional
            If given, write JSON to this file path.

        Returns
        -------
        str
            JSON representation.
        """
        text = json.dumps(self.to_dict(), indent=2, default=str)
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_json(cls, source: Union[str, Path]) -> "PipelineConfig":
        """Load from a JSON file path or raw JSON string.

        Parameters
        ----------
        source : str or Path
            File path or JSON string.

        Returns
        -------
        PipelineConfig
        """
        p = Path(source)
        if p.exists():
            text = p.read_text()
        else:
            text = str(source)
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PipelineConfig":
        """Load from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to YAML file.

        Returns
        -------
        PipelineConfig

        Raises
        ------
        ImportError
            If PyYAML is not installed.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML config loading: "
                "pip install pyyaml"
            ) from exc
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_yaml(self, path: Union[str, Path]) -> str:
        """Serialize to YAML and write to *path*.

        Parameters
        ----------
        path : str or Path
            Output file path.

        Returns
        -------
        str
            YAML representation.

        Raises
        ------
        ImportError
            If PyYAML is not installed.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML config writing: "
                "pip install pyyaml"
            ) from exc
        text = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        Path(path).write_text(text)
        return text

    # -----------------------------------------------------------------
    # Named profiles
    # -----------------------------------------------------------------

    @classmethod
    def fast(cls) -> "PipelineConfig":
        """Minimal-computation profile for quick testing.

        Uses small bootstrap/permutation counts, reduced search
        iterations, and sequential computation.

        Returns
        -------
        PipelineConfig
        """
        cfg = cls(
            discovery=DiscoveryConfig(
                method="pc",
                alpha=0.05,
                max_cond_set_size=3,
                stable=False,
            ),
            alignment=AlignmentConfig(
                strategy="greedy",
                max_alignment_iterations=50,
            ),
            descriptor=DescriptorConfig(
                n_bootstrap=50,
                n_permutations=100,
                auto_threshold=False,
                use_stability_selection=False,
            ),
            search=SearchConfig(
                strategy="map_elites",
                n_iterations=100,
                archive_size=64,
                n_cvt_samples=5000,
                batch_size=16,
            ),
            detection=DetectionConfig(
                method="pelt",
                n_permutations=50,
            ),
            certificate=CertificateConfig(
                n_stability_rounds=30,
                n_bootstrap=50,
            ),
            computation=ComputationConfig(
                n_jobs=1,
                backend="thread",
                progress=True,
            ),
            profile=ConfigProfile.FAST,
        )
        return cfg

    @classmethod
    def standard(cls) -> "PipelineConfig":
        """Balanced profile suitable for typical analyses.

        Returns
        -------
        PipelineConfig
        """
        return cls(profile=ConfigProfile.STANDARD)

    @classmethod
    def thorough(cls) -> "PipelineConfig":
        """High-fidelity profile for publication-quality results.

        Uses large bootstrap/permutation counts, extended search,
        automatic threshold calibration, and stability selection.

        Returns
        -------
        PipelineConfig
        """
        cfg = cls(
            discovery=DiscoveryConfig(
                method="pc",
                alpha=0.01,
                max_cond_set_size=-1,
                stable=True,
            ),
            alignment=AlignmentConfig(
                strategy="greedy",
                use_warm_start=True,
                max_alignment_iterations=200,
            ),
            descriptor=DescriptorConfig(
                n_bootstrap=1000,
                n_permutations=2000,
                auto_threshold=True,
                use_stability_selection=True,
                stability_n_subsamples=200,
            ),
            search=SearchConfig(
                strategy="curiosity_driven",
                n_iterations=2000,
                archive_size=1024,
                n_cvt_samples=100000,
                batch_size=64,
            ),
            detection=DetectionConfig(
                method="pelt",
                n_permutations=1000,
            ),
            certificate=CertificateConfig(
                n_stability_rounds=500,
                n_bootstrap=1000,
                validate_assumptions=True,
            ),
            computation=ComputationConfig(
                n_jobs=-1,
                backend="process",
                progress=True,
            ),
            profile=ConfigProfile.THOROUGH,
        )
        return cfg

    # -----------------------------------------------------------------
    # Merging and copying
    # -----------------------------------------------------------------

    def copy(self) -> "PipelineConfig":
        """Return a deep copy of this configuration."""
        return copy.deepcopy(self)

    def merge(self, overrides: Dict[str, Any]) -> "PipelineConfig":
        """Create a copy with *overrides* applied on top.

        Supports nested keys like ``{"discovery": {"alpha": 0.01}}``.

        Parameters
        ----------
        overrides : dict
            Nested dictionary of values to override.

        Returns
        -------
        PipelineConfig
            New config with overrides applied.
        """
        base = self.to_dict()
        _deep_update(base, overrides)
        return PipelineConfig.from_dict(base)

    def summary(self) -> str:
        """Return a human-readable one-line summary.

        Returns
        -------
        str
        """
        phases = []
        if self.run_phase_1:
            phases.append("Foundation")
        if self.run_phase_2:
            phases.append("Exploration")
        if self.run_phase_3:
            phases.append("Validation")
        return (
            f"PipelineConfig(profile={self.profile.value}, "
            f"phases=[{', '.join(phases)}], "
            f"method={self.discovery.method}, "
            f"n_jobs={self.computation.n_jobs})"
        )

    def __repr__(self) -> str:
        return self.summary()


# =====================================================================
# Helpers
# =====================================================================


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Recursively merge *override* into *base* in place."""
    for key, val in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(val, dict)
        ):
            _deep_update(base[key], val)
        else:
            base[key] = val


def load_config(source: Union[str, Path]) -> PipelineConfig:
    """Auto-detect format and load a PipelineConfig.

    Parameters
    ----------
    source : str or Path
        Path to a JSON or YAML configuration file.

    Returns
    -------
    PipelineConfig

    Raises
    ------
    ValueError
        If format cannot be determined.
    """
    p = Path(source)
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return PipelineConfig.from_yaml(p)
    if suffix == ".json":
        return PipelineConfig.from_json(p)
    raise ValueError(
        f"Cannot determine config format from extension {suffix!r}. "
        "Supported: .json, .yaml, .yml"
    )


def config_from_env() -> PipelineConfig:
    """Build a PipelineConfig from environment variables.

    Reads ``CPA_PROFILE``, ``CPA_N_JOBS``, ``CPA_SEED``, ``CPA_LOG_LEVEL``,
    ``CPA_CACHE_DIR``, and ``CPA_OUTPUT_DIR``.

    Returns
    -------
    PipelineConfig
    """
    profile_name = os.environ.get("CPA_PROFILE", "standard").lower()
    profiles = {
        "fast": PipelineConfig.fast,
        "standard": PipelineConfig.standard,
        "thorough": PipelineConfig.thorough,
    }
    factory = profiles.get(profile_name, PipelineConfig.standard)
    cfg = factory()

    n_jobs = os.environ.get("CPA_N_JOBS")
    if n_jobs is not None:
        cfg.computation.n_jobs = int(n_jobs)

    seed = os.environ.get("CPA_SEED")
    if seed is not None:
        cfg.computation.seed = int(seed)

    log_level = os.environ.get("CPA_LOG_LEVEL")
    if log_level is not None:
        cfg.computation.log_level = log_level.upper()

    cache_dir = os.environ.get("CPA_CACHE_DIR")
    if cache_dir is not None:
        cfg.computation.cache_dir = cache_dir

    output_dir = os.environ.get("CPA_OUTPUT_DIR")
    if output_dir is not None:
        cfg.computation.output_dir = output_dir

    return cfg
