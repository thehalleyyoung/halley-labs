"""
Global configuration for TN-Check.

Manages default parameters for TT arithmetic, CME compilation,
time integration, CSL model checking, and error certification.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import os
from pathlib import Path
from typing import Any, Optional


class IntegratorType(enum.Enum):
    """Time integration method."""
    TDVP_ONE_SITE = "tdvp_1site"
    TDVP_TWO_SITE = "tdvp_2site"
    KRYLOV = "krylov"
    UNIFORMIZATION = "uniformization"
    EULER = "euler"


class CanonicalForm(enum.Enum):
    """MPS canonical form."""
    LEFT = "left"
    RIGHT = "right"
    MIXED = "mixed"
    NONE = "none"


class OrderingMethod(enum.Enum):
    """Species ordering strategy."""
    IDENTITY = "identity"
    REVERSE_CUTHILL_MCKEE = "rcm"
    SPECTRAL = "spectral"
    GRAPH_PARTITION = "graph_partition"
    GREEDY_ENTANGLEMENT = "greedy_entanglement"


class CSLSemantics(enum.Enum):
    """CSL evaluation semantics."""
    TWO_VALUED = "two_valued"
    THREE_VALUED = "three_valued"


@dataclasses.dataclass
class TTConfig:
    """Tensor-train arithmetic configuration."""
    max_bond_dim: int = 200
    truncation_tolerance: float = 1e-10
    canonical_form: CanonicalForm = CanonicalForm.LEFT
    svd_driver: str = "gesdd"
    use_randomized_svd: bool = False
    randomized_svd_oversampling: int = 10
    qr_stabilization: bool = True
    relative_truncation: bool = True
    min_bond_dim: int = 1
    enable_caching: bool = True
    dtype: str = "float64"
    complex_dtype: str = "complex128"


@dataclasses.dataclass
class CMEConfig:
    """CME compilation configuration."""
    max_copy_number: int = 50
    fsp_tolerance: float = 1e-6
    fsp_expansion_factor: float = 1.5
    use_conservation_laws: bool = True
    propensity_interpolation_order: int = 4
    mpo_compression_tolerance: float = 1e-12
    kronecker_factorization: bool = True
    stochastic_interpretation: str = "kurtz"


@dataclasses.dataclass
class IntegratorConfig:
    """Time integration configuration."""
    method: IntegratorType = IntegratorType.TDVP_TWO_SITE
    dt: float = 0.01
    adaptive_dt: bool = True
    dt_min: float = 1e-8
    dt_max: float = 1.0
    dt_safety_factor: float = 0.9
    krylov_dim: int = 30
    krylov_tol: float = 1e-10
    max_krylov_restarts: int = 5
    uniformization_fox_glynn_epsilon: float = 1e-10
    uniformization_max_terms: int = 1000000
    tdvp_num_sweeps: int = 1
    tdvp_lanczos_dim: int = 20
    conservation_enforcement: bool = True
    max_steps: int = 100000
    steady_state_tol: float = 1e-8
    steady_state_max_sweeps: int = 200
    dmrg_energy_tol: float = 1e-10
    dmrg_max_sweeps: int = 100


@dataclasses.dataclass
class CheckerConfig:
    """CSL model checker configuration."""
    semantics: CSLSemantics = CSLSemantics.THREE_VALUED
    fixpoint_tol: float = 1e-8
    fixpoint_max_iter: int = 500
    bounded_until_substeps: int = 100
    threshold_epsilon: float = 1e-6
    indeterminate_tracking: bool = True
    satisfaction_set_compression: bool = True
    nested_error_propagation: bool = True


@dataclasses.dataclass
class ErrorConfig:
    """Error certification configuration."""
    track_truncation_error: bool = True
    track_fsp_error: bool = True
    track_integration_error: bool = True
    track_clamping_error: bool = True
    track_negativity: bool = True
    richardson_extrapolation: bool = True
    convergence_check_factor: int = 2
    error_budget_fraction: float = 0.01
    report_interval: int = 10
    max_total_error: float = 0.1


@dataclasses.dataclass
class OrderingConfig:
    """Species ordering configuration."""
    method: OrderingMethod = OrderingMethod.REVERSE_CUTHILL_MCKEE
    multi_ordering_trials: int = 5
    bond_dim_sensitivity_samples: int = 3
    metis_nparts: int = 4
    spectral_eigenvector_index: int = 1


@dataclasses.dataclass
class AdaptiveConfig:
    """Adaptive rank controller configuration."""
    initial_bond_dim: int = 10
    max_bond_dim: int = 200
    growth_factor: float = 2.0
    shrink_factor: float = 0.8
    convergence_threshold: float = 1e-6
    min_singular_value_ratio: float = 1e-10
    check_interval: int = 5
    bond_budget: Optional[int] = None
    enable_per_bond_monitoring: bool = True


@dataclasses.dataclass
class EvaluationConfig:
    """Benchmark and evaluation configuration."""
    ground_truth_max_species: int = 12
    timing_warmup_runs: int = 2
    timing_measurement_runs: int = 5
    memory_tracking: bool = True
    scaling_species_range: tuple = (4, 50)
    convergence_bond_dims: tuple = (10, 20, 50, 100, 200)
    output_dir: str = "benchmark_output"
    plot_format: str = "pdf"
    save_distributions: bool = False


@dataclasses.dataclass
class TNCheckConfig:
    """Master configuration for TN-Check."""
    tt: TTConfig = dataclasses.field(default_factory=TTConfig)
    cme: CMEConfig = dataclasses.field(default_factory=CMEConfig)
    integrator: IntegratorConfig = dataclasses.field(default_factory=IntegratorConfig)
    checker: CheckerConfig = dataclasses.field(default_factory=CheckerConfig)
    error: ErrorConfig = dataclasses.field(default_factory=ErrorConfig)
    ordering: OrderingConfig = dataclasses.field(default_factory=OrderingConfig)
    adaptive: AdaptiveConfig = dataclasses.field(default_factory=AdaptiveConfig)
    evaluation: EvaluationConfig = dataclasses.field(default_factory=EvaluationConfig)
    log_level: str = "INFO"
    seed: Optional[int] = None
    num_threads: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TNCheckConfig:
        """Deserialize configuration from dictionary."""
        cfg = cls()
        if "tt" in d:
            for k, v in d["tt"].items():
                if k == "canonical_form":
                    v = CanonicalForm(v)
                setattr(cfg.tt, k, v)
        if "cme" in d:
            for k, v in d["cme"].items():
                setattr(cfg.cme, k, v)
        if "integrator" in d:
            for k, v in d["integrator"].items():
                if k == "method":
                    v = IntegratorType(v)
                setattr(cfg.integrator, k, v)
        if "checker" in d:
            for k, v in d["checker"].items():
                if k == "semantics":
                    v = CSLSemantics(v)
                setattr(cfg.checker, k, v)
        if "error" in d:
            for k, v in d["error"].items():
                setattr(cfg.error, k, v)
        if "ordering" in d:
            for k, v in d["ordering"].items():
                if k == "method":
                    v = OrderingMethod(v)
                setattr(cfg.ordering, k, v)
        if "adaptive" in d:
            for k, v in d["adaptive"].items():
                setattr(cfg.adaptive, k, v)
        if "evaluation" in d:
            for k, v in d["evaluation"].items():
                setattr(cfg.evaluation, k, v)
        for k in ("log_level", "seed", "num_threads"):
            if k in d:
                setattr(cfg, k, d[k])
        return cfg

    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _convert(obj: Any) -> Any:
            if isinstance(obj, enum.Enum):
                return obj.value
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        d = self.to_dict()
        with open(path, "w") as f:
            json.dump(d, f, indent=2, default=_convert)

    @classmethod
    def load(cls, path: str | Path) -> TNCheckConfig:
        """Load configuration from JSON file."""
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_env(cls) -> TNCheckConfig:
        """Create configuration from environment variables."""
        cfg = cls()
        env_prefix = "TNCHECK_"
        if val := os.environ.get(f"{env_prefix}MAX_BOND_DIM"):
            cfg.tt.max_bond_dim = int(val)
        if val := os.environ.get(f"{env_prefix}TRUNCATION_TOL"):
            cfg.tt.truncation_tolerance = float(val)
        if val := os.environ.get(f"{env_prefix}MAX_COPY_NUMBER"):
            cfg.cme.max_copy_number = int(val)
        if val := os.environ.get(f"{env_prefix}INTEGRATOR"):
            cfg.integrator.method = IntegratorType(val)
        if val := os.environ.get(f"{env_prefix}LOG_LEVEL"):
            cfg.log_level = val
        if val := os.environ.get(f"{env_prefix}SEED"):
            cfg.seed = int(val)
        if val := os.environ.get(f"{env_prefix}NUM_THREADS"):
            cfg.num_threads = int(val)
        return cfg


def default_config() -> TNCheckConfig:
    """Return default configuration."""
    return TNCheckConfig()


def quick_config(
    max_bond_dim: int = 50,
    max_copy_number: int = 30,
    integrator: str = "tdvp_2site",
) -> TNCheckConfig:
    """Create a quick configuration for testing."""
    cfg = TNCheckConfig()
    cfg.tt.max_bond_dim = max_bond_dim
    cfg.cme.max_copy_number = max_copy_number
    cfg.integrator.method = IntegratorType(integrator)
    return cfg
