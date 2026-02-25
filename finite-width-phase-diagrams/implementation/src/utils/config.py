"""Configuration management for the finite-width phase diagram system.

Provides ``PhaseDiagramConfig`` with all tuneable parameters, YAML
loading/saving, validation, profile presets (quick, standard, thorough,
research), config merging and command-line override support.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ConfigProfile(Enum):
    """Pre-defined configuration profiles."""

    QUICK = "quick"
    STANDARD = "standard"
    THOROUGH = "thorough"
    RESEARCH = "research"


class OutputFormat(Enum):
    """Supported output formats."""

    HDF5 = "hdf5"
    NPZ = "npz"
    JSON = "json"


# ---------------------------------------------------------------------------
# Sub-configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ArchitectureSpec:
    """Specification of the neural network architecture to analyse."""

    arch_type: str = "mlp"
    depth: int = 2
    width: int = 256
    activation: str = "relu"
    init_scale: float = 1.0
    bias: bool = True
    normalization: Optional[str] = None
    input_dim: int = 10
    output_dim: int = 1
    # Conv-specific
    kernel_size: int = 3
    stride: int = 1
    padding: int = 0
    channels: int = 32
    # Residual
    skip_type: str = "additive"
    num_blocks: int = 1
    # DSL string for advanced architectures
    dsl: Optional[str] = None

    def validate(self) -> List[str]:
        """Return list of validation errors (empty means valid)."""
        errors: List[str] = []
        if self.depth < 1:
            errors.append("depth must be >= 1")
        if self.width < 1:
            errors.append("width must be >= 1")
        if self.activation not in ("relu", "gelu", "tanh", "sigmoid", "linear"):
            errors.append(f"unsupported activation: {self.activation}")
        if self.arch_type not in ("mlp", "conv1d", "conv2d", "resnet"):
            errors.append(f"unsupported arch_type: {self.arch_type}")
        if self.init_scale <= 0:
            errors.append("init_scale must be positive")
        if self.input_dim < 1:
            errors.append("input_dim must be >= 1")
        if self.output_dim < 1:
            errors.append("output_dim must be >= 1")
        return errors


@dataclass
class CalibrationSpec:
    """Settings for multi-width calibration."""

    widths: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    num_seeds: int = 5
    max_correction_order: int = 2
    bootstrap_samples: int = 1000
    bootstrap_ci_level: float = 0.95
    regression_type: str = "ols"
    constrained: bool = True
    min_width: int = 32
    max_width: int = 4096

    def validate(self) -> List[str]:
        errors: List[str] = []
        if len(self.widths) < 2:
            errors.append("need at least 2 calibration widths")
        if any(w < 1 for w in self.widths):
            errors.append("all widths must be positive")
        if self.num_seeds < 1:
            errors.append("num_seeds must be >= 1")
        if self.max_correction_order < 1:
            errors.append("max_correction_order must be >= 1")
        if not (0 < self.bootstrap_ci_level < 1):
            errors.append("bootstrap_ci_level must be in (0, 1)")
        if self.bootstrap_samples < 10:
            errors.append("bootstrap_samples should be >= 10")
        return errors


@dataclass
class GridSpec:
    """Settings for the hyperparameter grid sweep."""

    lr_range: Tuple[float, float] = (1e-4, 1.0)
    width_range: Tuple[int, int] = (32, 2048)
    depth_range: Tuple[int, int] = (1, 10)
    lr_points: int = 20
    width_points: int = 15
    depth_points: int = 5
    log_scale_lr: bool = True
    log_scale_width: bool = True
    sweep_dims: List[str] = field(default_factory=lambda: ["lr", "width"])
    adaptive_refine: bool = True
    refine_threshold: float = 0.3

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.lr_range[0] >= self.lr_range[1]:
            errors.append("lr_range must satisfy min < max")
        if self.lr_range[0] <= 0:
            errors.append("lr_range min must be positive")
        if self.width_range[0] >= self.width_range[1]:
            errors.append("width_range must satisfy min < max")
        if self.lr_points < 2:
            errors.append("lr_points must be >= 2")
        if self.width_points < 2:
            errors.append("width_points must be >= 2")
        for dim in self.sweep_dims:
            if dim not in ("lr", "width", "depth", "init_scale"):
                errors.append(f"unknown sweep dimension: {dim}")
        return errors


@dataclass
class NystromSpec:
    """Settings for Nyström low-rank approximation."""

    rank: int = 50
    landmark_strategy: str = "kmeans"
    adaptive_rank: bool = True
    max_rank: int = 200
    tolerance: float = 1e-6
    oversampling: int = 10

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.rank < 1:
            errors.append("rank must be >= 1")
        if self.landmark_strategy not in ("random", "kmeans", "leverage"):
            errors.append(f"unknown landmark strategy: {self.landmark_strategy}")
        if self.max_rank < self.rank:
            errors.append("max_rank must be >= rank")
        if self.tolerance <= 0:
            errors.append("tolerance must be positive")
        return errors


@dataclass
class ODESpec:
    """Settings for kernel ODE integration."""

    scheme: str = "rk45"
    atol: float = 1e-8
    rtol: float = 1e-5
    max_step: float = 0.1
    max_steps: int = 10000
    t_span: Tuple[float, float] = (0.0, 100.0)
    dense_output: bool = True
    eigenvalue_tracking: bool = True
    bifurcation_detection: bool = True
    bifurcation_tol: float = 1e-6

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.scheme not in ("euler", "rk4", "rk45", "implicit_midpoint"):
            errors.append(f"unknown ODE scheme: {self.scheme}")
        if self.atol <= 0:
            errors.append("atol must be positive")
        if self.rtol <= 0:
            errors.append("rtol must be positive")
        if self.max_step <= 0:
            errors.append("max_step must be positive")
        if self.t_span[0] >= self.t_span[1]:
            errors.append("t_span must satisfy start < end")
        return errors


@dataclass
class OutputSpec:
    """Settings for output paths and formats."""

    output_dir: str = "./output"
    save_kernels: bool = True
    save_trajectories: bool = True
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    format: str = "npz"
    plot_format: str = "png"
    plot_dpi: int = 150
    verbose: bool = True

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.format not in ("npz", "hdf5", "json"):
            errors.append(f"unknown output format: {self.format}")
        if self.plot_format not in ("png", "pdf", "svg"):
            errors.append(f"unknown plot format: {self.plot_format}")
        if self.plot_dpi < 50:
            errors.append("plot_dpi should be >= 50")
        return errors


@dataclass
class TrainingSpec:
    """Settings for ground-truth training evaluation."""

    n_train: int = 100
    n_test: int = 50
    num_seeds: int = 5
    max_epochs: int = 1000
    loss: str = "mse"
    optimizer: str = "sgd"
    measure_interval: int = 10
    kernel_alignment_tracking: bool = True

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.n_train < 2:
            errors.append("n_train must be >= 2")
        if self.num_seeds < 1:
            errors.append("num_seeds must be >= 1")
        if self.max_epochs < 1:
            errors.append("max_epochs must be >= 1")
        if self.loss not in ("mse", "cross_entropy"):
            errors.append(f"unknown loss: {self.loss}")
        return errors


@dataclass
class ParallelSpec:
    """Settings for parallel computation."""

    n_workers: int = 1
    backend: str = "multiprocessing"
    chunk_size: int = 10
    timeout: float = 3600.0
    progress: bool = True

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.n_workers < 1:
            errors.append("n_workers must be >= 1")
        if self.backend not in ("multiprocessing", "threading", "sequential"):
            errors.append(f"unknown backend: {self.backend}")
        if self.chunk_size < 1:
            errors.append("chunk_size must be >= 1")
        if self.timeout <= 0:
            errors.append("timeout must be positive")
        return errors


# ---------------------------------------------------------------------------
# Main configuration
# ---------------------------------------------------------------------------

@dataclass
class PhaseDiagramConfig:
    """Master configuration for a full phase-diagram computation.

    Contains all sub-configurations and provides validation, serialisation,
    and profile-based factory methods.
    """

    architecture: ArchitectureSpec = field(default_factory=ArchitectureSpec)
    calibration: CalibrationSpec = field(default_factory=CalibrationSpec)
    grid: GridSpec = field(default_factory=GridSpec)
    nystrom: NystromSpec = field(default_factory=NystromSpec)
    ode: ODESpec = field(default_factory=ODESpec)
    output: OutputSpec = field(default_factory=OutputSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)
    parallel: ParallelSpec = field(default_factory=ParallelSpec)

    # ---- validation -------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate all sub-configurations, returning list of errors."""
        errors: List[str] = []
        for name in (
            "architecture",
            "calibration",
            "grid",
            "nystrom",
            "ode",
            "output",
            "training",
            "parallel",
        ):
            sub = getattr(self, name)
            for err in sub.validate():
                errors.append(f"{name}: {err}")
        return errors

    def validate_or_raise(self) -> None:
        """Raise ``ValueError`` if configuration is invalid."""
        errors = self.validate()
        if errors:
            raise ValueError(
                "Invalid configuration:\n  " + "\n  ".join(errors)
            )

    # ---- serialisation ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict (JSON/YAML-safe)."""
        return _config_to_dict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PhaseDiagramConfig":
        """Construct from a plain dict."""
        return _config_from_dict(d)

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, s: str) -> "PhaseDiagramConfig":
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(s))

    # ---- YAML support (optional dependency) --------------------------------

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML support: pip install pyyaml")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: Union[str, Path]) -> "PhaseDiagramConfig":
        """Load configuration from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML support: pip install pyyaml")
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    # ---- merging / override ------------------------------------------------

    def merge(self, overrides: Dict[str, Any]) -> "PhaseDiagramConfig":
        """Return a new config with *overrides* merged on top of this one.

        Supports dotted keys like ``"grid.lr_points": 50``.
        """
        base = self.to_dict()
        _deep_merge(base, _expand_dotted(overrides))
        return PhaseDiagramConfig.from_dict(base)

    def with_overrides(self, **kwargs: Any) -> "PhaseDiagramConfig":
        """Convenience: ``cfg.with_overrides(grid__lr_points=50)``

        Double underscores are converted to dict nesting.
        """
        d: Dict[str, Any] = {}
        for key, val in kwargs.items():
            parts = key.split("__")
            cur = d
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = val
        return self.merge(d)

    # ---- profile presets ---------------------------------------------------

    @classmethod
    def from_profile(cls, profile: Union[str, ConfigProfile]) -> "PhaseDiagramConfig":
        """Create configuration from a named profile."""
        if isinstance(profile, str):
            profile = ConfigProfile(profile)
        presets = _profile_presets()
        if profile not in presets:
            raise ValueError(f"Unknown profile: {profile}")
        return cls.from_dict(presets[profile])

    @classmethod
    def quick(cls) -> "PhaseDiagramConfig":
        """Quick profile: small grids, few seeds, fast for smoke testing."""
        return cls.from_profile(ConfigProfile.QUICK)

    @classmethod
    def standard(cls) -> "PhaseDiagramConfig":
        """Standard profile: reasonable defaults for routine experiments."""
        return cls.from_profile(ConfigProfile.STANDARD)

    @classmethod
    def thorough(cls) -> "PhaseDiagramConfig":
        """Thorough profile: finer grids, more seeds, better statistics."""
        return cls.from_profile(ConfigProfile.THOROUGH)

    @classmethod
    def research(cls) -> "PhaseDiagramConfig":
        """Research profile: publication-quality settings."""
        return cls.from_profile(ConfigProfile.RESEARCH)

    # ---- repr & summary ----------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of key settings."""
        lines = [
            "PhaseDiagramConfig Summary",
            "=" * 40,
            f"  Architecture : {self.architecture.arch_type} "
            f"d={self.architecture.depth} w={self.architecture.width} "
            f"act={self.architecture.activation}",
            f"  Calibration  : widths={self.calibration.widths} "
            f"seeds={self.calibration.num_seeds}",
            f"  Grid         : {self.grid.sweep_dims} "
            f"lr={self.grid.lr_points} w={self.grid.width_points}",
            f"  Nyström      : rank={self.nystrom.rank} "
            f"strategy={self.nystrom.landmark_strategy}",
            f"  ODE          : scheme={self.ode.scheme} "
            f"atol={self.ode.atol} rtol={self.ode.rtol}",
            f"  Output       : {self.output.output_dir} "
            f"format={self.output.format}",
            f"  Training     : n={self.training.n_train} "
            f"seeds={self.training.num_seeds} "
            f"epochs={self.training.max_epochs}",
            f"  Parallel     : workers={self.parallel.n_workers} "
            f"backend={self.parallel.backend}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Profile presets
# ---------------------------------------------------------------------------

def _profile_presets() -> Dict[ConfigProfile, Dict[str, Any]]:
    """Return preset dictionaries for each profile."""
    return {
        ConfigProfile.QUICK: {
            "architecture": {"depth": 2, "width": 64, "input_dim": 5, "output_dim": 1},
            "calibration": {
                "widths": [32, 64, 128],
                "num_seeds": 2,
                "bootstrap_samples": 100,
            },
            "grid": {
                "lr_points": 8,
                "width_points": 6,
                "depth_points": 3,
                "adaptive_refine": False,
            },
            "nystrom": {"rank": 20, "max_rank": 50},
            "ode": {"max_steps": 2000, "t_span": [0.0, 50.0]},
            "output": {"save_trajectories": False, "verbose": False},
            "training": {
                "n_train": 30,
                "n_test": 20,
                "num_seeds": 2,
                "max_epochs": 200,
            },
            "parallel": {"n_workers": 1},
        },
        ConfigProfile.STANDARD: {
            "architecture": {"depth": 2, "width": 256, "input_dim": 10, "output_dim": 1},
            "calibration": {
                "widths": [64, 128, 256, 512, 1024],
                "num_seeds": 5,
                "bootstrap_samples": 1000,
            },
            "grid": {"lr_points": 20, "width_points": 15},
            "nystrom": {"rank": 50},
            "ode": {"max_steps": 10000},
            "output": {"verbose": True},
            "training": {"n_train": 100, "num_seeds": 5, "max_epochs": 1000},
            "parallel": {"n_workers": 4},
        },
        ConfigProfile.THOROUGH: {
            "architecture": {"depth": 3, "width": 512, "input_dim": 20, "output_dim": 1},
            "calibration": {
                "widths": [64, 128, 256, 512, 1024, 2048],
                "num_seeds": 10,
                "bootstrap_samples": 5000,
                "max_correction_order": 3,
            },
            "grid": {
                "lr_points": 40,
                "width_points": 30,
                "depth_points": 8,
                "adaptive_refine": True,
                "refine_threshold": 0.2,
            },
            "nystrom": {"rank": 100, "max_rank": 300},
            "ode": {"atol": 1e-10, "rtol": 1e-7, "max_steps": 50000},
            "output": {"format": "hdf5", "plot_dpi": 300},
            "training": {
                "n_train": 200,
                "n_test": 100,
                "num_seeds": 10,
                "max_epochs": 5000,
            },
            "parallel": {"n_workers": 8},
        },
        ConfigProfile.RESEARCH: {
            "architecture": {"depth": 4, "width": 1024, "input_dim": 50, "output_dim": 1},
            "calibration": {
                "widths": [32, 64, 128, 256, 512, 1024, 2048, 4096],
                "num_seeds": 20,
                "bootstrap_samples": 10000,
                "max_correction_order": 3,
            },
            "grid": {
                "lr_points": 60,
                "width_points": 50,
                "depth_points": 10,
                "adaptive_refine": True,
                "refine_threshold": 0.15,
            },
            "nystrom": {"rank": 200, "max_rank": 500, "adaptive_rank": True},
            "ode": {
                "atol": 1e-12,
                "rtol": 1e-9,
                "max_steps": 100000,
                "scheme": "rk45",
            },
            "output": {
                "format": "hdf5",
                "save_kernels": True,
                "save_trajectories": True,
                "plot_format": "pdf",
                "plot_dpi": 600,
            },
            "training": {
                "n_train": 500,
                "n_test": 200,
                "num_seeds": 30,
                "max_epochs": 10000,
            },
            "parallel": {"n_workers": 16, "backend": "multiprocessing"},
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_to_dict(cfg: PhaseDiagramConfig) -> Dict[str, Any]:
    """Recursively convert dataclass to dict, handling tuples → lists."""
    d = asdict(cfg)
    return _tuples_to_lists(d)


def _tuples_to_lists(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _tuples_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_tuples_to_lists(v) for v in obj]
    return obj


def _config_from_dict(d: Dict[str, Any]) -> PhaseDiagramConfig:
    """Reconstruct PhaseDiagramConfig from a dict, applying defaults."""
    cfg = PhaseDiagramConfig()
    mapping = {
        "architecture": (ArchitectureSpec, cfg.architecture),
        "calibration": (CalibrationSpec, cfg.calibration),
        "grid": (GridSpec, cfg.grid),
        "nystrom": (NystromSpec, cfg.nystrom),
        "ode": (ODESpec, cfg.ode),
        "output": (OutputSpec, cfg.output),
        "training": (TrainingSpec, cfg.training),
        "parallel": (ParallelSpec, cfg.parallel),
    }
    for section_name, (cls, default_obj) in mapping.items():
        section_data = d.get(section_name, {})
        if not isinstance(section_data, dict):
            continue
        merged = asdict(default_obj)
        merged.update(section_data)
        # Convert lists back to tuples where needed
        for fname in ("lr_range", "width_range", "depth_range", "t_span"):
            if fname in merged and isinstance(merged[fname], list):
                merged[fname] = tuple(merged[fname])
        try:
            setattr(cfg, section_name, cls(**{
                k: v for k, v in merged.items()
                if k in cls.__dataclass_fields__
            }))
        except TypeError:
            pass  # skip unknown fields gracefully
    return cfg


def _deep_merge(base: Dict, override: Dict) -> None:
    """In-place deep merge of *override* into *base*."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _expand_dotted(d: Dict[str, Any]) -> Dict[str, Any]:
    """Expand dotted keys: ``{"grid.lr_points": 50}`` → ``{"grid": {"lr_points": 50}}``."""
    result: Dict[str, Any] = {}
    for key, val in d.items():
        parts = key.split(".")
        cur = result
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    return result


# ---------------------------------------------------------------------------
# Top-level convenience functions
# ---------------------------------------------------------------------------

def load_config(path: Union[str, Path]) -> PhaseDiagramConfig:
    """Load configuration from YAML or JSON file.

    Determines format from extension.
    """
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        return PhaseDiagramConfig.load_yaml(path)
    elif path.suffix == ".json":
        with open(path) as f:
            return PhaseDiagramConfig.from_json(f.read())
    else:
        raise ValueError(f"Unsupported config file extension: {path.suffix}")


def save_config(cfg: PhaseDiagramConfig, path: Union[str, Path]) -> None:
    """Save configuration to YAML or JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in (".yaml", ".yml"):
        cfg.save_yaml(path)
    elif path.suffix == ".json":
        with open(path, "w") as f:
            f.write(cfg.to_json())
    else:
        raise ValueError(f"Unsupported config file extension: {path.suffix}")


def apply_cli_overrides(
    cfg: PhaseDiagramConfig, overrides: List[str]
) -> PhaseDiagramConfig:
    """Apply command-line overrides of the form ``key=value``.

    Examples::

        apply_cli_overrides(cfg, ["grid.lr_points=50", "ode.atol=1e-10"])
    """
    override_dict: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, val_str = item.split("=", 1)
        override_dict[key] = _parse_value(val_str)
    return cfg.merge(override_dict)


def _parse_value(s: str) -> Any:
    """Parse a string value to int, float, bool, or keep as str."""
    if s.lower() in ("true", "yes"):
        return True
    if s.lower() in ("false", "no"):
        return False
    if s.lower() == "none":
        return None
    # Try int
    try:
        return int(s)
    except ValueError:
        pass
    # Try float
    try:
        return float(s)
    except ValueError:
        pass
    # List syntax [a,b,c]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(x.strip()) for x in inner.split(",")]
    return s
