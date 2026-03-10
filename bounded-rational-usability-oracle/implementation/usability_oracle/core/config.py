"""
usability_oracle.core.config — Configuration dataclasses for every pipeline stage.

All configs are plain dataclasses with ``to_dict`` / ``from_dict`` round-trip
serialisation and a ``validate`` method that raises
:class:`~usability_oracle.core.errors.ValidationError` on bad values.

The master :class:`OracleConfig` nests every sub-config and can be loaded
from a YAML file via the ``from_yaml`` classmethod.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from usability_oracle.core.errors import ValidationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValidationError(
            f"{name} must be positive, got {value}",
            field=name, value=value, constraint="> 0",
        )


def _non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValidationError(
            f"{name} must be non-negative, got {value}",
            field=name, value=value, constraint=">= 0",
        )


def _in_range(name: str, value: float, lo: float, hi: float) -> None:
    if not (lo <= value <= hi):
        raise ValidationError(
            f"{name} must be in [{lo}, {hi}], got {value}",
            field=name, value=value, constraint=f"[{lo}, {hi}]",
        )


# ═══════════════════════════════════════════════════════════════════════════
# ParserConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ParserConfig:
    """Configuration for the accessibility-tree parser."""

    parser_type: str = "html"
    """One of 'html', 'axe_json', 'platform_native'."""

    strictness: str = "normal"
    """Parsing strictness: 'lenient', 'normal', 'strict'."""

    normalize_whitespace: bool = True
    """Collapse runs of whitespace in accessible names."""

    max_tree_depth: int = 64
    """Safety limit on tree depth (prevents stack overflow)."""

    include_hidden: bool = False
    """Whether to include aria-hidden nodes."""

    def validate(self) -> None:
        if self.parser_type not in ("html", "axe_json", "platform_native"):
            raise ValidationError(
                f"Unknown parser_type: {self.parser_type!r}",
                field="parser_type", value=self.parser_type,
            )
        if self.strictness not in ("lenient", "normal", "strict"):
            raise ValidationError(
                f"Unknown strictness: {self.strictness!r}",
                field="strictness", value=self.strictness,
            )
        _positive("max_tree_depth", self.max_tree_depth)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ParserConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════
# AlignmentConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AlignmentConfig:
    """Configuration for tree-alignment."""

    fuzzy_threshold: float = 0.6
    """Minimum similarity score to consider a fuzzy match."""

    max_edit_distance: int = 50
    """Maximum tree edit-distance to attempt (prune large diffs)."""

    weight_role: float = 0.4
    """Weight of role similarity in the matching score."""

    weight_label: float = 0.35
    """Weight of accessible-name similarity in the matching score."""

    weight_position: float = 0.25
    """Weight of spatial proximity in the matching score."""

    use_structural_hint: bool = True
    """If True, use tree structure (depth, sibling index) as a hint."""

    def validate(self) -> None:
        _in_range("fuzzy_threshold", self.fuzzy_threshold, 0.0, 1.0)
        _positive("max_edit_distance", self.max_edit_distance)
        total = self.weight_role + self.weight_label + self.weight_position
        if abs(total - 1.0) > 1e-6:
            raise ValidationError(
                f"Alignment weights must sum to 1.0, got {total:.6f}",
                field="weight_role+weight_label+weight_position",
                value=total, constraint="== 1.0",
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AlignmentConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════
# CognitiveConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CognitiveConfig:
    """Parameters for the cognitive cost models.

    Defaults are population medians from the HCI literature.
    """

    # Fitts' Law: MT = a + b * log2(D/W + 1)
    fitts_a: float = 0.050
    """Fitts intercept (seconds). MacKenzie 1992 median ~ 50ms."""

    fitts_b: float = 0.150
    """Fitts slope (seconds/bit). MacKenzie 1992 median ~ 150ms/bit."""

    # Hick-Hyman Law: RT = a + b * log2(n + 1)
    hick_a: float = 0.200
    """Hick intercept (seconds)."""

    hick_b: float = 0.150
    """Hick slope (seconds/bit)."""

    # Visual search
    visual_search_slope: float = 0.025
    """Visual search slope (seconds/item) for inefficient search."""

    visual_search_intercept: float = 0.400
    """Visual search base time (seconds)."""

    # Working memory
    working_memory_capacity: int = 4
    """Effective WM capacity (Cowan 2001: ~4 chunks)."""

    working_memory_decay_rate: float = 0.069
    """WM decay rate lambda (1/s) such that half-life ~ 10 s."""

    # Motor execution
    motor_preparation_time: float = 0.150
    """Motor preparation time (seconds)."""

    motor_execution_time: float = 0.100
    """Base motor execution time per action (seconds)."""

    # Perceptual
    perception_time: float = 0.100
    """Visual perceptual encoding time (seconds)."""

    def validate(self) -> None:
        _non_negative("fitts_a", self.fitts_a)
        _positive("fitts_b", self.fitts_b)
        _non_negative("hick_a", self.hick_a)
        _positive("hick_b", self.hick_b)
        _positive("visual_search_slope", self.visual_search_slope)
        _non_negative("visual_search_intercept", self.visual_search_intercept)
        _positive("working_memory_capacity", self.working_memory_capacity)
        _positive("working_memory_decay_rate", self.working_memory_decay_rate)
        _non_negative("motor_preparation_time", self.motor_preparation_time)
        _non_negative("motor_execution_time", self.motor_execution_time)
        _non_negative("perception_time", self.perception_time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CognitiveConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════
# MDPConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MDPConfig:
    """Configuration for MDP construction."""

    max_states: int = 10_000
    """Hard limit on the number of states in the MDP."""

    discount_factor: float = 0.99
    """Discount factor gamma in (0, 1]."""

    convergence_threshold: float = 1e-6
    """Bellman residual threshold for value iteration."""

    max_iterations: int = 1_000
    """Maximum value-iteration sweeps."""

    prune_unreachable: bool = True
    """Remove states unreachable from the initial state."""

    def validate(self) -> None:
        _positive("max_states", self.max_states)
        _in_range("discount_factor", self.discount_factor, 0.0, 1.0)
        _positive("convergence_threshold", self.convergence_threshold)
        _positive("max_iterations", self.max_iterations)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MDPConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════
# PolicyConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PolicyConfig:
    """Configuration for bounded-rational policy computation."""

    beta_range: Tuple[float, float] = (0.1, 20.0)
    """Range of rationality parameter beta to sweep."""

    beta_steps: int = 50
    """Number of beta values in the sweep."""

    softmax_temperature: float = 1.0
    """Temperature for soft value iteration (usually = 1/beta)."""

    num_trajectories: int = 1_000
    """Number of Monte-Carlo trajectory samples."""

    trajectory_length: int = 100
    """Maximum steps per sampled trajectory."""

    use_log_sum_exp_trick: bool = True
    """Use numerically stable log-sum-exp in softmax."""

    def validate(self) -> None:
        lo, hi = self.beta_range
        _positive("beta_range[0]", lo)
        if hi <= lo:
            raise ValidationError(
                f"beta_range upper ({hi}) must exceed lower ({lo})",
                field="beta_range", value=self.beta_range,
            )
        _positive("beta_steps", self.beta_steps)
        _positive("softmax_temperature", self.softmax_temperature)
        _positive("num_trajectories", self.num_trajectories)
        _positive("trajectory_length", self.trajectory_length)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["beta_range"] = list(self.beta_range)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PolicyConfig:
        d = dict(d)
        if "beta_range" in d and isinstance(d["beta_range"], list):
            d["beta_range"] = tuple(d["beta_range"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════
# BisimulationConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BisimulationConfig:
    """Configuration for bisimulation-based state reduction."""

    epsilon: float = 1e-4
    """Tolerance for approximate bisimulation metric."""

    max_partitions: int = 5_000
    """Hard limit on partition count (prevents runaway refinement)."""

    use_heuristic_fallback: bool = True
    """Fall back to a cheaper heuristic when exact refinement is too slow."""

    signature_dimensions: int = 8
    """Dimensionality of state signatures for heuristic grouping."""

    def validate(self) -> None:
        _positive("epsilon", self.epsilon)
        _positive("max_partitions", self.max_partitions)
        _positive("signature_dimensions", self.signature_dimensions)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BisimulationConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════
# ComparisonConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ComparisonConfig:
    """Configuration for statistical usability comparison."""

    alpha: float = 0.05
    """Significance level for hypothesis tests."""

    multiple_testing_correction: str = "holm"
    """Correction method: 'bonferroni', 'holm', 'bh', 'none'."""

    min_effect_size: float = 0.2
    """Minimum Cohen's d to report as a practical regression."""

    comparison_mode: str = "paired"
    """One of 'paired', 'independent', 'parameter_free'."""

    bootstrap_iterations: int = 10_000
    """Number of bootstrap samples for CI estimation."""

    def validate(self) -> None:
        _in_range("alpha", self.alpha, 0.001, 0.5)
        if self.multiple_testing_correction not in (
            "bonferroni", "holm", "bh", "none",
        ):
            raise ValidationError(
                f"Unknown correction: {self.multiple_testing_correction!r}",
                field="multiple_testing_correction",
                value=self.multiple_testing_correction,
            )
        _positive("min_effect_size", self.min_effect_size)
        _positive("bootstrap_iterations", self.bootstrap_iterations)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ComparisonConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════
# RepairConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RepairConfig:
    """Configuration for automated repair synthesis."""

    max_mutations: int = 5
    """Maximum number of tree edits in a single repair candidate."""

    timeout_seconds: float = 30.0
    """Hard timeout for the synthesis search."""

    solver_backend: str = "greedy"
    """Solver: 'greedy', 'beam_search', 'smt'."""

    beam_width: int = 10
    """Beam width for beam-search solver."""

    preserve_semantics: bool = True
    """Require that repairs do not alter task semantics."""

    def validate(self) -> None:
        _positive("max_mutations", self.max_mutations)
        _positive("timeout_seconds", self.timeout_seconds)
        if self.solver_backend not in ("greedy", "beam_search", "smt"):
            raise ValidationError(
                f"Unknown solver: {self.solver_backend!r}",
                field="solver_backend", value=self.solver_backend,
            )
        _positive("beam_width", self.beam_width)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RepairConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════
# PipelineConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Configuration for the pipeline orchestrator."""

    stages_enabled: List[str] = field(default_factory=lambda: [
        "parse", "align", "cost", "mdp_build", "bisimulate",
        "policy", "compare", "bottleneck", "repair",
    ])
    """Which pipeline stages to execute (in order)."""

    parallelism: int = 1
    """Number of parallel workers (1 = sequential)."""

    cache_dir: Optional[str] = None
    """Directory for caching intermediate results (None = no cache)."""

    log_level: str = "INFO"
    """Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL."""

    fail_fast: bool = True
    """If True, abort the pipeline on the first stage failure."""

    def validate(self) -> None:
        _positive("parallelism", self.parallelism)
        valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        if self.log_level.upper() not in valid_levels:
            raise ValidationError(
                f"Unknown log_level: {self.log_level!r}",
                field="log_level", value=self.log_level,
            )

    @property
    def log_level_int(self) -> int:
        return getattr(logging, self.log_level.upper(), logging.INFO)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PipelineConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════
# OutputConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OutputConfig:
    """Configuration for report formatting."""

    format: str = "console"
    """Output format: 'json', 'sarif', 'html', 'console'."""

    verbosity: int = 1
    """Verbosity level: 0 (minimal), 1 (normal), 2 (detailed)."""

    include_visualization: bool = False
    """Whether to include tree-diff visualisations in HTML output."""

    output_path: Optional[str] = None
    """If set, write the report to this file path."""

    include_raw_data: bool = False
    """Include raw trajectory / cost data in JSON output."""

    def validate(self) -> None:
        if self.format not in ("json", "sarif", "html", "console"):
            raise ValidationError(
                f"Unknown format: {self.format!r}",
                field="format", value=self.format,
            )
        _in_range("verbosity", self.verbosity, 0, 2)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> OutputConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════
# OracleConfig  —  master configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OracleConfig:
    """Master configuration aggregating all sub-configs.

    Can be constructed programmatically, from a dict, or from a YAML file.
    """

    parser: ParserConfig = field(default_factory=ParserConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    cognitive: CognitiveConfig = field(default_factory=CognitiveConfig)
    mdp: MDPConfig = field(default_factory=MDPConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    bisimulation: BisimulationConfig = field(default_factory=BisimulationConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    repair: RepairConfig = field(default_factory=RepairConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        """Validate every sub-config.  Raises on the first error."""
        self.parser.validate()
        self.alignment.validate()
        self.cognitive.validate()
        self.mdp.validate()
        self.policy.validate()
        self.bisimulation.validate()
        self.comparison.validate()
        self.repair.validate()
        self.pipeline.validate()
        self.output.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parser": self.parser.to_dict(),
            "alignment": self.alignment.to_dict(),
            "cognitive": self.cognitive.to_dict(),
            "mdp": self.mdp.to_dict(),
            "policy": self.policy.to_dict(),
            "bisimulation": self.bisimulation.to_dict(),
            "comparison": self.comparison.to_dict(),
            "repair": self.repair.to_dict(),
            "pipeline": self.pipeline.to_dict(),
            "output": self.output.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> OracleConfig:
        return cls(
            parser=ParserConfig.from_dict(d.get("parser", {})),
            alignment=AlignmentConfig.from_dict(d.get("alignment", {})),
            cognitive=CognitiveConfig.from_dict(d.get("cognitive", {})),
            mdp=MDPConfig.from_dict(d.get("mdp", {})),
            policy=PolicyConfig.from_dict(d.get("policy", {})),
            bisimulation=BisimulationConfig.from_dict(d.get("bisimulation", {})),
            comparison=ComparisonConfig.from_dict(d.get("comparison", {})),
            repair=RepairConfig.from_dict(d.get("repair", {})),
            pipeline=PipelineConfig.from_dict(d.get("pipeline", {})),
            output=OutputConfig.from_dict(d.get("output", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> OracleConfig:
        """Load configuration from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to the YAML configuration file.

        Returns
        -------
        OracleConfig
            Parsed and validated configuration.

        Raises
        ------
        ConfigError
            If the file cannot be read or parsed.
        """
        import yaml  # type: ignore[import-untyped]

        from usability_oracle.core.errors import ConfigError

        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Config file not found: {p}", key=str(p))
        try:
            with open(p) as f:
                raw = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Failed to parse YAML config: {exc}", key=str(p),
            ) from exc

        if not isinstance(raw, dict):
            raise ConfigError("YAML root must be a mapping", key=str(p))

        cfg = cls.from_dict(raw)
        cfg.validate()
        return cfg

    @classmethod
    def default(cls) -> OracleConfig:
        """Return a fully-defaulted, validated configuration."""
        cfg = cls()
        cfg.validate()
        return cfg

    def merge(self, overrides: Dict[str, Any]) -> OracleConfig:
        """Return a new config with *overrides* applied on top of self."""
        base = self.to_dict()
        for section, values in overrides.items():
            if section in base and isinstance(values, dict):
                base[section].update(values)
            else:
                base[section] = values
        return OracleConfig.from_dict(base)

    def __repr__(self) -> str:
        return (
            f"OracleConfig(parser={self.parser.parser_type!r}, "
            f"stages={len(self.pipeline.stages_enabled)})"
        )


__all__ = [
    "OracleConfig",
    "ParserConfig",
    "AlignmentConfig",
    "CognitiveConfig",
    "MDPConfig",
    "PolicyConfig",
    "BisimulationConfig",
    "ComparisonConfig",
    "RepairConfig",
    "PipelineConfig",
    "OutputConfig",
]
