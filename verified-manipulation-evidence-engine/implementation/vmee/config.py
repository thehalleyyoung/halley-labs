"""
Configuration management for VMEE pipeline.

Handles loading, validation, and default configuration for all
pipeline components via TOML files and Pydantic models.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import toml


class ManipulationType(str, Enum):
    """Supported manipulation types."""
    SPOOFING = "spoofing"
    LAYERING = "layering"
    WASH_TRADING = "wash_trading"


class InferenceMethod(str, Enum):
    """Bayesian inference methods."""
    EXACT_AC = "exact_arithmetic_circuit"
    CUTSET = "bounded_cutset"
    GIBBS = "gibbs_sampling"
    VARIATIONAL = "mean_field_variational"


class CausalAlgorithm(str, Enum):
    """Causal discovery algorithms."""
    PC = "pc"
    FCI = "fci"
    GES = "ges"
    SCORE_BASED = "score_based"


@dataclass
class LOBConfig:
    """Limit order book simulator configuration."""
    num_price_levels: int = 100
    tick_size: float = 0.01
    lot_size: int = 100
    initial_mid_price: float = 100.0
    initial_spread: int = 2
    initial_depth_per_level: int = 500
    max_queue_size: int = 10000
    num_instruments: int = 10
    trading_hours: float = 6.5
    events_per_second: float = 50.0
    seed: int = 42
    latency_mean_us: float = 50.0
    latency_std_us: float = 10.0
    fee_maker: float = -0.0002
    fee_taker: float = 0.0003
    min_order_size: int = 1
    max_order_size: int = 10000

    def total_events(self) -> int:
        return int(self.trading_hours * 3600 * self.events_per_second)

    def validate(self) -> list[str]:
        errors = []
        if self.tick_size <= 0:
            errors.append("tick_size must be positive")
        if self.lot_size <= 0:
            errors.append("lot_size must be positive")
        if self.num_price_levels < 10:
            errors.append("num_price_levels must be >= 10")
        if self.initial_mid_price <= 0:
            errors.append("initial_mid_price must be positive")
        if self.events_per_second <= 0:
            errors.append("events_per_second must be positive")
        return errors


@dataclass
class CausalConfig:
    """Causal discovery engine configuration."""
    algorithm: CausalAlgorithm = CausalAlgorithm.PC
    significance_level: float = 0.05
    max_conditioning_set: int = 5
    hsic_kernel: str = "gaussian"
    hsic_num_permutations: int = 500
    window_size: int = 1000
    window_stride: int = 200
    change_detection_threshold: float = 0.01
    max_parents: int = 8
    score_function: str = "bic"
    use_additive_noise: bool = True
    anm_regression: str = "gam"
    bootstrap_samples: int = 200
    edge_confidence_threshold: float = 0.7
    faithfulness_check: bool = True
    causal_sufficiency: bool = True
    do_calculus_max_depth: int = 10
    identification_method: str = "id_algorithm"
    seed: int = 42

    def validate(self) -> list[str]:
        errors = []
        if not 0 < self.significance_level < 1:
            errors.append("significance_level must be in (0, 1)")
        if self.max_conditioning_set < 0:
            errors.append("max_conditioning_set must be non-negative")
        if self.window_size < 10:
            errors.append("window_size must be >= 10")
        if self.bootstrap_samples < 10:
            errors.append("bootstrap_samples must be >= 10")
        return errors


@dataclass
class BayesianConfig:
    """Bayesian inference engine configuration."""
    method: InferenceMethod = InferenceMethod.EXACT_AC
    treewidth_bound: int = 15
    max_circuit_edges: int = 10_000_000
    cutset_max_size: int = 5
    num_discretization_bins: int = 50
    prior_type: str = "conjugate"
    prior_strength: float = 1.0
    gibbs_iterations: int = 10000
    gibbs_burnin: int = 2000
    gibbs_thin: int = 5
    variational_max_iter: int = 1000
    variational_tol: float = 1e-6
    map_timeout_seconds: float = 300.0
    bayes_factor_threshold: float = 10.0
    evidence_threshold: float = 0.95
    hmm_num_states: int = 4
    hmm_emission_type: str = "gaussian"
    seed: int = 42

    def validate(self) -> list[str]:
        errors = []
        if self.treewidth_bound < 1:
            errors.append("treewidth_bound must be >= 1")
        if self.num_discretization_bins < 2:
            errors.append("num_discretization_bins must be >= 2")
        if self.gibbs_iterations < 100:
            errors.append("gibbs_iterations must be >= 100")
        return errors


@dataclass
class TemporalConfig:
    """Temporal logic monitoring configuration."""
    max_temporal_horizon: int = 10000
    monitoring_memory_bound_mb: float = 512.0
    incremental: bool = True
    quantitative: bool = True
    formula_timeout_seconds: float = 60.0
    regulatory_specs: list[str] = field(
        default_factory=lambda: ["spoofing_basic", "layering_basic", "wash_trading_basic"]
    )
    time_granularity_us: int = 1000
    event_buffer_size: int = 100000
    parallel_monitors: int = 4
    satisfaction_threshold: float = 0.0
    robustness_margin: float = 0.1

    def validate(self) -> list[str]:
        errors = []
        if self.max_temporal_horizon < 1:
            errors.append("max_temporal_horizon must be >= 1")
        if self.monitoring_memory_bound_mb <= 0:
            errors.append("monitoring_memory_bound_mb must be positive")
        return errors


@dataclass
class ProofConfig:
    """SMT proof generation configuration."""
    solver: str = "z3"
    timeout_seconds: float = 300.0
    logic: str = "QF_LRA"
    cross_validate: bool = True
    proof_format: str = "smtlib2"
    incremental: bool = True
    certificate_format: str = "json"
    equisatisfiability_check: bool = True
    max_clauses: int = 1_000_000
    rational_precision: int = 64
    posterior_encoding: str = "interval"
    bayes_factor_encoding: str = "ratio"
    probability_tolerance: float = 1e-10

    def validate(self) -> list[str]:
        errors = []
        if self.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")
        if self.max_clauses < 100:
            errors.append("max_clauses must be >= 100")
        return errors


@dataclass
class AdversarialConfig:
    """RL adversarial stress-testing configuration."""
    algorithm: str = "ppo"
    num_episodes: int = 1_000_000
    max_episode_length: int = 500
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    num_epochs: int = 4
    batch_size: int = 64
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])
    activation: str = "tanh"
    strategy_types: list[str] = field(
        default_factory=lambda: ["spoofing", "layering", "wash_trading"]
    )
    reward_shaping: bool = True
    detection_penalty: float = -10.0
    profit_reward_scale: float = 1.0
    coverage_target: float = 0.8
    seed: int = 42
    training_hours: float = 24.0
    checkpoint_interval: int = 10000
    eval_interval: int = 5000

    def validate(self) -> list[str]:
        errors = []
        if self.num_episodes < 100:
            errors.append("num_episodes must be >= 100")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if not 0 < self.gamma <= 1:
            errors.append("gamma must be in (0, 1]")
        if not 0 < self.clip_epsilon < 1:
            errors.append("clip_epsilon must be in (0, 1)")
        return errors


@dataclass
class EvidenceConfig:
    """Evidence assembly configuration."""
    output_format: str = "json"
    include_causal_subgraph: bool = True
    include_posterior_summary: bool = True
    include_temporal_violations: bool = True
    include_smt_proofs: bool = True
    include_adversarial_coverage: bool = True
    bundle_compression: bool = True
    max_bundle_size_mb: float = 100.0
    verification_on_assembly: bool = True
    hash_algorithm: str = "sha256"
    sign_bundles: bool = False
    metadata_fields: list[str] = field(
        default_factory=lambda: ["timestamp", "version", "config_hash"]
    )


@dataclass
class CalibrationConfig:
    """Sim-to-real calibration configuration."""
    reference_dataset: str = "lobster"
    ks_threshold: float = 0.1
    num_calibration_samples: int = 10000
    features_to_calibrate: list[str] = field(
        default_factory=lambda: [
            "bid_ask_spread", "queue_length", "inter_arrival_time",
            "cancellation_rate", "order_size_distribution",
        ]
    )
    optimization_method: str = "nelder-mead"
    optimization_max_iter: int = 1000
    published_stats_source: str = "cont_2014"


@dataclass
class EvaluationConfig:
    """Evaluation and benchmarking configuration."""
    num_scenarios_per_type: int = 1000
    manipulation_types: list[str] = field(
        default_factory=lambda: ["spoofing", "layering", "wash_trading"]
    )
    baselines: list[str] = field(
        default_factory=lambda: ["rule_based", "isolation_forest", "uncertified_bayesian"]
    )
    confidence_level: float = 0.95
    false_positive_budget: float = 0.05
    metrics: list[str] = field(
        default_factory=lambda: [
            "detection_rate", "false_positive_rate", "bayes_factor_accuracy",
            "proof_validity_rate", "evidence_quality_score", "latency",
        ]
    )
    output_dir: str = "evaluation_results"


@dataclass
class VMEEConfig:
    """Top-level VMEE configuration."""
    lob: LOBConfig = field(default_factory=LOBConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    bayesian: BayesianConfig = field(default_factory=BayesianConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    proof: ProofConfig = field(default_factory=ProofConfig)
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    log_level: str = "INFO"
    output_dir: str = "output"
    seed: int = 42
    max_workers: int = 4
    dry_run: bool = False

    def validate(self) -> list[str]:
        """Validate all sub-configurations and return list of errors."""
        errors = []
        errors.extend(self.lob.validate())
        errors.extend(self.causal.validate())
        errors.extend(self.bayesian.validate())
        errors.extend(self.temporal.validate())
        errors.extend(self.proof.validate())
        errors.extend(self.adversarial.validate())
        return errors


def _apply_dict(obj: Any, d: dict) -> None:
    """Recursively apply a dict of values to a dataclass instance."""
    for key, val in d.items():
        if hasattr(obj, key):
            attr = getattr(obj, key)
            if isinstance(val, dict) and hasattr(attr, "__dataclass_fields__"):
                _apply_dict(attr, val)
            else:
                setattr(obj, key, val)


def load_config(path: Union[str, Path]) -> VMEEConfig:
    """Load configuration from a TOML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = toml.load(f)

    config = VMEEConfig()
    _apply_dict(config, raw)

    errors = config.validate()
    if errors:
        raise ValueError(
            f"Configuration validation failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return config


def save_config(config: VMEEConfig, path: Union[str, Path]) -> None:
    """Save configuration to a TOML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def to_dict(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            result = {}
            for fname in obj.__dataclass_fields__:
                val = getattr(obj, fname)
                result[fname] = to_dict(val)
            return result
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, list):
            return [to_dict(v) for v in obj]
        return obj

    with open(path, "w") as f:
        toml.dump(to_dict(config), f)


def default_config() -> VMEEConfig:
    """Return a default VMEEConfig instance."""
    return VMEEConfig()


def merge_configs(base: VMEEConfig, override: dict) -> VMEEConfig:
    """Merge an override dictionary into a base configuration."""
    import copy
    merged = copy.deepcopy(base)
    _apply_dict(merged, override)
    return merged


def config_hash(config: VMEEConfig) -> str:
    """Compute a deterministic hash of the configuration."""
    import hashlib

    def to_dict(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, list):
            return [to_dict(v) for v in obj]
        return obj

    serialized = json.dumps(to_dict(config), sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]
