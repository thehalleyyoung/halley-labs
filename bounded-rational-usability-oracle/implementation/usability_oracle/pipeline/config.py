"""
usability_oracle.pipeline.config — Pipeline configuration management.

Provides :class:`FullPipelineConfig` which aggregates per-stage configs
and :class:`StageConfig` for individual stage settings.  Supports loading
from YAML files, dicts, and environment variables.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from usability_oracle.core.config import (
    AlignmentConfig,
    BisimulationConfig,
    CognitiveConfig,
    ComparisonConfig,
    MDPConfig,
    OracleConfig,
    OutputConfig,
    ParserConfig,
    PipelineConfig,
    PolicyConfig,
    RepairConfig,
)
from usability_oracle.core.enums import PipelineStage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StageConfig
# ---------------------------------------------------------------------------

@dataclass
class StageConfig:
    """Configuration for a single pipeline stage.

    Attributes
    ----------
    enabled : bool
        Whether this stage should be executed.
    timeout : float
        Maximum execution time in seconds (0 = unlimited).
    retry : int
        Number of retries on transient failure.
    fail_fast : bool
        If True, abort the entire pipeline on failure.
    """

    enabled: bool = True
    timeout: float = 60.0
    retry: int = 0
    fail_fast: bool = True

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.timeout < 0:
            errors.append(f"timeout must be non-negative, got {self.timeout}")
        if self.retry < 0:
            errors.append(f"retry must be non-negative, got {self.retry}")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "timeout": self.timeout,
            "retry": self.retry,
            "fail_fast": self.fail_fast,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageConfig:
        return cls(
            enabled=data.get("enabled", True),
            timeout=float(data.get("timeout", 60.0)),
            retry=int(data.get("retry", 0)),
            fail_fast=data.get("fail_fast", True),
        )


# ---------------------------------------------------------------------------
# FullPipelineConfig
# ---------------------------------------------------------------------------

@dataclass
class FullPipelineConfig:
    """Aggregated configuration for the entire analysis pipeline.

    Combines the core :class:`OracleConfig` with per-stage execution
    settings, cache configuration, and output preferences.

    Attributes
    ----------
    oracle : OracleConfig
        Domain-specific configuration (cognitive parameters, etc.).
    stages : dict[PipelineStage, StageConfig]
        Per-stage execution settings.
    cache_dir : str | None
        Directory for result caching; None disables caching.
    cache_ttl : float
        Cache entry time-to-live in seconds (default 3600 = 1 hour).
    max_workers : int
        Maximum parallel workers for independent stages.
    verbose : bool
        Enable verbose logging.
    output_format : str
        Default output format (json, sarif, html, console).
    """

    oracle: OracleConfig = field(default_factory=OracleConfig)
    stages: dict[str, StageConfig] = field(default_factory=dict)
    cache_dir: Optional[str] = None
    cache_ttl: float = 3600.0
    max_workers: int = 4
    verbose: bool = False
    output_format: str = "json"

    def __post_init__(self) -> None:
        """Ensure all pipeline stages have a config entry."""
        for stage in PipelineStage:
            if stage.value not in self.stages:
                self.stages[stage.value] = StageConfig()

    def get_stage_config(self, stage: PipelineStage) -> StageConfig:
        """Return the StageConfig for the given stage."""
        return self.stages.get(stage.value, StageConfig())

    def is_stage_enabled(self, stage: PipelineStage) -> bool:
        return self.get_stage_config(stage).enabled

    # ── Validation --------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate the entire configuration.  Returns list of errors."""
        errors: list[str] = []

        # Validate per-stage configs
        for stage_name, sc in self.stages.items():
            for err in sc.validate():
                errors.append(f"Stage {stage_name}: {err}")

        # Validate oracle config sub-fields
        if self.oracle.policy.beta_range[0] >= self.oracle.policy.beta_range[1]:
            errors.append(
                f"beta_range lower ({self.oracle.policy.beta_range[0]}) must be < "
                f"upper ({self.oracle.policy.beta_range[1]})"
            )
        if self.oracle.mdp.discount_factor <= 0 or self.oracle.mdp.discount_factor > 1:
            errors.append(
                f"discount_factor must be in (0, 1], "
                f"got {self.oracle.mdp.discount_factor}"
            )
        if self.max_workers < 1:
            errors.append(f"max_workers must be ≥ 1, got {self.max_workers}")

        if self.cache_dir is not None:
            cache_path = Path(self.cache_dir)
            if cache_path.exists() and not cache_path.is_dir():
                errors.append(f"cache_dir {self.cache_dir!r} exists but is not a directory")

        return errors

    # ── Serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "oracle": {
                "parser": {"max_depth": self.oracle.parser.max_tree_depth,
                           "include_hidden": self.oracle.parser.include_hidden},
                "cognitive": {
                    "fitts_a": self.oracle.cognitive.fitts_a,
                    "fitts_b": self.oracle.cognitive.fitts_b,
                    "hick_a": self.oracle.cognitive.hick_a,
                    "hick_b": self.oracle.cognitive.hick_b,
                    "working_memory_capacity": self.oracle.cognitive.working_memory_capacity,
                },
                "mdp": {
                    "max_states": self.oracle.mdp.max_states,
                    "discount_factor": self.oracle.mdp.discount_factor,
                },
                "policy": {
                    "beta_range": list(self.oracle.policy.beta_range),
                    "beta_steps": self.oracle.policy.beta_steps,
                },
                "repair": {
                    "max_mutations": self.oracle.repair.max_mutations,
                    "timeout_seconds": self.oracle.repair.timeout_seconds,
                },
            },
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "cache_dir": self.cache_dir,
            "cache_ttl": self.cache_ttl,
            "max_workers": self.max_workers,
            "verbose": self.verbose,
            "output_format": self.output_format,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FullPipelineConfig:
        """Construct from a dictionary (e.g. parsed YAML)."""
        oracle_data = d.get("oracle", {})
        oracle = OracleConfig()

        # Parser
        parser_data = oracle_data.get("parser", {})
        oracle.parser = ParserConfig(
            max_tree_depth=parser_data.get("max_depth", 50),
            include_hidden=parser_data.get("include_hidden", False),
        )

        # Cognitive
        cog_data = oracle_data.get("cognitive", {})
        oracle.cognitive = CognitiveConfig(
            fitts_a=cog_data.get("fitts_a", 0.05),
            fitts_b=cog_data.get("fitts_b", 0.15),
            hick_a=cog_data.get("hick_a", 0.2),
            hick_b=cog_data.get("hick_b", 0.15),
            working_memory_capacity=cog_data.get("working_memory_capacity", 4),
        )

        # MDP
        mdp_data = oracle_data.get("mdp", {})
        oracle.mdp = MDPConfig(
            max_states=mdp_data.get("max_states", 10000),
            discount_factor=mdp_data.get("discount_factor", 0.99),
        )

        # Policy
        pol_data = oracle_data.get("policy", {})
        oracle.policy = PolicyConfig(
            beta_range=tuple(pol_data.get("beta_range", [0.1, 20.0])),
            beta_steps=pol_data.get("beta_steps", 50),
        )

        # Repair
        rep_data = oracle_data.get("repair", {})
        oracle.repair = RepairConfig(
            max_mutations=rep_data.get("max_mutations", 5),
            timeout_seconds=rep_data.get("timeout_seconds", 30.0),
        )

        # Stage configs
        stages: dict[str, StageConfig] = {}
        for stage_name, stage_data in d.get("stages", {}).items():
            stages[stage_name] = StageConfig.from_dict(stage_data)

        return cls(
            oracle=oracle,
            stages=stages,
            cache_dir=d.get("cache_dir"),
            cache_ttl=float(d.get("cache_ttl", 3600.0)),
            max_workers=int(d.get("max_workers", 4)),
            verbose=d.get("verbose", False),
            output_format=d.get("output_format", "json"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> FullPipelineConfig:
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML dict, got {type(data).__name__}")

        logger.info("Loaded pipeline config from %s", path)
        return cls.from_dict(data)

    @classmethod
    def DEFAULT(cls) -> FullPipelineConfig:
        """Return the default configuration."""
        return cls()

    @classmethod
    def from_env(cls) -> FullPipelineConfig:
        """Create config with overrides from environment variables."""
        config = cls.DEFAULT()

        if os.environ.get("USABILITY_ORACLE_VERBOSE"):
            config.verbose = True
        if os.environ.get("USABILITY_ORACLE_CACHE_DIR"):
            config.cache_dir = os.environ["USABILITY_ORACLE_CACHE_DIR"]
        if os.environ.get("USABILITY_ORACLE_MAX_WORKERS"):
            config.max_workers = int(os.environ["USABILITY_ORACLE_MAX_WORKERS"])
        if os.environ.get("USABILITY_ORACLE_OUTPUT_FORMAT"):
            config.output_format = os.environ["USABILITY_ORACLE_OUTPUT_FORMAT"]

        return config
