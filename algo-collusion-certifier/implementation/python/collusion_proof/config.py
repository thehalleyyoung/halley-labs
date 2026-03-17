"""Configuration management for CollusionProof."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------
DEFAULT_ALPHA = 0.05
DEFAULT_BOOTSTRAP_SAMPLES = 10000
DEFAULT_NUM_ROUNDS = 1_000_000
DEFAULT_CONVERGENCE_WINDOW = 100_000
DEFAULT_NUM_PLAYERS = 2
DEFAULT_NUM_ACTIONS = 15
DEFAULT_DISCOUNT_FACTOR = 0.95
DEFAULT_MARGINAL_COST = 1.0
DEFAULT_DEMAND_INTERCEPT = 10.0
DEFAULT_DEMAND_SLOPE = 1.0

# Alpha spending allocation across tiers
ALPHA_ALLOCATION: Dict[str, float] = {
    "tier1": 0.02,
    "tier2": 0.015,
    "tier3": 0.01,
    "tier4": 0.005,
}


# ---------------------------------------------------------------------------
# Dataclass configs
# ---------------------------------------------------------------------------
@dataclass
class TestConfig:
    """Configuration for statistical testing."""

    alpha: float = DEFAULT_ALPHA
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES
    permutation_samples: int = 5000
    confidence_level: float = 0.95
    min_effect_size: float = 0.01
    convergence_threshold: float = 0.001
    convergence_window: int = DEFAULT_CONVERGENCE_WINDOW
    alpha_allocation: Dict[str, float] = field(
        default_factory=lambda: dict(ALPHA_ALLOCATION)
    )
    multiple_testing_method: str = "holm"

    def validate(self) -> List[str]:
        """Return a list of validation error strings (empty == valid)."""
        errors: List[str] = []
        if not 0 < self.alpha < 1:
            errors.append(f"alpha must be in (0, 1), got {self.alpha}")
        if self.bootstrap_samples < 100:
            errors.append(
                f"bootstrap_samples must be >= 100, got {self.bootstrap_samples}"
            )
        if self.permutation_samples < 100:
            errors.append(
                f"permutation_samples must be >= 100, got {self.permutation_samples}"
            )
        if not 0 < self.confidence_level < 1:
            errors.append(
                f"confidence_level must be in (0, 1), got {self.confidence_level}"
            )
        if self.min_effect_size < 0:
            errors.append(
                f"min_effect_size must be >= 0, got {self.min_effect_size}"
            )
        if self.convergence_threshold <= 0:
            errors.append(
                f"convergence_threshold must be > 0, got {self.convergence_threshold}"
            )
        if self.convergence_window < 2:
            errors.append(
                f"convergence_window must be >= 2, got {self.convergence_window}"
            )
        alloc_sum = sum(self.alpha_allocation.values())
        if abs(alloc_sum - self.alpha) > 1e-9 and self.alpha_allocation:
            errors.append(
                f"alpha_allocation values sum to {alloc_sum}, expected {self.alpha}"
            )
        if self.multiple_testing_method not in ("holm", "bonferroni", "bh", "none"):
            errors.append(
                f"Unknown multiple_testing_method: {self.multiple_testing_method}"
            )
        return errors

    def effective_alpha(self, tier: str) -> float:
        """Return the alpha budget allocated to *tier*."""
        if tier in self.alpha_allocation:
            return self.alpha_allocation[tier]
        return self.alpha / max(len(self.alpha_allocation), 1)


@dataclass
class MarketConfig:
    """Market simulation configuration."""

    num_players: int = DEFAULT_NUM_PLAYERS
    num_rounds: int = DEFAULT_NUM_ROUNDS
    num_actions: int = DEFAULT_NUM_ACTIONS
    marginal_cost: float = DEFAULT_MARGINAL_COST
    demand_intercept: float = DEFAULT_DEMAND_INTERCEPT
    demand_slope: float = DEFAULT_DEMAND_SLOPE
    discount_factor: float = DEFAULT_DISCOUNT_FACTOR
    noise_std: float = 0.0

    @property
    def nash_price(self) -> float:
        """Bertrand-Nash equilibrium price for symmetric oligopoly.

        For a symmetric linear-demand Bertrand game with *n* firms the
        Nash equilibrium price is:
            p_N = (a + n*c) / (n + 1)
        where a = demand_intercept / demand_slope, c = marginal_cost.
        When n → ∞ this converges to marginal cost.
        """
        a = self.demand_intercept / self.demand_slope
        n = self.num_players
        return (a + n * self.marginal_cost) / (n + 1)

    @property
    def monopoly_price(self) -> float:
        """Joint-profit-maximising (monopoly/cartel) price.

        p_M = (a + c) / 2  where a = demand_intercept / demand_slope.
        """
        a = self.demand_intercept / self.demand_slope
        return (a + self.marginal_cost) / 2.0

    @property
    def competitive_profit(self) -> float:
        """Per-firm profit at the Nash equilibrium."""
        p = self.nash_price
        total_q = max(self.demand_intercept - self.demand_slope * p, 0.0)
        q = total_q / self.num_players
        return (p - self.marginal_cost) * q

    @property
    def monopoly_profit(self) -> float:
        """Per-firm profit when all firms set the monopoly price."""
        p = self.monopoly_price
        total_q = max(self.demand_intercept - self.demand_slope * p, 0.0)
        q = total_q / self.num_players
        return (p - self.marginal_cost) * q

    @property
    def action_space(self) -> list:
        """Evenly spaced prices between marginal_cost and 1.2 × monopoly_price."""
        import numpy as np

        low = self.marginal_cost
        high = self.monopoly_price * 1.2
        return np.linspace(low, high, self.num_actions).tolist()


@dataclass
class OracleConfig:
    """Oracle access configuration."""

    access_level: str = "passive"
    checkpoint_interval: int = 1000
    rewind_budget: int = 100
    counterfactual_samples: int = 50


@dataclass
class VisualizationConfig:
    """Visualization settings."""

    output_dir: str = "./output"
    figure_format: str = "png"
    dpi: int = 150
    figsize: tuple = (10, 6)
    style: str = "seaborn-v0_8-whitegrid"
    color_palette: str = "Set2"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
@dataclass
class SystemConfig:
    """Top-level system configuration."""

    test: TestConfig = field(default_factory=TestConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    random_seed: Optional[int] = 42
    verbose: bool = False
    n_jobs: int = 1

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert the full config tree to a plain dict."""
        d = asdict(self)
        # Convert tuple back to list for JSON compatibility
        d["visualization"]["figsize"] = list(d["visualization"]["figsize"])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SystemConfig":
        """Reconstruct a *SystemConfig* from a plain dict."""
        d = dict(d)  # shallow copy so we don't mutate caller's dict

        test_d = d.pop("test", {})
        market_d = d.pop("market", {})
        oracle_d = d.pop("oracle", {})
        vis_d = d.pop("visualization", {})

        if "figsize" in vis_d and isinstance(vis_d["figsize"], list):
            vis_d["figsize"] = tuple(vis_d["figsize"])

        return cls(
            test=TestConfig(**test_d),
            market=MarketConfig(**market_d),
            oracle=OracleConfig(**oracle_d),
            visualization=VisualizationConfig(**vis_d),
            random_seed=d.get("random_seed", 42),
            verbose=d.get("verbose", False),
            n_jobs=d.get("n_jobs", 1),
        )

    def save(self, path: str) -> None:
        """Persist the config to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "SystemConfig":
        """Load a *SystemConfig* from a JSON file."""
        with open(path) as fh:
            d = json.load(fh)
        return cls.from_dict(d)


# ---------------------------------------------------------------------------
# Singleton default config
# ---------------------------------------------------------------------------
_default_config: Optional[SystemConfig] = None


def get_config() -> SystemConfig:
    """Return the current default config, creating one if necessary."""
    global _default_config
    if _default_config is None:
        env_path = os.environ.get("COLLUSION_PROOF_CONFIG")
        if env_path and Path(env_path).exists():
            _default_config = SystemConfig.load(env_path)
        else:
            _default_config = SystemConfig()
    return _default_config


def set_config(config: SystemConfig) -> None:
    """Replace the global default config."""
    global _default_config
    _default_config = config


def load_config(path: str) -> SystemConfig:
    """Load a config from *path* and install it as the global default."""
    config = SystemConfig.load(path)
    set_config(config)
    return config
