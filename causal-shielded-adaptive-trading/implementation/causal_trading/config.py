"""
Configuration management for the Causal-Shielded Adaptive Trading system.

Provides hierarchical dataclass-based configuration with support for
YAML/JSON serialization, validation, merging, and preset profiles.

Example
-------
>>> from causal_trading.config import TradingConfig
>>> cfg = TradingConfig.get_default("conservative")
>>> cfg.regime.K_max
20
>>> cfg.save("my_config.yaml")
>>> loaded = TradingConfig.load("my_config.yaml")
"""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations for constrained configuration choices
# ---------------------------------------------------------------------------

class CITestType(str, Enum):
    """Conditional independence test types."""
    HSIC = "hsic"
    PARTIAL_CORR = "partial_corr"
    KERNEL_CI = "kernel_ci"


class CausalModelType(str, Enum):
    """Structural equation model types."""
    ANM = "anm"
    LINEAR = "linear"
    NONLINEAR_GP = "nonlinear_gp"


class EValueType(str, Enum):
    """E-value construction strategies."""
    PRODUCT = "product"
    MIXTURE = "mixture"
    GROW = "grow"


class CorrectionMethod(str, Enum):
    """Multiple-testing correction methods."""
    BONFERRONI = "bonferroni"
    EBH = "ebh"
    BY = "by"


class SafetySpecType(str, Enum):
    """Shield safety specification types."""
    BOUNDED_DRAWDOWN = "bounded_drawdown"
    POSITION_LIMIT = "position_limit"
    MARGIN = "margin"
    MAX_LOSS = "max_loss"
    TURNOVER = "turnover"


class DiscretizationMethod(str, Enum):
    """State-space discretization methods."""
    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"
    QUANTILE = "quantile"


class PresetProfile(str, Enum):
    """Configuration preset profiles."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    RESEARCH = "research"


# ---------------------------------------------------------------------------
# Sub-configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RegimeConfig:
    """Configuration for Bayesian nonparametric regime detection.

    Parameters
    ----------
    K_max : int
        Maximum number of regimes (truncation level for HDP).
    alpha : float
        Concentration parameter for the top-level DP.
    gamma : float
        Concentration parameter for the base DP.
    kappa : float
        Self-transition bias (stickiness) for the Sticky HDP-HMM.
    n_iterations : int
        Number of Gibbs sampling iterations.
    convergence_threshold : float
        Relative log-likelihood change for early stopping.
    burn_in : int
        Number of burn-in samples to discard.
    online_window : int
        Sliding window length for online regime tracking.
    min_regime_duration : int
        Minimum number of time steps before switching regimes.
    """
    K_max: int = 20
    alpha: float = 1.0
    gamma: float = 1.0
    kappa: float = 50.0
    n_iterations: int = 500
    convergence_threshold: float = 1e-6
    burn_in: int = 100
    online_window: int = 250
    min_regime_duration: int = 5


@dataclass
class CausalConfig:
    """Configuration for causal discovery.

    Parameters
    ----------
    ci_test : str
        Conditional independence test method.
    alpha : float
        Significance level for independence tests.
    max_conditioning_set : int
        Maximum conditioning set size for PC/FCI algorithms.
    model_type : str
        Structural equation model type.
    n_bootstrap : int
        Bootstrap resamples for stability analysis.
    stability_threshold : float
        Edge frequency threshold for the stable PC variant.
    kernel_bandwidth : float
        Bandwidth for HSIC kernel; ``None`` triggers median heuristic.
    max_parents : int
        Maximum number of parents per node in discovered graph.
    """
    ci_test: str = CITestType.HSIC.value
    alpha: float = 0.05
    max_conditioning_set: int = 3
    model_type: str = CausalModelType.ANM.value
    n_bootstrap: int = 100
    stability_threshold: float = 0.75
    kernel_bandwidth: Optional[float] = None
    max_parents: int = 5


@dataclass
class InvarianceConfig:
    """Configuration for anytime-valid invariance testing.

    Parameters
    ----------
    alpha : float
        Significance level for sequential tests.
    e_value_type : str
        E-value construction strategy.
    correction_method : str
        Multiple-testing correction method.
    initial_wealth : float
        Starting wealth for the GROW martingale.
    mixture_components : int
        Number of mixing components for mixture e-values.
    truncation : float
        Truncation threshold for numerical stability.
    min_samples : int
        Minimum observations before a decision can be made.
    power_target : float
        Desired statistical power for sample-size calculations.
    """
    alpha: float = 0.05
    e_value_type: str = EValueType.GROW.value
    correction_method: str = CorrectionMethod.EBH.value
    initial_wealth: float = 1.0
    mixture_components: int = 20
    truncation: float = 1e-10
    min_samples: int = 30
    power_target: float = 0.80


@dataclass
class SafetySpec:
    """A single safety specification for the shield.

    Parameters
    ----------
    spec_type : str
        The kind of safety property.
    params : dict
        Specification-specific parameters (e.g. ``max_drawdown``, ``limit``).
    priority : int
        Priority ordering (lower = higher priority).
    """
    spec_type: str = SafetySpecType.BOUNDED_DRAWDOWN.value
    params: Dict[str, Any] = field(default_factory=lambda: {"max_drawdown": 0.10})
    priority: int = 1


@dataclass
class StateDiscretization:
    """Parameters for state-space discretization.

    Parameters
    ----------
    method : str
        Discretization strategy.
    n_bins_per_dim : int
        Number of bins per continuous dimension.
    adaptive_threshold : float
        Refinement threshold for adaptive methods.
    max_states : int
        Hard cap on total number of discrete states.
    """
    method: str = DiscretizationMethod.UNIFORM.value
    n_bins_per_dim: int = 10
    adaptive_threshold: float = 0.01
    max_states: int = 10_000


@dataclass
class ShieldConfig:
    """Configuration for posterior-predictive shield synthesis.

    Parameters
    ----------
    delta : float
        PAC-Bayes confidence parameter (1 − δ guarantee).
    horizon : int
        Planning/look-ahead horizon (time steps).
    safety_specs : list
        List of :class:`SafetySpec` instances.
    state_discretization : StateDiscretization
        Discretization parameters for the abstract MDP.
    n_posterior_samples : int
        Number of posterior samples for shield construction.
    shield_update_frequency : int
        Re-synthesis interval (time steps).
    composition_method : str
        How multiple safety specs are combined (``intersection`` | ``priority``).
    liveness_check : bool
        Whether to verify shield liveness (non-vacuousness).
    """
    delta: float = 0.05
    horizon: int = 10
    safety_specs: List[SafetySpec] = field(default_factory=lambda: [
        SafetySpec(SafetySpecType.BOUNDED_DRAWDOWN.value, {"max_drawdown": 0.10}, 1),
        SafetySpec(SafetySpecType.POSITION_LIMIT.value, {"max_position": 1.0}, 2),
        SafetySpec(SafetySpecType.MAX_LOSS.value, {"max_loss": 0.05}, 3),
    ])
    state_discretization: StateDiscretization = field(default_factory=StateDiscretization)
    n_posterior_samples: int = 1000
    shield_update_frequency: int = 50
    composition_method: str = "intersection"
    liveness_check: bool = True


@dataclass
class PortfolioConfig:
    """Configuration for portfolio optimization.

    Parameters
    ----------
    risk_aversion : float
        Mean-variance risk aversion coefficient λ.
    max_position : float
        Maximum absolute position size per asset.
    transaction_costs : float
        Proportional transaction cost rate.
    action_levels : int
        Number of discrete action levels.
    rebalance_frequency : int
        Steps between rebalance decisions.
    min_trade_size : float
        Minimum trade size to execute.
    max_turnover : float
        Maximum per-period turnover fraction.
    cash_buffer : float
        Fraction of portfolio kept as cash buffer.
    """
    risk_aversion: float = 1.0
    max_position: float = 1.0
    transaction_costs: float = 0.001
    action_levels: int = 11
    rebalance_frequency: int = 1
    min_trade_size: float = 0.001
    max_turnover: float = 0.5
    cash_buffer: float = 0.02


@dataclass
class MarketConfig:
    """Configuration for market data generation and feature engineering.

    Parameters
    ----------
    n_features : int
        Number of raw market features to generate.
    lasso_alpha : float
        L1 regularization strength for feature selection.
    feature_groups : list
        Named groups of features for structured selection.
    n_assets : int
        Number of tradeable assets.
    frequency : str
        Data frequency (``daily``, ``hourly``, ``minute``).
    regime_params : dict
        Per-regime distribution parameters for synthetic generation.
    lookback_window : int
        Lookback for feature computation.
    """
    n_features: int = 20
    lasso_alpha: float = 0.01
    feature_groups: List[str] = field(default_factory=lambda: [
        "momentum", "volatility", "volume", "fundamental", "sentiment",
    ])
    n_assets: int = 1
    frequency: str = "daily"
    regime_params: Dict[str, Any] = field(default_factory=lambda: {
        "bull": {"mu": 0.08, "sigma": 0.15},
        "bear": {"mu": -0.05, "sigma": 0.30},
        "sideways": {"mu": 0.01, "sigma": 0.10},
    })
    lookback_window: int = 20


@dataclass
class WalkForwardParams:
    """Walk-forward analysis parameters.

    Parameters
    ----------
    train_window : int
        Training window size (time steps).
    test_window : int
        Testing window size (time steps).
    step_size : int
        Step size between successive folds.
    min_train_size : int
        Minimum training observations required.
    expanding : bool
        If *True*, use expanding window; otherwise sliding.
    """
    train_window: int = 252
    test_window: int = 63
    step_size: int = 21
    min_train_size: int = 126
    expanding: bool = False


@dataclass
class EvaluationConfig:
    """Configuration for backtesting and evaluation.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap resamples for confidence intervals.
    confidence_level : float
        Confidence level for reported intervals.
    walk_forward : WalkForwardParams
        Walk-forward split parameters.
    benchmark : str
        Benchmark strategy for comparison (``buy_and_hold``, ``equal_weight``).
    risk_free_rate : float
        Annualized risk-free rate for Sharpe ratio.
    metrics : list
        List of metric names to compute.
    """
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    walk_forward: WalkForwardParams = field(default_factory=WalkForwardParams)
    benchmark: str = "buy_and_hold"
    risk_free_rate: float = 0.02
    metrics: List[str] = field(default_factory=lambda: [
        "sharpe", "sortino", "max_drawdown", "calmar",
        "hit_rate", "profit_factor", "avg_trade_return",
        "shield_intervention_rate", "regime_accuracy",
    ])


@dataclass
class AlertThresholds:
    """Thresholds for monitoring alerts.

    Parameters
    ----------
    regime_change_prob : float
        Posterior probability threshold for regime-change alerts.
    causal_edge_change : float
        Stability threshold change for causal graph alerts.
    shield_violation_rate : float
        Maximum tolerated shield violation rate before alerting.
    anomaly_score : float
        Z-score threshold for the anomaly detector.
    drawdown : float
        Drawdown percentage that triggers an alert.
    """
    regime_change_prob: float = 0.8
    causal_edge_change: float = 0.3
    shield_violation_rate: float = 0.01
    anomaly_score: float = 3.0
    drawdown: float = 0.05


@dataclass
class MonitoringConfig:
    """Configuration for real-time monitoring.

    Parameters
    ----------
    alert_thresholds : AlertThresholds
        Thresholds for monitoring alerts.
    update_frequency : int
        Dashboard refresh interval in seconds.
    log_level : str
        Logging level for monitor output.
    history_length : int
        Number of historical steps to keep in memory.
    enable_alerts : bool
        Whether to emit alerts on threshold breaches.
    export_format : str
        Export format for monitoring snapshots (``json``, ``csv``).
    """
    alert_thresholds: AlertThresholds = field(default_factory=AlertThresholds)
    update_frequency: int = 5
    log_level: str = "INFO"
    history_length: int = 1000
    enable_alerts: bool = True
    export_format: str = "json"


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------

@dataclass
class TradingConfig:
    """Top-level configuration for the full CSAT pipeline.

    Aggregates all module-specific configurations and provides
    convenience methods for loading, saving, validation, and merging.

    Parameters
    ----------
    regime : RegimeConfig
        Regime detection configuration.
    causal : CausalConfig
        Causal discovery configuration.
    invariance : InvarianceConfig
        Invariance testing configuration.
    shield : ShieldConfig
        Shield synthesis configuration.
    portfolio : PortfolioConfig
        Portfolio optimization configuration.
    market : MarketConfig
        Market data / feature configuration.
    evaluation : EvaluationConfig
        Backtesting and evaluation configuration.
    monitoring : MonitoringConfig
        Real-time monitoring configuration.
    seed : int or None
        Global random seed for reproducibility.
    n_jobs : int
        Number of parallel workers (``-1`` = all CPUs).
    verbose : bool
        Enable verbose logging across all modules.
    output_dir : str
        Base directory for pipeline outputs.
    """

    regime: RegimeConfig = field(default_factory=RegimeConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    invariance: InvarianceConfig = field(default_factory=InvarianceConfig)
    shield: ShieldConfig = field(default_factory=ShieldConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    seed: Optional[int] = 42
    n_jobs: int = 1
    verbose: bool = False
    output_dir: str = "output"

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert the config tree to a plain dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: Union[str, Path]) -> None:
        """Persist the configuration to a YAML or JSON file.

        The format is inferred from the file extension (``.yaml``, ``.yml``,
        or ``.json``).

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                with open(path, "w") as fh:
                    yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
            except ImportError:
                # Fallback: write as JSON with .yaml extension
                logger.warning("PyYAML not installed; writing JSON to %s", path)
                with open(path, "w") as fh:
                    json.dump(data, fh, indent=2, default=str)
        else:
            with open(path, "w") as fh:
                json.dump(data, fh, indent=2, default=str)

        logger.info("Configuration saved to %s", path)

    # ------------------------------------------------------------------ #
    # Deserialization
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingConfig":
        """Reconstruct a :class:`TradingConfig` from a plain dictionary.

        Nested dictionaries are converted to the appropriate dataclass
        instances.  Unknown keys are silently ignored so that forward-
        compatible config files do not break older code.
        """
        def _build(dc_cls, raw: Any):
            if not isinstance(raw, dict):
                return raw
            kwargs = {}
            field_map = {f.name: f for f in fields(dc_cls)}
            for name, fld in field_map.items():
                if name not in raw:
                    continue
                val = raw[name]
                origin = getattr(fld.type, "__origin__", None)
                # Check if the field type is itself a dataclass
                ftype = fld.type
                if isinstance(ftype, str):
                    ftype = _resolve_type(ftype)
                if _is_dataclass_type(ftype):
                    kwargs[name] = _build(ftype, val)
                elif origin is list and isinstance(val, list):
                    # Handle List[SafetySpec], etc.
                    inner = _get_list_inner(fld.type)
                    if inner is not None and _is_dataclass_type(inner):
                        kwargs[name] = [_build(inner, v) for v in val]
                    else:
                        kwargs[name] = val
                else:
                    kwargs[name] = val
            return dc_cls(**kwargs)

        return _build(cls, data)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TradingConfig":
        """Load configuration from a YAML or JSON file.

        Parameters
        ----------
        path : str or Path
            Source file path.

        Returns
        -------
        TradingConfig
            Loaded and validated configuration.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file cannot be parsed or fails validation.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as fh:
            raw = fh.read()

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                data = yaml.safe_load(raw)
            except ImportError:
                # Attempt JSON parse as fallback
                data = json.loads(raw)
        else:
            data = json.loads(raw)

        config = cls.from_dict(data)
        config.validate()
        logger.info("Configuration loaded from %s", path)
        return config

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> List[str]:
        """Validate configuration values.

        Returns
        -------
        list of str
            Warning messages for non-critical issues.

        Raises
        ------
        ValueError
            If any value is invalid.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Regime
        if self.regime.K_max < 2:
            errors.append("regime.K_max must be >= 2")
        if self.regime.alpha <= 0:
            errors.append("regime.alpha must be positive")
        if self.regime.gamma <= 0:
            errors.append("regime.gamma must be positive")
        if self.regime.kappa < 0:
            errors.append("regime.kappa must be non-negative")
        if self.regime.n_iterations < 1:
            errors.append("regime.n_iterations must be >= 1")
        if self.regime.burn_in >= self.regime.n_iterations:
            errors.append("regime.burn_in must be < n_iterations")
        if self.regime.convergence_threshold <= 0:
            errors.append("regime.convergence_threshold must be positive")

        # Causal
        if self.causal.ci_test not in {e.value for e in CITestType}:
            errors.append(
                f"causal.ci_test must be one of {[e.value for e in CITestType]}"
            )
        if not (0 < self.causal.alpha < 1):
            errors.append("causal.alpha must be in (0, 1)")
        if self.causal.max_conditioning_set < 0:
            errors.append("causal.max_conditioning_set must be non-negative")
        if self.causal.model_type not in {e.value for e in CausalModelType}:
            errors.append(
                f"causal.model_type must be one of {[e.value for e in CausalModelType]}"
            )

        # Invariance
        if not (0 < self.invariance.alpha < 1):
            errors.append("invariance.alpha must be in (0, 1)")
        if self.invariance.e_value_type not in {e.value for e in EValueType}:
            errors.append(
                f"invariance.e_value_type must be one of {[e.value for e in EValueType]}"
            )
        if self.invariance.initial_wealth <= 0:
            errors.append("invariance.initial_wealth must be positive")

        # Shield
        if not (0 < self.shield.delta < 1):
            errors.append("shield.delta must be in (0, 1)")
        if self.shield.horizon < 1:
            errors.append("shield.horizon must be >= 1")
        if self.shield.n_posterior_samples < 10:
            errors.append("shield.n_posterior_samples must be >= 10")
        if not self.shield.safety_specs:
            warnings.append("shield.safety_specs is empty; shield will be vacuous")

        # Portfolio
        if self.portfolio.risk_aversion <= 0:
            errors.append("portfolio.risk_aversion must be positive")
        if self.portfolio.max_position <= 0:
            errors.append("portfolio.max_position must be positive")
        if self.portfolio.transaction_costs < 0:
            errors.append("portfolio.transaction_costs must be non-negative")
        if self.portfolio.action_levels < 3:
            errors.append("portfolio.action_levels must be >= 3")

        # Market
        if self.market.n_features < 1:
            errors.append("market.n_features must be >= 1")
        if self.market.lasso_alpha < 0:
            errors.append("market.lasso_alpha must be non-negative")

        # Evaluation
        if self.evaluation.n_bootstrap < 1:
            errors.append("evaluation.n_bootstrap must be >= 1")
        if not (0 < self.evaluation.confidence_level < 1):
            errors.append("evaluation.confidence_level must be in (0, 1)")

        # Cross-module consistency
        if self.portfolio.max_position > 1.0 and self.shield.liveness_check:
            warnings.append(
                "Large max_position with liveness checking may cause "
                "frequent shield interventions"
            )
        if (
            self.evaluation.walk_forward.train_window
            < self.regime.online_window
        ):
            warnings.append(
                "walk_forward.train_window < regime.online_window; "
                "regime detector may not have enough data"
            )

        if errors:
            raise ValueError(
                "Configuration validation failed:\n  - "
                + "\n  - ".join(errors)
            )

        for w in warnings:
            logger.warning("Config warning: %s", w)

        return warnings

    # ------------------------------------------------------------------ #
    # Merging
    # ------------------------------------------------------------------ #

    def merge(self, other: "TradingConfig") -> "TradingConfig":
        """Create a new config by merging *other* on top of *self*.

        Non-default values in *other* overwrite those in *self*.

        Parameters
        ----------
        other : TradingConfig
            Configuration whose non-default values take precedence.

        Returns
        -------
        TradingConfig
            Merged configuration (new object).
        """
        base = self.to_dict()
        overlay = other.to_dict()
        merged = _deep_merge(base, overlay)
        return TradingConfig.from_dict(merged)

    # ------------------------------------------------------------------ #
    # Presets
    # ------------------------------------------------------------------ #

    @classmethod
    def get_default(cls, profile: str = "conservative") -> "TradingConfig":
        """Return a preset configuration for a named profile.

        Parameters
        ----------
        profile : str
            One of ``conservative``, ``moderate``, ``aggressive``, ``research``.

        Returns
        -------
        TradingConfig
        """
        profile = profile.lower()
        if profile == "conservative":
            return cls._conservative()
        elif profile == "moderate":
            return cls._moderate()
        elif profile == "aggressive":
            return cls._aggressive()
        elif profile == "research":
            return cls._research()
        else:
            raise ValueError(
                f"Unknown profile '{profile}'. Choose from: "
                f"{[p.value for p in PresetProfile]}"
            )

    @classmethod
    def _conservative(cls) -> "TradingConfig":
        """Conservative defaults: tighter safety, lower risk."""
        return cls(
            regime=RegimeConfig(K_max=20, kappa=100.0, n_iterations=800),
            causal=CausalConfig(alpha=0.01, max_conditioning_set=2),
            invariance=InvarianceConfig(alpha=0.01),
            shield=ShieldConfig(
                delta=0.01,
                horizon=15,
                n_posterior_samples=2000,
                safety_specs=[
                    SafetySpec(SafetySpecType.BOUNDED_DRAWDOWN.value,
                               {"max_drawdown": 0.05}, 1),
                    SafetySpec(SafetySpecType.POSITION_LIMIT.value,
                               {"max_position": 0.5}, 2),
                    SafetySpec(SafetySpecType.MAX_LOSS.value,
                               {"max_loss": 0.02}, 3),
                    SafetySpec(SafetySpecType.TURNOVER.value,
                               {"max_turnover": 0.3}, 4),
                ],
            ),
            portfolio=PortfolioConfig(
                risk_aversion=2.0,
                max_position=0.5,
                transaction_costs=0.002,
                max_turnover=0.3,
            ),
            evaluation=EvaluationConfig(n_bootstrap=2000),
            seed=42,
        )

    @classmethod
    def _moderate(cls) -> "TradingConfig":
        """Moderate defaults: balanced safety and performance."""
        return cls(
            regime=RegimeConfig(K_max=20, kappa=50.0),
            shield=ShieldConfig(delta=0.05, horizon=10),
            portfolio=PortfolioConfig(risk_aversion=1.0, max_position=1.0),
            seed=42,
        )

    @classmethod
    def _aggressive(cls) -> "TradingConfig":
        """Aggressive defaults: looser safety, higher risk appetite."""
        return cls(
            regime=RegimeConfig(K_max=30, kappa=20.0, n_iterations=300),
            causal=CausalConfig(alpha=0.10, max_conditioning_set=4),
            invariance=InvarianceConfig(alpha=0.10),
            shield=ShieldConfig(
                delta=0.10,
                horizon=5,
                n_posterior_samples=500,
                safety_specs=[
                    SafetySpec(SafetySpecType.BOUNDED_DRAWDOWN.value,
                               {"max_drawdown": 0.20}, 1),
                    SafetySpec(SafetySpecType.POSITION_LIMIT.value,
                               {"max_position": 2.0}, 2),
                ],
                liveness_check=False,
            ),
            portfolio=PortfolioConfig(
                risk_aversion=0.5,
                max_position=2.0,
                transaction_costs=0.0005,
                max_turnover=0.8,
            ),
            evaluation=EvaluationConfig(n_bootstrap=500),
            seed=42,
        )

    @classmethod
    def _research(cls) -> "TradingConfig":
        """Research defaults: maximum detail, extra diagnostics."""
        return cls(
            regime=RegimeConfig(
                K_max=40, n_iterations=2000, burn_in=500,
                convergence_threshold=1e-8,
            ),
            causal=CausalConfig(
                n_bootstrap=500,
                stability_threshold=0.5,
                max_conditioning_set=5,
            ),
            invariance=InvarianceConfig(
                mixture_components=50,
                power_target=0.95,
            ),
            shield=ShieldConfig(
                delta=0.01,
                horizon=20,
                n_posterior_samples=5000,
                state_discretization=StateDiscretization(
                    method=DiscretizationMethod.ADAPTIVE.value,
                    n_bins_per_dim=20,
                    max_states=100_000,
                ),
            ),
            evaluation=EvaluationConfig(
                n_bootstrap=5000,
                confidence_level=0.99,
                walk_forward=WalkForwardParams(
                    train_window=504, test_window=126, step_size=21,
                    expanding=True,
                ),
            ),
            monitoring=MonitoringConfig(
                update_frequency=1,
                history_length=10_000,
            ),
            verbose=True,
            seed=None,
            n_jobs=-1,
        )

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        """Return a human-readable summary of key configuration values."""
        lines = [
            "=== Causal-Shielded Adaptive Trading Configuration ===",
            f"  Profile seed     : {self.seed}",
            f"  Output directory : {self.output_dir}",
            f"  Parallel workers : {self.n_jobs}",
            "",
            "  [Regime]",
            f"    K_max={self.regime.K_max}  α={self.regime.alpha}  "
            f"γ={self.regime.gamma}  κ={self.regime.kappa}",
            f"    iterations={self.regime.n_iterations}  "
            f"burn_in={self.regime.burn_in}",
            "",
            "  [Causal]",
            f"    ci_test={self.causal.ci_test}  α={self.causal.alpha}  "
            f"model={self.causal.model_type}",
            f"    max_cond_set={self.causal.max_conditioning_set}  "
            f"bootstrap={self.causal.n_bootstrap}",
            "",
            "  [Invariance]",
            f"    α={self.invariance.alpha}  "
            f"e_value={self.invariance.e_value_type}  "
            f"correction={self.invariance.correction_method}",
            "",
            "  [Shield]",
            f"    δ={self.shield.delta}  horizon={self.shield.horizon}  "
            f"samples={self.shield.n_posterior_samples}",
            f"    specs={len(self.shield.safety_specs)}  "
            f"liveness={self.shield.liveness_check}",
            "",
            "  [Portfolio]",
            f"    risk_aversion={self.portfolio.risk_aversion}  "
            f"max_pos={self.portfolio.max_position}  "
            f"costs={self.portfolio.transaction_costs}",
            "",
            "  [Evaluation]",
            f"    bootstrap={self.evaluation.n_bootstrap}  "
            f"CI={self.evaluation.confidence_level}  "
            f"benchmark={self.evaluation.benchmark}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"TradingConfig(seed={self.seed}, "
            f"K_max={self.regime.K_max}, "
            f"delta={self.shield.delta}, "
            f"risk_aversion={self.portfolio.risk_aversion})"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge *overlay* into *base*, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _is_dataclass_type(tp) -> bool:
    """Check if *tp* is a dataclass type (not instance)."""
    import dataclasses
    try:
        return dataclasses.is_dataclass(tp) and isinstance(tp, type)
    except TypeError:
        return False


def _get_list_inner(type_hint):
    """Extract the inner type from ``List[X]`` if it is a dataclass."""
    args = getattr(type_hint, "__args__", None)
    if args and len(args) == 1:
        return args[0]
    return None


_TYPE_REGISTRY = {
    "RegimeConfig": RegimeConfig,
    "CausalConfig": CausalConfig,
    "InvarianceConfig": InvarianceConfig,
    "ShieldConfig": ShieldConfig,
    "SafetySpec": SafetySpec,
    "StateDiscretization": StateDiscretization,
    "PortfolioConfig": PortfolioConfig,
    "MarketConfig": MarketConfig,
    "EvaluationConfig": EvaluationConfig,
    "WalkForwardParams": WalkForwardParams,
    "MonitoringConfig": MonitoringConfig,
    "AlertThresholds": AlertThresholds,
}


def _resolve_type(name: str):
    """Resolve a forward-reference string to an actual type."""
    return _TYPE_REGISTRY.get(name)
