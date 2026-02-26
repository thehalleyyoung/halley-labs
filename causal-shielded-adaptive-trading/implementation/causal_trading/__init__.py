"""
Causal-Shielded Adaptive Trading (CSAT)
========================================

A framework for regime-aware causal trading with formal safety guarantees.

The system integrates:
    - Bayesian nonparametric regime detection (Sticky HDP-HMM)
    - Causal discovery with anytime-valid invariance testing
    - Shield synthesis with PAC-Bayes safety certificates
    - Portfolio optimization under shield constraints

Architecture
------------
The pipeline flows as:

    Market Data → Regime Detection → Causal Discovery → Shield Synthesis
                                                              ↓
    Monitoring ← Portfolio Optimization ← Action Selection ← Shield

Modules
-------
regime
    Sticky HDP-HMM regime detection and online tracking.
causal
    Structural causal models and constraint-based discovery.
invariance
    E-value sequential testing and anytime-valid inference.
coupled
    Joint regime-causal EM-style inference.
shield
    Posterior predictive shields with PAC-Bayes bounds.
verification
    Symbolic model checking and credible set polytopes.
portfolio
    Shielded mean-variance optimization.
market
    Synthetic data generation and market replay.
monitoring
    Real-time regime, causal graph, and shield monitors.
proofs
    Formal PAC-Bayes bound computation and certificates.
evaluation
    Backtesting, walk-forward analysis, and statistical testing.
"""

__version__ = "0.1.0"
__author__ = "CSAT Research Team"
__license__ = "MIT"

# Regime detection
from causal_trading.regime import (
    StickyHDPHMM,
    BayesianRegimeDetector,
    TransitionMatrixEstimator,
    OnlineRegimeTracker,
    RegimePosterior,
)

# Causal discovery
from causal_trading.causal import (
    StructuralCausalModel,
    PCAlgorithm,
    FCIAlgorithm,
    StablePCAlgorithm,
    HSIC,
    AdditiveNoiseModel,
)

# Invariance testing
from causal_trading.invariance import (
    EValueConstructor,
    GROWMartingale,
    ConfidenceSequence,
    SCITAlgorithm,
    AnytimeInference,
)

# Coupled inference
from causal_trading.coupled import (
    CoupledInference,
    ConvergenceAnalyzer,
    IdentifiabilityAnalyzer,
    JointPosterior,
)

# Shield synthesis
from causal_trading.shield import (
    PosteriorPredictiveShield,
    SafetySpecification,
    BoundedDrawdownSpec,
    PACBayesBound,
    ShieldSoundnessCertificate,
    ShieldLiveness,
)

# Verification
from causal_trading.verification import (
    CredibleSetPolytope,
    SymbolicModelChecker,
    PTIMEVerifier,
)

# Portfolio optimization
from causal_trading.portfolio import (
    ShieldedMeanVarianceOptimizer,
    CausalFeatureSelector,
    ActionSpace,
)

# Market simulation
from causal_trading.market import (
    SyntheticMarketGenerator,
    FeatureGenerator,
    MarketReplay,
)

# Monitoring
from causal_trading.monitoring import (
    RegimeMonitor,
    CausalGraphMonitor,
    ShieldMonitor,
    AnomalyDetector,
)

# Proofs and certificates
from causal_trading.proofs import (
    PACBayesBoundComputer,
    ShieldSoundnessVerifier,
    CompositionChecker,
    Certificate,
)

# Evaluation
from causal_trading.evaluation import (
    BacktestEngine,
    BacktestConfig,
    WalkForwardAnalyzer,
    RegimeAccuracyEvaluator,
    CausalAccuracyEvaluator,
    ShieldMetricsEvaluator,
    StatisticalTestSuite,
)

# Configuration
from causal_trading.config import TradingConfig


def get_version() -> str:
    """Return the package version string."""
    return __version__


def create_default_pipeline(config: "TradingConfig | None" = None):
    """Convenience factory: build a full CSAT pipeline from config.

    Parameters
    ----------
    config : TradingConfig, optional
        Configuration for all pipeline components.  If *None*, the
        conservative default is used.

    Returns
    -------
    dict
        Mapping of component name to instantiated object, ready to run.
    """
    if config is None:
        config = TradingConfig.get_default("conservative")

    regime_detector = BayesianRegimeDetector(
        K_max=config.regime.K_max,
        alpha=config.regime.alpha,
        gamma=config.regime.gamma,
        kappa=config.regime.kappa,
    )
    causal_discoverer = PCAlgorithm(
        ci_test=config.causal.ci_test,
        alpha=config.causal.alpha,
    )
    shield = PosteriorPredictiveShield(delta=config.shield.delta)
    optimizer = ShieldedMeanVarianceOptimizer(
        risk_aversion=config.portfolio.risk_aversion,
        max_position=config.portfolio.max_position,
    )
    market_gen = SyntheticMarketGenerator(n_features=config.market.n_features)

    return {
        "regime_detector": regime_detector,
        "causal_discoverer": causal_discoverer,
        "shield": shield,
        "optimizer": optimizer,
        "market_generator": market_gen,
        "config": config,
    }


__all__ = [
    # Version
    "__version__",
    # Regime
    "StickyHDPHMM",
    "BayesianRegimeDetector",
    "TransitionMatrixEstimator",
    "OnlineRegimeTracker",
    "RegimePosterior",
    # Causal
    "StructuralCausalModel",
    "PCAlgorithm",
    "FCIAlgorithm",
    "StablePCAlgorithm",
    "HSIC",
    "AdditiveNoiseModel",
    # Invariance
    "EValueConstructor",
    "GROWMartingale",
    "ConfidenceSequence",
    "SCITAlgorithm",
    "AnytimeInference",
    # Coupled
    "CoupledInference",
    "ConvergenceAnalyzer",
    "IdentifiabilityAnalyzer",
    "JointPosterior",
    # Shield
    "PosteriorPredictiveShield",
    "SafetySpecification",
    "BoundedDrawdownSpec",
    "PACBayesBound",
    "ShieldSoundnessCertificate",
    "ShieldLiveness",
    # Verification
    "CredibleSetPolytope",
    "SymbolicModelChecker",
    "PTIMEVerifier",
    # Portfolio
    "ShieldedMeanVarianceOptimizer",
    "CausalFeatureSelector",
    "ActionSpace",
    # Market
    "SyntheticMarketGenerator",
    "FeatureGenerator",
    "MarketReplay",
    # Monitoring
    "RegimeMonitor",
    "CausalGraphMonitor",
    "ShieldMonitor",
    "AnomalyDetector",
    # Proofs
    "PACBayesBoundComputer",
    "ShieldSoundnessVerifier",
    "CompositionChecker",
    "Certificate",
    # Evaluation
    "BacktestEngine",
    "BacktestConfig",
    "WalkForwardAnalyzer",
    "RegimeAccuracyEvaluator",
    "CausalAccuracyEvaluator",
    "ShieldMetricsEvaluator",
    "StatisticalTestSuite",
    # Config
    "TradingConfig",
    # Convenience
    "get_version",
    "create_default_pipeline",
]
