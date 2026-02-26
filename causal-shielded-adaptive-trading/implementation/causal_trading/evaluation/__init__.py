"""
Evaluation module for Causal-Shielded Adaptive Trading.

Provides comprehensive backtesting, walk-forward analysis, regime detection
accuracy, causal discovery accuracy, shield safety metrics, and statistical
testing infrastructure.
"""

from causal_trading.evaluation.backtest import (
    BacktestEngine,
    BacktestConfig,
    TradeRecord,
    BacktestResults,
)
from causal_trading.evaluation.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WalkForwardSplit,
)
from causal_trading.evaluation.regime_accuracy import (
    RegimeAccuracyEvaluator,
    RegimeAccuracyMetrics,
)
from causal_trading.evaluation.causal_accuracy import (
    CausalAccuracyEvaluator,
    CausalAccuracyMetrics,
)
from causal_trading.evaluation.shield_metrics import (
    ShieldMetricsEvaluator,
    ShieldSafetyMetrics,
    ShieldPermissivityMetrics,
)
from causal_trading.evaluation.statistical_tests import (
    StatisticalTestSuite,
    BootstrapResult,
    CorrectedPValues,
)
from causal_trading.evaluation.sensitivity_analysis import (
    SensitivityAnalyzer,
    SensitivityReport,
)
from causal_trading.evaluation.error_decomposition import (
    ErrorDecompositionExperiment,
)

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "TradeRecord",
    "BacktestResults",
    "WalkForwardAnalyzer",
    "WalkForwardConfig",
    "WalkForwardSplit",
    "RegimeAccuracyEvaluator",
    "RegimeAccuracyMetrics",
    "CausalAccuracyEvaluator",
    "CausalAccuracyMetrics",
    "ShieldMetricsEvaluator",
    "ShieldSafetyMetrics",
    "ShieldPermissivityMetrics",
    "StatisticalTestSuite",
    "BootstrapResult",
    "CorrectedPValues",
    "SensitivityAnalyzer",
    "SensitivityReport",
    "ErrorDecompositionExperiment",
]
