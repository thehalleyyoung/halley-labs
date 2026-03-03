"""
Workload optimization for differential privacy mechanisms.

This package provides algorithms for optimizing query answering strategies
for differential privacy mechanisms, based on the HDMM framework (McKenna
et al., 2018/2021) and extensions for high-dimensional workloads.

Main Components:
    - ``HDMMOptimizer``: Multiplicative weights strategy optimization
    - ``KroneckerStrategy``: Kronecker product strategies for separable workloads
    - ``MarginalOptimizer``: Marginal query selection and optimization
    - ``StrategySelector``: Automatic strategy selection based on workload structure
    - ``CEGISStrategySynthesizer``: Joint mechanism + strategy synthesis via CEGIS

Key Concepts:
    A **strategy matrix** A ∈ R^{d×d} defines which measurements to make.
    The mechanism measures y = Ax + noise, then answers queries via post-processing.
    The goal is to minimize total squared error: E[||W(x̂ - x)||²].

    For workload W and strategy A, the expected squared error is:
        TSE(W, A) = (2/ε²) · trace(W (AᵀA)⁻¹ Wᵀ)

    The HDMM algorithm finds the optimal strategy A via multiplicative weights.

Typical Usage:
    >>> from dp_forge.workload_optimizer import StrategySelector
    >>> from dp_forge.workloads import WorkloadGenerator
    >>> 
    >>> W = WorkloadGenerator.prefix_sums(100)
    >>> selector = StrategySelector()
    >>> strategy = selector.select_strategy(W, epsilon=1.0)
    >>> error = strategy.total_squared_error(W, epsilon=1.0)
"""

from dp_forge.workload_optimizer.hdmm import (
    HDMMOptimizer,
    StrategyMatrix,
    optimize_strategy,
    multiplicative_weights_update,
    frank_wolfe_strategy,
)
from dp_forge.workload_optimizer.kronecker import (
    KroneckerStrategy,
    kronecker_decompose,
    optimize_kronecker,
)
from dp_forge.workload_optimizer.marginal_optimization import (
    MarginalOptimizer,
    greedy_marginal_selection,
    mutual_information_criterion,
)
from dp_forge.workload_optimizer.strategy_selection import (
    StrategySelector,
    WorkloadClassification,
)
from dp_forge.workload_optimizer.cegis_strategy import (
    CEGISStrategySynthesizer,
)

__all__ = [
    "HDMMOptimizer",
    "StrategyMatrix",
    "optimize_strategy",
    "multiplicative_weights_update",
    "frank_wolfe_strategy",
    "KroneckerStrategy",
    "kronecker_decompose",
    "optimize_kronecker",
    "MarginalOptimizer",
    "greedy_marginal_selection",
    "mutual_information_criterion",
    "StrategySelector",
    "WorkloadClassification",
    "CEGISStrategySynthesizer",
]
