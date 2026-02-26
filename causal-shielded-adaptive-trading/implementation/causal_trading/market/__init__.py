"""
Market data module for Causal-Shielded Adaptive Trading.

Provides synthetic market data generation with regime-switching dynamics,
realistic feature engineering pipelines, and look-ahead-bias-free market
replay for backtesting.
"""

from .synthetic import SyntheticMarketGenerator
from .features import FeatureGenerator
from .replay import MarketReplay

__all__ = [
    "SyntheticMarketGenerator",
    "FeatureGenerator",
    "MarketReplay",
]
