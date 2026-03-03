"""
Causal-Plasticity Atlas — Tipping-Point Detection Module.

Implements Algorithm 4 (ALG4): Tipping-Point Detection via PELT.
Provides changepoint detection in mechanism divergence sequences across
ordered causal contexts, with permutation validation and mechanism-level
attribution.

Main classes:
    PELTDetector        — Full ALG4 tipping-point detection pipeline.
    SegmentAnalyzer     — Between-tipping-point segment characterization.
    TippingPointReport  — Human-readable tipping-point reports.
    PELTSolver          — Reusable PELT dynamic programming engine.
    BinarySegmentation  — Binary segmentation changepoint detector.
    CUSUMDetector       — CUSUM-based changepoint detector.
"""

from __future__ import annotations

from cpa.detection.tipping_points import (
    PELTDetector,
    SegmentAnalyzer,
    TippingPointReport,
    TippingPointResult,
)
from cpa.detection.changepoint import (
    BinarySegmentation,
    CUSUMDetector,
    PELTSolver,
)

__all__ = [
    "PELTDetector",
    "SegmentAnalyzer",
    "TippingPointReport",
    "TippingPointResult",
    "BinarySegmentation",
    "CUSUMDetector",
    "PELTSolver",
]
