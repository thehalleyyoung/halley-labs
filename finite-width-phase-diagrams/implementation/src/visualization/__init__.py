"""Visualization module for finite-width phase diagrams.

Provides plotting utilities for phase diagrams, kernel analysis,
and training dynamics using matplotlib.
"""

from __future__ import annotations

from .phase_plots import PhaseColorMap, PhaseDiagramPlotter, PlotConfig
from .kernel_plots import KernelPlotter
from .training_plots import TrainingPlotter

__all__ = [
    "PlotConfig",
    "PhaseColorMap",
    "PhaseDiagramPlotter",
    "KernelPlotter",
    "TrainingPlotter",
]
