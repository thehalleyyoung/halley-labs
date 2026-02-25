"""
Visualization module — text-based automaton and report visualization.

Provides:
    AutomatonVisualizer — ASCII state diagrams, transition tables
    ReportVisualizer    — Bar charts, comparison tables, learning curves
"""

from caber.visualization.automaton_viz import AutomatonVisualizer
from caber.visualization.report_viz import ReportVisualizer

__all__ = [
    "AutomatonVisualizer",
    "ReportVisualizer",
]
