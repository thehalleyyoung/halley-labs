"""Visualization module: SVG-based charts and HTML dashboards.

All visualizations are generated as SVG/HTML with no external dependencies.
"""

from regsynth_py.visualization.pareto_plot import ParetoPlotter
from regsynth_py.visualization.timeline_plot import TimelinePlotter
from regsynth_py.visualization.conflict_graph import ConflictGraphPlotter
from regsynth_py.visualization.dashboard import DashboardGenerator

__all__ = [
    "ParetoPlotter", "TimelinePlotter", "ConflictGraphPlotter", "DashboardGenerator",
]
