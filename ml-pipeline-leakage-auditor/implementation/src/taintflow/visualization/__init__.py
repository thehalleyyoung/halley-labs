"""
taintflow.visualization -- Visual representations of leakage analysis.

This package provides SVG- and ASCII-based visualization utilities for
rendering pipeline DAGs, leakage heatmaps, and report charts.  All
rendering is done with the standard library only—no matplotlib, plotly,
or other external visualization dependencies are required.

Public API
----------
* :class:`DAGVisualizer`   – SVG / ASCII / DOT rendering of PI-DAGs.
* :class:`LeakageHeatmap`  – Feature × stage leakage heatmaps.
* :class:`ChartGenerator`  – Bar, pie, line, treemap charts for reports.
"""

from __future__ import annotations

from taintflow.visualization.dag_viz import (
    DAGVisualizer,
    EdgeRenderer,
    LayoutEngine,
    NodeRenderer,
    SVGCanvas,
    SVGElement,
    SugiyamaLayout,
)
from taintflow.visualization.leakage_heatmap import (
    ASCIIHeatmap,
    AggregationViews,
    ColorMapper,
    HTMLHeatmap,
    LeakageHeatmap,
)
from taintflow.visualization.report_charts import (
    Annotation,
    Axis,
    BarChart,
    ChartGenerator,
    ChartLegend,
    ChartTheme,
    DataSeries,
    LineChart,
    PieChart,
    ResponsiveLayout,
    StackedBarChart,
    TreemapChart,
)

__all__: list[str] = [
    "DAGVisualizer",
    "SVGElement",
    "SVGCanvas",
    "LayoutEngine",
    "SugiyamaLayout",
    "NodeRenderer",
    "EdgeRenderer",
    "LeakageHeatmap",
    "ColorMapper",
    "HTMLHeatmap",
    "ASCIIHeatmap",
    "AggregationViews",
    "ChartGenerator",
    "BarChart",
    "StackedBarChart",
    "PieChart",
    "LineChart",
    "TreemapChart",
    "ChartTheme",
    "Axis",
    "DataSeries",
    "ChartLegend",
    "Annotation",
    "ResponsiveLayout",
]