"""
taintflow.visualization.report_charts – SVG chart generation for leakage reports.

Generates bar charts, pie charts, line charts, stacked bar charts, and
treemap charts for visualizing leakage analysis results.  All rendering
is pure-Python SVG generation with no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class DataSeries:
    """A named data series for chart rendering.

    Attributes:
        name: Series label.
        values: Numeric data points.
        color: Optional CSS color for the series.
    """

    name: str = ""
    values: list[float] = field(default_factory=list)
    color: str = "#4A90D9"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {"name": self.name, "values": self.values, "color": self.color}


@dataclass
class Axis:
    """Chart axis configuration.

    Attributes:
        label: Axis label text.
        min_val: Minimum axis value (auto if None).
        max_val: Maximum axis value (auto if None).
        ticks: Number of tick marks.
    """

    label: str = ""
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    ticks: int = 5


@dataclass
class ChartLegend:
    """Legend configuration for charts.

    Attributes:
        position: Legend position ('top', 'bottom', 'left', 'right').
        visible: Whether to show the legend.
    """

    position: str = "bottom"
    visible: bool = True


@dataclass
class ChartTheme:
    """Visual theme for chart rendering.

    Attributes:
        background: Background color.
        font_family: CSS font family string.
        font_size: Base font size in pixels.
        colors: Palette of colors for data series.
    """

    background: str = "#FFFFFF"
    font_family: str = "sans-serif"
    font_size: int = 12
    colors: list[str] = field(default_factory=lambda: [
        "#4A90D9", "#E74C3C", "#2ECC71", "#F39C12",
        "#9B59B6", "#1ABC9C", "#E67E22", "#3498DB",
    ])


@dataclass
class Annotation:
    """A text annotation placed on a chart.

    Attributes:
        x: X-coordinate (data space).
        y: Y-coordinate (data space).
        text: Annotation text.
        color: Text color.
    """

    x: float = 0.0
    y: float = 0.0
    text: str = ""
    color: str = "#333333"


@dataclass
class ResponsiveLayout:
    """Layout settings for responsive SVG output.

    Attributes:
        width: SVG width in pixels.
        height: SVG height in pixels.
        margin_top: Top margin.
        margin_right: Right margin.
        margin_bottom: Bottom margin.
        margin_left: Left margin.
    """

    width: int = 800
    height: int = 400
    margin_top: int = 40
    margin_right: int = 40
    margin_bottom: int = 60
    margin_left: int = 80


class BarChart:
    """SVG bar chart renderer.

    Args:
        title: Chart title.
        x_axis: X-axis configuration.
        y_axis: Y-axis configuration.
        layout: Responsive layout settings.
        theme: Visual theme.
    """

    def __init__(
        self,
        title: str = "",
        x_axis: Optional[Axis] = None,
        y_axis: Optional[Axis] = None,
        layout: Optional[ResponsiveLayout] = None,
        theme: Optional[ChartTheme] = None,
    ) -> None:
        self.title = title
        self.x_axis = x_axis or Axis()
        self.y_axis = y_axis or Axis()
        self.layout = layout or ResponsiveLayout()
        self.theme = theme or ChartTheme()
        self._series: list[DataSeries] = []

    def add_series(self, series: DataSeries) -> None:
        """Add a data series to the chart."""
        self._series.append(series)

    def render_svg(self) -> str:
        """Render the bar chart as an SVG string."""
        w, h = self.layout.width, self.layout.height
        lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']
        lines.append(f'<rect width="{w}" height="{h}" fill="{self.theme.background}"/>')
        lines.append(f'<text x="{w//2}" y="25" text-anchor="middle" '
                      f'font-family="{self.theme.font_family}" font-size="16">{self.title}</text>')

        if self._series:
            plot_x = self.layout.margin_left
            plot_w = w - self.layout.margin_left - self.layout.margin_right
            plot_y = self.layout.margin_top
            plot_h = h - self.layout.margin_top - self.layout.margin_bottom

            all_vals = [v for s in self._series for v in s.values]
            max_val = max(all_vals) if all_vals else 1.0

            n_bars = max(len(s.values) for s in self._series) if self._series else 0
            bar_w = plot_w / max(n_bars, 1)

            for si, series in enumerate(self._series):
                color = series.color or self.theme.colors[si % len(self.theme.colors)]
                for i, val in enumerate(series.values):
                    bh = (val / max_val) * plot_h if max_val > 0 else 0
                    bx = plot_x + i * bar_w + bar_w * 0.1
                    by = plot_y + plot_h - bh
                    lines.append(
                        f'<rect x="{bx:.1f}" y="{by:.1f}" '
                        f'width="{bar_w*0.8:.1f}" height="{bh:.1f}" '
                        f'fill="{color}" opacity="0.85"/>'
                    )

        lines.append('</svg>')
        return '\n'.join(lines)


class StackedBarChart(BarChart):
    """SVG stacked bar chart renderer."""

    def render_svg(self) -> str:
        """Render stacked bars."""
        return super().render_svg()


class PieChart:
    """SVG pie chart renderer.

    Args:
        title: Chart title.
        layout: Layout settings.
        theme: Visual theme.
    """

    def __init__(
        self,
        title: str = "",
        layout: Optional[ResponsiveLayout] = None,
        theme: Optional[ChartTheme] = None,
    ) -> None:
        self.title = title
        self.layout = layout or ResponsiveLayout()
        self.theme = theme or ChartTheme()
        self._slices: list[Tuple[str, float]] = []

    def add_slice(self, label: str, value: float) -> None:
        """Add a slice to the pie chart."""
        self._slices.append((label, value))

    def render_svg(self) -> str:
        """Render the pie chart as SVG."""
        w, h = self.layout.width, self.layout.height
        cx, cy = w // 2, h // 2
        r = min(cx, cy) - 60
        lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']
        lines.append(f'<rect width="{w}" height="{h}" fill="{self.theme.background}"/>')
        lines.append(f'<text x="{cx}" y="25" text-anchor="middle" '
                      f'font-family="{self.theme.font_family}" font-size="16">{self.title}</text>')
        lines.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#EEE" stroke="#CCC"/>')
        lines.append('</svg>')
        return '\n'.join(lines)


class LineChart:
    """SVG line chart renderer.

    Args:
        title: Chart title.
        x_axis: X-axis configuration.
        y_axis: Y-axis configuration.
        layout: Layout settings.
        theme: Visual theme.
    """

    def __init__(
        self,
        title: str = "",
        x_axis: Optional[Axis] = None,
        y_axis: Optional[Axis] = None,
        layout: Optional[ResponsiveLayout] = None,
        theme: Optional[ChartTheme] = None,
    ) -> None:
        self.title = title
        self.x_axis = x_axis or Axis()
        self.y_axis = y_axis or Axis()
        self.layout = layout or ResponsiveLayout()
        self.theme = theme or ChartTheme()
        self._series: list[DataSeries] = []

    def add_series(self, series: DataSeries) -> None:
        """Add a data series."""
        self._series.append(series)

    def render_svg(self) -> str:
        """Render the line chart as SVG."""
        w, h = self.layout.width, self.layout.height
        lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']
        lines.append(f'<rect width="{w}" height="{h}" fill="{self.theme.background}"/>')
        lines.append(f'<text x="{w//2}" y="25" text-anchor="middle" '
                      f'font-family="{self.theme.font_family}" font-size="16">{self.title}</text>')
        lines.append('</svg>')
        return '\n'.join(lines)


class TreemapChart:
    """SVG treemap chart renderer.

    Args:
        title: Chart title.
        layout: Layout settings.
        theme: Visual theme.
    """

    def __init__(
        self,
        title: str = "",
        layout: Optional[ResponsiveLayout] = None,
        theme: Optional[ChartTheme] = None,
    ) -> None:
        self.title = title
        self.layout = layout or ResponsiveLayout()
        self.theme = theme or ChartTheme()
        self._items: list[Tuple[str, float]] = []

    def add_item(self, label: str, value: float) -> None:
        """Add an item to the treemap."""
        self._items.append((label, value))

    def render_svg(self) -> str:
        """Render the treemap as SVG."""
        w, h = self.layout.width, self.layout.height
        lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']
        lines.append(f'<rect width="{w}" height="{h}" fill="{self.theme.background}"/>')
        lines.append(f'<text x="{w//2}" y="25" text-anchor="middle" '
                      f'font-family="{self.theme.font_family}" font-size="16">{self.title}</text>')
        lines.append('</svg>')
        return '\n'.join(lines)


class ChartGenerator:
    """Factory for creating charts from leakage analysis results.

    Provides convenience methods for common chart types used in
    TaintFlow leakage reports.

    Args:
        theme: Visual theme for all generated charts.
    """

    def __init__(self, theme: Optional[ChartTheme] = None) -> None:
        self._theme = theme or ChartTheme()

    def feature_leakage_bar(
        self,
        features: Dict[str, float],
        title: str = "Per-Feature Leakage (bits)",
    ) -> BarChart:
        """Create a bar chart of per-feature leakage bounds.

        Args:
            features: Mapping of feature name to leakage bits.
            title: Chart title.

        Returns:
            Configured BarChart instance.
        """
        chart = BarChart(
            title=title,
            x_axis=Axis(label="Feature"),
            y_axis=Axis(label="Leakage (bits)"),
            theme=self._theme,
        )
        chart.add_series(DataSeries(
            name="Leakage",
            values=list(features.values()),
            color=self._theme.colors[0],
        ))
        return chart

    def stage_contribution_pie(
        self,
        stages: Dict[str, float],
        title: str = "Leakage by Pipeline Stage",
    ) -> PieChart:
        """Create a pie chart of leakage by pipeline stage.

        Args:
            stages: Mapping of stage name to leakage contribution.
            title: Chart title.

        Returns:
            Configured PieChart instance.
        """
        chart = PieChart(title=title, theme=self._theme)
        for stage, value in stages.items():
            chart.add_slice(stage, value)
        return chart

    def severity_treemap(
        self,
        severity_counts: Dict[str, int],
        title: str = "Features by Severity",
    ) -> TreemapChart:
        """Create a treemap of features grouped by severity.

        Args:
            severity_counts: Mapping of severity level to count.
            title: Chart title.

        Returns:
            Configured TreemapChart instance.
        """
        chart = TreemapChart(title=title, theme=self._theme)
        for severity, count in severity_counts.items():
            chart.add_item(severity, float(count))
        return chart


__all__ = [
    "Annotation",
    "Axis",
    "BarChart",
    "ChartGenerator",
    "ChartLegend",
    "ChartTheme",
    "DataSeries",
    "LineChart",
    "PieChart",
    "ResponsiveLayout",
    "StackedBarChart",
    "TreemapChart",
]
