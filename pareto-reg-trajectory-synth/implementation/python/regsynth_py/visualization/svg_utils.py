"""SVG generation utilities for RegSynth visualizations.

Pure-Python SVG building: no external dependencies required.
"""

from __future__ import annotations

import math
from typing import Optional

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

PALETTE: list[str] = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#af7aa1", "#86bcb6", "#d37295", "#6b6ecf", "#8ca252",
]

RISK_COLORS: dict[str, str] = {
    "critical": "#d62728",
    "high": "#ff7f0e",
    "medium": "#ffbb33",
    "low": "#2ca02c",
}

FRAMEWORK_COLORS: dict[str, str] = {
    "GDPR": "#4e79a7",
    "CCPA": "#f28e2b",
    "HIPAA": "#e15759",
    "SOX": "#76b7b2",
    "PCI-DSS": "#59a14f",
    "NIST": "#edc948",
    "ISO27001": "#b07aa1",
    "FISMA": "#ff9da7",
    "GLBA": "#9c755f",
    "FERPA": "#bab0ac",
    "SOC2": "#6b6ecf",
    "DORA": "#d37295",
    "AI-Act": "#8ca252",
    "NIS2": "#86bcb6",
}


# ---------------------------------------------------------------------------
# SVGElement
# ---------------------------------------------------------------------------

class SVGElement:
    """Lightweight DOM-like node for building SVG documents."""

    def __init__(
        self,
        tag: str,
        attrs: Optional[dict] = None,
        children: Optional[list["SVGElement"]] = None,
        text: Optional[str] = None,
    ) -> None:
        self.tag = tag
        self.attrs: dict[str, str] = {k: str(v) for k, v in (attrs or {}).items()}
        self.children: list[SVGElement] = list(children or [])
        self.text = text

    def add_child(self, child: "SVGElement") -> "SVGElement":
        self.children.append(child)
        return child

    def set_attr(self, key: str, value) -> None:
        self.attrs[key] = str(value)

    def render(self, indent: int = 0) -> str:
        pad = "  " * indent
        attr_str = "".join(f' {k}="{_escape_xml(v)}"' for k, v in self.attrs.items())
        if not self.children and not self.text:
            return f"{pad}<{self.tag}{attr_str}/>"
        parts = [f"{pad}<{self.tag}{attr_str}>"]
        if self.text is not None:
            parts.append(f"{pad}  {_escape_xml(self.text)}")
        for child in self.children:
            parts.append(child.render(indent + 1))
        parts.append(f"{pad}</{self.tag}>")
        return "\n".join(parts)


def _escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# SVG element factory helpers
# ---------------------------------------------------------------------------

def svg_document(width: int, height: int, viewbox: Optional[str] = None) -> SVGElement:
    attrs: dict = {
        "xmlns": "http://www.w3.org/2000/svg",
        "width": str(width),
        "height": str(height),
    }
    if viewbox:
        attrs["viewBox"] = viewbox
    else:
        attrs["viewBox"] = f"0 0 {width} {height}"
    return SVGElement("svg", attrs)


def svg_rect(
    x: float, y: float, width: float, height: float,
    fill: str, stroke: Optional[str] = None, rx: float = 0,
    opacity: float = 1.0,
) -> SVGElement:
    attrs: dict = {"x": x, "y": y, "width": width, "height": height, "fill": fill}
    if stroke:
        attrs["stroke"] = stroke
    if rx:
        attrs["rx"] = rx
    if opacity < 1.0:
        attrs["opacity"] = opacity
    return SVGElement("rect", attrs)


def svg_circle(
    cx: float, cy: float, r: float, fill: str,
    stroke: Optional[str] = None, opacity: float = 1.0,
) -> SVGElement:
    attrs: dict = {"cx": cx, "cy": cy, "r": r, "fill": fill}
    if stroke:
        attrs["stroke"] = stroke
    if opacity < 1.0:
        attrs["opacity"] = opacity
    return SVGElement("circle", attrs)


def svg_line(
    x1: float, y1: float, x2: float, y2: float,
    stroke: str, stroke_width: float = 1, dash: Optional[str] = None,
) -> SVGElement:
    attrs: dict = {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "stroke": stroke, "stroke-width": stroke_width,
    }
    if dash:
        attrs["stroke-dasharray"] = dash
    return SVGElement("line", attrs)


def svg_polyline(
    points: list[tuple[float, float]], stroke: str,
    fill: str = "none", stroke_width: float = 1,
) -> SVGElement:
    pts = " ".join(f"{x},{y}" for x, y in points)
    return SVGElement("polyline", {
        "points": pts, "stroke": stroke, "fill": fill,
        "stroke-width": stroke_width,
    })


def svg_polygon(
    points: list[tuple[float, float]], fill: str,
    stroke: Optional[str] = None,
) -> SVGElement:
    pts = " ".join(f"{x},{y}" for x, y in points)
    attrs: dict = {"points": pts, "fill": fill}
    if stroke:
        attrs["stroke"] = stroke
    return SVGElement("polygon", attrs)


def svg_text(
    x: float, y: float, text: str, font_size: int = 12,
    fill: str = "black", anchor: str = "start",
    font_family: str = "sans-serif", font_weight: str = "normal",
    rotate: Optional[float] = None,
) -> SVGElement:
    attrs: dict = {
        "x": x, "y": y, "font-size": font_size, "fill": fill,
        "text-anchor": anchor, "font-family": font_family,
        "font-weight": font_weight,
    }
    if rotate is not None:
        attrs["transform"] = f"rotate({rotate},{x},{y})"
    return SVGElement("text", attrs, text=text)


def svg_group(
    transform: Optional[str] = None, opacity: Optional[float] = None,
) -> SVGElement:
    attrs: dict = {}
    if transform:
        attrs["transform"] = transform
    if opacity is not None:
        attrs["opacity"] = opacity
    return SVGElement("g", attrs)


def svg_path(
    d: str, fill: str = "none", stroke: str = "black",
    stroke_width: float = 1,
) -> SVGElement:
    return SVGElement("path", {
        "d": d, "fill": fill, "stroke": stroke,
        "stroke-width": stroke_width,
    })


def svg_title(text: str) -> SVGElement:
    return SVGElement("title", text=text)


def svg_style(css: str) -> SVGElement:
    elem = SVGElement("style")
    elem.text = css
    return elem


def svg_defs(*elements: SVGElement) -> SVGElement:
    defs = SVGElement("defs")
    for el in elements:
        defs.add_child(el)
    return defs


def svg_linear_gradient(
    gid: str, x1: float, y1: float, x2: float, y2: float,
    stops: list[tuple[float, str]],
) -> SVGElement:
    grad = SVGElement("linearGradient", {
        "id": gid, "x1": f"{x1}%", "y1": f"{y1}%",
        "x2": f"{x2}%", "y2": f"{y2}%",
    })
    for offset, color in stops:
        grad.add_child(SVGElement("stop", {
            "offset": f"{offset}%", "stop-color": color,
        }))
    return grad


# ---------------------------------------------------------------------------
# Rendering & math helpers
# ---------------------------------------------------------------------------

def render_svg(element: SVGElement) -> str:
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + element.render()


def scale_linear(
    value: float, domain_min: float, domain_max: float,
    range_min: float, range_max: float,
) -> float:
    if domain_max == domain_min:
        return (range_min + range_max) / 2.0
    t = (value - domain_min) / (domain_max - domain_min)
    return range_min + t * (range_max - range_min)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}"


def color_interpolate(color1: str, color2: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)
    return rgb_to_hex(
        int(r1 + (r2 - r1) * t),
        int(g1 + (g2 - g1) * t),
        int(b1 + (b2 - b1) * t),
    )
