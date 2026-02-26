"""Counterexample visualization for BioProver.

Renders concrete counterexample traces as ASCII time-series plots,
phase portraits, text reports, basic SVG circuit diagrams, and
exports data for external plotting tools.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bioprover.cegar.cegar_engine import VerificationResult
from bioprover.cegar.counterexample import ConcreteCounterexample
from bioprover.models.bio_model import BioModel
from bioprover.models.regulatory_network import (
    GeneRegulatoryNetwork,
    InteractionSign,
)
from bioprover.models.species import Species

logger = logging.getLogger(__name__)

_EPS = 1e-12
_MARKER_VIOLATION = "X"
_MARKER_PHASE = "o"
_MARKER_START = "S"
_MARKER_END = "E"
_MARKER_TRACE = "*"
_SVG_NODE_RADIUS = 30
_SVG_FONT_SIZE = 12
_CHANGE_THRESHOLD = 0.1


class TraceFormat(Enum):
    """Supported output formats for counterexample data."""
    ASCII = auto()
    SVG = auto()
    JSON = auto()
    CSV = auto()


@dataclass
class CounterexampleVisualizer:
    """Renders counterexample traces and circuit topology.

    Attributes
    ----------
    width : int
        Character width of ASCII plots.
    height : int
        Character height of ASCII plots.
    """

    width: int = 72
    height: int = 20

    # -- ASCII time-series ---------------------------------------------------

    def plot_trace_ascii(
        self, cex: ConcreteCounterexample, species: Optional[List[str]] = None,
    ) -> str:
        """Plot species concentrations along the counterexample trace.

        Returns a multi-line ASCII chart with Y-axis labels, per-species
        symbol traces, and a violation marker at the final time-point.
        """
        if cex.length == 0:
            return "(empty counterexample)"
        var_names = species or sorted(cex.states[0].keys())
        if not var_names:
            return "(no species data)"

        trajs: Dict[str, List[float]] = {v: cex.trajectory(v) for v in var_names}
        all_v = [x for t in trajs.values() for x in t]
        lo, hi = min(all_v), max(all_v)
        if abs(hi - lo) < _EPS:
            hi = lo + 1.0

        pw, ph = self.width - 10, self.height
        canvas = [[" "] * pw for _ in range(ph)]
        syms = ".*+#@oxs^v"
        legend: List[str] = []

        for idx, var in enumerate(var_names):
            sym = syms[idx % len(syms)]
            legend.append(f"  {sym} = {var}")
            for ti, val in enumerate(trajs[var]):
                c = self._scale_value(ti, 0, max(len(trajs[var]) - 1, 1), pw - 1)
                r = (ph - 1) - self._scale_value(val, lo, hi, ph - 1)
                canvas[r][c] = sym

        # Mark violation at final column
        for var in var_names:
            val = trajs[var][-1]
            r = (ph - 1) - self._scale_value(val, lo, hi, ph - 1)
            canvas[r][pw - 1] = _MARKER_VIOLATION

        lines = [f"Trace: {cex.property_violated}", "=" * self.width]
        for r in range(ph):
            if r == 0:
                yl = self._format_axis_label(hi)
            elif r == ph - 1:
                yl = self._format_axis_label(lo)
            elif r == ph // 2:
                yl = self._format_axis_label((lo + hi) / 2)
            else:
                yl = "        "
            lines.append(f"{yl:>8s} |{''.join(canvas[r])}")

        lines.append(" " * 9 + "+" + "-" * pw)
        t0, t1 = cex.time_points[0], cex.time_points[-1]
        lines.append(" " * 9 + self._format_axis_label(t0)
                      + " " * max(0, pw - 16) + self._format_axis_label(t1))
        lines.append(f"         {'time':^{pw}s}")
        lines.append("")
        lines.append("Legend:")
        lines.extend(legend)
        lines.append(f"  {_MARKER_VIOLATION} = violation point")
        return "\n".join(lines)

    # -- Violation annotation ------------------------------------------------

    def annotate_violation(self, cex: ConcreteCounterexample) -> str:
        """Return a text explanation of where and why the spec fails.

        Shows the violated property, time of violation, and species
        concentrations at the violation point and initial state.
        """
        if cex.length == 0:
            return "No violation data (empty counterexample)."
        final = cex.final_state
        parts: List[str] = [
            f"Property violated: {cex.property_violated}",
            f"Violation time:    t = {cex.time_points[-1]:.6g}",
            f"Trace length:      {cex.length} step(s), duration {cex.duration:.6g}",
            "", "Species at violation:",
        ]
        for n in sorted(final):
            parts.append(f"  {n:>20s} = {final[n]:.6g}")
        parts += ["", "Initial state:"]
        for n in sorted(cex.initial_state):
            parts.append(f"  {n:>20s} = {cex.initial_state[n]:.6g}")
        if cex.parameter_values:
            parts += ["", "Parameters:"]
            for n in sorted(cex.parameter_values):
                parts.append(f"  {n:>20s} = {cex.parameter_values[n]:.6g}")
        return "\n".join(parts)

    # -- Phase portrait ------------------------------------------------------

    def plot_phase_portrait_ascii(
        self, cex: ConcreteCounterexample, x_var: str, y_var: str,
    ) -> str:
        """2-D phase portrait of *x_var* vs *y_var* as ASCII scatter.

        Start marked ``S``, end marked ``E``.
        """
        xs, ys = cex.trajectory(x_var), cex.trajectory(y_var)
        if not xs or not ys:
            return f"(no data for '{x_var}' or '{y_var}')"
        xlo, xhi = min(xs), max(xs)
        ylo, yhi = min(ys), max(ys)
        if abs(xhi - xlo) < _EPS:
            xhi = xlo + 1.0
        if abs(yhi - ylo) < _EPS:
            yhi = ylo + 1.0

        pw, ph = self.width - 10, self.height
        canvas = [[" "] * pw for _ in range(ph)]
        for i in range(len(xs)):
            c = self._scale_value(xs[i], xlo, xhi, pw - 1)
            r = (ph - 1) - self._scale_value(ys[i], ylo, yhi, ph - 1)
            canvas[r][c] = _MARKER_PHASE
        # start / end markers
        canvas[(ph - 1) - self._scale_value(ys[0], ylo, yhi, ph - 1)][
            self._scale_value(xs[0], xlo, xhi, pw - 1)] = _MARKER_START
        canvas[(ph - 1) - self._scale_value(ys[-1], ylo, yhi, ph - 1)][
            self._scale_value(xs[-1], xlo, xhi, pw - 1)] = _MARKER_END

        lines = [f"Phase portrait: {x_var} vs {y_var}", "=" * self.width]
        for r in range(ph):
            yl = self._format_axis_label(yhi) if r == 0 else (
                 self._format_axis_label(ylo) if r == ph - 1 else "        ")
            lines.append(f"{yl:>8s} |{''.join(canvas[r])}")
        lines.append(" " * 9 + "+" + "-" * pw)
        lines.append(" " * 9 + self._format_axis_label(xlo)
                      + " " * max(0, pw - 16) + self._format_axis_label(xhi))
        lines.append(f"         {x_var:^{pw}s}")
        lines.append(f"  {_MARKER_START}=start  {_MARKER_END}=end  {_MARKER_PHASE}=trajectory")
        return "\n".join(lines)

    # -- Reachable set -------------------------------------------------------

    def visualize_reachable_set(
        self,
        bounds: Dict[str, Tuple[float, float]],
        cex: Optional[ConcreteCounterexample] = None,
    ) -> str:
        """ASCII rendering of reachable-set interval bounds.

        Optionally overlays a counterexample trajectory range.
        """
        if not bounds:
            return "(no reachable-set bounds provided)"
        lines: List[str] = ["Reachable-set bounds", "=" * self.width]
        bar_w = self.width - 30
        for var in sorted(bounds):
            lo, hi = bounds[var]
            cex_info = ""
            if cex is not None:
                t = cex.trajectory(var)
                if t:
                    cex_info = f"  cex:[{min(t):.4g},{max(t):.4g}]"
            lines.append(f"  {var:>12s}: [{lo:+.4g}, {hi:+.4g}]{cex_info}")
            if cex is not None and abs(hi - lo) > _EPS:
                t = cex.trajectory(var)
                if t:
                    bar = [" "] * bar_w
                    for v in t:
                        p = max(0, min(self._scale_value(v, lo, hi, bar_w - 1), bar_w - 1))
                        bar[p] = _MARKER_TRACE
                    lines.append(f"  {'':>12s}  |{''.join(bar)}|")
        return "\n".join(lines)

    # -- Text report ---------------------------------------------------------

    def generate_text_report(
        self, cex: ConcreteCounterexample, result: Optional[VerificationResult] = None,
    ) -> str:
        """Step-by-step text report highlighting key transitions and violation."""
        parts: List[str] = ["Counterexample Report", "=" * self.width]
        if result is not None:
            parts.append(f"Property:    {result.property_name}")
            parts.append(f"Status:      {result.status.name}")
            if result.coverage is not None:
                parts.append(f"Coverage:    {result.coverage:.2%}")
            if result.robustness is not None:
                parts.append(f"Robustness:  {result.robustness:.6g}")
            parts.append("")
        parts += [f"Trace length: {cex.length} steps",
                  f"Duration:     {cex.duration:.6g}",
                  f"Violated:     {cex.property_violated}", ""]
        if cex.parameter_values:
            parts.append("Parameters:")
            for p in sorted(cex.parameter_values):
                parts.append(f"  {p:>20s} = {cex.parameter_values[p]:.6g}")
            parts.append("")

        parts += ["Step-by-step trace:", "-" * self.width]
        vnames = sorted(cex.states[0].keys()) if cex.states else []
        for i, (t, state) in enumerate(zip(cex.time_points, cex.states)):
            changes: List[str] = []
            if i > 0:
                prev = cex.states[i - 1]
                for v in vnames:
                    old, new = prev.get(v, 0.0), state.get(v, 0.0)
                    if abs(new - old) / max(abs(old), _EPS) > _CHANGE_THRESHOLD:
                        changes.append(f"{v} {'↑' if new > old else '↓'} {old:.4g}->{new:.4g}")
            mark = ">>>" if changes else "   "
            vals = ", ".join(f"{v}={state.get(v, 0.0):.4g}" for v in vnames)
            parts.append(f"{mark} t={t:.4g}: {vals}")
            if changes:
                parts.append(f"      key: {'; '.join(changes)}")

        parts += ["", "Violation analysis:", "-" * self.width, self.annotate_violation(cex)]
        return "\n".join(parts)

    # -- SVG circuit diagram -------------------------------------------------

    def generate_svg(
        self, model: BioModel, cex: Optional[ConcreteCounterexample] = None,
    ) -> str:
        """Basic SVG of the regulatory network.

        Species as circle nodes, activation edges as arrows, repression
        edges as blunt T-ends.  Counterexample concentrations annotated
        when provided.
        """
        sps: List[Species] = list(model.species)
        n = len(sps)
        if n == 0:
            return "<svg></svg>"

        rad = max(120, n * 40)
        sz = 2 * (rad + 80 + _SVG_NODE_RADIUS)
        cx, cy = sz / 2, sz / 2
        pos: Dict[str, Tuple[float, float]] = {}
        for i, sp in enumerate(sps):
            a = 2 * math.pi * i / n - math.pi / 2
            pos[sp.name] = (cx + rad * math.cos(a), cy + rad * math.sin(a))

        els = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{sz:.0f}" height="{sz:.0f}">',
            '<defs>'
            '<marker id="arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">'
            '<polygon points="0 0,10 3.5,0 7" fill="#2d8a4e"/></marker>'
            '<marker id="blunt" markerWidth="10" markerHeight="10" refX="5" refY="5" orient="auto">'
            '<line x1="5" y1="0" x2="5" y2="10" stroke="#c0392b" stroke-width="3"/></marker>'
            '</defs>',
        ]

        grn: Optional[GeneRegulatoryNetwork] = model.regulatory_network
        if grn is not None:
            for inter in getattr(grn, "interactions", []):
                src = inter.source if isinstance(inter.source, str) else inter.source.name
                tgt = inter.target if isinstance(inter.target, str) else inter.target.name
                if src not in pos or tgt not in pos:
                    continue
                x1, y1 = pos[src]
                x2, y2 = pos[tgt]
                d = math.hypot(x2 - x1, y2 - y1)
                if d < _EPS:
                    continue
                ux, uy = (x2 - x1) / d, (y2 - y1) / d
                x1s, y1s = x1 + ux * _SVG_NODE_RADIUS, y1 + uy * _SVG_NODE_RADIUS
                x2s, y2s = x2 - ux * _SVG_NODE_RADIUS, y2 - uy * _SVG_NODE_RADIUS
                sign = getattr(inter, "sign", None)
                col = "#c0392b" if sign == InteractionSign.NEGATIVE else "#2d8a4e"
                me = "url(#blunt)" if sign == InteractionSign.NEGATIVE else "url(#arrow)"
                els.append(f'<line x1="{x1s:.1f}" y1="{y1s:.1f}" x2="{x2s:.1f}" '
                           f'y2="{y2s:.1f}" stroke="{col}" stroke-width="2" marker-end="{me}"/>')

        final = cex.final_state if cex else {}
        max_val = max(final.values(), default=1.0) + _EPS
        for sp in sps:
            px, py = pos[sp.name]
            fill = "#eaf2f8"
            if sp.name in final:
                t = min(1.0, max(0.0, final[sp.name] / max_val))
                fill = f"#{int(234 - 100 * t):02x}{int(242 - 60 * t):02x}{int(248 - 40 * t):02x}"
            els.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{_SVG_NODE_RADIUS}" '
                       f'fill="{fill}" stroke="#2c3e50" stroke-width="2"/>')
            els.append(f'<text x="{px:.1f}" y="{py:.1f}" text-anchor="middle" '
                       f'dominant-baseline="central" font-size="{_SVG_FONT_SIZE}" '
                       f'font-family="monospace">{sp.name}</text>')
            if sp.name in final:
                els.append(f'<text x="{px:.1f}" y="{py + _SVG_NODE_RADIUS + 14:.1f}" '
                           f'text-anchor="middle" font-size="{_SVG_FONT_SIZE - 2}" '
                           f'fill="#666">{final[sp.name]:.4g}</text>')
        els.append("</svg>")
        return "\n".join(els)

    # -- Export helpers -------------------------------------------------------

    def export_json(self, cex: ConcreteCounterexample) -> str:
        """Serialize the full counterexample trace as JSON."""
        return json.dumps({
            "property_violated": cex.property_violated,
            "length": cex.length,
            "duration": cex.duration,
            "parameter_values": cex.parameter_values,
            "time_points": cex.time_points,
            "states": cex.states,
            "initial_state": cex.initial_state,
            "final_state": cex.final_state,
        }, indent=2, default=_json_default)

    def export_csv(self, cex: ConcreteCounterexample) -> str:
        """Export the counterexample as a CSV table (time, species…)."""
        if cex.length == 0:
            return ""
        var_names = sorted(cex.states[0].keys())
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["time"] + var_names)
        for t, state in zip(cex.time_points, cex.states):
            writer.writerow([t] + [state.get(v, 0.0) for v in var_names])
        return buf.getvalue()

    # -- Private helpers -----------------------------------------------------

    @staticmethod
    def _scale_value(val: float, min_val: float, max_val: float, extent: int) -> int:
        """Linearly map *val* from [min_val, max_val] to [0, extent], clamped."""
        if abs(max_val - min_val) < _EPS:
            return 0
        ratio = max(0.0, min(1.0, (val - min_val) / (max_val - min_val)))
        return int(round(ratio * extent))

    @staticmethod
    def _format_axis_label(val: float) -> str:
        """Format a numeric value as a fixed-width 8-char axis label."""
        if abs(val) < _EPS:
            return "   0.00 "
        if abs(val) >= 1e4 or abs(val) < 1e-2:
            return f"{val:8.1e}"
        return f"{val:8.4g}"


def _json_default(obj: Any) -> Any:
    """Fallback serializer for :func:`json.dumps`."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
