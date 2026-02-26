"""Result export for BioProver verification and synthesis workflows.

Generates machine-readable and human-readable output in multiple
formats: JSON, CSV, LaTeX, HTML, SBML, and SBOL.
"""

from __future__ import annotations

import csv
import html
import io
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from bioprover.cegar.cegar_engine import VerificationResult, VerificationStatus
from bioprover.cegar.counterexample import ConcreteCounterexample
from bioprover.models.bio_model import BioModel
from bioprover.models.parameters import Parameter
from bioprover.models.reactions import Reaction
from bioprover.models.species import Species

logger = logging.getLogger(__name__)

_VERSION = "0.1.0"


class ExportFormat(Enum):
    """Supported export formats."""

    JSON = auto()
    CSV = auto()
    LATEX = auto()
    HTML = auto()
    SBML = auto()
    SBOL = auto()


class ResultExporter:
    """Export verification results to multiple output formats.

    Supports JSON, CSV, LaTeX, HTML, SBML annotations, and SBOL
    annotations.  Every public method returns the generated content as
    a string and optionally writes it to *path* when provided.
    """

    def __init__(self) -> None:
        self._created_at = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Public export methods
    # ------------------------------------------------------------------

    def export_json(
        self,
        results: List[VerificationResult],
        path: Optional[Path] = None,
    ) -> str:
        """Export results as machine-readable JSON.

        The payload contains a *metadata* block (timestamp, version,
        result count) and a *results* array with full result data
        produced by ``VerificationResult.to_dict()``.
        """
        payload: Dict = {
            "metadata": {
                "generator": "BioProver",
                "version": _VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "result_count": len(results),
            },
            "results": [r.to_dict() for r in results],
        }
        content = json.dumps(payload, indent=2, default=str)
        return self._write_or_return(content, path)

    def export_csv(
        self,
        results: List[VerificationResult],
        path: Optional[Path] = None,
    ) -> str:
        """Export results as a CSV table.

        Columns: property, status, coverage, robustness, iterations,
        time, states, predicates.
        """
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "property",
            "status",
            "coverage",
            "robustness",
            "iterations",
            "time",
            "states",
            "predicates",
        ])
        for r in results:
            stats = r.statistics
            writer.writerow([
                r.property_name,
                r.status.name,
                f"{r.coverage:.4f}",
                f"{r.robustness:.4f}",
                stats.iterations if stats else "",
                f"{stats.total_time:.3f}" if stats else "",
                stats.num_states if stats else "",
                stats.num_predicates if stats else "",
            ])
        content = buf.getvalue()
        return self._write_or_return(content, path)

    def export_latex(
        self,
        results: List[VerificationResult],
        path: Optional[Path] = None,
    ) -> str:
        """Export results as a LaTeX *booktabs* table.

        Columns: Property, Status, Coverage, Robustness, Iters, Time(s).
        The output is a standalone ``tabular`` environment suitable for
        inclusion in academic papers.
        """
        lines: List[str] = [
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \caption{BioProver verification results}",
            r"  \label{tab:verification-results}",
            r"  \begin{tabular}{l l r r r r}",
            r"    \toprule",
            r"    Property & Status & Coverage & Robustness & Iters & Time(s) \\",
            r"    \midrule",
        ]
        for r in results:
            stats = r.statistics
            prop = self._escape_latex(r.property_name)
            status = self._escape_latex(r.status.name)
            cov = f"{r.coverage:.2f}"
            rob = f"{r.robustness:.2f}"
            iters = str(stats.iterations) if stats else "--"
            time_s = f"{stats.total_time:.2f}" if stats else "--"
            lines.append(
                f"    {prop} & {status} & {cov} & {rob} & {iters} & {time_s} \\\\"
            )
        lines += [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
        content = "\n".join(lines) + "\n"
        return self._write_or_return(content, path)

    def export_html(
        self,
        results: List[VerificationResult],
        model_name: str = "Model",
        path: Optional[Path] = None,
    ) -> str:
        """Export results as a self-contained HTML report.

        Includes a summary section with status badges, a results table,
        and counterexample trace data when available.
        """
        verified = sum(1 for r in results if r.is_verified)
        falsified = sum(1 for r in results if r.is_falsified)
        unknown = len(results) - verified - falsified

        rows: List[str] = []
        cex_sections: List[str] = []
        for r in results:
            badge = self._status_badge_html(r.status)
            stats = r.statistics
            iters = str(stats.iterations) if stats else "&ndash;"
            time_s = self._format_time(stats.total_time) if stats else "&ndash;"
            rows.append(
                f"<tr><td>{html.escape(r.property_name)}</td>"
                f"<td>{badge}</td>"
                f"<td>{r.coverage:.2f}</td>"
                f"<td>{r.robustness:.2f}</td>"
                f"<td>{iters}</td>"
                f"<td>{time_s}</td></tr>"
            )
            if r.counterexample is not None:
                cex_sections.append(self._counterexample_html(r))

        cex_block = ""
        if cex_sections:
            cex_block = (
                '<h2>Counterexamples</h2>\n' + "\n".join(cex_sections)
            )

        content = _HTML_TEMPLATE.format(
            model_name=html.escape(model_name),
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            verified=verified,
            falsified=falsified,
            unknown=unknown,
            total=len(results),
            table_rows="\n".join(rows),
            counterexamples=cex_block,
        )
        return self._write_or_return(content, path)

    def export_sbml_annotations(
        self,
        model: BioModel,
        results: List[VerificationResult],
    ) -> str:
        """Generate SBML-compatible XML annotations.

        Attaches verification status as ``<annotation>`` elements on
        each species and parameter present in *model*.
        """
        status_map = {r.property_name: r for r in results}
        lines: List[str] = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">',
            "  <model>",
            "    <listOfSpecies>",
        ]
        for sp in model.species:
            res = status_map.get(sp.name)
            status_str = res.status.name if res else "UNCHECKED"
            lines += [
                f'      <species id="{html.escape(sp.name)}">',
                "        <annotation>",
                "          <bioprover:verification "
                f'xmlns:bioprover="http://bioprover.org/ns" '
                f'status="{status_str}" />',
                "        </annotation>",
                "      </species>",
            ]
        lines.append("    </listOfSpecies>")
        lines.append("    <listOfParameters>")
        for param in model.parameters:
            res = status_map.get(param.name)
            status_str = res.status.name if res else "UNCHECKED"
            lines += [
                f'      <parameter id="{html.escape(param.name)}" '
                f'value="{param.value}">',
                "        <annotation>",
                "          <bioprover:verification "
                f'xmlns:bioprover="http://bioprover.org/ns" '
                f'status="{status_str}" />',
                "        </annotation>",
                "      </parameter>",
            ]
        lines += [
            "    </listOfParameters>",
            "  </model>",
            "</sbml>",
        ]
        return "\n".join(lines) + "\n"

    def export_sbol_annotations(
        self,
        model: BioModel,
        results: List[VerificationResult],
    ) -> str:
        """Generate SBOL-style verification metadata.

        Produces an RDF/XML fragment with verification results
        expressed using SBOL-compatible terms.
        """
        status_map = {r.property_name: r for r in results}
        lines: List[str] = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
            '         xmlns:sbol="http://sbols.org/v3#"',
            '         xmlns:bioprover="http://bioprover.org/ns#">',
        ]
        for sp in model.species:
            sp_id = html.escape(sp.name)
            res = status_map.get(sp.name)
            status_str = res.status.name if res else "UNCHECKED"
            cov = f"{res.coverage:.4f}" if res else "0.0000"
            rob = f"{res.robustness:.4f}" if res else "0.0000"
            lines += [
                f'  <sbol:Component rdf:about="http://bioprover.org/component/{sp_id}">',
                f"    <sbol:displayId>{sp_id}</sbol:displayId>",
                f"    <bioprover:verificationStatus>{status_str}</bioprover:verificationStatus>",
                f"    <bioprover:coverage>{cov}</bioprover:coverage>",
                f"    <bioprover:robustness>{rob}</bioprover:robustness>",
                "  </sbol:Component>",
            ]
        lines.append("</rdf:RDF>")
        return "\n".join(lines) + "\n"

    def summary_report(
        self,
        results: List[VerificationResult],
    ) -> str:
        """Generate a human-readable text summary.

        Includes overall statistics (total, verified, falsified,
        unknown), a per-property breakdown, and timing information.
        """
        verified = [r for r in results if r.is_verified]
        falsified = [r for r in results if r.is_falsified]
        unknown = [
            r for r in results
            if not r.is_verified and not r.is_falsified
        ]

        lines: List[str] = [
            "=" * 60,
            "  BioProver Verification Summary",
            "=" * 60,
            "",
            f"  Total properties : {len(results)}",
            f"  Verified         : {len(verified)}",
            f"  Falsified        : {len(falsified)}",
            f"  Unknown          : {len(unknown)}",
            "",
            "-" * 60,
            "  Per-property breakdown",
            "-" * 60,
        ]
        for r in results:
            marker = "✓" if r.is_verified else ("✗" if r.is_falsified else "?")
            lines.append(f"  [{marker}] {r.property_name}: {r.status.name}")
            lines.append(f"      coverage={r.coverage:.2f}  robustness={r.robustness:.2f}")
            if r.message:
                lines.append(f"      message: {r.message}")

        # Timing statistics
        times = [
            r.statistics.total_time
            for r in results
            if r.statistics is not None
        ]
        if times:
            lines += [
                "",
                "-" * 60,
                "  Timing statistics",
                "-" * 60,
                f"  Total time  : {self._format_time(sum(times))}",
                f"  Mean time   : {self._format_time(sum(times) / len(times))}",
                f"  Min time    : {self._format_time(min(times))}",
                f"  Max time    : {self._format_time(max(times))}",
            ]

        lines += ["", "=" * 60]
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_or_return(content: str, path: Optional[Path]) -> str:
        """Write *content* to *path* if given, then return *content*."""
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            logger.info("Wrote export to %s", path)
        return content

    @staticmethod
    def _escape_latex(text: str) -> str:
        """Escape special LaTeX characters in *text*."""
        replacements = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for char, escaped in replacements.items():
            text = text.replace(char, escaped)
        return text

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format *seconds* as a human-readable duration string."""
        if seconds < 1.0:
            return f"{seconds * 1000:.0f}ms"
        if seconds < 60.0:
            return f"{seconds:.2f}s"
        minutes, secs = divmod(seconds, 60)
        return f"{int(minutes)}m {secs:.1f}s"

    @staticmethod
    def _status_badge_html(status: VerificationStatus) -> str:
        """Return an inline HTML badge for *status*."""
        colors = {
            VerificationStatus.VERIFIED: ("#2e7d32", "#e8f5e9"),
            VerificationStatus.FALSIFIED: ("#c62828", "#ffebee"),
            VerificationStatus.UNKNOWN: ("#f57f17", "#fff8e1"),
            VerificationStatus.BOUNDED_GUARANTEE: ("#1565c0", "#e3f2fd"),
        }
        fg, bg = colors.get(status, ("#424242", "#f5f5f5"))
        return (
            f'<span style="background:{bg};color:{fg};'
            f'padding:2px 8px;border-radius:4px;font-weight:600;">'
            f"{html.escape(status.name)}</span>"
        )

    @staticmethod
    def _counterexample_html(result: VerificationResult) -> str:
        """Render a counterexample trace as an HTML detail block."""
        cex: ConcreteCounterexample = result.counterexample  # type: ignore[assignment]
        lines: List[str] = [
            f"<details><summary>Counterexample for "
            f"<strong>{html.escape(result.property_name)}</strong> "
            f"(length={cex.length}, duration={cex.duration:.3f})</summary>",
            "<pre>",
            f"Property violated: {html.escape(cex.property_violated)}",
            f"Parameters: {html.escape(json.dumps(cex.parameter_values, default=str))}",
            "",
            "Time points:",
        ]
        for i, t in enumerate(cex.time_points):
            state_str = ", ".join(
                f"{k}={v:.4g}" for k, v in cex.states[i].items()
            ) if i < len(cex.states) else ""
            lines.append(f"  t={t:.4f}: {state_str}")
        lines += ["</pre>", "</details>"]
        return "\n".join(lines)


# ------------------------------------------------------------------
# HTML template
# ------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BioProver Report &mdash; {model_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
         Helvetica, Arial, sans-serif; margin: 2em; color: #212121;
         background: #fafafa; }}
  h1 {{ border-bottom: 2px solid #1565c0; padding-bottom: .3em; }}
  h2 {{ color: #1565c0; }}
  .summary {{ display: flex; gap: 1.5em; margin-bottom: 1.5em; }}
  .summary .card {{ background: #fff; border: 1px solid #e0e0e0;
                    border-radius: 8px; padding: 1em 1.5em;
                    min-width: 120px; text-align: center; }}
  .summary .card .num {{ font-size: 2em; font-weight: 700; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; }}
  th, td {{ padding: .6em 1em; text-align: left; border-bottom: 1px solid #e0e0e0; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  details {{ margin: 1em 0; }}
  pre {{ background: #263238; color: #eeffff; padding: 1em;
         border-radius: 6px; overflow-x: auto; }}
  footer {{ margin-top: 3em; font-size: .85em; color: #757575; }}
</style>
</head>
<body>
<h1>BioProver Verification Report &mdash; {model_name}</h1>
<p>Generated: {timestamp}</p>

<div class="summary">
  <div class="card"><div class="num">{total}</div>Total</div>
  <div class="card" style="border-color:#2e7d32"><div class="num" style="color:#2e7d32">{verified}</div>Verified</div>
  <div class="card" style="border-color:#c62828"><div class="num" style="color:#c62828">{falsified}</div>Falsified</div>
  <div class="card" style="border-color:#f57f17"><div class="num" style="color:#f57f17">{unknown}</div>Unknown</div>
</div>

<h2>Results</h2>
<table>
<thead><tr><th>Property</th><th>Status</th><th>Coverage</th><th>Robustness</th><th>Iters</th><th>Time</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>

{counterexamples}

<footer>BioProver v{version} &mdash; CEGAR-based verification for synthetic biology</footer>
</body>
</html>
""".replace("{version}", _VERSION)
