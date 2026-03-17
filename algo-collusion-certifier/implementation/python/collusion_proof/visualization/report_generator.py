"""Report generation for CollusionProof analysis results."""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class ReportGenerator:
    """Generate formatted reports from analysis results."""

    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # colour helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _verdict_color(verdict: str) -> str:
        v = verdict.upper()
        if "COLLUSI" in v:
            return "#d62728"
        if "COMPETITI" in v:
            return "#2ca02c"
        return "#ff7f0e"

    @staticmethod
    def _confidence_bar(confidence: float) -> str:
        pct = int(round(confidence * 100))
        filled = pct // 2
        empty = 50 - filled
        return f"[{'█' * filled}{'░' * empty}] {pct}%"

    # ------------------------------------------------------------------
    # internal formatting helpers
    # ------------------------------------------------------------------

    def _format_verdict_section(self, verdict: str, confidence: float) -> str:
        color = self._verdict_color(verdict)
        return (
            f'<div class="verdict" style="border-left:6px solid {color}; padding:12px; margin:16px 0;">'
            f'<h2 style="color:{color}; margin:0 0 4px 0;">{verdict.upper()}</h2>'
            f'<p style="margin:0;">Confidence: <strong>{confidence:.1%}</strong></p>'
            f"</div>\n"
        )

    def _format_tier_results(self, tier_results: List[Dict]) -> str:
        if not tier_results:
            return "<p>No tier results available.</p>"
        rows: List[str] = []
        for tr in tier_results:
            tier = tr.get("tier", "?")
            pval = tr.get("p_value", float("nan"))
            stat = tr.get("statistic", float("nan"))
            reject = tr.get("reject", False)
            verdict_icon = "&#10060;" if reject else "&#9989;"
            rows.append(
                f"<tr><td>{tier}</td><td>{stat:.4f}</td><td>{pval:.4f}</td>"
                f"<td>{verdict_icon} {'Reject' if reject else 'Fail to reject'}</td></tr>"
            )
        return (
            '<table class="tier-table"><thead><tr>'
            "<th>Tier</th><th>Statistic</th><th>p-value</th><th>Decision</th>"
            "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>\n"
        )

    @staticmethod
    def _format_collusion_premium(premium_result: Dict) -> str:
        est = premium_result.get("estimate", float("nan"))
        ci = premium_result.get("ci", (float("nan"), float("nan")))
        return (
            f"<p><strong>Estimated collusion premium:</strong> {est:.4f} "
            f"(95 % CI: [{ci[0]:.4f}, {ci[1]:.4f}])</p>\n"
        )

    @staticmethod
    def _format_evidence_chain(evidence: List[str]) -> str:
        if not evidence:
            return "<p>No evidence chain recorded.</p>"
        items = "".join(f"<li>{e}</li>" for e in evidence)
        return f"<ol>{items}</ol>\n"

    def _format_recommendations(self, verdict: str, results: Dict) -> List[str]:
        recs: List[str] = []
        v = verdict.upper()
        if "COLLUSI" in v:
            recs.append("Conduct deeper investigation into pricing algorithms used by the firms.")
            recs.append("Review algorithmic parameters for reward-shaping that incentivises supra-competitive pricing.")
            recs.append("Consider running additional simulations with varied initial conditions.")
        elif "COMPETITI" in v:
            recs.append("No immediate regulatory action recommended based on current evidence.")
            recs.append("Schedule periodic re-assessment as market or algorithm conditions change.")
        else:
            recs.append("Evidence is inconclusive – consider collecting additional data.")
            recs.append("Refine test parameters or extend simulation duration.")

        premium = results.get("collusion_premium", {})
        if isinstance(premium, dict) and premium.get("estimate", 0) > 0.10:
            recs.append("Significant premium detected – prioritise consumer-welfare analysis.")

        return recs

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _generate_summary_table(self, results: Dict) -> str:
        """Generate HTML table with summary statistics."""
        summary = results.get("summary", {})
        if not summary:
            summary = {
                k: v
                for k, v in results.items()
                if isinstance(v, (int, float, str, bool))
            }
        if not summary:
            return ""
        rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in summary.items()
        )
        return (
            '<table class="summary-table"><thead><tr><th>Metric</th><th>Value</th></tr></thead>'
            f"<tbody>{rows}</tbody></table>\n"
        )

    @staticmethod
    def _generate_css() -> str:
        return """
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                   margin: 40px; color: #333; background: #fafafa; }
            h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }
            h2 { color: #555; }
            table { border-collapse: collapse; width: 100%; margin: 12px 0; }
            th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
            th { background: #f0f0f0; }
            .verdict { background: #fff; border-radius: 4px; }
            .tier-table th { background: #e8e8e8; }
            .summary-table { max-width: 500px; }
            .evidence-chain { background: #fffff0; padding: 12px; border: 1px solid #eee; border-radius: 4px; }
            .footer { margin-top: 32px; font-size: 0.85em; color: #999; }
            .recommendation { background: #f0f8ff; padding: 8px 12px; margin: 4px 0; border-left: 3px solid #1f77b4; }
        </style>
        """

    def _generate_regulatory_narrative(self, results: Dict) -> str:
        """Generate regulatory-style narrative paragraph."""
        verdict = results.get("verdict", "UNKNOWN")
        confidence = results.get("confidence", 0.0)
        n_tiers = len(results.get("tier_results", []))
        premium = results.get("collusion_premium", {})
        est = premium.get("estimate", 0.0) if isinstance(premium, dict) else 0.0

        return (
            f"<p>The automated analysis system evaluated the pricing behaviour across "
            f"<strong>{n_tiers}</strong> statistical tiers and reached a verdict of "
            f"<strong>{verdict}</strong> with confidence <strong>{confidence:.1%}</strong>. "
            f"The estimated collusion premium above competitive baseline is "
            f"<strong>{est:.2%}</strong>."
            f" This assessment was produced by the CollusionProof certification pipeline "
            f"and should be reviewed by a domain expert before regulatory action.</p>\n"
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def generate_html_report(
        self,
        results: Dict[str, Any],
        title: str = "CollusionProof Analysis Report",
        include_figures: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """Generate comprehensive HTML report. Returns path to file."""
        verdict = results.get("verdict", "UNKNOWN")
        confidence = results.get("confidence", 0.0)
        tier_results = results.get("tier_results", [])
        premium = results.get("collusion_premium", {})
        evidence = results.get("evidence_chain", [])
        recs = self._format_recommendations(verdict, results)

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        parts: List[str] = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'>",
            f"<title>{title}</title>",
            self._generate_css(),
            "</head><body>",
            f"<h1>{title}</h1>",
            f"<p class='footer'>Generated: {now}</p>",
            "<h2>Verdict</h2>",
            self._format_verdict_section(verdict, confidence),
            "<h2>Regulatory Narrative</h2>",
            self._generate_regulatory_narrative(results),
            "<h2>Summary Statistics</h2>",
            self._generate_summary_table(results),
            "<h2>Tier Results</h2>",
            self._format_tier_results(tier_results),
        ]

        if isinstance(premium, dict) and premium:
            parts.append("<h2>Collusion Premium</h2>")
            parts.append(self._format_collusion_premium(premium))

        if evidence:
            parts.append("<h2>Evidence Chain</h2>")
            parts.append('<div class="evidence-chain">')
            parts.append(self._format_evidence_chain(evidence))
            parts.append("</div>")

        if recs:
            parts.append("<h2>Recommendations</h2>")
            for r in recs:
                parts.append(f'<div class="recommendation">{r}</div>')

        if include_figures:
            parts.append("<h2>Figures</h2>")
            for fig_path in include_figures:
                parts.append(f'<img src="{fig_path}" style="max-width:100%; margin:8px 0;" />')

        parts.append('<div class="footer">CollusionProof Automated Report</div>')
        parts.append("</body></html>")

        html = "\n".join(parts)

        if save_path is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"report_{ts}.html")

        with open(save_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        return save_path

    # ------------------------------------------------------------------

    def generate_text_report(self, results: Dict[str, Any]) -> str:
        """Generate plain text report. Returns the text."""
        verdict = results.get("verdict", "UNKNOWN")
        confidence = results.get("confidence", 0.0)
        tier_results = results.get("tier_results", [])
        premium = results.get("collusion_premium", {})
        evidence = results.get("evidence_chain", [])
        recs = self._format_recommendations(verdict, results)

        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  CollusionProof Analysis Report")
        lines.append(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"  VERDICT: {verdict}")
        lines.append(f"  Confidence: {self._confidence_bar(confidence)}")
        lines.append("")

        # Summary
        summary = results.get("summary", {})
        if not summary:
            summary = {k: v for k, v in results.items() if isinstance(v, (int, float, str, bool))}
        if summary:
            lines.append("-" * 40)
            lines.append("  Summary Statistics")
            lines.append("-" * 40)
            for k, v in summary.items():
                lines.append(f"  {k:30s}: {v}")
            lines.append("")

        # Tier results
        if tier_results:
            lines.append("-" * 40)
            lines.append("  Tier Results")
            lines.append("-" * 40)
            header = f"  {'Tier':12s} {'Statistic':>12s} {'p-value':>10s} {'Decision':>18s}"
            lines.append(header)
            lines.append("  " + "-" * len(header.strip()))
            for tr in tier_results:
                tier = tr.get("tier", "?")
                stat = tr.get("statistic", float("nan"))
                pval = tr.get("p_value", float("nan"))
                reject = tr.get("reject", False)
                decision = "REJECT" if reject else "FAIL TO REJECT"
                lines.append(f"  {tier:12s} {stat:12.4f} {pval:10.4f} {decision:>18s}")
            lines.append("")

        # Collusion premium
        if isinstance(premium, dict) and premium:
            est = premium.get("estimate", float("nan"))
            ci = premium.get("ci", (float("nan"), float("nan")))
            lines.append("-" * 40)
            lines.append("  Collusion Premium")
            lines.append("-" * 40)
            lines.append(f"  Estimate: {est:.4f}  (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
            lines.append("")

        # Evidence chain
        if evidence:
            lines.append("-" * 40)
            lines.append("  Evidence Chain")
            lines.append("-" * 40)
            for idx, e in enumerate(evidence, 1):
                lines.append(f"  {idx}. {e}")
            lines.append("")

        # Recommendations
        if recs:
            lines.append("-" * 40)
            lines.append("  Recommendations")
            lines.append("-" * 40)
            for r in recs:
                lines.append(f"  • {r}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def generate_json_report(self, results: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """Generate JSON report. Returns path to file."""
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "results": results,
            "recommendations": self._format_recommendations(
                results.get("verdict", "UNKNOWN"), results
            ),
        }
        if save_path is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"report_{ts}.json")

        with open(save_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, cls=_NumpyEncoder)
        return save_path

    # ------------------------------------------------------------------

    def generate_certificate(
        self,
        results: Dict[str, Any],
        system_id: str = "UNKNOWN",
        save_path: Optional[str] = None,
    ) -> str:
        """Generate formal certificate document (HTML). Returns file path."""
        verdict = results.get("verdict", "UNKNOWN")
        confidence = results.get("confidence", 0.0)
        color = self._verdict_color(verdict)
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>CollusionProof Certificate</title>
<style>
    body {{ font-family: 'Georgia', serif; margin: 60px; color: #222; background: #fff; text-align: center; }}
    .border {{ border: 4px double #333; padding: 40px; max-width: 700px; margin: auto; }}
    h1 {{ font-size: 28px; letter-spacing: 2px; }}
    .verdict {{ font-size: 24px; color: {color}; margin: 20px 0; }}
    .meta {{ font-size: 13px; color: #888; }}
    .seal {{ font-size: 48px; margin: 16px 0; }}
</style></head><body>
<div class="border">
    <h1>CERTIFICATE OF ANALYSIS</h1>
    <p class="seal">&#9878;</p>
    <p>This certifies that algorithmic pricing system</p>
    <p><strong style="font-size:18px;">{system_id}</strong></p>
    <p>has been evaluated by the <em>CollusionProof</em> automated certification pipeline.</p>
    <p class="verdict"><strong>{verdict.upper()}</strong></p>
    <p>Confidence: <strong>{confidence:.1%}</strong></p>
    <hr style="width:60%; margin: 24px auto;">
    <p class="meta">Issued: {now}</p>
    <p class="meta">CollusionProof v1.0 &mdash; Automated Algorithmic Collusion Certifier</p>
</div>
</body></html>"""

        if save_path is None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"certificate_{ts}.html")

        with open(save_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        return save_path


# ======================================================================
# TableFormatter
# ======================================================================


class TableFormatter:
    """Format data as text / HTML tables."""

    def __init__(self) -> None:
        self._default_align = "left"

    # ------------------------------------------------------------------
    # HTML table
    # ------------------------------------------------------------------

    def format_html_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        caption: Optional[str] = None,
        highlight_max: bool = False,
    ) -> str:
        """Return an HTML ``<table>`` string.

        Args:
            headers: Column headers.
            rows: List of row data (each row is a list of cell strings).
            caption: Optional table caption.
            highlight_max: If True, bold the maximum numeric value in each column.
        """
        max_indices: Dict[int, int] = {}
        if highlight_max:
            for col_idx in range(len(headers)):
                best_row = -1
                best_val = -float("inf")
                for row_idx, row in enumerate(rows):
                    try:
                        v = float(row[col_idx])
                        if v > best_val:
                            best_val = v
                            best_row = row_idx
                    except (ValueError, IndexError):
                        continue
                if best_row >= 0:
                    max_indices[col_idx] = best_row

        parts: List[str] = ["<table>"]
        if caption:
            parts.append(f"<caption>{caption}</caption>")
        parts.append("<thead><tr>")
        for h in headers:
            parts.append(f"<th>{h}</th>")
        parts.append("</tr></thead><tbody>")
        for row_idx, row in enumerate(rows):
            parts.append("<tr>")
            for col_idx, cell in enumerate(row):
                bold = highlight_max and max_indices.get(col_idx) == row_idx
                cell_html = f"<strong>{cell}</strong>" if bold else cell
                parts.append(f"<td>{cell_html}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table>")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # text table
    # ------------------------------------------------------------------

    def format_text_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        padding: int = 2,
    ) -> str:
        """Return a fixed-width plain-text table."""
        all_rows = [headers] + rows
        num_cols = len(headers)
        col_widths = [0] * num_cols
        for row in all_rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        def _fmt_row(row: List[str]) -> str:
            cells = [str(cell).ljust(col_widths[i] + padding) for i, cell in enumerate(row)]
            return "  ".join(cells)

        lines = [_fmt_row(headers)]
        lines.append("-" * len(lines[0]))
        for row in rows:
            lines.append(_fmt_row(row))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # metrics shorthand
    # ------------------------------------------------------------------

    def format_metrics_table(
        self,
        metrics: Dict[str, float],
        title: Optional[str] = None,
        fmt: str = ".4f",
    ) -> str:
        """Return an HTML table of metric-name → value pairs."""
        headers = ["Metric", "Value"]
        rows = [[k, f"{v:{fmt}}"] for k, v in metrics.items()]
        return self.format_html_table(headers, rows, caption=title)
