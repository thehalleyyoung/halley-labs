"""Report experiment results to disk and terminal.

Provides:
  - ResultReporter: persist results to JSON, generate text/JSON reports
  - Summary statistics and comparison tables
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class ResultReporter:
    """Write experiment results to files and produce human-readable summaries.

    Parameters
    ----------
    output_dir :
        Directory where result files are written.
    """

    def __init__(self, output_dir: str) -> None:
        self._output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def report(self, results: Dict[str, Any]) -> Path:
        """Persist *results* to ``results.json`` in the output directory.

        Parameters
        ----------
        results :
            Dictionary returned by :class:`ExperimentRunner.run`.

        Returns
        -------
        Path
            Path to the written file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / "results.json"
        path.write_text(json.dumps(results, indent=2, default=_json_default))
        return path

    def report_multiple(self, results_list: List[Dict[str, Any]]) -> Path:
        """Persist multiple result dicts to ``all_results.json``.

        Parameters
        ----------
        results_list :
            List of result dicts from multiple runs.

        Returns
        -------
        Path
            Path to the written file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / "all_results.json"
        path.write_text(json.dumps(results_list, indent=2, default=_json_default))
        return path

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, results: Dict[str, Any]) -> str:
        """Return a concise human-readable summary of the results.

        Parameters
        ----------
        results :
            Dictionary returned by :class:`ExperimentRunner.run`.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            "=" * 45,
            "  CausalQD Experiment Summary",
            "=" * 45,
            f"  Best quality   : {_fmt(results.get('best_quality'))}",
            f"  QD-Score       : {_fmt(results.get('qd_score'))}",
            f"  Coverage       : {_fmt(results.get('coverage'))}",
            f"  Elites         : {results.get('n_elites', 'N/A')}",
            f"  Evaluated      : {results.get('n_evaluated', 'N/A')}",
            f"  Time (s)       : {_fmt(results.get('elapsed_seconds'))}",
        ]

        # Structural metrics if present
        if "shd" in results:
            lines.append(f"  SHD            : {results['shd']}")
        if "f1" in results:
            lines.append(f"  F1             : {_fmt(results['f1'])}")
        if "precision" in results:
            lines.append(f"  Precision      : {_fmt(results['precision'])}")
        if "recall" in results:
            lines.append(f"  Recall         : {_fmt(results['recall'])}")

        # Baselines if present
        if "baselines" in results:
            lines.append("-" * 45)
            lines.append("  Baselines:")
            for name, bl_results in results["baselines"].items():
                parts = [f"{k}={_fmt(v)}" for k, v in bl_results.items()]
                lines.append(f"    {name:12s}: {', '.join(parts)}")

        lines.append("=" * 45)
        return "\n".join(lines)

    def summary_json(self, results: Dict[str, Any]) -> str:
        """Return a JSON-formatted summary."""
        summary = {
            "best_quality": results.get("best_quality"),
            "qd_score": results.get("qd_score"),
            "coverage": results.get("coverage"),
            "n_elites": results.get("n_elites"),
            "elapsed_seconds": results.get("elapsed_seconds"),
        }
        for k in ["shd", "f1", "precision", "recall"]:
            if k in results:
                summary[k] = results[k]
        return json.dumps(summary, indent=2, default=_json_default)

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    @staticmethod
    def comparison_table(
        results_list: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
    ) -> str:
        """Generate a text comparison table across multiple runs.

        Parameters
        ----------
        results_list :
            List of result dicts.
        metrics :
            Metric keys to include (default: standard set).

        Returns
        -------
        str
            Formatted text table.
        """
        if metrics is None:
            metrics = ["qd_score", "coverage", "best_quality", "n_elites", "elapsed_seconds"]

        # Header
        header = f"{'Run':>5}"
        for m in metrics:
            header += f"  {m:>15}"
        lines = [header, "-" * len(header)]

        for idx, results in enumerate(results_list):
            row = f"{idx:>5}"
            for m in metrics:
                val = results.get(m, "N/A")
                row += f"  {_fmt(val):>15}"
            lines.append(row)

        # Summary row
        if len(results_list) > 1:
            lines.append("-" * len(header))
            summary_row = f"{'Mean':>5}"
            for m in metrics:
                vals = [r.get(m) for r in results_list if m in r and isinstance(r[m], (int, float))]
                if vals:
                    summary_row += f"  {np.mean(vals):>15.4f}"
                else:
                    summary_row += f"  {'N/A':>15}"
            lines.append(summary_row)

            std_row = f"{'Std':>5}"
            for m in metrics:
                vals = [r.get(m) for r in results_list if m in r and isinstance(r[m], (int, float))]
                if vals:
                    std_row += f"  {np.std(vals):>15.4f}"
                else:
                    std_row += f"  {'N/A':>15}"
            lines.append(std_row)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Per-generation CSV export
    # ------------------------------------------------------------------

    def export_history_csv(self, results: Dict[str, Any]) -> Path:
        """Export history metrics to CSV.

        Parameters
        ----------
        results :
            Result dict with ``*_history`` keys.

        Returns
        -------
        Path
            Path to the CSV file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / "history.csv"

        history_keys = sorted(
            k for k in results if k.endswith("_history") and isinstance(results[k], list)
        )
        if not history_keys:
            path.write_text("No history data\n")
            return path

        n_gens = max(len(results[k]) for k in history_keys)
        header = "generation," + ",".join(k.replace("_history", "") for k in history_keys)
        lines = [header]
        for gen in range(n_gens):
            row = [str(gen)]
            for k in history_keys:
                vals = results[k]
                row.append(str(vals[gen]) if gen < len(vals) else "")
            lines.append(",".join(row))

        path.write_text("\n".join(lines) + "\n")
        return path


# ======================================================================
# Helpers
# ======================================================================


def _fmt(val: Any) -> str:
    """Format a value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def _json_default(obj: Any) -> Any:
    """JSON serializer for non-standard types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)
