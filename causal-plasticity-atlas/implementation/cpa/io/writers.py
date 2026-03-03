"""Output writers for the CPA engine.

Provides writers for saving pipeline results in various formats:
JSON, CSV, numpy arrays, full atlas directory, and reports.
"""

from __future__ import annotations

import csv
import json
import os
import time
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from cpa.utils.logging import get_logger

logger = get_logger("io.writers")


# =====================================================================
# JSON default handler
# =====================================================================


def _json_default(obj: Any) -> Any:
    """Default JSON serializer for numpy and CPA types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    if hasattr(obj, "value"):
        return obj.value
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# =====================================================================
# JSONWriter
# =====================================================================


class JSONWriter:
    """Write CPA results to JSON format.

    Parameters
    ----------
    indent : int
        JSON indentation level.
    sort_keys : bool
        Whether to sort dictionary keys.

    Examples
    --------
    >>> writer = JSONWriter()
    >>> writer.write(atlas.to_dict(), "results/atlas.json")
    """

    def __init__(self, indent: int = 2, sort_keys: bool = False) -> None:
        self._indent = indent
        self._sort_keys = sort_keys

    def write(
        self,
        data: Any,
        path: Union[str, Path],
    ) -> Path:
        """Write data to a JSON file.

        Parameters
        ----------
        data : dict or list
            JSON-serializable data.
        path : str or Path
            Output file path.

        Returns
        -------
        Path
            Absolute path to written file.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        text = json.dumps(
            data,
            indent=self._indent,
            sort_keys=self._sort_keys,
            default=_json_default,
        )

        out.write_text(text)
        logger.debug("Wrote JSON: %s (%d bytes)", out, len(text))
        return out.resolve()

    def to_string(self, data: Any) -> str:
        """Serialize data to a JSON string.

        Parameters
        ----------
        data : dict or list
            JSON-serializable data.

        Returns
        -------
        str
        """
        return json.dumps(
            data,
            indent=self._indent,
            sort_keys=self._sort_keys,
            default=_json_default,
        )


# =====================================================================
# CSVWriter
# =====================================================================


class CSVWriter:
    """Write CPA results to CSV format.

    Parameters
    ----------
    delimiter : str
        CSV delimiter.
    quoting : int
        CSV quoting style.

    Examples
    --------
    >>> writer = CSVWriter()
    >>> writer.write_descriptors(atlas.foundation.descriptors, "desc.csv")
    """

    def __init__(
        self, delimiter: str = ",", quoting: int = csv.QUOTE_MINIMAL
    ) -> None:
        self._delimiter = delimiter
        self._quoting = quoting

    def write(
        self,
        rows: List[Dict[str, Any]],
        path: Union[str, Path],
        columns: Optional[List[str]] = None,
    ) -> Path:
        """Write a list of row dictionaries to CSV.

        Parameters
        ----------
        rows : list of dict
            Row data.
        path : str or Path
            Output file path.
        columns : list of str, optional
            Column order. If None, uses keys from first row.

        Returns
        -------
        Path
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if not rows:
            out.write_text("")
            return out.resolve()

        if columns is None:
            columns = list(rows[0].keys())

        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=columns,
                delimiter=self._delimiter,
                quoting=self._quoting,
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                clean_row = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in row.items()
                }
                writer.writerow(clean_row)

        logger.debug("Wrote CSV: %s (%d rows)", out, len(rows))
        return out.resolve()

    def write_descriptors(
        self,
        descriptors: Dict[str, Any],
        path: Union[str, Path],
    ) -> Path:
        """Write plasticity descriptors to CSV.

        Parameters
        ----------
        descriptors : dict of str → DescriptorResult
            Variable → descriptor mapping.
        path : str or Path
            Output path.

        Returns
        -------
        Path
        """
        rows: List[Dict[str, Any]] = []
        for var, dr in descriptors.items():
            if hasattr(dr, "to_dict"):
                row = dr.to_dict()
            elif isinstance(dr, dict):
                row = dict(dr)
            else:
                row = {"variable": var}
            rows.append(row)

        columns = [
            "variable", "structural", "parametric",
            "emergence", "sensitivity", "classification", "norm",
        ]
        return self.write(rows, path, columns=columns)

    def write_alignments(
        self,
        alignments: Dict[Any, Any],
        path: Union[str, Path],
    ) -> Path:
        """Write alignment results to CSV.

        Parameters
        ----------
        alignments : dict
            Alignment results.
        path : str or Path
            Output path.

        Returns
        -------
        Path
        """
        rows: List[Dict[str, Any]] = []
        for key, ar in alignments.items():
            if hasattr(ar, "to_dict"):
                row = ar.to_dict()
            elif isinstance(ar, dict):
                row = dict(ar)
            else:
                row = {}
            if isinstance(key, tuple):
                row["context_i"] = key[0]
                row["context_j"] = key[1]
            rows.append(row)

        columns = [
            "context_i", "context_j", "structural_cost",
            "parametric_cost", "total_cost", "shared_edges",
            "modified_edges", "context_specific_edges",
        ]
        return self.write(rows, path, columns=columns)

    def write_matrix(
        self,
        matrix: np.ndarray,
        path: Union[str, Path],
        row_names: Optional[List[str]] = None,
        col_names: Optional[List[str]] = None,
    ) -> Path:
        """Write a 2D matrix to CSV.

        Parameters
        ----------
        matrix : np.ndarray
            2D array.
        path : str or Path
            Output path.
        row_names : list of str, optional
            Row labels.
        col_names : list of str, optional
            Column headers.

        Returns
        -------
        Path
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", newline="") as f:
            writer = csv.writer(f, delimiter=self._delimiter)

            if col_names:
                header = [""] + col_names if row_names else col_names
                writer.writerow(header)

            for i in range(matrix.shape[0]):
                row = matrix[i].tolist()
                if row_names and i < len(row_names):
                    row = [row_names[i]] + row
                writer.writerow(row)

        return out.resolve()


# =====================================================================
# NumpyWriter
# =====================================================================


class NumpyWriter:
    """Write numpy arrays and collections to disk.

    Examples
    --------
    >>> writer = NumpyWriter()
    >>> writer.write_arrays({"adj": adj, "params": params}, "data.npz")
    """

    def write_array(
        self, array: np.ndarray, path: Union[str, Path]
    ) -> Path:
        """Write a single array to .npy.

        Parameters
        ----------
        array : np.ndarray
            Array to save.
        path : str or Path
            Output path.

        Returns
        -------
        Path
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, array)
        logger.debug("Wrote array: %s shape=%s", out, array.shape)
        return out.resolve()

    def write_arrays(
        self,
        arrays: Dict[str, np.ndarray],
        path: Union[str, Path],
        compressed: bool = True,
    ) -> Path:
        """Write multiple arrays to .npz.

        Parameters
        ----------
        arrays : dict of str → np.ndarray
            Named arrays.
        path : str or Path
            Output path.
        compressed : bool
            Use compressed format.

        Returns
        -------
        Path
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if compressed:
            np.savez_compressed(out, **arrays)
        else:
            np.savez(out, **arrays)

        actual = out.with_suffix(".npz") if out.suffix != ".npz" else out
        logger.debug(
            "Wrote %d arrays to %s", len(arrays), actual
        )
        return actual.resolve()


# =====================================================================
# AtlasWriter
# =====================================================================


class AtlasWriter:
    """Write a full atlas to a directory structure.

    Creates an organized directory with:
    - atlas.json (full serialized atlas)
    - summary.json (summary statistics)
    - descriptors.csv (plasticity descriptors)
    - alignments.csv (pairwise alignment costs)
    - arrays/ (numpy arrays)
    - plots/ (visualizations, if generated)

    Parameters
    ----------
    include_arrays : bool
        Whether to save numpy arrays separately.
    include_plots : bool
        Whether to generate and save plots.

    Examples
    --------
    >>> writer = AtlasWriter()
    >>> writer.write(atlas, "output/my_atlas/")
    """

    def __init__(
        self,
        include_arrays: bool = True,
        include_plots: bool = False,
    ) -> None:
        self._include_arrays = include_arrays
        self._include_plots = include_plots

    def write(
        self,
        atlas: Any,
        directory: Union[str, Path],
    ) -> Path:
        """Write the full atlas to a directory.

        Parameters
        ----------
        atlas : AtlasResult
            Atlas to serialize.
        directory : str or Path
            Output directory (created if needed).

        Returns
        -------
        Path
            Path to the output directory.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        json_writer = JSONWriter()
        csv_writer = CSVWriter()
        np_writer = NumpyWriter()

        if hasattr(atlas, "to_dict"):
            atlas_dict = atlas.to_dict()
        else:
            atlas_dict = dict(atlas) if isinstance(atlas, dict) else {}

        json_writer.write(atlas_dict, out / "atlas.json")

        if hasattr(atlas, "summary_statistics"):
            json_writer.write(
                atlas.summary_statistics(), out / "summary.json"
            )

        foundation = getattr(atlas, "foundation", None)
        if foundation is not None:
            if hasattr(foundation, "descriptors") and foundation.descriptors:
                csv_writer.write_descriptors(
                    foundation.descriptors, out / "descriptors.csv"
                )

            if (
                hasattr(foundation, "alignment_results")
                and foundation.alignment_results
            ):
                csv_writer.write_alignments(
                    foundation.alignment_results, out / "alignments.csv"
                )

            if self._include_arrays:
                arrays_dir = out / "arrays"
                arrays_dir.mkdir(exist_ok=True)

                if hasattr(foundation, "alignment_cost_matrix"):
                    cost_mat = foundation.alignment_cost_matrix
                    if cost_mat.size > 0:
                        np_writer.write_array(
                            cost_mat, arrays_dir / "alignment_costs.npy"
                        )

                if hasattr(foundation, "descriptor_matrix"):
                    desc_mat = foundation.descriptor_matrix
                    if desc_mat.size > 0:
                        np_writer.write_array(
                            desc_mat, arrays_dir / "descriptors.npy"
                        )

                for cid, scm in foundation.scm_results.items():
                    if hasattr(scm, "adjacency") and scm.adjacency is not None:
                        np_writer.write_array(
                            scm.adjacency,
                            arrays_dir / f"adj_{cid}.npy",
                        )
                    if hasattr(scm, "parameters") and scm.parameters is not None:
                        np_writer.write_array(
                            scm.parameters,
                            arrays_dir / f"params_{cid}.npy",
                        )

        exploration = getattr(atlas, "exploration", None)
        if exploration is not None:
            if hasattr(exploration, "to_dict"):
                json_writer.write(
                    exploration.to_dict(), out / "exploration.json"
                )

        validation = getattr(atlas, "validation", None)
        if validation is not None:
            if hasattr(validation, "to_dict"):
                json_writer.write(
                    validation.to_dict(), out / "validation.json"
                )

        config = getattr(atlas, "config", None)
        if config:
            json_writer.write(config, out / "config.json")

        logger.info("Atlas written to %s", out)
        return out.resolve()


# =====================================================================
# ReportWriter
# =====================================================================


class ReportWriter:
    """Generate markdown or HTML reports from atlas results.

    Parameters
    ----------
    format : str
        Output format ('markdown' or 'html').

    Examples
    --------
    >>> writer = ReportWriter(format="markdown")
    >>> writer.write(atlas, "report.md")
    """

    def __init__(self, format: str = "markdown") -> None:
        if format not in ("markdown", "html"):
            raise ValueError(
                f"format must be 'markdown' or 'html', got {format!r}"
            )
        self._format = format

    def write(
        self, atlas: Any, path: Union[str, Path]
    ) -> Path:
        """Generate and write a report.

        Parameters
        ----------
        atlas : AtlasResult
            Atlas results.
        path : str or Path
            Output file path.

        Returns
        -------
        Path
        """
        if self._format == "markdown":
            content = self._generate_markdown(atlas)
        else:
            content = self._generate_html(atlas)

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content)
        logger.info("Report written to %s", out)
        return out.resolve()

    def _generate_markdown(self, atlas: Any) -> str:
        """Generate a Markdown report."""
        lines: List[str] = []
        lines.append("# Causal-Plasticity Atlas Report")
        lines.append("")

        metadata = getattr(atlas, "metadata", {})
        lines.append("## Run Information")
        lines.append("")
        lines.append(f"- **Profile**: {metadata.get('profile', 'unknown')}")
        lines.append(f"- **Contexts**: {getattr(atlas, 'n_contexts', '?')}")
        lines.append(f"- **Variables**: {getattr(atlas, 'n_variables', '?')}")
        total_time = metadata.get("total_time", 0.0)
        lines.append(f"- **Total Time**: {total_time:.2f}s")
        lines.append("")

        if hasattr(atlas, "summary_statistics"):
            stats = atlas.summary_statistics()

            lines.append("## Summary Statistics")
            lines.append("")

            cls_summary = stats.get("classification_summary", {})
            if cls_summary:
                lines.append("### Classification Distribution")
                lines.append("")
                lines.append("| Class | Count |")
                lines.append("|-------|-------|")
                for cls, count in sorted(cls_summary.items()):
                    lines.append(f"| {cls} | {count} |")
                lines.append("")

            desc_stats = stats.get("descriptors", {})
            if desc_stats:
                lines.append("### Descriptor Statistics")
                lines.append("")
                lines.append("| Component | Mean | Std |")
                lines.append("|-----------|------|-----|")
                for comp in ["structural", "parametric", "emergence", "sensitivity"]:
                    mean = desc_stats.get(f"mean_{comp}", 0.0)
                    std = desc_stats.get(f"std_{comp}", 0.0)
                    lines.append(f"| {comp} | {mean:.4f} | {std:.4f} |")
                lines.append("")

            align_stats = stats.get("alignment", {})
            if align_stats:
                lines.append("### Alignment Statistics")
                lines.append("")
                lines.append(
                    f"- Mean cost: {align_stats.get('mean_cost', 0):.4f}"
                )
                lines.append(
                    f"- Median cost: {align_stats.get('median_cost', 0):.4f}"
                )
                lines.append("")

            expl_stats = stats.get("exploration", {})
            if expl_stats:
                lines.append("### Exploration Statistics")
                lines.append("")
                lines.append(
                    f"- Archive size: {expl_stats.get('archive_size', 0)}"
                )
                lines.append(
                    f"- Coverage: {expl_stats.get('coverage', 0):.2%}"
                )
                lines.append(
                    f"- QD Score: {expl_stats.get('qd_score', 0):.4f}"
                )
                lines.append("")

            val_stats = stats.get("validation", {})
            if val_stats:
                lines.append("### Validation Statistics")
                lines.append("")
                lines.append(
                    f"- Certified: {val_stats.get('n_certified', 0)}"
                )
                lines.append(
                    f"- Rate: {val_stats.get('certification_rate', 0):.2%}"
                )
                lines.append("")

        foundation = getattr(atlas, "foundation", None)
        if foundation is not None and hasattr(foundation, "descriptors"):
            lines.append("## Variable Details")
            lines.append("")
            lines.append(
                "| Variable | Structural | Parametric | "
                "Emergence | Sensitivity | Class |"
            )
            lines.append("|----------|-----------|-----------|"
                         "-----------|-------------|-------|")

            for var in getattr(foundation, "variable_names", []):
                dr = foundation.descriptors.get(var)
                if dr is not None:
                    cls_val = (
                        dr.classification.value
                        if hasattr(dr.classification, "value")
                        else str(dr.classification)
                    )
                    lines.append(
                        f"| {var} | {dr.structural:.4f} | "
                        f"{dr.parametric:.4f} | {dr.emergence:.4f} | "
                        f"{dr.sensitivity:.4f} | {cls_val} |"
                    )
            lines.append("")

        validation = getattr(atlas, "validation", None)
        if validation is not None:
            tp = getattr(validation, "tipping_points", None)
            if tp is not None and hasattr(tp, "validated_changepoints"):
                if tp.validated_changepoints:
                    lines.append("## Tipping Points")
                    lines.append("")
                    lines.append(
                        f"- Detected: {len(tp.validated_changepoints)}"
                    )
                    lines.append(
                        f"- Locations: {tp.validated_changepoints}"
                    )
                    lines.append("")

        lines.append("---")
        lines.append(
            f"*Generated by CPA v0.1.0 at "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}*"
        )
        lines.append("")

        return "\n".join(lines)

    def _generate_html(self, atlas: Any) -> str:
        """Generate an HTML report wrapping the markdown content."""
        md_content = self._generate_markdown(atlas)

        html_lines: List[str] = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<meta charset='utf-8'>",
            "<title>CPA Atlas Report</title>",
            "<style>",
            "body { font-family: system-ui, sans-serif; max-width: 900px; "
            "margin: 0 auto; padding: 20px; line-height: 1.6; }",
            "table { border-collapse: collapse; width: 100%; margin: 1em 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background: #f5f5f5; }",
            "h1 { color: #333; border-bottom: 2px solid #0366d6; padding-bottom: 8px; }",
            "h2 { color: #444; border-bottom: 1px solid #eee; padding-bottom: 4px; }",
            "code { background: #f6f8fa; padding: 2px 4px; border-radius: 3px; }",
            "</style>",
            "</head><body>",
        ]

        for line in md_content.split("\n"):
            line = line.rstrip()

            if line.startswith("# ") and not line.startswith("## "):
                html_lines.append(f"<h1>{_escape_html(line[2:])}</h1>")
            elif line.startswith("## "):
                html_lines.append(f"<h2>{_escape_html(line[3:])}</h2>")
            elif line.startswith("### "):
                html_lines.append(f"<h3>{_escape_html(line[4:])}</h3>")
            elif line.startswith("| ") and "---" not in line:
                cells = [c.strip() for c in line.split("|")[1:-1]]
                if all("-" in c for c in cells):
                    continue
                tag = "td"
                html_lines.append(
                    "<tr>"
                    + "".join(f"<{tag}>{_escape_html(c)}</{tag}>" for c in cells)
                    + "</tr>"
                )
            elif line.startswith("- "):
                content = line[2:]
                content = content.replace("**", "<strong>", 1).replace(
                    "**", "</strong>", 1
                )
                html_lines.append(f"<li>{content}</li>")
            elif line == "---":
                html_lines.append("<hr>")
            elif line.startswith("*") and line.endswith("*"):
                html_lines.append(f"<em>{_escape_html(line[1:-1])}</em>")
            elif line:
                html_lines.append(f"<p>{_escape_html(line)}</p>")

        html_lines.extend(["</body></html>", ""])
        return "\n".join(html_lines)


def _escape_html(text: str) -> str:
    """Basic HTML escaping."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
