"""TaintFlow CLI – command implementations.

Each public command class exposes three methods:

* :meth:`validate_args` – check that parsed arguments are consistent and
  that referenced files exist.  Returns a list of error strings (empty on
  success).
* :meth:`execute` – run the command logic and return an integer exit code.
* :meth:`format_output` – render results for terminal or file output.

Exit codes follow :mod:`taintflow.cli.main`:

* 0 – success / no leakage
* 1 – leakage detected
* 2 – user or runtime error
"""

from __future__ import annotations

import abc
import json
import os
import sys
import time
from argparse import Namespace
from dataclasses import asdict
from typing import Any, Dict, List, Optional, TextIO

from taintflow.cli.formatters import (
    DiffFormatter,
    JSONFormatter,
    SummaryFormatter,
    TableFormatter,
    TerminalFormatter,
    TreeFormatter,
)
from taintflow.cli.main import (
    EXIT_CLEAN,
    EXIT_ERROR,
    EXIT_LEAKAGE,
    ProgressReporter,
    _error,
    _info,
    _warn,
    color_bold,
    color_dim,
    color_green,
    color_red,
    severity_color,
)
from taintflow.core.config import TaintFlowConfig
from taintflow.core.types import LeakageReport, Severity

# ---------------------------------------------------------------------------
# Base command
# ---------------------------------------------------------------------------


class BaseCommand(abc.ABC):
    """Abstract base for all CLI commands."""

    @abc.abstractmethod
    def validate_args(self, args: Namespace) -> List[str]:
        """Return a list of validation error messages (empty = OK)."""

    @abc.abstractmethod
    def execute(self, args: Namespace) -> int:
        """Run the command and return an exit code."""

    def format_output(self, data: Any, *, fmt: str = "text", stream: TextIO = sys.stdout) -> None:
        """Render *data* in the requested format to *stream*."""
        if fmt == "json":
            JSONFormatter(stream=stream).write(data)
        else:
            stream.write(str(data))
            stream.write("\n")
            stream.flush()


# ---------------------------------------------------------------------------
# Result formatter (shared utility)
# ---------------------------------------------------------------------------


class ResultFormatter:
    """Format a :class:`LeakageReport` for terminal output."""

    def __init__(self, *, stream: TextIO = sys.stdout, verbose: int = 0) -> None:
        self.stream = stream
        self.verbose = verbose
        self._terminal = TerminalFormatter(stream=stream)
        self._table = TableFormatter(stream=stream)
        self._tree = TreeFormatter(stream=stream)
        self._summary = SummaryFormatter(stream=stream)

    def render(self, report: LeakageReport, *, fmt: str = "text") -> None:
        """Render *report* in the given format."""
        if fmt == "json":
            JSONFormatter(stream=self.stream).write(asdict(report))
        elif fmt == "text":
            self._render_text(report)
        elif fmt == "html":
            self._render_html(report)
        elif fmt == "sarif":
            self._render_sarif(report)
        else:
            self._render_text(report)

    # -- text ----------------------------------------------------------------

    def _render_text(self, report: LeakageReport) -> None:
        self._terminal.header("TaintFlow Audit Report")
        self._summary.write_report_summary(report)
        self.stream.write("\n")

        if not report.stage_leakages:
            self._terminal.write_line(color_green("✓ No leakage detected."))
            return

        self._terminal.header("Stage Details")
        sorted_stages = report.stages_by_severity()
        rows: List[List[str]] = []
        for stage in sorted_stages:
            sev = severity_color(stage.severity.value)
            rows.append([
                stage.stage_name,
                stage.op_type.value if hasattr(stage.op_type, "value") else str(stage.op_type),
                sev,
                f"{stage.max_bit_bound:.2f}",
                str(stage.n_leaking_features),
            ])
        self._table.write(
            headers=["Stage", "Operation", "Severity", "Max Bits", "Leaking Cols"],
            rows=rows,
        )

        if self.verbose >= 1:
            self.stream.write("\n")
            self._terminal.header("Feature Breakdown")
            for stage in sorted_stages:
                if not stage.feature_leakages:
                    continue
                self._tree.write_stage_tree(stage)

    # -- html ----------------------------------------------------------------

    def _render_html(self, report: LeakageReport) -> None:
        html_parts: List[str] = [
            "<!DOCTYPE html>",
            "<html lang='en'><head><meta charset='utf-8'>",
            "<title>TaintFlow Report</title>",
            "<style>",
            "body{font-family:system-ui,sans-serif;margin:2em}",
            "table{border-collapse:collapse;width:100%}",
            "th,td{border:1px solid #ccc;padding:6px 10px;text-align:left}",
            "th{background:#f5f5f5}",
            ".critical{color:#d32f2f}.warning{color:#f9a825}.negligible{color:#388e3c}",
            "</style></head><body>",
            f"<h1>TaintFlow Audit – {_html_escape(report.pipeline_name)}</h1>",
            f"<p>Overall severity: <strong class='{report.overall_severity.value}'>"
            f"{report.overall_severity.value}</strong></p>",
            f"<p>Total bit bound: {report.total_bit_bound:.2f} bits</p>",
            f"<p>Leaking features: {report.n_leaking_features}/{report.n_features}</p>",
        ]
        if report.stage_leakages:
            html_parts.append("<table><thead><tr>")
            for hdr in ("Stage", "Operation", "Severity", "Max Bits", "Leaking Cols"):
                html_parts.append(f"<th>{hdr}</th>")
            html_parts.append("</tr></thead><tbody>")
            for stage in report.stages_by_severity():
                sev = stage.severity.value
                html_parts.append(
                    f"<tr><td>{_html_escape(stage.stage_name)}</td>"
                    f"<td>{_html_escape(str(getattr(stage.op_type, 'value', stage.op_type)))}</td>"
                    f"<td class='{sev}'>{sev}</td>"
                    f"<td>{stage.max_bit_bound:.2f}</td>"
                    f"<td>{stage.n_leaking_features}</td></tr>"
                )
            html_parts.append("</tbody></table>")
        else:
            html_parts.append("<p style='color:#388e3c'>✓ No leakage detected.</p>")
        html_parts.append(f"<footer><small>Generated by TaintFlow at {report.timestamp}</small></footer>")
        html_parts.append("</body></html>")
        self.stream.write("\n".join(html_parts) + "\n")
        self.stream.flush()

    # -- sarif ---------------------------------------------------------------

    def _render_sarif(self, report: LeakageReport) -> None:
        sarif: Dict[str, Any] = {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "TaintFlow",
                            "informationUri": "https://github.com/taintflow/taintflow",
                            "rules": [],
                        }
                    },
                    "results": [],
                }
            ],
        }
        run = sarif["runs"][0]
        severity_to_sarif = {
            "negligible": "note",
            "warning": "warning",
            "critical": "error",
        }
        for stage in report.stage_leakages:
            for feat in stage.feature_leakages:
                run["results"].append({
                    "ruleId": "taintflow/leakage",
                    "level": severity_to_sarif.get(feat.severity.value, "note"),
                    "message": {
                        "text": (
                            f"Feature '{feat.column_name}' leaks "
                            f"{feat.bit_bound:.2f} bits from "
                            f"{', '.join(o.value for o in feat.origins)} "
                            f"in stage '{stage.stage_name}'."
                        ),
                    },
                    "properties": {
                        "severity": feat.severity.value,
                        "bit_bound": feat.bit_bound,
                        "stage": stage.stage_name,
                    },
                })
        json.dump(sarif, self.stream, indent=2)
        self.stream.write("\n")
        self.stream.flush()


def _html_escape(text: str) -> str:
    """Minimal HTML entity escaping."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ---------------------------------------------------------------------------
# AuditCommand
# ---------------------------------------------------------------------------


class AuditCommand(BaseCommand):
    """Run a full leakage audit on a pipeline script."""

    def validate_args(self, args: Namespace) -> List[str]:
        """Validate audit-specific arguments."""
        errors: List[str] = []
        path = getattr(args, "input", None)
        if not path:
            errors.append("--input is required")
        elif not os.path.isfile(path):
            errors.append(f"input file not found: {path}")
        config_path = getattr(args, "config", None)
        if config_path and not os.path.isfile(config_path):
            errors.append(f"config file not found: {config_path}")
        max_iter = getattr(args, "max_iterations", None)
        if max_iter is not None and max_iter < 1:
            errors.append("--max-iterations must be >= 1")
        return errors

    def execute(self, args: Namespace) -> int:
        """Execute the audit pipeline and return an exit code."""
        verbose: int = getattr(args, "verbose", 0)
        if verbose:
            _info(f"loading pipeline from {args.input}")

        # -- load configuration ------------------------------------------------
        cli_overrides: Dict[str, Any] = {}
        if getattr(args, "max_iterations", None) is not None:
            cli_overrides["max_iterations"] = args.max_iterations
        cli_overrides["verbosity"] = verbose

        try:
            config = TaintFlowConfig.load(
                config_path=getattr(args, "config", None),
                cli_overrides=cli_overrides if cli_overrides else None,
            )
        except Exception as exc:
            _error(f"failed to load configuration: {exc}")

        # -- read pipeline script ----------------------------------------------
        try:
            with open(args.input, "r", encoding="utf-8") as fh:
                pipeline_source = fh.read()
        except OSError as exc:
            _error(f"cannot read pipeline script: {exc}")

        pipeline_name = os.path.splitext(os.path.basename(args.input))[0]

        # -- run analysis phases -----------------------------------------------
        progress = ProgressReporter(6, label="audit")
        start_time = time.monotonic()

        progress.update("parsing pipeline")
        pipeline_ast = _parse_pipeline(pipeline_source, args.input)

        progress.update("setting up instrumentation")
        instrumented = _instrument_pipeline(pipeline_ast, config)

        progress.update("executing pipeline")
        skip_empirical = getattr(args, "no_empirical", False)
        trace = _execute_pipeline(instrumented, config, skip_empirical=skip_empirical)

        progress.update("building DAG & estimating capacities")
        dag = _build_dag(trace, config)

        progress.update("running taint propagation")
        analysis_result = _run_analysis(dag, config)

        progress.update("generating report")
        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        report = _build_report(analysis_result, pipeline_name, config, elapsed_ms)
        progress.finish("complete")

        # -- determine minimum severity filter ---------------------------------
        min_severity = _parse_severity(getattr(args, "severity", "negligible"))
        report = _filter_report(report, min_severity)

        # -- output ------------------------------------------------------------
        out_fmt: str = getattr(args, "format", "text")
        out_path: Optional[str] = getattr(args, "output", None)
        formatter = ResultFormatter(verbose=verbose)

        if out_path:
            try:
                with open(out_path, "w", encoding="utf-8") as fh:
                    ResultFormatter(stream=fh, verbose=verbose).render(report, fmt=out_fmt)
                _info(f"report written to {out_path}")
            except OSError as exc:
                _error(f"cannot write report: {exc}")
        else:
            formatter.render(report, fmt=out_fmt)

        return EXIT_LEAKAGE if not report.is_clean else EXIT_CLEAN


# -- audit helpers (thin wrappers around core modules) -----------------------


def _parse_pipeline(source: str, filename: str) -> Any:
    """Parse a pipeline script into an AST representation."""
    import ast as _ast

    try:
        tree = _ast.parse(source, filename=filename)
    except SyntaxError as exc:
        _error(f"syntax error in pipeline script: {exc}")
    return tree


def _instrument_pipeline(tree: Any, config: TaintFlowConfig) -> Any:
    """Return an instrumented version of the pipeline AST."""
    try:
        from taintflow.instrument import instrument_ast  # type: ignore[import-untyped]

        return instrument_ast(tree, config)
    except ImportError:
        return tree


def _execute_pipeline(instrumented: Any, config: TaintFlowConfig, *, skip_empirical: bool = False) -> Any:
    """Execute the instrumented pipeline and collect a trace."""
    try:
        from taintflow.instrument import collect_trace  # type: ignore[import-untyped]

        return collect_trace(instrumented, config, skip_empirical=skip_empirical)
    except ImportError:
        return {}


def _build_dag(trace: Any, config: TaintFlowConfig) -> Any:
    """Construct the dataflow DAG from the collected trace."""
    try:
        from taintflow.dag import build_dag  # type: ignore[import-untyped]

        return build_dag(trace, config)
    except ImportError:
        return {}


def _run_analysis(dag: Any, config: TaintFlowConfig) -> Any:
    """Run taint propagation on the DAG and return raw analysis results."""
    try:
        from taintflow.analysis import run_analysis  # type: ignore[import-untyped]

        return run_analysis(dag, config)
    except ImportError:
        return {}


def _build_report(
    analysis_result: Any,
    pipeline_name: str,
    config: TaintFlowConfig,
    elapsed_ms: float,
) -> LeakageReport:
    """Build a :class:`LeakageReport` from raw analysis results."""
    try:
        from taintflow.report import build_report  # type: ignore[import-untyped]

        return build_report(analysis_result, pipeline_name, config, elapsed_ms)
    except ImportError:
        return LeakageReport(
            pipeline_name=pipeline_name,
            analysis_duration_ms=elapsed_ms,
            config_snapshot=config.to_dict(),
        )


def _parse_severity(value: str) -> Severity:
    """Convert a CLI severity string to a :class:`Severity` enum member."""
    mapping = {
        "negligible": Severity.NEGLIGIBLE,
        "warning": Severity.WARNING,
        "critical": Severity.CRITICAL,
    }
    return mapping.get(value.lower(), Severity.NEGLIGIBLE)


def _filter_report(report: LeakageReport, min_severity: Severity) -> LeakageReport:
    """Remove stages / features below *min_severity* from *report*."""
    if min_severity == Severity.NEGLIGIBLE:
        return report

    filtered_stages = []
    total_leaking = 0
    for stage in report.stage_leakages:
        kept = [f for f in stage.feature_leakages if f.severity >= min_severity]
        if not kept:
            continue
        stage.feature_leakages = kept
        stage.max_bit_bound = max(f.bit_bound for f in kept)
        stage.mean_bit_bound = sum(f.bit_bound for f in kept) / len(kept)
        stage.severity = max(f.severity for f in kept)
        total_leaking += len(kept)
        filtered_stages.append(stage)

    report.stage_leakages = filtered_stages
    report.n_leaking_features = total_leaking
    if filtered_stages:
        report.overall_severity = max(s.severity for s in filtered_stages)
        report.total_bit_bound = sum(s.max_bit_bound for s in filtered_stages)
    else:
        report.overall_severity = Severity.NEGLIGIBLE
        report.total_bit_bound = 0.0
    return report


# ---------------------------------------------------------------------------
# ScanCommand
# ---------------------------------------------------------------------------


class ScanCommand(BaseCommand):
    """Quick static scan for common leakage patterns (no execution)."""

    # Known built-in patterns
    BUILTIN_PATTERNS = (
        "fit_before_split",
        "target_in_features",
        "test_in_train_index",
        "global_scaling",
        "feature_selection_leakage",
        "duplicated_rows",
        "temporal_leakage",
    )

    def validate_args(self, args: Namespace) -> List[str]:
        errors: List[str] = []
        path = getattr(args, "input", None)
        if not path:
            errors.append("--input is required")
        elif not os.path.isfile(path):
            errors.append(f"input file not found: {path}")
        return errors

    def execute(self, args: Namespace) -> int:
        """Run pattern-based scan."""
        pattern_arg: str = getattr(args, "patterns", "all")
        if pattern_arg.lower() == "all":
            patterns = list(self.BUILTIN_PATTERNS)
        else:
            patterns = [p.strip() for p in pattern_arg.split(",") if p.strip()]
            unknown = [p for p in patterns if p not in self.BUILTIN_PATTERNS]
            if unknown:
                _warn(f"unknown patterns ignored: {', '.join(unknown)}")
                patterns = [p for p in patterns if p in self.BUILTIN_PATTERNS]
            if not patterns:
                _error("no valid patterns specified")

        try:
            with open(args.input, "r", encoding="utf-8") as fh:
                source = fh.read()
        except OSError as exc:
            _error(f"cannot read pipeline script: {exc}")

        _info(f"scanning {args.input} for {len(patterns)} pattern(s)")
        hits = self._scan_source(source, patterns)

        if hits:
            sys.stdout.write(color_bold("Scan results:\n"))
            for name, description, line in hits:
                loc = f"line {line}" if line else "unknown location"
                sys.stdout.write(f"  {severity_color('warning')} {name} ({loc}): {description}\n")
            sys.stdout.write(f"\n{len(hits)} potential issue(s) found.\n")
            return EXIT_LEAKAGE

        sys.stdout.write(color_green("✓ No common leakage patterns detected.\n"))
        return EXIT_CLEAN

    def _scan_source(
        self, source: str, patterns: List[str]
    ) -> List[tuple[str, str, int]]:
        """Return list of ``(pattern_name, description, line_number)`` hits."""
        import re

        hits: List[tuple[str, str, int]] = []
        lines = source.splitlines()

        pattern_checks: Dict[str, tuple[str, str]] = {
            "fit_before_split": (
                r"\.fit\s*\(",
                "Model fitting detected before train/test split",
            ),
            "target_in_features": (
                r"(target|label|y_true)\s*=",
                "Target variable may be included in feature matrix",
            ),
            "test_in_train_index": (
                r"(iloc|loc)\[.*test.*\].*train",
                "Test indices may overlap with training data",
            ),
            "global_scaling": (
                r"(StandardScaler|MinMaxScaler|RobustScaler)\(\)\.fit\(",
                "Scaler fitted on full dataset before split",
            ),
            "feature_selection_leakage": (
                r"(SelectKBest|SelectFromModel|VarianceThreshold)\(\)\.fit\(",
                "Feature selection fitted on full dataset",
            ),
            "duplicated_rows": (
                r"\.duplicated\(",
                "Duplicate check may not account for train/test partition",
            ),
            "temporal_leakage": (
                r"(shift|lag|rolling)\s*\(",
                "Time-series operation may introduce future leakage",
            ),
        }

        for pat_name in patterns:
            if pat_name not in pattern_checks:
                continue
            regex, description = pattern_checks[pat_name]
            compiled = re.compile(regex)
            for idx, line in enumerate(lines, start=1):
                if compiled.search(line):
                    hits.append((pat_name, description, idx))

        return hits

    def format_output(self, data: Any, *, fmt: str = "text", stream: TextIO = sys.stdout) -> None:
        stream.write(json.dumps(data, indent=2) if fmt == "json" else str(data))
        stream.write("\n")
        stream.flush()


# ---------------------------------------------------------------------------
# ReportCommand
# ---------------------------------------------------------------------------


class ReportCommand(BaseCommand):
    """Regenerate a report from saved analysis results."""

    def validate_args(self, args: Namespace) -> List[str]:
        errors: List[str] = []
        path = getattr(args, "input", None)
        if not path:
            errors.append("--input is required")
        elif not os.path.isfile(path):
            errors.append(f"input file not found: {path}")
        return errors

    def execute(self, args: Namespace) -> int:
        """Load results JSON and re-render in the requested format."""
        try:
            with open(args.input, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            _error(f"cannot load results file: {exc}")

        report = self._dict_to_report(data)

        out_fmt: str = getattr(args, "format", "text")
        out_path: Optional[str] = getattr(args, "output", None)

        if out_path:
            try:
                with open(out_path, "w", encoding="utf-8") as fh:
                    ResultFormatter(stream=fh).render(report, fmt=out_fmt)
                _info(f"report written to {out_path}")
            except OSError as exc:
                _error(f"cannot write report: {exc}")
        else:
            ResultFormatter().render(report, fmt=out_fmt)

        return EXIT_LEAKAGE if not report.is_clean else EXIT_CLEAN

    @staticmethod
    def _dict_to_report(data: Dict[str, Any]) -> LeakageReport:
        """Reconstruct a :class:`LeakageReport` from a plain dictionary."""
        from taintflow.core.types import (
            FeatureLeakage,
            NodeKind,
            OpType,
            Origin,
            StageLeakage,
        )

        stage_leakages: List[StageLeakage] = []
        for sd in data.get("stage_leakages", []):
            features = []
            for fd in sd.get("feature_leakages", []):
                origins = frozenset(
                    Origin(o) for o in fd.get("origins", [])
                )
                features.append(FeatureLeakage(
                    column_name=fd.get("column_name", ""),
                    bit_bound=fd.get("bit_bound", 0.0),
                    severity=Severity(fd.get("severity", "negligible")),
                    origins=origins,
                    contributing_stages=fd.get("contributing_stages", []),
                    remediation=fd.get("remediation", ""),
                    explanation=fd.get("explanation", ""),
                    confidence=fd.get("confidence", 1.0),
                ))
            try:
                op_type = OpType(sd.get("op_type", "UNKNOWN"))
            except (ValueError, KeyError):
                op_type = OpType.UNKNOWN
            try:
                node_kind = NodeKind(sd.get("node_kind", "UNKNOWN"))
            except (ValueError, KeyError):
                node_kind = NodeKind.UNKNOWN
            stage_leakages.append(StageLeakage(
                stage_id=sd.get("stage_id", ""),
                stage_name=sd.get("stage_name", ""),
                op_type=op_type,
                node_kind=node_kind,
                max_bit_bound=sd.get("max_bit_bound", 0.0),
                mean_bit_bound=sd.get("mean_bit_bound", 0.0),
                feature_leakages=features,
                severity=Severity(sd.get("severity", "negligible")),
                description=sd.get("description", ""),
            ))

        return LeakageReport(
            pipeline_name=data.get("pipeline_name", "unknown"),
            timestamp=data.get("timestamp", ""),
            overall_severity=Severity(data.get("overall_severity", "negligible")),
            total_bit_bound=data.get("total_bit_bound", 0.0),
            n_stages=data.get("n_stages", len(stage_leakages)),
            n_features=data.get("n_features", 0),
            n_leaking_features=data.get("n_leaking_features", 0),
            stage_leakages=stage_leakages,
            metadata=data.get("metadata", {}),
            config_snapshot=data.get("config_snapshot", {}),
            analysis_duration_ms=data.get("analysis_duration_ms", 0.0),
        )


# ---------------------------------------------------------------------------
# CompareCommand
# ---------------------------------------------------------------------------


class CompareCommand(BaseCommand):
    """Compare two analysis result sets and show regressions / improvements."""

    def validate_args(self, args: Namespace) -> List[str]:
        errors: List[str] = []
        for attr in ("before", "after"):
            path = getattr(args, attr, None)
            if not path:
                errors.append(f"--{attr} is required")
            elif not os.path.isfile(path):
                errors.append(f"{attr} file not found: {path}")
        return errors

    def execute(self, args: Namespace) -> int:
        """Load two result files, compare, and print a diff summary."""
        try:
            with open(args.before, "r", encoding="utf-8") as fh:
                before_data = json.load(fh)
            with open(args.after, "r", encoding="utf-8") as fh:
                after_data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            _error(f"cannot load results file: {exc}")

        before = ReportCommand._dict_to_report(before_data)
        after = ReportCommand._dict_to_report(after_data)

        diff = self._compute_diff(before, after)
        DiffFormatter(stream=sys.stdout).write(diff)

        has_regression = any(
            item.get("change") == "regression" for item in diff.get("changes", [])
        )
        return EXIT_LEAKAGE if has_regression else EXIT_CLEAN

    @staticmethod
    def _compute_diff(
        before: LeakageReport, after: LeakageReport
    ) -> Dict[str, Any]:
        """Compute a structured diff between two reports."""
        before_features: Dict[str, float] = {}
        after_features: Dict[str, float] = {}

        for stage in before.stage_leakages:
            for feat in stage.feature_leakages:
                key = f"{stage.stage_name}::{feat.column_name}"
                before_features[key] = feat.bit_bound

        for stage in after.stage_leakages:
            for feat in stage.feature_leakages:
                key = f"{stage.stage_name}::{feat.column_name}"
                after_features[key] = feat.bit_bound

        all_keys = sorted(set(before_features) | set(after_features))
        changes: List[Dict[str, Any]] = []
        for key in all_keys:
            b_val = before_features.get(key, 0.0)
            a_val = after_features.get(key, 0.0)
            delta = a_val - b_val
            if abs(delta) < 1e-9:
                change_type = "unchanged"
            elif key not in before_features:
                change_type = "new"
            elif key not in after_features:
                change_type = "fixed"
            elif delta > 0:
                change_type = "regression"
            else:
                change_type = "improvement"

            changes.append({
                "feature": key,
                "before_bits": b_val,
                "after_bits": a_val,
                "delta": delta,
                "change": change_type,
            })

        return {
            "before_pipeline": before.pipeline_name,
            "after_pipeline": after.pipeline_name,
            "before_severity": before.overall_severity.value,
            "after_severity": after.overall_severity.value,
            "before_total_bits": before.total_bit_bound,
            "after_total_bits": after.total_bit_bound,
            "changes": changes,
            "summary": {
                "regressions": sum(1 for c in changes if c["change"] == "regression"),
                "improvements": sum(1 for c in changes if c["change"] == "improvement"),
                "new": sum(1 for c in changes if c["change"] == "new"),
                "fixed": sum(1 for c in changes if c["change"] == "fixed"),
                "unchanged": sum(1 for c in changes if c["change"] == "unchanged"),
            },
        }


# ---------------------------------------------------------------------------
# ConfigCommand
# ---------------------------------------------------------------------------


class ConfigCommand(BaseCommand):
    """Manage TaintFlow configuration files."""

    def validate_args(self, args: Namespace) -> List[str]:
        errors: List[str] = []
        action = getattr(args, "config_action", None)
        if not action:
            errors.append("config subcommand required (init | show | validate)")
        return errors

    def execute(self, args: Namespace) -> int:
        action = args.config_action
        if action == "init":
            return self._init_config()
        if action == "show":
            return self._show_config()
        if action == "validate":
            return self._validate_config(getattr(args, "file", None))
        _error(f"unknown config action: {action}")

    def _init_config(self) -> int:
        """Write a default ``taintflow.toml`` to the current directory."""
        target = os.path.join(os.getcwd(), "taintflow.toml")
        if os.path.exists(target):
            _warn(f"{target} already exists – not overwriting")
            return EXIT_ERROR

        default_toml = _default_toml()
        try:
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(default_toml)
            _info(f"created {target}")
        except OSError as exc:
            _error(f"cannot write config file: {exc}")
        return EXIT_CLEAN

    def _show_config(self) -> int:
        """Load and display the resolved configuration."""
        try:
            config = TaintFlowConfig.load()
        except Exception as exc:
            _error(f"failed to load configuration: {exc}")
        sys.stdout.write(config.summary())
        sys.stdout.write("\n")
        return EXIT_CLEAN

    def _validate_config(self, path: Optional[str]) -> int:
        """Validate a configuration file and report errors."""
        try:
            if path:
                if path.endswith(".toml"):
                    config = TaintFlowConfig.from_toml(path)
                else:
                    config = TaintFlowConfig.from_json(path)
            else:
                config = TaintFlowConfig.load()
        except Exception as exc:
            _error(f"cannot load config: {exc}")

        errors = config.validate()
        if errors:
            sys.stderr.write(color_red("Configuration errors:\n"))
            for err in errors:
                sys.stderr.write(f"  • {err}\n")
            return EXIT_ERROR

        sys.stdout.write(color_green("✓ Configuration is valid.\n"))
        return EXIT_CLEAN


def _default_toml() -> str:
    """Return the contents of a default ``taintflow.toml``."""
    return """\
# TaintFlow configuration
# See https://taintflow.readthedocs.io/en/latest/configuration.html

[analysis]
b_max = 64.0
alpha = 0.05
max_iterations = 1000
use_widening = true
widening_delay = 3
use_narrowing = true
narrowing_iterations = 5
epsilon = 1e-10

[execution]
parallel = false
n_workers = 1
verbosity = 1

[severity]
negligible_max = 1.0
warning_max = 8.0

[channel]
tier_preference = "analytic"
fallback_tier = "sampling"
analytic_timeout_ms = 5000
sampling_n_samples = 10000
sampling_seed = 42

[instrumentation]
trace_depth = 50
include_numpy = true
include_pandas = true
include_sklearn = true
record_shapes = true
record_dtypes = true

[report]
format = "text"
include_remediation = true
include_summary = true
sort_by = "severity"
"""


# ---------------------------------------------------------------------------
# PluginsCommand
# ---------------------------------------------------------------------------


class PluginsCommand(BaseCommand):
    """List installed TaintFlow plugins."""

    def validate_args(self, args: Namespace) -> List[str]:
        return []

    def execute(self, args: Namespace) -> int:
        """Discover and display installed plugins."""
        plugins = self._discover_plugins()

        if not plugins:
            sys.stdout.write(color_dim("No TaintFlow plugins installed.\n"))
            return EXIT_CLEAN

        sys.stdout.write(color_bold("Installed plugins:\n\n"))
        rows: List[List[str]] = []
        for name, version, description in plugins:
            rows.append([name, version, description])
        TableFormatter(stream=sys.stdout).write(
            headers=["Name", "Version", "Description"],
            rows=rows,
        )
        return EXIT_CLEAN

    @staticmethod
    def _discover_plugins() -> List[tuple[str, str, str]]:
        """Return ``(name, version, description)`` for each installed plugin."""
        plugins: List[tuple[str, str, str]] = []
        try:
            from importlib.metadata import entry_points

            eps = entry_points()
            taintflow_eps = eps.get("taintflow.plugins", []) if isinstance(eps, dict) else [
                ep for ep in eps if ep.group == "taintflow.plugins"
            ]
            for ep in taintflow_eps:
                try:
                    plugin_mod = ep.load()
                    version = getattr(plugin_mod, "__version__", "unknown")
                    doc = getattr(plugin_mod, "__doc__", "") or ""
                    description = doc.strip().split("\n")[0] if doc else ""
                    plugins.append((ep.name, version, description))
                except Exception:
                    plugins.append((ep.name, "error", "failed to load"))
        except Exception:
            pass
        return plugins
