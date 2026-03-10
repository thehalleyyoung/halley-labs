"""
Click CLI entry point for CausalCert.

Provides commands:
  causalcert audit     -- Full structural-robustness audit
  causalcert fragility -- Quick fragility scan
  causalcert radius    -- Compute robustness radius only
  causalcert validate  -- Validate DAG-data consistency
  causalcert report    -- Generate report from saved results
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import click
import numpy as np

# ---------------------------------------------------------------------------
# Shared option groups
# ---------------------------------------------------------------------------

_SOLVER_CHOICES = click.Choice(["auto", "ilp", "lp", "fpt", "cdcl"])
_FORMAT_CHOICES = click.Choice(["json", "html", "latex", "table"])
_CI_CHOICES = click.Choice([
    "ensemble", "partial_correlation", "kernel", "rank", "crt",
])


def _resolve_variable(name: str, node_names: list[str]) -> int:
    """Resolve a variable name-or-index to an integer index."""
    try:
        return int(name)
    except ValueError:
        pass
    # Look up by name
    for i, n in enumerate(node_names):
        if n == name:
            return i
    raise click.BadParameter(
        f"Variable {name!r} not found.  Available: {node_names}"
    )


def _load_dag(path: str) -> tuple[Any, list[str]]:
    """Load DAG from file, returning (adj_matrix, node_names)."""
    from causalcert.data.dag_io import load_dag
    adj, names = load_dag(path)
    return adj, names


def _load_data(path: str) -> Any:
    """Load data from file."""
    from causalcert.data.loader import load_auto
    return load_auto(path)


def _setup_logging(verbose: bool, quiet: bool = False) -> None:
    """Configure logging based on verbosity."""
    from causalcert.pipeline.logging_config import configure_logging
    if quiet:
        configure_logging(level="WARNING", json_format=False)
    elif verbose:
        configure_logging(level="DEBUG", json_format=False)
    else:
        configure_logging(level="INFO", json_format=False)


def _print_json(obj: dict) -> None:
    """Pretty-print a dict as JSON."""
    click.echo(json.dumps(obj, indent=2, default=str))


def _print_table(rows: list[dict], columns: list[str]) -> None:
    """Print a simple ASCII table."""
    widths = {c: len(c) for c in columns}
    for row in rows:
        for c in columns:
            widths[c] = max(widths[c], len(str(row.get(c, ""))))

    header = " | ".join(c.ljust(widths[c]) for c in columns)
    sep = "-+-".join("-" * widths[c] for c in columns)
    click.echo(header)
    click.echo(sep)
    for row in rows:
        line = " | ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns)
        click.echo(line)


# ---------------------------------------------------------------------------
# Main group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="causalcert")
@click.option("--config", "config_file", default=None,
              type=click.Path(exists=True),
              help="Path to YAML/JSON configuration file.")
@click.pass_context
def main(ctx: click.Context, config_file: str | None) -> None:
    """CausalCert — Causal Robustness Radii.

    Compute the minimum number of edge edits that would overturn a causal
    conclusion derived from an assumed DAG and observational data.
    """
    ctx.ensure_object(dict)
    if config_file:
        from causalcert.pipeline.config import PipelineRunConfig
        ctx.obj["config_file"] = config_file
        ctx.obj["base_config"] = PipelineRunConfig.from_file(config_file)
    else:
        ctx.obj["config_file"] = None
        ctx.obj["base_config"] = None


# ---------------------------------------------------------------------------
# audit command
# ---------------------------------------------------------------------------


@main.command()
@click.option("--dag", required=True, type=click.Path(exists=True),
              help="Path to DAG file (DOT, JSON, or CSV).")
@click.option("--data", required=True, type=click.Path(exists=True),
              help="Path to data file (CSV or Parquet).")
@click.option("--treatment", required=True, type=str,
              help="Treatment variable name or index.")
@click.option("--outcome", required=True, type=str,
              help="Outcome variable name or index.")
@click.option("--alpha", default=0.05, type=float,
              help="Significance level for CI tests.")
@click.option("--max-k", default=10, type=int,
              help="Maximum edit distance to search.")
@click.option("--solver", default="auto", type=_SOLVER_CHOICES,
              help="Solver strategy.")
@click.option("--ci-method", default="ensemble", type=_CI_CHOICES,
              help="CI test method.")
@click.option("--fdr-method", default="by",
              type=click.Choice(["by", "bh", "bonferroni", "holm"]),
              help="Multiplicity correction method.")
@click.option("--n-folds", default=5, type=int,
              help="Cross-fitting folds for estimation.")
@click.option("--n-jobs", default=1, type=int,
              help="Number of parallel workers (-1 = all CPUs).")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Output directory.")
@click.option("--format", "fmt", default="json", type=_FORMAT_CHOICES,
              help="Report format.")
@click.option("--seed", default=42, type=int,
              help="Random seed.")
@click.option("--cache-dir", default=".causalcert_cache", type=str,
              help="Cache directory (empty string to disable).")
@click.option("--verbose", "-v", is_flag=True,
              help="Verbose output.")
@click.option("--quiet", "-q", is_flag=True,
              help="Suppress non-error output.")
@click.option("--no-cache", is_flag=True, help="Disable caching.")
@click.pass_context
def audit(
    ctx: click.Context,
    dag: str,
    data: str,
    treatment: str,
    outcome: str,
    alpha: float,
    max_k: int,
    solver: str,
    ci_method: str,
    fdr_method: str,
    n_folds: int,
    n_jobs: int,
    output: str | None,
    fmt: str,
    seed: int,
    cache_dir: str,
    verbose: bool,
    quiet: bool,
    no_cache: bool,
) -> None:
    """Run a full structural-robustness audit.

    Loads the DAG and data, runs CI tests, computes fragility scores,
    solves for the robustness radius, estimates treatment effects, and
    generates an audit report.

    \b
    Examples:
      causalcert audit --dag dag.dot --data data.csv --treatment X --outcome Y
      causalcert audit --dag dag.json --data data.parquet -t 0 -y 3 --solver ilp
    """
    _setup_logging(verbose, quiet)

    from causalcert.pipeline.config import PipelineRunConfig
    from causalcert.pipeline.orchestrator import CausalCertPipeline
    from causalcert.types import CITestMethod, SolverStrategy

    # Load inputs
    if not quiet:
        click.echo(f"Loading DAG from {dag} ...")
    adj_matrix, node_names = _load_dag(dag)

    if not quiet:
        click.echo(f"Loading data from {data} ...")
    df = _load_data(data)

    # Resolve variable names
    treatment_idx = _resolve_variable(treatment, node_names)
    outcome_idx = _resolve_variable(outcome, node_names)

    if not quiet:
        click.echo(
            f"DAG: {adj_matrix.shape[0]} nodes, {int(adj_matrix.sum())} edges  |  "
            f"Data: {df.shape[0]} rows × {df.shape[1]} cols"
        )
        click.echo(
            f"Treatment: {node_names[treatment_idx]} ({treatment_idx})  |  "
            f"Outcome: {node_names[outcome_idx]} ({outcome_idx})"
        )

    solver_map = {
        "auto": SolverStrategy.AUTO,
        "ilp": SolverStrategy.ILP,
        "lp": SolverStrategy.LP_RELAXATION,
        "fpt": SolverStrategy.FPT,
        "cdcl": SolverStrategy.CDCL,
    }
    ci_map = {
        "ensemble": CITestMethod.ENSEMBLE,
        "partial_correlation": CITestMethod.PARTIAL_CORRELATION,
        "kernel": CITestMethod.KERNEL,
        "rank": CITestMethod.RANK,
        "crt": CITestMethod.CRT,
    }

    # Build config (merge with file config if provided)
    base = ctx.obj.get("base_config")
    if base is None:
        base = PipelineRunConfig()

    base.treatment = treatment_idx
    base.outcome = outcome_idx
    base.alpha = alpha
    base.max_k = max_k
    base.solver_strategy = solver_map[solver]
    base.ci_method = ci_map[ci_method]
    base.fdr_method = fdr_method
    base.n_folds = n_folds
    base.n_jobs = n_jobs
    base.seed = seed
    base.verbose = verbose
    base.output_dir = output
    base.report_formats = [fmt] if fmt != "table" else ["json"]
    base.cache_dir = None if no_cache else (cache_dir or None)

    # Validate config
    issues = base.validate()
    if issues:
        for issue in issues:
            click.echo(f"  ⚠ {issue}", err=True)
        raise click.Abort()

    # Run pipeline
    pipeline = CausalCertPipeline(base)
    t0 = time.perf_counter()
    report = pipeline.run(adj_matrix, df)
    elapsed = time.perf_counter() - t0

    # Output
    result_dict = _audit_report_to_dict(report)
    result_dict["elapsed_s"] = round(elapsed, 3)

    if fmt == "table":
        _print_audit_table(result_dict, node_names, verbose)
    else:
        _print_json(result_dict)

    if output:
        out_path = Path(output)
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / "audit_report.json"
        out_file.write_text(json.dumps(result_dict, indent=2, default=str))
        if not quiet:
            click.echo(f"\nReport written to {out_file}")


def _audit_report_to_dict(report: Any) -> dict[str, Any]:
    """Convert an AuditReport to a JSON-serialisable dict."""
    d: dict[str, Any] = {
        "treatment": report.treatment,
        "outcome": report.outcome,
        "n_nodes": report.n_nodes,
        "n_edges": report.n_edges,
        "radius": {
            "lower_bound": report.radius.lower_bound,
            "upper_bound": report.radius.upper_bound,
            "certified": report.radius.certified,
            "gap": report.radius.gap,
            "solver_strategy": report.radius.solver_strategy.value,
            "solver_time_s": round(report.radius.solver_time_s, 3),
        },
        "n_fragile_edges": len(report.fragility_ranking),
        "n_ci_tests": len(report.ci_results),
    }
    if report.fragility_ranking:
        d["top_fragile_edges"] = [
            {
                "edge": list(fs.edge),
                "total_score": round(fs.total_score, 4),
            }
            for fs in report.fragility_ranking[:10]
        ]
    if report.baseline_estimate is not None:
        d["baseline_ate"] = round(report.baseline_estimate.ate, 4)
        d["baseline_se"] = round(report.baseline_estimate.se, 4)
        d["baseline_ci"] = [
            round(report.baseline_estimate.ci_lower, 4),
            round(report.baseline_estimate.ci_upper, 4),
        ]
    if report.perturbed_estimates:
        d["n_perturbed_estimates"] = len(report.perturbed_estimates)
    d["metadata"] = {k: str(v) for k, v in report.metadata.items()}
    return d


def _print_audit_table(d: dict, node_names: list[str], verbose: bool) -> None:
    """Print audit results as a readable table."""
    click.echo("\n" + "=" * 60)
    click.echo("  CausalCert Structural-Robustness Audit")
    click.echo("=" * 60)
    click.echo(f"  Treatment : {node_names[d['treatment']]} ({d['treatment']})")
    click.echo(f"  Outcome   : {node_names[d['outcome']]} ({d['outcome']})")
    click.echo(f"  Nodes     : {d['n_nodes']}")
    click.echo(f"  Edges     : {d['n_edges']}")
    click.echo("-" * 60)
    r = d["radius"]
    click.echo(f"  Robustness Radius: [{r['lower_bound']}, {r['upper_bound']}]")
    click.echo(f"  Certified        : {'Yes' if r['certified'] else 'No'}")
    click.echo(f"  Solver           : {r['solver_strategy']} ({r['solver_time_s']}s)")
    if "baseline_ate" in d:
        click.echo("-" * 60)
        click.echo(f"  Baseline ATE : {d['baseline_ate']} ± {d['baseline_se']}")
        click.echo(f"  95% CI       : [{d['baseline_ci'][0]}, {d['baseline_ci'][1]}]")
    if d.get("top_fragile_edges"):
        click.echo("-" * 60)
        click.echo("  Top Fragile Edges:")
        for i, fe in enumerate(d["top_fragile_edges"][:5], 1):
            src, tgt = fe["edge"]
            sn = node_names[src] if src < len(node_names) else str(src)
            tn = node_names[tgt] if tgt < len(node_names) else str(tgt)
            click.echo(f"    {i}. {sn} → {tn}  (score={fe['total_score']})")
    click.echo("-" * 60)
    click.echo(f"  CI tests   : {d['n_ci_tests']}")
    click.echo(f"  Elapsed    : {d.get('elapsed_s', '?')}s")
    click.echo("=" * 60)


# ---------------------------------------------------------------------------
# fragility command
# ---------------------------------------------------------------------------


@main.command()
@click.option("--dag", required=True, type=click.Path(exists=True),
              help="Path to DAG file.")
@click.option("--data", required=True, type=click.Path(exists=True),
              help="Path to data file.")
@click.option("--treatment", required=True, type=str,
              help="Treatment variable.")
@click.option("--outcome", required=True, type=str,
              help="Outcome variable.")
@click.option("--top-k", default=10, type=int,
              help="Number of top fragile edges to display.")
@click.option("--alpha", default=0.05, type=float,
              help="Significance level.")
@click.option("--output-format", default="table",
              type=click.Choice(["table", "json"]),
              help="Output format.")
@click.option("--verbose", "-v", is_flag=True,
              help="Verbose output.")
@click.option("--quiet", "-q", is_flag=True,
              help="Suppress non-error output.")
def fragility(
    dag: str,
    data: str,
    treatment: str,
    outcome: str,
    top_k: int,
    alpha: float,
    output_format: str,
    verbose: bool,
    quiet: bool,
) -> None:
    """Run a quick fragility scan on the DAG.

    Computes per-edge fragility scores and ranks edges from most to
    least fragile.  Does NOT compute the full robustness radius.

    \b
    Examples:
      causalcert fragility --dag dag.dot --data data.csv -t X -y Y
      causalcert fragility --dag dag.json --data data.csv -t 0 -y 3 --top-k 20
    """
    _setup_logging(verbose, quiet)

    from causalcert.pipeline.config import PipelineRunConfig
    from causalcert.pipeline.orchestrator import CausalCertPipeline

    adj_matrix, node_names = _load_dag(dag)
    df = _load_data(data)
    treatment_idx = _resolve_variable(treatment, node_names)
    outcome_idx = _resolve_variable(outcome, node_names)

    if not quiet:
        click.echo(f"Fragility scan: {adj_matrix.shape[0]} nodes, "
                    f"treatment={node_names[treatment_idx]}, "
                    f"outcome={node_names[outcome_idx]}")

    cfg = PipelineRunConfig(
        treatment=treatment_idx,
        outcome=outcome_idx,
        alpha=alpha,
        verbose=verbose,
    )
    cfg.steps.radius = False
    cfg.steps.estimation = False

    pipeline = CausalCertPipeline(cfg)
    report = pipeline.run(adj_matrix, df)

    scores = report.fragility_ranking[:top_k]
    if output_format == "json":
        out = [
            {
                "edge": list(s.edge),
                "source": node_names[s.edge[0]],
                "target": node_names[s.edge[1]],
                "total_score": round(s.total_score, 4),
                "channels": {ch.value: round(v, 4) for ch, v in s.channel_scores.items()},
            }
            for s in scores
        ]
        _print_json(out)
    else:
        rows = []
        for rank, s in enumerate(scores, 1):
            src_name = node_names[s.edge[0]] if s.edge[0] < len(node_names) else str(s.edge[0])
            tgt_name = node_names[s.edge[1]] if s.edge[1] < len(node_names) else str(s.edge[1])
            rows.append({
                "Rank": str(rank),
                "Edge": f"{src_name} → {tgt_name}",
                "Score": f"{s.total_score:.4f}",
            })
        _print_table(rows, ["Rank", "Edge", "Score"])


# ---------------------------------------------------------------------------
# radius command
# ---------------------------------------------------------------------------


@main.command()
@click.option("--dag", required=True, type=click.Path(exists=True),
              help="Path to DAG file.")
@click.option("--data", required=True, type=click.Path(exists=True),
              help="Path to data file.")
@click.option("--treatment", required=True, type=str,
              help="Treatment variable.")
@click.option("--outcome", required=True, type=str,
              help="Outcome variable.")
@click.option("--k-max", default=10, type=int,
              help="Maximum edit distance.")
@click.option("--solver", default="auto", type=_SOLVER_CHOICES,
              help="Solver strategy.")
@click.option("--alpha", default=0.05, type=float,
              help="Significance level.")
@click.option("--output-format", default="table",
              type=click.Choice(["table", "json"]),
              help="Output format.")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--quiet", "-q", is_flag=True)
def radius(
    dag: str,
    data: str,
    treatment: str,
    outcome: str,
    k_max: int,
    solver: str,
    alpha: float,
    output_format: str,
    verbose: bool,
    quiet: bool,
) -> None:
    """Compute the robustness radius for a DAG.

    \b
    Examples:
      causalcert radius --dag dag.dot --data data.csv -t X -y Y --k-max 5
      causalcert radius --dag dag.json --data d.csv -t 0 -y 3 --solver ilp
    """
    _setup_logging(verbose, quiet)

    from causalcert.pipeline.config import PipelineRunConfig
    from causalcert.pipeline.orchestrator import CausalCertPipeline
    from causalcert.types import SolverStrategy

    adj_matrix, node_names = _load_dag(dag)
    df = _load_data(data)
    treatment_idx = _resolve_variable(treatment, node_names)
    outcome_idx = _resolve_variable(outcome, node_names)

    solver_map = {
        "auto": SolverStrategy.AUTO,
        "ilp": SolverStrategy.ILP,
        "lp": SolverStrategy.LP_RELAXATION,
        "fpt": SolverStrategy.FPT,
        "cdcl": SolverStrategy.CDCL,
    }

    cfg = PipelineRunConfig(
        treatment=treatment_idx,
        outcome=outcome_idx,
        alpha=alpha,
        max_k=k_max,
        solver_strategy=solver_map[solver],
        verbose=verbose,
    )
    cfg.steps.estimation = False

    pipeline = CausalCertPipeline(cfg)
    report = pipeline.run(adj_matrix, df)
    r = report.radius

    if output_format == "json":
        _print_json({
            "lower_bound": r.lower_bound,
            "upper_bound": r.upper_bound,
            "certified": r.certified,
            "gap": r.gap,
            "solver": r.solver_strategy.value,
            "time_s": round(r.solver_time_s, 3),
            "n_witness_edits": len(r.witness_edits),
        })
    else:
        click.echo(f"Robustness Radius: [{r.lower_bound}, {r.upper_bound}]")
        click.echo(f"Certified: {'Yes' if r.certified else 'No'}")
        click.echo(f"Solver: {r.solver_strategy.value} ({r.solver_time_s:.3f}s)")
        if r.witness_edits:
            click.echo(f"Witness edits ({len(r.witness_edits)}):")
            for e in r.witness_edits:
                src = node_names[e.source] if e.source < len(node_names) else str(e.source)
                tgt = node_names[e.target] if e.target < len(node_names) else str(e.target)
                click.echo(f"  {e.edit_type.value}: {src} → {tgt}")


# ---------------------------------------------------------------------------
# validate command
# ---------------------------------------------------------------------------


@main.command()
@click.option("--dag", required=True, type=click.Path(exists=True),
              help="Path to DAG file.")
@click.option("--data", required=True, type=click.Path(exists=True),
              help="Path to data file.")
@click.option("--verbose", "-v", is_flag=True)
def validate(dag: str, data: str, verbose: bool) -> None:
    """Validate DAG structure and data-DAG consistency.

    Checks acyclicity, binary adjacency, dimension agreement, and
    reports missing values.

    \b
    Examples:
      causalcert validate --dag dag.dot --data data.csv
    """
    _setup_logging(verbose)

    from causalcert.dag.validation import validate_adjacency_matrix
    from causalcert.data.loader import missing_value_report

    adj_matrix, node_names = _load_dag(dag)
    df = _load_data(data)

    click.echo(f"DAG: {len(node_names)} nodes, {int(adj_matrix.sum())} edges")
    click.echo(f"Data: {df.shape[0]} rows × {df.shape[1]} columns")

    # Structural checks
    issues = validate_adjacency_matrix(adj_matrix)
    if issues:
        click.echo("\n⚠ DAG issues:")
        for issue in issues:
            click.echo(f"  - {issue}")
    else:
        click.echo("✓ DAG structure is valid")

    # Dimension check
    if df.shape[1] < adj_matrix.shape[0]:
        click.echo(
            f"\n⚠ Data has {df.shape[1]} columns but DAG has "
            f"{adj_matrix.shape[0]} nodes"
        )
    else:
        click.echo("✓ Data dimensions consistent with DAG")

    # Missing values
    mv = missing_value_report(df)
    if mv["total_missing"] > 0:
        click.echo(f"\n⚠ Missing values: {mv['total_missing']} total")
        for col, pct in mv["pct_missing"].items():
            if pct > 0:
                click.echo(f"  {col}: {pct:.1f}%")
    else:
        click.echo("✓ No missing values")


# ---------------------------------------------------------------------------
# report command
# ---------------------------------------------------------------------------


@main.command()
@click.option("--results", required=True, type=click.Path(exists=True),
              help="Path to results JSON file.")
@click.option("--format", "fmt", default="json", type=_FORMAT_CHOICES,
              help="Output format.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Output file path.")
@click.option("--verbose", "-v", is_flag=True)
def report(results: str, fmt: str, output: str | None, verbose: bool) -> None:
    """Generate a report from saved pipeline results.

    \b
    Examples:
      causalcert report --results audit.json --format html -o report.html
      causalcert report --results audit.json --format table
    """
    _setup_logging(verbose)

    path = Path(results)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if fmt == "table":
        # Print a summary table
        click.echo("\n=== CausalCert Report ===")
        for key, value in data.items():
            if isinstance(value, dict):
                click.echo(f"\n{key}:")
                for k, v in value.items():
                    click.echo(f"  {k}: {v}")
            elif isinstance(value, list):
                click.echo(f"\n{key}: ({len(value)} items)")
            else:
                click.echo(f"{key}: {value}")
    elif fmt == "json":
        _print_json(data)
    elif fmt in ("html", "latex"):
        if output is None:
            ext = ".html" if fmt == "html" else ".tex"
            output = str(path.with_suffix(ext))
        click.echo(f"Generating {fmt} report → {output}")
        # Write a simple wrapper
        if fmt == "html":
            html = _simple_html_report(data)
            Path(output).write_text(html, encoding="utf-8")
        else:
            tex = _simple_latex_report(data)
            Path(output).write_text(tex, encoding="utf-8")
        click.echo(f"Report written to {output}")


def _simple_html_report(data: dict) -> str:
    """Generate a minimal HTML report."""
    lines = [
        "<!DOCTYPE html><html><head><title>CausalCert Report</title></head><body>",
        "<h1>CausalCert Structural-Robustness Audit</h1>",
        "<table border='1' cellpadding='4'>",
    ]
    for k, v in data.items():
        if not isinstance(v, (dict, list)):
            lines.append(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>")
    lines.append("</table>")
    if "radius" in data and isinstance(data["radius"], dict):
        lines.append("<h2>Robustness Radius</h2><table border='1' cellpadding='4'>")
        for k, v in data["radius"].items():
            lines.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
        lines.append("</table>")
    if "top_fragile_edges" in data:
        lines.append("<h2>Top Fragile Edges</h2><table border='1' cellpadding='4'>")
        lines.append("<tr><th>Edge</th><th>Score</th></tr>")
        for fe in data["top_fragile_edges"]:
            lines.append(f"<tr><td>{fe['edge']}</td><td>{fe['total_score']}</td></tr>")
        lines.append("</table>")
    lines.append("</body></html>")
    return "\n".join(lines)


def _simple_latex_report(data: dict) -> str:
    """Generate a minimal LaTeX report."""
    lines = [
        r"\documentclass{article}",
        r"\begin{document}",
        r"\section{CausalCert Structural-Robustness Audit}",
        r"\begin{tabular}{ll}",
    ]
    for k, v in data.items():
        if not isinstance(v, (dict, list)):
            lines.append(f"  {k} & {v} \\\\")
    lines.append(r"\end{tabular}")
    if "radius" in data and isinstance(data["radius"], dict):
        lines.append(r"\subsection{Robustness Radius}")
        lines.append(r"\begin{tabular}{ll}")
        for k, v in data["radius"].items():
            lines.append(f"  {k} & {v} \\\\")
        lines.append(r"\end{tabular}")
    lines.append(r"\end{document}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
