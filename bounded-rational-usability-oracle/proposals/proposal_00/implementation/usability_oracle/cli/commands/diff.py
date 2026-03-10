"""
usability_oracle.cli.commands.diff — Compare two UI versions.

Implements the ``usability-oracle diff`` command which loads two UI
sources, runs the full comparison pipeline, and outputs a regression
report.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import click

logger = logging.getLogger(__name__)


def diff_command(
    before: str,
    after: str,
    task_spec: Optional[str] = None,
    config: Optional[str] = None,
    output_format: str = "console",
    beta_range: tuple[float, float] = (0.1, 20.0),
    output_path: Optional[str] = None,
) -> int:
    """Compare two UI versions for usability regressions.

    Parameters
    ----------
    before : str
        Path to the "before" UI source (HTML or JSON).
    after : str
        Path to the "after" UI source.
    task_spec : str, optional
        Path to task specification file.
    config : str, optional
        Path to YAML configuration file.
    output_format : str
        Output format: json, sarif, html, console.
    beta_range : tuple[float, float]
        Rationality parameter β range.
    output_path : str, optional
        Output file path (stdout if None).

    Returns
    -------
    int
        Exit code: 0 = no regression, 1 = regression detected, 2 = error.
    """
    from usability_oracle.pipeline.config import FullPipelineConfig
    from usability_oracle.pipeline.runner import PipelineRunner
    from usability_oracle.cli.formatters import format_result

    try:
        # Load configuration
        if config:
            cfg = FullPipelineConfig.from_yaml(config)
        else:
            cfg = FullPipelineConfig.DEFAULT()

        cfg.oracle.policy.beta_min = beta_range[0]
        cfg.oracle.policy.beta_max = beta_range[1]

        # Load UI sources
        before_path = Path(before)
        after_path = Path(after)
        source_a = before_path.read_text(encoding="utf-8")
        source_b = after_path.read_text(encoding="utf-8")

        # Load task spec if provided
        task_data = None
        if task_spec:
            task_path = Path(task_spec)
            task_text = task_path.read_text(encoding="utf-8")
            if task_path.suffix in (".yaml", ".yml"):
                import yaml
                task_data = yaml.safe_load(task_text)
            else:
                task_data = json.loads(task_text)

        # Run pipeline
        click.echo("Running usability comparison…")
        t0 = time.monotonic()

        runner = PipelineRunner(config=cfg)
        result = runner.run(
            config=cfg,
            source_a=source_a,
            source_b=source_b,
            task_spec=task_data,
        )

        elapsed = time.monotonic() - t0

        # Format output
        output = format_result(result, output_format)

        # Write output
        if output_path:
            Path(output_path).write_text(output, encoding="utf-8")
            click.echo(f"Results written to {output_path}")
        else:
            click.echo(output)

        # Summary
        click.echo(f"\nCompleted in {elapsed:.2f}s")

        if result.final_result:
            verdict = _extract_verdict(result.final_result)
            if verdict == "regression":
                click.echo(
                    click.style("⚠ Usability regression detected", fg="red", bold=True)
                )
                return 1
            elif verdict == "improvement":
                click.echo(
                    click.style("✓ Usability improved", fg="green", bold=True)
                )
                return 0
            else:
                click.echo("— No significant change detected")
                return 0

        return 0

    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        return 2
    except Exception as exc:
        logger.exception("Diff command failed")
        click.echo(f"Error: {exc}", err=True)
        return 2


def _extract_verdict(result: Any) -> str:
    """Extract the verdict string from a pipeline result."""
    if isinstance(result, dict):
        return result.get("verdict", "no_change")
    if hasattr(result, "verdict"):
        return result.verdict
    return "no_change"


def _format_diff_summary(result: Any) -> str:
    """Format a compact diff summary."""
    lines: list[str] = []

    if isinstance(result, dict):
        details = result.get("details", {})
        lines.append(f"Common states compared: {details.get('common_states', '?')}")
        lines.append(
            f"Average policy divergence: {details.get('avg_policy_diff', 0):.4f}"
        )

    return "\n".join(lines) if lines else "No details available"


def _load_task_spec(path: str) -> Any:
    """Load a task specification from file.

    Supports both YAML and JSON formats, determined by file extension.

    Parameters
    ----------
    path : str
        Path to the task spec file.

    Returns
    -------
    Any
        Parsed task specification data.
    """
    task_path = Path(path)
    task_text = task_path.read_text(encoding="utf-8")

    if task_path.suffix in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(task_text)
    return json.loads(task_text)


def _format_stage_timing(result: Any) -> str:
    """Format per-stage timing as a summary string.

    Parameters
    ----------
    result : PipelineResult-like
        Pipeline result with timing information.

    Returns
    -------
    str
        Formatted timing breakdown.
    """
    lines: list[str] = []

    if not hasattr(result, "timing"):
        return "No timing data"

    timing = result.timing
    for stage, elapsed in sorted(timing.items()):
        if stage == "total":
            continue
        lines.append(f"  {stage:20s}  {elapsed:.3f}s")

    if "total" in timing:
        lines.append(f"  {'TOTAL':20s}  {timing['total']:.3f}s")

    return "\n".join(lines) if lines else "No timing data"


def _format_regression_details(result: Any) -> str:
    """Format detailed regression information.

    Examines the comparison result and extracts specific metrics
    that indicate where the regression occurred.

    Parameters
    ----------
    result : PipelineResult-like
        Pipeline result.

    Returns
    -------
    str
        Multi-line string with regression details.
    """
    lines: list[str] = []

    if not hasattr(result, "stages"):
        return "No stage data available"

    stages = result.stages

    # Cost comparison
    cost_a = stages.get("cost")
    cost_b = stages.get("cost_b")
    if cost_a and cost_b and hasattr(cost_a, "output") and hasattr(cost_b, "output"):
        if cost_a.output and cost_b.output:
            costs_a = cost_a.output.get("node_costs", {})
            costs_b = cost_b.output.get("node_costs", {})

            total_a = sum(
                c.get("total", 0) for c in costs_a.values()
            )
            total_b = sum(
                c.get("total", 0) for c in costs_b.values()
            )

            lines.append(f"Total cost (before): {total_a:.3f}s")
            lines.append(f"Total cost (after):  {total_b:.3f}s")
            diff = total_b - total_a
            if diff > 0:
                lines.append(f"Regression:          +{diff:.3f}s ({diff/max(total_a,0.001)*100:.1f}%)")
            elif diff < 0:
                lines.append(f"Improvement:         {diff:.3f}s ({abs(diff)/max(total_a,0.001)*100:.1f}%)")

    # Bottleneck comparison
    bn_stage = stages.get("bottleneck")
    if bn_stage and hasattr(bn_stage, "output") and bn_stage.output:
        bottlenecks = bn_stage.output
        if isinstance(bottlenecks, list):
            lines.append(f"\nBottlenecks detected: {len(bottlenecks)}")
            by_type: dict[str, int] = {}
            for bn in bottlenecks:
                bt = bn.get("bottleneck_type", "?") if isinstance(bn, dict) else "?"
                by_type[bt] = by_type.get(bt, 0) + 1
            for bt, count in sorted(by_type.items()):
                lines.append(f"  {bt}: {count}")

    return "\n".join(lines) if lines else "No regression details"
