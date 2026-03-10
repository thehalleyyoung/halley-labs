"""
usability_oracle.cli.commands.analyze — Analyse a single UI.

Implements the ``usability-oracle analyze`` command which loads a UI
source, runs the analysis pipeline, and outputs bottleneck and repair
reports.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import click

logger = logging.getLogger(__name__)


def analyze_command(
    source: str,
    task_spec: Optional[str] = None,
    config: Optional[str] = None,
    output_format: str = "console",
    output_path: Optional[str] = None,
) -> int:
    """Analyse a single UI for usability issues.

    Parameters
    ----------
    source : str
        Path to the UI source (HTML or JSON accessibility tree).
    task_spec : str, optional
        Path to task specification file.
    config : str, optional
        Path to YAML configuration file.
    output_format : str
        Output format.
    output_path : str, optional
        Output file path.

    Returns
    -------
    int
        Exit code: 0 = no issues, 1 = issues found, 2 = error.
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

        # Disable comparison stage (single-UI analysis)
        cfg.stages["comparison"].enabled = False

        # Load source
        source_path = Path(source)
        source_content = source_path.read_text(encoding="utf-8")

        # Load task spec
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
        click.echo(f"Analysing {source_path.name}…")
        t0 = time.monotonic()

        runner = PipelineRunner(config=cfg)
        result = runner.run(
            config=cfg,
            source_a=source_content,
            task_spec=task_data,
        )

        elapsed = time.monotonic() - t0

        # Format output
        output = format_result(result, output_format)

        if output_path:
            Path(output_path).write_text(output, encoding="utf-8")
            click.echo(f"Results written to {output_path}")
        else:
            click.echo(output)

        # Summary
        n_bottlenecks = _count_bottlenecks(result)
        click.echo(f"\nCompleted in {elapsed:.2f}s")

        if n_bottlenecks > 0:
            click.echo(
                click.style(
                    f"Found {n_bottlenecks} usability bottleneck(s)",
                    fg="yellow", bold=True,
                )
            )
            return 1
        else:
            click.echo(
                click.style("No usability issues detected", fg="green")
            )
            return 0

    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        return 2
    except Exception as exc:
        logger.exception("Analyze command failed")
        click.echo(f"Error: {exc}", err=True)
        return 2


def _count_bottlenecks(result: Any) -> int:
    """Count bottlenecks from a pipeline result."""
    stages = getattr(result, "stages", {})
    bn_stage = stages.get("bottleneck")
    if bn_stage and hasattr(bn_stage, "output") and bn_stage.output:
        output = bn_stage.output
        if isinstance(output, list):
            return len(output)
    return 0


def _format_analysis_summary(result: Any) -> list[str]:
    """Format analysis summary lines."""
    lines: list[str] = []

    stages = getattr(result, "stages", {})

    # Cost summary
    cost_stage = stages.get("cost")
    if cost_stage and hasattr(cost_stage, "output") and cost_stage.output:
        costs = cost_stage.output
        if isinstance(costs, dict):
            n_nodes = costs.get("n_interactive", 0)
            lines.append(f"Interactive elements: {n_nodes}")
            node_costs = costs.get("node_costs", {})
            if node_costs:
                total = sum(
                    c.get("total", 0) for c in node_costs.values()
                )
                lines.append(f"Total cognitive cost: {total:.2f}")

    # Bottleneck summary
    bn_stage = stages.get("bottleneck")
    if bn_stage and hasattr(bn_stage, "output") and bn_stage.output:
        bottlenecks = bn_stage.output
        if isinstance(bottlenecks, list):
            by_type: dict[str, int] = {}
            for bn in bottlenecks:
                bt = bn.get("bottleneck_type", "unknown") if isinstance(bn, dict) else "unknown"
                by_type[bt] = by_type.get(bt, 0) + 1
            for bt, count in sorted(by_type.items()):
                lines.append(f"  {bt}: {count}")

    # Repair summary
    repair_stage = stages.get("repair")
    if repair_stage and hasattr(repair_stage, "output") and repair_stage.output:
        repair = repair_stage.output
        if hasattr(repair, "n_feasible"):
            lines.append(f"Feasible repairs: {repair.n_feasible}")
        elif isinstance(repair, dict):
            lines.append(f"Repairs: {len(repair.get('candidates', []))}")

    return lines


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


def _format_cost_breakdown(result: Any) -> str:
    """Format a per-node cost breakdown from the analysis result.

    Extracts cost information from the cost stage output and formats
    it as a human-readable table.

    Parameters
    ----------
    result : PipelineResult-like
        Pipeline result with cost stage output.

    Returns
    -------
    str
        Formatted cost table.
    """
    stages = getattr(result, "stages", {})
    cost_stage = stages.get("cost")

    if not cost_stage or not hasattr(cost_stage, "output") or not cost_stage.output:
        return "No cost data available"

    costs = cost_stage.output
    from usability_oracle.cli.formatters import format_cost_table
    return format_cost_table(costs)


def _format_repair_suggestions(result: Any) -> str:
    """Format repair suggestions from the analysis result.

    Parameters
    ----------
    result : PipelineResult-like
        Pipeline result.

    Returns
    -------
    str
        Formatted repair suggestions.
    """
    stages = getattr(result, "stages", {})
    repair_stage = stages.get("repair")

    if not repair_stage or not hasattr(repair_stage, "output") or not repair_stage.output:
        return "No repair suggestions available"

    repair = repair_stage.output
    lines: list[str] = ["Repair Suggestions:"]

    candidates = []
    if hasattr(repair, "candidates"):
        candidates = repair.candidates
    elif isinstance(repair, dict):
        candidates = repair.get("candidates", [])

    for i, c in enumerate(candidates[:5], 1):
        if hasattr(c, "description"):
            desc = c.description
            cr = c.expected_cost_reduction
            conf = c.confidence
        elif isinstance(c, dict):
            desc = c.get("description", "")
            cr = c.get("expected_cost_reduction", 0)
            conf = c.get("confidence", 0)
        else:
            desc = str(c)
            cr = 0
            conf = 0

        lines.append(f"  {i}. {desc}")
        lines.append(f"     Cost reduction: {cr:.3f}, Confidence: {conf:.2f}")

    return "\n".join(lines)
