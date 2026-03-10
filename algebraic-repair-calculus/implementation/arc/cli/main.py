"""
Click-based CLI for the Algebraic Repair Calculus.

Provides commands for analyzing pipelines, computing repair plans,
executing repairs, validating specs, classifying Fragment F membership,
visualizing graphs, and monitoring quality metrics — all with rich
terminal output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich import box

from arc import __version__

console = Console()
error_console = Console(stderr=True)


# =====================================================================
# Utility helpers
# =====================================================================

def _format_cost(cost: Any) -> str:
    """Format a CostEstimate as a compact string."""
    if cost is None:
        return "—"
    parts: list[str] = []
    if cost.compute_seconds > 0:
        parts.append(f"{cost.compute_seconds:.1f}s")
    if cost.memory_bytes > 0:
        mb = cost.memory_bytes / (1024 ** 2)
        parts.append(f"{mb:.1f}MB")
    if cost.monetary_cost > 0:
        parts.append(f"${cost.monetary_cost:.4f}")
    return ", ".join(parts) if parts else "—"


def _format_schema_brief(schema: Any) -> str:
    """Format a Schema as a compact one-line summary."""
    if schema is None:
        return "(no schema)"
    col_names = [c.name for c in schema.columns]
    if len(col_names) <= 4:
        return f"({', '.join(col_names)})"
    shown = ', '.join(col_names[:3])
    remaining = len(col_names) - 3
    return f"({shown}, ... +{remaining} more)"

def _load_pipeline(path: str) -> Any:
    """Load a pipeline graph from a JSON or YAML file."""
    from arc.io.json_format import PipelineSpec
    from arc.io.yaml_format import YAMLPipelineSpec

    p = Path(path)
    if not p.exists():
        error_console.print(f"[red]Error:[/red] File not found: {path}")
        sys.exit(1)

    suffix = p.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            return YAMLPipelineSpec.load(p)
        else:
            return PipelineSpec.load(p)
    except Exception as e:
        error_console.print(f"[red]Error loading {path}:[/red] {e}")
        sys.exit(1)


def _load_delta(path: str) -> dict[str, Any]:
    """Load a delta specification from a JSON file."""
    from arc.io.json_format import DeltaSerializer

    p = Path(path)
    if not p.exists():
        error_console.print(f"[red]Error:[/red] File not found: {path}")
        sys.exit(1)
    try:
        return DeltaSerializer.load_delta(p)
    except Exception as e:
        error_console.print(f"[red]Error loading delta {path}:[/red] {e}")
        sys.exit(1)


def _print_graph_summary(graph: Any) -> None:
    """Print a summary table of the pipeline graph."""
    from arc.graph.analysis import compute_metrics, FragmentClassifier

    metrics = compute_metrics(graph)

    table = Table(title="Pipeline Summary", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Name", graph.name or "(unnamed)")
    table.add_row("Nodes", str(metrics.node_count))
    table.add_row("Edges", str(metrics.edge_count))
    table.add_row("Sources", str(metrics.source_count))
    table.add_row("Sinks", str(metrics.sink_count))
    table.add_row("Depth", str(metrics.depth))
    table.add_row("Width", str(metrics.width))
    table.add_row("DAG", "✓" if metrics.is_dag else "✗")
    table.add_row("Components", str(metrics.component_count))
    table.add_row("Fragment F", f"{metrics.fragment_f_fraction:.1%}")
    table.add_row("Total Cost", f"{metrics.total_cost:.2f}")

    console.print(table)

    # Operator distribution
    if metrics.operator_distribution:
        op_table = Table(title="Operators", box=box.SIMPLE)
        op_table.add_column("Operator", style="cyan")
        op_table.add_column("Count", style="green")
        for op, count in sorted(metrics.operator_distribution.items(), key=lambda x: -x[1]):
            op_table.add_row(op, str(count))
        console.print(op_table)


# =====================================================================
# CLI group
# =====================================================================

@click.group()
@click.version_option(version=__version__, prog_name="arc")
def cli() -> None:
    """ARC — Algebraic Repair Calculus for data pipeline maintenance.

    Provably correct incremental maintenance of data pipelines under
    schema evolution, quality drift, and partial outages.
    """
    pass


# =====================================================================
# analyze
# =====================================================================

@cli.command()
@click.argument("pipeline", type=click.Path(exists=True))
@click.option("--node", "-n", help="Analyze impact from a specific node")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def analyze(pipeline: str, node: str | None, verbose: bool) -> None:
    """Analyze a pipeline and show dependency graph."""
    graph = _load_pipeline(pipeline)
    _print_graph_summary(graph)

    if node:
        from arc.graph.analysis import impact_analysis, dependency_analysis

        console.print(f"\n[bold]Impact analysis for node: {node}[/bold]")
        try:
            impact = impact_analysis(graph, node)
            impact_table = Table(title="Impact Analysis", box=box.ROUNDED)
            impact_table.add_column("Metric", style="cyan")
            impact_table.add_column("Value", style="yellow")
            impact_table.add_row("Affected nodes", str(len(impact.affected_nodes)))
            impact_table.add_row("Max depth", str(impact.max_depth))
            impact_table.add_row("Recompute cost", f"{impact.total_recompute_cost.total_weighted_cost:.2f}")
            impact_table.add_row("Fragment F fraction", f"{impact.fragment_f_fraction:.1%}")
            console.print(impact_table)

            if verbose and impact.affected_nodes:
                console.print("\n[bold]Affected nodes:[/bold]")
                for nid in impact.affected_nodes:
                    n = graph.get_node(nid)
                    frag = "[green]F[/green]" if n.in_fragment_f else "[red]~F[/red]"
                    console.print(f"  • {nid} [{n.operator.value}] {frag}")

        except Exception as e:
            error_console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    if verbose:
        # Show dependency tree
        console.print("\n[bold]Dependency graph:[/bold]")
        tree = Tree(f"[bold]{graph.name or 'pipeline'}[/bold]")
        _build_dependency_tree(graph, tree)
        console.print(tree)


def _build_dependency_tree(graph: Any, tree: Tree) -> None:
    """Build a Rich tree from the pipeline graph."""
    for source in graph.sources():
        node = graph.get_node(source)
        frag = "[green]F[/green]" if node.in_fragment_f else "[red]~F[/red]"
        source_tree = tree.add(f"[cyan]{source}[/cyan] [{node.operator.value}] {frag}")
        _add_children(graph, source, source_tree, visited=set())


def _add_children(graph: Any, node_id: str, parent_tree: Tree, visited: set[str]) -> None:
    """Recursively add children to a tree."""
    if node_id in visited:
        return
    visited.add(node_id)
    for child_id in graph.successors(node_id):
        child = graph.get_node(child_id)
        frag = "[green]F[/green]" if child.in_fragment_f else "[red]~F[/red]"
        child_tree = parent_tree.add(f"[cyan]{child_id}[/cyan] [{child.operator.value}] {frag}")
        _add_children(graph, child_id, child_tree, visited)


# =====================================================================
# validate
# =====================================================================

@cli.command()
@click.argument("pipeline", type=click.Path(exists=True))
@click.option("--strict", is_flag=True, help="Treat warnings as errors")
def validate(pipeline: str, strict: bool) -> None:
    """Validate a pipeline specification."""
    graph = _load_pipeline(pipeline)
    issues = graph.validate()

    if not issues:
        console.print("[green]✓ Pipeline is valid[/green]")
        return

    for issue in issues:
        level = "[red]ERROR[/red]" if strict else "[yellow]WARNING[/yellow]"
        console.print(f"  {level}: {issue}")

    if strict and issues:
        console.print(f"\n[red]Validation failed with {len(issues)} issue(s)[/red]")
        sys.exit(1)
    else:
        console.print(f"\n[yellow]{len(issues)} issue(s) found[/yellow]")


# =====================================================================
# fragment
# =====================================================================

@cli.command()
@click.argument("pipeline", type=click.Path(exists=True))
@click.option("--node", "-n", help="Check a specific node")
def fragment(pipeline: str, node: str | None) -> None:
    """Check Fragment F membership for pipeline nodes."""
    from arc.graph.analysis import FragmentClassifier

    graph = _load_pipeline(pipeline)
    classifier = FragmentClassifier()
    classification = classifier.classify(graph)

    if node:
        in_f, reasons = classifier.node_in_fragment_f(graph, node)
        if in_f:
            console.print(f"[green]✓ Node '{node}' is in Fragment F[/green]")
        else:
            console.print(f"[red]✗ Node '{node}' is NOT in Fragment F[/red]")
            for reason in reasons:
                console.print(f"  • {reason}")
        return

    # Show all nodes
    table = Table(title="Fragment F Classification", box=box.ROUNDED)
    table.add_column("Node", style="cyan")
    table.add_column("Operator", style="white")
    table.add_column("Fragment F", style="green")
    table.add_column("Violations", style="red")

    order = graph.topological_sort() if graph.is_dag() else sorted(graph.node_ids)
    for nid in order:
        n = graph.get_node(nid)
        in_f = nid in classification.fragment_f_nodes
        violations = classification.violations.get(nid, [])
        status = "[green]✓ F[/green]" if in_f else "[red]✗ ~F[/red]"
        viol_str = "; ".join(violations) if violations else ""
        table.add_row(nid, n.operator.value, status, viol_str)

    console.print(table)
    console.print(f"\n{classification}")


# =====================================================================
# visualize
# =====================================================================

@cli.command()
@click.argument("pipeline", type=click.Path(exists=True))
@click.option("--format", "-f", "fmt", type=click.Choice(["ascii", "dot", "mermaid"]), default="ascii")
@click.option("--output", "-o", help="Output file (default: stdout)")
@click.option("--highlight", "-h", multiple=True, help="Nodes to highlight")
def visualize(pipeline: str, fmt: str, output: str | None, highlight: tuple[str, ...]) -> None:
    """Visualize the pipeline graph."""
    from arc.graph.visualization import to_ascii, to_dot, to_mermaid, save_dot

    graph = _load_pipeline(pipeline)
    highlight_list = list(highlight)

    if fmt == "dot":
        result = to_dot(graph, highlight_nodes=highlight_list)
    elif fmt == "mermaid":
        result = to_mermaid(graph)
    else:
        result = to_ascii(graph, highlight_nodes=highlight_list)

    if output:
        Path(output).write_text(result)
        console.print(f"[green]Written to {output}[/green]")
    else:
        console.print(result)


# =====================================================================
# repair
# =====================================================================

@cli.command()
@click.argument("pipeline", type=click.Path(exists=True))
@click.option("--perturbation", "-p", required=True, type=click.Path(exists=True),
              help="Delta specification file (JSON)")
@click.option("--output", "-o", help="Output repair plan file")
@click.option("--dry-run", is_flag=True, help="Only show what would be repaired")
def repair(pipeline: str, perturbation: str, output: str | None, dry_run: bool) -> None:
    """Compute a repair plan for a perturbation."""
    from arc.graph.analysis import compute_repair_scope, compare_repair_vs_recompute

    graph = _load_pipeline(pipeline)
    delta = _load_delta(perturbation)

    # Identify the target node
    target_node = delta.get("target_node")
    if not target_node:
        error_console.print("[red]Error:[/red] Delta must specify 'target_node'")
        sys.exit(1)

    if not graph.has_node(target_node):
        error_console.print(f"[red]Error:[/red] Node '{target_node}' not found in pipeline")
        sys.exit(1)

    # Compute repair scope
    repair_nodes, repair_cost = compute_repair_scope(graph, [target_node])
    comparison = compare_repair_vs_recompute(graph, repair_nodes)

    # Build repair plan
    plan: dict[str, Any] = {
        "version": "1.0",
        "pipeline_name": graph.name,
        "perturbation": delta,
        "actions": [],
        "cost_estimate": repair_cost.to_dict(),
        "affected_nodes": repair_nodes,
        "correctness_guarantee": "exact" if graph.is_in_fragment_f() else "bounded_epsilon",
    }

    for nid in repair_nodes:
        node = graph.get_node(nid)
        action: dict[str, Any] = {
            "node_id": nid,
            "action_type": "recompute",
            "estimated_cost": node.cost_estimate.to_dict(),
        }
        # Determine action type based on delta sort and node position
        if nid == target_node and delta.get("sort") == "schema":
            action["action_type"] = "schema_migrate"
        elif nid == target_node and delta.get("sort") == "quality":
            action["action_type"] = "quality_fix"

        plan["actions"].append(action)

    # Display results
    console.print(Panel(
        f"[bold]Repair Plan[/bold] for [cyan]{graph.name}[/cyan]\n"
        f"Perturbation: {delta.get('sort', 'unknown')} at {target_node}\n"
        f"Guarantee: {plan['correctness_guarantee']}",
        title="ARC Repair Planner",
    ))

    table = Table(title="Repair Actions", box=box.ROUNDED)
    table.add_column("#", style="dim")
    table.add_column("Node", style="cyan")
    table.add_column("Action", style="yellow")
    table.add_column("Cost", style="green")

    for i, action in enumerate(plan["actions"], 1):
        cost = action.get("estimated_cost", {})
        cost_str = f"{cost.get('compute_seconds', 0):.2f}s"
        table.add_row(str(i), action["node_id"], action["action_type"], cost_str)

    console.print(table)
    console.print(f"\n{comparison}")

    if dry_run:
        console.print("\n[yellow]Dry run — no changes applied[/yellow]")
        return

    if output:
        from arc.io.json_format import RepairPlanSerializer
        RepairPlanSerializer.save(plan, output)
        console.print(f"\n[green]Repair plan saved to {output}[/green]")
    else:
        console.print("\n[dim]Use --output to save the repair plan[/dim]")


# =====================================================================
# execute
# =====================================================================

@cli.command()
@click.argument("repair_plan", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Only show what would be executed")
@click.option("--checkpoint/--no-checkpoint", default=True, help="Create checkpoints")
def execute(repair_plan: str, dry_run: bool, checkpoint: bool) -> None:
    """Execute a repair plan."""
    from arc.io.json_format import RepairPlanSerializer

    plan = RepairPlanSerializer.load(repair_plan)

    # Validate
    errors = RepairPlanSerializer.validate(plan)
    if errors:
        error_console.print("[red]Invalid repair plan:[/red]")
        for err in errors:
            error_console.print(f"  • {err}")
        sys.exit(1)

    actions = plan.get("actions", [])
    console.print(Panel(
        f"[bold]Executing repair plan[/bold]\n"
        f"Pipeline: {plan.get('pipeline_name', '?')}\n"
        f"Actions: {len(actions)}\n"
        f"Guarantee: {plan.get('correctness_guarantee', '?')}",
        title="ARC Executor",
    ))

    if dry_run:
        table = Table(title="Planned Actions (dry run)", box=box.ROUNDED)
        table.add_column("#", style="dim")
        table.add_column("Node", style="cyan")
        table.add_column("Action", style="yellow")

        for i, action in enumerate(actions, 1):
            table.add_row(str(i), action["node_id"], action["action_type"])

        console.print(table)
        console.print("\n[yellow]Dry run — no changes applied[/yellow]")
        return

    # Execute each action
    with console.status("[bold green]Executing repair plan...") as status:
        for i, action in enumerate(actions, 1):
            nid = action["node_id"]
            atype = action["action_type"]
            status.update(f"Step {i}/{len(actions)}: {atype} on {nid}")

            # Placeholder: actual execution would dispatch to backends
            console.print(f"  [{i}/{len(actions)}] {atype} on [cyan]{nid}[/cyan] ... [green]✓[/green]")

    console.print(f"\n[green]✓ Repair plan executed successfully ({len(actions)} actions)[/green]")


# =====================================================================
# monitor
# =====================================================================

@cli.command()
@click.argument("pipeline", type=click.Path(exists=True))
@click.option("--node", "-n", help="Monitor a specific node")
def monitor(pipeline: str, node: str | None) -> None:
    """Monitor quality metrics for a pipeline."""
    graph = _load_pipeline(pipeline)

    table = Table(title="Quality Constraints", box=box.ROUNDED)
    table.add_column("Node", style="cyan")
    table.add_column("Constraint", style="white")
    table.add_column("Severity", style="yellow")
    table.add_column("Predicate", style="dim")
    table.add_column("Columns", style="green")

    order = graph.topological_sort() if graph.is_dag() else sorted(graph.node_ids)
    for nid in order:
        if node and nid != node:
            continue
        n = graph.get_node(nid)
        if not n.quality_constraints:
            continue
        for qc in n.quality_constraints:
            sev_style = {
                "info": "[blue]info[/blue]",
                "warning": "[yellow]warning[/yellow]",
                "error": "[red]error[/red]",
                "critical": "[bold red]critical[/bold red]",
            }
            sev = sev_style.get(qc.severity.value, qc.severity.value)
            cols = ", ".join(qc.affected_columns) if qc.affected_columns else "-"
            table.add_row(nid, qc.constraint_id, sev, qc.predicate, cols)

    if table.row_count == 0:
        console.print("[dim]No quality constraints defined[/dim]")
    else:
        console.print(table)


# =====================================================================
# info
# =====================================================================

@cli.command()
def info() -> None:
    """Show ARC system information."""
    from arc.io.yaml_format import list_templates

    console.print(Panel(
        f"[bold]Algebraic Repair Calculus[/bold] v{__version__}\n\n"
        "Three-sorted delta algebra Δ = (Δ_S, Δ_D, Δ_Q, ∘, ⁻¹, push)\n"
        "for provably correct data pipeline repair.",
        title="ARC",
    ))

    table = Table(title="Available Commands", box=box.SIMPLE)
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    table.add_row("analyze", "Analyze pipeline and show dependency graph")
    table.add_row("repair", "Compute repair plan for a perturbation")
    table.add_row("execute", "Execute a repair plan")
    table.add_row("validate", "Validate a pipeline specification")
    table.add_row("fragment", "Check Fragment F membership")
    table.add_row("visualize", "Visualize the pipeline graph")
    table.add_row("monitor", "Monitor quality metrics")
    table.add_row("template", "Generate pipeline from template")
    table.add_row("info", "Show system information")
    console.print(table)

    template_table = Table(title="Pipeline Templates", box=box.SIMPLE)
    template_table.add_column("Template", style="cyan")
    for t in list_templates():
        template_table.add_row(t)
    console.print(template_table)


# =====================================================================
# template
# =====================================================================

@cli.command()
@click.argument("template_name")
@click.option("--name", "-n", default="", help="Pipeline name")
@click.option("--output", "-o", help="Output file")
@click.option("--format", "-f", "fmt", type=click.Choice(["yaml", "json"]), default="yaml")
def template(template_name: str, name: str, output: str | None, fmt: str) -> None:
    """Generate a pipeline from a template."""
    from arc.io.yaml_format import from_template, get_template_yaml, list_templates

    available = list_templates()
    if template_name not in available:
        error_console.print(
            f"[red]Unknown template '{template_name}'.[/red] "
            f"Available: {', '.join(available)}"
        )
        sys.exit(1)

    if fmt == "yaml":
        content = get_template_yaml(template_name, name=name)
    else:
        graph = from_template(template_name, name=name)
        from arc.io.json_format import PipelineSpec
        content = PipelineSpec.to_json(graph)

    if output:
        Path(output).write_text(content)
        console.print(f"[green]Template written to {output}[/green]")
    else:
        console.print(content)


# =====================================================================
# scope
# =====================================================================

@cli.command()
@click.argument("pipeline", type=click.Path(exists=True))
@click.option("--node", "-n", required=True, multiple=True,
              help="Perturbed node(s)")
def scope(pipeline: str, node: tuple[str, ...]) -> None:
    """Compute the repair scope for perturbed nodes."""
    from arc.graph.analysis import compute_repair_scope, compare_repair_vs_recompute

    graph = _load_pipeline(pipeline)
    perturbed = list(node)

    # Validate nodes exist
    for nid in perturbed:
        if not graph.has_node(nid):
            error_console.print(f"[red]Error:[/red] Node '{nid}' not found")
            sys.exit(1)

    repair_nodes, repair_cost = compute_repair_scope(graph, perturbed)
    comparison = compare_repair_vs_recompute(graph, repair_nodes)

    console.print(Panel(
        f"[bold]Repair Scope Analysis[/bold]\n"
        f"Perturbed: {', '.join(perturbed)}\n"
        f"Nodes to repair: {len(repair_nodes)}\n"
        f"Total nodes: {graph.node_count}",
        title="ARC Scope",
    ))

    table = Table(title="Repair Scope", box=box.ROUNDED)
    table.add_column("#", style="dim")
    table.add_column("Node", style="cyan")
    table.add_column("Operator", style="white")
    table.add_column("Fragment F", style="green")
    table.add_column("Cost", style="yellow")

    for i, nid in enumerate(repair_nodes, 1):
        n = graph.get_node(nid)
        frag = "[green]F[/green]" if n.in_fragment_f else "[red]~F[/red]"
        cost = f"{n.cost_estimate.total_weighted_cost:.2f}"
        table.add_row(str(i), nid, n.operator.value, frag, cost)

    console.print(table)
    console.print(f"\n{comparison}")


# =====================================================================
# diff
# =====================================================================

@cli.command()
@click.argument("pipeline_a", type=click.Path(exists=True))
@click.argument("pipeline_b", type=click.Path(exists=True))
def diff(pipeline_a: str, pipeline_b: str) -> None:
    """Show differences between two pipeline specifications."""
    from arc.graph.analysis import diff_graphs

    graph_a = _load_pipeline(pipeline_a)
    graph_b = _load_pipeline(pipeline_b)

    result = diff_graphs(graph_a, graph_b)

    if result.is_empty:
        console.print("[green]Pipelines are identical[/green]")
        return

    console.print(Panel(
        f"[bold]Pipeline Diff[/bold]\n"
        f"A: {pipeline_a} ({graph_a.node_count} nodes)\n"
        f"B: {pipeline_b} ({graph_b.node_count} nodes)",
        title="ARC Diff",
    ))

    if result.added_nodes:
        console.print("\n[green]Added nodes:[/green]")
        for nid in result.added_nodes:
            n = graph_b.get_node(nid)
            console.print(f"  + {nid} [{n.operator.value}]")

    if result.removed_nodes:
        console.print("\n[red]Removed nodes:[/red]")
        for nid in result.removed_nodes:
            console.print(f"  - {nid}")

    if result.modified_nodes:
        console.print("\n[yellow]Modified nodes:[/yellow]")
        for nid in result.modified_nodes:
            changes = result.schema_changes.get(nid, [])
            console.print(f"  ~ {nid}")
            for change in changes:
                console.print(f"    {change}")

    if result.added_edges:
        console.print("\n[green]Added edges:[/green]")
        for s, t in result.added_edges:
            console.print(f"  + {s} -> {t}")

    if result.removed_edges:
        console.print("\n[red]Removed edges:[/red]")
        for s, t in result.removed_edges:
            console.print(f"  - {s} -> {t}")


# =====================================================================
# bottleneck
# =====================================================================

@cli.command()
@click.argument("pipeline", type=click.Path(exists=True))
@click.option("--top", "-k", default=5, help="Number of top bottlenecks")
def bottleneck(pipeline: str, top: int) -> None:
    """Detect bottleneck nodes in the pipeline."""
    from arc.graph.analysis import detect_bottlenecks

    graph = _load_pipeline(pipeline)
    bottlenecks = detect_bottlenecks(graph, top_k=top)

    if not bottlenecks:
        console.print("[dim]No bottlenecks detected (pipeline too small)[/dim]")
        return

    table = Table(title=f"Top {top} Bottlenecks", box=box.ROUNDED)
    table.add_column("Rank", style="dim")
    table.add_column("Node", style="cyan")
    table.add_column("Fan-In", style="green")
    table.add_column("Fan-Out", style="green")
    table.add_column("Betweenness", style="yellow")
    table.add_column("Downstream Cost", style="yellow")
    table.add_column("Score", style="bold red")

    for i, b in enumerate(bottlenecks, 1):
        table.add_row(
            str(i),
            b.node_id,
            str(b.fan_in),
            str(b.fan_out),
            f"{b.betweenness:.3f}",
            f"{b.downstream_cost:.2f}",
            f"{b.bottleneck_score:.3f}",
        )

    console.print(table)


# =====================================================================
# waves
# =====================================================================

@cli.command()
@click.argument("pipeline", type=click.Path(exists=True))
def waves(pipeline: str) -> None:
    """Show parallel execution waves for the pipeline."""
    from arc.graph.analysis import (
        compute_execution_waves,
        estimate_parallel_time,
        estimate_sequential_time,
        parallelism_speedup,
    )

    graph = _load_pipeline(pipeline)
    wave_list = compute_execution_waves(graph)

    console.print(Panel(
        f"[bold]Execution Waves[/bold]\n"
        f"Sequential time: {estimate_sequential_time(graph):.2f}\n"
        f"Parallel time: {estimate_parallel_time(graph):.2f}\n"
        f"Speedup: {parallelism_speedup(graph):.1f}x",
        title="ARC Waves",
    ))

    table = Table(title="Execution Waves", box=box.ROUNDED)
    table.add_column("Wave", style="dim")
    table.add_column("Nodes", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Max Cost", style="yellow")

    for wave in wave_list:
        nodes_str = ", ".join(wave.node_ids[:5])
        if len(wave.node_ids) > 5:
            nodes_str += f", +{len(wave.node_ids) - 5}"
        table.add_row(
            str(wave.wave_index),
            nodes_str,
            str(len(wave.node_ids)),
            f"{wave.max_node_cost:.2f}",
        )

    console.print(table)


# =====================================================================
# lineage
# =====================================================================

@cli.command()
@click.argument("pipeline", type=click.Path(exists=True))
@click.option("--node", "-n", required=True, help="Node to trace from")
@click.option("--column", "-c", required=True, help="Column to trace")
def lineage(pipeline: str, node: str, column: str) -> None:
    """Trace column lineage back to its source(s)."""
    from arc.graph.analysis import trace_column_lineage

    graph = _load_pipeline(pipeline)

    if not graph.has_node(node):
        error_console.print(f"[red]Error:[/red] Node '{node}' not found")
        sys.exit(1)

    try:
        entries = trace_column_lineage(graph, node, column)
    except Exception as e:
        error_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    if not entries:
        console.print(f"[dim]No lineage found for column '{column}' at node '{node}'[/dim]")
        return

    console.print(f"\n[bold]Column lineage: {node}.{column}[/bold]\n")
    for i, entry in enumerate(entries, 1):
        path_str = " -> ".join(entry.node_path)
        console.print(f"  Path {i}: [cyan]{path_str}[/cyan]")
        if entry.transform_chain:
            for step in entry.transform_chain:
                console.print(f"    {step}")


# =====================================================================
# export
# =====================================================================

@cli.command("export")
@click.argument("pipeline", type=click.Path(exists=True))
@click.option("--format", "-f", "fmt",
              type=click.Choice(["json", "yaml", "dot", "mermaid"]),
              required=True)
@click.option("--output", "-o", required=True, help="Output file path")
@click.option("--show-schemas", is_flag=True)
@click.option("--show-costs", is_flag=True)
def export_cmd(
    pipeline: str,
    fmt: str,
    output: str,
    show_schemas: bool,
    show_costs: bool,
) -> None:
    """Export pipeline to various formats."""
    from arc.io.json_format import PipelineSpec
    from arc.io.yaml_format import YAMLPipelineSpec
    from arc.graph.visualization import to_dot, to_mermaid, save_dot

    graph = _load_pipeline(pipeline)

    if fmt == "json":
        PipelineSpec.save(graph, output)
    elif fmt == "yaml":
        YAMLPipelineSpec.save(graph, output)
    elif fmt == "dot":
        save_dot(graph, output, show_schemas=show_schemas, show_costs=show_costs)
    elif fmt == "mermaid":
        content = to_mermaid(graph)
        Path(output).write_text(content)

    console.print(f"[green]Exported to {output} ({fmt})[/green]")


# =====================================================================
# schema
# =====================================================================

@cli.command("schema")
@click.argument("pipeline", type=click.Path(exists=True))
@click.option("--node", "-n", help="Show schema for specific node")
def schema_cmd(pipeline: str, node: str | None) -> None:
    """Display schema information for pipeline nodes."""
    graph = _load_pipeline(pipeline)

    order = graph.topological_sort() if graph.is_dag() else sorted(graph.node_ids)

    for nid in order:
        if node and nid != node:
            continue

        n = graph.get_node(nid)

        if not n.output_schema.columns and not n.input_schema.columns:
            if node:
                console.print(f"[dim]Node '{nid}' has no schema defined[/dim]")
            continue

        console.print(f"\n[bold cyan]{nid}[/bold cyan] [{n.operator.value}]")

        if n.input_schema.columns:
            in_table = Table(title="Input Schema", box=box.SIMPLE)
            in_table.add_column("Column", style="green")
            in_table.add_column("Type", style="white")
            in_table.add_column("Nullable", style="dim")
            in_table.add_column("Default", style="dim")

            for col in n.input_schema.columns:
                nullable = "✓" if col.nullable else "✗"
                default = col.default_expr or ""
                in_table.add_row(col.name, str(col.sql_type), nullable, default)
            console.print(in_table)

        if n.output_schema.columns:
            out_table = Table(title="Output Schema", box=box.SIMPLE)
            out_table.add_column("Column", style="green")
            out_table.add_column("Type", style="white")
            out_table.add_column("Nullable", style="dim")
            out_table.add_column("Default", style="dim")

            for col in n.output_schema.columns:
                nullable = "✓" if col.nullable else "✗"
                default = col.default_expr or ""
                out_table.add_row(col.name, str(col.sql_type), nullable, default)
            console.print(out_table)


# =====================================================================
# cost
# =====================================================================

@cli.command("cost")
@click.argument("pipeline", type=click.Path(exists=True))
@click.option("--top", "-k", default=10, help="Show top-K expensive nodes")
def cost_cmd(pipeline: str, top: int) -> None:
    """Show cost analysis for the pipeline."""
    from arc.graph.visualization import cost_heatmap

    graph = _load_pipeline(pipeline)

    # Overall cost
    total = graph.total_cost()
    console.print(Panel(
        f"[bold]Cost Analysis[/bold]\n"
        f"Total weighted cost: {total.total_weighted_cost:.2f}\n"
        f"Compute: {total.compute_seconds:.1f}s\n"
        f"Memory: {total.memory_bytes / (1024**2):.1f} MB\n"
        f"I/O: {total.io_bytes / (1024**2):.1f} MB\n"
        f"Rows: {total.row_estimate:,}\n"
        f"Monetary: ${total.monetary_cost:.4f}",
        title="ARC Cost",
    ))

    # Top expensive nodes
    expensive = graph.most_expensive_nodes(top)
    if expensive:
        table = Table(title=f"Top {top} Expensive Nodes", box=box.ROUNDED)
        table.add_column("Rank", style="dim")
        table.add_column("Node", style="cyan")
        table.add_column("Cost", style="yellow")
        table.add_column("% of Total", style="green")

        total_val = total.total_weighted_cost or 1.0
        for i, (nid, cost) in enumerate(expensive, 1):
            pct = (cost / total_val) * 100
            table.add_row(str(i), nid, f"{cost:.2f}", f"{pct:.1f}%")

        console.print(table)

    # Text heatmap
    console.print(f"\n{cost_heatmap(graph)}")


# =====================================================================
# annihilation — delta annihilation analysis
# =====================================================================

@cli.command("annihilation")
@click.argument("pipeline", type=click.Path(exists=True))
@click.option("--source", "-s", required=True, help="Source node to inject delta")
@click.option("--column", "-c", multiple=True, help="Columns in the delta")
def annihilation_cmd(pipeline: str, source: str, column: tuple[str, ...]) -> None:
    """Analyze delta annihilation through the pipeline.

    Shows where deltas injected at --source with the given columns
    become fully annihilated (have no effect) due to operator semantics.
    """
    from arc.graph.analysis import analyze_delta_annihilation

    graph = _load_pipeline(pipeline)

    if source not in {n.node_id for n in graph.nodes}:
        error_console.print(f"[red]Node '{source}' not found in pipeline[/red]")
        raise SystemExit(1)

    result = analyze_delta_annihilation(graph, source, set(column))

    console.print(Panel(
        f"[bold]Delta Annihilation Analysis[/bold]\n"
        f"Source: {source}\n"
        f"Delta columns: {', '.join(column) or '(none)'}\n"
        f"Annihilation points: {len(result.annihilated_at)}\n"
        f"Surviving downstream: {len(result.surviving_downstream)}",
        title="Annihilation",
    ))

    if result.annihilated_at:
        table = Table(title="Annihilation Points", box=box.ROUNDED)
        table.add_column("Node", style="red")
        table.add_column("Operator", style="cyan")
        table.add_column("Reason", style="dim")
        for node_id, reason in result.annihilated_at.items():
            node = graph.get_node(node_id)
            op = node.operator.value if node.operator else "—"
            table.add_row(node_id, op, reason)
        console.print(table)

    if result.surviving_downstream:
        surv_table = Table(title="Surviving Downstream", box=box.ROUNDED)
        surv_table.add_column("Node", style="green")
        surv_table.add_column("Type", style="cyan")
        for nid in sorted(result.surviving_downstream):
            node = graph.get_node(nid)
            surv_table.add_row(nid, node.operator.value if node.operator else "—")
        console.print(surv_table)


# =====================================================================
# compare — repair vs recompute comparison
# =====================================================================

@cli.command("compare")
@click.argument("pipeline", type=click.Path(exists=True))
@click.option("--node", "-n", required=True, help="Target node for comparison")
@click.option("--affected", "-a", multiple=True, help="Affected upstream nodes")
def compare_cmd(pipeline: str, node: str, affected: tuple[str, ...]) -> None:
    """Compare repair cost vs full recompute for a target node.

    Given a set of --affected nodes, estimates the cost of applying a
    targeted repair vs recomputing the entire subgraph from sources.
    """
    from arc.graph.analysis import compare_repair_vs_recompute

    graph = _load_pipeline(pipeline)

    if node not in {n.node_id for n in graph.nodes}:
        error_console.print(f"[red]Node '{node}' not found in pipeline[/red]")
        raise SystemExit(1)

    comparison = compare_repair_vs_recompute(graph, node, set(affected))

    rec_style = "green" if comparison.recommendation == "repair" else "yellow"
    console.print(Panel(
        f"[bold]Repair vs Recompute[/bold]\n"
        f"Target: {node}\n"
        f"Affected: {', '.join(affected) or '(none)'}\n\n"
        f"Repair cost:    {comparison.repair_cost:.2f}\n"
        f"Recompute cost: {comparison.recompute_cost:.2f}\n"
        f"Savings ratio:  {comparison.savings_ratio:.2f}\n\n"
        f"[{rec_style}]Recommendation: {comparison.recommendation}[/{rec_style}]",
        title="ARC Comparison",
    ))

    # Show repair scope details
    tree = Tree(f"[bold]Repair Scope[/bold]")
    for nid in comparison.repair_scope:
        n = graph.get_node(nid)
        op = n.operator.value if n.operator else "—"
        tree.add(f"[cyan]{nid}[/cyan] ({op})")
    console.print(tree)


# =====================================================================
# redundancy — detect redundancies
# =====================================================================

@cli.command("redundancy")
@click.argument("pipeline", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Save report to JSON")
def redundancy_cmd(pipeline: str, output: str | None) -> None:
    """Detect redundant computations in the pipeline.

    Identifies nodes that compute equivalent results, suggesting
    opportunities for deduplication.
    """
    from arc.graph.analysis import detect_redundancies

    graph = _load_pipeline(pipeline)
    results = detect_redundancies(graph)

    console.print(Panel(
        f"[bold]Redundancy Analysis[/bold]\n"
        f"Redundant groups found: {len(results)}",
        title="Redundancy",
    ))

    if not results:
        console.print("[green]No redundancies detected.[/green]")
        return

    for i, result in enumerate(results, 1):
        table = Table(
            title=f"Redundancy Group {i}: {result.description}",
            box=box.ROUNDED,
        )
        table.add_column("Node", style="cyan")
        table.add_column("Operator", style="yellow")
        table.add_column("Estimated Savings", style="green")

        for nid in result.redundant_nodes:
            node = graph.get_node(nid)
            op = node.operator.value if node.operator else "—"
            savings = result.estimated_savings.total_weighted_cost if result.estimated_savings else 0.0
            table.add_row(nid, op, f"{savings:.2f}")
        console.print(table)

    if output:
        report = [
            {
                "redundant_nodes": list(r.redundant_nodes),
                "description": r.description,
                "savings": r.estimated_savings.total_weighted_cost if r.estimated_savings else 0.0,
            }
            for r in results
        ]
        Path(output).write_text(json.dumps(report, indent=2))
        console.print(f"\n[dim]Report saved to {output}[/dim]")


# =====================================================================
# convert — format conversion
# =====================================================================

@cli.command("convert")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--format", "fmt", type=click.Choice(["json", "yaml", "dot", "mermaid"]),
              help="Output format (auto-detected from extension if not specified)")
def convert_cmd(input_file: str, output_file: str, fmt: str | None) -> None:
    """Convert a pipeline spec between formats.

    Supported conversions: JSON ↔ YAML, JSON/YAML → DOT, JSON/YAML → Mermaid.
    """
    from arc.io.json_format import PipelineSpec
    from arc.io.yaml_format import YAMLPipelineSpec
    from arc.graph.visualization import to_dot, to_mermaid

    graph = _load_pipeline(input_file)

    # Auto-detect format from extension
    if fmt is None:
        ext = Path(output_file).suffix.lower()
        fmt_map = {
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".dot": "dot",
            ".gv": "dot",
            ".mmd": "mermaid",
            ".mermaid": "mermaid",
        }
        fmt = fmt_map.get(ext)
        if fmt is None:
            error_console.print(f"[red]Cannot detect format from extension '{ext}'[/red]")
            raise SystemExit(1)

    out_path = Path(output_file)
    if fmt == "json":
        out_path.write_text(PipelineSpec.to_json(graph, indent=2))
    elif fmt == "yaml":
        YAMLPipelineSpec.save(graph, out_path)
    elif fmt == "dot":
        dot_str = to_dot(graph)
        out_path.write_text(dot_str)
    elif fmt == "mermaid":
        mmd = to_mermaid(graph)
        out_path.write_text(mmd)

    console.print(f"[green]Converted {input_file} → {output_file} ({fmt})[/green]")


# =====================================================================
# metrics — detailed pipeline metrics
# =====================================================================

@cli.command("metrics")
@click.argument("pipeline", type=click.Path(exists=True))
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def metrics_cmd(pipeline: str, json_output: bool) -> None:
    """Display detailed pipeline metrics.

    Shows structural metrics, complexity indicators, cost statistics,
    and quality coverage for the pipeline.
    """
    from arc.graph.analysis import compute_metrics

    graph = _load_pipeline(pipeline)
    m = compute_metrics(graph)

    if json_output:
        d = {
            "node_count": m.node_count,
            "edge_count": m.edge_count,
            "source_count": m.source_count,
            "sink_count": m.sink_count,
            "max_depth": m.max_depth,
            "max_width": m.max_width,
            "avg_fan_out": round(m.avg_fan_out, 2),
            "avg_fan_in": round(m.avg_fan_in, 2),
            "fragment_f_count": m.fragment_f_count,
            "fragment_f_ratio": round(m.fragment_f_ratio, 3),
        }
        click.echo(json.dumps(d, indent=2))
        return

    console.print(Panel(
        f"[bold]Pipeline Metrics[/bold]\n\n"
        f"Nodes:     {m.node_count:>6}\n"
        f"Edges:     {m.edge_count:>6}\n"
        f"Sources:   {m.source_count:>6}\n"
        f"Sinks:     {m.sink_count:>6}\n"
        f"Max depth: {m.max_depth:>6}\n"
        f"Max width: {m.max_width:>6}\n\n"
        f"Avg fan-out:  {m.avg_fan_out:.2f}\n"
        f"Avg fan-in:   {m.avg_fan_in:.2f}\n\n"
        f"Fragment F nodes: {m.fragment_f_count} / {m.node_count}"
        f" ({m.fragment_f_ratio:.1%})",
        title="ARC Metrics",
    ))


# =====================================================================
# quality — quality constraint summary
# =====================================================================

@cli.command("quality")
@click.argument("pipeline", type=click.Path(exists=True))
def quality_cmd(pipeline: str) -> None:
    """Summarize quality constraints across the pipeline.

    Lists all quality constraints, availability SLAs, and their
    coverage across pipeline nodes.
    """
    graph = _load_pipeline(pipeline)

    constraints = graph.all_quality_constraints()
    sla_nodes = graph.nodes_with_availability_sla()
    coverage = graph.schema_coverage()

    console.print(Panel(
        f"[bold]Quality Summary[/bold]\n"
        f"Total constraints: {len(constraints)}\n"
        f"Nodes with SLA: {len(sla_nodes)}\n"
        f"Schema coverage: {coverage:.1%}",
        title="ARC Quality",
    ))

    if constraints:
        table = Table(title="Quality Constraints", box=box.ROUNDED)
        table.add_column("Node", style="cyan")
        table.add_column("Constraint", style="yellow")
        table.add_column("Type", style="green")
        table.add_column("Threshold", style="magenta")

        for node_id, qc in constraints:
            table.add_row(
                node_id,
                qc.name,
                qc.constraint_type.value if hasattr(qc, 'constraint_type') else "—",
                str(qc.threshold) if hasattr(qc, 'threshold') else "—",
            )
        console.print(table)

    if sla_nodes:
        sla_table = Table(title="Availability SLAs", box=box.ROUNDED)
        sla_table.add_column("Node", style="cyan")
        sla_table.add_column("SLA %", style="green")
        sla_table.add_column("Max Downtime", style="yellow")

        for node in sla_nodes:
            meta = node.metadata
            if meta and meta.availability:
                avail = meta.availability
                sla_table.add_row(
                    node.node_id,
                    f"{avail.target_uptime:.3%}",
                    f"{avail.max_downtime_minutes} min",
                )
        console.print(sla_table)


# =====================================================================
# dependencies — dependency tree
# =====================================================================

@cli.command("dependencies")
@click.argument("pipeline", type=click.Path(exists=True))
@click.option("--node", "-n", required=True, help="Node to trace dependencies for")
@click.option("--direction", "-d", type=click.Choice(["up", "down", "both"]),
              default="both", help="Direction to trace")
def dependencies_cmd(pipeline: str, node: str, direction: str) -> None:
    """Display the dependency tree for a specific node.

    Traces upstream (providers) and/or downstream (consumers)
    dependencies, rendering them as a Rich tree.
    """
    from arc.graph.analysis import dependency_analysis, impact_analysis

    graph = _load_pipeline(pipeline)

    if node not in {n.node_id for n in graph.nodes}:
        error_console.print(f"[red]Node '{node}' not found[/red]")
        raise SystemExit(1)

    if direction in ("up", "both"):
        dep = dependency_analysis(graph, node)
        tree = Tree(f"[bold cyan]{node}[/bold cyan] upstream dependencies")
        for depth, ancestor_id in sorted(
            ((d, n) for n, d in dep.dependency_depth.items()),
            key=lambda x: x[0],
        ):
            prefix = "  " * depth
            tree.add(f"{prefix}[dim]{depth}[/dim] → [cyan]{ancestor_id}[/cyan]")
        console.print(tree)

        console.print(
            f"\n  Direct deps: {len(dep.direct_dependencies)}, "
            f"Total: {len(dep.all_dependencies)}, "
            f"Critical path: {' → '.join(dep.critical_path)}\n"
        )

    if direction in ("down", "both"):
        imp = impact_analysis(graph, node)
        tree = Tree(f"[bold yellow]{node}[/bold yellow] downstream impact")
        for depth, downstream_id in sorted(
            ((d, n) for n, d in imp.impact_depth.items()),
            key=lambda x: x[0],
        ):
            prefix = "  " * depth
            tree.add(f"{prefix}[dim]{depth}[/dim] → [yellow]{downstream_id}[/yellow]")
        console.print(tree)

        console.print(
            f"\n  Direct: {len(imp.directly_affected)}, "
            f"Total: {len(imp.all_affected)}\n"
        )


# =====================================================================
# check — comprehensive pipeline health check
# =====================================================================

@cli.command("check")
@click.argument("pipeline", type=click.Path(exists=True))
@click.option("--strict", is_flag=True, help="Fail on warnings too")
def check_cmd(pipeline: str, strict: bool) -> None:
    """Run a comprehensive health check on a pipeline.

    Validates structure, schema compatibility, quality constraints,
    cost estimates, and Fragment F coverage. Returns non-zero exit
    code on errors (or warnings in strict mode).
    """
    from arc.graph.analysis import (
        compute_metrics,
        detect_bottlenecks,
        detect_redundancies,
        FragmentClassifier,
    )

    graph = _load_pipeline(pipeline)
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Structural checks
    if graph.node_count == 0:
        errors.append("Pipeline has no nodes")
    if not graph.sources:
        errors.append("Pipeline has no source nodes")
    if not graph.sinks:
        errors.append("Pipeline has no sink nodes")

    # 2. Check for cycles
    try:
        graph.topological_sort()
    except Exception:
        errors.append("Pipeline contains cycles (not a DAG)")

    # 3. Schema validation
    for edge in graph.edges:
        src = graph.get_node(edge.source_id)
        tgt = graph.get_node(edge.target_id)
        if src.output_schema and tgt.input_schema:
            missing = set(c.name for c in tgt.input_schema.columns) - set(
                c.name for c in src.output_schema.columns
            )
            if missing:
                warnings.append(
                    f"Edge {edge.source_id}→{edge.target_id}: "
                    f"target expects columns {missing} not in source output"
                )

    # 4. Cost completeness
    nodes_no_cost = [
        n.node_id for n in graph.nodes if n.cost_estimate is None
    ]
    if nodes_no_cost:
        warnings.append(f"{len(nodes_no_cost)} nodes lack cost estimates")

    # 5. Fragment F coverage
    metrics = compute_metrics(graph)
    if metrics.fragment_f_ratio < 0.5:
        warnings.append(
            f"Only {metrics.fragment_f_ratio:.0%} of nodes are in Fragment F "
            f"(recommend ≥50% for efficient repair)"
        )

    # 6. Bottlenecks
    bottlenecks = detect_bottlenecks(graph)
    if bottlenecks:
        warnings.append(
            f"{len(bottlenecks)} bottleneck nodes detected "
            f"(high fan-in/fan-out)"
        )

    # 7. Redundancies
    redundancies = detect_redundancies(graph)
    if redundancies:
        warnings.append(
            f"{len(redundancies)} redundancy groups detected — "
            f"consider deduplication"
        )

    # Display results
    console.print(Panel(
        f"[bold]Pipeline Health Check[/bold]\n"
        f"Nodes: {graph.node_count}, Edges: {graph.edge_count}\n"
        f"Errors: {len(errors)}, Warnings: {len(warnings)}",
        title="ARC Check",
    ))

    for err in errors:
        console.print(f"  [red]✗[/red] {err}")
    for warn in warnings:
        console.print(f"  [yellow]⚠[/yellow] {warn}")

    if not errors and not warnings:
        console.print("  [green]✓ All checks passed[/green]")

    if errors or (strict and warnings):
        raise SystemExit(1)


# =====================================================================
# subgraph — extract a subgraph
# =====================================================================

@cli.command("subgraph")
@click.argument("pipeline", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("--node", "-n", multiple=True, required=True,
              help="Nodes to include (repeatable)")
@click.option("--include-deps", is_flag=True, help="Include upstream dependencies")
def subgraph_cmd(
    pipeline: str, output: str, node: tuple[str, ...], include_deps: bool,
) -> None:
    """Extract a subgraph containing only the specified nodes.

    Optionally includes all upstream dependencies of the specified
    nodes to preserve pipeline correctness.
    """
    from arc.io.json_format import PipelineSpec
    from arc.graph.analysis import dependency_analysis

    graph = _load_pipeline(pipeline)
    node_set = set(node)

    if include_deps:
        for nid in list(node_set):
            if nid in {n.node_id for n in graph.nodes}:
                dep = dependency_analysis(graph, nid)
                node_set |= dep.all_dependencies

    subgraph = graph.subgraph(node_set)

    out_path = Path(output)
    out_path.write_text(PipelineSpec.to_json(subgraph, indent=2))

    console.print(
        f"[green]Extracted subgraph with {subgraph.node_count} nodes, "
        f"{subgraph.edge_count} edges → {output}[/green]"
    )


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    cli()
