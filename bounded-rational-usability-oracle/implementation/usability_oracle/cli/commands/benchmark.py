"""
usability_oracle.cli.commands.benchmark — Run benchmark suite.

Implements the ``usability-oracle benchmark`` command which runs
performance and accuracy benchmarks across synthetic and real-world
UI examples.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Optional

import click

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

_SUITES: dict[str, list[dict[str, Any]]] = {
    "small": [
        {
            "name": "simple_form",
            "description": "3-field login form",
            "n_nodes": 10,
            "n_interactive": 3,
        },
        {
            "name": "nav_menu",
            "description": "5-item navigation menu",
            "n_nodes": 15,
            "n_interactive": 5,
        },
    ],
    "medium": [
        {
            "name": "dashboard",
            "description": "Analytics dashboard with 20 widgets",
            "n_nodes": 80,
            "n_interactive": 20,
        },
        {
            "name": "settings_page",
            "description": "Settings page with tabs and sections",
            "n_nodes": 60,
            "n_interactive": 25,
        },
        {
            "name": "data_table",
            "description": "Sortable data table with pagination",
            "n_nodes": 100,
            "n_interactive": 30,
        },
    ],
    "large": [
        {
            "name": "email_client",
            "description": "Full email client UI",
            "n_nodes": 500,
            "n_interactive": 80,
        },
        {
            "name": "ide_layout",
            "description": "IDE-like layout with panels",
            "n_nodes": 300,
            "n_interactive": 100,
        },
    ],
}


def benchmark_command(
    suite: str = "small",
    n_runs: int = 3,
    output: Optional[str] = None,
) -> int:
    """Run the benchmark suite.

    Parameters
    ----------
    suite : str
        Which suite to run: small, medium, large, all.
    n_runs : int
        Number of runs per benchmark.
    output : str, optional
        Output file path for results.

    Returns
    -------
    int
        Exit code: 0 = success, 2 = error.
    """
    from usability_oracle.accessibility.models import (
        AccessibilityNode,
        AccessibilityState,
        AccessibilityTree,
        BoundingBox,
    )
    from usability_oracle.pipeline.config import FullPipelineConfig
    from usability_oracle.pipeline.runner import PipelineRunner

    try:
        if suite == "all":
            benchmarks = []
            for s in ("small", "medium", "large"):
                benchmarks.extend(_SUITES[s])
        else:
            benchmarks = _SUITES.get(suite, _SUITES["small"])

        click.echo(f"Running {len(benchmarks)} benchmarks × {n_runs} runs\n")

        results: list[dict[str, Any]] = []

        for bm in benchmarks:
            name = bm["name"]
            click.echo(f"  {name}: ", nl=False)

            # Generate synthetic tree
            tree = _generate_synthetic_tree(
                n_nodes=bm["n_nodes"],
                n_interactive=bm["n_interactive"],
            )

            timings: list[float] = []
            for run in range(n_runs):
                cfg = FullPipelineConfig.DEFAULT()
                cfg.stages["compare"].enabled = False
                cfg.stages["repair"].enabled = False

                runner = PipelineRunner(config=cfg)

                t0 = time.monotonic()
                # Run core stages with the pre-built tree
                from usability_oracle.pipeline.stages import (
                    CostStageExecutor,
                    MDPStageExecutor,
                    BottleneckStageExecutor,
                )

                cost_exec = CostStageExecutor()
                cost_result = cost_exec.execute(tree=tree)

                mdp_exec = MDPStageExecutor()
                mdp_result = mdp_exec.execute(tree=tree)

                bn_exec = BottleneckStageExecutor()
                bn_result = bn_exec.execute(mdp=mdp_result)

                elapsed = time.monotonic() - t0
                timings.append(elapsed)

            avg = statistics.mean(timings)
            std = statistics.stdev(timings) if len(timings) > 1 else 0.0

            click.echo(f"{avg:.3f}s ± {std:.3f}s")

            results.append({
                "name": name,
                "description": bm["description"],
                "n_nodes": bm["n_nodes"],
                "n_interactive": bm["n_interactive"],
                "n_runs": n_runs,
                "mean_time": avg,
                "std_time": std,
                "min_time": min(timings),
                "max_time": max(timings),
            })

        # Output
        click.echo(f"\n{'='*60}")
        click.echo("Benchmark Results Summary")
        click.echo(f"{'='*60}")

        for r in results:
            click.echo(
                f"  {r['name']:20s}  "
                f"nodes={r['n_nodes']:4d}  "
                f"interactive={r['n_interactive']:3d}  "
                f"time={r['mean_time']:.3f}s"
            )

        if output:
            output_path = Path(output)
            output_path.write_text(
                json.dumps(results, indent=2),
                encoding="utf-8",
            )
            click.echo(f"\nDetailed results written to {output}")

        return 0

    except Exception as exc:
        logger.exception("Benchmark command failed")
        click.echo(f"Error: {exc}", err=True)
        return 2


def _generate_synthetic_tree(
    n_nodes: int, n_interactive: int
) -> Any:
    """Generate a synthetic accessibility tree for benchmarking."""
    from usability_oracle.accessibility.models import (
        AccessibilityNode,
        AccessibilityState,
        AccessibilityTree,
        BoundingBox,
    )

    import random
    random.seed(42)

    interactive_roles = ["button", "link", "textfield", "checkbox"]
    static_roles = ["generic", "heading", "region", "list"]

    children: list[AccessibilityNode] = []

    for i in range(min(n_interactive, n_nodes)):
        role = interactive_roles[i % len(interactive_roles)]
        children.append(AccessibilityNode(
            id=f"node_{i}",
            role=role,
            name=f"{role.title()} {i}",
            bounding_box=BoundingBox(
                x=random.uniform(0, 800),
                y=random.uniform(0, 600),
                width=random.uniform(20, 100),
                height=random.uniform(20, 60),
            ),
            state=AccessibilityState(),
        ))

    for i in range(n_interactive, n_nodes):
        role = static_roles[i % len(static_roles)]
        children.append(AccessibilityNode(
            id=f"node_{i}",
            role=role,
            name=f"{role.title()} {i}",
            bounding_box=BoundingBox(
                x=random.uniform(0, 800),
                y=random.uniform(0, 600),
                width=random.uniform(50, 200),
                height=random.uniform(20, 100),
            ),
            state=AccessibilityState(),
        ))

    root = AccessibilityNode(
        id="root",
        role="document",
        name="Benchmark Page",
        children=children,
        state=AccessibilityState(),
        bounding_box=BoundingBox(x=0, y=0, width=1024, height=768),
    )

    return AccessibilityTree(root=root)
