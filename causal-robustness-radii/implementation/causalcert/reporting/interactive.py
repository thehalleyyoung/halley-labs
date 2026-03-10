"""
Interactive analysis reporting for CausalCert.

Provides step-by-step walkthrough, what-if scenario analysis, comparison
mode, sensitivity knobs, and session export.
"""

from __future__ import annotations

import copy
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from causalcert.types import (
    AdjacencyMatrix,
    AuditReport,
    EditType,
    EstimationResult,
    FragilityChannel,
    FragilityScore,
    NodeId,
    NodeSet,
    RobustnessRadius,
    SolverStrategy,
    StructuralEdit,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WhatIfResult:
    """Result of a what-if scenario."""
    description: str
    original_radius: RobustnessRadius
    modified_radius: RobustnessRadius | None
    radius_change: int
    fragility_change: list[tuple[tuple[int, int], float]]
    new_n_edges: int
    is_still_dag: bool


@dataclass(slots=True)
class ComparisonReport:
    """Comparison between two DAG analyses."""
    dag_a_name: str
    dag_b_name: str
    shared_edges: list[tuple[int, int]]
    edges_only_in_a: list[tuple[int, int]]
    edges_only_in_b: list[tuple[int, int]]
    radius_a: int
    radius_b: int
    fragility_correlation: float
    summary: str


@dataclass(slots=True)
class SensitivityKnobResult:
    """Result of varying a sensitivity knob."""
    parameter_name: str
    values: list[Any]
    radii: list[int]
    top_fragile_edges: list[list[tuple[int, int]]]


@dataclass(slots=True)
class AnalysisSession:
    """Recorded interactive analysis session."""
    steps: list[dict[str, Any]] = field(default_factory=list)
    whatif_results: list[WhatIfResult] = field(default_factory=list)
    comparisons: list[ComparisonReport] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_dag(adj: np.ndarray) -> bool:
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int).copy()
    queue = deque(i for i in range(n) if in_deg[i] == 0)
    count = 0
    while queue:
        v = queue.popleft()
        count += 1
        for c in range(n):
            if adj[v, c]:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
    return count == n


def _compute_radius_simple(
    adj: np.ndarray, treatment: int, outcome: int
) -> int:
    """Simple BFS-based upper bound on robustness radius."""
    if treatment == outcome:
        return 0
    visited: set[int] = set()
    queue = deque([treatment])
    while queue:
        node = queue.popleft()
        for c in np.nonzero(adj[node])[0]:
            c = int(c)
            if c == outcome:
                # Path exists
                from causalcert.dag.paths import shortest_directed_path
                sp = shortest_directed_path(adj, treatment, outcome)
                return len(sp) - 1 if sp else 0
            if c not in visited:
                visited.add(c)
                queue.append(c)
    return 0


# ---------------------------------------------------------------------------
# Interactive walkthrough
# ---------------------------------------------------------------------------


class InteractiveAnalysis:
    """Step-by-step interactive analysis of causal robustness.

    Provides methods for exploring the analysis results, running what-if
    scenarios, comparing DAGs, and exporting sessions.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Original DAG adjacency matrix.
    treatment : NodeId
        Treatment node.
    outcome : NodeId
        Outcome node.
    audit_report : AuditReport | None
        Pre-computed audit report (optional).
    node_names : list[str] | None
        Human-readable node names.
    """

    def __init__(
        self,
        adj: AdjacencyMatrix,
        treatment: NodeId,
        outcome: NodeId,
        audit_report: AuditReport | None = None,
        node_names: list[str] | None = None,
    ) -> None:
        self._adj = np.asarray(adj, dtype=np.int8).copy()
        self._treatment = treatment
        self._outcome = outcome
        self._report = audit_report
        self._node_names = node_names or [f"X{i}" for i in range(adj.shape[0])]
        self._session = AnalysisSession()

    @property
    def session(self) -> AnalysisSession:
        return self._session

    # -- Step 1: DAG overview -----------------------------------------------

    def dag_overview(self) -> dict[str, Any]:
        """Get an overview of the DAG structure."""
        adj = self._adj
        n = adj.shape[0]
        n_edges = int(adj.sum())
        density = n_edges / (n * (n - 1)) if n > 1 else 0.0

        # In/out degrees
        in_deg = adj.sum(axis=0).astype(int)
        out_deg = adj.sum(axis=1).astype(int)

        overview = {
            "n_nodes": n,
            "n_edges": n_edges,
            "density": density,
            "treatment": self._node_names[self._treatment],
            "outcome": self._node_names[self._outcome],
            "max_in_degree": int(in_deg.max()),
            "max_out_degree": int(out_deg.max()),
            "roots": [self._node_names[i] for i in range(n) if in_deg[i] == 0],
            "leaves": [self._node_names[i] for i in range(n) if out_deg[i] == 0],
        }

        self._session.steps.append({"action": "dag_overview", "result": overview})
        return overview

    # -- Step 2: Path analysis ----------------------------------------------

    def path_analysis(self) -> dict[str, Any]:
        """Analyze causal and backdoor paths."""
        from causalcert.dag.paths import (
            all_directed_paths,
            backdoor_paths,
            shortest_directed_path,
        )

        causal = all_directed_paths(self._adj, self._treatment, self._outcome)
        backdoor = backdoor_paths(self._adj, self._treatment, self._outcome)
        shortest = shortest_directed_path(self._adj, self._treatment, self._outcome)

        def format_path(p: list[int]) -> str:
            return " → ".join(self._node_names[i] for i in p)

        result = {
            "n_causal_paths": len(causal),
            "n_backdoor_paths": len(backdoor),
            "causal_paths": [format_path(p) for p in causal],
            "backdoor_paths": [format_path(p) for p in backdoor],
            "shortest_causal_path": format_path(shortest) if shortest else None,
        }

        self._session.steps.append({"action": "path_analysis", "result": result})
        return result

    # -- Step 3: Fragility summary ------------------------------------------

    def fragility_summary(self) -> dict[str, Any]:
        """Summarize fragility scores."""
        if self._report is None:
            return {"error": "No audit report available"}

        ranking = self._report.fragility_ranking
        result: dict[str, Any] = {
            "n_scored_edges": len(ranking),
            "top_5": [],
        }

        for fs in ranking[:5]:
            result["top_5"].append({
                "edge": f"{self._node_names[fs.edge[0]]} → {self._node_names[fs.edge[1]]}",
                "score": fs.total_score,
                "channels": {
                    ch.value: sc for ch, sc in fs.channel_scores.items()
                },
            })

        if ranking:
            scores = [fs.total_score for fs in ranking]
            result["mean_score"] = float(np.mean(scores))
            result["max_score"] = float(np.max(scores))
            result["min_score"] = float(np.min(scores))

        self._session.steps.append({"action": "fragility_summary", "result": result})
        return result

    # -- What-if scenarios --------------------------------------------------

    def what_if_add_edge(
        self, source: NodeId, target: NodeId
    ) -> WhatIfResult:
        """What if we add edge source → target?"""
        return self._what_if_edit(
            f"Add edge {self._node_names[source]} → {self._node_names[target]}",
            [(EditType.ADD, source, target)],
        )

    def what_if_remove_edge(
        self, source: NodeId, target: NodeId
    ) -> WhatIfResult:
        """What if we remove edge source → target?"""
        return self._what_if_edit(
            f"Remove edge {self._node_names[source]} → {self._node_names[target]}",
            [(EditType.DELETE, source, target)],
        )

    def what_if_reverse_edge(
        self, source: NodeId, target: NodeId
    ) -> WhatIfResult:
        """What if we reverse edge source → target?"""
        return self._what_if_edit(
            f"Reverse edge {self._node_names[source]} → {self._node_names[target]}",
            [(EditType.REVERSE, source, target)],
        )

    def what_if_multi_edit(
        self,
        edits: list[tuple[EditType, NodeId, NodeId]],
        description: str = "",
    ) -> WhatIfResult:
        """What if we apply multiple edits simultaneously?"""
        if not description:
            description = f"Apply {len(edits)} edits"
        return self._what_if_edit(description, edits)

    def _what_if_edit(
        self,
        description: str,
        edits: list[tuple[EditType, int, int]],
    ) -> WhatIfResult:
        """Internal what-if computation."""
        modified = self._adj.copy()

        for edit_type, src, tgt in edits:
            if edit_type == EditType.ADD:
                modified[src, tgt] = 1
            elif edit_type == EditType.DELETE:
                modified[src, tgt] = 0
            elif edit_type == EditType.REVERSE:
                modified[src, tgt] = 0
                modified[tgt, src] = 1

        still_dag = _is_dag(modified)
        new_n_edges = int(modified.sum())

        original_radius = (
            self._report.radius if self._report
            else RobustnessRadius(lower_bound=0, upper_bound=0)
        )

        modified_radius = None
        radius_change = 0
        fragility_change: list[tuple[tuple[int, int], float]] = []

        if still_dag:
            new_ub = _compute_radius_simple(modified, self._treatment, self._outcome)
            modified_radius = RobustnessRadius(
                lower_bound=0,
                upper_bound=new_ub,
                solver_strategy=SolverStrategy.AUTO,
            )
            radius_change = new_ub - original_radius.upper_bound

            # Compute fragility changes
            from causalcert.fragility.theoretical import structural_fragility_ranking
            try:
                new_scores = structural_fragility_ranking(
                    modified, self._treatment, self._outcome
                )
                old_map = {}
                if self._report:
                    old_map = {fs.edge: fs.total_score for fs in self._report.fragility_ranking}
                for fs in new_scores:
                    old_score = old_map.get(fs.edge, 0.0)
                    if abs(fs.total_score - old_score) > 0.01:
                        fragility_change.append((fs.edge, fs.total_score - old_score))
            except Exception:
                pass

        result = WhatIfResult(
            description=description,
            original_radius=original_radius,
            modified_radius=modified_radius,
            radius_change=radius_change,
            fragility_change=fragility_change,
            new_n_edges=new_n_edges,
            is_still_dag=still_dag,
        )

        self._session.whatif_results.append(result)
        return result

    # -- DAG comparison -----------------------------------------------------

    def compare_with(
        self,
        other_adj: AdjacencyMatrix,
        other_name: str = "DAG B",
        self_name: str = "DAG A",
    ) -> ComparisonReport:
        """Compare this DAG with another."""
        adj_a = self._adj
        adj_b = np.asarray(other_adj, dtype=np.int8)

        edges_a = set()
        edges_b = set()
        for i in range(adj_a.shape[0]):
            for j in range(adj_a.shape[0]):
                if adj_a[i, j]:
                    edges_a.add((i, j))
                if i < adj_b.shape[0] and j < adj_b.shape[0] and adj_b[i, j]:
                    edges_b.add((i, j))

        shared = sorted(edges_a & edges_b)
        only_a = sorted(edges_a - edges_b)
        only_b = sorted(edges_b - edges_a)

        radius_a = _compute_radius_simple(adj_a, self._treatment, self._outcome)
        radius_b = _compute_radius_simple(adj_b, self._treatment, self._outcome)

        # Fragility correlation
        from causalcert.fragility.theoretical import structural_fragility_ranking
        try:
            scores_a = structural_fragility_ranking(adj_a, self._treatment, self._outcome)
            scores_b = structural_fragility_ranking(adj_b, self._treatment, self._outcome)
            map_a = {fs.edge: fs.total_score for fs in scores_a}
            map_b = {fs.edge: fs.total_score for fs in scores_b}
            common_edges = set(map_a.keys()) & set(map_b.keys())
            if len(common_edges) >= 2:
                vals_a = [map_a[e] for e in common_edges]
                vals_b = [map_b[e] for e in common_edges]
                corr = float(np.corrcoef(vals_a, vals_b)[0, 1])
            else:
                corr = 0.0
        except Exception:
            corr = 0.0

        summary_lines = [
            f"Comparing {self_name} vs {other_name}:",
            f"  Shared edges: {len(shared)}",
            f"  Only in {self_name}: {len(only_a)}",
            f"  Only in {other_name}: {len(only_b)}",
            f"  Radius {self_name}: {radius_a}, {other_name}: {radius_b}",
            f"  Fragility correlation: {corr:.3f}",
        ]

        report = ComparisonReport(
            dag_a_name=self_name,
            dag_b_name=other_name,
            shared_edges=shared,
            edges_only_in_a=only_a,
            edges_only_in_b=only_b,
            radius_a=radius_a,
            radius_b=radius_b,
            fragility_correlation=corr,
            summary="\n".join(summary_lines),
        )

        self._session.comparisons.append(report)
        return report

    # -- Sensitivity knobs --------------------------------------------------

    def vary_alpha(
        self,
        alphas: Sequence[float] = (0.01, 0.025, 0.05, 0.1, 0.2),
    ) -> SensitivityKnobResult:
        """Vary alpha and observe impact on results.

        Since structural fragility is alpha-independent, this shows
        the theoretical sensitivity envelope.
        """
        from causalcert.fragility.theoretical import structural_fragility_ranking

        radii: list[int] = []
        top_edges: list[list[tuple[int, int]]] = []

        for alpha in alphas:
            scores = structural_fragility_ranking(
                self._adj, self._treatment, self._outcome
            )
            radius = _compute_radius_simple(
                self._adj, self._treatment, self._outcome
            )
            radii.append(radius)
            top_edges.append([fs.edge for fs in scores[:3]])

        return SensitivityKnobResult(
            parameter_name="alpha",
            values=list(alphas),
            radii=radii,
            top_fragile_edges=top_edges,
        )

    def vary_max_k(
        self,
        k_values: Sequence[int] = (1, 2, 3, 5, 10),
    ) -> SensitivityKnobResult:
        """Vary max_k and observe impact."""
        from causalcert.fragility.theoretical import structural_fragility_ranking

        base_radius = _compute_radius_simple(
            self._adj, self._treatment, self._outcome
        )
        scores = structural_fragility_ranking(
            self._adj, self._treatment, self._outcome
        )

        radii = [min(k, base_radius) for k in k_values]
        top_edges = [[fs.edge for fs in scores[:3]] for _ in k_values]

        return SensitivityKnobResult(
            parameter_name="max_k",
            values=list(k_values),
            radii=radii,
            top_fragile_edges=top_edges,
        )

    # -- Session management -------------------------------------------------

    def add_note(self, note: str) -> None:
        """Add a note to the session."""
        self._session.notes.append(note)

    def export_session(self, path: str | Path) -> None:
        """Export the analysis session to a JSON file."""
        session_data = {
            "n_nodes": int(self._adj.shape[0]),
            "n_edges": int(self._adj.sum()),
            "treatment": self._treatment,
            "outcome": self._outcome,
            "steps": self._session.steps,
            "whatif_count": len(self._session.whatif_results),
            "comparison_count": len(self._session.comparisons),
            "notes": self._session.notes,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(session_data, indent=2, default=str))

    def session_summary(self) -> str:
        """Return a summary of the current session."""
        lines = [
            "=== Interactive Analysis Session ===",
            f"Steps taken: {len(self._session.steps)}",
            f"What-if scenarios: {len(self._session.whatif_results)}",
            f"Comparisons: {len(self._session.comparisons)}",
            f"Notes: {len(self._session.notes)}",
        ]
        for i, step in enumerate(self._session.steps):
            lines.append(f"  Step {i+1}: {step['action']}")
        return "\n".join(lines)
