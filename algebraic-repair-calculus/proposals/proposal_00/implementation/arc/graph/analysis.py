"""
Graph analysis utilities for pipeline DAGs.

Provides impact analysis, dependency analysis, bottleneck detection,
redundancy detection, Fragment F classification, and pipeline complexity
metrics — all operating on :class:`PipelineGraph`.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from typing import Any, Sequence

import attr
from attr import validators as v

from arc.types.base import CostEstimate, Schema
from arc.types.errors import NodeNotFoundError
from arc.types.operators import OperatorProperties, SQLOperator
from arc.graph.pipeline import PipelineGraph, PipelineNode


# =====================================================================
# Impact analysis
# =====================================================================

@attr.s(frozen=True, slots=True, repr=True)
class ImpactResult:
    """Result of an impact analysis: what downstream nodes are affected."""

    origin_node: str = attr.ib(validator=v.instance_of(str))
    affected_nodes: tuple[str, ...] = attr.ib(converter=tuple)
    affected_edges: tuple[tuple[str, str], ...] = attr.ib(converter=tuple)
    total_recompute_cost: CostEstimate = attr.ib(factory=CostEstimate.zero)
    fragment_f_fraction: float = attr.ib(default=0.0)
    max_depth: int = attr.ib(default=0)

    def __str__(self) -> str:
        return (
            f"Impact({self.origin_node}): {len(self.affected_nodes)} nodes affected, "
            f"cost={self.total_recompute_cost.total_weighted_cost:.2f}, "
            f"F-fraction={self.fragment_f_fraction:.1%}"
        )


def impact_analysis(graph: PipelineGraph, node_id: str) -> ImpactResult:
    """Compute the downstream impact of a perturbation at *node_id*.

    Uses BFS from *node_id* through all successors to identify every
    node and edge that would need recomputation.
    """
    if not graph.has_node(node_id):
        raise NodeNotFoundError(node_id)

    desc = graph.descendants(node_id)
    affected = sorted(desc)

    affected_edges: list[tuple[str, str]] = []
    for (s, t) in graph.edges:
        if s in desc or t in desc or s == node_id:
            if t in desc:
                affected_edges.append((s, t))

    total_cost = CostEstimate.zero()
    f_count = 0
    for nid in affected:
        node = graph.get_node(nid)
        total_cost = total_cost + node.cost_estimate
        if node.in_fragment_f:
            f_count += 1

    f_fraction = f_count / len(affected) if affected else 1.0

    # Compute max depth from origin
    max_depth = 0
    if affected:
        depths: dict[str, int] = {node_id: 0}
        queue = deque([node_id])
        while queue:
            cur = queue.popleft()
            for succ in graph.successors(cur):
                if succ not in depths:
                    depths[succ] = depths[cur] + 1
                    max_depth = max(max_depth, depths[succ])
                    queue.append(succ)

    return ImpactResult(
        origin_node=node_id,
        affected_nodes=tuple(affected),
        affected_edges=tuple(affected_edges),
        total_recompute_cost=total_cost,
        fragment_f_fraction=f_fraction,
        max_depth=max_depth,
    )


# =====================================================================
# Dependency analysis
# =====================================================================

@attr.s(frozen=True, slots=True, repr=True)
class DependencyResult:
    """Result of a dependency analysis for a single node."""

    node_id: str = attr.ib(validator=v.instance_of(str))
    direct_dependencies: tuple[str, ...] = attr.ib(converter=tuple)
    transitive_dependencies: tuple[str, ...] = attr.ib(converter=tuple)
    source_dependencies: tuple[str, ...] = attr.ib(converter=tuple)
    depth: int = attr.ib(default=0)

    def __str__(self) -> str:
        return (
            f"Deps({self.node_id}): "
            f"{len(self.direct_dependencies)} direct, "
            f"{len(self.transitive_dependencies)} transitive, "
            f"{len(self.source_dependencies)} sources, "
            f"depth={self.depth}"
        )


def dependency_analysis(graph: PipelineGraph, node_id: str) -> DependencyResult:
    """Analyse what a node depends on (upstream)."""
    if not graph.has_node(node_id):
        raise NodeNotFoundError(node_id)

    direct = graph.predecessors(node_id)
    transitive = sorted(graph.ancestors(node_id))
    sources = [nid for nid in transitive if graph.in_degree(nid) == 0]
    if graph.in_degree(node_id) == 0:
        sources = [node_id]

    # Compute depth from sources
    depth = 0
    if transitive:
        for src in graph.sources():
            path = graph.find_path(src, node_id)
            if path:
                depth = max(depth, len(path) - 1)

    return DependencyResult(
        node_id=node_id,
        direct_dependencies=tuple(sorted(direct)),
        transitive_dependencies=tuple(transitive),
        source_dependencies=tuple(sorted(sources)),
        depth=depth,
    )


# =====================================================================
# Bottleneck detection
# =====================================================================

@attr.s(frozen=True, slots=True, repr=True)
class BottleneckResult:
    """A potential bottleneck node."""

    node_id: str = attr.ib(validator=v.instance_of(str))
    fan_in: int = attr.ib(default=0)
    fan_out: int = attr.ib(default=0)
    betweenness: float = attr.ib(default=0.0)
    downstream_cost: float = attr.ib(default=0.0)
    bottleneck_score: float = attr.ib(default=0.0)

    def __str__(self) -> str:
        return (
            f"Bottleneck({self.node_id}): "
            f"fan_in={self.fan_in}, fan_out={self.fan_out}, "
            f"score={self.bottleneck_score:.3f}"
        )


def detect_bottlenecks(graph: PipelineGraph, top_k: int = 5) -> list[BottleneckResult]:
    """Identify the top-K bottleneck nodes by a composite score.

    The score combines fan-out (high fan-out = many dependents), fan-in,
    betweenness centrality, and downstream recomputation cost.
    """
    import networkx as nx

    if graph.node_count == 0:
        return []

    betweenness = nx.betweenness_centrality(graph.nx_graph)

    results: list[BottleneckResult] = []
    for nid in graph.node_ids:
        fi = graph.in_degree(nid)
        fo = graph.out_degree(nid)
        bc = betweenness.get(nid, 0.0)

        # Downstream cost
        desc = graph.descendants(nid)
        dc = sum(
            graph.get_node(d).cost_estimate.total_weighted_cost
            for d in desc
        )

        # Composite score: weighted combination
        max_cost = max(
            (graph.get_node(n).cost_estimate.total_weighted_cost for n in graph.node_ids),
            default=1.0,
        ) or 1.0
        score = (
            0.3 * (fo / max(graph.node_count, 1))
            + 0.2 * bc
            + 0.3 * (dc / (max_cost * max(graph.node_count, 1)))
            + 0.2 * (fi / max(graph.node_count, 1))
        )

        results.append(BottleneckResult(
            node_id=nid,
            fan_in=fi,
            fan_out=fo,
            betweenness=bc,
            downstream_cost=dc,
            bottleneck_score=score,
        ))

    results.sort(key=lambda r: r.bottleneck_score, reverse=True)
    return results[:top_k]


# =====================================================================
# Redundancy detection
# =====================================================================

@attr.s(frozen=True, slots=True, repr=True)
class RedundancyResult:
    """A pair of nodes that appear to compute the same thing."""

    node_a: str = attr.ib(validator=v.instance_of(str))
    node_b: str = attr.ib(validator=v.instance_of(str))
    similarity: float = attr.ib(default=0.0)
    reason: str = attr.ib(default="")


def detect_redundancies(graph: PipelineGraph) -> list[RedundancyResult]:
    """Find pairs of nodes that may be redundant (duplicate computations).

    Checks for identical operator + same input sources + same output schema.
    """
    results: list[RedundancyResult] = []
    nodes = list(graph.nodes.values())

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            similarity = _node_similarity(graph, a, b)
            if similarity >= 0.8:
                results.append(RedundancyResult(
                    node_a=a.node_id,
                    node_b=b.node_id,
                    similarity=similarity,
                    reason=_redundancy_reason(graph, a, b),
                ))

    results.sort(key=lambda r: r.similarity, reverse=True)
    return results


def _node_similarity(graph: PipelineGraph, a: PipelineNode, b: PipelineNode) -> float:
    """Compute a [0, 1] similarity score between two nodes."""
    score = 0.0
    weights = 0.0

    # Same operator type
    w = 0.3
    weights += w
    if a.operator == b.operator:
        score += w

    # Same query text
    w = 0.3
    weights += w
    if a.query_text and b.query_text and a.query_text.strip() == b.query_text.strip():
        score += w

    # Same output schema columns
    w = 0.2
    weights += w
    a_cols = a.output_schema.column_names
    b_cols = b.output_schema.column_names
    if a_cols and b_cols:
        intersection = len(a_cols & b_cols)
        union = len(a_cols | b_cols)
        if union > 0:
            score += w * (intersection / union)

    # Same upstream nodes
    w = 0.2
    weights += w
    a_preds = set(graph.predecessors(a.node_id))
    b_preds = set(graph.predecessors(b.node_id))
    if a_preds and b_preds:
        intersection = len(a_preds & b_preds)
        union = len(a_preds | b_preds)
        if union > 0:
            score += w * (intersection / union)

    return score / weights if weights > 0 else 0.0


def _redundancy_reason(graph: PipelineGraph, a: PipelineNode, b: PipelineNode) -> str:
    reasons: list[str] = []
    if a.operator == b.operator:
        reasons.append(f"same operator ({a.operator.value})")
    if a.query_text and a.query_text.strip() == b.query_text.strip():
        reasons.append("identical query")
    if a.output_schema.column_names == b.output_schema.column_names:
        reasons.append("same output columns")
    a_preds = set(graph.predecessors(a.node_id))
    b_preds = set(graph.predecessors(b.node_id))
    if a_preds == b_preds:
        reasons.append("same inputs")
    return "; ".join(reasons) if reasons else "structural similarity"


# =====================================================================
# Fragment F classifier
# =====================================================================

@attr.s(frozen=True, slots=True, repr=True)
class FragmentClassification:
    """Classification of nodes into Fragment F vs non-F."""

    fragment_f_nodes: tuple[str, ...] = attr.ib(converter=tuple)
    non_fragment_f_nodes: tuple[str, ...] = attr.ib(converter=tuple)
    violations: dict[str, list[str]] = attr.ib(factory=dict)

    @property
    def fragment_f_fraction(self) -> float:
        total = len(self.fragment_f_nodes) + len(self.non_fragment_f_nodes)
        return len(self.fragment_f_nodes) / total if total > 0 else 1.0

    def __str__(self) -> str:
        return (
            f"FragmentF: {len(self.fragment_f_nodes)}/{len(self.fragment_f_nodes) + len(self.non_fragment_f_nodes)} "
            f"nodes in F ({self.fragment_f_fraction:.1%})"
        )


class FragmentClassifier:
    """Classify pipeline nodes into Fragment F (deterministic,
    order-independent) vs non-F.

    A node is in F iff:
    1. Its operator is deterministic
    2. Its operator is order-independent
    3. It has no side effects
    4. All its upstream dependencies are also in F (transitivity)
    """

    def classify(self, graph: PipelineGraph) -> FragmentClassification:
        """Classify all nodes."""
        in_f: set[str] = set()
        not_in_f: set[str] = set()
        violations: dict[str, list[str]] = {}

        # Process in topological order so we can check upstream
        if not graph.is_dag():
            # If there are cycles, nothing is in F
            for nid in graph.node_ids:
                not_in_f.add(nid)
                violations[nid] = ["Pipeline contains cycles"]
            return FragmentClassification(
                fragment_f_nodes=tuple(sorted(in_f)),
                non_fragment_f_nodes=tuple(sorted(not_in_f)),
                violations=violations,
            )

        topo = graph.topological_sort()
        for nid in topo:
            node = graph.get_node(nid)
            node_violations: list[str] = []

            # Check own properties
            if not node.properties.deterministic:
                node_violations.append("Non-deterministic operator")
            if not node.properties.order_independent:
                node_violations.append("Order-dependent operator")
            if node.properties.has_side_effects:
                node_violations.append("Has side effects")

            # Check upstream
            for pred in graph.predecessors(nid):
                if pred in not_in_f:
                    node_violations.append(f"Upstream node '{pred}' is not in Fragment F")
                    break

            if node_violations:
                not_in_f.add(nid)
                violations[nid] = node_violations
            else:
                in_f.add(nid)

        return FragmentClassification(
            fragment_f_nodes=tuple(sorted(in_f)),
            non_fragment_f_nodes=tuple(sorted(not_in_f)),
            violations=violations,
        )

    def node_in_fragment_f(self, graph: PipelineGraph, node_id: str) -> tuple[bool, list[str]]:
        """Check a single node and return (is_in_F, reasons_if_not)."""
        classification = self.classify(graph)
        if node_id in classification.fragment_f_nodes:
            return True, []
        return False, classification.violations.get(node_id, ["Unknown reason"])


# =====================================================================
# Pipeline complexity metrics
# =====================================================================

@attr.s(frozen=True, slots=True, repr=True)
class PipelineMetrics:
    """Aggregate complexity metrics for a pipeline graph."""

    node_count: int = attr.ib(default=0)
    edge_count: int = attr.ib(default=0)
    source_count: int = attr.ib(default=0)
    sink_count: int = attr.ib(default=0)
    depth: int = attr.ib(default=0)
    width: int = attr.ib(default=0)
    avg_fan_out: float = attr.ib(default=0.0)
    max_fan_out: int = attr.ib(default=0)
    avg_fan_in: float = attr.ib(default=0.0)
    max_fan_in: int = attr.ib(default=0)
    density: float = attr.ib(default=0.0)
    fragment_f_fraction: float = attr.ib(default=0.0)
    total_cost: float = attr.ib(default=0.0)
    component_count: int = attr.ib(default=0)
    is_dag: bool = attr.ib(default=True)
    operator_distribution: dict[str, int] = attr.ib(factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Pipeline Metrics:",
            f"  Nodes: {self.node_count} ({self.source_count} sources, {self.sink_count} sinks)",
            f"  Edges: {self.edge_count}",
            f"  Depth: {self.depth}, Width: {self.width}",
            f"  Fan-out: avg={self.avg_fan_out:.1f}, max={self.max_fan_out}",
            f"  Fan-in:  avg={self.avg_fan_in:.1f}, max={self.max_fan_in}",
            f"  Density: {self.density:.3f}",
            f"  Fragment F: {self.fragment_f_fraction:.1%}",
            f"  Total Cost: {self.total_cost:.2f}",
            f"  Components: {self.component_count}",
            f"  DAG: {self.is_dag}",
        ]
        if self.operator_distribution:
            lines.append("  Operators:")
            for op, count in sorted(self.operator_distribution.items(), key=lambda x: -x[1]):
                lines.append(f"    {op}: {count}")
        return "\n".join(lines)


def compute_metrics(graph: PipelineGraph) -> PipelineMetrics:
    """Compute aggregate complexity metrics."""
    n = graph.node_count
    e = graph.edge_count

    if n == 0:
        return PipelineMetrics()

    fan_outs = [graph.out_degree(nid) for nid in graph.node_ids]
    fan_ins = [graph.in_degree(nid) for nid in graph.node_ids]

    density = e / (n * (n - 1)) if n > 1 else 0.0

    classifier = FragmentClassifier()
    classification = classifier.classify(graph)

    op_dist: dict[str, int] = Counter()
    for node in graph.nodes.values():
        op_dist[node.operator.value] += 1

    return PipelineMetrics(
        node_count=n,
        edge_count=e,
        source_count=len(graph.sources()),
        sink_count=len(graph.sinks()),
        depth=graph.depth(),
        width=graph.width(),
        avg_fan_out=sum(fan_outs) / n,
        max_fan_out=max(fan_outs),
        avg_fan_in=sum(fan_ins) / n,
        max_fan_in=max(fan_ins),
        density=density,
        fragment_f_fraction=classification.fragment_f_fraction,
        total_cost=graph.total_cost().total_weighted_cost,
        component_count=len(graph.connected_components()),
        is_dag=graph.is_dag(),
        operator_distribution=dict(op_dist),
    )


# =====================================================================
# Schema impact propagation
# =====================================================================

def schema_impact_propagation(
    graph: PipelineGraph,
    node_id: str,
    column_name: str,
) -> dict[str, list[str]]:
    """Trace which downstream nodes/columns are affected when a column
    changes at *node_id*.

    Returns a mapping of downstream node_id -> list of affected column names.
    """
    if not graph.has_node(node_id):
        raise NodeNotFoundError(node_id)

    affected: dict[str, list[str]] = {}
    queue = deque([(node_id, column_name)])
    visited: set[tuple[str, str]] = set()

    while queue:
        cur_node, cur_col = queue.popleft()
        if (cur_node, cur_col) in visited:
            continue
        visited.add((cur_node, cur_col))

        for succ_id in graph.successors(cur_node):
            edge = graph.get_edge(cur_node, succ_id)

            # Determine which columns in the successor are affected
            if edge.column_mapping:
                # Explicit mapping: check if cur_col is mapped
                if cur_col in edge.column_mapping:
                    mapped_col = edge.column_mapping[cur_col]
                    affected.setdefault(succ_id, []).append(mapped_col)
                    queue.append((succ_id, mapped_col))
            else:
                # No mapping: assume same column name propagates
                succ_node = graph.get_node(succ_id)
                if cur_col in succ_node.output_schema:
                    affected.setdefault(succ_id, []).append(cur_col)
                    queue.append((succ_id, cur_col))
                elif cur_col in succ_node.input_schema:
                    # Column consumed but may produce different outputs
                    # Conservative: mark all output columns as affected
                    for out_col in succ_node.output_schema.column_list:
                        affected.setdefault(succ_id, []).append(out_col)
                        queue.append((succ_id, out_col))

    return affected


# =====================================================================
# Repair scope computation
# =====================================================================

def compute_repair_scope(
    graph: PipelineGraph,
    perturbed_nodes: Sequence[str],
) -> tuple[list[str], CostEstimate]:
    """Given a set of perturbed nodes, compute the minimal set of nodes
    that need recomputation and the total cost.

    Returns ``(ordered_nodes_to_repair, total_cost)``.
    """
    affected: set[str] = set()
    for nid in perturbed_nodes:
        affected.add(nid)
        affected.update(graph.descendants(nid))

    # Order by topological sort
    if graph.is_dag():
        topo = graph.topological_sort()
        ordered = [n for n in topo if n in affected]
    else:
        ordered = sorted(affected)

    total = CostEstimate.zero()
    for nid in ordered:
        total = total + graph.get_node(nid).cost_estimate

    return ordered, total


# =====================================================================
# Cost comparison
# =====================================================================

@attr.s(frozen=True, slots=True, repr=True)
class RepairVsRecomputeComparison:
    """Compare repair cost vs full recomputation cost."""

    repair_nodes: tuple[str, ...] = attr.ib(converter=tuple)
    repair_cost: CostEstimate = attr.ib(factory=CostEstimate.zero)
    full_recompute_cost: CostEstimate = attr.ib(factory=CostEstimate.zero)
    savings_fraction: float = attr.ib(default=0.0)

    def __str__(self) -> str:
        return (
            f"Repair: {len(self.repair_nodes)} nodes, "
            f"cost={self.repair_cost.total_weighted_cost:.2f} "
            f"(vs full={self.full_recompute_cost.total_weighted_cost:.2f}, "
            f"savings={self.savings_fraction:.1%})"
        )


def compare_repair_vs_recompute(
    graph: PipelineGraph,
    repair_nodes: Sequence[str],
) -> RepairVsRecomputeComparison:
    """Compare the cost of repairing specific nodes vs full recomputation."""
    repair_cost = CostEstimate.zero()
    for nid in repair_nodes:
        repair_cost = repair_cost + graph.get_node(nid).cost_estimate

    full_cost = graph.total_cost()
    full_val = full_cost.total_weighted_cost
    savings = 1.0 - (repair_cost.total_weighted_cost / full_val) if full_val > 0 else 0.0

    return RepairVsRecomputeComparison(
        repair_nodes=tuple(repair_nodes),
        repair_cost=repair_cost,
        full_recompute_cost=full_cost,
        savings_fraction=max(0.0, savings),
    )


# =====================================================================
# Delta annihilation analysis
# =====================================================================

@attr.s(frozen=True, slots=True, repr=True)
class AnnihilationResult:
    """Result of delta annihilation analysis.

    Identifies nodes where a propagated delta provably becomes zero
    (i.e., the transformation is invariant to the perturbation), meaning
    those nodes and all their exclusive descendants can be skipped.
    """

    annihilated_at: tuple[str, ...] = attr.ib(converter=tuple)
    skippable_nodes: tuple[str, ...] = attr.ib(converter=tuple)
    savings: CostEstimate = attr.ib(factory=CostEstimate.zero)
    reasons: dict[str, str] = attr.ib(factory=dict)

    def __str__(self) -> str:
        return (
            f"Annihilation: {len(self.annihilated_at)} annihilation points, "
            f"{len(self.skippable_nodes)} skippable nodes, "
            f"savings={self.savings.total_weighted_cost:.2f}"
        )


def analyze_delta_annihilation(
    graph: PipelineGraph,
    perturbed_node: str,
    affected_columns: set[str],
) -> AnnihilationResult:
    """Analyse where a column-level delta is annihilated.

    A delta is annihilated at a node if none of the affected columns
    contribute to that node's output.  This is a conservative static
    analysis — the actual algebra engine may find additional annihilation
    points through more precise propagation.

    Parameters
    ----------
    graph:
        The pipeline graph.
    perturbed_node:
        The node where the perturbation originates.
    affected_columns:
        Columns affected by the perturbation at the origin.
    """
    if not graph.has_node(perturbed_node):
        raise NodeNotFoundError(perturbed_node)

    annihilated_at: list[str] = []
    reasons: dict[str, str] = {}

    # Propagate column sets through the graph
    # At each node, track which perturbed columns reach it
    column_reach: dict[str, set[str]] = {perturbed_node: set(affected_columns)}

    topo = graph.topological_sort()
    start_idx = topo.index(perturbed_node) if perturbed_node in topo else 0

    for nid in topo[start_idx + 1:]:
        if nid not in graph.descendants(perturbed_node):
            continue

        # Gather reaching columns from predecessors
        reaching: set[str] = set()
        for pred in graph.predecessors(nid):
            if pred in column_reach:
                edge = graph.get_edge(pred, nid)
                if edge.column_mapping:
                    for src_col, tgt_col in edge.column_mapping.items():
                        if src_col in column_reach[pred]:
                            reaching.add(tgt_col)
                else:
                    reaching.update(column_reach.get(pred, set()))

        node = graph.get_node(nid)

        # Check if any reaching column is used by this node
        if node.output_schema.columns:
            output_cols = node.output_schema.column_names
            used_cols = reaching & output_cols
            if not used_cols and reaching:
                annihilated_at.append(nid)
                reasons[nid] = (
                    f"None of the perturbed columns ({', '.join(sorted(reaching))}) "
                    f"appear in output schema"
                )
                column_reach[nid] = set()
                continue

        # For GROUP_BY, check if affected columns are in grouping keys or aggregates
        if node.operator == SQLOperator.GROUP_BY:
            if node.input_schema.columns and reaching:
                # If perturbed columns are not in the input, annihilate
                input_cols = node.input_schema.column_names
                if not (reaching & input_cols):
                    annihilated_at.append(nid)
                    reasons[nid] = "Perturbed columns not in GROUP BY input"
                    column_reach[nid] = set()
                    continue

        column_reach[nid] = reaching

    # Find skippable nodes: annihilated nodes and their exclusive descendants
    annihilated_set = set(annihilated_at)
    skippable: set[str] = set(annihilated_set)
    for ann_node in annihilated_at:
        for desc in graph.descendants(ann_node):
            # A descendant is skippable only if ALL its ancestors that are
            # in the affected subgraph are also skippable
            desc_anc = graph.ancestors(desc) & graph.descendants(perturbed_node)
            if desc_anc <= skippable:
                skippable.add(desc)

    # Compute savings
    savings = CostEstimate.zero()
    for nid in skippable:
        savings = savings + graph.get_node(nid).cost_estimate

    return AnnihilationResult(
        annihilated_at=tuple(sorted(annihilated_at)),
        skippable_nodes=tuple(sorted(skippable)),
        savings=savings,
        reasons=reasons,
    )


# =====================================================================
# Topological level assignment
# =====================================================================

def assign_topological_levels(graph: PipelineGraph) -> dict[str, int]:
    """Assign a topological level to each node.

    Sources are at level 0.  Each other node's level is
    ``max(predecessor levels) + 1``.
    """
    if not graph.is_dag():
        return {nid: 0 for nid in graph.node_ids}

    levels: dict[str, int] = {}
    for nid in graph.topological_sort():
        preds = graph.predecessors(nid)
        if not preds:
            levels[nid] = 0
        else:
            levels[nid] = max(levels[p] for p in preds) + 1
    return levels


def nodes_at_level(graph: PipelineGraph, level: int) -> list[str]:
    """Return all node ids at a given topological level."""
    levels = assign_topological_levels(graph)
    return [nid for nid, lv in levels.items() if lv == level]


def max_topological_level(graph: PipelineGraph) -> int:
    """Return the maximum topological level (= depth)."""
    levels = assign_topological_levels(graph)
    return max(levels.values()) if levels else 0


# =====================================================================
# Parallel execution grouping
# =====================================================================

@attr.s(frozen=True, slots=True, repr=True)
class ExecutionWave:
    """A group of nodes that can be executed in parallel."""

    wave_index: int = attr.ib(default=0)
    node_ids: tuple[str, ...] = attr.ib(converter=tuple, factory=tuple)
    estimated_cost: CostEstimate = attr.ib(factory=CostEstimate.zero)
    max_node_cost: float = attr.ib(default=0.0)

    def __str__(self) -> str:
        return (
            f"Wave {self.wave_index}: {len(self.node_ids)} nodes, "
            f"max_cost={self.max_node_cost:.2f}"
        )


def compute_execution_waves(graph: PipelineGraph) -> list[ExecutionWave]:
    """Partition nodes into waves for parallel execution.

    Nodes in the same wave have no dependencies on each other and
    all their dependencies are in earlier waves.
    """
    if not graph.is_dag():
        return [ExecutionWave(
            wave_index=0,
            node_ids=tuple(graph.node_ids),
        )]

    levels = assign_topological_levels(graph)
    max_level = max(levels.values()) if levels else 0

    waves: list[ExecutionWave] = []
    for lv in range(max_level + 1):
        nids = [nid for nid, l in levels.items() if l == lv]
        if not nids:
            continue

        total_cost = CostEstimate.zero()
        max_cost = 0.0
        for nid in nids:
            node_cost = graph.get_node(nid).cost_estimate
            total_cost = total_cost + node_cost
            max_cost = max(max_cost, node_cost.total_weighted_cost)

        waves.append(ExecutionWave(
            wave_index=lv,
            node_ids=tuple(nids),
            estimated_cost=total_cost,
            max_node_cost=max_cost,
        ))

    return waves


def estimate_parallel_time(graph: PipelineGraph) -> float:
    """Estimate the wall-clock time assuming perfect parallelism.

    This is the sum of max-node-costs across all waves.
    """
    waves = compute_execution_waves(graph)
    return sum(w.max_node_cost for w in waves)


def estimate_sequential_time(graph: PipelineGraph) -> float:
    """Estimate the wall-clock time for sequential execution."""
    return sum(
        graph.get_node(nid).cost_estimate.total_weighted_cost
        for nid in graph.node_ids
    )


def parallelism_speedup(graph: PipelineGraph) -> float:
    """Estimate the speedup from parallel execution."""
    seq = estimate_sequential_time(graph)
    par = estimate_parallel_time(graph)
    return seq / par if par > 0 else 1.0


# =====================================================================
# Schema lineage tracing
# =====================================================================

@attr.s(frozen=True, slots=True, repr=True)
class SchemaLineageEntry:
    """Traces a column back to its origin across the pipeline."""

    column_name: str = attr.ib(validator=v.instance_of(str))
    node_path: tuple[str, ...] = attr.ib(converter=tuple)
    transform_chain: tuple[str, ...] = attr.ib(converter=tuple)

    def __str__(self) -> str:
        return f"{self.column_name}: {' -> '.join(self.node_path)}"


def trace_column_lineage(
    graph: PipelineGraph,
    node_id: str,
    column_name: str,
) -> list[SchemaLineageEntry]:
    """Trace a column backward to its source(s).

    Returns all possible origin paths for the specified column.
    """
    if not graph.has_node(node_id):
        raise NodeNotFoundError(node_id)

    results: list[SchemaLineageEntry] = []

    def _trace(
        cur_node: str,
        cur_col: str,
        path: list[str],
        transforms: list[str],
    ) -> None:
        path = path + [cur_node]
        preds = graph.predecessors(cur_node)

        if not preds:
            # Reached a source
            results.append(SchemaLineageEntry(
                column_name=cur_col,
                node_path=tuple(reversed(path)),
                transform_chain=tuple(reversed(transforms)),
            ))
            return

        for pred_id in preds:
            edge = graph.get_edge(pred_id, cur_node)
            pred_node = graph.get_node(pred_id)

            if edge.column_mapping:
                # Check reverse mapping
                for src_col, tgt_col in edge.column_mapping.items():
                    if tgt_col == cur_col:
                        xform = f"{pred_id}.{src_col} -> {cur_node}.{cur_col}"
                        _trace(pred_id, src_col, path, transforms + [xform])
            else:
                # Assume same column name propagates
                if pred_node.output_schema.columns:
                    if cur_col in pred_node.output_schema:
                        xform = f"{pred_id}.{cur_col} -> {cur_node}.{cur_col}"
                        _trace(pred_id, cur_col, path, transforms + [xform])
                else:
                    xform = f"{pred_id}.? -> {cur_node}.{cur_col}"
                    _trace(pred_id, cur_col, path, transforms + [xform])

    _trace(node_id, column_name, [], [])
    return results


# =====================================================================
# Graph diff
# =====================================================================

@attr.s(frozen=True, slots=True, repr=True)
class GraphDiff:
    """Difference between two pipeline graphs."""

    added_nodes: tuple[str, ...] = attr.ib(converter=tuple)
    removed_nodes: tuple[str, ...] = attr.ib(converter=tuple)
    modified_nodes: tuple[str, ...] = attr.ib(converter=tuple)
    added_edges: tuple[tuple[str, str], ...] = attr.ib(converter=tuple)
    removed_edges: tuple[tuple[str, str], ...] = attr.ib(converter=tuple)
    schema_changes: dict[str, list[str]] = attr.ib(factory=dict)

    @property
    def is_empty(self) -> bool:
        return (
            not self.added_nodes
            and not self.removed_nodes
            and not self.modified_nodes
            and not self.added_edges
            and not self.removed_edges
        )

    def __str__(self) -> str:
        lines = ["Graph Diff:"]
        if self.added_nodes:
            lines.append(f"  +nodes: {list(self.added_nodes)}")
        if self.removed_nodes:
            lines.append(f"  -nodes: {list(self.removed_nodes)}")
        if self.modified_nodes:
            lines.append(f"  ~nodes: {list(self.modified_nodes)}")
        if self.added_edges:
            lines.append(f"  +edges: {list(self.added_edges)}")
        if self.removed_edges:
            lines.append(f"  -edges: {list(self.removed_edges)}")
        if self.schema_changes:
            lines.append("  Schema changes:")
            for nid, changes in self.schema_changes.items():
                for change in changes:
                    lines.append(f"    {nid}: {change}")
        return "\n".join(lines)


def diff_graphs(before: PipelineGraph, after: PipelineGraph) -> GraphDiff:
    """Compute the structural difference between two pipeline graphs."""
    before_nodes = set(before.node_ids)
    after_nodes = set(after.node_ids)

    added = sorted(after_nodes - before_nodes)
    removed = sorted(before_nodes - after_nodes)

    # Modified nodes: same id but different properties
    modified: list[str] = []
    schema_changes: dict[str, list[str]] = {}
    for nid in before_nodes & after_nodes:
        b_node = before.get_node(nid)
        a_node = after.get_node(nid)

        changes: list[str] = []
        if b_node.operator != a_node.operator:
            changes.append(f"operator: {b_node.operator.value} -> {a_node.operator.value}")
        if b_node.query_text != a_node.query_text:
            changes.append("query_text changed")
        if b_node.output_schema != a_node.output_schema:
            # Compute column-level diff
            b_cols = b_node.output_schema.column_names
            a_cols = a_node.output_schema.column_names
            new_cols = a_cols - b_cols
            dropped_cols = b_cols - a_cols
            if new_cols:
                changes.append(f"+columns: {sorted(new_cols)}")
            if dropped_cols:
                changes.append(f"-columns: {sorted(dropped_cols)}")
            # Check type changes
            for col_name in b_cols & a_cols:
                b_col = b_node.output_schema[col_name]
                a_col = a_node.output_schema[col_name]
                if b_col.sql_type != a_col.sql_type:
                    changes.append(
                        f"type({col_name}): {b_col.sql_type} -> {a_col.sql_type}"
                    )

        if changes:
            modified.append(nid)
            schema_changes[nid] = changes

    # Edge diffs
    before_edges = set(before.edges.keys())
    after_edges = set(after.edges.keys())
    added_edges = sorted(after_edges - before_edges)
    removed_edges = sorted(before_edges - after_edges)

    return GraphDiff(
        added_nodes=tuple(added),
        removed_nodes=tuple(removed),
        modified_nodes=tuple(modified),
        added_edges=tuple(added_edges),
        removed_edges=tuple(removed_edges),
        schema_changes=schema_changes,
    )
