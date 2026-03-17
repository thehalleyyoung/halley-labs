"""
taintflow.attribution.mincut -- Information-theoretic min-cut computation.

Implements max-flow / min-cut on the PI-DAG where edge capacities represent
leakage bounds in bits.  The min-cut identifies the *bottleneck* set of
edges whose collective capacity determines the total leakage flowing from
test-origin sources to training sinks.

Key classes:

* :class:`MinCutSolver` -- Ford–Fulkerson on the leakage DAG.
* :class:`MinCutResult` -- edges forming the cut with per-feature attribution.
* :class:`BottleneckStage` -- single-stage contribution to total leakage.
* :class:`MinCutDecomposition` -- per-stage decomposition of total leakage.
* :class:`BottleneckRanking` -- stages ranked by leakage contribution.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)
from typing import TYPE_CHECKING

from taintflow.core.types import (
    FeatureLeakage,
    OpType,
    Severity,
    StageLeakage,
)
from taintflow.core.config import TaintFlowConfig, SeverityThresholds

if TYPE_CHECKING:
    from taintflow.dag.pidag import PIDAG, PipelineStage
    from taintflow.dag.edge import PipelineEdge
    from taintflow.dag.node import PipelineNode

# ===================================================================
#  Constants
# ===================================================================

_INF: float = float("inf")
_EPSILON: float = 1e-12

# ===================================================================
#  Result dataclasses
# ===================================================================


@dataclass(frozen=True)
class BottleneckStage:
    """A single pipeline stage's contribution to total leakage."""

    stage_id: str
    leakage_bits: float
    fraction_of_total: float

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.stage_id:
            errors.append("stage_id must be non-empty")
        if self.leakage_bits < 0.0:
            errors.append(f"leakage_bits must be >= 0, got {self.leakage_bits}")
        if not (0.0 <= self.fraction_of_total <= 1.0 + _EPSILON):
            errors.append(
                f"fraction_of_total must be in [0, 1], got {self.fraction_of_total}"
            )
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "leakage_bits": self.leakage_bits,
            "fraction_of_total": self.fraction_of_total,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BottleneckStage:
        return cls(
            stage_id=str(data["stage_id"]),
            leakage_bits=float(data["leakage_bits"]),
            fraction_of_total=float(data["fraction_of_total"]),
        )

    def __repr__(self) -> str:
        return (
            f"BottleneckStage({self.stage_id!r}, "
            f"{self.leakage_bits:.2f} bits, "
            f"{self.fraction_of_total:.1%})"
        )


@dataclass
class MinCutResult:
    """Result of a min-cut computation on the PI-DAG.

    Attributes
    ----------
    cut_edges : list[tuple[str, str]]
        Edges forming the min-cut, as ``(source_id, target_id)`` pairs.
    total_capacity : float
        Sum of capacities across the cut edges (= max-flow value).
    per_feature_attribution : dict[str, float]
        Maps each feature (column) to its attributed leakage bits
        flowing through the cut.
    source_side : set[str]
        Node IDs on the source side of the cut.
    sink_side : set[str]
        Node IDs on the sink side of the cut.
    """

    cut_edges: list[tuple[str, str]] = field(default_factory=list)
    total_capacity: float = 0.0
    per_feature_attribution: dict[str, float] = field(default_factory=dict)
    source_side: set[str] = field(default_factory=set)
    sink_side: set[str] = field(default_factory=set)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.total_capacity < 0.0:
            errors.append(f"total_capacity must be >= 0, got {self.total_capacity}")
        for feat, bits in self.per_feature_attribution.items():
            if bits < 0.0:
                errors.append(f"attribution for {feat!r} must be >= 0, got {bits}")
        return errors

    @property
    def n_cut_edges(self) -> int:
        return len(self.cut_edges)

    @property
    def is_trivial(self) -> bool:
        """True when the cut carries zero capacity (no leakage)."""
        return self.total_capacity < _EPSILON

    def to_dict(self) -> dict[str, Any]:
        return {
            "cut_edges": [list(e) for e in self.cut_edges],
            "total_capacity": self.total_capacity,
            "per_feature_attribution": dict(self.per_feature_attribution),
            "source_side": sorted(self.source_side),
            "sink_side": sorted(self.sink_side),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MinCutResult:
        return cls(
            cut_edges=[tuple(e) for e in data.get("cut_edges", [])],
            total_capacity=float(data.get("total_capacity", 0.0)),
            per_feature_attribution=dict(data.get("per_feature_attribution", {})),
            source_side=set(data.get("source_side", [])),
            sink_side=set(data.get("sink_side", [])),
        )

    def __repr__(self) -> str:
        return (
            f"MinCutResult(edges={self.n_cut_edges}, "
            f"capacity={self.total_capacity:.2f} bits)"
        )


@dataclass
class MinCutDecomposition:
    """Per-stage decomposition of total min-cut leakage.

    Distributes the min-cut's total capacity among the pipeline stages
    that participate in the cut.
    """

    total_leakage_bits: float = 0.0
    stage_contributions: list[BottleneckStage] = field(default_factory=list)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.total_leakage_bits < 0.0:
            errors.append(f"total_leakage_bits must be >= 0, got {self.total_leakage_bits}")
        for bs in self.stage_contributions:
            errors.extend(bs.validate())
        contrib_sum = sum(bs.leakage_bits for bs in self.stage_contributions)
        if self.total_leakage_bits > _EPSILON and abs(contrib_sum - self.total_leakage_bits) > 0.01:
            errors.append(
                f"Stage contributions sum to {contrib_sum:.4f}, "
                f"expected {self.total_leakage_bits:.4f}"
            )
        return errors

    @property
    def n_stages(self) -> int:
        return len(self.stage_contributions)

    def top_k(self, k: int) -> list[BottleneckStage]:
        """Return the top-*k* stages by leakage bits."""
        return sorted(
            self.stage_contributions,
            key=lambda b: b.leakage_bits,
            reverse=True,
        )[:k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_leakage_bits": self.total_leakage_bits,
            "stage_contributions": [s.to_dict() for s in self.stage_contributions],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MinCutDecomposition:
        return cls(
            total_leakage_bits=float(data.get("total_leakage_bits", 0.0)),
            stage_contributions=[
                BottleneckStage.from_dict(s) for s in data.get("stage_contributions", [])
            ],
        )

    def __repr__(self) -> str:
        return (
            f"MinCutDecomposition({self.total_leakage_bits:.2f} bits, "
            f"{self.n_stages} stages)"
        )


@dataclass
class SensitivityEntry:
    """Result of sensitivity analysis for a single stage."""

    stage_id: str
    original_leakage: float
    leakage_without_stage: float

    @property
    def delta(self) -> float:
        """Reduction in leakage if this stage is removed."""
        return self.original_leakage - self.leakage_without_stage

    @property
    def relative_delta(self) -> float:
        if self.original_leakage < _EPSILON:
            return 0.0
        return self.delta / self.original_leakage

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "original_leakage": self.original_leakage,
            "leakage_without_stage": self.leakage_without_stage,
            "delta": self.delta,
            "relative_delta": self.relative_delta,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SensitivityEntry:
        return cls(
            stage_id=str(data["stage_id"]),
            original_leakage=float(data["original_leakage"]),
            leakage_without_stage=float(data["leakage_without_stage"]),
        )

    def __repr__(self) -> str:
        return (
            f"SensitivityEntry({self.stage_id!r}, "
            f"Δ={self.delta:+.2f} bits)"
        )


@dataclass
class BottleneckRanking:
    """Stages ranked by their contribution to total leakage.

    Combines min-cut decomposition with sensitivity analysis to produce
    a single prioritised ranking.
    """

    ranked_stages: list[BottleneckStage] = field(default_factory=list)
    sensitivities: list[SensitivityEntry] = field(default_factory=list)

    @property
    def n_stages(self) -> int:
        return len(self.ranked_stages)

    def top_k(self, k: int) -> list[BottleneckStage]:
        return self.ranked_stages[:k]

    def sensitivity_for(self, stage_id: str) -> SensitivityEntry | None:
        for se in self.sensitivities:
            if se.stage_id == stage_id:
                return se
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranked_stages": [s.to_dict() for s in self.ranked_stages],
            "sensitivities": [s.to_dict() for s in self.sensitivities],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BottleneckRanking:
        return cls(
            ranked_stages=[
                BottleneckStage.from_dict(s) for s in data.get("ranked_stages", [])
            ],
            sensitivities=[
                SensitivityEntry.from_dict(s) for s in data.get("sensitivities", [])
            ],
        )

    def __repr__(self) -> str:
        return f"BottleneckRanking({self.n_stages} stages)"


# ===================================================================
#  Internal flow-network helpers
# ===================================================================


class _FlowNetwork:
    """Residual-graph representation for Ford-Fulkerson on a DAG.

    Stores forward capacities and current flow in adjacency dicts.
    Reverse edges are tracked implicitly in the residual.
    """

    __slots__ = ("_capacity", "_flow", "_adj", "_nodes")

    def __init__(self) -> None:
        self._capacity: dict[tuple[str, str], float] = {}
        self._flow: dict[tuple[str, str], float] = {}
        self._adj: dict[str, set[str]] = defaultdict(set)
        self._nodes: set[str] = set()

    def add_edge(self, u: str, v: str, cap: float) -> None:
        key = (u, v)
        self._capacity[key] = self._capacity.get(key, 0.0) + max(cap, 0.0)
        self._flow.setdefault(key, 0.0)
        self._flow.setdefault((v, u), 0.0)
        self._capacity.setdefault((v, u), 0.0)
        self._adj[u].add(v)
        self._adj[v].add(u)
        self._nodes.update((u, v))

    def residual(self, u: str, v: str) -> float:
        return self._capacity.get((u, v), 0.0) - self._flow.get((u, v), 0.0)

    def push_flow(self, u: str, v: str, amount: float) -> None:
        self._flow[(u, v)] = self._flow.get((u, v), 0.0) + amount
        self._flow[(v, u)] = self._flow.get((v, u), 0.0) - amount

    def neighbours(self, u: str) -> set[str]:
        return self._adj.get(u, set())

    @property
    def nodes(self) -> set[str]:
        return set(self._nodes)

    def bfs_augmenting_path(
        self,
        source: str,
        sink: str,
    ) -> list[str] | None:
        """BFS on residual graph (Edmonds–Karp variant)."""
        visited: set[str] = {source}
        parent: dict[str, str] = {}
        queue: deque[str] = deque([source])

        while queue:
            u = queue.popleft()
            if u == sink:
                path = [sink]
                cur = sink
                while cur != source:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path
            for v in self._adj.get(u, set()):
                if v not in visited and self.residual(u, v) > _EPSILON:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)
        return None

    def reachable_from(self, source: str) -> set[str]:
        """Nodes reachable from *source* in the residual graph."""
        visited: set[str] = {source}
        stack: list[str] = [source]
        while stack:
            u = stack.pop()
            for v in self._adj.get(u, set()):
                if v not in visited and self.residual(u, v) > _EPSILON:
                    visited.add(v)
                    stack.append(v)
        return visited


# ===================================================================
#  MinCutSolver
# ===================================================================


class MinCutSolver:
    """Information-theoretic min-cut computation on the PI-DAG.

    Constructs a flow network from the DAG's leakage edges and solves
    max-flow / min-cut using Edmonds–Karp (BFS-based Ford–Fulkerson).
    Since the PI-DAG is acyclic, the algorithm terminates in polynomial
    time without needing cycle-breaking heuristics.

    Parameters
    ----------
    dag : PIDAG
        The pipeline information DAG with capacity-annotated edges.
    config : TaintFlowConfig | None
        Optional configuration; uses defaults if *None*.
    """

    def __init__(
        self,
        dag: PIDAG,
        config: TaintFlowConfig | None = None,
    ) -> None:
        self._dag = dag
        self._config = config
        self._thresholds: SeverityThresholds = (
            config.severity if config is not None else SeverityThresholds()
        )

    # -- public API ----------------------------------------------------------

    def compute_mincut(
        self,
        source_ids: Sequence[str] | None = None,
        sink_ids: Sequence[str] | None = None,
    ) -> MinCutResult:
        """Compute the global min-cut between source and sink node sets.

        If *source_ids* / *sink_ids* are not specified the solver
        auto-detects test-origin sources and training sinks from the DAG.
        """
        sources = list(source_ids) if source_ids else self._auto_sources()
        sinks = list(sink_ids) if sink_ids else self._auto_sinks()

        if not sources or not sinks:
            return MinCutResult()

        net, super_s, super_t = self._build_network(sources, sinks)
        max_flow = self._edmonds_karp(net, super_s, super_t)

        source_side = net.reachable_from(super_s)
        all_nodes = net.nodes
        sink_side = all_nodes - source_side

        cut_edges = self._extract_cut_edges(net, source_side, sink_side)
        feat_attr = self._attribute_features(cut_edges)

        real_sources = source_side - {super_s}
        real_sinks = sink_side - {super_t}

        return MinCutResult(
            cut_edges=cut_edges,
            total_capacity=max_flow,
            per_feature_attribution=feat_attr,
            source_side=real_sources,
            sink_side=real_sinks,
        )

    def per_feature_mincut(
        self,
        features: Sequence[str] | None = None,
    ) -> dict[str, MinCutResult]:
        """Compute an independent min-cut for each output feature.

        For each feature, restrict the DAG to edges carrying that feature
        and solve min-cut on the restricted graph.

        Parameters
        ----------
        features : sequence of str, optional
            Features to analyse.  Defaults to all features observed in the
            DAG's sink nodes.
        """
        if features is None:
            features = self._all_sink_features()

        results: dict[str, MinCutResult] = {}
        for feat in features:
            results[feat] = self._single_feature_mincut(feat)
        return results

    def multi_commodity_flow(
        self,
        features: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Multi-commodity flow formulation for concurrent feature leakage.

        Each feature is a commodity.  Returns the total flow (leakage) per
        feature under shared edge capacities.  This gives a tighter bound
        than summing independent per-feature cuts when features share edges.
        """
        if features is None:
            features = self._all_sink_features()

        sources = self._auto_sources()
        sinks = self._auto_sinks()
        if not sources or not sinks:
            return {f: 0.0 for f in features}

        edge_caps: dict[tuple[str, str], float] = {}
        edge_feats: dict[tuple[str, str], set[str]] = defaultdict(set)
        for edge in self._dag.iter_data_flow_edges():
            key = (edge.source_id, edge.target_id)
            edge_caps[key] = edge.capacity
            for col in edge.columns:
                if col in features:
                    edge_feats[key].add(col)

        feature_flow: dict[str, float] = {f: 0.0 for f in features}
        remaining_cap: dict[tuple[str, str], float] = dict(edge_caps)

        feat_priority = sorted(features, key=lambda f: sum(
            remaining_cap.get(k, 0.0) for k, fs in edge_feats.items() if f in fs
        ))

        for feat in feat_priority:
            net = _FlowNetwork()
            super_s = "__mc_source__"
            super_t = "__mc_sink__"

            for src_id in sources:
                net.add_edge(super_s, src_id, _INF)
            for snk_id in sinks:
                net.add_edge(snk_id, super_t, _INF)

            for (u, v), feats_on_edge in edge_feats.items():
                if feat in feats_on_edge:
                    cap = remaining_cap.get((u, v), 0.0)
                    n_feats_here = len(feats_on_edge)
                    fair_share = cap / max(n_feats_here, 1)
                    net.add_edge(u, v, fair_share)

            flow = self._edmonds_karp(net, super_s, super_t)
            feature_flow[feat] = flow

            for (u, v), feats_on_edge in edge_feats.items():
                if feat in feats_on_edge and (u, v) in remaining_cap:
                    n_feats_here = len(feats_on_edge)
                    used = flow / max(n_feats_here, 1)
                    remaining_cap[(u, v)] = max(remaining_cap[(u, v)] - used, 0.0)

        return feature_flow

    def decompose(
        self,
        mincut_result: MinCutResult | None = None,
    ) -> MinCutDecomposition:
        """Decompose total min-cut leakage into per-stage contributions.

        Each cut edge is attributed to the stage containing its source node.
        If a stage has multiple cut edges the contributions are summed.
        """
        if mincut_result is None:
            mincut_result = self.compute_mincut()

        if mincut_result.is_trivial:
            return MinCutDecomposition()

        stage_map = self._build_node_to_stage_map()
        stage_bits: dict[str, float] = defaultdict(float)

        for src, tgt in mincut_result.cut_edges:
            edge = self._find_edge(src, tgt)
            cap = edge.capacity if edge is not None else 0.0
            stage_id = stage_map.get(src, src)
            stage_bits[stage_id] += cap

        total = mincut_result.total_capacity
        contributions: list[BottleneckStage] = []
        for sid, bits in sorted(stage_bits.items(), key=lambda kv: -kv[1]):
            frac = bits / total if total > _EPSILON else 0.0
            contributions.append(BottleneckStage(
                stage_id=sid,
                leakage_bits=bits,
                fraction_of_total=frac,
            ))

        return MinCutDecomposition(
            total_leakage_bits=total,
            stage_contributions=contributions,
        )

    def sensitivity_analysis(
        self,
        stage_ids: Sequence[str] | None = None,
    ) -> list[SensitivityEntry]:
        """Measure how much leakage changes if each stage is removed.

        For every stage, zero-out the capacities of edges involving that
        stage and re-run the min-cut.  The difference is the stage's
        sensitivity.

        Parameters
        ----------
        stage_ids : sequence of str, optional
            Stages to test.  Defaults to all stages in the DAG.
        """
        baseline = self.compute_mincut()
        original_leakage = baseline.total_capacity

        if stage_ids is None:
            stage_ids = self._all_stage_ids()

        stage_map = self._build_node_to_stage_map()
        node_to_stage: dict[str, str] = {}
        for nid, sid in stage_map.items():
            node_to_stage[nid] = sid

        entries: list[SensitivityEntry] = []
        for sid in stage_ids:
            stage_nodes = {nid for nid, s in node_to_stage.items() if s == sid}
            leakage_without = self._leakage_without_nodes(stage_nodes)
            entries.append(SensitivityEntry(
                stage_id=sid,
                original_leakage=original_leakage,
                leakage_without_stage=leakage_without,
            ))

        entries.sort(key=lambda e: e.delta, reverse=True)
        return entries

    def rank_bottlenecks(self) -> BottleneckRanking:
        """Produce a combined ranking of bottleneck stages.

        Merges decomposition (how much each stage contributes to the cut)
        with sensitivity analysis (how much removing the stage would help).
        The final ranking is sorted by decomposition contribution.
        """
        decomp = self.decompose()
        sensitivities = self.sensitivity_analysis(
            stage_ids=[s.stage_id for s in decomp.stage_contributions],
        )
        return BottleneckRanking(
            ranked_stages=decomp.stage_contributions,
            sensitivities=sensitivities,
        )

    # -- private: network construction ---------------------------------------

    def _build_network(
        self,
        sources: list[str],
        sinks: list[str],
    ) -> tuple[_FlowNetwork, str, str]:
        """Build the flow network with super-source and super-sink."""
        net = _FlowNetwork()
        super_s = "__super_source__"
        super_t = "__super_sink__"

        for src_id in sources:
            net.add_edge(super_s, src_id, _INF)
        for snk_id in sinks:
            net.add_edge(snk_id, super_t, _INF)

        for edge in self._dag.iter_data_flow_edges():
            cap = edge.capacity
            if cap > _EPSILON:
                net.add_edge(edge.source_id, edge.target_id, cap)

        return net, super_s, super_t

    def _edmonds_karp(
        self,
        net: _FlowNetwork,
        source: str,
        sink: str,
    ) -> float:
        """Edmonds-Karp max-flow (BFS augmenting paths)."""
        total_flow = 0.0
        while True:
            path = net.bfs_augmenting_path(source, sink)
            if path is None:
                break
            bottleneck = _INF
            for i in range(len(path) - 1):
                res = net.residual(path[i], path[i + 1])
                bottleneck = min(bottleneck, res)
            if bottleneck < _EPSILON:
                break
            for i in range(len(path) - 1):
                net.push_flow(path[i], path[i + 1], bottleneck)
            total_flow += bottleneck
        return total_flow

    def _extract_cut_edges(
        self,
        net: _FlowNetwork,
        source_side: set[str],
        sink_side: set[str],
    ) -> list[tuple[str, str]]:
        """Edges crossing from source-side to sink-side in the residual."""
        cut: list[tuple[str, str]] = []
        for u in source_side:
            for v in net.neighbours(u):
                if v in sink_side and net.residual(u, v) < _EPSILON:
                    if u.startswith("__") or v.startswith("__"):
                        continue
                    cut.append((u, v))
        return cut

    def _attribute_features(
        self,
        cut_edges: list[tuple[str, str]],
    ) -> dict[str, float]:
        """Distribute cut capacity across features using column sets."""
        attr: dict[str, float] = defaultdict(float)
        for src, tgt in cut_edges:
            edge = self._find_edge(src, tgt)
            if edge is None:
                continue
            cols = edge.columns
            if not cols:
                continue
            per_col = edge.capacity / len(cols)
            for col in cols:
                attr[col] += per_col
        return dict(attr)

    # -- private: per-feature cut -------------------------------------------

    def _single_feature_mincut(self, feature: str) -> MinCutResult:
        """Min-cut on the sub-DAG restricted to a single feature."""
        sources = self._auto_sources()
        sinks = self._auto_sinks()
        if not sources or not sinks:
            return MinCutResult()

        net = _FlowNetwork()
        super_s = "__feat_source__"
        super_t = "__feat_sink__"

        for src_id in sources:
            net.add_edge(super_s, src_id, _INF)
        for snk_id in sinks:
            net.add_edge(snk_id, super_t, _INF)

        for edge in self._dag.iter_data_flow_edges():
            if feature in edge.columns:
                col_fraction = 1.0 / max(len(edge.columns), 1)
                cap = edge.capacity * col_fraction
                if cap > _EPSILON:
                    net.add_edge(edge.source_id, edge.target_id, cap)

        max_flow = self._edmonds_karp(net, super_s, super_t)
        source_side = net.reachable_from(super_s)
        all_nodes = net.nodes
        sink_side = all_nodes - source_side
        cut_edges = self._extract_cut_edges(net, source_side, sink_side)

        return MinCutResult(
            cut_edges=cut_edges,
            total_capacity=max_flow,
            per_feature_attribution={feature: max_flow},
            source_side=source_side - {super_s},
            sink_side=sink_side - {super_t},
        )

    # -- private: auto-detect sources / sinks --------------------------------

    def _auto_sources(self) -> list[str]:
        """Detect test-origin source nodes."""
        from taintflow.dag.node import DataSourceNode
        from taintflow.core.types import Origin

        sources: list[str] = []
        for nid, node in self._dag.nodes.items():
            if isinstance(node, DataSourceNode) and node.origin == Origin.TEST:
                sources.append(nid)
            elif node.max_test_fraction() > 0.0 and self._dag.in_degree(nid) == 0:
                sources.append(nid)
        if not sources:
            for nid, node in self._dag.nodes.items():
                if node.max_test_fraction() > 0.0:
                    sources.append(nid)
        return sources

    def _auto_sinks(self) -> list[str]:
        """Detect training-sink nodes (nodes with fit steps)."""
        from taintflow.dag.node import TransformNode

        sinks: list[str] = []
        for nid, node in self._dag.nodes.items():
            if node.has_fit:
                sinks.append(nid)
            elif isinstance(node, TransformNode) and node.is_fitted:
                sinks.append(nid)
        if not sinks:
            sinks = [n.node_id for n in self._dag.sinks]
        return sinks

    def _all_sink_features(self) -> list[str]:
        """All feature column names observed at sink nodes."""
        features: set[str] = set()
        for node in self._dag.sinks:
            for col in node.output_schema:
                if not col.is_target and not col.is_index:
                    features.add(col.name)
            for col in node.input_schema:
                if not col.is_target and not col.is_index:
                    features.add(col.name)
        if not features:
            for edge in self._dag.iter_data_flow_edges():
                features.update(edge.columns)
        return sorted(features)

    # -- private: helpers ----------------------------------------------------

    def _find_edge(self, source_id: str, target_id: str) -> PipelineEdge | None:
        """Look up a DAG edge by endpoint IDs."""
        for edge in self._dag.in_edges(target_id):
            if edge.source_id == source_id and edge.is_data_flow:
                return edge
        return None

    def _build_node_to_stage_map(self) -> dict[str, str]:
        """Map each node ID to its containing stage ID."""
        stage_map: dict[str, str] = {}
        stages = self._dag.get_pipeline_stages()
        for stage in stages:
            for nid in stage.node_ids:
                stage_map[nid] = stage.stage_id
        for nid in self._dag.nodes:
            if nid not in stage_map:
                stage_map[nid] = nid
        return stage_map

    def _all_stage_ids(self) -> list[str]:
        stages = self._dag.get_pipeline_stages()
        return [s.stage_id for s in stages]

    def _leakage_without_nodes(self, removed_nodes: set[str]) -> float:
        """Compute max-flow after zeroing edges incident to *removed_nodes*."""
        sources = self._auto_sources()
        sinks = self._auto_sinks()
        remaining_sources = [s for s in sources if s not in removed_nodes]
        remaining_sinks = [s for s in sinks if s not in removed_nodes]
        if not remaining_sources or not remaining_sinks:
            return 0.0

        net = _FlowNetwork()
        super_s = "__sens_source__"
        super_t = "__sens_sink__"

        for src_id in remaining_sources:
            net.add_edge(super_s, src_id, _INF)
        for snk_id in remaining_sinks:
            net.add_edge(snk_id, super_t, _INF)

        for edge in self._dag.iter_data_flow_edges():
            if edge.source_id in removed_nodes or edge.target_id in removed_nodes:
                continue
            if edge.capacity > _EPSILON:
                net.add_edge(edge.source_id, edge.target_id, edge.capacity)

        return self._edmonds_karp(net, super_s, super_t)
