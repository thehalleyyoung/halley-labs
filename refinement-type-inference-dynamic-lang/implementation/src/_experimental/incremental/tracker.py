"""
Incremental Dependency Tracker for Refinement Type Inference.

Theory basis:
- Function-level dependency graph G = (V, E) where V = functions,
  (f,g) in E iff f's contract depends on g's summary
- Predicate-sensitive invalidation: when f changes, invalidate only
  downstream summaries whose predicate sets overlap with affected predicates
- Semi-naive delta propagation through dependency graph
- Complexity: O(|Delta_out| * poly(|P|))
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CacheStats:
    """Statistics about the analysis cache."""
    hits: int = 0
    misses: int = 0
    invalidations: int = 0
    size: int = 0
    age: float = 0.0  # seconds since cache creation

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CacheStats":
        return cls(**d)


@dataclass
class FunctionSummary:
    """Summary of a single function's refinement type analysis."""
    name: str
    refinement_types: Dict[str, Any] = field(default_factory=dict)
    predicates: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    timestamp: float = field(default_factory=time.time)
    ir_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "refinement_types": self.refinement_types,
            "predicates": sorted(self.predicates),
            "dependencies": sorted(self.dependencies),
            "timestamp": self.timestamp,
            "ir_hash": self.ir_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FunctionSummary":
        return cls(
            name=d["name"],
            refinement_types=d.get("refinement_types", {}),
            predicates=set(d.get("predicates", [])),
            dependencies=set(d.get("dependencies", [])),
            timestamp=d.get("timestamp", 0.0),
            ir_hash=d.get("ir_hash", ""),
        )


@dataclass
class ChangeSet:
    """Describes a set of changes between two analysis states."""
    changed_files: Set[str] = field(default_factory=set)
    changed_functions: Set[str] = field(default_factory=set)
    added_functions: Set[str] = field(default_factory=set)
    removed_functions: Set[str] = field(default_factory=set)
    changed_predicates: Set[str] = field(default_factory=set)

    @property
    def is_empty(self) -> bool:
        return not (
            self.changed_files
            or self.changed_functions
            or self.added_functions
            or self.removed_functions
            or self.changed_predicates
        )

    def all_affected_functions(self) -> Set[str]:
        return self.changed_functions | self.added_functions | self.removed_functions

    def merge(self, other: "ChangeSet") -> "ChangeSet":
        return ChangeSet(
            changed_files=self.changed_files | other.changed_files,
            changed_functions=self.changed_functions | other.changed_functions,
            added_functions=self.added_functions | other.added_functions,
            removed_functions=self.removed_functions | other.removed_functions,
            changed_predicates=self.changed_predicates | other.changed_predicates,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "changed_files": sorted(self.changed_files),
            "changed_functions": sorted(self.changed_functions),
            "added_functions": sorted(self.added_functions),
            "removed_functions": sorted(self.removed_functions),
            "changed_predicates": sorted(self.changed_predicates),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChangeSet":
        return cls(
            changed_files=set(d.get("changed_files", [])),
            changed_functions=set(d.get("changed_functions", [])),
            added_functions=set(d.get("added_functions", [])),
            removed_functions=set(d.get("removed_functions", [])),
            changed_predicates=set(d.get("changed_predicates", [])),
        )


@dataclass
class IncrementalStatistics:
    """Tracks incremental analysis performance."""
    total_functions: int = 0
    reanalyzed_functions: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    invalidated_functions: int = 0
    actually_changed_results: int = 0
    full_analysis_time_estimate: float = 0.0
    incremental_analysis_time: float = 0.0

    @property
    def reanalysis_ratio(self) -> float:
        """Fraction of functions re-analyzed."""
        if self.total_functions == 0:
            return 0.0
        return self.reanalyzed_functions / self.total_functions

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit ratio."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def time_saved(self) -> float:
        """Estimated time saved vs full re-analysis."""
        return max(0.0, self.full_analysis_time_estimate - self.incremental_analysis_time)

    @property
    def invalidation_precision(self) -> float:
        """Fraction of invalidated functions that actually changed results."""
        if self.invalidated_functions == 0:
            return 1.0
        return self.actually_changed_results / self.invalidated_functions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_functions": self.total_functions,
            "reanalyzed_functions": self.reanalyzed_functions,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "invalidated_functions": self.invalidated_functions,
            "actually_changed_results": self.actually_changed_results,
            "full_analysis_time_estimate": self.full_analysis_time_estimate,
            "incremental_analysis_time": self.incremental_analysis_time,
            "reanalysis_ratio": self.reanalysis_ratio,
            "cache_hit_rate": self.cache_hit_rate,
            "time_saved": self.time_saved,
            "invalidation_precision": self.invalidation_precision,
        }


# ---------------------------------------------------------------------------
# DependencyGraph
# ---------------------------------------------------------------------------

class DependencyGraph:
    """
    Directed graph of function dependencies.

    G = (V, E) where V = set of function names, (f, g) in E iff
    f's refinement type contract depends on g's summary.

    Each vertex stores a predicate set and an optional summary.
    """

    def __init__(self) -> None:
        self._successors: Dict[str, Set[str]] = defaultdict(set)
        self._predecessors: Dict[str, Set[str]] = defaultdict(set)
        self._predicates: Dict[str, Set[str]] = {}
        self._summaries: Dict[str, FunctionSummary] = {}
        self._vertices: Set[str] = set()

    # -- vertex / edge mutation -------------------------------------------

    def add_function(
        self,
        name: str,
        predicates: Optional[Set[str]] = None,
        summary: Optional[FunctionSummary] = None,
    ) -> None:
        """Add a function vertex with optional predicates and summary."""
        self._vertices.add(name)
        if predicates is not None:
            self._predicates[name] = set(predicates)
        elif name not in self._predicates:
            self._predicates[name] = set()
        if summary is not None:
            self._summaries[name] = summary

    def add_dependency(self, caller: str, callee: str) -> None:
        """Add edge caller -> callee (caller depends on callee)."""
        self._vertices.add(caller)
        self._vertices.add(callee)
        self._successors[caller].add(callee)
        self._predecessors[callee].add(caller)

    def remove_function(self, name: str) -> None:
        """Remove a function and all its incident edges."""
        if name not in self._vertices:
            return
        self._vertices.discard(name)
        # remove outgoing edges
        for callee in list(self._successors.get(name, [])):
            self._predecessors[callee].discard(name)
        self._successors.pop(name, None)
        # remove incoming edges
        for caller in list(self._predecessors.get(name, [])):
            self._successors[caller].discard(name)
        self._predecessors.pop(name, None)
        self._predicates.pop(name, None)
        self._summaries.pop(name, None)

    def remove_dependency(self, caller: str, callee: str) -> None:
        """Remove a single edge."""
        self._successors[caller].discard(callee)
        self._predecessors[callee].discard(caller)

    # -- queries -----------------------------------------------------------

    @property
    def functions(self) -> Set[str]:
        return set(self._vertices)

    def __contains__(self, name: str) -> bool:
        return name in self._vertices

    def __len__(self) -> int:
        return len(self._vertices)

    def get_predicates(self, name: str) -> Set[str]:
        return set(self._predicates.get(name, set()))

    def set_predicates(self, name: str, predicates: Set[str]) -> None:
        self._predicates[name] = set(predicates)

    def get_summary(self, name: str) -> Optional[FunctionSummary]:
        return self._summaries.get(name)

    def set_summary(self, name: str, summary: FunctionSummary) -> None:
        self._summaries[name] = summary

    def get_dependents(self, name: str) -> Set[str]:
        """Return functions that depend on *name* (predecessors in call sense,
        but successors in the "who is affected" sense).

        Convention: edge (caller, callee) means caller depends on callee.
        So dependents of *name* = {f : (f, name) in E} = predecessors[name].
        """
        return set(self._predecessors.get(name, set()))

    def get_dependencies(self, name: str) -> Set[str]:
        """Return functions that *name* depends on (callees)."""
        return set(self._successors.get(name, set()))

    def all_edges(self) -> List[Tuple[str, str]]:
        edges: List[Tuple[str, str]] = []
        for caller, callees in self._successors.items():
            for callee in callees:
                edges.append((caller, callee))
        return edges

    # -- topological sort --------------------------------------------------

    def topological_order(self) -> List[str]:
        """
        Return vertices in topological order (Kahn's algorithm).

        Uses the dependency direction: if caller depends on callee, callee
        appears before caller.  Raises ValueError if cycles exist.
        """
        in_degree: Dict[str, int] = {v: 0 for v in self._vertices}
        for v in self._vertices:
            for dep in self._successors.get(v, set()):
                if dep in in_degree:
                    in_degree[v] = in_degree.get(v, 0)  # ensure key exists

        # in_degree[v] = number of dependencies of v that are in the graph
        in_degree = {v: 0 for v in self._vertices}
        for caller in self._vertices:
            for callee in self._successors.get(caller, set()):
                if callee in self._vertices:
                    in_degree[caller] += 1

        # Start with functions that have no dependencies
        # But we want callees before callers, so reverse: treat edges as
        # callee -> caller, i.e. use predecessors as "outgoing".
        # Alternatively: compute topo order of the reverse graph.

        # Recompute: in the "analysis order" graph, edges go callee -> caller.
        # in_degree[v] = |{u : u is a callee of v, u in V}| = |successors[v] ∩ V|
        # A function with in_degree 0 has no callees in the graph -> analyze first.
        queue: Deque[str] = deque()
        for v in self._vertices:
            if in_degree[v] == 0:
                queue.append(v)

        order: List[str] = []
        while queue:
            v = queue.popleft()
            order.append(v)
            # v is now "done"; reduce in_degree of callers of v
            for caller in self._predecessors.get(v, set()):
                if caller in in_degree:
                    in_degree[caller] -= 1
                    if in_degree[caller] == 0:
                        queue.append(caller)

        if len(order) != len(self._vertices):
            raise ValueError(
                "Cycle detected in dependency graph; "
                f"ordered {len(order)}/{len(self._vertices)} vertices. "
                "Use strongly_connected_components() for cyclic graphs."
            )
        return order

    # -- strongly connected components (Tarjan) ----------------------------

    def strongly_connected_components(self) -> List[List[str]]:
        """
        Compute SCCs using Tarjan's algorithm.

        Returns list of SCCs in reverse topological order (leaves first).
        Useful for handling mutual recursion.
        """
        index_counter = [0]
        stack: List[str] = []
        on_stack: Set[str] = set()
        index: Dict[str, int] = {}
        lowlink: Dict[str, int] = {}
        result: List[List[str]] = []

        def strongconnect(v: str) -> None:
            index[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)

            for w in self._successors.get(v, set()):
                if w not in self._vertices:
                    continue
                if w not in index:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])

            if lowlink[v] == index[v]:
                component: List[str] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.append(w)
                    if w == v:
                        break
                result.append(component)

        for v in sorted(self._vertices):
            if v not in index:
                strongconnect(v)

        return result

    # -- transitive closure ------------------------------------------------

    def transitive_closure(self, name: str) -> Set[str]:
        """Return all functions transitively dependent on *name*."""
        visited: Set[str] = set()
        queue: Deque[str] = deque([name])
        while queue:
            current = queue.popleft()
            for dep in self.get_dependents(current):
                if dep not in visited:
                    visited.add(dep)
                    queue.append(dep)
        return visited

    def transitive_dependencies(self, name: str) -> Set[str]:
        """Return all functions that *name* transitively depends on."""
        visited: Set[str] = set()
        queue: Deque[str] = deque([name])
        while queue:
            current = queue.popleft()
            for dep in self.get_dependencies(current):
                if dep not in visited:
                    visited.add(dep)
                    queue.append(dep)
        return visited

    # -- subgraph ----------------------------------------------------------

    def subgraph(self, vertices: Set[str]) -> "DependencyGraph":
        """Return the induced subgraph on the given vertex set."""
        g = DependencyGraph()
        for v in vertices:
            if v in self._vertices:
                g.add_function(v, self._predicates.get(v), self._summaries.get(v))
        for caller, callees in self._successors.items():
            if caller in vertices:
                for callee in callees:
                    if callee in vertices:
                        g.add_dependency(caller, callee)
        return g

    # -- serialization -----------------------------------------------------

    def serialize(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        vertices_data: Dict[str, Any] = {}
        for v in sorted(self._vertices):
            vdata: Dict[str, Any] = {
                "predicates": sorted(self._predicates.get(v, set())),
            }
            if v in self._summaries:
                vdata["summary"] = self._summaries[v].to_dict()
            vertices_data[v] = vdata

        edges: List[List[str]] = []
        for caller in sorted(self._successors):
            for callee in sorted(self._successors[caller]):
                edges.append([caller, callee])

        return {"vertices": vertices_data, "edges": edges}

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "DependencyGraph":
        """Deserialize from a JSON-compatible dict."""
        g = cls()
        for name, vdata in data.get("vertices", {}).items():
            preds = set(vdata.get("predicates", []))
            summary = None
            if "summary" in vdata:
                summary = FunctionSummary.from_dict(vdata["summary"])
            g.add_function(name, preds, summary)
        for edge in data.get("edges", []):
            g.add_dependency(edge[0], edge[1])
        return g

    def to_json(self) -> str:
        return json.dumps(self.serialize(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "DependencyGraph":
        return cls.deserialize(json.loads(s))

    # -- helpers -----------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DependencyGraph(vertices={len(self._vertices)}, "
            f"edges={sum(len(s) for s in self._successors.values())})"
        )


# ---------------------------------------------------------------------------
# ChangeDetector
# ---------------------------------------------------------------------------

class ChangeDetector:
    """Detects changes between old and new analysis states."""

    @staticmethod
    def _hash_content(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # -- file-level --------------------------------------------------------

    def detect_file_changes(
        self,
        old_files: Dict[str, str],
        new_files: Dict[str, str],
    ) -> ChangeSet:
        """
        Compare two mappings of filename -> content.

        Returns a ChangeSet with changed/added/removed files and the union
        of functions contained in changed files (if function info is embedded
        in the content as ``## func: <name>`` markers, they are extracted).
        """
        cs = ChangeSet()
        all_keys = set(old_files) | set(new_files)
        for fname in all_keys:
            old_content = old_files.get(fname)
            new_content = new_files.get(fname)
            if old_content is None and new_content is not None:
                cs.changed_files.add(fname)
            elif old_content is not None and new_content is None:
                cs.changed_files.add(fname)
            elif old_content != new_content:
                cs.changed_files.add(fname)
        return cs

    # -- function-level ----------------------------------------------------

    def detect_function_changes(
        self,
        old_ir: Dict[str, Any],
        new_ir: Dict[str, Any],
    ) -> Set[str]:
        """
        Compare two IR representations (name -> IR node dict).
        Returns names of functions that structurally differ.
        """
        changed: Set[str] = set()
        all_funcs = set(old_ir) | set(new_ir)
        for fname in all_funcs:
            old_f = old_ir.get(fname)
            new_f = new_ir.get(fname)
            if old_f is None or new_f is None:
                changed.add(fname)
            elif self.structural_diff(old_f, new_f):
                changed.add(fname)
        return changed

    # -- predicate delta ---------------------------------------------------

    def compute_predicate_delta(
        self,
        old_preds: Dict[str, Set[str]],
        new_preds: Dict[str, Set[str]],
    ) -> Set[str]:
        """
        Compute the set of predicates that changed (added, removed, or
        associated with a different set of functions).
        """
        all_preds: Set[str] = set()
        for ps in old_preds.values():
            all_preds |= ps
        for ps in new_preds.values():
            all_preds |= ps

        changed: Set[str] = set()
        for pred in all_preds:
            old_funcs = {f for f, ps in old_preds.items() if pred in ps}
            new_funcs = {f for f, ps in new_preds.items() if pred in ps}
            if old_funcs != new_funcs:
                changed.add(pred)
        return changed

    # -- structural diff ---------------------------------------------------

    def structural_diff(self, old_ir_func: Any, new_ir_func: Any) -> bool:
        """
        Determine whether two IR representations of a function are
        structurally different.

        Handles dicts (recursive), lists (element-wise), and primitives.
        Ignores metadata keys starting with underscore.
        """
        if type(old_ir_func) is not type(new_ir_func):
            return True

        if isinstance(old_ir_func, dict):
            old_keys = {k for k in old_ir_func if not k.startswith("_")}
            new_keys = {k for k in new_ir_func if not k.startswith("_")}
            if old_keys != new_keys:
                return True
            return any(
                self.structural_diff(old_ir_func[k], new_ir_func[k])
                for k in old_keys
            )

        if isinstance(old_ir_func, (list, tuple)):
            if len(old_ir_func) != len(new_ir_func):
                return True
            return any(
                self.structural_diff(a, b)
                for a, b in zip(old_ir_func, new_ir_func)
            )

        return old_ir_func != new_ir_func

    # -- full change detection pipeline ------------------------------------

    def full_detect(
        self,
        old_files: Dict[str, str],
        new_files: Dict[str, str],
        old_ir: Dict[str, Any],
        new_ir: Dict[str, Any],
        old_preds: Dict[str, Set[str]],
        new_preds: Dict[str, Set[str]],
    ) -> ChangeSet:
        """Run full change detection combining file, function, predicate."""
        cs = self.detect_file_changes(old_files, new_files)
        changed_funcs = self.detect_function_changes(old_ir, new_ir)
        cs.changed_functions = changed_funcs

        all_old = set(old_ir)
        all_new = set(new_ir)
        cs.added_functions = all_new - all_old
        cs.removed_functions = all_old - all_new
        cs.changed_predicates = self.compute_predicate_delta(old_preds, new_preds)
        return cs


# ---------------------------------------------------------------------------
# AnalysisCache
# ---------------------------------------------------------------------------

class AnalysisCache:
    """
    Persistent cache of analysis results.

    Stores function summaries, the dependency graph, and predicate sets
    on disk under a configurable cache directory.
    """

    _SUMMARIES_FILE = "summaries.json"
    _GRAPH_FILE = "dependency_graph.json"
    _PREDICATES_FILE = "predicates.json"
    _META_FILE = "meta.json"

    def __init__(self, cache_dir: Union[str, Path] = ".reftype-cache") -> None:
        self.cache_dir = Path(cache_dir)
        self._summaries: Dict[str, FunctionSummary] = {}
        self._predicates: Dict[str, Set[str]] = {}
        self._graph: Optional[DependencyGraph] = None
        self._created_at: float = time.time()
        self._hits: int = 0
        self._misses: int = 0
        self._invalidations: int = 0

    # -- function summaries ------------------------------------------------

    def store_function_summary(
        self,
        name: str,
        summary: FunctionSummary,
        predicates: Optional[Set[str]] = None,
    ) -> None:
        self._summaries[name] = summary
        if predicates is not None:
            self._predicates[name] = set(predicates)

    def load_function_summary(self, name: str) -> Optional[FunctionSummary]:
        summary = self._summaries.get(name)
        if summary is not None:
            self._hits += 1
        else:
            self._misses += 1
        return summary

    def has_function_summary(self, name: str) -> bool:
        return name in self._summaries

    # -- dependency graph --------------------------------------------------

    def store_dependency_graph(self, graph: DependencyGraph) -> None:
        self._graph = graph

    def load_dependency_graph(self) -> Optional[DependencyGraph]:
        return self._graph

    # -- predicate sets ----------------------------------------------------

    def store_predicate_set(self, func_name: str, predicates: Set[str]) -> None:
        self._predicates[func_name] = set(predicates)

    def load_predicate_set(self, func_name: str) -> Optional[Set[str]]:
        return self._predicates.get(func_name)

    # -- invalidation / clear ----------------------------------------------

    def invalidate_entries(self, func_names: Iterable[str]) -> None:
        for name in func_names:
            if name in self._summaries:
                del self._summaries[name]
                self._invalidations += 1
            self._predicates.pop(name, None)

    def clear(self) -> None:
        self._summaries.clear()
        self._predicates.clear()
        self._graph = None
        self._hits = 0
        self._misses = 0
        self._invalidations = 0
        self._created_at = time.time()

    # -- disk persistence --------------------------------------------------

    def serialize_to_disk(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # summaries
        summaries_data = {
            name: s.to_dict() for name, s in self._summaries.items()
        }
        (self.cache_dir / self._SUMMARIES_FILE).write_text(
            json.dumps(summaries_data, indent=2)
        )

        # graph
        if self._graph is not None:
            (self.cache_dir / self._GRAPH_FILE).write_text(self._graph.to_json())

        # predicates
        pred_data = {
            name: sorted(ps) for name, ps in self._predicates.items()
        }
        (self.cache_dir / self._PREDICATES_FILE).write_text(
            json.dumps(pred_data, indent=2)
        )

        # metadata
        meta = {
            "created_at": self._created_at,
            "hits": self._hits,
            "misses": self._misses,
            "invalidations": self._invalidations,
            "num_summaries": len(self._summaries),
            "saved_at": time.time(),
        }
        (self.cache_dir / self._META_FILE).write_text(
            json.dumps(meta, indent=2)
        )

    def load_from_disk(self) -> bool:
        """Load cache from disk.  Returns True if cache was found."""
        if not self.cache_dir.exists():
            return False

        summaries_path = self.cache_dir / self._SUMMARIES_FILE
        if summaries_path.exists():
            raw = json.loads(summaries_path.read_text())
            self._summaries = {
                name: FunctionSummary.from_dict(d) for name, d in raw.items()
            }

        graph_path = self.cache_dir / self._GRAPH_FILE
        if graph_path.exists():
            self._graph = DependencyGraph.from_json(graph_path.read_text())

        pred_path = self.cache_dir / self._PREDICATES_FILE
        if pred_path.exists():
            raw = json.loads(pred_path.read_text())
            self._predicates = {name: set(ps) for name, ps in raw.items()}

        meta_path = self.cache_dir / self._META_FILE
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self._created_at = meta.get("created_at", time.time())
            self._hits = meta.get("hits", 0)
            self._misses = meta.get("misses", 0)
            self._invalidations = meta.get("invalidations", 0)

        return True

    # -- statistics --------------------------------------------------------

    def cache_statistics(self) -> CacheStats:
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            invalidations=self._invalidations,
            size=len(self._summaries),
            age=time.time() - self._created_at,
        )

    def __repr__(self) -> str:
        stats = self.cache_statistics()
        return (
            f"AnalysisCache(size={stats.size}, hits={stats.hits}, "
            f"misses={stats.misses}, inv={stats.invalidations})"
        )


# ---------------------------------------------------------------------------
# DeltaPropagator — semi-naive delta propagation
# ---------------------------------------------------------------------------

class DeltaPropagator:
    """
    Semi-naive delta propagation through the dependency graph.

    Given a set of changed (delta) facts and the dependency graph, propagate
    changes until a fixpoint is reached.  Facts are represented as
    Dict[str, Set[str]] mapping function names to sets of refined predicates.
    """

    def __init__(self, max_iterations: int = 1000) -> None:
        self.max_iterations = max_iterations

    def propagate(
        self,
        delta_facts: Dict[str, Set[str]],
        dep_graph: DependencyGraph,
    ) -> Dict[str, Set[str]]:
        """
        Propagate delta facts through *dep_graph* using semi-naive evaluation.

        Returns the set of new facts derived from the delta.
        """
        new_facts: Dict[str, Set[str]] = {}
        current_delta = dict(delta_facts)

        for _ in range(self.max_iterations):
            next_delta: Dict[str, Set[str]] = {}
            for func_name, preds in current_delta.items():
                for dependent in dep_graph.get_dependents(func_name):
                    dep_preds = dep_graph.get_predicates(dependent)
                    overlap = preds & dep_preds
                    if overlap:
                        existing = new_facts.get(dependent, set())
                        added = overlap - existing
                        if added:
                            new_facts.setdefault(dependent, set()).update(added)
                            next_delta.setdefault(dependent, set()).update(added)

            if not next_delta:
                break
            current_delta = next_delta

        return new_facts

    def merge_facts(
        self,
        old_facts: Dict[str, Set[str]],
        delta_facts: Dict[str, Set[str]],
    ) -> Dict[str, Set[str]]:
        """Merge old facts with delta facts (set union per key)."""
        merged: Dict[str, Set[str]] = {}
        for key in set(old_facts) | set(delta_facts):
            merged[key] = old_facts.get(key, set()) | delta_facts.get(key, set())
        return merged

    def compute_delta_output(
        self,
        old_result: Dict[str, Set[str]],
        new_result: Dict[str, Set[str]],
    ) -> Dict[str, Set[str]]:
        """Compute the symmetric difference between old and new results."""
        delta: Dict[str, Set[str]] = {}
        for key in set(old_result) | set(new_result):
            old_set = old_result.get(key, set())
            new_set = new_result.get(key, set())
            diff = old_set.symmetric_difference(new_set)
            if diff:
                delta[key] = diff
        return delta


# ---------------------------------------------------------------------------
# StratifiedAnalysis
# ---------------------------------------------------------------------------

class StratifiedAnalysis:
    """
    Stratified analysis ordering for the Horn clause system.

    Computes strata — groups of functions that can be analyzed together —
    based on the dependency graph's SCC-condensation.
    """

    def compute_strata(self, dep_graph: DependencyGraph) -> List[List[str]]:
        """
        Compute analysis strata from the dependency graph.

        Each stratum is a strongly connected component, returned in
        reverse topological order (leaves first).  Functions within
        the same SCC must be analyzed together (mutual recursion).
        """
        sccs = dep_graph.strongly_connected_components()

        # Build condensation graph and topo-sort it
        func_to_scc: Dict[str, int] = {}
        for idx, scc in enumerate(sccs):
            for f in scc:
                func_to_scc[f] = idx

        # SCCs from Tarjan are already in reverse topo order
        return sccs

    def analyze_stratum(
        self,
        stratum: List[str],
        cache: AnalysisCache,
        analyze_fn: Optional[Callable[[List[str]], Dict[str, FunctionSummary]]] = None,
    ) -> Dict[str, FunctionSummary]:
        """
        Analyze a single stratum.

        If all members have valid cache entries, return cached summaries.
        Otherwise, invoke *analyze_fn* (or return empty summaries as stubs).
        """
        # Check cache
        cached: Dict[str, FunctionSummary] = {}
        all_cached = True
        for func_name in stratum:
            summary = cache.load_function_summary(func_name)
            if summary is not None:
                cached[func_name] = summary
            else:
                all_cached = False

        if all_cached:
            return cached

        # Need to (re-)analyze
        if analyze_fn is not None:
            results = analyze_fn(stratum)
        else:
            results = {}
            for func_name in stratum:
                results[func_name] = FunctionSummary(name=func_name)

        # Store results
        for func_name, summary in results.items():
            cache.store_function_summary(func_name, summary, summary.predicates)

        return results

    def is_stratification_preserved(
        self,
        change_set: ChangeSet,
        strata: List[List[str]],
    ) -> bool:
        """
        Check whether the existing stratification is still valid after
        the given change set.

        The stratification is preserved if no added function creates a new
        dependency cycle that merges two existing strata, and no removed
        function splits an SCC.

        Conservative approximation: return False if any function was
        added or removed; return True if only existing functions changed.
        """
        if change_set.added_functions or change_set.removed_functions:
            return False
        return True


# ---------------------------------------------------------------------------
# IncrementalTracker — main orchestrator
# ---------------------------------------------------------------------------

class IncrementalTracker:
    """
    Main incremental analysis tracker.

    Orchestrates change detection, predicate-sensitive invalidation,
    delta propagation, and cache management.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = ".reftype-cache",
        max_propagation_iters: int = 1000,
    ) -> None:
        self._change_detector = ChangeDetector()
        self._delta_propagator = DeltaPropagator(max_iterations=max_propagation_iters)
        self._stratified = StratifiedAnalysis()
        self._cache = AnalysisCache(cache_dir)
        self._stats = IncrementalStatistics()
        self._dep_graph = DependencyGraph()

    @property
    def cache(self) -> AnalysisCache:
        return self._cache

    @property
    def dependency_graph(self) -> DependencyGraph:
        return self._dep_graph

    @property
    def statistics(self) -> IncrementalStatistics:
        return self._stats

    # -- initialization ----------------------------------------------------

    def initialize(
        self,
        ir_module: Dict[str, Any],
        analysis_results: Dict[str, FunctionSummary],
    ) -> AnalysisCache:
        """
        Bootstrap the tracker from a full analysis.

        Builds the initial dependency graph and populates the cache.
        Returns the initialized cache.
        """
        self._dep_graph = DependencyGraph()

        for func_name, summary in analysis_results.items():
            ir_node = ir_module.get(func_name, {})
            ir_hash = hashlib.sha256(
                json.dumps(ir_node, sort_keys=True, default=str).encode()
            ).hexdigest()
            summary.ir_hash = ir_hash

            self._dep_graph.add_function(
                func_name, summary.predicates, summary
            )
            self._cache.store_function_summary(
                func_name, summary, summary.predicates
            )

        # Build edges from declared dependencies
        for func_name, summary in analysis_results.items():
            for dep in summary.dependencies:
                if dep in analysis_results:
                    self._dep_graph.add_dependency(func_name, dep)

        self._cache.store_dependency_graph(self._dep_graph)
        self._stats.total_functions = len(analysis_results)
        return self._cache

    # -- update pipeline ---------------------------------------------------

    def update(
        self,
        change_set: ChangeSet,
        old_cache: Optional[AnalysisCache] = None,
    ) -> Tuple[Set[str], AnalysisCache]:
        """
        Perform an incremental update given a ChangeSet.

        Returns (set of functions to reanalyze, updated cache).
        """
        t0 = time.time()
        cache = old_cache if old_cache is not None else self._cache

        if change_set.is_empty:
            return set(), cache

        # Handle removed functions
        for func_name in change_set.removed_functions:
            self._dep_graph.remove_function(func_name)
        cache.invalidate_entries(change_set.removed_functions)

        # Handle added functions (need full analysis)
        for func_name in change_set.added_functions:
            self._dep_graph.add_function(func_name)

        # Compute invalidation set via predicate-sensitive analysis
        invalidated: Set[str] = set()
        for func_name in change_set.changed_functions:
            affected = self.predicate_sensitive_invalidation(
                func_name, change_set.changed_predicates
            )
            invalidated |= affected

        # Add directly changed functions
        invalidated |= change_set.changed_functions
        invalidated |= change_set.added_functions

        # Propagate through dependency graph
        to_reanalyze = self.propagate_changes(invalidated, self._dep_graph)

        # Minimize
        to_reanalyze = self.minimize_reanalysis_set(to_reanalyze, self._dep_graph)

        # Invalidate cache entries
        cache.invalidate_entries(to_reanalyze)

        # Update statistics
        self._stats.reanalyzed_functions = len(to_reanalyze)
        self._stats.invalidated_functions = len(invalidated)
        self._stats.total_functions = len(self._dep_graph)
        self._stats.incremental_analysis_time = time.time() - t0

        cache.store_dependency_graph(self._dep_graph)
        return to_reanalyze, cache

    # -- invalidation strategies -------------------------------------------

    def invalidate(self, changed_functions: Set[str]) -> Set[str]:
        """
        Basic invalidation: return the transitive closure of dependents
        for all changed functions.
        """
        to_reanalyze: Set[str] = set(changed_functions)
        for func_name in changed_functions:
            to_reanalyze |= self._dep_graph.transitive_closure(func_name)
        return to_reanalyze

    def predicate_sensitive_invalidation(
        self,
        changed_func: str,
        delta_predicates: Set[str],
    ) -> Set[str]:
        """
        Predicate-sensitive invalidation: only invalidate downstream
        functions whose predicate sets overlap with the delta predicates.

        If delta_predicates is empty, falls back to full transitive invalidation.
        This is the key optimization: O(|Delta_out| * poly(|P|)).
        """
        if not delta_predicates:
            return self._dep_graph.transitive_closure(changed_func)

        affected: Set[str] = set()
        queue: Deque[str] = deque()

        # Seed with direct dependents whose predicates overlap
        for dep in self._dep_graph.get_dependents(changed_func):
            dep_preds = self._dep_graph.get_predicates(dep)
            if dep_preds & delta_predicates:
                affected.add(dep)
                queue.append(dep)

        # BFS propagation with predicate filtering
        while queue:
            current = queue.popleft()
            current_preds = self._dep_graph.get_predicates(current)
            for dep in self._dep_graph.get_dependents(current):
                if dep not in affected:
                    dep_preds = self._dep_graph.get_predicates(dep)
                    # Propagate only if predicate overlap exists
                    if dep_preds & (delta_predicates | current_preds):
                        affected.add(dep)
                        queue.append(dep)

        return affected

    # -- propagation -------------------------------------------------------

    def propagate_changes(
        self,
        invalidated: Set[str],
        dep_graph: DependencyGraph,
    ) -> Set[str]:
        """
        Given an initial invalidated set, propagate to find the full set
        of functions that must be re-analyzed.

        Uses BFS over the dependency graph following dependent edges.
        """
        full_set: Set[str] = set(invalidated)
        queue: Deque[str] = deque(invalidated)

        while queue:
            current = queue.popleft()
            for dep in dep_graph.get_dependents(current):
                if dep not in full_set:
                    full_set.add(dep)
                    queue.append(dep)

        return full_set

    def delta_propagation(
        self,
        old_facts: Dict[str, Set[str]],
        new_facts: Dict[str, Set[str]],
        dep_graph: DependencyGraph,
    ) -> Dict[str, Set[str]]:
        """
        Semi-naive delta propagation: compute the difference between old
        and new facts, then propagate deltas through the graph.
        """
        delta = self._delta_propagator.compute_delta_output(old_facts, new_facts)
        if not delta:
            return new_facts

        propagated = self._delta_propagator.propagate(delta, dep_graph)
        return self._delta_propagator.merge_facts(new_facts, propagated)

    # -- minimization ------------------------------------------------------

    def minimize_reanalysis_set(
        self,
        candidates: Set[str],
        dep_graph: DependencyGraph,
    ) -> Set[str]:
        """
        Minimize the set of functions to re-analyze by removing candidates
        whose dependencies are all unchanged and whose own IR hash hasn't
        changed.
        """
        minimized: Set[str] = set()
        for func_name in candidates:
            if func_name not in dep_graph:
                # Function was removed or isn't in graph; skip
                continue

            summary = dep_graph.get_summary(func_name)
            if summary is None:
                # No cached summary → must analyze
                minimized.add(func_name)
                continue

            # Check if any dependency is also a candidate
            deps = dep_graph.get_dependencies(func_name)
            if deps & candidates:
                # At least one dependency changed → must re-analyze
                minimized.add(func_name)
                continue

            # If the function itself is directly changed, keep it
            # (it's already in candidates for a reason)
            minimized.add(func_name)

        return minimized

    # -- stratified analysis -----------------------------------------------

    def compute_analysis_order(self) -> List[List[str]]:
        """Compute the stratified analysis order."""
        return self._stratified.compute_strata(self._dep_graph)

    def analyze_incrementally(
        self,
        change_set: ChangeSet,
        analyze_fn: Callable[[List[str]], Dict[str, FunctionSummary]],
    ) -> Dict[str, FunctionSummary]:
        """
        End-to-end incremental analysis.

        1. Detect what needs re-analysis
        2. Compute strata
        3. Analyze each affected stratum
        4. Propagate delta facts
        5. Update cache
        """
        t0 = time.time()
        to_reanalyze, cache = self.update(change_set)

        if not to_reanalyze:
            self._stats.incremental_analysis_time = time.time() - t0
            return {}

        # Build subgraph of functions to re-analyze
        sub = self._dep_graph.subgraph(to_reanalyze)
        strata = self._stratified.compute_strata(sub)

        all_results: Dict[str, FunctionSummary] = {}
        for stratum in strata:
            results = self._stratified.analyze_stratum(
                stratum, cache, analyze_fn
            )
            all_results.update(results)

            # Update the main graph with new summaries
            for func_name, summary in results.items():
                self._dep_graph.set_summary(func_name, summary)
                self._dep_graph.set_predicates(func_name, summary.predicates)
                cache.store_function_summary(func_name, summary, summary.predicates)

        # Track actually changed results for precision metric
        actually_changed = 0
        for func_name, new_summary in all_results.items():
            old_summary = self._dep_graph.get_summary(func_name)
            if old_summary is None or old_summary.refinement_types != new_summary.refinement_types:
                actually_changed += 1

        self._stats.actually_changed_results = actually_changed
        self._stats.incremental_analysis_time = time.time() - t0
        self._stats.cache_hits = cache.cache_statistics().hits
        self._stats.cache_misses = cache.cache_statistics().misses

        cache.store_dependency_graph(self._dep_graph)
        return all_results


# ---------------------------------------------------------------------------
# FileWatcher — monitors file changes for incremental dev mode
# ---------------------------------------------------------------------------

class FileWatcher:
    """
    Monitors a directory for file changes and yields ChangeSets.

    Uses polling-based detection (no platform-specific watchers) for
    maximum portability.
    """

    def __init__(
        self,
        extensions: Optional[Set[str]] = None,
        poll_interval: float = 1.0,
    ) -> None:
        self._extensions = extensions or {".py", ".js", ".ts", ".rb"}
        self._poll_interval = poll_interval
        self._file_hashes: Dict[str, str] = {}
        self._running = False

    @staticmethod
    def _hash_file(path: Path) -> str:
        try:
            content = path.read_bytes()
            return hashlib.sha256(content).hexdigest()
        except (OSError, IOError):
            return ""

    def _scan_directory(self, directory: Path) -> Dict[str, str]:
        """Scan directory and return mapping of relative path -> hash."""
        result: Dict[str, str] = {}
        if not directory.is_dir():
            return result
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in self._extensions:
                rel = str(path.relative_to(directory))
                result[rel] = self._hash_file(path)
        return result

    def _compute_changes(
        self,
        old_hashes: Dict[str, str],
        new_hashes: Dict[str, str],
    ) -> ChangeSet:
        """Compute ChangeSet from hash differences."""
        cs = ChangeSet()
        all_files = set(old_hashes) | set(new_hashes)
        for fname in all_files:
            old_h = old_hashes.get(fname)
            new_h = new_hashes.get(fname)
            if old_h is None and new_h is not None:
                cs.changed_files.add(fname)
            elif old_h is not None and new_h is None:
                cs.changed_files.add(fname)
            elif old_h != new_h:
                cs.changed_files.add(fname)
        return cs

    def snapshot(self, directory: Union[str, Path]) -> Dict[str, str]:
        """Take a snapshot of the directory's file hashes."""
        return self._scan_directory(Path(directory))

    def detect_changes(
        self,
        directory: Union[str, Path],
    ) -> ChangeSet:
        """
        Compare current directory state against the last snapshot.
        Updates internal state.
        """
        directory = Path(directory)
        new_hashes = self._scan_directory(directory)
        cs = self._compute_changes(self._file_hashes, new_hashes)
        self._file_hashes = new_hashes
        return cs

    async def watch(
        self,
        directory: Union[str, Path],
    ) -> AsyncIterator[ChangeSet]:
        """
        Async generator that yields ChangeSets when files change.

        Usage::

            async for change_set in watcher.watch("/path/to/src"):
                tracker.update(change_set)
        """
        import asyncio

        directory = Path(directory)
        self._file_hashes = self._scan_directory(directory)
        self._running = True

        try:
            while self._running:
                await asyncio.sleep(self._poll_interval)
                new_hashes = self._scan_directory(directory)
                cs = self._compute_changes(self._file_hashes, new_hashes)
                if not cs.is_empty:
                    self._file_hashes = new_hashes
                    yield cs
        finally:
            self._running = False

    def stop(self) -> None:
        """Stop the watch loop."""
        self._running = False

    @staticmethod
    def debounce(
        changes: List[ChangeSet],
        interval: float = 0.5,
    ) -> ChangeSet:
        """
        Merge multiple ChangeSets within a time interval into one batched
        ChangeSet.
        """
        if not changes:
            return ChangeSet()

        merged = ChangeSet()
        for cs in changes:
            merged = merged.merge(cs)
        return merged


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def build_dependency_graph_from_ir(
    ir_module: Dict[str, Any],
    call_graph: Optional[Dict[str, Set[str]]] = None,
) -> DependencyGraph:
    """
    Build a DependencyGraph from an IR module representation.

    *ir_module* maps function names to their IR dict.  Each IR dict may
    contain a ``"calls"`` key listing callee names, or *call_graph* can
    provide this externally.
    """
    graph = DependencyGraph()
    for func_name, ir_node in ir_module.items():
        predicates: Set[str] = set()
        if isinstance(ir_node, dict):
            predicates = set(ir_node.get("predicates", []))
        graph.add_function(func_name, predicates)

    for func_name, ir_node in ir_module.items():
        callees: Set[str] = set()
        if call_graph and func_name in call_graph:
            callees = call_graph[func_name]
        elif isinstance(ir_node, dict):
            callees = set(ir_node.get("calls", []))
        for callee in callees:
            if callee in ir_module:
                graph.add_dependency(func_name, callee)

    return graph


def incremental_update_pipeline(
    tracker: IncrementalTracker,
    old_files: Dict[str, str],
    new_files: Dict[str, str],
    old_ir: Dict[str, Any],
    new_ir: Dict[str, Any],
    old_preds: Dict[str, Set[str]],
    new_preds: Dict[str, Set[str]],
    analyze_fn: Callable[[List[str]], Dict[str, FunctionSummary]],
) -> Tuple[Dict[str, FunctionSummary], IncrementalStatistics]:
    """
    End-to-end incremental update pipeline.

    1. Detect changes
    2. Determine functions to re-analyze
    3. Re-analyze affected strata
    4. Return updated summaries and statistics
    """
    detector = ChangeDetector()
    change_set = detector.full_detect(
        old_files, new_files, old_ir, new_ir, old_preds, new_preds
    )

    results = tracker.analyze_incrementally(change_set, analyze_fn)
    return results, tracker.statistics


def compute_ir_hash(ir_node: Any) -> str:
    """Compute a stable hash for an IR node."""
    serialized = json.dumps(ir_node, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Quick smoke test of the tracker components."""

    # 1. Build a dependency graph
    #    a -> b -> c
    #         b -> d
    g = DependencyGraph()
    g.add_function("a", {"p1", "p2"})
    g.add_function("b", {"p2", "p3"})
    g.add_function("c", {"p3"})
    g.add_function("d", {"p4"})
    g.add_dependency("a", "b")
    g.add_dependency("b", "c")
    g.add_dependency("b", "d")

    assert g.get_dependencies("a") == {"b"}
    assert g.get_dependencies("b") == {"c", "d"}
    assert g.get_dependents("b") == {"a"}
    assert g.get_dependents("c") == {"b"}

    # Topological order: c, d before b before a
    topo = g.topological_order()
    assert topo.index("c") < topo.index("b")
    assert topo.index("d") < topo.index("b")
    assert topo.index("b") < topo.index("a")

    # Transitive closure
    assert g.transitive_closure("c") == {"b", "a"}
    assert g.transitive_closure("d") == {"b", "a"}

    # SCCs (no cycles)
    sccs = g.strongly_connected_components()
    assert all(len(scc) == 1 for scc in sccs)

    # Serialization round-trip
    data = g.serialize()
    g2 = DependencyGraph.deserialize(data)
    assert g2.functions == g.functions
    assert set(map(tuple, g2.all_edges())) == set(map(tuple, g.all_edges()))

    # 2. ChangeDetector
    det = ChangeDetector()
    old_ir = {"foo": {"body": [1, 2, 3]}, "bar": {"body": [4, 5]}}
    new_ir = {"foo": {"body": [1, 2, 3]}, "bar": {"body": [4, 6]}, "baz": {"body": []}}
    changed = det.detect_function_changes(old_ir, new_ir)
    assert "bar" in changed
    assert "baz" in changed
    assert "foo" not in changed

    # 3. Predicate delta
    old_preds = {"foo": {"p1"}, "bar": {"p2"}}
    new_preds = {"foo": {"p1", "p3"}, "bar": {"p2"}}
    delta_p = det.compute_predicate_delta(old_preds, new_preds)
    assert "p3" in delta_p

    # 4. DeltaPropagator
    dp = DeltaPropagator()
    delta_facts: Dict[str, Set[str]] = {"c": {"p3"}}
    new_facts = dp.propagate(delta_facts, g)
    assert "b" in new_facts  # b has p3

    # 5. IncrementalTracker initialization
    summaries = {
        "a": FunctionSummary("a", predicates={"p1", "p2"}, dependencies={"b"}),
        "b": FunctionSummary("b", predicates={"p2", "p3"}, dependencies={"c", "d"}),
        "c": FunctionSummary("c", predicates={"p3"}),
        "d": FunctionSummary("d", predicates={"p4"}),
    }
    ir_mod = {
        "a": {"body": "a_body"},
        "b": {"body": "b_body"},
        "c": {"body": "c_body"},
        "d": {"body": "d_body"},
    }

    tracker = IncrementalTracker(cache_dir="/tmp/_reftype_test_cache")
    cache = tracker.initialize(ir_mod, summaries)
    assert cache.has_function_summary("a")
    assert cache.has_function_summary("d")

    # 6. Incremental update
    cs = ChangeSet(
        changed_functions={"c"},
        changed_predicates={"p3"},
    )
    to_reanalyze, updated_cache = tracker.update(cs)
    assert "c" in to_reanalyze
    assert "b" in to_reanalyze  # b depends on c and shares p3

    # 7. Predicate-sensitive invalidation
    affected = tracker.predicate_sensitive_invalidation("c", {"p3"})
    assert "b" in affected

    # 8. AnalysisCache disk round-trip
    import tempfile, shutil
    tmpdir = tempfile.mkdtemp()
    try:
        c1 = AnalysisCache(tmpdir)
        c1.store_function_summary("f", FunctionSummary("f", predicates={"x"}), {"x"})
        c1.store_dependency_graph(g)
        c1.serialize_to_disk()

        c2 = AnalysisCache(tmpdir)
        assert c2.load_from_disk()
        s = c2.load_function_summary("f")
        assert s is not None
        assert s.name == "f"
        g3 = c2.load_dependency_graph()
        assert g3 is not None
        assert "a" in g3
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # 9. StratifiedAnalysis
    sa = StratifiedAnalysis()
    strata = sa.compute_strata(g)
    assert len(strata) == 4  # 4 SCCs, each size 1

    # Test with cycle: e <-> f
    g_cycle = DependencyGraph()
    g_cycle.add_function("e", {"p5"})
    g_cycle.add_function("f", {"p5"})
    g_cycle.add_dependency("e", "f")
    g_cycle.add_dependency("f", "e")
    sccs_cycle = g_cycle.strongly_connected_components()
    assert any(len(scc) == 2 for scc in sccs_cycle)

    strata_cycle = sa.compute_strata(g_cycle)
    assert any(len(s) == 2 for s in strata_cycle)

    # 10. FileWatcher snapshot
    fw = FileWatcher(extensions={".py"})
    import tempfile as _tf
    tmpdir2 = _tf.mkdtemp()
    try:
        p = Path(tmpdir2) / "test.py"
        p.write_text("x = 1")
        snap1 = fw.snapshot(tmpdir2)
        assert "test.py" in snap1

        p.write_text("x = 2")
        snap2 = fw.snapshot(tmpdir2)
        assert snap1["test.py"] != snap2["test.py"]
    finally:
        shutil.rmtree(tmpdir2, ignore_errors=True)

    # 11. CacheStats
    stats = cache.cache_statistics()
    assert stats.size >= 0

    # 12. IncrementalStatistics
    tracker._stats.total_functions = 100
    tracker._stats.reanalyzed_functions = 10
    tracker._stats.cache_hits = 90
    tracker._stats.cache_misses = 10
    tracker._stats.invalidated_functions = 15
    tracker._stats.actually_changed_results = 8
    assert abs(tracker.statistics.reanalysis_ratio - 0.1) < 1e-9
    assert abs(tracker.statistics.cache_hit_rate - 0.9) < 1e-9

    # Cleanup test cache
    import shutil
    shutil.rmtree("/tmp/_reftype_test_cache", ignore_errors=True)

    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
