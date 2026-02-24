"""Code evolution tracker using file version history.

Analyzes how code changes over time by tracking complexity, churn,
bug introduction patterns, ownership, technical debt markers,
API stability, and file coupling.
"""
from __future__ import annotations

import ast
import copy
import re
import textwrap
import hashlib
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _normalize_function_body(node: ast.FunctionDef) -> str:
    """Normalize a function body to a canonical string for hashing."""
    node = copy.deepcopy(node)
    node.decorator_list = []
    if (node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, (ast.Constant, ast.Str))):
        node.body = node.body[1:]
    return ast.dump(node, annotate_fields=False)


def hash_function_body(node: ast.FunctionDef) -> str:
    """Return a SHA-256 hex digest of the normalized function body."""
    canonical = _normalize_function_body(node)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def extract_functions(source: str) -> Dict[str, ast.FunctionDef]:
    """Parse *source* and return {qualified_name: FunctionDef}."""
    tree = ast.parse(source)
    functions: Dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            functions[node.name] = node
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions[f"{node.name}.{item.name}"] = item
    return functions


@dataclass
class FunctionSignature:
    """Captures the externally visible signature of a function."""
    name: str
    params: Tuple[str, ...]
    defaults_count: int
    has_varargs: bool
    has_kwargs: bool
    return_annotation: Optional[str]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FunctionSignature):
            return NotImplemented
        return (self.name == other.name
                and self.params == other.params
                and self.defaults_count == other.defaults_count
                and self.has_varargs == other.has_varargs
                and self.has_kwargs == other.has_kwargs
                and self.return_annotation == other.return_annotation)

    def __hash__(self) -> int:
        return hash((self.name, self.params, self.defaults_count,
                      self.has_varargs, self.has_kwargs, self.return_annotation))


def extract_signature(node: ast.FunctionDef) -> FunctionSignature:
    """Extract the public signature from a function AST node."""
    params = [arg.arg for arg in node.args.args]
    ret_ann = ast.dump(node.returns) if node.returns is not None else None
    return FunctionSignature(
        name=node.name, params=tuple(params),
        defaults_count=len(node.args.defaults),
        has_varargs=node.args.vararg is not None,
        has_kwargs=node.args.kwarg is not None,
        return_annotation=ret_ann,
    )


def diff_function_sets(
    old_funcs: Dict[str, ast.FunctionDef],
    new_funcs: Dict[str, ast.FunctionDef],
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Return (added, removed, modified) sets of function names."""
    old_names, new_names = set(old_funcs), set(new_funcs)
    added = new_names - old_names
    removed = old_names - new_names
    modified = {n for n in old_names & new_names
                if hash_function_body(old_funcs[n]) != hash_function_body(new_funcs[n])}
    return added, removed, modified


def _string_diff_ratio(a: str, b: str) -> float:
    """Cheap diff ratio: fraction of characters that differ."""
    if not a and not b:
        return 0.0
    max_len = max(len(a), len(b))
    matching = sum(ca == cb for ca, cb in zip(a, b))
    return 1.0 - matching / max_len


def compute_all_hashes(source: str) -> Dict[str, str]:
    """Return {func_name: body_hash} for every function in *source*."""
    funcs = extract_functions(source)
    return {name: hash_function_body(node) for name, node in funcs.items()}


def version_diff_summary(old_source: str, new_source: str) -> Dict[str, Any]:
    """Produce a human-readable diff summary between two versions."""
    old_funcs = extract_functions(old_source)
    new_funcs = extract_functions(new_source)
    added, removed, modified = diff_function_sets(old_funcs, new_funcs)
    unchanged = set(old_funcs) & set(new_funcs) - modified
    return {"added": sorted(added), "removed": sorted(removed),
            "modified": sorted(modified), "unchanged": sorted(unchanged)}

# ---------------------------------------------------------------------------
# 1. Complexity tracking (McCabe cyclomatic complexity)
# ---------------------------------------------------------------------------

class _ComplexityVisitor(ast.NodeVisitor):
    """Walk a function AST and count McCabe decision points."""
    def __init__(self) -> None:
        self.complexity = 1  # base path

    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.complexity += len(node.ifs)
        self.generic_visit(node)


def compute_cyclomatic_complexity(func_node: ast.FunctionDef) -> int:
    """Return cyclomatic complexity of a single function node."""
    visitor = _ComplexityVisitor()
    visitor.visit(func_node)
    return visitor.complexity


@dataclass
class ComplexityTrajectory:
    """Per-function complexity over successive versions."""
    function_trajectories: Dict[str, List[int]] = field(default_factory=dict)
    version_count: int = 0

    def mean_complexity(self, func_name: str) -> float:
        vals = self.function_trajectories.get(func_name, [])
        return float(np.mean(vals)) if vals else 0.0

    def max_complexity(self, func_name: str) -> int:
        vals = self.function_trajectories.get(func_name, [])
        return max(vals) if vals else 0

    def trend(self, func_name: str) -> float:
        """Return slope of a linear fit over versions (positive = growing)."""
        vals = self.function_trajectories.get(func_name, [])
        if len(vals) < 2:
            return 0.0
        x = np.arange(len(vals), dtype=np.float64)
        coeffs = np.polyfit(x, np.array(vals, dtype=np.float64), 1)
        return float(coeffs[0])


class ComplexityTracker:
    """Track cyclomatic complexity of every function across versions."""
    def track(self, versions: List[str]) -> ComplexityTrajectory:
        traj = ComplexityTrajectory(version_count=len(versions))
        all_funcs: Dict[str, List[int]] = defaultdict(list)
        for source in versions:
            funcs = extract_functions(source)
            for name, node in funcs.items():
                all_funcs[name].append(compute_cyclomatic_complexity(node))
        traj.function_trajectories = dict(all_funcs)
        return traj

# ---------------------------------------------------------------------------
# 2. Churn analysis
# ---------------------------------------------------------------------------

@dataclass
class ChurnReport:
    change_counts: Dict[str, int] = field(default_factory=dict)
    churn_rates: Dict[str, float] = field(default_factory=dict)
    hotspots: List[Tuple[str, float]] = field(default_factory=list)
    version_count: int = 0

    def top_hotspots(self, n: int = 5) -> List[Tuple[str, float]]:
        return self.hotspots[:n]


class ChurnAnalyzer:
    """Detect which functions change most frequently across versions."""
    def analyze(self, versions: List[str]) -> ChurnReport:
        if len(versions) < 2:
            return ChurnReport(version_count=len(versions))
        change_counts: Counter[str] = Counter()
        prev_hashes = {n: hash_function_body(f) for n, f in extract_functions(versions[0]).items()}
        for source in versions[1:]:
            cur_hashes = {n: hash_function_body(f) for n, f in extract_functions(source).items()}
            for name in cur_hashes:
                if name in prev_hashes and cur_hashes[name] != prev_hashes[name]:
                    change_counts[name] += 1
                elif name not in prev_hashes:
                    change_counts[name] += 1
            prev_hashes = cur_hashes
        num_transitions = len(versions) - 1
        churn_rates = {n: c / num_transitions for n, c in change_counts.items()}
        hotspots = sorted(churn_rates.items(), key=lambda t: t[1], reverse=True)
        return ChurnReport(change_counts=dict(change_counts), churn_rates=churn_rates,
                           hotspots=hotspots, version_count=len(versions))

# ---------------------------------------------------------------------------
# 3. Bug introduction estimation
# ---------------------------------------------------------------------------

class ChangeKind(Enum):
    NONE = auto()
    ADDED = auto()
    REMOVED = auto()
    MODIFIED = auto()


@dataclass
class BugIntroduction:
    function_name: str
    introduced_version: int
    fixed_version: int
    complexity_spike: int
    confidence: float


class BugEstimator:
    """Heuristic bug-introduction estimator.

    A complexity spike followed by addition of error-handling constructs
    suggests the spike introduced a bug that was subsequently fixed.
    """
    def __init__(self, spike_threshold: float = 1.5, lookahead: int = 3) -> None:
        self.spike_threshold = spike_threshold
        self.lookahead = lookahead

    def _count_error_handling(self, node: ast.FunctionDef) -> int:
        count = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.Try, ast.ExceptHandler, ast.Raise)):
                count += 1
        return count

    def estimate(self, versions: List[str]) -> List[BugIntroduction]:
        if len(versions) < 3:
            return []
        traj = ComplexityTracker().track(versions)
        eh_counts: Dict[str, List[int]] = defaultdict(list)
        for source in versions:
            funcs = extract_functions(source)
            for name, node in funcs.items():
                eh_counts[name].append(self._count_error_handling(node))
        bugs: List[BugIntroduction] = []
        for func_name, complexities in traj.function_trajectories.items():
            if len(complexities) < 3:
                continue
            arr = np.array(complexities, dtype=np.float64)
            for i in range(1, len(arr)):
                if arr[i - 1] == 0:
                    continue
                ratio = arr[i] / arr[i - 1]
                if ratio < self.spike_threshold:
                    continue
                eh_vals = eh_counts.get(func_name, [])
                if i >= len(eh_vals):
                    continue
                eh_at_spike = eh_vals[i]
                for j in range(i + 1, min(i + 1 + self.lookahead, len(eh_vals))):
                    if eh_vals[j] > eh_at_spike:
                        confidence = min(1.0, (ratio - 1.0) / 3.0)
                        confidence *= min(1.0, (eh_vals[j] - eh_at_spike) / 2.0)
                        bugs.append(BugIntroduction(
                            function_name=func_name, introduced_version=i,
                            fixed_version=j, complexity_spike=int(arr[i]),
                            confidence=round(confidence, 3)))
                        break
        return bugs

# ---------------------------------------------------------------------------
# 4. Code ownership
# ---------------------------------------------------------------------------

@dataclass
class OwnershipRecord:
    creator_version: int
    last_major_change_version: int
    total_modifications: int


@dataclass
class OwnershipReport:
    ownership: Dict[str, OwnershipRecord] = field(default_factory=dict)
    version_count: int = 0

    def functions_created_at(self, version: int) -> List[str]:
        return [n for n, r in self.ownership.items() if r.creator_version == version]

    def most_modified(self, n: int = 5) -> List[Tuple[str, int]]:
        ranked = sorted(self.ownership.items(), key=lambda t: t[1].total_modifications, reverse=True)
        return [(name, rec.total_modifications) for name, rec in ranked[:n]]


class OwnershipTracker:
    """Attribute each function to the version that created or last heavily modified it."""
    MAJOR_CHANGE_THRESHOLD = 0.3

    def track(self, versions: List[str]) -> OwnershipReport:
        report = OwnershipReport(version_count=len(versions))
        prev_funcs: Dict[str, ast.FunctionDef] = {}
        prev_hashes: Dict[str, str] = {}
        for idx, source in enumerate(versions):
            cur_funcs = extract_functions(source)
            cur_hashes = {n: hash_function_body(f) for n, f in cur_funcs.items()}
            for name, node in cur_funcs.items():
                if name not in report.ownership:
                    report.ownership[name] = OwnershipRecord(
                        creator_version=idx, last_major_change_version=idx,
                        total_modifications=0)
                elif name in prev_hashes and cur_hashes[name] != prev_hashes[name]:
                    rec = report.ownership[name]
                    rec.total_modifications += 1
                    old_body = _normalize_function_body(prev_funcs[name])
                    new_body = _normalize_function_body(node)
                    if _string_diff_ratio(old_body, new_body) >= self.MAJOR_CHANGE_THRESHOLD:
                        rec.last_major_change_version = idx
            prev_funcs = cur_funcs
            prev_hashes = cur_hashes
        return report

# ---------------------------------------------------------------------------
# 5. Technical debt tracking
# ---------------------------------------------------------------------------

_DEBT_PATTERN = re.compile(r"\b(TODO|FIXME|HACK|XXX|NOQA)\b", re.IGNORECASE)


@dataclass
class DebtTrajectory:
    counts: List[int] = field(default_factory=list)
    per_marker: Dict[str, List[int]] = field(default_factory=dict)
    velocity: float = 0.0

    def total_at(self, version: int) -> int:
        if 0 <= version < len(self.counts):
            return self.counts[version]
        return 0

    def accumulation_rate(self) -> float:
        if len(self.counts) < 2:
            return 0.0
        diffs = np.diff(np.array(self.counts, dtype=np.float64))
        return float(np.mean(diffs))


class DebtTracker:
    """Count TODO / FIXME / HACK / XXX / NOQA across versions."""
    def track(self, versions: List[str]) -> DebtTrajectory:
        traj = DebtTrajectory()
        marker_names = ["TODO", "FIXME", "HACK", "XXX", "NOQA"]
        per_marker: Dict[str, List[int]] = {m: [] for m in marker_names}
        for source in versions:
            total = 0
            local_counts: Dict[str, int] = Counter()
            for match in _DEBT_PATTERN.finditer(source):
                key = match.group(1).upper()
                local_counts[key] += 1
                total += 1
            traj.counts.append(total)
            for m in marker_names:
                per_marker[m].append(local_counts.get(m, 0))
        traj.per_marker = per_marker
        if len(traj.counts) >= 2:
            x = np.arange(len(traj.counts), dtype=np.float64)
            coeffs = np.polyfit(x, np.array(traj.counts, dtype=np.float64), 1)
            traj.velocity = float(coeffs[0])
        return traj

# ---------------------------------------------------------------------------
# 6. API stability
# ---------------------------------------------------------------------------

@dataclass
class StabilityReport:
    scores: Dict[str, float] = field(default_factory=dict)
    signature_changes: Dict[str, List[int]] = field(default_factory=dict)
    version_count: int = 0

    def most_stable(self, n: int = 5) -> List[Tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda t: t[1], reverse=True)[:n]

    def least_stable(self, n: int = 5) -> List[Tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda t: t[1])[:n]


class APIStabilityAnalyzer:
    """Track function signatures across versions and compute stability."""
    def analyze(self, versions: List[str]) -> StabilityReport:
        if not versions:
            return StabilityReport()
        all_sigs: Dict[str, List[Optional[FunctionSignature]]] = defaultdict(list)
        all_func_names: Set[str] = set()
        for source in versions:
            funcs = extract_functions(source)
            current_names = set(funcs.keys())
            all_func_names |= current_names
            for name in all_func_names:
                if name in funcs:
                    all_sigs[name].append(extract_signature(funcs[name]))
                else:
                    all_sigs[name].append(None)
        report = StabilityReport(version_count=len(versions))
        for name, sigs in all_sigs.items():
            present = [s for s in sigs if s is not None]
            if not present:
                report.scores[name] = 0.0
                report.signature_changes[name] = []
                continue
            changes: List[int] = []
            unchanged_count = 0
            total_present = 0
            for i in range(1, len(sigs)):
                if sigs[i] is None or sigs[i - 1] is None:
                    continue
                total_present += 1
                if sigs[i] == sigs[i - 1]:
                    unchanged_count += 1
                else:
                    changes.append(i)
            score = unchanged_count / total_present if total_present > 0 else 1.0
            report.scores[name] = round(score, 4)
            report.signature_changes[name] = changes
        return report

# ---------------------------------------------------------------------------
# 7. File coupling analysis
# ---------------------------------------------------------------------------

@dataclass
class CouplingReport:
    file_names: List[str] = field(default_factory=list)
    coupling_matrix: Optional[np.ndarray] = None
    top_pairs: List[Tuple[str, str, float]] = field(default_factory=list)

    def coupling_score(self, file_a: str, file_b: str) -> float:
        if self.coupling_matrix is None:
            return 0.0
        try:
            ia = self.file_names.index(file_a)
            ib = self.file_names.index(file_b)
        except ValueError:
            return 0.0
        return float(self.coupling_matrix[ia, ib])


class CouplingAnalyzer:
    """Detect files that change together across version snapshots."""
    def analyze(self, file_histories: Dict[str, List[str]]) -> CouplingReport:
        file_names = sorted(file_histories.keys())
        n = len(file_names)
        if n < 2:
            return CouplingReport(file_names=file_names)
        num_versions = min(len(v) for v in file_histories.values())
        if num_versions < 2:
            return CouplingReport(file_names=file_names)
        change_vectors: Dict[str, List[bool]] = {}
        for fname in file_names:
            versions = file_histories[fname]
            vec: List[bool] = []
            for i in range(1, num_versions):
                h_prev = hashlib.sha256(versions[i - 1].encode()).hexdigest()
                h_cur = hashlib.sha256(versions[i].encode()).hexdigest()
                vec.append(h_prev != h_cur)
            change_vectors[fname] = vec
        matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i, n):
                vec_i = change_vectors[file_names[i]]
                vec_j = change_vectors[file_names[j]]
                co_changes = sum(a and b for a, b in zip(vec_i, vec_j))
                denom = min(sum(vec_i), sum(vec_j))
                score = co_changes / denom if denom > 0 else 0.0
                matrix[i, j] = score
                matrix[j, i] = score
        pairs: List[Tuple[str, str, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((file_names[i], file_names[j], float(matrix[i, j])))
        pairs.sort(key=lambda t: t[2], reverse=True)
        return CouplingReport(file_names=file_names, coupling_matrix=matrix, top_pairs=pairs)

# ---------------------------------------------------------------------------
# 9. Evolution report dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvolutionReport:
    """Aggregated evolution analysis results."""
    complexity_trajectory: Dict[str, List[int]] = field(default_factory=dict)
    churn_hotspots: List[Tuple[str, float]] = field(default_factory=list)
    debt_trajectory: List[int] = field(default_factory=list)
    stability_scores: Dict[str, float] = field(default_factory=dict)
    coupling_matrix: Optional[np.ndarray] = None
    bug_estimates: List[BugIntroduction] = field(default_factory=list)
    ownership: Dict[str, OwnershipRecord] = field(default_factory=dict)
    version_count: int = 0

    def summary_stats(self) -> Dict[str, Any]:
        avg_churn = float(np.mean([r for _, r in self.churn_hotspots])) if self.churn_hotspots else 0.0
        avg_stability = float(np.mean(list(self.stability_scores.values()))) if self.stability_scores else 0.0
        total_debt_last = self.debt_trajectory[-1] if self.debt_trajectory else 0
        return {
            "version_count": self.version_count,
            "tracked_functions": len(self.complexity_trajectory),
            "avg_churn_rate": round(avg_churn, 4),
            "avg_api_stability": round(avg_stability, 4),
            "total_debt_markers_latest": total_debt_last,
            "estimated_bugs": len(self.bug_estimates),
        }

    def risk_score(self) -> float:
        """Compute a composite risk score in [0, 1]."""
        weights = np.array([0.3, 0.25, 0.25, 0.2])
        churn_component = min(1.0, float(np.mean([r for _, r in self.churn_hotspots]))) if self.churn_hotspots else 0.0
        instability = 1.0 - (float(np.mean(list(self.stability_scores.values()))) if self.stability_scores else 1.0)
        debt_growth = 0.0
        if len(self.debt_trajectory) >= 2:
            diffs = np.diff(np.array(self.debt_trajectory, dtype=np.float64))
            debt_growth = min(1.0, max(0.0, float(np.mean(diffs)) / 5.0))
        bug_density = min(1.0, len(self.bug_estimates) / max(self.version_count, 1))
        components = np.array([churn_component, instability, debt_growth, bug_density])
        return float(np.clip(np.dot(weights, components), 0.0, 1.0))

# ---------------------------------------------------------------------------
# 8. Main orchestrator
# ---------------------------------------------------------------------------

class CodeEvolutionTracker:
    """Facade that runs all analyses on a file's version history.

    Usage::

        tracker = CodeEvolutionTracker()
        report = tracker.analyze(["v1 source ...", "v2 source ...", ...])
        print(report.summary_stats())
    """
    def __init__(self, spike_threshold: float = 1.5, lookahead: int = 3) -> None:
        self._complexity = ComplexityTracker()
        self._churn = ChurnAnalyzer()
        self._bugs = BugEstimator(spike_threshold=spike_threshold, lookahead=lookahead)
        self._ownership = OwnershipTracker()
        self._debt = DebtTracker()
        self._api = APIStabilityAnalyzer()
        self._coupling = CouplingAnalyzer()

    def analyze(self, file_history: List[str]) -> EvolutionReport:
        """Run the full analysis pipeline on *file_history*."""
        ct = self._complexity.track(file_history)
        churn = self._churn.analyze(file_history)
        bugs = self._bugs.estimate(file_history)
        own = self._ownership.track(file_history)
        debt = self._debt.track(file_history)
        api = self._api.analyze(file_history)
        return EvolutionReport(
            complexity_trajectory=ct.function_trajectories,
            churn_hotspots=churn.hotspots,
            debt_trajectory=debt.counts,
            stability_scores=api.scores,
            coupling_matrix=None,
            bug_estimates=bugs,
            ownership=own.ownership,
            version_count=len(file_history),
        )

    def analyze_multi(self, file_histories: Dict[str, List[str]]) -> Dict[str, EvolutionReport]:
        """Analyze multiple files and include coupling information."""
        coupling = self._coupling.analyze(file_histories)
        reports: Dict[str, EvolutionReport] = {}
        for fname, versions in file_histories.items():
            report = self.analyze(versions)
            report.coupling_matrix = coupling.coupling_matrix
            reports[fname] = report
        return reports

# ---------------------------------------------------------------------------
# 10. Additional helper utilities
# ---------------------------------------------------------------------------

def complexity_heatmap(versions: List[str]) -> np.ndarray:
    """Build a 2-D array of shape (n_functions, n_versions).

    Rows correspond to functions (sorted by name), columns to versions.
    Missing values (function absent in a version) are filled with 0.
    """
    all_func_names: Set[str] = set()
    parsed: List[Dict[str, ast.FunctionDef]] = []
    for source in versions:
        funcs = extract_functions(source)
        all_func_names |= set(funcs.keys())
        parsed.append(funcs)
    sorted_names = sorted(all_func_names)
    heatmap = np.zeros((len(sorted_names), len(versions)), dtype=np.int64)
    for vi, funcs in enumerate(parsed):
        for fi, name in enumerate(sorted_names):
            if name in funcs:
                heatmap[fi, vi] = compute_cyclomatic_complexity(funcs[name])
    return heatmap


def sliding_window_churn(versions: List[str], window: int = 3) -> List[Dict[str, int]]:
    """Compute churn counts in a sliding window over versions."""
    if len(versions) < 2:
        return []
    hashes_per_version = [compute_all_hashes(source) for source in versions]
    results: List[Dict[str, int]] = []
    for start in range(len(versions) - window + 1):
        window_slice = hashes_per_version[start: start + window]
        counts: Counter[str] = Counter()
        for i in range(1, len(window_slice)):
            for name in window_slice[i]:
                if name in window_slice[i - 1]:
                    if window_slice[i][name] != window_slice[i - 1][name]:
                        counts[name] += 1
                else:
                    counts[name] += 1
        results.append(dict(counts))
    return results


def aggregate_debt_by_category(versions: List[str]) -> Dict[str, np.ndarray]:
    """Return per-category debt marker counts as numpy arrays."""
    traj = DebtTracker().track(versions)
    return {marker: np.array(counts, dtype=np.int64) for marker, counts in traj.per_marker.items()}


def function_lifetime(versions: List[str]) -> Dict[str, Tuple[int, int]]:
    """Return {func_name: (first_version, last_version)}."""
    lifetimes: Dict[str, List[int]] = defaultdict(list)
    for idx, source in enumerate(versions):
        for name in extract_functions(source):
            lifetimes[name].append(idx)
    return {name: (indices[0], indices[-1]) for name, indices in lifetimes.items()}


def detect_signature_breaks(
    versions: List[str],
) -> Dict[str, List[Tuple[int, FunctionSignature, FunctionSignature]]]:
    """For each function, list (version, old_sig, new_sig) where signature changed."""
    breaks: Dict[str, List[Tuple[int, FunctionSignature, FunctionSignature]]] = defaultdict(list)
    prev_sigs: Dict[str, FunctionSignature] = {}
    for idx, source in enumerate(versions):
        funcs = extract_functions(source)
        cur_sigs = {n: extract_signature(f) for n, f in funcs.items()}
        if idx > 0:
            for name, sig in cur_sigs.items():
                if name in prev_sigs and sig != prev_sigs[name]:
                    breaks[name].append((idx, prev_sigs[name], sig))
        prev_sigs = cur_sigs
    return dict(breaks)
