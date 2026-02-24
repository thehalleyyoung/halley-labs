"""Comprehensive code metrics dashboard.

Computes LOC, complexity, maintainability, duplication, dependency,
and quality gate metrics for Python projects. Produces an executive
summary with health grades.
"""
from __future__ import annotations

import ast
import copy
import re
import textwrap
import math
import hashlib
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np

# ── Data classes ──────────────────────────────────────────────────────────
@dataclass
class LOCMetrics:
    total_lines: int = 0
    blank_lines: int = 0
    comment_lines: int = 0
    code_lines: int = 0
    def __add__(self, other: LOCMetrics) -> LOCMetrics:
        return LOCMetrics(self.total_lines + other.total_lines,
                          self.blank_lines + other.blank_lines,
                          self.comment_lines + other.comment_lines,
                          self.code_lines + other.code_lines)

@dataclass
class ComplexityMetrics:
    cyclomatic: int = 1
    cognitive: int = 0
    max_nesting: int = 0
    per_function: Dict[str, int] = field(default_factory=dict)
    distribution: Dict[str, int] = field(
        default_factory=lambda: {"low": 0, "medium": 0, "high": 0, "very_high": 0})

@dataclass
class DuplicationReport:
    duplicate_blocks: List[Tuple[int, int, int]] = field(default_factory=list)
    duplicate_line_count: int = 0
    total_lines: int = 0
    duplication_pct: float = 0.0

@dataclass
class CouplingMetrics:
    afferent: int = 0
    efferent: int = 0
    instability: float = 0.0
    abstractness: float = 0.0
    imports: List[str] = field(default_factory=list)
    abstract_classes: int = 0
    total_classes: int = 0

@dataclass
class QualityGateResult:
    passed: bool = True
    details: Dict[str, bool] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)

class TrendDirection(Enum):
    IMPROVED = auto()
    REGRESSED = auto()
    STABLE = auto()

@dataclass
class TrendItem:
    metric_name: str
    previous: float
    current: float
    direction: TrendDirection
    delta: float = 0.0

@dataclass
class TrendReport:
    items: List[TrendItem] = field(default_factory=list)
    @property
    def improved_count(self) -> int:
        return sum(1 for i in self.items if i.direction == TrendDirection.IMPROVED)
    @property
    def regressed_count(self) -> int:
        return sum(1 for i in self.items if i.direction == TrendDirection.REGRESSED)

@dataclass
class Summary:
    grade: str = "C"
    score: float = 50.0
    recommendations: List[str] = field(default_factory=list)
    key_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class Dashboard:
    loc_metrics: Dict[str, LOCMetrics] = field(default_factory=dict)
    complexity_metrics: Dict[str, ComplexityMetrics] = field(default_factory=dict)
    maintainability_scores: Dict[str, float] = field(default_factory=dict)
    duplication: Dict[str, DuplicationReport] = field(default_factory=dict)
    coupling: Dict[str, CouplingMetrics] = field(default_factory=dict)
    quality_gate_result: Optional[QualityGateResult] = None
    grade: str = "C"
    recommendations: List[str] = field(default_factory=list)

@dataclass
class QualityThresholds:
    max_complexity: int = 10
    max_duplication_pct: float = 5.0
    min_maintainability: float = 20.0
    max_file_loc: int = 500

# ── 1. LOC Counter ───────────────────────────────────────────────────────
class LOCCounter:
    _comment_re = re.compile(r"^\s*#")
    _blank_re = re.compile(r"^\s*$")

    def count(self, source_code: str) -> LOCMetrics:
        lines = source_code.splitlines()
        total = len(lines)
        blank = sum(1 for l in lines if self._blank_re.match(l))
        comment = sum(1 for l in lines if self._comment_re.match(l))
        in_docstring = False
        doc_lines = 0
        for line in lines:
            stripped = line.strip()
            if not in_docstring:
                for delim in ('"""', "'''"):
                    if stripped.startswith(delim):
                        if stripped.count(delim) >= 2 and len(stripped) > 3:
                            doc_lines += 1
                        else:
                            in_docstring = True
                            doc_lines += 1
                        break
            else:
                doc_lines += 1
                if '"""' in stripped or "'''" in stripped:
                    in_docstring = False
        code = total - blank - comment
        return LOCMetrics(total_lines=total, blank_lines=blank,
                          comment_lines=comment + doc_lines,
                          code_lines=max(0, code - doc_lines))

    def count_function(self, func_node: ast.FunctionDef) -> LOCMetrics:
        start = func_node.lineno
        end = getattr(func_node, "end_lineno", start)
        total = end - start + 1
        docstr_lines = 0
        for node in ast.walk(func_node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                docstr_lines += node.value.count("\n")
        return LOCMetrics(total_lines=total, blank_lines=0,
                          comment_lines=docstr_lines,
                          code_lines=max(1, total - docstr_lines))

# ── 2. Complexity Computer ───────────────────────────────────────────────
class ComplexityComputer:
    _branch_types = (ast.If, ast.For, ast.While, ast.ExceptHandler,
                     ast.With, ast.Assert)

    def compute(self, tree: ast.AST) -> ComplexityMetrics:
        per_func: Dict[str, int] = {}
        total_cc, total_cog, max_nest = 1, 0, 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cc = self._cyclomatic(node)
                cog = self._cognitive(node, 0)
                nest = self._max_nesting(node, 0)
                per_func[node.name] = cc
                total_cc += cc - 1
                total_cog += cog
                max_nest = max(max_nest, nest)
        if not per_func:
            total_cc = self._cyclomatic(tree)
            total_cog = self._cognitive(tree, 0)
            max_nest = self._max_nesting(tree, 0)
        return ComplexityMetrics(cyclomatic=total_cc, cognitive=total_cog,
                                max_nesting=max_nest, per_function=per_func,
                                distribution=self._distribution(per_func))

    def _cyclomatic(self, node: ast.AST) -> int:
        cc = 1
        for child in ast.walk(node):
            if isinstance(child, self._branch_types):
                cc += 1
            elif isinstance(child, ast.BoolOp):
                cc += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                cc += 1 + len(child.ifs)
        return cc

    def _cognitive(self, node: ast.AST, depth: int) -> int:
        score = 0
        for child in ast.iter_child_nodes(node):
            inc, nest = 0, 0
            if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                inc, nest = 1, depth
            elif isinstance(child, ast.BoolOp):
                inc = 1
            elif isinstance(child, (ast.Break, ast.Continue)):
                inc = 1
            score += inc + nest
            new_depth = depth + 1 if isinstance(
                child, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With)
            ) else depth
            score += self._cognitive(child, new_depth)
        return score

    def _max_nesting(self, node: ast.AST, cur: int) -> int:
        best = cur
        for child in ast.iter_child_nodes(node):
            deeper = cur + 1 if isinstance(
                child, (ast.If, ast.For, ast.While, ast.With, ast.ExceptHandler)
            ) else cur
            best = max(best, self._max_nesting(child, deeper))
        return best

    @staticmethod
    def _distribution(pf: Dict[str, int]) -> Dict[str, int]:
        d = {"low": 0, "medium": 0, "high": 0, "very_high": 0}
        for cc in pf.values():
            if cc <= 5: d["low"] += 1
            elif cc <= 10: d["medium"] += 1
            elif cc <= 20: d["high"] += 1
            else: d["very_high"] += 1
        return d

# ── 3. Maintainability Computer ──────────────────────────────────────────
class MaintainabilityComputer:
    def compute(self, source_code: str, tree: ast.AST) -> float:
        volume = self._halstead_volume(tree)
        cc = ComplexityComputer().compute(tree).cyclomatic
        loc = max(LOCCounter().count(source_code).code_lines, 1)
        mi = 171.0 - 5.2 * math.log(max(volume, 1)) - 0.23 * cc - 16.2 * math.log(loc)
        return float(np.clip(mi, 0.0, 100.0))

    def _halstead_volume(self, tree: ast.AST) -> float:
        operators: Counter = Counter()
        operands: Counter = Counter()
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                operators[type(node.op).__name__] += 1
            elif isinstance(node, ast.UnaryOp):
                operators[type(node.op).__name__] += 1
            elif isinstance(node, ast.BoolOp):
                operators[type(node.op).__name__] += len(node.values) - 1
            elif isinstance(node, ast.Compare):
                for op in node.ops:
                    operators[type(op).__name__] += 1
            elif isinstance(node, ast.Assign):
                operators["Assign"] += 1
            elif isinstance(node, ast.AugAssign):
                operators["AugAssign"] += 1
            elif isinstance(node, ast.Return):
                operators["Return"] += 1
            elif isinstance(node, ast.Call):
                operators["Call"] += 1
            elif isinstance(node, ast.Constant):
                operands[repr(node.value)] += 1
            elif isinstance(node, ast.Name):
                operands[node.id] += 1
            elif isinstance(node, ast.Attribute):
                operands[node.attr] += 1
        n1, n2 = sum(operators.values()), sum(operands.values())
        eta = len(operators) + len(operands)
        n = n1 + n2
        if eta <= 0 or n <= 0:
            return 1.0
        return n * math.log2(max(eta, 2))

# ── 4. Duplication Detector ──────────────────────────────────────────────
class DuplicationDetector:
    @staticmethod
    def _normalize(line: str) -> str:
        return re.sub(r"\s+", " ", line.strip())

    def detect(self, source_code: str, min_block: int = 4) -> DuplicationReport:
        raw_lines = source_code.splitlines()
        total = len(raw_lines)
        if total < min_block:
            return DuplicationReport(total_lines=total)
        normed = [self._normalize(l) for l in raw_lines]
        hashes: Dict[str, List[int]] = defaultdict(list)
        for i in range(total - min_block + 1):
            block = "\n".join(normed[i:i + min_block])
            h = hashlib.md5(block.encode("utf-8")).hexdigest()
            hashes[h].append(i)
        dup_lines: Set[int] = set()
        blocks: List[Tuple[int, int, int]] = []
        for h, positions in hashes.items():
            if len(positions) < 2:
                continue
            for idx in range(1, len(positions)):
                start_a, start_b = positions[0], positions[idx]
                length = min_block
                while (start_a + length < total and start_b + length < total
                       and start_a + length < start_b
                       and normed[start_a + length] == normed[start_b + length]):
                    length += 1
                blocks.append((start_a, start_b, length))
                for off in range(length):
                    dup_lines.add(start_b + off)
        dup_count = len(dup_lines)
        pct = (dup_count / total * 100.0) if total > 0 else 0.0
        return DuplicationReport(duplicate_blocks=blocks,
                                 duplicate_line_count=dup_count,
                                 total_lines=total, duplication_pct=round(pct, 2))

# ── 5. Dependency Analyzer ───────────────────────────────────────────────
class DependencyAnalyzer:
    def compute(self, tree: ast.AST) -> CouplingMetrics:
        imports = self._collect_imports(tree)
        ce = len(imports)
        abstract_count, total_classes = self._count_abstractions(tree)
        abstractness = (abstract_count / total_classes) if total_classes > 0 else 0.0
        instability = 1.0 if ce > 0 else 0.0
        return CouplingMetrics(afferent=0, efferent=ce,
                               instability=round(instability, 4),
                               abstractness=round(abstractness, 4),
                               imports=imports, abstract_classes=abstract_count,
                               total_classes=total_classes)

    @staticmethod
    def _collect_imports(tree: ast.AST) -> List[str]:
        imports: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for alias in node.names:
                    imports.append(f"{mod}.{alias.name}")
        return imports

    @staticmethod
    def _count_abstractions(tree: ast.AST) -> Tuple[int, int]:
        total, abstract = 0, 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            total += 1
            bases = [getattr(b, "id", getattr(b, "attr", "")) for b in node.bases]
            if "ABC" in bases or "ABCMeta" in bases:
                abstract += 1
                continue
            for item in ast.walk(node):
                if isinstance(item, ast.FunctionDef):
                    for dec in item.decorator_list:
                        nm = getattr(dec, "id", getattr(dec, "attr", ""))
                        if nm == "abstractmethod":
                            abstract += 1
                            break
        return abstract, total

    def compute_cross_file(self, trees: Dict[str, ast.AST]) -> Dict[str, CouplingMetrics]:
        per_file: Dict[str, CouplingMetrics] = {}
        all_imports: Dict[str, List[str]] = {}
        for fname, tree in trees.items():
            cm = self.compute(tree)
            per_file[fname] = cm
            all_imports[fname] = cm.imports
        module_names: Dict[str, str] = {}
        for fname in trees:
            mod = fname.replace("/", ".").replace("\\", ".")
            if mod.endswith(".py"):
                mod = mod[:-3]
            module_names[fname] = mod
        for fname in trees:
            ca = 0
            my_mod = module_names[fname]
            for other, imp_list in all_imports.items():
                if other == fname:
                    continue
                for imp in imp_list:
                    if imp.startswith(my_mod) or my_mod.endswith(imp):
                        ca += 1
                        break
            per_file[fname].afferent = ca
            tot = ca + per_file[fname].efferent
            per_file[fname].instability = (
                round(per_file[fname].efferent / tot, 4) if tot > 0 else 1.0)
        return per_file

# ── 6. Quality Gate ──────────────────────────────────────────────────────
class QualityGate:
    def __init__(self, thresholds: Optional[QualityThresholds] = None) -> None:
        self.thresholds = thresholds or QualityThresholds()

    def check(self, dashboard: Dashboard) -> QualityGateResult:
        details: Dict[str, bool] = {}
        messages: List[str] = []
        t = self.thresholds
        max_cc = max((c.cyclomatic for c in dashboard.complexity_metrics.values()), default=0)
        ok = max_cc <= t.max_complexity
        details["complexity"] = ok
        if not ok:
            messages.append(f"Complexity {max_cc} exceeds threshold {t.max_complexity}")
        max_dup = max((d.duplication_pct for d in dashboard.duplication.values()), default=0.0)
        ok = max_dup <= t.max_duplication_pct
        details["duplication"] = ok
        if not ok:
            messages.append(f"Duplication {max_dup:.1f}% exceeds {t.max_duplication_pct}%")
        mi_vals = list(dashboard.maintainability_scores.values())
        min_mi = min(mi_vals) if mi_vals else 0.0
        ok = min_mi >= t.min_maintainability
        details["maintainability"] = ok
        if not ok:
            messages.append(f"Maintainability {min_mi:.1f} below {t.min_maintainability}")
        max_loc = max((l.total_lines for l in dashboard.loc_metrics.values()), default=0)
        ok = max_loc <= t.max_file_loc
        details["file_loc"] = ok
        if not ok:
            messages.append(f"File LOC {max_loc} exceeds threshold {t.max_file_loc}")
        return QualityGateResult(passed=all(details.values()),
                                 details=details, messages=messages)

# ── 7. Trend Analyzer ───────────────────────────────────────────────────
class TrendAnalyzer:
    _lower_better = {"complexity", "duplication_pct", "loc"}
    _higher_better = {"maintainability"}

    def compare(self, current: Dict[str, float],
                baseline: Dict[str, float]) -> TrendReport:
        items: List[TrendItem] = []
        for key in sorted(set(current) | set(baseline)):
            cur, prev = current.get(key, 0.0), baseline.get(key, 0.0)
            delta = cur - prev
            items.append(TrendItem(metric_name=key, previous=prev, current=cur,
                                   direction=self._classify(key, delta),
                                   delta=round(delta, 4)))
        return TrendReport(items=items)

    def _classify(self, metric: str, delta: float) -> TrendDirection:
        if abs(delta) < 1e-6:
            return TrendDirection.STABLE
        if metric in self._lower_better:
            return TrendDirection.IMPROVED if delta < 0 else TrendDirection.REGRESSED
        if metric in self._higher_better:
            return TrendDirection.IMPROVED if delta > 0 else TrendDirection.REGRESSED
        return TrendDirection.STABLE

    @staticmethod
    def extract_metrics(dashboard: Dashboard) -> Dict[str, float]:
        result: Dict[str, float] = {}
        if dashboard.loc_metrics:
            result["loc"] = float(sum(l.total_lines for l in dashboard.loc_metrics.values()))
        if dashboard.complexity_metrics:
            result["complexity"] = float(max(
                c.cyclomatic for c in dashboard.complexity_metrics.values()))
        if dashboard.maintainability_scores:
            result["maintainability"] = round(float(
                np.mean(list(dashboard.maintainability_scores.values()))), 2)
        if dashboard.duplication:
            result["duplication_pct"] = max(
                d.duplication_pct for d in dashboard.duplication.values())
        return result

# ── 8. Executive Summary ─────────────────────────────────────────────────
class ExecutiveSummary:
    _grade_thresholds = [(90, "A"), (75, "B"), (55, "C"), (35, "D"), (0, "F")]
    _weights = {"maintainability": 0.35, "complexity": 0.25,
                "duplication": 0.20, "loc_efficiency": 0.20}

    def summarize(self, dashboard: Dashboard) -> Summary:
        scores: Dict[str, float] = {}
        if dashboard.maintainability_scores:
            scores["maintainability"] = float(
                np.mean(list(dashboard.maintainability_scores.values())))
        else:
            scores["maintainability"] = 50.0
        if dashboard.complexity_metrics:
            max_cc = max(c.cyclomatic for c in dashboard.complexity_metrics.values())
            scores["complexity"] = max(0.0, 100.0 - max_cc * 5.0)
        else:
            scores["complexity"] = 80.0
        if dashboard.duplication:
            max_dup = max(d.duplication_pct for d in dashboard.duplication.values())
            scores["duplication"] = max(0.0, 100.0 - max_dup * 5.0)
        else:
            scores["duplication"] = 100.0
        if dashboard.loc_metrics:
            max_loc = max(l.total_lines for l in dashboard.loc_metrics.values())
            scores["loc_efficiency"] = max(0.0, 100.0 - max(0, max_loc - 200) * 0.2)
        else:
            scores["loc_efficiency"] = 80.0
        weighted = sum(scores.get(k, 50.0) * w for k, w in self._weights.items())
        grade = self._to_grade(weighted)
        recs = self._build_recommendations(dashboard, scores)
        return Summary(grade=grade, score=round(weighted, 2),
                       recommendations=recs[:5], key_metrics=scores)

    def _to_grade(self, score: float) -> str:
        for threshold, letter in self._grade_thresholds:
            if score >= threshold:
                return letter
        return "F"

    @staticmethod
    def _build_recommendations(dashboard: Dashboard,
                                scores: Dict[str, float]) -> List[str]:
        recs: List[str] = []
        if scores.get("complexity", 100) < 50:
            recs.append("Refactor high-complexity functions to reduce cyclomatic complexity.")
        if scores.get("duplication", 100) < 60:
            recs.append("Extract duplicated code blocks into shared utility functions.")
        if scores.get("maintainability", 100) < 40:
            recs.append("Improve maintainability by reducing function length and complexity.")
        if scores.get("loc_efficiency", 100) < 60:
            recs.append("Consider splitting large files into smaller, focused modules.")
        for fname, cm in dashboard.complexity_metrics.items():
            if cm.max_nesting > 4:
                recs.append(f"{fname}: reduce nesting depth (currently {cm.max_nesting}).")
        for fname, dr in dashboard.duplication.items():
            if dr.duplication_pct > 10:
                recs.append(f"{fname}: {dr.duplication_pct:.0f}% duplication — deduplicate.")
        for fname, lm in dashboard.loc_metrics.items():
            if lm.code_lines > 400:
                recs.append(f"{fname}: {lm.code_lines} code lines — consider splitting.")
        if not recs:
            recs.append("Code quality is good. Maintain current practices.")
        return recs

# ── 9. MetricDashboard (orchestrator) ────────────────────────────────────
class MetricDashboard:
    def __init__(self, thresholds: Optional[QualityThresholds] = None,
                 min_dup_block: int = 4) -> None:
        self._loc = LOCCounter()
        self._complexity = ComplexityComputer()
        self._maintainability = MaintainabilityComputer()
        self._duplication = DuplicationDetector()
        self._dependency = DependencyAnalyzer()
        self._gate = QualityGate(thresholds)
        self._summary_gen = ExecutiveSummary()
        self._min_dup_block = min_dup_block

    def compute(self, source_files: Dict[str, str]) -> Dashboard:
        db = Dashboard()
        trees: Dict[str, ast.AST] = {}
        for fname, src in source_files.items():
            try:
                tree = ast.parse(src, filename=fname)
            except SyntaxError:
                continue
            trees[fname] = tree
            db.loc_metrics[fname] = self._loc.count(src)
            db.complexity_metrics[fname] = self._complexity.compute(tree)
            db.maintainability_scores[fname] = self._maintainability.compute(src, tree)
            db.duplication[fname] = self._duplication.detect(src, self._min_dup_block)
        db.coupling = self._dependency.compute_cross_file(trees)
        db.quality_gate_result = self._gate.check(db)
        summary = self._summary_gen.summarize(db)
        db.grade = summary.grade
        db.recommendations = summary.recommendations
        return db

    def compute_with_trend(self, source_files: Dict[str, str],
                           baseline: Optional[Dict[str, float]] = None,
                           ) -> Tuple[Dashboard, Optional[TrendReport]]:
        dashboard = self.compute(source_files)
        trend: Optional[TrendReport] = None
        if baseline is not None:
            analyzer = TrendAnalyzer()
            trend = analyzer.compare(analyzer.extract_metrics(dashboard), baseline)
        return dashboard, trend

# ── Utility helpers ──────────────────────────────────────────────────────
def format_dashboard_text(dashboard: Dashboard) -> str:
    parts: List[str] = []
    parts.append("=" * 60)
    parts.append("  CODE METRICS DASHBOARD")
    parts.append("=" * 60)
    parts.append(f"\nOverall Grade: {dashboard.grade}")
    if dashboard.quality_gate_result:
        status = "PASSED" if dashboard.quality_gate_result.passed else "FAILED"
        parts.append(f"Quality Gate:  {status}")
        for msg in dashboard.quality_gate_result.messages:
            parts.append(f"  - {msg}")
    parts.append("\n--- LOC Metrics ---")
    for fname, lm in dashboard.loc_metrics.items():
        parts.append(f"  {fname}: total={lm.total_lines}  code={lm.code_lines}  "
                     f"comment={lm.comment_lines}  blank={lm.blank_lines}")
    parts.append("\n--- Complexity ---")
    for fname, cm in dashboard.complexity_metrics.items():
        parts.append(f"  {fname}: CC={cm.cyclomatic}  cognitive={cm.cognitive}  "
                     f"max_nest={cm.max_nesting}")
        for fn, cc in cm.per_function.items():
            parts.append(f"    {fn}: CC={cc}")
    parts.append("\n--- Maintainability ---")
    for fname, mi in dashboard.maintainability_scores.items():
        parts.append(f"  {fname}: MI={mi:.1f}")
    parts.append("\n--- Duplication ---")
    for fname, dr in dashboard.duplication.items():
        parts.append(f"  {fname}: {dr.duplication_pct:.1f}% "
                     f"({dr.duplicate_line_count}/{dr.total_lines} lines)")
    parts.append("\n--- Coupling ---")
    for fname, cp in dashboard.coupling.items():
        parts.append(f"  {fname}: Ca={cp.afferent}  Ce={cp.efferent}  "
                     f"I={cp.instability:.2f}  A={cp.abstractness:.2f}")
    parts.append("\n--- Recommendations ---")
    for i, rec in enumerate(dashboard.recommendations, 1):
        parts.append(f"  {i}. {rec}")
    parts.append("=" * 60)
    return "\n".join(parts)

def format_trend_text(report: TrendReport) -> str:
    parts: List[str] = ["--- Trend Analysis ---"]
    for item in report.items:
        arrow = {TrendDirection.IMPROVED: "↑ improved",
                 TrendDirection.REGRESSED: "↓ regressed",
                 TrendDirection.STABLE: "— stable"}[item.direction]
        parts.append(f"  {item.metric_name}: {item.previous:.2f} → {item.current:.2f}  "
                     f"(Δ {item.delta:+.2f})  {arrow}")
    parts.append(f"  Summary: {report.improved_count} improved, "
                 f"{report.regressed_count} regressed")
    return "\n".join(parts)

def aggregate_loc(metrics: Dict[str, LOCMetrics]) -> LOCMetrics:
    total = LOCMetrics()
    for m in metrics.values():
        total = total + m
    return total

def compute_halstead_for_source(source_code: str) -> float:
    tree = ast.parse(source_code)
    return MaintainabilityComputer()._halstead_volume(tree)

def batch_compute(file_sources: Dict[str, str],
                  thresholds: Optional[QualityThresholds] = None,
                  ) -> Tuple[Dashboard, str]:
    md = MetricDashboard(thresholds=thresholds)
    db = md.compute(file_sources)
    return db, format_dashboard_text(db)
