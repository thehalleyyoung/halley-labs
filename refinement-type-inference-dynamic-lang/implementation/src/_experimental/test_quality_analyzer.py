"""Test quality analyzer for Python test suites.

Analyzes test code to assess quality through coverage estimation,
assertion analysis, test independence checking, smell detection,
mock analysis, and mutation testing simulation.
"""
from __future__ import annotations
import ast
import copy
import re
import textwrap
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union
import numpy as np


class SmellKind(Enum):
    EMPTY_TEST = auto()
    NO_ASSERTIONS = auto()
    COMMENTED_OUT_CODE = auto()
    MAGIC_NUMBER = auto()
    TOO_LONG = auto()
    CATCH_ALL_EXCEPTION = auto()
    DUPLICATE_LOGIC = auto()
    SLEEP_CALL = auto()


class MockIssueKind(Enum):
    OVER_MOCKING = auto()
    MOCK_NOT_ASSERTED = auto()
    MOCK_LEAKAGE = auto()
    UNTESTED_MOCK_CONFIG = auto()


@dataclass
class CoverageEstimate:
    total_source_functions: int
    covered_functions: List[str]
    uncovered_functions: List[str]
    coverage_ratio: float


@dataclass
class AssertionReport:
    total_assertions: int
    assertions_by_type: Dict[str, int]
    assertion_density: float
    per_test: Dict[str, int]


@dataclass
class IndependenceIssue:
    test_name: str
    kind: str
    detail: str
    line: int


@dataclass
class IndependenceReport:
    issues: List[IndependenceIssue]
    score: float


@dataclass
class TestSmell:
    kind: SmellKind
    test_name: str
    line: int
    message: str


@dataclass
class MockIssue:
    kind: MockIssueKind
    test_name: str
    line: int
    message: str


@dataclass
class MockReport:
    issues: List[MockIssue]
    total_mocks: int
    quality_score: float


@dataclass
class ParameterizeOpportunity:
    test_names: List[str]
    shared_structure_hash: str
    differing_literals: List[Any]
    suggested_params: List[str]


@dataclass
class MutationReport:
    total_mutations: int
    killed_mutations: int
    survived_mutations: int
    mutation_score: float


@dataclass
class TestQualityReport:
    coverage_estimate: Optional[CoverageEstimate]
    assertion_density: float
    smells: List[TestSmell]
    independence_score: float
    mock_quality_score: float
    parameterize_opportunities: List[ParameterizeOpportunity]
    mutation_score_estimate: float
    recommendations: List[str]
    overall_grade: str


_ASSERTION_METHODS = frozenset({
    "assertEqual", "assertNotEqual", "assertTrue", "assertFalse",
    "assertIs", "assertIsNot", "assertIsNone", "assertIsNotNone",
    "assertIn", "assertNotIn", "assertRaises", "assertWarns",
    "assertAlmostEqual", "assertNotAlmostEqual", "assertGreater",
    "assertGreaterEqual", "assertLess", "assertLessEqual",
    "assertRegex", "assertNotRegex", "assertCountEqual",
    "assertMultiLineEqual", "assertSequenceEqual", "assertListEqual",
    "assertTupleEqual", "assertSetEqual", "assertDictEqual",
})


def _get_test_functions(tree: ast.Module) -> List[ast.FunctionDef]:
    return [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name.startswith("test_")]


def _count_statements(body: List[ast.stmt]) -> int:
    total = 0
    for stmt in body:
        total += 1
        for child in ast.walk(stmt):
            if isinstance(child, ast.stmt) and child is not stmt:
                total += 1
    return total


def _normalize_body(func: ast.FunctionDef) -> str:
    class _R(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.Constant:
            if isinstance(node.value, (int, float)):
                return ast.Constant(value=0)
            if isinstance(node.value, str):
                return ast.Constant(value="")
            return node
    mod = ast.Module(body=copy.deepcopy(func.body), type_ignores=[])
    return ast.dump(_R().visit(mod))


def _collect_literals(func: ast.FunctionDef) -> List[Any]:
    return [n.value for n in ast.walk(func)
            if isinstance(n, ast.Constant) and n.value is not None]


class CoverageEstimator:
    """Match test functions to source functions by naming convention."""

    def estimate(self, source_code: str, test_code: str) -> CoverageEstimate:
        src_tree, test_tree = ast.parse(source_code), ast.parse(test_code)
        src_funcs = self._source_names(src_tree)
        test_targets = self._test_targets(test_tree)
        covered = [f for f in src_funcs if f in test_targets]
        uncovered = [f for f in src_funcs if f not in test_targets]
        return CoverageEstimate(len(src_funcs), covered, uncovered,
                                round(len(covered) / max(len(src_funcs), 1), 4))

    @staticmethod
    def _source_names(tree: ast.Module) -> List[str]:
        return [n.name for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not n.name.startswith("test_")
                and (not n.name.startswith("_") or n.name.startswith("__"))]

    @staticmethod
    def _test_targets(tree: ast.Module) -> Set[str]:
        targets: Set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_"):
                target = n.name[5:]
                targets.add(target)
                parts = target.split("_")
                for i in range(1, len(parts)):
                    targets.add("_".join(parts[:i]))
        return targets


class AssertionAnalyzer:
    """Count and classify assertions per test."""

    def analyze(self, test_tree: ast.Module) -> AssertionReport:
        tests = _get_test_functions(test_tree)
        by_type: Counter = Counter()
        per_test: Dict[str, int] = {}
        total = 0
        for func in tests:
            count = 0
            for node in ast.walk(func):
                kind = self._classify(node)
                if kind:
                    by_type[kind] += 1
                    count += 1
                    total += 1
            per_test[func.name] = count
        density = total / max(len(tests), 1)
        return AssertionReport(total, dict(by_type), round(density, 3), per_test)

    @staticmethod
    def _classify(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Assert):
            return "assert_statement"
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            c = node.value
            if isinstance(c.func, ast.Attribute) and c.func.attr in _ASSERTION_METHODS:
                return c.func.attr
        if isinstance(node, ast.With):
            for item in node.items:
                ctx = item.context_expr
                if (isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Attribute)
                        and ctx.func.attr == "raises"):
                    return "pytest.raises"
        return None


class IndependenceChecker:
    """Detect tests that depend on shared mutable state."""
    _GLOBAL_MUT = frozenset({"os.environ", "os.putenv", "os.chdir",
                             "sys.path.append", "sys.path.insert"})

    def check(self, test_tree: ast.Module) -> IndependenceReport:
        issues: List[IndependenceIssue] = []
        tests = _get_test_functions(test_tree)
        for func in tests:
            self._check_global_kw(func, issues)
            self._check_class_state(func, test_tree, issues)
            self._check_global_calls(func, issues)
        self._check_order(tests, issues)
        affected = len({i.test_name for i in issues})
        score = 1.0 - affected / max(len(tests), 1)
        return IndependenceReport(issues, round(max(score, 0.0), 3))

    @staticmethod
    def _check_global_kw(func: ast.FunctionDef, out: List[IndependenceIssue]) -> None:
        for node in ast.walk(func):
            if isinstance(node, ast.Global):
                out.append(IndependenceIssue(func.name, "global_keyword",
                           f"Uses 'global {', '.join(node.names)}'", node.lineno))

    @staticmethod
    def _check_class_state(func: ast.FunctionDef, tree: ast.Module,
                           out: List[IndependenceIssue]) -> None:
        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            if func not in ast.walk(cls):
                continue
            cls_attrs = {t.id for s in cls.body if isinstance(s, ast.Assign)
                         for t in s.targets if isinstance(t, ast.Name)}
            for node in ast.walk(func):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                                and t.value.id == "self" and t.attr in cls_attrs):
                            out.append(IndependenceIssue(
                                func.name, "class_state_mutation",
                                f"Mutates class attr 'self.{t.attr}'", node.lineno))

    def _check_global_calls(self, func: ast.FunctionDef,
                            out: List[IndependenceIssue]) -> None:
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                src = ast.dump(node.func)
                for pat in self._GLOBAL_MUT:
                    if pat.replace(".", "") in src.replace("'", ""):
                        out.append(IndependenceIssue(func.name, "global_side_effect",
                                   f"Calls {pat}", node.lineno))

    @staticmethod
    def _check_order(tests: List[ast.FunctionDef],
                     out: List[IndependenceIssue]) -> None:
        written: Dict[str, str] = {}
        for func in tests:
            for node in ast.walk(func):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            written[t.id] = func.name
            for node in ast.walk(func):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    writer = written.get(node.id)
                    if writer and writer != func.name:
                        out.append(IndependenceIssue(
                            func.name, "order_dependence",
                            f"Reads '{node.id}' written by '{writer}'", node.lineno))


class SmellDetector:
    """Detect common test smells."""
    _MAX_STMTS = 50

    def detect(self, test_tree: ast.Module) -> List[TestSmell]:
        smells: List[TestSmell] = []
        tests = _get_test_functions(test_tree)
        for f in tests:
            smells.extend(self._empty(f))
            smells.extend(self._no_assert(f))
            smells.extend(self._too_long(f))
            smells.extend(self._magic(f))
            smells.extend(self._catch_all(f))
            smells.extend(self._sleep(f))
        smells.extend(self._duplicate(tests))
        smells.extend(self._commented(test_tree))
        return smells

    @staticmethod
    def _empty(f: ast.FunctionDef) -> List[TestSmell]:
        if not f.body:
            return [TestSmell(SmellKind.EMPTY_TEST, f.name, f.lineno, "Empty test")]
        if len(f.body) == 1:
            s = f.body[0]
            if isinstance(s, ast.Pass):
                return [TestSmell(SmellKind.EMPTY_TEST, f.name, f.lineno, "Only pass")]
            if (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                    and isinstance(s.value.value, str)):
                return [TestSmell(SmellKind.EMPTY_TEST, f.name, f.lineno, "Only docstring")]
        return []

    @staticmethod
    def _no_assert(f: ast.FunctionDef) -> List[TestSmell]:
        for n in ast.walk(f):
            if isinstance(n, ast.Assert):
                return []
            if (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Attribute)
                    and n.value.func.attr in _ASSERTION_METHODS):
                return []
        return [TestSmell(SmellKind.NO_ASSERTIONS, f.name, f.lineno, "No assertions")]

    def _too_long(self, f: ast.FunctionDef) -> List[TestSmell]:
        c = _count_statements(f.body)
        if c > self._MAX_STMTS:
            return [TestSmell(SmellKind.TOO_LONG, f.name, f.lineno,
                              f"{c} statements (>{self._MAX_STMTS})")]
        return []

    @staticmethod
    def _magic(f: ast.FunctionDef) -> List[TestSmell]:
        trivial = {0, 1, -1, 2, 0.0, 1.0, True, False}
        out: List[TestSmell] = []
        for n in ast.walk(f):
            if (isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
                    and n.value not in trivial):
                out.append(TestSmell(SmellKind.MAGIC_NUMBER, f.name,
                           getattr(n, "lineno", f.lineno), f"Magic number {n.value}"))
        return out

    @staticmethod
    def _catch_all(f: ast.FunctionDef) -> List[TestSmell]:
        out: List[TestSmell] = []
        for n in ast.walk(f):
            if isinstance(n, ast.ExceptHandler):
                if n.type is None:
                    out.append(TestSmell(SmellKind.CATCH_ALL_EXCEPTION, f.name,
                               n.lineno, "Bare except"))
                elif isinstance(n.type, ast.Name) and n.type.id in ("Exception", "BaseException"):
                    out.append(TestSmell(SmellKind.CATCH_ALL_EXCEPTION, f.name,
                               n.lineno, f"Catches {n.type.id}"))
        return out

    @staticmethod
    def _sleep(f: ast.FunctionDef) -> List[TestSmell]:
        out: List[TestSmell] = []
        for n in ast.walk(f):
            if isinstance(n, ast.Call):
                is_sleep = ((isinstance(n.func, ast.Attribute) and n.func.attr == "sleep")
                            or (isinstance(n.func, ast.Name) and n.func.id == "sleep"))
                if is_sleep:
                    out.append(TestSmell(SmellKind.SLEEP_CALL, f.name, n.lineno,
                               "Calls sleep()"))
        return out

    @staticmethod
    def _duplicate(tests: List[ast.FunctionDef]) -> List[TestSmell]:
        groups: Dict[str, List[str]] = defaultdict(list)
        for f in tests:
            groups[_normalize_body(f)].append(f.name)
        out: List[TestSmell] = []
        for names in groups.values():
            if len(names) > 1:
                for nm in names:
                    others = [x for x in names if x != nm]
                    out.append(TestSmell(SmellKind.DUPLICATE_LOGIC, nm, 0,
                               f"Duplicate with {others}"))
        return out

    @staticmethod
    def _commented(tree: ast.Module) -> List[TestSmell]:
        out: List[TestSmell] = []
        try:
            src = ast.unparse(tree)
        except Exception:
            return out
        for i, line in enumerate(src.splitlines(), 1):
            if line.strip().startswith("#") and "def test_" in line:
                out.append(TestSmell(SmellKind.COMMENTED_OUT_CODE, "<module>", i,
                           "Commented-out test"))
        return out


class MockAnalyzer:
    """Detect mock usage patterns and issues."""
    _MOCK_ASSERTS = frozenset({
        "assert_called", "assert_called_once", "assert_called_with",
        "assert_called_once_with", "assert_any_call",
        "assert_not_called", "assert_has_calls",
    })

    def analyze(self, test_tree: ast.Module) -> MockReport:
        issues: List[MockIssue] = []
        total = 0
        for func in _get_test_functions(test_tree):
            mocks = self._find_mocks(func)
            total += len(mocks)
            issues.extend(self._over_mocking(func, mocks))
            issues.extend(self._not_asserted(func, mocks))
            issues.extend(self._leakage(func))
            issues.extend(self._untested_cfg(func, mocks))
        q = 1.0 - len(issues) / max(total, 1)
        return MockReport(issues, total, round(max(q, 0.0), 3))

    @staticmethod
    def _find_mocks(func: ast.FunctionDef) -> Set[str]:
        names: Set[str] = set()
        for n in ast.walk(func):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
                fn = ""
                if isinstance(n.value.func, ast.Name):
                    fn = n.value.func.id
                elif isinstance(n.value.func, ast.Attribute):
                    fn = n.value.func.attr
                if fn in ("Mock", "MagicMock", "patch", "create_autospec"):
                    for t in n.targets:
                        if isinstance(t, ast.Name):
                            names.add(t.id)
        for arg in func.args.args:
            if "mock" in arg.arg.lower():
                names.add(arg.arg)
        return names

    @staticmethod
    def _over_mocking(func: ast.FunctionDef, mocks: Set[str]) -> List[MockIssue]:
        if not mocks:
            return []
        real, mock_c = 0, 0
        for n in ast.walk(func):
            if not isinstance(n, ast.Call):
                continue
            if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name):
                if n.func.value.id in mocks:
                    mock_c += 1
                else:
                    real += 1
            elif isinstance(n.func, ast.Name) and n.func.id not in mocks:
                real += 1
        if mock_c > real and mock_c > 2:
            return [MockIssue(MockIssueKind.OVER_MOCKING, func.name, func.lineno,
                    f"Mock calls ({mock_c}) > real ({real})")]
        return []

    def _not_asserted(self, func: ast.FunctionDef, mocks: Set[str]) -> List[MockIssue]:
        asserted: Set[str] = set()
        for n in ast.walk(func):
            if (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Attribute)
                    and n.value.func.attr in self._MOCK_ASSERTS
                    and isinstance(n.value.func.value, ast.Name)):
                asserted.add(n.value.func.value.id)
        return [MockIssue(MockIssueKind.MOCK_NOT_ASSERTED, func.name, func.lineno,
                f"Mock '{m}' never asserted") for m in mocks - asserted]

    @staticmethod
    def _leakage(func: ast.FunctionDef) -> List[MockIssue]:
        out: List[MockIssue] = []
        for n in ast.walk(func):
            if not (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)):
                continue
            fn = ""
            if isinstance(n.value.func, ast.Name):
                fn = n.value.func.id
            elif isinstance(n.value.func, ast.Attribute):
                fn = n.value.func.attr
            if fn != "patch":
                continue
            for t in n.targets:
                if not isinstance(t, ast.Name):
                    continue
                has_start = has_stop = False
                for n2 in ast.walk(func):
                    if (isinstance(n2, ast.Call) and isinstance(n2.func, ast.Attribute)
                            and isinstance(n2.func.value, ast.Name)
                            and n2.func.value.id == t.id):
                        has_start = has_start or n2.func.attr == "start"
                        has_stop = has_stop or n2.func.attr == "stop"
                if has_start and not has_stop:
                    out.append(MockIssue(MockIssueKind.MOCK_LEAKAGE, func.name,
                               n.lineno, f"patch '{t.id}' started but never stopped"))
        return out

    @staticmethod
    def _untested_cfg(func: ast.FunctionDef, mocks: Set[str]) -> List[MockIssue]:
        configured: Set[str] = set()
        triggered: Set[str] = set()
        for n in ast.walk(func):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if (isinstance(t, ast.Attribute)
                            and t.attr in ("return_value", "side_effect")
                            and isinstance(t.value, ast.Name)
                            and t.value.id in mocks):
                        configured.add(t.value.id)
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id in mocks):
                triggered.add(n.func.id)
        return [MockIssue(MockIssueKind.UNTESTED_MOCK_CONFIG, func.name, func.lineno,
                f"Mock '{m}' configured but never called") for m in configured - triggered]


class ParameterizeDetector:
    """Find tests that could benefit from parameterization."""

    def detect(self, test_tree: ast.Module) -> List[ParameterizeOpportunity]:
        tests = _get_test_functions(test_tree)
        already = self._already_param(tests)
        candidates = [t for t in tests if t.name not in already]
        groups: Dict[str, List[ast.FunctionDef]] = defaultdict(list)
        for f in candidates:
            groups[_normalize_body(f)].append(f)
        opps: List[ParameterizeOpportunity] = []
        for key, funcs in groups.items():
            if len(funcs) < 2:
                continue
            names = [f.name for f in funcs]
            all_lits = [_collect_literals(f) for f in funcs]
            common = set(all_lits[0]) if all_lits else set()
            for ls in all_lits[1:]:
                common &= set(ls)
            diff = [v for ls in all_lits for v in ls if v not in common]
            opps.append(ParameterizeOpportunity(
                names, str(hash(key)), diff[:20], self._param_names(funcs)))
        return opps

    @staticmethod
    def _already_param(tests: List[ast.FunctionDef]) -> Set[str]:
        return {f.name for f in tests for d in f.decorator_list
                if "parametrize" in ast.dump(d).lower()}

    @staticmethod
    def _param_names(funcs: List[ast.FunctionDef]) -> List[str]:
        if len(funcs) < 2:
            return []
        names = [f.name for f in funcs]
        prefix = names[0]
        for n in names[1:]:
            while not n.startswith(prefix) and prefix:
                prefix = prefix[:-1]
        suffixes = [n[len(prefix):].strip("_") for n in names]
        return ["input_value", "expected"] if all(suffixes) else ["param"]


class MutationSimulator:
    """Simulate code mutations and estimate test kill rate."""
    _OP_SWAPS: Dict[type, type] = {
        ast.Add: ast.Sub, ast.Sub: ast.Add,
        ast.Mult: ast.Div, ast.Div: ast.Mult,
        ast.Gt: ast.Lt, ast.Lt: ast.Gt,
        ast.GtE: ast.LtE, ast.LtE: ast.GtE,
        ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
    }

    def simulate(self, source_tree: ast.Module,
                 test_tree: ast.Module) -> MutationReport:
        mutations = self._gen_mutations(source_tree)
        if not mutations:
            return MutationReport(0, 0, 0, 0.0)
        targets = self._assertion_targets(test_tree)
        killed = sum(1 for m in mutations if self._likely_killed(m, targets))
        survived = len(mutations) - killed
        return MutationReport(len(mutations), killed, survived,
                              round(killed / len(mutations), 4))

    def _gen_mutations(self, tree: ast.Module) -> List[Dict[str, Any]]:
        muts: List[Dict[str, Any]] = []
        for node in ast.walk(tree):
            ln = getattr(node, "lineno", 0)
            ctx = self._enclosing(tree, node)
            if isinstance(node, ast.BinOp):
                sw = self._OP_SWAPS.get(type(node.op))
                if sw:
                    muts.append({"kind": "op_swap", "line": ln,
                                 "original": type(node.op).__name__,
                                 "mutated": sw.__name__, "context": ctx})
            if isinstance(node, ast.Compare):
                for op in node.ops:
                    sw = self._OP_SWAPS.get(type(op))
                    if sw:
                        muts.append({"kind": "cmp_swap", "line": ln,
                                     "original": type(op).__name__,
                                     "mutated": sw.__name__, "context": ctx})
            if isinstance(node, ast.If):
                muts.append({"kind": "negate_cond", "line": ln,
                             "original": "if cond", "mutated": "if not cond",
                             "context": ctx})
            if (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
                    and node.value not in (0, 1)):
                muts.append({"kind": "const_swap", "line": ln,
                             "original": str(node.value),
                             "mutated": str(node.value + 1), "context": ctx})
            if isinstance(node, ast.Return) and node.value is not None:
                muts.append({"kind": "return_none", "line": ln,
                             "original": "return expr", "mutated": "return None",
                             "context": ctx})
        return muts

    @staticmethod
    def _enclosing(tree: ast.Module, target: ast.AST) -> str:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if child is target:
                        return node.name
        return "<module>"

    @staticmethod
    def _assertion_targets(test_tree: ast.Module) -> Set[str]:
        names: Set[str] = set()
        for node in ast.walk(test_tree):
            is_assert = isinstance(node, ast.Assert)
            is_call = (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                       and isinstance(node.value.func, ast.Attribute)
                       and node.value.func.attr in _ASSERTION_METHODS)
            if not (is_assert or is_call):
                continue
            for n in ast.walk(node):
                if isinstance(n, ast.Name):
                    names.add(n.id)
                elif isinstance(n, ast.Attribute):
                    names.add(n.attr)
                elif isinstance(n, ast.Call):
                    if isinstance(n.func, ast.Name):
                        names.add(n.func.id)
                    elif isinstance(n.func, ast.Attribute):
                        names.add(n.func.attr)
        return names

    @staticmethod
    def _likely_killed(mutation: Dict[str, Any], targets: Set[str]) -> bool:
        ctx = mutation.get("context", "")
        if ctx in targets:
            return True
        kind = mutation["kind"]
        if kind in ("op_swap", "cmp_swap", "negate_cond"):
            rng = np.random.default_rng(hash(str(mutation)) & 0xFFFFFFFF)
            return bool(rng.random() < 0.6)
        return False


def _compute_grade(scores: List[float]) -> str:
    avg = float(np.mean(scores)) if scores else 0.0
    if avg >= 0.90:
        return "A"
    if avg >= 0.80:
        return "B"
    if avg >= 0.65:
        return "C"
    if avg >= 0.50:
        return "D"
    return "F"


def _generate_recommendations(r: TestQualityReport) -> List[str]:
    recs: List[str] = []
    if r.coverage_estimate and r.coverage_estimate.coverage_ratio < 0.5:
        missing = r.coverage_estimate.uncovered_functions[:5]
        recs.append(f"Low coverage ({r.coverage_estimate.coverage_ratio:.0%}). "
                    f"Add tests for: {', '.join(missing)}")
    if r.assertion_density < 1.0:
        recs.append(f"Low assertion density ({r.assertion_density:.2f}/test).")
    smell_counts = Counter(s.kind for s in r.smells)
    for kind, cnt in smell_counts.most_common(3):
        recs.append(f"Fix {cnt} '{kind.name}' smell(s).")
    if r.independence_score < 0.8:
        recs.append("Improve test independence — shared mutable state detected.")
    if r.mock_quality_score < 0.7:
        recs.append("Review mock usage — unasserted or leaking mocks found.")
    if r.parameterize_opportunities:
        recs.append(f"Parameterize {len(r.parameterize_opportunities)} group(s).")
    if r.mutation_score_estimate < 0.5:
        recs.append("Low mutation score — tests may miss regressions.")
    return recs


class TestAnalyzer:
    """Main entry point: analyze test quality."""

    def __init__(self) -> None:
        self._coverage = CoverageEstimator()
        self._assertions = AssertionAnalyzer()
        self._independence = IndependenceChecker()
        self._smells = SmellDetector()
        self._mocks = MockAnalyzer()
        self._parameterize = ParameterizeDetector()
        self._mutation = MutationSimulator()

    def analyze(self, test_source: str,
                source_code: Optional[str] = None) -> TestQualityReport:
        test_tree = ast.parse(test_source)
        cov = (self._coverage.estimate(source_code, test_source)
               if source_code else None)
        assertion_rep = self._assertions.analyze(test_tree)
        indep = self._independence.check(test_tree)
        smells = self._smells.detect(test_tree)
        mock_rep = self._mocks.analyze(test_tree)
        param_opps = self._parameterize.detect(test_tree)
        mut: Optional[MutationReport] = None
        if source_code:
            mut = self._mutation.simulate(ast.parse(source_code), test_tree)
        mut_score = mut.mutation_score if mut else 0.0
        cov_ratio = cov.coverage_ratio if cov else 0.0
        smell_penalty = min(len(smells) * 0.02, 0.3)
        scores = [cov_ratio, min(assertion_rep.assertion_density / 3.0, 1.0),
                  indep.score, mock_rep.quality_score,
                  max(1.0 - smell_penalty, 0.0), mut_score]
        report = TestQualityReport(
            coverage_estimate=cov,
            assertion_density=assertion_rep.assertion_density,
            smells=smells,
            independence_score=indep.score,
            mock_quality_score=mock_rep.quality_score,
            parameterize_opportunities=param_opps,
            mutation_score_estimate=mut_score,
            recommendations=[],
            overall_grade=_compute_grade(scores),
        )
        report.recommendations = _generate_recommendations(report)
        return report
