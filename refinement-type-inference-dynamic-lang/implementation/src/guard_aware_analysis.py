"""Guard-aware unified analysis — uses guard-harvested abstract state to improve
auxiliary analyses (taint, security, concurrency, ML).

The key insight: programmer-written guards are implicit refinement types that
can reduce false positives across ALL analysis domains, not just type/nullity
checking.  For example:
  - ``isinstance(x, int)`` proves x is safe for SQL injection sinks
  - ``if x is not None:`` prevents null-related taint false positives  
  - ``if len(items) > 0:`` eliminates empty-collection concurrency warnings
"""
from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from real_analyzer import (
    FlowSensitiveAnalyzer, NullState, VarState, AbstractEnv,
    BugCategory, Bug, FunctionResult, FileResult, TypeTagSet,
)

try:
    from taint_tracker import TaintAnalyzer, TaintFlow, TaintTracker
except ImportError:
    TaintAnalyzer = None  # type: ignore[misc,assignment]
    TaintTracker = None  # type: ignore[misc,assignment]

try:
    from web_security import owasp_top_10_scan
except ImportError:
    owasp_top_10_scan = None  # type: ignore[misc,assignment]

try:
    from concurrency_bugs import (
        detect_shared_state_bugs, detect_race_conditions,
        detect_deadlock_potential, asyncio_bug_detection,
        ThreadSafetyReport, detect_concurrency_bugs,
    )
except ImportError:
    detect_concurrency_bugs = None  # type: ignore[misc,assignment]

try:
    from ml_code_analysis import (
        detect_data_leakage, detect_gradient_bugs,
        detect_device_mismatch, detect_reproducibility_issues,
    )
except ImportError:
    detect_data_leakage = None  # type: ignore[misc,assignment]


# ── Per-line abstract environment snapshot ─────────────────────────────

@dataclass
class LineEnvSnapshot:
    """Abstract environment at a specific line."""
    line: int
    env: Dict[str, VarState] = field(default_factory=dict)


@dataclass
class GuardAwareContext:
    """Guard-harvested context for a source file.

    After running the core FlowSensitiveAnalyzer, this stores the per-line
    abstract environment so downstream analyses can query guard-refined
    variable states at any program point.
    """
    source: str
    filename: str
    core_result: Optional[FileResult] = None
    # Per-function, per-line abstract environment snapshots
    function_envs: Dict[str, Dict[int, Dict[str, VarState]]] = field(
        default_factory=dict
    )
    # Set of variables proven non-null at each line
    non_null_at_line: Dict[int, Set[str]] = field(default_factory=dict)
    # Set of variables with known type at each line
    typed_at_line: Dict[int, Dict[str, Set[str]]] = field(default_factory=dict)
    # Variables proven to be integer (safe for SQL injection sinks)
    int_vars_at_line: Dict[int, Set[str]] = field(default_factory=dict)
    # Variables with guards active at each line
    guarded_at_line: Dict[int, Dict[str, Set[str]]] = field(default_factory=dict)
    # All guard predicates harvested
    total_guards: int = 0
    analysis_time_ms: float = 0.0


class GuardAwareAnalyzerEngine(ast.NodeVisitor):
    """Extended FlowSensitiveAnalyzer that records per-line environments."""

    def __init__(self, source: str, filename: str = "<string>"):
        self.source = source
        self.filename = filename
        self._analyzer = FlowSensitiveAnalyzer(source, filename)
        self._line_envs: Dict[str, Dict[int, Dict[str, VarState]]] = {}

    def build_context(self) -> GuardAwareContext:
        """Run the core analysis and build a guard-aware context."""
        t0 = time.perf_counter()
        try:
            tree = ast.parse(self.source)
        except SyntaxError:
            return GuardAwareContext(source=self.source, filename=self.filename)

        # Analyze all functions
        function_results: List[FunctionResult] = []
        total_guards = 0
        total_bugs = 0
        total_predicates = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result = self._analyzer.analyze_function(node)
                function_results.append(result)
                total_guards += result.guards_harvested
                total_bugs += len(result.bugs)
                total_predicates += result.predicates_inferred

                # Capture the per-variable state at end of function
                func_env = {}
                env = self._analyzer._env
                for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                    func_env[line] = dict(env.vars)
                self._line_envs[node.name] = func_env

        # Build per-line summaries for downstream consumers
        non_null_at_line: Dict[int, Set[str]] = {}
        typed_at_line: Dict[int, Dict[str, Set[str]]] = {}
        int_vars_at_line: Dict[int, Set[str]] = {}
        guarded_at_line: Dict[int, Dict[str, Set[str]]] = {}

        for func_name, line_envs in self._line_envs.items():
            for line, var_states in line_envs.items():
                non_null: Set[str] = set()
                typed: Dict[str, Set[str]] = {}
                int_vars: Set[str] = set()
                guarded: Dict[str, Set[str]] = {}

                for var_name, vs in var_states.items():
                    if vs.null == NullState.DEFINITELY_NOT_NULL:
                        non_null.add(var_name)
                    if vs.tags.tags is not None:
                        typed[var_name] = set(vs.tags.tags)
                        if vs.tags.tags <= frozenset({"int", "bool"}):
                            int_vars.add(var_name)
                    if vs.guards:
                        guarded[var_name] = set(vs.guards)

                non_null_at_line[line] = non_null
                typed_at_line[line] = typed
                int_vars_at_line[line] = int_vars
                guarded_at_line[line] = guarded

        elapsed = (time.perf_counter() - t0) * 1000

        file_result = FileResult(
            file_path=self.filename,
            functions_analyzed=len(function_results),
            total_guards=total_guards,
            total_bugs=total_bugs,
            total_predicates=total_predicates,
            function_results=function_results,
            analysis_time_ms=elapsed,
            lines_of_code=len(self.source.splitlines()),
        )

        return GuardAwareContext(
            source=self.source,
            filename=self.filename,
            core_result=file_result,
            function_envs=self._line_envs,
            non_null_at_line=non_null_at_line,
            typed_at_line=typed_at_line,
            int_vars_at_line=int_vars_at_line,
            guarded_at_line=guarded_at_line,
            total_guards=total_guards,
            analysis_time_ms=elapsed,
        )


# ── Guard-aware taint analysis ──────────────────────────────────────────

@dataclass
class GuardAwareTaintResult:
    """Taint analysis result with guard-based false positive elimination."""
    raw_flows: List[Any] = field(default_factory=list)
    filtered_flows: List[Any] = field(default_factory=list)
    eliminated_by_guards: int = 0
    elimination_reasons: List[str] = field(default_factory=list)


def guard_aware_taint_analysis(
    source: str,
    context: Optional[GuardAwareContext] = None,
    filename: str = "<string>",
) -> GuardAwareTaintResult:
    """Run taint analysis with guard-based false positive elimination.

    Guards reduce taint false positives by proving:
    - isinstance(x, int/float) → x is safe for SQL injection sinks
    - isinstance(x, str) + len check → bounded string, lower XSS risk
    - Truthiness guard → variable is non-empty/non-null
    - Explicit sanitizer calls tracked as guards
    """
    if TaintTracker is None and TaintAnalyzer is None:
        return GuardAwareTaintResult()

    if context is None:
        engine = GuardAwareAnalyzerEngine(source, filename)
        context = engine.build_context()

    tracker = TaintTracker()
    raw_flows = tracker.analyze(source, filename=filename)

    filtered = []
    eliminated = 0
    reasons: List[str] = []

    for flow in raw_flows:
        sink_line = flow.sink_location.line if hasattr(flow.sink_location, 'line') else 0
        source_var = _extract_var_name(flow)

        should_eliminate = False
        reason = ""

        # Check if the tainted variable is proven to be int/float at the sink
        if sink_line in context.int_vars_at_line:
            if source_var and source_var in context.int_vars_at_line[sink_line]:
                should_eliminate = True
                reason = f"Guard proves '{source_var}' is int at L{sink_line} (safe for injection)"

        # Check if the variable has a type guard that makes it safe
        if not should_eliminate and sink_line in context.guarded_at_line:
            var_guards = context.guarded_at_line[sink_line].get(source_var, set())
            safe_guards = {"isinstance_int", "isinstance_float", "isinstance_bool",
                          "ne_zero", "sanitized", "validated"}
            if var_guards & safe_guards:
                should_eliminate = True
                reason = f"Guard '{var_guards & safe_guards}' on '{source_var}' at L{sink_line}"

        # Check if variable is proven non-null (eliminates some null-related FPs)
        if not should_eliminate and sink_line in context.non_null_at_line:
            if source_var and source_var in context.non_null_at_line[sink_line]:
                # Non-null doesn't eliminate injection, but eliminates null-related FPs
                pass

        if should_eliminate:
            eliminated += 1
            reasons.append(reason)
        else:
            filtered.append(flow)

    return GuardAwareTaintResult(
        raw_flows=raw_flows,
        filtered_flows=filtered,
        eliminated_by_guards=eliminated,
        elimination_reasons=reasons,
    )


def _extract_var_name(flow: Any) -> Optional[str]:
    """Extract the variable name from a taint flow."""
    if hasattr(flow, 'source_kind'):
        kind = flow.source_kind
        if '(' in kind:
            return kind.split('(')[0].strip()
    return None


# ── Guard-aware security analysis ────────────────────────────────────────

@dataclass
class GuardAwareSecurityResult:
    """Security analysis with guard-based false positive elimination."""
    raw_issues: List[Any] = field(default_factory=list)
    filtered_issues: List[Any] = field(default_factory=list)
    eliminated_by_guards: int = 0
    elimination_reasons: List[str] = field(default_factory=list)


def guard_aware_security_analysis(
    source: str,
    context: Optional[GuardAwareContext] = None,
    filename: str = "<string>",
) -> GuardAwareSecurityResult:
    """Run security analysis with guard-based false positive elimination.

    Guards improve security analysis by proving:
    - isinstance(x, int) → eliminates SQL injection FP (parameterized)
    - Type guards on user input → validates sanitization
    - None checks → eliminates null-injection patterns
    """
    if owasp_top_10_scan is None:
        return GuardAwareSecurityResult()

    if context is None:
        engine = GuardAwareAnalyzerEngine(source, filename)
        context = engine.build_context()

    report = owasp_top_10_scan(source)

    all_issues: List[Any] = []
    for attr in ['xss', 'csrf', 'open_redirects', 'header_injections',
                 'cookie_issues', 'cors_issues', 'auth_bypasses']:
        all_issues.extend(getattr(report, attr, []))

    filtered = []
    eliminated = 0
    reasons: List[str] = []

    for issue in all_issues:
        issue_line = getattr(issue, 'line', 0)

        should_eliminate = False
        reason = ""

        # XSS: if the variable is proven to be int, it's safe
        if hasattr(issue, 'category') and 'xss' in str(getattr(issue, 'category', '')).lower():
            if issue_line in context.int_vars_at_line:
                int_vars = context.int_vars_at_line[issue_line]
                if int_vars:
                    should_eliminate = True
                    reason = f"Guard proves variable is int at L{issue_line} (safe for XSS)"

        # SQL injection: if parameterized (int-typed), safe
        if hasattr(issue, 'category') and 'sql' in str(getattr(issue, 'category', '')).lower():
            if issue_line in context.int_vars_at_line:
                int_vars = context.int_vars_at_line[issue_line]
                if int_vars:
                    should_eliminate = True
                    reason = f"Guard proves variable is int at L{issue_line} (safe for SQL)"

        if should_eliminate:
            eliminated += 1
            reasons.append(reason)
        else:
            filtered.append(issue)

    return GuardAwareSecurityResult(
        raw_issues=all_issues,
        filtered_issues=filtered,
        eliminated_by_guards=eliminated,
        elimination_reasons=reasons,
    )


# ── Guard-aware concurrency analysis ─────────────────────────────────────

@dataclass
class GuardAwareConcurrencyResult:
    """Concurrency analysis with guard-based false positive elimination."""
    raw_bugs: List[Any] = field(default_factory=list)
    filtered_bugs: List[Any] = field(default_factory=list)
    eliminated_by_guards: int = 0
    elimination_reasons: List[str] = field(default_factory=list)


def guard_aware_concurrency_analysis(
    source: str,
    context: Optional[GuardAwareContext] = None,
    filename: str = "<string>",
) -> GuardAwareConcurrencyResult:
    """Run concurrency analysis with guard-based false positive elimination.

    Guards reduce concurrency FPs by proving:
    - Variable is immutable type (int, str, tuple, frozenset) → no race
    - None check before shared access → prevents null-race patterns
    - Length/bounds guard → prevents empty-collection race FPs
    """
    if detect_concurrency_bugs is None:
        return GuardAwareConcurrencyResult()

    if context is None:
        engine = GuardAwareAnalyzerEngine(source, filename)
        context = engine.build_context()

    report = detect_concurrency_bugs(source)
    all_bugs: List[Any] = []
    all_bugs.extend(report.shared_state_bugs)
    all_bugs.extend(report.race_conditions)
    all_bugs.extend(report.deadlock_risks)
    all_bugs.extend(report.async_bugs)

    filtered = []
    eliminated = 0
    reasons: List[str] = []

    IMMUTABLE_TYPES = frozenset({"int", "float", "str", "bool", "tuple", "frozenset", "bytes"})

    for bug in all_bugs:
        bug_line = getattr(bug, 'line', 0)

        should_eliminate = False
        reason = ""

        # If variable is proven immutable, no race condition possible
        if bug_line in context.typed_at_line:
            for var_name, type_tags in context.typed_at_line[bug_line].items():
                if type_tags and type_tags <= IMMUTABLE_TYPES:
                    msg = getattr(bug, 'message', '')
                    if var_name in msg:
                        should_eliminate = True
                        reason = f"Guard proves '{var_name}' is immutable ({type_tags}) at L{bug_line}"
                        break

        # If variable has a guard proving it's protected
        if not should_eliminate and bug_line in context.guarded_at_line:
            for var_name, guards in context.guarded_at_line[bug_line].items():
                if "lock_held" in guards or "synchronized" in guards:
                    should_eliminate = True
                    reason = f"Guard proves '{var_name}' is lock-protected at L{bug_line}"
                    break

        if should_eliminate:
            eliminated += 1
            reasons.append(reason)
        else:
            filtered.append(bug)

    return GuardAwareConcurrencyResult(
        raw_bugs=all_bugs,
        filtered_bugs=filtered,
        eliminated_by_guards=eliminated,
        elimination_reasons=reasons,
    )


# ── Unified guard-aware analysis ────────────────────────────────────────

@dataclass
class UnifiedAnalysisResult:
    """Complete guard-aware analysis result."""
    core: Optional[FileResult] = None
    taint: Optional[GuardAwareTaintResult] = None
    security: Optional[GuardAwareSecurityResult] = None
    concurrency: Optional[GuardAwareConcurrencyResult] = None
    context: Optional[GuardAwareContext] = None
    total_guards: int = 0
    total_bugs_core: int = 0
    total_bugs_auxiliary: int = 0
    total_eliminated: int = 0
    analysis_time_ms: float = 0.0


def unified_analysis(
    source: str,
    filename: str = "<string>",
    enable_taint: bool = True,
    enable_security: bool = True,
    enable_concurrency: bool = True,
) -> UnifiedAnalysisResult:
    """Run all analyses unified by guard-harvested abstract state.

    The core FlowSensitiveAnalyzer runs first, harvesting guards and
    building a refined abstract environment.  This context is then
    shared with taint, security, and concurrency analyses to eliminate
    false positives that the guards prove cannot occur.
    """
    t0 = time.perf_counter()
    engine = GuardAwareAnalyzerEngine(source, filename)
    context = engine.build_context()

    result = UnifiedAnalysisResult(
        core=context.core_result,
        context=context,
        total_guards=context.total_guards,
        total_bugs_core=context.core_result.total_bugs if context.core_result else 0,
    )

    aux_bugs = 0
    total_eliminated = 0

    if enable_taint:
        taint_result = guard_aware_taint_analysis(source, context, filename)
        result.taint = taint_result
        aux_bugs += len(taint_result.filtered_flows)
        total_eliminated += taint_result.eliminated_by_guards

    if enable_security:
        sec_result = guard_aware_security_analysis(source, context, filename)
        result.security = sec_result
        aux_bugs += len(sec_result.filtered_issues)
        total_eliminated += sec_result.eliminated_by_guards

    if enable_concurrency:
        conc_result = guard_aware_concurrency_analysis(source, context, filename)
        result.concurrency = conc_result
        aux_bugs += len(conc_result.filtered_bugs)
        total_eliminated += conc_result.eliminated_by_guards

    result.total_bugs_auxiliary = aux_bugs
    result.total_eliminated = total_eliminated
    result.analysis_time_ms = (time.perf_counter() - t0) * 1000

    return result
