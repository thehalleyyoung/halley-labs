"""Pattern matching analysis — exhaustiveness, redundancy, and refactoring.

Analyzes Python 3.10+ ``match``/``case`` statements for exhaustiveness,
redundant patterns, and suggests improvements.  Also provides an automated
refactoring from ``isinstance`` chains to ``match`` statements.
"""
from __future__ import annotations

import ast
import copy
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ── Data types ───────────────────────────────────────────────────────────────

class PatternKind(Enum):
    LITERAL = "literal"
    CAPTURE = "capture"
    WILDCARD = "wildcard"
    CLASS = "class"
    SEQUENCE = "sequence"
    MAPPING = "mapping"
    OR = "or"
    STAR = "star"
    VALUE = "value"
    GUARD = "guard"


@dataclass
class PatternInfo:
    kind: PatternKind
    value: Any = None
    class_name: str = ""
    sub_patterns: List["PatternInfo"] = field(default_factory=list)
    keys: List[Any] = field(default_factory=list)
    guard: Optional[str] = None
    line: int = 0
    column: int = 0
    is_irrefutable: bool = False

    def __str__(self) -> str:
        if self.kind == PatternKind.WILDCARD:
            return "_"
        if self.kind == PatternKind.CAPTURE:
            return str(self.value)
        if self.kind == PatternKind.LITERAL:
            return repr(self.value)
        if self.kind == PatternKind.CLASS:
            args = ", ".join(str(s) for s in self.sub_patterns)
            return f"{self.class_name}({args})"
        if self.kind == PatternKind.SEQUENCE:
            elts = ", ".join(str(s) for s in self.sub_patterns)
            return f"[{elts}]"
        if self.kind == PatternKind.MAPPING:
            pairs = ", ".join(
                f"{k!r}: {v}" for k, v in zip(self.keys, self.sub_patterns)
            )
            return f"{{{pairs}}}"
        if self.kind == PatternKind.OR:
            return " | ".join(str(s) for s in self.sub_patterns)
        if self.kind == PatternKind.STAR:
            return f"*{self.value}" if self.value else "*_"
        if self.kind == PatternKind.VALUE:
            return str(self.value)
        return "?"


@dataclass
class NonExhaustiveMatch:
    line: int
    column: int
    subject: str
    missing_patterns: List[str]
    message: str
    severity: str = "warning"
    confidence: float = 0.8

    def __str__(self) -> str:
        missing = ", ".join(self.missing_patterns)
        return f"{self.line}:{self.column} Non-exhaustive match on '{self.subject}': missing {missing}"


@dataclass
class RedundantPattern:
    line: int
    column: int
    pattern: str
    subsumed_by: str
    subsumed_by_line: int
    message: str
    severity: str = "warning"
    confidence: float = 0.85

    def __str__(self) -> str:
        return (
            f"{self.line}:{self.column} Redundant pattern '{self.pattern}' "
            f"already covered by '{self.subsumed_by}' at line {self.subsumed_by_line}"
        )


class ImprovementKind(Enum):
    ADD_WILDCARD = "add_wildcard"
    REMOVE_REDUNDANT = "remove_redundant"
    SIMPLIFY_OR = "simplify_or"
    REORDER_PATTERNS = "reorder_patterns"
    USE_GUARD = "use_guard"
    MERGE_CASES = "merge_cases"
    CONVERT_TO_MATCH = "convert_to_match"


@dataclass
class MatchImprovement:
    kind: ImprovementKind
    line: int
    message: str
    suggested_code: Optional[str] = None
    severity: str = "info"

    def __str__(self) -> str:
        return f"{self.line} [{self.kind.value}] {self.message}"


@dataclass
class PatternReport:
    match_statements: int = 0
    total_patterns: int = 0
    exhaustive: int = 0
    non_exhaustive: List[NonExhaustiveMatch] = field(default_factory=list)
    redundant: List[RedundantPattern] = field(default_factory=list)
    improvements: List[MatchImprovement] = field(default_factory=list)
    isinstance_chains: int = 0

    @property
    def issues_count(self) -> int:
        return len(self.non_exhaustive) + len(self.redundant)


# ── Pattern extraction from AST ──────────────────────────────────────────────

def _extract_pattern(node: ast.pattern) -> PatternInfo:
    """Convert a ``match_case`` pattern AST node to ``PatternInfo``."""
    if isinstance(node, ast.MatchValue):
        val = None
        if isinstance(node.value, ast.Constant):
            val = node.value.value
        elif isinstance(node.value, ast.Attribute):
            val = _attr_str(node.value)
        return PatternInfo(
            kind=PatternKind.VALUE if isinstance(node.value, ast.Attribute) else PatternKind.LITERAL,
            value=val,
            line=node.value.lineno if hasattr(node.value, "lineno") else 0,
        )

    if isinstance(node, ast.MatchSingleton):
        return PatternInfo(
            kind=PatternKind.LITERAL,
            value=node.value,
            is_irrefutable=False,
        )

    if isinstance(node, ast.MatchSequence):
        subs = [_extract_pattern(p) for p in node.patterns]
        return PatternInfo(
            kind=PatternKind.SEQUENCE,
            sub_patterns=subs,
        )

    if isinstance(node, ast.MatchMapping):
        keys = []
        for k in node.keys:
            if isinstance(k, ast.Constant):
                keys.append(k.value)
            else:
                keys.append(str(k))
        subs = [_extract_pattern(p) for p in node.patterns]
        return PatternInfo(
            kind=PatternKind.MAPPING,
            keys=keys,
            sub_patterns=subs,
        )

    if isinstance(node, ast.MatchClass):
        cls_name = ""
        if isinstance(node.cls, ast.Name):
            cls_name = node.cls.id
        elif isinstance(node.cls, ast.Attribute):
            cls_name = _attr_str(node.cls)
        subs = [_extract_pattern(p) for p in node.patterns]
        for kw_pat in node.kwd_patterns:
            subs.append(_extract_pattern(kw_pat))
        return PatternInfo(
            kind=PatternKind.CLASS,
            class_name=cls_name,
            sub_patterns=subs,
        )

    if isinstance(node, ast.MatchStar):
        name = node.name
        return PatternInfo(
            kind=PatternKind.STAR,
            value=name,
            is_irrefutable=True,
        )

    if isinstance(node, ast.MatchAs):
        if node.pattern is None:
            if node.name is None:
                return PatternInfo(kind=PatternKind.WILDCARD, is_irrefutable=True)
            return PatternInfo(kind=PatternKind.CAPTURE, value=node.name, is_irrefutable=True)
        inner = _extract_pattern(node.pattern)
        inner.value = node.name
        return inner

    if isinstance(node, ast.MatchOr):
        subs = [_extract_pattern(p) for p in node.patterns]
        return PatternInfo(
            kind=PatternKind.OR,
            sub_patterns=subs,
            is_irrefutable=any(s.is_irrefutable for s in subs),
        )

    return PatternInfo(kind=PatternKind.WILDCARD, is_irrefutable=True)


def _attr_str(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_attr_str(node.value)}.{node.attr}"
    return ""


def _subject_str(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _attr_str(node)
    if isinstance(node, ast.Call):
        return f"{_attr_str(node.func)}()"
    if isinstance(node, ast.Subscript):
        return f"{_subject_str(node.value)}[...]"
    return "<expr>"


# ── Exhaustiveness checking ─────────────────────────────────────────────────

def _pattern_is_irrefutable(pat: PatternInfo) -> bool:
    """Check if a pattern is irrefutable (matches everything)."""
    if pat.is_irrefutable:
        return True
    if pat.kind in (PatternKind.WILDCARD, PatternKind.CAPTURE):
        return True
    if pat.kind == PatternKind.OR:
        return any(_pattern_is_irrefutable(s) for s in pat.sub_patterns)
    if pat.kind == PatternKind.STAR:
        return True
    return False


def _collect_literal_values(patterns: List[PatternInfo]) -> Set[Any]:
    """Collect all literal values from a list of patterns."""
    values: Set[Any] = set()
    for pat in patterns:
        if pat.kind == PatternKind.LITERAL:
            if pat.value is not None:
                try:
                    values.add(pat.value)
                except TypeError:
                    pass
        elif pat.kind == PatternKind.OR:
            values.update(_collect_literal_values(pat.sub_patterns))
    return values


def _collect_class_names(patterns: List[PatternInfo]) -> Set[str]:
    """Collect all class names from class patterns."""
    names: Set[str] = set()
    for pat in patterns:
        if pat.kind == PatternKind.CLASS:
            names.add(pat.class_name)
        elif pat.kind == PatternKind.OR:
            names.update(_collect_class_names(pat.sub_patterns))
    return names


KNOWN_ENUM_MEMBERS: Dict[str, List[str]] = {
    "bool": ["True", "False"],
    "NoneType": ["None"],
}

COMMON_UNION_TYPES: Dict[str, List[str]] = {
    "Optional": ["Some", "None"],
    "Result": ["Ok", "Err"],
    "Either": ["Left", "Right"],
}


def _check_match_exhaustiveness(
    subject: str,
    patterns: List[PatternInfo],
    line: int,
    column: int,
) -> Optional[NonExhaustiveMatch]:
    """Check if a match statement's patterns are exhaustive."""
    if any(_pattern_is_irrefutable(p) for p in patterns):
        return None

    missing: List[str] = []

    literal_vals = _collect_literal_values(patterns)
    class_names = _collect_class_names(patterns)

    if literal_vals:
        # Check for boolean exhaustiveness
        if literal_vals <= {True, False}:
            if True not in literal_vals:
                missing.append("True")
            if False not in literal_vals:
                missing.append("False")
            if not missing:
                return None
        else:
            missing.append("_ (wildcard)")
    elif class_names:
        missing.append("_ (wildcard)")
    else:
        missing.append("_ (wildcard)")

    if missing:
        return NonExhaustiveMatch(
            line=line,
            column=column,
            subject=subject,
            missing_patterns=missing,
            message=f"Match on '{subject}' is non-exhaustive; missing: {', '.join(missing)}",
        )
    return None


def check_exhaustiveness(source: str) -> List[NonExhaustiveMatch]:
    """Check all ``match`` statements in *source* for exhaustiveness."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results: List[NonExhaustiveMatch] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Match):
            continue

        patterns = [_extract_pattern(case.pattern) for case in node.cases]
        subject = _subject_str(node.subject)

        issue = _check_match_exhaustiveness(
            subject, patterns, node.lineno, node.col_offset
        )
        if issue:
            results.append(issue)

    return results


# ── Redundancy checking ─────────────────────────────────────────────────────

def _pattern_subsumes(a: PatternInfo, b: PatternInfo) -> bool:
    """Check whether pattern *a* subsumes (is more general than) pattern *b*."""
    if _pattern_is_irrefutable(a):
        return True

    if a.kind == b.kind:
        if a.kind == PatternKind.LITERAL:
            return a.value == b.value
        if a.kind == PatternKind.CLASS:
            if a.class_name != b.class_name:
                return False
            if not a.sub_patterns:
                return True
            if len(a.sub_patterns) != len(b.sub_patterns):
                return False
            return all(
                _pattern_subsumes(sa, sb)
                for sa, sb in zip(a.sub_patterns, b.sub_patterns)
            )
        if a.kind == PatternKind.SEQUENCE:
            if len(a.sub_patterns) != len(b.sub_patterns):
                return False
            return all(
                _pattern_subsumes(sa, sb)
                for sa, sb in zip(a.sub_patterns, b.sub_patterns)
            )
        if a.kind == PatternKind.MAPPING:
            if set(a.keys) != set(b.keys):
                return False
            a_map = dict(zip(a.keys, a.sub_patterns))
            b_map = dict(zip(b.keys, b.sub_patterns))
            return all(
                _pattern_subsumes(a_map[k], b_map[k])
                for k in a.keys if k in b_map
            )
        if a.kind == PatternKind.VALUE:
            return a.value == b.value

    if a.kind == PatternKind.OR:
        return any(_pattern_subsumes(s, b) for s in a.sub_patterns)

    if b.kind == PatternKind.OR:
        return all(_pattern_subsumes(a, s) for s in b.sub_patterns)

    return False


def check_redundant_patterns(source: str) -> List[RedundantPattern]:
    """Check all ``match`` statements for redundant case patterns."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results: List[RedundantPattern] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Match):
            continue

        case_patterns: List[Tuple[PatternInfo, int]] = []
        for case in node.cases:
            pat = _extract_pattern(case.pattern)
            case_line = case.pattern.lineno if hasattr(case.pattern, "lineno") else node.lineno
            case_patterns.append((pat, case_line))

        for i, (pat_b, line_b) in enumerate(case_patterns):
            for j, (pat_a, line_a) in enumerate(case_patterns[:i]):
                if _pattern_subsumes(pat_a, pat_b) and not _pattern_is_irrefutable(pat_b):
                    results.append(RedundantPattern(
                        line=line_b,
                        column=0,
                        pattern=str(pat_b),
                        subsumed_by=str(pat_a),
                        subsumed_by_line=line_a,
                        message=(
                            f"Pattern '{pat_b}' at line {line_b} is redundant; "
                            f"already covered by '{pat_a}' at line {line_a}"
                        ),
                    ))

    return results


# ── Improvement suggestions ─────────────────────────────────────────────────

def suggest_match_improvements(source: str) -> List[MatchImprovement]:
    """Suggest improvements for ``match`` statements and ``isinstance`` chains."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    improvements: List[MatchImprovement] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Match):
            improvements.extend(_suggest_match_stmt_improvements(node))
        if isinstance(node, ast.If):
            chain = _detect_isinstance_chain(node)
            if chain and len(chain) >= 3:
                improvements.append(MatchImprovement(
                    kind=ImprovementKind.CONVERT_TO_MATCH,
                    line=node.lineno,
                    message=(
                        f"isinstance chain with {len(chain)} branches could be "
                        f"converted to a match statement"
                    ),
                ))

    return improvements


def _suggest_match_stmt_improvements(node: ast.Match) -> List[MatchImprovement]:
    improvements: List[MatchImprovement] = []
    patterns = [_extract_pattern(case.pattern) for case in node.cases]

    # Check: no wildcard at end
    if patterns and not _pattern_is_irrefutable(patterns[-1]):
        improvements.append(MatchImprovement(
            kind=ImprovementKind.ADD_WILDCARD,
            line=node.lineno,
            message="Consider adding a wildcard `case _:` for exhaustiveness",
            suggested_code="    case _:\n        pass",
        ))

    # Check: irrefutable pattern not at end
    for i, pat in enumerate(patterns[:-1]):
        if _pattern_is_irrefutable(pat):
            improvements.append(MatchImprovement(
                kind=ImprovementKind.REORDER_PATTERNS,
                line=node.lineno,
                message=(
                    f"Irrefutable pattern '{pat}' at position {i + 1} shadows "
                    f"subsequent patterns"
                ),
            ))

    # Check: consecutive literal patterns that could use OR
    i = 0
    while i < len(patterns) - 1:
        if (patterns[i].kind == PatternKind.LITERAL and
                patterns[i + 1].kind == PatternKind.LITERAL):
            # Check if the bodies are the same
            body_i = ast.dump(node.cases[i].body[0]) if node.cases[i].body else ""
            body_j = ast.dump(node.cases[i + 1].body[0]) if node.cases[i + 1].body else ""
            if body_i and body_i == body_j:
                improvements.append(MatchImprovement(
                    kind=ImprovementKind.MERGE_CASES,
                    line=node.cases[i].pattern.lineno if hasattr(node.cases[i].pattern, "lineno") else node.lineno,
                    message=(
                        f"Cases '{patterns[i]}' and '{patterns[i + 1]}' have identical bodies; "
                        f"merge with 'case {patterns[i]} | {patterns[i + 1]}:'"
                    ),
                ))
        i += 1

    # Check: single-element OR patterns
    for pat in patterns:
        if pat.kind == PatternKind.OR and len(pat.sub_patterns) == 1:
            improvements.append(MatchImprovement(
                kind=ImprovementKind.SIMPLIFY_OR,
                line=node.lineno,
                message=f"OR pattern with single alternative '{pat}' can be simplified",
            ))

    return improvements


# ── isinstance chain detection and conversion ────────────────────────────────

@dataclass
class _IsinstanceBranch:
    variable: str
    type_name: str
    body: List[ast.stmt]
    line: int


def _detect_isinstance_chain(node: ast.If) -> List[_IsinstanceBranch]:
    """Detect if an ``if``/``elif`` chain is an isinstance dispatch."""
    chain: List[_IsinstanceBranch] = []
    current: Optional[ast.If] = node

    while current is not None:
        branch = _extract_isinstance_test(current.test)
        if branch is None:
            break
        branch.body = current.body
        branch.line = current.lineno
        chain.append(branch)

        if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
            current = current.orelse[0]
        else:
            if current.orelse:
                chain.append(_IsinstanceBranch(
                    variable=chain[0].variable if chain else "",
                    type_name="_",
                    body=current.orelse,
                    line=current.orelse[0].lineno if current.orelse else current.lineno,
                ))
            current = None

    if len(chain) < 2:
        return []

    # Verify all branches test the same variable
    var = chain[0].variable
    if all(b.variable == var or b.type_name == "_" for b in chain):
        return chain
    return []


def _extract_isinstance_test(test: ast.expr) -> Optional[_IsinstanceBranch]:
    if isinstance(test, ast.Call):
        if isinstance(test.func, ast.Name) and test.func.id == "isinstance":
            if len(test.args) == 2:
                var_name = ""
                if isinstance(test.args[0], ast.Name):
                    var_name = test.args[0].id
                type_name = ""
                if isinstance(test.args[1], ast.Name):
                    type_name = test.args[1].id
                elif isinstance(test.args[1], ast.Tuple):
                    names = [e.id for e in test.args[1].elts if isinstance(e, ast.Name)]
                    type_name = " | ".join(names)
                if var_name and type_name:
                    return _IsinstanceBranch(variable=var_name, type_name=type_name, body=[], line=0)
    return None


def isinstance_chain_to_match(source: str) -> str:
    """Refactor ``isinstance`` chains to ``match`` statements.

    Finds ``if isinstance(x, A): ... elif isinstance(x, B): ...`` chains
    and converts them to ``match x: case A(): ... case B(): ...``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines(keepends=True)
    replacements: List[Tuple[int, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        chain = _detect_isinstance_chain(node)
        if len(chain) < 2:
            continue

        var = chain[0].variable
        start_line = node.lineno - 1
        end_line = _find_chain_end(node)
        indent = _get_indent(lines[start_line])

        match_lines: List[str] = [f"{indent}match {var}:\n"]
        for branch in chain:
            if branch.type_name == "_":
                match_lines.append(f"{indent}    case _:\n")
            else:
                type_parts = branch.type_name.split(" | ")
                if len(type_parts) > 1:
                    pattern = " | ".join(f"{t}()" for t in type_parts)
                else:
                    pattern = f"{branch.type_name}()"
                match_lines.append(f"{indent}    case {pattern}:\n")

            if branch.body:
                body_indent = indent + "        "
                for stmt in branch.body:
                    stmt_lines = ast.get_source_segment(source, stmt)
                    if stmt_lines:
                        for sl in stmt_lines.splitlines(keepends=True):
                            stripped = sl.lstrip()
                            match_lines.append(f"{body_indent}{stripped}")
                            if not stripped.endswith("\n"):
                                match_lines.append("\n")
                    else:
                        match_lines.append(f"{body_indent}pass\n")
            else:
                match_lines.append(f"{indent}        pass\n")

        replacement = "".join(match_lines)
        replacements.append((start_line, end_line, replacement))

    for start, end, replacement in reversed(replacements):
        lines[start:end] = [replacement]

    return "".join(lines)


def _find_chain_end(node: ast.If) -> int:
    """Find the last line of an if/elif chain."""
    end = node.end_lineno or node.lineno
    if node.orelse:
        last = node.orelse[-1]
        if isinstance(last, ast.If):
            return _find_chain_end(last)
        end = last.end_lineno or last.lineno
    return end


def _get_indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


# ── Structural pattern report ───────────────────────────────────────────────

def analyze_pattern_matching(source: str) -> PatternReport:
    """Full pattern matching analysis entry point."""
    return structural_pattern_analysis(source)


def structural_pattern_analysis(source: str) -> PatternReport:
    """Produce a full report on all ``match`` statements and isinstance chains."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return PatternReport()

    report = PatternReport()

    for node in ast.walk(tree):
        if isinstance(node, ast.Match):
            report.match_statements += 1
            patterns = [_extract_pattern(case.pattern) for case in node.cases]
            report.total_patterns += len(patterns)

            subject = _subject_str(node.subject)
            exhaustive = _check_match_exhaustiveness(
                subject, patterns, node.lineno, node.col_offset
            )
            if exhaustive:
                report.non_exhaustive.append(exhaustive)
            else:
                report.exhaustive += 1

        if isinstance(node, ast.If):
            chain = _detect_isinstance_chain(node)
            if len(chain) >= 2:
                report.isinstance_chains += 1

    report.redundant = check_redundant_patterns(source)
    report.improvements = suggest_match_improvements(source)

    return report
