"""
Automated fix suggestions for detected bugs and code smells.
Given a bug, suggest concrete code fixes with before/after examples.
"""

import ast
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from .python_ast_analyzer import Bug, Severity


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class BreakingRisk(Enum):
    NONE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


@dataclass
class Fix:
    description: str
    before_code: str
    after_code: str
    confidence: float = 0.0    # 0.0 – 1.0
    breaking_risk: BreakingRisk = BreakingRisk.LOW
    category: str = ""

    def __str__(self) -> str:
        return f"[{self.confidence:.0%}] {self.description}"


# ---------------------------------------------------------------------------
# Fix generators per category
# ---------------------------------------------------------------------------

class _NoneDerefFixer:
    """Suggest fixes for None dereference bugs."""

    def suggest(self, bug: Bug) -> List[Fix]:
        fixes: List[Fix] = []
        # Extract variable name from message
        match = re.search(r"'(\w+)'", bug.message)
        var_name = match.group(1) if match else "x"

        attr_match = re.search(r"'\.(\w+)'", bug.message)
        attr_name = attr_match.group(1) if attr_match else "attr"

        # Fix 1: Guard with None check
        fixes.append(Fix(
            description=f"Add None check before accessing '.{attr_name}' on '{var_name}'",
            before_code=f"result = {var_name}.{attr_name}",
            after_code=(
                f"if {var_name} is not None:\n"
                f"    result = {var_name}.{attr_name}\n"
                f"else:\n"
                f"    result = None  # or a default value"
            ),
            confidence=0.9,
            breaking_risk=BreakingRisk.LOW,
            category="none-dereference",
        ))

        # Fix 2: Use getattr with default
        fixes.append(Fix(
            description=f"Use getattr() with default value",
            before_code=f"result = {var_name}.{attr_name}",
            after_code=f"result = getattr({var_name}, '{attr_name}', None)",
            confidence=0.7,
            breaking_risk=BreakingRisk.LOW,
            category="none-dereference",
        ))

        # Fix 3: Use conditional expression
        fixes.append(Fix(
            description=f"Use conditional expression (ternary)",
            before_code=f"result = {var_name}.{attr_name}",
            after_code=f"result = {var_name}.{attr_name} if {var_name} is not None else None",
            confidence=0.85,
            breaking_risk=BreakingRisk.NONE,
            category="none-dereference",
        ))

        return fixes


class _DivByZeroFixer:
    """Suggest fixes for division by zero bugs."""

    def suggest(self, bug: Bug) -> List[Fix]:
        fixes: List[Fix] = []
        match = re.search(r"'(\w+)'", bug.message)
        var_name = match.group(1) if match else "divisor"

        # Fix 1: Guard with zero check
        fixes.append(Fix(
            description=f"Add zero check before division",
            before_code=f"result = value / {var_name}",
            after_code=(
                f"if {var_name} != 0:\n"
                f"    result = value / {var_name}\n"
                f"else:\n"
                f"    result = 0  # or float('inf'), or raise"
            ),
            confidence=0.9,
            breaking_risk=BreakingRisk.LOW,
            category="division-by-zero",
        ))

        # Fix 2: Try/except
        fixes.append(Fix(
            description=f"Wrap in try/except ZeroDivisionError",
            before_code=f"result = value / {var_name}",
            after_code=(
                f"try:\n"
                f"    result = value / {var_name}\n"
                f"except ZeroDivisionError:\n"
                f"    result = 0  # default value"
            ),
            confidence=0.8,
            breaking_risk=BreakingRisk.NONE,
            category="division-by-zero",
        ))

        # Fix 3: Use max() to ensure non-zero
        fixes.append(Fix(
            description=f"Use max() to ensure divisor is non-zero",
            before_code=f"result = value / {var_name}",
            after_code=f"result = value / max({var_name}, 1)",
            confidence=0.6,
            breaking_risk=BreakingRisk.MEDIUM,
            category="division-by-zero",
        ))

        return fixes


class _TypeErrorFixer:
    """Suggest fixes for type errors."""

    def suggest(self, bug: Bug) -> List[Fix]:
        fixes: List[Fix] = []

        if "not callable" in bug.message:
            match = re.search(r"'(\w+)'.*type '(\w+)'", bug.message)
            var = match.group(1) if match else "x"
            ty = match.group(2) if match else "int"

            fixes.append(Fix(
                description=f"Remove function call on non-callable '{var}'",
                before_code=f"result = {var}()",
                after_code=f"result = {var}  # '{var}' is {ty}, not callable",
                confidence=0.8,
                breaking_risk=BreakingRisk.MEDIUM,
                category="type-error",
            ))

        if "Unsupported operand" in bug.message:
            match = re.search(r"'(\w+)' and '(\w+)'", bug.message)
            lt = match.group(1) if match else "str"
            rt = match.group(2) if match else "int"

            if lt == "str" and rt == "int":
                fixes.append(Fix(
                    description="Convert int to str before concatenation",
                    before_code='result = "text" + number',
                    after_code='result = "text" + str(number)',
                    confidence=0.9,
                    breaking_risk=BreakingRisk.NONE,
                    category="type-error",
                ))
                fixes.append(Fix(
                    description="Use f-string for concatenation",
                    before_code='result = "text" + number',
                    after_code='result = f"text{number}"',
                    confidence=0.85,
                    breaking_risk=BreakingRisk.NONE,
                    category="type-error",
                ))
            elif lt == "int" and rt == "str":
                fixes.append(Fix(
                    description="Convert str to int before addition",
                    before_code="result = number + text",
                    after_code="result = number + int(text)",
                    confidence=0.7,
                    breaking_risk=BreakingRisk.MEDIUM,
                    category="type-error",
                ))

            fixes.append(Fix(
                description="Add isinstance check before operation",
                before_code=f"result = a + b",
                after_code=(
                    f"if isinstance(a, str) and isinstance(b, str):\n"
                    f"    result = a + b\n"
                    f"else:\n"
                    f"    result = str(a) + str(b)"
                ),
                confidence=0.6,
                breaking_risk=BreakingRisk.LOW,
                category="type-error",
            ))

        return fixes


class _IndexOutOfBoundsFixer:
    """Suggest fixes for index out of bounds bugs."""

    def suggest(self, bug: Bug) -> List[Fix]:
        fixes: List[Fix] = []
        match = re.search(r"Index (\d+).*'(\w+)'.*length (\d+)", bug.message)
        idx = match.group(1) if match else "i"
        var = match.group(2) if match else "lst"
        length = match.group(3) if match else "n"

        fixes.append(Fix(
            description=f"Check index bounds before access",
            before_code=f"value = {var}[{idx}]",
            after_code=(
                f"if {idx} < len({var}):\n"
                f"    value = {var}[{idx}]\n"
                f"else:\n"
                f"    value = None  # or default"
            ),
            confidence=0.9,
            breaking_risk=BreakingRisk.LOW,
            category="index-out-of-bounds",
        ))

        fixes.append(Fix(
            description=f"Use try/except IndexError",
            before_code=f"value = {var}[{idx}]",
            after_code=(
                f"try:\n"
                f"    value = {var}[{idx}]\n"
                f"except IndexError:\n"
                f"    value = None"
            ),
            confidence=0.85,
            breaking_risk=BreakingRisk.NONE,
            category="index-out-of-bounds",
        ))

        return fixes


class _UnusedImportFixer:
    """Suggest fixes for unused imports."""

    def suggest(self, bug: Bug) -> List[Fix]:
        match = re.search(r"'(\w+)'", bug.message)
        name = match.group(1) if match else "module"

        return [Fix(
            description=f"Remove unused import '{name}'",
            before_code=f"import {name}",
            after_code=f"# import {name}  # removed: unused",
            confidence=0.95,
            breaking_risk=BreakingRisk.NONE,
            category="unused-import",
        )]


class _UnusedVariableFixer:
    """Suggest fixes for unused variables."""

    def suggest(self, bug: Bug) -> List[Fix]:
        match = re.search(r"'(\w+)'", bug.message)
        name = match.group(1) if match else "x"

        return [
            Fix(
                description=f"Prefix with underscore to indicate intentionally unused",
                before_code=f"{name} = value",
                after_code=f"_{name} = value",
                confidence=0.8,
                breaking_risk=BreakingRisk.NONE,
                category="unused-variable",
            ),
            Fix(
                description=f"Remove the unused variable assignment",
                before_code=f"{name} = value",
                after_code=f"# {name} = value  # removed: unused",
                confidence=0.7,
                breaking_risk=BreakingRisk.LOW,
                category="unused-variable",
            ),
        ]


class _UnreachableCodeFixer:
    """Suggest fixes for unreachable code."""

    def suggest(self, bug: Bug) -> List[Fix]:
        return [Fix(
            description="Remove unreachable code",
            before_code="return result\nmore_code()",
            after_code="return result\n# more_code()  # unreachable: removed",
            confidence=0.95,
            breaking_risk=BreakingRisk.NONE,
            category="unreachable-code",
        )]


class _UninitializedVarFixer:
    """Suggest fixes for uninitialized variables."""

    def suggest(self, bug: Bug) -> List[Fix]:
        match = re.search(r"'(\w+)'", bug.message)
        name = match.group(1) if match else "x"

        return [
            Fix(
                description=f"Initialize '{name}' before use",
                before_code=f"# ... code uses {name} ...\nprint({name})",
                after_code=f"{name} = None  # initialize\n# ... code uses {name} ...\nprint({name})",
                confidence=0.7,
                breaking_risk=BreakingRisk.LOW,
                category="uninitialized-variable",
            ),
        ]


class _SecurityFixer:
    """Suggest fixes for security issues (taint flow)."""

    def suggest(self, vuln_type: str, source_kind: str, sink_kind: str) -> List[Fix]:
        fixes: List[Fix] = []

        if "sql" in vuln_type.lower():
            fixes.append(Fix(
                description="Use parameterized queries instead of string formatting",
                before_code='cursor.execute("SELECT * FROM users WHERE id = " + user_id)',
                after_code='cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))',
                confidence=0.95,
                breaking_risk=BreakingRisk.LOW,
                category="sql-injection",
            ))

        if "command" in vuln_type.lower():
            fixes.append(Fix(
                description="Use subprocess with list arguments instead of shell=True",
                before_code='os.system("ls " + user_input)',
                after_code='subprocess.run(["ls", user_input], shell=False)',
                confidence=0.9,
                breaking_risk=BreakingRisk.MEDIUM,
                category="command-injection",
            ))
            fixes.append(Fix(
                description="Use shlex.quote to escape shell arguments",
                before_code='os.system("cmd " + user_input)',
                after_code='os.system("cmd " + shlex.quote(user_input))',
                confidence=0.85,
                breaking_risk=BreakingRisk.LOW,
                category="command-injection",
            ))

        if "path" in vuln_type.lower():
            fixes.append(Fix(
                description="Validate and sanitize file path",
                before_code='open(user_path)',
                after_code=(
                    'import os\n'
                    'safe_path = os.path.normpath(user_path)\n'
                    'if not safe_path.startswith(ALLOWED_DIR):\n'
                    '    raise ValueError("Path traversal attempt")\n'
                    'open(safe_path)'
                ),
                confidence=0.85,
                breaking_risk=BreakingRisk.LOW,
                category="path-traversal",
            ))

        if "xss" in vuln_type.lower():
            fixes.append(Fix(
                description="Escape HTML output",
                before_code='return "<div>" + user_input + "</div>"',
                after_code=(
                    'from markupsafe import escape\n'
                    'return "<div>" + escape(user_input) + "</div>"'
                ),
                confidence=0.9,
                breaking_risk=BreakingRisk.NONE,
                category="xss",
            ))

        if "ssrf" in vuln_type.lower():
            fixes.append(Fix(
                description="Validate URL against allowlist",
                before_code='requests.get(user_url)',
                after_code=(
                    'from urllib.parse import urlparse\n'
                    'parsed = urlparse(user_url)\n'
                    'if parsed.hostname not in ALLOWED_HOSTS:\n'
                    '    raise ValueError("URL not allowed")\n'
                    'requests.get(user_url)'
                ),
                confidence=0.85,
                breaking_risk=BreakingRisk.LOW,
                category="ssrf",
            ))

        if not fixes:
            fixes.append(Fix(
                description="Sanitize user input before use",
                before_code="dangerous_func(user_input)",
                after_code="dangerous_func(sanitize(user_input))",
                confidence=0.6,
                breaking_risk=BreakingRisk.MEDIUM,
                category="security",
            ))

        return fixes


class _PatternFixer:
    """Suggest fixes for anti-patterns and code smells."""

    def suggest(self, pattern_type: str) -> List[Fix]:
        fixes: List[Fix] = []

        if pattern_type == "mutable-default-argument":
            fixes.append(Fix(
                description="Use None as default and create mutable inside function",
                before_code="def f(items=[]):\n    items.append(1)",
                after_code="def f(items=None):\n    if items is None:\n        items = []\n    items.append(1)",
                confidence=0.95,
                breaking_risk=BreakingRisk.LOW,
                category="mutable-default-argument",
            ))

        elif pattern_type == "bare-except":
            fixes.append(Fix(
                description="Catch specific exception type",
                before_code="try:\n    risky()\nexcept:\n    pass",
                after_code="try:\n    risky()\nexcept Exception:\n    pass",
                confidence=0.9,
                breaking_risk=BreakingRisk.NONE,
                category="bare-except",
            ))

        elif pattern_type == "bad-comparison":
            fixes.append(Fix(
                description="Use 'is' for None/True/False comparisons",
                before_code="if x == None:",
                after_code="if x is None:",
                confidence=0.95,
                breaking_risk=BreakingRisk.NONE,
                category="bad-comparison",
            ))

        elif pattern_type == "god-class":
            fixes.append(Fix(
                description="Split class into smaller, focused classes",
                before_code="class GodClass:\n    # 20+ methods...",
                after_code="class DataHandler:\n    # data methods\n\nclass Renderer:\n    # display methods",
                confidence=0.6,
                breaking_risk=BreakingRisk.HIGH,
                category="god-class",
            ))

        elif pattern_type == "long-method":
            fixes.append(Fix(
                description="Extract logical blocks into helper functions",
                before_code="def long_method():\n    # 100+ lines...",
                after_code="def long_method():\n    step1()\n    step2()\n    step3()",
                confidence=0.7,
                breaking_risk=BreakingRisk.MEDIUM,
                category="long-method",
            ))

        elif pattern_type == "unnecessary-list-comprehension":
            fixes.append(Fix(
                description="Replace list comprehension with generator expression",
                before_code="sum([x*x for x in range(100)])",
                after_code="sum(x*x for x in range(100))",
                confidence=0.95,
                breaking_risk=BreakingRisk.NONE,
                category="unnecessary-list-comprehension",
            ))

        else:
            fixes.append(Fix(
                description=f"Refactor to address '{pattern_type}'",
                before_code="# current code with anti-pattern",
                after_code="# refactored code",
                confidence=0.5,
                breaking_risk=BreakingRisk.MEDIUM,
                category=pattern_type,
            ))

        return fixes


class _ComplexityFixer:
    """Suggest fixes for complexity issues."""

    def suggest(self, metric: str, value: float) -> List[Fix]:
        fixes: List[Fix] = []

        if metric == "cyclomatic":
            fixes.append(Fix(
                description="Extract complex conditions into named boolean variables",
                before_code="if a and b or c and (d or e):\n    ...",
                after_code="is_valid = a and b\nis_special = c and (d or e)\nif is_valid or is_special:\n    ...",
                confidence=0.7,
                breaking_risk=BreakingRisk.NONE,
                category="high-complexity",
            ))
            fixes.append(Fix(
                description="Use early returns to reduce nesting",
                before_code="def f(x):\n    if x:\n        if y:\n            return z",
                after_code="def f(x):\n    if not x:\n        return None\n    if not y:\n        return None\n    return z",
                confidence=0.8,
                breaking_risk=BreakingRisk.LOW,
                category="high-complexity",
            ))

        if metric == "nesting":
            fixes.append(Fix(
                description="Flatten deeply nested code with early returns or guard clauses",
                before_code="if a:\n    if b:\n        if c:\n            do()",
                after_code="if not a:\n    return\nif not b:\n    return\nif c:\n    do()",
                confidence=0.75,
                breaking_risk=BreakingRisk.LOW,
                category="deep-nesting",
            ))

        return fixes


# ---------------------------------------------------------------------------
# Auto-fix: apply fix and verify syntax
# ---------------------------------------------------------------------------

def auto_fix(source: str, fix: Fix) -> Optional[str]:
    """Apply a fix to source code and verify the result is valid Python.
    Returns the fixed source or None if the fix cannot be applied."""
    if not fix.before_code or not fix.after_code:
        return None

    # Try direct replacement
    if fix.before_code in source:
        result = source.replace(fix.before_code, fix.after_code, 1)
        try:
            ast.parse(result)
            return result
        except SyntaxError:
            return None

    return None


# ---------------------------------------------------------------------------
# Main suggester
# ---------------------------------------------------------------------------

class FixSuggester:
    """Suggest fixes for detected bugs, patterns, and security issues."""

    def __init__(self) -> None:
        self._none_fixer = _NoneDerefFixer()
        self._div_fixer = _DivByZeroFixer()
        self._type_fixer = _TypeErrorFixer()
        self._index_fixer = _IndexOutOfBoundsFixer()
        self._unused_import_fixer = _UnusedImportFixer()
        self._unused_var_fixer = _UnusedVariableFixer()
        self._unreachable_fixer = _UnreachableCodeFixer()
        self._uninit_fixer = _UninitializedVarFixer()
        self._security_fixer = _SecurityFixer()
        self._pattern_fixer = _PatternFixer()
        self._complexity_fixer = _ComplexityFixer()

    def suggest(self, bug: Bug) -> List[Fix]:
        """Suggest fixes for a detected bug."""
        cat = bug.category

        if cat == "none-dereference":
            return self._none_fixer.suggest(bug)
        if cat == "division-by-zero":
            return self._div_fixer.suggest(bug)
        if cat == "type-error":
            return self._type_fixer.suggest(bug)
        if cat == "index-out-of-bounds":
            return self._index_fixer.suggest(bug)
        if cat == "unused-import":
            return self._unused_import_fixer.suggest(bug)
        if cat == "unused-variable":
            return self._unused_var_fixer.suggest(bug)
        if cat == "unreachable-code":
            return self._unreachable_fixer.suggest(bug)
        if cat == "uninitialized-variable":
            return self._uninit_fixer.suggest(bug)

        return [Fix(
            description=f"Review and fix: {bug.message}",
            before_code="# current code",
            after_code="# fixed code",
            confidence=0.3,
            breaking_risk=BreakingRisk.MEDIUM,
            category=cat,
        )]

    def suggest_for_security(self, vuln_type: str,
                              source_kind: str = "",
                              sink_kind: str = "") -> List[Fix]:
        """Suggest fixes for security vulnerabilities."""
        return self._security_fixer.suggest(vuln_type, source_kind, sink_kind)

    def suggest_for_pattern(self, pattern_type: str) -> List[Fix]:
        """Suggest fixes for anti-patterns."""
        return self._pattern_fixer.suggest(pattern_type)

    def suggest_for_complexity(self, metric: str, value: float) -> List[Fix]:
        """Suggest fixes for complexity issues."""
        return self._complexity_fixer.suggest(metric, value)
