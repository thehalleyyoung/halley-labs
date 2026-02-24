#!/usr/bin/env python3
"""
Full benchmark: test all analysis tools against code snippets
with known bugs/clean code, measure precision/recall/F1.
"""

import ast
import json
import os
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Add parent paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.python_ast_analyzer import PythonASTAnalyzer, Severity
from src.type_inference_engine_v2 import TypeInferencer, INT, FLOAT, STR, BOOL, NONE_TYPE, List_, Dict_, Optional_
from src.taint_tracker import TaintTracker, VulnerabilityType
from src.complexity_analyzer import ComplexityAnalyzer, cyclomatic_complexity
from src.pattern_detector import PatternDetector
from src.refactoring_engine import RefactoringEngine, Refactoring, RefactoringKind
from src.fix_suggester import FixSuggester


# ===================================================================
# Buggy code snippets (30)
# ===================================================================

BUGGY_SNIPPETS: List[Tuple[str, str, str]] = [
    # (code, expected_bug_category, description)

    # -- None dereference (1-5) --
    ("x = None\nprint(x.strip())", "none-dereference", "attr on None"),
    ("val = {}.get('key')\ny = val.upper()", "none-dereference", "dict.get may return None"),
    ("import re\nm = re.search('a', 'b')\nprint(m.group())", "none-dereference", "search returns None"),
    ("result = None\nresult.append(1)", "none-dereference", "append on None"),
    ("obj = None\nobj.save()", "none-dereference", "method call on None"),

    # -- Division by zero (6-10) --
    ("x = 10 / 0", "division-by-zero", "literal div by zero"),
    ("y = 0\nresult = 100 / y", "division-by-zero", "var set to 0 then divide"),
    ("n = 0\nq = 50 // n", "division-by-zero", "floor div by zero"),
    ("z = 0\nr = 42 % z", "division-by-zero", "modulo by zero"),
    ("d = 0\nval = 1.0 / d", "division-by-zero", "float div by zero var"),

    # -- Index out of bounds (11-13) --
    ("items = [1, 2, 3]\nprint(items[5])", "index-out-of-bounds", "index beyond list length"),
    ("a = [10]\nprint(a[3])", "index-out-of-bounds", "single-element list, idx 3"),
    ("arr = [1, 2]\nprint(arr[-5])", "index-out-of-bounds", "negative index out of range"),

    # -- Type errors (14-17) --
    ('result = "hello" + 42', "type-error", "str + int"),
    ("x = 5\ny = x()", "type-error", "calling an int"),
    ('a = 10\nb = "text"\nc = a + b', "type-error", "int + str"),
    ('s = "hi"\nn = 3\nresult = s + n', "type-error", "str + int variable"),

    # -- Uninitialized variables (18-20) --
    ("def f():\n    print(undefined_var)", "uninitialized-variable", "use before assignment"),
    ("def g():\n    y = x + 1\n    x = 5", "uninitialized-variable", "use before define in func"),
    ("def h():\n    return mystery", "uninitialized-variable", "return uninitialized"),

    # -- Unreachable code (21-24) --
    ("def f():\n    return 1\n    print('dead')", "unreachable-code", "code after return"),
    ("def g():\n    raise ValueError()\n    x = 5", "unreachable-code", "code after raise"),
    ("for i in range(10):\n    break\n    print(i)", "unreachable-code", "code after break"),
    ("while True:\n    continue\n    x = 1", "unreachable-code", "code after continue"),

    # -- Unused imports (25-27) --
    ("import os\nimport sys\nprint('hello')", "unused-import", "unused os and sys"),
    ("from collections import Counter\nx = 1", "unused-import", "unused Counter"),
    ("import json\nimport re\nprint(json.dumps({}))", "unused-import", "unused re"),

    # -- Unused variables (28-29) --
    ("unused_var = 42\nprint('done')", "unused-variable", "assigned never used"),
    ("temp = compute_something()\nresult = 10", "unused-variable", "temp never used"),

    # -- Bare except (30) --  (detected by pattern detector, not AST analyzer directly)
    ("try:\n    x = 1\nexcept:\n    pass", "bare-except", "bare except clause"),
]


# ===================================================================
# Clean code snippets (30)
# ===================================================================

CLEAN_SNIPPETS: List[Tuple[str, str]] = [
    # (code, description)
    ("x = 5\nprint(x + 3)", "simple arithmetic"),
    ("def add(a, b):\n    return a + b\nprint(add(1, 2))", "simple function"),
    ("items = [1, 2, 3]\nfor i in items:\n    print(i)", "simple loop"),
    ("if True:\n    x = 1\nelse:\n    x = 2\nprint(x)", "if-else"),
    ("name = 'Alice'\nprint(f'Hello {name}')", "f-string"),
    ("d = {'a': 1}\nval = d.get('a', 0)\nprint(val + 1)", "dict get with default"),
    ("x = 10\nif x != 0:\n    y = 100 / x", "guarded division"),
    ("items = [1, 2, 3]\nif len(items) > 2:\n    print(items[2])", "bounds-checked access"),
    ("import os\nprint(os.getcwd())", "used import"),
    ("def greet(name):\n    return f'Hi {name}'", "clean function"),
    ("class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y", "clean class"),
    ("try:\n    x = int('5')\nexcept ValueError:\n    x = 0", "specific except"),
    ("result = sum(range(10))", "builtin use"),
    ("pairs = [(1, 'a'), (2, 'b')]\nfor k, v in pairs:\n    print(k, v)", "tuple unpack"),
    ("x = None\nif x is not None:\n    print(x.strip())", "guarded None access"),
    ("values = [1, 2, 3]\nresult = [v * 2 for v in values]", "list comprehension"),
    ("a, b = 1, 2\nprint(a + b)", "tuple assignment"),
    ("def safe_div(a, b):\n    if b == 0:\n        return 0\n    return a / b", "safe division"),
    ("import math\nprint(math.sqrt(16))", "used math import"),
    ("data = {'x': 1}\nfor key in data:\n    print(key)", "dict iteration"),
    ("class Animal:\n    def __init__(self, name):\n        self.name = name\n    def speak(self):\n        return self.name", "simple class"),
    ("numbers = [3, 1, 2]\nnumbers.sort()\nprint(numbers)", "list sort"),
    ("text = 'hello world'\nwords = text.split()\nprint(len(words))", "string ops"),
    ("for i in range(5):\n    if i % 2 == 0:\n        print(i)", "loop with condition"),
    ("def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)", "recursion"),
    ("with open(__file__) as f:\n    content = f.read()", "context manager"),
    ("x = [1, 2, 3]\ny = x[:]\nprint(y)", "list copy"),
    ("s = set([1, 2, 3, 2, 1])\nprint(len(s))", "set usage"),
    ("d = {i: i**2 for i in range(5)}\nprint(d)", "dict comprehension"),
    ("a = lambda x: x + 1\nprint(a(5))", "lambda"),
]


# ===================================================================
# Type inference test cases (20)
# ===================================================================

TYPE_INFERENCE_CASES: List[Tuple[str, str, str]] = [
    # (code, variable_name, expected_type_str_contains)
    ("x = 42", "x", "Int"),
    ("y = 3.14", "y", "Float"),
    ('s = "hello"', "s", "Str"),
    ("b = True", "b", "Bool"),
    ("n = None", "n", "None"),
    ("lst = [1, 2, 3]", "lst", "List"),
    ("d = {'a': 1}", "d", "Dict"),
    ("t = (1, 2)", "t", "Tuple"),
    ("def f(x):\n    return x + 1", "f", "Callable"),
    ("result = 10 / 3", "result", "Float"),
    ("items = [1, 2]\nfirst = items[0]", "first", "Int"),
    ('words = "hello world".split()', "words", "List"),
    ('length = len("test")', "length", "Int"),
    ("def add(a, b):\n    return a + b", "add", "Callable"),
    ('greeting = "hi " + "there"', "greeting", "Str"),
    ("flag = 1 > 0", "flag", "Bool"),
    ("x = 5\ny = x * 2", "y", "Int"),
    ("pairs = [(1, 'a')]", "pairs", "List"),
    ('msg = f"value={42}"', "msg", "Str"),
    ("def noop():\n    pass", "noop", "Callable"),
]


# ===================================================================
# Taint tracking test cases (10)
# ===================================================================

TAINT_CASES: List[Tuple[str, bool, str]] = [
    # (code, should_have_flow, description)
    (
        'user = input("name: ")\nos.system("echo " + user)',
        True, "command injection via input"
    ),
    (
        'data = input("query: ")\ncursor.execute("SELECT * FROM t WHERE id=" + data)',
        True, "SQL injection via input"
    ),
    (
        'path = input("file: ")\nopen(path)',
        True, "path traversal via input"
    ),
    (
        'url = input("url: ")\nrequests.get(url)',
        True, "SSRF via input"
    ),
    (
        'code = input("expr: ")\neval(code)',
        True, "code injection via eval"
    ),
    (
        'x = input("x: ")\ny = int(x)\nprint(y)',
        False, "sanitized by int()"
    ),
    (
        'x = 42\nprint(x)',
        False, "no taint source"
    ),
    (
        'name = "Alice"\nprint(f"Hello {name}")',
        False, "literal string, not tainted"
    ),
    (
        'import os\nval = os.environ.get("KEY")\nexec(val)',
        True, "env var to exec"
    ),
    (
        'user = input("cmd: ")\nsubprocess.run(user)',
        True, "command injection via subprocess"
    ),
]


# ===================================================================
# Complexity test cases (10)
# ===================================================================

COMPLEXITY_CASES: List[Tuple[str, int, str]] = [
    # (code, expected_cyclomatic, description)
    ("def f():\n    return 1", 1, "trivial function"),
    ("def f(x):\n    if x:\n        return 1\n    return 0", 2, "single if"),
    ("def f(x):\n    if x > 0:\n        return 1\n    elif x < 0:\n        return -1\n    return 0", 3, "if-elif"),
    ("def f(x):\n    for i in range(x):\n        if i > 5:\n            break", 3, "for + if + break path"),
    ("def f(x):\n    if x and y:\n        return 1", 3, "boolean and"),
    ("def f(x):\n    while x > 0:\n        x -= 1", 2, "while loop"),
    ("def f(x):\n    try:\n        return 1/x\n    except ZeroDivisionError:\n        return 0", 2, "try-except"),
    ("def f(x):\n    return 1 if x else 0", 2, "ternary"),
    ("def f(lst):\n    return [x for x in lst if x > 0]", 2, "list comp with filter"),
    ("def f(a, b, c):\n    if a:\n        if b:\n            if c:\n                return 1", 4, "nested ifs"),
]


# ===================================================================
# Pattern detection test cases (10)
# ===================================================================

PATTERN_CASES: List[Tuple[str, str, str]] = [
    # (code, expected_pattern_type, description)
    ("def f(items=[]):\n    items.append(1)\n    return items", "mutable-default-argument", "mutable default"),
    ("try:\n    x = 1\nexcept:\n    pass", "bare-except", "bare except"),
    ("if x == None:\n    pass", "bad-comparison", "== None"),
    ("if y == True:\n    pass", "bad-comparison", "== True"),
    ('result = "hello %s" % name', "unsafe-string-format", "% formatting"),
    ("sum([x*x for x in range(10)])", "unnecessary-list-comprehension", "list comp in sum"),
    (
        "def f(a, b, c, d, e, f_param):\n    return a",
        "primitive-obsession", "too many params"
    ),
    (
        "\n".join([f"    def method_{i}(self): pass" for i in range(20)]),
        "god-class",
        "too many methods"
    ),
    ("if x == False:\n    pass", "bad-comparison", "== False"),
    ("all([x > 0 for x in items])", "unnecessary-list-comprehension", "list comp in all"),
]

# Fix god class snippet - needs class wrapper
PATTERN_CASES[7] = (
    "class Big:\n" + "\n".join([f"    def method_{i}(self): pass" for i in range(20)]),
    "god-class",
    "too many methods"
)


# ===================================================================
# Refactoring test cases (5)
# ===================================================================

REFACTORING_CASES: List[Tuple[Refactoring, str, str]] = [
    # (refactoring, source, description)
    (
        Refactoring(kind=RefactoringKind.RENAME, target="old_name", new_name="new_name"),
        "old_name = 42\nprint(old_name)",
        "rename variable"
    ),
    (
        Refactoring(kind=RefactoringKind.SIMPLIFY_BOOLEAN),
        "result = not not x",
        "simplify double negation"
    ),
    (
        Refactoring(kind=RefactoringKind.REMOVE_DEAD_CODE),
        "import os\nimport sys\nprint(os.getcwd())",
        "remove unused import"
    ),
    (
        Refactoring(kind=RefactoringKind.CONVERT_TO_FSTRING),
        'msg = "hello {}".format(name)',
        "convert to f-string"
    ),
    (
        Refactoring(kind=RefactoringKind.SIMPLIFY_BOOLEAN),
        "if x == True:\n    pass",
        "simplify == True"
    ),
]


# ===================================================================
# Benchmark runner
# ===================================================================

@dataclass
class BenchmarkResults:
    # AST analyzer
    ast_true_positives: int = 0
    ast_false_negatives: int = 0
    ast_false_positives: int = 0
    ast_true_negatives: int = 0
    ast_precision: float = 0.0
    ast_recall: float = 0.0
    ast_f1: float = 0.0

    # Type inference
    type_inference_correct: int = 0
    type_inference_total: int = 0
    type_inference_accuracy: float = 0.0

    # Taint tracking
    taint_true_positives: int = 0
    taint_false_negatives: int = 0
    taint_false_positives: int = 0
    taint_true_negatives: int = 0
    taint_precision: float = 0.0
    taint_recall: float = 0.0

    # Complexity
    complexity_correct: int = 0
    complexity_total: int = 0
    complexity_accuracy: float = 0.0

    # Patterns
    pattern_correct: int = 0
    pattern_total: int = 0
    pattern_accuracy: float = 0.0

    # Refactoring
    refactoring_valid: int = 0
    refactoring_total: int = 0
    refactoring_success_rate: float = 0.0

    # Timing
    total_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ast_analysis": {
                "true_positives": self.ast_true_positives,
                "false_negatives": self.ast_false_negatives,
                "false_positives": self.ast_false_positives,
                "true_negatives": self.ast_true_negatives,
                "precision": round(self.ast_precision, 4),
                "recall": round(self.ast_recall, 4),
                "f1": round(self.ast_f1, 4),
            },
            "type_inference": {
                "correct": self.type_inference_correct,
                "total": self.type_inference_total,
                "accuracy": round(self.type_inference_accuracy, 4),
            },
            "taint_tracking": {
                "true_positives": self.taint_true_positives,
                "false_negatives": self.taint_false_negatives,
                "false_positives": self.taint_false_positives,
                "true_negatives": self.taint_true_negatives,
                "precision": round(self.taint_precision, 4),
                "recall": round(self.taint_recall, 4),
            },
            "complexity": {
                "correct": self.complexity_correct,
                "total": self.complexity_total,
                "accuracy": round(self.complexity_accuracy, 4),
            },
            "patterns": {
                "correct": self.pattern_correct,
                "total": self.pattern_total,
                "accuracy": round(self.pattern_accuracy, 4),
            },
            "refactoring": {
                "valid": self.refactoring_valid,
                "total": self.refactoring_total,
                "success_rate": round(self.refactoring_success_rate, 4),
            },
            "total_time_seconds": round(self.total_time_s, 3),
        }


def run_benchmark() -> BenchmarkResults:
    results = BenchmarkResults()
    start = time.time()

    print("=" * 70)
    print("FULL BENCHMARK: Bug Detection & Analysis Tools")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. AST Analyzer: Buggy snippets (true positives)
    # ---------------------------------------------------------------
    print("\n--- AST Analyzer: Buggy Snippets (30) ---")
    analyzer = PythonASTAnalyzer()
    pattern_detector = PatternDetector()

    for i, (code, expected_cat, desc) in enumerate(BUGGY_SNIPPETS):
        result = analyzer.analyze_source(code)
        patterns = pattern_detector.detect(code)

        # Check if any bug matches the expected category
        found = any(b.category == expected_cat for b in result.bugs)
        # Also check pattern detector for bare-except etc.
        if not found:
            found = any(p.type == expected_cat for p in patterns)

        if found:
            results.ast_true_positives += 1
            status = "✓ TP"
        else:
            results.ast_false_negatives += 1
            status = "✗ FN"
            # Show what was found instead
            found_cats = [b.category for b in result.bugs] + [p.type for p in patterns]

        print(f"  [{status}] #{i+1:2d} {desc}: expected={expected_cat}"
              + ("" if found else f"  got={found_cats}"))

    # ---------------------------------------------------------------
    # 2. AST Analyzer: Clean snippets (true negatives)
    # ---------------------------------------------------------------
    print("\n--- AST Analyzer: Clean Snippets (30) ---")
    error_categories = {
        "none-dereference", "division-by-zero", "index-out-of-bounds",
        "type-error", "syntax-error",
    }

    for i, (code, desc) in enumerate(CLEAN_SNIPPETS):
        result = analyzer.analyze_source(code)
        # Only count error-level bugs as false positives
        errors = [b for b in result.bugs if b.category in error_categories]

        if not errors:
            results.ast_true_negatives += 1
            status = "✓ TN"
        else:
            results.ast_false_positives += 1
            status = "✗ FP"

        if status == "✗ FP":
            print(f"  [{status}] #{i+1:2d} {desc}: false alarm={[b.category for b in errors]}")
        else:
            print(f"  [{status}] #{i+1:2d} {desc}")

    # Compute precision/recall/F1
    tp = results.ast_true_positives
    fp = results.ast_false_positives
    fn = results.ast_false_negatives
    results.ast_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    results.ast_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if results.ast_precision + results.ast_recall > 0:
        results.ast_f1 = (2 * results.ast_precision * results.ast_recall /
                          (results.ast_precision + results.ast_recall))

    print(f"\n  Precision: {results.ast_precision:.2%}")
    print(f"  Recall:    {results.ast_recall:.2%}")
    print(f"  F1:        {results.ast_f1:.2%}")

    # ---------------------------------------------------------------
    # 3. Type Inference (20 cases)
    # ---------------------------------------------------------------
    print("\n--- Type Inference (20 cases) ---")
    inferencer = TypeInferencer()
    results.type_inference_total = len(TYPE_INFERENCE_CASES)

    for i, (code, var_name, expected_contains) in enumerate(TYPE_INFERENCE_CASES):
        env = inferencer.infer(code)
        inferred = env.get(var_name)
        type_str = str(inferred)

        if expected_contains in type_str:
            results.type_inference_correct += 1
            status = "✓"
        else:
            status = "✗"

        print(f"  [{status}] #{i+1:2d} {var_name}: inferred={type_str}, expected contains '{expected_contains}'")

    results.type_inference_accuracy = (results.type_inference_correct /
                                        results.type_inference_total if results.type_inference_total else 0)
    print(f"\n  Accuracy: {results.type_inference_accuracy:.2%}")

    # ---------------------------------------------------------------
    # 4. Taint Tracking (10 cases)
    # ---------------------------------------------------------------
    print("\n--- Taint Tracking (10 cases) ---")
    tracker = TaintTracker()

    for i, (code, should_have_flow, desc) in enumerate(TAINT_CASES):
        flows = tracker.analyze(code)
        has_flow = len(flows) > 0

        if should_have_flow and has_flow:
            results.taint_true_positives += 1
            status = "✓ TP"
        elif not should_have_flow and not has_flow:
            results.taint_true_negatives += 1
            status = "✓ TN"
        elif should_have_flow and not has_flow:
            results.taint_false_negatives += 1
            status = "✗ FN"
        else:
            results.taint_false_positives += 1
            status = "✗ FP"

        print(f"  [{status}] #{i+1:2d} {desc}")

    t_tp = results.taint_true_positives
    t_fp = results.taint_false_positives
    t_fn = results.taint_false_negatives
    results.taint_precision = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0.0
    results.taint_recall = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0.0
    print(f"\n  Precision: {results.taint_precision:.2%}")
    print(f"  Recall:    {results.taint_recall:.2%}")

    # ---------------------------------------------------------------
    # 5. Complexity Analysis (10 cases)
    # ---------------------------------------------------------------
    print("\n--- Complexity Analysis (10 cases) ---")
    comp_analyzer = ComplexityAnalyzer()
    results.complexity_total = len(COMPLEXITY_CASES)

    for i, (code, expected_cc, desc) in enumerate(COMPLEXITY_CASES):
        report = comp_analyzer.analyze(code)
        actual_cc = report.per_function[0].cyclomatic if report.per_function else 0

        if actual_cc == expected_cc:
            results.complexity_correct += 1
            status = "✓"
        else:
            status = "✗"

        print(f"  [{status}] #{i+1:2d} {desc}: expected={expected_cc}, actual={actual_cc}")

    results.complexity_accuracy = (results.complexity_correct /
                                    results.complexity_total if results.complexity_total else 0)
    print(f"\n  Accuracy: {results.complexity_accuracy:.2%}")

    # ---------------------------------------------------------------
    # 6. Pattern Detection (10 cases)
    # ---------------------------------------------------------------
    print("\n--- Pattern Detection (10 cases) ---")
    results.pattern_total = len(PATTERN_CASES)

    for i, (code, expected_type, desc) in enumerate(PATTERN_CASES):
        pd = PatternDetector()
        patterns = pd.detect(code)
        found = any(p.type == expected_type for p in patterns)

        if found:
            results.pattern_correct += 1
            status = "✓"
        else:
            status = "✗"
            found_types = [p.type for p in patterns]

        print(f"  [{status}] #{i+1:2d} {desc}: expected={expected_type}"
              + ("" if found else f"  got={found_types}"))

    results.pattern_accuracy = (results.pattern_correct /
                                 results.pattern_total if results.pattern_total else 0)
    print(f"\n  Accuracy: {results.pattern_accuracy:.2%}")

    # ---------------------------------------------------------------
    # 7. Refactoring (5 cases)
    # ---------------------------------------------------------------
    print("\n--- Refactoring Engine (5 cases) ---")
    engine = RefactoringEngine()
    results.refactoring_total = len(REFACTORING_CASES)

    for i, (refactoring, source, desc) in enumerate(REFACTORING_CASES):
        result = engine.refactor(source, refactoring)

        # Verify output is valid Python
        is_valid = False
        if result.success and result.new_source:
            try:
                ast.parse(result.new_source)
                is_valid = True
            except SyntaxError:
                pass

        if is_valid:
            results.refactoring_valid += 1
            status = "✓"
        else:
            status = "✗"

        print(f"  [{status}] #{i+1:2d} {desc}: success={result.success}, valid_python={is_valid}")

    results.refactoring_success_rate = (results.refactoring_valid /
                                         results.refactoring_total if results.refactoring_total else 0)
    print(f"\n  Success rate: {results.refactoring_success_rate:.2%}")

    # ---------------------------------------------------------------
    # 8. Fix Suggester (quick check)
    # ---------------------------------------------------------------
    print("\n--- Fix Suggester (spot check) ---")
    suggester = FixSuggester()
    # Test on first few bugs
    for code, cat, desc in BUGGY_SNIPPETS[:5]:
        result = analyzer.analyze_source(code)
        for bug in result.bugs:
            fixes = suggester.suggest(bug)
            if fixes:
                print(f"  ✓ {bug.category}: {len(fixes)} fix(es) suggested")
                break

    # ---------------------------------------------------------------
    # Verify ALL snippets parse
    # ---------------------------------------------------------------
    print("\n--- Syntax Validation ---")
    all_valid = True
    all_codes = (
        [(c, d) for c, _, d in BUGGY_SNIPPETS] +
        [(c, d) for c, d in CLEAN_SNIPPETS] +
        [(c, d) for c, _, d in TAINT_CASES] +
        [(c, d) for c, _, d in COMPLEXITY_CASES] +
        [(c, d) for c, _, d in PATTERN_CASES] +
        [(s, d) for _, s, d in REFACTORING_CASES]
    )
    for code, desc in all_codes:
        try:
            ast.parse(code)
        except SyntaxError as e:
            print(f"  ✗ SYNTAX ERROR in '{desc}': {e}")
            all_valid = False

    if all_valid:
        print(f"  ✓ All {len(all_codes)} code snippets are syntactically valid Python")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    results.total_time_s = time.time() - start

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  AST Bug Detection:   P={results.ast_precision:.2%}  R={results.ast_recall:.2%}  F1={results.ast_f1:.2%}")
    print(f"  Type Inference:      {results.type_inference_correct}/{results.type_inference_total} = {results.type_inference_accuracy:.2%}")
    print(f"  Taint Tracking:      P={results.taint_precision:.2%}  R={results.taint_recall:.2%}")
    print(f"  Complexity Analysis: {results.complexity_correct}/{results.complexity_total} = {results.complexity_accuracy:.2%}")
    print(f"  Pattern Detection:   {results.pattern_correct}/{results.pattern_total} = {results.pattern_accuracy:.2%}")
    print(f"  Refactoring:         {results.refactoring_valid}/{results.refactoring_total} = {results.refactoring_success_rate:.2%}")
    print(f"  Total Time:          {results.total_time_s:.2f}s")
    print("=" * 70)

    return results


def main() -> None:
    results = run_benchmark()

    # Write results JSON
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "guard_harvest_benchmark_results.json")

    with open(output_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)

    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
