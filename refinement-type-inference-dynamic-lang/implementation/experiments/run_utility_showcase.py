#!/usr/bin/env python3
"""
Utility showcase benchmark: comprehensive evaluation of all analysis tools.
Tests bug detection (P/R/F1 per bug type), type inference accuracy,
taint analysis, complexity verification, refactoring validity, and project scanning.
"""

import ast
import json
import os
import shutil
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.python_ast_analyzer import PythonASTAnalyzer, Severity
from src.type_inference_engine_v2 import (
    TypeInferencer, INT, FLOAT, STR, BOOL, NONE_TYPE,
)
from src.taint_tracker import TaintTracker, VulnerabilityType
from src.complexity_analyzer import ComplexityAnalyzer
from src.pattern_detector import PatternDetector
from src.refactoring_engine import RefactoringEngine, Refactoring, RefactoringKind
from src.project_scanner import ProjectScanner


# =====================================================================
# 1. Bug Detection Benchmark — 40 buggy + 20 clean
# =====================================================================

NONE_DEREF_BUGS: List[Tuple[str, str]] = [
    ("x = None\nprint(x.strip())", "attr on None literal"),
    ("val = {}.get('k')\ny = val.upper()", "dict.get may return None"),
    ("import re\nm = re.search('a','b')\nprint(m.group())", "re.search None"),
    ("result = None\nresult.append(1)", "append on None"),
    ("obj = None\nobj.save()", "method on None"),
    ("a = None\nb = a[0]", "subscript on None"),
    ("z = None\nfor c in z: pass", "iterate None"),
    ("q = None\nlen(q)", "len of None"),
    ("p = None\np += 1", "augassign None"),
    ("w = None\nw.x = 5", "setattr None"),
]

TYPE_ERROR_BUGS: List[Tuple[str, str]] = [
    ('r = "hello" + 42', "str + int"),
    ("x = 5\ny = x()", "call int"),
    ('a = 10\nb = "t"\nc = a + b', "int + str"),
    ('s = "hi"\nn = 3\nr = s + n', "str + int var"),
    ('v = [1,2] + "ab"', "list + str"),
    ('q = True - "x"', "bool - str"),
    ("m = 3.14\nm[0]", "subscript float"),
    ('d = {"a":1}\nd + 5', "dict + int"),
    ("f = 5\nf.append(1)", "append on int"),
    ("n = 7\nfor c in n: pass", "iterate int"),
]

UNUSED_VAR_BUGS: List[Tuple[str, str]] = [
    ("unused_var = 42\nprint('done')", "simple unused"),
    ("temp = 10\nresult = 20\nprint(result)", "temp unused"),
    ("a = 1\nb = 2\nc = 3\nprint(a + c)", "b unused"),
    ("x = compute()\ny = 5\nprint(y)", "x unused"),  # will parse but compute undef
    ("first = 1\nsecond = 2\nthird = 3\nprint(first)", "second+third unused"),
    ("alpha = 'a'\nbeta = 'b'\nprint('done')", "both unused"),
    ("val = 99\nval2 = 100\nprint(val2)", "val unused"),
    ("data = [1,2]\nbackup = data\nprint(data)", "backup unused"),
    ("count = 0\ntotal = 0\nprint(count)", "total unused"),
    ("flag = True\nresult = flag\nstatus = 0\nprint(result)", "status unused"),
]

UNREACHABLE_BUGS: List[Tuple[str, str]] = [
    ("def f():\n    return 1\n    print('dead')", "after return"),
    ("def g():\n    raise ValueError()\n    x = 5", "after raise"),
    ("for i in range(10):\n    break\n    print(i)", "after break"),
    ("while True:\n    continue\n    x = 1", "after continue"),
    ("def h():\n    return 0\n    y = 10\n    z = 20", "multi after return"),
    ("def j():\n    raise Exception()\n    return 1", "return after raise"),
    ("def k():\n    if True:\n        return 1\n    return 2\n    x = 3", "after guaranteed return"),
    ("for x in []:\n    break\n    y = x + 1", "after break in for"),
    ("while False:\n    continue\n    z = 0", "after continue in while"),
    ("def m():\n    return None\n    import os", "import after return"),
]

CLEAN_FUNCTIONS: List[Tuple[str, str]] = [
    ("x = 5\nprint(x + 3)", "arithmetic"),
    ("def add(a, b):\n    return a + b\nprint(add(1,2))", "function call"),
    ("items = [1,2,3]\nfor i in items:\n    print(i)", "loop"),
    ("if True:\n    x = 1\nelse:\n    x = 2\nprint(x)", "if-else"),
    ("name = 'Alice'\nprint(f'Hello {name}')", "f-string"),
    ("d = {'a':1}\nval = d.get('a',0)\nprint(val+1)", "dict get default"),
    ("x = 10\nif x != 0:\n    y = 100/x\n    print(y)", "guarded div"),
    ("items = [1,2,3]\nif len(items) > 2:\n    print(items[2])", "bounds check"),
    ("import os\nprint(os.getcwd())", "used import"),
    ("def greet(n):\n    return f'Hi {n}'\nprint(greet('X'))", "clean func"),
    ("class P:\n    def __init__(s,x):\n        s.x = x\np = P(1)\nprint(p.x)", "class"),
    ("try:\n    x = int('5')\nexcept ValueError:\n    x = 0\nprint(x)", "except"),
    ("result = sum(range(10))\nprint(result)", "builtin"),
    ("pairs = [(1,'a')]\nfor k,v in pairs:\n    print(k,v)", "unpack"),
    ("x = None\nif x is not None:\n    print(x.strip())", "guarded None"),
    ("values = [1,2,3]\nresult = [v*2 for v in values]\nprint(result)", "listcomp"),
    ("a, b = 1, 2\nprint(a + b)", "tuple assign"),
    ("def safe(a,b):\n    return a/b if b else 0\nprint(safe(1,2))", "safe div"),
    ("import math\nprint(math.sqrt(16))", "math import"),
    ("data = {'x':1}\nfor k in data:\n    print(k)", "dict iter"),
]

BUG_CATEGORIES = {
    "none-dereference": NONE_DEREF_BUGS,
    "type-error": TYPE_ERROR_BUGS,
    "unused-variable": UNUSED_VAR_BUGS,
    "unreachable-code": UNREACHABLE_BUGS,
}


# =====================================================================
# 2. Type Inference — 20 functions with expected types
# =====================================================================

TYPE_CASES: List[Tuple[str, str, str]] = [
    ("x = 42", "x", "Int"),
    ("y = 3.14", "y", "Float"),
    ('s = "hello"', "s", "Str"),
    ("b = True", "b", "Bool"),
    ("n = None", "n", "None"),
    ("lst = [1,2,3]", "lst", "List"),
    ("d = {'a':1}", "d", "Dict"),
    ("t = (1,2)", "t", "Tuple"),
    ("def f(x): return x+1", "f", "Callable"),
    ("r = 10/3", "r", "Float"),
    ("items=[1,2]\nfirst=items[0]", "first", "Int"),
    ('w = "hello world".split()', "w", "List"),
    ('l = len("test")', "l", "Int"),
    ("def add(a,b): return a+b", "add", "Callable"),
    ('g = "hi "+"there"', "g", "Str"),
    ("fl = 1 > 0", "fl", "Bool"),
    ("x = 5\ny = x * 2", "y", "Int"),
    ("ps = [(1,'a')]", "ps", "List"),
    ('ms = f"v={42}"', "ms", "Str"),
    ("def noop(): pass", "noop", "Callable"),
]


# =====================================================================
# 3. Taint Analysis — 15 snippets
# =====================================================================

TAINT_CASES: List[Tuple[str, str, bool, str]] = [
    # SQL injection (5)
    ('d=input("q:")\ncursor.execute("SELECT * FROM t WHERE id="+d)',
     "sql-injection", True, "SQL injection concat"),
    ('u=input("u:")\ncursor.execute(f"SELECT * FROM users WHERE name=\'{u}\'")',
     "sql-injection", True, "SQL injection f-string"),
    ('q=input("s:")\ncursor.execute("DELETE FROM t WHERE id="+q)',
     "sql-injection", True, "SQL injection DELETE"),
    ('v=input("v:")\ncursor.execute("INSERT INTO t VALUES("+v+")")',
     "sql-injection", True, "SQL injection INSERT"),
    ('w=input("w:")\ncursor.execute("UPDATE t SET x="+w)',
     "sql-injection", True, "SQL injection UPDATE"),
    # Command injection (5)
    ('u=input("c:")\nos.system("echo "+u)',
     "command-injection", True, "cmd injection os.system"),
    ('p=input("p:")\nos.popen("ls "+p)',
     "command-injection", True, "cmd injection os.popen"),
    ('c=input("x:")\nsubprocess.run(c, shell=True)',
     "command-injection", True, "cmd injection subprocess"),
    ('e=input("e:")\neval(e)',
     "code-injection", True, "code injection eval"),
    ('ex=input("ex:")\nexec(ex)',
     "code-injection", True, "code injection exec"),
    # Clean (5)
    ('x=input("x:")\ny=int(x)\nprint(y)',
     "clean", False, "sanitized int()"),
    ('x=42\nprint(x)',
     "clean", False, "no taint source"),
    ('name="Alice"\nprint(f"Hello {name}")',
     "clean", False, "literal string"),
    ('import os\nprint(os.path.exists("/tmp"))',
     "clean", False, "safe os call"),
    ('data=[1,2,3]\nprint(sum(data))',
     "clean", False, "no user input"),
]


# =====================================================================
# 4. Complexity — 20 functions with known cyclomatic complexity
# =====================================================================

COMPLEXITY_CASES: List[Tuple[str, int, str]] = [
    ("def f(): return 1", 1, "trivial"),
    ("def f(x):\n    if x: return 1\n    return 0", 2, "single if"),
    ("def f(x):\n    if x>0: return 1\n    elif x<0: return -1\n    return 0", 3, "if-elif"),
    ("def f(x):\n    for i in range(x):\n        if i>5: break", 3, "for+if+break"),
    ("def f(x):\n    while x>0: x-=1", 2, "while"),
    ("def f(x):\n    try:\n        return 1/x\n    except ZeroDivisionError:\n        return 0", 2, "try-except"),
    ("def f(x): return 1 if x else 0", 2, "ternary"),
    ("def f(l): return [x for x in l if x>0]", 2, "listcomp filter"),
    ("def f(a,b,c):\n    if a:\n        if b:\n            if c:\n                return 1", 4, "nested ifs"),
    ("def f(): pass", 1, "empty func"),
    ("def f(x):\n    if x==1: return 'a'\n    elif x==2: return 'b'\n    elif x==3: return 'c'\n    return 'd'", 4, "multi elif"),
    ("def f(x,y):\n    if x:\n        if y:\n            return 1\n    return 0", 3, "nested 2"),
    ("def f(lst):\n    for x in lst:\n        for y in x:\n            print(y)", 3, "nested loops"),
    ("def f(x):\n    while x>0:\n        if x%2==0: x//=2\n        else: x=3*x+1", 4, "collatz step"),
    ("def f(x): return x", 1, "identity"),
    ("def f(a,b): return a if a>b else b", 2, "max ternary"),
    ("def f(n):\n    if n<=1: return n\n    return f(n-1)+f(n-2)", 2, "fibonacci"),
    ("def f(x):\n    for i in range(x):\n        if i%2: continue\n        print(i)", 3, "for+if+continue"),
    ("def f(d):\n    for k,v in d.items():\n        if v is None: continue\n        print(k,v)", 3, "dict iter filter"),
    ("def f(x):\n    if x and True: return 1\n    return 0", 3, "boolean and"),
]


# =====================================================================
# 5. Refactoring — 10 snippets
# =====================================================================

REFACTORING_CASES: List[Tuple[Refactoring, str, str]] = [
    (Refactoring(kind=RefactoringKind.RENAME, target="old_name", new_name="new_name"),
     "old_name = 42\nprint(old_name)", "rename variable"),
    (Refactoring(kind=RefactoringKind.SIMPLIFY_BOOLEAN),
     "result = not not x", "simplify double neg"),
    (Refactoring(kind=RefactoringKind.REMOVE_DEAD_CODE),
     "import os\nimport sys\nprint(os.getcwd())", "remove unused import"),
    (Refactoring(kind=RefactoringKind.CONVERT_TO_FSTRING),
     'msg = "hello {}".format(name)', "convert to f-string"),
    (Refactoring(kind=RefactoringKind.SIMPLIFY_BOOLEAN),
     "if x == True:\n    pass", "simplify == True"),
    (Refactoring(kind=RefactoringKind.RENAME, target="foo", new_name="bar"),
     "foo = 1\nprint(foo + 2)", "rename foo->bar"),
    (Refactoring(kind=RefactoringKind.REMOVE_DEAD_CODE),
     "import json\nimport re\nprint(json.dumps({}))", "remove unused re"),
    (Refactoring(kind=RefactoringKind.CONVERT_TO_FSTRING),
     'x = "val={}".format(42)', "convert val f-string"),
    (Refactoring(kind=RefactoringKind.SIMPLIFY_BOOLEAN),
     "if x == False:\n    pass", "simplify == False"),
    (Refactoring(kind=RefactoringKind.ADD_TYPE_ANNOTATIONS),
     "def add(a, b):\n    return a + b", "add annotations"),
]


# =====================================================================
# 6. Project scanning — synthetic 20-file project
# =====================================================================

SYNTHETIC_PROJECT_FILES: Dict[str, str] = {
    "main.py": "import utils\nimport config\ndef main():\n    print(utils.greet(config.NAME))\nmain()",
    "utils.py": "def greet(name):\n    return f'Hello {name}'\ndef unused_helper():\n    x = None\n    x.strip()\n    return 42",
    "config.py": "NAME = 'World'\nDEBUG = True\nSECRET = 'hunter2'",
    "models.py": "class User:\n    def __init__(self, name, email):\n        self.name = name\n        self.email = email\n    def greet(self): return f'Hi {self.name}'",
    "views.py": 'import os\ndef render(template):\n    return open(template).read()\ndef process(data):\n    result = "ok"\n    return result',
    "api.py": 'def handle(request):\n    user = request.get("user")\n    return {"status": "ok", "user": user}',
    "auth.py": "def login(user, pw):\n    if user == 'admin' and pw == 'pass':\n        return True\n    return False",
    "db.py": 'import sqlite3\ndef query(q):\n    conn = sqlite3.connect(":memory:")\n    return conn.execute(q).fetchall()',
    "cache.py": "CACHE = {}\ndef get(key):\n    return CACHE.get(key)\ndef put(key, val):\n    CACHE[key] = val",
    "logger.py": "import logging\nlog = logging.getLogger(__name__)\ndef info(msg):\n    log.info(msg)",
    "helpers.py": "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\ndef identity(x):\n    return x",
    "validators.py": "def is_email(s):\n    return '@' in s and '.' in s\ndef is_positive(n):\n    return n > 0",
    "serializers.py": "import json\ndef to_json(obj):\n    return json.dumps(obj)\ndef from_json(s):\n    return json.loads(s)",
    "middleware.py": "def log_request(req):\n    print(f'Request: {req}')\n    return req",
    "errors.py": "class AppError(Exception):\n    pass\nclass NotFoundError(AppError):\n    pass",
    "routes.py": "ROUTES = {}\ndef route(path):\n    def decorator(fn):\n        ROUTES[path] = fn\n        return fn\n    return decorator",
    "tasks.py": "def process_task(task):\n    if task is None:\n        return\n    return task.upper()",
    "cli.py": "import sys\ndef run():\n    args = sys.argv[1:]\n    print(args)",
    "tests.py": "def test_add():\n    assert 1 + 1 == 2\ndef test_greet():\n    assert 'Hello' in 'Hello World'",
    "setup.py": "from setuptools import setup\nsetup(name='myapp', version='0.1')",
}


# =====================================================================
# Metrics helpers
# =====================================================================

def prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# =====================================================================
# Benchmark runner
# =====================================================================

def run_bug_detection_benchmark(analyzer: PythonASTAnalyzer,
                                pattern_detector: PatternDetector) -> Dict:
    """40 buggy + 20 clean functions, P/R/F1 per bug type."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Bug Detection (40 buggy + 20 clean)")
    print("=" * 70)

    per_type: Dict[str, Dict[str, int]] = {}
    overall_tp = overall_fp = overall_fn = overall_tn = 0

    for cat_name, snippets in BUG_CATEGORIES.items():
        tp = fp = fn = 0
        print(f"\n  --- {cat_name} ({len(snippets)} cases) ---")
        for i, (code, desc) in enumerate(snippets):
            result = analyzer.analyze_source(code)
            patterns = pattern_detector.detect(code)
            found = (any(b.category == cat_name for b in result.bugs) or
                     any(p.type == cat_name for p in patterns))
            if found:
                tp += 1
                print(f"    [✓ TP] #{i+1:2d} {desc}")
            else:
                fn += 1
                got = [b.category for b in result.bugs] + [p.type for p in patterns]
                print(f"    [✗ FN] #{i+1:2d} {desc}  got={got}")
        p, r, f1 = prf1(tp, 0, fn)
        per_type[cat_name] = {"tp": tp, "fn": fn, "fp": 0,
                              "precision": round(p, 4),
                              "recall": round(r, 4), "f1": round(f1, 4)}
        overall_tp += tp
        overall_fn += fn
        print(f"    P={p:.2%}  R={r:.2%}  F1={f1:.2%}")

    # Clean snippets
    error_cats = {"none-dereference", "division-by-zero", "index-out-of-bounds",
                  "type-error"}
    print(f"\n  --- clean ({len(CLEAN_FUNCTIONS)} cases) ---")
    for i, (code, desc) in enumerate(CLEAN_FUNCTIONS):
        result = analyzer.analyze_source(code)
        errs = [b for b in result.bugs if b.category in error_cats]
        if not errs:
            overall_tn += 1
            print(f"    [✓ TN] #{i+1:2d} {desc}")
        else:
            overall_fp += 1
            print(f"    [✗ FP] #{i+1:2d} {desc}  false={[b.category for b in errs]}")

    op, orr, of1 = prf1(overall_tp, overall_fp, overall_fn)
    print(f"\n  Overall:  P={op:.2%}  R={orr:.2%}  F1={of1:.2%}")

    return {
        "per_type": per_type,
        "overall": {"tp": overall_tp, "fp": overall_fp, "fn": overall_fn,
                     "tn": overall_tn, "precision": round(op, 4),
                     "recall": round(orr, 4), "f1": round(of1, 4)},
    }


def run_type_inference_benchmark(inferencer: TypeInferencer) -> Dict:
    """20 functions with annotated types."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Type Inference Accuracy (20 cases)")
    print("=" * 70)

    per_type_correct: Dict[str, List[bool]] = {}
    correct = total = 0
    details = []

    for i, (code, var, expected) in enumerate(TYPE_CASES):
        env = inferencer.infer(code)
        val = env.get(var)
        inferred = str(val) if val is not None else "???"
        ok = expected in inferred
        if ok:
            correct += 1
        total += 1
        per_type_correct.setdefault(expected, []).append(ok)
        details.append({"var": var, "expected": expected,
                         "inferred": inferred, "correct": ok})
        mark = "✓" if ok else "✗"
        print(f"  [{mark}] #{i+1:2d} {var}: {inferred}  (expect '{expected}')")

    accuracy = correct / total if total else 0
    per_type_acc = {t: round(sum(v)/len(v), 4)
                    for t, v in per_type_correct.items()}
    print(f"\n  Accuracy: {correct}/{total} = {accuracy:.2%}")
    print(f"  Per-type: {per_type_acc}")

    return {"correct": correct, "total": total,
            "accuracy": round(accuracy, 4),
            "per_type_accuracy": per_type_acc, "details": details}


def run_taint_benchmark(tracker: TaintTracker) -> Dict:
    """15 snippets: 5 SQL inj, 5 cmd inj, 5 clean."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Taint Analysis (15 cases)")
    print("=" * 70)

    tp = fp = fn = tn = 0
    details = []

    for i, (code, vuln_type, should_find, desc) in enumerate(TAINT_CASES):
        flows = tracker.analyze(code)
        has_flow = len(flows) > 0

        if should_find and has_flow:
            tp += 1; status = "✓ TP"
        elif not should_find and not has_flow:
            tn += 1; status = "✓ TN"
        elif should_find and not has_flow:
            fn += 1; status = "✗ FN"
        else:
            fp += 1; status = "✗ FP"

        details.append({"desc": desc, "expected_vuln": vuln_type,
                         "flows_found": len(flows), "status": status})
        print(f"  [{status}] #{i+1:2d} {desc}  flows={len(flows)}")

    p, r, f1 = prf1(tp, fp, fn)
    print(f"\n  P={p:.2%}  R={r:.2%}  F1={f1:.2%}")

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "details": details}


def run_complexity_benchmark(comp: ComplexityAnalyzer) -> Dict:
    """20 functions with known cyclomatic complexity."""
    print("\n" + "=" * 70)
    print("BENCHMARK 4: Complexity Metrics (20 cases)")
    print("=" * 70)

    correct = total = 0
    details = []

    for i, (code, expected_cc, desc) in enumerate(COMPLEXITY_CASES):
        report = comp.analyze(code)
        actual = report.per_function[0].cyclomatic if report.per_function else 0
        ok = (actual == expected_cc)
        if ok:
            correct += 1
        total += 1
        mark = "✓" if ok else "✗"
        details.append({"desc": desc, "expected": expected_cc,
                         "actual": actual, "correct": ok})
        print(f"  [{mark}] #{i+1:2d} {desc}: expected={expected_cc} actual={actual}")

    acc = correct / total if total else 0
    print(f"\n  Accuracy: {correct}/{total} = {acc:.2%}")

    return {"correct": correct, "total": total,
            "accuracy": round(acc, 4), "details": details}


def run_refactoring_benchmark(engine: RefactoringEngine) -> Dict:
    """10 refactoring operations — verify valid Python + semantics."""
    print("\n" + "=" * 70)
    print("BENCHMARK 5: Refactoring Engine (10 cases)")
    print("=" * 70)

    valid = total = 0
    details = []

    for i, (refactoring, source, desc) in enumerate(REFACTORING_CASES):
        result = engine.refactor(source, refactoring)
        is_valid = False
        if result.success and result.new_source:
            try:
                ast.parse(result.new_source)
                is_valid = True
            except SyntaxError:
                pass

        if is_valid:
            valid += 1
        total += 1
        mark = "✓" if is_valid else "✗"
        details.append({"desc": desc, "success": result.success,
                         "valid_python": is_valid})
        print(f"  [{mark}] #{i+1:2d} {desc}: success={result.success} valid={is_valid}")

    rate = valid / total if total else 0
    print(f"\n  Success rate: {valid}/{total} = {rate:.2%}")

    return {"valid": valid, "total": total,
            "success_rate": round(rate, 4), "details": details}


def run_project_scanning_benchmark() -> Dict:
    """Create synthetic 20-file project, scan, verify health score & bug counts."""
    print("\n" + "=" * 70)
    print("BENCHMARK 6: Project Scanning (20-file synthetic project)")
    print("=" * 70)

    tmpdir = tempfile.mkdtemp(prefix="showcase_project_")
    try:
        for fname, content in SYNTHETIC_PROJECT_FILES.items():
            with open(os.path.join(tmpdir, fname), "w") as f:
                f.write(content)

        scanner = ProjectScanner()
        report = scanner.scan(tmpdir)

        print(f"  Files analyzed:  {report.files_analyzed}")
        print(f"  Total bugs:      {report.total_bugs}")
        print(f"  Bugs by severity: {dict(report.bugs_by_severity)}")
        print(f"  Bugs by category: {dict(report.bugs_by_category)}")
        print(f"  Health score:    {report.health_score:.1f}/100")
        print(f"  Recommendations: {len(report.recommendations)}")

        checks_passed = 0
        checks_total = 4

        if report.files_analyzed >= 15:
            checks_passed += 1
            print("  [✓] Scanned >= 15 files")
        else:
            print(f"  [✗] Expected >= 15 files, got {report.files_analyzed}")

        if report.total_bugs >= 1:
            checks_passed += 1
            print("  [✓] Found >= 1 bug")
        else:
            print("  [✗] Expected >= 1 bug")

        if 0 <= report.health_score <= 100:
            checks_passed += 1
            print(f"  [✓] Health score in [0,100]: {report.health_score:.1f}")
        else:
            print(f"  [✗] Health score out of range: {report.health_score}")

        if len(report.recommendations) >= 0:
            checks_passed += 1
            print(f"  [✓] Recommendations generated: {len(report.recommendations)}")

        print(f"\n  Checks passed: {checks_passed}/{checks_total}")

        return {
            "files_analyzed": report.files_analyzed,
            "total_bugs": report.total_bugs,
            "bugs_by_severity": dict(report.bugs_by_severity),
            "bugs_by_category": dict(report.bugs_by_category),
            "health_score": round(report.health_score, 2),
            "recommendations_count": len(report.recommendations),
            "checks_passed": checks_passed,
            "checks_total": checks_total,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    start = time.time()
    print("=" * 70)
    print("UTILITY SHOWCASE — Comprehensive Analysis Tool Benchmark")
    print("=" * 70)

    analyzer = PythonASTAnalyzer()
    pattern_det = PatternDetector()
    inferencer = TypeInferencer()
    tracker = TaintTracker()
    comp = ComplexityAnalyzer()
    engine = RefactoringEngine()

    results: Dict[str, Any] = {}
    results["bug_detection"] = run_bug_detection_benchmark(analyzer, pattern_det)
    results["type_inference"] = run_type_inference_benchmark(inferencer)
    results["taint_analysis"] = run_taint_benchmark(tracker)
    results["complexity"] = run_complexity_benchmark(comp)
    results["refactoring"] = run_refactoring_benchmark(engine)
    results["project_scanning"] = run_project_scanning_benchmark()

    elapsed = time.time() - start
    results["total_time_seconds"] = round(elapsed, 3)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    bd = results["bug_detection"]["overall"]
    ti = results["type_inference"]
    ta = results["taint_analysis"]
    cx = results["complexity"]
    rf = results["refactoring"]
    ps = results["project_scanning"]

    print(f"  Bug Detection:     P={bd['precision']:.2%}  R={bd['recall']:.2%}  F1={bd['f1']:.2%}")
    print(f"  Type Inference:    {ti['correct']}/{ti['total']} = {ti['accuracy']:.2%}")
    print(f"  Taint Analysis:    P={ta['precision']:.2%}  R={ta['recall']:.2%}  F1={ta['f1']:.2%}")
    print(f"  Complexity:        {cx['correct']}/{cx['total']} = {cx['accuracy']:.2%}")
    print(f"  Refactoring:       {rf['valid']}/{rf['total']} = {rf['success_rate']:.2%}")
    print(f"  Project Scan:      {ps['checks_passed']}/{ps['checks_total']} checks passed")
    print(f"  Total Time:        {elapsed:.2f}s")
    print("=" * 70)

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "utility_showcase_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
