"""Experiment: Guard-Aware Unified Analysis.

Demonstrates that guard-harvested abstract state reduces false positives
in taint, security, and concurrency analyses.  Compares:
  (A) Standalone auxiliary analysis (no guard context)
  (B) Guard-aware auxiliary analysis (with guard-harvested context)

This is the key experiment showing that guards are a UNIFYING ABSTRACTION
that improves ALL forms of analysis.
"""
import json
import os
import sys
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from guard_aware_analysis import (
    unified_analysis, guard_aware_taint_analysis,
    guard_aware_security_analysis, guard_aware_concurrency_analysis,
    GuardAwareAnalyzerEngine,
)
from taint_tracker import TaintTracker
from web_security import owasp_top_10_scan
from concurrency_bugs import detect_concurrency_bugs


# ── Test cases: code with guards that should eliminate FPs ────────────

TAINT_TEST_CASES = [
    {
        "name": "int_validated_sql",
        "description": "User input validated as int before SQL query",
        "source": '''
def get_user(user_id_str):
    user_id = int(user_id_str)  # sanitized by int()
    if isinstance(user_id, int):
        import sqlite3
        conn = sqlite3.connect("db.sqlite")
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
        return cursor.fetchone()
''',
        "has_guard": True,
        "expected_raw_taint": True,
        "expected_guard_eliminates": False,  # int() is already a sanitizer in taint
    },
    {
        "name": "none_checked_before_use",
        "description": "Variable checked for None before passing to sink",
        "source": '''
def process_input(data):
    val = data.get("cmd")
    if val is not None:
        if isinstance(val, str) and val.isalnum():
            import os
            os.system(val)
''',
        "has_guard": True,
        "expected_raw_taint": True,
        "expected_guard_eliminates": False,  # guards don't eliminate cmd injection
    },
    {
        "name": "unguarded_sql_injection",
        "description": "No guard — real SQL injection",
        "source": '''
def search(query):
    import sqlite3
    conn = sqlite3.connect("db.sqlite")
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM items WHERE name = '{query}'")
    return cursor.fetchall()
''',
        "has_guard": False,
        "expected_raw_taint": True,
        "expected_guard_eliminates": False,
    },
    {
        "name": "type_guarded_path",
        "description": "Path traversal guarded by isinstance check",
        "source": '''
import os

def read_config(path):
    if isinstance(path, int):
        # path is an integer file descriptor, safe
        return os.read(path, 1024)
    return None
''',
        "has_guard": True,
        "expected_raw_taint": False,
        "expected_guard_eliminates": False,
    },
]

SECURITY_TEST_CASES = [
    {
        "name": "csrf_with_auth_decorator",
        "description": "POST handler with csrf_exempt but has login_required",
        "source": '''
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required

@login_required
@csrf_exempt
def update_profile(request):
    if request.method == "POST":
        name = request.POST.get("name")
        return HttpResponse("Updated")
''',
        "has_guard": True,
    },
    {
        "name": "xss_int_context",
        "description": "Reflected value is proven integer — no XSS risk",
        "source": '''
def show_count(request):
    count = int(request.GET.get("count", "0"))
    if isinstance(count, int):
        return f"<p>Count: {count}</p>"
''',
        "has_guard": True,
    },
    {
        "name": "unguarded_xss",
        "description": "Reflected user input without sanitization",
        "source": '''
def search_results(request):
    query = request.GET.get("q", "")
    return f"<h1>Results for: {query}</h1>"
''',
        "has_guard": False,
    },
]

CONCURRENCY_TEST_CASES = [
    {
        "name": "immutable_shared_state",
        "description": "Shared variable is proven immutable (int/str)",
        "source": '''
import threading

counter = 0
lock = threading.Lock()

def worker():
    global counter
    name = "worker"
    if isinstance(name, str):
        print(name)
''',
        "has_guard": True,
    },
    {
        "name": "unguarded_shared_mutation",
        "description": "Shared mutable state without lock",
        "source": '''
import threading

results = []

def worker(item):
    global results
    results.append(item)

threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
''',
        "has_guard": False,
    },
    {
        "name": "none_checked_shared_access",
        "description": "Shared resource checked for None before access",
        "source": '''
import threading

shared_resource = None

def worker():
    global shared_resource
    if shared_resource is not None:
        shared_resource.process()
''',
        "has_guard": True,
    },
]


def run_taint_comparison():
    """Compare taint analysis with and without guard context."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Guard-Aware Taint Analysis")
    print("=" * 70)

    results = []
    for tc in TAINT_TEST_CASES:
        print(f"\n--- {tc['name']}: {tc['description']} ---")

        # (A) Standalone taint analysis
        tracker = TaintTracker()
        raw_flows = tracker.analyze(tc["source"])
        raw_count = len(raw_flows)

        # (B) Guard-aware taint analysis
        ga_result = guard_aware_taint_analysis(tc["source"])
        filtered_count = len(ga_result.filtered_flows)
        eliminated = ga_result.eliminated_by_guards

        print(f"  Standalone: {raw_count} taint flows")
        print(f"  Guard-aware: {filtered_count} flows ({eliminated} eliminated by guards)")
        if ga_result.elimination_reasons:
            for r in ga_result.elimination_reasons:
                print(f"    → {r}")

        results.append({
            "name": tc["name"],
            "has_guard": tc["has_guard"],
            "raw_flows": raw_count,
            "filtered_flows": filtered_count,
            "eliminated": eliminated,
        })

    return results


def run_security_comparison():
    """Compare security analysis with and without guard context."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Guard-Aware Security Analysis")
    print("=" * 70)

    results = []
    for tc in SECURITY_TEST_CASES:
        print(f"\n--- {tc['name']}: {tc['description']} ---")

        # (A) Standalone security analysis
        report = owasp_top_10_scan(tc["source"])
        raw_issues = []
        for attr in ['xss', 'csrf', 'open_redirects', 'header_injections',
                     'cookie_issues', 'cors_issues', 'auth_bypasses']:
            raw_issues.extend(getattr(report, attr, []))
        raw_count = len(raw_issues)

        # (B) Guard-aware security analysis
        ga_result = guard_aware_security_analysis(tc["source"])
        filtered_count = len(ga_result.filtered_issues)
        eliminated = ga_result.eliminated_by_guards

        print(f"  Standalone: {raw_count} security issues")
        print(f"  Guard-aware: {filtered_count} issues ({eliminated} eliminated by guards)")
        if ga_result.elimination_reasons:
            for r in ga_result.elimination_reasons:
                print(f"    → {r}")

        results.append({
            "name": tc["name"],
            "has_guard": tc["has_guard"],
            "raw_issues": raw_count,
            "filtered_issues": filtered_count,
            "eliminated": eliminated,
        })

    return results


def run_concurrency_comparison():
    """Compare concurrency analysis with and without guard context."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Guard-Aware Concurrency Analysis")
    print("=" * 70)

    results = []
    for tc in CONCURRENCY_TEST_CASES:
        print(f"\n--- {tc['name']}: {tc['description']} ---")

        # (A) Standalone concurrency analysis
        report = detect_concurrency_bugs(tc["source"])
        raw_bugs = (len(report.shared_state_bugs) + len(report.race_conditions) +
                   len(report.deadlock_risks) + len(report.async_bugs))

        # (B) Guard-aware concurrency analysis
        ga_result = guard_aware_concurrency_analysis(tc["source"])
        filtered_count = len(ga_result.filtered_bugs)
        eliminated = ga_result.eliminated_by_guards

        print(f"  Standalone: {raw_bugs} concurrency bugs")
        print(f"  Guard-aware: {filtered_count} bugs ({eliminated} eliminated by guards)")
        if ga_result.elimination_reasons:
            for r in ga_result.elimination_reasons:
                print(f"    → {r}")

        results.append({
            "name": tc["name"],
            "has_guard": tc["has_guard"],
            "raw_bugs": raw_bugs,
            "filtered_bugs": filtered_count,
            "eliminated": eliminated,
        })

    return results


def run_unified_demo():
    """Run unified analysis on a comprehensive example."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Unified Guard-Aware Analysis")
    print("=" * 70)

    source = '''
import sqlite3
import threading
import os

shared_data = {}
lock = threading.Lock()

def process_request(user_input, db_path):
    """Process user request with multiple potential bug categories."""
    # Guard 1: None check
    if user_input is None:
        return {"error": "No input"}

    # Guard 2: Type check — proves user_id is int (safe for SQL)
    user_id = user_input.get("id")
    if user_id is not None and isinstance(user_id, int):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # SAFE: user_id is proven int by guard
        cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
        result = cursor.fetchone()

        # Guard 3: Length check
        items = user_input.get("items", [])
        if len(items) > 0:
            first = items[0]  # SAFE: length proven > 0

        # Guard 4: Truthiness
        name = user_input.get("name")
        if name:
            # SAFE: name is non-null and non-empty
            greeting = f"<h1>Hello, {name}</h1>"  # potential XSS but guarded

        return result

    return {"error": "Invalid input"}


def unguarded_handler(query):
    """Unguarded handler — should have bugs detected."""
    import sqlite3
    conn = sqlite3.connect("test.db")
    cursor = conn.cursor()
    # BUG: unguarded SQL injection
    cursor.execute(f"SELECT * FROM items WHERE name = '{query}'")
    return cursor.fetchall()
'''

    result = unified_analysis(source)

    print(f"\n  Core analysis:")
    print(f"    Guards harvested: {result.total_guards}")
    print(f"    Core bugs: {result.total_bugs_core}")

    if result.taint:
        print(f"\n  Taint analysis:")
        print(f"    Raw flows: {len(result.taint.raw_flows)}")
        print(f"    After guard filtering: {len(result.taint.filtered_flows)}")
        print(f"    Eliminated by guards: {result.taint.eliminated_by_guards}")

    if result.security:
        print(f"\n  Security analysis:")
        print(f"    Raw issues: {len(result.security.raw_issues)}")
        print(f"    After guard filtering: {len(result.security.filtered_issues)}")
        print(f"    Eliminated by guards: {result.security.eliminated_by_guards}")

    if result.concurrency:
        print(f"\n  Concurrency analysis:")
        print(f"    Raw bugs: {len(result.concurrency.raw_bugs)}")
        print(f"    After guard filtering: {len(result.concurrency.filtered_bugs)}")
        print(f"    Eliminated by guards: {result.concurrency.eliminated_by_guards}")

    print(f"\n  Total eliminated across all domains: {result.total_eliminated}")
    print(f"  Analysis time: {result.analysis_time_ms:.1f}ms")

    return {
        "guards_harvested": result.total_guards,
        "core_bugs": result.total_bugs_core,
        "taint_raw": len(result.taint.raw_flows) if result.taint else 0,
        "taint_filtered": len(result.taint.filtered_flows) if result.taint else 0,
        "taint_eliminated": result.taint.eliminated_by_guards if result.taint else 0,
        "security_raw": len(result.security.raw_issues) if result.security else 0,
        "security_filtered": len(result.security.filtered_issues) if result.security else 0,
        "security_eliminated": result.security.eliminated_by_guards if result.security else 0,
        "concurrency_raw": len(result.concurrency.raw_bugs) if result.concurrency else 0,
        "concurrency_filtered": len(result.concurrency.filtered_bugs) if result.concurrency else 0,
        "concurrency_eliminated": result.concurrency.eliminated_by_guards if result.concurrency else 0,
        "total_eliminated": result.total_eliminated,
        "analysis_time_ms": result.analysis_time_ms,
    }


def main():
    print("Guard-Aware Unified Analysis Experiment")
    print("=" * 70)

    all_results = {}

    all_results["taint"] = run_taint_comparison()
    all_results["security"] = run_security_comparison()
    all_results["concurrency"] = run_concurrency_comparison()
    all_results["unified"] = run_unified_demo()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_raw = 0
    total_filtered = 0
    total_eliminated = 0

    for domain in ["taint", "security", "concurrency"]:
        domain_results = all_results[domain]
        for r in domain_results:
            raw_key = [k for k in r if k.startswith("raw_")][0]
            filt_key = [k for k in r if k.startswith("filtered")][0]
            total_raw += r[raw_key]
            total_filtered += r[filt_key]
            total_eliminated += r["eliminated"]

    print(f"  Total raw issues (without guards): {total_raw}")
    print(f"  Total after guard filtering: {total_filtered}")
    print(f"  Total eliminated by guard harvesting: {total_eliminated}")
    if total_raw > 0:
        print(f"  False positive reduction: {total_eliminated/total_raw*100:.1f}%")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), 'results',
                               'guard_aware_unified.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
