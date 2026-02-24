"""
Compare guard-harvesting analyzer against mypy and pyright on the benchmark suite.

Runs all three tools on experiments/benchmarks/benchmark_suite.py, maps reported
issues to per-function results, and computes precision / recall / F1 for each tool.
Results are saved to experiments/results/RW_tool_comparison.json.
"""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.real_analyzer import analyze_source

BENCHMARK_PATH = ROOT / "experiments" / "benchmarks" / "benchmark_suite.py"
RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "RW_tool_comparison.json"


# ── helpers ──────────────────────────────────────────────────────────────

def load_ground_truth():
    """Import GROUND_TRUTH dict from the benchmark module."""
    source = BENCHMARK_PATH.read_text()
    ns: dict = {}
    exec(compile(source, str(BENCHMARK_PATH), "exec"), ns)
    return ns["GROUND_TRUTH"]


def build_function_line_map() -> dict[str, tuple[int, int]]:
    """Return {func_name: (start_line, end_line)} for every top-level def."""
    source = BENCHMARK_PATH.read_text()
    tree = ast.parse(source)
    mapping: dict[str, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            mapping[node.name] = (node.lineno, node.end_lineno or node.lineno)
    return mapping


def line_to_function(line: int, func_map: dict[str, tuple[int, int]]) -> str | None:
    """Map a source line number to the enclosing function name."""
    for name, (start, end) in func_map.items():
        if start <= line <= end:
            return name
    return None


# ── tool installation ────────────────────────────────────────────────────

def ensure_tool(module_name: str, pip_name: str | None = None) -> bool:
    """Ensure a tool is importable / runnable; pip-install if missing.
    Returns True if the tool is available."""
    pip_name = pip_name or module_name
    try:
        subprocess.run(
            [sys.executable, "-m", module_name, "--version"],
            capture_output=True, check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    print(f"  Installing {pip_name} …")
    for extra_flags in [[], ["--user"], ["--break-system-packages"]]:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name, "-q"] + extra_flags,
                capture_output=True, check=True,
            )
            return True
        except subprocess.CalledProcessError:
            continue
    print(f"  WARNING: could not install {pip_name} – will skip")
    return False


# ── mypy ─────────────────────────────────────────────────────────────────

def run_mypy(func_map: dict) -> dict[str, bool]:
    """Run mypy, return {func_name: detected} for every benchmark function."""
    detected: dict[str, bool] = {fn: False for fn in func_map}
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "mypy",
                "--no-error-summary",
                "--show-column-numbers",
                str(BENCHMARK_PATH),
            ],
            capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:
        print("  mypy not found – skipping")
        return detected

    for line in result.stdout.splitlines():
        # format: file.py:LINE:COL: error: message  [code]
        m = re.match(r".+?:(\d+):\d+:\s*(error|warning)", line)
        if m:
            lineno = int(m.group(1))
            fn = line_to_function(lineno, func_map)
            if fn is not None:
                detected[fn] = True
    return detected


# ── pyright ──────────────────────────────────────────────────────────────

def run_pyright(func_map: dict) -> dict[str, bool]:
    """Run pyright with --outputjson, return {func_name: detected}."""
    detected: dict[str, bool] = {fn: False for fn in func_map}
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pyright",
                "--outputjson",
                str(BENCHMARK_PATH),
            ],
            capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:
        print("  pyright not found – skipping")
        return detected

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("  pyright produced non-JSON output – skipping")
        return detected

    diagnostics = data.get("generalDiagnostics", [])
    for diag in diagnostics:
        lineno = diag.get("range", {}).get("start", {}).get("line", -1)
        # pyright lines are 0-indexed
        lineno += 1
        fn = line_to_function(lineno, func_map)
        if fn is not None:
            detected[fn] = True
    return detected


# ── our analyzer ─────────────────────────────────────────────────────────

def run_our_analyzer(func_map: dict) -> dict[str, bool]:
    """Run our guard-harvesting analyzer, return {func_name: detected}."""
    detected: dict[str, bool] = {fn: False for fn in func_map}
    source = BENCHMARK_PATH.read_text()
    file_result = analyze_source(source, str(BENCHMARK_PATH))
    for fr in file_result.function_results:
        if fr.name in detected and fr.bugs:
            detected[fr.name] = True
    return detected


# ── metrics ──────────────────────────────────────────────────────────────

def compute_metrics(ground_truth: dict, detected: dict[str, bool]) -> dict:
    tp = fp = fn_ = tn = 0
    for name, info in ground_truth.items():
        has_bug = info["has_bug"]
        tool_says = detected.get(name, False)
        if has_bug and tool_says:
            tp += 1
        elif has_bug and not tool_says:
            fn_ += 1
        elif not has_bug and tool_says:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn_) if (tp + fn_) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "TP": tp, "FP": fp, "FN": fn_, "TN": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "F1": round(f1, 4),
    }


# ── main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Tool Comparison: Our Analyzer vs mypy vs pyright")
    print("=" * 60)

    ground_truth = load_ground_truth()
    func_map = build_function_line_map()
    total = len(ground_truth)
    buggy = sum(1 for v in ground_truth.values() if v["has_bug"])
    safe = total - buggy
    print(f"\nBenchmark: {total} functions ({buggy} buggy, {safe} safe)\n")

    # Ensure tools are installed
    print("Checking tool availability …")
    mypy_available = ensure_tool("mypy")
    pyright_available = ensure_tool("pyright")

    # Run each tool
    print("\n[1/3] Running mypy …")
    mypy_det = run_mypy(func_map) if mypy_available else {fn: False for fn in func_map}
    if not mypy_available:
        print("  (skipped – not installed)")

    print("[2/3] Running pyright …")
    pyright_det = run_pyright(func_map) if pyright_available else {fn: False for fn in func_map}
    if not pyright_available:
        print("  (skipped – not installed)")

    print("[3/3] Running our analyzer …")
    our_det = run_our_analyzer(func_map)

    # Per-function comparison table
    per_function: list[dict] = []
    for name, info in ground_truth.items():
        per_function.append({
            "function": name,
            "ground_truth_has_bug": info["has_bug"],
            "category": info["category"],
            "our_tool_detected": our_det.get(name, False),
            "mypy_detected": mypy_det.get(name, False),
            "pyright_detected": pyright_det.get(name, False),
        })

    # Compute metrics
    our_metrics = compute_metrics(ground_truth, our_det)
    mypy_metrics = compute_metrics(ground_truth, mypy_det)
    pyright_metrics = compute_metrics(ground_truth, pyright_det)

    results = {
        "benchmark_functions": total,
        "buggy_functions": buggy,
        "safe_functions": safe,
        "metrics": {
            "our_analyzer": our_metrics,
            "mypy": mypy_metrics,
            "pyright": pyright_metrics,
        },
        "per_function": per_function,
    }

    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {OUTPUT_PATH}\n")

    # Summary table
    print(f"{'Tool':<20} {'Prec':>7} {'Recall':>7} {'F1':>7}  (TP/FP/FN/TN)")
    print("-" * 70)
    for tool_name, m in [("Our Analyzer", our_metrics),
                         ("mypy", mypy_metrics),
                         ("pyright", pyright_metrics)]:
        print(f"{tool_name:<20} {m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{m['F1']:>7.4f}  ({m['TP']}/{m['FP']}/{m['FN']}/{m['TN']})")
    print()


if __name__ == "__main__":
    main()
