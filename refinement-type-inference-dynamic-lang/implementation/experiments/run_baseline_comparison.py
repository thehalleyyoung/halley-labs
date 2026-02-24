#!/usr/bin/env python3
"""
Baseline comparison: TensorGuard vs Pyright vs MyPy on PyTorch shape bugs.

For each benchmark model, runs all three tools and classifies results as
TP / FP / TN / FN, then computes precision, recall, and F1 per tool.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

# Ensure we can import from the project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.shape_bug_benchmarks import ALL_BENCHMARKS, BenchmarkModel
from src.pipeline import analyze_python_source
from src.model_checker import verify_model

PYRIGHT = "/opt/homebrew/bin/pyright"
MYPY = "/opt/homebrew/bin/mypy"


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    detected_bug: bool
    raw_errors: List[str] = field(default_factory=list)
    time_ms: float = 0.0


@dataclass
class BenchmarkResult:
    name: str
    has_bug: bool
    bug_description: str
    tensorguard: Optional[ToolResult] = None
    pyright: Optional[ToolResult] = None
    mypy: Optional[ToolResult] = None


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_tensorguard(model: BenchmarkModel) -> ToolResult:
    """Run TensorGuard pipeline + model checker on a benchmark model."""
    errors: List[str] = []
    detected = False
    t0 = time.monotonic()

    try:
        # Pipeline analysis (Z3-enhanced for nn.Module sources)
        analysis = analyze_python_source(model.source, filename=f"{model.name}.py")
        if analysis.total_bugs > 0:
            detected = True
            for s in analysis.summaries:
                for b in getattr(s, "bugs", []):
                    errors.append(str(b))

        # Constraint-based verification
        vr = verify_model(model.source)
        if not vr.safe:
            detected = True
            if vr.counterexample:
                errors.append(vr.counterexample.pretty())
            errors.extend(vr.errors)
    except Exception as exc:
        errors.append(f"TensorGuard error: {exc}")

    elapsed = (time.monotonic() - t0) * 1000
    return ToolResult(detected_bug=detected, raw_errors=errors, time_ms=elapsed)


def _write_tmp(source: str, name: str) -> str:
    """Write source to a temp file and return the path."""
    d = tempfile.mkdtemp(prefix="shapebench_")
    p = os.path.join(d, f"{name}.py")
    with open(p, "w") as f:
        f.write(source)
    return p


def run_pyright(model: BenchmarkModel) -> ToolResult:
    """Run Pyright on a benchmark model."""
    path = _write_tmp(model.source, model.name)
    t0 = time.monotonic()
    errors: List[str] = []
    detected = False

    try:
        proc = subprocess.run(
            [PYRIGHT, "--outputjson", path],
            capture_output=True, text=True, timeout=60,
        )
        elapsed = (time.monotonic() - t0) * 1000
        try:
            data = json.loads(proc.stdout)
            diagnostics = data.get("generalDiagnostics", [])
            for d in diagnostics:
                sev = d.get("severity", "")
                msg = d.get("message", "")
                if sev in ("error", "warning"):
                    errors.append(f"[{sev}] {msg}")
                    detected = True
        except json.JSONDecodeError:
            # Fall back to stderr / stdout text
            if proc.returncode != 0 and proc.stderr.strip():
                errors.append(proc.stderr.strip())
    except subprocess.TimeoutExpired:
        elapsed = (time.monotonic() - t0) * 1000
        errors.append("Pyright timed out")
    finally:
        os.unlink(path)
        os.rmdir(os.path.dirname(path))

    return ToolResult(detected_bug=detected, raw_errors=errors, time_ms=elapsed)


def run_mypy(model: BenchmarkModel) -> ToolResult:
    """Run MyPy on a benchmark model."""
    path = _write_tmp(model.source, model.name)
    t0 = time.monotonic()
    errors: List[str] = []
    detected = False

    try:
        proc = subprocess.run(
            [MYPY, "--no-error-summary", "--show-column-numbers",
             "--ignore-missing-imports", path],
            capture_output=True, text=True, timeout=60,
        )
        elapsed = (time.monotonic() - t0) * 1000
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            if ": error:" in line or ": warning:" in line:
                errors.append(line)
                detected = True
    except subprocess.TimeoutExpired:
        elapsed = (time.monotonic() - t0) * 1000
        errors.append("MyPy timed out")
    finally:
        os.unlink(path)
        os.rmdir(os.path.dirname(path))

    return ToolResult(detected_bug=detected, raw_errors=errors, time_ms=elapsed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: List[BenchmarkResult], tool: str) -> Dict:
    tp = fp = tn = fn = 0
    for r in results:
        tr: Optional[ToolResult] = getattr(r, tool)
        if tr is None:
            continue
        if r.has_bug:
            if tr.detected_bug:
                tp += 1
            else:
                fn += 1
        else:
            if tr.detected_bug:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results: List[BenchmarkResult] = []

    for model in ALL_BENCHMARKS:
        tag = "BUGGY" if model.has_bug else "CLEAN"
        print(f"[{tag}] {model.name} ... ", end="", flush=True)

        br = BenchmarkResult(
            name=model.name,
            has_bug=model.has_bug,
            bug_description=model.bug_description,
        )

        br.tensorguard = run_tensorguard(model)
        br.pyright = run_pyright(model)
        br.mypy = run_mypy(model)

        lp = "✓" if br.tensorguard.detected_bug else "✗"
        pr = "✓" if br.pyright.detected_bug else "✗"
        mp = "✓" if br.mypy.detected_bug else "✗"
        print(f"TensorGuard={lp}  Pyright={pr}  MyPy={mp}")

        results.append(br)

    # Compute per-tool metrics
    metrics = {}
    for tool in ("tensorguard", "pyright", "mypy"):
        metrics[tool] = compute_metrics(results, tool)

    # Pretty-print summary
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 70)
    for tool, m in metrics.items():
        print(f"\n  {tool:10s}  Precision={m['precision']:.2f}  "
              f"Recall={m['recall']:.2f}  F1={m['f1']:.2f}  "
              f"(TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']})")

    # Serialize and save
    out = {
        "benchmarks": [
            {
                "name": r.name,
                "has_bug": r.has_bug,
                "bug_description": r.bug_description,
                "tensorguard": asdict(r.tensorguard) if r.tensorguard else None,
                "pyright": asdict(r.pyright) if r.pyright else None,
                "mypy": asdict(r.mypy) if r.mypy else None,
            }
            for r in results
        ],
        "metrics": metrics,
    }

    out_path = Path(__file__).resolve().parent / "baseline_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
