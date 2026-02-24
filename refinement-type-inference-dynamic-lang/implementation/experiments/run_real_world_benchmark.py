#!/usr/bin/env python3
"""
Run GuardHarvest, Mypy, Pyright, and Pylint on the real-world bug benchmark.
Collects per-function detection results and produces summary statistics.
"""
import ast
import json
import subprocess
import sys
import os
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# Add src to path for GuardHarvest
IMPL_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = IMPL_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

BUGS_DIR = Path(__file__).resolve().parent / "real_world_bugs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Ground truth: which functions have bugs and what category
# Format: (filename, function_name) -> (has_bug: bool, category: str, description: str)
GROUND_TRUTH = {
    # null_deref_bugs.py
    ("null_deref_bugs.py", "pandas_get_column"): (True, "null_dereference", "dict.get returns None"),
    ("null_deref_bugs.py", "requests_json_parse"): (True, "null_dereference", "json() may return None"),
    ("null_deref_bugs.py", "flask_get_param"): (True, "null_dereference", "args.get returns None"),
    ("null_deref_bugs.py", "django_first_user"): (True, "null_dereference", ".first() returns None"),
    ("null_deref_bugs.py", "config_lookup"): (True, "null_dereference", "dict.get returns None"),
    ("null_deref_bugs.py", "parse_video_id"): (True, "null_dereference", "re.match returns None"),
    ("null_deref_bugs.py", "get_db_connection"): (True, "null_dereference", "environ.get returns None"),
    ("null_deref_bugs.py", "extract_title"): (True, "null_dereference", "find returns None"),
    ("null_deref_bugs.py", "find_first_match"): (True, "null_dereference", "next with None default"),
    ("null_deref_bugs.py", "parse_api_response"): (True, "null_dereference", "json.loads null"),
    ("null_deref_bugs.py", "find_project_root"): (True, "null_dereference", "function returns None"),
    ("null_deref_bugs.py", "process_headers"): (True, "null_dereference", "pop with None default"),
    ("null_deref_bugs.py", "get_model_name"): (True, "null_dereference", "getattr None default"),
    ("null_deref_bugs.py", "safe_get_column"): (False, "null_dereference", "guarded with is not None"),
    ("null_deref_bugs.py", "safe_parse"): (False, "null_dereference", "guarded with isinstance"),
    ("null_deref_bugs.py", "safe_match"): (False, "null_dereference", "early return guard"),
    ("null_deref_bugs.py", "tornado_get_cookie"): (True, "null_dereference", "get returns None"),
    ("null_deref_bugs.py", "get_user_role"): (True, "null_dereference", "one_or_none returns None"),
    # type_error_bugs.py
    ("type_error_bugs.py", "format_progress"): (True, "type_error", "str + int concat"),
    ("type_error_bugs.py", "apply_transform"): (True, "type_error", "calling non-callable"),
    ("type_error_bugs.py", "process_dependencies"): (True, "type_error", "iterating non-iterable"),
    ("type_error_bugs.py", "build_url"): (True, "type_error", "int in string concat"),
    ("type_error_bugs.py", "format_score"): (True, "type_error", "format with wrong type"),
    ("type_error_bugs.py", "compute_ratio"): (False, "type_error", "bool/int division is legal"),
    ("type_error_bugs.py", "safe_concat"): (False, "type_error", "isinstance guard"),
    ("type_error_bugs.py", "process_input"): (False, "type_error", "isinstance narrowing"),
    ("type_error_bugs.py", "get_series_value"): (True, "null_dereference", "name may be None"),
    ("type_error_bugs.py", "get_display_name"): (True, "null_dereference", "may return None"),
    ("type_error_bugs.py", "normalize"): (True, "null_dereference", "returns None on empty"),
    ("type_error_bugs.py", "decode_response"): (False, "type_error", "works on both str/bytes"),
    ("type_error_bugs.py", "safe_normalize"): (False, "null_dereference", "early return guard"),
    # div_by_zero_bugs.py
    ("div_by_zero_bugs.py", "compute_average"): (True, "division_by_zero", "empty list"),
    ("div_by_zero_bugs.py", "show_progress"): (True, "division_by_zero", "total may be 0"),
    ("div_by_zero_bugs.py", "class_weight"): (True, "division_by_zero", "count may be 0"),
    ("div_by_zero_bugs.py", "normalize_features"): (True, "division_by_zero", "span may be 0"),
    ("div_by_zero_bugs.py", "calculate_fps"): (True, "division_by_zero", "time may be 0"),
    ("div_by_zero_bugs.py", "throughput"): (True, "division_by_zero", "duration may be 0"),
    ("div_by_zero_bugs.py", "safe_average"): (False, "division_by_zero", "guarded with if not"),
    ("div_by_zero_bugs.py", "safe_ratio"): (False, "division_by_zero", "guarded with == 0"),
    ("div_by_zero_bugs.py", "circular_index"): (True, "division_by_zero", "step may be 0"),
    ("div_by_zero_bugs.py", "chunk_data"): (True, "division_by_zero", "num_chunks may be 0"),
    ("div_by_zero_bugs.py", "std_dev"): (True, "division_by_zero", "n or n-1 may be 0"),
    ("div_by_zero_bugs.py", "event_rate"): (True, "division_by_zero", "delta may be 0"),
    # index_oob_bugs.py
    ("index_oob_bugs.py", "get_first_result"): (True, "index_out_of_bounds", "empty list"),
    ("index_oob_bugs.py", "parse_csv_line"): (True, "index_out_of_bounds", "insufficient parts"),
    ("index_oob_bugs.py", "get_extension"): (False, "index_out_of_bounds", "[-1] always works"),
    ("index_oob_bugs.py", "get_last_segment"): (False, "index_out_of_bounds", "[-1] always works"),
    ("index_oob_bugs.py", "get_pairs"): (True, "index_out_of_bounds", "off-by-one i+1"),
    ("index_oob_bugs.py", "lookup_status_code"): (True, "index_out_of_bounds", "key not in dict"),
    ("index_oob_bugs.py", "safe_first"): (False, "index_out_of_bounds", "truthiness guard"),
    ("index_oob_bugs.py", "safe_csv_parse"): (False, "index_out_of_bounds", "len guard"),
    ("index_oob_bugs.py", "extract_protocol"): (False, "index_out_of_bounds", "slice is safe"),
    ("index_oob_bugs.py", "parse_args"): (True, "index_out_of_bounds", "argv too short"),
    ("index_oob_bugs.py", "get_cell"): (True, "index_out_of_bounds", "row/col OOB"),
    ("index_oob_bugs.py", "process_stack"): (True, "index_out_of_bounds", "pop from empty"),
}


def run_guardharvest(filepath: Path) -> Dict[str, List[str]]:
    """Run GuardHarvest on a file, return per-function bug messages."""
    from real_analyzer import analyze_source
    try:
        source = filepath.read_text()
        result = analyze_source(source, filename=str(filepath), use_cegar=True)
    except Exception as e:
        return {"_error": [str(e)]}

    func_bugs: Dict[str, List[str]] = {}
    for fr in result.function_results:
        bugs = []
        for b in fr.bugs:
            bugs.append(f"{b.category.name}: {b.message}")
        if bugs:
            func_bugs[fr.name] = bugs
    return func_bugs


def _copy_to_tmp(filepath: Path) -> Path:
    """Copy file to /tmp to avoid package name issues with mypy/pyright."""
    import shutil
    tmp = Path("/tmp/guardharvest_bench")
    tmp.mkdir(exist_ok=True)
    dest = tmp / filepath.name
    shutil.copy2(filepath, dest)
    return dest


def run_mypy(filepath: Path) -> Dict[str, List[str]]:
    """Run mypy on a file, return per-line error messages."""
    tmp_file = _copy_to_tmp(filepath)
    try:
        result = subprocess.run(
            ["mypy", "--no-error-summary", "--no-color-output",
             "--ignore-missing-imports", str(tmp_file)],
            capture_output=True, text=True, timeout=30,
            cwd="/tmp"
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    
    errors: Dict[str, List[str]] = {}
    for line in result.stdout.strip().split("\n"):
        if ": error:" in line:
            # Extract line number: filename.py:LINE: error: msg
            parts = line.split(":")
            if len(parts) >= 3:
                try:
                    lineno = int(parts[1].strip())
                    errors.setdefault("_all", []).append(f"{lineno}:{line}")
                except ValueError:
                    errors.setdefault("_all", []).append(line)
            else:
                errors.setdefault("_all", []).append(line)
    return errors


def run_pyright(filepath: Path) -> Dict[str, List[str]]:
    """Run pyright on a file."""
    tmp_file = _copy_to_tmp(filepath)
    try:
        result = subprocess.run(
            ["pyright", "--outputjson", str(tmp_file)],
            capture_output=True, text=True, timeout=30,
            cwd="/tmp"
        )
        data = json.loads(result.stdout)
        errors: Dict[str, List[str]] = {}
        for diag in data.get("generalDiagnostics", []):
            if diag.get("severity") == "error":
                line = diag.get("range", {}).get("start", {}).get("line", 0) + 1
                msg = diag.get("message", "")
                errors.setdefault("_all", []).append(f"{line}:{msg}")
        return errors
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {}


def run_pylint(filepath: Path) -> Dict[str, List[str]]:
    """Run pylint on a file."""
    tmp_file = _copy_to_tmp(filepath)
    try:
        result = subprocess.run(
            ["pylint", "--output-format=json", "--disable=all",
             "--enable=E", str(tmp_file)],
            capture_output=True, text=True, timeout=30,
            cwd="/tmp"
        )
        data = json.loads(result.stdout) if result.stdout.strip() else []
        errors: Dict[str, List[str]] = {}
        for item in data:
            if item.get("type") == "error":
                line = item.get("line", 0)
                msg = item.get("message", "")
                errors.setdefault("_all", []).append(f"{line}:{msg}")
        return errors
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {}


def map_guardharvest_to_functions(gh_results, source: str) -> Dict[str, bool]:
    """Map GuardHarvest results to per-function detection."""
    detected = {}
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            detected[node.name] = node.name in gh_results
    return detected


def evaluate_tool(tool_func, filepath, source, ground_truth_for_file):
    """Evaluate a tool's detection on the ground truth functions."""
    results = tool_func(filepath)
    
    # For GuardHarvest: per-function results
    if tool_func == run_guardharvest:
        tp = fp = fn = tn = 0
        details = []
        for (fname, func_name), (has_bug, cat, desc) in ground_truth_for_file.items():
            detected = func_name in results
            if has_bug and detected:
                tp += 1
                details.append({"func": func_name, "result": "TP", "category": cat})
            elif has_bug and not detected:
                fn += 1
                details.append({"func": func_name, "result": "FN", "category": cat})
            elif not has_bug and detected:
                fp += 1
                details.append({"func": func_name, "result": "FP", "category": cat})
            else:
                tn += 1
                details.append({"func": func_name, "result": "TN", "category": cat})
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "details": details}
    
    # For other tools: check if any error reported (coarse)
    has_any_errors = bool(results.get("_all", []))
    tp = fp = fn = tn = 0
    details = []
    for (fname, func_name), (has_bug, cat, desc) in ground_truth_for_file.items():
        # Coarse: other tools report file-level, not per-function
        detected = has_any_errors  # We'll refine this below
        
        # Try to match line-level errors to functions
        func_detected = False
        tree = ast.parse(filepath.read_text())
        func_lines = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end = getattr(node, 'end_lineno', node.lineno + 20)
                func_lines[node.name] = (node.lineno, end)
        
        for err_msg in results.get("_all", []):
            # Line number is at the start of the error message (format: "LINE:msg")
            try:
                line_no = int(err_msg.split(":")[0].strip())
                if func_name in func_lines:
                    start, end = func_lines[func_name]
                    if start <= line_no <= end:
                        func_detected = True
                        break
            except (ValueError, IndexError):
                continue
        
        if has_bug and func_detected:
            tp += 1
            details.append({"func": func_name, "result": "TP", "category": cat})
        elif has_bug and not func_detected:
            fn += 1
            details.append({"func": func_name, "result": "FN", "category": cat})
        elif not has_bug and func_detected:
            fp += 1
            details.append({"func": func_name, "result": "FP", "category": cat})
        else:
            tn += 1
            details.append({"func": func_name, "result": "TN", "category": cat})
    
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "details": details}


def main():
    print("=" * 70)
    print("Real-World Bug Benchmark Evaluation")
    print("=" * 70)
    
    bug_files = sorted(BUGS_DIR.glob("*.py"))
    if not bug_files:
        print("ERROR: No bug files found in", BUGS_DIR)
        return
    
    tools = {
        "guardharvest": run_guardharvest,
        "mypy": run_mypy,
        "pyright": run_pyright,
        "pylint": run_pylint,
    }
    
    all_results = {}
    
    for tool_name, tool_func in tools.items():
        print(f"\n--- Running {tool_name} ---")
        tool_totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "details": []}
        
        for filepath in bug_files:
            fname = filepath.name
            source = filepath.read_text()
            
            # Get ground truth for this file
            gt_for_file = {k: v for k, v in GROUND_TRUTH.items() if k[0] == fname}
            if not gt_for_file:
                continue
            
            result = evaluate_tool(tool_func, filepath, source, gt_for_file)
            tool_totals["tp"] += result["tp"]
            tool_totals["fp"] += result["fp"]
            tool_totals["fn"] += result["fn"]
            tool_totals["tn"] += result["tn"]
            tool_totals["details"].extend(result["details"])
            
            print(f"  {fname}: TP={result['tp']} FP={result['fp']} FN={result['fn']} TN={result['tn']}")
        
        tp, fp, fn, tn = tool_totals["tp"], tool_totals["fp"], tool_totals["fn"], tool_totals["tn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        tool_totals["precision"] = round(precision, 4)
        tool_totals["recall"] = round(recall, 4)
        tool_totals["f1"] = round(f1, 4)
        
        print(f"\n  {tool_name} TOTALS: TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"  Precision={precision:.1%} Recall={recall:.1%} F1={f1:.1%}")
        
        all_results[tool_name] = tool_totals
    
    # Per-category breakdown for GuardHarvest
    print("\n" + "=" * 70)
    print("GuardHarvest Per-Category Breakdown")
    print("=" * 70)
    gh_details = all_results.get("guardharvest", {}).get("details", [])
    categories = set(d["category"] for d in gh_details)
    for cat in sorted(categories):
        cat_items = [d for d in gh_details if d["category"] == cat]
        tp = sum(1 for d in cat_items if d["result"] == "TP")
        fp = sum(1 for d in cat_items if d["result"] == "FP")
        fn = sum(1 for d in cat_items if d["result"] == "FN")
        tn = sum(1 for d in cat_items if d["result"] == "TN")
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"  {cat}: TP={tp} FP={fp} FN={fn} TN={tn} P={prec:.1%} R={rec:.1%}")
    
    # Save results
    output_path = RESULTS_DIR / "real_world_benchmark_results.json"
    # Remove non-serializable details for summary
    summary = {}
    for tool, data in all_results.items():
        summary[tool] = {
            "tp": data["tp"], "fp": data["fp"], "fn": data["fn"], "tn": data["tn"],
            "precision": data["precision"], "recall": data["recall"], "f1": data["f1"],
            "per_function": data["details"]
        }
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
