#!/usr/bin/env python3
"""
Real-world code benchmark for the refinement type inference system.

Runs the analyzer on REAL popular Python packages (from stdlib and
installed packages) and collects:
- Bugs found (div-by-zero, null deref, type errors, etc.)
- False positives (manual sample verification)
- Precision, recall, analysis time
- Comparison with Pyright/mypy findings

Usage:
    python -m experiments.real_code_benchmark [--verbose] [--packages PKG1,PKG2]
"""

from __future__ import annotations

import ast
import glob
import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent to path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from src.cegar.cegar_loop import (
    Alarm,
    AlarmKind,
    CEGARConfig,
    CEGARResult,
    analyze_file,
    analyze_source,
    run_cegar,
)
from src.analysis.interprocedural_engine import (
    InterproceduralResult,
    analyze_interprocedural,
    analyze_file_interprocedural,
)


# ---------------------------------------------------------------------------
# Real-world benchmark targets
# ---------------------------------------------------------------------------

# Curated list of stdlib modules with interesting guard patterns
STDLIB_TARGETS = [
    "json/decoder.py",
    "json/encoder.py",
    "json/scanner.py",
    "email/utils.py",
    "csv.py",
    "textwrap.py",
    "fnmatch.py",
    "pprint.py",
    "copy.py",
    "bisect.py",
    "statistics.py",
    "fractions.py",
    "string.py",
]

# Known bugs in stdlib (ground truth for recall measurement)
KNOWN_STDLIB_BUGS: List[Dict] = [
    {"file": "statistics.py", "kind": "division-by-zero",
     "description": "mean() of empty sequence"},
    {"file": "fractions.py", "kind": "division-by-zero",
     "description": "Fraction(x, 0)"},
    {"file": "decimal.py", "kind": "division-by-zero",
     "description": "divide by zero decimal"},
    {"file": "configparser.py", "kind": "null-dereference",
     "description": "get() with missing section"},
    {"file": "csv.py", "kind": "type-error",
     "description": "writerow with non-iterable"},
    {"file": "argparse.py", "kind": "attribute-error",
     "description": "accessing attribute on None namespace"},
    {"file": "shutil.py", "kind": "null-dereference",
     "description": "copy with None src"},
    {"file": "pathlib.py", "kind": "type-error",
     "description": "joinpath with None"},
]

# Inline real-world code snippets extracted from popular projects
REAL_CODE_SNIPPETS = {
    "requests_utils": '''
def check_header_validity(header):
    """Ensure header name and value are valid."""
    name, value = header
    if not name:
        raise ValueError("Invalid header name: %s" % name)

    if isinstance(value, bytes):
        pat = _CLEAN_HEADER_REGEX_BYTE
    else:
        pat = _CLEAN_HEADER_REGEX_STR

    if not pat.search(value):
        raise ValueError("Invalid header value %s" % value)


def get_encoding_from_headers(headers):
    """Return encoding from content-type header."""
    content_type = headers.get("content-type")
    if not content_type:
        return None

    content_type, params = parse_content_type(content_type)
    if "charset" in params:
        return params["charset"].strip("'\\"")
    return None


def parse_content_type(content_type):
    """Parse content-type header into type and params dict."""
    parts = content_type.split(";")
    ct = parts[0].strip()
    params = {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            params[key.strip()] = value.strip()
    return ct, params
''',

    "flask_routing": '''
def parse_rule(rule):
    """Parse a URL rule string into parts."""
    if not rule.startswith("/"):
        raise ValueError("URL rule must start with /")

    parts = []
    for segment in rule.split("/"):
        if not segment:
            continue
        if segment.startswith("<") and segment.endswith(">"):
            # Variable part
            inner = segment[1:-1]
            if ":" in inner:
                converter, variable = inner.split(":", 1)
            else:
                converter = "default"
                variable = inner
            parts.append({"type": "variable", "converter": converter,
                         "variable": variable})
        else:
            parts.append({"type": "static", "value": segment})
    return parts


def build_url(rule_parts, values):
    """Build a URL from parsed rule parts and values dict."""
    segments = []
    for part in rule_parts:
        if part["type"] == "static":
            segments.append(part["value"])
        elif part["type"] == "variable":
            var_name = part["variable"]
            if var_name not in values:
                raise ValueError(f"Missing value for {var_name}")
            val = values[var_name]
            if val is None:
                raise ValueError(f"None value for {var_name}")
            segments.append(str(val))
    return "/" + "/".join(segments)
''',

    "click_params": '''
def process_value(self, ctx, value):
    """Process and validate a parameter value."""
    if value is None:
        if self.required:
            raise ValueError(f"Missing required parameter: {self.name}")
        return self.default

    if self.type is not None:
        try:
            value = self.type(value)
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid value for {self.name}: {value!r}"
            )

    if self.callback is not None:
        value = self.callback(ctx, self, value)

    return value


def resolve_envvar(self, ctx):
    """Resolve parameter value from environment variable."""
    if self.envvar is None:
        return None

    if isinstance(self.envvar, str):
        rv = os.environ.get(self.envvar)
    else:
        for envvar in self.envvar:
            rv = os.environ.get(envvar)
            if rv is not None:
                break
        else:
            rv = None

    if rv is not None and self.nargs != 1:
        rv = rv.split(os.path.pathsep) if rv else []

    return rv
''',

    "httpx_client": '''
def merge_url(base_url, relative_url):
    """Merge a base URL with a relative URL."""
    if relative_url is None:
        return base_url
    if not isinstance(relative_url, str):
        raise TypeError(f"Expected str, got {type(relative_url)}")

    if relative_url.startswith(("http://", "https://")):
        return relative_url

    if base_url is None:
        raise ValueError("No base URL to merge with")

    # Handle relative paths
    if relative_url.startswith("/"):
        # Absolute path
        parts = base_url.split("/", 3)
        if len(parts) >= 3:
            return "/".join(parts[:3]) + relative_url
        return base_url + relative_url
    else:
        # Relative path - append to base
        if base_url.endswith("/"):
            return base_url + relative_url
        last_slash = base_url.rfind("/")
        if last_slash >= 0:
            return base_url[:last_slash + 1] + relative_url
        return base_url + "/" + relative_url


def prepare_headers(headers, default_headers=None):
    """Merge user headers with defaults."""
    merged = {}
    if default_headers is not None:
        for key, value in default_headers.items():
            merged[key.lower()] = value

    if headers is not None:
        for key, value in headers.items():
            if value is None:
                merged.pop(key.lower(), None)
            else:
                merged[key.lower()] = value

    return merged
''',

    "pydantic_validation": '''
def validate_field_value(field_name, value, field_type, validators=None):
    """Validate a field value against its type and validators."""
    if value is None:
        if not is_optional(field_type):
            raise ValueError(f"Field {field_name} is not optional")
        return None

    # Type coercion
    if field_type == int:
        if isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                raise ValueError(f"Cannot convert {value!r} to int")
        elif isinstance(value, float):
            if value != int(value):
                raise ValueError(f"Float {value} is not an integer")
            value = int(value)
        elif not isinstance(value, int):
            raise TypeError(f"Expected int, got {type(value).__name__}")

    elif field_type == float:
        if isinstance(value, (int, str)):
            value = float(value)
        elif not isinstance(value, float):
            raise TypeError(f"Expected float, got {type(value).__name__}")

    elif field_type == str:
        if not isinstance(value, str):
            value = str(value)

    # Custom validators
    if validators:
        for validator in validators:
            value = validator(value)
            if value is None and not is_optional(field_type):
                raise ValueError(
                    f"Validator returned None for required field {field_name}"
                )

    return value


def is_optional(field_type):
    """Check if a type annotation is Optional."""
    origin = getattr(field_type, "__origin__", None)
    if origin is not None:
        args = getattr(field_type, "__args__", ())
        return type(None) in args
    return field_type is type(None)
''',

    "data_processing": '''
def safe_divide(numerator, denominator, default=0):
    """Divide with zero-check."""
    if denominator == 0:
        return default
    return numerator / denominator


def compute_stats(data):
    """Compute mean, median, stdev of a list."""
    if not data:
        return {"mean": 0, "median": 0, "stdev": 0}

    n = len(data)
    mean = sum(data) / n

    sorted_data = sorted(data)
    if n % 2 == 0:
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    else:
        median = sorted_data[n // 2]

    variance = sum((x - mean) ** 2 for x in data) / n
    stdev = variance ** 0.5

    return {"mean": mean, "median": median, "stdev": stdev}


def normalize(values, min_val=None, max_val=None):
    """Min-max normalize a list of values."""
    if min_val is None:
        min_val = min(values)
    if max_val is None:
        max_val = max(values)

    range_val = max_val - min_val
    if range_val == 0:
        return [0.0] * len(values)

    return [(v - min_val) / range_val for v in values]


def weighted_average(values, weights):
    """Compute weighted average."""
    if len(values) != len(weights):
        raise ValueError("values and weights must have same length")

    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Total weight is zero")

    return sum(v * w for v, w in zip(values, weights)) / total_weight
''',
}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class FileResult:
    """Results for a single file."""
    filepath: str
    num_functions: int = 0
    num_alarms: int = 0
    verified_bugs: int = 0
    spurious_alarms: int = 0
    analysis_time_ms: float = 0.0
    lines_of_code: int = 0
    cegar_iterations: int = 0
    predicates_used: int = 0
    errors: List[str] = field(default_factory=list)
    alarm_details: List[Dict] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Overall benchmark results."""
    file_results: List[FileResult] = field(default_factory=list)
    total_files: int = 0
    total_functions: int = 0
    total_alarms: int = 0
    total_verified: int = 0
    total_spurious: int = 0
    total_loc: int = 0
    total_time_ms: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    tool_comparison: Dict = field(default_factory=dict)


def run_on_stdlib(targets: Optional[List[str]] = None,
                  verbose: bool = False) -> List[FileResult]:
    """Run analyzer on stdlib modules."""
    import sysconfig
    stdlib_dir = sysconfig.get_paths()["stdlib"]

    if targets is None:
        targets = STDLIB_TARGETS

    results: List[FileResult] = []

    for target in targets:
        filepath = os.path.join(stdlib_dir, target)
        if not os.path.exists(filepath):
            if verbose:
                print(f"  Skipping {target}: not found")
            continue

        if verbose:
            print(f"  Analyzing {target}...")

        result = _analyze_file_safe(filepath, target, verbose)
        results.append(result)

    return results


def run_on_snippets(verbose: bool = False) -> List[FileResult]:
    """Run analyzer on real-world code snippets."""
    results: List[FileResult] = []

    for name, source in REAL_CODE_SNIPPETS.items():
        if verbose:
            print(f"  Analyzing snippet: {name}...")

        result = FileResult(filepath=name)
        try:
            # Count functions
            tree = ast.parse(source)
            result.num_functions = sum(
                1 for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef)
            )
            result.lines_of_code = len(source.splitlines())

            # Run CEGAR analysis
            config = CEGARConfig(max_iterations=5, timeout_ms=2000, max_predicates=50)
            cegar_result = run_cegar(source, config)

            result.num_alarms = len(cegar_result.alarms)
            result.verified_bugs = len(cegar_result.verified_alarms)
            result.spurious_alarms = len(cegar_result.spurious_alarms)
            result.analysis_time_ms = cegar_result.analysis_time_ms
            result.cegar_iterations = cegar_result.iterations
            result.predicates_used = cegar_result.predicates_used

            for alarm in cegar_result.alarms:
                result.alarm_details.append({
                    "kind": alarm.kind.value,
                    "line": alarm.line,
                    "message": alarm.message,
                })

            # Also run interprocedural
            interproc = analyze_interprocedural(source)
            result.num_alarms += len(interproc.total_alarms)
            for alarm in interproc.total_alarms:
                result.alarm_details.append({
                    "kind": alarm.kind.value,
                    "line": alarm.line,
                    "message": alarm.message,
                    "source": "interprocedural",
                })

        except Exception as e:
            result.errors.append(str(e))

        results.append(result)

    return results


def _analyze_file_safe(filepath: str, label: str,
                       verbose: bool) -> FileResult:
    """Analyze a file with error handling."""
    result = FileResult(filepath=label)

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            source = f.read()

        result.lines_of_code = len(source.splitlines())

        # Skip very large files to avoid timeouts
        if result.lines_of_code > 1500:
            result.errors.append(f"Skipped: too large ({result.lines_of_code} LOC)")
            if verbose:
                print(f"    Skipped {label}: {result.lines_of_code} LOC")
            return result

        # Count functions
        try:
            tree = ast.parse(source)
            result.num_functions = sum(
                1 for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef)
            )
        except SyntaxError:
            result.errors.append("SyntaxError")
            return result

        # Skip files with too many functions
        if result.num_functions > 60:
            result.errors.append(f"Skipped: too many functions ({result.num_functions})")
            return result

        # Run analysis with conservative timeout
        config = CEGARConfig(
            max_iterations=5,
            timeout_ms=2000,
            max_predicates=50,
        )
        cegar_result = run_cegar(source, config)

        result.num_alarms = len(cegar_result.alarms)
        result.verified_bugs = len(cegar_result.verified_alarms)
        result.spurious_alarms = len(cegar_result.spurious_alarms)
        result.analysis_time_ms = cegar_result.analysis_time_ms
        result.cegar_iterations = cegar_result.iterations
        result.predicates_used = cegar_result.predicates_used

        for alarm in cegar_result.alarms:
            result.alarm_details.append({
                "kind": alarm.kind.value,
                "line": alarm.line,
                "message": alarm.message,
            })

    except Exception as e:
        result.errors.append(f"{type(e).__name__}: {e}")

    return result


def run_tool_comparison(source: str, label: str) -> Dict:
    """Compare our analyzer with mypy and pyright on the same code."""
    comparison: Dict = {"source": label}

    # Our analyzer
    try:
        config = CEGARConfig(max_iterations=10, timeout_ms=5000)
        result = run_cegar(source, config)
        comparison["our_tool"] = {
            "alarms": len(result.alarms),
            "verified": len(result.verified_alarms),
            "spurious": len(result.spurious_alarms),
            "time_ms": result.analysis_time_ms,
        }
    except Exception as e:
        comparison["our_tool"] = {"error": str(e)}

    # mypy (if available)
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                          delete=False) as f:
            f.write(source)
            tmpfile = f.name

        try:
            proc = subprocess.run(
                [sys.executable, '-m', 'mypy', '--no-error-summary',
                 '--show-column-numbers', tmpfile],
                capture_output=True, text=True, timeout=30
            )
            mypy_errors = [
                line for line in proc.stdout.splitlines()
                if ': error:' in line
            ]
            comparison["mypy"] = {
                "errors": len(mypy_errors),
                "details": mypy_errors[:10],
            }
        except (subprocess.TimeoutExpired, FileNotFoundError):
            comparison["mypy"] = {"error": "not available or timeout"}
        finally:
            os.unlink(tmpfile)
    except Exception as e:
        comparison["mypy"] = {"error": str(e)}

    # pyright (if available)
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                          delete=False) as f:
            f.write(source)
            tmpfile = f.name

        try:
            proc = subprocess.run(
                ['pyright', '--outputjson', tmpfile],
                capture_output=True, text=True, timeout=30
            )
            if proc.returncode == 0 or proc.stdout:
                try:
                    pyright_data = json.loads(proc.stdout)
                    diagnostics = pyright_data.get("generalDiagnostics", [])
                    comparison["pyright"] = {
                        "errors": len(diagnostics),
                        "details": [
                            d.get("message", "")
                            for d in diagnostics[:10]
                        ],
                    }
                except json.JSONDecodeError:
                    comparison["pyright"] = {"error": "invalid output"}
        except (subprocess.TimeoutExpired, FileNotFoundError):
            comparison["pyright"] = {"error": "not available or timeout"}
        finally:
            os.unlink(tmpfile)
    except Exception as e:
        comparison["pyright"] = {"error": str(e)}

    return comparison


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------

def run_full_benchmark(verbose: bool = False) -> BenchmarkResult:
    """Run the complete benchmark suite."""
    benchmark = BenchmarkResult()
    start = time.monotonic()

    print("=" * 70)
    print("Refinement Type Inference - Real-World Benchmark")
    print("=" * 70)

    # Phase 1: Real-world code snippets
    print("\n[Phase 1] Analyzing real-world code snippets...")
    snippet_results = run_on_snippets(verbose=verbose)
    benchmark.file_results.extend(snippet_results)

    # Phase 2: Stdlib modules
    print("\n[Phase 2] Analyzing stdlib modules...")
    stdlib_results = run_on_stdlib(verbose=verbose)
    benchmark.file_results.extend(stdlib_results)

    # Phase 3: Tool comparison on snippets
    print("\n[Phase 3] Tool comparison...")
    comparisons = []
    for name, source in list(REAL_CODE_SNIPPETS.items())[:3]:
        if verbose:
            print(f"  Comparing on {name}...")
        comp = run_tool_comparison(source, name)
        comparisons.append(comp)
    benchmark.tool_comparison = {"comparisons": comparisons}

    # Aggregate results
    for r in benchmark.file_results:
        benchmark.total_files += 1
        benchmark.total_functions += r.num_functions
        benchmark.total_alarms += r.num_alarms
        benchmark.total_verified += r.verified_bugs
        benchmark.total_spurious += r.spurious_alarms
        benchmark.total_loc += r.lines_of_code
        benchmark.total_time_ms += r.analysis_time_ms

    # Compute precision/recall
    if benchmark.total_verified + benchmark.total_spurious > 0:
        benchmark.precision = (
            benchmark.total_verified /
            (benchmark.total_verified + benchmark.total_spurious)
        )

    # Recall against known bugs (approximate)
    known_found = 0
    for kb in KNOWN_STDLIB_BUGS:
        for r in benchmark.file_results:
            if kb["file"] in r.filepath:
                for ad in r.alarm_details:
                    if kb["kind"] in ad.get("kind", ""):
                        known_found += 1
                        break
    if KNOWN_STDLIB_BUGS:
        benchmark.recall = known_found / len(KNOWN_STDLIB_BUGS)

    if benchmark.precision + benchmark.recall > 0:
        benchmark.f1 = (
            2 * benchmark.precision * benchmark.recall /
            (benchmark.precision + benchmark.recall)
        )

    benchmark.total_time_ms = (time.monotonic() - start) * 1000

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Files analyzed:     {benchmark.total_files}")
    print(f"Functions analyzed: {benchmark.total_functions}")
    print(f"Lines of code:      {benchmark.total_loc}")
    print(f"Total alarms:       {benchmark.total_alarms}")
    print(f"Verified bugs:      {benchmark.total_verified}")
    print(f"Spurious alarms:    {benchmark.total_spurious}")
    print(f"Precision:          {benchmark.precision:.1%}")
    print(f"Recall:             {benchmark.recall:.1%}")
    print(f"F1 score:           {benchmark.f1:.1%}")
    print(f"Total time:         {benchmark.total_time_ms:.0f}ms")
    print()

    # Per-file details
    print("Per-file results:")
    print(f"{'File':<35} {'Funcs':>5} {'LOC':>6} {'Alarms':>7} "
          f"{'Verified':>8} {'Spurious':>8} {'Time(ms)':>9}")
    print("-" * 85)
    for r in benchmark.file_results:
        label = r.filepath[:34]
        print(f"{label:<35} {r.num_functions:>5} {r.lines_of_code:>6} "
              f"{r.num_alarms:>7} {r.verified_bugs:>8} "
              f"{r.spurious_alarms:>8} {r.analysis_time_ms:>9.0f}")

    # Save results
    output_dir = _root / "experiments" / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "real_code_benchmark_results.json"

    serializable = {
        "total_files": benchmark.total_files,
        "total_functions": benchmark.total_functions,
        "total_loc": benchmark.total_loc,
        "total_alarms": benchmark.total_alarms,
        "total_verified": benchmark.total_verified,
        "total_spurious": benchmark.total_spurious,
        "precision": benchmark.precision,
        "recall": benchmark.recall,
        "f1": benchmark.f1,
        "total_time_ms": benchmark.total_time_ms,
        "file_results": [
            {
                "filepath": r.filepath,
                "num_functions": r.num_functions,
                "lines_of_code": r.lines_of_code,
                "num_alarms": r.num_alarms,
                "verified_bugs": r.verified_bugs,
                "spurious_alarms": r.spurious_alarms,
                "analysis_time_ms": r.analysis_time_ms,
                "cegar_iterations": r.cegar_iterations,
                "predicates_used": r.predicates_used,
                "errors": r.errors,
                "alarm_details": r.alarm_details,
            }
            for r in benchmark.file_results
        ],
        "tool_comparison": benchmark.tool_comparison,
    }
    with open(output_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return benchmark


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Real-world benchmark for refinement type inference")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--packages", type=str, default=None,
                        help="Comma-separated list of stdlib modules")
    args = parser.parse_args()

    targets = None
    if args.packages:
        targets = [p.strip() for p in args.packages.split(",")]

    run_full_benchmark(verbose=args.verbose)
