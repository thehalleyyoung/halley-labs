#!/usr/bin/env python3
"""Parse hardware litmus-test results and compare against model predictions.

Reads JSON/CSV output from run_hardware_tests.py, loads the model's allowed
outcomes, computes consistency checks and statistical analysis, and generates
a validation report.

Usage:
    python parse_results.py results/litmus_cuda_20240101_120000.json
    python parse_results.py results/*.json --model-dir models/
    python parse_results.py results.csv --format csv --output report.json
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ModelOutcome:
    """An outcome predicted by the memory model."""
    key: str
    classification: str  # "allowed", "forbidden", "required"


@dataclass
class ModelSpec:
    """Model specification for a single litmus test."""
    test_name: str
    outcomes: List[ModelOutcome] = field(default_factory=list)

    @property
    def allowed_keys(self) -> Set[str]:
        return {o.key for o in self.outcomes if o.classification in ("allowed", "required")}

    @property
    def forbidden_keys(self) -> Set[str]:
        return {o.key for o in self.outcomes if o.classification == "forbidden"}

    @property
    def required_keys(self) -> Set[str]:
        return {o.key for o in self.outcomes if o.classification == "required"}


@dataclass
class ObservedOutcome:
    """An observed outcome from hardware."""
    key: str
    count: int
    fraction: float


@dataclass
class TestValidation:
    """Validation result for a single test."""
    test_name: str
    backend: str
    total_iterations: int
    observed: List[ObservedOutcome]
    allowed_observed: List[Tuple[str, int]]  # (key, count)
    forbidden_observed: List[Tuple[str, int]]
    required_missing: List[str]
    unknown_observed: List[Tuple[str, int]]
    is_consistent: bool
    chi_square: Optional[float] = None
    p_value: Optional[float] = None
    entropy: float = 0.0
    notes: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Complete validation report across all tests."""
    title: str
    backend: str
    timestamp: str
    total_tests: int = 0
    consistent_tests: int = 0
    inconsistent_tests: int = 0
    skipped_tests: int = 0
    test_validations: List[TestValidation] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_specs(model_dir: Path) -> Dict[str, ModelSpec]:
    """Load model outcome specifications from a directory.

    Expects JSON files named <test_name>.json with format:
    {
        "test_name": "MP",
        "outcomes": [
            {"key": "T1:r0=1,T1:r1=0", "classification": "forbidden"},
            {"key": "T1:r0=0,T1:r1=0", "classification": "allowed"},
            ...
        ]
    }
    """
    specs = {}

    if not model_dir.exists():
        return specs

    for path in model_dir.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            name = data.get("test_name", path.stem)
            outcomes = [
                ModelOutcome(key=o["key"], classification=o["classification"])
                for o in data.get("outcomes", [])
            ]
            specs[name] = ModelSpec(test_name=name, outcomes=outcomes)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[!] Warning: failed to load {path}: {e}", file=sys.stderr)

    return specs


def create_default_model_specs() -> Dict[str, ModelSpec]:
    """Create default model specs for common litmus tests."""
    specs = {}

    # MP (Message Passing): forbidden outcome is r0=1, r1=0
    specs["MP"] = ModelSpec(
        test_name="MP",
        outcomes=[
            ModelOutcome(key="T1:r0=0,T1:r1=0", classification="allowed"),
            ModelOutcome(key="T1:r0=0,T1:r1=1", classification="allowed"),
            ModelOutcome(key="T1:r0=1,T1:r1=1", classification="allowed"),
            ModelOutcome(key="T1:r0=1,T1:r1=0", classification="forbidden"),
        ],
    )

    # SB (Store Buffering): both reading 0 may be forbidden under TSO
    specs["SB"] = ModelSpec(
        test_name="SB",
        outcomes=[
            ModelOutcome(key="T0:r0=0,T1:r0=0", classification="allowed"),
            ModelOutcome(key="T0:r0=0,T1:r0=1", classification="allowed"),
            ModelOutcome(key="T0:r0=1,T1:r0=0", classification="allowed"),
            ModelOutcome(key="T0:r0=1,T1:r0=1", classification="allowed"),
        ],
    )

    # LB (Load Buffering)
    specs["LB"] = ModelSpec(
        test_name="LB",
        outcomes=[
            ModelOutcome(key="T0:r0=0,T1:r0=0", classification="allowed"),
            ModelOutcome(key="T0:r0=0,T1:r0=1", classification="allowed"),
            ModelOutcome(key="T0:r0=1,T1:r0=0", classification="allowed"),
            ModelOutcome(key="T0:r0=1,T1:r0=1", classification="forbidden"),
        ],
    )

    return specs


# ---------------------------------------------------------------------------
# Parsing results
# ---------------------------------------------------------------------------

def parse_json_results(path: Path) -> List[dict]:
    """Parse a JSON results file."""
    with open(path) as f:
        data = json.load(f)
    return data.get("results", [])


def parse_csv_results(path: Path) -> List[dict]:
    """Parse CSV results files.

    Expects a main CSV and an optional .outcomes.csv companion file.
    """
    results = []
    outcome_path = path.with_suffix(".outcomes.csv")

    # Load main results
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "test_name": row["test_name"],
                "backend": row["backend"],
                "total_iterations": int(row.get("iterations", 0)),
                "duration_seconds": float(row.get("duration_s", 0)),
                "distinct_outcomes": int(row.get("distinct_outcomes", 0)),
                "consistent": row.get("consistent", "True").lower() == "true",
                "error": row.get("error") or None,
                "outcomes": [],
            })

    # Load outcome details if available
    if outcome_path.exists():
        outcome_map = defaultdict(list)
        with open(outcome_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                outcome_map[row["test_name"]].append({
                    "outcome_key": row["outcome"],
                    "count": int(row["count"]),
                    "fraction": float(row.get("fraction", 0)),
                })
        for r in results:
            r["outcomes"] = outcome_map.get(r["test_name"], [])

    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_test(
    result: dict,
    model: Optional[ModelSpec],
) -> TestValidation:
    """Validate a single test result against the model."""
    test_name = result["test_name"]
    backend = result.get("backend", "unknown")
    total = result.get("total_iterations", 0)
    error = result.get("error")

    observed = []
    for o in result.get("outcomes", []):
        observed.append(ObservedOutcome(
            key=o["outcome_key"],
            count=o["count"],
            fraction=o.get("fraction", 0.0),
        ))

    if error:
        return TestValidation(
            test_name=test_name,
            backend=backend,
            total_iterations=0,
            observed=[],
            allowed_observed=[],
            forbidden_observed=[],
            required_missing=[],
            unknown_observed=[],
            is_consistent=True,  # Can't tell
            notes=[f"Test errored: {error}"],
        )

    if model is None:
        # No model — can only report observations.
        return TestValidation(
            test_name=test_name,
            backend=backend,
            total_iterations=total,
            observed=observed,
            allowed_observed=[(o.key, o.count) for o in observed],
            forbidden_observed=[],
            required_missing=[],
            unknown_observed=[],
            is_consistent=True,
            entropy=compute_entropy(observed, total),
            notes=["No model available; all outcomes treated as allowed"],
        )

    allowed_obs = []
    forbidden_obs = []
    unknown_obs = []

    for o in observed:
        if o.key in model.allowed_keys:
            allowed_obs.append((o.key, o.count))
        elif o.key in model.forbidden_keys:
            forbidden_obs.append((o.key, o.count))
        else:
            unknown_obs.append((o.key, o.count))

    required_missing = []
    observed_keys = {o.key for o in observed}
    for rk in model.required_keys:
        if rk not in observed_keys:
            required_missing.append(rk)

    is_consistent = len(forbidden_obs) == 0 and len(required_missing) == 0

    # Chi-square test (if model has expected fractions)
    chi2, pval = None, None
    if model.outcomes and total > 0:
        chi2, pval = chi_square_test(observed, model, total)

    notes = []
    if unknown_obs:
        notes.append(
            f"{len(unknown_obs)} outcomes not in model specification"
        )

    return TestValidation(
        test_name=test_name,
        backend=backend,
        total_iterations=total,
        observed=observed,
        allowed_observed=allowed_obs,
        forbidden_observed=forbidden_obs,
        required_missing=required_missing,
        unknown_observed=unknown_obs,
        is_consistent=is_consistent,
        chi_square=chi2,
        p_value=pval,
        entropy=compute_entropy(observed, total),
        notes=notes,
    )


def compute_entropy(outcomes: List[ObservedOutcome], total: int) -> float:
    """Compute Shannon entropy (bits) of the outcome distribution."""
    if total == 0:
        return 0.0
    h = 0.0
    for o in outcomes:
        if o.count > 0:
            p = o.count / total
            h -= p * math.log2(p)
    return h


def chi_square_test(
    observed: List[ObservedOutcome],
    model: ModelSpec,
    total: int,
) -> Tuple[float, float]:
    """Perform a chi-square goodness-of-fit test.

    Uses uniform distribution among allowed outcomes as the expected
    distribution (in the absence of more specific model predictions).
    """
    allowed = model.allowed_keys | model.required_keys
    if not allowed:
        return 0.0, 1.0

    expected_per = total / len(allowed) if allowed else 0

    chi2 = 0.0
    for o in observed:
        if o.key in allowed:
            contrib = (o.count - expected_per) ** 2 / max(expected_per, 1)
            chi2 += contrib
        else:
            # Unexpected outcome contributes its full count.
            chi2 += o.count

    df = max(len(allowed) - 1, 1)
    p_value = approximate_chi2_p(chi2, df)
    return chi2, p_value


def approximate_chi2_p(chi2: float, df: int) -> float:
    """Wilson-Hilferty approximation for chi-square p-value."""
    if df == 0:
        return 1.0
    k = float(df)
    term = 2.0 / (9.0 * k)
    z = ((chi2 / k) ** (1.0 / 3.0) - (1.0 - term)) / math.sqrt(term)
    # Logistic approximation of standard normal CDF upper tail
    p = 1.0 / (1.0 + math.exp(1.7 * z))
    return max(0.0, min(1.0, p))


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    results_files: List[Path],
    model_specs: Dict[str, ModelSpec],
    fmt: str = "json",
) -> ValidationReport:
    """Generate a validation report from result files."""
    all_results = []
    backend = "unknown"

    for path in results_files:
        if path.suffix == ".json":
            file_results = parse_json_results(path)
        elif path.suffix == ".csv":
            file_results = parse_csv_results(path)
        else:
            print(f"[!] Unsupported format: {path}", file=sys.stderr)
            continue

        all_results.extend(file_results)
        if file_results:
            backend = file_results[0].get("backend", backend)

    report = ValidationReport(
        title=f"Hardware Validation ({backend.upper()})",
        backend=backend,
        timestamp=__import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        total_tests=len(all_results),
    )

    for result in all_results:
        test_name = result["test_name"]
        model = model_specs.get(test_name)
        validation = validate_test(result, model)

        if result.get("error"):
            report.skipped_tests += 1
        elif validation.is_consistent:
            report.consistent_tests += 1
        else:
            report.inconsistent_tests += 1

        report.test_validations.append(validation)

    return report


def print_report(report: ValidationReport):
    """Print the report to stdout."""
    print()
    print("=" * 70)
    print(f"  {report.title}")
    print("=" * 70)
    print(f"  Total tests:        {report.total_tests}")
    print(f"  Consistent:         {report.consistent_tests}")
    print(f"  Inconsistent:       {report.inconsistent_tests}")
    print(f"  Skipped/Errored:    {report.skipped_tests}")
    print(f"  Timestamp:          {report.timestamp}")
    print("-" * 70)

    for v in report.test_validations:
        status = "PASS" if v.is_consistent else "FAIL"
        if v.notes and "errored" in v.notes[0].lower():
            status = "SKIP"

        print(f"\n  [{status}] {v.test_name}")
        print(f"    Iterations: {v.total_iterations}")
        print(f"    Distinct outcomes: {len(v.observed)}")
        print(f"    Entropy: {v.entropy:.3f} bits")

        if v.allowed_observed:
            print(f"    Allowed (observed): {len(v.allowed_observed)}")
            for key, count in v.allowed_observed[:5]:
                frac = count / v.total_iterations if v.total_iterations else 0
                print(f"      {key}: {count} ({frac:.4%})")

        if v.forbidden_observed:
            print(f"    FORBIDDEN (observed): {len(v.forbidden_observed)}")
            for key, count in v.forbidden_observed:
                frac = count / v.total_iterations if v.total_iterations else 0
                print(f"      ⚠ {key}: {count} ({frac:.4%})")

        if v.required_missing:
            print(f"    REQUIRED (missing): {len(v.required_missing)}")
            for key in v.required_missing:
                print(f"      ⚠ {key}")

        if v.unknown_observed:
            print(f"    Unknown outcomes: {len(v.unknown_observed)}")
            for key, count in v.unknown_observed[:3]:
                print(f"      ? {key}: {count}")

        if v.chi_square is not None:
            print(f"    χ² = {v.chi_square:.2f}, p = {v.p_value:.4f}")

        for note in v.notes:
            print(f"    Note: {note}")

    print()
    print("=" * 70)
    overall = "PASS" if report.inconsistent_tests == 0 else "FAIL"
    print(f"  Overall: {overall}")
    print("=" * 70)


def write_report_json(report: ValidationReport, path: Path):
    """Write the validation report as JSON."""
    data = {
        "title": report.title,
        "backend": report.backend,
        "timestamp": report.timestamp,
        "summary": {
            "total_tests": report.total_tests,
            "consistent": report.consistent_tests,
            "inconsistent": report.inconsistent_tests,
            "skipped": report.skipped_tests,
        },
        "validations": [asdict(v) for v in report.test_validations],
        "notes": report.notes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[*] Report written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse and validate hardware litmus test results"
    )
    parser.add_argument(
        "results", nargs="+", type=Path,
        help="Result files (JSON or CSV)"
    )
    parser.add_argument(
        "--model-dir", type=Path, default=None,
        help="Directory with model outcome specs"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output report file (JSON)"
    )
    parser.add_argument(
        "--format", choices=["json", "csv"], default="json",
        help="Input format hint (default: auto-detect from extension)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress detailed output"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model specs
    model_specs = {}
    if args.model_dir:
        model_specs = load_model_specs(args.model_dir)
        print(f"[*] Loaded {len(model_specs)} model specifications")
    else:
        model_specs = create_default_model_specs()
        print(f"[*] Using {len(model_specs)} default model specifications")

    # Validate result files exist
    for p in args.results:
        if not p.exists():
            print(f"[!] File not found: {p}", file=sys.stderr)
            sys.exit(1)

    # Generate report
    report = generate_report(args.results, model_specs, args.format)

    # Output
    if not args.quiet:
        print_report(report)

    if args.output:
        write_report_json(report, args.output)

    sys.exit(0 if report.inconsistent_tests == 0 else 1)


if __name__ == "__main__":
    main()
