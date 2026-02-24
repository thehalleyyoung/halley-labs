#!/usr/bin/env python3
"""Run hardware litmus tests using the Rust litmus_infinity runner.

Builds the Rust binary, runs litmus tests on actual GPU hardware, collects
results, and generates reports in CSV/JSON format.

Usage:
    python run_hardware_tests.py --backend cuda --iterations 100000
    python run_hardware_tests.py --backend vulkan --tests MP,SB,IRIW
    python run_hardware_tests.py --backend opencl --output results.json --plot
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TestConfig:
    """Configuration for a hardware test run."""
    backend: str = "cuda"
    iterations: int = 100_000
    threads: int = 256
    workgroups: int = 1
    stress_mode: str = "none"
    timeout: int = 120
    seed: int = 0
    tests: Optional[List[str]] = None
    output_dir: str = "results"
    output_format: str = "json"
    keep_intermediates: bool = False
    verbose: bool = False

    def validate(self):
        """Validate the configuration."""
        valid_backends = {"cuda", "opencl", "vulkan", "metal"}
        if self.backend not in valid_backends:
            raise ValueError(
                f"Invalid backend '{self.backend}'. Must be one of {valid_backends}"
            )
        if self.iterations <= 0:
            raise ValueError("iterations must be > 0")
        if self.threads <= 0:
            raise ValueError("threads must be > 0")
        if self.workgroups <= 0:
            raise ValueError("workgroups must be > 0")

    def to_args(self) -> List[str]:
        """Convert to command-line arguments for the Rust runner."""
        args = [
            "--backend", self.backend,
            "--iterations", str(self.iterations),
            "--threads", str(self.threads),
            "--workgroups", str(self.workgroups),
            "--stress-mode", self.stress_mode,
            "--timeout", str(self.timeout),
        ]
        if self.seed > 0:
            args.extend(["--seed", str(self.seed)])
        if self.tests:
            args.extend(["--tests", ",".join(self.tests)])
        return args


@dataclass
class OutcomeRecord:
    """A single observed outcome with its count."""
    outcome_key: str
    count: int
    fraction: float = 0.0


@dataclass
class TestResult:
    """Result from running a single litmus test."""
    test_name: str
    backend: str
    total_iterations: int
    duration_seconds: float
    outcomes: List[OutcomeRecord] = field(default_factory=list)
    distinct_outcomes: int = 0
    consistent: bool = True
    violations: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class BatchResult:
    """Result from running a batch of tests."""
    config: TestConfig
    results: List[TestResult] = field(default_factory=list)
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    total_duration: float = 0.0
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    """Find the Cargo project root."""
    here = Path(__file__).resolve().parent
    # Walk up looking for Cargo.toml
    candidate = here
    for _ in range(10):
        if (candidate / "Cargo.toml").exists():
            return candidate
        candidate = candidate.parent
    # Fallback: assume two levels up from experiments/hardware_validation/
    return here.parent.parent


def build_runner(project_root: Path, verbose: bool = False) -> Path:
    """Build the Rust litmus CLI binary and return its path."""
    print("[*] Building litmus runner...")
    cmd = ["cargo", "build", "--release", "--bin", "litmus-cli"]
    kwargs = {"cwd": str(project_root)}
    if not verbose:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        stderr = getattr(result, "stderr", b"")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        print(f"[!] Build failed:\n{stderr}", file=sys.stderr)
        sys.exit(1)

    binary = project_root / "target" / "release" / "litmus-cli"
    if not binary.exists():
        print("[!] Binary not found after build", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Binary: {binary}")
    return binary


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

def run_single_test(
    binary: Path,
    test_name: str,
    config: TestConfig,
) -> TestResult:
    """Run a single litmus test and parse output."""
    cmd = [
        str(binary),
        "hardware-test",
        "--test", test_name,
        *config.to_args(),
    ]

    if config.verbose:
        print(f"  [>] {' '.join(cmd)}")

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout,
        )
        elapsed = time.time() - start

        if proc.returncode != 0:
            return TestResult(
                test_name=test_name,
                backend=config.backend,
                total_iterations=0,
                duration_seconds=elapsed,
                error=proc.stderr.strip() or f"exit code {proc.returncode}",
            )

        return parse_test_output(test_name, config.backend, proc.stdout, elapsed)

    except subprocess.TimeoutExpired:
        return TestResult(
            test_name=test_name,
            backend=config.backend,
            total_iterations=0,
            duration_seconds=config.timeout,
            error=f"timeout after {config.timeout}s",
        )
    except Exception as e:
        return TestResult(
            test_name=test_name,
            backend=config.backend,
            total_iterations=0,
            duration_seconds=time.time() - start,
            error=str(e),
        )


def parse_test_output(
    test_name: str,
    backend: str,
    stdout: str,
    elapsed: float,
) -> TestResult:
    """Parse the structured output from the Rust runner."""
    outcomes: Dict[str, int] = {}
    total = 0
    violations = []

    for line in stdout.splitlines():
        line = line.strip()

        # Parse outcome lines: OUTCOME key COUNT n
        m = re.match(r"^OUTCOME\s+(\S+)\s+COUNT\s+(\d+)$", line)
        if m:
            key, count = m.group(1), int(m.group(2))
            outcomes[key] = outcomes.get(key, 0) + count
            total += count
            continue

        # Parse violation lines
        if line.startswith("VIOLATION"):
            violations.append(line)

    outcome_records = []
    for key, count in sorted(outcomes.items(), key=lambda kv: -kv[1]):
        frac = count / total if total > 0 else 0.0
        outcome_records.append(OutcomeRecord(
            outcome_key=key,
            count=count,
            fraction=frac,
        ))

    return TestResult(
        test_name=test_name,
        backend=backend,
        total_iterations=total,
        duration_seconds=elapsed,
        outcomes=outcome_records,
        distinct_outcomes=len(outcomes),
        consistent=len(violations) == 0,
        violations=violations,
    )


def discover_tests(binary: Path, config: TestConfig) -> List[str]:
    """Discover available litmus tests from the runner."""
    if config.tests:
        return config.tests

    cmd = [str(binary), "list-tests", "--backend", config.backend]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            tests = [
                line.strip()
                for line in proc.stdout.splitlines()
                if line.strip() and not line.startswith("#")
            ]
            if tests:
                return tests
    except Exception:
        pass

    # Fallback: standard litmus tests
    return [
        "MP", "MP+pos", "SB", "SB+fences", "LB", "LB+fences",
        "IRIW", "2+2W", "WRC", "ISA2",
    ]


def run_batch(binary: Path, config: TestConfig) -> BatchResult:
    """Run a batch of litmus tests."""
    tests = discover_tests(binary, config)
    print(f"[*] Running {len(tests)} tests on {config.backend.upper()}")
    print(f"    Iterations: {config.iterations}, Threads: {config.threads}")

    batch = BatchResult(
        config=config,
        total_tests=len(tests),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    start = time.time()
    for i, test_name in enumerate(tests, 1):
        print(f"  [{i}/{len(tests)}] {test_name}...", end=" ", flush=True)
        result = run_single_test(binary, test_name, config)

        if result.error:
            batch.errors += 1
            print(f"ERROR: {result.error}")
        elif result.consistent:
            batch.passed += 1
            print(
                f"PASS ({result.distinct_outcomes} outcomes, "
                f"{result.duration_seconds:.1f}s)"
            )
        else:
            batch.failed += 1
            print(
                f"FAIL ({len(result.violations)} violations, "
                f"{result.duration_seconds:.1f}s)"
            )

        batch.results.append(result)

    batch.total_duration = time.time() - start
    return batch


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_json(batch: BatchResult, path: Path):
    """Write results to JSON."""
    data = {
        "config": asdict(batch.config),
        "summary": {
            "total_tests": batch.total_tests,
            "passed": batch.passed,
            "failed": batch.failed,
            "errors": batch.errors,
            "total_duration": batch.total_duration,
            "timestamp": batch.timestamp,
        },
        "results": [asdict(r) for r in batch.results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[*] JSON results written to {path}")


def write_csv(batch: BatchResult, path: Path):
    """Write results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "test_name", "backend", "iterations", "duration_s",
            "distinct_outcomes", "consistent", "violations", "error",
        ])
        for r in batch.results:
            writer.writerow([
                r.test_name,
                r.backend,
                r.total_iterations,
                f"{r.duration_seconds:.3f}",
                r.distinct_outcomes,
                r.consistent,
                len(r.violations),
                r.error or "",
            ])

    # Also write per-outcome detail
    detail_path = path.with_suffix(".outcomes.csv")
    with open(detail_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_name", "outcome", "count", "fraction"])
        for r in batch.results:
            for o in r.outcomes:
                writer.writerow([
                    r.test_name,
                    o.outcome_key,
                    o.count,
                    f"{o.fraction:.6f}",
                ])

    print(f"[*] CSV results written to {path}")
    print(f"[*] Outcome details written to {detail_path}")


def write_results(batch: BatchResult, config: TestConfig):
    """Write results in the configured format."""
    out_dir = Path(config.output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    basename = f"litmus_{config.backend}_{timestamp}"

    if config.output_format in ("json", "both"):
        write_json(batch, out_dir / f"{basename}.json")
    if config.output_format in ("csv", "both"):
        write_csv(batch, out_dir / f"{basename}.csv")


def print_summary(batch: BatchResult):
    """Print a summary to stdout."""
    print()
    print("=" * 60)
    print(f"  Hardware Litmus Test Summary ({batch.config.backend.upper()})")
    print("=" * 60)
    print(f"  Tests:      {batch.total_tests}")
    print(f"  Passed:     {batch.passed}")
    print(f"  Failed:     {batch.failed}")
    print(f"  Errors:     {batch.errors}")
    print(f"  Duration:   {batch.total_duration:.1f}s")
    print(f"  Timestamp:  {batch.timestamp}")
    print("=" * 60)

    if batch.failed > 0:
        print("\n  Failed tests:")
        for r in batch.results:
            if not r.consistent:
                print(f"    - {r.test_name}: {len(r.violations)} violations")
                for v in r.violations[:3]:
                    print(f"        {v}")

    if batch.errors > 0:
        print("\n  Errored tests:")
        for r in batch.results:
            if r.error:
                print(f"    - {r.test_name}: {r.error}")


# ---------------------------------------------------------------------------
# Plotting (optional)
# ---------------------------------------------------------------------------

def plot_results(batch: BatchResult, output_dir: Path):
    """Generate plots of test results (requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[!] matplotlib not installed; skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bar chart of outcome counts per test
    fig, ax = plt.subplots(figsize=(12, 6))
    test_names = [r.test_name for r in batch.results if not r.error]
    outcome_counts = [r.distinct_outcomes for r in batch.results if not r.error]
    colours = [
        "green" if r.consistent else "red"
        for r in batch.results if not r.error
    ]

    ax.bar(range(len(test_names)), outcome_counts, color=colours)
    ax.set_xticks(range(len(test_names)))
    ax.set_xticklabels(test_names, rotation=45, ha="right")
    ax.set_ylabel("Distinct outcomes")
    ax.set_title(f"Litmus Test Results ({batch.config.backend.upper()})")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(output_dir / "outcome_counts.png", dpi=150)
    plt.close()

    # 2. Pie chart per test (up to 6)
    interesting = [r for r in batch.results if not r.error and r.distinct_outcomes > 1]
    if interesting:
        n = min(len(interesting), 6)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, r in zip(axes, interesting[:n]):
            labels = [o.outcome_key for o in r.outcomes]
            sizes = [o.count for o in r.outcomes]
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.set_title(r.test_name)
        plt.tight_layout()
        plt.savefig(output_dir / "outcome_distributions.png", dpi=150)
        plt.close()

    # 3. Duration bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    durations = [r.duration_seconds for r in batch.results if not r.error]
    ax.bar(range(len(test_names)), durations, color="steelblue")
    ax.set_xticks(range(len(test_names)))
    ax.set_xticklabels(test_names, rotation=45, ha="right")
    ax.set_ylabel("Duration (s)")
    ax.set_title("Test Durations")
    plt.tight_layout()
    plt.savefig(output_dir / "durations.png", dpi=150)
    plt.close()

    print(f"[*] Plots saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> TestConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run hardware litmus tests using litmus_infinity"
    )
    parser.add_argument(
        "--backend", choices=["cuda", "opencl", "vulkan", "metal"],
        default="cuda", help="GPU backend (default: cuda)"
    )
    parser.add_argument(
        "--iterations", type=int, default=100_000,
        help="Iterations per test (default: 100000)"
    )
    parser.add_argument(
        "--threads", type=int, default=256,
        help="Threads per workgroup (default: 256)"
    )
    parser.add_argument(
        "--workgroups", type=int, default=1,
        help="Number of workgroups (default: 1)"
    )
    parser.add_argument(
        "--stress", choices=["none", "light", "medium", "heavy"],
        default="none", help="Stress-testing mode (default: none)"
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Per-test timeout in seconds (default: 120)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed (0 = system entropy)"
    )
    parser.add_argument(
        "--tests", type=str, default=None,
        help="Comma-separated list of test names"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory (default: results)"
    )
    parser.add_argument(
        "--format", choices=["json", "csv", "both"],
        default="json", help="Output format (default: json)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate plots (requires matplotlib)"
    )
    parser.add_argument(
        "--keep-intermediates", action="store_true",
        help="Keep generated kernel source files"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    tests = None
    if args.tests:
        tests = [t.strip() for t in args.tests.split(",") if t.strip()]

    return TestConfig(
        backend=args.backend,
        iterations=args.iterations,
        threads=args.threads,
        workgroups=args.workgroups,
        stress_mode=args.stress,
        timeout=args.timeout,
        seed=args.seed,
        tests=tests,
        output_dir=args.output_dir,
        output_format=args.format,
        keep_intermediates=args.keep_intermediates,
        verbose=args.verbose,
    )


def main():
    config = parse_args()
    config.validate()

    project_root = find_project_root()
    print(f"[*] Project root: {project_root}")

    binary = build_runner(project_root, verbose=config.verbose)
    batch = run_batch(binary, config)

    write_results(batch, config)
    print_summary(batch)

    if hasattr(config, "verbose") and "--plot" in sys.argv:
        plot_results(batch, Path(config.output_dir))

    sys.exit(0 if batch.failed == 0 and batch.errors == 0 else 1)


if __name__ == "__main__":
    main()
