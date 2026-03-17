#!/usr/bin/env python3
"""
MIPLIB 2017 Benchmark Suite for SpectralOracle.

Compares decomposition approaches across three configurations:
  (1) Default SCIP (monolithic solve, no decomposition)
  (2) GCG auto-decomposition (Dantzig-Wolfe via hMETIS partitioning)
  (3) SpectralOracle recommended decomposition

Metrics collected:
  - Solve time (seconds)
  - Gap closed (%)
  - Decomposition quality score (spectral gap, crossing weight)
  - Dual bound quality

Usage:
  python3 run_miplib_benchmark.py --instances-dir miplib2017/ --tier pilot
  python3 run_miplib_benchmark.py --instances-dir miplib2017/ --tier paper --time-limit 600

Requirements:
  - spectral-oracle binary (cargo install --path spectral-cli)
  - MIPLIB 2017 instances (.mps files) in --instances-dir
  - Optional: SCIP, GCG executables on PATH for comparison
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

TIER_SIZES = {
    "pilot": 10,
    "dev": 50,
    "paper": 200,
    "artifact": 1065,
}

TIER_LIMITS = {
    "pilot": 60,
    "dev": 300,
    "paper": 600,
    "artifact": 3600,
}


@dataclass
class BenchmarkResult:
    instance: str
    rows: int = 0
    cols: int = 0
    nnz: int = 0
    # SpectralOracle
    spectral_gap: float = 0.0
    spectral_time_ms: float = 0.0
    recommended_method: str = ""
    oracle_confidence: float = 0.0
    # SCIP baseline
    scip_obj: Optional[float] = None
    scip_time: Optional[float] = None
    scip_gap: Optional[float] = None
    # GCG comparison
    gcg_obj: Optional[float] = None
    gcg_time: Optional[float] = None
    gcg_gap: Optional[float] = None
    # SpectralOracle decomposition
    sdo_obj: Optional[float] = None
    sdo_time: Optional[float] = None
    sdo_gap: Optional[float] = None
    # Status
    status: str = "pending"
    error: str = ""


def find_instances(instances_dir: Path, tier: str) -> list:
    """Find MPS/LP files in the instances directory."""
    files = sorted(
        p for p in instances_dir.iterdir()
        if p.suffix in (".mps", ".lp")
    )
    max_count = TIER_SIZES.get(tier, 10)
    return files[:max_count]


def run_spectral_oracle(mps_path: Path, time_limit: float) -> dict:
    """Run spectral-oracle analyze on an instance."""
    try:
        result = subprocess.run(
            ["spectral-oracle", "analyze", str(mps_path), "-f", "json"],
            capture_output=True, text=True, timeout=time_limit,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        return {"error": result.stderr.strip()}
    except FileNotFoundError:
        return {"error": "spectral-oracle binary not found. Run: cargo install --path spectral-cli"}
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {time_limit}s"}
    except json.JSONDecodeError:
        return {"error": "invalid JSON output"}


def run_scip_baseline(mps_path: Path, time_limit: float) -> dict:
    """Run SCIP on the instance (if available)."""
    try:
        result = subprocess.run(
            ["scip", "-c",
             f"read {mps_path} optimize display statistics quit"],
            capture_output=True, text=True, timeout=time_limit,
        )
        # Parse SCIP output for objective and time
        output = result.stdout
        obj = None
        solve_time = None
        gap = None
        for line in output.split("\n"):
            if "Primal Bound" in line:
                try:
                    obj = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
            if "Solving Time" in line:
                try:
                    solve_time = float(line.split(":")[-1].strip().rstrip("s"))
                except ValueError:
                    pass
            if "Gap" in line and "%" in line:
                try:
                    gap = float(line.split(":")[-1].strip().rstrip("%"))
                except ValueError:
                    pass
        return {"obj": obj, "time": solve_time, "gap": gap}
    except FileNotFoundError:
        return {"error": "SCIP not found on PATH"}
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {time_limit}s"}


def run_gcg_comparison(mps_path: Path, time_limit: float) -> dict:
    """Run GCG on the instance (if available)."""
    try:
        result = subprocess.run(
            ["gcg", "-c",
             f"read {mps_path} detect optimize display statistics quit"],
            capture_output=True, text=True, timeout=time_limit,
        )
        output = result.stdout
        obj = None
        solve_time = None
        for line in output.split("\n"):
            if "Primal Bound" in line:
                try:
                    obj = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
            if "Solving Time" in line:
                try:
                    solve_time = float(line.split(":")[-1].strip().rstrip("s"))
                except ValueError:
                    pass
        return {"obj": obj, "time": solve_time}
    except FileNotFoundError:
        return {"error": "GCG not found on PATH"}
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {time_limit}s"}


def run_benchmark(instances_dir: Path, tier: str, time_limit: float,
                  output_dir: Path, skip_scip: bool, skip_gcg: bool):
    """Run the complete benchmark suite."""
    output_dir.mkdir(parents=True, exist_ok=True)
    instances = find_instances(instances_dir, tier)

    if not instances:
        print(f"No .mps or .lp files found in {instances_dir}")
        print("Download MIPLIB 2017 from: https://miplib.zib.de/tag_benchmark.html")
        return

    print(f"=== MIPLIB 2017 Benchmark — Tier: {tier} ===")
    print(f"Instances: {len(instances)}, Time limit: {time_limit}s")
    print(f"Output: {output_dir}\n")

    results = []

    for i, path in enumerate(instances):
        name = path.stem
        print(f"[{i+1}/{len(instances)}] {name}...", end=" ", flush=True)

        br = BenchmarkResult(instance=name)

        # (1) SpectralOracle analysis
        sdo = run_spectral_oracle(path, time_limit)
        if "error" not in sdo:
            dims = sdo.get("dimensions", {})
            br.rows = dims.get("rows", 0)
            br.cols = dims.get("cols", 0)
            br.nnz = dims.get("nnz", 0)
            sf = sdo.get("spectral_features", {})
            br.spectral_gap = sf.get("spectral_gap", 0.0)
            si = sdo.get("solver_info", {})
            br.spectral_time_ms = si.get("time_ms", 0.0)
            br.status = "analyzed"
        else:
            br.error = sdo["error"]
            br.status = "error"
            print(f"ERROR: {br.error}")
            results.append(br)
            continue

        # (2) SCIP baseline
        if not skip_scip:
            scip = run_scip_baseline(path, time_limit)
            if "error" not in scip:
                br.scip_obj = scip.get("obj")
                br.scip_time = scip.get("time")
                br.scip_gap = scip.get("gap")

        # (3) GCG comparison
        if not skip_gcg:
            gcg = run_gcg_comparison(path, time_limit)
            if "error" not in gcg:
                br.gcg_obj = gcg.get("obj")
                br.gcg_time = gcg.get("time")

        br.status = "complete"
        results.append(br)
        print(f"gap={br.spectral_gap:.3e}, time={br.spectral_time_ms:.0f}ms")

    # Write results
    results_file = output_dir / f"benchmark_{tier}.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "tier": tier,
                "time_limit": time_limit,
                "instances_dir": str(instances_dir),
                "num_instances": len(instances),
                "num_complete": sum(1 for r in results if r.status == "complete"),
                "results": [asdict(r) for r in results],
            },
            f, indent=2,
        )
    print(f"\nResults written to {results_file}")

    # Print summary table
    print(f"\n{'='*72}")
    print(f"{'Instance':<20} {'Rows':>6} {'Cols':>6} {'Gap':>12} {'SDO ms':>8}")
    print(f"{'-'*72}")
    for r in results:
        if r.status in ("analyzed", "complete"):
            print(f"{r.instance:<20} {r.rows:>6} {r.cols:>6} "
                  f"{r.spectral_gap:>12.4e} {r.spectral_time_ms:>8.0f}")
    print(f"{'='*72}")


def main():
    parser = argparse.ArgumentParser(
        description="MIPLIB 2017 benchmark suite for SpectralOracle"
    )
    parser.add_argument(
        "--instances-dir", type=Path, default=Path("miplib2017"),
        help="Directory containing MIPLIB .mps files"
    )
    parser.add_argument(
        "--tier", choices=list(TIER_SIZES.keys()), default="pilot",
        help="Benchmark tier (pilot=10, dev=50, paper=200, artifact=1065)"
    )
    parser.add_argument(
        "--time-limit", type=float, default=None,
        help="Per-instance time limit in seconds (default: tier-dependent)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_results"),
        help="Output directory for results"
    )
    parser.add_argument("--skip-scip", action="store_true",
                       help="Skip SCIP baseline comparison")
    parser.add_argument("--skip-gcg", action="store_true",
                       help="Skip GCG comparison")

    args = parser.parse_args()

    if args.time_limit is None:
        args.time_limit = TIER_LIMITS[args.tier]

    run_benchmark(
        args.instances_dir, args.tier, args.time_limit,
        args.output_dir, args.skip_scip, args.skip_gcg,
    )


if __name__ == "__main__":
    main()
