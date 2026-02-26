"""
Master experiment runner for CoaCert-TLA evaluation.

Single entry point that runs all experiments:
    - Benchmarks (via BenchmarkRunner)
    - Ablation studies (via AblationRunner)
    - Baseline comparisons (via BaselineComparisonRunner)
    - Scalability analysis (via ScalabilityRunner)
    - Bloom filter soundness analysis (via BloomSoundnessAnalyzer)

Usage::

    python3 -m coacert.evaluation.run_all_experiments [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

from .baseline_comparison import (
    LTS,
    BaselineComparisonRunner,
    PaigeTarjanBaseline,
)
from .bloom_soundness import (
    BloomSoundnessAnalyzer,
    build_soundness_certificate,
    compare_witness_sizes,
)
from .scalability import (
    ScalabilityRunner,
    ParameterizedBenchmark,
    SpecFamily,
    ALL_SCALABILITY_EXPERIMENTS,
    generate_two_phase_commit,
    generate_peterson,
    generate_dining_philosophers,
    generate_token_ring,
)
from .statistical import compute_summary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "experiment_results"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Run all evaluation experiments and save results.

    Parameters
    ----------
    output_dir : str
        Directory to write JSON result files.
    verbose : bool
        Print progress to stdout.
    """

    def __init__(
        self,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        verbose: bool = True,
    ) -> None:
        self._output_dir = output_dir
        self._verbose = verbose
        os.makedirs(output_dir, exist_ok=True)

    def run_all(self) -> Dict[str, Any]:
        """Run all experiments and return combined results."""
        t0 = time.monotonic()
        results: Dict[str, Any] = {}

        self._log("=" * 60)
        self._log("CoaCert-TLA: Running all experiments")
        self._log("=" * 60)

        results["bloom_soundness"] = self.run_bloom_analysis()
        results["baseline_comparison"] = self.run_baseline_comparisons()
        results["scalability"] = self.run_scalability()
        results["summary"] = {
            "total_time_seconds": time.monotonic() - t0,
            "output_dir": self._output_dir,
        }

        # Save combined results
        combined_path = os.path.join(self._output_dir, "all_results.json")
        with open(combined_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        self._log(f"\nAll results saved to {combined_path}")

        return results

    # -- Bloom soundness analysis -------------------------------------------

    def run_bloom_analysis(self) -> Dict[str, Any]:
        """Run Bloom filter soundness analysis experiments."""
        self._log("\n--- Bloom Filter Soundness Analysis ---")

        analyzer = BloomSoundnessAnalyzer(target_soundness=0.999)

        # Analyze typical configurations
        configs = [
            {"m": 1024, "k": 7, "n": 100, "V": 500},
            {"m": 4096, "k": 10, "n": 200, "V": 1000},
            {"m": 16384, "k": 12, "n": 500, "V": 5000},
            {"m": 65536, "k": 14, "n": 1000, "V": 10000},
        ]

        analyses = []
        for cfg in configs:
            bound = analyzer.analyze(cfg["m"], cfg["k"], cfg["n"], cfg["V"])
            analyses.append(bound.to_dict())
            self._log(
                f"  m={cfg['m']:>6}, k={cfg['k']:>2}, n={cfg['n']:>5}, "
                f"V={cfg['V']:>6} => FPR={bound.per_query_fpr:.2e}, "
                f"soundness={bound.soundness_level:.6f}"
            )

        # Witness size comparison
        witness_comp = compare_witness_sizes(
            num_states=1000, num_transitions=5000,
            bloom_bits=16384,
        )

        # Build a certificate
        cert = build_soundness_certificate(
            bloom_bits=16384, bloom_k=12, bloom_n=500,
            verification_checks=5000,
            full_witness_size_bytes=witness_comp["full_witness_bytes"],
            bloom_witness_size_bytes=witness_comp["bloom_witness_bytes"],
        )

        result = {
            "analyses": analyses,
            "witness_comparison": witness_comp,
            "certificate": cert.to_dict(),
            "sensitivity": analyzer.sensitivity_table(500, 5000),
        }

        self._save("bloom_soundness.json", result)
        return result

    # -- Baseline comparisons -----------------------------------------------

    def run_baseline_comparisons(self) -> Dict[str, Any]:
        """Run baseline comparison experiments."""
        self._log("\n--- Baseline Comparisons ---")

        runner = BaselineComparisonRunner(num_runs=3, timeout=60.0)

        # Generate small test LTS instances
        specs = [
            ("dining_phil_3", generate_dining_philosophers(3)),
            ("token_ring_4", generate_token_ring(4)),
            ("two_phase_2", generate_two_phase_commit(2)),
        ]

        reports = []
        for name, lts in specs:
            self._log(f"  Comparing on {name} ({lts.num_states} states)")
            report = runner.compare(name, lts)
            reports.append(report.to_dict())
            if report.blocks_match():
                self._log(f"    ✓ All algorithms agree on partition size")
            else:
                self._log(f"    ✗ Partition sizes differ!")

        result = {"comparisons": reports}
        self._save("baseline_comparison.json", result)
        return result

    # -- Scalability --------------------------------------------------------

    def run_scalability(self) -> Dict[str, Any]:
        """Run scalability experiments."""
        self._log("\n--- Scalability Analysis ---")

        runner = ScalabilityRunner(num_runs=1, timeout=30.0)

        reports = []
        # Use smaller parameter ranges for tractability
        experiments = [
            ParameterizedBenchmark(
                family=SpecFamily.DINING_PHILOSOPHERS,
                parameter_name="philosophers",
                parameter_values=[2, 3, 4],
            ),
            ParameterizedBenchmark(
                family=SpecFamily.TOKEN_RING,
                parameter_name="processes",
                parameter_values=[4, 8, 16],
            ),
        ]

        for benchmark in experiments:
            self._log(f"  Running {benchmark.family.name}")
            report = runner.run(benchmark)
            reports.append(report.to_dict())
            self._log(
                f"    {len(report.valid_points)} valid data points"
            )
            if report.time_fit:
                self._log(f"    Time fit: {report.time_fit.formula}")

        result = {"reports": reports}
        self._save("scalability.json", result)
        return result

    # -- Helpers ------------------------------------------------------------

    def _save(self, filename: str, data: Any) -> None:
        path = os.path.join(self._output_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(msg, flush=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all CoaCert-TLA evaluation experiments",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for result JSON files (default: %(default)s)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    runner = ExperimentRunner(
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
    runner.run_all()


if __name__ == "__main__":
    main()
