#!/usr/bin/env python3
"""
Run sensitivity analysis experiment.

Generates synthetic data with known ground truth, runs all
hyperparameter sensitivity sweeps, saves results as JSON, and
generates pgfplots-compatible .dat files.

Usage
-----
    python experiments/run_sensitivity.py [--output-dir results/sensitivity]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the implementation package is importable
_IMPL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

from causal_trading.market.synthetic import SyntheticMarketGenerator
from causal_trading.evaluation.sensitivity_analysis import (
    SensitivityAnalyzer,
    SensitivityReport,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def generate_data(
    n_regimes: int = 3,
    n_features: int = 5,
    T: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic market data with known ground truth."""
    gen = SyntheticMarketGenerator(
        n_features=n_features,
        n_regimes=n_regimes,
        seed=seed,
    )
    dataset = gen.generate(T=T, n_regimes=n_regimes, n_features=n_features)
    logger.info(
        "Generated %d-step dataset with %d regimes and %d features",
        T, n_regimes, n_features,
    )
    return dataset.features


def save_pgfplots(report: SensitivityReport, output_dir: Path) -> None:
    """Save pgfplots-compatible .dat files."""
    dat_dir = output_dir / "pgfplots"
    dat_dir.mkdir(parents=True, exist_ok=True)
    for name, content in report.to_pgfplots_data().items():
        path = dat_dir / f"{name}.dat"
        path.write_text(content)
        logger.info("Wrote %s", path)


def save_latex_table(report: SensitivityReport, output_dir: Path) -> None:
    """Save LaTeX table."""
    path = output_dir / "sensitivity_table.tex"
    path.write_text(report.to_latex_table())
    logger.info("Wrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sensitivity analysis")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sensitivity",
        help="Output directory",
    )
    parser.add_argument("--n-regimes", type=int, default=3)
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate data
    logger.info("=== Step 1: Generate synthetic data ===")
    data = generate_data(
        n_regimes=args.n_regimes,
        n_features=args.n_features,
        T=args.T,
        seed=args.seed,
    )

    # 2. Run sweeps
    logger.info("=== Step 2: Run sensitivity sweeps ===")
    analyzer = SensitivityAnalyzer()
    t0 = time.time()
    report = analyzer.full_sensitivity_report(
        data,
        n_iter=50,
        burn_in=10,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    logger.info("Sweeps completed in %.1f seconds", elapsed)

    # 3. Save JSON
    logger.info("=== Step 3: Save results ===")
    report.metadata["data_config"] = {
        "n_regimes": args.n_regimes,
        "n_features": args.n_features,
        "T": args.T,
        "seed": args.seed,
    }
    report.save(output_dir / "sensitivity_report.json")
    logger.info("Saved JSON report to %s", output_dir / "sensitivity_report.json")

    # 4. Generate pgfplots data
    logger.info("=== Step 4: Generate pgfplots data ===")
    save_pgfplots(report, output_dir)

    # 5. LaTeX table
    save_latex_table(report, output_dir)

    # 6. Summary
    most_sensitive = report.most_sensitive_param()
    logger.info("=== Summary ===")
    logger.info("Most sensitive parameter: %s", most_sensitive)
    for name, points in report.sweeps.items():
        lo, hi = report.robust_range(name)
        logger.info("  %s: robust range [%.4g, %.4g]", name, lo, hi)

    logger.info("Done. Results in %s", output_dir)


if __name__ == "__main__":
    main()
