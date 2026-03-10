#!/usr/bin/env python3
"""Sensitivity to near-faithfulness violations and power envelope.

The faithfulness assumption states that every conditional independence in the
data corresponds to d-separation in the true DAG.  Near-violations — where a
path exists but the effect nearly cancels — can fool CI tests.

This script demonstrates:

  1. How CausalCert behaves under *exact* faithfulness (strong signal).
  2. How results degrade as edge weights approach cancellation.
  3. The *power envelope*: the minimum detectable effect size at a given
     sample size and significance level.
  4. Discussion of limitations and practical guidance.

Run::

    python examples/sensitivity_to_faithfulness.py [--samples 2000]
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from causalcert.data.synthetic import generate_linear_gaussian
from causalcert.pipeline.config import PipelineRunConfig
from causalcert.pipeline.orchestrator import CausalCertPipeline
from causalcert.types import AuditReport


# =====================================================================
# DAG Definition
# =====================================================================

def _build_cancellation_dag() -> tuple[np.ndarray, list[str]]:
    """Build a 4-node diamond DAG where two paths can cancel.

        X  →  M1  →  Y
        X  →  M2  →  Y

    When w(X→M1)·w(M1→Y) = −w(X→M2)·w(M2→Y), the total effect of X on Y
    vanishes even though causal paths exist — a faithfulness violation.
    """
    names = ["X", "M1", "M2", "Y"]
    adj = np.zeros((4, 4), dtype=np.int8)
    adj[0, 1] = 1  # X → M1
    adj[0, 2] = 1  # X → M2
    adj[1, 3] = 1  # M1 → Y
    adj[2, 3] = 1  # M2 → Y
    return adj, names


def _make_weights(
    adj: np.ndarray,
    direct_strength: float = 0.6,
    cancellation_factor: float = 0.0,
) -> np.ndarray:
    """Construct edge weights with controllable cancellation.

    Parameters
    ----------
    direct_strength : baseline weight magnitude on each edge.
    cancellation_factor : float in [0, 1].
        0 = no cancellation (faithful), 1 = exact cancellation.
    """
    w = np.zeros_like(adj, dtype=np.float64)
    w[0, 1] = direct_strength                        # X → M1
    w[1, 3] = direct_strength                        # M1 → Y
    w[0, 2] = direct_strength                        # X → M2
    # Negate and scale M2 → Y to create cancellation
    w[2, 3] = -direct_strength * cancellation_factor  # M2 → Y
    return w


def _generate_data(
    adj: np.ndarray,
    weights: np.ndarray,
    names: list[str],
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    data = generate_linear_gaussian(
        adj_matrix=adj,
        weights=weights,
        n_samples=n_samples,
        noise_scale=1.0,
        seed=seed,
    )
    return pd.DataFrame(data, columns=names)


# =====================================================================
# Experiment Results
# =====================================================================

@dataclass
class FaithfulnessResult:
    cancellation: float
    total_effect_true: float
    radius_lower: int
    radius_upper: int
    n_fragile: int
    ate_estimate: float | None
    runtime_s: float


# =====================================================================
# 1. Cancellation Sweep
# =====================================================================

CANCELLATION_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]


def cancellation_sweep(
    n_samples: int = 2000,
    seed: int = 42,
) -> list[FaithfulnessResult]:
    """Run CausalCert at varying levels of path cancellation."""
    adj, names = _build_cancellation_dag()
    results: list[FaithfulnessResult] = []

    for cancel in CANCELLATION_LEVELS:
        weights = _make_weights(adj, direct_strength=0.6, cancellation_factor=cancel)
        # Theoretical total effect: path1 + path2 = 0.6*0.6 + 0.6*(-0.6*cancel)
        total_true = 0.36 + 0.6 * (-0.6 * cancel)

        df = _generate_data(adj, weights, names, n_samples, seed)
        config = PipelineRunConfig(
            treatment=0, outcome=3, alpha=0.05, solver_strategy="auto",
        )
        pipeline = CausalCertPipeline(config)
        t0 = time.perf_counter()
        report = pipeline.run(adj_matrix=adj, data=df)
        elapsed = time.perf_counter() - t0

        n_frag = sum(1 for fs in report.fragility_ranking if fs.score >= 0.4)
        ate = report.estimation_result.estimate if report.estimation_result else None

        results.append(FaithfulnessResult(
            cancellation=cancel,
            total_effect_true=total_true,
            radius_lower=report.radius.lower_bound,
            radius_upper=report.radius.upper_bound,
            n_fragile=n_frag,
            ate_estimate=ate,
            runtime_s=elapsed,
        ))

        status = "✓" if report.radius.lower_bound > 0 else "⚠"
        print(f"  cancel={cancel:.2f}  ATE_true={total_true:>+.4f}  "
              f"r=[{report.radius.lower_bound},{report.radius.upper_bound}]  {status}")

    return results


def print_cancellation_table(results: list[FaithfulnessResult]) -> None:
    print()
    print("=" * 78)
    print("Cancellation Sweep Results")
    print("=" * 78)
    header = (f"{'Cancel':>7s}  {'ATE_true':>9s}  {'ATE_est':>9s}  "
              f"{'r_lo':>4s}  {'r_hi':>4s}  {'#Frag':>5s}  {'Time':>7s}")
    print(header)
    print("-" * 78)
    for r in results:
        ate_str = f"{r.ate_estimate:.4f}" if r.ate_estimate is not None else "N/A"
        print(f"{r.cancellation:>7.2f}  {r.total_effect_true:>+9.4f}  "
              f"{ate_str:>9s}  {r.radius_lower:>4d}  {r.radius_upper:>4d}  "
              f"{r.n_fragile:>5d}  {r.runtime_s:>6.2f}s")
    print()


# =====================================================================
# 2. Power Envelope
# =====================================================================

SAMPLE_SIZES_POWER = [100, 250, 500, 1000, 2000, 5000]
EFFECT_SIZES = [0.05, 0.10, 0.20, 0.40]


@dataclass
class PowerResult:
    n_samples: int
    effect_size: float
    detected: bool
    radius_lower: int


def power_envelope(seed: int = 42) -> list[PowerResult]:
    """Estimate the power envelope: minimum effect size detectable at each n."""
    adj, names = _build_cancellation_dag()
    results: list[PowerResult] = []

    for n in SAMPLE_SIZES_POWER:
        for eff in EFFECT_SIZES:
            # Set cancellation so that total effect = eff
            # total = 0.36 - 0.36*cancel → cancel = (0.36 - eff) / 0.36
            cancel = max(0.0, min(1.0, (0.36 - eff) / 0.36))
            weights = _make_weights(adj, 0.6, cancel)
            df = _generate_data(adj, weights, names, n, seed)

            config = PipelineRunConfig(
                treatment=0, outcome=3, alpha=0.05, solver_strategy="auto",
            )
            pipeline = CausalCertPipeline(config)
            report = pipeline.run(adj_matrix=adj, data=df)

            detected = report.radius.lower_bound >= 1
            results.append(PowerResult(
                n_samples=n,
                effect_size=eff,
                detected=detected,
                radius_lower=report.radius.lower_bound,
            ))

    return results


def print_power_matrix(results: list[PowerResult]) -> None:
    """Print a sample-size × effect-size detection matrix."""
    print("=" * 60)
    print("Power Envelope  (✓ = detected, · = missed)")
    print("=" * 60)

    # Build matrix
    ns = sorted(set(r.n_samples for r in results))
    effs = sorted(set(r.effect_size for r in results))

    lookup: dict[tuple[int, float], bool] = {
        (r.n_samples, r.effect_size): r.detected for r in results
    }

    header = f"{'n':>6s}  " + "  ".join(f"{e:.2f}" for e in effs)
    print(header)
    print("-" * 60)
    for n in ns:
        cells = []
        for e in effs:
            cells.append(" ✓ " if lookup.get((n, e), False) else " · ")
        print(f"{n:>6d}  {'  '.join(cells)}")
    print()


# =====================================================================
# 3. Discussion
# =====================================================================

DISCUSSION = """\
╔══════════════════════════════════════════════════════════════════╗
║                          Discussion                             ║
╚══════════════════════════════════════════════════════════════════╝

Near-faithfulness violations reduce the statistical signal available
for conditional-independence tests, which has two downstream effects:

  1. **Inflated fragility** — edges on near-cancelling paths receive
     higher fragility scores because their removal or reversal changes
     d-separation verdicts that are already borderline.

  2. **Radius underestimation** — when a CI test fails to detect a
     true dependence, the solver may find a smaller edit set that
     "overturns" the (erroneously weaker) conclusion.

Practical guidance:

  • If CausalCert reports a radius ≤ 1 **and** the estimated ATE is
    small, investigate whether path cancellation is plausible.
  • Use the Cauchy-ensemble CI test (default) to improve power.
  • Increase the sample size: the power envelope shows the minimum n
    needed to detect effects of a given magnitude.
  • Consider domain knowledge: if two paths are known to have opposite
    signs, faithfulness may be violated by design, not by noise.

Limitations:

  • CausalCert assumes an *exact* DAG.  If the true generating process
    involves latent variables or feedback, the radius may be misleading.
  • The radius is defined with respect to *single-edge* edits.  Block
    perturbations (e.g., adding a latent common cause) are not covered.
  • Statistical power bounds are approximate and assume linear Gaussian
    SEMs; real-world power may differ.
"""


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sensitivity to near-faithfulness violations.",
    )
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(textwrap.dedent("""\
    ╔══════════════════════════════════════════════════════════╗
    ║    CausalCert — Faithfulness Sensitivity Analysis        ║
    ╚══════════════════════════════════════════════════════════╝
    """))

    # 1. Cancellation sweep
    print("1. Cancellation sweep:\n")
    cancel_results = cancellation_sweep(n_samples=args.samples, seed=args.seed)
    print_cancellation_table(cancel_results)

    # 2. Power envelope
    print("2. Power envelope:\n")
    power_results = power_envelope(seed=args.seed)
    print_power_matrix(power_results)

    # 3. Discussion
    print(DISCUSSION)

    print("Done.")


if __name__ == "__main__":
    main()
