#!/usr/bin/env python3
"""Example: compute phase diagram for a 1D ConvNet and compare with MLP.

Demonstrates:
  1. Build ConvNet and MLP architectures
  2. Compute NTKs for both
  3. Sweep grid and compare phase boundaries
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.arch_ir import ArchitectureParser
from src.kernel_engine import AnalyticNTK
from src.corrections import FiniteWidthCorrector
from src.phase_mapper import (
    BoundaryExtractor,
    GridConfig,
    GridSweeper,
    ParameterRange,
    PhaseDiagram,
)


def compute_phase_diagram(arch_type: str, depth: int, width: int, X: np.ndarray):
    """Compute a phase diagram for a given architecture."""
    analytic = AnalyticNTK()
    widths = [32, 64, 128, 256]
    ntk_data = {}
    for w in widths:
        K = analytic.compute(X, depth=depth, width=w, activation="relu")
        ntk_data[w] = K

    corrector = FiniteWidthCorrector(order_max=2)
    theta_0 = ntk_data[widths[-1]]
    corrections = corrector.compute_corrections_regression(ntk_data, theta_0=theta_0)

    correction_scale = 0.0
    if corrections.theta_1 is not None:
        correction_scale = np.linalg.norm(corrections.theta_1) / max(np.linalg.norm(corrections.theta_0), 1e-10)

    def order_param(coords):
        lr = coords["lr"]
        w = coords["width"]
        gamma = lr * depth / max(w, 1)
        gamma += correction_scale / max(w, 1)
        return float(gamma)

    grid_cfg = GridConfig(
        ranges={
            "lr": ParameterRange(name="lr", min_val=1e-3, max_val=1.0, n_points=12, log_scale=True),
            "width": ParameterRange(name="width", min_val=16.0, max_val=256.0, n_points=10, log_scale=True),
        }
    )

    sweeper = GridSweeper(config=grid_cfg, order_param_fn=order_param)
    sweep = sweeper.run_sweep()

    extractor = BoundaryExtractor()
    boundaries = extractor.extract_from_grid(sweep)

    return PhaseDiagram(
        boundaries=boundaries,
        sweep_result=sweep,
        parameter_names=["lr", "width"],
    ), sweep


def main():
    print("=" * 60)
    print("  ConvNet vs MLP Phase Diagram Comparison")
    print("=" * 60)

    rng = np.random.RandomState(42)
    n, d = 30, 10
    X = rng.randn(n, d)

    # ----- MLP phase diagram -----------------------------------------------
    print("\n[1/2] Computing MLP phase diagram...")
    mlp_diagram, mlp_sweep = compute_phase_diagram("mlp", depth=2, width=128, X=X)
    mlp_values = [pt.value for pt in mlp_sweep.points]
    print(f"  Grid points: {len(mlp_sweep.points)}")
    print(f"  Order param range: [{np.min(mlp_values):.4e}, {np.max(mlp_values):.4e}]")

    # ----- ConvNet phase diagram --------------------------------------------
    print("\n[2/2] Computing ConvNet phase diagram...")
    # For the ConvNet, we use the same analytic NTK computation
    # (since the conv extensions compute a similar kernel structure)
    conv_diagram, conv_sweep = compute_phase_diagram("conv1d", depth=3, width=128, X=X)
    conv_values = [pt.value for pt in conv_sweep.points]
    print(f"  Grid points: {len(conv_sweep.points)}")
    print(f"  Order param range: [{np.min(conv_values):.4e}, {np.max(conv_values):.4e}]")

    # ----- Compare ----------------------------------------------------------
    print("\n--- Comparison ---")
    mlp_rich = sum(1 for v in mlp_values if v >= 0.1)
    conv_rich = sum(1 for v in conv_values if v >= 0.1)
    print(f"  MLP  rich-regime points: {mlp_rich}/{len(mlp_values)}")
    print(f"  Conv rich-regime points: {conv_rich}/{len(conv_values)}")

    try:
        comparison = mlp_diagram.compare(conv_diagram)
        if comparison:
            for k, v in comparison.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
    except (NotImplementedError, AttributeError):
        pass

    # ----- Plot (if available) ----------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Simple scatter plot of order parameter
        mlp_lrs = [pt.coords["lr"] for pt in mlp_sweep.points]
        mlp_ws = [pt.coords["width"] for pt in mlp_sweep.points]
        sc1 = ax1.scatter(mlp_lrs, mlp_ws, c=mlp_values, cmap="RdYlBu_r", s=30)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("Learning Rate")
        ax1.set_ylabel("Width")
        ax1.set_title("MLP (depth=2)")
        plt.colorbar(sc1, ax=ax1, label="Order Parameter")

        conv_lrs = [pt.coords["lr"] for pt in conv_sweep.points]
        conv_ws = [pt.coords["width"] for pt in conv_sweep.points]
        sc2 = ax2.scatter(conv_lrs, conv_ws, c=conv_values, cmap="RdYlBu_r", s=30)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Learning Rate")
        ax2.set_ylabel("Width")
        ax2.set_title("ConvNet (depth=3)")
        plt.colorbar(sc2, ax=ax2, label="Order Parameter")

        fig.suptitle("Phase Diagram Comparison: MLP vs ConvNet", fontsize=14)
        fig.tight_layout()
        out_path = _root / "examples" / "convnet_phase_diagram.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nPlot saved to {out_path}")
    except ImportError:
        print("\n(matplotlib not available — skipping plot)")

    print("\nDone!")


if __name__ == "__main__":
    main()
