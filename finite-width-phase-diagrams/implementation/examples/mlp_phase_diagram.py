#!/usr/bin/env python3
"""Example: compute a phase diagram for a 2-layer ReLU MLP on synthetic data.

Demonstrates the full workflow:
  1. Specify architecture via dict
  2. Compute analytic NTK at multiple widths
  3. Fit finite-width corrections
  4. Sweep hyperparameter grid (lr × width)
  5. Extract phase boundaries
  6. Plot the phase diagram
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure src/ is importable
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.arch_ir import ArchitectureParser
from src.kernel_engine import AnalyticNTK, EmpiricalNTK
from src.corrections import FiniteWidthCorrector
from src.phase_mapper import (
    BoundaryExtractor,
    GridConfig,
    GridSweeper,
    ParameterRange,
    PhaseDiagram,
)
from src.utils.config import PhaseDiagramConfig


def main():
    print("=" * 60)
    print("  MLP Phase Diagram Example")
    print("=" * 60)

    # ----- 1. Architecture specification ------------------------------------
    parser = ArchitectureParser()
    graph = parser.parse_dict({
        "type": "mlp",
        "depth": 2,
        "width": 128,
        "activation": "relu",
        "input_dim": 10,
        "output_dim": 1,
    })
    print(f"\nArchitecture: {graph}")
    print(f"  Total parameters: {graph.total_parameters()}")

    # ----- 2. Generate synthetic data ---------------------------------------
    rng = np.random.RandomState(42)
    n_train = 50
    d_in = 10
    X = rng.randn(n_train, d_in)
    print(f"\nSynthetic data: n={n_train}, d={d_in}")

    # ----- 3. Compute NTKs at multiple widths -------------------------------
    print("\nComputing NTKs at calibration widths...")
    analytic = AnalyticNTK(depth=2, activation="relu")
    widths = [32, 64, 128, 256, 512]
    # Compute infinite-width NTK as the base
    K_inf = analytic.compute_ntk(X)
    # Simulate finite-width NTKs with 1/N perturbations for calibration
    ntk_list = []
    for w in widths:
        perturbation = rng.randn(n_train, n_train)
        perturbation = (perturbation + perturbation.T) / (2 * w)
        K_w = K_inf + perturbation
        ntk_list.append(K_w)
        print(f"  width={w:4d}: ||K||_F={np.linalg.norm(K_w, 'fro'):.2f}, "
              f"cond={np.linalg.cond(K_w):.1e}")
    ntk_measurements = np.array(ntk_list)  # (K, n, n)

    # ----- 4. Fit finite-width corrections ----------------------------------
    print("\nFitting 1/N corrections...")
    corrector = FiniteWidthCorrector()
    result = corrector.compute_corrections_regression(ntk_measurements, widths)
    print(f"  θ₀ norm: {np.linalg.norm(result.theta_0):.4f}")
    if result.theta_1 is not None:
        print(f"  θ₁ norm: {np.linalg.norm(result.theta_1):.4f}")

    # ----- 5. Grid sweep: order parameter over (lr, width) ------------------
    print("\nSweeping hyperparameter grid...")
    grid_cfg = GridConfig(
        learning_rate=ParameterRange(
            min_val=1e-3, max_val=1.0,
            num_points=15, log_scale=True,
        ),
        width=ParameterRange(
            min_val=16.0, max_val=512.0,
            num_points=12, log_scale=True,
        ),
    )

    corrections = result

    def order_param(coords):
        lr = coords["learning_rate"]
        width = coords["width"]
        gamma = lr * 2.0 / max(width, 1)  # depth=2
        if corrections.theta_1 is not None:
            correction_scale = np.linalg.norm(corrections.theta_1) / np.linalg.norm(corrections.theta_0)
            gamma += correction_scale / max(width, 1)
        return float(gamma)

    sweeper = GridSweeper(config=grid_cfg, order_param_fn=order_param)
    sweep_result = sweeper.run_sweep()
    print(f"  Grid points evaluated: {len(sweep_result.grid_points)}")

    # ----- 6. Extract boundaries --------------------------------------------
    print("\nExtracting phase boundaries...")
    extractor = BoundaryExtractor()
    boundaries = extractor.extract_from_grid(sweep_result)
    print(f"  Boundary curves found: {len(boundaries)}")

    # ----- 7. Build phase diagram -------------------------------------------
    diagram = PhaseDiagram(
        boundary_curves=boundaries,
        parameter_names=("learning_rate", "width"),
        parameter_ranges={
            "learning_rate": (1e-3, 1.0),
            "width": (16.0, 512.0),
        },
    )
    print("\nPhase diagram constructed.")

    # ----- 8. Analyse -------------------------------------------------------
    values = [pt.order_parameter_value for pt in sweep_result.grid_points]
    print(f"\nOrder parameter statistics:")
    print(f"  min = {np.min(values):.4e}")
    print(f"  max = {np.max(values):.4e}")
    print(f"  mean = {np.mean(values):.4e}")

    lazy_count = sum(1 for v in values if v < 0.1)
    rich_count = sum(1 for v in values if v >= 0.1)
    print(f"\nRegime counts (threshold=0.1):")
    print(f"  Lazy: {lazy_count}")
    print(f"  Rich: {rich_count}")

    # ----- 9. Plot (if matplotlib available) --------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.visualization import PhaseDiagramPlotter, PlotConfig

        fig, ax = plt.subplots(figsize=(10, 8))
        plotter = PhaseDiagramPlotter(config=PlotConfig(dpi=150))
        plotter.plot_phase_diagram(diagram, ax=ax)
        out_path = _root / "examples" / "mlp_phase_diagram.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nPlot saved to {out_path}")
    except ImportError:
        print("\n(matplotlib not available — skipping plot)")

    print("\nDone!")


if __name__ == "__main__":
    main()
