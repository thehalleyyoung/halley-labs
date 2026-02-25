"""Command-line interface for the finite-width phase diagram system.

Provides subcommands: compute, calibrate, map, evaluate, visualize,
retrodiction. Each subcommand parses arguments, loads config, runs the
appropriate pipeline stage, and saves results.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        if getattr(args, "verbose", False):
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="fwpd",
        description="Finite-Width Phase Diagram computation system.",
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to configuration YAML/JSON file.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress non-error output.",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Config overrides as key=value pairs.",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ---- compute -----------------------------------------------------------
    p_compute = sub.add_parser("compute", help="Run full phase diagram computation.")
    p_compute.add_argument("--profile", choices=["quick", "standard", "thorough", "research"],
                           default="standard", help="Configuration profile.")
    p_compute.add_argument("--arch", type=str, default=None, help="Architecture DSL string.")
    p_compute.add_argument("--depth", type=int, default=None)
    p_compute.add_argument("--width", type=int, default=None)
    p_compute.add_argument("--activation", type=str, default=None)
    p_compute.add_argument("--resume", action="store_true", help="Resume from checkpoint.")
    p_compute.add_argument("--workers", type=int, default=None)
    p_compute.set_defaults(func=cmd_compute)

    # ---- calibrate ---------------------------------------------------------
    p_cal = sub.add_parser("calibrate", help="Run calibration pipeline.")
    p_cal.add_argument("--widths", nargs="+", type=int, default=None)
    p_cal.add_argument("--seeds", type=int, default=None)
    p_cal.add_argument("--bootstrap", type=int, default=None)
    p_cal.add_argument("--profile", choices=["quick", "standard", "thorough", "research"],
                       default="standard")
    p_cal.set_defaults(func=cmd_calibrate)

    # ---- map ---------------------------------------------------------------
    p_map = sub.add_parser("map", help="Compute phase map / grid sweep.")
    p_map.add_argument("--lr-range", nargs=2, type=float, default=None)
    p_map.add_argument("--width-range", nargs=2, type=int, default=None)
    p_map.add_argument("--lr-points", type=int, default=None)
    p_map.add_argument("--width-points", type=int, default=None)
    p_map.add_argument("--adaptive", action="store_true")
    p_map.add_argument("--profile", choices=["quick", "standard", "thorough", "research"],
                       default="standard")
    p_map.set_defaults(func=cmd_map)

    # ---- evaluate ----------------------------------------------------------
    p_eval = sub.add_parser("evaluate", help="Evaluate predictions against ground truth.")
    p_eval.add_argument("--predicted", type=str, required=False,
                        help="Path to predicted phase diagram.")
    p_eval.add_argument("--ground-truth-seeds", type=int, default=None)
    p_eval.add_argument("--profile", choices=["quick", "standard", "thorough", "research"],
                        default="standard")
    p_eval.set_defaults(func=cmd_evaluate)

    # ---- visualize ---------------------------------------------------------
    p_viz = sub.add_parser("visualize", help="Generate phase diagram plots.")
    p_viz.add_argument("--input", type=str, required=False,
                       help="Path to phase diagram data.")
    p_viz.add_argument("--format", choices=["png", "pdf", "svg"], default="png")
    p_viz.add_argument("--dpi", type=int, default=150)
    p_viz.add_argument("--comparison", type=str, default=None,
                       help="Path to ground-truth diagram for comparison plot.")
    p_viz.set_defaults(func=cmd_visualize)

    # ---- retrodiction ------------------------------------------------------
    p_retro = sub.add_parser("retrodiction", help="Validate against known theoretical results.")
    p_retro.add_argument("--profile", choices=["quick", "standard", "thorough", "research"],
                         default="standard")
    p_retro.set_defaults(func=cmd_retrodiction)

    return parser


# ---------------------------------------------------------------------------
# Config loading helper
# ---------------------------------------------------------------------------

def _load_config(args: argparse.Namespace) -> "PhaseDiagramConfig":
    """Load and merge configuration from file, profile, and CLI overrides."""
    from .utils.config import PhaseDiagramConfig, apply_cli_overrides

    profile = getattr(args, "profile", "standard")
    if args.config:
        cfg = PhaseDiagramConfig.load_yaml(args.config) if args.config.endswith(
            (".yaml", ".yml")
        ) else PhaseDiagramConfig.from_json(Path(args.config).read_text())
    else:
        cfg = PhaseDiagramConfig.from_profile(profile)

    # Apply direct CLI arguments
    overrides: Dict[str, Any] = {}
    if getattr(args, "depth", None) is not None:
        overrides["architecture.depth"] = args.depth
    if getattr(args, "width", None) is not None:
        overrides["architecture.width"] = args.width
    if getattr(args, "activation", None) is not None:
        overrides["architecture.activation"] = args.activation
    if args.output_dir:
        overrides["output.output_dir"] = args.output_dir
    if getattr(args, "workers", None) is not None:
        overrides["parallel.n_workers"] = args.workers
    if getattr(args, "widths", None) is not None:
        overrides["calibration.widths"] = args.widths
    if getattr(args, "seeds", None) is not None:
        overrides["calibration.num_seeds"] = args.seeds
    if getattr(args, "bootstrap", None) is not None:
        overrides["calibration.bootstrap_samples"] = args.bootstrap
    if getattr(args, "lr_range", None) is not None:
        overrides["grid.lr_range"] = tuple(args.lr_range)
    if getattr(args, "width_range", None) is not None:
        overrides["grid.width_range"] = tuple(args.width_range)
    if getattr(args, "lr_points", None) is not None:
        overrides["grid.lr_points"] = args.lr_points
    if getattr(args, "width_points", None) is not None:
        overrides["grid.width_points"] = args.width_points

    if overrides:
        cfg = cfg.merge(overrides)
    if args.override:
        cfg = apply_cli_overrides(cfg, args.override)

    # Verbosity
    if args.verbose:
        cfg = cfg.with_overrides(output__verbose=True)
    if args.quiet:
        cfg = cfg.with_overrides(output__verbose=False)

    return cfg


def _setup_logging(args: argparse.Namespace) -> None:
    """Configure logging based on verbosity flags."""
    from .utils.logging import set_global_level
    import logging

    if args.quiet:
        set_global_level(logging.WARNING)
    elif args.verbose:
        set_global_level(logging.DEBUG)
    else:
        set_global_level(logging.INFO)


def _print_header(command: str, cfg: Any) -> None:
    """Print a header with command and key config info."""
    print(f"{'=' * 60}")
    print(f"  Finite-Width Phase Diagrams — {command}")
    print(f"{'=' * 60}")
    if hasattr(cfg, "summary"):
        print(cfg.summary())
    print()


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_compute(args: argparse.Namespace) -> int:
    """Run full phase diagram computation."""
    _setup_logging(args)
    cfg = _load_config(args)
    cfg.validate_or_raise()

    if not args.quiet:
        _print_header("compute", cfg)

    from .pipeline import PhaseDiagramPipeline

    pipeline = PhaseDiagramPipeline(cfg)
    result = pipeline.run(resume=getattr(args, "resume", False))

    out_dir = Path(cfg.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from .utils.io import save_phase_diagram
    save_phase_diagram(result["phase_diagram"], out_dir / "phase_diagram", format=cfg.output.format)

    if not args.quiet:
        print(f"\nResults saved to {out_dir}")
        if "report" in result:
            print(result["report"])
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Run calibration pipeline."""
    _setup_logging(args)
    cfg = _load_config(args)
    cfg.validate_or_raise()

    if not args.quiet:
        _print_header("calibrate", cfg)

    from .pipeline import PhaseDiagramPipeline

    pipeline = PhaseDiagramPipeline(cfg)
    result = pipeline.run_calibration()

    out_dir = Path(cfg.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from .utils.io import save_calibration
    save_calibration(result, out_dir / "calibration")

    if not args.quiet:
        print(f"\nCalibration results saved to {out_dir}")
    return 0


def cmd_map(args: argparse.Namespace) -> int:
    """Run phase mapping / grid sweep."""
    _setup_logging(args)
    cfg = _load_config(args)
    cfg.validate_or_raise()

    if not args.quiet:
        _print_header("map", cfg)

    from .pipeline import PhaseDiagramPipeline

    pipeline = PhaseDiagramPipeline(cfg)
    result = pipeline.run_mapping()

    out_dir = Path(cfg.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from .utils.io import save_phase_diagram
    save_phase_diagram(result, out_dir / "phase_map", format=cfg.output.format)

    if not args.quiet:
        print(f"\nPhase map saved to {out_dir}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate predictions against ground truth."""
    _setup_logging(args)
    cfg = _load_config(args)
    cfg.validate_or_raise()

    if not args.quiet:
        _print_header("evaluate", cfg)

    from .pipeline import PhaseDiagramPipeline

    pipeline = PhaseDiagramPipeline(cfg)
    result = pipeline.run_evaluation(
        predicted_path=getattr(args, "predicted", None)
    )

    out_dir = Path(cfg.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from .utils.io import save_json
    save_json(result, out_dir / "evaluation.json")

    if not args.quiet:
        print(f"\nEvaluation results saved to {out_dir}")
        if isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, (int, float, str)):
                    print(f"  {k}: {v}")
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    """Generate phase diagram plots."""
    _setup_logging(args)
    cfg = _load_config(args)

    if not args.quiet:
        _print_header("visualize", cfg)

    from .utils.io import load_phase_diagram
    from .visualization import PhaseDiagramPlotter, PlotConfig

    input_path = getattr(args, "input", None)
    if input_path is None:
        input_path = str(Path(cfg.output.output_dir) / "phase_diagram")

    diagram_data = load_phase_diagram(input_path)

    plot_cfg = PlotConfig(dpi=args.dpi)
    plotter = PhaseDiagramPlotter(config=plot_cfg)

    out_dir = Path(cfg.output.output_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        plotter.plot_phase_diagram(diagram_data, ax=ax)
        fig_path = out_dir / f"phase_diagram.{args.format}"
        fig.savefig(str(fig_path), dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        if not args.quiet:
            print(f"Phase diagram plot saved to {fig_path}")

        if args.comparison:
            gt_data = load_phase_diagram(args.comparison)
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            plotter.plot_comparison(diagram_data, gt_data, ax=ax2)
            cmp_path = out_dir / f"comparison.{args.format}"
            fig2.savefig(str(cmp_path), dpi=args.dpi, bbox_inches="tight")
            plt.close(fig2)
            if not args.quiet:
                print(f"Comparison plot saved to {cmp_path}")

    except ImportError:
        print("matplotlib is required for visualization", file=sys.stderr)
        return 1

    return 0


def cmd_retrodiction(args: argparse.Namespace) -> int:
    """Validate against known theoretical results."""
    _setup_logging(args)
    cfg = _load_config(args)
    cfg.validate_or_raise()

    if not args.quiet:
        _print_header("retrodiction", cfg)

    from .pipeline import PhaseDiagramPipeline

    pipeline = PhaseDiagramPipeline(cfg)
    results = pipeline.run_retrodiction()

    out_dir = Path(cfg.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from .utils.io import save_json
    save_json(results, out_dir / "retrodiction.json")

    if not args.quiet:
        print(f"\nRetrodiction results saved to {out_dir}")
        if isinstance(results, list):
            for r in results:
                name = r.get("name", "?")
                passed = r.get("passed", False)
                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] {name}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
