"""Command-line interface for the Causal-Plasticity Atlas.

Provides a ``cpa`` CLI with commands for running the pipeline,
analyzing results, generating visualizations, and running benchmarks.

Usage
-----
::

    python -m cpa run --data data/ --output results/ --profile standard
    python -m cpa analyze results/ --summary
    python -m cpa visualize results/ --heatmap --dashboard
    python -m cpa benchmark --generator fsvp --n-reps 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entry point.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments (defaults to sys.argv[1:]).

    Returns
    -------
    int
        Exit code (0 = success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "command") or args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "run":
            return _cmd_run(args)
        elif args.command == "analyze":
            return _cmd_analyze(args)
        elif args.command == "visualize":
            return _cmd_visualize(args)
        elif args.command == "benchmark":
            return _cmd_benchmark(args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


# =====================================================================
# Argument parser
# =====================================================================


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="cpa",
        description="Causal-Plasticity Atlas: Map mechanism invariance, "
                    "plasticity, and emergence across causal contexts.",
    )
    parser.add_argument(
        "--version", action="version", version="cpa 0.1.0"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # --- run ---
    run_parser = subparsers.add_parser(
        "run", help="Run the CPA pipeline"
    )
    run_parser.add_argument(
        "--data", "-d", required=True,
        help="Path to data directory or file (CSV, NPZ, Parquet)",
    )
    run_parser.add_argument(
        "--output", "-o", default="cpa_output",
        help="Output directory (default: cpa_output)",
    )
    run_parser.add_argument(
        "--config", "-c",
        help="Path to configuration file (JSON or YAML)",
    )
    run_parser.add_argument(
        "--profile", choices=["fast", "standard", "thorough"],
        default="standard",
        help="Named configuration profile (default: standard)",
    )
    run_parser.add_argument(
        "--context-column",
        help="Column identifying contexts (single-CSV mode)",
    )
    run_parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    run_parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Number of parallel workers (default: 1, -1 = all CPUs)",
    )
    run_parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if available",
    )
    run_parser.add_argument(
        "--checkpoint-dir",
        help="Directory for checkpoints",
    )
    run_parser.add_argument(
        "--ordered", action="store_true",
        help="Contexts are ordered (enables tipping-point detection)",
    )
    run_parser.add_argument(
        "--no-phase-2", action="store_true",
        help="Skip Phase 2 (Exploration)",
    )
    run_parser.add_argument(
        "--no-phase-3", action="store_true",
        help="Skip Phase 3 (Validation)",
    )

    # --- analyze ---
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze atlas results"
    )
    analyze_parser.add_argument(
        "atlas_dir",
        help="Path to atlas output directory",
    )
    analyze_parser.add_argument(
        "--summary", action="store_true",
        help="Print summary statistics",
    )
    analyze_parser.add_argument(
        "--descriptors", action="store_true",
        help="Print descriptor table",
    )
    analyze_parser.add_argument(
        "--filter-class",
        help="Filter variables by classification",
    )
    analyze_parser.add_argument(
        "--json", action="store_true",
        help="Output in JSON format",
    )

    # --- visualize ---
    viz_parser = subparsers.add_parser(
        "visualize", help="Generate visualizations"
    )
    viz_parser.add_argument(
        "atlas_dir",
        help="Path to atlas output directory",
    )
    viz_parser.add_argument(
        "--output", "-o",
        help="Output directory for plots",
    )
    viz_parser.add_argument(
        "--heatmap", action="store_true",
        help="Generate plasticity heatmap",
    )
    viz_parser.add_argument(
        "--classification", action="store_true",
        help="Generate classification distribution",
    )
    viz_parser.add_argument(
        "--embedding", action="store_true",
        help="Generate context embedding plot",
    )
    viz_parser.add_argument(
        "--dashboard", action="store_true",
        help="Generate summary dashboard",
    )
    viz_parser.add_argument(
        "--all", action="store_true",
        help="Generate all visualizations",
    )
    viz_parser.add_argument(
        "--format", choices=["png", "pdf", "svg"], default="png",
        help="Output format (default: png)",
    )

    # --- benchmark ---
    bench_parser = subparsers.add_parser(
        "benchmark", help="Run benchmarks"
    )
    bench_parser.add_argument(
        "--generator", "-g",
        choices=["fsvp", "csvm", "tps", "all"],
        default="all",
        help="Benchmark generator (default: all)",
    )
    bench_parser.add_argument(
        "--n-reps", type=int, default=5,
        help="Number of replications (default: 5)",
    )
    bench_parser.add_argument(
        "--output", "-o", default="benchmark_results",
        help="Output directory",
    )
    bench_parser.add_argument(
        "--p", type=int, default=5,
        help="Number of variables (default: 5)",
    )
    bench_parser.add_argument(
        "--K", type=int, default=3,
        help="Number of contexts (default: 3)",
    )
    bench_parser.add_argument(
        "--n", type=int, default=200,
        help="Samples per context (default: 200)",
    )
    bench_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    bench_parser.add_argument(
        "--profile", choices=["fast", "standard", "thorough"],
        default="fast",
        help="Pipeline profile for benchmarks (default: fast)",
    )

    return parser


# =====================================================================
# Command implementations
# =====================================================================


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute the 'run' command."""
    from cpa.pipeline.config import PipelineConfig
    from cpa.pipeline.orchestrator import CPAOrchestrator
    from cpa.io.readers import CSVReader, NumpyReader, ParquetReader

    # Build config
    if args.config:
        from cpa.pipeline.config import load_config
        config = load_config(args.config)
    else:
        profiles = {
            "fast": PipelineConfig.fast,
            "standard": PipelineConfig.standard,
            "thorough": PipelineConfig.thorough,
        }
        config = profiles[args.profile]()

    if args.seed is not None:
        config.computation.seed = args.seed
    config.computation.n_jobs = args.n_jobs
    config.computation.output_dir = args.output
    config.computation.progress = True

    if args.verbose:
        config.computation.log_level = "DEBUG"

    if args.checkpoint_dir:
        config.computation.checkpoint_dir = args.checkpoint_dir

    if args.ordered:
        config.detection.contexts_are_ordered = True

    if args.no_phase_2:
        config.run_phase_2 = False
    if args.no_phase_3:
        config.run_phase_3 = False

    config.validate_or_raise()

    # Load data
    data_path = Path(args.data)
    if data_path.suffix in (".npz", ".npy"):
        reader = NumpyReader(data_path)
    elif data_path.suffix in (".parquet", ".pq"):
        reader = ParquetReader(
            data_path, context_column=args.context_column,
        )
    else:
        reader = CSVReader(data_path, context_column=args.context_column)

    print(f"Loading data from {data_path}...")
    dataset = reader.read()
    print(f"  {dataset}")

    # Run pipeline
    orch = CPAOrchestrator(config)
    print(f"Running CPA pipeline ({args.profile} profile)...")

    t0 = time.time()
    atlas = orch.run(dataset, resume=args.resume)
    elapsed = time.time() - t0

    print(f"\nPipeline complete in {elapsed:.2f}s")
    print(f"  Contexts: {atlas.n_contexts}")
    print(f"  Variables: {atlas.n_variables}")

    summary = atlas.classification_summary()
    if summary:
        print("  Classifications:")
        for cls, count in sorted(summary.items()):
            print(f"    {cls}: {count}")

    print(f"\nResults saved to {args.output}/")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Execute the 'analyze' command."""
    from cpa.pipeline.results import AtlasResult, MechanismClass

    atlas = AtlasResult.load(args.atlas_dir)

    if args.summary or (not args.descriptors and not args.filter_class):
        stats = atlas.summary_statistics()
        if args.json:
            print(json.dumps(stats, indent=2, default=str))
        else:
            print("=== CPA Atlas Summary ===")
            print(f"Contexts: {stats.get('n_contexts', '?')}")
            print(f"Variables: {stats.get('n_variables', '?')}")
            print(f"Total time: {stats.get('total_time_seconds', 0):.2f}s")

            cls_summary = stats.get("classification_summary", {})
            if cls_summary:
                print("\nClassifications:")
                for cls, count in sorted(cls_summary.items()):
                    print(f"  {cls}: {count}")

            desc_stats = stats.get("descriptors", {})
            if desc_stats:
                print("\nDescriptor means:")
                for comp in ["structural", "parametric", "emergence", "sensitivity"]:
                    m = desc_stats.get(f"mean_{comp}", 0)
                    s = desc_stats.get(f"std_{comp}", 0)
                    print(f"  {comp}: {m:.4f} ± {s:.4f}")

    if args.descriptors and atlas.foundation:
        if args.json:
            descs = {
                v: d.to_dict()
                for v, d in atlas.foundation.descriptors.items()
            }
            print(json.dumps(descs, indent=2, default=str))
        else:
            print("\n=== Plasticity Descriptors ===")
            header = f"{'Variable':<12} {'Struct':>7} {'Param':>7} {'Emerg':>7} {'Sens':>7} {'Class'}"
            print(header)
            print("-" * len(header))
            for var in atlas.variable_names:
                dr = atlas.get_descriptor(var)
                if dr:
                    cls = dr.classification.value if hasattr(dr.classification, "value") else str(dr.classification)
                    print(
                        f"{var:<12} {dr.structural:>7.4f} "
                        f"{dr.parametric:>7.4f} {dr.emergence:>7.4f} "
                        f"{dr.sensitivity:>7.4f} {cls}"
                    )

    if args.filter_class and atlas.foundation:
        try:
            cls = MechanismClass(args.filter_class)
        except ValueError:
            print(f"Unknown class: {args.filter_class}", file=sys.stderr)
            return 1

        variables = atlas.variables_by_class(cls)
        if args.json:
            print(json.dumps(variables))
        else:
            print(f"\nVariables classified as '{args.filter_class}':")
            for v in variables:
                print(f"  {v}")
            print(f"  Total: {len(variables)}")

    return 0


def _cmd_visualize(args: argparse.Namespace) -> int:
    """Execute the 'visualize' command."""
    from cpa.pipeline.results import AtlasResult

    atlas = AtlasResult.load(args.atlas_dir)
    out_dir = Path(args.output or args.atlas_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = args.format
    gen_all = args.all

    try:
        from cpa.visualization.atlas_viz import AtlasVisualizer
        viz = AtlasVisualizer(save_format=fmt)
    except ImportError:
        print("matplotlib required: pip install matplotlib", file=sys.stderr)
        return 1

    if gen_all or args.heatmap:
        print("Generating plasticity heatmap...")
        viz.plasticity_heatmap(atlas, save_path=out_dir / f"heatmap.{fmt}")

    if gen_all or args.classification:
        print("Generating classification distribution...")
        viz.classification_distribution(
            atlas, save_path=out_dir / f"classification.{fmt}"
        )

    if gen_all or args.embedding:
        print("Generating context embedding...")
        viz.context_embedding(
            atlas, save_path=out_dir / f"embedding.{fmt}"
        )

    if gen_all or args.dashboard:
        print("Generating summary dashboard...")
        viz.summary_dashboard(
            atlas, save_path=out_dir / f"dashboard.{fmt}"
        )

    if gen_all:
        viz.alignment_cost_heatmap(
            atlas, save_path=out_dir / f"alignment_costs.{fmt}"
        )
        if atlas.exploration:
            viz.archive_coverage(
                atlas, save_path=out_dir / f"archive.{fmt}"
            )
            viz.convergence_plot(
                atlas, save_path=out_dir / f"convergence.{fmt}"
            )
        if atlas.validation:
            viz.certificate_dashboard(
                atlas, save_path=out_dir / f"certificates.{fmt}"
            )
            if atlas.validation.tipping_points:
                viz.tipping_point_timeline(
                    atlas, save_path=out_dir / f"tipping_points.{fmt}"
                )
            viz.sensitivity_plot(
                atlas, save_path=out_dir / f"sensitivity.{fmt}"
            )

    print(f"Visualizations saved to {out_dir}/")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Execute the 'benchmark' command."""
    from cpa.pipeline.config import PipelineConfig
    from cpa.pipeline.orchestrator import CPAOrchestrator, MultiContextDataset

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = {
        "fast": PipelineConfig.fast,
        "standard": PipelineConfig.standard,
        "thorough": PipelineConfig.thorough,
    }
    config = profiles[args.profile]()
    config.computation.seed = args.seed

    generators_to_run = []
    if args.generator == "all":
        generators_to_run = ["fsvp", "csvm", "tps"]
    else:
        generators_to_run = [args.generator]

    results: Dict[str, List[Dict[str, Any]]] = {}

    for gen_name in generators_to_run:
        print(f"\n=== Benchmark: {gen_name.upper()} ===")
        gen_results: List[Dict[str, Any]] = []

        for rep in range(args.n_reps):
            seed = args.seed + rep
            print(f"  Replication {rep + 1}/{args.n_reps} (seed={seed})...")

            dataset = _generate_benchmark_data(
                gen_name, p=args.p, K=args.K, n=args.n, seed=seed,
            )

            config_copy = config.copy()
            config_copy.computation.seed = seed
            config_copy.computation.progress = False
            config_copy.computation.log_level = "WARNING"

            orch = CPAOrchestrator(config_copy)

            t0 = time.time()
            try:
                atlas = orch.run(dataset)
                elapsed = time.time() - t0

                rep_result = {
                    "replication": rep,
                    "seed": seed,
                    "elapsed": elapsed,
                    "n_contexts": atlas.n_contexts,
                    "n_variables": atlas.n_variables,
                    "classification_summary": atlas.classification_summary(),
                }

                if atlas.validation:
                    rep_result["certification_rate"] = atlas.certification_rate()
                    rep_result["n_certified"] = atlas.validation.n_certified

                gen_results.append(rep_result)
                print(f"    Done in {elapsed:.2f}s")

            except Exception as e:
                print(f"    Failed: {e}")
                gen_results.append({
                    "replication": rep,
                    "seed": seed,
                    "error": str(e),
                })

        results[gen_name] = gen_results

    # Save results
    results_path = out_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nBenchmark results saved to {results_path}")

    # Print summary
    for gen_name, gen_results in results.items():
        successful = [r for r in gen_results if "elapsed" in r]
        if successful:
            times = [r["elapsed"] for r in successful]
            print(
                f"\n{gen_name.upper()}: "
                f"{len(successful)}/{len(gen_results)} succeeded, "
                f"mean time {np.mean(times):.2f}s ± {np.std(times):.2f}s"
            )

    return 0


def _generate_benchmark_data(
    generator: str,
    p: int = 5,
    K: int = 3,
    n: int = 200,
    seed: int = 42,
) -> "MultiContextDataset":
    """Generate benchmark data for a named generator."""
    from cpa.pipeline.orchestrator import MultiContextDataset

    try:
        from benchmarks.generators import (
            FSVPGenerator, CSVMGenerator, TPSGenerator,
        )
        gen_map = {
            "fsvp": FSVPGenerator,
            "csvm": CSVMGenerator,
            "tps": TPSGenerator,
        }
        gen_cls = gen_map.get(generator)
        if gen_cls:
            gen = gen_cls(p=p, K=K, n=n, seed=seed)
            result = gen.generate()
            if isinstance(result, MultiContextDataset):
                return result
            if isinstance(result, tuple):
                return result[0]
    except ImportError:
        pass

    # Fallback: simple synthetic data
    rng = np.random.RandomState(seed)
    context_data: Dict[str, np.ndarray] = {}

    for k in range(K):
        adj = np.zeros((p, p))
        for i in range(p - 1):
            if rng.random() < 0.5:
                adj[i, i + 1] = rng.uniform(0.3, 1.0)

        data = rng.randn(n, p)
        for i in range(p):
            parents = np.where(adj[:, i] != 0)[0]
            for pa in parents:
                data[:, i] += adj[pa, i] * data[:, pa]

        context_data[f"context_{k}"] = data

    return MultiContextDataset(
        context_data=context_data,
        variable_names=[f"X{i}" for i in range(p)],
    )


# =====================================================================
# Entry point
# =====================================================================


if __name__ == "__main__":
    sys.exit(main())
