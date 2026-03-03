#!/usr/bin/env python3
"""CPA Visualization Gallery — produce all available plots.

Generates each type of visualization supported by CPA, saving them
to a temporary directory.  Uses the FSVP benchmark generator for data.

Usage
-----
    python examples/visualization_gallery.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPA Visualization Gallery")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for saved plots (default: temp dir)",
    )
    return parser.parse_args()


def run_pipeline():
    """Run a small pipeline and return the AtlasResult."""
    from benchmarks.generators import FSVPGenerator
    from cpa.pipeline import CPAOrchestrator, PipelineConfig
    from cpa.pipeline.orchestrator import MultiContextDataset

    gen = FSVPGenerator(p=8, K=4, n=300, density=0.3,
                        plasticity_fraction=0.5, seed=42)
    bench = gen.generate()

    dataset = MultiContextDataset(
        context_data=bench.context_data,
        variable_names=bench.variable_names,
        context_ids=bench.context_ids,
    )

    config = PipelineConfig.fast()
    config.search.n_iterations = 10
    config.certificate.n_bootstrap = 30
    config.certificate.n_permutations = 30

    orch = CPAOrchestrator(config)
    atlas = orch.run(dataset)
    return atlas


def make_gallery(atlas, output_dir: Path) -> None:
    """Generate all visualizations and save to output_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cpa.visualization.atlas_viz import AtlasVisualizer
    from cpa.visualization.dag_viz import DAGVisualizer
    from cpa.visualization.descriptor_viz import DescriptorVisualizer

    atlas_viz = AtlasVisualizer()
    dag_viz = DAGVisualizer()
    desc_viz = DescriptorVisualizer()

    saved = []

    # ------------------------------------------------------------------
    # 1.  Plasticity heatmap
    # ------------------------------------------------------------------
    try:
        path = output_dir / "01_plasticity_heatmap.png"
        fig = atlas_viz.plasticity_heatmap(atlas, save_path=str(path))
        plt.close("all")
        saved.append(path)
        print(f"  [✓] Plasticity heatmap → {path.name}")
    except Exception as exc:
        print(f"  [✗] Plasticity heatmap: {exc}")

    # ------------------------------------------------------------------
    # 2.  Classification distribution
    # ------------------------------------------------------------------
    try:
        path = output_dir / "02_classification_distribution.png"
        fig = atlas_viz.classification_distribution(atlas, save_path=str(path))
        plt.close("all")
        saved.append(path)
        print(f"  [✓] Classification distribution → {path.name}")
    except Exception as exc:
        print(f"  [✗] Classification distribution: {exc}")

    # ------------------------------------------------------------------
    # 3.  DAG comparison
    # ------------------------------------------------------------------
    try:
        foundation = atlas.foundation
        if foundation and len(foundation.scm_results) >= 2:
            ctx_ids = sorted(foundation.scm_results.keys())
            adj_i = foundation.scm_results[ctx_ids[0]].adjacency
            adj_j = foundation.scm_results[ctx_ids[1]].adjacency
            var_names = foundation.variable_names

            path = output_dir / "03_dag_comparison.png"
            fig = dag_viz.draw_dag_diff(
                adj_i, adj_j, var_names,
                context_i=ctx_ids[0], context_j=ctx_ids[1],
                save_path=str(path),
            )
            plt.close("all")
            saved.append(path)
            print(f"  [✓] DAG comparison → {path.name}")
    except Exception as exc:
        print(f"  [✗] DAG comparison: {exc}")

    # ------------------------------------------------------------------
    # 4.  Single DAG visualisation
    # ------------------------------------------------------------------
    try:
        foundation = atlas.foundation
        if foundation and foundation.scm_results:
            ctx_id = sorted(foundation.scm_results.keys())[0]
            scm_res = foundation.scm_results[ctx_id]

            path = output_dir / "04_single_dag.png"
            fig = dag_viz.draw_dag(
                scm_res.adjacency,
                variable_names=scm_res.variable_names,
                title=f"Causal DAG — {ctx_id}",
                save_path=str(path),
            )
            plt.close("all")
            saved.append(path)
            print(f"  [✓] Single DAG → {path.name}")
    except Exception as exc:
        print(f"  [✗] Single DAG: {exc}")

    # ------------------------------------------------------------------
    # 5.  Archive coverage map
    # ------------------------------------------------------------------
    try:
        path = output_dir / "05_archive_coverage.png"
        fig = atlas_viz.archive_coverage(atlas, save_path=str(path))
        plt.close("all")
        saved.append(path)
        print(f"  [✓] Archive coverage → {path.name}")
    except Exception as exc:
        print(f"  [✗] Archive coverage: {exc}")

    # ------------------------------------------------------------------
    # 6.  Tipping-point timeline
    # ------------------------------------------------------------------
    try:
        path = output_dir / "06_tipping_timeline.png"
        fig = atlas_viz.tipping_point_timeline(atlas, save_path=str(path))
        plt.close("all")
        saved.append(path)
        print(f"  [✓] Tipping-point timeline → {path.name}")
    except Exception as exc:
        print(f"  [✗] Tipping-point timeline: {exc}")

    # ------------------------------------------------------------------
    # 7.  Convergence plot
    # ------------------------------------------------------------------
    try:
        path = output_dir / "07_convergence.png"
        fig = atlas_viz.convergence_plot(atlas, save_path=str(path))
        plt.close("all")
        saved.append(path)
        print(f"  [✓] Convergence plot → {path.name}")
    except Exception as exc:
        print(f"  [✗] Convergence plot: {exc}")

    # ------------------------------------------------------------------
    # 8.  Alignment cost heatmap
    # ------------------------------------------------------------------
    try:
        path = output_dir / "08_alignment_cost.png"
        fig = atlas_viz.alignment_cost_heatmap(atlas, save_path=str(path))
        plt.close("all")
        saved.append(path)
        print(f"  [✓] Alignment cost heatmap → {path.name}")
    except Exception as exc:
        print(f"  [✗] Alignment cost heatmap: {exc}")

    # ------------------------------------------------------------------
    # 9.  Descriptor 2D scatter
    # ------------------------------------------------------------------
    try:
        foundation = atlas.foundation
        if foundation and foundation.descriptors:
            path = output_dir / "09_descriptor_scatter.png"
            fig = desc_viz.scatter_2d(
                foundation.descriptors,
                save_path=str(path),
            )
            plt.close("all")
            saved.append(path)
            print(f"  [✓] Descriptor scatter → {path.name}")
    except Exception as exc:
        print(f"  [✗] Descriptor scatter: {exc}")

    # ------------------------------------------------------------------
    # 10.  Radar chart
    # ------------------------------------------------------------------
    try:
        foundation = atlas.foundation
        if foundation and foundation.descriptors:
            var_name = list(foundation.descriptors.keys())[0]
            path = output_dir / "10_radar_chart.png"
            fig = desc_viz.radar_chart(
                foundation.descriptors,
                variable=var_name,
                save_path=str(path),
            )
            plt.close("all")
            saved.append(path)
            print(f"  [✓] Radar chart ({var_name}) → {path.name}")
    except Exception as exc:
        print(f"  [✗] Radar chart: {exc}")

    # ------------------------------------------------------------------
    # 11.  Component distributions
    # ------------------------------------------------------------------
    try:
        foundation = atlas.foundation
        if foundation and foundation.descriptors:
            path = output_dir / "11_component_distributions.png"
            fig = desc_viz.component_distributions(
                foundation.descriptors,
                save_path=str(path),
            )
            plt.close("all")
            saved.append(path)
            print(f"  [✓] Component distributions → {path.name}")
    except Exception as exc:
        print(f"  [✗] Component distributions: {exc}")

    # ------------------------------------------------------------------
    # 12.  Summary dashboard
    # ------------------------------------------------------------------
    try:
        path = output_dir / "12_summary_dashboard.png"
        fig = atlas_viz.summary_dashboard(atlas, save_path=str(path))
        plt.close("all")
        saved.append(path)
        print(f"  [✓] Summary dashboard → {path.name}")
    except Exception as exc:
        print(f"  [✗] Summary dashboard: {exc}")

    # ------------------------------------------------------------------
    # 13.  Certificate dashboard
    # ------------------------------------------------------------------
    try:
        path = output_dir / "13_certificate_dashboard.png"
        fig = atlas_viz.certificate_dashboard(atlas, save_path=str(path))
        plt.close("all")
        saved.append(path)
        print(f"  [✓] Certificate dashboard → {path.name}")
    except Exception as exc:
        print(f"  [✗] Certificate dashboard: {exc}")

    # ------------------------------------------------------------------
    # 14.  Sensitivity plot
    # ------------------------------------------------------------------
    try:
        path = output_dir / "14_sensitivity.png"
        fig = atlas_viz.sensitivity_plot(atlas, save_path=str(path))
        plt.close("all")
        saved.append(path)
        print(f"  [✓] Sensitivity plot → {path.name}")
    except Exception as exc:
        print(f"  [✗] Sensitivity plot: {exc}")

    print(f"\n  Saved {len(saved)} / 14 plots to {output_dir}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        tempfile.mkdtemp(prefix="cpa_gallery_")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CPA Visualization Gallery")
    print("=" * 60)
    print(f"Output directory: {output_dir}\n")

    print("Running pipeline …")
    atlas = run_pipeline()
    print(f"Pipeline complete ({atlas.total_time:.2f}s)\n")

    print("Generating visualizations:")
    make_gallery(atlas, output_dir)

    print("\n" + "=" * 60)
    print("Gallery complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
