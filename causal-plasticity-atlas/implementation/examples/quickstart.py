#!/usr/bin/env python3
"""CPA Quick Start — minimal working example.

Generate synthetic data with the FSVP generator, run the full
Causal-Plasticity Atlas pipeline, print classification results,
and save the atlas to disk.

Usage
-----
    python examples/quickstart.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    # ------------------------------------------------------------------
    # 1.  Generate synthetic data (Generator 1 — FSVP)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("CPA Quick Start Example")
    print("=" * 60)

    from benchmarks.generators import FSVPGenerator

    gen = FSVPGenerator(
        p=8,         # 8 variables
        K=4,         # 4 observational contexts
        n=300,       # 300 samples per context
        density=0.3,
        plasticity_fraction=0.5,  # half of mechanisms are plastic
        seed=42,
    )
    bench = gen.generate()
    print(f"\nGenerated FSVP scenario:")
    print(f"  Variables      : {bench.variable_names}")
    print(f"  Contexts       : {bench.context_ids}")
    print(f"  Samples/context: {bench.context_data[bench.context_ids[0]].shape[0]}")
    print(f"  Ground-truth invariant : {bench.ground_truth.invariant_variables}")
    print(f"  Ground-truth plastic   : {bench.ground_truth.plastic_variables}")

    # ------------------------------------------------------------------
    # 2.  Configure the CPA pipeline
    # ------------------------------------------------------------------
    from cpa.pipeline import CPAOrchestrator, PipelineConfig
    from cpa.pipeline.orchestrator import MultiContextDataset

    dataset = MultiContextDataset(
        context_data=bench.context_data,
        variable_names=bench.variable_names,
        context_ids=bench.context_ids,
    )

    # Use "fast" profile for a quick demonstration
    config = PipelineConfig.fast()
    config.search.n_iterations = 10
    config.certificate.n_bootstrap = 50
    config.certificate.n_permutations = 50
    print(f"\nPipeline config profile: {config.profile.value}")

    # ------------------------------------------------------------------
    # 3.  Run the pipeline
    # ------------------------------------------------------------------
    print("\nRunning CPA pipeline …")
    orch = CPAOrchestrator(config)
    atlas = orch.run(dataset)
    print(f"Pipeline complete in {atlas.total_time:.2f}s")

    # ------------------------------------------------------------------
    # 4.  Inspect results
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)

    # 4a.  Classification summary
    summary = atlas.classification_summary()
    print(f"\nClassification summary (count per category):")
    for cls_name, count in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"  {cls_name:30s}  {count}")

    # 4b.  Per-variable descriptors
    print(f"\nPer-variable plasticity descriptors:")
    print(f"  {'Variable':>10s}  {'Struct':>7s}  {'Param':>7s}  "
          f"{'Emerg':>7s}  {'Sens':>7s}  {'Norm':>7s}  Classification")
    for var in atlas.variable_names:
        desc = atlas.get_descriptor(var)
        cls = atlas.get_classification(var)
        if desc is not None:
            print(f"  {var:>10s}  {desc.structural:7.3f}  {desc.parametric:7.3f}  "
                  f"{desc.emergence:7.3f}  {desc.sensitivity:7.3f}  "
                  f"{desc.norm:7.3f}  {cls.value}")

    # 4c.  Most similar / different context pairs
    print(f"\nMost similar context pairs:")
    for ci, cj, cost in atlas.most_similar_contexts(n=3):
        print(f"  ({ci}, {cj})  cost={cost:.4f}")

    print(f"\nMost different context pairs:")
    for ci, cj, cost in atlas.most_different_contexts(n=3):
        print(f"  ({ci}, {cj})  cost={cost:.4f}")

    # 4d.  Exploration phase summary
    if atlas.exploration is not None:
        expl = atlas.exploration
        print(f"\nQD Search:")
        print(f"  Iterations    : {expl.n_iterations}")
        print(f"  Best fitness  : {expl.best_fitness:.4f}")
        print(f"  Coverage      : {expl.coverage:.4f}")
        print(f"  QD-score      : {expl.qd_score:.4f}")
        print(f"  Archive size  : {len(expl.archive)}")

    # 4e.  Validation phase summary
    if atlas.validation is not None:
        val = atlas.validation
        print(f"\nValidation:")
        print(f"  Certificates issued : {len(val.certificates)}")
        if val.tipping_points is not None:
            tp = val.tipping_points
            n_tp = 0
            if hasattr(tp, "tipping_points"):
                n_tp = len(tp.tipping_points) if tp.tipping_points else 0
            print(f"  Tipping points      : {n_tp}")

    # ------------------------------------------------------------------
    # 5.  Save atlas to disk
    # ------------------------------------------------------------------
    save_dir = Path(tempfile.mkdtemp(prefix="cpa_quickstart_"))
    atlas_path = save_dir / "atlas.json"

    from cpa.io.serialization import save_atlas

    saved = save_atlas(atlas, atlas_path)
    print(f"\nAtlas saved to: {saved}")

    # Verify round-trip
    from cpa.io.serialization import load_atlas

    loaded = load_atlas(saved)
    print(f"Atlas loaded back successfully ({type(loaded).__name__})")

    # ------------------------------------------------------------------
    # 6.  Optional: generate a quick heatmap
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")

        from cpa.visualization.atlas_viz import AtlasVisualizer

        viz = AtlasVisualizer()
        fig = viz.plasticity_heatmap(atlas, save_path=str(save_dir / "heatmap.png"))
        print(f"Heatmap saved to: {save_dir / 'heatmap.png'}")

        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception as exc:
        print(f"(Visualization skipped: {exc})")

    print("\n" + "=" * 60)
    print("Quick start complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
