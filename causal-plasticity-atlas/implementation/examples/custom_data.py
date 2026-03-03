#!/usr/bin/env python3
"""CPA Custom Data Example — bring your own data.

Demonstrates how to:
  1. Construct SCMs manually from known adjacency matrices.
  2. Build a MultiContextCausalModel (MCCM) from user data.
  3. Run the CPA analysis with a custom configuration.
  4. Interpret and query the results.

Usage
-----
    python examples/custom_data.py
"""

from __future__ import annotations

import numpy as np

from cpa.core.scm import StructuralCausalModel
from cpa.core.mccm import build_mccm_from_data, build_mccm_from_scms
from cpa.pipeline import CPAOrchestrator, PipelineConfig
from cpa.pipeline.orchestrator import MultiContextDataset
from cpa.pipeline.results import MechanismClass


def generate_custom_data() -> dict:
    """Simulate two contexts with known causal structure.

    Context A:  X0 → X1 → X2 → X3   (chain with weights [0.8, 0.6, 0.4])
    Context B:  Same topology but different weights  [0.3, 0.9, 0.7]
    """
    rng = np.random.default_rng(123)
    p, n = 4, 500
    var_names = ["Temperature", "Pressure", "Flow", "Output"]

    # ----- Context A -----
    adj_a = np.zeros((p, p))
    adj_a[0, 1] = 0.8
    adj_a[1, 2] = 0.6
    adj_a[2, 3] = 0.4

    noise_a = rng.standard_normal((n, p))
    data_a = np.zeros((n, p))
    data_a[:, 0] = noise_a[:, 0]
    data_a[:, 1] = adj_a[0, 1] * data_a[:, 0] + noise_a[:, 1]
    data_a[:, 2] = adj_a[1, 2] * data_a[:, 1] + noise_a[:, 2]
    data_a[:, 3] = adj_a[2, 3] * data_a[:, 2] + noise_a[:, 3]

    # ----- Context B — same structure, different parameters -----
    adj_b = np.zeros((p, p))
    adj_b[0, 1] = 0.3
    adj_b[1, 2] = 0.9
    adj_b[2, 3] = 0.7

    noise_b = rng.standard_normal((n, p))
    data_b = np.zeros((n, p))
    data_b[:, 0] = noise_b[:, 0]
    data_b[:, 1] = adj_b[0, 1] * data_b[:, 0] + noise_b[:, 1]
    data_b[:, 2] = adj_b[1, 2] * data_b[:, 1] + noise_b[:, 2]
    data_b[:, 3] = adj_b[2, 3] * data_b[:, 2] + noise_b[:, 3]

    return {
        "var_names": var_names,
        "adj_a": adj_a,
        "adj_b": adj_b,
        "data_a": data_a,
        "data_b": data_b,
    }


def demo_manual_scm(custom: dict) -> None:
    """Show how to create StructuralCausalModel objects directly."""
    print("\n--- 1. Manual SCM Construction ---")

    scm_a = StructuralCausalModel(
        custom["adj_a"],
        variable_names=custom["var_names"],
    )
    scm_b = StructuralCausalModel(
        custom["adj_b"],
        variable_names=custom["var_names"],
    )
    print(f"SCM A: {scm_a.adjacency_matrix.shape}, "
          f"edges={int(np.sum(scm_a.adjacency_matrix != 0))}")
    print(f"SCM B: {scm_b.adjacency_matrix.shape}, "
          f"edges={int(np.sum(scm_b.adjacency_matrix != 0))}")

    # Build MCCM from SCM objects
    mccm = build_mccm_from_scms(
        {"context_A": scm_a, "context_B": scm_b},
    )
    print(f"MCCM contexts: {list(mccm.context_ids) if hasattr(mccm, 'context_ids') else 'N/A'}")


def demo_mccm_from_data(custom: dict) -> None:
    """Show how to build an MCCM from raw data + adjacency matrices."""
    print("\n--- 2. MCCM from Data ---")

    datasets = {
        "context_A": custom["data_a"],
        "context_B": custom["data_b"],
    }
    adjacencies = {
        "context_A": custom["adj_a"],
        "context_B": custom["adj_b"],
    }

    mccm = build_mccm_from_data(
        datasets=datasets,
        adjacency_matrices=adjacencies,
        variable_names=custom["var_names"],
    )
    print(f"MCCM built with {len(datasets)} contexts, "
          f"{len(custom['var_names'])} variables")


def demo_pipeline(custom: dict) -> None:
    """Run the full CPA pipeline with custom configuration."""
    print("\n--- 3. Run CPA Pipeline ---")

    dataset = MultiContextDataset(
        context_data={
            "context_A": custom["data_a"],
            "context_B": custom["data_b"],
        },
        variable_names=custom["var_names"],
    )

    # Create a custom configuration
    config = PipelineConfig.fast()
    config.discovery.alpha = 0.01
    config.search.n_iterations = 5
    config.certificate.n_bootstrap = 30
    config.certificate.n_permutations = 30

    print(f"Config profile: {config.profile.value}")
    print(f"Discovery alpha: {config.discovery.alpha}")

    orch = CPAOrchestrator(config)
    atlas = orch.run(dataset)

    print(f"\nPipeline completed in {atlas.total_time:.2f}s")
    print(f"  Contexts  : {atlas.n_contexts}")
    print(f"  Variables : {atlas.n_variables}")

    # ------------------------------------------------------------------
    # 4.  Interpret results
    # ------------------------------------------------------------------
    print("\n--- 4. Interpret Results ---")

    summary = atlas.classification_summary()
    print(f"\nClassification summary:")
    for cls_name, count in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"  {cls_name:30s}  {count}")

    print(f"\nDetailed descriptors:")
    for var in atlas.variable_names:
        desc = atlas.get_descriptor(var)
        cls = atlas.get_classification(var)
        if desc is not None:
            print(f"  {var:>15s}  struct={desc.structural:.3f}  "
                  f"param={desc.parametric:.3f}  "
                  f"class={cls.value}")

    # Query specific mechanism types
    invariant = atlas.variables_by_class(MechanismClass.INVARIANT)
    parametric = atlas.variables_by_class(MechanismClass.PARAMETRICALLY_PLASTIC)
    print(f"\nInvariant variables   : {invariant}")
    print(f"Parametric-plastic    : {parametric}")

    # Filter by descriptor criteria
    sensitive = atlas.filter_variables(min_parametric=0.3)
    print(f"Variables with param >= 0.3: {sensitive}")

    # Alignment information
    for (ci, cj), aln in atlas.foundation.alignment_results.items():
        print(f"\nAlignment ({ci} ↔ {cj}):")
        print(f"  Structural cost : {aln.structural_cost:.4f}")
        print(f"  Parametric cost : {aln.parametric_cost:.4f}")
        print(f"  Total cost      : {aln.total_cost:.4f}")

    # Certificates
    if atlas.validation and atlas.validation.certificates:
        print(f"\nRobustness certificates:")
        for var, cert in atlas.validation.certificates.items():
            print(f"  {var}: {cert}")


def main() -> None:
    print("=" * 60)
    print("CPA Custom Data Example")
    print("=" * 60)

    custom = generate_custom_data()
    demo_manual_scm(custom)
    demo_mccm_from_data(custom)
    demo_pipeline(custom)

    print("\n" + "=" * 60)
    print("Custom data example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
