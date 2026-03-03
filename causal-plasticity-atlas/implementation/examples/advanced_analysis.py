#!/usr/bin/env python3
"""CPA Advanced Analysis — power-user features.

Demonstrates:
  1. Custom alignment anchors
  2. Robustness certificate inspection
  3. Sensitivity analysis
  4. QD search with custom parameters

Usage
-----
    python examples/advanced_analysis.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from benchmarks.generators import FSVPGenerator, TPSGenerator
from cpa.core.scm import StructuralCausalModel, random_dag
from cpa.alignment.cada import CADAAligner
from cpa.certificates.robustness import CertificateGenerator
from cpa.diagnostics.sensitivity import SensitivityAnalyzer, DescriptorSensitivity
from cpa.exploration.qd_search import QDSearchEngine
from cpa.pipeline import CPAOrchestrator, PipelineConfig
from cpa.pipeline.orchestrator import MultiContextDataset
from cpa.pipeline.results import MechanismClass


# =====================================================================
# 1.  Custom Alignment Anchors
# =====================================================================

def demo_custom_anchors() -> None:
    """Run CADA alignment with user-specified anchor variables."""
    print("\n" + "=" * 60)
    print("1. Custom Alignment Anchors")
    print("=" * 60)

    rng = np.random.default_rng(42)
    p, n = 6, 300
    var_names = [f"X{i}" for i in range(p)]

    # Two contexts with same structure but different weights
    adj = np.zeros((p, p))
    adj[0, 1] = 0.6
    adj[1, 2] = 0.4
    adj[2, 3] = 0.5
    adj[3, 4] = 0.3
    adj[4, 5] = 0.7

    data_a = rng.standard_normal((n, p))
    data_b = rng.standard_normal((n, p))

    # Simulate data from the DAG
    for j in range(p):
        parents = np.where(adj[:, j] != 0)[0]
        for pa in parents:
            data_a[:, j] += adj[pa, j] * data_a[:, pa]
            data_b[:, j] += adj[pa, j] * 1.5 * data_b[:, pa]

    aligner = CADAAligner()

    # Build SCM objects (align() requires SCM, not raw arrays)
    scm_a = StructuralCausalModel(adj, variable_names=var_names)
    scm_b = StructuralCausalModel(adj, variable_names=var_names)

    # Run without anchors
    result_no_anchor = aligner.align(scm_a, scm_b)
    print(f"\nWithout anchors:")
    print(f"  Alignment quality     : {result_no_anchor.alignment_quality:.4f}")
    print(f"  Structural divergence : {result_no_anchor.structural_divergence:.4f}")
    print(f"  Normalized divergence : {result_no_anchor.normalized_divergence:.4f}")

    # Run with anchors: force X0↔X0 and X5↔X5
    try:
        result_anchored = aligner.align(
            scm_a, scm_b,
            anchors={0: 0, 5: 5},
        )
        print(f"\nWith anchors (X0↔X0, X5↔X5):")
        print(f"  Alignment quality     : {result_anchored.alignment_quality:.4f}")
        print(f"  Structural divergence : {result_anchored.structural_divergence:.4f}")
        print(f"  Normalized divergence : {result_anchored.normalized_divergence:.4f}")
    except (TypeError, AttributeError) as exc:
        print(f"\n  Anchor API: {exc}")


# =====================================================================
# 2.  Robustness Certificate Inspection
# =====================================================================

def demo_certificate_inspection() -> None:
    """Generate and inspect robustness certificates in detail."""
    print("\n" + "=" * 60)
    print("2. Robustness Certificate Inspection")
    print("=" * 60)

    gen = FSVPGenerator(p=6, K=3, n=300, density=0.3,
                        plasticity_fraction=0.5, seed=42)
    bench = gen.generate()

    dataset = MultiContextDataset(
        context_data=bench.context_data,
        variable_names=bench.variable_names,
        context_ids=bench.context_ids,
    )

    cfg = PipelineConfig.fast()
    cfg.search.n_iterations = 5
    cfg.certificate.n_bootstrap = 50
    cfg.certificate.n_permutations = 50
    orch = CPAOrchestrator(cfg)
    atlas = orch.run(dataset)

    print(f"\nPipeline completed in {atlas.total_time:.2f}s")

    if atlas.validation and atlas.validation.certificates:
        certs = atlas.validation.certificates
        print(f"\nCertificates issued: {len(certs)}")

        for var, cert in certs.items():
            cls = atlas.get_classification(var)
            desc = atlas.get_descriptor(var)
            print(f"\n  Variable: {var}")
            print(f"    Classification : {cls.value if cls else 'N/A'}")
            if desc:
                print(f"    Descriptor     : struct={desc.structural:.3f} "
                      f"param={desc.parametric:.3f} "
                      f"emerg={desc.emergence:.3f}")
            print(f"    Certificate    : {cert}")
    else:
        print("\n  No certificates available (validation phase may not have run)")


# =====================================================================
# 3.  Sensitivity Analysis
# =====================================================================

def demo_sensitivity_analysis() -> None:
    """Run sensitivity analysis on a discovered DAG."""
    print("\n" + "=" * 60)
    print("3. Sensitivity Analysis")
    print("=" * 60)

    rng = np.random.default_rng(42)
    p, n = 5, 400
    var_names = [f"X{i}" for i in range(p)]

    adj = np.zeros((p, p))
    adj[0, 1] = 0.7
    adj[1, 2] = 0.5
    adj[0, 3] = 0.4
    adj[3, 4] = 0.6

    data = rng.standard_normal((n, p))
    order = list(range(p))
    for j in order:
        parents = np.where(adj[:, j] != 0)[0]
        for pa in parents:
            data[:, j] += adj[pa, j] * data[:, pa]

    analyzer = SensitivityAnalyzer(
        n_perturbations=20,
        noise_scale=0.1,
        seed=42,
    )

    try:
        report = analyzer.analyze(
            adj_matrix=adj,
            data=data,
            variable_names=var_names,
        )
        print(f"\nSensitivity report:")
        print(f"  Robustness score: "
              f"{getattr(report, 'robustness_score', 'N/A')}")

        if hasattr(report, "summary_table"):
            print(f"\n  Summary table:")
            for row in report.summary_table:
                print(f"    {row}")

        if hasattr(report, "recommendations"):
            print(f"\n  Recommendations:")
            for rec in report.recommendations:
                print(f"    • {rec}")
    except (TypeError, AttributeError) as exc:
        print(f"\n  Sensitivity analysis: {exc}")

    # Descriptor sensitivity
    desc_sens = DescriptorSensitivity(epsilon=0.01)
    try:
        sens_matrix = desc_sens.compute_sensitivity_matrix(adj)
        print(f"\n  Descriptor sensitivity matrix shape: {sens_matrix.shape}")

        critical = desc_sens.identify_critical_edges(adj, threshold=0.3)
        print(f"  Critical edges (threshold=0.3): {critical}")
    except (TypeError, AttributeError) as exc:
        print(f"\n  Descriptor sensitivity: {exc}")


# =====================================================================
# 4.  QD Search with Custom Parameters
# =====================================================================

def demo_custom_qd_search() -> None:
    """Run QD search with customised parameters."""
    print("\n" + "=" * 60)
    print("4. QD Search with Custom Parameters")
    print("=" * 60)

    gen = FSVPGenerator(p=6, K=3, n=200, density=0.3,
                        plasticity_fraction=0.5, seed=42)
    bench = gen.generate()

    dataset = MultiContextDataset(
        context_data=bench.context_data,
        variable_names=bench.variable_names,
        context_ids=bench.context_ids,
    )

    # Custom search configuration
    cfg = PipelineConfig.fast()
    cfg.search.n_iterations = 20
    cfg.search.population_size = 30
    cfg.run_phase_3 = False  # skip validation for speed

    print(f"\nQD Search config:")
    print(f"  Iterations      : {cfg.search.n_iterations}")
    print(f"  Population size : {cfg.search.population_size}")

    orch = CPAOrchestrator(cfg)
    atlas = orch.run(dataset)

    if atlas.exploration is not None:
        expl = atlas.exploration
        print(f"\nQD Search results:")
        print(f"  Archive size       : {len(expl.archive)}")
        print(f"  Iterations run     : {expl.n_iterations}")
        print(f"  Best fitness       : {expl.best_fitness:.4f}")
        print(f"  Coverage           : {expl.coverage:.4f}")
        print(f"  QD-score           : {expl.qd_score:.4f}")

        if expl.convergence_history:
            print(f"  Convergence (last 5):")
            for val in expl.convergence_history[-5:]:
                print(f"    {val:.4f}")

        if expl.patterns:
            print(f"\n  Discovered patterns ({len(expl.patterns)}):")
            for i, pat in enumerate(expl.patterns[:5]):
                print(f"    Pattern {i}: {pat}")
    else:
        print("\n  Exploration phase did not run")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    print("=" * 60)
    print("CPA Advanced Analysis Example")
    print("=" * 60)

    demo_custom_anchors()
    demo_certificate_inspection()
    demo_sensitivity_analysis()
    demo_custom_qd_search()

    print("\n" + "=" * 60)
    print("Advanced analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
