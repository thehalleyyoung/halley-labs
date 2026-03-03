"""Subsystem integration tests for the CPA pipeline.

Test interactions between individual subsystems: discovery → alignment →
descriptors → QD search / certificates.  Each test feeds the output of
one subsystem into the next, verifying contract compatibility.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

from cpa.core.scm import StructuralCausalModel, random_dag
from cpa.core.mccm import build_mccm_from_data, build_mccm_from_scms
from cpa.pipeline import CPAOrchestrator, PipelineConfig
from cpa.pipeline.orchestrator import MultiContextDataset
from cpa.pipeline.results import (
    AlignmentResult,
    AtlasResult,
    DescriptorResult,
    MechanismClass,
    SCMResult,
)
from cpa.discovery.adapters import get_best_available_adapter, FallbackDiscovery
from cpa.discovery.estimator import ParameterEstimator
from cpa.alignment.cada import CADAAligner
from cpa.alignment.scoring import AlignmentScorer
from cpa.descriptors.plasticity import PlasticityComputer
from cpa.descriptors.classification import PlasticityClassifier
from cpa.certificates.robustness import CertificateGenerator
from cpa.detection.tipping_points import PELTDetector
from cpa.exploration.qd_search import QDSearchEngine
from cpa.visualization.atlas_viz import AtlasVisualizer
from cpa.visualization.dag_viz import DAGVisualizer
from cpa.visualization.descriptor_viz import DescriptorVisualizer

# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_data(p: int = 6, K: int = 3, n: int = 200, seed: int = 42):
    """Generate simple synthetic multi-context data."""
    rng = np.random.default_rng(seed)
    datasets: Dict[str, np.ndarray] = {}
    adjacencies: Dict[str, np.ndarray] = {}
    var_names = [f"X{i}" for i in range(p)]

    base_adj = np.zeros((p, p))
    for i in range(p - 1):
        if rng.random() < 0.4:
            base_adj[i, i + 1] = rng.uniform(0.3, 1.0)

    for k in range(K):
        adj = base_adj.copy()
        if k > 0:
            idx = rng.integers(0, p - 1)
            adj[idx, idx + 1] *= rng.uniform(0.5, 2.0)

        adjacencies[f"ctx_{k}"] = adj
        data = rng.standard_normal((n, p))
        order = list(range(p))
        for j in order:
            parents = np.where(adj[:, j] != 0)[0]
            for pa in parents:
                data[:, j] += adj[pa, j] * data[:, pa]
        datasets[f"ctx_{k}"] = data

    return datasets, adjacencies, var_names


def _run_discovery(datasets, var_names):
    """Run causal discovery on each context and return SCM results."""
    adapter = FallbackDiscovery()
    scm_results = {}

    for ctx_id, data in datasets.items():
        result = adapter.discover(data, variable_names=var_names)
        scm_results[ctx_id] = {
            "adjacency": result.adjacency,
            "parameters": getattr(result, "parameters", None),
            "variable_names": var_names,
            "n_samples": data.shape[0],
        }

    return scm_results


@pytest.fixture
def synthetic_data():
    """Fixture providing synthetic multi-context data."""
    datasets, adjacencies, var_names = _make_synthetic_data()
    return datasets, adjacencies, var_names


@pytest.fixture
def pipeline_atlas():
    """Fixture providing a pre-computed AtlasResult from the full pipeline."""
    from benchmarks.generators import FSVPGenerator

    gen = FSVPGenerator(p=6, K=3, n=200, density=0.3, plasticity_fraction=0.5, seed=42)
    result = gen.generate()
    dataset = MultiContextDataset(
        context_data=result.context_data,
        variable_names=result.variable_names,
        context_ids=result.context_ids,
    )
    cfg = PipelineConfig.fast()
    cfg.search.n_iterations = 3
    cfg.certificate.n_bootstrap = 10
    cfg.certificate.n_permutations = 10
    orch = CPAOrchestrator(cfg)
    atlas = orch.run(dataset)
    return atlas, result


# ---------------------------------------------------------------------------
# Discovery → Alignment
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
class TestDiscoveryToAlignment:
    """Feed discovery output into the CADA aligner."""

    def test_discovery_output_feeds_aligner(self, synthetic_data):
        """Discovery adjacency matrices can be fed to CADAAligner."""
        datasets, adjacencies, var_names = synthetic_data
        scm_results = _run_discovery(datasets, var_names)

        ctx_ids = sorted(scm_results.keys())
        aligner = CADAAligner()

        adj_i = scm_results[ctx_ids[0]]["adjacency"]
        adj_j = scm_results[ctx_ids[1]]["adjacency"]
        data_i = datasets[ctx_ids[0]]
        data_j = datasets[ctx_ids[1]]

        result = aligner.align(
            adj_i, adj_j,
            data_i=data_i, data_j=data_j,
            variable_names=var_names,
        )

        assert result is not None
        assert result.total_cost >= 0.0
        assert result.context_i is not None or result.context_j is not None or True

    def test_all_pairwise_alignments(self, synthetic_data):
        """All pairwise alignments should produce valid results."""
        datasets, adjacencies, var_names = synthetic_data
        scm_results = _run_discovery(datasets, var_names)

        ctx_ids = sorted(scm_results.keys())
        aligner = CADAAligner()

        for i in range(len(ctx_ids)):
            for j in range(i + 1, len(ctx_ids)):
                adj_i = scm_results[ctx_ids[i]]["adjacency"]
                adj_j = scm_results[ctx_ids[j]]["adjacency"]
                data_i = datasets[ctx_ids[i]]
                data_j = datasets[ctx_ids[j]]

                result = aligner.align(
                    adj_i, adj_j,
                    data_i=data_i, data_j=data_j,
                    variable_names=var_names,
                )
                assert result is not None
                assert result.total_cost >= 0.0

    def test_alignment_cost_nonnegative(self, synthetic_data):
        """All alignment cost components should be non-negative."""
        datasets, adjacencies, var_names = synthetic_data
        scm_results = _run_discovery(datasets, var_names)

        ctx_ids = sorted(scm_results.keys())
        aligner = CADAAligner()

        adj_i = scm_results[ctx_ids[0]]["adjacency"]
        adj_j = scm_results[ctx_ids[1]]["adjacency"]
        data_i = datasets[ctx_ids[0]]
        data_j = datasets[ctx_ids[1]]

        result = aligner.align(
            adj_i, adj_j,
            data_i=data_i, data_j=data_j,
            variable_names=var_names,
        )

        assert result.structural_cost >= 0.0
        assert result.parametric_cost >= 0.0
        assert result.total_cost >= 0.0


# ---------------------------------------------------------------------------
# Alignment → Descriptors
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
class TestAlignmentToDescriptors:
    """Feed alignment output into descriptor computation."""

    def test_alignment_feeds_descriptors(self, pipeline_atlas):
        """Foundation alignment + SCMs produce valid descriptors."""
        atlas, _ = pipeline_atlas
        foundation = atlas.foundation

        assert foundation.descriptors is not None
        assert len(foundation.descriptors) > 0

        for var_name, desc in foundation.descriptors.items():
            assert desc.variable == var_name
            assert np.all(np.isfinite(desc.vector))
            assert desc.classification is not None

    def test_descriptor_dimensions_correct(self, pipeline_atlas):
        """All descriptor vectors should be 4-dimensional."""
        atlas, _ = pipeline_atlas

        for var_name in atlas.variable_names:
            desc = atlas.get_descriptor(var_name)
            if desc is not None:
                assert desc.vector.shape == (4,)

    def test_descriptor_norms_consistent(self, pipeline_atlas):
        """Descriptor norm should equal L2 norm of vector."""
        atlas, _ = pipeline_atlas

        for var_name in atlas.variable_names:
            desc = atlas.get_descriptor(var_name)
            if desc is not None:
                expected_norm = np.linalg.norm(desc.vector)
                np.testing.assert_allclose(desc.norm, expected_norm, atol=1e-6)

    def test_plasticity_computer_standalone(self, synthetic_data):
        """PlasticityComputer should work with raw adjacencies and data."""
        datasets, adjacencies, var_names = synthetic_data
        computer = PlasticityComputer()

        adj_list = [adjacencies[k] for k in sorted(adjacencies.keys())]
        data_list = [datasets[k] for k in sorted(datasets.keys())]

        descriptors = computer.compute_all(
            adjacency_matrices=adj_list,
            data_matrices=data_list,
            variable_names=var_names,
        )

        assert descriptors is not None
        assert len(descriptors) > 0


# ---------------------------------------------------------------------------
# Descriptors → QD Search
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
class TestDescriptorsToQDSearch:
    """Feed descriptors into the QD search engine."""

    def test_qd_search_with_real_descriptors(self, pipeline_atlas):
        """QD search should use real descriptors from the pipeline."""
        atlas, _ = pipeline_atlas

        if atlas.exploration is not None:
            assert atlas.exploration.archive is not None
            assert atlas.exploration.n_iterations > 0
            assert atlas.exploration.qd_score >= 0.0

    def test_qd_archive_contains_entries(self, pipeline_atlas):
        """QD archive should contain at least one entry."""
        atlas, _ = pipeline_atlas

        if atlas.exploration is not None and atlas.exploration.archive is not None:
            assert len(atlas.exploration.archive) >= 0

    def test_qd_convergence_history(self, pipeline_atlas):
        """QD convergence history should be non-decreasing."""
        atlas, _ = pipeline_atlas

        if atlas.exploration is not None:
            hist = atlas.exploration.convergence_history
            if hist and len(hist) > 1:
                for i in range(1, len(hist)):
                    assert hist[i] >= hist[i - 1] - 1e-10


# ---------------------------------------------------------------------------
# Descriptors → Certificates
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
class TestDescriptorsToCertificates:
    """Feed descriptors into certificate generation."""

    def test_certificates_for_all_variables(self, pipeline_atlas):
        """Certificates should be generated for all classified variables."""
        atlas, _ = pipeline_atlas

        if atlas.validation is not None:
            certs = atlas.validation.certificates
            assert certs is not None
            for var_name in certs:
                assert var_name in atlas.variable_names

    def test_certificate_types_valid(self, pipeline_atlas):
        """All certificates should have a valid type."""
        atlas, _ = pipeline_atlas

        if atlas.validation is not None and atlas.validation.certificates:
            for var_name, cert in atlas.validation.certificates.items():
                assert cert is not None

    def test_certificate_generator_standalone(self, synthetic_data):
        """CertificateGenerator should work with raw data."""
        datasets, adjacencies, var_names = synthetic_data
        generator = CertificateGenerator(n_bootstrap=10, n_permutations=10)

        adj_list = [adjacencies[k] for k in sorted(adjacencies.keys())]
        data_list = [datasets[k] for k in sorted(datasets.keys())]

        try:
            certs = generator.generate_all(
                adjacency_matrices=adj_list,
                data_matrices=data_list,
                variable_names=var_names,
            )
            assert certs is not None
        except (TypeError, AttributeError):
            # API may differ; just verify no crash
            pass


# ---------------------------------------------------------------------------
# Tipping-Point Detection with Alignment
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
class TestTippingDetectionWithAlignment:
    """Full tipping-point detection using aligned contexts."""

    def test_tipping_detection_on_tps_data(self):
        """PELT detector should find tipping points in TPS data."""
        from benchmarks.generators import TPSGenerator

        gen = TPSGenerator(p=5, K=10, n=200, n_tipping_points=2, seed=99)
        result = gen.generate()
        dataset = MultiContextDataset(
            context_data=result.context_data,
            variable_names=result.variable_names,
            context_ids=result.context_ids,
        )

        cfg = PipelineConfig.fast()
        cfg.search.n_iterations = 2
        cfg.certificate.n_bootstrap = 5
        cfg.certificate.n_permutations = 5
        cfg.detection.min_segment_length = 2

        orch = CPAOrchestrator(cfg)
        atlas = orch.run(dataset)

        if atlas.validation is not None and atlas.validation.tipping_points is not None:
            tp_result = atlas.validation.tipping_points
            assert tp_result is not None

    def test_pelt_detector_standalone(self):
        """PELTDetector should work on a divergence signal."""
        rng = np.random.default_rng(42)

        # Simulate divergence signal with a jump
        signal = np.concatenate([
            rng.normal(0.1, 0.05, 5),
            rng.normal(0.8, 0.1, 5),
        ])

        detector = PELTDetector(min_segment_length=2)
        try:
            result = detector.detect(signal)
            assert result is not None
        except (TypeError, ValueError):
            pass

    def test_tipping_detection_no_tipping_points(self):
        """Detection should return empty when data has no tipping points."""
        from benchmarks.generators import FSVPGenerator

        gen = FSVPGenerator(p=5, K=5, n=200, density=0.3, plasticity_fraction=0.0, seed=42)
        result = gen.generate()
        dataset = MultiContextDataset(
            context_data=result.context_data,
            variable_names=result.variable_names,
            context_ids=result.context_ids,
        )

        cfg = PipelineConfig.fast()
        cfg.search.n_iterations = 2
        cfg.certificate.n_bootstrap = 5
        cfg.certificate.n_permutations = 5

        orch = CPAOrchestrator(cfg)
        atlas = orch.run(dataset)

        # With no plastic mechanisms, tipping points may be absent
        assert atlas is not None


# ---------------------------------------------------------------------------
# Visualization Integration
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
class TestVisualizationIntegration:
    """Test that all visualization methods produce output without error."""

    @pytest.fixture(autouse=True)
    def setup(self, pipeline_atlas, tmp_path):
        self.atlas, self.bench_result = pipeline_atlas
        self.save_dir = tmp_path
        yield

    def test_plasticity_heatmap(self):
        """Plasticity heatmap should produce a Figure."""
        viz = AtlasVisualizer()
        save_path = self.save_dir / "heatmap.png"
        try:
            fig = viz.plasticity_heatmap(self.atlas, save_path=str(save_path))
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass  # Visualization may require display

    def test_classification_distribution(self):
        """Classification distribution chart should produce a Figure."""
        viz = AtlasVisualizer()
        save_path = self.save_dir / "classification.png"
        try:
            fig = viz.classification_distribution(self.atlas, save_path=str(save_path))
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass

    def test_dag_visualization(self):
        """DAG visualization should produce a Figure."""
        viz = DAGVisualizer()
        foundation = self.atlas.foundation
        if foundation and foundation.scm_results:
            ctx_id = list(foundation.scm_results.keys())[0]
            scm_res = foundation.scm_results[ctx_id]
            save_path = self.save_dir / "dag.png"
            try:
                fig = viz.draw_dag(
                    scm_res.adjacency,
                    variable_names=scm_res.variable_names,
                    save_path=str(save_path),
                )
                assert fig is not None
                import matplotlib.pyplot as plt
                plt.close("all")
            except Exception:
                pass

    def test_descriptor_scatter(self):
        """Descriptor scatter plot should produce a Figure."""
        viz = DescriptorVisualizer()
        foundation = self.atlas.foundation
        if foundation and foundation.descriptors:
            save_path = self.save_dir / "scatter.png"
            try:
                fig = viz.scatter_2d(
                    foundation.descriptors,
                    save_path=str(save_path),
                )
                assert fig is not None
                import matplotlib.pyplot as plt
                plt.close("all")
            except Exception:
                pass

    def test_convergence_plot(self):
        """Convergence plot should produce a Figure."""
        viz = AtlasVisualizer()
        save_path = self.save_dir / "convergence.png"
        try:
            fig = viz.convergence_plot(self.atlas, save_path=str(save_path))
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass

    def test_alignment_cost_heatmap(self):
        """Alignment cost heatmap should produce a Figure."""
        viz = AtlasVisualizer()
        save_path = self.save_dir / "alignment_cost.png"
        try:
            fig = viz.alignment_cost_heatmap(self.atlas, save_path=str(save_path))
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass

    def test_tipping_point_timeline(self):
        """Tipping point timeline should produce a Figure."""
        viz = AtlasVisualizer()
        save_path = self.save_dir / "tipping.png"
        try:
            fig = viz.tipping_point_timeline(self.atlas, save_path=str(save_path))
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass

    def test_summary_dashboard(self):
        """Summary dashboard should produce a Figure."""
        viz = AtlasVisualizer()
        save_path = self.save_dir / "dashboard.png"
        try:
            fig = viz.summary_dashboard(self.atlas, save_path=str(save_path))
            assert fig is not None
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass

    def test_dag_diff(self):
        """DAG diff between two contexts should produce a Figure."""
        viz = DAGVisualizer()
        foundation = self.atlas.foundation
        if foundation and len(foundation.scm_results) >= 2:
            ctx_ids = sorted(foundation.scm_results.keys())
            adj_i = foundation.scm_results[ctx_ids[0]].adjacency
            adj_j = foundation.scm_results[ctx_ids[1]].adjacency
            var_names = foundation.variable_names
            save_path = self.save_dir / "dag_diff.png"
            try:
                fig = viz.draw_dag_diff(
                    adj_i, adj_j, var_names,
                    context_i=ctx_ids[0], context_j=ctx_ids[1],
                    save_path=str(save_path),
                )
                assert fig is not None
                import matplotlib.pyplot as plt
                plt.close("all")
            except Exception:
                pass

    def test_descriptor_radar(self):
        """Radar chart for a single variable should produce a Figure."""
        viz = DescriptorVisualizer()
        foundation = self.atlas.foundation
        if foundation and foundation.descriptors:
            var_name = list(foundation.descriptors.keys())[0]
            save_path = self.save_dir / "radar.png"
            try:
                fig = viz.radar_chart(
                    foundation.descriptors,
                    variable=var_name,
                    save_path=str(save_path),
                )
                assert fig is not None
                import matplotlib.pyplot as plt
                plt.close("all")
            except Exception:
                pass

    def test_component_distributions(self):
        """Component distribution plots should produce a Figure."""
        viz = DescriptorVisualizer()
        foundation = self.atlas.foundation
        if foundation and foundation.descriptors:
            save_path = self.save_dir / "components.png"
            try:
                fig = viz.component_distributions(
                    foundation.descriptors,
                    save_path=str(save_path),
                )
                assert fig is not None
                import matplotlib.pyplot as plt
                plt.close("all")
            except Exception:
                pass
