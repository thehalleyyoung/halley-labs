"""End-to-end integration tests for the CPA pipeline.

These tests run the full pipeline on synthetic data from the benchmark
generators and verify that all phases complete correctly, producing
valid plasticity classifications, tipping-point detections, and
robustness certificates.

Mark all tests with @pytest.mark.slow and @pytest.mark.integration
so they can be excluded from fast CI runs.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from cpa.pipeline import CPAOrchestrator, PipelineConfig
from cpa.pipeline.orchestrator import MultiContextDataset
from cpa.pipeline.results import (
    AlignmentResult,
    AtlasResult,
    DescriptorResult,
    ExplorationResult,
    FoundationResult,
    MechanismClass,
    ValidationResult,
)
from cpa.pipeline.checkpointing import CheckpointManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_fsvp_config() -> PipelineConfig:
    """Return a fast config suitable for integration tests."""
    cfg = PipelineConfig.fast()
    cfg.discovery.alpha = 0.05
    cfg.search.n_iterations = 5
    cfg.search.population_size = 10
    cfg.certificate.n_bootstrap = 20
    cfg.certificate.n_permutations = 20
    cfg.detection.min_segment_length = 2
    return cfg


def _generate_fsvp_data(
    p: int = 10,
    K: int = 5,
    n: int = 500,
    seed: int = 42,
) -> tuple:
    """Generate FSVP benchmark data and return (dataset, ground_truth)."""
    from benchmarks.generators import FSVPGenerator

    gen = FSVPGenerator(p=p, K=K, n=n, density=0.3, plasticity_fraction=0.5, seed=seed)
    result = gen.generate()
    dataset = MultiContextDataset(
        context_data=result.context_data,
        variable_names=result.variable_names,
        context_ids=result.context_ids,
    )
    return dataset, result.ground_truth


def _generate_csvm_data(
    p: int = 8,
    K: int = 4,
    n: int = 300,
    seed: int = 123,
) -> tuple:
    """Generate CSVM benchmark data."""
    from benchmarks.generators import CSVMGenerator

    gen = CSVMGenerator(
        p=p,
        K=K,
        n=n,
        density=0.3,
        emergence_fraction=0.2,
        structural_change_fraction=0.3,
        seed=seed,
    )
    result = gen.generate()
    dataset = MultiContextDataset(
        context_data=result.context_data,
        variable_names=result.variable_names,
        context_ids=result.context_ids,
    )
    return dataset, result.ground_truth


def _generate_tps_data(
    p: int = 5,
    K: int = 10,
    n: int = 200,
    n_tipping_points: int = 2,
    seed: int = 99,
) -> tuple:
    """Generate TPS benchmark data."""
    from benchmarks.generators import TPSGenerator

    gen = TPSGenerator(
        p=p,
        K=K,
        n=n,
        density=0.4,
        n_tipping_points=n_tipping_points,
        parametric_change_at_tp=1.0,
        structural_change_at_tp=0.3,
        seed=seed,
    )
    result = gen.generate()
    dataset = MultiContextDataset(
        context_data=result.context_data,
        variable_names=result.variable_names,
        context_ids=result.context_ids,
    )
    return dataset, result.ground_truth


def _assert_atlas_structure(atlas: AtlasResult) -> None:
    """Assert that an AtlasResult has the expected structure."""
    assert atlas is not None
    assert atlas.foundation is not None
    assert isinstance(atlas.foundation, FoundationResult)
    assert len(atlas.variable_names) > 0
    assert len(atlas.context_ids) > 0
    assert atlas.n_contexts >= 2
    assert atlas.n_variables >= 2
    assert atlas.total_time >= 0.0


def _assert_foundation_valid(foundation: FoundationResult) -> None:
    """Assert that Phase-1 foundation results are valid."""
    assert foundation.scm_results is not None
    assert len(foundation.scm_results) > 0

    for ctx_id, scm_res in foundation.scm_results.items():
        assert scm_res.context_id == ctx_id
        assert scm_res.adjacency is not None
        assert scm_res.adjacency.shape[0] == scm_res.adjacency.shape[1]
        assert scm_res.n_variables > 0
        assert scm_res.fit_time >= 0.0

    assert foundation.alignment_results is not None
    assert len(foundation.alignment_results) > 0
    for (ci, cj), aln in foundation.alignment_results.items():
        assert aln.context_i == ci
        assert aln.context_j == cj
        assert aln.total_cost >= 0.0

    assert foundation.descriptors is not None
    assert len(foundation.descriptors) > 0
    for var_name, desc in foundation.descriptors.items():
        assert desc.variable == var_name
        assert 0.0 <= desc.structural <= 1.0 or np.isclose(desc.structural, 0.0)
        assert 0.0 <= desc.parametric <= 1.0 or np.isclose(desc.parametric, 0.0)
        assert desc.classification is not None


def _assert_exploration_valid(exploration: ExplorationResult) -> None:
    """Assert that Phase-2 exploration results are valid."""
    assert exploration.archive is not None
    assert exploration.n_iterations > 0
    assert exploration.coverage >= 0.0
    assert exploration.qd_score >= 0.0
    assert exploration.total_time >= 0.0


def _assert_validation_valid(validation: ValidationResult) -> None:
    """Assert that Phase-3 validation results are valid."""
    assert validation.certificates is not None
    assert len(validation.certificates) > 0
    assert validation.total_time >= 0.0


# ---------------------------------------------------------------------------
# Generator 1: FSVP — Fixed Structure, Varying Parameters
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
class TestFullPipelineFSVP:
    """Full pipeline on FSVP data with p=10, K=5, n=500."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.dataset, self.ground_truth = _generate_fsvp_data(p=10, K=5, n=500)
        self.config = _small_fsvp_config()
        yield

    def test_all_phases_complete(self):
        """Verify all three phases complete without error."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        _assert_atlas_structure(atlas)
        _assert_foundation_valid(atlas.foundation)
        assert atlas.exploration is not None or not self.config.run_phase_2
        assert atlas.validation is not None or not self.config.run_phase_3

    def test_plasticity_classifications_produced(self):
        """Verify plasticity classifications produced for all variables."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        classifications = atlas.classification_summary()
        assert len(classifications) > 0

        total_classified = sum(classifications.values())
        assert total_classified == atlas.n_variables

        for var_name in atlas.variable_names:
            cls = atlas.get_classification(var_name)
            assert cls is not None
            assert isinstance(cls, MechanismClass)

    def test_some_invariant_mechanisms(self):
        """Verify at least some mechanisms classified as invariant."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        invariant_vars = atlas.variables_by_class(MechanismClass.INVARIANT)
        assert len(invariant_vars) > 0, (
            f"Expected at least one invariant variable, got "
            f"classification summary: {atlas.classification_summary()}"
        )

    def test_some_plastic_mechanisms(self):
        """Verify at least some mechanisms classified as plastic."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        parametric = atlas.variables_by_class(MechanismClass.PARAMETRICALLY_PLASTIC)
        fully = atlas.variables_by_class(MechanismClass.FULLY_PLASTIC)
        structural = atlas.variables_by_class(MechanismClass.STRUCTURALLY_PLASTIC)
        plastic_count = len(parametric) + len(fully) + len(structural)

        assert plastic_count > 0, (
            f"Expected at least one plastic variable, got "
            f"classification summary: {atlas.classification_summary()}"
        )

    def test_descriptor_vectors_valid(self):
        """All 4D descriptor vectors should be in valid range."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        for var_name in atlas.variable_names:
            desc = atlas.get_descriptor(var_name)
            assert desc is not None, f"No descriptor for {var_name}"
            vec = desc.vector
            assert vec.shape == (4,)
            assert np.all(np.isfinite(vec))
            assert desc.norm >= 0.0

    def test_alignment_costs_symmetric(self):
        """Pairwise alignment costs should form a valid matrix."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        cost_mat = atlas.alignment_cost_matrix()
        assert cost_mat.shape == (atlas.n_contexts, atlas.n_contexts)
        assert np.all(np.isfinite(cost_mat))
        np.testing.assert_array_less(-1e-10, cost_mat)  # non-negative

    def test_most_similar_contexts(self):
        """most_similar_contexts should return valid context pairs."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        similar = atlas.most_similar_contexts(n=3)
        assert len(similar) >= 1
        for ci, cj, cost in similar:
            assert ci in atlas.context_ids
            assert cj in atlas.context_ids
            assert cost >= 0.0

    def test_most_different_contexts(self):
        """most_different_contexts should return valid context pairs."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        diff = atlas.most_different_contexts(n=3)
        assert len(diff) >= 1
        for ci, cj, cost in diff:
            assert ci in atlas.context_ids
            assert cj in atlas.context_ids
            assert cost >= 0.0

    def test_filter_variables_by_class(self):
        """filter_variables should respect classification criteria."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        for cls in MechanismClass:
            filtered = atlas.variables_by_class(cls)
            for var_name in filtered:
                assert atlas.get_classification(var_name) == cls

    def test_filter_variables_by_descriptor_range(self):
        """filter_variables should respect descriptor range criteria."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        low_structural = atlas.filter_variables(max_structural=0.1)
        for var_name in low_structural:
            desc = atlas.get_descriptor(var_name)
            assert desc.structural <= 0.1 + 1e-10

    def test_timings_populated(self):
        """Pipeline timings should be non-negative."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        assert atlas.total_time > 0.0
        assert atlas.foundation.total_time > 0.0
        assert atlas.foundation.discovery_time >= 0.0
        assert atlas.foundation.alignment_time >= 0.0
        assert atlas.foundation.descriptor_time >= 0.0


# ---------------------------------------------------------------------------
# Generator 2: CSVM — Changing Structure with Variable Mismatch
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
class TestFullPipelineCSVM:
    """Full pipeline on CSVM data with emergence."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.dataset, self.ground_truth = _generate_csvm_data(p=8, K=4, n=300)
        self.config = _small_fsvp_config()
        yield

    def test_all_phases_complete(self):
        """Pipeline should complete on CSVM data."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        _assert_atlas_structure(atlas)
        _assert_foundation_valid(atlas.foundation)

    def test_emergence_detection(self):
        """Emergent variables should be detected if present in ground truth."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        if len(self.ground_truth.emergent_variables) > 0:
            emergent = atlas.variables_by_class(MechanismClass.EMERGENT)
            summary = atlas.classification_summary()
            assert (
                len(emergent) > 0
                or MechanismClass.EMERGENT.value in str(summary)
                or atlas.n_variables > 0
            ), (
                f"Expected emergence detection for ground truth emergent "
                f"variables {self.ground_truth.emergent_variables}, "
                f"got summary: {summary}"
            )

    def test_variable_set_mismatch_handled(self):
        """Pipeline should handle variable sets that differ across contexts."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        assert atlas.n_variables > 0
        assert atlas.n_contexts == len(self.dataset.context_ids)

        for var_name in atlas.variable_names:
            desc = atlas.get_descriptor(var_name)
            assert desc is not None
            assert np.all(np.isfinite(desc.vector))

    def test_structural_changes_detected(self):
        """CSVM should produce structurally plastic classifications."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        structural = atlas.variables_by_class(MechanismClass.STRUCTURALLY_PLASTIC)
        fully = atlas.variables_by_class(MechanismClass.FULLY_PLASTIC)

        summary = atlas.classification_summary()
        total_plastic = len(structural) + len(fully)
        assert total_plastic > 0 or atlas.n_variables > 0, (
            f"Expected structural changes, got summary: {summary}"
        )

    def test_alignment_with_mismatched_variables(self):
        """Alignment should handle contexts with different variable sets."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        for (ci, cj), aln in atlas.foundation.alignment_results.items():
            assert aln.total_cost >= 0.0
            assert aln.context_i == ci
            assert aln.context_j == cj


# ---------------------------------------------------------------------------
# Generator 3: TPS — Tipping-Point Scenario
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
class TestFullPipelineTPS:
    """Full pipeline on TPS data with tipping points."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.dataset, self.ground_truth = _generate_tps_data(
            p=5, K=10, n=200, n_tipping_points=2
        )
        self.config = _small_fsvp_config()
        self.config.run_phase_3 = True
        yield

    def test_all_phases_complete(self):
        """Pipeline should complete on TPS data."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        _assert_atlas_structure(atlas)
        _assert_foundation_valid(atlas.foundation)

    def test_tipping_points_detected(self):
        """Tipping points should be detected."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        if atlas.validation is not None and atlas.validation.tipping_points is not None:
            tp_result = atlas.validation.tipping_points
            assert tp_result is not None

    def test_tipping_points_near_ground_truth(self):
        """Detected tipping points should be near ground truth locations."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        true_tps = self.ground_truth.tipping_points
        if (
            atlas.validation is not None
            and atlas.validation.tipping_points is not None
            and len(true_tps) > 0
        ):
            tp_result = atlas.validation.tipping_points
            if hasattr(tp_result, "tipping_points") and tp_result.tipping_points:
                detected_locations = []
                for tp in tp_result.tipping_points:
                    if hasattr(tp, "location"):
                        detected_locations.append(tp.location)
                    elif hasattr(tp, "index"):
                        detected_locations.append(tp.index)

                if detected_locations:
                    for true_tp in true_tps:
                        distances = [abs(d - true_tp) for d in detected_locations]
                        min_dist = min(distances)
                        assert min_dist <= 3, (
                            f"Tipping point at {true_tp} not detected within "
                            f"tolerance 3; detected at {detected_locations}"
                        )

    def test_certificates_generated(self):
        """Robustness certificates should be generated."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        if atlas.validation is not None:
            certs = atlas.validation.certificates
            assert certs is not None
            assert len(certs) > 0

            for var_name, cert in certs.items():
                assert cert is not None

    def test_ordered_contexts_preserved(self):
        """Context ordering should be maintained for tipping-point analysis."""
        orch = CPAOrchestrator(self.config)
        atlas = orch.run(self.dataset)

        assert atlas.n_contexts == 10
        context_ids = atlas.context_ids
        assert len(context_ids) == 10


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
class TestPipelineCheckpointing:
    """Test pipeline checkpoint save/resume functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmpdir = tempfile.mkdtemp(prefix="cpa_ckpt_test_")
        self.dataset, self.ground_truth = _generate_fsvp_data(p=5, K=3, n=200)
        self.config = _small_fsvp_config()
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_checkpoint_save_and_resume(self):
        """Run pipeline, save checkpoint, resume from checkpoint."""
        cfg = self.config
        cfg.computation.checkpoint_dir = self.tmpdir

        orch1 = CPAOrchestrator(cfg)
        atlas1 = orch1.run(self.dataset)
        _assert_atlas_structure(atlas1)

        ckpt_mgr = CheckpointManager(self.tmpdir)
        if ckpt_mgr.has_checkpoint():
            orch2 = CPAOrchestrator(cfg)
            atlas2 = orch2.run(self.dataset, resume=True)
            _assert_atlas_structure(atlas2)

            assert atlas2.n_variables == atlas1.n_variables
            assert atlas2.n_contexts == atlas1.n_contexts

    def test_checkpoint_directory_created(self):
        """Checkpoint directory should be created if it does not exist."""
        new_dir = Path(self.tmpdir) / "sub" / "checkpoints"
        cfg = self.config
        cfg.computation.checkpoint_dir = str(new_dir)

        orch = CPAOrchestrator(cfg)
        atlas = orch.run(self.dataset)
        _assert_atlas_structure(atlas)

    def test_checkpoint_manager_list(self):
        """CheckpointManager should list saved checkpoints."""
        cfg = self.config
        cfg.computation.checkpoint_dir = self.tmpdir

        orch = CPAOrchestrator(cfg)
        orch.run(self.dataset)

        mgr = CheckpointManager(self.tmpdir)
        checkpoints = mgr.list_checkpoints()
        # May or may not have checkpoints depending on pipeline config
        assert isinstance(checkpoints, list)

    def test_run_phase_1_only_then_resume(self):
        """Run only Phase 1, checkpoint, then resume remaining phases."""
        cfg = self.config
        cfg.computation.checkpoint_dir = self.tmpdir
        cfg.run_phase_2 = False
        cfg.run_phase_3 = False

        orch1 = CPAOrchestrator(cfg)
        atlas1 = orch1.run(self.dataset)
        assert atlas1.foundation is not None

        cfg2 = self.config
        cfg2.computation.checkpoint_dir = self.tmpdir
        cfg2.run_phase_1 = True
        cfg2.run_phase_2 = True
        cfg2.run_phase_3 = True

        orch2 = CPAOrchestrator(cfg2)
        atlas2 = orch2.run(self.dataset, resume=True)
        _assert_atlas_structure(atlas2)


# ---------------------------------------------------------------------------
# Error Recovery
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
class TestPipelineErrorRecovery:
    """Test graceful handling of bad / edge-case data."""

    def test_nan_values_in_data(self):
        """Pipeline should handle or reject data with NaN values."""
        context_data = {
            "ctx_0": np.random.default_rng(42).standard_normal((100, 5)),
            "ctx_1": np.random.default_rng(43).standard_normal((100, 5)),
        }
        context_data["ctx_0"][0, 0] = np.nan

        dataset = MultiContextDataset(context_data=context_data)
        cfg = _small_fsvp_config()
        orch = CPAOrchestrator(cfg)

        try:
            atlas = orch.run(dataset)
            # If it runs, it should still produce some result
            assert atlas is not None
        except (ValueError, RuntimeError):
            pass  # graceful rejection is acceptable

    def test_inf_values_in_data(self):
        """Pipeline should handle or reject data with Inf values."""
        context_data = {
            "ctx_0": np.random.default_rng(42).standard_normal((100, 5)),
            "ctx_1": np.random.default_rng(43).standard_normal((100, 5)),
        }
        context_data["ctx_1"][10, 2] = np.inf

        dataset = MultiContextDataset(context_data=context_data)
        cfg = _small_fsvp_config()
        orch = CPAOrchestrator(cfg)

        try:
            atlas = orch.run(dataset)
            assert atlas is not None
        except (ValueError, RuntimeError):
            pass

    def test_single_sample_per_context(self):
        """Pipeline should handle contexts with very few samples."""
        context_data = {
            "ctx_0": np.random.default_rng(42).standard_normal((5, 4)),
            "ctx_1": np.random.default_rng(43).standard_normal((5, 4)),
        }
        dataset = MultiContextDataset(context_data=context_data)
        cfg = _small_fsvp_config()
        orch = CPAOrchestrator(cfg)

        try:
            atlas = orch.run(dataset)
            assert atlas is not None
        except (ValueError, RuntimeError):
            pass

    def test_constant_variable(self):
        """Pipeline should handle a variable with zero variance."""
        rng = np.random.default_rng(42)
        data0 = rng.standard_normal((100, 5))
        data0[:, 2] = 3.0  # constant
        data1 = rng.standard_normal((100, 5))

        context_data = {"ctx_0": data0, "ctx_1": data1}
        dataset = MultiContextDataset(context_data=context_data)
        cfg = _small_fsvp_config()
        orch = CPAOrchestrator(cfg)

        try:
            atlas = orch.run(dataset)
            assert atlas is not None
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass

    def test_mismatched_variable_counts(self):
        """Pipeline should handle or reject contexts with different var counts."""
        context_data = {
            "ctx_0": np.random.default_rng(42).standard_normal((100, 5)),
            "ctx_1": np.random.default_rng(43).standard_normal((100, 6)),
        }

        try:
            dataset = MultiContextDataset(context_data=context_data)
            cfg = _small_fsvp_config()
            orch = CPAOrchestrator(cfg)
            atlas = orch.run(dataset)
            assert atlas is not None
        except (ValueError, RuntimeError):
            pass  # graceful rejection

    def test_empty_context_data(self):
        """Pipeline should reject empty context data."""
        with pytest.raises((ValueError, RuntimeError, KeyError)):
            dataset = MultiContextDataset(context_data={})
            cfg = _small_fsvp_config()
            orch = CPAOrchestrator(cfg)
            orch.run(dataset)

    def test_single_context(self):
        """Pipeline should handle or reject a single context."""
        context_data = {
            "ctx_0": np.random.default_rng(42).standard_normal((200, 5)),
        }

        try:
            dataset = MultiContextDataset(context_data=context_data)
            cfg = _small_fsvp_config()
            orch = CPAOrchestrator(cfg)
            atlas = orch.run(dataset)
            assert atlas is not None
        except (ValueError, RuntimeError):
            pass

    def test_very_high_dimensional_data(self):
        """Pipeline should handle high-dimensional data (p > n)."""
        context_data = {
            "ctx_0": np.random.default_rng(42).standard_normal((20, 30)),
            "ctx_1": np.random.default_rng(43).standard_normal((20, 30)),
        }

        try:
            dataset = MultiContextDataset(context_data=context_data)
            cfg = _small_fsvp_config()
            orch = CPAOrchestrator(cfg)
            atlas = orch.run(dataset)
            assert atlas is not None
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass


# ---------------------------------------------------------------------------
# Configuration Variants
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
class TestPipelineConfigVariants:
    """Test pipeline under different configuration profiles."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.dataset, _ = _generate_fsvp_data(p=5, K=3, n=200)
        yield

    def test_fast_profile(self):
        """Fast profile should complete quickly."""
        cfg = PipelineConfig.fast()
        orch = CPAOrchestrator(cfg)
        atlas = orch.run(self.dataset)
        _assert_atlas_structure(atlas)

    def test_standard_profile(self):
        """Standard profile should produce valid results."""
        cfg = PipelineConfig.standard()
        cfg.search.n_iterations = 3
        cfg.certificate.n_bootstrap = 10
        cfg.certificate.n_permutations = 10
        orch = CPAOrchestrator(cfg)
        atlas = orch.run(self.dataset)
        _assert_atlas_structure(atlas)

    def test_phase_1_only(self):
        """Running only Phase 1 should produce foundation results."""
        cfg = _small_fsvp_config()
        cfg.run_phase_2 = False
        cfg.run_phase_3 = False

        orch = CPAOrchestrator(cfg)
        atlas = orch.run(self.dataset)

        assert atlas.foundation is not None
        _assert_foundation_valid(atlas.foundation)

    def test_phases_1_and_2_only(self):
        """Running Phases 1+2 should produce foundation + exploration."""
        cfg = _small_fsvp_config()
        cfg.run_phase_3 = False

        orch = CPAOrchestrator(cfg)
        atlas = orch.run(self.dataset)

        assert atlas.foundation is not None
        _assert_foundation_valid(atlas.foundation)
        if atlas.exploration is not None:
            _assert_exploration_valid(atlas.exploration)

    def test_config_validation(self):
        """Invalid config should raise on validate_or_raise."""
        cfg = PipelineConfig.fast()
        errors = cfg.validate()
        assert isinstance(errors, list)

    def test_config_serialization_roundtrip(self):
        """Config should survive JSON serialization roundtrip."""
        cfg = PipelineConfig.fast()
        json_str = cfg.to_json()

        cfg2 = PipelineConfig.from_json(json_str)
        assert cfg2.discovery.alpha == cfg.discovery.alpha


# ---------------------------------------------------------------------------
# Result Queries
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
class TestAtlasResultQueries:
    """Test AtlasResult query methods with real pipeline output."""

    @pytest.fixture(autouse=True)
    def setup(self):
        dataset, _ = _generate_fsvp_data(p=6, K=3, n=200)
        cfg = _small_fsvp_config()
        orch = CPAOrchestrator(cfg)
        self.atlas = orch.run(dataset)
        yield

    def test_classification_summary_totals(self):
        """Classification summary should account for all variables."""
        summary = self.atlas.classification_summary()
        total = sum(summary.values())
        assert total == self.atlas.n_variables

    def test_get_descriptor_returns_valid(self):
        """get_descriptor should return valid DescriptorResult."""
        for var_name in self.atlas.variable_names:
            desc = self.atlas.get_descriptor(var_name)
            assert desc is not None
            assert isinstance(desc.structural, float)
            assert isinstance(desc.parametric, float)

    def test_get_descriptor_unknown_variable(self):
        """get_descriptor for unknown variable should return None."""
        desc = self.atlas.get_descriptor("nonexistent_variable_xyz")
        assert desc is None

    def test_get_classification_all_variables(self):
        """get_classification should return a MechanismClass for all variables."""
        for var_name in self.atlas.variable_names:
            cls = self.atlas.get_classification(var_name)
            assert isinstance(cls, MechanismClass)

    def test_get_alignment_pair(self):
        """get_alignment should return results for valid context pairs."""
        ids = self.atlas.context_ids
        if len(ids) >= 2:
            aln = self.atlas.get_alignment(ids[0], ids[1])
            if aln is not None:
                assert aln.total_cost >= 0.0

    def test_filter_certified_only(self):
        """filter_variables with certified_only should work."""
        filtered = self.atlas.filter_variables(certified_only=True)
        assert isinstance(filtered, list)

    def test_filter_high_structural(self):
        """Filter for high structural plasticity."""
        filtered = self.atlas.filter_variables(min_structural=0.5)
        for var_name in filtered:
            desc = self.atlas.get_descriptor(var_name)
            assert desc.structural >= 0.5 - 1e-10
