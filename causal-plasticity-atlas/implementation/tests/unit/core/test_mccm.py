"""Unit tests for cpa.core.mccm.MultiContextCausalModel."""

from __future__ import annotations

import numpy as np
import pytest

from cpa.core.context import ContextSpace
from cpa.core.mccm import MultiContextCausalModel, build_mccm_from_scms
from cpa.core.scm import StructuralCausalModel
from cpa.core.types import Context


# ===================================================================
# Helpers / fixtures
# ===================================================================


def _make_scm(
    adj: list[list[float]],
    names: list[str] | None = None,
    coefs: list[list[float]] | None = None,
    sample_size: int = 100,
) -> StructuralCausalModel:
    """Convenience factory for a StructuralCausalModel."""
    a = np.array(adj, dtype=np.float64)
    c = np.array(coefs, dtype=np.float64) if coefs is not None else None
    return StructuralCausalModel(
        adjacency_matrix=a,
        variable_names=names,
        regression_coefficients=c,
        sample_size=sample_size,
    )


@pytest.fixture()
def scm_abc() -> StructuralCausalModel:
    """3-variable DAG: A->B->C with variables {A, B, C}."""
    return _make_scm(
        adj=[[0, 1, 0], [0, 0, 1], [0, 0, 0]],
        names=["A", "B", "C"],
        coefs=[[0, 0.5, 0], [0, 0, 0.8], [0, 0, 0]],
    )


@pytest.fixture()
def scm_abc_modified() -> StructuralCausalModel:
    """Same structure as scm_abc but different coefficients."""
    return _make_scm(
        adj=[[0, 1, 0], [0, 0, 1], [0, 0, 0]],
        names=["A", "B", "C"],
        coefs=[[0, 0.9, 0], [0, 0, 0.3], [0, 0, 0]],
    )


@pytest.fixture()
def scm_abc_extra_edge() -> StructuralCausalModel:
    """A->B->C plus A->C."""
    return _make_scm(
        adj=[[0, 1, 1], [0, 0, 1], [0, 0, 0]],
        names=["A", "B", "C"],
        coefs=[[0, 0.5, 0.4], [0, 0, 0.8], [0, 0, 0]],
    )


@pytest.fixture()
def scm_bcd() -> StructuralCausalModel:
    """3-variable DAG with variables {B, C, D}: B->C->D."""
    return _make_scm(
        adj=[[0, 1, 0], [0, 0, 1], [0, 0, 0]],
        names=["B", "C", "D"],
        coefs=[[0, 0.6, 0], [0, 0, 0.7], [0, 0, 0]],
    )


@pytest.fixture()
def scm_xy() -> StructuralCausalModel:
    """2-variable DAG with variables {X, Y}: X->Y (disjoint from ABC)."""
    return _make_scm(
        adj=[[0, 1], [0, 0]],
        names=["X", "Y"],
    )


@pytest.fixture()
def ctx_a() -> Context:
    return Context(id="ctx_a", metadata={"env": "lab"})


@pytest.fixture()
def ctx_b() -> Context:
    return Context(id="ctx_b", metadata={"env": "field"})


@pytest.fixture()
def ctx_c() -> Context:
    return Context(id="ctx_c", metadata={"env": "sim"})


# ===================================================================
# Construction
# ===================================================================


class TestConstruction:
    """Test MCCM construction and mode validation."""

    def test_default_construction(self):
        mccm = MultiContextCausalModel()
        assert mccm.mode == "intersection"
        assert mccm.num_contexts == 0
        assert mccm.context_ids == []

    def test_union_mode(self):
        mccm = MultiContextCausalModel(mode="union")
        assert mccm.mode == "union"

    def test_intersection_mode(self):
        mccm = MultiContextCausalModel(mode="intersection")
        assert mccm.mode == "intersection"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            MultiContextCausalModel(mode="invalid")

    def test_with_context_space(self):
        cs = ContextSpace([Context(id="pre")])
        mccm = MultiContextCausalModel(context_space=cs, mode="union")
        assert mccm.context_space is cs

    def test_default_context_space_created(self):
        mccm = MultiContextCausalModel()
        assert isinstance(mccm.context_space, ContextSpace)


# ===================================================================
# Context management
# ===================================================================


class TestContextManagement:
    """Test adding, removing, and retrieving contexts."""

    def test_add_context(self, ctx_a, scm_abc):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        assert mccm.num_contexts == 1
        assert mccm.context_ids == ["ctx_a"]

    def test_add_multiple_contexts(self, ctx_a, ctx_b, scm_abc, scm_abc_modified):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_abc_modified)
        assert mccm.num_contexts == 2
        assert set(mccm.context_ids) == {"ctx_a", "ctx_b"}

    def test_add_duplicate_context_raises(self, ctx_a, scm_abc):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        with pytest.raises(ValueError, match="already exists"):
            mccm.add_context(ctx_a, scm_abc)

    def test_add_context_with_metadata(self, ctx_a, scm_abc):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc, metadata={"source": "experiment_1"})
        assert mccm._metadata["ctx_a"]["source"] == "experiment_1"

    def test_get_scm(self, ctx_a, scm_abc):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        retrieved = mccm.get_scm("ctx_a")
        assert retrieved is scm_abc

    def test_get_scm_missing_raises(self):
        mccm = MultiContextCausalModel()
        with pytest.raises(KeyError, match="not found"):
            mccm.get_scm("nonexistent")

    def test_get_context(self, ctx_a, scm_abc):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        retrieved = mccm.get_context("ctx_a")
        assert retrieved.id == "ctx_a"

    def test_remove_context(self, ctx_a, scm_abc):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        removed = mccm.remove_context("ctx_a")
        assert removed is scm_abc
        assert mccm.num_contexts == 0

    def test_remove_missing_context_raises(self):
        mccm = MultiContextCausalModel()
        with pytest.raises(KeyError, match="not found"):
            mccm.remove_context("nonexistent")

    def test_remove_and_readd(self, ctx_a, scm_abc, scm_abc_modified):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.remove_context("ctx_a")
        mccm.add_context(ctx_a, scm_abc_modified)
        assert mccm.get_scm("ctx_a") is scm_abc_modified


# ===================================================================
# Variable alignment
# ===================================================================


class TestVariableAlignment:
    """Test variable union, intersection, and presence matrix."""

    def test_variable_union_single_context(self, ctx_a, scm_abc):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        assert mccm.variable_union() == {"A", "B", "C"}

    def test_variable_union_overlapping(self, ctx_a, ctx_b, scm_abc, scm_bcd):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_bcd)
        assert mccm.variable_union() == {"A", "B", "C", "D"}

    def test_variable_intersection_identical(self, ctx_a, ctx_b, scm_abc, scm_abc_modified):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_abc_modified)
        assert mccm.variable_intersection() == {"A", "B", "C"}

    def test_variable_intersection_partial(self, ctx_a, ctx_b, scm_abc, scm_bcd):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_bcd)
        assert mccm.variable_intersection() == {"B", "C"}

    def test_variable_intersection_empty(self):
        mccm = MultiContextCausalModel()
        assert mccm.variable_intersection() == set()

    def test_shared_variables_alias(self, ctx_a, ctx_b, scm_abc, scm_bcd):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_bcd)
        assert mccm.shared_variables() == mccm.variable_intersection()

    def test_context_specific_variables(self, ctx_a, ctx_b, scm_abc, scm_bcd):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_bcd)
        assert mccm.context_specific_variables("ctx_a") == {"A"}
        assert mccm.context_specific_variables("ctx_b") == {"D"}

    def test_context_specific_variables_none(self, ctx_a, ctx_b, scm_abc, scm_abc_modified):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_abc_modified)
        assert mccm.context_specific_variables("ctx_a") == set()

    def test_effective_variables_intersection_mode(self, ctx_a, ctx_b, scm_abc, scm_bcd):
        mccm = MultiContextCausalModel(mode="intersection")
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_bcd)
        assert mccm.effective_variables() == {"B", "C"}

    def test_effective_variables_union_mode(self, ctx_a, ctx_b, scm_abc, scm_bcd):
        mccm = MultiContextCausalModel(mode="union")
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_bcd)
        assert mccm.effective_variables() == {"A", "B", "C", "D"}

    def test_variable_presence_matrix(self, ctx_a, ctx_b, scm_abc, scm_bcd):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_bcd)
        names, mat = mccm.variable_presence_matrix()
        assert names == ["A", "B", "C", "D"]
        assert mat.shape == (2, 4)
        # ctx_a has A,B,C (indices 0,1,2) but not D (index 3)
        ctx_a_idx = mccm.context_ids.index("ctx_a")
        assert mat[ctx_a_idx, 0]  # A
        assert mat[ctx_a_idx, 1]  # B
        assert mat[ctx_a_idx, 2]  # C
        assert not mat[ctx_a_idx, 3]  # D
        # ctx_b has B,C,D but not A
        ctx_b_idx = mccm.context_ids.index("ctx_b")
        assert not mat[ctx_b_idx, 0]  # A
        assert mat[ctx_b_idx, 1]  # B
        assert mat[ctx_b_idx, 2]  # C
        assert mat[ctx_b_idx, 3]  # D

    def test_variable_presence_matrix_dtype(self, ctx_a, scm_abc):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        _, mat = mccm.variable_presence_matrix()
        assert mat.dtype == bool


# ===================================================================
# Aligned subgraph
# ===================================================================


class TestAlignedSubgraph:
    """Test aligned_scm_pair extraction."""

    def test_aligned_pair_shared_variables(self, ctx_a, ctx_b, scm_abc, scm_bcd):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_bcd)
        sub_a, sub_b, shared = mccm.aligned_scm_pair("ctx_a", "ctx_b")
        assert shared == ["B", "C"]
        assert sub_a.num_variables == 2
        assert sub_b.num_variables == 2
        assert set(sub_a.variable_names) == {"B", "C"}
        assert set(sub_b.variable_names) == {"B", "C"}

    def test_aligned_pair_identical_variables(self, ctx_a, ctx_b, scm_abc, scm_abc_modified):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_abc_modified)
        sub_a, sub_b, shared = mccm.aligned_scm_pair("ctx_a", "ctx_b")
        assert shared == ["A", "B", "C"]
        assert sub_a.num_variables == 3

    def test_aligned_pair_no_shared_raises(self, ctx_a, ctx_b, scm_abc, scm_xy):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_xy)
        with pytest.raises(ValueError, match="share no variables"):
            mccm.aligned_scm_pair("ctx_a", "ctx_b")

    def test_aligned_pair_preserves_edges(self, ctx_a, ctx_b, scm_abc, scm_bcd):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_bcd)
        sub_a, sub_b, shared = mccm.aligned_scm_pair("ctx_a", "ctx_b")
        # B->C should be preserved in both subgraphs
        b_idx_a = sub_a.variable_index("B")
        c_idx_a = sub_a.variable_index("C")
        assert sub_a.has_edge(b_idx_a, c_idx_a)
        b_idx_b = sub_b.variable_index("B")
        c_idx_b = sub_b.variable_index("C")
        assert sub_b.has_edge(b_idx_b, c_idx_b)


# ===================================================================
# Pairwise comparison
# ===================================================================


class TestPairwiseComparison:
    """Test pairwise SHD and edge comparison."""

    def test_pairwise_shd_identical(self, ctx_a, ctx_b, scm_abc, scm_abc_modified):
        """Same structure → SHD = 0."""
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_abc_modified)
        D = mccm.pairwise_shd()
        assert D.shape == (2, 2)
        assert D[0, 0] == 0.0
        assert D[1, 1] == 0.0
        assert D[0, 1] == 0.0  # same structure
        assert D[1, 0] == 0.0

    def test_pairwise_shd_different_structure(self, ctx_a, ctx_b, scm_abc, scm_abc_extra_edge):
        """Extra edge → SHD > 0."""
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_abc_extra_edge)
        D = mccm.pairwise_shd()
        assert D[0, 1] > 0
        assert D[0, 1] == D[1, 0]  # symmetric

    def test_pairwise_shd_single_context(self, ctx_a, scm_abc):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        D = mccm.pairwise_shd()
        assert D.shape == (1, 1)
        assert D[0, 0] == 0.0

    def test_pairwise_shd_no_shared_vars_is_nan(self, ctx_a, ctx_b, scm_abc, scm_xy):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_xy)
        D = mccm.pairwise_shd()
        assert np.isnan(D[0, 1])
        assert np.isnan(D[1, 0])

    def test_pairwise_shd_three_contexts(
        self, ctx_a, ctx_b, ctx_c, scm_abc, scm_abc_modified, scm_abc_extra_edge
    ):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_abc_modified)
        mccm.add_context(ctx_c, scm_abc_extra_edge)
        D = mccm.pairwise_shd()
        assert D.shape == (3, 3)
        # diagonal is zero
        np.testing.assert_array_equal(np.diag(D), [0, 0, 0])
        # symmetric
        np.testing.assert_array_equal(D, D.T)

    def test_edge_comparison_same_structure_diff_coefs(
        self, ctx_a, ctx_b, scm_abc, scm_abc_modified
    ):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_abc_modified)
        result = mccm.pairwise_edge_comparison("ctx_a", "ctx_b")
        assert "shared" in result
        assert "modified" in result
        assert "context_specific_a" in result
        assert "context_specific_b" in result
        # Same structure, no context-specific edges
        assert result["context_specific_a"] == set()
        assert result["context_specific_b"] == set()
        # Both edges exist, but coefficients differ → modified
        assert len(result["modified"]) == 2  # A->B and B->C differ

    def test_edge_comparison_extra_edge(self, ctx_a, ctx_b, scm_abc, scm_abc_extra_edge):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_abc_extra_edge)
        result = mccm.pairwise_edge_comparison("ctx_a", "ctx_b")
        # A->C only in ctx_b
        assert ("A", "C") in result["context_specific_b"]
        assert result["context_specific_a"] == set()

    def test_edge_comparison_no_shared_raises(self, ctx_a, ctx_b, scm_abc, scm_xy):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_xy)
        with pytest.raises(ValueError, match="share no variables"):
            mccm.pairwise_edge_comparison("ctx_a", "ctx_b")


# ===================================================================
# Alignment mapping
# ===================================================================


class TestAlignmentMapping:
    """Test alignment_mapping between context pairs."""

    def test_alignment_mapping_basic(self, ctx_a, ctx_b, scm_abc, scm_abc_modified):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_abc_modified)
        am = mccm.alignment_mapping("ctx_a", "ctx_b")
        # Identity mapping for shared variables
        assert am.pi == {"A": "A", "B": "B", "C": "C"}
        assert 0.0 <= am.quality_score <= 1.0
        assert am.structural_divergence >= 0.0

    def test_alignment_mapping_has_edge_partition(self, ctx_a, ctx_b, scm_abc, scm_abc_extra_edge):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_abc_extra_edge)
        am = mccm.alignment_mapping("ctx_a", "ctx_b")
        assert "shared" in am.edge_partition
        assert "modified" in am.edge_partition
        assert "context_specific_a" in am.edge_partition
        assert "context_specific_b" in am.edge_partition


# ===================================================================
# Serialization
# ===================================================================


class TestSerialization:
    """Test to_dict / from_dict round-trip."""

    def test_round_trip(self, ctx_a, ctx_b, scm_abc, scm_abc_modified):
        mccm = MultiContextCausalModel(mode="union")
        mccm.add_context(ctx_a, scm_abc, metadata={"run": 1})
        mccm.add_context(ctx_b, scm_abc_modified, metadata={"run": 2})

        d = mccm.to_dict()
        restored = MultiContextCausalModel.from_dict(d)

        assert restored.mode == "union"
        assert restored.num_contexts == 2
        assert set(restored.context_ids) == {"ctx_a", "ctx_b"}

    def test_round_trip_preserves_adjacency(self, ctx_a, scm_abc):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        d = mccm.to_dict()
        restored = MultiContextCausalModel.from_dict(d)
        original_adj = scm_abc.adjacency_matrix
        restored_adj = restored.get_scm("ctx_a").adjacency_matrix
        np.testing.assert_array_equal(original_adj, restored_adj)

    def test_round_trip_preserves_variable_names(self, ctx_a, scm_abc):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        d = mccm.to_dict()
        restored = MultiContextCausalModel.from_dict(d)
        assert restored.get_scm("ctx_a").variable_names == ["A", "B", "C"]

    def test_to_mccm_dataclass(self, ctx_a, ctx_b, scm_abc, scm_bcd):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_bcd)
        lightweight = mccm.to_mccm()
        assert set(lightweight.scms.keys()) == {"ctx_a", "ctx_b"}
        assert set(lightweight.shared_variables) == {"B", "C"}


# ===================================================================
# Factory function
# ===================================================================


class TestBuildFromSCMs:
    """Test the build_mccm_from_scms factory."""

    def test_build_basic(self, scm_abc, scm_abc_modified):
        mccm = build_mccm_from_scms(
            {"c1": scm_abc, "c2": scm_abc_modified}, mode="union"
        )
        assert mccm.num_contexts == 2
        assert mccm.mode == "union"

    def test_build_with_contexts(self, scm_abc, scm_bcd):
        ctx1 = Context(id="c1")
        ctx2 = Context(id="c2")
        mccm = build_mccm_from_scms(
            {"c1": scm_abc, "c2": scm_bcd},
            contexts={"c1": ctx1, "c2": ctx2},
        )
        assert mccm.get_context("c1").id == "c1"


# ===================================================================
# Summary and batch operations
# ===================================================================


class TestSummaryAndBatch:
    """Test summary statistics and batch operations."""

    def test_summary_string(self, ctx_a, ctx_b, scm_abc, scm_abc_modified):
        mccm = MultiContextCausalModel(mode="intersection")
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_abc_modified)
        s = mccm.summary()
        assert "MultiContextCausalModel" in s
        assert "intersection" in s

    def test_repr(self, ctx_a, scm_abc):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        r = repr(mccm)
        assert "MultiContextCausalModel" in r

    def test_edge_count_summary(self, ctx_a, ctx_b, scm_abc, scm_abc_extra_edge):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_abc_extra_edge)
        counts = mccm.edge_count_summary()
        assert counts["ctx_a"] == 2  # A->B, B->C
        assert counts["ctx_b"] == 3  # A->B, B->C, A->C

    def test_density_summary(self, ctx_a, scm_abc):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        densities = mccm.density_summary()
        assert "ctx_a" in densities
        assert 0.0 < densities["ctx_a"] < 1.0

    def test_apply_to_all(self, ctx_a, ctx_b, scm_abc, scm_abc_modified):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_abc_modified)
        result = mccm.apply_to_all(lambda cid, scm: scm.num_variables)
        assert result == {"ctx_a": 3, "ctx_b": 3}

    def test_filter_contexts(self, ctx_a, ctx_b, scm_abc, scm_abc_extra_edge):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_abc_extra_edge)
        filtered = mccm.filter_contexts(lambda cid, scm: scm.num_edges > 2)
        assert filtered.num_contexts == 1
        assert filtered.context_ids == ["ctx_b"]

    def test_shared_structure_fraction_identical(self, ctx_a, ctx_b, scm_abc, scm_abc_modified):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_abc_modified)
        frac = mccm.shared_structure_fraction()
        assert frac == 1.0  # same edges in both

    def test_shared_structure_fraction_extra_edge(
        self, ctx_a, ctx_b, scm_abc, scm_abc_extra_edge
    ):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_abc_extra_edge)
        frac = mccm.shared_structure_fraction()
        assert 0.0 < frac < 1.0

    def test_shared_structure_fraction_single_context(self, ctx_a, scm_abc):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        assert mccm.shared_structure_fraction() == 1.0


# ===================================================================
# Validation
# ===================================================================


class TestValidation:
    """Test the validate() method."""

    def test_validate_no_warnings(self, ctx_a, scm_abc):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        warns = mccm.validate()
        assert isinstance(warns, list)

    def test_validate_no_shared_variables(self, ctx_a, ctx_b, scm_abc, scm_xy):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_xy)
        warns = mccm.validate()
        assert any("No shared variables" in w for w in warns)

    def test_validate_low_overlap(self, ctx_a, ctx_b, scm_abc, scm_bcd):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_bcd)
        warns = mccm.validate()
        # 2/4 shared = 50%, borderline
        assert isinstance(warns, list)


# ===================================================================
# Divergence matrix
# ===================================================================


class TestDivergenceMatrix:
    """Test divergence_matrix method."""

    def test_shd_metric(self, ctx_a, ctx_b, scm_abc, scm_abc_extra_edge):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        mccm.add_context(ctx_b, scm_abc_extra_edge)
        D = mccm.divergence_matrix(metric="shd")
        assert D.shape == (2, 2)
        assert D[0, 1] > 0

    def test_unknown_metric_raises(self, ctx_a, scm_abc):
        mccm = MultiContextCausalModel()
        mccm.add_context(ctx_a, scm_abc)
        with pytest.raises(ValueError, match="Unknown metric"):
            mccm.divergence_matrix(metric="unknown")
