"""Unit tests for cpa.core.types."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cpa.core.types import (
    AlignmentMapping,
    ChangeType,
    CertificateType,
    Context,
    EdgeClassification,
    MCCM,
    PlasticityClass,
    PlasticityDescriptor,
    QDGenome,
    RobustnessCertificate,
    SCM,
    TippingPoint,
)


# ── helpers ───────────────────────────────────────────────────────────

def _make_scm(p: int = 3, *, names: list[str] | None = None,
              sample_size: int = 100) -> SCM:
    """Build a simple chain DAG  0 → 1 → … → p-1."""
    adj = np.zeros((p, p))
    for i in range(p - 1):
        adj[i, i + 1] = 1.0
    names = names or [f"X{i}" for i in range(p)]
    return SCM(adj, adj * 0.5, np.ones(p), names, sample_size)


def _make_context(cid: str = "ctx_a", **kw) -> Context:
    return Context(id=cid, **kw)


def _edge_partition(shared=None, modified=None, csa=None, csb=None):
    return {
        "shared": shared or set(),
        "modified": modified or set(),
        "context_specific_a": csa or set(),
        "context_specific_b": csb or set(),
    }


def _make_descriptor(**overrides) -> PlasticityDescriptor:
    defaults = dict(
        psi_S=0.3, psi_P=0.5, psi_E=0.1, psi_CS=0.2,
        confidence_intervals={"psi_S": (0.2, 0.4), "psi_P": (0.4, 0.6)},
        classification=PlasticityClass.PARAMETRIC_PLASTIC,
        variable_index=1, variable_name="Y",
    )
    defaults.update(overrides)
    return PlasticityDescriptor(**defaults)


def _make_tipping_point(**overrides) -> TippingPoint:
    defaults = dict(
        context_location=3, p_value=0.01, effect_size=1.5,
        affected_mechanisms=["X0", "X1"],
        change_types=[ChangeType.STRUCTURAL, ChangeType.PARAMETRIC],
        left_segment=["c0", "c1"], right_segment=["c2", "c3"],
    )
    defaults.update(overrides)
    return TippingPoint(**defaults)


def _make_certificate(**overrides) -> RobustnessCertificate:
    defaults = dict(
        type=CertificateType.STRONG_INVARIANCE, validity=True,
        max_sqrt_jsd=0.05, upper_confidence_bound=0.1,
        robustness_margin=0.3, stability_selection_probs={"X0": 0.9},
        assumptions=["linearity"], validity_conditions=["n>=50"],
        min_sample_size_warning=False,
    )
    defaults.update(overrides)
    return RobustnessCertificate(**defaults)


# =====================================================================
# Enums
# =====================================================================

class TestEnums:
    @pytest.mark.parametrize("cls,member,value", [
        (PlasticityClass, "INVARIANT", "invariant"),
        (PlasticityClass, "MIXED", "mixed"),
        (CertificateType, "CANNOT_ISSUE", "cannot_issue"),
        (EdgeClassification, "SHARED", "shared"),
        (ChangeType, "STRUCTURAL", "structural"),
    ])
    def test_enum_values(self, cls, member, value):
        assert cls[member].value == value

    def test_plasticity_class_members(self):
        assert len(PlasticityClass) == 5

    def test_certificate_type_members(self):
        assert len(CertificateType) == 4

    def test_edge_classification_members(self):
        assert len(EdgeClassification) == 4

    def test_change_type_members(self):
        assert len(ChangeType) == 2

    def test_enum_repr(self):
        assert "INVARIANT" in repr(PlasticityClass.INVARIANT)
        assert "STRUCTURAL" in repr(ChangeType.STRUCTURAL)


# =====================================================================
# SCM
# =====================================================================

class TestSCM:
    def test_basic_construction(self):
        scm = _make_scm(3)
        assert scm.num_variables == 3
        assert scm.num_edges == 2
        assert scm.variable_names == ["X0", "X1", "X2"]
        assert scm.sample_size == 100

    def test_parents_children(self):
        scm = _make_scm(4)
        assert scm.parents(0) == []
        assert scm.parents(1) == [0]
        assert scm.children(0) == [1]
        assert scm.children(3) == []

    def test_variable_index(self):
        scm = _make_scm(3)
        assert scm.variable_index("X1") == 1
        with pytest.raises(ValueError, match="not in model"):
            scm.variable_index("Z")

    def test_markov_blanket(self):
        # chain: 0→1→2
        scm = _make_scm(3)
        mb = scm.markov_blanket(1)
        assert mb == {0, 2}

    def test_markov_blanket_with_coparents(self):
        adj = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=float)
        scm = SCM(adj, adj * 0.5, np.ones(3), ["A", "B", "C"], 50)
        # MB of A: child={C}, coparents of C={B}
        assert scm.markov_blanket(0) == {1, 2}

    def test_topological_sort_chain(self):
        scm = _make_scm(4)
        assert scm.topological_sort() == [0, 1, 2, 3]

    def test_topological_sort_cycle_raises(self):
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        scm = SCM(adj, adj, np.ones(2), ["A", "B"], 10)
        with pytest.raises(ValueError, match="cycle"):
            scm.topological_sort()

    def test_is_dag_valid(self):
        assert _make_scm(3).is_dag_valid()
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        scm = SCM(adj, adj, np.ones(2), ["A", "B"], 10)
        assert not scm.is_dag_valid()

    def test_serialization_roundtrip(self):
        scm = _make_scm(3)
        d = scm.to_dict()
        scm2 = SCM.from_dict(d)
        np.testing.assert_array_equal(scm2.adjacency_matrix, scm.adjacency_matrix)
        assert scm2.variable_names == scm.variable_names
        assert scm2.sample_size == scm.sample_size

    # ── validation errors ──

    @pytest.mark.parametrize("field,bad,match", [
        ("adjacency_matrix", np.ones((2, 3)), "square"),
        ("regression_coefficients", np.ones((2, 2)), "regression_coefficients shape"),
        ("residual_variances", np.ones(2), "residual_variances shape"),
    ])
    def test_shape_mismatch(self, field, bad, match):
        kw = dict(
            adjacency_matrix=np.zeros((3, 3)),
            regression_coefficients=np.zeros((3, 3)),
            residual_variances=np.ones(3),
            variable_names=["A", "B", "C"],
            sample_size=10,
        )
        kw[field] = bad
        with pytest.raises(ValueError, match=match):
            SCM(**kw)

    def test_duplicate_variable_names(self):
        with pytest.raises(ValueError, match="Duplicate"):
            _make_scm(3, names=["X", "X", "Y"])

    def test_negative_sample_size(self):
        with pytest.raises(ValueError, match="sample_size"):
            _make_scm(3, sample_size=-1)

    def test_zero_sample_size_ok(self):
        scm = _make_scm(3, sample_size=0)
        assert scm.sample_size == 0

    def test_isolated_node(self):
        adj = np.zeros((3, 3))
        scm = SCM(adj, adj, np.ones(3), ["A", "B", "C"], 50)
        assert scm.num_edges == 0
        assert scm.parents(0) == []
        assert scm.children(0) == []
        assert scm.markov_blanket(0) == set()
        assert set(scm.topological_sort()) == {0, 1, 2}


# =====================================================================
# Context
# =====================================================================

class TestContext:
    def test_basic(self):
        ctx = _make_context("c1", metadata={"temp": 37})
        assert ctx.id == "c1"
        assert ctx.metadata == {"temp": 37}
        assert not ctx.is_ordered

    def test_ordered(self):
        ctx = _make_context("c1", ordering_value=2.5)
        assert ctx.is_ordered
        assert ctx.ordering_value == 2.5

    def test_empty_id_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            Context(id="")

    def test_whitespace_id_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            Context(id="   ")

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_ordering_raises(self, bad_val):
        with pytest.raises(ValueError, match="finite"):
            Context(id="c", ordering_value=bad_val)

    def test_hash_and_eq(self):
        a = Context(id="c1", metadata={"x": 1})
        b = Context(id="c1", metadata={"y": 2})
        assert a == b
        assert hash(a) == hash(b)

    def test_ne_different_id(self):
        assert Context(id="a") != Context(id="b")

    def test_serialization_roundtrip(self):
        ctx = Context(id="c1", metadata={"k": "v"}, ordering_value=3.0)
        ctx2 = Context.from_dict(ctx.to_dict())
        assert ctx2 == ctx
        assert ctx2.ordering_value == 3.0
        assert ctx2.metadata == {"k": "v"}

    def test_serialization_no_ordering(self):
        ctx = Context(id="c1")
        d = ctx.to_dict()
        assert "ordering_value" not in d
        ctx2 = Context.from_dict(d)
        assert ctx2.ordering_value is None

    def test_metadata_type_check(self):
        with pytest.raises(TypeError, match="dict"):
            Context(id="c1", metadata="not_a_dict")  # type: ignore[arg-type]


# =====================================================================
# MCCM
# =====================================================================

class TestMCCM:
    def _two_context_mccm(self):
        scm_a = _make_scm(3, names=["X", "Y", "Z"])
        scm_b = _make_scm(3, names=["X", "Y", "Z"])
        ca, cb = Context(id="a"), Context(id="b")
        return MCCM(
            scms={"a": scm_a, "b": scm_b},
            context_space=[ca, cb],
            shared_variables=["X", "Y", "Z"],
        )

    def test_basic_props(self):
        m = self._two_context_mccm()
        assert m.num_contexts == 2
        assert set(m.context_ids) == {"a", "b"}

    def test_get_scm(self):
        m = self._two_context_mccm()
        scm = m.get_scm("a")
        assert scm.num_variables == 3

    def test_get_scm_missing_raises(self):
        m = self._two_context_mccm()
        with pytest.raises(KeyError):
            m.get_scm("missing")

    def test_add_remove_context(self):
        m = MCCM()
        scm = _make_scm(2, names=["A", "B"])
        ctx = Context(id="new")
        m.add_context(ctx, scm, extra_variables=["extra"])
        assert m.num_contexts == 1
        assert "extra" in m.context_specific_variables["new"]
        removed = m.remove_context("new")
        assert removed.num_variables == 2
        assert m.num_contexts == 0

    def test_add_duplicate_raises(self):
        m = MCCM()
        scm = _make_scm(2, names=["A", "B"])
        m.add_context(Context(id="x"), scm)
        with pytest.raises(ValueError, match="already exists"):
            m.add_context(Context(id="x"), scm)

    def test_remove_missing_raises(self):
        with pytest.raises(KeyError):
            MCCM().remove_context("nope")

    def test_variable_union_intersection(self):
        scm_a = _make_scm(3, names=["X", "Y", "Z"])
        scm_b = _make_scm(2, names=["X", "Y"])
        m = MCCM(
            scms={"a": scm_a, "b": scm_b},
            context_space=[Context(id="a"), Context(id="b")],
        )
        assert m.variable_union() == {"X", "Y", "Z"}
        assert m.variable_intersection() == {"X", "Y"}

    def test_variable_intersection_empty(self):
        assert MCCM().variable_intersection() == set()

    def test_validate_small_sample(self):
        scm = _make_scm(2, names=["A", "B"], sample_size=10)
        m = MCCM(scms={"a": scm}, context_space=[Context(id="a")])
        warnings = m.validate()
        assert any("small sample" in w for w in warnings)

    def test_validate_missing_shared_var(self):
        scm = _make_scm(2, names=["A", "B"])
        m = MCCM(
            scms={"a": scm},
            context_space=[Context(id="a")],
            shared_variables=["A", "B", "C"],
        )
        warnings = m.validate()
        assert any("C" in w for w in warnings)

    def test_mismatch_ids_raises(self):
        scm = _make_scm(2, names=["A", "B"])
        with pytest.raises(ValueError, match="mismatch"):
            MCCM(
                scms={"a": scm},
                context_space=[Context(id="b")],
            )

    def test_serialization_roundtrip(self):
        m = self._two_context_mccm()
        d = m.to_dict()
        m2 = MCCM.from_dict(d)
        assert m2.num_contexts == 2
        assert set(m2.context_ids) == {"a", "b"}
        np.testing.assert_array_equal(
            m2.get_scm("a").adjacency_matrix,
            m.get_scm("a").adjacency_matrix,
        )


# =====================================================================
# AlignmentMapping
# =====================================================================

class TestAlignmentMapping:
    def _make_am(self, **kw):
        defaults = dict(
            pi={"X": "X", "Y": "Y"},
            quality_score=0.9,
            edge_partition=_edge_partition(
                shared={("X", "Y")}, modified={("Y", "Z")}
            ),
            structural_divergence=0.1,
        )
        defaults.update(kw)
        return AlignmentMapping(**defaults)

    def test_basic(self):
        am = self._make_am()
        assert am.num_shared == 1
        assert am.num_modified == 1
        assert am.quality_score == 0.9

    def test_jaccard_index(self):
        am = self._make_am()
        # 1 shared out of 2 total unique edges
        assert am.jaccard_index == pytest.approx(0.5)

    def test_jaccard_no_edges(self):
        am = self._make_am(edge_partition=_edge_partition())
        assert am.jaccard_index == 1.0

    def test_quality_score_out_of_range(self):
        with pytest.raises(ValueError, match="quality_score"):
            self._make_am(quality_score=1.5)

    def test_negative_divergence(self):
        with pytest.raises(ValueError, match="structural_divergence"):
            self._make_am(structural_divergence=-0.1)

    def test_missing_edge_partition_key(self):
        bad_partition = {"shared": set(), "modified": set()}
        with pytest.raises(ValueError, match="edge_partition keys"):
            self._make_am(edge_partition=bad_partition)

    def test_serialization_roundtrip(self):
        am = self._make_am()
        d = am.to_dict()
        am2 = AlignmentMapping.from_dict(d)
        assert am2.pi == am.pi
        assert am2.quality_score == am.quality_score
        assert am2.structural_divergence == am.structural_divergence
        assert am2.num_shared == am.num_shared

    def test_list_coerced_to_set(self):
        ep = {
            "shared": [("X", "Y")],
            "modified": [],
            "context_specific_a": [],
            "context_specific_b": [],
        }
        am = AlignmentMapping(
            pi={}, quality_score=0.5,
            edge_partition=ep,  # type: ignore[arg-type]
            structural_divergence=0.0,
        )
        assert isinstance(am.edge_partition["shared"], set)


# =====================================================================
# PlasticityDescriptor
# =====================================================================

class TestPlasticityDescriptor:
    def test_vector(self):
        pd = _make_descriptor(psi_S=0.1, psi_P=0.2, psi_E=0.3, psi_CS=0.4)
        np.testing.assert_array_almost_equal(pd.vector, [0.1, 0.2, 0.3, 0.4])

    def test_magnitude(self):
        pd = _make_descriptor(psi_S=0.0, psi_P=0.0, psi_E=0.0, psi_CS=1.0)
        assert pd.magnitude == pytest.approx(1.0)

    def test_dominant_dimension(self):
        pd = _make_descriptor(psi_S=0.1, psi_P=0.9, psi_E=0.0, psi_CS=0.0)
        assert pd.dominant_dimension == "psi_P"

    def test_distance_to(self):
        a = _make_descriptor(psi_S=0.0, psi_P=0.0, psi_E=0.0, psi_CS=0.0)
        b = _make_descriptor(psi_S=1.0, psi_P=0.0, psi_E=0.0, psi_CS=0.0)
        assert a.distance_to(b) == pytest.approx(1.0)

    def test_distance_to_self_zero(self):
        pd = _make_descriptor()
        assert pd.distance_to(pd) == pytest.approx(0.0)

    @pytest.mark.parametrize("field", ["psi_S", "psi_P", "psi_E", "psi_CS"])
    def test_out_of_range(self, field):
        with pytest.raises(ValueError, match=field):
            _make_descriptor(**{field: 1.5})

    @pytest.mark.parametrize("field", ["psi_S", "psi_P", "psi_E", "psi_CS"])
    def test_negative_raises(self, field):
        with pytest.raises(ValueError, match=field):
            _make_descriptor(**{field: -0.1})

    def test_boundary_values_ok(self):
        pd = _make_descriptor(psi_S=0.0, psi_P=1.0, psi_E=0.0, psi_CS=1.0)
        assert pd.magnitude == pytest.approx(math.sqrt(2.0))

    def test_bad_classification_type(self):
        with pytest.raises(TypeError, match="PlasticityClass"):
            _make_descriptor(classification="invariant")  # type: ignore

    def test_serialization_roundtrip(self):
        pd = _make_descriptor()
        d = pd.to_dict()
        pd2 = PlasticityDescriptor.from_dict(d)
        np.testing.assert_array_equal(pd2.vector, pd.vector)
        assert pd2.classification == pd.classification
        assert pd2.variable_name == pd.variable_name
        assert pd2.variable_index == pd.variable_index


# =====================================================================
# TippingPoint
# =====================================================================

class TestTippingPoint:
    def test_basic(self):
        tp = _make_tipping_point()
        assert tp.context_location == 3
        assert tp.num_affected == 2
        assert tp.is_significant  # p=0.01 < 0.05

    def test_not_significant(self):
        tp = _make_tipping_point(p_value=0.5)
        assert not tp.is_significant

    def test_has_structural_change(self):
        tp = _make_tipping_point()
        assert tp.has_structural_change

    def test_no_structural_change(self):
        tp = _make_tipping_point(
            affected_mechanisms=["X0"],
            change_types=[ChangeType.PARAMETRIC],
        )
        assert not tp.has_structural_change

    def test_negative_location_raises(self):
        with pytest.raises(ValueError, match="context_location"):
            _make_tipping_point(context_location=-1)

    def test_p_value_out_of_range(self):
        with pytest.raises(ValueError, match="p_value"):
            _make_tipping_point(p_value=2.0)

    def test_negative_effect_size(self):
        with pytest.raises(ValueError, match="effect_size"):
            _make_tipping_point(effect_size=-1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            _make_tipping_point(
                affected_mechanisms=["X0", "X1"],
                change_types=[ChangeType.STRUCTURAL],
            )

    def test_zero_location_ok(self):
        tp = _make_tipping_point(context_location=0)
        assert tp.context_location == 0

    def test_serialization_roundtrip(self):
        tp = _make_tipping_point()
        d = tp.to_dict()
        tp2 = TippingPoint.from_dict(d)
        assert tp2.context_location == tp.context_location
        assert tp2.p_value == tp.p_value
        assert tp2.effect_size == tp.effect_size
        assert tp2.affected_mechanisms == tp.affected_mechanisms
        assert tp2.change_types == tp.change_types
        assert tp2.left_segment == tp.left_segment
        assert tp2.right_segment == tp.right_segment

    def test_empty_mechanisms_ok(self):
        tp = _make_tipping_point(affected_mechanisms=[], change_types=[])
        assert tp.num_affected == 0
        assert not tp.has_structural_change


# =====================================================================
# RobustnessCertificate
# =====================================================================

class TestRobustnessCertificate:
    def test_is_strong(self):
        cert = _make_certificate()
        assert cert.is_strong

    def test_not_strong_if_invalid(self):
        cert = _make_certificate(validity=False)
        assert not cert.is_strong

    def test_not_strong_if_wrong_type(self):
        cert = _make_certificate(type=CertificateType.PARAMETRIC_STABILITY)
        assert not cert.is_strong

    def test_summary_valid(self):
        cert = _make_certificate()
        s = cert.summary
        assert "VALID" in s
        assert "sqrt-JSD" in s

    def test_summary_small_n_warning(self):
        cert = _make_certificate(min_sample_size_warning=True)
        assert "small n" in cert.summary

    def test_negative_max_sqrt_jsd_raises(self):
        with pytest.raises(ValueError, match="max_sqrt_jsd"):
            _make_certificate(max_sqrt_jsd=-0.01)

    def test_negative_ucb_raises(self):
        with pytest.raises(ValueError, match="upper_confidence_bound"):
            _make_certificate(upper_confidence_bound=-1.0)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError, match="CertificateType"):
            _make_certificate(type="strong_invariance")  # type: ignore

    def test_serialization_roundtrip(self):
        cert = _make_certificate()
        d = cert.to_dict()
        cert2 = RobustnessCertificate.from_dict(d)
        assert cert2.type == cert.type
        assert cert2.validity == cert.validity
        assert cert2.max_sqrt_jsd == cert.max_sqrt_jsd
        assert cert2.robustness_margin == cert.robustness_margin
        assert cert2.assumptions == cert.assumptions
        assert cert2.min_sample_size_warning == cert.min_sample_size_warning


# =====================================================================
# QDGenome
# =====================================================================

class TestQDGenome:
    def _make_genome(self, **kw):
        defaults = dict(
            context_subset={"c1", "c2", "c3"},
            mechanism_subset={"X", "Y"},
            analysis_params={"alpha": 0.05, "n_bootstrap": 200},
        )
        defaults.update(kw)
        return QDGenome(**defaults)

    def test_size(self):
        g = self._make_genome()
        assert g.size == 5  # 3 contexts + 2 mechanisms

    def test_coercion_to_set(self):
        g = QDGenome(
            context_subset=["c1", "c2"],  # type: ignore[arg-type]
            mechanism_subset=["X"],  # type: ignore[arg-type]
            analysis_params={},
        )
        assert isinstance(g.context_subset, set)
        assert isinstance(g.mechanism_subset, set)

    def test_bad_params_type(self):
        with pytest.raises(TypeError, match="dict"):
            QDGenome(set(), set(), "not_a_dict")  # type: ignore

    def test_mutate_deterministic_seed(self):
        g = self._make_genome()
        rng = np.random.default_rng(42)
        child = g.mutate({"c1", "c2", "c3", "c4"}, {"X", "Y", "Z"}, rng=rng)
        assert isinstance(child, QDGenome)
        assert len(child.context_subset) >= 2  # minimum 2 enforced

    def test_mutate_preserves_min_contexts(self):
        g = QDGenome({"c1", "c2"}, {"X"}, {"alpha": 0.05})
        rng = np.random.default_rng(0)
        child = g.mutate({"c1", "c2"}, {"X"}, mutation_rate=1.0, rng=rng)
        assert len(child.context_subset) >= 2

    def test_mutate_preserves_min_mechanisms(self):
        g = QDGenome({"c1", "c2"}, {"X"}, {"alpha": 0.05})
        rng = np.random.default_rng(0)
        child = g.mutate({"c1", "c2"}, {"X"}, mutation_rate=1.0, rng=rng)
        assert len(child.mechanism_subset) >= 1

    def test_crossover(self):
        a = self._make_genome(context_subset={"c1", "c2", "c3"})
        b = self._make_genome(context_subset={"c3", "c4", "c5"})
        rng = np.random.default_rng(99)
        child = a.crossover(b, rng=rng)
        assert isinstance(child, QDGenome)
        assert len(child.context_subset) >= 2

    def test_crossover_child_has_params(self):
        a = QDGenome({"c1", "c2"}, {"X"}, {"alpha": 0.05})
        b = QDGenome({"c2", "c3"}, {"Y"}, {"beta": 0.1})
        rng = np.random.default_rng(7)
        child = a.crossover(b, rng=rng)
        # child should have at least one key from the union
        assert len(child.analysis_params) >= 1

    def test_serialization_roundtrip(self):
        g = self._make_genome()
        d = g.to_dict()
        g2 = QDGenome.from_dict(d)
        assert g2.context_subset == g.context_subset
        assert g2.mechanism_subset == g.mechanism_subset
        assert g2.analysis_params == g.analysis_params

    def test_to_dict_sorts(self):
        g = self._make_genome(context_subset={"c3", "c1", "c2"})
        d = g.to_dict()
        assert d["context_subset"] == ["c1", "c2", "c3"]


# =====================================================================
# Cross-cutting serialization & edge cases
# =====================================================================

class TestCrossCutting:
    def test_scm_single_variable(self):
        adj = np.zeros((1, 1))
        scm = SCM(adj, adj, np.ones(1), ["X"], 50)
        assert scm.num_variables == 1
        assert scm.num_edges == 0
        assert scm.topological_sort() == [0]

    def test_context_eq_not_context(self):
        ctx = Context(id="a")
        assert ctx != "a"

    def test_alignment_pi_type_check(self):
        with pytest.raises(TypeError, match="dict"):
            AlignmentMapping(
                pi="bad",  # type: ignore
                quality_score=0.5,
                edge_partition=_edge_partition(),
                structural_divergence=0.0,
            )

    def test_mccm_empty_valid(self):
        m = MCCM()
        assert m.num_contexts == 0
        assert m.variable_union() == set()
        assert m.validate() == []

    def test_scm_weighted_edges(self):
        adj = np.array([[0, 2.5, 0], [0, 0, 0.1], [0, 0, 0]])
        scm = SCM(adj, adj, np.ones(3), ["A", "B", "C"], 100)
        assert scm.num_edges == 2
        assert scm.parents(1) == [0]

    def test_plasticity_descriptor_all_zero(self):
        pd = _make_descriptor(psi_S=0, psi_P=0, psi_E=0, psi_CS=0)
        assert pd.magnitude == pytest.approx(0.0)

    def test_tipping_point_boundary_p_value(self):
        tp = _make_tipping_point(p_value=0.05)
        assert not tp.is_significant  # p < 0.05 is threshold, 0.05 is not significant

    def test_tipping_point_p_value_exactly_zero(self):
        tp = _make_tipping_point(p_value=0.0)
        assert tp.is_significant

    def test_certificate_zero_margin(self):
        cert = _make_certificate(robustness_margin=0.0)
        assert cert.robustness_margin == 0.0

    def test_genome_empty_subsets(self):
        g = QDGenome(set(), set(), {})
        assert g.size == 0
