"""
Tests for 5-theory product domain composition soundness.
"""

import pytest
from src.composition_soundness import (
    verify_product_domain_soundness,
    check_signature_disjointness,
    check_stably_infinite_or_finite,
    check_quantifier_free,
    check_convexity,
    check_cross_theory_deduction_soundness,
    compute_arrangement_complexity,
    verify_composition_properties_z3,
    TheorySignature,
    ALL_THEORIES,
    T_SHAPE,
    T_DEVICE,
    T_PHASE,
    T_STRIDE,
    T_PERM,
    _stirling_second,
    _bell_bounded,
)


class TestSignatureDisjointness:
    def test_all_theories_disjoint(self):
        result = check_signature_disjointness(ALL_THEORIES)
        assert result.satisfied, result.details

    def test_empty_theories(self):
        result = check_signature_disjointness([])
        assert result.satisfied

    def test_single_theory(self):
        result = check_signature_disjointness([T_SHAPE])
        assert result.satisfied

    def test_overlapping_signatures_detected(self):
        t1 = TheorySignature(
            name="t1",
            sorts=frozenset({"S"}),
            functions=frozenset({"f"}),
            predicates=frozenset(),
            domain_kind="stably_infinite",
        )
        t2 = TheorySignature(
            name="t2",
            sorts=frozenset({"S"}),
            functions=frozenset({"f"}),
            predicates=frozenset(),
            domain_kind="stably_infinite",
        )
        result = check_signature_disjointness([t1, t2])
        assert not result.satisfied
        assert "f" in result.details


class TestDomainCharacterization:
    def test_all_theories_characterized(self):
        result = check_stably_infinite_or_finite(ALL_THEORIES)
        assert result.satisfied, result.details

    def test_shape_is_stably_infinite(self):
        assert T_SHAPE.domain_kind == "stably_infinite"

    def test_stride_is_stably_infinite(self):
        assert T_STRIDE.domain_kind == "stably_infinite"

    def test_device_is_finite(self):
        assert T_DEVICE.domain_kind == "finite"
        assert T_DEVICE.domain_size == 5

    def test_phase_is_finite(self):
        assert T_PHASE.domain_kind == "finite"
        assert T_PHASE.domain_size == 2

    def test_perm_is_finite(self):
        assert T_PERM.domain_kind == "finite"
        assert T_PERM.domain_size == 24

    def test_unknown_domain_fails(self):
        bad_theory = TheorySignature(
            name="bad",
            sorts=frozenset({"X"}),
            functions=frozenset(),
            predicates=frozenset(),
            domain_kind="unknown",
        )
        result = check_stably_infinite_or_finite([bad_theory])
        assert not result.satisfied

    def test_finite_without_size_fails(self):
        bad_theory = TheorySignature(
            name="bad",
            sorts=frozenset({"X"}),
            functions=frozenset(),
            predicates=frozenset(),
            domain_kind="finite",
            domain_size=None,
        )
        result = check_stably_infinite_or_finite([bad_theory])
        assert not result.satisfied


class TestQuantifierFree:
    def test_all_qf(self):
        result = check_quantifier_free(ALL_THEORIES)
        assert result.satisfied


class TestConvexity:
    def test_all_convex(self):
        result = check_convexity(ALL_THEORIES)
        assert result.satisfied, result.details


class TestCrossTheoryDeduction:
    def test_deduction_sound(self):
        result = check_cross_theory_deduction_soundness()
        assert result.satisfied

    def test_z3_machine_check_included(self):
        """Z3 model-extraction lemma is machine-checked when Z3 available."""
        result = check_cross_theory_deduction_soundness()
        assert "machine-check" in result.details or "model-extraction" in result.details

    def test_z3_all_instances_verified(self):
        """All model-extraction instances pass."""
        result = check_cross_theory_deduction_soundness()
        if "machine-check" in result.details:
            # Parse "3/3 model-extraction lemma instances verified"
            import re
            m = re.search(r"(\d+)/(\d+) model-extraction", result.details)
            assert m is not None
            assert m.group(1) == m.group(2)  # all passed


class TestStirlingNumbers:
    def test_stirling_base_cases(self):
        assert _stirling_second(0, 0) == 1
        assert _stirling_second(1, 0) == 0
        assert _stirling_second(0, 1) == 0
        assert _stirling_second(1, 1) == 1

    def test_stirling_known_values(self):
        # S(3, 2) = 3 (ways to partition {1,2,3} into 2 non-empty subsets)
        assert _stirling_second(3, 2) == 3
        # S(4, 2) = 7
        assert _stirling_second(4, 2) == 7
        # S(4, 3) = 6
        assert _stirling_second(4, 3) == 6

    def test_bell_bounded(self):
        # B(3, 3) = S(3,1) + S(3,2) + S(3,3) = 1 + 3 + 1 = 5
        assert _bell_bounded(3, 3) == 5
        # B(2, 2) = S(2,1) + S(2,2) = 1 + 1 = 2
        assert _bell_bounded(2, 2) == 2


class TestArrangementComplexity:
    def test_default_complexity(self):
        result = compute_arrangement_complexity(ALL_THEORIES)
        assert result["total_arrangements"] > 0
        assert result["tractable"]

    def test_custom_shared_vars(self):
        result = compute_arrangement_complexity(
            ALL_THEORIES,
            {"device": 2, "phase": 1, "permutation": 2},
        )
        assert result["total_arrangements"] > 0
        assert result["tractable"]

    def test_no_finite_theories(self):
        inf_only = [T_SHAPE, T_STRIDE]
        result = compute_arrangement_complexity(inf_only)
        assert result["total_arrangements"] == 1  # product of empty = 1


class TestFullSoundnessVerdict:
    def test_five_theory_sound(self):
        verdict = verify_product_domain_soundness()
        assert verdict.sound, (
            "Product domain soundness verification failed: "
            + "; ".join(p.details for p in verdict.preconditions if not p.satisfied)
        )

    def test_combination_method_is_hybrid(self):
        verdict = verify_product_domain_soundness()
        assert verdict.combination_method == "tinelli_zarba_hybrid"

    def test_proof_sketch_nonempty(self):
        verdict = verify_product_domain_soundness()
        assert len(verdict.proof_sketch) > 100
        assert "PROOF" in verdict.proof_sketch
        assert "Tinelli-Zarba" in verdict.proof_sketch

    def test_complexity_bound_nonempty(self):
        verdict = verify_product_domain_soundness()
        assert len(verdict.complexity_bound) > 0

    def test_all_preconditions_checked(self):
        verdict = verify_product_domain_soundness()
        names = {p.name for p in verdict.preconditions}
        assert "signature_disjointness" in names
        assert "domain_characterization" in names
        assert "quantifier_free" in names
        assert "convexity" in names
        assert "cross_theory_deduction_soundness" in names


class TestZ3CompositionProperties:
    def test_z3_verification(self):
        try:
            import z3
        except ImportError:
            pytest.skip("Z3 not available")

        results = verify_composition_properties_z3()
        assert "error" not in results

        # Device sort should be categorical
        assert results["device_categoricity"]["is_categorical"]

        # Phase sort should be categorical
        assert results["phase_categoricity"]["is_categorical"]

        # Equality transitivity should hold (UNSAT = no counterexample)
        assert results["equality_transitivity"]["sound"]

        # Arrangement completeness should hold
        assert results["arrangement_completeness_2elem"]["complete"]
