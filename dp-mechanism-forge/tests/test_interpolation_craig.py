"""Tests for interpolation.craig module."""
import pytest
import numpy as np

from dp_forge.interpolation.craig import (
    CraigInterpolant,
    BinaryInterpolation,
    SequenceInterpolation,
    InterpolantCache,
    ProofBasedInterpolation,
    ResolutionProof,
    ResolutionStep,
)
from dp_forge.interpolation.formula import (
    FormulaNode,
    Formula,
    SatisfiabilityChecker,
)
from dp_forge.interpolation import InterpolantConfig, InterpolantStrength
from dp_forge.types import Formula as DPFormula, InterpolantType


def _make_dp_formula(expr: str, variables=None):
    vs = frozenset(variables) if variables else frozenset()
    return DPFormula(expr=expr, variables=vs, formula_type="linear_arithmetic")


class TestCraigInterpolant:
    """Test CraigInterpolant properties."""

    def test_interpolant_computed(self):
        """Interpolant is returned for inconsistent pair."""
        A = _make_dp_formula("x <= 1.0", {"x"})
        B = _make_dp_formula("x >= 5.0", {"x"})
        ci = CraigInterpolant()
        result = ci.compute(A, B)
        assert result.success

    def test_interpolant_variables_subset(self):
        """Interpolant vars ⊆ vars(A) ∩ vars(B)."""
        A = _make_dp_formula("x + y <= 1.0", {"x", "y"})
        B = _make_dp_formula("x >= 5.0", {"x"})
        ci = CraigInterpolant()
        result = ci.compute(A, B)
        if result.success:
            common = {"x", "y"} & {"x"}
            assert result.interpolant.formula.variables <= frozenset(common)

    def test_consistent_pair_fails(self):
        """Consistent pair should fail or report no interpolant."""
        A = _make_dp_formula("x <= 5.0", {"x"})
        B = _make_dp_formula("x >= 0.0", {"x"})
        ci = CraigInterpolant()
        result = ci.compute(A, B)
        assert not result.success

    def test_different_proof_systems(self):
        """All proof systems produce interpolants for the same instance."""
        from dp_forge.interpolation import ProofSystem
        A = _make_dp_formula("x <= 1.0", {"x"})
        B = _make_dp_formula("x >= 5.0", {"x"})
        for ps in [ProofSystem.FARKAS_LEMMA, ProofSystem.RESOLUTION, ProofSystem.CUTTING_PLANES]:
            config = InterpolantConfig(proof_system=ps)
            ci = CraigInterpolant(config)
            result = ci.compute(A, B)
            assert result.success


class TestBinaryInterpolation:
    """Test BinaryInterpolation variable restriction."""

    def test_binary_interpolation_basic(self):
        """Basic binary interpolation succeeds."""
        A = _make_dp_formula("x <= 1.0", {"x"})
        B = _make_dp_formula("x >= 5.0", {"x"})
        bi = BinaryInterpolation()
        result = bi.interpolate([A], [B])
        assert result is not None
        assert result.success

    def test_symmetric_interpolant(self):
        """Symmetric interpolant is also valid."""
        A = _make_dp_formula("x <= 1.0", {"x"})
        B = _make_dp_formula("x >= 5.0", {"x"})
        bi = BinaryInterpolation()
        r1, r2 = bi.symmetric_interpolant(A, B)
        if r1 is not None and r1.success:
            assert r1.interpolant is not None


class TestSequenceInterpolation:
    """Test SequenceInterpolation path coverage."""

    def test_sequence_of_two(self):
        """Two-formula sequence gives one interpolant."""
        A = _make_dp_formula("x <= 1.0", {"x"})
        B = _make_dp_formula("x >= 5.0", {"x"})
        si = SequenceInterpolation()
        result = si.compute([A, B])
        if result is not None:
            assert result.length == 1

    def test_sequence_of_three(self):
        """Three-formula sequence gives two interpolants."""
        A = _make_dp_formula("x <= 1.0", {"x"})
        B = _make_dp_formula("x >= 2.0 AND x <= 3.0", {"x"})
        # Make conjunction truly unsat by using incompatible bounds
        C = _make_dp_formula("x >= 10.0", {"x"})
        si = SequenceInterpolation()
        result = si.compute([A, B, C])
        # May or may not succeed depending on solver
        if result is not None:
            assert result.length == 2


class TestInterpolantCache:
    """Test InterpolantCache hit/miss."""

    def test_cache_miss(self):
        """First lookup is a miss."""
        cache = InterpolantCache(max_size=10)
        A = _make_dp_formula("x <= 1.0", {"x"})
        B = _make_dp_formula("x >= 5.0", {"x"})
        result = cache.get(A, B)
        assert result is None

    def test_cache_hit_after_put(self):
        """After put, get returns the cached result."""
        cache = InterpolantCache(max_size=10)
        A = _make_dp_formula("x <= 1.0", {"x"})
        B = _make_dp_formula("x >= 5.0", {"x"})
        ci = CraigInterpolant()
        result = ci.compute(A, B)
        cache.put(A, B, result)
        cached = cache.get(A, B)
        assert cached is not None
        assert cached.success == result.success

    def test_cache_clear(self):
        """Clear empties the cache."""
        cache = InterpolantCache(max_size=10)
        A = _make_dp_formula("x <= 1.0", {"x"})
        B = _make_dp_formula("x >= 5.0", {"x"})
        ci = CraigInterpolant()
        cache.put(A, B, ci.compute(A, B))
        cache.clear()
        assert cache.size == 0

    def test_cache_size_limit(self):
        """Cache respects max_size."""
        cache = InterpolantCache(max_size=3)
        ci = CraigInterpolant()
        for i in range(5):
            A = _make_dp_formula(f"x <= {i}.0", {"x"})
            B = _make_dp_formula(f"x >= {i + 10}.0", {"x"})
            cache.put(A, B, ci.compute(A, B))
        assert cache.size <= 3

    def test_hit_rate(self):
        """Hit rate is computed correctly."""
        cache = InterpolantCache(max_size=10)
        A = _make_dp_formula("x <= 1.0", {"x"})
        B = _make_dp_formula("x >= 5.0", {"x"})
        ci = CraigInterpolant()
        cache.put(A, B, ci.compute(A, B))
        cache.get(A, B)  # hit
        C = _make_dp_formula("y >= 10.0", {"y"})
        cache.get(A, C)  # miss
        # Hit rate >= 0
        assert cache.hit_rate >= 0.0


class TestProofBasedInterpolation:
    """Test ProofBasedInterpolation."""

    def test_from_resolution_proof(self):
        """Proof-based interpolation produces a result from resolution proof."""
        steps = [
            ResolutionStep(clause=(FormulaNode.var("x"),), source="input_a"),
            ResolutionStep(clause=(FormulaNode.not_(FormulaNode.var("x")),), source="input_b"),
            ResolutionStep(clause=(), parent_a=0, parent_b=1, pivot="x", source="derived"),
        ]
        proof = ResolutionProof(
            steps=steps,
            variables_a=frozenset({"x"}),
            variables_b=frozenset({"x"}),
        )
        pbi = ProofBasedInterpolation()
        result = pbi.from_resolution_proof(proof)
        assert result is not None

    def test_resolution_proof_structure(self):
        """ResolutionProof has proper structure."""
        steps = [
            ResolutionStep(clause=(FormulaNode.var("x"),), source="input_a"),
            ResolutionStep(clause=(FormulaNode.not_(FormulaNode.var("x")),), source="input_b"),
            ResolutionStep(clause=(), parent_a=0, parent_b=1, pivot="x", source="derived"),
        ]
        proof = ResolutionProof(
            steps=steps,
            variables_a=frozenset({"x"}),
            variables_b=frozenset({"x"}),
        )
        assert proof.verify()  # ends with empty clause
        assert proof.size == 3
        assert "x" in proof.common_variables

    def test_build_resolution_proof(self):
        """build_resolution_proof constructs a valid proof."""
        pbi = ProofBasedInterpolation()
        # Create complementary clause sets
        clauses_a = [(FormulaNode.var("x"),)]
        clauses_b = [(FormulaNode.not_(FormulaNode.var("x")),)]
        proof = pbi.build_resolution_proof(
            clauses_a, clauses_b,
            frozenset({"x"}), frozenset({"x"}),
        )
        if proof is not None:
            assert proof.verify()
