"""
Unit tests for usability_oracle.algebra.soundness — Soundness verification.

Tests the SoundnessVerifier and VerificationResult classes which verify that
cost compositions satisfy axiomatic algebraic properties:

1. Positivity:        μ ≥ 0, σ² ≥ 0, λ ∈ [0, 1]
2. Monotonicity:      composed μ ≥ max(individual μ)
3. Identity:          a ⊕ 0 = a, a ⊗ 0 = a
4. Variance bound:    σ²_{composed} ≥ max(σ²_a, σ²_b)
5. Triangle inequality: (a ⊕ c).μ ≤ (a ⊕ b).μ + (b ⊕ c).μ
6. Commutativity:     a ⊗ b = b ⊗ a
"""

import math
import pytest

from usability_oracle.algebra.models import (
    CostElement,
    CostExpression,
    Leaf,
    Sequential,
    Parallel,
    ContextMod,
)
from usability_oracle.algebra.sequential import SequentialComposer
from usability_oracle.algebra.parallel import ParallelComposer
from usability_oracle.algebra.soundness import (
    SoundnessVerifier,
    VerificationResult,
    VerificationStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def verifier():
    """Return a fresh SoundnessVerifier instance."""
    return SoundnessVerifier()


@pytest.fixture
def elem_a():
    """A typical cost element."""
    return CostElement(mu=2.0, sigma_sq=0.5, kappa=0.1, lambda_=0.05)


@pytest.fixture
def elem_b():
    """Another typical cost element."""
    return CostElement(mu=1.0, sigma_sq=0.2, kappa=0.05, lambda_=0.03)


@pytest.fixture
def elem_c():
    """A third cost element for triangle inequality tests."""
    return CostElement(mu=1.5, sigma_sq=0.3, kappa=0.0, lambda_=0.02)


@pytest.fixture
def zero():
    """The additive identity element."""
    return CostElement.zero()


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------

class TestVerificationResult:
    """Tests for the VerificationResult data class."""

    def test_pass_result(self):
        """A PASS result has passed == True."""
        r = VerificationResult(
            property_name="test_prop",
            status=VerificationStatus.PASS,
            message="All good.",
        )
        assert r.passed is True
        assert r.property_name == "test_prop"

    def test_fail_result(self):
        """A FAIL result has passed == False."""
        r = VerificationResult(
            property_name="test_prop",
            status=VerificationStatus.FAIL,
            message="Something broke.",
        )
        assert r.passed is False

    def test_warn_result_not_passed(self):
        """A WARN result has passed == False (only PASS counts)."""
        r = VerificationResult(
            property_name="test_prop",
            status=VerificationStatus.WARN,
            message="Marginal.",
        )
        assert r.passed is False

    def test_to_dict(self):
        """to_dict() serialises the result correctly."""
        r = VerificationResult(
            property_name="monotonicity",
            status=VerificationStatus.PASS,
            message="OK",
            details={"mu": 3.0},
        )
        d = r.to_dict()
        assert d["property"] == "monotonicity"
        assert d["status"] == "pass"
        assert d["message"] == "OK"
        assert d["details"]["mu"] == 3.0

    def test_default_details_empty(self):
        """Default details is an empty dict."""
        r = VerificationResult(
            property_name="x",
            status=VerificationStatus.PASS,
            message="ok",
        )
        assert r.details == {}


# ---------------------------------------------------------------------------
# VerificationStatus enum
# ---------------------------------------------------------------------------

class TestVerificationStatus:
    """Tests for the VerificationStatus enum."""

    def test_values(self):
        """Enum has PASS, FAIL, WARN values."""
        assert VerificationStatus.PASS.value == "pass"
        assert VerificationStatus.FAIL.value == "fail"
        assert VerificationStatus.WARN.value == "warn"

    def test_members(self):
        """Enum has exactly 3 members."""
        assert len(VerificationStatus) == 3


# ---------------------------------------------------------------------------
# verify_sequential
# ---------------------------------------------------------------------------

class TestVerifySequential:
    """Tests for verify_sequential — sequential composition axioms."""

    def test_valid_composition_passes(self, verifier, elem_a, elem_b):
        """A correctly composed sequential result passes verification."""
        composed = SequentialComposer().compose(elem_a, elem_b, coupling=0.2)
        assert verifier.verify_sequential(elem_a, elem_b, composed) is True

    def test_valid_with_zero_coupling(self, verifier, elem_a, elem_b):
        """Zero-coupling composition passes verification."""
        composed = SequentialComposer().compose(elem_a, elem_b, coupling=0.0)
        assert verifier.verify_sequential(elem_a, elem_b, composed) is True

    def test_valid_with_full_coupling(self, verifier, elem_a, elem_b):
        """Full-coupling composition passes verification."""
        composed = SequentialComposer().compose(elem_a, elem_b, coupling=1.0)
        assert verifier.verify_sequential(elem_a, elem_b, composed) is True

    def test_invalid_mu_too_small(self, verifier, elem_a, elem_b):
        """A composed result with μ < μ_a + μ_b fails verification."""
        fake_composed = CostElement(mu=0.5, sigma_sq=1.0, kappa=0.0, lambda_=0.05)
        assert verifier.verify_sequential(elem_a, elem_b, fake_composed) is False

    def test_invalid_negative_mu(self, verifier, elem_a, elem_b):
        """A composed result with negative μ fails positivity check."""
        fake = CostElement(mu=-1.0, sigma_sq=0.5, kappa=0.0, lambda_=0.05)
        assert verifier.verify_sequential(elem_a, elem_b, fake) is False

    def test_invalid_lambda_out_of_range(self, verifier, elem_a, elem_b):
        """Composed λ > 1 fails positivity check."""
        fake = CostElement(mu=5.0, sigma_sq=1.0, kappa=0.0, lambda_=1.5)
        assert verifier.verify_sequential(elem_a, elem_b, fake) is False


# ---------------------------------------------------------------------------
# verify_parallel
# ---------------------------------------------------------------------------

class TestVerifyParallel:
    """Tests for verify_parallel — parallel composition axioms."""

    def test_valid_composition_passes(self, verifier, elem_a, elem_b):
        """A correctly composed parallel result passes verification."""
        composed = ParallelComposer().compose(elem_a, elem_b, interference=0.3)
        assert verifier.verify_parallel(elem_a, elem_b, composed) is True

    def test_valid_zero_interference(self, verifier, elem_a, elem_b):
        """Zero-interference composition passes verification."""
        composed = ParallelComposer().compose(elem_a, elem_b, interference=0.0)
        assert verifier.verify_parallel(elem_a, elem_b, composed) is True

    def test_invalid_mu_below_max(self, verifier, elem_a, elem_b):
        """Composed μ < max(μ_a, μ_b) fails monotonicity check."""
        fake = CostElement(mu=0.5, sigma_sq=0.5, kappa=0.0, lambda_=0.05)
        assert verifier.verify_parallel(elem_a, elem_b, fake) is False

    def test_invalid_variance_below_max(self, verifier, elem_a, elem_b):
        """Composed σ² < max(σ²_a, σ²_b) fails variance bound check."""
        fake = CostElement(mu=3.0, sigma_sq=0.01, kappa=0.0, lambda_=0.05)
        assert verifier.verify_parallel(elem_a, elem_b, fake) is False


# ---------------------------------------------------------------------------
# verify_monotonicity
# ---------------------------------------------------------------------------

class TestVerifyMonotonicity:
    """Tests for verify_monotonicity — composed ≥ max individual."""

    def test_valid_sequential_monotonicity(self, verifier, elem_a, elem_b):
        """Sequential composition satisfies monotonicity."""
        composed = SequentialComposer().compose(elem_a, elem_b)
        assert verifier.verify_monotonicity([elem_a, elem_b], composed) is True

    def test_valid_parallel_monotonicity(self, verifier, elem_a, elem_b):
        """Parallel composition satisfies monotonicity."""
        composed = ParallelComposer().compose(elem_a, elem_b, interference=0.5)
        assert verifier.verify_monotonicity([elem_a, elem_b], composed) is True

    def test_empty_elements_trivially_true(self, verifier):
        """Monotonicity with empty element list is trivially true."""
        assert verifier.verify_monotonicity([], CostElement.zero()) is True

    def test_violation_detected(self, verifier, elem_a, elem_b):
        """A composed cost below max individual μ fails monotonicity."""
        fake = CostElement(mu=0.1, sigma_sq=0.5, kappa=0.0, lambda_=0.05)
        assert verifier.verify_monotonicity([elem_a, elem_b], fake) is False

    def test_single_element(self, verifier, elem_a):
        """Monotonicity holds trivially for a single element composed with itself."""
        composed = SequentialComposer().compose(elem_a, CostElement.zero())
        assert verifier.verify_monotonicity([elem_a], composed) is True


# ---------------------------------------------------------------------------
# verify_triangle_inequality
# ---------------------------------------------------------------------------

class TestVerifyTriangleInequality:
    """Tests for verify_triangle_inequality — metric property."""

    def test_triangle_holds(self, verifier, elem_a, elem_b, elem_c):
        """Triangle inequality holds for independent (ρ=0) composition."""
        assert verifier.verify_triangle_inequality(elem_a, elem_b, elem_c) is True

    def test_triangle_with_identical_elements(self, verifier, elem_a):
        """Triangle inequality holds when all three elements are identical."""
        assert verifier.verify_triangle_inequality(elem_a, elem_a, elem_a) is True

    def test_triangle_with_zero(self, verifier, elem_a, elem_b, zero):
        """Triangle inequality holds with zero as the middle element."""
        assert verifier.verify_triangle_inequality(elem_a, zero, elem_b) is True

    def test_triangle_checks_mu_dimension(self, verifier):
        """Triangle inequality specifically checks the μ dimension."""
        a = CostElement(mu=1.0, sigma_sq=0.1, kappa=0.0, lambda_=0.01)
        b = CostElement(mu=2.0, sigma_sq=0.1, kappa=0.0, lambda_=0.01)
        c = CostElement(mu=3.0, sigma_sq=0.1, kappa=0.0, lambda_=0.01)
        # (a ⊕ c).μ = 4.0 ≤ (a ⊕ b).μ + (b ⊕ c).μ = 3.0 + 5.0 = 8.0
        assert verifier.verify_triangle_inequality(a, b, c) is True


# ---------------------------------------------------------------------------
# verify_commutativity
# ---------------------------------------------------------------------------

class TestVerifyCommutativity:
    """Tests for verify_commutativity — parallel a ⊗ b = b ⊗ a."""

    def test_commutativity_holds(self, verifier, elem_a, elem_b):
        """Parallel composition is commutative."""
        assert verifier.verify_commutativity(elem_a, elem_b) is True

    def test_commutativity_with_interference(self, verifier, elem_a, elem_b):
        """Commutativity holds with non-zero interference."""
        assert verifier.verify_commutativity(elem_a, elem_b, interference=0.5) is True

    def test_commutativity_with_equal_elements(self, verifier, elem_a):
        """Commutativity is trivially true for identical elements."""
        assert verifier.verify_commutativity(elem_a, elem_a, interference=0.3) is True

    def test_commutativity_with_zero(self, verifier, elem_a, zero):
        """Commutativity holds when one element is zero."""
        assert verifier.verify_commutativity(elem_a, zero) is True


# ---------------------------------------------------------------------------
# verify_identity
# ---------------------------------------------------------------------------

class TestVerifyIdentity:
    """Tests for verify_identity — a ⊕ 0 = a and a ⊗ 0 = a."""

    def test_identity_holds(self, verifier, elem_a):
        """Identity property holds for a typical element."""
        assert verifier.verify_identity(elem_a) is True

    def test_identity_for_zero(self, verifier, zero):
        """Identity property holds for the zero element itself."""
        assert verifier.verify_identity(zero) is True

    def test_identity_for_degenerate(self, verifier):
        """Identity holds for a degenerate (zero-variance) element."""
        degenerate = CostElement(mu=5.0, sigma_sq=0.0, kappa=0.0, lambda_=0.0)
        assert verifier.verify_identity(degenerate) is True

    def test_identity_for_high_lambda(self, verifier):
        """Identity holds even when λ is large."""
        high_lambda = CostElement(mu=1.0, sigma_sq=0.1, kappa=0.1, lambda_=0.9)
        assert verifier.verify_identity(high_lambda) is True


# ---------------------------------------------------------------------------
# verify_all (expression tree)
# ---------------------------------------------------------------------------

class TestVerifyAll:
    """Tests for verify_all — recursive expression tree verification."""

    def test_leaf_passes(self, verifier, elem_a):
        """A valid leaf expression passes verification."""
        expr = Leaf(elem_a)
        results = verifier.verify_all(expr)
        assert all(r.passed for r in results)

    def test_sequential_expression_passes(self, verifier, elem_a, elem_b):
        """A valid Sequential expression tree passes all checks."""
        expr = Sequential(Leaf(elem_a), Leaf(elem_b), coupling=0.2)
        results = verifier.verify_all(expr)
        assert all(r.passed for r in results)

    def test_parallel_expression_passes(self, verifier, elem_a, elem_b):
        """A valid Parallel expression tree passes all checks."""
        expr = Parallel(Leaf(elem_a), Leaf(elem_b), interference=0.3)
        results = verifier.verify_all(expr)
        assert all(r.passed for r in results)

    def test_nested_expression_passes(self, verifier, elem_a, elem_b, elem_c):
        """A nested expression tree (seq ∘ par) passes all checks."""
        par = Parallel(Leaf(elem_a), Leaf(elem_b), interference=0.2)
        expr = Sequential(par, Leaf(elem_c), coupling=0.1)
        results = verifier.verify_all(expr)
        assert all(r.passed for r in results)

    def test_verify_all_returns_list(self, verifier, elem_a):
        """verify_all returns a list of VerificationResult objects."""
        results = verifier.verify_all(Leaf(elem_a))
        assert isinstance(results, list)
        assert all(isinstance(r, VerificationResult) for r in results)

    def test_sequential_produces_multiple_results(self, verifier, elem_a, elem_b):
        """A Sequential node produces results for leaves + composition checks."""
        expr = Sequential(Leaf(elem_a), Leaf(elem_b))
        results = verifier.verify_all(expr)
        # At minimum: 2 leaf checks + positivity + monotonicity + variance
        assert len(results) >= 5

    def test_parallel_checks_commutativity(self, verifier, elem_a, elem_b):
        """Parallel expression verification includes commutativity check."""
        expr = Parallel(Leaf(elem_a), Leaf(elem_b), interference=0.3)
        results = verifier.verify_all(expr)
        prop_names = [r.property_name for r in results]
        assert "parallel_commutativity" in prop_names

    def test_invalid_leaf_detected(self, verifier):
        """An invalid leaf element is detected by verify_all."""
        bad = CostElement(mu=-1.0, sigma_sq=-0.5, kappa=0.0, lambda_=2.0)
        expr = Leaf(bad)
        results = verifier.verify_all(expr)
        assert any(not r.passed for r in results)

    def test_context_mod_expression(self, verifier, elem_a):
        """A ContextMod expression preserves positivity."""
        expr = ContextMod(Leaf(elem_a), context={"stress_level": 0.8})
        results = verifier.verify_all(expr)
        prop_names = [r.property_name for r in results]
        assert "context_positivity" in prop_names


# ---------------------------------------------------------------------------
# verify_elements (batch)
# ---------------------------------------------------------------------------

class TestVerifyElements:
    """Tests for verify_elements — batch element validation."""

    def test_all_valid(self, verifier, elem_a, elem_b):
        """All valid elements produce PASS results."""
        results = verifier.verify_elements([elem_a, elem_b])
        assert all(r.passed for r in results)
        assert len(results) == 2

    def test_invalid_negative_mu(self, verifier):
        """An element with negative σ² and out-of-range λ is detected as invalid."""
        bad = CostElement(mu=1.0, sigma_sq=-0.1, kappa=0.0, lambda_=1.5)
        results = verifier.verify_elements([bad])
        assert not results[0].passed

    def test_invalid_negative_sigma_sq(self, verifier):
        """An element with negative σ² is detected as invalid."""
        bad = CostElement(mu=1.0, sigma_sq=-0.1, kappa=0.0, lambda_=0.05)
        results = verifier.verify_elements([bad])
        assert not results[0].passed

    def test_invalid_lambda_above_one(self, verifier):
        """An element with λ > 1 is detected as invalid."""
        bad = CostElement(mu=1.0, sigma_sq=0.1, kappa=0.0, lambda_=1.5)
        results = verifier.verify_elements([bad])
        assert not results[0].passed

    def test_mixed_valid_invalid(self, verifier, elem_a):
        """Mixed batch correctly identifies valid and invalid elements."""
        bad = CostElement(mu=1.0, sigma_sq=-0.1, kappa=0.0, lambda_=1.5)
        results = verifier.verify_elements([elem_a, bad])
        assert results[0].passed is True
        assert results[1].passed is False

    def test_empty_list(self, verifier):
        """Empty element list returns empty result list."""
        results = verifier.verify_elements([])
        assert results == []

    def test_result_property_names_indexed(self, verifier, elem_a, elem_b):
        """Result property names are indexed (element_0, element_1, ...)."""
        results = verifier.verify_elements([elem_a, elem_b])
        assert results[0].property_name == "element_0_validity"
        assert results[1].property_name == "element_1_validity"

    def test_inf_mu_detected(self, verifier):
        """An element with infinite μ is detected as invalid."""
        bad = CostElement(mu=float("inf"), sigma_sq=0.1, kappa=0.0, lambda_=0.05)
        results = verifier.verify_elements([bad])
        assert not results[0].passed


# ---------------------------------------------------------------------------
# Soundness violation detection
# ---------------------------------------------------------------------------

class TestSoundnessViolation:
    """Tests that intentionally-broken compositions are caught."""

    def test_fabricated_sequential_violation(self, verifier, elem_a, elem_b):
        """A hand-crafted result violating sequential monotonicity is caught."""
        fake = CostElement(mu=0.01, sigma_sq=0.01, kappa=0.0, lambda_=0.0)
        assert verifier.verify_sequential(elem_a, elem_b, fake) is False

    def test_fabricated_parallel_violation(self, verifier, elem_a, elem_b):
        """A hand-crafted result violating parallel monotonicity is caught."""
        fake = CostElement(mu=0.5, sigma_sq=0.01, kappa=0.0, lambda_=0.0)
        assert verifier.verify_parallel(elem_a, elem_b, fake) is False

    def test_fabricated_monotonicity_violation(self, verifier, elem_a, elem_b):
        """A composed μ below max(individual μ) fails monotonicity."""
        fake = CostElement(mu=0.5, sigma_sq=1.0, kappa=0.0, lambda_=0.05)
        assert verifier.verify_monotonicity([elem_a, elem_b], fake) is False

    def test_expression_with_invalid_leaf_fails(self, verifier):
        """An expression tree containing an invalid leaf fails verify_all."""
        bad = CostElement(mu=-5.0, sigma_sq=-1.0, kappa=0.0, lambda_=3.0)
        good = CostElement(mu=1.0, sigma_sq=0.1, kappa=0.0, lambda_=0.01)
        expr = Sequential(Leaf(bad), Leaf(good))
        results = verifier.verify_all(expr)
        failed = [r for r in results if not r.passed]
        assert len(failed) > 0
