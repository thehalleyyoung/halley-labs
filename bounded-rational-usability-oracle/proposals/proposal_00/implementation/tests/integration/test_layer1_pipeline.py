"""Integration tests for the Layer-1 (additive, parameter-free) pipeline.

Layer 1 uses the ``SequentialComposer`` to combine ``CostElement`` values
representing individual task steps.  No MDP is needed — the comparison is
a direct algebraic composition of cognitive costs followed by interval-based
comparison.  All tests verify parameter-free reasoning via interval arithmetic.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from usability_oracle.algebra.models import CostElement, Leaf, Sequential
from usability_oracle.algebra.sequential import SequentialComposer
from usability_oracle.algebra.composer import TaskGraphComposer
from usability_oracle.algebra.soundness import SoundnessVerifier, VerificationResult
from usability_oracle.interval import Interval as IvInterval
from usability_oracle.core.enums import RegressionVerdict
from usability_oracle.taskspec.models import TaskStep, TaskFlow, TaskSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cost(mu: float, sigma_sq: float = 0.0, kappa: float = 0.0,
           lambda_: float = 0.0) -> CostElement:
    """Shorthand for ``CostElement`` construction."""
    return CostElement(mu=mu, sigma_sq=sigma_sq, kappa=kappa, lambda_=lambda_)


def _make_step_costs(n: int = 3, base_mu: float = 1.0) -> List[CostElement]:
    """Create *n* cost elements with increasing mean cost."""
    return [_cost(mu=base_mu * (i + 1), sigma_sq=0.1 * (i + 1))
            for i in range(n)]


def _make_simple_task(n_steps: int = 3) -> TaskSpec:
    """Create a linear task spec with *n_steps* steps."""
    steps = [
        TaskStep(
            step_id=f"s{i}",
            action_type="click",
            target_role="button",
            target_name=f"Btn{i}",
            description=f"Step {i}",
            depends_on=[f"s{i-1}"] if i > 0 else [],
        )
        for i in range(n_steps)
    ]
    flow = TaskFlow(flow_id="f1", name="Main", steps=steps)
    return TaskSpec(spec_id="t1", name="Test Task", flows=[flow])


# ===================================================================
# Tests – CostElement basics
# ===================================================================


class TestCostElementAlgebra:
    """Verify cost element arithmetic and properties."""

    def test_addition_mu(self) -> None:
        """``(a + b).mu == a.mu + b.mu``."""
        a = _cost(1.0)
        b = _cost(2.0)
        assert (a + b).mu == pytest.approx(3.0)

    def test_addition_sigma_sq(self) -> None:
        """Variance adds under independence assumption."""
        a = _cost(1.0, sigma_sq=0.5)
        b = _cost(2.0, sigma_sq=0.3)
        assert (a + b).sigma_sq == pytest.approx(0.8)

    def test_scalar_multiplication(self) -> None:
        """``(2 * a).mu == 2 * a.mu``."""
        a = _cost(3.0, sigma_sq=1.0)
        result = 2 * a
        assert result.mu == pytest.approx(6.0)

    def test_zero_element(self) -> None:
        """``CostElement.zero()`` must be the additive identity."""
        z = CostElement.zero()
        a = _cost(5.0, sigma_sq=1.0)
        assert (a + z).mu == pytest.approx(a.mu)
        assert (a + z).sigma_sq == pytest.approx(a.sigma_sq)

    def test_negation(self) -> None:
        """``(-a).mu == -a.mu``."""
        a = _cost(3.0)
        assert (-a).mu == pytest.approx(-3.0)

    def test_subtraction(self) -> None:
        """``(a - b).mu == a.mu - b.mu``."""
        a = _cost(5.0)
        b = _cost(2.0)
        assert (a - b).mu == pytest.approx(3.0)

    def test_to_interval(self) -> None:
        """``to_interval`` should produce a sensible confidence interval."""
        a = _cost(10.0, sigma_sq=4.0)
        lo, hi = a.to_interval(confidence=0.95)
        assert lo < a.mu < hi

    def test_is_valid(self) -> None:
        """A standard cost element should be valid."""
        a = _cost(5.0, sigma_sq=1.0)
        assert a.is_valid

    def test_degenerate_detection(self) -> None:
        """Zero-variance elements are degenerate."""
        a = _cost(5.0, sigma_sq=0.0)
        assert a.is_degenerate


# ===================================================================
# Tests – SequentialComposer
# ===================================================================


class TestSequentialComposer:
    """Additive cost composition via ``SequentialComposer``."""

    def test_compose_two_elements(self) -> None:
        """Composing two elements must yield their sum."""
        a, b = _cost(1.0, sigma_sq=0.1), _cost(2.0, sigma_sq=0.2)
        composed = SequentialComposer().compose(a, b)
        assert composed.mu == pytest.approx(3.0)

    def test_compose_chain(self) -> None:
        """``compose_chain`` must be equivalent to pairwise composition."""
        elems = _make_step_costs(4)
        composed = SequentialComposer().compose_chain(elems)
        expected_mu = sum(e.mu for e in elems)
        assert composed.mu == pytest.approx(expected_mu)

    def test_compose_with_coupling(self) -> None:
        """Non-zero coupling should affect the composed variance."""
        a = _cost(1.0, sigma_sq=0.5)
        b = _cost(1.0, sigma_sq=0.5)
        no_coupling = SequentialComposer().compose(a, b, coupling=0.0)
        with_coupling = SequentialComposer().compose(a, b, coupling=0.5)
        assert with_coupling.sigma_sq != no_coupling.sigma_sq

    def test_compose_interval(self) -> None:
        """``compose_interval`` must return a pair of cost elements."""
        a = _cost(1.0, sigma_sq=0.1)
        b = _cost(2.0, sigma_sq=0.2)
        lo_elem, hi_elem = SequentialComposer().compose_interval(
            a, b, coupling_interval=(0.0, 0.5),
        )
        assert lo_elem.mu <= hi_elem.mu or lo_elem.mu == pytest.approx(hi_elem.mu)

    def test_sensitivity(self) -> None:
        """``sensitivity`` should return a non-empty dict."""
        a = _cost(1.0, sigma_sq=0.1)
        b = _cost(2.0, sigma_sq=0.2)
        sens = SequentialComposer().sensitivity(a, b)
        assert isinstance(sens, dict)
        assert len(sens) > 0

    def test_compose_single_element_chain(self) -> None:
        """Composing a single-element chain should return that element."""
        a = _cost(5.0, sigma_sq=1.0)
        composed = SequentialComposer().compose_chain([a])
        assert composed.mu == pytest.approx(a.mu)


# ===================================================================
# Tests – Layer 1 cost comparison
# ===================================================================


class TestLayer1Comparison:
    """Direct cost comparison without MDP construction."""

    def test_identical_costs_neutral(self) -> None:
        """Identical before/after costs should indicate no regression."""
        steps_before = _make_step_costs(3, base_mu=2.0)
        steps_after = _make_step_costs(3, base_mu=2.0)
        cost_before = SequentialComposer().compose_chain(steps_before)
        cost_after = SequentialComposer().compose_chain(steps_after)
        delta = cost_after - cost_before
        assert abs(delta.mu) < 1e-8

    def test_increased_cost_detected(self) -> None:
        """Higher after-cost should produce a positive delta."""
        steps_before = _make_step_costs(3, base_mu=1.0)
        steps_after = _make_step_costs(3, base_mu=2.0)
        cost_before = SequentialComposer().compose_chain(steps_before)
        cost_after = SequentialComposer().compose_chain(steps_after)
        delta = cost_after - cost_before
        assert delta.mu > 0

    def test_decreased_cost_detected(self) -> None:
        """Lower after-cost should produce a negative delta."""
        steps_before = _make_step_costs(3, base_mu=3.0)
        steps_after = _make_step_costs(3, base_mu=1.0)
        cost_before = SequentialComposer().compose_chain(steps_before)
        cost_after = SequentialComposer().compose_chain(steps_after)
        delta = cost_after - cost_before
        assert delta.mu < 0

    def test_interval_comparison(self) -> None:
        """Interval-based comparison should be sound for disjoint intervals."""
        before = _cost(10.0, sigma_sq=1.0)
        after = _cost(15.0, sigma_sq=1.0)
        lo_b, hi_b = before.to_interval(0.95)
        lo_a, hi_a = after.to_interval(0.95)
        # after is strictly higher, so lower bound of delta > 0
        assert lo_a > hi_b or lo_a > lo_b


# ===================================================================
# Tests – SoundnessVerifier
# ===================================================================


class TestSoundnessVerification:
    """Algebraic soundness checks on composed costs."""

    def test_verify_sequential(self) -> None:
        """Sequential composition must satisfy the verifier."""
        a = _cost(1.0, sigma_sq=0.1)
        b = _cost(2.0, sigma_sq=0.2)
        composed = SequentialComposer().compose(a, b)
        assert SoundnessVerifier().verify_sequential(a, b, composed)

    def test_verify_identity(self) -> None:
        """The zero element must be the additive identity."""
        z = CostElement.zero()
        assert SoundnessVerifier().verify_identity(z)

    def test_verify_monotonicity(self) -> None:
        """Composed cost ≥ max component cost (monotonicity)."""
        elems = _make_step_costs(4)
        composed = SequentialComposer().compose_chain(elems)
        assert SoundnessVerifier().verify_monotonicity(elems, composed)

    def test_verify_all_on_expression(self) -> None:
        """``verify_all`` should return a list of VerificationResult."""
        a = _cost(1.0, sigma_sq=0.1)
        b = _cost(2.0, sigma_sq=0.2)
        expr = Sequential(left=Leaf(a), right=Leaf(b), coupling=0.0)
        results = SoundnessVerifier().verify_all(expr)
        assert all(isinstance(r, VerificationResult) for r in results)

    def test_verify_elements(self) -> None:
        """``verify_elements`` on valid costs should return results."""
        elems = _make_step_costs(3)
        results = SoundnessVerifier().verify_elements(elems)
        assert isinstance(results, list)
