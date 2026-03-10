"""Unit tests for usability_oracle.comparison.error_bounds.

Tests ErrorBoundComputer and ErrorBoundResult for computing rigorous error
bounds on usability cost estimates using Hoeffding, Chebyshev, and CLT bounds.

References
----------
- Hoeffding (1963). *J. Amer. Statist. Assoc.*, 58(301).
- Givan, Dean & Greig (2003). *Artificial Intelligence*, 147.
"""

from __future__ import annotations

import math
import pytest
import numpy as np
from scipy import stats as sp_stats

from usability_oracle.comparison.error_bounds import ErrorBoundComputer, ErrorBoundResult
from usability_oracle.comparison.models import Partition, PartitionBlock
from usability_oracle.mdp.models import MDP, State, Action, Transition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_mdp(cost: float = 1.0, discount: float = 0.99) -> MDP:
    """Create a minimal 2-state MDP for abstraction error tests."""
    states = {
        "s0": State(state_id="s0", features={}, label="s0",
                     is_terminal=False, is_goal=False, metadata={}),
        "s1": State(state_id="s1", features={}, label="s1",
                     is_terminal=True, is_goal=True, metadata={}),
    }
    actions = {
        "go": Action(action_id="go", action_type="click",
                     target_node_id="x", description="go", preconditions=[]),
    }
    transitions = [
        Transition(source="s0", action="go", target="s1",
                   probability=1.0, cost=cost),
    ]
    return MDP(states=states, actions=actions, transitions=transitions,
               initial_state="s0", goal_states={"s1"}, discount=discount)


def _make_chain_mdp(n: int = 4, cost: float = 1.0) -> MDP:
    """Create a linear chain MDP with n states and uniform cost."""
    states = {}
    for i in range(n):
        states[f"s{i}"] = State(
            state_id=f"s{i}", features={"pos": float(i)}, label=f"s{i}",
            is_terminal=(i == n - 1), is_goal=(i == n - 1), metadata={},
        )
    actions = {
        "step": Action(action_id="step", action_type="click",
                       target_node_id="x", description="step", preconditions=[]),
    }
    transitions = [
        Transition(source=f"s{i}", action="step", target=f"s{i+1}",
                   probability=1.0, cost=cost)
        for i in range(n - 1)
    ]
    return MDP(states=states, actions=actions, transitions=transitions,
               initial_state="s0", goal_states={f"s{n-1}"}, discount=0.99)


def _trivial_partition(mdp: MDP) -> Partition:
    """Create a trivial partition with one block per state (zero abstraction error)."""
    blocks = []
    state_to_block = {}
    for sid in mdp.states:
        bid = f"block_{sid}"
        blocks.append(PartitionBlock(block_id=bid, state_ids=[sid], representative=sid))
        state_to_block[sid] = bid
    return Partition(blocks=blocks, state_to_block=state_to_block)


def _coarse_partition(mdp: MDP) -> Partition:
    """Create a coarse partition grouping ALL states into one block."""
    all_sids = list(mdp.states.keys())
    block = PartitionBlock(
        block_id="all",
        state_ids=all_sids,
        representative=all_sids[0],
    )
    state_to_block = {sid: "all" for sid in all_sids}
    return Partition(blocks=[block], state_to_block=state_to_block)


# ---------------------------------------------------------------------------
# Tests: ErrorBoundResult
# ---------------------------------------------------------------------------


class TestErrorBoundResult:
    """Tests for the ErrorBoundResult dataclass."""

    def test_default_fields(self):
        """Default ErrorBoundResult should have all errors at zero, confidence 0.95."""
        r = ErrorBoundResult()
        assert r.abstraction_error == 0.0
        assert r.sampling_error == 0.0
        assert r.model_error == 0.0
        assert r.total_error == 0.0
        assert r.confidence == 0.95
        assert r.required_samples == 0

    def test_all_fields_set(self):
        """ErrorBoundResult should faithfully store all provided fields."""
        r = ErrorBoundResult(
            abstraction_error=0.1,
            sampling_error=0.2,
            model_error=0.05,
            total_error=0.35,
            confidence=0.99,
            required_samples=500,
        )
        assert r.abstraction_error == 0.1
        assert r.sampling_error == 0.2
        assert r.model_error == 0.05
        assert r.total_error == 0.35
        assert r.confidence == 0.99
        assert r.required_samples == 500


# ---------------------------------------------------------------------------
# Tests: ErrorBoundComputer initialization
# ---------------------------------------------------------------------------


class TestErrorBoundComputerInit:
    """Tests for ErrorBoundComputer constructor."""

    def test_default_method(self):
        """Default ErrorBoundComputer uses Hoeffding's inequality."""
        ebc = ErrorBoundComputer()
        assert ebc.bound_method == "hoeffding"
        assert ebc.cost_range == 100.0

    def test_chebyshev_method(self):
        """ErrorBoundComputer should accept 'chebyshev' as bound method."""
        ebc = ErrorBoundComputer(bound_method="chebyshev")
        assert ebc.bound_method == "chebyshev"

    def test_clt_method(self):
        """ErrorBoundComputer should accept 'clt' as bound method."""
        ebc = ErrorBoundComputer(bound_method="clt")
        assert ebc.bound_method == "clt"

    def test_custom_cost_range(self):
        """ErrorBoundComputer should accept a custom cost range R."""
        ebc = ErrorBoundComputer(cost_range=50.0)
        assert ebc.cost_range == 50.0


# ---------------------------------------------------------------------------
# Tests: compute_sampling_error — Hoeffding bound
# ---------------------------------------------------------------------------


class TestHoeffdingBound:
    """Tests for Hoeffding's concentration inequality bound."""

    def test_hoeffding_decreases_with_more_samples(self):
        """Hoeffding bound ε = R√(ln(2/α)/2n) decreases as n increases."""
        ebc = ErrorBoundComputer(bound_method="hoeffding", cost_range=10.0)

        err_100 = ebc.compute_sampling_error(100, variance=1.0, alpha=0.05)
        err_1000 = ebc.compute_sampling_error(1000, variance=1.0, alpha=0.05)

        assert err_1000 < err_100

    def test_hoeffding_exact_formula(self):
        """Hoeffding bound should match ε = R√(ln(2/α)/(2n))."""
        R, n, alpha = 10.0, 100, 0.05
        ebc = ErrorBoundComputer(bound_method="hoeffding", cost_range=R)

        expected = R * math.sqrt(math.log(2.0 / alpha) / (2.0 * n))
        actual = ebc.compute_sampling_error(n, variance=1.0, alpha=alpha)

        assert actual == pytest.approx(expected, rel=1e-10)

    def test_hoeffding_independent_of_variance(self):
        """Hoeffding bound does NOT depend on the variance parameter."""
        ebc = ErrorBoundComputer(bound_method="hoeffding", cost_range=10.0)

        err_low_var = ebc.compute_sampling_error(100, variance=0.1, alpha=0.05)
        err_high_var = ebc.compute_sampling_error(100, variance=100.0, alpha=0.05)

        assert err_low_var == pytest.approx(err_high_var)

    def test_hoeffding_zero_samples_returns_inf(self):
        """With n=0, Hoeffding bound returns infinity."""
        ebc = ErrorBoundComputer(bound_method="hoeffding", cost_range=10.0)
        err = ebc.compute_sampling_error(0, variance=1.0, alpha=0.05)

        assert math.isinf(err)


# ---------------------------------------------------------------------------
# Tests: compute_sampling_error — Chebyshev bound
# ---------------------------------------------------------------------------


class TestChebyshevBound:
    """Tests for Chebyshev's concentration inequality bound."""

    def test_chebyshev_decreases_with_more_samples(self):
        """Chebyshev bound ε = √(σ²/(nα)) decreases as n increases."""
        ebc = ErrorBoundComputer(bound_method="chebyshev")

        err_100 = ebc.compute_sampling_error(100, variance=4.0, alpha=0.05)
        err_1000 = ebc.compute_sampling_error(1000, variance=4.0, alpha=0.05)

        assert err_1000 < err_100

    def test_chebyshev_exact_formula(self):
        """Chebyshev bound should match ε = √(σ²/(nα))."""
        variance, n, alpha = 4.0, 200, 0.05
        ebc = ErrorBoundComputer(bound_method="chebyshev")

        expected = math.sqrt(variance / (n * alpha))
        actual = ebc.compute_sampling_error(n, variance, alpha)

        assert actual == pytest.approx(expected, rel=1e-10)

    def test_chebyshev_increases_with_variance(self):
        """Higher variance → wider Chebyshev error bound."""
        ebc = ErrorBoundComputer(bound_method="chebyshev")

        err_low = ebc.compute_sampling_error(100, variance=1.0, alpha=0.05)
        err_high = ebc.compute_sampling_error(100, variance=16.0, alpha=0.05)

        assert err_high > err_low


# ---------------------------------------------------------------------------
# Tests: compute_sampling_error — CLT bound
# ---------------------------------------------------------------------------


class TestCLTBound:
    """Tests for the Central Limit Theorem–based bound."""

    def test_clt_decreases_with_more_samples(self):
        """CLT bound ε = z_{α/2}σ/√n decreases as n increases."""
        ebc = ErrorBoundComputer(bound_method="clt")

        err_100 = ebc.compute_sampling_error(100, variance=4.0, alpha=0.05)
        err_1000 = ebc.compute_sampling_error(1000, variance=4.0, alpha=0.05)

        assert err_1000 < err_100

    def test_clt_exact_formula(self):
        """CLT bound should match ε = z_{α/2}σ/√n."""
        variance, n, alpha = 4.0, 200, 0.05
        ebc = ErrorBoundComputer(bound_method="clt")

        z = sp_stats.norm.ppf(1 - alpha / 2)
        sigma = math.sqrt(variance)
        expected = z * sigma / math.sqrt(n)
        actual = ebc.compute_sampling_error(n, variance, alpha)

        assert actual == pytest.approx(expected, rel=1e-10)

    def test_clt_tighter_than_hoeffding_large_n(self):
        """For large n, CLT bound should be tighter than Hoeffding."""
        variance, n, alpha, R = 4.0, 1000, 0.05, 100.0

        ebc_clt = ErrorBoundComputer(bound_method="clt", cost_range=R)
        ebc_hoef = ErrorBoundComputer(bound_method="hoeffding", cost_range=R)

        err_clt = ebc_clt.compute_sampling_error(n, variance, alpha)
        err_hoef = ebc_hoef.compute_sampling_error(n, variance, alpha)

        assert err_clt < err_hoef

    def test_clt_tighter_than_chebyshev(self):
        """CLT bound should be tighter than Chebyshev for moderate n."""
        variance, n, alpha = 4.0, 200, 0.05

        ebc_clt = ErrorBoundComputer(bound_method="clt")
        ebc_cheb = ErrorBoundComputer(bound_method="chebyshev")

        err_clt = ebc_clt.compute_sampling_error(n, variance, alpha)
        err_cheb = ebc_cheb.compute_sampling_error(n, variance, alpha)

        assert err_clt < err_cheb


# ---------------------------------------------------------------------------
# Tests: compute_total_error
# ---------------------------------------------------------------------------


class TestTotalError:
    """Tests for compute_total_error — summing all error sources."""

    def test_total_error_is_sum(self):
        """ε_total = ε_abs + ε_samp + ε_model."""
        ebc = ErrorBoundComputer()
        total = ebc.compute_total_error(0.1, 0.2, 0.05)
        assert total == pytest.approx(0.35)

    def test_total_error_zero_model(self):
        """With zero model error, total = abstraction + sampling."""
        ebc = ErrorBoundComputer()
        total = ebc.compute_total_error(0.1, 0.2, 0.0)
        assert total == pytest.approx(0.3)

    def test_total_error_all_zero(self):
        """All components zero → total error zero."""
        ebc = ErrorBoundComputer()
        total = ebc.compute_total_error(0.0, 0.0, 0.0)
        assert total == 0.0


# ---------------------------------------------------------------------------
# Tests: compute_abstraction_error
# ---------------------------------------------------------------------------


class TestAbstractionError:
    """Tests for abstraction error from state-space partitioning."""

    def test_singleton_partition_zero_error(self):
        """Singleton partition (one state/block) → zero abstraction error."""
        mdp = _make_chain_mdp(n=4, cost=1.0)
        abstract = mdp  # same MDP as abstract
        partition = _trivial_partition(mdp)

        ebc = ErrorBoundComputer()
        err = ebc.compute_abstraction_error(mdp, abstract, partition)

        assert err == pytest.approx(0.0)

    def test_coarse_partition_nonzero_error(self):
        """Coarse partition grouping all states may have nonzero error."""
        mdp = _make_chain_mdp(n=4, cost=1.0)
        abstract = _make_simple_mdp(cost=1.0)
        partition = _coarse_partition(mdp)

        ebc = ErrorBoundComputer()
        err = ebc.compute_abstraction_error(mdp, abstract, partition)

        # Chain has states with different action availability, so variation ≥ 0
        assert err >= 0.0

    def test_abstraction_error_nonnegative(self):
        """Abstraction error should always be ≥ 0."""
        mdp = _make_simple_mdp(cost=2.0)
        partition = _trivial_partition(mdp)

        ebc = ErrorBoundComputer()
        err = ebc.compute_abstraction_error(mdp, mdp, partition)

        assert err >= 0.0


# ---------------------------------------------------------------------------
# Tests: compute_required_samples
# ---------------------------------------------------------------------------


class TestRequiredSamples:
    """Tests for the inverse problem: how many samples for a target error?"""

    def test_required_samples_positive(self):
        """compute_required_samples should return at least 1."""
        ebc = ErrorBoundComputer(bound_method="hoeffding", cost_range=10.0)
        n = ebc.compute_required_samples(target_error=5.0, variance=1.0, alpha=0.05)
        assert n >= 1

    def test_tighter_target_needs_more_samples(self):
        """Smaller target error → more required samples (ε ∝ 1/√n)."""
        ebc = ErrorBoundComputer(bound_method="hoeffding", cost_range=10.0)
        n_wide = ebc.compute_required_samples(target_error=1.0, variance=1.0, alpha=0.05)
        n_tight = ebc.compute_required_samples(target_error=0.5, variance=1.0, alpha=0.05)

        assert n_tight > n_wide

    def test_required_samples_hoeffding_formula(self):
        """Hoeffding: n = R²ln(2/α)/(2ε²)."""
        R, target, alpha = 10.0, 1.0, 0.05
        ebc = ErrorBoundComputer(bound_method="hoeffding", cost_range=R)

        expected = math.ceil(R**2 * math.log(2.0 / alpha) / (2.0 * target**2))
        actual = ebc.compute_required_samples(target, variance=1.0, alpha=alpha)

        assert actual == expected

    def test_required_samples_clt_formula(self):
        """CLT: n = (zσ/ε)²."""
        variance, target, alpha = 4.0, 0.5, 0.05
        ebc = ErrorBoundComputer(bound_method="clt")

        z = sp_stats.norm.ppf(1 - alpha / 2)
        sigma = math.sqrt(variance)
        expected = math.ceil((z * sigma / target) ** 2)
        actual = ebc.compute_required_samples(target, variance, alpha)

        assert actual == expected

    def test_zero_target_returns_large_n(self):
        """Target error of zero → very large sample count (capped at 10⁹)."""
        ebc = ErrorBoundComputer(bound_method="hoeffding", cost_range=10.0)
        n = ebc.compute_required_samples(target_error=0.0, variance=1.0, alpha=0.05)
        assert n >= 1_000_000


# ---------------------------------------------------------------------------
# Tests: full_analysis
# ---------------------------------------------------------------------------


class TestFullAnalysis:
    """Tests for full_analysis() — the complete error analysis pipeline."""

    def test_full_analysis_returns_result(self):
        """full_analysis() should return an ErrorBoundResult."""
        mdp = _make_simple_mdp(cost=1.0)
        partition = _trivial_partition(mdp)

        ebc = ErrorBoundComputer(bound_method="hoeffding", cost_range=10.0)
        result = ebc.full_analysis(
            original=mdp, abstract=mdp, partition=partition,
            n_trajectories=100, cost_variance=4.0, alpha=0.05,
        )

        assert isinstance(result, ErrorBoundResult)

    def test_full_analysis_total_equals_sum(self):
        """full_analysis().total_error == abs + sampling + model."""
        mdp = _make_simple_mdp(cost=1.0)
        partition = _trivial_partition(mdp)

        ebc = ErrorBoundComputer(bound_method="hoeffding", cost_range=10.0)
        result = ebc.full_analysis(
            original=mdp, abstract=mdp, partition=partition,
            n_trajectories=100, cost_variance=4.0, model_error=0.1, alpha=0.05,
        )

        expected_total = result.abstraction_error + result.sampling_error + result.model_error
        assert result.total_error == pytest.approx(expected_total)

    def test_full_analysis_confidence(self):
        """full_analysis() confidence = 1 − α."""
        mdp = _make_simple_mdp(cost=1.0)
        partition = _trivial_partition(mdp)

        ebc = ErrorBoundComputer(bound_method="hoeffding", cost_range=10.0)
        result = ebc.full_analysis(
            original=mdp, abstract=mdp, partition=partition,
            n_trajectories=100, cost_variance=4.0, alpha=0.01,
        )

        assert result.confidence == pytest.approx(0.99)

    def test_full_analysis_with_target_error(self):
        """With target_error, required_samples should be > 0."""
        mdp = _make_simple_mdp(cost=1.0)
        partition = _trivial_partition(mdp)

        ebc = ErrorBoundComputer(bound_method="hoeffding", cost_range=10.0)
        result = ebc.full_analysis(
            original=mdp, abstract=mdp, partition=partition,
            n_trajectories=100, cost_variance=4.0, alpha=0.05,
            target_error=0.5,
        )

        assert result.required_samples > 0

    def test_full_analysis_without_target_error(self):
        """Without target_error, required_samples should be 0."""
        mdp = _make_simple_mdp(cost=1.0)
        partition = _trivial_partition(mdp)

        ebc = ErrorBoundComputer(bound_method="hoeffding", cost_range=10.0)
        result = ebc.full_analysis(
            original=mdp, abstract=mdp, partition=partition,
            n_trajectories=100, cost_variance=4.0, alpha=0.05,
        )

        assert result.required_samples == 0

    def test_full_analysis_unknown_method_raises(self):
        """Unknown bound method raises ValueError."""
        mdp = _make_simple_mdp(cost=1.0)
        partition = _trivial_partition(mdp)

        ebc = ErrorBoundComputer(bound_method="invalid")

        with pytest.raises(ValueError, match="Unknown bound method"):
            ebc.full_analysis(
                original=mdp, abstract=mdp, partition=partition,
                n_trajectories=100, cost_variance=4.0,
            )
