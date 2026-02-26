"""
Tests for bounded liveness specifications.
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from causal_trading.shield.bounded_liveness_specs import (
    BoundedLivenessLibrary,
    BoundedLivenessSpec,
    DrawdownRecoverySpec,
    LossRecoverySpec,
    PositionReductionSpec,
    RegimeTransitionSpec,
    TrajectoryResult,
)
from causal_trading.shield.safety_specs import (
    LTLFormula,
    SafetySpecification,
    TrajectoryChecker,
)
from causal_trading.verification.temporal_logic import (
    BoundedLTL,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def make_trajectory(n: int, **series) -> list:
    """Build a trajectory of n state dicts from keyword series."""
    traj = []
    for i in range(n):
        state = {}
        for key, vals in series.items():
            state[key] = vals[i] if i < len(vals) else vals[-1]
        traj.append(state)
    return traj


# -----------------------------------------------------------------------
# DrawdownRecoverySpec tests
# -----------------------------------------------------------------------

class TestDrawdownRecoverySpec:
    def test_inherits_safety_spec(self):
        spec = DrawdownRecoverySpec()
        assert isinstance(spec, SafetySpecification)
        assert isinstance(spec, BoundedLivenessSpec)

    def test_satisfying_trajectory(self):
        """Drawdown spikes then recovers within horizon."""
        spec = DrawdownRecoverySpec(threshold=0.05, recovery_level=0.02,
                                   horizon=10)
        # 30 steps: low, spike at step 10, recover by step 15
        dd = [0.01] * 10 + [0.06] + [0.04, 0.03, 0.02, 0.015, 0.01] + [0.01] * 14
        traj = make_trajectory(30, drawdown=dd)
        assert spec.check(traj) is True

    def test_violating_trajectory(self):
        """Drawdown spikes and never recovers."""
        spec = DrawdownRecoverySpec(threshold=0.05, recovery_level=0.02,
                                   horizon=5)
        dd = [0.01] * 5 + [0.06] * 20
        traj = make_trajectory(25, drawdown=dd)
        assert spec.check(traj) is False

    def test_no_trigger_always_satisfied(self):
        """If drawdown never exceeds threshold, spec is trivially true."""
        spec = DrawdownRecoverySpec(threshold=0.05, recovery_level=0.02,
                                   horizon=5)
        dd = [0.01] * 50
        traj = make_trajectory(50, drawdown=dd)
        assert spec.check(traj) is True

    def test_evaluate_trajectory_details(self):
        spec = DrawdownRecoverySpec(threshold=0.05, recovery_level=0.02,
                                   horizon=5)
        dd = [0.01] * 5 + [0.06] * 10
        traj = make_trajectory(15, drawdown=dd)
        result = spec.evaluate_trajectory(traj)
        assert isinstance(result, TrajectoryResult)
        assert result.trigger_count > 0
        assert not result.satisfied
        assert len(result.violation_times) > 0

    def test_is_satisfied_state(self):
        spec = DrawdownRecoverySpec(threshold=0.05, recovery_level=0.02)
        assert spec.is_satisfied({"drawdown": 0.01}) is True
        assert spec.is_satisfied({"drawdown": 0.03}) is False

    def test_to_ltl_formula_string(self):
        spec = DrawdownRecoverySpec(threshold=0.05, recovery_level=0.02,
                                   horizon=20)
        formula_str = spec.to_ltl_formula()
        assert "G(" in formula_str
        assert "F[0,20]" in formula_str

    def test_to_ltl_returns_formula(self):
        spec = DrawdownRecoverySpec()
        formula = spec.to_ltl()
        assert isinstance(formula, LTLFormula)

    def test_safe_state_mask(self):
        spec = DrawdownRecoverySpec(threshold=0.05)
        mask = spec.get_safe_state_mask(100)
        assert mask.shape == (100,)
        assert mask.dtype == bool
        # Low-index states (low drawdown) should be safe
        assert mask[0] is np.bool_(True)

    def test_get_constraints(self):
        spec = DrawdownRecoverySpec()
        constraints = spec.get_constraints()
        assert len(constraints) >= 1
        assert callable(constraints[0])


# -----------------------------------------------------------------------
# LossRecoverySpec tests
# -----------------------------------------------------------------------

class TestLossRecoverySpec:
    def test_satisfying_trajectory(self):
        spec = LossRecoverySpec(threshold=0.03, recovery_level=0.01,
                                horizon=5)
        loss = [0.00] * 5 + [0.04, 0.03, 0.02, 0.01, 0.005] + [0.005] * 10
        traj = make_trajectory(20, loss=loss)
        assert spec.check(traj) is True

    def test_violating_trajectory(self):
        spec = LossRecoverySpec(threshold=0.03, recovery_level=0.01,
                                horizon=3)
        loss = [0.00] * 3 + [0.04] * 10
        traj = make_trajectory(13, loss=loss)
        assert spec.check(traj) is False

    def test_no_trigger(self):
        spec = LossRecoverySpec(threshold=0.03, recovery_level=0.01,
                                horizon=3)
        loss = [0.01] * 20
        traj = make_trajectory(20, loss=loss)
        assert spec.check(traj) is True

    def test_to_ltl_formula(self):
        spec = LossRecoverySpec(threshold=0.03, recovery_level=0.01,
                                horizon=5)
        s = spec.to_ltl_formula()
        assert "F[0,5]" in s

    def test_safe_state_mask(self):
        spec = LossRecoverySpec(threshold=0.03)
        mask = spec.get_safe_state_mask(100)
        assert mask.shape == (100,)


# -----------------------------------------------------------------------
# PositionReductionSpec tests
# -----------------------------------------------------------------------

class TestPositionReductionSpec:
    def test_satisfying(self):
        spec = PositionReductionSpec(limit=100, safe_level=80, horizon=5)
        exp = [50] * 5 + [110, 100, 90, 75, 70] + [60] * 10
        traj = make_trajectory(20, exposure=exp)
        assert spec.check(traj) is True

    def test_violating(self):
        spec = PositionReductionSpec(limit=100, safe_level=80, horizon=3)
        exp = [50] * 3 + [110] * 10
        traj = make_trajectory(13, exposure=exp)
        assert spec.check(traj) is False

    def test_to_ltl_formula(self):
        spec = PositionReductionSpec(limit=100, safe_level=80, horizon=10)
        s = spec.to_ltl_formula()
        assert "F[0,10]" in s

    def test_evaluate_trajectory(self):
        spec = PositionReductionSpec(limit=100, safe_level=80, horizon=3)
        exp = [50] * 3 + [110] * 10
        traj = make_trajectory(13, exposure=exp)
        result = spec.evaluate_trajectory(traj)
        assert result.trigger_count > 0
        assert not result.satisfied


# -----------------------------------------------------------------------
# RegimeTransitionSpec tests
# -----------------------------------------------------------------------

class TestRegimeTransitionSpec:
    def test_satisfying(self):
        spec = RegimeTransitionSpec(adaptation_window=5)
        rc = [0] * 10 + [1] + [0] * 3 + [0] * 6
        adapted = [0] * 10 + [0, 0, 0, 1] + [0] * 6
        traj = make_trajectory(20, regime_change=rc, strategy_adapted=adapted)
        assert spec.check(traj) is True

    def test_violating(self):
        spec = RegimeTransitionSpec(adaptation_window=3)
        rc = [0] * 5 + [1] + [0] * 10
        adapted = [0] * 16
        traj = make_trajectory(16, regime_change=rc, strategy_adapted=adapted)
        assert spec.check(traj) is False

    def test_no_regime_change(self):
        spec = RegimeTransitionSpec(adaptation_window=5)
        traj = make_trajectory(20, regime_change=[0] * 20,
                               strategy_adapted=[0] * 20)
        assert spec.check(traj) is True

    def test_to_ltl_formula(self):
        spec = RegimeTransitionSpec(adaptation_window=10)
        s = spec.to_ltl_formula()
        assert "F[0,10]" in s
        assert "regime_change" in s
        assert "strategy_adapted" in s

    def test_safe_state_mask(self):
        spec = RegimeTransitionSpec()
        mask = spec.get_safe_state_mask(50)
        # All states are safe for regime transition
        assert np.all(mask)


# -----------------------------------------------------------------------
# LTL formula parsing via BoundedLTL
# -----------------------------------------------------------------------

class TestLTLFormulaParsing:
    """Test that the formula strings produced by specs can be parsed."""

    def test_drawdown_formula_parseable(self):
        ltl = BoundedLTL()
        spec = DrawdownRecoverySpec(threshold=0.05, recovery_level=0.02,
                                   horizon=20)
        formula_str = spec.to_ltl_formula()
        parsed = ltl.parse(formula_str)
        assert parsed is not None

    def test_loss_formula_parseable(self):
        ltl = BoundedLTL()
        spec = LossRecoverySpec(threshold=0.03, recovery_level=0.01,
                                horizon=5)
        formula_str = spec.to_ltl_formula()
        parsed = ltl.parse(formula_str)
        assert parsed is not None

    def test_position_formula_parseable(self):
        ltl = BoundedLTL()
        spec = PositionReductionSpec(limit=100, safe_level=80, horizon=10)
        formula_str = spec.to_ltl_formula()
        parsed = ltl.parse(formula_str)
        assert parsed is not None

    def test_regime_formula_parseable(self):
        ltl = BoundedLTL()
        spec = RegimeTransitionSpec(adaptation_window=10)
        formula_str = spec.to_ltl_formula()
        parsed = ltl.parse(formula_str)
        assert parsed is not None


# -----------------------------------------------------------------------
# BoundedLivenessLibrary tests
# -----------------------------------------------------------------------

class TestBoundedLivenessLibrary:
    def test_conservative_suite(self):
        specs = BoundedLivenessLibrary.conservative_suite()
        assert len(specs) == 4
        assert all(isinstance(s, BoundedLivenessSpec) for s in specs)
        assert all(isinstance(s, SafetySpecification) for s in specs)

    def test_moderate_suite(self):
        specs = BoundedLivenessLibrary.moderate_suite()
        assert len(specs) == 4

    def test_aggressive_suite(self):
        specs = BoundedLivenessLibrary.aggressive_suite()
        assert len(specs) == 4

    def test_conservative_tighter_than_aggressive(self):
        con = BoundedLivenessLibrary.conservative_suite()
        agg = BoundedLivenessLibrary.aggressive_suite()
        # Conservative has smaller horizons
        for c, a in zip(con, agg):
            assert c.horizon <= a.horizon

    def test_from_config(self):
        config = {
            "drawdown_recovery": {
                "threshold": 0.04, "recovery_level": 0.01, "horizon": 15,
            },
            "loss_recovery": {
                "threshold": 0.025, "recovery_level": 0.01, "horizon": 4,
            },
        }
        specs = BoundedLivenessLibrary.from_config(config)
        assert len(specs) == 2
        assert isinstance(specs[0], DrawdownRecoverySpec)
        assert isinstance(specs[1], LossRecoverySpec)

    def test_from_config_empty(self):
        specs = BoundedLivenessLibrary.from_config({})
        assert len(specs) == 0

    def test_from_config_all(self):
        config = {
            "drawdown_recovery": {
                "threshold": 0.05, "recovery_level": 0.02, "horizon": 20,
            },
            "loss_recovery": {
                "threshold": 0.03, "recovery_level": 0.01, "horizon": 5,
            },
            "position_reduction": {
                "limit": 100, "safe_level": 80, "horizon": 10,
            },
            "regime_transition": {
                "adaptation_window": 10,
            },
        }
        specs = BoundedLivenessLibrary.from_config(config)
        assert len(specs) == 4

    def test_all_spec_names(self):
        names = BoundedLivenessLibrary.all_spec_names()
        assert len(names) == 4
        assert "drawdown_recovery" in names

    def test_suite_specs_check_conforming_trajectory(self):
        """All specs in moderate suite should pass on a well-behaved traj."""
        specs = BoundedLivenessLibrary.moderate_suite()
        # Build a trajectory that never triggers any spec
        traj = make_trajectory(
            100,
            drawdown=[0.01] * 100,
            loss=[0.005] * 100,
            exposure=[50.0] * 100,
            regime_change=[0] * 100,
            strategy_adapted=[0] * 100,
        )
        for spec in specs:
            assert spec.check(traj) is True

    def test_quantitative_robustness(self):
        """Quantitative robustness should be positive for conforming traj."""
        spec = DrawdownRecoverySpec(threshold=0.05, recovery_level=0.02,
                                   horizon=20)
        traj = make_trajectory(
            30,
            drawdown=[0.01] * 30,
        )
        robustness = spec.check_quantitative(traj)
        assert robustness > 0


# -----------------------------------------------------------------------
# Integration: TrajectoryChecker with bounded liveness formulas
# -----------------------------------------------------------------------

class TestTrajectoryCheckerIntegration:
    def test_checker_on_satisfying(self):
        spec = DrawdownRecoverySpec(threshold=0.05, recovery_level=0.02,
                                   horizon=10)
        dd = [0.01] * 5 + [0.06, 0.04, 0.015] + [0.01] * 12
        traj = make_trajectory(20, drawdown=dd)

        formula = spec.to_ltl()
        checker = TrajectoryChecker(traj)
        assert checker.check(formula, 0) is True

    def test_checker_on_violating(self):
        spec = DrawdownRecoverySpec(threshold=0.05, recovery_level=0.02,
                                   horizon=3)
        dd = [0.01] * 3 + [0.06] * 10
        traj = make_trajectory(13, drawdown=dd)

        formula = spec.to_ltl()
        checker = TrajectoryChecker(traj)
        assert checker.check(formula, 0) is False
