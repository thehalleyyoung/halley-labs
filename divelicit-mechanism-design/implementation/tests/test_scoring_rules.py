"""Tests for proper scoring rules."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scoring_rules import (
    LogarithmicRule,
    BrierRule,
    SphericalRule,
    CRPSRule,
    PowerRule,
    EnergyAugmentedRule,
    verify_properness,
)


def _random_dist(n, rng=None):
    """Generate a random probability distribution."""
    if rng is None:
        rng = np.random.RandomState(42)
    p = rng.dirichlet(np.ones(n))
    return p


def test_log_properness():
    """LogarithmicRule is proper: E_q[S(q,Y)] >= E_q[S(p,Y)]."""
    rule = LogarithmicRule()
    rng = np.random.RandomState(1)
    for _ in range(20):
        q = _random_dist(5, rng)
        p = _random_dist(5, rng)
        gap = rule.properness_gap(p, q)
        assert gap >= -1e-10, f"Properness violated: gap={gap}"


def test_brier_properness():
    rule = BrierRule()
    rng = np.random.RandomState(2)
    for _ in range(20):
        q = _random_dist(5, rng)
        p = _random_dist(5, rng)
        gap = rule.properness_gap(p, q)
        assert gap >= -1e-10, f"Properness violated: gap={gap}"


def test_spherical_properness():
    rule = SphericalRule()
    rng = np.random.RandomState(3)
    for _ in range(20):
        q = _random_dist(5, rng)
        p = _random_dist(5, rng)
        gap = rule.properness_gap(p, q)
        assert gap >= -1e-10, f"Properness violated: gap={gap}"


def test_crps_properness():
    rule = CRPSRule()
    rng = np.random.RandomState(4)
    for _ in range(20):
        q = _random_dist(5, rng)
        p = _random_dist(5, rng)
        gap = rule.properness_gap(p, q)
        assert gap >= -1e-10, f"Properness violated: gap={gap}"


def test_power_properness_alpha_2():
    rule = PowerRule(alpha=2.0)
    rng = np.random.RandomState(5)
    for _ in range(20):
        q = _random_dist(5, rng)
        p = _random_dist(5, rng)
        gap = rule.properness_gap(p, q)
        assert gap >= -1e-10, f"Properness violated: gap={gap}"


def test_energy_augmented_properness():
    """Energy-augmented rule preserves properness because energy depends on y, not p."""
    base = LogarithmicRule()
    energy_fn = lambda y, hist: float(y)  # arbitrary function of y
    rule = EnergyAugmentedRule(base, energy_fn, lambda_=0.5)
    rule.set_history(np.array([]))

    rng = np.random.RandomState(6)
    for _ in range(20):
        q = _random_dist(5, rng)
        p = _random_dist(5, rng)
        gap = rule.properness_gap(p, q)
        assert gap >= -1e-10, f"Energy-augmented properness violated: gap={gap}"


def test_properness_gap_equals_kl_for_log():
    """For LogarithmicRule, properness gap = KL(q||p)."""
    rule = LogarithmicRule()
    rng = np.random.RandomState(7)
    q = _random_dist(5, rng)
    p = _random_dist(5, rng)

    gap = rule.properness_gap(p, q)
    kl = np.sum(q * np.log(q / np.clip(p, 1e-15, None)))
    assert abs(gap - kl) < 1e-10, f"Gap {gap} != KL {kl}"


def test_properness_gap_nonnegative():
    """Properness gap should be non-negative for all proper rules."""
    rules = [LogarithmicRule(), BrierRule(), SphericalRule(), CRPSRule(), PowerRule(2.0)]
    rng = np.random.RandomState(8)
    for rule in rules:
        for _ in range(10):
            q = _random_dist(5, rng)
            p = _random_dist(5, rng)
            gap = rule.properness_gap(p, q)
            assert gap >= -1e-10


def test_verify_properness_passes_proper_rule():
    rule = BrierRule()
    rng = np.random.RandomState(9)
    q = _random_dist(5, rng)
    p = _random_dist(5, rng)
    assert verify_properness(rule, p, q, n_samples=5000)


def test_verify_properness_detects_improper():
    """An improper rule that always favors uniform should fail."""

    class ImproperRule:
        def score(self, p, y):
            # Rewards uniform regardless of p
            return -np.sum((p - 1.0 / len(p)) ** 2)

        def expected_score(self, p, q):
            return float(-np.sum((p - 1.0 / len(p)) ** 2))

        def properness_gap(self, p, q):
            return self.expected_score(q, q) - self.expected_score(p, q)

    rule = ImproperRule()
    # q is non-uniform, uniform p should score better -> improper
    q = np.array([0.8, 0.1, 0.1])
    p = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    # For this improper rule, uniform p scores higher than true q
    gap = rule.properness_gap(p, q)
    assert gap < 0, "Improper rule should have negative gap"


def test_score_range():
    """Scores should be finite."""
    rule = BrierRule()
    p = np.array([0.5, 0.3, 0.2])
    for y in range(3):
        s = rule.score(p, y)
        assert np.isfinite(s)


def test_energy_augmented_increases_diversity_reward():
    """Energy-augmented rule gives higher scores to outcomes far from history."""
    base = BrierRule()
    # Energy function that rewards distance from history indices
    def energy_fn(y, hist):
        if len(hist) == 0:
            return 0.0
        return float(min(abs(y - h) for h in hist))

    rule = EnergyAugmentedRule(base, energy_fn, lambda_=1.0)
    rule.set_history(np.array([0]))

    p = np.array([0.25, 0.25, 0.25, 0.25])
    # Outcome far from history (index 0) should get higher energy bonus
    score_close = rule.score(p, 0)
    score_far = rule.score(p, 3)
    assert score_far > score_close
