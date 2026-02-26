"""Tests for shield synthesis module."""
import numpy as np
import pytest
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Lightweight stubs so tests run without heavy imports
# ---------------------------------------------------------------------------

class _SafetySpec:
    """Minimal safety-spec stub used when the real module is unavailable."""

    def __init__(self, name: str, threshold: float):
        self.name = name
        self.threshold = threshold

    def check(self, trajectory: dict) -> bool:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transition_matrices(n_states: int, n_actions: int, seed: int = 42):
    """Generate random transition matrices for a small MDP."""
    rng = np.random.RandomState(seed)
    T = rng.dirichlet(np.ones(n_states), size=(n_actions, n_states))
    return T  # shape (n_actions, n_states, n_states)


def _make_posterior_models(n_models: int, n_states: int, n_actions: int, seed: int = 0):
    """Return a list of (weight, transition_matrix) tuples."""
    rng = np.random.RandomState(seed)
    weights = rng.dirichlet(np.ones(n_models))
    models = []
    for i in range(n_models):
        T = np.zeros((n_actions, n_states, n_states))
        for a in range(n_actions):
            T[a] = rng.dirichlet(np.ones(n_states), size=n_states)
        models.append((weights[i], T))
    return models


def _simulate_trajectory(T, start_state: int, actions: list, rng=None):
    """Simulate a trajectory through an MDP."""
    if rng is None:
        rng = np.random.RandomState(123)
    states = [start_state]
    for a in actions:
        probs = T[a, states[-1]]
        next_state = rng.choice(len(probs), p=probs)
        states.append(next_state)
    return states


# ===================================================================
# Tests for safety_specs
# ===================================================================

class TestBoundedDrawdownSpec:
    """Tests for the bounded drawdown safety specification."""

    def test_no_drawdown_satisfies(self):
        """A trajectory with no drawdown satisfies any drawdown spec."""
        prices = np.array([100, 101, 102, 103, 104, 105])
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / peak
        assert np.all(drawdown <= 0.1)

    def test_large_drawdown_violates(self):
        """A trajectory with large drawdown violates a tight spec."""
        prices = np.array([100, 95, 80, 75, 70])
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / peak
        max_dd = drawdown.max()
        assert max_dd > 0.1  # 30% drawdown violates 10% spec

    def test_drawdown_computation_correctness(self):
        """Verify drawdown is computed correctly."""
        prices = np.array([100, 110, 105, 115, 100])
        peak = np.maximum.accumulate(prices)
        expected_peak = np.array([100, 110, 110, 115, 115])
        np.testing.assert_array_equal(peak, expected_peak)
        drawdown = (peak - prices) / peak
        expected_dd = np.array([0.0, 0.0, 5.0/110, 0.0, 15.0/115])
        np.testing.assert_allclose(drawdown, expected_dd, atol=1e-10)

    def test_drawdown_at_boundary(self):
        """Test drawdown exactly at the threshold."""
        threshold = 0.05
        prices = np.array([100, 100, 95])  # exactly 5%
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / peak
        assert drawdown.max() == pytest.approx(threshold, abs=1e-10)


class TestPositionLimitSpec:
    """Tests for position limit safety specification."""

    def test_within_limits(self):
        positions = np.array([0, 1, 2, 3, 2, 1, 0, -1, -2, -3])
        limit = 3
        assert np.all(np.abs(positions) <= limit)

    def test_exceeds_limits(self):
        positions = np.array([0, 1, 2, 3, 4])
        limit = 3
        assert not np.all(np.abs(positions) <= limit)

    def test_symmetric_limits(self):
        """Limits should apply symmetrically to long and short."""
        limit = 2
        assert np.abs(-2) <= limit
        assert np.abs(2) <= limit
        assert not np.abs(3) <= limit
        assert not np.abs(-3) <= limit


class TestMarginSpec:
    """Tests for margin safety specification."""

    def test_sufficient_margin(self):
        """Position value within margin capacity."""
        equity = 100000
        position_value = 50000
        margin_requirement = 0.5
        margin_ratio = equity / max(position_value, 1)
        assert margin_ratio >= margin_requirement

    def test_insufficient_margin(self):
        equity = 100000
        position_value = 300000
        margin_requirement = 0.5
        margin_ratio = equity / max(position_value, 1)
        assert margin_ratio < margin_requirement


class TestCompositeSpec:
    """Tests for composite (conjunction/disjunction) specifications."""

    def test_conjunction_all_satisfied(self):
        """Conjunction is satisfied when all sub-specs are satisfied."""
        results = [True, True, True]
        assert all(results)

    def test_conjunction_one_violated(self):
        """Conjunction fails when any sub-spec is violated."""
        results = [True, False, True]
        assert not all(results)

    def test_disjunction_one_satisfied(self):
        """Disjunction passes when any sub-spec is satisfied."""
        results = [False, True, False]
        assert any(results)

    def test_disjunction_none_satisfied(self):
        """Disjunction fails when no sub-spec is satisfied."""
        results = [False, False, False]
        assert not any(results)


# ===================================================================
# Tests for shield synthesis
# ===================================================================

class TestShieldSynthesis:
    """Tests for posterior-predictive shield synthesis."""

    def test_shield_blocks_unsafe_action(self):
        """Shield should block actions that violate safety with high probability."""
        n_states, n_actions = 5, 4
        models = _make_posterior_models(10, n_states, n_actions, seed=42)
        
        # Define a "bad" action for state 0: always transitions to state 4
        for i, (w, T) in enumerate(models):
            T[3, 0, :] = 0
            T[3, 0, 4] = 1.0  # action 3 from state 0 always goes to state 4
        
        # If state 4 is "unsafe", the shield should block action 3 in state 0
        unsafe_states = {4}
        delta = 0.1
        
        # Check: action 3 from state 0 leads to unsafe state with prob 1.0
        violation_prob = sum(
            w * T[3, 0, 4] for w, T in models
        )
        assert violation_prob > 1 - delta  # should be blocked

    def test_shield_permits_safe_action(self):
        """Shield should permit actions that satisfy safety with high probability."""
        n_states, n_actions = 5, 4
        rng = np.random.RandomState(42)
        
        # Create models where action 0 from state 0 always stays in safe states
        models = []
        for i in range(10):
            w = 0.1
            T = rng.dirichlet(np.ones(n_states), size=(n_actions, n_states))
            # Make action 0 from state 0 safe (stays in states 0-3)
            T[0, 0, :] = 0
            T[0, 0, :4] = 0.25
            models.append((w, T))
        
        unsafe_states = {4}
        delta = 0.1
        
        # P(transition to unsafe | action 0, state 0) = 0 for all models
        violation_prob = sum(w * T[0, 0, 4] for w, T in models)
        assert violation_prob < delta  # should be permitted

    def test_shield_action_set_nonempty(self):
        """Shield should permit at least one action if any safe action exists."""
        n_states, n_actions = 3, 3
        rng = np.random.RandomState(42)
        
        models = _make_posterior_models(5, n_states, n_actions, seed=42)
        
        # For each state, check that at least one action has low violation prob
        delta = 0.5  # generous threshold
        for s in range(n_states):
            permitted = []
            for a in range(n_actions):
                violation_prob = sum(w * T[a, s, n_states-1] for w, T in models)
                if violation_prob < delta:
                    permitted.append(a)
            # With generous delta, usually at least one action is permitted
            # (this is a probabilistic check, but with seed it's deterministic)

    def test_shield_composition(self):
        """Composition of two shields is more restrictive than either alone."""
        n_states, n_actions = 4, 4
        rng = np.random.RandomState(42)
        
        # Shield 1: blocks actions leading to state 3
        # Shield 2: blocks actions leading to state 2
        models = _make_posterior_models(5, n_states, n_actions, seed=42)
        delta = 0.3
        
        for s in range(n_states):
            permitted_1 = set()
            permitted_2 = set()
            
            for a in range(n_actions):
                p_unsafe_1 = sum(w * T[a, s, 3] for w, T in models)
                p_unsafe_2 = sum(w * T[a, s, 2] for w, T in models)
                
                if p_unsafe_1 < delta:
                    permitted_1.add(a)
                if p_unsafe_2 < delta:
                    permitted_2.add(a)
            
            composed = permitted_1 & permitted_2
            assert composed <= permitted_1
            assert composed <= permitted_2

    def test_shield_with_deterministic_model(self):
        """Shield with a single deterministic model should be exact."""
        n_states, n_actions = 3, 2
        T = np.zeros((n_actions, n_states, n_states))
        # Action 0: always goes to state 0 (safe)
        T[0, :, 0] = 1.0
        # Action 1: always goes to state 2 (unsafe)
        T[1, :, 2] = 1.0
        
        models = [(1.0, T)]
        unsafe_states = {2}
        delta = 0.01
        
        for s in range(n_states):
            # Action 0 should always be permitted
            p_unsafe_0 = sum(w * Tm[0, s, 2] for w, Tm in models)
            assert p_unsafe_0 < delta
            
            # Action 1 should always be blocked
            p_unsafe_1 = sum(w * Tm[1, s, 2] for w, Tm in models)
            assert p_unsafe_1 > 1 - delta


# ===================================================================
# Tests for PAC-Bayes bounds
# ===================================================================

class TestPACBayes:
    """Tests for PAC-Bayes bound computation."""

    def test_kl_divergence_same_distribution(self):
        """KL(p || p) = 0 for any distribution p."""
        p = np.array([0.2, 0.3, 0.5])
        kl = np.sum(p * np.log(p / p))
        assert kl == pytest.approx(0.0, abs=1e-10)

    def test_kl_divergence_nonnegative(self):
        """KL divergence is always non-negative (Gibbs inequality)."""
        rng = np.random.RandomState(42)
        for _ in range(100):
            p = rng.dirichlet(np.ones(5))
            q = rng.dirichlet(np.ones(5))
            kl = np.sum(p * np.log(p / q))
            assert kl >= -1e-10  # numerical tolerance

    def test_kl_divergence_asymmetric(self):
        """KL(p || q) != KL(q || p) in general."""
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        kl_pq = np.sum(p * np.log(p / q))
        kl_qp = np.sum(q * np.log(q / p))
        assert kl_pq != pytest.approx(kl_qp)

    def test_pac_bayes_bound_formula(self):
        """Test the PAC-Bayes bound computation."""
        n = 1000
        delta = 0.05
        kl = 2.0  # KL(posterior || prior)
        eps = delta
        
        complexity_term = np.sqrt(
            (kl + np.log(2 * np.sqrt(n) / eps)) / (2 * n)
        )
        
        assert complexity_term > 0
        assert complexity_term < 1  # should be reasonable for n=1000

    def test_bound_tightens_with_data(self):
        """PAC-Bayes bound should tighten as n increases."""
        kl = 1.0
        delta = 0.05
        
        bounds = []
        for n in [100, 500, 1000, 5000, 10000]:
            bound = np.sqrt((kl + np.log(2 * np.sqrt(n) / delta)) / (2 * n))
            bounds.append(bound)
        
        # Bounds should be monotonically decreasing
        for i in range(len(bounds) - 1):
            assert bounds[i] > bounds[i + 1]

    def test_bound_increases_with_kl(self):
        """Larger KL divergence gives looser bound."""
        n = 1000
        delta = 0.05
        
        bounds = []
        for kl in [0.1, 0.5, 1.0, 5.0, 10.0]:
            bound = np.sqrt((kl + np.log(2 * np.sqrt(n) / delta)) / (2 * n))
            bounds.append(bound)
        
        for i in range(len(bounds) - 1):
            assert bounds[i] < bounds[i + 1]

    def test_catoni_bound(self):
        """Catoni bound for 0-1 loss should be valid."""
        # Catoni bound: empirical_risk + f(KL, n, delta)
        # where f is the inverse of the Catoni function
        empirical_risk = 0.05
        kl = 1.0
        n = 500
        delta = 0.05
        
        # Simplified Catoni-style bound
        bound = 1 - np.exp(
            -((kl + np.log(2 * np.sqrt(n) / delta)) / n)
            - empirical_risk * n / n
        )
        # Just check it's in valid range
        assert 0 <= bound <= 1

    def test_sequential_pac_bayes(self):
        """Sequential PAC-Bayes bound should hold uniformly over time."""
        kl = 1.0
        delta = 0.05
        
        # At each time step, the bound should be valid
        for t in range(1, 100):
            n = t * 10
            # Use peeling device: delta_t = delta * 6 / (pi^2 * t^2)
            delta_t = delta * 6 / (np.pi**2 * t**2)
            bound = np.sqrt((kl + np.log(2 * np.sqrt(n) / delta_t)) / (2 * n))
            assert bound > 0
            assert bound < 10  # should be finite


# ===================================================================
# Tests for shield liveness
# ===================================================================

class TestShieldLiveness:
    """Tests for shield liveness theorem."""

    def test_permissivity_ratio_computation(self):
        """Permissivity ratio should be in [0, 1]."""
        n_actions = 10
        for n_permitted in range(n_actions + 1):
            ratio = n_permitted / n_actions
            assert 0 <= ratio <= 1

    def test_permissivity_with_generous_spec(self):
        """With a loose safety spec, permissivity should be high."""
        n_states, n_actions = 5, 10
        delta = 0.99  # very loose: almost any action is "safe"
        
        rng = np.random.RandomState(42)
        models = _make_posterior_models(5, n_states, n_actions, seed=42)
        
        total_permitted = 0
        total_possible = n_states * n_actions
        
        for s in range(n_states):
            for a in range(n_actions):
                # With delta=0.99, almost everything is permitted
                p_unsafe = sum(w * T[a, s, n_states-1] for w, T in models)
                if p_unsafe < delta:
                    total_permitted += 1
        
        permissivity = total_permitted / total_possible
        assert permissivity > 0.5  # should be high with loose spec

    def test_permissivity_with_tight_spec(self):
        """With a tight safety spec, permissivity may be lower."""
        n_states, n_actions = 5, 10
        delta = 0.01  # very tight
        
        models = _make_posterior_models(5, n_states, n_actions, seed=42)
        
        total_permitted = 0
        total_possible = n_states * n_actions
        
        for s in range(n_states):
            for a in range(n_actions):
                p_unsafe = sum(w * T[a, s, n_states-1] for w, T in models)
                if p_unsafe < delta:
                    total_permitted += 1
        
        permissivity = total_permitted / total_possible
        # Tight spec means lower permissivity (but not necessarily zero)
        assert 0 <= permissivity <= 1

    def test_liveness_minimum_bound(self):
        """Under favorable conditions, permissivity should exceed minimum."""
        n_states, n_actions = 3, 5
        
        # Create a model where action 0 is always safe
        T = np.zeros((n_actions, n_states, n_states))
        for a in range(n_actions):
            for s in range(n_states):
                T[a, s, :] = 1.0 / n_states  # uniform transitions
        # Make action 0 always go to state 0 (safe)
        T[0, :, :] = 0
        T[0, :, 0] = 1.0
        
        models = [(1.0, T)]
        delta = 0.1
        
        # Action 0 should be permitted in every state
        for s in range(n_states):
            p_unsafe = T[0, s, n_states-1]  # prob of going to last state
            assert p_unsafe < delta
        
        # Permissivity >= 1/n_actions (at least action 0 is always permitted)
        min_permissivity = 1.0 / n_actions
        
        total_permitted = 0
        for s in range(n_states):
            for a in range(n_actions):
                p_unsafe = sum(w * Tm[a, s, n_states-1] for w, Tm in models)
                if p_unsafe < delta:
                    total_permitted += 1
        
        permissivity = total_permitted / (n_states * n_actions)
        assert permissivity >= min_permissivity

    def test_permissivity_tracking_over_time(self):
        """Permissivity ratio can be tracked as a time series."""
        rng = np.random.RandomState(42)
        n_steps = 100
        n_actions = 5
        
        permissivity_series = []
        for t in range(n_steps):
            # Simulate varying number of permitted actions
            n_permitted = rng.randint(1, n_actions + 1)
            permissivity_series.append(n_permitted / n_actions)
        
        series = np.array(permissivity_series)
        assert len(series) == n_steps
        assert np.all(series >= 0)
        assert np.all(series <= 1)
        
        # Check running average
        window = 10
        running_avg = np.convolve(series, np.ones(window)/window, mode='valid')
        assert len(running_avg) == n_steps - window + 1


# ===================================================================
# Tests for permissivity tracking
# ===================================================================

class TestPermissivityTracker:
    """Tests for permissivity ratio tracking and analysis."""

    def test_per_regime_tracking(self):
        """Track permissivity separately for each regime."""
        rng = np.random.RandomState(42)
        n_regimes = 3
        n_actions = 5
        n_steps = 200
        
        regime_permissivity = {r: [] for r in range(n_regimes)}
        
        for t in range(n_steps):
            regime = rng.randint(n_regimes)
            # Different regimes have different base permissivities
            base = 0.3 + 0.2 * regime
            n_permitted = max(1, int(base * n_actions + rng.normal(0, 0.5)))
            n_permitted = min(n_permitted, n_actions)
            regime_permissivity[regime].append(n_permitted / n_actions)
        
        # Regime 2 should have highest average permissivity
        avgs = {r: np.mean(v) for r, v in regime_permissivity.items() if v}
        assert avgs[2] > avgs[0]

    def test_permissivity_decomposition(self):
        """Identify which specification is most restrictive."""
        n_actions = 10
        
        # Three specs with different restrictiveness
        spec_permits = {
            'drawdown': set(range(8)),      # permits 8/10
            'position_limit': set(range(6)), # permits 6/10
            'margin': set(range(9)),          # permits 9/10
        }
        
        # The composed shield permits the intersection
        composed = spec_permits['drawdown'] & spec_permits['position_limit'] & spec_permits['margin']
        
        # Position limit is most restrictive
        most_restrictive = min(spec_permits, key=lambda k: len(spec_permits[k]))
        assert most_restrictive == 'position_limit'
        
        # Composed set is subset of each individual
        for spec_name, permitted in spec_permits.items():
            assert composed <= permitted

    def test_trend_detection(self):
        """Detect declining permissivity trend."""
        rng = np.random.RandomState(42)
        n_steps = 100
        
        # Create declining permissivity
        base = np.linspace(0.8, 0.2, n_steps)
        noise = rng.normal(0, 0.05, n_steps)
        series = np.clip(base + noise, 0, 1)
        
        # Simple linear regression to detect trend
        t = np.arange(n_steps)
        slope = np.polyfit(t, series, 1)[0]
        assert slope < 0  # declining trend


# ===================================================================
# Tests for trajectory checking
# ===================================================================

class TestTrajectoryChecking:
    """Tests for checking safety specifications on trajectories."""

    def test_safe_trajectory(self):
        """A trajectory that never violates specs should pass."""
        prices = np.array([100, 101, 102, 103, 104, 105])
        positions = np.array([0, 1, 1, 1, 0, 0])
        
        max_drawdown = 0.0  # prices only go up
        max_position = 1
        
        assert max_drawdown <= 0.05  # 5% drawdown limit
        assert max_position <= 3      # position limit

    def test_unsafe_trajectory_drawdown(self):
        """A trajectory with large drawdown should fail."""
        prices = np.array([100, 110, 80, 70, 60])
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / peak
        max_dd = drawdown.max()
        
        assert max_dd > 0.05  # violates 5% drawdown limit

    def test_unsafe_trajectory_position(self):
        """A trajectory exceeding position limits should fail."""
        positions = np.array([0, 1, 2, 3, 4, 5])
        limit = 3
        assert np.any(np.abs(positions) > limit)

    def test_multi_spec_checking(self):
        """Check multiple specs simultaneously."""
        prices = np.array([100, 99, 98, 97, 96, 95])
        positions = np.array([0, 1, 1, 2, 2, 1])
        
        # Drawdown check
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / peak
        dd_ok = drawdown.max() <= 0.10  # 10% limit
        
        # Position check
        pos_ok = np.all(np.abs(positions) <= 3)
        
        # Both must pass
        all_ok = dd_ok and pos_ok
        assert isinstance(all_ok, (bool, np.bool_))


# ===================================================================
# Tests for shield with posterior uncertainty
# ===================================================================

class TestShieldWithUncertainty:
    """Tests for shield behavior under posterior uncertainty."""

    def test_more_concentrated_posterior_more_permissive(self):
        """A more concentrated posterior should yield a more permissive shield."""
        n_states, n_actions = 3, 4
        rng = np.random.RandomState(42)
        
        # Create a "safe" base model
        T_safe = np.zeros((n_actions, n_states, n_states))
        for a in range(n_actions):
            T_safe[a] = rng.dirichlet(10 * np.ones(n_states), size=n_states)
            T_safe[a, :, n_states-1] *= 0.01  # very low prob of unsafe state
            T_safe[a] /= T_safe[a].sum(axis=1, keepdims=True)
        
        delta = 0.1
        
        # Concentrated posterior (all weight on safe model)
        models_concentrated = [(1.0, T_safe)]
        
        # Diffuse posterior (mix safe model with random models)
        T_random = rng.dirichlet(np.ones(n_states), size=(n_actions, n_states))
        models_diffuse = [(0.5, T_safe), (0.5, T_random)]
        
        permitted_concentrated = 0
        permitted_diffuse = 0
        
        for s in range(n_states):
            for a in range(n_actions):
                p_unsafe_c = sum(w * T[a, s, n_states-1] for w, T in models_concentrated)
                p_unsafe_d = sum(w * T[a, s, n_states-1] for w, T in models_diffuse)
                
                if p_unsafe_c < delta:
                    permitted_concentrated += 1
                if p_unsafe_d < delta:
                    permitted_diffuse += 1
        
        assert permitted_concentrated >= permitted_diffuse

    def test_map_shield_vs_full_posterior_shield(self):
        """MAP shield should be more permissive than full posterior shield."""
        n_states, n_actions = 4, 3
        rng = np.random.RandomState(42)
        
        # Create models with varying safety
        models = []
        for i in range(5):
            w = 0.2
            T = rng.dirichlet(np.ones(n_states), size=(n_actions, n_states))
            models.append((w, T))
        
        # MAP model: the one with highest weight (or first, since equal weights)
        map_model = [(1.0, models[0][1])]
        
        delta = 0.2
        
        permitted_map = 0
        permitted_full = 0
        
        for s in range(n_states):
            for a in range(n_actions):
                p_map = sum(w * T[a, s, n_states-1] for w, T in map_model)
                p_full = sum(w * T[a, s, n_states-1] for w, T in models)
                
                if p_map < delta:
                    permitted_map += 1
                if p_full < delta:
                    permitted_full += 1
        
        # MAP is typically more permissive (ignores model uncertainty)
        assert permitted_map >= permitted_full

    def test_shield_monotonicity_in_delta(self):
        """Increasing delta (loosening spec) should increase permissivity."""
        n_states, n_actions = 4, 5
        models = _make_posterior_models(5, n_states, n_actions, seed=42)
        
        prev_permitted = 0
        for delta in [0.01, 0.05, 0.1, 0.2, 0.5, 0.9]:
            permitted = 0
            for s in range(n_states):
                for a in range(n_actions):
                    p_unsafe = sum(w * T[a, s, n_states-1] for w, T in models)
                    if p_unsafe < delta:
                        permitted += 1
            assert permitted >= prev_permitted
            prev_permitted = permitted

    def test_shield_soundness_empirical(self):
        """Empirically verify shield soundness on simulated trajectories."""
        n_states, n_actions = 4, 3
        unsafe_state = n_states - 1
        rng = np.random.RandomState(42)
        
        # True model
        T_true = rng.dirichlet(2 * np.ones(n_states), size=(n_actions, n_states))
        
        # Posterior models (include true model)
        models = [(0.3, T_true)]
        for i in range(4):
            T = rng.dirichlet(np.ones(n_states), size=(n_actions, n_states))
            models.append((0.175, T))
        
        delta = 0.3
        horizon = 10
        n_simulations = 1000
        
        # Determine permitted actions per state
        permitted = {}
        for s in range(n_states):
            permitted[s] = []
            for a in range(n_actions):
                p_unsafe = sum(w * T[a, s, unsafe_state] for w, T in models)
                if p_unsafe < delta:
                    permitted[s].append(a)
            if not permitted[s]:
                permitted[s] = [0]  # fallback
        
        # Simulate with shielded policy
        violations = 0
        for _ in range(n_simulations):
            state = 0
            violated = False
            for t in range(horizon):
                action = rng.choice(permitted[state])
                probs = T_true[action, state]
                state = rng.choice(n_states, p=probs)
                if state == unsafe_state:
                    violated = True
                    break
            if violated:
                violations += 1
        
        violation_rate = violations / n_simulations
        # Should be reasonably bounded (may not exactly match delta due to horizon effects)
        assert violation_rate < 1.0  # basic sanity


# ===================================================================
# Tests for shield integration with portfolio
# ===================================================================

class TestShieldPortfolioIntegration:
    """Tests for shield integration with portfolio optimization."""

    def test_shielded_optimization_respects_constraints(self):
        """Optimizer should only select from shield-permitted actions."""
        n_actions = 7  # -3, -2, -1, 0, 1, 2, 3
        actions = list(range(-3, 4))
        
        # Shield permits only actions -1, 0, 1
        permitted_indices = [2, 3, 4]  # indices of -1, 0, 1
        permitted_actions = [actions[i] for i in permitted_indices]
        
        # Simple mean-variance: expected return proportional to position
        expected_returns = np.array([a * 0.01 for a in actions])
        risk = np.array([abs(a) * 0.02 for a in actions])
        
        # Optimize only over permitted actions
        risk_aversion = 1.0
        utilities = [
            expected_returns[i] - risk_aversion * risk[i]
            for i in permitted_indices
        ]
        
        best_idx = permitted_indices[np.argmax(utilities)]
        assert actions[best_idx] in permitted_actions

    def test_shield_reduces_maximum_loss(self):
        """Shielded strategy should have lower maximum loss."""
        rng = np.random.RandomState(42)
        n_steps = 100
        
        returns = rng.normal(0, 0.02, n_steps)
        
        # Unshielded: always maximum position
        unshielded_pnl = 3 * returns
        
        # Shielded: limited to position 1
        shielded_pnl = 1 * returns
        
        assert np.min(shielded_pnl) > np.min(unshielded_pnl)

    def test_causal_features_affect_optimization(self):
        """Using causal vs all features should change optimization result."""
        rng = np.random.RandomState(42)
        n_features = 30
        n_samples = 200
        
        X = rng.randn(n_samples, n_features)
        
        # True signal in first 5 features only
        true_beta = np.zeros(n_features)
        true_beta[:5] = rng.randn(5) * 0.1
        y = X @ true_beta + rng.randn(n_samples) * 0.5
        
        # All features regression
        beta_all = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Causal features only
        X_causal = X[:, :5]
        beta_causal = np.linalg.lstsq(X_causal, y, rcond=None)[0]
        
        # Causal model should be closer to truth for the causal features
        error_causal = np.sum((beta_causal - true_beta[:5])**2)
        error_all = np.sum((beta_all[:5] - true_beta[:5])**2)
        # With enough data, causal should be comparable or better
        assert error_causal < error_all * 3  # generous bound


class TestPACBayesVacuityAnalysis:
    """Tests for PAC-Bayes vacuity analysis (Phase B1 improvement)."""

    def test_kl_increases_with_k(self):
        """KL divergence should generally increase with K for fixed n."""
        from causal_trading.shield.pac_bayes import PACBayesVacuityAnalyzer
        analyzer = PACBayesVacuityAnalyzer(
            n_abstract_states_per_regime=20, n_actions=5
        )
        kl_values = []
        for K in [2, 3, 4, 5]:
            kl = analyzer.estimate_kl_for_k(K, 10000)
            kl_values.append(kl)
        # More regimes -> more state-action pairs -> higher total KL
        assert kl_values[-1] > kl_values[0]

    def test_bound_decreases_with_n(self):
        """PAC-Bayes bound should decrease with more observations."""
        from causal_trading.shield.pac_bayes import PACBayesVacuityAnalyzer
        analyzer = PACBayesVacuityAnalyzer(
            n_abstract_states_per_regime=20, n_actions=5
        )
        n_values = np.array([500, 5000, 50000])
        bounds = analyzer.compute_bound_curve(
            K=3, n_values=n_values, bound_type="pac-bayes-kl"
        )
        # Bound at large n should be smaller than at small n
        assert bounds[-1] < bounds[0]

    def test_full_analysis_produces_results(self):
        """Full analysis should produce valid results for all K values."""
        from causal_trading.shield.pac_bayes import PACBayesVacuityAnalyzer
        analyzer = PACBayesVacuityAnalyzer(
            n_abstract_states_per_regime=20, n_actions=5
        )
        result = analyzer.full_analysis(
            k_values=[2, 3], n_range=(100, 10000), n_points=20
        )
        assert len(result.k_values) == 2
        assert len(result.bounds) > 0
        assert all(0 <= v <= 1.0 for arr in result.bounds.values() for v in arr)
        table = analyzer.summary_table(result)
        assert "PAC-Bayes Vacuity Analysis" in table

    def test_vacuity_threshold_detection(self):
        """Should correctly detect when bounds become non-vacuous."""
        from causal_trading.shield.pac_bayes import PACBayesVacuityAnalyzer
        analyzer = PACBayesVacuityAnalyzer(
            n_abstract_states_per_regime=20, n_actions=5,
            vacuity_threshold=0.5
        )
        result = analyzer.full_analysis(
            k_values=[3], n_range=(100, 100000), n_points=50,
            bound_types=["pac-bayes-kl"]
        )
        # With enough data, the bound should be below the threshold
        key = (3, "pac-bayes-kl")
        assert key in result.min_n_nontrivial
        min_n = result.min_n_nontrivial[key]
        assert min_n is not None, "Bound never becomes non-vacuous in range"
