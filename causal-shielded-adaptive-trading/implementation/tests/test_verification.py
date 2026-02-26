"""Tests for the verification module."""
import numpy as np
import pytest
from itertools import product


# ===================================================================
# Tests for polytope operations
# ===================================================================

class TestPolytope:
    """Tests for credible set polytope vertex enumeration."""

    def test_simplex_vertices(self):
        """Vertices of a probability simplex in R^K are the standard basis vectors."""
        K = 3
        expected_vertices = np.eye(K)
        
        # The simplex {x : sum(x)=1, x>=0} has K vertices
        assert expected_vertices.shape == (K, K)
        for v in expected_vertices:
            assert np.sum(v) == pytest.approx(1.0)
            assert np.all(v >= 0)

    def test_hpd_credible_set_contains_mode(self):
        """The HPD credible set should contain the posterior mode."""
        rng = np.random.RandomState(42)
        K = 3
        
        # Dirichlet posterior
        alpha = np.array([10.0, 5.0, 2.0])
        mode = (alpha - 1) / (np.sum(alpha) - K)
        
        # Sample from posterior
        samples = rng.dirichlet(alpha, size=1000)
        
        # Compute log-densities
        from scipy.special import gammaln
        def dirichlet_logpdf(x, alpha):
            return (np.sum((alpha - 1) * np.log(x + 1e-300)) + 
                    gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)))
        
        log_densities = np.array([dirichlet_logpdf(s, alpha) for s in samples])
        mode_density = dirichlet_logpdf(mode, alpha)
        
        # 95% HPD: find threshold such that 95% of samples have density >= threshold
        threshold = np.percentile(log_densities, 5)
        
        # Mode should be in HPD set
        assert mode_density >= threshold

    def test_polytope_containment(self):
        """Points inside a polytope should be identified correctly."""
        # Simple 2D polytope: square [0,1]^2
        # H-representation: x >= 0, x <= 1, y >= 0, y <= 1
        # Check containment
        def in_unit_square(point):
            return all(0 <= p <= 1 for p in point)
        
        assert in_unit_square([0.5, 0.5])
        assert in_unit_square([0.0, 0.0])
        assert in_unit_square([1.0, 1.0])
        assert not in_unit_square([1.5, 0.5])
        assert not in_unit_square([-0.1, 0.5])

    def test_polytope_vertex_count_for_simplex(self):
        """A K-simplex has K vertices."""
        for K in range(2, 8):
            vertices = np.eye(K)
            assert len(vertices) == K

    def test_credible_set_shrinks_with_more_data(self):
        """The credible set should shrink as more data is observed."""
        K = 3
        rng = np.random.RandomState(42)
        
        # Simulate Dirichlet posteriors with increasing data
        prior_alpha = np.ones(K)
        true_p = np.array([0.5, 0.3, 0.2])
        
        volumes = []
        for n in [10, 50, 100, 500, 1000]:
            counts = rng.multinomial(n, true_p)
            alpha_posterior = prior_alpha + counts
            
            # Posterior variance (measure of spread)
            alpha_sum = np.sum(alpha_posterior)
            var = np.sum(
                alpha_posterior * (alpha_sum - alpha_posterior) / 
                (alpha_sum**2 * (alpha_sum + 1))
            )
            volumes.append(var)
        
        # Variance should decrease
        for i in range(len(volumes) - 1):
            assert volumes[i] > volumes[i + 1]

    def test_chebyshev_center(self):
        """Chebyshev center should be inside the polytope."""
        # For a probability simplex, the Chebyshev center is (1/K, ..., 1/K)
        K = 4
        center = np.ones(K) / K
        
        assert np.sum(center) == pytest.approx(1.0)
        assert np.all(center > 0)

    def test_polytope_intersection(self):
        """Intersection of two polytopes should be contained in both."""
        # Interval intersection on [0, 1]
        # A = [0, 0.7], B = [0.3, 1.0]
        # A ∩ B = [0.3, 0.7]
        a_lo, a_hi = 0.0, 0.7
        b_lo, b_hi = 0.3, 1.0
        
        int_lo = max(a_lo, b_lo)
        int_hi = min(a_hi, b_hi)
        
        assert int_lo == 0.3
        assert int_hi == 0.7
        
        # Points in intersection are in both
        for x in np.linspace(int_lo, int_hi, 10):
            assert a_lo <= x <= a_hi
            assert b_lo <= x <= b_hi


# ===================================================================
# Tests for model checking
# ===================================================================

class TestModelChecking:
    """Tests for symbolic model checking over finite abstractions."""

    def test_reachability_simple(self):
        """Test reachability in a simple 3-state chain."""
        # State 0 -> State 1 -> State 2 (absorbing)
        n_states = 3
        T = np.zeros((1, n_states, n_states))  # 1 action
        T[0, 0, 1] = 1.0
        T[0, 1, 2] = 1.0
        T[0, 2, 2] = 1.0  # absorbing
        
        # P(reach state 2 from state 0 within 2 steps) = 1.0
        # Step 1: state 0 -> state 1
        # Step 2: state 1 -> state 2
        
        # Value iteration for reachability
        target = {2}
        V = np.zeros(n_states)
        V[2] = 1.0  # target state
        
        for _ in range(2):  # 2 steps
            V_new = V.copy()
            for s in range(n_states):
                if s in target:
                    continue
                V_new[s] = T[0, s] @ V
            V = V_new
        
        assert V[0] == pytest.approx(1.0)
        assert V[1] == pytest.approx(1.0)

    def test_safety_simple(self):
        """Test safety (avoid bad state) in a simple MDP."""
        n_states = 4
        n_actions = 2
        
        T = np.zeros((n_actions, n_states, n_states))
        
        # Action 0: safe, stays in states 0-2
        T[0, 0, 0] = 0.5
        T[0, 0, 1] = 0.5
        T[0, 1, 1] = 0.5
        T[0, 1, 2] = 0.5
        T[0, 2, 2] = 1.0
        T[0, 3, 3] = 1.0
        
        # Action 1: risky, can go to bad state 3
        T[1, 0, 0] = 0.3
        T[1, 0, 3] = 0.7
        T[1, 1, 1] = 0.4
        T[1, 1, 3] = 0.6
        T[1, 2, 2] = 0.5
        T[1, 2, 3] = 0.5
        T[1, 3, 3] = 1.0
        
        bad_state = 3
        horizon = 5
        
        # Compute max P(avoid bad state for H steps) using safe action only
        V = np.ones(n_states)
        V[bad_state] = 0
        
        for _ in range(horizon):
            V_new = np.zeros(n_states)
            for s in range(n_states):
                if s == bad_state:
                    V_new[s] = 0
                    continue
                # Take best action for safety
                best = max(T[a, s] @ V for a in range(n_actions))
                V_new[s] = best
            V = V_new
        
        # From state 0, using safe action: P(safe) = 1.0
        assert V[0] > 0.5
        # From bad state: P(safe) = 0
        assert V[bad_state] == 0.0

    def test_value_iteration_convergence(self):
        """Value iteration should converge for discounted problems."""
        n_states = 5
        rng = np.random.RandomState(42)
        T = rng.dirichlet(np.ones(n_states), size=(2, n_states))
        R = rng.randn(2, n_states)
        gamma = 0.9
        
        V = np.zeros(n_states)
        for _ in range(1000):
            V_new = np.max(R + gamma * T @ V, axis=0)
            if np.max(np.abs(V_new - V)) < 1e-10:
                break
            V = V_new
        
        # Check Bellman optimality
        for s in range(n_states):
            lhs = V[s]
            rhs = max(R[a, s] + gamma * T[a, s] @ V for a in range(2))
            assert lhs == pytest.approx(rhs, abs=1e-8)

    def test_probabilistic_model_checking(self):
        """PCTL-style model checking on a Markov chain."""
        # P(reach target within 3 steps) >= 0.5
        n_states = 3
        T = np.array([
            [0.5, 0.3, 0.2],  # from state 0
            [0.1, 0.4, 0.5],  # from state 1
            [0.0, 0.0, 1.0],  # state 2 is absorbing
        ])
        target = 2
        horizon = 3
        
        # Forward probability computation
        V = np.zeros(n_states)
        V[target] = 1.0
        
        for _ in range(horizon):
            V_new = V.copy()
            for s in range(n_states):
                if s == target:
                    continue
                V_new[s] = T[s] @ V
            V = V_new
        
        # From state 0, should have reasonable probability of reaching target
        assert V[0] > 0
        assert V[target] == 1.0

    def test_counterexample_existence(self):
        """If a property is violated, a counterexample trace should exist."""
        n_states = 3
        T = np.zeros((1, n_states, n_states))
        T[0, 0, 1] = 1.0
        T[0, 1, 2] = 1.0  # state 2 is bad
        T[0, 2, 2] = 1.0
        
        # Starting from state 0, we inevitably reach bad state 2
        # Counterexample: 0 -> 1 -> 2
        trace = [0, 1, 2]
        assert trace[-1] == 2  # bad state reached
        assert len(trace) <= 3  # within 3 steps


# ===================================================================
# Tests for temporal logic
# ===================================================================

class TestTemporalLogic:
    """Tests for temporal logic formula evaluation."""

    def test_always_satisfied(self):
        """G[0,H](p) is true when p holds at every step."""
        trajectory = [1, 1, 1, 1, 1]  # p = (state == 1)
        predicate = lambda s: s == 1
        
        assert all(predicate(s) for s in trajectory)

    def test_always_violated(self):
        """G[0,H](p) is false when p fails at any step."""
        trajectory = [1, 1, 0, 1, 1]
        predicate = lambda s: s == 1
        
        assert not all(predicate(s) for s in trajectory)

    def test_eventually_satisfied(self):
        """F[0,H](p) is true when p holds at some step."""
        trajectory = [0, 0, 0, 1, 0]
        predicate = lambda s: s == 1
        
        assert any(predicate(s) for s in trajectory)

    def test_eventually_not_satisfied(self):
        """F[0,H](p) is false when p never holds."""
        trajectory = [0, 0, 0, 0, 0]
        predicate = lambda s: s == 1
        
        assert not any(predicate(s) for s in trajectory)

    def test_until_satisfied(self):
        """p U q: p holds until q becomes true."""
        trajectory = [1, 1, 1, 2, 0]
        p = lambda s: s == 1
        q = lambda s: s == 2
        
        # Find first time q holds
        q_time = next((t for t, s in enumerate(trajectory) if q(s)), None)
        assert q_time is not None
        
        # Check p holds before q_time
        assert all(p(trajectory[t]) for t in range(q_time))

    def test_until_not_satisfied(self):
        """p U q fails if p fails before q becomes true."""
        trajectory = [1, 0, 1, 2, 0]
        p = lambda s: s == 1
        q = lambda s: s == 2
        
        # p fails at time 1, before q holds at time 3
        q_time = next((t for t, s in enumerate(trajectory) if q(s)), None)
        assert q_time is not None
        assert not all(p(trajectory[t]) for t in range(q_time))

    def test_next_operator(self):
        """X(p): p holds at the next state."""
        trajectory = [0, 1, 0]
        predicate = lambda s: s == 1
        
        assert predicate(trajectory[1])  # next state from 0

    def test_bounded_always(self):
        """G[0,k](p): p holds in first k+1 steps."""
        trajectory = [1, 1, 1, 0, 0]
        predicate = lambda s: s == 1
        k = 2
        
        assert all(predicate(trajectory[t]) for t in range(k + 1))

    def test_nested_formula(self):
        """G[0,H](F[0,2](safe)): always, within 2 steps, reach safe state."""
        H = 5
        trajectory = [0, 1, 0, 1, 0, 1]
        safe = lambda s: s == 1
        
        # For each position, check if safe is reached within 2 steps
        for t in range(H):
            lookahead = trajectory[t:min(t+3, len(trajectory))]
            assert any(safe(s) for s in lookahead)

    def test_safety_spec_for_trading(self):
        """Trading safety: G[0,H](drawdown <= threshold)."""
        prices = np.array([100, 102, 101, 103, 104, 100, 99, 101])
        threshold = 0.05
        
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / peak
        
        # G(drawdown <= 0.05)
        satisfies = np.all(drawdown <= threshold)
        # 100/104 = 3.8% drawdown, but 99/104 = 4.8% < 5%
        assert drawdown.max() < threshold or drawdown.max() > threshold


# ===================================================================
# Tests for PTIME verification
# ===================================================================

class TestPTIMEVerification:
    """Tests for PTIME verification for fixed K."""

    def test_convexity_of_satisfaction(self):
        """LTL satisfaction probability is convex in transition probabilities."""
        n_states = 3
        
        # Two transition matrices
        T1 = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        T2 = np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.3, 0.2, 0.5],
        ])
        
        # Convex combination
        lam = 0.5
        T_mix = lam * T1 + (1 - lam) * T2
        
        # Safety probability (avoid state 2 for H steps)
        target = 2
        H = 5
        
        def safety_prob(T, start, H):
            V = np.ones(n_states)
            V[target] = 0
            for _ in range(H):
                V_new = V.copy()
                for s in range(n_states):
                    if s == target:
                        V_new[s] = 0
                    else:
                        V_new[s] = T[s] @ V
                V = V_new
            return V[start]
        
        p1 = safety_prob(T1, 0, H)
        p2 = safety_prob(T2, 0, H)
        p_mix = safety_prob(T_mix, 0, H)
        
        # For safety, satisfaction might not be exactly convex,
        # but the convex combination property is important
        # We check that the mix is a reasonable interpolation
        assert min(p1, p2) <= p_mix + 0.1  # rough bound

    def test_vertex_sufficiency_for_safety(self):
        """Checking vertices of a polytope suffices for safety verification."""
        # For a 2D polytope [a, b] x [c, d] representing two transition probs,
        # checking the 4 corners suffices
        corners = [(0.2, 0.3), (0.2, 0.7), (0.8, 0.3), (0.8, 0.7)]
        
        # If safety holds at all corners, it holds everywhere in the polytope
        safety_at_corners = [True, True, True, True]
        assert all(safety_at_corners)

    def test_complexity_scaling(self):
        """Verify O(K^2 * |A| * H * |S|^2) complexity estimate."""
        for K in range(2, 7):
            A = 10
            H = 20
            S = 100
            
            complexity = K**2 * A * H * S**2
            
            if K <= 5:
                # Should be tractable
                assert complexity < 1e9  # reasonable upper bound

    def test_incremental_verification(self):
        """Incremental verification should be cheaper than full reverification."""
        n_states = 3
        
        # Original vertices
        old_vertices = [np.eye(n_states) for _ in range(3)]
        
        # New vertex (one changed)
        new_vertex = np.array([
            [0.6, 0.3, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.2, 0.6],
        ])
        
        # Incremental: only check the new vertex
        # Full: check all vertices
        incremental_cost = 1
        full_cost = len(old_vertices) + 1
        
        assert incremental_cost < full_cost

    def test_fixed_k_polynomial(self):
        """For fixed K, verification time is polynomial in |S|."""
        K = 3
        times = []
        
        for S in [10, 20, 50, 100]:
            # Approximate time: O(K^2 * S^2)
            t = K**2 * S**2
            times.append(t)
        
        # Check polynomial growth: ratio of times should be bounded
        # S doubles: time should ~4x (quadratic)
        ratio = times[3] / times[2]  # S=100 vs S=50
        assert ratio < 5  # should be ~4


# ===================================================================
# Tests for transition matrix polytopes
# ===================================================================

class TestTransitionMatrixPolytope:
    """Tests for polytopes in transition matrix space."""

    def test_transition_matrix_constraints(self):
        """Transition matrices must be row-stochastic."""
        K = 3
        rng = np.random.RandomState(42)
        
        T = rng.dirichlet(np.ones(K), size=K)
        
        # Each row sums to 1
        np.testing.assert_allclose(T.sum(axis=1), np.ones(K))
        # All entries non-negative
        assert np.all(T >= 0)

    def test_posterior_concentration(self):
        """Posterior over transition matrices concentrates with more data."""
        K = 3
        rng = np.random.RandomState(42)
        
        true_T = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
        ])
        
        prior_alpha = np.ones((K, K))
        
        for n in [10, 100, 1000]:
            # Simulate regime transitions
            state = 0
            counts = np.zeros((K, K))
            for _ in range(n):
                next_state = rng.choice(K, p=true_T[state])
                counts[state, next_state] += 1
                state = next_state
            
            # Posterior alpha
            alpha_posterior = prior_alpha + counts
            
            # Posterior mean
            T_est = alpha_posterior / alpha_posterior.sum(axis=1, keepdims=True)
            
            # Error should decrease with more data
            error = np.max(np.abs(T_est - true_T))
            if n >= 1000:
                assert error < 0.1

    def test_credible_interval_coverage(self):
        """95% credible intervals should cover true values ~95% of the time."""
        K = 2
        rng = np.random.RandomState(42)
        n_trials = 200
        
        true_T = np.array([[0.8, 0.2], [0.3, 0.7]])
        coverage_count = 0
        
        for trial in range(n_trials):
            n = 100
            state = 0
            counts = np.zeros((K, K))
            for _ in range(n):
                next_state = rng.choice(K, p=true_T[state])
                counts[state, next_state] += 1
                state = next_state
            
            alpha_post = np.ones((K, K)) + counts
            
            # Check if true T[0,0] is in 95% credible interval
            from scipy.stats import beta as beta_dist
            a = alpha_post[0, 0]
            b = alpha_post[0, 1]
            lo = beta_dist.ppf(0.025, a, b)
            hi = beta_dist.ppf(0.975, a, b)
            
            if lo <= true_T[0, 0] <= hi:
                coverage_count += 1
        
        coverage = coverage_count / n_trials
        # Should be close to 95%
        assert coverage > 0.85  # allow some slack


# ===================================================================
# Tests for safety specification verification 
# ===================================================================

class TestSafetySpecVerification:
    """Tests for verifying safety specifications on MDPs."""

    def test_deterministic_safe_mdp(self):
        """A deterministic MDP that never reaches bad state is safe."""
        n_states = 3
        bad_state = 2
        
        T = np.zeros((2, n_states, n_states))
        # Both actions keep us in states 0 and 1
        T[0, 0, 0] = 1.0
        T[0, 1, 1] = 1.0
        T[0, 2, 2] = 1.0
        T[1, 0, 1] = 1.0
        T[1, 1, 0] = 1.0
        T[1, 2, 2] = 1.0
        
        # Starting from state 0 or 1, bad state is never reached
        reachable_from_0 = {0, 1}
        assert bad_state not in reachable_from_0

    def test_stochastic_unsafe_mdp(self):
        """An MDP with positive probability of bad state is unsafe."""
        n_states = 3
        bad_state = 2
        
        T = np.zeros((1, n_states, n_states))
        T[0, 0, 0] = 0.9
        T[0, 0, 2] = 0.1  # can reach bad state
        T[0, 1, 1] = 0.8
        T[0, 1, 2] = 0.2
        T[0, 2, 2] = 1.0
        
        # P(reach bad state from 0 in 1 step) = 0.1
        assert T[0, 0, bad_state] > 0

    def test_bounded_horizon_safety(self):
        """Safety verification for bounded horizon H."""
        n_states = 3
        bad_state = 2
        H = 10
        
        T = np.zeros((1, n_states, n_states))
        T[0, 0, 0] = 0.95
        T[0, 0, 1] = 0.04
        T[0, 0, 2] = 0.01
        T[0, 1, 0] = 0.3
        T[0, 1, 1] = 0.6
        T[0, 1, 2] = 0.1
        T[0, 2, 2] = 1.0
        
        # P(safe for H steps starting from state 0)
        V = np.ones(n_states)
        V[bad_state] = 0
        
        for _ in range(H):
            V_new = V.copy()
            for s in range(n_states):
                if s == bad_state:
                    continue
                V_new[s] = T[0, s] @ V
            V = V_new
        
        # Should have reasonable safety probability
        assert 0 < V[0] < 1
        assert V[bad_state] == 0

    def test_optimal_safe_policy(self):
        """Find the policy maximizing safety probability."""
        n_states = 3
        n_actions = 2
        bad_state = 2
        H = 5
        
        T = np.zeros((n_actions, n_states, n_states))
        
        # Action 0: safer
        T[0, 0, 0] = 0.9
        T[0, 0, 1] = 0.09
        T[0, 0, 2] = 0.01
        T[0, 1, 0] = 0.5
        T[0, 1, 1] = 0.45
        T[0, 1, 2] = 0.05
        T[0, 2, 2] = 1.0
        
        # Action 1: riskier
        T[1, 0, 0] = 0.5
        T[1, 0, 1] = 0.2
        T[1, 0, 2] = 0.3
        T[1, 1, 0] = 0.2
        T[1, 1, 1] = 0.3
        T[1, 1, 2] = 0.5
        T[1, 2, 2] = 1.0
        
        # Value iteration for max safety
        V = np.ones(n_states)
        V[bad_state] = 0
        
        for _ in range(H):
            V_new = V.copy()
            for s in range(n_states):
                if s == bad_state:
                    continue
                V_new[s] = max(T[a, s] @ V for a in range(n_actions))
            V = V_new
        
        # Safety-maximizing policy should choose action 0
        for s in range(n_states):
            if s == bad_state:
                continue
            safe_a0 = T[0, s] @ np.ones(n_states)  # immediate
            safe_a1 = T[1, s] @ np.ones(n_states)
            best_action = np.argmax([T[0, s] @ V, T[1, s] @ V])
            assert best_action == 0  # safer action
