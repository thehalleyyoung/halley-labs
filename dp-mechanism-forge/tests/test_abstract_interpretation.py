"""
Comprehensive tests for abstract interpretation framework.

Tests:
- IntervalDomain abstract operations
- OctagonDomain constraint propagation
- Fixpoint iteration convergence
- Widening/narrowing operators
- Abstract CEGIS pruning
"""

import numpy as np
import numpy.testing as npt
import pytest

from dp_forge.verification.abstract_interpretation import (
    AbstractValue,
    IntervalBounds,
    IntervalDomain,
    OctagonDomain,
    PolyhedralDomain,
    PrivacyAbstractTransformer,
    PrivacyDomainLifting,
    AbstractInterpreter,
    fixpoint_iteration,
    abstract_verify_dp,
    abstract_cegis_prune,
    compute_abstract_sensitivities,
)


class TestIntervalBounds:
    """Test IntervalBounds dataclass."""
    
    def test_interval_bounds_creation(self):
        bounds = IntervalBounds(lower=1.0, upper=2.0)
        assert bounds.lower == 1.0
        assert bounds.upper == 2.0
        assert bounds.is_bottom is False
    
    def test_interval_bounds_bottom_on_invalid(self):
        bounds = IntervalBounds(lower=2.0, upper=1.0)
        assert bounds.is_bottom is True
    
    def test_interval_bounds_top(self):
        bounds = IntervalBounds.top()
        assert bounds.lower == -np.inf
        assert bounds.upper == np.inf
    
    def test_interval_bounds_bottom(self):
        bounds = IntervalBounds.bottom()
        assert bounds.is_bottom is True
    
    def test_interval_bounds_singleton(self):
        bounds = IntervalBounds.singleton(5.0)
        assert bounds.lower == 5.0
        assert bounds.upper == 5.0
    
    def test_interval_bounds_contains(self):
        bounds = IntervalBounds(lower=1.0, upper=3.0)
        assert bounds.contains(2.0) is True
        assert bounds.contains(0.5) is False
        assert bounds.contains(4.0) is False
    
    def test_interval_bounds_join(self):
        b1 = IntervalBounds(1.0, 2.0)
        b2 = IntervalBounds(1.5, 3.0)
        
        joined = b1.join(b2)
        
        assert joined.lower == 1.0
        assert joined.upper == 3.0
    
    def test_interval_bounds_join_with_bottom(self):
        b1 = IntervalBounds(1.0, 2.0)
        b2 = IntervalBounds.bottom()
        
        joined = b1.join(b2)
        
        assert joined.lower == 1.0
        assert joined.upper == 2.0
    
    def test_interval_bounds_meet(self):
        b1 = IntervalBounds(1.0, 3.0)
        b2 = IntervalBounds(2.0, 4.0)
        
        met = b1.meet(b2)
        
        assert met.lower == 2.0
        assert met.upper == 3.0
    
    def test_interval_bounds_meet_disjoint(self):
        b1 = IntervalBounds(1.0, 2.0)
        b2 = IntervalBounds(3.0, 4.0)
        
        met = b1.meet(b2)
        
        assert met.is_bottom is True
    
    def test_interval_bounds_widen(self):
        b1 = IntervalBounds(1.0, 2.0)
        b2 = IntervalBounds(0.5, 2.5)
        
        widened = b1.widen(b2, threshold=100.0)
        
        assert widened.lower <= b1.lower
        assert widened.upper >= b1.upper


class TestIntervalDomain:
    """Test IntervalDomain class."""
    
    def test_interval_domain_creation(self):
        domain = IntervalDomain()
        assert len(domain.intervals) == 0
    
    def test_interval_domain_set_get(self):
        domain = IntervalDomain()
        bounds = IntervalBounds(1.0, 2.0)
        
        domain.set_interval("x", bounds)
        retrieved = domain.get_interval("x")
        
        assert retrieved.lower == 1.0
        assert retrieved.upper == 2.0
    
    def test_interval_domain_get_nonexistent(self):
        domain = IntervalDomain()
        bounds = domain.get_interval("nonexistent")
        
        assert bounds.lower == -np.inf
        assert bounds.upper == np.inf
    
    def test_interval_domain_top(self):
        domain = IntervalDomain(intervals={"x": IntervalBounds(1.0, 2.0)})
        top = domain.top()
        
        assert "x" in top.intervals
        assert top.intervals["x"].lower == -np.inf
    
    def test_interval_domain_bottom(self):
        domain = IntervalDomain(intervals={"x": IntervalBounds(1.0, 2.0)})
        bottom = domain.bottom()
        
        assert bottom.is_bottom() is True
    
    def test_interval_domain_is_bottom(self):
        domain = IntervalDomain()
        assert domain.is_bottom() is False
        
        domain.set_interval("x", IntervalBounds.bottom())
        assert domain.is_bottom() is True
    
    def test_interval_domain_join(self):
        d1 = IntervalDomain()
        d1.set_interval("x", IntervalBounds(1.0, 2.0))
        
        d2 = IntervalDomain()
        d2.set_interval("x", IntervalBounds(1.5, 3.0))
        
        joined = d1.join(d2)
        
        x_bounds = joined.get_interval("x")
        assert x_bounds.lower == 1.0
        assert x_bounds.upper == 3.0
    
    def test_interval_domain_meet(self):
        d1 = IntervalDomain()
        d1.set_interval("x", IntervalBounds(1.0, 3.0))
        
        d2 = IntervalDomain()
        d2.set_interval("x", IntervalBounds(2.0, 4.0))
        
        met = d1.meet(d2)
        
        x_bounds = met.get_interval("x")
        assert x_bounds.lower == 2.0
        assert x_bounds.upper == 3.0
    
    def test_interval_domain_widen(self):
        d1 = IntervalDomain()
        d1.set_interval("x", IntervalBounds(1.0, 2.0))
        
        d2 = IntervalDomain()
        d2.set_interval("x", IntervalBounds(0.5, 2.5))
        
        widened = d1.widen(d2)
        
        x_bounds = widened.get_interval("x")
        assert x_bounds.lower <= 1.0
    
    def test_interval_domain_is_less_or_equal(self):
        d1 = IntervalDomain()
        d1.set_interval("x", IntervalBounds(1.5, 2.5))
        
        d2 = IntervalDomain()
        d2.set_interval("x", IntervalBounds(1.0, 3.0))
        
        assert d1.is_less_or_equal(d2) is True
        assert d2.is_less_or_equal(d1) is False


class TestOctagonDomain:
    """Test OctagonDomain class."""
    
    def test_octagon_domain_creation(self):
        domain = OctagonDomain(n_vars=2)
        assert domain.n_vars == 2
        assert domain.matrix.shape == (4, 4)
    
    def test_octagon_domain_top(self):
        domain = OctagonDomain(n_vars=2)
        top = domain.top()
        
        assert np.all(np.isposinf(top.matrix) | (top.matrix == 0.0))
    
    def test_octagon_domain_bottom(self):
        domain = OctagonDomain(n_vars=2)
        bottom = domain.bottom()
        
        assert bottom.is_bottom() == True
    
    def test_octagon_domain_is_bottom(self):
        domain = OctagonDomain(n_vars=2)
        assert domain.is_bottom() == False
        
        domain.matrix[0, 0] = -1.0
        assert domain.is_bottom() == True
    
    def test_octagon_domain_join(self):
        d1 = OctagonDomain(n_vars=2)
        d1.matrix[0, 1] = 5.0
        
        d2 = OctagonDomain(n_vars=2)
        d2.matrix[0, 1] = 10.0
        
        joined = d1.join(d2)
        
        assert joined.matrix[0, 1] == 10.0
    
    def test_octagon_domain_meet(self):
        d1 = OctagonDomain(n_vars=2)
        d1.matrix[0, 1] = 10.0
        
        d2 = OctagonDomain(n_vars=2)
        d2.matrix[0, 1] = 5.0
        
        met = d1.meet(d2)
        
        assert met.matrix[0, 1] == 5.0
    
    def test_octagon_domain_add_constraint(self):
        domain = OctagonDomain(n_vars=2)
        domain.add_constraint(0, 1, 5.0)
        
        assert domain.matrix[0, 3] == 5.0
    
    def test_octagon_domain_get_interval(self):
        domain = OctagonDomain(n_vars=2)
        domain.matrix[0, 1] = 4.0  # x_0 upper bound constraint
        domain.matrix[1, 0] = 2.0  # x_0 lower bound constraint
        
        bounds = domain.get_interval(0)
        
        assert bounds.lower == -1.0
        assert bounds.upper == 2.0


class TestPrivacyAbstractTransformer:
    """Test PrivacyAbstractTransformer."""
    
    def test_transformer_initialization(self):
        transformer = PrivacyAbstractTransformer()
        assert transformer.domain_type == "interval"
    
    def test_abstract_probability(self):
        transformer = PrivacyAbstractTransformer()
        domain = IntervalDomain()
        
        bounds = transformer.abstract_probability(domain, 0.5, 0.01)
        
        assert bounds.lower == 0.49
        assert bounds.upper == 0.51
    
    def test_abstract_probability_clips_to_zero_one(self):
        transformer = PrivacyAbstractTransformer()
        domain = IntervalDomain()
        
        bounds = transformer.abstract_probability(domain, 0.99, 0.02)
        
        assert bounds.lower >= 0.0
        assert bounds.upper <= 1.0
    
    def test_abstract_privacy_loss(self):
        transformer = PrivacyAbstractTransformer()
        domain = IntervalDomain()
        
        domain.set_interval("p", IntervalBounds(0.4, 0.6))
        domain.set_interval("q", IntervalBounds(0.3, 0.5))
        
        loss_bounds = transformer.abstract_privacy_loss(domain, "p", "q")
        
        assert loss_bounds.lower <= loss_bounds.upper
        assert np.isfinite(loss_bounds.lower)
        assert np.isfinite(loss_bounds.upper)
    
    def test_abstract_privacy_loss_handles_zero_q(self):
        transformer = PrivacyAbstractTransformer()
        domain = IntervalDomain()
        
        domain.set_interval("p", IntervalBounds(0.5, 0.5))
        domain.set_interval("q", IntervalBounds(0.0, 0.0))
        
        loss_bounds = transformer.abstract_privacy_loss(domain, "p", "q")
        
        assert loss_bounds.lower == -np.inf or np.isfinite(loss_bounds.lower)
    
    def test_abstract_hockey_stick(self):
        transformer = PrivacyAbstractTransformer()
        domain = IntervalDomain()
        
        p_vars = ["p0", "p1"]
        q_vars = ["q0", "q1"]
        
        for var in p_vars:
            domain.set_interval(var, IntervalBounds(0.4, 0.6))
        for var in q_vars:
            domain.set_interval(var, IntervalBounds(0.3, 0.5))
        
        hs_bounds = transformer.abstract_hockey_stick(domain, p_vars, q_vars, 0.5)
        
        assert hs_bounds.lower >= 0.0
        assert hs_bounds.upper >= hs_bounds.lower
    
    def test_abstract_composition(self):
        transformer = PrivacyAbstractTransformer()
        
        loss_bounds = [
            IntervalBounds(0.4, 0.6),
            IntervalBounds(0.3, 0.5),
            IntervalBounds(0.2, 0.4),
        ]
        
        composed = transformer.abstract_composition(loss_bounds)
        
        assert composed.lower == 0.9
        assert composed.upper == 1.5


class TestFixpointIteration:
    """Test fixpoint iteration algorithm."""
    
    def test_fixpoint_iteration_converges(self):
        initial = IntervalDomain()
        initial.set_interval("x", IntervalBounds(0.0, 1.0))
        
        def transfer(domain):
            return domain
        
        result = fixpoint_iteration(initial, transfer, max_iterations=10)
        
        assert result.is_less_or_equal(initial) or initial.is_less_or_equal(result)
    
    def test_fixpoint_iteration_with_widening(self):
        initial = IntervalDomain()
        initial.set_interval("x", IntervalBounds(0.0, 1.0))
        
        iteration_count = [0]
        
        def transfer(domain):
            iteration_count[0] += 1
            new_domain = IntervalDomain()
            current_bounds = domain.get_interval("x")
            new_domain.set_interval("x", IntervalBounds(
                current_bounds.lower - 0.1,
                current_bounds.upper + 0.1
            ))
            return new_domain
        
        result = fixpoint_iteration(initial, transfer, max_iterations=100, use_widening=True)
        
        assert iteration_count[0] <= 100
    
    def test_fixpoint_iteration_detects_convergence(self):
        initial = IntervalDomain()
        initial.set_interval("x", IntervalBounds(1.0, 2.0))
        
        def identity_transfer(domain):
            return domain
        
        result = fixpoint_iteration(initial, identity_transfer, max_iterations=100)
        
        assert result.get_interval("x").lower == 1.0
        assert result.get_interval("x").upper == 2.0


class TestAbstractVerifyDP:
    """Test abstract DP verification."""
    
    def test_abstract_verify_dp_valid_mechanism(self):
        prob_table = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        edges = [(0, 1)]
        epsilon = 1.0
        delta = 0.0
        
        is_valid, domain = abstract_verify_dp(prob_table, edges, epsilon, delta)
        
        assert is_valid is True
        assert isinstance(domain, IntervalDomain)
    
    def test_abstract_verify_dp_invalid_mechanism(self):
        prob_table = np.array([
            [0.99, 0.01],
            [0.01, 0.99]
        ])
        edges = [(0, 1)]
        epsilon = 0.1
        delta = 0.0
        
        is_valid, domain = abstract_verify_dp(prob_table, edges, epsilon, delta)
        
        assert is_valid is False
    
    def test_abstract_verify_dp_with_delta(self):
        prob_table = np.array([
            [0.6, 0.4],
            [0.4, 0.6]
        ])
        edges = [(0, 1)]
        epsilon = 0.5
        delta = 0.1
        
        is_valid, domain = abstract_verify_dp(prob_table, edges, epsilon, delta)
        
        assert isinstance(domain, IntervalDomain)
    
    def test_abstract_verify_dp_builds_domain(self):
        prob_table = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        edges = [(0, 1)]
        
        is_valid, domain = abstract_verify_dp(prob_table, edges, 1.0, 0.0)
        
        assert len(domain.intervals) > 0


class TestAbstractCEGISPrune:
    """Test abstract CEGIS pruning."""
    
    def test_abstract_cegis_prune_basic(self):
        prob_table = np.array([
            [0.5, 0.5],
            [0.5, 0.5],
            [0.4, 0.6]
        ])
        candidate_pairs = [(0, 1), (0, 2), (1, 2)]
        epsilon = 1.0
        delta = 0.0
        
        pruned = abstract_cegis_prune(prob_table, candidate_pairs, epsilon, delta)
        
        assert len(pruned) <= len(candidate_pairs)
    
    def test_abstract_cegis_prune_removes_safe_pairs(self):
        prob_table = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        candidate_pairs = [(0, 1)]
        epsilon = 1.0
        delta = 0.0
        
        pruned = abstract_cegis_prune(prob_table, candidate_pairs, epsilon, delta)
        
        assert len(pruned) == 0
    
    def test_abstract_cegis_prune_keeps_violating_pairs(self):
        prob_table = np.array([
            [0.9, 0.1],
            [0.1, 0.9]
        ])
        candidate_pairs = [(0, 1)]
        epsilon = 0.1
        delta = 0.0
        
        pruned = abstract_cegis_prune(prob_table, candidate_pairs, epsilon, delta)
        
        assert len(pruned) == 1


class TestPolyhedralDomain:
    """Test PolyhedralDomain class."""
    
    def test_polyhedral_domain_creation(self):
        domain = PolyhedralDomain(n_vars=2)
        assert domain.n_vars == 2
        assert domain.A.shape[1] == 2
    
    def test_polyhedral_domain_top(self):
        domain = PolyhedralDomain(n_vars=2)
        top = domain.top()
        
        assert len(top.b) == 0
    
    def test_polyhedral_domain_add_constraint(self):
        domain = PolyhedralDomain(n_vars=2)
        
        a = np.array([1.0, 0.0])
        b = 5.0
        
        domain.add_constraint(a, b)
        
        assert domain.A.shape[0] == 1
        assert domain.b[0] == 5.0
    
    def test_polyhedral_domain_meet(self):
        d1 = PolyhedralDomain(n_vars=2)
        d1.add_constraint(np.array([1.0, 0.0]), 5.0)
        
        d2 = PolyhedralDomain(n_vars=2)
        d2.add_constraint(np.array([0.0, 1.0]), 3.0)
        
        met = d1.meet(d2)
        
        assert met.A.shape[0] == 2


class TestPrivacyDomainLifting:
    """Test PrivacyDomainLifting."""
    
    def test_lifting_initialization(self):
        lifting = PrivacyDomainLifting()
        assert lifting.transformer is not None
    
    def test_lift_dp_constraint(self):
        lifting = PrivacyDomainLifting()
        
        domain = IntervalDomain()
        domain.set_interval("p_0_0", IntervalBounds(0.5, 0.7))
        domain.set_interval("p_1_0", IntervalBounds(0.3, 0.5))
        
        refined = lifting.lift_dp_constraint(domain, 0, 1, epsilon=1.0, n_outputs=1)
        
        assert isinstance(refined, IntervalDomain)
    
    def test_lift_composition(self):
        lifting = PrivacyDomainLifting()
        
        d1 = IntervalDomain()
        d1.set_interval("x", IntervalBounds(1.0, 2.0))
        
        d2 = IntervalDomain()
        d2.set_interval("x", IntervalBounds(1.5, 2.5))
        
        composed = lifting.lift_composition([d1, d2])
        
        assert isinstance(composed, IntervalDomain)
        x_bounds = composed.get_interval("x")
        assert x_bounds.lower == 1.0
        assert x_bounds.upper == 2.5


class TestComputeAbstractSensitivities:
    """Test sensitivity computation."""
    
    def test_compute_abstract_sensitivities(self):
        query_matrix = np.array([
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.0],
        ])
        adjacency = [(0, 1)]
        
        domain = compute_abstract_sensitivities(query_matrix, adjacency)
        
        assert isinstance(domain, IntervalDomain)
        assert len(domain.intervals) > 0
    
    def test_sensitivities_are_nonnegative(self):
        query_matrix = np.array([
            [1.0, 2.0],
            [2.0, 1.0],
        ])
        adjacency = [(0, 1)]
        
        domain = compute_abstract_sensitivities(query_matrix, adjacency)
        
        for bounds in domain.intervals.values():
            assert bounds.lower >= 0.0


class TestAbstractInterpreter:
    """Test AbstractInterpreter class."""
    
    def test_interpreter_initialization(self):
        interpreter = AbstractInterpreter(domain_type="interval")
        assert interpreter.domain_type == "interval"
    
    def test_analyze_mechanism(self):
        interpreter = AbstractInterpreter()
        
        prob_table = np.array([
            [0.5, 0.5],
            [0.5, 0.5]
        ])
        epsilon = 1.0
        delta = 0.0
        
        is_private, domain = interpreter.analyze_mechanism(prob_table, epsilon, delta)
        
        assert isinstance(is_private, bool)
        assert isinstance(domain, IntervalDomain)


class TestPropertyTests:
    """Property-based tests for abstract interpretation."""
    
    def test_join_is_commutative(self):
        d1 = IntervalDomain()
        d1.set_interval("x", IntervalBounds(1.0, 2.0))
        
        d2 = IntervalDomain()
        d2.set_interval("x", IntervalBounds(1.5, 3.0))
        
        j1 = d1.join(d2)
        j2 = d2.join(d1)
        
        assert j1.get_interval("x").lower == j2.get_interval("x").lower
        assert j1.get_interval("x").upper == j2.get_interval("x").upper
    
    def test_join_is_associative(self):
        d1 = IntervalDomain()
        d1.set_interval("x", IntervalBounds(1.0, 2.0))
        
        d2 = IntervalDomain()
        d2.set_interval("x", IntervalBounds(1.5, 3.0))
        
        d3 = IntervalDomain()
        d3.set_interval("x", IntervalBounds(2.0, 4.0))
        
        j1 = d1.join(d2).join(d3)
        j2 = d1.join(d2.join(d3))
        
        assert j1.get_interval("x").lower == j2.get_interval("x").lower
        assert j1.get_interval("x").upper == j2.get_interval("x").upper
    
    def test_meet_is_commutative(self):
        d1 = IntervalDomain()
        d1.set_interval("x", IntervalBounds(1.0, 3.0))
        
        d2 = IntervalDomain()
        d2.set_interval("x", IntervalBounds(2.0, 4.0))
        
        m1 = d1.meet(d2)
        m2 = d2.meet(d1)
        
        assert m1.get_interval("x").lower == m2.get_interval("x").lower
        assert m1.get_interval("x").upper == m2.get_interval("x").upper
    
    def test_widening_ensures_convergence(self):
        d1 = IntervalDomain()
        d1.set_interval("x", IntervalBounds(0.0, 1.0))
        
        current = d1
        for i in range(100):
            next_d = IntervalDomain()
            current_bounds = current.get_interval("x")
            next_d.set_interval("x", IntervalBounds(
                current_bounds.lower - 0.1,
                current_bounds.upper + 0.1
            ))
            
            if i > 50:
                current = current.widen(next_d)
            else:
                current = next_d
            
            if current.get_interval("x").lower == -np.inf:
                break
        
        assert current.get_interval("x").lower == -np.inf or current.get_interval("x").lower < -100


class TestSoundness:
    """Test soundness properties of abstract interpretation."""
    
    def test_abstract_over_approximates_concrete(self):
        np.random.seed(42)
        
        concrete_p = 0.6
        tolerance = 0.01
        
        transformer = PrivacyAbstractTransformer()
        domain = IntervalDomain()
        
        abstract_bounds = transformer.abstract_probability(domain, concrete_p, tolerance)
        
        assert abstract_bounds.contains(concrete_p)
    
    def test_abstract_privacy_loss_sound(self):
        np.random.seed(43)
        
        p_true = 0.6
        q_true = 0.4
        
        transformer = PrivacyAbstractTransformer()
        domain = IntervalDomain()
        
        domain.set_interval("p", IntervalBounds(p_true - 0.01, p_true + 0.01))
        domain.set_interval("q", IntervalBounds(q_true - 0.01, q_true + 0.01))
        
        loss_bounds = transformer.abstract_privacy_loss(domain, "p", "q")
        
        true_loss = np.log(p_true / q_true)
        
        assert loss_bounds.lower <= true_loss + 1e-10
        assert loss_bounds.upper >= true_loss - 1e-10
    
    def test_abstract_verification_is_conservative(self):
        prob_table = np.array([
            [0.5, 0.5],
            [0.4, 0.6]
        ])
        edges = [(0, 1)]
        epsilon = 0.5
        
        is_valid_abstract, _ = abstract_verify_dp(prob_table, edges, epsilon, 0.0)
        
        from dp_forge.verifier import verify, VerificationMode
        result_concrete = verify(
            prob_table, epsilon, 0.0, edges,
            mode=VerificationMode.MOST_VIOLATING
        )
        
        if result_concrete.valid:
            pass
        else:
            assert is_valid_abstract is False
