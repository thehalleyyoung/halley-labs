"""
Comprehensive tests for Pufferfish privacy framework.

Tests PufferfishFramework, PufferfishMechanism, WassersteinMechanism,
discriminative pairs, LP synthesis, and custom privacy guarantees.
"""

import math
import pytest
import numpy as np
import numpy.testing as npt
from hypothesis import given, strategies as st, settings

from dp_forge.mechanisms.pufferfish import (
    DiscriminativePair,
    PufferfishFramework,
    PufferfishMechanism,
    WassersteinMechanism,
)
from dp_forge.exceptions import ConfigurationError, InvalidMechanismError


class TestDiscriminativePair:
    """Test DiscriminativePair dataclass."""
    
    def test_initialization(self):
        """Test pair initialization."""
        pair = DiscriminativePair(
            secret1="healthy",
            secret2="diabetes",
            epsilon=1.0,
            description="Health status privacy",
        )
        
        assert pair.secret1 == "healthy"
        assert pair.secret2 == "diabetes"
        assert pair.epsilon == 1.0
        assert pair.description == "Health status privacy"
    
    def test_initialization_errors(self):
        """Test initialization error handling."""
        with pytest.raises(ValueError):
            DiscriminativePair("s1", "s2", epsilon=-1.0)
        
        with pytest.raises(ValueError):
            DiscriminativePair("s1", "s2", epsilon=0.0)
    
    def test_equality(self):
        """Test pair equality."""
        pair1 = DiscriminativePair("A", "B", epsilon=1.0)
        pair2 = DiscriminativePair("A", "B", epsilon=1.0)
        pair3 = DiscriminativePair("B", "A", epsilon=1.0)  # Symmetric
        pair4 = DiscriminativePair("A", "B", epsilon=2.0)
        
        assert pair1 == pair2
        assert pair1 == pair3  # Symmetric
        assert pair1 != pair4  # Different epsilon
    
    def test_hashing(self):
        """Test pair hashing."""
        pair1 = DiscriminativePair("A", "B", epsilon=1.0)
        pair2 = DiscriminativePair("B", "A", epsilon=1.0)
        
        # Symmetric pairs should have same hash
        assert hash(pair1) == hash(pair2)
        
        # Can use in sets
        pair_set = {pair1, pair2}
        assert len(pair_set) == 1  # Only one unique pair


class TestPufferfishFramework:
    """Test PufferfishFramework."""
    
    def test_initialization(self):
        """Test framework initialization."""
        secrets = ["healthy", "diabetes", "heart_disease"]
        pairs = [
            DiscriminativePair("healthy", "diabetes", epsilon=1.0),
            DiscriminativePair("healthy", "heart_disease", epsilon=1.0),
        ]
        
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
        
        assert framework.num_secrets == 3
        assert framework.num_pairs == 2
        assert framework.num_scenarios == 1
    
    def test_initialization_errors(self):
        """Test initialization error handling."""
        # Empty secrets
        with pytest.raises(ConfigurationError):
            PufferfishFramework(secrets=[], pairs=[])
        
        # Empty pairs
        with pytest.raises(ConfigurationError):
            PufferfishFramework(secrets=["A", "B"], pairs=[])
        
        # Invalid pair reference
        secrets = ["A", "B"]
        pairs = [DiscriminativePair("A", "C", epsilon=1.0)]  # C not in secrets
        
        with pytest.raises(ConfigurationError):
            PufferfishFramework(secrets=secrets, pairs=pairs)
    
    def test_max_epsilon(self):
        """Test max epsilon computation."""
        secrets = ["A", "B", "C"]
        pairs = [
            DiscriminativePair("A", "B", epsilon=1.0),
            DiscriminativePair("B", "C", epsilon=2.0),
            DiscriminativePair("A", "C", epsilon=0.5),
        ]
        
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
        
        assert framework.max_epsilon() == 2.0
    
    def test_is_complete(self):
        """Test completeness checking."""
        secrets = ["A", "B", "C"]
        
        # Complete: all pairs
        complete_pairs = [
            DiscriminativePair("A", "B", epsilon=1.0),
            DiscriminativePair("B", "C", epsilon=1.0),
            DiscriminativePair("A", "C", epsilon=1.0),
        ]
        framework_complete = PufferfishFramework(
            secrets=secrets, pairs=complete_pairs
        )
        assert framework_complete.is_complete()
        
        # Incomplete: missing pairs
        incomplete_pairs = [
            DiscriminativePair("A", "B", epsilon=1.0),
        ]
        framework_incomplete = PufferfishFramework(
            secrets=secrets, pairs=incomplete_pairs
        )
        assert not framework_incomplete.is_complete()
    
    def test_custom_scenarios(self):
        """Test custom scenarios."""
        secrets = ["A", "B"]
        pairs = [DiscriminativePair("A", "B", epsilon=1.0)]
        scenarios = ["scenario1", "scenario2", "scenario3"]
        
        framework = PufferfishFramework(
            secrets=secrets, pairs=pairs, scenarios=scenarios
        )
        
        assert framework.num_scenarios == 3
        assert framework.scenarios == scenarios


class TestPufferfishMechanism:
    """Test PufferfishMechanism."""
    
    def test_initialization(self):
        """Test mechanism initialization."""
        secrets = ["A", "B"]
        pairs = [DiscriminativePair("A", "B", epsilon=1.0)]
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
        
        # Probability table: (num_secrets, num_scenarios, num_outputs)
        p_table = np.ones((2, 1, 10)) / 10.0
        
        mech = PufferfishMechanism(
            framework=framework,
            probability_table=p_table,
        )
        
        assert mech.num_secrets == 2
        assert mech.num_scenarios == 1
        assert mech.num_outputs == 10
    
    def test_initialization_errors(self):
        """Test initialization error handling."""
        secrets = ["A", "B"]
        pairs = [DiscriminativePair("A", "B", epsilon=1.0)]
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
        
        # Wrong dimensionality
        p_table_wrong = np.ones((2, 10))  # 2D instead of 3D
        with pytest.raises(InvalidMechanismError):
            PufferfishMechanism(framework=framework, probability_table=p_table_wrong)
        
        # Wrong number of secrets
        p_table_wrong = np.ones((3, 1, 10)) / 10.0
        with pytest.raises(InvalidMechanismError):
            PufferfishMechanism(framework=framework, probability_table=p_table_wrong)
    
    def test_sample(self):
        """Test sampling."""
        secrets = ["A", "B", "C"]
        pairs = [
            DiscriminativePair("A", "B", epsilon=1.0),
            DiscriminativePair("B", "C", epsilon=1.0),
        ]
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
        
        # Uniform distribution
        p_table = np.ones((3, 1, 20)) / 20.0
        mech = PufferfishMechanism(
            framework=framework,
            probability_table=p_table,
            seed=42,
        )
        
        # Sample for secret 0
        output = mech.sample(secret_index=0, scenario_index=0)
        
        assert isinstance(output, float)
        assert 0 <= output < 20  # Within output range
    
    def test_sample_errors(self):
        """Test sampling error handling."""
        secrets = ["A", "B"]
        pairs = [DiscriminativePair("A", "B", epsilon=1.0)]
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
        
        p_table = np.ones((2, 1, 10)) / 10.0
        mech = PufferfishMechanism(
            framework=framework,
            probability_table=p_table,
        )
        
        # Invalid secret index
        with pytest.raises(ConfigurationError):
            mech.sample(secret_index=5)
        
        # Invalid scenario index
        with pytest.raises(ConfigurationError):
            mech.sample(secret_index=0, scenario_index=5)
    
    def test_pdf(self):
        """Test PDF evaluation."""
        secrets = ["A", "B"]
        pairs = [DiscriminativePair("A", "B", epsilon=1.0)]
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
        
        p_table = np.array([[[0.1, 0.2, 0.7]], [[0.3, 0.4, 0.3]]])
        mech = PufferfishMechanism(
            framework=framework,
            probability_table=p_table,
        )
        
        # Check PDF values
        assert abs(mech.pdf(0, 0, 0) - 0.1) < 1e-10
        assert abs(mech.pdf(0, 0, 1) - 0.2) < 1e-10
        assert abs(mech.pdf(1, 0, 0) - 0.3) < 1e-10
    
    def test_privacy_guarantee(self):
        """Test privacy guarantee reporting."""
        secrets = ["A", "B", "C"]
        pairs = [
            DiscriminativePair("A", "B", epsilon=1.0),
            DiscriminativePair("B", "C", epsilon=2.0),
        ]
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
        
        p_table = np.ones((3, 1, 10)) / 10.0
        mech = PufferfishMechanism(
            framework=framework,
            probability_table=p_table,
        )
        
        eps, delta = mech.privacy_guarantee()
        
        # Should return max epsilon
        assert eps == 2.0
        assert delta == 0.0
    
    def test_verify_privacy_uniform(self):
        """Test privacy verification for uniform mechanism."""
        secrets = ["A", "B"]
        pairs = [DiscriminativePair("A", "B", epsilon=1.0)]
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
        
        # Uniform distribution satisfies any epsilon
        p_table = np.ones((2, 1, 10)) / 10.0
        mech = PufferfishMechanism(
            framework=framework,
            probability_table=p_table,
        )
        
        is_private, violations = mech.verify_privacy()
        
        assert is_private
        assert len(violations) == 0
    
    def test_verify_privacy_violation(self):
        """Test privacy verification detects violations."""
        secrets = ["A", "B"]
        pairs = [DiscriminativePair("A", "B", epsilon=0.1)]  # Very small epsilon
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
        
        # Very different distributions
        p_table = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        mech = PufferfishMechanism(
            framework=framework,
            probability_table=p_table,
        )
        
        is_private, violations = mech.verify_privacy(tol=1e-3)
        
        # Should detect violation
        assert not is_private
        assert len(violations) > 0
    
    def test_synthesize_uniform(self):
        """Test synthesis with uniform baseline."""
        secrets = ["A", "B", "C"]
        pairs = [
            DiscriminativePair("A", "B", epsilon=1.0),
            DiscriminativePair("B", "C", epsilon=1.0),
        ]
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
        
        mech = PufferfishMechanism.synthesize(
            framework=framework,
            output_size=20,
            solver="scipy",
        )
        
        assert mech.num_secrets == 3
        assert mech.num_outputs == 20
        
        # Should satisfy privacy constraints
        is_private, _ = mech.verify_privacy()
        assert is_private
    
    def test_synthesize_random(self):
        """Test synthesis with random baseline."""
        secrets = ["A", "B"]
        pairs = [DiscriminativePair("A", "B", epsilon=1.0)]
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
        
        mech = PufferfishMechanism.synthesize(
            framework=framework,
            output_size=10,
            solver="random",
            seed=42,
        )
        
        assert mech.num_secrets == 2
        assert mech.num_outputs == 10
    
    def test_synthesize_constraint_budget_exceeded(self):
        """Test constraint budget checking."""
        # Create framework with too many constraints
        n_secrets = 100
        secrets = [f"s{i}" for i in range(n_secrets)]
        
        # All pairs
        pairs = [
            DiscriminativePair(f"s{i}", f"s{j}", epsilon=1.0)
            for i in range(n_secrets)
            for j in range(i + 1, n_secrets)
        ]
        
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
        
        # Should raise error due to too many constraints
        with pytest.raises(ConfigurationError):
            PufferfishMechanism.synthesize(
                framework=framework,
                output_size=100,
            )


class TestWassersteinMechanism:
    """Test WassersteinMechanism (optimal transport)."""
    
    def test_initialization(self):
        """Test mechanism initialization."""
        secrets = ["A", "B"]
        pairs = [DiscriminativePair("A", "B", epsilon=1.0)]
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
        
        # Cost function: squared difference
        def cost_fn(x, y):
            return (x - y) ** 2
        
        input_domain = np.linspace(0, 10, 11)
        output_domain = np.linspace(0, 10, 11)
        
        mech = WassersteinMechanism(
            framework=framework,
            cost_fn=cost_fn,
            input_domain=input_domain,
            output_domain=output_domain,
        )
        
        assert mech._cost_matrix.shape == (11, 11)
    
    def test_sample(self):
        """Test sampling."""
        secrets = ["A", "B"]
        pairs = [DiscriminativePair("A", "B", epsilon=1.0)]
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
        
        def cost_fn(x, y):
            return abs(x - y)
        
        input_domain = np.linspace(0, 10, 11)
        output_domain = np.linspace(0, 10, 11)
        
        mech = WassersteinMechanism(
            framework=framework,
            cost_fn=cost_fn,
            input_domain=input_domain,
            output_domain=output_domain,
            seed=42,
        )
        
        output = mech.sample(input_value=5.0)
        
        assert isinstance(output, float)
        assert 0 <= output <= 10
    
    def test_expected_cost(self):
        """Test expected cost computation."""
        secrets = ["A", "B"]
        pairs = [DiscriminativePair("A", "B", epsilon=1.0)]
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
        
        def cost_fn(x, y):
            return (x - y) ** 2
        
        input_domain = np.linspace(0, 10, 11)
        output_domain = np.linspace(0, 10, 11)
        
        mech = WassersteinMechanism(
            framework=framework,
            cost_fn=cost_fn,
            input_domain=input_domain,
            output_domain=output_domain,
        )
        
        cost = mech.expected_cost()
        
        assert cost >= 0
        assert math.isfinite(cost)
    
    def test_cost_matrix_structure(self):
        """Test cost matrix has correct structure."""
        secrets = ["A"]
        pairs = []
        framework = PufferfishFramework(
            secrets=secrets, pairs=[DiscriminativePair("A", "A", epsilon=1.0)]
        )
        
        # Linear cost
        def cost_fn(x, y):
            return abs(x - y)
        
        input_domain = np.array([0, 1, 2])
        output_domain = np.array([0, 1, 2])
        
        mech = WassersteinMechanism(
            framework=framework,
            cost_fn=cost_fn,
            input_domain=input_domain,
            output_domain=output_domain,
        )
        
        # Cost matrix should have linear structure
        C = mech._cost_matrix
        
        assert C[0, 0] == 0  # Same point
        assert C[0, 1] == 1  # Distance 1
        assert C[0, 2] == 2  # Distance 2


class TestPufferfishApplications:
    """Test Pufferfish for specific applications."""
    
    def test_health_data_privacy(self):
        """Test Pufferfish for health data privacy."""
        # Secrets: health conditions
        secrets = ["healthy", "diabetes", "heart_disease"]
        
        # Privacy requirements:
        # - Protect diabetes vs healthy
        # - Protect heart_disease vs healthy
        # - Weaker protection between diseases
        pairs = [
            DiscriminativePair("healthy", "diabetes", epsilon=0.5),
            DiscriminativePair("healthy", "heart_disease", epsilon=0.5),
            DiscriminativePair("diabetes", "heart_disease", epsilon=2.0),
        ]
        
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
        
        assert framework.num_secrets == 3
        assert framework.num_pairs == 3
        
        # Max epsilon should be 2.0
        assert framework.max_epsilon() == 2.0
    
    def test_location_privacy(self):
        """Test Pufferfish for location privacy."""
        # Secrets: locations
        secrets = ["home", "work", "hospital", "restaurant"]
        
        # Privacy requirements:
        # - Strong protection for hospital
        # - Moderate protection for home/work
        pairs = [
            DiscriminativePair("home", "hospital", epsilon=0.1),
            DiscriminativePair("work", "hospital", epsilon=0.1),
            DiscriminativePair("restaurant", "hospital", epsilon=0.1),
            DiscriminativePair("home", "work", epsilon=1.0),
            DiscriminativePair("home", "restaurant", epsilon=1.0),
            DiscriminativePair("work", "restaurant", epsilon=1.0),
        ]
        
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
        
        assert framework.num_secrets == 4
        assert framework.num_pairs == 6
        assert framework.is_complete()


@given(
    epsilon=st.floats(min_value=0.1, max_value=5.0),
)
@settings(max_examples=10, deadline=None)
def test_uniform_mechanism_satisfies_any_epsilon_hypothesis(epsilon):
    """Property test: uniform mechanism satisfies any epsilon."""
    secrets = ["A", "B", "C"]
    pairs = [
        DiscriminativePair("A", "B", epsilon=epsilon),
        DiscriminativePair("B", "C", epsilon=epsilon),
    ]
    framework = PufferfishFramework(secrets=secrets, pairs=pairs)
    
    # Uniform distribution
    output_size = 20
    p_table = np.ones((3, 1, output_size)) / output_size
    
    mech = PufferfishMechanism(
        framework=framework,
        probability_table=p_table,
    )
    
    is_private, violations = mech.verify_privacy()
    
    assert is_private
    assert len(violations) == 0


class TestPufferfishIntegration:
    """Integration tests for Pufferfish privacy."""
    
    def test_custom_privacy_policy(self):
        """Test implementing a custom privacy policy."""
        # Application: salary data
        # Secrets: income brackets
        secrets = ["low", "medium", "high", "very_high"]
        
        # Policy: stronger protection between non-adjacent brackets
        pairs = [
            DiscriminativePair("low", "medium", epsilon=1.0),
            DiscriminativePair("medium", "high", epsilon=1.0),
            DiscriminativePair("high", "very_high", epsilon=1.0),
            DiscriminativePair("low", "high", epsilon=0.5),  # Stronger
            DiscriminativePair("medium", "very_high", epsilon=0.5),  # Stronger
            DiscriminativePair("low", "very_high", epsilon=0.1),  # Strongest
        ]
        
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
        
        # Synthesize mechanism
        mech = PufferfishMechanism.synthesize(
            framework=framework,
            output_size=50,
            solver="scipy",
        )
        
        # Verify it satisfies the policy
        is_private, violations = mech.verify_privacy()
        
        assert is_private
        assert len(violations) == 0
    
    def test_sampling_distribution(self):
        """Test sampling distribution properties."""
        secrets = ["A", "B"]
        pairs = [DiscriminativePair("A", "B", epsilon=1.0)]
        framework = PufferfishFramework(secrets=secrets, pairs=pairs)
        
        # Create mechanism with known distribution
        output_size = 10
        p_table = np.zeros((2, 1, output_size))
        
        # Secret A: uniform on first half
        p_table[0, 0, :5] = 0.2
        
        # Secret B: uniform on second half
        p_table[1, 0, 5:] = 0.2
        
        mech = PufferfishMechanism(
            framework=framework,
            probability_table=p_table,
            seed=42,
        )
        
        # Sample many times for secret A
        n_samples = 1000
        samples = [mech.sample(secret_index=0) for _ in range(n_samples)]
        
        # Most samples should be in first half [0, 5)
        first_half = sum(1 for s in samples if s < 5)
        
        # Should be close to 100% (with statistical tolerance)
        assert first_half / n_samples > 0.8
