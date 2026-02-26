"""Tests for policy module."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.policy.policy_utils import (
    DummyPolicy,
    RandomPolicy,
    LinearPolicy,
    NormalizationWrapper,
    PolicyCache,
    PolicySampler,
)
from marace.policy.lipschitz import (
    LipschitzExtractor,
    SpectralNormComputation,
    ReLULipschitz,
    TanhLipschitz,
    LocalLipschitz,
)
from marace.policy.abstract_policy import (
    AbstractPolicyEvaluator,
    DeepZTransformer,
    ReLUAbstractTransformer,
    TanhAbstractTransformer,
    LinearAbstractTransformer,
)
from marace.policy.onnx_loader import (
    ActivationType,
    LayerInfo,
    NetworkArchitecture,
)
from marace.abstract.zonotope import Zonotope


class TestDummyPolicy:
    """Test dummy policy for testing."""

    def test_creation(self):
        """Test creating a dummy policy."""
        policy = DummyPolicy(
            output_dim=2,
            constant_output=np.array([1.0, 0.0]),
        )
        action = policy.evaluate(np.zeros(6))
        assert action.shape == (2,)

    def test_evaluation(self):
        """Test evaluating dummy policy."""
        policy = DummyPolicy(
            output_dim=2,
            constant_output=np.array([0.5, -0.5]),
        )
        obs = np.random.randn(4)
        action = policy.evaluate(obs)
        np.testing.assert_allclose(action, [0.5, -0.5])

    def test_batch_evaluation(self):
        """Test batch evaluation."""
        policy = DummyPolicy(
            output_dim=2,
            constant_output=np.array([1.0, 0.0]),
        )
        obs_batch = np.random.randn(10, 4)
        actions = policy.evaluate_batch(obs_batch)
        assert actions.shape == (10, 2)


class TestRandomPolicy:
    """Test random policy."""

    def test_creation(self):
        """Test creating random policy."""
        policy = RandomPolicy(output_dim=2)
        action = policy.evaluate(np.zeros(4))
        assert action.shape == (2,)

    def test_random_output(self):
        """Test random policy produces outputs."""
        policy = RandomPolicy(output_dim=2, seed=42)
        obs = np.zeros(4)
        a1 = policy.evaluate(obs)
        a2 = policy.evaluate(obs)
        assert a1.shape == (2,)
        assert a2.shape == (2,)


class TestLinearPolicy:
    """Test linear policy."""

    def test_creation(self):
        """Test creating linear policy."""
        W = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        b = np.array([0.1, -0.1])
        policy = LinearPolicy(weight=W, bias=b)
        action = policy.evaluate(np.ones(4))
        assert action.shape == (2,)

    def test_evaluation(self):
        """Test evaluating linear policy."""
        W = np.eye(2)
        b = np.zeros(2)
        policy = LinearPolicy(weight=W, bias=b)
        obs = np.array([1.0, 2.0])
        action = policy.evaluate(obs)
        np.testing.assert_allclose(action, [1.0, 2.0])

    def test_with_bias(self):
        """Test linear policy with bias."""
        W = np.array([[2.0, 0.0], [0.0, 3.0]])
        b = np.array([1.0, -1.0])
        policy = LinearPolicy(weight=W, bias=b)
        obs = np.array([1.0, 1.0])
        action = policy.evaluate(obs)
        np.testing.assert_allclose(action, [3.0, 2.0])


class TestNormalizationWrapper:
    """Test normalization wrapper."""

    def test_normalize(self):
        """Test observation normalization."""
        wrapper = NormalizationWrapper(
            obs_mean=np.array([1.0, 2.0]),
            obs_std=np.array([2.0, 4.0]),
        )
        obs = np.array([3.0, 6.0])
        normalized = wrapper.normalize_obs(obs)
        np.testing.assert_allclose(normalized, [1.0, 1.0])

    def test_denormalize_action(self):
        """Test action denormalization."""
        wrapper = NormalizationWrapper(
            obs_mean=np.zeros(2),
            obs_std=np.ones(2),
            act_mean=np.array([0.5, 0.5]),
            act_std=np.array([0.5, 0.5]),
        )
        action = np.array([0.0, 0.0])
        denorm = wrapper.denormalize_action(action)
        np.testing.assert_allclose(denorm, [0.5, 0.5])


class TestPolicyCache:
    """Test policy evaluation cache."""

    def test_cache_hit(self):
        """Test cache hit."""
        cache = PolicyCache(max_size=100)
        obs = np.array([1.0, 2.0])
        action = np.array([0.5, -0.5])
        cache.put(obs, action)
        result = cache.get(obs)
        assert result is not None
        np.testing.assert_allclose(result, action)

    def test_cache_miss(self):
        """Test cache miss."""
        cache = PolicyCache(max_size=100)
        obs = np.array([1.0, 2.0])
        result = cache.get(obs)
        assert result is None

    def test_cache_eviction(self):
        """Test cache eviction when full."""
        cache = PolicyCache(max_size=5)
        for i in range(10):
            cache.put(np.array([float(i)]), np.array([float(i)]))
        assert cache.size <= 5


class TestSpectralNorm:
    """Test spectral norm computation."""

    def test_identity_matrix(self):
        """Test spectral norm of identity."""
        snc = SpectralNormComputation()
        W = np.eye(5)
        sigma = snc.compute(W)
        assert np.isclose(sigma, 1.0, atol=0.01)

    def test_scaled_identity(self):
        """Test spectral norm of scaled identity."""
        snc = SpectralNormComputation()
        W = 3.0 * np.eye(4)
        sigma = snc.compute(W)
        assert np.isclose(sigma, 3.0, atol=0.01)

    def test_rectangular_matrix(self):
        """Test spectral norm of rectangular matrix."""
        snc = SpectralNormComputation()
        np.random.seed(42)
        W = np.random.randn(3, 5)
        sigma = snc.compute(W)
        sigma_true = np.linalg.svd(W, compute_uv=False)[0]
        assert np.isclose(sigma, sigma_true, atol=0.01)

    def test_zero_matrix(self):
        """Test spectral norm of zero matrix."""
        snc = SpectralNormComputation()
        W = np.zeros((3, 3))
        sigma = snc.compute(W)
        assert np.isclose(sigma, 0.0, atol=0.01)


class TestLipschitzExtraction:
    """Test Lipschitz bound extraction."""

    def _make_arch(self, weights_list, activation=ActivationType.RELU):
        """Helper to create a NetworkArchitecture from weight matrices."""
        layers = []
        for i, W in enumerate(weights_list):
            layers.append(LayerInfo(
                name=f"fc{i}",
                layer_type="dense",
                input_size=W.shape[1],
                output_size=W.shape[0],
                activation=activation,
                weights=W,
                bias=np.zeros(W.shape[0]),
            ))
        return NetworkArchitecture(
            layers=layers,
            input_dim=weights_list[0].shape[1],
            output_dim=weights_list[-1].shape[0],
        )

    def test_relu_lipschitz_single_layer(self):
        """Test Lipschitz bound for single ReLU layer."""
        extractor = ReLULipschitz()
        W = np.array([[2.0, 0.0], [0.0, 3.0]])
        arch = self._make_arch([W])
        cert = extractor.compute(arch)
        assert np.isclose(cert.global_bound, 3.0, atol=0.01)

    def test_relu_lipschitz_multi_layer(self):
        """Test Lipschitz bound for multi-layer ReLU network."""
        extractor = ReLULipschitz()
        weights = [
            np.array([[1.0, 0.5], [-0.3, 1.0], [0.7, -0.2]]),
            np.array([[0.5, -0.3, 0.7], [0.2, 0.6, -0.4]]),
        ]
        arch = self._make_arch(weights)
        cert = extractor.compute(arch)
        # Product of spectral norms
        s1 = np.linalg.svd(weights[0], compute_uv=False)[0]
        s2 = np.linalg.svd(weights[1], compute_uv=False)[0]
        expected = s1 * s2
        assert np.isclose(cert.global_bound, expected, atol=0.1)

    def test_tanh_lipschitz(self):
        """Test Lipschitz for Tanh network."""
        extractor = TanhLipschitz()
        weights = [np.array([[2.0, 0.0], [0.0, 2.0]])]
        arch = self._make_arch(weights, activation=ActivationType.TANH)
        cert = extractor.compute(arch)
        assert cert.global_bound > 0
        assert cert.global_bound <= 2.0 + 0.1


class TestAbstractPolicyEvaluation:
    """Test abstract policy evaluation."""

    def test_linear_abstract_transformer(self):
        """Test abstract transformer for linear layer."""
        W = np.array([[1.0, -0.5], [0.3, 1.0]])
        b = np.array([0.1, -0.1])
        z = Zonotope(
            center=np.array([1.0, 2.0]),
            generators=np.array([[0.5, 0.0], [0.0, 0.5]])
        )
        z2 = LinearAbstractTransformer.transform(z, W, b)
        assert z2.dimension == 2
        expected_center = W @ z.center + b
        np.testing.assert_allclose(z2.center, expected_center, atol=0.01)

    def test_relu_abstract_transformer(self):
        """Test abstract transformer for ReLU."""
        transformer = ReLUAbstractTransformer()
        z = Zonotope(
            center=np.array([0.5, -0.5]),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        z2 = transformer.transform(z)
        assert z2.dimension == 2
        bbox = z2.bounding_box()
        # ReLU output upper bound for dim 0 should be positive (center+gen = 1.5)
        assert bbox[0, 1] >= 0.5
        # Over-approximation may give slightly negative lower bounds; just check soundness
        assert bbox[0, 1] >= bbox[0, 0]

    def test_deepz_transformer_soundness(self):
        """Test DeepZ transformer soundness for simple network."""
        np.random.seed(42)
        W1 = np.array([[1.0, -0.5], [0.3, 1.0]])
        b1 = np.array([0.1, -0.1])
        W2 = np.array([[0.5, 0.7], [-0.3, 0.6]])
        b2 = np.array([0.0, 0.1])

        layers = [
            LayerInfo(name="fc1", layer_type="dense", input_size=2, output_size=2,
                      activation=ActivationType.RELU, weights=W1, bias=b1),
            LayerInfo(name="fc2", layer_type="dense", input_size=2, output_size=2,
                      activation=ActivationType.LINEAR, weights=W2, bias=b2),
        ]
        arch = NetworkArchitecture(layers=layers, input_dim=2, output_dim=2)
        transformer = DeepZTransformer(architecture=arch)
        z_in = Zonotope(
            center=np.array([0.5, 0.5]),
            generators=np.array([[0.2, 0.0], [0.0, 0.2]])
        )
        result = transformer.transform(z_in)
        bbox = result.output_zonotope.bounding_box()

        # Verify soundness by sampling
        for _ in range(200):
            x = z_in.sample(1)[0]
            h = np.maximum(W1 @ x + b1, 0)
            y = W2 @ h + b2
            for d in range(2):
                assert bbox[d, 0] - 0.01 <= y[d] <= bbox[d, 1] + 0.01


class TestPolicySampler:
    """Test policy sampler."""

    def test_sampling(self):
        """Test sampling state-action pairs from policy."""
        policy = LinearPolicy(
            weight=np.eye(2),
            bias=np.zeros(2),
        )
        sampler = PolicySampler(
            policy_fn=policy.evaluate,
            input_dim=2,
        )
        states, actions = sampler.sample_uniform(
            lower=np.array([-1.0, -1.0]),
            upper=np.array([1.0, 1.0]),
            n_samples=100,
        )
        assert states.shape == (100, 2)
        assert actions.shape == (100, 2)
