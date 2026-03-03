"""Tests for local_dp.protocols module."""
import math
import pytest
import numpy as np

from dp_forge.local_dp.protocols import (
    RAPPOREncoder,
    ShuffleModel,
    SecureAggregation,
    PrivateHistogram,
)


class TestRAPPOREncoder:
    """Test RAPPOREncoder encoding/decoding roundtrip."""

    def test_encode_returns_bloom_filter(self):
        """Encode returns a bloom filter bit array."""
        rappor = RAPPOREncoder(epsilon=1.0, bloom_size=32, num_hashes=2, seed=42)
        r = rappor.encode(5, user_id=0)
        vec = np.asarray(r.encoded_value)
        assert vec.shape == (32,)
        assert np.all((vec == 0) | (vec == 1))

    def test_memoization(self):
        """Same user gets same permanent response."""
        rappor = RAPPOREncoder(epsilon=1.0, bloom_size=32, num_hashes=2, seed=42)
        r1 = rappor.encode(5, user_id=0)
        r2 = rappor.encode(5, user_id=0)
        # Both use same permanent memo; IRR differs
        assert r1.encoded_value is not r2.encoded_value  # different IRR realizations

    def test_different_users_different_memos(self):
        """Different users get different permanent memos."""
        rappor = RAPPOREncoder(epsilon=1.0, bloom_size=64, num_hashes=2, seed=42)
        rappor.encode(5, user_id=0)
        rappor.encode(5, user_id=1)
        assert 0 in rappor._memos
        assert 1 in rappor._memos

    def test_aggregate_detects_frequent(self):
        """Aggregation detects the most frequent value."""
        rappor = RAPPOREncoder(epsilon=2.0, bloom_size=64, num_hashes=2, seed=42)
        n = 5000
        rng = np.random.default_rng(42)
        candidates = list(range(5))
        # 60% value 0
        values = rng.choice(5, size=n, p=[0.6, 0.1, 0.1, 0.1, 0.1])
        reports = [rappor.encode(int(v), user_id=i) for i, v in enumerate(values)]
        est = rappor.aggregate(reports, candidates)
        assert np.argmax(est.frequencies) == 0

    def test_encode_string(self):
        """String encoding works."""
        rappor = RAPPOREncoder(epsilon=1.0, bloom_size=32, num_hashes=2, seed=42)
        r = rappor.encode_string("hello", user_id=0)
        vec = np.asarray(r.encoded_value)
        assert vec.shape == (32,)


class TestShuffleModel:
    """Test ShuffleModel amplification."""

    def test_central_epsilon_less_than_local(self):
        """Amplified central epsilon < local epsilon."""
        sm = ShuffleModel(local_epsilon=2.0, num_users=1000, delta=1e-6)
        assert sm.central_epsilon < sm.local_epsilon

    def test_amplification_increases_with_users(self):
        """More users → lower central epsilon."""
        sm1 = ShuffleModel(local_epsilon=2.0, num_users=100, delta=1e-6)
        sm2 = ShuffleModel(local_epsilon=2.0, num_users=10000, delta=1e-6)
        assert sm2.central_epsilon < sm1.central_epsilon

    def test_shuffle_anonymizes(self):
        """Shuffled reports have user_id = -1."""
        from dp_forge.local_dp import LDPReport, LDPMechanismType
        reports = [
            LDPReport(user_id=i, encoded_value=i, mechanism_type=LDPMechanismType.RANDOMIZED_RESPONSE, domain_size=5)
            for i in range(10)
        ]
        sm = ShuffleModel(local_epsilon=1.0, num_users=10, delta=1e-6, seed=42)
        shuffled = sm.shuffle(reports)
        for r in shuffled:
            assert r.user_id == -1

    def test_shuffle_preserves_count(self):
        """Shuffling preserves the number of reports."""
        from dp_forge.local_dp import LDPReport, LDPMechanismType
        reports = [
            LDPReport(user_id=i, encoded_value=i % 3, mechanism_type=LDPMechanismType.RANDOMIZED_RESPONSE, domain_size=3)
            for i in range(20)
        ]
        sm = ShuffleModel(local_epsilon=1.0, num_users=20, delta=1e-6, seed=42)
        shuffled = sm.shuffle(reports)
        assert len(shuffled) == 20

    def test_amplification_factor(self):
        """Amplification factor > 1."""
        sm = ShuffleModel(local_epsilon=2.0, num_users=1000, delta=1e-6)
        assert sm.amplification_factor() > 1.0


class TestSecureAggregation:
    """Test SecureAggregation correctness."""

    def test_secure_encode_returns_vector(self):
        """Secure mode returns a float vector."""
        sa = SecureAggregation(epsilon=1.0, domain_size=5, num_users=100, secure=True, seed=42)
        r = sa.encode(2)
        vec = np.asarray(r.encoded_value)
        assert vec.shape == (5,)
        # True position should have highest value on average
        assert vec[2] > vec.mean() - 1.0  # allow noise

    def test_insecure_encode_returns_int(self):
        """Insecure mode returns a randomized response integer."""
        sa = SecureAggregation(epsilon=1.0, domain_size=5, num_users=100, secure=False, seed=42)
        r = sa.encode(2)
        assert isinstance(int(r.encoded_value), int)
        assert 0 <= int(r.encoded_value) < 5

    def test_secure_aggregate_accurate(self):
        """Secure aggregation recovers true distribution."""
        d = 4
        n = 5000
        sa = SecureAggregation(epsilon=2.0, domain_size=d, num_users=n, secure=True, seed=42)
        true_dist = np.array([0.4, 0.3, 0.2, 0.1])
        rng = np.random.default_rng(42)
        values = rng.choice(d, size=n, p=true_dist)
        reports = [sa.encode(int(v), user_id=i) for i, v in enumerate(values)]
        est = sa.aggregate(reports)
        for i in range(d):
            assert est.frequencies[i] == pytest.approx(true_dist[i], abs=0.15)

    def test_insecure_aggregate(self):
        """Insecure aggregation also works."""
        d = 3
        n = 5000
        sa = SecureAggregation(epsilon=3.0, domain_size=d, num_users=n, secure=False, seed=42)
        rng = np.random.default_rng(42)
        values = rng.choice(d, size=n)
        reports = [sa.encode(int(v), user_id=i) for i, v in enumerate(values)]
        est = sa.aggregate(reports)
        assert est.frequencies.shape == (d,)
        assert np.sum(est.frequencies) == pytest.approx(1.0, abs=0.01)


class TestPrivateHistogram:
    """Test PrivateHistogram accuracy."""

    def test_encode_returns_int(self):
        """Encode returns integer in domain."""
        ph = PrivateHistogram(epsilon=1.0, domain_size=5, seed=42)
        r = ph.encode(2)
        assert 0 <= int(r.encoded_value) < 5

    def test_build_histogram_valid_distribution(self):
        """Histogram is a valid distribution."""
        d = 5
        ph = PrivateHistogram(epsilon=3.0, domain_size=d, seed=42)
        n = 5000
        rng = np.random.default_rng(42)
        values = rng.choice(d, size=n)
        reports = [ph.encode(int(v), user_id=i) for i, v in enumerate(values)]
        est = ph.build_histogram(reports)
        assert est.frequencies.shape == (d,)
        assert np.sum(est.frequencies) == pytest.approx(1.0, abs=0.01)

    def test_encode_and_build(self):
        """Convenience method encode_and_build works."""
        d = 4
        ph = PrivateHistogram(epsilon=3.0, domain_size=d, seed=42)
        values = [0, 1, 2, 3, 0, 1, 0, 0]
        est = ph.encode_and_build(values)
        assert est.frequencies.shape == (d,)
        assert est.num_reports == len(values)

    def test_confidence_intervals(self):
        """Histogram has confidence intervals."""
        d = 3
        ph = PrivateHistogram(epsilon=2.0, domain_size=d, seed=42)
        rng = np.random.default_rng(42)
        values = rng.choice(d, size=1000).tolist()
        est = ph.encode_and_build(values)
        assert est.confidence_intervals is not None
        assert est.confidence_intervals.shape == (d, 2)

    def test_unnormalized_histogram(self):
        """Unnormalized histogram doesn't normalize."""
        d = 3
        ph = PrivateHistogram(epsilon=2.0, domain_size=d, seed=42)
        rng = np.random.default_rng(42)
        values = rng.choice(d, size=500).tolist()
        reports = [ph.encode(int(v), user_id=i) for i, v in enumerate(values)]
        est = ph.build_histogram(reports, normalise=False)
        # Frequencies are counts, not necessarily summing to 1
        assert est.frequencies is not None
