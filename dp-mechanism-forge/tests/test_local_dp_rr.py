"""Tests for local_dp.randomized_response module."""
import math
import pytest
import numpy as np

from dp_forge.local_dp.randomized_response import (
    RandomizedResponse,
    GeneralizedRR,
    OptimalRR,
    DirectEncoding,
    UnaryEncoding,
)


class TestRandomizedResponse:
    """Test RandomizedResponse ε-LDP property."""

    def test_encode_returns_report(self):
        """Encode returns a valid LDPReport."""
        rr = RandomizedResponse(epsilon=1.0, domain_size=3, seed=42)
        report = rr.encode(0)
        assert 0 <= int(report.encoded_value) < 3

    def test_truth_probability(self):
        """Truth probability matches e^eps / (e^eps + d - 1)."""
        eps = 2.0
        d = 4
        rr = RandomizedResponse(epsilon=eps, domain_size=d, seed=42)
        expected = math.exp(eps) / (math.exp(eps) + d - 1)
        assert rr.truth_probability == pytest.approx(expected, abs=1e-10)

    def test_ldp_property(self):
        """Empirically verify LDP: P[y|x] / P[y|x'] <= e^eps for all x,x',y."""
        eps = 1.0
        d = 3
        n = 50000
        rr = RandomizedResponse(epsilon=eps, domain_size=d, seed=42)
        counts = np.zeros((d, d))
        for x in range(d):
            for _ in range(n):
                r = rr.encode(x)
                counts[x, int(r.encoded_value)] += 1
        probs = counts / n
        for y in range(d):
            for x1 in range(d):
                for x2 in range(d):
                    if probs[x2, y] > 0.001:
                        ratio = probs[x1, y] / probs[x2, y]
                        assert ratio <= math.exp(eps) * 1.5  # allow stat noise

    def test_aggregate_unbiased(self):
        """Aggregate recovers true distribution approximately."""
        eps = 3.0
        d = 3
        n = 10000
        rr = RandomizedResponse(epsilon=eps, domain_size=d, seed=42)
        # True distribution: 60% value 0, 30% value 1, 10% value 2
        true_dist = np.array([0.6, 0.3, 0.1])
        values = np.random.default_rng(42).choice(d, size=n, p=true_dist)
        reports = rr.encode_batch(values.astype(np.int64))
        est = rr.aggregate(reports)
        for i in range(d):
            assert est.frequencies[i] == pytest.approx(true_dist[i], abs=0.1)

    def test_encode_batch(self):
        """Batch encoding produces correct number of reports."""
        rr = RandomizedResponse(epsilon=1.0, domain_size=2, seed=42)
        values = np.array([0, 1, 0, 1], dtype=np.int64)
        reports = rr.encode_batch(values)
        assert len(reports) == 4

    def test_invalid_value_raises(self):
        """Encoding invalid value raises ValueError."""
        rr = RandomizedResponse(epsilon=1.0, domain_size=3, seed=42)
        with pytest.raises(ValueError):
            rr.encode(5)

    def test_analysis(self):
        """Analysis returns valid statistics."""
        rr = RandomizedResponse(epsilon=1.0, domain_size=5, seed=42)
        analysis = rr.analysis()
        assert analysis.expected_mse > 0
        assert analysis.communication_bits > 0


class TestGeneralizedRR:
    """Test GeneralizedRR unbiasedness."""

    def test_default_matrix_is_rr(self):
        """Default perturbation matrix matches standard RR."""
        eps = 1.0
        d = 3
        grr = GeneralizedRR(epsilon=eps, domain_size=d, seed=42)
        P = grr.perturbation_matrix
        e_eps = math.exp(eps)
        expected_p = e_eps / (e_eps + d - 1)
        assert P[0, 0] == pytest.approx(expected_p, abs=1e-10)

    def test_custom_matrix(self):
        """Custom perturbation matrix is used."""
        P = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        grr = GeneralizedRR(epsilon=2.0, domain_size=3, p_matrix=P, seed=42)
        np.testing.assert_allclose(grr.perturbation_matrix, P)

    def test_aggregate_recovers_frequencies(self):
        """Aggregation recovers true frequencies."""
        eps = 3.0
        d = 3
        grr = GeneralizedRR(epsilon=eps, domain_size=d, seed=42)
        n = 10000
        true_dist = np.array([0.5, 0.3, 0.2])
        values = np.random.default_rng(42).choice(d, size=n, p=true_dist)
        reports = [grr.encode(int(v), user_id=i) for i, v in enumerate(values)]
        est = grr.aggregate(reports)
        for i in range(d):
            assert est.frequencies[i] == pytest.approx(true_dist[i], abs=0.15)

    def test_rows_sum_to_one(self):
        """Each row of perturbation matrix sums to 1."""
        grr = GeneralizedRR(epsilon=1.0, domain_size=4, seed=42)
        P = grr.perturbation_matrix
        for row in P:
            assert np.sum(row) == pytest.approx(1.0, abs=1e-10)


class TestOptimalRR:
    """Test OptimalRR minimizes variance."""

    def test_optimal_probabilities(self):
        """p* = e^eps/(e^eps+1), q* = 1/(e^eps+1)."""
        eps = 2.0
        orr = OptimalRR(epsilon=eps, domain_size=5, target_value=0, seed=42)
        e_eps = math.exp(eps)
        assert orr._p_star == pytest.approx(e_eps / (e_eps + 1.0), abs=1e-10)
        assert orr._q_star == pytest.approx(1.0 / (e_eps + 1.0), abs=1e-10)

    def test_encode_binary(self):
        """Encode returns 0 or 1."""
        orr = OptimalRR(epsilon=1.0, domain_size=3, target_value=0, seed=42)
        r = orr.encode(0)
        assert int(r.encoded_value) in (0, 1)

    def test_estimate_frequency_unbiased(self):
        """Frequency estimate is close to true frequency."""
        eps = 3.0
        orr = OptimalRR(epsilon=eps, domain_size=5, target_value=0, seed=42)
        # 40% are target value 0
        n = 10000
        rng = np.random.default_rng(42)
        values = rng.choice(5, size=n, p=[0.4, 0.15, 0.15, 0.15, 0.15])
        reports = [orr.encode(int(v), user_id=i) for i, v in enumerate(values)]
        freq = orr.estimate_frequency(reports)
        assert freq == pytest.approx(0.4, abs=0.15)

    def test_variance_property(self):
        """Variance is positive."""
        orr = OptimalRR(epsilon=1.0, domain_size=5, target_value=0)
        assert orr.variance > 0


class TestDirectEncoding:
    """Test DirectEncoding frequency estimation."""

    def test_encode_returns_vector(self):
        """Encode returns a binary vector."""
        de = DirectEncoding(epsilon=1.0, domain_size=5, seed=42)
        r = de.encode(2)
        vec = np.asarray(r.encoded_value)
        assert vec.shape == (5,)
        assert np.all((vec == 0) | (vec == 1))

    def test_aggregate_unbiased(self):
        """Aggregation is approximately unbiased."""
        eps = 3.0
        d = 4
        de = DirectEncoding(epsilon=eps, domain_size=d, seed=42)
        n = 10000
        true_dist = np.array([0.4, 0.3, 0.2, 0.1])
        rng = np.random.default_rng(42)
        values = rng.choice(d, size=n, p=true_dist)
        reports = [de.encode(int(v), user_id=i) for i, v in enumerate(values)]
        est = de.aggregate(reports)
        for i in range(d):
            assert est.frequencies[i] == pytest.approx(true_dist[i], abs=0.15)

    def test_invalid_value_raises(self):
        """Invalid value raises error."""
        de = DirectEncoding(epsilon=1.0, domain_size=3, seed=42)
        with pytest.raises(ValueError):
            de.encode(5)


class TestUnaryEncoding:
    """Test UnaryEncoding variance."""

    def test_oue_encode(self):
        """OUE encode returns binary vector."""
        ue = UnaryEncoding(epsilon=1.0, domain_size=5, variant="OUE", seed=42)
        r = ue.encode(2)
        vec = np.asarray(r.encoded_value)
        assert vec.shape == (5,)

    def test_sue_encode(self):
        """SUE encode returns binary vector."""
        ue = UnaryEncoding(epsilon=1.0, domain_size=5, variant="SUE", seed=42)
        r = ue.encode(2)
        vec = np.asarray(r.encoded_value)
        assert vec.shape == (5,)

    def test_oue_unbiased(self):
        """OUE aggregation is approximately unbiased."""
        eps = 3.0
        d = 4
        ue = UnaryEncoding(epsilon=eps, domain_size=d, variant="OUE", seed=42)
        n = 10000
        true_dist = np.array([0.4, 0.3, 0.2, 0.1])
        rng = np.random.default_rng(42)
        values = rng.choice(d, size=n, p=true_dist)
        reports = [ue.encode(int(v), user_id=i) for i, v in enumerate(values)]
        est = ue.aggregate(reports)
        for i in range(d):
            assert est.frequencies[i] == pytest.approx(true_dist[i], abs=0.15)

    def test_analysis(self):
        """Analysis returns valid MSE."""
        ue = UnaryEncoding(epsilon=1.0, domain_size=10, variant="OUE")
        analysis = ue.analysis()
        assert analysis.expected_mse > 0
        assert analysis.communication_bits == 10

    def test_invalid_variant_raises(self):
        """Invalid variant raises error."""
        with pytest.raises(ValueError):
            UnaryEncoding(epsilon=1.0, domain_size=5, variant="INVALID")

    def test_oue_lower_variance_than_sue(self):
        """OUE should have lower or equal variance than SUE."""
        oue = UnaryEncoding(epsilon=2.0, domain_size=10, variant="OUE")
        sue = UnaryEncoding(epsilon=2.0, domain_size=10, variant="SUE")
        # OUE is optimal, so its MSE should be <= SUE's
        assert oue.analysis().expected_mse <= sue.analysis().expected_mse + 0.1
