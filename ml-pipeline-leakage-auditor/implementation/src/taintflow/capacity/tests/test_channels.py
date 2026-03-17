"""
Tests for taintflow.capacity.channels – information-theoretic channel models.

Covers GaussianChannel, DiscreteChannel, BinaryChannel, DataProcessingInequality,
ChannelComposition, and the convenience functions for mean/variance statistics.
"""

from __future__ import annotations

import math
import unittest

from taintflow.capacity.channels import (
    BinaryChannel,
    Channel,
    ChannelCapacityBound,
    ChannelComposition,
    ChannelKind,
    DataProcessingInequality,
    DiscreteChannel,
    GaussianChannel,
    capacity_mean_statistic,
    capacity_variance_statistic,
)


class TestChannelCapacityBound(unittest.TestCase):
    """Tests for ChannelCapacityBound dataclass."""

    def test_default_values(self) -> None:
        b = ChannelCapacityBound(bits=1.5)
        self.assertAlmostEqual(b.bits, 1.5)
        self.assertAlmostEqual(b.tightness_factor, 1.0)
        self.assertFalse(b.is_tight)
        self.assertAlmostEqual(b.confidence, 1.0)

    def test_negative_bits_clamped(self) -> None:
        b = ChannelCapacityBound(bits=-5.0)
        self.assertAlmostEqual(b.bits, 0.0)

    def test_tightness_factor_clamped(self) -> None:
        b = ChannelCapacityBound(bits=1.0, tightness_factor=0.5)
        self.assertAlmostEqual(b.tightness_factor, 1.0)

    def test_validate_valid(self) -> None:
        b = ChannelCapacityBound(bits=2.0, confidence=0.95)
        self.assertEqual(b.validate(), [])

    def test_validate_invalid_confidence(self) -> None:
        b = ChannelCapacityBound(bits=1.0, confidence=1.5)
        errors = b.validate()
        self.assertTrue(any("confidence" in e for e in errors))

    def test_attenuated(self) -> None:
        b = ChannelCapacityBound(bits=4.0)
        a = b.attenuated(0.5)
        self.assertAlmostEqual(a.bits, 2.0)

    def test_attenuated_negative_factor(self) -> None:
        b = ChannelCapacityBound(bits=4.0)
        a = b.attenuated(-1.0)
        self.assertAlmostEqual(a.bits, 0.0)

    def test_zero_factory(self) -> None:
        z = ChannelCapacityBound.zero()
        self.assertAlmostEqual(z.bits, 0.0)
        self.assertTrue(z.is_tight)

    def test_infinite_factory(self) -> None:
        inf = ChannelCapacityBound.infinite()
        self.assertAlmostEqual(inf.bits, 64.0)
        self.assertFalse(inf.is_tight)


class TestGaussianChannel(unittest.TestCase):
    """Tests for GaussianChannel capacity computations."""

    def test_kind(self) -> None:
        gc = GaussianChannel()
        self.assertEqual(gc.kind, ChannelKind.GAUSSIAN)

    def test_zero_test_fraction(self) -> None:
        gc = GaussianChannel()
        cap = gc.capacity(rho=0.0, n=1000, d=5)
        self.assertAlmostEqual(cap, 0.0, places=5)

    def test_half_test_fraction(self) -> None:
        gc = GaussianChannel()
        cap = gc.capacity(rho=0.5, n=1000, d=5)
        expected = 0.5 * math.log2(1.0 + 1.0)  # SNR=1 → 0.5 bits per feature
        self.assertAlmostEqual(cap, expected * 5, places=3)

    def test_small_test_fraction_low_capacity(self) -> None:
        gc = GaussianChannel()
        cap = gc.capacity(rho=0.01, n=10000, d=1)
        self.assertLess(cap, 0.02)  # very small leakage

    def test_capacity_per_feature(self) -> None:
        gc = GaussianChannel()
        cpf = gc.capacity_per_feature(rho=0.2, n=500)
        expected = 0.5 * math.log2(1.0 + 0.2 / 0.8)
        self.assertAlmostEqual(cpf, expected, places=4)

    def test_capacity_bound_returns_bound(self) -> None:
        gc = GaussianChannel()
        b = gc.capacity_bound(rho=0.3, n=500, d=10)
        self.assertIsInstance(b, ChannelCapacityBound)
        self.assertGreater(b.bits, 0.0)
        self.assertEqual(b.channel_kind, ChannelKind.GAUSSIAN)

    def test_snr_from_test_fraction(self) -> None:
        snr = GaussianChannel._snr_from_test_fraction(0.5, 1000)
        self.assertAlmostEqual(snr, 1.0, places=3)

    def test_gaussian_capacity_formula(self) -> None:
        cap = GaussianChannel.gaussian_capacity_from_snr(1.0)
        self.assertAlmostEqual(cap, 0.5 * math.log2(2.0), places=6)


class TestDiscreteChannel(unittest.TestCase):
    """Tests for DiscreteChannel capacity computations."""

    def test_kind(self) -> None:
        dc = DiscreteChannel(input_alphabet_size=4, output_alphabet_size=4)
        self.assertEqual(dc.kind, ChannelKind.DISCRETE)

    def test_noiseless_capacity(self) -> None:
        dc = DiscreteChannel(input_alphabet_size=8, output_alphabet_size=8)
        self.assertAlmostEqual(dc.noiseless_capacity(), math.log2(8), places=5)

    def test_binary_entropy(self) -> None:
        self.assertAlmostEqual(DiscreteChannel.binary_entropy(0.5), 1.0, places=5)
        self.assertAlmostEqual(DiscreteChannel.binary_entropy(0.0), 0.0, places=5)
        self.assertAlmostEqual(DiscreteChannel.binary_entropy(1.0), 0.0, places=5)

    def test_capacity_bound(self) -> None:
        dc = DiscreteChannel(input_alphabet_size=4, output_alphabet_size=4)
        b = dc.capacity_bound(rho=0.2, n=500, d=5)
        self.assertIsInstance(b, ChannelCapacityBound)
        self.assertGreaterEqual(b.bits, 0.0)


class TestBinaryChannel(unittest.TestCase):
    """Tests for BinaryChannel (binary symmetric channel)."""

    def test_kind(self) -> None:
        bc = BinaryChannel(crossover_prob=0.1)
        self.assertEqual(bc.kind, ChannelKind.BINARY)

    def test_capacity_no_noise(self) -> None:
        bc = BinaryChannel(crossover_prob=0.0)
        self.assertAlmostEqual(bc.capacity_fixed(), 1.0, places=5)

    def test_capacity_full_noise(self) -> None:
        bc = BinaryChannel(crossover_prob=0.5)
        self.assertAlmostEqual(bc.capacity_fixed(), 0.0, places=5)

    def test_capacity_bound(self) -> None:
        bc = BinaryChannel(crossover_prob=0.1)
        b = bc.capacity_bound(rho=0.3, n=500, d=1)
        self.assertIsInstance(b, ChannelCapacityBound)
        self.assertGreater(b.bits, 0.0)


class TestDataProcessingInequality(unittest.TestCase):
    """Tests for DPI application."""

    def test_apply_returns_minimum(self) -> None:
        result = DataProcessingInequality.apply(
            i_xy=3.0, i_yz=5.0
        )
        self.assertAlmostEqual(result, 3.0)

    def test_apply_chain(self) -> None:
        result = DataProcessingInequality.apply_chain([10.0, 3.0, 7.0, 2.0])
        self.assertAlmostEqual(result, 2.0)

    def test_apply_chain_single(self) -> None:
        result = DataProcessingInequality.apply_chain([5.0])
        self.assertAlmostEqual(result, 5.0)


class TestChannelComposition(unittest.TestCase):
    """Tests for sequential and parallel channel composition."""

    def test_sequential_minimum(self) -> None:
        g1 = GaussianChannel()
        g2 = GaussianChannel()
        bound = ChannelComposition.sequential([g1, g2], rho=0.3, n=500, d=5)
        self.assertIsInstance(bound, ChannelCapacityBound)
        individual1 = g1.capacity_bound(0.3, 500, 5)
        individual2 = g2.capacity_bound(0.3, 500, 5)
        self.assertLessEqual(bound.bits, min(individual1.bits, individual2.bits) + 1e-10)

    def test_parallel_sum(self) -> None:
        g1 = GaussianChannel()
        g2 = GaussianChannel()
        bound = ChannelComposition.parallel([g1, g2], rho=0.3, n=500, d=5)
        self.assertIsInstance(bound, ChannelCapacityBound)


class TestCapacityConvenienceFunctions(unittest.TestCase):
    """Tests for capacity_mean_statistic and capacity_variance_statistic."""

    def test_mean_statistic_positive(self) -> None:
        b = capacity_mean_statistic(rho=0.2, n=1000, d=10)
        self.assertIsInstance(b, ChannelCapacityBound)
        self.assertGreater(b.bits, 0.0)

    def test_mean_statistic_zero_rho(self) -> None:
        b = capacity_mean_statistic(rho=0.0, n=1000, d=10)
        self.assertAlmostEqual(b.bits, 0.0, places=5)

    def test_variance_statistic_greater_than_mean(self) -> None:
        bm = capacity_mean_statistic(rho=0.3, n=500, d=5)
        bv = capacity_variance_statistic(rho=0.3, n=500, d=5)
        self.assertGreaterEqual(bv.bits, bm.bits - 0.01)

    def test_mean_statistic_increases_with_rho(self) -> None:
        b1 = capacity_mean_statistic(rho=0.1, n=1000, d=5)
        b2 = capacity_mean_statistic(rho=0.3, n=1000, d=5)
        b3 = capacity_mean_statistic(rho=0.5, n=1000, d=5)
        self.assertLess(b1.bits, b2.bits)
        self.assertLess(b2.bits, b3.bits)


if __name__ == "__main__":
    unittest.main()
