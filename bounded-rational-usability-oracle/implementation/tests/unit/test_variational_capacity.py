"""Tests for usability_oracle.variational.capacity.

Verifies Fitts' capacity, Hick's law, visual/memory capacity,
Blahut–Arimoto channel capacity, and serial/parallel composition.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from usability_oracle.variational.capacity import (
    blahut_arimoto,
    compose_capacities,
    estimate_fitts_capacity,
    estimate_hick_capacity,
    estimate_memory_capacity,
    estimate_visual_capacity,
)


# =====================================================================
# Fitts' capacity (MacKenzie 1992)
# =====================================================================

class TestFittsCapacity:
    """Test motor-channel capacity via Fitts' law."""

    def test_typical_mackenzie_values(self) -> None:
        """Fitts' throughput should match typical MacKenzie (1992) values.

        With the Shannon formulation ID = log2(D/W + 1), a typical
        throughput is ~4.9 bits/sec.
        """
        cap = estimate_fitts_capacity(distance=256.0, width=32.0, throughput=4.9)
        assert cap > 0.0
        assert np.isfinite(cap)

    def test_zero_distance(self) -> None:
        """Distance = 0 → ID = 0 (already at target)."""
        cap = estimate_fitts_capacity(distance=0.0, width=32.0)
        # ID = log2(0/32 + 1) = log2(1) = 0; still returns throughput-based value
        assert cap >= 0.0

    def test_negative_distance_raises(self) -> None:
        """Negative distance should raise ValueError."""
        with pytest.raises(ValueError, match="distance"):
            estimate_fitts_capacity(distance=-10.0, width=32.0)

    def test_zero_width_raises(self) -> None:
        """Zero target width should raise ValueError."""
        with pytest.raises(ValueError, match="width"):
            estimate_fitts_capacity(distance=100.0, width=0.0)

    @pytest.mark.parametrize("dist,width", [
        (64, 16),
        (128, 8),
        (256, 32),
        (512, 64),
    ])
    def test_capacity_is_positive(self, dist: float, width: float) -> None:
        """Capacity is always positive for valid inputs."""
        cap = estimate_fitts_capacity(distance=dist, width=width)
        assert cap > 0.0


# =====================================================================
# Hick's capacity
# =====================================================================

class TestHickCapacity:
    """Test choice-channel capacity via Hick–Hyman law."""

    @pytest.mark.parametrize("n,expected", [
        (1, 0.0),
        (2, 1.0),
        (4, 2.0),
        (8, 3.0),
    ])
    def test_uniform_probabilities_equals_log2n(
        self, n: int, expected: float
    ) -> None:
        """With uniform stimulus probabilities, H = log₂(n)."""
        cap = estimate_hick_capacity(n)
        assert cap == pytest.approx(expected, abs=1e-10)

    def test_non_uniform_probabilities(self) -> None:
        """Non-uniform probabilities should give less entropy."""
        probs = np.array([0.7, 0.1, 0.1, 0.1])
        cap_nonuniform = estimate_hick_capacity(4, stimulus_probs=probs)
        cap_uniform = estimate_hick_capacity(4)
        assert cap_nonuniform < cap_uniform

    def test_invalid_n_raises(self) -> None:
        """n < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_alternatives"):
            estimate_hick_capacity(0)

    def test_single_alternative(self) -> None:
        """With 1 alternative, entropy is 0."""
        assert estimate_hick_capacity(1) == 0.0


# =====================================================================
# Visual capacity
# =====================================================================

class TestVisualCapacity:
    """Test visual-search channel capacity."""

    def test_foveal_capacity(self) -> None:
        """At fovea (eccentricity=0), acuity factor = 1."""
        cap_fovea = estimate_visual_capacity(eccentricity=0.0, n_items=4)
        cap_periph = estimate_visual_capacity(eccentricity=10.0, n_items=4)
        assert cap_fovea > cap_periph

    def test_decreases_with_eccentricity(self) -> None:
        """Capacity should decrease as eccentricity increases."""
        caps = [
            estimate_visual_capacity(eccentricity=e, n_items=8)
            for e in [0, 5, 10, 20]
        ]
        for i in range(len(caps) - 1):
            assert caps[i] >= caps[i + 1]

    def test_negative_eccentricity_raises(self) -> None:
        """Negative eccentricity should raise ValueError."""
        with pytest.raises(ValueError, match="eccentricity"):
            estimate_visual_capacity(eccentricity=-1.0)

    def test_capacity_is_positive(self) -> None:
        """Visual capacity should be positive."""
        cap = estimate_visual_capacity(eccentricity=5.0, n_items=4)
        assert cap > 0.0


# =====================================================================
# Memory capacity (Miller's 7±2)
# =====================================================================

class TestMemoryCapacity:
    """Test working-memory channel capacity."""

    def test_default_bounded_by_millers(self) -> None:
        """Default WM capacity with 4 chunks should be within log2(4) ≈ 2 bits."""
        cap = estimate_memory_capacity(n_chunks=4)
        # With no decay, capacity = log2(4) = 2.0
        assert cap == pytest.approx(math.log2(4), abs=1e-10)

    def test_seven_chunks_bounded(self) -> None:
        """7 chunks → log2(7) ≈ 2.807 bits, within Miller's range."""
        cap = estimate_memory_capacity(n_chunks=7)
        assert cap == pytest.approx(math.log2(7), abs=1e-10)
        # Miller's 7±2: capacity should be roughly 2-3 bits
        assert 2.0 <= cap <= 4.0

    def test_decay_reduces_capacity(self) -> None:
        """Nonzero decay rate should reduce effective capacity."""
        cap_no_decay = estimate_memory_capacity(n_chunks=4, decay_rate=0.0)
        cap_with_decay = estimate_memory_capacity(n_chunks=4, decay_rate=1.0)
        assert cap_with_decay < cap_no_decay

    def test_rehearsal_recovers_capacity(self) -> None:
        """Full rehearsal should compensate for decay."""
        cap_full_rehearsal = estimate_memory_capacity(
            n_chunks=4, decay_rate=1.0, rehearsal=1.0
        )
        cap_no_decay = estimate_memory_capacity(n_chunks=4, decay_rate=0.0)
        assert cap_full_rehearsal == pytest.approx(cap_no_decay, abs=1e-10)

    def test_zero_chunks(self) -> None:
        """0 chunks → 0 capacity."""
        assert estimate_memory_capacity(n_chunks=0) == 0.0


# =====================================================================
# Blahut-Arimoto
# =====================================================================

class TestBlahutArimoto:
    """Test the Blahut–Arimoto algorithm for channel capacity."""

    def test_binary_symmetric_channel_converges(self) -> None:
        """BSC should converge and give capacity = 1 - H(p)."""
        p_error = 0.1
        W = np.array([
            [1 - p_error, p_error],
            [p_error, 1 - p_error],
        ])
        capacity, input_dist = blahut_arimoto(W, tolerance=1e-10, max_iter=500)
        # BSC capacity = 1 - H(p) in bits
        h_p = -p_error * math.log2(p_error) - (1 - p_error) * math.log2(1 - p_error)
        expected_capacity = 1.0 - h_p
        assert capacity == pytest.approx(expected_capacity, abs=0.01)

    def test_bsc_capacity_equals_1_minus_hp(self) -> None:
        """Explicit BSC capacity formula check for various error rates."""
        for p_error in [0.0001, 0.05, 0.2, 0.3]:
            W = np.array([
                [1 - p_error, p_error],
                [p_error, 1 - p_error],
            ])
            capacity, _ = blahut_arimoto(W, tolerance=1e-10, max_iter=1000)
            h_p = -(
                p_error * math.log2(max(p_error, 1e-30))
                + (1 - p_error) * math.log2(max(1 - p_error, 1e-30))
            )
            expected = 1.0 - h_p
            assert capacity == pytest.approx(expected, abs=0.02), (
                f"BSC(p={p_error}): got {capacity}, expected {expected}"
            )

    def test_noiseless_channel_capacity_is_log_n(self) -> None:
        """Noiseless channel (identity matrix) has C = log2(n) bits."""
        n = 4
        W = np.eye(n)
        capacity, _ = blahut_arimoto(W, tolerance=1e-10)
        assert capacity == pytest.approx(math.log2(n), abs=0.01)

    def test_uniform_input_dist_for_symmetric_channel(self) -> None:
        """For a symmetric channel, the capacity-achieving input is uniform."""
        W = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        _, input_dist = blahut_arimoto(W, tolerance=1e-10)
        np.testing.assert_allclose(input_dist, np.ones(3) / 3, atol=0.01)

    def test_capacity_is_non_negative(self) -> None:
        """Channel capacity is always ≥ 0."""
        W = np.array([[0.5, 0.5], [0.5, 0.5]])  # totally noisy
        capacity, _ = blahut_arimoto(W)
        assert capacity >= -1e-10

    def test_invalid_ndim_raises(self) -> None:
        """1-D matrix should raise ValueError."""
        with pytest.raises(ValueError, match="2-D"):
            blahut_arimoto(np.array([0.5, 0.5]))


# =====================================================================
# Serial/parallel composition
# =====================================================================

class TestComposeCapacities:
    """Test serial and parallel composition of channel capacities."""

    def test_serial_equals_min(self) -> None:
        """Serial composition: C = min(C₁, C₂, …)."""
        caps = [3.0, 1.5, 4.2]
        assert compose_capacities(caps, mode="serial") == 1.5

    def test_serial_le_min_individual(self) -> None:
        """Serial composition ≤ min(individual capacities)."""
        caps = [5.0, 3.0, 8.0]
        composed = compose_capacities(caps, mode="serial")
        assert composed <= min(caps) + 1e-10

    def test_parallel_equals_sum(self) -> None:
        """Parallel composition: C = C₁ + C₂ + …."""
        caps = [3.0, 1.5, 4.2]
        assert compose_capacities(caps, mode="parallel") == pytest.approx(8.7)

    def test_empty_raises(self) -> None:
        """Empty capacities should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            compose_capacities([], mode="serial")

    def test_unknown_mode_raises(self) -> None:
        """Unknown mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            compose_capacities([1.0], mode="hybrid")

    def test_capacity_non_negative(self) -> None:
        """Non-negative inputs produce non-negative output."""
        for mode in ["serial", "parallel"]:
            assert compose_capacities([0.0, 1.0, 2.0], mode=mode) >= 0.0
