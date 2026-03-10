"""Unit tests for usability_oracle.channel.allocator.

Tests cover resource allocation, water-filling algorithm,
capacity-constrained allocation, and multi-channel allocation.
"""

from __future__ import annotations

import numpy as np
import pytest

from usability_oracle.channel.types import (
    ChannelAllocation,
    InterferenceMatrix,
    ResourceChannel,
    ResourcePool,
    WickensResource,
)
from usability_oracle.channel.allocator import (
    AllocationResult,
    MRTAllocator,
    water_filling_allocate,
)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _pool(*channels: tuple[WickensResource, float]) -> ResourcePool:
    """Create a pool from (resource, capacity) pairs."""
    chs = tuple(
        ResourceChannel(resource=r, capacity_bits_per_s=c)
        for r, c in channels
    )
    return ResourcePool(channels=chs)


# ------------------------------------------------------------------ #
# Water-filling algorithm
# ------------------------------------------------------------------ #


class TestWaterFilling:
    """Tests for the classic water-filling resource allocation."""

    def test_equal_noise_equal_allocation(self) -> None:
        """Equal noise → equal power split."""
        caps = np.array([10.0, 10.0, 10.0])
        noise = np.array([1.0, 1.0, 1.0])
        alloc = water_filling_allocate(caps, noise, total_power=6.0)
        np.testing.assert_allclose(alloc, [2.0, 2.0, 2.0], atol=1e-8)

    def test_budget_constraint(self) -> None:
        """Total allocation ≤ total power."""
        caps = np.array([10.0, 10.0])
        noise = np.array([0.5, 1.5])
        alloc = water_filling_allocate(caps, noise, total_power=5.0)
        assert np.sum(alloc) == pytest.approx(5.0, abs=1e-8)

    def test_all_non_negative(self) -> None:
        """No negative allocations."""
        caps = np.array([10.0, 10.0, 10.0])
        noise = np.array([0.1, 1.0, 10.0])
        alloc = water_filling_allocate(caps, noise, total_power=3.0)
        assert np.all(alloc >= -1e-12)

    def test_low_noise_gets_more(self) -> None:
        """Lower noise channel gets more power."""
        caps = np.array([10.0, 10.0])
        noise = np.array([0.1, 10.0])
        alloc = water_filling_allocate(caps, noise, total_power=5.0)
        assert alloc[0] >= alloc[1]

    def test_very_noisy_channel_gets_zero(self) -> None:
        """A very noisy channel may get zero allocation."""
        caps = np.array([10.0, 10.0])
        noise = np.array([0.01, 1000.0])
        alloc = water_filling_allocate(caps, noise, total_power=1.0)
        assert alloc[1] == pytest.approx(0.0, abs=0.1)

    def test_zero_budget(self) -> None:
        """Zero budget → all zeros."""
        caps = np.array([10.0, 10.0])
        noise = np.array([1.0, 1.0])
        alloc = water_filling_allocate(caps, noise, total_power=0.0)
        np.testing.assert_allclose(alloc, [0.0, 0.0], atol=1e-12)

    def test_single_channel(self) -> None:
        """Single channel gets all power."""
        caps = np.array([10.0])
        noise = np.array([1.0])
        alloc = water_filling_allocate(caps, noise, total_power=5.0)
        assert alloc[0] == pytest.approx(5.0, abs=1e-8)


# ------------------------------------------------------------------ #
# MRTAllocator — single task
# ------------------------------------------------------------------ #


class TestMRTAllocatorSingle:
    """Tests for single-task MRT allocation."""

    def test_demand_within_capacity(self) -> None:
        """When demand < capacity, full demand is allocated."""
        pool = _pool(
            (WickensResource.VISUAL, 10.0),
            (WickensResource.MANUAL, 10.0),
        )
        allocator = MRTAllocator()
        result = allocator.allocate(
            {WickensResource.VISUAL: 5.0, WickensResource.MANUAL: 3.0},
            pool,
        )
        assert result.demands[WickensResource.VISUAL] == pytest.approx(5.0)
        assert result.demands[WickensResource.MANUAL] == pytest.approx(3.0)

    def test_demand_exceeds_capacity(self) -> None:
        """When demand > capacity, allocation is capped."""
        pool = _pool((WickensResource.VISUAL, 5.0),)
        allocator = MRTAllocator()
        result = allocator.allocate(
            {WickensResource.VISUAL: 8.0},
            pool,
        )
        assert result.demands[WickensResource.VISUAL] <= 5.0

    def test_bottleneck_identified(self) -> None:
        """Bottleneck is the resource with highest demand/capacity ratio."""
        pool = _pool(
            (WickensResource.VISUAL, 10.0),
            (WickensResource.COGNITIVE, 3.0),
        )
        allocator = MRTAllocator()
        result = allocator.allocate(
            {WickensResource.VISUAL: 5.0, WickensResource.COGNITIVE: 2.5},
            pool,
        )
        # COGNITIVE has ratio 2.5/3 ≈ 0.83 vs VISUAL 5/10 = 0.5
        assert result.bottleneck_resource == WickensResource.COGNITIVE

    def test_unknown_resource_passthrough(self) -> None:
        """Resource not in pool → demand passes through as-is."""
        pool = _pool((WickensResource.VISUAL, 10.0),)
        allocator = MRTAllocator()
        result = allocator.allocate(
            {WickensResource.VISUAL: 5.0, WickensResource.AUDITORY: 3.0},
            pool,
        )
        assert WickensResource.AUDITORY in result.demands


# ------------------------------------------------------------------ #
# Capacity-constrained (concurrent) allocation
# ------------------------------------------------------------------ #


class TestConcurrentAllocation:
    """Tests for multi-task concurrent allocation."""

    def test_no_contention(self) -> None:
        """Tasks on different channels → full allocation."""
        pool = _pool(
            (WickensResource.VISUAL, 10.0),
            (WickensResource.AUDITORY, 10.0),
        )
        allocator = MRTAllocator()
        results = allocator.allocate_concurrent(
            [
                {WickensResource.VISUAL: 5.0},
                {WickensResource.AUDITORY: 5.0},
            ],
            pool,
        )
        assert len(results) == 2
        assert results[0].demands[WickensResource.VISUAL] == pytest.approx(5.0)
        assert results[1].demands[WickensResource.AUDITORY] == pytest.approx(5.0)

    def test_contention_proportional(self) -> None:
        """When total demand > capacity, allocation is proportional."""
        pool = _pool((WickensResource.VISUAL, 10.0),)
        allocator = MRTAllocator()
        results = allocator.allocate_concurrent(
            [
                {WickensResource.VISUAL: 8.0},
                {WickensResource.VISUAL: 12.0},
            ],
            pool,
        )
        # Total demand = 20, capacity = 10
        # Proportional: task0 gets 8/20 * 10 = 4, task1 gets 12/20 * 10 = 6
        assert results[0].demands[WickensResource.VISUAL] == pytest.approx(4.0, rel=0.1)
        assert results[1].demands[WickensResource.VISUAL] == pytest.approx(6.0, rel=0.1)

    def test_three_tasks(self) -> None:
        pool = _pool(
            (WickensResource.VISUAL, 10.0),
            (WickensResource.COGNITIVE, 7.0),
        )
        allocator = MRTAllocator()
        results = allocator.allocate_concurrent(
            [
                {WickensResource.VISUAL: 3.0},
                {WickensResource.VISUAL: 3.0},
                {WickensResource.COGNITIVE: 5.0},
            ],
            pool,
        )
        assert len(results) == 3


# ------------------------------------------------------------------ #
# Priority-based allocation
# ------------------------------------------------------------------ #


class TestPriorityAllocation:
    """Tests for priority-based allocation with preemption."""

    def test_high_priority_gets_full(self) -> None:
        """High-priority task gets its full demand first."""
        pool = _pool((WickensResource.VISUAL, 10.0),)
        allocator = MRTAllocator(priority_weights={"high": 10.0, "low": 1.0})
        results = allocator.allocate_with_preemption(
            [
                {WickensResource.VISUAL: 8.0},
                {WickensResource.VISUAL: 8.0},
            ],
            pool,
            task_ids=["high", "low"],
        )
        high_alloc = next(r for r in results if r.task_id == "high")
        low_alloc = next(r for r in results if r.task_id == "low")
        assert high_alloc.demands[WickensResource.VISUAL] == pytest.approx(8.0)
        assert low_alloc.demands[WickensResource.VISUAL] <= 2.0 + 1e-6

    def test_no_priority_equal_preemption(self) -> None:
        """Without explicit priorities, default weight is 1.0 for all."""
        pool = _pool((WickensResource.VISUAL, 10.0),)
        allocator = MRTAllocator()
        results = allocator.allocate_with_preemption(
            [
                {WickensResource.VISUAL: 6.0},
                {WickensResource.VISUAL: 6.0},
            ],
            pool,
        )
        assert len(results) == 2


# ------------------------------------------------------------------ #
# Multi-channel allocation (water-filling via allocator)
# ------------------------------------------------------------------ #


class TestWaterFillingAllocation:
    """Tests for MRTAllocator.allocate_water_filling."""

    def test_water_filling_returns_allocation(self) -> None:
        pool = _pool(
            (WickensResource.VISUAL, 10.0),
            (WickensResource.AUDITORY, 8.0),
        )
        allocator = MRTAllocator()
        result = allocator.allocate_water_filling(
            {WickensResource.VISUAL: 5.0, WickensResource.AUDITORY: 3.0},
            pool,
        )
        assert isinstance(result, ChannelAllocation)

    def test_water_filling_non_negative(self) -> None:
        pool = _pool(
            (WickensResource.VISUAL, 10.0),
            (WickensResource.COGNITIVE, 7.0),
        )
        allocator = MRTAllocator()
        result = allocator.allocate_water_filling(
            {WickensResource.VISUAL: 8.0, WickensResource.COGNITIVE: 6.0},
            pool,
        )
        for d in result.demands.values():
            assert d >= -1e-12

    def test_water_filling_bottleneck(self) -> None:
        pool = _pool(
            (WickensResource.VISUAL, 10.0),
            (WickensResource.COGNITIVE, 3.0),
        )
        allocator = MRTAllocator()
        result = allocator.allocate_water_filling(
            {WickensResource.VISUAL: 5.0, WickensResource.COGNITIVE: 5.0},
            pool,
        )
        assert result.bottleneck_resource is not None


# ------------------------------------------------------------------ #
# Dynamic reallocation
# ------------------------------------------------------------------ #


class TestDynamicReallocation:
    """Tests for MRTAllocator.reallocate."""

    def test_realloc_unchanged_tasks_stable(self) -> None:
        """Tasks with unchanged demands should get similar allocation."""
        pool = _pool(
            (WickensResource.VISUAL, 10.0),
            (WickensResource.AUDITORY, 8.0),
        )
        allocator = MRTAllocator()
        initial = allocator.allocate_concurrent(
            [
                {WickensResource.VISUAL: 5.0},
                {WickensResource.AUDITORY: 4.0},
            ],
            pool,
            task_ids=["t1", "t2"],
        )
        # Reallocate with t1 changed, t2 unchanged
        updated = allocator.reallocate(
            initial,
            {"t1": {WickensResource.VISUAL: 8.0}},
            pool,
        )
        assert len(updated) == 2

    def test_realloc_empty_new_demands(self) -> None:
        """No changes → same allocation."""
        pool = _pool((WickensResource.VISUAL, 10.0),)
        allocator = MRTAllocator()
        initial = allocator.allocate_concurrent(
            [{WickensResource.VISUAL: 5.0}],
            pool,
            task_ids=["t1"],
        )
        updated = allocator.reallocate(initial, {}, pool)
        assert len(updated) == 1
