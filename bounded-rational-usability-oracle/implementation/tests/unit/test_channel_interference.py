"""Unit tests for usability_oracle.channel.interference.

Tests cover interference computation, four-dimensional resource conflict,
time-sharing efficiency, and interference matrix construction.
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
from usability_oracle.channel.interference import (
    ResourceProfile,
    WickensInterferenceModel,
    build_standard_interference_matrix,
    dimension_conflict,
    profile_interference,
)


# ------------------------------------------------------------------ #
# Dimension conflict
# ------------------------------------------------------------------ #


class TestDimensionConflict:
    """Tests for pairwise dimension-conflict weights."""

    def test_same_modality_high_conflict(self) -> None:
        """VISUAL–VISUAL should have high conflict weight."""
        w = dimension_conflict(WickensResource.VISUAL, WickensResource.VISUAL)
        assert w >= 0.7

    def test_cross_modality_low_conflict(self) -> None:
        """VISUAL–AUDITORY should have low conflict."""
        w = dimension_conflict(WickensResource.VISUAL, WickensResource.AUDITORY)
        assert w <= 0.3

    def test_same_stage_cognitive(self) -> None:
        """COGNITIVE–COGNITIVE conflict should be 1.0."""
        w = dimension_conflict(WickensResource.COGNITIVE, WickensResource.COGNITIVE)
        assert w == pytest.approx(1.0)

    def test_different_dimensions_zero(self) -> None:
        """Resources from different dimensions → 0 conflict."""
        w = dimension_conflict(WickensResource.VISUAL, WickensResource.COGNITIVE)
        assert w == pytest.approx(0.0)

    def test_spatial_spatial(self) -> None:
        w = dimension_conflict(WickensResource.SPATIAL, WickensResource.SPATIAL)
        assert w == pytest.approx(1.0)

    def test_spatial_verbal_low(self) -> None:
        w = dimension_conflict(WickensResource.SPATIAL, WickensResource.VERBAL)
        assert w <= 0.3

    def test_manual_manual(self) -> None:
        w = dimension_conflict(WickensResource.MANUAL, WickensResource.MANUAL)
        assert w == pytest.approx(1.0)

    def test_manual_vocal_low(self) -> None:
        w = dimension_conflict(WickensResource.MANUAL, WickensResource.VOCAL)
        assert w <= 0.2

    def test_focal_focal(self) -> None:
        w = dimension_conflict(WickensResource.FOCAL, WickensResource.FOCAL)
        assert w == pytest.approx(1.0)

    def test_focal_ambient_low(self) -> None:
        w = dimension_conflict(WickensResource.FOCAL, WickensResource.AMBIENT)
        assert w <= 0.4


# ------------------------------------------------------------------ #
# Four-dimensional resource conflict (profile interference)
# ------------------------------------------------------------------ #


class TestProfileInterference:
    """Tests for multi-dimensional profile interference."""

    def test_identical_profiles_high(self) -> None:
        """Two identical profiles → high interference."""
        p = ResourceProfile(
            stage=WickensResource.COGNITIVE,
            modality=WickensResource.VISUAL,
            code=WickensResource.SPATIAL,
            effector=WickensResource.MANUAL,
            demand=1.0,
        )
        intf = profile_interference(p, p)
        assert intf > 0.3

    def test_different_modality_lower(self) -> None:
        """Different modalities → lower interference."""
        pa = ResourceProfile(
            stage=WickensResource.COGNITIVE,
            modality=WickensResource.VISUAL,
            demand=0.8,
        )
        pb = ResourceProfile(
            stage=WickensResource.COGNITIVE,
            modality=WickensResource.AUDITORY,
            demand=0.8,
        )
        pc = ResourceProfile(
            stage=WickensResource.COGNITIVE,
            modality=WickensResource.VISUAL,
            demand=0.8,
        )
        # Same modality should interfere more
        intf_same = profile_interference(pa, pc)
        intf_diff = profile_interference(pa, pb)
        assert intf_same > intf_diff

    def test_zero_demand_zero_interference(self) -> None:
        pa = ResourceProfile(demand=0.0)
        pb = ResourceProfile(demand=1.0)
        intf = profile_interference(pa, pb)
        assert intf == pytest.approx(0.0, abs=1e-12)

    def test_interference_bounded(self) -> None:
        """Profile interference should be in [0, 1]."""
        pa = ResourceProfile(
            stage=WickensResource.COGNITIVE,
            modality=WickensResource.VISUAL,
            code=WickensResource.SPATIAL,
            effector=WickensResource.MANUAL,
            demand=1.0,
        )
        intf = profile_interference(pa, pa)
        assert 0.0 <= intf <= 1.0

    def test_different_stages_zero(self) -> None:
        """Different processing stages → 0 interference (stage conflict = 0)."""
        pa = ResourceProfile(stage=WickensResource.PERCEPTUAL, demand=1.0)
        pb = ResourceProfile(stage=WickensResource.RESPONSE, demand=1.0)
        intf = profile_interference(pa, pb)
        assert intf == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------ #
# WickensInterferenceModel
# ------------------------------------------------------------------ #


class TestWickensInterferenceModel:
    """Tests for the interference model class."""

    @pytest.fixture
    def model(self) -> WickensInterferenceModel:
        return WickensInterferenceModel()

    def test_compute_interference_zero_for_independent(self, model: WickensInterferenceModel) -> None:
        """Non-overlapping allocations → ~0 interference."""
        a = ChannelAllocation(task_id="a", demands={WickensResource.VISUAL: 0.5})
        b = ChannelAllocation(task_id="b", demands={WickensResource.AUDITORY: 0.5})
        intf = model.compute_interference(a, b)
        # Visual and Auditory share "modality" dimension but with low weight
        assert intf <= 0.15

    def test_compute_interference_positive_for_shared(self, model: WickensInterferenceModel) -> None:
        """Shared resources → positive interference."""
        a = ChannelAllocation(task_id="a", demands={WickensResource.VISUAL: 0.8})
        b = ChannelAllocation(task_id="b", demands={WickensResource.VISUAL: 0.7})
        intf = model.compute_interference(a, b)
        assert intf > 0

    def test_interference_clamped(self, model: WickensInterferenceModel) -> None:
        """Interference should be clamped to [0, 1]."""
        a = ChannelAllocation(task_id="a", demands={
            WickensResource.VISUAL: 1.0,
            WickensResource.COGNITIVE: 1.0,
            WickensResource.MANUAL: 1.0,
        })
        b = ChannelAllocation(task_id="b", demands={
            WickensResource.VISUAL: 1.0,
            WickensResource.COGNITIVE: 1.0,
            WickensResource.MANUAL: 1.0,
        })
        intf = model.compute_interference(a, b)
        assert intf <= 1.0

    def test_build_interference_matrix(self, model: WickensInterferenceModel) -> None:
        resources = [WickensResource.VISUAL, WickensResource.AUDITORY, WickensResource.COGNITIVE]
        mat = model.build_interference_matrix(resources)
        assert mat.n_channels == 3
        # Diagonal should be 0
        for i in range(3):
            assert mat.interference(i, i) == pytest.approx(0.0)

    def test_interference_matrix_symmetric(self, model: WickensInterferenceModel) -> None:
        resources = [WickensResource.VISUAL, WickensResource.AUDITORY]
        mat = model.build_interference_matrix(resources)
        assert mat.interference(0, 1) == pytest.approx(mat.interference(1, 0))


# ------------------------------------------------------------------ #
# Time-sharing efficiency
# ------------------------------------------------------------------ #


class TestTimeSharingEfficiency:
    """Tests for time_sharing_efficiency."""

    def test_single_task_perfect(self) -> None:
        """One task → η = 1.0."""
        model = WickensInterferenceModel()
        a = ChannelAllocation(task_id="a", demands={WickensResource.VISUAL: 0.5})
        eta = model.time_sharing_efficiency([a])
        assert eta == pytest.approx(1.0)

    def test_more_interference_lower_efficiency(self) -> None:
        """Higher interference → lower efficiency."""
        model = WickensInterferenceModel()
        a = ChannelAllocation(task_id="a", demands={WickensResource.VISUAL: 0.8})
        b_same = ChannelAllocation(task_id="b", demands={WickensResource.VISUAL: 0.8})
        b_diff = ChannelAllocation(task_id="b", demands={WickensResource.AUDITORY: 0.8})
        eta_same = model.time_sharing_efficiency([a, b_same])
        eta_diff = model.time_sharing_efficiency([a, b_diff])
        assert eta_diff >= eta_same

    def test_efficiency_positive(self) -> None:
        """Efficiency is always > 0."""
        model = WickensInterferenceModel()
        a = ChannelAllocation(task_id="a", demands={WickensResource.VISUAL: 1.0})
        b = ChannelAllocation(task_id="b", demands={WickensResource.VISUAL: 1.0})
        eta = model.time_sharing_efficiency([a, b])
        assert eta > 0

    def test_temporal_overlap_modulation(self) -> None:
        """Zero overlap → zero temporal interference."""
        model = WickensInterferenceModel()
        a = ChannelAllocation(task_id="a", demands={WickensResource.VISUAL: 0.8})
        b = ChannelAllocation(task_id="b", demands={WickensResource.VISUAL: 0.8})
        intf_full = model.temporal_overlap_interference(a, b, 1.0)
        intf_none = model.temporal_overlap_interference(a, b, 0.0)
        assert intf_none == pytest.approx(0.0)
        assert intf_full > 0


# ------------------------------------------------------------------ #
# Interference matrix construction
# ------------------------------------------------------------------ #


class TestInterferenceMatrixConstruction:
    """Tests for build_standard_interference_matrix."""

    def test_standard_matrix_covers_all_resources(self) -> None:
        mat = build_standard_interference_matrix()
        assert mat.n_channels == len(list(WickensResource))

    def test_standard_matrix_diagonal_zero(self) -> None:
        mat = build_standard_interference_matrix()
        for i in range(mat.n_channels):
            assert mat.interference(i, i) == pytest.approx(0.0)

    def test_standard_matrix_symmetric(self) -> None:
        mat = build_standard_interference_matrix()
        n = mat.n_channels
        for i in range(n):
            for j in range(i + 1, n):
                assert mat.interference(i, j) == pytest.approx(
                    mat.interference(j, i), abs=1e-12,
                )

    def test_effective_capacity_with_no_load(self) -> None:
        """Effective capacity without load = base capacity."""
        model = WickensInterferenceModel()
        ch = ResourceChannel(resource=WickensResource.VISUAL, capacity_bits_per_s=10.0)
        mat = model.build_interference_matrix([WickensResource.VISUAL])
        eff = model.effective_capacity(ch, {}, mat)
        assert eff == pytest.approx(10.0)

    def test_effective_capacity_reduced_by_load(self) -> None:
        """Concurrent load should reduce effective capacity."""
        model = WickensInterferenceModel()
        resources = [WickensResource.VISUAL, WickensResource.AUDITORY]
        mat = model.build_interference_matrix(resources)
        ch = ResourceChannel(resource=WickensResource.VISUAL, capacity_bits_per_s=10.0)
        # No concurrent load
        eff_alone = model.effective_capacity(ch, {}, mat)
        # With concurrent auditory load
        eff_loaded = model.effective_capacity(
            ch, {WickensResource.AUDITORY: 0.8}, mat,
        )
        assert eff_loaded <= eff_alone
