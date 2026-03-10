"""Unit tests for usability_oracle.channel.wickens.

Tests cover the Wickens MRT model, resource demand vectors, conflict
matrix computation, performance prediction, and SEEV attention model.
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
from usability_oracle.channel.wickens import (
    AOI,
    MRTDemandVector,
    PerformancePrediction,
    SEEVWeights,
    WickensMRTModel,
    compute_conflict_matrix,
    compute_demand_vector,
    predict_performance,
    seev_attention_allocation,
)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _default_pool() -> ResourcePool:
    """A minimal resource pool for testing."""
    channels = (
        ResourceChannel(resource=WickensResource.VISUAL, capacity_bits_per_s=10.0),
        ResourceChannel(resource=WickensResource.AUDITORY, capacity_bits_per_s=8.0),
        ResourceChannel(resource=WickensResource.COGNITIVE, capacity_bits_per_s=7.0),
        ResourceChannel(resource=WickensResource.MANUAL, capacity_bits_per_s=10.0),
        ResourceChannel(resource=WickensResource.VOCAL, capacity_bits_per_s=6.0),
    )
    return ResourcePool(channels=channels, label="test_pool")


# ------------------------------------------------------------------ #
# MRT demand vector
# ------------------------------------------------------------------ #


class TestMRTDemandVector:
    """Tests for demand vector computation from task descriptions."""

    def test_basic_demand_vector(self) -> None:
        desc = {
            "task_id": "read_text",
            "visual_load": 0.8,
            "cognitive_load": 0.5,
            "manual_load": 0.2,
        }
        dv = compute_demand_vector(desc)
        assert dv.task_id == "read_text"
        assert WickensResource.VISUAL in dv.modality_demands
        assert dv.modality_demands[WickensResource.VISUAL] == pytest.approx(0.8)
        assert WickensResource.COGNITIVE in dv.stage_demands
        assert dv.stage_demands[WickensResource.COGNITIVE] == pytest.approx(0.5)

    def test_total_demand(self) -> None:
        dv = MRTDemandVector(
            task_id="t",
            stage_demands={WickensResource.COGNITIVE: 0.5},
            modality_demands={WickensResource.VISUAL: 0.3},
        )
        assert dv.total_demand == pytest.approx(0.8)

    def test_all_demands_flat(self) -> None:
        dv = MRTDemandVector(
            task_id="t",
            stage_demands={WickensResource.COGNITIVE: 0.5},
            modality_demands={WickensResource.VISUAL: 0.3},
            code_demands={WickensResource.SPATIAL: 0.2},
        )
        all_d = dv.all_demands
        assert len(all_d) == 3

    def test_as_numpy(self) -> None:
        dv = MRTDemandVector(
            task_id="t",
            stage_demands={WickensResource.COGNITIVE: 0.5},
        )
        order = [WickensResource.COGNITIVE, WickensResource.VISUAL]
        arr = dv.as_numpy(order)
        assert arr[0] == pytest.approx(0.5)
        assert arr[1] == pytest.approx(0.0)

    def test_empty_description(self) -> None:
        dv = compute_demand_vector({})
        assert dv.total_demand == pytest.approx(0.0)

    def test_clamping(self) -> None:
        """Loads outside [0,1] should be clamped."""
        desc = {"visual_load": 1.5, "auditory_load": -0.2}
        dv = compute_demand_vector(desc)
        assert dv.modality_demands.get(WickensResource.VISUAL, 0) <= 1.0


# ------------------------------------------------------------------ #
# Conflict matrix
# ------------------------------------------------------------------ #


class TestConflictMatrix:
    """Tests for compute_conflict_matrix."""

    def test_self_conflict_zero(self) -> None:
        """Diagonal should be 0 (no self-conflict)."""
        dv = MRTDemandVector(
            task_id="t",
            stage_demands={WickensResource.COGNITIVE: 0.5},
            modality_demands={WickensResource.VISUAL: 0.5},
        )
        mat = compute_conflict_matrix([dv, dv])
        assert mat[0, 0] == pytest.approx(0.0)
        assert mat[1, 1] == pytest.approx(0.0)

    def test_symmetric(self) -> None:
        """Conflict matrix should be symmetric."""
        dv1 = compute_demand_vector({"visual_load": 0.8, "cognitive_load": 0.5})
        dv2 = compute_demand_vector({"auditory_load": 0.6, "cognitive_load": 0.4})
        mat = compute_conflict_matrix([dv1, dv2])
        assert mat[0, 1] == pytest.approx(mat[1, 0])

    def test_no_overlap_zero_conflict(self) -> None:
        """Tasks with no shared resources should have 0 conflict."""
        dv1 = MRTDemandVector(
            task_id="a", modality_demands={WickensResource.VISUAL: 1.0},
        )
        dv2 = MRTDemandVector(
            task_id="b", modality_demands={WickensResource.AUDITORY: 1.0},
        )
        mat = compute_conflict_matrix([dv1, dv2])
        assert mat[0, 1] == pytest.approx(0.0)

    def test_shared_modality_positive_conflict(self) -> None:
        """Two visual tasks should conflict."""
        dv1 = MRTDemandVector(
            task_id="a", modality_demands={WickensResource.VISUAL: 0.8},
        )
        dv2 = MRTDemandVector(
            task_id="b", modality_demands={WickensResource.VISUAL: 0.6},
        )
        mat = compute_conflict_matrix([dv1, dv2])
        assert mat[0, 1] > 0

    def test_conflict_bounded(self) -> None:
        """Conflict values should be in [0, 1]."""
        dv1 = compute_demand_vector({
            "visual_load": 1.0, "cognitive_load": 1.0, "manual_load": 1.0,
        })
        dv2 = compute_demand_vector({
            "visual_load": 1.0, "cognitive_load": 1.0, "manual_load": 1.0,
        })
        mat = compute_conflict_matrix([dv1, dv2])
        assert mat[0, 1] <= 1.0

    def test_three_tasks(self) -> None:
        """3 tasks → 3×3 matrix."""
        dvs = [
            compute_demand_vector({"visual_load": 0.5}),
            compute_demand_vector({"auditory_load": 0.5}),
            compute_demand_vector({"visual_load": 0.3, "auditory_load": 0.3}),
        ]
        mat = compute_conflict_matrix(dvs)
        assert mat.shape == (3, 3)


# ------------------------------------------------------------------ #
# Performance prediction
# ------------------------------------------------------------------ #


class TestPerformancePrediction:
    """Tests for predict_performance under resource competition."""

    def test_single_task_baseline(self) -> None:
        """Single task with low load → time close to baseline."""
        dv = MRTDemandVector(
            task_id="t",
            stage_demands={WickensResource.COGNITIVE: 0.1},
        )
        pool = _default_pool()
        pred = predict_performance(dv, pool, base_completion_time_s=2.0)
        assert pred.predicted_completion_time_s >= 2.0

    def test_workload_in_range(self) -> None:
        dv = compute_demand_vector({"visual_load": 0.5, "cognitive_load": 0.5})
        pool = _default_pool()
        pred = predict_performance(dv, pool)
        assert 0.0 <= pred.workload_index <= 1.0

    def test_error_rate_in_range(self) -> None:
        dv = compute_demand_vector({"visual_load": 0.9, "cognitive_load": 0.9})
        pool = _default_pool()
        pred = predict_performance(dv, pool)
        assert 0.0 <= pred.predicted_error_rate <= 1.0

    def test_concurrent_tasks_increase_time(self) -> None:
        """Concurrent task demands should increase completion time."""
        dv = compute_demand_vector({"visual_load": 0.5, "cognitive_load": 0.5})
        concurrent = [compute_demand_vector({"visual_load": 0.8, "cognitive_load": 0.7})]
        pool = _default_pool()
        pred_alone = predict_performance(dv, pool)
        pred_dual = predict_performance(dv, pool, concurrent)
        assert pred_dual.predicted_completion_time_s >= pred_alone.predicted_completion_time_s

    def test_bottleneck_identified(self) -> None:
        """Bottleneck resource should be the one with highest demand/capacity ratio."""
        dv = MRTDemandVector(
            task_id="t",
            stage_demands={WickensResource.COGNITIVE: 0.9},
            modality_demands={WickensResource.VISUAL: 0.1},
        )
        pool = _default_pool()
        pred = predict_performance(dv, pool)
        assert pred.bottleneck_resource is not None

    def test_performance_decrement_non_negative(self) -> None:
        dv = compute_demand_vector({"visual_load": 0.5})
        pool = _default_pool()
        pred = predict_performance(dv, pool)
        assert pred.performance_decrement >= 0


# ------------------------------------------------------------------ #
# SEEV attention model
# ------------------------------------------------------------------ #


class TestSEEVAttentionModel:
    """Tests for SEEV-based attention allocation."""

    def test_single_aoi(self) -> None:
        """Single AOI gets 100% attention."""
        aois = [AOI(aoi_id="a", salience=0.5, effort=0.3, expectancy=0.5, value=0.8)]
        result = seev_attention_allocation(aois)
        assert result["a"] == pytest.approx(1.0)

    def test_two_aois_sum_to_one(self) -> None:
        aois = [
            AOI(aoi_id="a", salience=0.8, effort=0.2, expectancy=0.6, value=0.9),
            AOI(aoi_id="b", salience=0.3, effort=0.5, expectancy=0.4, value=0.3),
        ]
        result = seev_attention_allocation(aois)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-10)

    def test_higher_value_gets_more_attention(self) -> None:
        """AOI with higher value should get more attention (all else equal)."""
        aois = [
            AOI(aoi_id="high", salience=0.5, effort=0.3, expectancy=0.5, value=0.9),
            AOI(aoi_id="low", salience=0.5, effort=0.3, expectancy=0.5, value=0.1),
        ]
        result = seev_attention_allocation(aois)
        assert result["high"] > result["low"]

    def test_empty_aois(self) -> None:
        result = seev_attention_allocation([])
        assert result == {}

    def test_custom_weights(self) -> None:
        """Custom SEEV weights should affect allocation."""
        aois = [
            AOI(aoi_id="a", salience=1.0, effort=0.0, expectancy=0.0, value=0.0),
            AOI(aoi_id="b", salience=0.0, effort=0.0, expectancy=0.0, value=1.0),
        ]
        w_sal = SEEVWeights(salience=1.0, effort=0.0, expectancy=0.0, value=0.0)
        result_sal = seev_attention_allocation(aois, w_sal)
        assert result_sal["a"] > result_sal["b"]

        w_val = SEEVWeights(salience=0.0, effort=0.0, expectancy=0.0, value=1.0)
        result_val = seev_attention_allocation(aois, w_val)
        assert result_val["b"] > result_val["a"]


# ------------------------------------------------------------------ #
# WickensMRTModel integration
# ------------------------------------------------------------------ #


class TestWickensMRTModel:
    """Tests for the full WickensMRTModel class."""

    def test_analyse_task(self) -> None:
        model = WickensMRTModel()
        dv = model.analyse_task({"visual_load": 0.5, "cognitive_load": 0.3})
        assert dv.total_demand > 0

    def test_conflict_matrix(self) -> None:
        model = WickensMRTModel()
        dvs = model.analyse_tasks([
            {"visual_load": 0.8},
            {"visual_load": 0.6},
        ])
        mat = model.conflict_matrix(dvs)
        assert mat.shape == (2, 2)

    def test_total_conflict(self) -> None:
        model = WickensMRTModel()
        dvs = model.analyse_tasks([
            {"visual_load": 0.8, "cognitive_load": 0.5},
            {"visual_load": 0.6, "cognitive_load": 0.4},
        ])
        tc = model.total_conflict(dvs)
        assert tc >= 0

    def test_predict(self) -> None:
        pool = _default_pool()
        model = WickensMRTModel(pool=pool)
        pred = model.predict({"visual_load": 0.5, "cognitive_load": 0.3})
        assert isinstance(pred, PerformancePrediction)
        assert pred.predicted_completion_time_s > 0

    def test_seev_allocate(self) -> None:
        model = WickensMRTModel()
        aois = [
            AOI(aoi_id="a", value=0.9),
            AOI(aoi_id="b", value=0.1),
        ]
        result = model.seev_allocate(aois)
        assert "a" in result and "b" in result
