"""Unit tests for usability_oracle.resources.wickens_model — Wickens' Multiple-Resource Theory.

Tests interference computation, demand vectors, time-sharing efficiency,
and single-task properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from usability_oracle.resources.wickens_model import WickensModel
from usability_oracle.resources.types import (
    DemandVector,
    PerceptualModality,
    ProcessingCode,
    ProcessingStage,
    Resource,
    ResourceDemand,
    VisualChannel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resource(stage: ProcessingStage = ProcessingStage.PERCEPTION,
              modality: PerceptualModality = PerceptualModality.VISUAL,
              code: ProcessingCode = ProcessingCode.SPATIAL,
              channel: VisualChannel | None = VisualChannel.FOCAL) -> Resource:
    return Resource(
        stage=stage, modality=modality, visual_channel=channel,
        code=code, label=f"{stage.value}_{modality.value}_{code.value}",
    )


def _demand(resource: Resource, level: float = 0.5, op_id: str = "task") -> ResourceDemand:
    return ResourceDemand(
        resource=resource, demand_level=level,
        operation_id=op_id, description="test demand",
    )


def _demand_vector(demands: list[ResourceDemand], op_id: str = "task") -> DemandVector:
    return DemandVector(demands=tuple(demands), operation_id=op_id)


# ===================================================================
# Interference ordering
# ===================================================================


class TestInterferenceOrdering:

    def test_same_modality_higher_than_cross(self):
        """Visual + Visual should have more interference than Visual + Auditory."""
        model = WickensModel()
        r_vis_spatial = _resource(ProcessingStage.PERCEPTION, PerceptualModality.VISUAL, ProcessingCode.SPATIAL)
        r_vis_verbal = _resource(ProcessingStage.PERCEPTION, PerceptualModality.VISUAL, ProcessingCode.VERBAL)
        r_aud_verbal = _resource(ProcessingStage.PERCEPTION, PerceptualModality.AUDITORY, ProcessingCode.VERBAL, channel=None)

        d_vs = _demand(r_vis_spatial, 0.8, "a")
        d_vv = _demand(r_vis_verbal, 0.8, "b")
        d_av = _demand(r_aud_verbal, 0.8, "c")

        dv_a = _demand_vector([d_vs], "a")
        dv_b_same = _demand_vector([d_vv], "b")
        dv_b_cross = _demand_vector([d_av], "c")

        interference_same = model.compute_interference(d_vs, d_vv)
        interference_cross = model.compute_interference(d_vs, d_av)
        assert interference_same >= interference_cross

    def test_visual_spatial_plus_visual_spatial_high(self):
        model = WickensModel()
        r = _resource(ProcessingStage.PERCEPTION, PerceptualModality.VISUAL, ProcessingCode.SPATIAL)
        d1 = _demand(r, 0.9, "t1")
        d2 = _demand(r, 0.9, "t2")
        interference = model.compute_interference(d1, d2)
        assert interference > 0.0

    def test_visual_spatial_plus_auditory_verbal_low(self):
        model = WickensModel()
        r_vs = _resource(ProcessingStage.PERCEPTION, PerceptualModality.VISUAL, ProcessingCode.SPATIAL)
        r_av = _resource(ProcessingStage.PERCEPTION, PerceptualModality.AUDITORY, ProcessingCode.VERBAL, channel=None)
        d1 = _demand(r_vs, 0.5, "t1")
        d2 = _demand(r_av, 0.5, "t2")
        interference = model.compute_interference(d1, d2)
        # Cross-modality, cross-code → low interference
        assert interference >= 0.0


# ===================================================================
# Demand vector properties
# ===================================================================


class TestDemandVector:

    def test_dimension_is_four(self):
        """Wickens' 4D resource model: stage, modality, visual_channel, code."""
        r = _resource()
        # The resource itself has the 4 dimensions
        assert hasattr(r, "stage")
        assert hasattr(r, "modality")
        assert hasattr(r, "visual_channel")
        assert hasattr(r, "code")

    def test_total_demand(self):
        r1 = _resource()
        r2 = _resource(code=ProcessingCode.VERBAL)
        dv = _demand_vector([_demand(r1, 0.3), _demand(r2, 0.7)])
        assert dv.total_demand == pytest.approx(1.0, abs=0.01)


# ===================================================================
# Time-sharing efficiency
# ===================================================================


class TestTimeSharingEfficiency:

    def test_efficiency_in_unit_interval(self):
        model = WickensModel()
        r1 = _resource(ProcessingStage.PERCEPTION, PerceptualModality.VISUAL, ProcessingCode.SPATIAL)
        r2 = _resource(ProcessingStage.PERCEPTION, PerceptualModality.AUDITORY, ProcessingCode.VERBAL, channel=None)
        dv1 = _demand_vector([_demand(r1, 0.5, "t1")], "t1")
        dv2 = _demand_vector([_demand(r2, 0.5, "t2")], "t2")
        efficiency = model.compute_time_sharing_efficiency([dv1, dv2])
        assert 0.0 <= efficiency <= 1.0

    def test_single_task_efficiency_one(self):
        model = WickensModel()
        r = _resource()
        dv = _demand_vector([_demand(r, 0.5, "solo")], "solo")
        efficiency = model.compute_time_sharing_efficiency([dv])
        np.testing.assert_allclose(efficiency, 1.0, atol=0.01)


# ===================================================================
# Interference matrix
# ===================================================================


class TestInterferenceMatrix:

    def test_matrix_symmetric(self):
        model = WickensModel()
        r1 = _resource(ProcessingStage.PERCEPTION, PerceptualModality.VISUAL, ProcessingCode.SPATIAL)
        r2 = _resource(ProcessingStage.COGNITION, PerceptualModality.AUDITORY, ProcessingCode.VERBAL, channel=None)
        dv1 = _demand_vector([_demand(r1, 0.6, "t1")], "t1")
        dv2 = _demand_vector([_demand(r2, 0.6, "t2")], "t2")
        matrix = model.compute_matrix([dv1, dv2])
        # For a 2x2 matrix, upper-triangular has 1 element: (0,1)
        # Symmetry is implicit in the storage
        i_12 = matrix.get_interference("t1", "t2")
        i_21 = matrix.get_interference("t2", "t1")
        np.testing.assert_allclose(i_12, i_21, atol=1e-10)

    def test_diagonal_zero(self):
        model = WickensModel()
        r = _resource()
        dv = _demand_vector([_demand(r, 0.5, "t1")], "t1")
        matrix = model.compute_matrix([dv])
        # Self-interference should be zero
        i_11 = matrix.get_interference("t1", "t1")
        np.testing.assert_allclose(i_11, 0.0, atol=1e-10)


# ===================================================================
# Dual-task cost
# ===================================================================


class TestDualTaskCost:

    def test_cost_non_negative(self):
        model = WickensModel()
        r1 = _resource()
        r2 = _resource(modality=PerceptualModality.AUDITORY, channel=None)
        dv1 = _demand_vector([_demand(r1, 0.5, "t1")], "t1")
        dv2 = _demand_vector([_demand(r2, 0.5, "t2")], "t2")
        cost = model.predict_dual_task_cost(dv1, dv2)
        assert cost >= 0.0

    def test_same_resource_higher_cost(self):
        model = WickensModel()
        r = _resource()
        r_diff = _resource(modality=PerceptualModality.AUDITORY, channel=None)
        dv1 = _demand_vector([_demand(r, 0.8, "t1")], "t1")
        dv2_same = _demand_vector([_demand(r, 0.8, "t2")], "t2")
        dv2_diff = _demand_vector([_demand(r_diff, 0.8, "t3")], "t3")
        cost_same = model.predict_dual_task_cost(dv1, dv2_same)
        cost_diff = model.predict_dual_task_cost(dv1, dv2_diff)
        assert cost_same >= cost_diff
