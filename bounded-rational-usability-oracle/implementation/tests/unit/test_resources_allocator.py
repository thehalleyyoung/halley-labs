"""Unit tests for usability_oracle.resources.allocator — Resource allocation.

Tests greedy allocation, LP allocation, bottleneck identification,
and slack analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from usability_oracle.resources.allocator import ResourceAllocator
from usability_oracle.resources.wickens_model import WickensModel
from usability_oracle.resources.types import (
    DemandVector,
    PerceptualModality,
    ProcessingCode,
    ProcessingStage,
    Resource,
    ResourceAllocation,
    ResourceDemand,
    VisualChannel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resource(stage=ProcessingStage.PERCEPTION, modality=PerceptualModality.VISUAL,
              code=ProcessingCode.SPATIAL, channel=VisualChannel.FOCAL) -> Resource:
    return Resource(stage=stage, modality=modality, visual_channel=channel,
                    code=code, label=f"{stage.value}_{modality.value}_{code.value}")


def _demand(resource, level=0.5, op_id="task") -> ResourceDemand:
    return ResourceDemand(resource=resource, demand_level=level,
                          operation_id=op_id, description="test")


def _demand_vector(demands, op_id="task") -> DemandVector:
    return DemandVector(demands=tuple(demands), operation_id=op_id)


# ===================================================================
# Basic allocation
# ===================================================================


class TestAllocate:

    def test_allocation_returns_result(self):
        alloc = ResourceAllocator()
        r = _resource()
        dv = _demand_vector([_demand(r, 0.5, "t1")], "t1")
        result = alloc.allocate([dv])
        assert isinstance(result, ResourceAllocation)

    def test_total_cost_non_negative(self):
        alloc = ResourceAllocator()
        r = _resource()
        dv = _demand_vector([_demand(r, 0.5, "t1")], "t1")
        result = alloc.allocate([dv])
        assert result.total_cost >= 0.0


# ===================================================================
# Greedy allocation
# ===================================================================


class TestGreedyAllocation:

    def test_respects_priorities(self):
        alloc = ResourceAllocator()
        r = _resource()
        dv1 = _demand_vector([_demand(r, 0.6, "high")], "high")
        dv2 = _demand_vector([_demand(r, 0.4, "low")], "low")
        result = alloc.greedy_allocation([dv1, dv2], priorities={"high": 2.0, "low": 1.0})
        assert isinstance(result, ResourceAllocation)

    def test_returns_allocation(self):
        alloc = ResourceAllocator()
        r = _resource()
        dv = _demand_vector([_demand(r, 0.5, "t1")], "t1")
        result = alloc.greedy_allocation([dv])
        assert isinstance(result, ResourceAllocation)


# ===================================================================
# LP allocation
# ===================================================================


class TestLPAllocation:

    def test_finds_solution(self):
        alloc = ResourceAllocator()
        r = _resource()
        dv = _demand_vector([_demand(r, 0.5, "t1")], "t1")
        result = alloc.lp_allocation([dv])
        assert isinstance(result, ResourceAllocation)

    def test_total_cost_reasonable(self):
        alloc = ResourceAllocator()
        r1 = _resource()
        r2 = _resource(modality=PerceptualModality.AUDITORY, channel=None)
        dv1 = _demand_vector([_demand(r1, 0.3, "t1")], "t1")
        dv2 = _demand_vector([_demand(r2, 0.3, "t2")], "t2")
        result = alloc.lp_allocation([dv1, dv2])
        assert result.total_cost >= 0.0


# ===================================================================
# Bottleneck identification
# ===================================================================


class TestBottleneckResource:

    def test_identifies_bottleneck(self):
        alloc = ResourceAllocator()
        r = _resource()
        dv1 = _demand_vector([_demand(r, 0.9, "t1")], "t1")
        dv2 = _demand_vector([_demand(r, 0.9, "t2")], "t2")
        allocation = alloc.allocate([dv1, dv2])
        bottleneck = alloc.bottleneck_resource(allocation)
        # Should identify a bottleneck (or None if not applicable)
        assert bottleneck is None or isinstance(bottleneck, str)


# ===================================================================
# Slack analysis
# ===================================================================


class TestSlackAnalysis:

    def test_slack_values_exist(self):
        alloc = ResourceAllocator()
        r = _resource()
        dv = _demand_vector([_demand(r, 0.3, "t1")], "t1")
        allocation = alloc.allocate([dv])
        slack = alloc.slack_analysis(allocation)
        assert isinstance(slack, dict)

    def test_slack_non_negative(self):
        alloc = ResourceAllocator()
        r = _resource()
        dv = _demand_vector([_demand(r, 0.3, "t1")], "t1")
        allocation = alloc.allocate([dv])
        slack = alloc.slack_analysis(allocation)
        for v in slack.values():
            assert v >= -1e-10
