"""Shared pytest fixtures for MARACE test suite."""

import numpy as np
import pytest

from marace.trace.events import (
    Event, EventType, ActionEvent, ObservationEvent,
    CommunicationEvent, EnvironmentEvent, SyncEvent,
    VectorClock as VCDict,
    make_action_event, make_observation_event,
    make_communication_event, make_environment_event, make_sync_event,
    vc_zero, vc_increment, vc_merge,
)
from marace.trace.trace import ExecutionTrace, MultiAgentTrace
from marace.hb.vector_clock import VectorClock
from marace.hb.hb_graph import HBGraph
from marace.abstract.zonotope import Zonotope
from marace.abstract.hb_constraints import HBConstraint, HBConstraintSet


AGENTS = ["agent_0", "agent_1", "agent_2"]


@pytest.fixture
def agent_ids():
    return list(AGENTS)


@pytest.fixture
def two_agents():
    return ["agent_0", "agent_1"]


@pytest.fixture
def vc_a0():
    return vc_increment(vc_zero(AGENTS), "agent_0")


@pytest.fixture
def vc_a1():
    return vc_increment(vc_zero(AGENTS), "agent_1")


@pytest.fixture
def sample_action_event(vc_a0):
    return make_action_event(
        agent_id="agent_0", timestamp=0.1,
        action_vector=np.array([1.0, 0.0]),
        vector_clock=vc_a0,
    )


@pytest.fixture
def sample_obs_event(vc_a1):
    return make_observation_event(
        agent_id="agent_1", timestamp=0.2,
        observation_vector=np.array([0.5, 0.5, 0.5]),
        vector_clock=vc_a1,
        source_agents=["agent_0"],
    )


@pytest.fixture
def sample_comm_event(vc_a0):
    return make_communication_event(
        sender="agent_0", receiver="agent_1", timestamp=0.15,
        message_payload={"cmd": "go"},
        vector_clock=vc_a0,
    )


@pytest.fixture
def sample_env_event():
    vc = vc_zero(AGENTS)
    return make_environment_event(
        timestamp=0.05,
        env_state_delta=np.array([0.1, -0.1]),
        affected_agents=["agent_0", "agent_1"],
        vector_clock=vc,
    )


@pytest.fixture
def sample_sync_event():
    vc = vc_zero(AGENTS)
    return make_sync_event(
        agent_id="agent_0", timestamp=0.3,
        barrier_id="b0", barrier_agents=AGENTS,
        vector_clock=vc,
    )


@pytest.fixture
def simple_trace(sample_action_event, sample_obs_event):
    trace = ExecutionTrace(trace_id="test_trace", agents=AGENTS)
    trace.append_event(sample_action_event)
    trace.append_event(sample_obs_event)
    return trace


@pytest.fixture
def simple_zonotope():
    return Zonotope(center=np.array([1.0, 2.0]),
                    generators=np.array([[1.0, 0.5], [0.0, 1.0]]))


@pytest.fixture
def unit_zonotope():
    return Zonotope.from_interval(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))


@pytest.fixture
def simple_hb_graph():
    g = HBGraph(name="test_hb")
    g.add_event("e0", agent_id="agent_0", timestep=0)
    g.add_event("e1", agent_id="agent_1", timestep=0)
    g.add_event("e2", agent_id="agent_0", timestep=1)
    g.add_hb_edge("e0", "e2", source="program_order")
    g.add_hb_edge("e1", "e2", source="communication")
    return g


@pytest.fixture
def simple_constraints():
    cs = HBConstraintSet()
    cs.add(HBConstraint(normal=np.array([1.0, 0.0]), bound=3.0, label="x<=3"))
    cs.add(HBConstraint(normal=np.array([0.0, 1.0]), bound=5.0, label="y<=5"))
    return cs
