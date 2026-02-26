"""Tests for marace.trace.events — event creation, validation, serialization."""

import numpy as np
import pytest

from marace.trace.events import (
    Event, EventType, ActionEvent, ObservationEvent,
    CommunicationEvent, EnvironmentEvent, SyncEvent,
    VectorClock,
    vc_zero, vc_increment, vc_merge, vc_leq, vc_strictly_less, vc_concurrent,
    make_action_event, make_observation_event,
    make_communication_event, make_environment_event, make_sync_event,
    event_from_dict, validate_event, EventValidationError,
)


# ── Vector clock helpers (dict-based, from events module) ─────────────

class TestVCDictHelpers:
    def test_vc_zero(self, agent_ids):
        vc = vc_zero(agent_ids)
        assert all(vc[a] == 0 for a in agent_ids)

    def test_vc_increment(self, agent_ids):
        vc = vc_zero(agent_ids)
        vc2 = vc_increment(vc, "agent_0")
        assert vc2["agent_0"] == 1
        assert vc2["agent_1"] == 0
        # original unchanged (returns new dict)
        assert vc["agent_0"] == 0

    def test_vc_merge(self):
        a = {"x": 2, "y": 1}
        b = {"x": 1, "y": 3}
        m = vc_merge(a, b)
        assert m == {"x": 2, "y": 3}

    def test_vc_leq(self):
        a = {"x": 1, "y": 2}
        b = {"x": 2, "y": 3}
        assert vc_leq(a, b) is True
        assert vc_leq(b, a) is False
        assert vc_leq(a, a) is True

    def test_vc_strictly_less(self):
        a = {"x": 1, "y": 2}
        b = {"x": 2, "y": 3}
        assert vc_strictly_less(a, b) is True
        assert vc_strictly_less(a, a) is False

    def test_vc_concurrent(self):
        a = {"x": 2, "y": 1}
        b = {"x": 1, "y": 3}
        assert vc_concurrent(a, b) is True
        assert vc_concurrent(a, a) is False

    def test_vc_merge_disjoint_keys(self):
        a = {"x": 5}
        b = {"y": 3}
        m = vc_merge(a, b)
        assert m == {"x": 5, "y": 3}

    def test_vc_leq_missing_keys(self):
        a = {"x": 1}
        b = {"x": 1, "y": 0}
        assert vc_leq(a, b) is True


# ── EventType enum ────────────────────────────────────────────────────

class TestEventType:
    def test_members(self):
        assert EventType.ACTION.name == "ACTION"
        assert len(EventType) == 5

    def test_by_name(self):
        assert EventType["SYNC"] is EventType.SYNC


# ── Base Event ────────────────────────────────────────────────────────

class TestEvent:
    def test_create_base_event(self):
        ev = Event(agent_id="a0", timestamp=1.0, event_type=EventType.ACTION)
        assert ev.agent_id == "a0"
        assert ev.timestamp == 1.0
        assert ev.event_id  # auto-generated

    def test_happens_before(self):
        ev1 = Event(agent_id="a0", timestamp=0.1, event_type=EventType.ACTION,
                    vector_clock={"a0": 1, "a1": 0})
        ev2 = Event(agent_id="a1", timestamp=0.2, event_type=EventType.ACTION,
                    vector_clock={"a0": 1, "a1": 1})
        assert ev1.happens_before(ev2) is True
        assert ev2.happens_before(ev1) is False

    def test_is_concurrent_with(self):
        ev1 = Event(agent_id="a0", timestamp=0.1, event_type=EventType.ACTION,
                    vector_clock={"a0": 1, "a1": 0})
        ev2 = Event(agent_id="a1", timestamp=0.2, event_type=EventType.OBSERVATION,
                    vector_clock={"a0": 0, "a1": 1})
        assert ev1.is_concurrent_with(ev2) is True

    def test_serialization_roundtrip(self):
        ev = Event(agent_id="a0", timestamp=1.5, event_type=EventType.SYNC,
                   data={"key": "val"}, vector_clock={"a0": 2})
        d = ev.to_dict()
        ev2 = Event.from_dict(d)
        assert ev2.agent_id == ev.agent_id
        assert ev2.timestamp == ev.timestamp
        assert ev2.event_type == ev.event_type
        assert ev2.data == ev.data

    def test_hash_and_eq(self):
        ev1 = Event(agent_id="a0", timestamp=0.0, event_type=EventType.ACTION,
                    event_id="abc")
        ev2 = Event(agent_id="a1", timestamp=1.0, event_type=EventType.SYNC,
                    event_id="abc")
        assert ev1 == ev2
        assert hash(ev1) == hash(ev2)


# ── ActionEvent ───────────────────────────────────────────────────────

class TestActionEvent:
    def test_auto_type(self):
        ev = ActionEvent(agent_id="a0", timestamp=0.0,
                         event_type=EventType.OBSERVATION)
        assert ev.event_type is EventType.ACTION

    def test_action_magnitude(self):
        ev = ActionEvent(agent_id="a0", timestamp=0.0,
                         event_type=EventType.ACTION,
                         action_vector=np.array([3.0, 4.0]))
        assert np.isclose(ev.action_magnitude(), 5.0)

    def test_action_magnitude_none(self):
        ev = ActionEvent(agent_id="a0", timestamp=0.0,
                         event_type=EventType.ACTION)
        assert ev.action_magnitude() == 0.0

    def test_state_delta_norm(self):
        ev = ActionEvent(agent_id="a0", timestamp=0.0,
                         event_type=EventType.ACTION,
                         state_before=np.array([0.0, 0.0]),
                         state_after=np.array([3.0, 4.0]))
        assert np.isclose(ev.state_delta_norm(), 5.0)

    def test_serialization_roundtrip(self):
        ev = ActionEvent(agent_id="a0", timestamp=0.5,
                         event_type=EventType.ACTION,
                         action_vector=np.array([1.0, 2.0]),
                         vector_clock={"a0": 1})
        d = ev.to_dict()
        ev2 = ActionEvent.from_dict(d)
        assert np.allclose(ev2.action_vector, ev.action_vector)
        assert ev2.agent_id == ev.agent_id


# ── ObservationEvent ──────────────────────────────────────────────────

class TestObservationEvent:
    def test_auto_type(self):
        ev = ObservationEvent(agent_id="a0", timestamp=0.0,
                              event_type=EventType.ACTION)
        assert ev.event_type is EventType.OBSERVATION

    def test_observation_norm(self):
        ev = ObservationEvent(agent_id="a0", timestamp=0.0,
                              event_type=EventType.OBSERVATION,
                              observation_vector=np.array([3.0, 4.0]))
        assert np.isclose(ev.observation_norm(), 5.0)

    def test_is_multi_source(self, sample_obs_event):
        assert sample_obs_event.is_multi_source() is False  # only one source

    def test_multi_source_true(self):
        ev = ObservationEvent(agent_id="a0", timestamp=0.0,
                              event_type=EventType.OBSERVATION,
                              source_agents=["a1", "a2"])
        assert ev.is_multi_source() is True

    def test_serialization_roundtrip(self):
        ev = ObservationEvent(agent_id="a0", timestamp=0.0,
                              event_type=EventType.OBSERVATION,
                              observation_vector=np.array([1.0, 2.0, 3.0]),
                              source_agents=["a1"])
        d = ev.to_dict()
        ev2 = ObservationEvent.from_dict(d)
        assert np.allclose(ev2.observation_vector, ev.observation_vector)
        assert ev2.source_agents == ["a1"]


# ── CommunicationEvent ───────────────────────────────────────────────

class TestCommunicationEvent:
    def test_auto_type_and_sender(self):
        ev = CommunicationEvent(agent_id="a0", timestamp=0.0,
                                event_type=EventType.ACTION)
        assert ev.event_type is EventType.COMMUNICATION
        assert ev.sender == "a0"

    def test_message_size(self, sample_comm_event):
        assert sample_comm_event.message_size() > 0

    def test_serialization_roundtrip(self, sample_comm_event):
        d = sample_comm_event.to_dict()
        ev2 = CommunicationEvent.from_dict(d)
        assert ev2.sender == sample_comm_event.sender
        assert ev2.receiver == sample_comm_event.receiver
        assert ev2.channel_id == "default"


# ── EnvironmentEvent ──────────────────────────────────────────────────

class TestEnvironmentEvent:
    def test_auto_agent_id(self):
        ev = EnvironmentEvent(agent_id="", timestamp=0.0,
                              event_type=EventType.ACTION)
        assert ev.agent_id == "__env__"

    def test_delta_magnitude(self, sample_env_event):
        assert sample_env_event.delta_magnitude() > 0

    def test_serialization_roundtrip(self, sample_env_event):
        d = sample_env_event.to_dict()
        ev2 = EnvironmentEvent.from_dict(d)
        assert np.allclose(ev2.env_state_delta, sample_env_event.env_state_delta)


# ── SyncEvent ─────────────────────────────────────────────────────────

class TestSyncEvent:
    def test_auto_type(self):
        ev = SyncEvent(agent_id="a0", timestamp=0.0,
                       event_type=EventType.ACTION)
        assert ev.event_type is EventType.SYNC

    def test_all_arrived(self, agent_ids):
        clocks = {a: vc_increment(vc_zero(agent_ids), a) for a in agent_ids}
        ev = SyncEvent(agent_id="a0", timestamp=0.0,
                       event_type=EventType.SYNC,
                       barrier_id="b0", barrier_agents=agent_ids,
                       arrived_clocks=clocks)
        assert ev.all_arrived() is True

    def test_not_all_arrived(self, agent_ids):
        ev = SyncEvent(agent_id="a0", timestamp=0.0,
                       event_type=EventType.SYNC,
                       barrier_id="b0", barrier_agents=agent_ids,
                       arrived_clocks={})
        assert ev.all_arrived() is False

    def test_merged_clock(self, agent_ids):
        clocks = {
            "agent_0": {"agent_0": 2, "agent_1": 0},
            "agent_1": {"agent_0": 0, "agent_1": 3},
        }
        ev = SyncEvent(agent_id="a0", timestamp=0.0,
                       event_type=EventType.SYNC,
                       barrier_id="b0", barrier_agents=["agent_0", "agent_1"],
                       arrived_clocks=clocks)
        mc = ev.merged_clock()
        assert mc["agent_0"] == 2
        assert mc["agent_1"] == 3


# ── Factory functions ─────────────────────────────────────────────────

class TestFactories:
    def test_make_action_event(self, vc_a0):
        ev = make_action_event("agent_0", 0.1, np.array([1.0]), vc_a0)
        assert isinstance(ev, ActionEvent)
        assert ev.event_type is EventType.ACTION

    def test_make_observation_event(self, vc_a1):
        ev = make_observation_event("agent_1", 0.2, np.array([0.5]), vc_a1)
        assert isinstance(ev, ObservationEvent)

    def test_make_communication_event(self, vc_a0):
        ev = make_communication_event("agent_0", "agent_1", 0.15,
                                      {"msg": "hi"}, vc_a0)
        assert isinstance(ev, CommunicationEvent)
        assert ev.sender == "agent_0"
        assert ev.receiver == "agent_1"

    def test_make_environment_event(self):
        vc = vc_zero(["a0"])
        ev = make_environment_event(0.05, np.array([0.1]), ["a0"], vc)
        assert isinstance(ev, EnvironmentEvent)
        assert ev.agent_id == "__env__"

    def test_make_sync_event(self, agent_ids):
        vc = vc_zero(agent_ids)
        ev = make_sync_event("agent_0", 0.3, "b0", agent_ids, vc)
        assert isinstance(ev, SyncEvent)


# ── event_from_dict polymorphic deserialization ───────────────────────

class TestEventFromDict:
    def test_action_roundtrip(self, sample_action_event):
        d = sample_action_event.to_dict()
        ev = event_from_dict(d)
        assert isinstance(ev, ActionEvent)

    def test_obs_roundtrip(self, sample_obs_event):
        d = sample_obs_event.to_dict()
        ev = event_from_dict(d)
        assert isinstance(ev, ObservationEvent)

    def test_comm_roundtrip(self, sample_comm_event):
        d = sample_comm_event.to_dict()
        ev = event_from_dict(d)
        assert isinstance(ev, CommunicationEvent)

    def test_env_roundtrip(self, sample_env_event):
        d = sample_env_event.to_dict()
        ev = event_from_dict(d)
        assert isinstance(ev, EnvironmentEvent)

    def test_sync_roundtrip(self, sample_sync_event):
        d = sample_sync_event.to_dict()
        ev = event_from_dict(d)
        assert isinstance(ev, SyncEvent)


# ── Validation ────────────────────────────────────────────────────────

class TestValidation:
    def test_valid_event(self, sample_action_event):
        warnings = validate_event(sample_action_event)
        assert isinstance(warnings, list)

    def test_empty_id_raises(self):
        ev = Event(agent_id="a0", timestamp=0.0, event_type=EventType.ACTION,
                   event_id="")
        with pytest.raises(EventValidationError):
            validate_event(ev)

    def test_empty_agent_id_raises(self):
        ev = Event(agent_id="", timestamp=0.0, event_type=EventType.ACTION)
        with pytest.raises(EventValidationError):
            validate_event(ev)

    def test_negative_timestamp_raises(self):
        ev = Event(agent_id="a0", timestamp=-1.0, event_type=EventType.ACTION)
        with pytest.raises(EventValidationError):
            validate_event(ev)

    def test_empty_vector_clock_warning(self):
        ev = Event(agent_id="a0", timestamp=0.0, event_type=EventType.ACTION,
                   vector_clock={})
        w = validate_event(ev)
        assert any("empty vector clock" in s for s in w)

    def test_unknown_agent_warning(self):
        ev = Event(agent_id="unknown", timestamp=0.0,
                   event_type=EventType.ACTION,
                   vector_clock={"unknown": 1})
        w = validate_event(ev, known_agents={"a0", "a1"})
        assert any("not in known set" in s for s in w)

    def test_nonfinite_action_warning(self):
        ev = ActionEvent(agent_id="a0", timestamp=0.0,
                         event_type=EventType.ACTION,
                         action_vector=np.array([float("inf")]),
                         vector_clock={"a0": 1})
        w = validate_event(ev)
        assert any("non-finite" in s for s in w)

    def test_shape_mismatch_raises(self):
        ev = ActionEvent(agent_id="a0", timestamp=0.0,
                         event_type=EventType.ACTION,
                         state_before=np.array([0.0]),
                         state_after=np.array([0.0, 1.0]),
                         vector_clock={"a0": 1})
        with pytest.raises(EventValidationError):
            validate_event(ev)

    def test_self_send_warning(self):
        ev = CommunicationEvent(agent_id="a0", timestamp=0.0,
                                event_type=EventType.COMMUNICATION,
                                sender="a0", receiver="a0",
                                vector_clock={"a0": 1})
        w = validate_event(ev)
        assert any("sender == receiver" in s for s in w)
