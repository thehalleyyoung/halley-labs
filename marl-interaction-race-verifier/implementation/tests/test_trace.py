"""Tests for marace.trace.trace — ExecutionTrace, MultiAgentTrace, TraceStatistics."""

import numpy as np
import pytest

from marace.trace.events import (
    Event, EventType, ActionEvent, ObservationEvent,
    make_action_event, make_observation_event,
    vc_zero, vc_increment, vc_merge,
)
from marace.trace.trace import (
    ExecutionTrace, MultiAgentTrace, TraceSegment,
    TraceStatistics, TraceValidator, TraceValidationResult,
)


AGENTS = ["agent_0", "agent_1"]


def _make_event(agent_id, timestamp, vc, eid=None):
    ev = Event(agent_id=agent_id, timestamp=timestamp,
               event_type=EventType.ACTION, vector_clock=vc)
    if eid:
        ev.event_id = eid
    return ev


# ── TraceSegment ──────────────────────────────────────────────────────

class TestTraceSegment:
    def test_len_and_iter(self):
        events = [_make_event("a0", 0.0, {"a0": 1}),
                  _make_event("a1", 0.1, {"a1": 1})]
        seg = TraceSegment(events=events, start_time=0.0, end_time=0.1)
        assert len(seg) == 2
        assert list(seg) == events

    def test_agent_ids(self):
        events = [_make_event("a0", 0.0, {}), _make_event("a1", 0.1, {})]
        seg = TraceSegment(events=events, start_time=0.0, end_time=0.1)
        assert seg.agent_ids == {"a0", "a1"}

    def test_filter_by_type(self):
        e1 = Event(agent_id="a0", timestamp=0.0, event_type=EventType.ACTION)
        e2 = Event(agent_id="a0", timestamp=0.1, event_type=EventType.OBSERVATION)
        seg = TraceSegment(events=[e1, e2], start_time=0.0, end_time=0.1)
        filtered = seg.filter_by_type(EventType.ACTION)
        assert len(filtered) == 1


# ── ExecutionTrace ────────────────────────────────────────────────────

class TestExecutionTrace:
    def test_empty_trace(self):
        t = ExecutionTrace(trace_id="empty")
        assert len(t) == 0
        assert t.agents == set()

    def test_append_and_length(self):
        t = ExecutionTrace()
        ev = _make_event("a0", 0.0, {"a0": 1}, eid="ev1")
        t.append_event(ev)
        assert len(t) == 1
        assert t[0] == ev

    def test_agents_set(self):
        t = ExecutionTrace(agents=AGENTS)
        ev = _make_event("a0", 0.0, {})
        t.append_event(ev)
        assert "a0" in t.agents
        assert "agent_0" in t.agents
        assert "agent_1" in t.agents

    def test_extend(self):
        t = ExecutionTrace()
        events = [_make_event("a0", i * 0.1, {"a0": i + 1}) for i in range(5)]
        t.extend(events)
        assert len(t) == 5

    def test_get_event_by_id(self):
        t = ExecutionTrace()
        ev = _make_event("a0", 0.0, {}, eid="test_id")
        t.append_event(ev)
        assert t.get_event_by_id("test_id") is ev
        assert t.get_event_by_id("nonexistent") is None

    def test_get_agent_events(self):
        t = ExecutionTrace()
        t.append_event(_make_event("a0", 0.0, {}))
        t.append_event(_make_event("a1", 0.1, {}))
        t.append_event(_make_event("a0", 0.2, {}))
        assert len(t.get_agent_events("a0")) == 2
        assert len(t.get_agent_events("a1")) == 1

    def test_filter_by_type(self):
        t = ExecutionTrace()
        t.append_event(Event(agent_id="a0", timestamp=0.0,
                             event_type=EventType.ACTION))
        t.append_event(Event(agent_id="a0", timestamp=0.1,
                             event_type=EventType.OBSERVATION))
        assert len(t.filter_by_type(EventType.ACTION)) == 1

    def test_get_events_in_window(self):
        t = ExecutionTrace(trace_id="tw")
        for i in range(10):
            t.append_event(_make_event("a0", i * 0.1, {}))
        seg = t.get_events_in_window(0.2, 0.5)
        assert isinstance(seg, TraceSegment)
        assert all(0.2 <= e.timestamp <= 0.5 for e in seg)

    def test_slice_by_time(self):
        t = ExecutionTrace(trace_id="sl")
        for i in range(10):
            t.append_event(_make_event("a0", i * 0.1, {}))
        sub = t.slice_by_time(0.3, 0.7)
        assert isinstance(sub, ExecutionTrace)
        assert all(0.3 <= e.timestamp <= 0.7 for e in sub)

    def test_get_concurrent_events(self):
        t = ExecutionTrace()
        t.append_event(_make_event("a0", 0.0, {"a0": 1, "a1": 0}))
        t.append_event(_make_event("a1", 0.1, {"a0": 0, "a1": 1}))
        pairs = t.get_concurrent_events()
        assert len(pairs) == 1

    def test_get_concurrent_events_for(self):
        t = ExecutionTrace()
        ev0 = _make_event("a0", 0.0, {"a0": 1, "a1": 0})
        ev1 = _make_event("a1", 0.1, {"a0": 0, "a1": 1})
        t.append_event(ev0)
        t.append_event(ev1)
        conc = t.get_concurrent_events_for(ev0)
        assert ev1 in conc

    def test_no_concurrent_ordered_events(self):
        t = ExecutionTrace()
        t.append_event(_make_event("a0", 0.0, {"a0": 1, "a1": 0}))
        t.append_event(_make_event("a0", 0.1, {"a0": 2, "a1": 0}))
        assert len(t.get_concurrent_events()) == 0

    def test_causal_predecessors_transitive(self):
        t = ExecutionTrace()
        e0 = _make_event("a0", 0.0, {}, eid="e0")
        e1 = _make_event("a0", 0.1, {}, eid="e1")
        e1.causal_predecessors = frozenset(["e0"])
        e2 = _make_event("a0", 0.2, {}, eid="e2")
        e2.causal_predecessors = frozenset(["e1"])
        t.extend([e0, e1, e2])
        preds = t.causal_predecessors_transitive("e2")
        assert "e0" in preds
        assert "e1" in preds

    def test_to_dict_list(self, simple_trace):
        dl = simple_trace.to_dict_list()
        assert isinstance(dl, list)
        assert len(dl) == len(simple_trace)

    def test_iter(self, simple_trace):
        count = sum(1 for _ in simple_trace)
        assert count == len(simple_trace)


# ── MultiAgentTrace ───────────────────────────────────────────────────

class TestMultiAgentTrace:
    def _make_agent_trace(self, agent_id, n_events):
        t = ExecutionTrace(trace_id=f"trace_{agent_id}")
        for i in range(n_events):
            vc = {agent_id: i + 1}
            t.append_event(_make_event(agent_id, i * 0.1, vc, eid=f"{agent_id}_{i}"))
        return t

    def test_add_and_agents(self):
        mt = MultiAgentTrace(trace_id="multi")
        mt.add_agent_trace("a0", self._make_agent_trace("a0", 3))
        mt.add_agent_trace("a1", self._make_agent_trace("a1", 2))
        assert mt.agent_ids == {"a0", "a1"}

    def test_merged_trace_length(self):
        mt = MultiAgentTrace(trace_id="multi")
        mt.add_agent_trace("a0", self._make_agent_trace("a0", 3))
        mt.add_agent_trace("a1", self._make_agent_trace("a1", 2))
        merged = mt.merged_trace()
        assert len(merged) == 5

    def test_merged_trace_cached(self):
        mt = MultiAgentTrace(trace_id="multi")
        mt.add_agent_trace("a0", self._make_agent_trace("a0", 2))
        m1 = mt.merged_trace()
        m2 = mt.merged_trace()
        assert m1 is m2  # cached

    def test_cache_invalidation(self):
        mt = MultiAgentTrace(trace_id="multi")
        mt.add_agent_trace("a0", self._make_agent_trace("a0", 2))
        m1 = mt.merged_trace()
        mt.add_agent_trace("a1", self._make_agent_trace("a1", 1))
        m2 = mt.merged_trace()
        assert m1 is not m2

    def test_merged_trace_hb_order(self):
        """Events with HB ordering should appear in correct order."""
        t0 = ExecutionTrace(trace_id="t0")
        t1 = ExecutionTrace(trace_id="t1")
        e0 = _make_event("a0", 0.0, {"a0": 1, "a1": 0}, eid="e0")
        e1 = _make_event("a1", 0.1, {"a0": 1, "a1": 1}, eid="e1")
        e1.causal_predecessors = frozenset(["e0"])
        t0.append_event(e0)
        t1.append_event(e1)
        mt = MultiAgentTrace(trace_id="ordered")
        mt.add_agent_trace("a0", t0)
        mt.add_agent_trace("a1", t1)
        merged = mt.merged_trace()
        ids = [e.event_id for e in merged]
        assert ids.index("e0") < ids.index("e1")

    def test_get_agent_trace(self):
        mt = MultiAgentTrace()
        t = self._make_agent_trace("a0", 2)
        mt.add_agent_trace("a0", t)
        assert mt.get_agent_trace("a0") is t

    def test_interaction_pairs(self):
        t0 = ExecutionTrace(trace_id="t0")
        t1 = ExecutionTrace(trace_id="t1")
        e0 = _make_event("a0", 0.0, {"a0": 1, "a1": 0}, eid="e0")
        e1 = _make_event("a1", 0.1, {"a0": 1, "a1": 1}, eid="e1")
        e1.causal_predecessors = frozenset(["e0"])
        t0.append_event(e0)
        t1.append_event(e1)
        mt = MultiAgentTrace()
        mt.add_agent_trace("a0", t0)
        mt.add_agent_trace("a1", t1)
        pairs = mt.interaction_pairs()
        assert len(pairs) >= 1
        assert any(a == "a0" and b == "a1" for a, b, _ in pairs)


# ── TraceStatistics ───────────────────────────────────────────────────

class TestTraceStatistics:
    def test_empty_trace(self):
        t = ExecutionTrace()
        stats = TraceStatistics.compute(t)
        assert stats.total_events == 0

    def test_basic_stats(self):
        t = ExecutionTrace()
        for i in range(5):
            t.append_event(_make_event("a0", i * 0.1, {"a0": i + 1}))
        for i in range(3):
            t.append_event(_make_event("a1", i * 0.1 + 0.05, {"a1": i + 1}))
        stats = TraceStatistics.compute(t)
        assert stats.total_events == 8
        assert stats.events_per_agent["a0"] == 5
        assert stats.events_per_agent["a1"] == 3
        assert stats.trace_duration > 0
        assert stats.event_density > 0

    def test_concurrency_degree(self):
        t = ExecutionTrace()
        # All concurrent events
        t.append_event(_make_event("a0", 0.0, {"a0": 1, "a1": 0}))
        t.append_event(_make_event("a1", 0.1, {"a0": 0, "a1": 1}))
        stats = TraceStatistics.compute(t)
        assert stats.concurrency_degree == 1.0

    def test_max_vector_clock_skew(self):
        t = ExecutionTrace()
        t.append_event(_make_event("a0", 0.0, {"a0": 10, "a1": 1}))
        stats = TraceStatistics.compute(t)
        assert stats.max_vector_clock_skew == 9


# ── TraceValidator ────────────────────────────────────────────────────

class TestTraceValidator:
    def test_valid_trace(self, simple_trace):
        result = TraceValidator.validate(simple_trace)
        assert isinstance(result, TraceValidationResult)
        assert result.is_valid

    def test_duplicate_ids(self):
        t = ExecutionTrace()
        e1 = _make_event("a0", 0.0, {"a0": 1}, eid="dup")
        e2 = _make_event("a0", 0.1, {"a0": 2}, eid="dup")
        t.append_event(e1)
        t.append_event(e2)
        result = TraceValidator.validate(t)
        assert not result.is_valid

    def test_vc_inconsistency(self):
        t = ExecutionTrace()
        e0 = _make_event("a0", 0.0, {"a0": 2}, eid="e0")
        e1 = _make_event("a0", 0.1, {"a0": 1}, eid="e1")
        e1.causal_predecessors = frozenset(["e0"])
        t.extend([e0, e1])
        result = TraceValidator.validate(t)
        assert not result.is_valid

    def test_per_agent_monotonicity(self):
        t = ExecutionTrace()
        t.append_event(_make_event("a0", 0.0, {"a0": 3}, eid="x1"))
        t.append_event(_make_event("a0", 0.1, {"a0": 1}, eid="x2"))
        result = TraceValidator.validate(t)
        assert not result.is_valid

    def test_validation_result_repr(self):
        r = TraceValidationResult()
        assert "valid=True" in repr(r)
        r.errors.append("err")
        assert "valid=False" in repr(r)
