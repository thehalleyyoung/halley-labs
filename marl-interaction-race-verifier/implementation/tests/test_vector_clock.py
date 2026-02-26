"""Tests for vector clock implementation."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.hb.vector_clock import VectorClock, VectorClockManager


class TestVectorClock:
    """Test VectorClock basic operations."""

    def test_creation_default(self):
        """Test creating a vector clock with default values."""
        vc = VectorClock()
        assert vc.get("a") == 0
        assert vc.get("b") == 0
        assert vc.get("c") == 0

    def test_increment(self):
        """Test incrementing a clock component."""
        vc = VectorClock()
        vc.increment("a")
        assert vc.get("a") == 1
        assert vc.get("b") == 0
        vc.increment("a")
        assert vc.get("a") == 2

    def test_merge(self):
        """Test merging two vector clocks (component-wise max)."""
        vc1 = VectorClock()
        vc2 = VectorClock()
        vc1.increment("a")
        vc1.increment("a")
        vc1.increment("b")
        vc2.increment("b")
        vc2.increment("b")
        vc2.increment("c")
        merged = vc1.merge(vc2)
        assert merged.get("a") == 2
        assert merged.get("b") == 2
        assert merged.get("c") == 1

    def test_happens_before_true(self):
        """Test happens-before relation when it holds."""
        vc1 = VectorClock()
        vc2 = VectorClock()
        vc1.increment("a")
        vc2.increment("a")
        vc2.increment("a")
        vc2.increment("b")
        assert vc1.happens_before(vc2)
        assert not vc2.happens_before(vc1)

    def test_happens_before_false(self):
        """Test happens-before when clocks are concurrent."""
        vc1 = VectorClock()
        vc2 = VectorClock()
        vc1.increment("a")
        vc2.increment("b")
        assert not vc1.happens_before(vc2)
        assert not vc2.happens_before(vc1)

    def test_concurrent_with(self):
        """Test concurrent_with relation."""
        vc1 = VectorClock()
        vc2 = VectorClock()
        vc1.increment("a")
        vc2.increment("b")
        assert vc1.concurrent_with(vc2)
        assert vc2.concurrent_with(vc1)

    def test_concurrent_not_with_ordered(self):
        """Test that ordered events are not concurrent."""
        vc1 = VectorClock()
        vc2 = VectorClock()
        vc1.increment("a")
        vc2.increment("a")
        vc2.increment("a")
        vc2.increment("b")
        assert not vc1.concurrent_with(vc2)

    def test_equal_clocks(self):
        """Test equality of identical clocks."""
        vc1 = VectorClock()
        vc2 = VectorClock()
        vc1.increment("a")
        vc2.increment("a")
        assert not vc1.happens_before(vc2)
        assert not vc2.happens_before(vc1)

    def test_copy(self):
        """Test that copy creates independent clock."""
        vc1 = VectorClock()
        vc1.increment("a")
        vc2 = vc1.copy()
        vc2.increment("b")
        assert vc1.get("b") == 0
        assert vc2.get("b") == 1

    def test_serialization(self):
        """Test serialization round-trip."""
        vc = VectorClock()
        vc.increment("a")
        vc.increment("b")
        vc.increment("b")
        data = vc.to_dict()
        vc2 = VectorClock.from_dict(data)
        assert vc2.get("a") == 1
        assert vc2.get("b") == 2
        assert vc2.get("c") == 0


class TestVectorClockManager:
    """Test VectorClockManager for multi-agent scenarios."""

    def test_creation(self):
        """Test manager creation."""
        manager = VectorClockManager(["a", "b", "c"])
        assert len(manager.agent_ids) == 3

    def test_local_event(self):
        """Test recording a local event."""
        manager = VectorClockManager(["a", "b"])
        vc = manager.record_local_event("a")
        assert vc.get("a") == 1
        assert vc.get("b") == 0

    def test_send_receive(self):
        """Test send/receive event pattern."""
        manager = VectorClockManager(["a", "b"])
        manager.record_local_event("a")
        send_vc = manager.record_send("a", "b")
        manager.record_receive("b", "a", send_vc)
        vc_b = manager.get_clock("b")
        assert vc_b.get("a") == 2
        assert vc_b.get("b") == 1

    def test_multiple_agents_ordering(self):
        """Test ordering across multiple agents."""
        agents = ["a", "b", "c"]
        manager = VectorClockManager(agents)
        manager.record_local_event("a")
        send_ab = manager.record_send("a", "b")
        manager.record_receive("b", "a", send_ab)
        manager.record_local_event("b")
        send_bc = manager.record_send("b", "c")
        manager.record_receive("c", "b", send_bc)
        vc_a = manager.get_clock("a")
        vc_c = manager.get_clock("c")
        assert vc_a.happens_before(vc_c)

    def test_concurrent_agents(self):
        """Test concurrent activity detection."""
        manager = VectorClockManager(["a", "b"])
        manager.record_local_event("a")
        manager.record_local_event("b")
        vc_a = manager.get_clock("a")
        vc_b = manager.get_clock("b")
        assert vc_a.concurrent_with(vc_b)

    def test_batch_comparison(self):
        """Test batch comparison of clocks."""
        manager = VectorClockManager(["a", "b", "c"])
        manager.record_local_event("a")
        manager.record_local_event("b")
        send = manager.record_send("a", "c")
        manager.record_receive("c", "a", send)
        clocks = [manager.get_clock(aid) for aid in ["a", "b", "c"]]
        assert clocks[0].happens_before(clocks[2])
        assert clocks[1].concurrent_with(clocks[0])


class TestVectorClockEdgeCases:
    """Test edge cases for vector clocks."""

    def test_single_agent(self):
        """Test with single agent."""
        vc = VectorClock()
        vc.increment("a")
        vc.increment("a")
        assert vc.get("a") == 2

    def test_many_agents(self):
        """Test with many agents."""
        agents = [f"agent_{i}" for i in range(50)]
        vc = VectorClock()
        for a in agents:
            vc.increment(a)
        for a in agents:
            assert vc.get(a) == 1

    def test_merge_preserves_max(self):
        """Test merge takes component-wise max."""
        vc1 = VectorClock()
        vc2 = VectorClock()
        for _ in range(5):
            vc1.increment("a")
        for _ in range(3):
            vc2.increment("a")
        for _ in range(7):
            vc2.increment("b")
        merged = vc1.merge(vc2)
        assert merged.get("a") == 5
        assert merged.get("b") == 7
        assert merged.get("c") == 0

    def test_transitivity(self):
        """Test happens-before transitivity."""
        vc1 = VectorClock()
        vc2 = VectorClock()
        vc3 = VectorClock()
        vc1.increment("a")
        vc2.increment("a")
        vc2.increment("a")
        vc2.increment("b")
        vc3.increment("a")
        vc3.increment("a")
        vc3.increment("a")
        vc3.increment("b")
        vc3.increment("b")
        vc3.increment("c")
        assert vc1.happens_before(vc2)
        assert vc2.happens_before(vc3)
        assert vc1.happens_before(vc3)
