"""Tests for environment adapters."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.env.base import (
    MultiAgentEnv,
    AgentTimingConfig,
    EnvironmentClock,
    EnvironmentState,
    StaleObservationModel,
    AsyncSteppingSemantics,
)
from marace.env.highway import HighwayEnv
from marace.env.warehouse import WarehouseEnv
from marace.env.timing import (
    DistributionType,
    FixedLatencyModel,
    HardwareClass,
    HardwareProfile,
    LatencyDistribution,
    LatencyScheduler,
    StochasticLatencyModel,
    TimingJitter,
    TimingAnalyzer,
)


class TestEnvironmentClock:
    """Test environment clock."""

    def test_clock_creation(self):
        """Test creating a clock."""
        clock = EnvironmentClock(dt=0.1)
        assert clock.current_time == 0.0

    def test_clock_tick(self):
        """Test clock advancing."""
        clock = EnvironmentClock(dt=0.1)
        clock.advance()
        assert np.isclose(clock.current_time, 0.1)
        clock.advance()
        assert np.isclose(clock.current_time, 0.2)

    def test_clock_reset(self):
        """Test clock reset."""
        clock = EnvironmentClock(dt=0.1)
        clock.advance()
        clock.advance()
        clock.reset()
        assert clock.current_time == 0.0


class TestAgentTimingConfig:
    """Test agent timing configuration."""

    def test_default_config(self):
        """Test default timing config."""
        config = AgentTimingConfig(agent_id="agent_0")
        assert config.perception_latency >= 0
        assert config.compute_latency >= 0
        assert config.actuation_latency >= 0

    def test_custom_config(self):
        """Test custom timing config."""
        config = AgentTimingConfig(
            agent_id="agent_0",
            perception_latency=0.02,
            compute_latency=0.05,
            actuation_latency=0.01,
        )
        assert config.total_latency == pytest.approx(0.08)

    def test_different_profiles(self):
        """Test different hardware profiles have different latencies."""
        fast = AgentTimingConfig(
            agent_id="fast",
            perception_latency=0.01,
            compute_latency=0.02,
            actuation_latency=0.005,
        )
        slow = AgentTimingConfig(
            agent_id="slow",
            perception_latency=0.05,
            compute_latency=0.1,
            actuation_latency=0.02,
        )
        assert fast.total_latency < slow.total_latency


class TestHighwayEnv:
    """Test highway driving environment."""

    def test_creation(self):
        """Test creating highway environment."""
        env = HighwayEnv(
            num_agents=4,
            dt=0.1,
        )
        assert env.num_agents == 4

    def test_reset(self):
        """Test environment reset."""
        env = HighwayEnv(num_agents=3, dt=0.1)
        obs = env.reset()
        assert len(obs) == 3
        for agent_id, ob in obs.items():
            assert isinstance(ob, np.ndarray)

    def test_step_sync(self):
        """Test synchronous stepping."""
        env = HighwayEnv(num_agents=2, dt=0.1)
        env.reset()
        actions = {}
        for aid in env.get_agent_ids():
            actions[aid] = np.array([0.0, 0.0])  # No acceleration
        obs, rewards, done, info = env.step_sync(actions)
        assert len(obs) == 2

    def test_step_async(self):
        """Test asynchronous stepping."""
        env = HighwayEnv(num_agents=2, dt=0.1)
        env.reset()
        agent_ids = env.get_agent_ids()
        obs, reward, done, info = env.step_async(agent_ids[0], np.array([1.0, 0.0]))
        assert isinstance(obs, np.ndarray)

    def test_get_state(self):
        """Test getting environment state."""
        env = HighwayEnv(num_agents=2, dt=0.1)
        env.reset()
        state = env.get_state()
        assert state is not None

    def test_state_save_restore(self):
        """Test saving and restoring state."""
        env = HighwayEnv(num_agents=2, dt=0.1)
        env.reset()
        state1 = env.get_state()
        # Step forward
        for aid in env.get_agent_ids():
            env.step_async(aid, np.array([1.0, 0.0]))
        state2 = env.get_state()
        # Restore
        env.set_state(state1)
        state_restored = env.get_state()
        # States should match
        assert state_restored.fingerprint() == state1.fingerprint()

    def test_multiple_steps(self):
        """Test multiple steps."""
        env = HighwayEnv(num_agents=3, dt=0.1)
        env.reset()
        for _ in range(10):
            actions = {aid: np.array([0.5, 0.0]) for aid in env.get_agent_ids()}
            obs, rewards, done, info = env.step_sync(actions)
            assert len(obs) == 3


class TestWarehouseEnv:
    """Test warehouse environment."""

    def test_creation(self):
        """Test creating warehouse environment."""
        env = WarehouseEnv(
            num_robots=4,
            dt=0.1,
        )
        assert env.num_agents == 4

    def test_reset(self):
        """Test warehouse reset."""
        env = WarehouseEnv(num_robots=4, dt=0.1)
        obs = env.reset()
        assert len(obs) == 4

    def test_step(self):
        """Test warehouse stepping."""
        env = WarehouseEnv(num_robots=4, dt=0.1)
        env.reset()
        actions = {aid: np.array([1.0, 0.0]) for aid in env.get_agent_ids()}
        obs, rewards, done, info = env.step_sync(actions)
        assert len(obs) == 4

    def test_agent_ids(self):
        """Test agent IDs."""
        env = WarehouseEnv(num_robots=6, dt=0.1)
        env.reset()
        ids = env.get_agent_ids()
        assert len(ids) == 6
        assert len(set(ids)) == 6  # All unique


class TestTimingModels:
    """Test timing models."""

    def test_fixed_latency(self):
        """Test fixed latency model."""
        model = FixedLatencyModel(default_latency=0.05)
        assert model.sample("agent_0", 0) == 0.05
        assert model.sample("agent_0", 1) == 0.05

    def test_stochastic_latency(self):
        """Test stochastic latency model."""
        dist = LatencyDistribution(
            dist_type=DistributionType.NORMAL,
            mean_latency=0.05,
            std_latency=0.01,
        )
        model = StochasticLatencyModel(
            default_dist=dist,
            seed=42,
        )
        latencies = [model.sample("agent_0", i) for i in range(100)]
        assert all(lat >= 0.0 for lat in latencies)
        assert np.std(latencies) > 0

    def test_hardware_profiles(self):
        """Test hardware profile presets."""
        fast = HardwareProfile(HardwareClass.GPU_ACCELERATED)
        slow = HardwareProfile(HardwareClass.EMBEDDED_LOW)
        assert fast.total_latency < slow.total_latency

    def test_latency_scheduler(self):
        """Test latency-based scheduling."""
        model = FixedLatencyModel(
            latencies={"agent_0": 0.035, "agent_1": 0.17},
        )
        scheduler = LatencyScheduler(model)
        sa0 = scheduler.submit("agent_0", None, current_time=0.0)
        sa1 = scheduler.submit("agent_1", None, current_time=0.0)
        ready = scheduler.pop_ready(current_time=0.2)
        assert len(ready) == 2
        # Faster agent should execute first
        assert ready[0].agent_id == "agent_0"

    def test_timing_jitter(self):
        """Test timing jitter."""
        base_model = FixedLatencyModel(default_latency=1.0)
        jitter = TimingJitter(base_model=base_model, jitter_scale=0.01, seed=42)
        jittered = [jitter.sample("agent_0", i) for i in range(100)]
        assert all(t >= 0.0 for t in jittered)
        assert np.std(jittered) > 0


class TestStaleObservationModel:
    """Test stale observation model."""

    def test_no_staleness(self):
        """Test with zero latency (no staleness)."""
        model = StaleObservationModel()
        state = EnvironmentState()
        model.record(1.0, state)
        result = model.get_observation_state(current_time=1.0, perception_latency=0.0)
        assert result is not None

    def test_with_staleness(self):
        """Test with positive latency."""
        model = StaleObservationModel()
        # Record states at different times
        s0 = EnvironmentState()
        s0.data = np.array([0.0, 0.0])
        s1 = EnvironmentState()
        s1.data = np.array([1.0, 1.0])
        s2 = EnvironmentState()
        s2.data = np.array([2.0, 2.0])
        model.record(0.0, s0)
        model.record(0.1, s1)
        model.record(0.2, s2)
        # With 0.1s perception latency at time 0.2, should get state from t=0.1
        result = model.get_observation_state(current_time=0.2, perception_latency=0.1)
        assert result is not None
        np.testing.assert_allclose(result.data, np.array([1.0, 1.0]))


class TestTimingAnalyzer:
    """Test timing analysis."""

    def test_analyze_latencies(self):
        """Test latency analysis from trace data."""
        analyzer = TimingAnalyzer()
        # Record latency samples
        for lat in [0.11, 0.11, 0.09]:
            analyzer.record("agent_0", lat)
        for lat in [0.10, 0.13, 0.10]:
            analyzer.record("agent_1", lat)
        result = analyzer.all_stats()
        assert "agent_0" in result
        assert result["agent_0"].mean_latency > 0

    def test_detect_async(self):
        """Test detecting ordering violations via scheduler."""
        model = FixedLatencyModel(
            latencies={"agent_0": 0.2, "agent_1": 0.05},
        )
        scheduler = LatencyScheduler(model)
        # agent_0 submits first but has higher latency
        scheduler.submit("agent_0", "act_a", current_time=0.0)
        scheduler.submit("agent_1", "act_b", current_time=0.01)
        scheduler.pop_ready(current_time=1.0)
        analyzer = TimingAnalyzer()
        violations = analyzer.ordering_violations(scheduler)
        assert len(violations) > 0  # Submission order differs from execution order
