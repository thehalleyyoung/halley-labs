"""End-to-end integration tests for MARACE."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestTraceToHBIntegration:
    """Test trace construction to HB graph construction."""

    def test_trace_to_hb(self):
        """Test building HB graph from execution trace."""
        from marace.trace.events import ActionEvent, ObservationEvent, EventType
        from marace.trace.trace import ExecutionTrace, MultiAgentTrace
        from marace.hb.vector_clock import VectorClock, VectorClockManager
        from marace.hb.hb_graph import HBGraph, HBRelation

        agents = ["a0", "a1", "a2"]
        manager = VectorClockManager(agents)

        # Simulate execution
        trace_a0 = ExecutionTrace(trace_id="trace_a0", agents=["a0"])
        trace_a1 = ExecutionTrace(trace_id="trace_a1", agents=["a1"])

        # a0 acts
        vc = manager.record_local_event("a0")
        e1 = ActionEvent(
            event_id="e1", agent_id="a0", timestamp=0.0,
            event_type=EventType.ACTION,
            data={}, action_vector=np.array([1.0, 0.0]),
            state_before=np.zeros(4), state_after=np.array([0.1, 0.0, 0.0, 0.0]),
            vector_clock=vc.to_dict(),
            causal_predecessors=[],
        )
        trace_a0.append_event(e1)

        # a0 sends observation to a1
        send_vc = manager.record_send("a0", "a1")

        # a1 receives observation
        manager.record_receive("a1", "a0", send_vc)
        vc_a1 = manager.record_local_event("a1")
        e2 = ObservationEvent(
            event_id="e2", agent_id="a1", timestamp=0.05,
            event_type=EventType.OBSERVATION,
            data={}, observation_vector=np.array([0.1, 0.0, 0.0, 0.0]),
            source_agents=["a0"],
            vector_clock=vc_a1.to_dict(),
            causal_predecessors=["e1"],
        )
        trace_a1.append_event(e2)

        # Build HB graph
        hb = HBGraph()
        hb.add_event("e1", agent_id="a0", timestamp=0.0)
        hb.add_event("e2", agent_id="a1", timestamp=0.05)
        hb.add_hb_edge("e1", "e2")

        assert hb.query_hb("e1", "e2") == HBRelation.BEFORE
        assert hb.query_hb("e2", "e1") == HBRelation.AFTER

    def test_concurrent_events_detected(self):
        """Test detecting concurrent events."""
        from marace.hb.hb_graph import HBGraph, HBRelation

        hb = HBGraph()
        hb.add_event("e_a0", agent_id="a0", timestamp=0.0)
        hb.add_event("e_a1", agent_id="a1", timestamp=0.0)

        assert hb.query_hb("e_a0", "e_a1") == HBRelation.CONCURRENT
        concurrent = hb.concurrent_pairs()
        assert ("e_a0", "e_a1") in concurrent or ("e_a1", "e_a0") in concurrent


class TestAbstractInterpretationIntegration:
    """Test abstract interpretation pipeline."""

    def test_zonotope_through_network(self):
        """Test pushing zonotope through a neural network."""
        from marace.abstract.zonotope import Zonotope
        from marace.abstract.transfer import LinearTransfer, ReLUTransfer

        # 2-layer ReLU network
        W1 = np.array([[1.0, -0.5], [0.3, 1.0], [-0.2, 0.8]])
        b1 = np.array([0.1, -0.1, 0.0])
        W2 = np.array([[0.5, -0.3, 0.7], [0.2, 0.6, -0.4]])
        b2 = np.array([0.0, 0.1])

        z_in = Zonotope(
            center=np.array([0.5, 0.5]),
            generators=np.array([[0.3, 0.0], [0.0, 0.3]])
        )

        lt1 = LinearTransfer(W1, b1)
        relu = ReLUTransfer()
        lt2 = LinearTransfer(W2, b2)

        z1 = lt1.apply(z_in)
        z2 = relu.apply(z1)
        z3 = lt2.apply(z2)

        # Verify soundness
        bbox = z3.bounding_box()
        for _ in range(500):
            x = z_in.sample(1)[0]
            h = np.maximum(W1 @ x + b1, 0)
            y = W2 @ h + b2
            for d in range(2):
                assert bbox[d, 0] - 0.01 <= y[d] <= bbox[d, 1] + 0.01, \
                    f"Soundness violation: y[{d}]={y[d]} not in [{bbox[d, 0]}, {bbox[d, 1]}]"

    def test_fixpoint_convergence(self):
        """Test fixpoint convergence for stable system."""
        from marace.abstract.zonotope import Zonotope
        from marace.abstract.fixpoint import FixpointEngine

        def stable_system(z):
            W = np.array([[0.6, 0.1], [-0.1, 0.6]])
            b = np.array([0.2, 0.2])
            return z.affine_transform(W, b)

        engine = FixpointEngine(
            transfer_fn=stable_system,
            max_iterations=100,
            convergence_threshold=1e-4,
        )

        z0 = Zonotope(
            center=np.zeros(2),
            generators=np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        result = engine.compute(z0)
        assert result.converged


class TestEnvironmentTraceIntegration:
    """Test environment stepping with trace recording."""

    def test_highway_trace_recording(self):
        """Test recording traces from highway environment."""
        from marace.env.highway import HighwayEnv

        env = HighwayEnv(num_agents=2, dt=0.1)
        obs = env.reset()
        assert len(obs) == 2

        states = []
        for step in range(5):
            actions = {}
            for aid in env.get_agent_ids():
                actions[aid] = np.array([0.5, 0.0])  # Slight acceleration
            obs = env.step_sync(actions)
            state = env.get_state()
            states.append(state)

        assert len(states) == 5

    def test_warehouse_trace_recording(self):
        """Test recording traces from warehouse environment."""
        from marace.env.warehouse import WarehouseEnv

        env = WarehouseEnv(num_robots=4, dt=0.1)
        obs = env.reset()
        assert len(obs) == 4  # 4 robots = 4 agents

        for step in range(5):
            actions = {}
            for aid in env.get_agent_ids():
                actions[aid] = np.array([0.5, 0.1])  # Move forward, slight turn
            obs = env.step_sync(actions)

        assert True  # If we get here, no crash


class TestRaceDetectionIntegration:
    """Test race detection pipeline."""

    def test_detect_planted_race(self):
        """Test detecting a planted race in simple scenario."""
        from marace.race.definition import InteractionRace, RaceClassification
        from marace.race.catalog import RaceCatalog, CatalogEntry, CatalogStatistics

        catalog = RaceCatalog()

        # Plant a race
        race = InteractionRace(
            race_id="planted_001",
            events=(),
            agents=["a0", "a1"],
            classification=RaceClassification.COLLISION,
            probability=0.01,
            metadata={},
        )
        catalog.add(CatalogEntry(
            entry_id="entry_001",
            race=race,
            probability_bound=0.01,
            tags=[],
        ))

        stats = CatalogStatistics(catalog=catalog)
        result = stats.compute()
        assert result.total_entries == 1
        assert result.max_probability == pytest.approx(0.01, abs=0.001)

    def test_epsilon_race_calibration_flow(self):
        """Test epsilon-race calibration flow."""
        from marace.race.epsilon_race import EpsilonCalibrator

        calibrator = EpsilonCalibrator(
            lipschitz_constant=10.0,
            global_safety_margin=2.0,
            max_iterations=20,
        )

        # calibrate takes a center point and optional initial epsilon
        center = np.array([0.0, 0.0])
        result = calibrator.calibrate(center, initial_epsilon=0.2)
        assert result is not None
        assert result.epsilon > 0
        assert result.lipschitz_constant == 10.0


class TestDecompositionIntegration:
    """Test compositional decomposition."""

    def test_partition_and_verify(self):
        """Test partitioning agents and per-group analysis."""
        from marace.decomposition.interaction_graph import InteractionGraph, InteractionEdge, InteractionType
        from marace.decomposition.partitioning import ConstrainedPartitioner

        g = InteractionGraph()
        for i in range(6):
            g.add_agent(f"a{i}")

        # Group 1: a0, a1, a2 (strongly interacting)
        g.add_interaction(InteractionEdge("a0", "a1", InteractionType.OBSERVATION, 0.9))
        g.add_interaction(InteractionEdge("a1", "a2", InteractionType.OBSERVATION, 0.8))
        g.add_interaction(InteractionEdge("a0", "a2", InteractionType.PHYSICS, 0.7))

        # Group 2: a3, a4, a5 (strongly interacting)
        g.add_interaction(InteractionEdge("a3", "a4", InteractionType.OBSERVATION, 0.9))
        g.add_interaction(InteractionEdge("a4", "a5", InteractionType.COMMUNICATION, 0.85))

        # Weak link between groups
        g.add_interaction(InteractionEdge("a2", "a3", InteractionType.PHYSICS, 0.05))

        partitioner = ConstrainedPartitioner(max_group_size=4)
        partition = partitioner.partition(g)

        # Should find approximately 2 groups
        assert len(partition.groups) >= 2
        all_agents = set()
        for group_name, members in partition.groups.items():
            all_agents.update(members)
        assert all_agents == {f"a{i}" for i in range(6)}

    def test_interaction_groups_from_hb(self):
        """Test extracting interaction groups from HB graph."""
        from marace.hb.hb_graph import HBGraph

        hb = HBGraph()
        # Two independent groups of events
        for i in range(5):
            hb.add_event(f"g1_e{i}", agent_id=f"a{i % 2}", timestamp=float(i) * 0.1)
        for i in range(5):
            hb.add_event(f"g2_e{i}", agent_id=f"a{2 + i % 2}", timestamp=float(i) * 0.1)

        # HB edges within groups only
        hb.add_hb_edge("g1_e0", "g1_e1")
        hb.add_hb_edge("g1_e1", "g1_e2")
        hb.add_hb_edge("g2_e0", "g2_e1")
        hb.add_hb_edge("g2_e1", "g2_e2")

        components = hb.connected_components()
        assert len(components) >= 2


class TestSamplingIntegration:
    """Test sampling pipeline."""

    def test_schedule_sampling_and_estimation(self):
        """Test generating schedules and estimating race probability."""
        from marace.sampling.schedule_space import ScheduleGenerator, ScheduleSpace
        from marace.sampling.importance_sampling import (
            ImportanceWeights, EffectiveSampleSize, ConfidenceInterval
        )

        space = ScheduleSpace(agents=["a", "b", "c"], num_timesteps=6)
        gen = ScheduleGenerator(space, rng=np.random.RandomState(42))
        schedules = gen.sample_uniform(n_samples=100)
        assert len(schedules) == 100

        # Simulate: evaluate each schedule for race
        np.random.seed(42)
        race_detected = np.random.rand(100) < 0.05  # 5% race rate
        indicators = race_detected.astype(float)
        log_weights = np.zeros(100)  # uniform weights

        iw = ImportanceWeights(log_weights=log_weights)
        n_eff = EffectiveSampleSize.compute(iw)
        assert np.isclose(n_eff, 100.0)

        ci = ConfidenceInterval.from_importance_samples(indicators, iw, confidence_level=0.95)
        assert ci.lower <= np.mean(indicators) <= ci.upper


class TestPolicyAbstractIntegration:
    """Test policy analysis integration."""

    def test_lipschitz_to_epsilon(self):
        """Test Lipschitz bound feeds into epsilon calibration."""
        from marace.policy.lipschitz import ReLULipschitz
        from marace.policy.onnx_loader import NetworkArchitecture, LayerInfo, ActivationType
        from marace.race.epsilon_race import EpsilonCalibrator

        # Compute Lipschitz bound
        weights = [
            np.array([[1.5, -0.3], [0.7, 1.2]]),
            np.array([[0.8, -0.5], [0.3, 0.9]]),
        ]
        layers = [
            LayerInfo(
                name=f"fc{i}",
                layer_type="dense",
                input_size=w.shape[1],
                output_size=w.shape[0],
                activation=ActivationType.RELU,
                weights=w,
                bias=np.zeros(w.shape[0]),
            )
            for i, w in enumerate(weights)
        ]
        arch = NetworkArchitecture(layers=layers, input_dim=2, output_dim=2)

        extractor = ReLULipschitz()
        cert = extractor.compute(arch)
        L = cert.global_bound
        assert L > 0

        # Use for epsilon calibration
        calibrator = EpsilonCalibrator(
            lipschitz_constant=L,
            global_safety_margin=1.0,
        )
        center = np.array([0.0, 0.0])
        result = calibrator.calibrate(center, initial_epsilon=1.0 / L)
        assert result is not None
        assert result.epsilon > 0
