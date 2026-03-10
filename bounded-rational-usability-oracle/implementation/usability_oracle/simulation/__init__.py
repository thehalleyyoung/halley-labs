"""
usability_oracle.simulation — Simulated user interaction.

Provides bounded-rational user agent simulation, UI environments,
interaction event models, recording/replay, metrics collection,
pre-built simulation scenarios, discrete-event cognitive simulation,
KLM/GOMS task analysis, ACT-R activation dynamics, and task networks.
"""

from __future__ import annotations

from usability_oracle.simulation.agent import SimulatedAgent, AgentConfig
from usability_oracle.simulation.environment import UIEnvironment
from usability_oracle.simulation.interaction import InteractionEvent, InteractionSequence
from usability_oracle.simulation.recorder import SimulationRecorder, Recording
from usability_oracle.simulation.metrics import SimulationMetrics
from usability_oracle.simulation.scenarios import ScenarioLibrary

from usability_oracle.simulation.event_queue import (
    EventQueue,
    EventType,
    EventPriority,
    SimulationEvent,
    ConditionalEvent,
    RecurringEvent,
    EventFilter,
)
from usability_oracle.simulation.processors import (
    CognitiveProcessor,
    PerceptualProcessor,
    MotorProcessor,
    CognitiveControlProcessor,
    WorkingMemoryProcessor,
    VisualAttentionProcessor,
    ProcessorState,
    ProcessorBuffer,
    MemoryChunk,
)
from usability_oracle.simulation.engine import (
    DiscreteEventSimulator,
    SimulationConfig,
    CognitiveProcessorConfig,
    SimulationState,
    SimulationTrace,
    ResourceBus,
    TrialResult,
    ExperimentResult,
)
from usability_oracle.simulation.klm import (
    KLMModel,
    KLMOperator,
    KLMTimings,
    KLMStep,
    apply_heuristic_rules,
)
from usability_oracle.simulation.goms import (
    GOMSModel,
    Goal,
    Operator,
    Method,
    SelectionRule,
    ResourceType,
)
from usability_oracle.simulation.activation import (
    ChunkActivation,
    base_level_learning,
    spreading_activation,
    partial_matching,
    noise,
    retrieval_probability,
    retrieval_time,
    decay_curve,
    fan_effect,
)
from usability_oracle.simulation.task_network import (
    TaskNetwork,
    TaskNode,
    TaskStatus,
    Resource,
)

__all__ = [
    # Original exports
    "SimulatedAgent",
    "AgentConfig",
    "UIEnvironment",
    "InteractionEvent",
    "InteractionSequence",
    "SimulationRecorder",
    "Recording",
    "SimulationMetrics",
    "ScenarioLibrary",
    # Event queue
    "EventQueue",
    "EventType",
    "EventPriority",
    "SimulationEvent",
    "ConditionalEvent",
    "RecurringEvent",
    "EventFilter",
    # Processors
    "CognitiveProcessor",
    "PerceptualProcessor",
    "MotorProcessor",
    "CognitiveControlProcessor",
    "WorkingMemoryProcessor",
    "VisualAttentionProcessor",
    "ProcessorState",
    "ProcessorBuffer",
    "MemoryChunk",
    # Engine
    "DiscreteEventSimulator",
    "SimulationConfig",
    "CognitiveProcessorConfig",
    "SimulationState",
    "SimulationTrace",
    "ResourceBus",
    "TrialResult",
    "ExperimentResult",
    # KLM
    "KLMModel",
    "KLMOperator",
    "KLMTimings",
    "KLMStep",
    "apply_heuristic_rules",
    # GOMS
    "GOMSModel",
    "Goal",
    "Operator",
    "Method",
    "SelectionRule",
    "ResourceType",
    # Activation
    "ChunkActivation",
    "base_level_learning",
    "spreading_activation",
    "partial_matching",
    "noise",
    "retrieval_probability",
    "retrieval_time",
    "decay_curve",
    "fan_effect",
    # Task network
    "TaskNetwork",
    "TaskNode",
    "TaskStatus",
    "Resource",
]
