"""
usability_oracle.simulation — Simulated user interaction.

Provides bounded-rational user agent simulation, UI environments,
interaction event models, recording/replay, metrics collection,
and pre-built simulation scenarios.
"""

from __future__ import annotations

from usability_oracle.simulation.agent import SimulatedAgent, AgentConfig
from usability_oracle.simulation.environment import UIEnvironment
from usability_oracle.simulation.interaction import InteractionEvent, InteractionSequence
from usability_oracle.simulation.recorder import SimulationRecorder, Recording
from usability_oracle.simulation.metrics import SimulationMetrics
from usability_oracle.simulation.scenarios import ScenarioLibrary

__all__ = [
    "SimulatedAgent",
    "AgentConfig",
    "UIEnvironment",
    "InteractionEvent",
    "InteractionSequence",
    "SimulationRecorder",
    "Recording",
    "SimulationMetrics",
    "ScenarioLibrary",
]
