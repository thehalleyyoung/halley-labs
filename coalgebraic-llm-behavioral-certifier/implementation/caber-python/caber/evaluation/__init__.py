"""
Evaluation module — harness, metrics, baselines, and drift simulation.

Provides:
    EvaluationHarness    — Full CABER evaluation orchestrator
    EvaluationMetrics    — Automaton fidelity, coverage, certificate metrics
    HELMBaseline         — Benchmark-style evaluation baseline
    CheckListBaseline    — Template-based testing baseline
    DirectStatisticalBaseline — Statistical hypothesis tests on raw responses
    HMMBaseline          — Hidden Markov Model on behavioral traces
    AALpyPRISMBaseline   — Classical automata learning + model checking
    DriftSimulator       — Simulated drift for consistency validation
"""

from caber.evaluation.harness import EvaluationHarness
from caber.evaluation.metrics import EvaluationMetrics
from caber.evaluation.baselines import (
    HELMBaseline,
    CheckListBaseline,
    DirectStatisticalBaseline,
    HMMBaseline,
    AALpyPRISMBaseline,
)
from caber.evaluation.drift_simulator import DriftSimulator

__all__ = [
    "EvaluationHarness",
    "EvaluationMetrics",
    "HELMBaseline",
    "CheckListBaseline",
    "DirectStatisticalBaseline",
    "HMMBaseline",
    "AALpyPRISMBaseline",
    "DriftSimulator",
]
