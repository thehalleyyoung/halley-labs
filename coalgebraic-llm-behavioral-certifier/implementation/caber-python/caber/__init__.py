"""
CABER — Coalgebraic Behavioral Auditing of Foundation Models.

This package provides a Python evaluation harness for the CABER framework,
which uses coalgebraic automata theory to audit black-box LLM behavior.

Modules:
    interface   — LLM API clients, query generation, response parsing
    evaluation  — Evaluation harness, metrics, baselines, drift simulation
    classifiers — Refusal, safety, and behavioral atom classifiers
    visualization — Automaton and report visualization
"""

__version__ = "0.1.0"
__author__ = "CABER Team"

from caber.interface.model_client import ModelClient, MockClient
from caber.interface.query_generator import QueryGenerator
from caber.interface.response_parser import ResponseParser
from caber.evaluation.harness import EvaluationHarness
from caber.evaluation.metrics import EvaluationMetrics

__all__ = [
    "ModelClient",
    "MockClient",
    "QueryGenerator",
    "ResponseParser",
    "EvaluationHarness",
    "EvaluationMetrics",
    "__version__",
]
