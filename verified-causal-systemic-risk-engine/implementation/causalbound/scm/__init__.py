"""
SCM (Structural Causal Model) construction module for CausalBound.

Implements SCM construction from partially-observed data with causal discovery,
orientation rules, Markov equivalence class enumeration, and efficient DAG
representations for downstream causal inference tasks.
"""

from .dag import DAGRepresentation
from .builder import SCMBuilder
from .causal_discovery import FastCausalInference
from .orientation import OrientationRules
from .equivalence import MarkovEquivalenceClass
from .partial_observability import PartialObservabilityHandler
from .sensitivity import DAGSensitivityAnalyzer

__all__ = [
    "DAGRepresentation",
    "SCMBuilder",
    "FastCausalInference",
    "OrientationRules",
    "MarkovEquivalenceClass",
    "PartialObservabilityHandler",
    "DAGSensitivityAnalyzer",
]
