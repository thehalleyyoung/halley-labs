"""
DP-Forge: Counterexample-Guided Synthesis of Provably Optimal DP Mechanisms.

This package implements a CEGIS-based engine that automatically discovers
provably-optimal differentially private noise mechanisms for arbitrary query
types. It encodes mechanism design as LP/SDP optimization with formal
verification to produce deployable mechanisms that dominate Laplace/Gaussian
baselines on accuracy at equivalent privacy guarantees.

Architecture Overview:
    - ``types``: Core type system (QuerySpec, LPStruct, SDPStruct, etc.)
    - ``exceptions``: Structured exception hierarchy
    - ``config``: Configuration management with solver auto-detection
    - ``logging_config``: Structured logging with rich output

Downstream modules (not yet implemented):
    - ``lp_builder``: LP construction for discrete mechanism synthesis
    - ``sdp_builder``: SDP construction for Gaussian workload mechanisms
    - ``verifier``: (ε,δ)-DP verification with counterexample generation
    - ``cegis_loop``: Main CEGIS orchestrator
    - ``extractor``: Post-processing LP solutions into deployable mechanisms
"""

__version__ = "0.1.0"
__author__ = "DP-Forge Team"

from dp_forge.types import (
    AdjacencyRelation,
    BenchmarkResult,
    CEGISResult,
    ExtractedMechanism,
    LossFunction,
    LPStruct,
    MechanismFamily,
    NumericalConfig,
    OptimalityCertificate,
    PrivacyBudget,
    QuerySpec,
    QueryType,
    SamplingConfig,
    SDPStruct,
    SynthesisConfig,
    VerifyResult,
    WorkloadSpec,
)
from dp_forge.exceptions import (
    BudgetExhaustedError,
    ConfigurationError,
    ConvergenceError,
    CycleDetectedError,
    DPForgeError,
    InfeasibleSpecError,
    InvalidMechanismError,
    NumericalInstabilityError,
    SensitivityError,
    SolverError,
    VerificationError,
)
from dp_forge.config import DPForgeConfig, get_config

__all__ = [
    # Version
    "__version__",
    # Types
    "AdjacencyRelation",
    "BenchmarkResult",
    "CEGISResult",
    "ExtractedMechanism",
    "LossFunction",
    "LPStruct",
    "MechanismFamily",
    "NumericalConfig",
    "OptimalityCertificate",
    "PrivacyBudget",
    "QuerySpec",
    "QueryType",
    "SamplingConfig",
    "SDPStruct",
    "SynthesisConfig",
    "VerifyResult",
    "WorkloadSpec",
    # Exceptions
    "BudgetExhaustedError",
    "ConfigurationError",
    "ConvergenceError",
    "CycleDetectedError",
    "DPForgeError",
    "InfeasibleSpecError",
    "InvalidMechanismError",
    "NumericalInstabilityError",
    "SensitivityError",
    "SolverError",
    "VerificationError",
    # Config
    "DPForgeConfig",
    "get_config",
]
