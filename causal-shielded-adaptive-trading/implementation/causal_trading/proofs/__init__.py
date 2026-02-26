"""
Proofs module for Causal-Shielded Adaptive Trading.

Provides PAC-Bayes bound computation, shield soundness verification,
composition theorem checking, and formal certificate generation.
"""

from .pac_bayes_bound import PACBayesBoundComputer
from .shield_soundness import ShieldSoundnessVerifier
from .composition_checker import CompositionChecker
from .certificate import Certificate
from .decomposed_composition import (
    PipelineErrorBudget,
    DecomposedCompositionTheorem,
    DecomposedCertificate,
)

__all__ = [
    "PACBayesBoundComputer",
    "ShieldSoundnessVerifier",
    "CompositionChecker",
    "Certificate",
    "PipelineErrorBudget",
    "DecomposedCompositionTheorem",
    "DecomposedCertificate",
]
