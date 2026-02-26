"""
Verification module for Causal-Shielded Adaptive Trading.

Provides polytope-based credible set computation, symbolic model checking,
temporal logic satisfaction checking, PTIME verification for fixed
regime count K, sound interval arithmetic, conservative state abstraction,
and independent certificate verification.
"""

from .polytope import CredibleSetPolytope
from .model_checking import SymbolicModelChecker
from .temporal_logic import BoundedLTL, LTLFormula, BuchiAutomaton
from .ptime_verification import PTIMEVerifier
from .state_abstraction import ConservativeOverapproximation, AbstractState, AbstractionFunction
from .interval_arithmetic import Interval, IntervalVector, IntervalMatrix
from .independent_verifier import IndependentVerifier, VerificationReport

__all__ = [
    "CredibleSetPolytope",
    "SymbolicModelChecker",
    "BoundedLTL",
    "LTLFormula",
    "BuchiAutomaton",
    "PTIMEVerifier",
    "ConservativeOverapproximation",
    "AbstractState",
    "AbstractionFunction",
    "Interval",
    "IntervalVector",
    "IntervalMatrix",
    "IndependentVerifier",
    "VerificationReport",
]
