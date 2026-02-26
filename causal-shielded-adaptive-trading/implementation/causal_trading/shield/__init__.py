"""
Shield synthesis module for Causal-Shielded Adaptive Trading.

Provides posterior-predictive shield synthesis with PAC-Bayes soundness
guarantees, temporal logic safety specifications, liveness theorems,
and permissivity tracking for safe reinforcement learning in trading.
"""

from .shield_synthesis import (
    PosteriorPredictiveShield,
    ShieldResult,
    ShieldQuery,
    ComposedShield,
)
from .safety_specs import (
    SafetySpecification,
    BoundedDrawdownSpec,
    PositionLimitSpec,
    MarginSpec,
    MaxLossSpec,
    TurnoverSpec,
    CompositeSpec,
    LTLFormula,
    TrajectoryChecker,
)
from .pac_bayes import (
    PACBayesBound,
    McAllesterBound,
    MaurerBound,
    CatoniBound,
    SequentialPACBayes,
    EmpiricalBernsteinPACBayes,
    ShieldSoundnessCertificate,
)
from .liveness import (
    ShieldLiveness,
    LivenessCertificate,
    PermissivityAlert,
    DegradationMonitor,
)
from .permissivity import (
    PermissivityTracker,
    PermissivityDecomposition,
    PermissivityForecaster,
    PermissivityReport,
)
from .bounded_liveness_specs import (
    DrawdownRecoverySpec,
    LossRecoverySpec,
    PositionReductionSpec,
    RegimeTransitionSpec,
    BoundedLivenessLibrary,
)

__all__ = [
    # Shield synthesis
    "PosteriorPredictiveShield",
    "ShieldResult",
    "ShieldQuery",
    "ComposedShield",
    # Safety specifications
    "SafetySpecification",
    "BoundedDrawdownSpec",
    "PositionLimitSpec",
    "MarginSpec",
    "MaxLossSpec",
    "TurnoverSpec",
    "CompositeSpec",
    "LTLFormula",
    "TrajectoryChecker",
    # PAC-Bayes
    "PACBayesBound",
    "McAllesterBound",
    "MaurerBound",
    "CatoniBound",
    "SequentialPACBayes",
    "EmpiricalBernsteinPACBayes",
    "ShieldSoundnessCertificate",
    # Liveness
    "ShieldLiveness",
    "LivenessCertificate",
    "PermissivityAlert",
    "DegradationMonitor",
    # Permissivity
    "PermissivityTracker",
    "PermissivityDecomposition",
    "PermissivityForecaster",
    "PermissivityReport",
    # Bounded liveness
    "DrawdownRecoverySpec",
    "LossRecoverySpec",
    "PositionReductionSpec",
    "RegimeTransitionSpec",
    "BoundedLivenessLibrary",
]
