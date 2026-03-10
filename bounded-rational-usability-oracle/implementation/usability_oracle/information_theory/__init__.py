"""usability_oracle.information_theory — Deep information-theoretic primitives."""

from usability_oracle.information_theory.types import (
    Channel,
    ChannelCapacity,
    InformationState,
    MutualInfoResult,
    RateDistortionPoint,
)
from usability_oracle.information_theory.protocols import (
    ChannelComputer,
    InformationMeasure,
    RateDistortionSolver,
)

# ── algorithmic modules ──────────────────────────────────────────────────
from usability_oracle.information_theory import (
    entropy,
    mutual_information,
    channel_capacity,
    rate_distortion,
    free_energy,
    bounds,
    estimators,
)

__all__ = [
    # types
    "Channel",
    "ChannelCapacity",
    "InformationState",
    "MutualInfoResult",
    "RateDistortionPoint",
    # protocols
    "ChannelComputer",
    "InformationMeasure",
    "RateDistortionSolver",
    # algorithmic modules
    "entropy",
    "mutual_information",
    "channel_capacity",
    "rate_distortion",
    "free_energy",
    "bounds",
    "estimators",
]
