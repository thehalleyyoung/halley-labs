"""
Privacy amplification by subsampling for DP-Forge.

This subpackage implements privacy amplification theorems and a subsampled
CEGIS synthesis engine.  When a differentially private mechanism is applied
to a random subsample of the dataset (rather than the full dataset), the
effective privacy guarantee is tighter than the base mechanism's guarantee.

Subpackage Structure:
    - :mod:`amplification` — Core amplification formulas (Poisson,
      without-replacement, shuffle model, RDP).
    - :mod:`budget_inversion` — Numerical inversion of amplification
      curves to find base privacy from target privacy.
    - :mod:`shuffle_amplification` — Detailed shuffle model analysis
      with blanket decomposition and privacy profiles.
    - :mod:`protocol` — Subsampling protocol execution (mask generation,
      mechanism application, error estimation).
    - :mod:`subsampled_cegis` — SubsampledCEGIS engine that integrates
      budget inversion with the core CEGIS synthesiser.

Quick Start::

    from dp_forge.subsampling import poisson_amplify, SubsampledCEGIS

    # Direct amplification computation
    result = poisson_amplify(base_eps=1.0, base_delta=1e-5, q_rate=0.01)
    print(result)  # ε ≈ 0.01, δ = 1e-7

    # Subsampled mechanism synthesis
    from dp_forge.types import QuerySpec
    spec = QuerySpec.counting(n=5, epsilon=1.0)
    engine = SubsampledCEGIS()
    mechanism = engine.synthesize(spec, q_rate=0.01)
    print(mechanism.amplified.eps)
"""

from dp_forge.subsampling.amplification import (
    AmplificationBound,
    AmplificationResult,
    compare_amplification_bounds,
    compute_amplification_factor,
    poisson_amplify,
    poisson_amplify_rdp,
    replacement_amplify,
    shuffle_amplify,
)
from dp_forge.subsampling.budget_inversion import (
    BudgetInverter,
    InversionResult,
    invert_poisson,
    invert_replacement,
)
from dp_forge.subsampling.protocol import (
    ExecutionResult,
    SubsamplingMode,
    SubsamplingProtocol,
)
from dp_forge.subsampling.shuffle_amplification import (
    PrivacyProfilePoint,
    ShuffleAmplifier,
    ShuffleComparison,
)
from dp_forge.subsampling.subsampled_cegis import (
    SubsampledCEGIS,
    SubsampledMechanism,
    synthesize_subsampled,
)

__all__ = [
    # amplification
    "AmplificationBound",
    "AmplificationResult",
    "compare_amplification_bounds",
    "compute_amplification_factor",
    "poisson_amplify",
    "poisson_amplify_rdp",
    "replacement_amplify",
    "shuffle_amplify",
    # budget_inversion
    "BudgetInverter",
    "InversionResult",
    "invert_poisson",
    "invert_replacement",
    # protocol
    "ExecutionResult",
    "SubsamplingMode",
    "SubsamplingProtocol",
    # shuffle_amplification
    "PrivacyProfilePoint",
    "ShuffleAmplifier",
    "ShuffleComparison",
    # subsampled_cegis
    "SubsampledCEGIS",
    "SubsampledMechanism",
    "synthesize_subsampled",
]
