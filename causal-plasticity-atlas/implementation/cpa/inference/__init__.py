"""CPA inference subpackage.

Causal inference engine providing do-calculus, counterfactual reasoning,
interventional query processing, and identifiability analysis.

Modules
-------
do_calculus
    Do-calculus engine implementing the three rules.
counterfactual
    Counterfactual reasoning via twin networks.
interventional
    Interventional query processing.
identifiability
    Causal effect identifiability checks.
"""

from __future__ import annotations

from cpa.inference.do_calculus import DoCalculusEngine, DoCalculusResult
from cpa.inference.counterfactual import (
    CounterfactualEngine,
    TwinNetwork,
    CounterfactualResult,
)
from cpa.inference.interventional import (
    InterventionalQuery,
    InterventionalEstimator,
)
from cpa.inference.identifiability import (
    IdentifiabilityChecker,
    IdentifiabilityResult,
)

__all__ = [
    # do_calculus.py
    "DoCalculusEngine",
    "DoCalculusResult",
    # counterfactual.py
    "CounterfactualEngine",
    "TwinNetwork",
    "CounterfactualResult",
    # interventional.py
    "InterventionalQuery",
    "InterventionalEstimator",
    # identifiability.py
    "IdentifiabilityChecker",
    "IdentifiabilityResult",
]
