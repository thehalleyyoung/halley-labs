"""
usability_oracle.policy — Bounded-rational policy computation.

Implements the free-energy formulation for bounded-rational decision-making:

    F(π) = E_π[cost] + (1/β) · D_KL(π ‖ p₀)

where β is the *rationality parameter* (inverse temperature), π is the
agent's policy, and p₀ is a prior (default) policy.

Modules
-------
- **models** — Policy, QValues, PolicyResult data structures
- **softmax** — Boltzmann / softmax policy construction
- **free_energy** — Free-energy computation and decomposition
- **value_iteration** — Soft (entropy-regularised) value iteration
- **monte_carlo** — Monte Carlo estimation of value and free energy
- **optimal** — Optimal and bounded-rational policy computation

Re-exports
----------
>>> from usability_oracle.policy import Policy, SoftmaxPolicy, FreeEnergyComputer
"""

from __future__ import annotations

from usability_oracle.policy.models import Policy, QValues, PolicyResult
from usability_oracle.policy.softmax import SoftmaxPolicy
from usability_oracle.policy.free_energy import (
    FreeEnergyComputer,
    FreeEnergyDecomposition,
)
from usability_oracle.policy.value_iteration import SoftValueIteration
from usability_oracle.policy.monte_carlo import MonteCarloEstimator
from usability_oracle.policy.optimal import OptimalPolicyComputer

__all__ = [
    "Policy",
    "QValues",
    "PolicyResult",
    "SoftmaxPolicy",
    "FreeEnergyComputer",
    "FreeEnergyDecomposition",
    "SoftValueIteration",
    "MonteCarloEstimator",
    "OptimalPolicyComputer",
]
