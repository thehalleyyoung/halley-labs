"""
Rényi Differential Privacy (RDP) accounting subpackage for DP-Forge.

Provides exact and approximate Rényi divergence computation, RDP-based
privacy accounting with optimal composition, privacy framework conversions
(RDP ↔ (ε,δ)-DP ↔ zCDP), budget allocation optimisation under RDP
composition constraints, and CEGIS integration with composition-aware
synthesis.

Modules:
    renyi_divergence         — Exact and approximate Rényi divergence computation.
    accountant               — RDP accountant with composition and conversion.
    conversion               — Privacy framework conversions (RDP, DP, zCDP).
    budget_optimizer         — Budget allocation optimisation under RDP.
    composition_aware_cegis  — CEGIS with RDP composition awareness.
    mechanisms               — RDP characterisation of standard mechanisms.

Example::

    from dp_forge.rdp import RDPAccountant, RDPCurve, rdp_to_dp

    acct = RDPAccountant()
    acct.add_mechanism("gaussian", sigma=1.0, sensitivity=1.0)
    acct.add_mechanism("gaussian", sigma=2.0, sensitivity=1.0)
    budget = acct.to_dp(delta=1e-5)
    print(f"Composed (ε, δ)-DP: ε={budget.epsilon:.4f}, δ={budget.delta}")
"""

from dp_forge.rdp.renyi_divergence import RenyiDivergenceComputer
from dp_forge.rdp.accountant import RDPAccountant, RDPCurve
from dp_forge.rdp.conversion import (
    rdp_to_dp,
    dp_to_rdp_bound,
    zcdp_to_rdp,
    rdp_to_zcdp,
)
from dp_forge.rdp.budget_optimizer import RDPBudgetOptimizer
from dp_forge.rdp.composition_aware_cegis import CompositionAwareCEGIS
from dp_forge.rdp.mechanisms import RDPMechanismCharacterizer

__all__ = [
    "RenyiDivergenceComputer",
    "RDPAccountant",
    "RDPCurve",
    "rdp_to_dp",
    "dp_to_rdp_bound",
    "zcdp_to_rdp",
    "rdp_to_zcdp",
    "RDPBudgetOptimizer",
    "CompositionAwareCEGIS",
    "RDPMechanismCharacterizer",
]
