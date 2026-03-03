"""
Composition engine for DP-Forge.

This package provides advanced privacy composition accounting via multiple
formalisms: Privacy Loss Distributions (PLD), Fourier accountant (Koskela
et al. 2020), optimal composition theorems (KOV 2015), privacy filters,
and mixed accounting across (ε,δ), RDP, zCDP, and f-DP frameworks.

Modules:
    pld                    — Privacy Loss Distribution accounting with FFT convolution
    fourier_accountant     — Characteristic function FFT accountant
    advanced_composition   — Optimal composition theorems (KOV, heterogeneous, group)
    privacy_filter         — Privacy filters and odometers for budget tracking
    mixed_accounting       — Unified accounting across multiple privacy definitions

Key Classes:
    PrivacyLossDistribution — PLD representation and composition
    FourierAccountant       — FFT-based accountant
    PrivacyFilter           — Halt when cumulative privacy exceeds budget
    PrivacyOdometer         — Track running privacy expenditure
    MixedAccountant         — Unified accounting across privacy definitions

Example::

    from dp_forge.composition import PrivacyLossDistribution, PrivacyFilter
    
    # Create PLD from mechanism
    pld = PrivacyLossDistribution.from_mechanism(mechanism, adjacent_pair)
    
    # Compose multiple mechanisms
    composed_pld = pld.compose(other_pld)
    
    # Convert to (ε, δ)
    epsilon = composed_pld.to_epsilon_delta(delta=1e-5)
    
    # Track budget with filter
    filter = PrivacyFilter(epsilon_budget=1.0, delta_budget=1e-5)
    filter.add_mechanism(pld)
    if not filter.budget_available():
        print("Budget exhausted")
"""

from dp_forge.composition.pld import (
    PrivacyLossDistribution,
    compose,
    worst_case_pld,
    discretize,
)
from dp_forge.composition.fourier_accountant import (
    FourierAccountant,
    characteristic_function,
)
from dp_forge.composition.advanced_composition import (
    optimal_advanced_composition,
    heterogeneous_basic_composition,
    group_privacy_composition,
    parallel_composition,
    sequential_composition_optimal_order,
    mixed_composition,
)
from dp_forge.composition.privacy_filter import (
    PrivacyFilter,
    PrivacyOdometer,
    AdaptiveFilter,
    FilteredComposition,
)
from dp_forge.composition.mixed_accounting import (
    MixedAccountant,
    convert_rdp_to_pld,
    convert_zcdp_to_pld,
    convert_fdp_to_pld,
)

__all__ = [
    "PrivacyLossDistribution",
    "compose",
    "worst_case_pld",
    "discretize",
    "FourierAccountant",
    "characteristic_function",
    "optimal_advanced_composition",
    "heterogeneous_basic_composition",
    "group_privacy_composition",
    "parallel_composition",
    "sequential_composition_optimal_order",
    "mixed_composition",
    "PrivacyFilter",
    "PrivacyOdometer",
    "AdaptiveFilter",
    "FilteredComposition",
    "MixedAccountant",
    "convert_rdp_to_pld",
    "convert_zcdp_to_pld",
    "convert_fdp_to_pld",
]
