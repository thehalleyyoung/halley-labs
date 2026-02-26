"""
Species ordering strategies for entanglement minimization.

The ordering of species along the TT chain affects bond dimensions:
weakly correlated species placed far apart reduce entanglement at
intermediate bonds.
"""

from tn_check.ordering.strategies import (
    identity_ordering,
    reverse_cuthill_mckee,
    spectral_ordering,
    greedy_entanglement_ordering,
)

__all__ = [
    "identity_ordering",
    "reverse_cuthill_mckee",
    "spectral_ordering",
    "greedy_entanglement_ordering",
]
