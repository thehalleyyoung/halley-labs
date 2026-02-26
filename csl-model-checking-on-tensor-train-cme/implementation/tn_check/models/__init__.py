"""
Standard stochastic reaction network models for benchmarking.

Provides parameterized model builders for:
- Toggle switch (bistable, 2+ species)
- Repressilator (oscillatory, 3+ species)
- MAPK cascade (modular, multi-species)
- Birth-death (simple, 1 species)
- Schlögl model (bistable, 1 species)
"""

from tn_check.models.library import (
    birth_death,
    toggle_switch,
    repressilator,
    cascade,
    schlogl,
    gene_expression,
    exclusive_switch,
    sir_epidemic,
    michaelis_menten_enzyme,
    multi_species_cascade,
)

__all__ = [
    "birth_death",
    "toggle_switch",
    "repressilator",
    "cascade",
    "schlogl",
    "gene_expression",
    "exclusive_switch",
    "sir_epidemic",
    "michaelis_menten_enzyme",
    "multi_species_cascade",
]
