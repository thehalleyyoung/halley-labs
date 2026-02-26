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
)

__all__ = [
    "birth_death",
    "toggle_switch",
    "repressilator",
    "cascade",
    "schlogl",
]
