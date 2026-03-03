"""Curiosity-Driven Quality-Diversity exploration for the CPA engine.

Provides the QD-MAP-Elites search loop that systematically explores the
space of causal mechanism configurations across heterogeneous contexts,
producing a structured archive of diverse plasticity patterns.

Modules
-------
qd_search
    Main search engine implementing ALG3 (Curiosity-Driven QD Search).
genome
    Genome representation and genetic operators.
cvt
    Centroidal Voronoi Tessellation for archive cell management.
curiosity
    Curiosity signal computation (novelty + surprise).
"""

from cpa.exploration.genome import QDGenome, BehaviorDescriptor
from cpa.exploration.cvt import CVTTessellation, AdaptiveCVT
from cpa.exploration.curiosity import CuriosityComputer
from cpa.exploration.qd_search import QDSearchEngine, QDArchive

__all__ = [
    "QDSearchEngine",
    "QDArchive",
    "QDGenome",
    "BehaviorDescriptor",
    "CVTTessellation",
    "AdaptiveCVT",
    "CuriosityComputer",
]
