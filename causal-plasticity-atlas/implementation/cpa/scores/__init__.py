"""CPA scoring functions subpackage.

Scoring functions for causal structure learning including BIC, BGe,
interventional BIC, BDeu, and decomposable score interfaces.

Modules
-------
bic
    BIC score variants.
bge
    BGe (Bayesian Gaussian equivalent) score.
interventional_bic
    Interventional BIC score for multi-context learning.
bdeu
    BDeu score for discrete data.
decomposable
    Decomposable score interface with caching.
"""

from cpa.scores.bic import BICScore, ExtendedBICScore
from cpa.scores.bge import BGeScore
from cpa.scores.interventional_bic import InterventionalBICScore
from cpa.scores.bdeu import BDeuScore
from cpa.scores.decomposable import DecomposableScore, ScoreCache

__all__ = [
    # bic.py
    "BICScore",
    "ExtendedBICScore",
    # bge.py
    "BGeScore",
    # interventional_bic.py
    "InterventionalBICScore",
    # bdeu.py
    "BDeuScore",
    # decomposable.py
    "DecomposableScore",
    "ScoreCache",
]
