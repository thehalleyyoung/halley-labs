"""
Junction-tree exact inference engine with do-operator support.

Implements Hugin/Shafer-Shenoy message passing on clique trees built from
triangulated moral graphs of bounded-treewidth Bayesian networks.  Supports
observational queries, interventional (do-calculus) queries, and adaptive
discretization of continuous variables for systemic-risk applications.
"""

from .potential_table import PotentialTable
from .discretization import AdaptiveDiscretizer
from .cache import InferenceCache
from .clique_tree import CliqueTree, CliqueNode
from .message_passing import MessagePasser
from .do_operator import DoOperator
from .engine import JunctionTreeEngine
from .discretization_error import DiscretizationErrorAnalyzer

__all__ = [
    "JunctionTreeEngine",
    "CliqueTree",
    "CliqueNode",
    "MessagePasser",
    "PotentialTable",
    "AdaptiveDiscretizer",
    "DoOperator",
    "InferenceCache",
    "DiscretizationErrorAnalyzer",
]
