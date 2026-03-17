"""
taintflow.utils – Utility modules for the TaintFlow system.

Provides bitmap-based provenance tracking, information-theoretic math,
graph algorithms, serialization helpers, and parallel execution utilities.
"""

from __future__ import annotations

from taintflow.utils.bitmap import (
    ArrayContainer,
    BitmapContainer,
    ProvenanceBitmap,
    RoaringBitmap,
    RunContainer,
)
from taintflow.utils.math_utils import (
    binary_entropy,
    channel_capacity_bec,
    channel_capacity_bsc,
    channel_capacity_gaussian,
    entropy,
    kl_divergence,
    log2_safe,
    mutual_information_discrete,
)
from taintflow.utils.graph_utils import (
    DiGraph,
    topological_sort,
    strongly_connected_components,
    transitive_closure,
)

__all__ = [
    "ArrayContainer",
    "BitmapContainer",
    "DiGraph",
    "ProvenanceBitmap",
    "RoaringBitmap",
    "RunContainer",
    "binary_entropy",
    "channel_capacity_bec",
    "channel_capacity_bsc",
    "channel_capacity_gaussian",
    "entropy",
    "kl_divergence",
    "log2_safe",
    "mutual_information_discrete",
    "strongly_connected_components",
    "topological_sort",
    "transitive_closure",
]