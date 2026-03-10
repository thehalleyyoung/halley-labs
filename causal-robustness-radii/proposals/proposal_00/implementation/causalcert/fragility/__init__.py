"""
Fragility scoring sub-package — per-edge vulnerability analysis (ALG 3).

Assigns each edge (and each candidate absent edge) a fragility score
measuring how strongly a single edit to that edge position could affect
the causal conclusion.  Scores are decomposed into channels: d-separation,
identification, and estimation.
"""

from causalcert.fragility.scorer import FragilityScorerImpl
from causalcert.fragility.channels import DSepChannel, IdentificationChannel, EstimationChannel
from causalcert.fragility.aggregation import aggregate_scores, AggregationMethod
from causalcert.fragility.ranking import rank_edges, top_k_fragile
from causalcert.fragility.bfs import single_edit_bfs
from causalcert.fragility.theoretical import (
    theoretical_fragility_scores,
    structural_robustness_lower_bound,
    structural_robustness_upper_bound,
    skeleton_bridge_edges,
    causal_articulation_points,
)
from causalcert.fragility.bootstrap import (
    BootstrapFragilityResult,
    bootstrap_fragility_scores,
    rank_stability_analysis,
    full_bootstrap_analysis,
)

__all__ = [
    "FragilityScorerImpl",
    "DSepChannel",
    "IdentificationChannel",
    "EstimationChannel",
    "aggregate_scores",
    "AggregationMethod",
    "rank_edges",
    "top_k_fragile",
    "single_edit_bfs",
    # theoretical
    "theoretical_fragility_scores",
    "structural_robustness_lower_bound",
    "structural_robustness_upper_bound",
    "skeleton_bridge_edges",
    "causal_articulation_points",
    # bootstrap
    "BootstrapFragilityResult",
    "bootstrap_fragility_scores",
    "rank_stability_analysis",
    "full_bootstrap_analysis",
]
