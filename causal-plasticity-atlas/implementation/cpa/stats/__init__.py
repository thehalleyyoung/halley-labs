"""CPA statistics subpackage.

Provides distributional computations, divergence measures,
information-theoretic quantities, and statistical testing utilities
used throughout the Causal-Plasticity Atlas engine.
"""

from cpa.stats.distributions import (
    GaussianConditional,
    jsd_discrete,
    jsd_gaussian,
    sqrt_jsd_discrete,
    sqrt_jsd_gaussian,
    kl_discrete,
    kl_gaussian,
    partial_correlation,
    partial_correlation_test,
    fisher_z_test,
    bonferroni_correction,
    bh_fdr_correction,
    bootstrap_ci,
    cohens_d,
    hedges_g,
    permutation_test,
)
from cpa.stats.information_theory import (
    shannon_entropy_discrete,
    shannon_entropy_gaussian,
    mutual_information_discrete,
    mutual_information_gaussian,
    conditional_mutual_information,
    transfer_entropy,
    multi_distribution_jsd,
    normalized_information_distance,
)

__all__ = [
    "GaussianConditional",
    "jsd_discrete",
    "jsd_gaussian",
    "sqrt_jsd_discrete",
    "sqrt_jsd_gaussian",
    "kl_discrete",
    "kl_gaussian",
    "partial_correlation",
    "partial_correlation_test",
    "fisher_z_test",
    "bonferroni_correction",
    "bh_fdr_correction",
    "bootstrap_ci",
    "cohens_d",
    "hedges_g",
    "permutation_test",
    "shannon_entropy_discrete",
    "shannon_entropy_gaussian",
    "mutual_information_discrete",
    "mutual_information_gaussian",
    "conditional_mutual_information",
    "transfer_entropy",
    "multi_distribution_jsd",
    "normalized_information_distance",
]
