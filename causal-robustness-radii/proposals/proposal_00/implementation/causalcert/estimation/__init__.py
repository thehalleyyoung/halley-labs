"""
Estimation sub-package — causal effect estimation under a given DAG.

Provides back-door criterion checking, valid adjustment set enumeration,
augmented inverse-probability weighting (AIPW), cross-fitting, and
influence function computation.
"""

from causalcert.estimation.backdoor import satisfies_backdoor, enumerate_adjustment_sets
from causalcert.estimation.adjustment import find_optimal_adjustment_set
from causalcert.estimation.aipw import AIPWEstimator
from causalcert.estimation.crossfit import CrossFitter
from causalcert.estimation.influence import influence_function
from causalcert.estimation.effects import estimate_ate, estimate_att
from causalcert.estimation.propensity import PropensityModel
from causalcert.estimation.outcome import OutcomeModel
from causalcert.estimation.sensitivity import (
    e_value_rr,
    e_value_or,
    e_value_hr,
    breakdown_point_fraction,
    rosenbaum_bounds,
    sensitivity_contour,
    robustness_value,
)
from causalcert.estimation.semiparametric import (
    tmle_ate,
    one_step_ate,
    super_learner,
    eif_ate,
)
from causalcert.estimation.mediation import (
    MediationResult,
    mediation_analysis,
    estimate_nde,
    estimate_nie,
    path_specific_effect,
)

__all__ = [
    "satisfies_backdoor",
    "enumerate_adjustment_sets",
    "find_optimal_adjustment_set",
    "AIPWEstimator",
    "CrossFitter",
    "influence_function",
    "estimate_ate",
    "estimate_att",
    "PropensityModel",
    "OutcomeModel",
    # sensitivity
    "e_value_rr",
    "e_value_or",
    "e_value_hr",
    "breakdown_point_fraction",
    "rosenbaum_bounds",
    "sensitivity_contour",
    "robustness_value",
    # semiparametric
    "tmle_ate",
    "one_step_ate",
    "super_learner",
    "eif_ate",
    # mediation
    "MediationResult",
    "mediation_analysis",
    "estimate_nde",
    "estimate_nie",
    "path_specific_effect",
]
