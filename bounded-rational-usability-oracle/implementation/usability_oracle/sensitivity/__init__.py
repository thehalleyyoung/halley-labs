"""usability_oracle.sensitivity — Parametric sensitivity analysis."""

from usability_oracle.sensitivity.types import (
    MorrisResult,
    ParameterRange,
    SensitivityConfig,
    SensitivityResult,
    SobolIndices,
)
from usability_oracle.sensitivity.protocols import (
    GlobalSensitivity,
    LocalSensitivity,
    SensitivityAnalyzer,
)
from usability_oracle.sensitivity.sobol import (
    SobolAnalyzer,
    ConvergenceRecord,
    saltelli_sample,
    sobol_sequence,
    monitor_convergence,
    rank_parameters_by_total_order,
    identify_interactions,
)
from usability_oracle.sensitivity.morris import (
    MorrisAnalyzer,
    ParameterEffect,
    classify_parameter,
    optimized_trajectories,
    compute_elementary_effects,
)
from usability_oracle.sensitivity.local import (
    LocalSensitivityAnalyzer,
    compute_gradient,
    compute_jacobian,
    elasticity,
    normalized_sensitivity,
    condition_number,
    collinearity_index,
    central_difference,
    forward_difference,
    complex_step,
)
from usability_oracle.sensitivity.variance import (
    VarianceAnalyzer,
    VarianceComponent,
    VarianceDecomposition,
    functional_anova,
    first_order_variance,
    interaction_variance,
    correlation_ratio,
    variance_importance_ranking,
)
from usability_oracle.sensitivity.robustness import (
    RobustnessAnalyzer,
    RobustnessCertificate,
    RobustnessRegion,
    ScenarioResult,
    compute_robustness_region,
    worst_case_analysis,
    probabilistic_robustness,
    vertex_analysis,
    scenario_robustness,
    identify_critical_parameters,
)
from usability_oracle.sensitivity.calibration import (
    CalibrationResult,
    CrossValidationResult,
    ProfileLikelihoodResult,
    maximum_likelihood,
    bayesian_map,
    fisher_information,
    cramer_rao_bounds,
    gaussian_log_likelihood,
    profile_likelihood,
    parameter_correlations,
    cross_validate,
)

__all__ = [
    # types
    "MorrisResult",
    "ParameterRange",
    "SensitivityConfig",
    "SensitivityResult",
    "SobolIndices",
    # protocols
    "GlobalSensitivity",
    "LocalSensitivity",
    "SensitivityAnalyzer",
    # sobol
    "SobolAnalyzer",
    "ConvergenceRecord",
    "saltelli_sample",
    "sobol_sequence",
    "monitor_convergence",
    "rank_parameters_by_total_order",
    "identify_interactions",
    # morris
    "MorrisAnalyzer",
    "ParameterEffect",
    "classify_parameter",
    "optimized_trajectories",
    "compute_elementary_effects",
    # local
    "LocalSensitivityAnalyzer",
    "compute_gradient",
    "compute_jacobian",
    "elasticity",
    "normalized_sensitivity",
    "condition_number",
    "collinearity_index",
    "central_difference",
    "forward_difference",
    "complex_step",
    # variance
    "VarianceAnalyzer",
    "VarianceComponent",
    "VarianceDecomposition",
    "functional_anova",
    "first_order_variance",
    "interaction_variance",
    "correlation_ratio",
    "variance_importance_ranking",
    # robustness
    "RobustnessAnalyzer",
    "RobustnessCertificate",
    "RobustnessRegion",
    "ScenarioResult",
    "compute_robustness_region",
    "worst_case_analysis",
    "probabilistic_robustness",
    "vertex_analysis",
    "scenario_robustness",
    "identify_critical_parameters",
    # calibration
    "CalibrationResult",
    "CrossValidationResult",
    "ProfileLikelihoodResult",
    "maximum_likelihood",
    "bayesian_map",
    "fisher_information",
    "cramer_rao_bounds",
    "gaussian_log_likelihood",
    "profile_likelihood",
    "parameter_correlations",
    "cross_validate",
]
