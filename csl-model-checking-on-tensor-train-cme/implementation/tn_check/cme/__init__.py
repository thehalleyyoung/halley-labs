"""
CME-to-MPO compiler.

Converts reaction network descriptions into MPO representations of the
Chemical Master Equation generator matrix. Supports mass-action kinetics,
Hill functions, Michaelis-Menten kinetics, and general propensity functions.
"""

from tn_check.cme.reaction_network import (
    Species,
    Reaction,
    ReactionNetwork,
    KineticsType,
    PropensityFunction,
    MassActionPropensity,
    HillPropensity,
    MichaelisMentenPropensity,
    CustomPropensity,
)
from tn_check.cme.compiler import (
    CMECompiler,
    compile_reaction_to_mpo,
    compile_network_to_mpo,
    compile_propensity_to_diagonal,
)
from tn_check.cme.stoichiometry import (
    StoichiometryMatrix,
    compute_stoichiometry,
    find_conservation_laws,
    compute_reachable_bounds,
)
from tn_check.cme.fsp import (
    FSPBounds,
    compute_fsp_bounds,
    adaptive_fsp_expansion,
    validate_fsp_truncation,
)
from tn_check.cme.initial_state import (
    deterministic_initial_state,
    poisson_initial_state,
    binomial_initial_state,
    thermal_initial_state,
)

__all__ = [
    "Species", "Reaction", "ReactionNetwork",
    "KineticsType", "PropensityFunction",
    "MassActionPropensity", "HillPropensity",
    "MichaelisMentenPropensity", "CustomPropensity",
    "CMECompiler", "compile_reaction_to_mpo", "compile_network_to_mpo",
    "StoichiometryMatrix", "compute_stoichiometry",
    "find_conservation_laws", "compute_reachable_bounds",
    "FSPBounds", "compute_fsp_bounds", "adaptive_fsp_expansion",
    "deterministic_initial_state", "poisson_initial_state",
]
