"""
Solver sub-package — ILP, LP relaxation, FPT-DP, and CDCL search.

Computes the robustness radius (minimum edit distance to overturn a causal
conclusion) via exact and approximate solvers.  The unified search interface
automatically selects the best strategy based on DAG size and treewidth.
"""

from causalcert.solver.ilp import ILPSolver
from causalcert.solver.lp_relaxation import LPRelaxationSolver
from causalcert.solver.fpt import FPTSolver
from causalcert.solver.cdcl import CDCLSolver, ConflictClause
from causalcert.solver.constraints import (
    AcyclicityConstraint,
    BackDoorConstraint,
    BudgetConstraint,
    CIConsistencyConstraint,
    ConclusionNegationConstraint,
    MutualExclusionConstraint,
    SymmetryBreakingConstraint,
    ValidInequalityTightener,
)
from causalcert.solver.bounds import BoundManager, BoundState
from causalcert.solver.search import UnifiedSolver
from causalcert.solver.symmetry import (
    SymmetryAwareBrancher,
    automorphism_group,
    orbit_fixing_constraints,
    lex_leader_constraints,
)
from causalcert.solver.cutting_planes import (
    CuttingPlaneLoop,
    CutPool,
    SeparationOracle,
    generate_gomory_cuts,
)
from causalcert.solver.heuristics import (
    HeuristicResult,
    greedy_constructive,
    local_search,
    simulated_annealing,
    tabu_search,
    genetic_algorithm,
    hybrid_heuristic_ilp,
)

__all__ = [
    "ILPSolver",
    "LPRelaxationSolver",
    "FPTSolver",
    "CDCLSolver",
    "ConflictClause",
    "AcyclicityConstraint",
    "BackDoorConstraint",
    "BudgetConstraint",
    "CIConsistencyConstraint",
    "ConclusionNegationConstraint",
    "MutualExclusionConstraint",
    "SymmetryBreakingConstraint",
    "ValidInequalityTightener",
    "BoundManager",
    "BoundState",
    "UnifiedSolver",
    # symmetry
    "SymmetryAwareBrancher",
    "automorphism_group",
    "orbit_fixing_constraints",
    "lex_leader_constraints",
    # cutting_planes
    "CuttingPlaneLoop",
    "CutPool",
    "SeparationOracle",
    "generate_gomory_cuts",
    # heuristics
    "HeuristicResult",
    "greedy_constructive",
    "local_search",
    "simulated_annealing",
    "tabu_search",
    "genetic_algorithm",
    "hybrid_heuristic_ilp",
]
