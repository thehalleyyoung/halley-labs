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
from causalcert.solver.clause_database import (
    Clause,
    ClauseDatabase,
    ClauseStats,
    EditLiteral,
)
from causalcert.solver.watched_literals import (
    WatchedLiteralEngine,
    PropagationResult,
    ImplicationRecord,
    LiteralValue,
)
from causalcert.solver.restart_strategy import (
    RestartScheduler,
    RestartPolicy,
    LubyRestart,
    GeometricRestart,
    GlucoseRestart,
    AdaptiveRestart,
    luby_sequence,
)
from causalcert.solver.phase_saving import PhaseSaver, Polarity
from causalcert.solver.conflict_analysis import (
    ConflictAnalyzer,
    ConflictResult,
    ImplicationGraph,
)
from causalcert.solver.branching import (
    BranchingEngine,
    BranchingStrategy,
    EVSIDS,
    LRB,
    CHB,
    RandomBranching,
)
from causalcert.solver.preprocessing import Preprocessor, PreprocessStats

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
    # clause_database
    "Clause",
    "ClauseDatabase",
    "ClauseStats",
    "EditLiteral",
    # watched_literals
    "WatchedLiteralEngine",
    "PropagationResult",
    "ImplicationRecord",
    "LiteralValue",
    # restart_strategy
    "RestartScheduler",
    "RestartPolicy",
    "LubyRestart",
    "GeometricRestart",
    "GlucoseRestart",
    "AdaptiveRestart",
    "luby_sequence",
    # phase_saving
    "PhaseSaver",
    "Polarity",
    # conflict_analysis
    "ConflictAnalyzer",
    "ConflictResult",
    "ImplicationGraph",
    # branching
    "BranchingEngine",
    "BranchingStrategy",
    "EVSIDS",
    "LRB",
    "CHB",
    "RandomBranching",
    # preprocessing
    "Preprocessor",
    "PreprocessStats",
]
