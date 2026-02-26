"""
Causal Polytope LP Solver Module
=================================

Implements column-generation-based LP solving for computing worst-case bounds
on interventional queries over causal DAG models with discrete variables.

Core classes:
    - CausalPolytopeSolver: top-level orchestrator
    - ColumnGenerationSolver: master LP + column generation loop
    - PricingSubproblem: DAG-aware pricing via junction-tree DP
    - ConstraintEncoder: d-separation, marginal, interventional constraints
    - InterventionalPolytope: do-operator polytope construction
    - BoundExtractor: worst-case bound extraction and sensitivity analysis
"""

from .causal_polytope import CausalPolytopeSolver, SolverResult, SolverConfig
from .column_generation import ColumnGenerationSolver, ColumnPool, MasterProblem
from .pricing import PricingSubproblem, PricingStrategy, PricingResult
from .constraints import ConstraintEncoder, ConstraintType, ConstraintBlock
from .interventional import InterventionalPolytope, DoOperator, IdentifiabilityResult
from .bounds import BoundExtractor, BoundResult, SensitivityReport

__all__ = [
    "CausalPolytopeSolver",
    "SolverResult",
    "SolverConfig",
    "ColumnGenerationSolver",
    "ColumnPool",
    "MasterProblem",
    "PricingSubproblem",
    "PricingStrategy",
    "PricingResult",
    "ConstraintEncoder",
    "ConstraintType",
    "ConstraintBlock",
    "InterventionalPolytope",
    "DoOperator",
    "IdentifiabilityResult",
    "BoundExtractor",
    "BoundResult",
    "SensitivityReport",
]
