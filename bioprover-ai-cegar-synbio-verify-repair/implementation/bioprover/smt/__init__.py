"""SMT solver interface module for BioProver.

Provides Z3 and dReal solver interfaces, proof post-processing,
interpolant extraction, and portfolio solving for biological system
verification and counterexample-guided abstraction refinement.
"""

from bioprover.smt.solver_base import (
    AbstractSMTSolver,
    CounterexampleTrace,
    Model,
    SMTResult,
    SolverStatistics,
)
from bioprover.smt.z3_interface import Z3Solver
from bioprover.smt.dreal_interface import DRealSolver
from bioprover.smt.interpolation import (
    CraigInterpolant,
    InterpolantExtractor,
)
from bioprover.smt.portfolio import PortfolioSolver
from bioprover.smt.proof_checker import ProofStep, ProofTree

__all__ = [
    "AbstractSMTSolver",
    "CounterexampleTrace",
    "CraigInterpolant",
    "DRealSolver",
    "InterpolantExtractor",
    "Model",
    "PortfolioSolver",
    "ProofStep",
    "ProofTree",
    "SMTResult",
    "SolverStatistics",
    "Z3Solver",
]
