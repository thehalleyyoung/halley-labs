"""
Causal discovery module for the Causal-Shielded Adaptive Trading system.

Provides structural causal models, additive noise model identification,
kernel independence testing (HSIC), constraint-based structure learning
(PC/FCI), score-based structure learning, and Markov equivalence class
utilities.
"""

from .scm import StructuralCausalModel, StructuralEquation, LinearEquation, ANMEquation
from .additive_noise import AdditiveNoiseModel, ANMDirectionTest
from .hsic import HSIC, ConditionalHSIC, GaussianKernel, PolynomialKernel, LinearKernel
from .pc_algorithm import PCAlgorithm, FCIAlgorithm, StablePCAlgorithm
from .dag_scoring import BICScore, BDeuScore, BGeScore, GreedyHillClimbing
from .markov_equivalence import CPDAG, PAG, MarkovEquivalenceClass

__all__ = [
    "StructuralCausalModel",
    "StructuralEquation",
    "LinearEquation",
    "ANMEquation",
    "AdditiveNoiseModel",
    "ANMDirectionTest",
    "HSIC",
    "ConditionalHSIC",
    "GaussianKernel",
    "PolynomialKernel",
    "LinearKernel",
    "PCAlgorithm",
    "FCIAlgorithm",
    "StablePCAlgorithm",
    "BICScore",
    "BDeuScore",
    "BGeScore",
    "GreedyHillClimbing",
    "CPDAG",
    "PAG",
    "MarkovEquivalenceClass",
]
