"""Causal-Plasticity Atlas (CPA) Engine.

A framework for analysing causal mechanism plasticity across multiple
observational or experimental contexts.  Provides structural causal
model operations, multi-context comparison, plasticity quantification,
tipping-point detection, and robustness certification.

Subpackages
-----------
core
    Core types, SCM operations, context management, and MCCM.
stats
    Statistical distributions, divergence measures, and testing.
utils
    Validation, caching, parallel computation, and logging.
inference
    Causal inference engine (do-calculus, counterfactuals, identifiability).
mec
    Markov Equivalence Class operations (CPDAG, PAG, orientation rules).
ci_tests
    Conditional independence testing (Fisher-z, kernel, discrete, CMI).
scores
    Scoring functions for structure learning (BIC, BGe, BDeu).
sampling
    MCMC methods for DAG space (order, partition, structure, tempering).
operators
    Advanced genetic operators for QD search.
baselines
    Evaluation baselines (ICP, CD-NOD, JCI, GES, pooled).
analysis
    Post-hoc analysis (convergence, sensitivity, mechanism comparison).
data
    Synthetic data generation (SB1-SB5, nonlinear SCMs, semi-synthetic).
streaming
    Online/streaming algorithms (IPU, windowed detection).
scalability
    Scalability infrastructure (caching, sparse ops, distributed).
config
    Extended configuration (experiments, hyperparameters, registry).
"""

__version__ = "0.1.0"
__author__ = "CPA Team"

from cpa.core.types import (
    PlasticityClass,
    CertificateType,
    EdgeClassification,
    ChangeType,
    SCM,
    Context,
    MCCM,
    AlignmentMapping,
    PlasticityDescriptor,
    TippingPoint,
    RobustnessCertificate,
    QDGenome,
    QDArchiveEntry,
    CVTCell,
)
from cpa.core.scm import StructuralCausalModel
from cpa.core.context import ContextSpace, ContextPartition
from cpa.core.mccm import MultiContextCausalModel

__all__ = [
    "__version__",
    # Enums
    "PlasticityClass",
    "CertificateType",
    "EdgeClassification",
    "ChangeType",
    # Data types
    "SCM",
    "Context",
    "MCCM",
    "AlignmentMapping",
    "PlasticityDescriptor",
    "TippingPoint",
    "RobustnessCertificate",
    "QDGenome",
    "QDArchiveEntry",
    "CVTCell",
    # Classes
    "StructuralCausalModel",
    "ContextSpace",
    "ContextPartition",
    "MultiContextCausalModel",
]
