"""
CausalCert — Causal Robustness Radii.

Certifies the stability of causal conclusions under structural perturbations
to an assumed causal DAG.  Computes the minimum number of edge edits
(insertions, deletions, reversals) required to overturn a causal finding.

Modules
-------
dag         Graph representation, d-separation, MEC operations.
ci_testing  Conditional independence testing with multiplicity control.
solver      ILP / LP / FPT solvers for the robustness radius.
fragility   Per-edge fragility scoring and ranking.
estimation  Causal effect estimation (AIPW, back-door).
data        Data loading, validation, synthetic generation.
pipeline    Orchestrator, CLI, caching.
reporting   Audit report generation.
evaluation  Benchmarks, DGPs, ablation studies.
"""

from causalcert.types import (
    AuditReport,
    CITestResult,
    EditType,
    EstimationResult,
    FragilityChannel,
    FragilityScore,
    RobustnessRadius,
    StructuralEdit,
)
from causalcert.dag.graph import CausalDAG
from causalcert.pipeline.orchestrator import CausalCertPipeline
from causalcert.fragility.scorer import FragilityScorerImpl
from causalcert.pipeline.api import CausalCertAnalysis

__all__ = [
    "EditType",
    "StructuralEdit",
    "FragilityScore",
    "FragilityChannel",
    "RobustnessRadius",
    "AuditReport",
    "CITestResult",
    "EstimationResult",
    "CausalDAG",
    "CausalCertPipeline",
    "FragilityScorerImpl",
    "CausalCertAnalysis",
]

__version__ = "0.1.0"
