"""
Pipeline sub-package — orchestration, CLI, caching, and parallel execution.

The pipeline orchestrator (ALG 8) coordinates all sub-modules to produce
a full structural-robustness audit report.
"""

from causalcert.pipeline.orchestrator import CausalCertPipeline
from causalcert.pipeline.config import PipelineRunConfig
from causalcert.pipeline.cache import ResultCache
from causalcert.pipeline.api import CausalCertAnalysis, AnalysisExplorer, compare_analyses
from causalcert.pipeline.batch import BatchRunner, BatchResult, batch_summary_report
from causalcert.pipeline.diagnostics import (
    preflight_check,
    posthoc_diagnostics,
    recommend_parameters,
    full_diagnostics,
)

__all__ = [
    "CausalCertPipeline",
    "PipelineRunConfig",
    "ResultCache",
    # api
    "CausalCertAnalysis",
    "AnalysisExplorer",
    "compare_analyses",
    # batch
    "BatchRunner",
    "BatchResult",
    "batch_summary_report",
    # diagnostics
    "preflight_check",
    "posthoc_diagnostics",
    "recommend_parameters",
    "full_diagnostics",
]
