"""
usability_oracle.output — Output formatting for usability analysis results.

Supports JSON, SARIF 2.1.0, HTML, and rich console output.
"""

from __future__ import annotations

from usability_oracle.output.models import (
    AnnotatedElement,
    OutputResult,
    OutputSection,
    PipelineResult,
    StageTimingInfo,
)
from usability_oracle.output.json_output import JSONFormatter
from usability_oracle.output.sarif import SARIFFormatter
from usability_oracle.output.html_report import HTMLReportGenerator
from usability_oracle.output.console import ConsoleFormatter
from usability_oracle.output.sarif_typed import SarifFormatter
from usability_oracle.output.diff_view import (
    DiffFormatter,
    DiffEntry,
    DiffStatus,
    TreeDiffResult,
    CostDiffResult,
    BottleneckDiffResult,
)
from usability_oracle.output.github_integration import GitHubCommentFormatter

__all__ = [
    "AnnotatedElement",
    "OutputResult",
    "OutputSection",
    "PipelineResult",
    "StageTimingInfo",
    "JSONFormatter",
    "SARIFFormatter",
    "HTMLReportGenerator",
    "ConsoleFormatter",
    "SarifFormatter",
    "DiffFormatter",
    "DiffEntry",
    "DiffStatus",
    "TreeDiffResult",
    "CostDiffResult",
    "BottleneckDiffResult",
    "GitHubCommentFormatter",
]
