"""
Reporting sub-package — audit report generation in multiple formats.

Generates structural audit reports in JSON, HTML (via Jinja2), and LaTeX,
plus plain-language narrative summaries.
"""

from causalcert.reporting.audit import generate_audit_report
from causalcert.reporting.json_report import to_json_report
from causalcert.reporting.html_report import to_html_report
from causalcert.reporting.latex_report import to_latex_tables
from causalcert.reporting.narrative import generate_narrative
from causalcert.reporting.interactive import (
    InteractiveAnalysis,
    AnalysisSession,
    ComparisonReport,
)

__all__ = [
    "generate_audit_report",
    "to_json_report",
    "to_html_report",
    "to_latex_tables",
    "generate_narrative",
    # interactive
    "InteractiveAnalysis",
    "AnalysisSession",
    "ComparisonReport",
]
