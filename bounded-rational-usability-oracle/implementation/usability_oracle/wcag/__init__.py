"""usability_oracle.wcag — WCAG 2.2 conformance checking."""

from usability_oracle.wcag.types import (
    ConformanceLevel,
    ImpactLevel,
    SuccessCriterion,
    WCAGGuideline,
    WCAGPrinciple,
    WCAGResult,
    WCAGViolation,
)
from usability_oracle.wcag.protocols import (
    WCAGEvaluator,
    WCAGParser,
    WCAGReporter,
)
from usability_oracle.wcag.contrast import (
    Color,
    ContrastResult,
    check_contrast,
    contrast_ratio,
    relative_luminance,
    simulate_cvd,
)
from usability_oracle.wcag.keyboard import (
    FocusableElement,
    FocusTrap,
    NavigationCost,
    SkipNavResult,
    extract_tab_order,
    compute_navigation_cost,
    verify_skip_navigation,
)
from usability_oracle.wcag.semantic import (
    SemanticAnalysisResult,
    analyse_semantics,
    validate_heading_hierarchy,
    validate_landmarks,
    validate_aria_roles,
)
from usability_oracle.wcag.parser import (
    WCAGSpecification,
    WCAGXMLParser,
    TechniqueRef,
)
from usability_oracle.wcag.evaluator import WCAGConformanceEvaluator
from usability_oracle.wcag.reporter import (
    WCAGConformanceReporter,
    ReportSummary,
    RemediationItem,
    compute_summary,
    rank_remediations,
)
from usability_oracle.wcag.mapping import (
    CognitiveCostDelta,
    compute_cost_delta,
    compute_violation_cost,
    to_cost_element,
    wcag_cost_summary,
)

__all__ = [
    # types
    "ConformanceLevel",
    "ImpactLevel",
    "SuccessCriterion",
    "WCAGGuideline",
    "WCAGPrinciple",
    "WCAGResult",
    "WCAGViolation",
    # protocols
    "WCAGEvaluator",
    "WCAGParser",
    "WCAGReporter",
    # contrast
    "Color",
    "ContrastResult",
    "check_contrast",
    "contrast_ratio",
    "relative_luminance",
    "simulate_cvd",
    # keyboard
    "FocusableElement",
    "FocusTrap",
    "NavigationCost",
    "SkipNavResult",
    "extract_tab_order",
    "compute_navigation_cost",
    "verify_skip_navigation",
    # semantic
    "SemanticAnalysisResult",
    "analyse_semantics",
    "validate_heading_hierarchy",
    "validate_landmarks",
    "validate_aria_roles",
    # parser
    "WCAGSpecification",
    "WCAGXMLParser",
    "TechniqueRef",
    # evaluator
    "WCAGConformanceEvaluator",
    # reporter
    "WCAGConformanceReporter",
    "ReportSummary",
    "RemediationItem",
    "compute_summary",
    "rank_remediations",
    # mapping
    "CognitiveCostDelta",
    "compute_cost_delta",
    "compute_violation_cost",
    "to_cost_element",
    "wcag_cost_summary",
]
