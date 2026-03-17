"""Analysis module: cost, risk, coverage, sensitivity, and statistics.

Provides quantitative analysis of regulatory compliance strategies.
"""

from regsynth_py.analysis.cost_analysis import CostAnalyzer
from regsynth_py.analysis.risk_analysis import RiskAnalyzer
from regsynth_py.analysis.coverage_analysis import CoverageAnalyzer
from regsynth_py.analysis.sensitivity_analysis import SensitivityAnalyzer
from regsynth_py.analysis.statistics import compute_hypervolume, generational_distance

__all__ = [
    "CostAnalyzer", "RiskAnalyzer", "CoverageAnalyzer",
    "SensitivityAnalyzer", "compute_hypervolume", "generational_distance",
]
