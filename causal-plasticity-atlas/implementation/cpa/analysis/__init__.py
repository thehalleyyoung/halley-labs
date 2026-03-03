"""CPA analysis subpackage.

Post-hoc analysis tools for convergence analysis, stopping criteria,
ergodicity checking, global sensitivity analysis, and mechanism comparison.

Modules
-------
convergence
    Convergence analysis for QD search.
supermartingale
    Supermartingale stopping criterion.
ergodicity
    Ergodicity checking for atlas completeness.
sensitivity_global
    Global sensitivity analysis (Sobol, Morris).
mechanism_comparison
    Cross-context mechanism comparison.
"""

from cpa.analysis.convergence import ConvergenceAnalyzer, ConvergenceMetrics
from cpa.analysis.supermartingale import SupermartingaleStopper, StoppingResult
from cpa.analysis.ergodicity import ErgodicityChecker, ErgodicityResult
from cpa.analysis.sensitivity_global import (
    SobolAnalyzer,
    MorrisScreening,
    GlobalSensitivityResult,
)
from cpa.analysis.mechanism_comparison import (
    MechanismComparator,
    MechanismComparisonResult,
)

__all__ = [
    # convergence.py
    "ConvergenceAnalyzer",
    "ConvergenceMetrics",
    # supermartingale.py
    "SupermartingaleStopper",
    "StoppingResult",
    # ergodicity.py
    "ErgodicityChecker",
    "ErgodicityResult",
    # sensitivity_global.py
    "SobolAnalyzer",
    "MorrisScreening",
    "GlobalSensitivityResult",
    # mechanism_comparison.py
    "MechanismComparator",
    "MechanismComparisonResult",
]
