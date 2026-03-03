"""Diagnostics and sensitivity analysis for the CPA engine.

Provides tools for assessing the structural stability of causal
discoveries and comparing alternative causal models.

Modules
-------
sensitivity
    Perturbation-based structural stability analysis (T8).
model_comparison
    BIC/AIC model selection and equivalence class analysis.
"""

from cpa.diagnostics.sensitivity import (
    SensitivityAnalyzer,
    DescriptorSensitivity,
    DiagnosticReport,
)
from cpa.diagnostics.model_comparison import (
    ModelSelector,
    EquivalenceClassAnalyzer,
)

__all__ = [
    "SensitivityAnalyzer",
    "DescriptorSensitivity",
    "DiagnosticReport",
    "ModelSelector",
    "EquivalenceClassAnalyzer",
]
