"""CPA conditional independence testing subpackage.

Conditional independence tests for structure learning including
Fisher-z, kernel-based, discrete, CMI-based, and adaptive tests.

Modules
-------
fisher_z
    Fisher-z transform test for linear Gaussian data.
kernel_ci
    Kernel-based CI tests (HSIC, KCI).
discrete_ci
    Chi-squared and G-test for discrete data.
conditional_mutual_info
    CMI-based independence tests.
adaptive
    Adaptive test selection based on data characteristics.
"""

from __future__ import annotations

from cpa.ci_tests.fisher_z import FisherZTest, PartialCorrelation, CITestResult
from cpa.ci_tests.kernel_ci import KernelCITest, HSICTest
from cpa.ci_tests.discrete_ci import (
    ChiSquaredTest, GTest, FisherExactTest, discretize,
)
from cpa.ci_tests.conditional_mutual_info import CMITest, KSGEstimator
from cpa.ci_tests.adaptive import AdaptiveCITest, DataCharacteristics, CITestSuite

__all__ = [
    # fisher_z.py
    "FisherZTest",
    "PartialCorrelation",
    "CITestResult",
    # kernel_ci.py
    "KernelCITest",
    "HSICTest",
    # discrete_ci.py
    "ChiSquaredTest",
    "GTest",
    "FisherExactTest",
    "discretize",
    # conditional_mutual_info.py
    "CMITest",
    "KSGEstimator",
    # adaptive.py
    "AdaptiveCITest",
    "DataCharacteristics",
    "CITestSuite",
]
