"""
Conditional independence testing sub-package.

Provides multiple CI test implementations (kernel, partial-correlation,
rank-based, conditional randomization, HSIC, mutual information,
classifier-based) together with Cauchy-combination ensemble aggregation,
adaptive test selection, Benjamini–Yekutieli FDR control, result caching
with warm-start kernel matrices, shared kernel operations, and diagnostics.
"""

from causalcert.ci_testing.base import BaseCITest
from causalcert.ci_testing.kci import KernelCITest
from causalcert.ci_testing.rank import RankCITest
from causalcert.ci_testing.partial_corr import PartialCorrelationTest
from causalcert.ci_testing.crt import ConditionalRandomizationTest
from causalcert.ci_testing.ensemble import CauchyCombinationTest
from causalcert.ci_testing.multiplicity import BenjaminiYekutieli, ancestral_pruning
from causalcert.ci_testing.power import power_envelope, required_sample_size
from causalcert.ci_testing.cache import CITestCache
from causalcert.ci_testing.nonparametric import (
    HSICTest as _HSICTestLegacy,
    DistanceCorrelationTest,
    ClassifierCITest as _ClassifierCITestLegacy,
    ConditionalMITest,
)
from causalcert.ci_testing.parametric import (
    LinearGaussianCITest,
    BayesianCITest,
    LikelihoodRatioCITest,
    FTestCI,
)
from causalcert.ci_testing.hsic import HSICTest
from causalcert.ci_testing.mutual_info import MutualInfoCITest
from causalcert.ci_testing.classifier import ClassifierCITest
from causalcert.ci_testing.adaptive import AdaptiveEnsemble
from causalcert.ci_testing.kernel_ops import KernelCache
from causalcert.ci_testing.diagnostics import CIDiagnostics

__all__ = [
    "BaseCITest",
    "KernelCITest",
    "RankCITest",
    "PartialCorrelationTest",
    "ConditionalRandomizationTest",
    "CauchyCombinationTest",
    "BenjaminiYekutieli",
    "ancestral_pruning",
    "power_envelope",
    "required_sample_size",
    "CITestCache",
    # nonparametric
    "HSICTest",
    "DistanceCorrelationTest",
    "ClassifierCITest",
    "ConditionalMITest",
    # parametric
    "LinearGaussianCITest",
    "BayesianCITest",
    "LikelihoodRatioCITest",
    "FTestCI",
    # new modules
    "MutualInfoCITest",
    "AdaptiveEnsemble",
    "KernelCache",
    "CIDiagnostics",
]
