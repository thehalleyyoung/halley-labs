"""Financial instruments module for CausalBound.

Provides payoff encoding, valuation, and CPD discretization
for credit default swaps, interest rate swaps, repos, and equity options.
"""

from .cds import CDSModel
from .irs import IRSModel
from .repo import RepoModel
from .equity_option import EquityOptionModel
from .discretization import InstrumentDiscretizer
from .exposure import ExposureProfile

__all__ = [
    "CDSModel",
    "IRSModel",
    "RepoModel",
    "EquityOptionModel",
    "InstrumentDiscretizer",
    "ExposureProfile",
]
