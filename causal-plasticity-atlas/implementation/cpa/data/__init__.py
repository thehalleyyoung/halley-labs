"""CPA data subpackage.

Synthetic data generation utilities for benchmarking and testing
including standard benchmarks, nonlinear SCM simulation,
semi-synthetic datasets, and perturbation methods.

Modules
-------
generators
    Synthetic data generators (SB1-SB5).
nonlinear_scm
    Nonlinear SCM simulation.
semi_synthetic
    Semi-synthetic benchmark datasets.
perturbation
    Data perturbation for robustness testing.
"""

from cpa.data.generators import (
    SyntheticGenerator,
    generate_sb1,
    generate_sb2,
    generate_sb3,
    generate_sb4,
    generate_sb5,
)
from cpa.data.nonlinear_scm import NonlinearSCMGenerator, MechanismType
from cpa.data.semi_synthetic import (
    SemiSyntheticLoader,
    sachs_network,
    alarm_network,
    insurance_network,
)
from cpa.data.perturbation import DataPerturbation, PerturbationType

__all__ = [
    # generators.py
    "SyntheticGenerator",
    "generate_sb1",
    "generate_sb2",
    "generate_sb3",
    "generate_sb4",
    "generate_sb5",
    # nonlinear_scm.py
    "NonlinearSCMGenerator",
    "MechanismType",
    # semi_synthetic.py
    "SemiSyntheticLoader",
    "sachs_network",
    "alarm_network",
    "insurance_network",
    # perturbation.py
    "DataPerturbation",
    "PerturbationType",
]
