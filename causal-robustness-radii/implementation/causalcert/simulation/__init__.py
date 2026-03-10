"""
Simulation sub-package — data-generating processes and synthetic data.

Provides interfaces and type definitions for specifying structural causal
models, generating synthetic observational and interventional datasets,
and recording ground-truth causal quantities for benchmarking and power
analysis.
"""

from causalcert.simulation.types import (
    DGPSpec,
    GroundTruth,
    SimulationResult,
)
from causalcert.simulation.protocols import (
    DataGenerator,
    NoiseModel,
)
from causalcert.simulation.noise_models import (
    GaussianNoise,
    StudentTNoise,
    MixtureNoise,
    HeteroskedasticNoise,
    NonAdditiveNoise,
    DiscreteNoise,
    create_noise,
)
from causalcert.simulation.engines import (
    LinearGaussianEngine,
    NonlinearEngine,
    MixedTypeEngine,
    InterventionalEngine,
)
from causalcert.simulation.perturbation import (
    PerturbedDAG,
    PerturbationImpact,
    PerturbationGenerator,
    ImpactCategory,
)
from causalcert.simulation.faithfulness import (
    FaithfulnessChecker,
    FaithfulnessReport,
    FaithfulnessViolation,
    PathCancellation,
)
from causalcert.simulation.monte_carlo import (
    MonteCarloRunner,
    SimStudyConfig,
    SimStudyResult,
    run_simulation_study,
)
from causalcert.simulation.dgp_library import (
    LaLondeDGP,
    SmokingBirthweightDGP,
    IHDPSimulation,
    InstrumentDGP,
    MediationDGP,
    ConfoundedDGP,
    FaithfulnessViolationDGP,
    SparseHighDimDGP,
    create_dgp,
    list_dgps,
)

__all__ = [
    "DGPSpec",
    "GroundTruth",
    "SimulationResult",
    # protocols
    "DataGenerator",
    "NoiseModel",
    # noise models
    "GaussianNoise",
    "StudentTNoise",
    "MixtureNoise",
    "HeteroskedasticNoise",
    "NonAdditiveNoise",
    "DiscreteNoise",
    "create_noise",
    # engines
    "LinearGaussianEngine",
    "NonlinearEngine",
    "MixedTypeEngine",
    "InterventionalEngine",
    # perturbation
    "PerturbedDAG",
    "PerturbationImpact",
    "PerturbationGenerator",
    "ImpactCategory",
    # faithfulness
    "FaithfulnessChecker",
    "FaithfulnessReport",
    "FaithfulnessViolation",
    "PathCancellation",
    # monte carlo
    "MonteCarloRunner",
    "SimStudyConfig",
    "SimStudyResult",
    "run_simulation_study",
    # dgp library
    "LaLondeDGP",
    "SmokingBirthweightDGP",
    "IHDPSimulation",
    "InstrumentDGP",
    "MediationDGP",
    "ConfoundedDGP",
    "FaithfulnessViolationDGP",
    "SparseHighDimDGP",
    "create_dgp",
    "list_dgps",
]
