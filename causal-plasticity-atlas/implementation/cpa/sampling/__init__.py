"""CPA sampling subpackage.

MCMC methods for exploring DAG space including order MCMC,
partition MCMC, structure MCMC, parallel tempering, and
DAG bootstrap methods.

Modules
-------
order_mcmc
    Order MCMC (Friedman & Koller).
partition_mcmc
    Partition MCMC for improved mixing.
structure_mcmc
    Structure MCMC with edge proposals.
parallel_tempering
    Parallel tempering for multi-modal posteriors.
dag_bootstrap
    DAG bootstrap for uncertainty quantification.
"""

from cpa.sampling.order_mcmc import OrderMCMC, OrderSample, DAGPosteriorSamples
from cpa.sampling.partition_mcmc import PartitionMCMC, Partition
from cpa.sampling.structure_mcmc import StructureMCMC, EdgeProposal
from cpa.sampling.parallel_tempering import ParallelTempering, TemperingConfig
from cpa.sampling.dag_bootstrap import DAGBootstrap, BootstrapResult

__all__ = [
    # order_mcmc.py
    "OrderMCMC",
    "OrderSample",
    "DAGPosteriorSamples",
    # partition_mcmc.py
    "PartitionMCMC",
    "Partition",
    # structure_mcmc.py
    "StructureMCMC",
    "EdgeProposal",
    # parallel_tempering.py
    "ParallelTempering",
    "TemperingConfig",
    # dag_bootstrap.py
    "DAGBootstrap",
    "BootstrapResult",
]
