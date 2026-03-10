"""
Type definitions for the simulation and data-generating process sub-package.

Provides data structures for specifying data-generating processes (DGPs),
capturing simulation results, and representing ground-truth causal
quantities used in benchmark evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from causalcert.types import (
    AdjacencyMatrix,
    EdgeTuple,
    NodeId,
    NodeSet,
    VariableType,
)


# ---------------------------------------------------------------------------
# DGPSpec
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DGPSpec:
    """Specification of a data-generating process.

    Fully describes the structural causal model (SCM) used to generate
    synthetic datasets: the DAG structure, functional form per variable,
    noise distribution, coefficient ranges, and variable types.

    Attributes
    ----------
    adjacency : AdjacencyMatrix
        Ground-truth DAG adjacency matrix.
    n_nodes : int
        Number of nodes in the DAG.
    variable_types : tuple[VariableType, ...]
        Statistical type of each variable (indexed by node id).
    noise_type : str
        Name of the noise distribution (e.g. ``"gaussian"``, ``"uniform"``).
    noise_scale : float
        Scale parameter for the noise distribution.
    functional_form : str
        Functional form for structural equations (``"linear"``, ``"nonlinear"``).
    coefficient_range : tuple[float, float]
        ``(min, max)`` range for random edge coefficients.
    intervention_type : str
        Type of intervention applied (``"do"``, ``"soft"``, ``"shift"``).
    treatment : NodeId
        Treatment variable index.
    outcome : NodeId
        Outcome variable index.
    seed : int
        Random seed for reproducibility.
    extra : dict[str, Any]
        Free-form additional parameters for custom DGPs.
    """

    adjacency: AdjacencyMatrix
    n_nodes: int = 0
    variable_types: tuple[VariableType, ...] = ()
    noise_type: str = "gaussian"
    noise_scale: float = 1.0
    functional_form: str = "linear"
    coefficient_range: tuple[float, float] = (0.5, 2.0)
    intervention_type: str = "do"
    treatment: NodeId = 0
    outcome: NodeId = 1
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_nodes == 0:
            self.n_nodes = self.adjacency.shape[0]
        if not self.variable_types:
            self.variable_types = tuple(
                VariableType.CONTINUOUS for _ in range(self.n_nodes)
            )


# ---------------------------------------------------------------------------
# GroundTruth
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Ground-truth causal quantities from the data-generating process.

    These are the "oracle answers" known only because we control the DGP,
    used for benchmark evaluation.

    Attributes
    ----------
    true_ate : float
        True average treatment effect under the DGP.
    true_dag : AdjacencyMatrix
        The ground-truth DAG adjacency matrix.
    true_robustness_radius : int | None
        Exact minimum edit distance, if known analytically.
    confounders : NodeSet
        Set of true confounding variables between treatment and outcome.
    mediators : NodeSet
        Set of true mediating variables on causal paths.
    valid_adjustment_sets : tuple[NodeSet, ...]
        All minimal valid adjustment sets for the back-door criterion.
    true_causal_effect_fn : Callable[..., float] | None
        Exact structural equation for the causal effect, if available.
    """

    true_ate: float
    true_dag: AdjacencyMatrix
    true_robustness_radius: int | None = None
    confounders: NodeSet = frozenset()
    mediators: NodeSet = frozenset()
    valid_adjustment_sets: tuple[NodeSet, ...] = ()
    true_causal_effect_fn: Callable[..., float] | None = None


# ---------------------------------------------------------------------------
# SimulationResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Result of generating a synthetic dataset from a DGP.

    Bundles the sampled data with the DGP specification and ground truth
    so that downstream evaluation code has everything it needs.

    Attributes
    ----------
    data : NDArray[np.float64]
        Simulated dataset of shape ``(n_samples, n_nodes)``.
    n_samples : int
        Number of observations generated.
    dgp : DGPSpec
        The data-generating process that produced this dataset.
    ground_truth : GroundTruth
        Oracle causal quantities for evaluation.
    observational : NDArray[np.float64] | None
        Purely observational data (no interventions), if generated separately.
    interventional : NDArray[np.float64] | None
        Interventional data under ``do(treatment)``, if generated.
    seed_used : int
        Actual random seed used (may differ from ``dgp.seed`` if batched).
    """

    data: NDArray[np.float64]
    n_samples: int
    dgp: DGPSpec
    ground_truth: GroundTruth
    observational: NDArray[np.float64] | None = None
    interventional: NDArray[np.float64] | None = None
    seed_used: int = 0

    @property
    def n_nodes(self) -> int:
        """Number of variables in the simulated data."""
        return self.data.shape[1]
