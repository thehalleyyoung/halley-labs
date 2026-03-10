"""
Protocols for the simulation and data-generating process sub-system.

Defines structural sub-typing interfaces for data generators and noise
models, enabling pluggable simulation back-ends for benchmarking and
power analysis.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from causalcert.types import AdjacencyMatrix, NodeId, VariableType
from causalcert.simulation.types import DGPSpec, GroundTruth, SimulationResult


# ---------------------------------------------------------------------------
# NoiseModel
# ---------------------------------------------------------------------------


@runtime_checkable
class NoiseModel(Protocol):
    """Generates noise vectors for structural equation models.

    Each implementation encapsulates a family of noise distributions
    (Gaussian, uniform, Student-t, etc.) and can generate i.i.d. noise
    samples for every variable in the SCM.
    """

    @property
    def name(self) -> str:
        """Short identifier for this noise family (e.g. ``"gaussian"``)."""
        ...

    def sample(
        self,
        n_samples: int,
        n_variables: int,
        *,
        scale: float = 1.0,
        variable_types: tuple[VariableType, ...] = (),
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Draw i.i.d. noise samples.

        Parameters
        ----------
        n_samples : int
            Number of observations to generate.
        n_variables : int
            Number of variables (columns) in the noise matrix.
        scale : float, optional
            Scale parameter governing the spread of the noise.
        variable_types : tuple[VariableType, ...], optional
            Per-variable type hints for heterogeneous noise.  If empty,
            all variables are treated as continuous.
        rng : numpy.random.Generator | None, optional
            Random number generator for reproducibility.

        Returns
        -------
        NDArray[np.float64]
            Noise matrix of shape ``(n_samples, n_variables)``.
        """
        ...

    def log_density(
        self,
        noise: NDArray[np.float64],
        *,
        scale: float = 1.0,
    ) -> NDArray[np.float64]:
        """Evaluate the log-density of each noise observation.

        Parameters
        ----------
        noise : NDArray[np.float64]
            Noise matrix of shape ``(n_samples, n_variables)``.
        scale : float, optional
            Scale parameter used during generation.

        Returns
        -------
        NDArray[np.float64]
            Log-density values of shape ``(n_samples,)``.
        """
        ...


# ---------------------------------------------------------------------------
# DataGenerator
# ---------------------------------------------------------------------------


@runtime_checkable
class DataGenerator(Protocol):
    """Generates synthetic datasets from a data-generating process.

    Implementations take a :class:`DGPSpec` and produce a
    :class:`SimulationResult` containing both observational and
    (optionally) interventional data, together with ground-truth causal
    quantities.
    """

    def generate(
        self,
        dgp: DGPSpec,
        n_samples: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> SimulationResult:
        """Generate a synthetic dataset.

        Parameters
        ----------
        dgp : DGPSpec
            Full specification of the structural causal model.
        n_samples : int
            Number of observations to draw.
        rng : numpy.random.Generator | None, optional
            Random number generator.  If ``None``, a generator is created
            from ``dgp.seed``.

        Returns
        -------
        SimulationResult
            Simulated data bundled with ground truth.
        """
        ...

    def generate_interventional(
        self,
        dgp: DGPSpec,
        n_samples: int,
        intervention_value: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Generate interventional data under ``do(treatment = v)``.

        Parameters
        ----------
        dgp : DGPSpec
            Specification of the structural causal model.
        n_samples : int
            Number of interventional observations.
        intervention_value : float
            Value to set the treatment variable to.
        rng : numpy.random.Generator | None, optional
            Random number generator.

        Returns
        -------
        NDArray[np.float64]
            Interventional data of shape ``(n_samples, n_nodes)``.
        """
        ...

    def compute_ground_truth(
        self,
        dgp: DGPSpec,
    ) -> GroundTruth:
        """Compute analytical ground-truth causal quantities.

        Parameters
        ----------
        dgp : DGPSpec
            Specification of the structural causal model.

        Returns
        -------
        GroundTruth
            Oracle causal quantities derived analytically from the DGP.
        """
        ...
