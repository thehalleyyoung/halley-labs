"""Linear Gaussian Structural Causal Model."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt

from causal_qd.core.dag import DAG
from causal_qd.types import DataMatrix, WeightedAdjacencyMatrix


class LinearGaussianSCM:
    """A linear Gaussian structural causal model (SCM).

    Each variable *Xj* is generated as::

        Xj = Σ_i w[i,j] * Xi + εj,    εj ~ N(0, σj²)

    where the sum runs over the parents of *j* in the DAG.

    Parameters
    ----------
    dag : DAG
        Causal DAG.
    weights : WeightedAdjacencyMatrix
        Edge weight matrix (``weights[i, j]`` is the linear coefficient
        for parent *i* on child *j*).
    noise_std : npt.NDArray[np.float64]
        Per-variable noise standard deviations (length *n*).
    """

    def __init__(
        self,
        dag: DAG,
        weights: WeightedAdjacencyMatrix,
        noise_std: npt.NDArray[np.float64],
    ) -> None:
        self._dag = dag
        self._weights = np.array(weights, dtype=np.float64)
        self._noise_std = np.array(noise_std, dtype=np.float64)
        self._order = dag.topological_order

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dag(self) -> DAG:
        """The underlying DAG."""
        return self._dag

    @property
    def weights(self) -> WeightedAdjacencyMatrix:
        """Edge weight matrix."""
        return self._weights.copy()

    @property
    def noise_std(self) -> npt.NDArray[np.float64]:
        """Per-variable noise standard deviations."""
        return self._noise_std.copy()

    @property
    def n_nodes(self) -> int:
        """Number of variables."""
        return self._dag.n_nodes

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self, n: int, rng: np.random.Generator | None = None
    ) -> DataMatrix:
        """Draw *n* i.i.d. samples from the SCM.

        Parameters
        ----------
        n : int
            Number of samples.
        rng : numpy.random.Generator or None
            Random number generator.

        Returns
        -------
        DataMatrix
            ``(n, n_nodes)`` data matrix.
        """
        if rng is None:
            rng = np.random.default_rng()

        p = self.n_nodes
        data = np.zeros((n, p), dtype=np.float64)
        for node in self._order:
            noise = rng.normal(0.0, self._noise_std[node], size=n)
            parent_contrib = data @ self._weights[:, node]
            data[:, node] = parent_contrib + noise

        return data

    def intervene(self, node: int, value: float) -> LinearGaussianSCM:
        """Return a new SCM under the hard intervention ``do(X_node = value)``.

        The returned model has all incoming edges to *node* removed and
        the noise for that variable set to zero (the variable is fixed
        at *value*).

        Parameters
        ----------
        node : int
            Variable to intervene on.
        value : float
            Fixed value for the intervened variable.

        Returns
        -------
        LinearGaussianSCM
            A modified copy of the SCM.
        """
        new_adj = self._dag.adjacency.copy()
        new_adj[:, node] = 0  # sever incoming edges

        new_weights = self._weights.copy()
        new_weights[:, node] = 0.0

        new_noise = self._noise_std.copy()
        new_noise[node] = 0.0

        new_dag = DAG(new_adj)
        scm = LinearGaussianSCM(new_dag, new_weights, new_noise)
        # Monkey-patch the topological order sampling to inject the
        # intervention value at the right place.
        original_sample = scm.sample

        def _interventional_sample(
            n_: int, rng: np.random.Generator | None = None
        ) -> DataMatrix:
            data = original_sample(n_, rng)
            data[:, node] = value
            return data

        scm.sample = _interventional_sample  # type: ignore[assignment]
        return scm

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dag(
        cls,
        dag: DAG,
        weight_range: Tuple[float, float] = (0.25, 1.0),
        noise_std_range: Tuple[float, float] = (0.5, 1.5),
        rng: np.random.Generator | None = None,
    ) -> LinearGaussianSCM:
        """Construct a random linear Gaussian SCM from a DAG.

        Parameters
        ----------
        dag : DAG
        weight_range : tuple of float
            ``(low, high)`` for the absolute value of edge weights.
        noise_std_range : tuple of float
            ``(low, high)`` for noise standard deviations.
        rng : numpy.random.Generator or None

        Returns
        -------
        LinearGaussianSCM
        """
        if rng is None:
            rng = np.random.default_rng()

        n = dag.n_nodes
        adj = dag.adjacency.astype(np.float64)

        low, high = weight_range
        raw = rng.uniform(low, high, size=(n, n))
        signs = rng.choice([-1.0, 1.0], size=(n, n))
        weights = adj * raw * signs

        noise_low, noise_high = noise_std_range
        noise_std = rng.uniform(noise_low, noise_high, size=n)

        return cls(dag, weights, noise_std)
