"""Nonlinear Structural Causal Models for data generation.

Extends beyond the linear-Gaussian case to support polynomial, sigmoid,
tanh, quadratic, and additive-noise mechanisms with various noise
distributions.
"""
from __future__ import annotations

import enum
from typing import Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from causal_qd.core.dag import DAG
from causal_qd.types import DataMatrix


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MechanismType(enum.Enum):
    """Type of structural mechanism for a variable."""
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    QUADRATIC = "quadratic"
    ADDITIVE_NOISE = "additive_noise"


class NoiseType(enum.Enum):
    """Distribution family for exogenous noise."""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    LAPLACE = "laplace"
    STUDENT_T = "student_t"


# ---------------------------------------------------------------------------
# NonlinearSCM
# ---------------------------------------------------------------------------

class NonlinearSCM:
    """A nonlinear structural causal model.

    Each variable *Xj* is generated as::

        Xj = f_j(parents(Xj)) + εj

    where *f_j* is one of several nonlinear mechanism types and *εj* is
    drawn from the specified noise distribution.

    Parameters
    ----------
    dag : DAG
        Causal DAG.
    mechanisms : dict[int, MechanismType] or MechanismType
        Per-node mechanism type, or a single type applied to all nodes.
    noise_type : NoiseType
        Noise distribution family.
    noise_scale : float
        Scale parameter for the noise distribution.
    coefficient_range : tuple[float, float]
        ``(low, high)`` for the absolute value of mechanism coefficients.
    """

    def __init__(
        self,
        dag: DAG,
        mechanisms: Union[Dict[int, MechanismType], MechanismType] = MechanismType.POLYNOMIAL,
        noise_type: NoiseType = NoiseType.GAUSSIAN,
        noise_scale: float = 1.0,
        coefficient_range: Tuple[float, float] = (0.5, 2.0),
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._dag = dag
        self._noise_type = noise_type
        self._noise_scale = noise_scale
        self._coefficient_range = coefficient_range
        self._order = dag.topological_order
        n = dag.n_nodes

        # Resolve per-node mechanisms
        if isinstance(mechanisms, MechanismType):
            self._mechanisms: Dict[int, MechanismType] = {i: mechanisms for i in range(n)}
        else:
            self._mechanisms = dict(mechanisms)
            for i in range(n):
                if i not in self._mechanisms:
                    self._mechanisms[i] = MechanismType.POLYNOMIAL

        # Generate random coefficients for each node
        rng = rng or np.random.default_rng()
        self._coefficients: Dict[int, npt.NDArray[np.float64]] = {}
        self._poly_coefficients: Dict[int, npt.NDArray[np.float64]] = {}

        low, high = coefficient_range
        for node in range(n):
            parents = sorted(dag.parents(node))
            n_parents = len(parents)
            if n_parents == 0:
                self._coefficients[node] = np.array([], dtype=np.float64)
                self._poly_coefficients[node] = np.array([], dtype=np.float64)
                continue

            signs = rng.choice([-1.0, 1.0], size=n_parents)
            coeffs = rng.uniform(low, high, size=n_parents) * signs
            self._coefficients[node] = coeffs

            if self._mechanisms[node] == MechanismType.POLYNOMIAL:
                # 3 coefficients per parent (degree 1, 2, 3)
                poly_signs = rng.choice([-1.0, 1.0], size=(n_parents, 3))
                poly_coeffs = rng.uniform(low, high, size=(n_parents, 3)) * poly_signs
                self._poly_coefficients[node] = poly_coeffs

        self._ground_truth_adjacency = dag.adjacency

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dag(self) -> DAG:
        return self._dag

    @property
    def n_nodes(self) -> int:
        return self._dag.n_nodes

    @property
    def ground_truth_adjacency(self) -> npt.NDArray[np.int8]:
        return self._ground_truth_adjacency.copy()

    @property
    def mechanisms(self) -> Dict[int, MechanismType]:
        return dict(self._mechanisms)

    @property
    def noise_type(self) -> NoiseType:
        return self._noise_type

    @property
    def coefficients(self) -> Dict[int, npt.NDArray[np.float64]]:
        return {k: v.copy() for k, v in self._coefficients.items()}

    # ------------------------------------------------------------------
    # Noise generation
    # ------------------------------------------------------------------

    def _sample_noise(
        self, n_samples: int, rng: np.random.Generator
    ) -> npt.NDArray[np.float64]:
        """Draw noise for a single variable."""
        s = self._noise_scale
        if self._noise_type == NoiseType.GAUSSIAN:
            return rng.normal(0.0, s, size=n_samples)
        elif self._noise_type == NoiseType.UNIFORM:
            return rng.uniform(-s, s, size=n_samples)
        elif self._noise_type == NoiseType.LAPLACE:
            return rng.laplace(0.0, s, size=n_samples)
        elif self._noise_type == NoiseType.STUDENT_T:
            return rng.standard_t(df=3, size=n_samples) * s
        raise ValueError(f"Unknown noise type: {self._noise_type}")

    # ------------------------------------------------------------------
    # Mechanism evaluation
    # ------------------------------------------------------------------

    def _apply_mechanism(
        self,
        node: int,
        parent_values: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute f_node(parent_values).

        Parameters
        ----------
        node : int
            Target node.
        parent_values : ndarray, shape (n_samples, n_parents)
            Values of the parent variables.

        Returns
        -------
        ndarray, shape (n_samples,)
        """
        n_samples = parent_values.shape[0]
        n_parents = parent_values.shape[1] if parent_values.ndim == 2 else 0
        if n_parents == 0:
            return np.zeros(n_samples, dtype=np.float64)

        mech = self._mechanisms[node]
        coeffs = self._coefficients[node]

        if mech == MechanismType.LINEAR or mech == MechanismType.ADDITIVE_NOISE:
            return parent_values @ coeffs

        elif mech == MechanismType.SIGMOID:
            # sum of c_j * sigmoid(X_pa_j)
            sigmoid_vals = 1.0 / (1.0 + np.exp(-parent_values))
            return sigmoid_vals @ coeffs

        elif mech == MechanismType.TANH:
            return np.tanh(parent_values) @ coeffs

        elif mech == MechanismType.QUADRATIC:
            return (parent_values ** 2) @ coeffs

        elif mech == MechanismType.POLYNOMIAL:
            poly_coeffs = self._poly_coefficients[node]
            result = np.zeros(n_samples, dtype=np.float64)
            for j in range(n_parents):
                x = parent_values[:, j]
                result += poly_coeffs[j, 0] * x
                result += poly_coeffs[j, 1] * x ** 2
                result += poly_coeffs[j, 2] * x ** 3
            return result

        raise ValueError(f"Unknown mechanism: {mech}")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self, n_samples: int, rng: Optional[np.random.Generator] = None
    ) -> DataMatrix:
        """Draw *n_samples* i.i.d. observations from the SCM.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        rng : numpy.random.Generator or None
            Random number generator.

        Returns
        -------
        DataMatrix
            ``(n_samples, n_nodes)`` data matrix.
        """
        rng = rng or np.random.default_rng()
        p = self.n_nodes
        data = np.zeros((n_samples, p), dtype=np.float64)

        for node in self._order:
            parents = sorted(self._dag.parents(node))
            noise = self._sample_noise(n_samples, rng)
            if parents:
                parent_vals = data[:, parents]
                data[:, node] = self._apply_mechanism(node, parent_vals) + noise
            else:
                data[:, node] = noise

        return data

    # ------------------------------------------------------------------
    # Interventions
    # ------------------------------------------------------------------

    def intervene(
        self,
        targets: Dict[int, float],
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
    ) -> DataMatrix:
        """Sample under hard (do) interventions.

        Sets each target variable to its specified value, severing all
        incoming edges.

        Parameters
        ----------
        targets : dict[int, float]
            ``{node: value}`` pairs for hard interventions.
        n_samples : int
            Number of samples to draw.
        rng : numpy.random.Generator or None

        Returns
        -------
        DataMatrix
        """
        rng = rng or np.random.default_rng()
        p = self.n_nodes
        data = np.zeros((n_samples, p), dtype=np.float64)

        for node in self._order:
            if node in targets:
                data[:, node] = targets[node]
                continue
            parents = sorted(self._dag.parents(node))
            noise = self._sample_noise(n_samples, rng)
            if parents:
                parent_vals = data[:, parents]
                data[:, node] = self._apply_mechanism(node, parent_vals) + noise
            else:
                data[:, node] = noise

        return data

    def soft_intervene(
        self,
        targets: Dict[int, Tuple[float, float]],
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
    ) -> DataMatrix:
        """Sample under soft interventions (shift the mechanism).

        For each target node, the mechanism output is shifted by *shift*
        and the noise scale is replaced by *new_scale*.

        Parameters
        ----------
        targets : dict[int, tuple[float, float]]
            ``{node: (shift, new_noise_scale)}`` pairs.
        n_samples : int
        rng : numpy.random.Generator or None

        Returns
        -------
        DataMatrix
        """
        rng = rng or np.random.default_rng()
        p = self.n_nodes
        data = np.zeros((n_samples, p), dtype=np.float64)

        for node in self._order:
            parents = sorted(self._dag.parents(node))
            noise = self._sample_noise(n_samples, rng)

            if node in targets:
                shift, new_scale = targets[node]
                # Replace noise with the new scale
                if self._noise_type == NoiseType.GAUSSIAN:
                    noise = rng.normal(0.0, new_scale, size=n_samples)
                elif self._noise_type == NoiseType.UNIFORM:
                    noise = rng.uniform(-new_scale, new_scale, size=n_samples)
                elif self._noise_type == NoiseType.LAPLACE:
                    noise = rng.laplace(0.0, new_scale, size=n_samples)
                elif self._noise_type == NoiseType.STUDENT_T:
                    noise = rng.standard_t(df=3, size=n_samples) * new_scale

                if parents:
                    parent_vals = data[:, parents]
                    data[:, node] = self._apply_mechanism(node, parent_vals) + shift + noise
                else:
                    data[:, node] = shift + noise
            else:
                if parents:
                    parent_vals = data[:, parents]
                    data[:, node] = self._apply_mechanism(node, parent_vals) + noise
                else:
                    data[:, node] = noise

        return data

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_random(
        cls,
        n_nodes: int,
        edge_prob: float = 0.3,
        mechanism: MechanismType = MechanismType.POLYNOMIAL,
        noise_type: NoiseType = NoiseType.GAUSSIAN,
        noise_scale: float = 1.0,
        coefficient_range: Tuple[float, float] = (0.5, 2.0),
        rng: Optional[np.random.Generator] = None,
    ) -> "NonlinearSCM":
        """Create a random nonlinear SCM.

        Parameters
        ----------
        n_nodes : int
            Number of variables.
        edge_prob : float
            Probability of each forward edge.
        mechanism : MechanismType
            Mechanism type applied to all nodes.
        noise_type : NoiseType
            Noise distribution.
        noise_scale : float
            Scale of the noise.
        coefficient_range : tuple[float, float]
            Range for coefficient magnitudes.
        rng : numpy.random.Generator or None

        Returns
        -------
        NonlinearSCM
        """
        rng = rng or np.random.default_rng()
        dag = DAG.random_dag(n_nodes, edge_prob=edge_prob, rng=rng)
        return cls(
            dag=dag,
            mechanisms=mechanism,
            noise_type=noise_type,
            noise_scale=noise_scale,
            coefficient_range=coefficient_range,
            rng=rng,
        )
