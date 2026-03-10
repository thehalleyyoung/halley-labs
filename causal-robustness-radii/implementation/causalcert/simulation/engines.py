"""
Data generation engines for structural causal models.

Each engine generates observational (and optionally interventional) data
from a DAG by constructing an SCM with configurable structural equations,
coefficient distributions, and noise models.

Engines
-------
- :class:`LinearGaussianEngine` — Linear SEM with Gaussian noise.
- :class:`NonlinearEngine` — Polynomial / sigmoid / sinusoidal SCMs.
- :class:`MixedTypeEngine` — Mixed continuous / discrete / ordinal data.
- :class:`InterventionalEngine` — Wraps any engine to apply do-interventions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from causalcert.types import AdjacencyMatrix, NodeId, VariableType
from causalcert.simulation.noise_models import (
    GaussianNoise,
    NonAdditiveNoise,
    DiscreteNoise,
    create_noise,
)


# ============================================================================
# Internal helpers
# ============================================================================


def _topological_order(adj: NDArray) -> list[int]:
    """Kahn's algorithm topological sort."""
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int).copy()
    queue = deque(i for i in range(n) if in_deg[i] == 0)
    order: list[int] = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in range(n):
            if adj[v, c]:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
    if len(order) != n:
        raise ValueError("Graph contains a cycle; cannot topologically sort.")
    return order


def _parents(adj: NDArray, v: int) -> list[int]:
    """Return parent indices of node *v*."""
    return [int(p) for p in np.nonzero(adj[:, v])[0]]


def _sample_coefficients(
    n_edges: int,
    coef_range: tuple[float, float],
    rng: np.random.Generator,
    *,
    sign_symmetric: bool = True,
) -> NDArray[np.float64]:
    """Sample edge coefficients uniformly, with optional random sign flips."""
    lo, hi = coef_range
    coefs = rng.uniform(lo, hi, size=n_edges)
    if sign_symmetric:
        signs = rng.choice([-1.0, 1.0], size=n_edges)
        coefs *= signs
    return coefs


def _standardize_columns(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Column-wise z-score standardization."""
    mu = data.mean(axis=0)
    sd = data.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (data - mu) / sd


# ============================================================================
# LinearGaussianEngine
# ============================================================================


@dataclass(slots=True)
class LinearGaussianEngine:
    r"""Linear Gaussian structural equation model.

    Each variable is generated as

    .. math::
        X_j = \sum_{i \in \text{pa}(j)} w_{ij}\,X_i
              + \text{intercept}_j + \varepsilon_j

    where :math:`\varepsilon_j \sim N(0, \sigma^2)`.

    Parameters
    ----------
    noise_scale : float
        Standard deviation of the Gaussian noise.
    coefficient_range : tuple[float, float]
        ``(min, max)`` absolute range for edge weights.
    sign_symmetric : bool
        If ``True``, signs are drawn uniformly from {-1, +1}.
    standardize : bool
        If ``True``, output columns are z-scored after generation.
    intercept_range : tuple[float, float]
        Range for intercepts drawn per variable.
    """

    noise_scale: float = 1.0
    coefficient_range: tuple[float, float] = (0.5, 2.0)
    sign_symmetric: bool = True
    standardize: bool = False
    intercept_range: tuple[float, float] = (0.0, 0.0)

    def generate(
        self,
        dag: AdjacencyMatrix,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """Generate an observational dataset.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Binary adjacency matrix defining the causal graph.
        n_samples : int
            Number of observations to draw.
        rng : Generator | None
            Random number generator.

        Returns
        -------
        pd.DataFrame
            Generated data with columns ``X0, X1, …``.
        """
        rng = rng if rng is not None else np.random.default_rng()
        adj = np.asarray(dag, dtype=np.int8)
        n = adj.shape[0]
        order = _topological_order(adj)

        # Sample structural parameters
        weights = self._build_weight_matrix(adj, rng)
        intercepts = rng.uniform(
            self.intercept_range[0], self.intercept_range[1], size=n
        )
        noise = GaussianNoise(sigma=self.noise_scale)
        eps = noise.sample(n_samples, n, rng=rng)

        data = np.zeros((n_samples, n), dtype=np.float64)
        for v in order:
            pa = _parents(adj, v)
            signal = intercepts[v] + eps[:, v]
            if pa:
                signal += data[:, pa] @ weights[pa, v]
            data[:, v] = signal

        if self.standardize:
            data = _standardize_columns(data)

        return pd.DataFrame(data, columns=[f"X{i}" for i in range(n)])

    def _build_weight_matrix(
        self, adj: NDArray, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Build the full weight matrix with sampled coefficients."""
        n = adj.shape[0]
        W = np.zeros((n, n), dtype=np.float64)
        edges = list(zip(*np.nonzero(adj)))
        if edges:
            coefs = _sample_coefficients(
                len(edges),
                self.coefficient_range,
                rng,
                sign_symmetric=self.sign_symmetric,
            )
            for (u, v), c in zip(edges, coefs):
                W[u, v] = c
        return W

    def get_weight_matrix(
        self,
        dag: AdjacencyMatrix,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Return the sampled weight matrix (useful for ground-truth ATE)."""
        rng = rng if rng is not None else np.random.default_rng()
        return self._build_weight_matrix(np.asarray(dag, dtype=np.int8), rng)

    def compute_total_effect(
        self,
        dag: AdjacencyMatrix,
        treatment: int,
        outcome: int,
        weights: NDArray[np.float64],
    ) -> float:
        """Compute the true ATE in a linear SEM via Wright's path method.

        Parameters
        ----------
        dag : AdjacencyMatrix
            The causal DAG.
        treatment, outcome : int
            Treatment and outcome node indices.
        weights : NDArray
            Weight matrix of shape ``(n, n)``.

        Returns
        -------
        float
            True total causal effect.
        """
        n = dag.shape[0]
        # Total effect = (I - W^T)^{-1}[treatment, outcome]
        W = weights.copy()
        M = np.eye(n) - W.T
        try:
            inv_M = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return 0.0
        return float(inv_M[treatment, outcome])


# ============================================================================
# NonlinearEngine
# ============================================================================

_NL_FORMS = ("polynomial", "sigmoid", "sinusoidal", "mixed")


@dataclass(slots=True)
class NonlinearEngine:
    """Nonlinear structural equation model.

    Supports polynomial, sigmoid, sinusoidal, or mixed functional forms
    per edge.

    Parameters
    ----------
    noise_scale : float
        Noise standard deviation.
    functional_form : str
        One of ``"polynomial"``, ``"sigmoid"``, ``"sinusoidal"``, ``"mixed"``.
    coefficient_range : tuple[float, float]
        Coefficient absolute range.
    polynomial_degree : int
        Maximum polynomial degree when *functional_form* is ``"polynomial"``
        or ``"mixed"``.
    standardize : bool
        Z-score output columns.
    """

    noise_scale: float = 1.0
    functional_form: str = "mixed"
    coefficient_range: tuple[float, float] = (0.5, 2.0)
    polynomial_degree: int = 3
    standardize: bool = False

    def __post_init__(self) -> None:
        if self.functional_form not in _NL_FORMS:
            raise ValueError(
                f"Unknown functional_form {self.functional_form!r}; "
                f"choose from {_NL_FORMS}"
            )

    # -- function factories ---------------------------------------------------

    @staticmethod
    def _poly_fn(
        degree: int, coef: float
    ) -> Callable[[NDArray], NDArray]:
        def fn(x: NDArray) -> NDArray:
            return coef * x ** degree
        return fn

    @staticmethod
    def _sigmoid_fn(coef: float) -> Callable[[NDArray], NDArray]:
        def fn(x: NDArray) -> NDArray:
            return coef * 2.0 / (1.0 + np.exp(-x)) - coef
        return fn

    @staticmethod
    def _sin_fn(coef: float) -> Callable[[NDArray], NDArray]:
        def fn(x: NDArray) -> NDArray:
            return coef * np.sin(x)
        return fn

    def _pick_fn(
        self, rng: np.random.Generator, coef: float
    ) -> Callable[[NDArray], NDArray]:
        form = self.functional_form
        if form == "mixed":
            form = rng.choice(["polynomial", "sigmoid", "sinusoidal"])
        if form == "polynomial":
            deg = rng.integers(2, self.polynomial_degree + 1)
            return self._poly_fn(int(deg), coef)
        if form == "sigmoid":
            return self._sigmoid_fn(coef)
        return self._sin_fn(coef)

    # -- generate -------------------------------------------------------------

    def generate(
        self,
        dag: AdjacencyMatrix,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """Generate data from a nonlinear SCM.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Binary DAG adjacency matrix.
        n_samples : int
            Number of observations.
        rng : Generator | None
            Random state.

        Returns
        -------
        pd.DataFrame
        """
        rng = rng if rng is not None else np.random.default_rng()
        adj = np.asarray(dag, dtype=np.int8)
        n = adj.shape[0]
        order = _topological_order(adj)

        # Pre-sample one function per edge
        edge_fns: dict[tuple[int, int], Callable] = {}
        for u, v in zip(*np.nonzero(adj)):
            u, v = int(u), int(v)
            coef = _sample_coefficients(
                1, self.coefficient_range, rng, sign_symmetric=True
            )[0]
            edge_fns[(u, v)] = self._pick_fn(rng, coef)

        noise = GaussianNoise(sigma=self.noise_scale)
        eps = noise.sample(n_samples, n, rng=rng)

        data = np.zeros((n_samples, n), dtype=np.float64)
        for v in order:
            pa = _parents(adj, v)
            value = eps[:, v].copy()
            for p in pa:
                fn = edge_fns[(p, v)]
                value += fn(data[:, p])
            data[:, v] = value

        if self.standardize:
            data = _standardize_columns(data)

        return pd.DataFrame(data, columns=[f"X{i}" for i in range(n)])


# ============================================================================
# MixedTypeEngine
# ============================================================================


@dataclass(slots=True)
class MixedTypeEngine:
    """Generate mixed continuous / discrete / ordinal data.

    For continuous variables, a linear SEM is used.  Discrete and ordinal
    variables are generated via a latent-threshold model: a latent
    continuous variable is computed from the SEM and then discretized.

    Parameters
    ----------
    variable_types : tuple[VariableType, ...]
        Type of each variable.  Must match the number of DAG nodes.
    noise_scale : float
        Noise standard deviation for continuous components.
    coefficient_range : tuple[float, float]
        Coefficient range.
    n_categories : int
        Default number of categories for discrete / ordinal variables.
    standardize : bool
        Z-score continuous columns.
    """

    variable_types: tuple[VariableType, ...] = ()
    noise_scale: float = 1.0
    coefficient_range: tuple[float, float] = (0.5, 2.0)
    n_categories: int = 4
    standardize: bool = False

    def generate(
        self,
        dag: AdjacencyMatrix,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """Generate a mixed-type dataset.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Binary DAG adjacency matrix.
        n_samples : int
            Number of observations.
        rng : Generator | None
            Random state.

        Returns
        -------
        pd.DataFrame
        """
        rng = rng if rng is not None else np.random.default_rng()
        adj = np.asarray(dag, dtype=np.int8)
        n = adj.shape[0]
        order = _topological_order(adj)

        vtypes = self.variable_types
        if not vtypes:
            vtypes = tuple(VariableType.CONTINUOUS for _ in range(n))
        if len(vtypes) != n:
            raise ValueError(
                f"variable_types length {len(vtypes)} != DAG size {n}"
            )

        # Build weight matrix
        lin_engine = LinearGaussianEngine(
            noise_scale=self.noise_scale,
            coefficient_range=self.coefficient_range,
        )
        W = lin_engine._build_weight_matrix(adj, rng)
        eps = GaussianNoise(sigma=self.noise_scale).sample(n_samples, n, rng=rng)

        # Generate latent continuous values
        latent = np.zeros((n_samples, n), dtype=np.float64)
        for v in order:
            pa = _parents(adj, v)
            val = eps[:, v].copy()
            if pa:
                val += latent[:, pa] @ W[pa, v]
            latent[:, v] = val

        # Discretize selected columns
        data = latent.copy()
        for j in range(n):
            if vtypes[j] == VariableType.BINARY:
                data[:, j] = (latent[:, j] > 0).astype(np.float64)
            elif vtypes[j] in (VariableType.ORDINAL, VariableType.NOMINAL):
                k = self.n_categories
                quantiles = np.linspace(0, 1, k + 1)[1:-1]
                thresholds = np.quantile(latent[:, j], quantiles)
                data[:, j] = np.digitize(latent[:, j], thresholds).astype(
                    np.float64
                )

        if self.standardize:
            cont_mask = [
                j for j in range(n) if vtypes[j] == VariableType.CONTINUOUS
            ]
            if cont_mask:
                mu = data[:, cont_mask].mean(axis=0)
                sd = data[:, cont_mask].std(axis=0)
                sd[sd < 1e-12] = 1.0
                data[:, cont_mask] = (data[:, cont_mask] - mu) / sd

        return pd.DataFrame(data, columns=[f"X{i}" for i in range(n)])


# ============================================================================
# InterventionalEngine
# ============================================================================


@dataclass(slots=True)
class InterventionalEngine:
    """Wrapper that generates data under do-interventions.

    Combines any base engine with a hard or soft intervention on a
    specified treatment variable.

    Parameters
    ----------
    base_engine : LinearGaussianEngine | NonlinearEngine | MixedTypeEngine
        The underlying SCM engine.
    noise_scale : float
        Noise scale (forwarded if the base engine needs it).
    """

    base_engine: Any = field(default_factory=LinearGaussianEngine)
    noise_scale: float = 1.0

    def generate(
        self,
        dag: AdjacencyMatrix,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """Generate observational data (delegates to base engine)."""
        return self.base_engine.generate(dag, n_samples, rng=rng)

    def generate_do(
        self,
        dag: AdjacencyMatrix,
        n_samples: int,
        treatment: int,
        value: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """Generate data under ``do(treatment = value)``.

        Severs all incoming edges to *treatment* and fixes its value.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Original DAG.
        n_samples : int
            Number of observations.
        treatment : int
            Index of the intervened variable.
        value : float
            Fixed intervention value.
        rng : Generator | None
            Random state.

        Returns
        -------
        pd.DataFrame
        """
        rng = rng if rng is not None else np.random.default_rng()
        adj = np.asarray(dag, dtype=np.int8).copy()
        # Sever incoming edges (do-intervention)
        adj[:, treatment] = 0
        df = self.base_engine.generate(adj, n_samples, rng=rng)
        df.iloc[:, treatment] = value
        # Re-propagate descendants
        return self._repropagate(dag, df, treatment, value, rng)

    def generate_shift(
        self,
        dag: AdjacencyMatrix,
        n_samples: int,
        treatment: int,
        shift: float,
        *,
        rng: np.random.Generator | None = None,
    ) -> pd.DataFrame:
        """Generate data under a soft shift intervention.

        Parameters
        ----------
        dag : AdjacencyMatrix
            Original DAG.
        n_samples : int
            Number of observations.
        treatment : int
            Index of the shifted variable.
        shift : float
            Additive shift applied to the treatment.
        rng : Generator | None
            Random state.

        Returns
        -------
        pd.DataFrame
        """
        rng = rng if rng is not None else np.random.default_rng()
        df = self.base_engine.generate(dag, n_samples, rng=rng)
        df.iloc[:, treatment] = df.iloc[:, treatment].values + shift
        return df

    def compute_ate(
        self,
        dag: AdjacencyMatrix,
        outcome: int,
        treatment: int,
        treat_value: float = 1.0,
        control_value: float = 0.0,
        n_samples: int = 50_000,
        rng: np.random.Generator | None = None,
    ) -> float:
        """Estimate the ATE via Monte Carlo simulation.

        Parameters
        ----------
        dag : AdjacencyMatrix
            The causal DAG.
        outcome : int
            Outcome variable index.
        treatment : int
            Treatment variable index.
        treat_value : float
            Treatment group value.
        control_value : float
            Control group value.
        n_samples : int
            Monte Carlo samples.
        rng : Generator | None
            Random state.

        Returns
        -------
        float
            Estimated ATE.
        """
        rng = rng if rng is not None else np.random.default_rng()
        rng1, rng2 = rng.spawn(2)
        df1 = self.generate_do(dag, n_samples, treatment, treat_value, rng=rng1)
        df0 = self.generate_do(dag, n_samples, treatment, control_value, rng=rng2)
        return float(df1.iloc[:, outcome].mean() - df0.iloc[:, outcome].mean())

    # -- internal helpers -----------------------------------------------------

    def _repropagate(
        self,
        dag: AdjacencyMatrix,
        df: pd.DataFrame,
        treatment: int,
        value: float,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Re-generate values for descendants of *treatment*."""
        adj = np.asarray(dag, dtype=np.int8)
        n = adj.shape[0]
        order = _topological_order(adj)
        data = df.values.copy()
        data[:, treatment] = value

        # Only propagate from treatment onward
        t_pos = order.index(treatment)
        # Build a simple linear weight matrix for re-propagation
        if isinstance(self.base_engine, LinearGaussianEngine):
            W = self.base_engine._build_weight_matrix(adj, rng)
            eps = GaussianNoise(sigma=self.base_engine.noise_scale).sample(
                data.shape[0], n, rng=rng
            )
            for v in order[t_pos + 1:]:
                pa = _parents(adj, v)
                if treatment not in pa and not _is_descendant(adj, treatment, v):
                    continue
                data[:, v] = eps[:, v]
                if pa:
                    data[:, v] += data[:, pa] @ W[pa, v]

        return pd.DataFrame(data, columns=df.columns)


def _is_descendant(adj: NDArray, u: int, v: int) -> bool:
    """Check if *v* is a descendant of *u* in *adj*."""
    visited: set[int] = set()
    queue = deque([u])
    while queue:
        node = queue.popleft()
        for c in np.nonzero(adj[node])[0]:
            c = int(c)
            if c == v:
                return True
            if c not in visited:
                visited.add(c)
                queue.append(c)
    return False
