"""Nonlinear structural causal model simulation.

Generates data from SCMs with configurable nonlinear mechanisms
(quadratic, sigmoid, GP via random Fourier features, additive noise,
post-nonlinear) and supports interventional and counterfactual queries.

Provides:

* :class:`MechanismType` – enum of supported mechanism families.
* :class:`MechanismFunction` – evaluable mechanism with parameters.
* :class:`NonlinearSCMGenerator` – full SCM simulator.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


# ===================================================================
# Enum
# ===================================================================


class MechanismType(Enum):
    """Supported mechanism function families."""

    LINEAR = "linear"
    QUADRATIC = "quadratic"
    SIGMOID = "sigmoid"
    GP = "gp"
    ADDITIVE_NOISE = "additive_noise"
    POST_NONLINEAR = "post_nonlinear"


# ===================================================================
# MechanismFunction
# ===================================================================


class MechanismFunction:
    """Evaluable mechanism function with stored parameters.

    Parameters
    ----------
    mechanism_type : MechanismType
        Type of mechanism.
    n_parents : int
        Number of parent variables.
    params : dict or None
        Pre-initialised parameters.  If ``None``, random parameters
        are drawn on first evaluation.
    seed : int or None
        Random seed for parameter initialisation.
    """

    def __init__(
        self,
        mechanism_type: MechanismType,
        n_parents: int,
        params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._type = mechanism_type
        self._n_parents = n_parents
        self._rng = np.random.default_rng(seed)
        self._params = params or self._init_params()

    def _init_params(self) -> Dict[str, Any]:
        """Draw random parameters for the mechanism."""
        d = max(self._n_parents, 1)
        if self._type == MechanismType.LINEAR:
            return {"weights": self._rng.standard_normal(d)}
        elif self._type == MechanismType.QUADRATIC:
            return {
                "weights_linear": self._rng.standard_normal(d),
                "weights_quad": self._rng.standard_normal(d) * 0.3,
                "interaction": self._rng.standard_normal(max(d * (d - 1) // 2, 1)) * 0.2,
            }
        elif self._type == MechanismType.SIGMOID:
            return {
                "weights": self._rng.standard_normal(d) * 2.0,
                "bias": self._rng.standard_normal() * 0.5,
                "scale": self._rng.uniform(1.0, 3.0),
            }
        elif self._type == MechanismType.GP:
            n_features = 50
            return {
                "omega": self._rng.standard_normal((d, n_features)),
                "bias": self._rng.uniform(0, 2 * np.pi, size=n_features),
                "beta": self._rng.standard_normal(n_features) / np.sqrt(n_features),
                "lengthscale": self._rng.uniform(0.5, 2.0),
            }
        elif self._type == MechanismType.ADDITIVE_NOISE:
            return {
                "weights": self._rng.standard_normal(d),
                "nonlinearity": "tanh",
            }
        elif self._type == MechanismType.POST_NONLINEAR:
            return {
                "weights": self._rng.standard_normal(d),
                "outer_scale": self._rng.uniform(0.5, 2.0),
                "outer_bias": self._rng.standard_normal() * 0.5,
            }
        return {"weights": self._rng.standard_normal(d)}

    def evaluate(self, parents_data: NDArray) -> NDArray:
        """Evaluate the mechanism on parent data.

        Parameters
        ----------
        parents_data : NDArray, shape (n, d)
            Parent variable values.  If the node has no parents,
            pass a column of zeros.

        Returns
        -------
        NDArray, shape (n,)
            Mechanism output (before noise).
        """
        X = np.atleast_2d(parents_data)
        n = X.shape[0]
        d = X.shape[1] if X.ndim > 1 else 1

        if self._type == MechanismType.LINEAR:
            w = self._params["weights"][:d]
            return X @ w

        elif self._type == MechanismType.QUADRATIC:
            w_lin = self._params["weights_linear"][:d]
            w_quad = self._params["weights_quad"][:d]
            linear = X @ w_lin
            quad = np.sum(X ** 2 * w_quad, axis=1)
            inter = self._params["interaction"]
            interaction = 0.0
            idx = 0
            for i in range(d):
                for j in range(i + 1, d):
                    if idx < len(inter):
                        interaction = interaction + inter[idx] * X[:, i] * X[:, j]
                        idx += 1
            return linear + quad + interaction

        elif self._type == MechanismType.SIGMOID:
            w = self._params["weights"][:d]
            b = self._params["bias"]
            s = self._params["scale"]
            z = X @ w + b
            return s * (1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))) - 0.5)

        elif self._type == MechanismType.GP:
            return self._rbf_random_features(X)

        elif self._type == MechanismType.ADDITIVE_NOISE:
            w = self._params["weights"][:d]
            nl = self._params.get("nonlinearity", "tanh")
            z = X @ w
            if nl == "tanh":
                return np.tanh(z)
            elif nl == "relu":
                return np.maximum(z, 0.0)
            return z

        elif self._type == MechanismType.POST_NONLINEAR:
            w = self._params["weights"][:d]
            s = self._params["outer_scale"]
            b = self._params["outer_bias"]
            inner = X @ w
            return s * np.tanh(inner + b)

        return np.zeros(n)

    def _rbf_random_features(self, X: NDArray) -> NDArray:
        """Approximate GP mechanism via random Fourier features.

        Parameters
        ----------
        X : NDArray, shape (n, d)

        Returns
        -------
        NDArray, shape (n,)
        """
        d = X.shape[1]
        omega = self._params["omega"][:d, :]
        bias = self._params["bias"]
        beta = self._params["beta"]
        ls = self._params["lengthscale"]

        Z = np.cos(X @ omega / ls + bias) * np.sqrt(2.0 / len(beta))
        return Z @ beta

    @staticmethod
    def _rbf_random_features_static(
        X: NDArray,
        n_features: int,
        rng: np.random.Generator,
    ) -> NDArray:
        """Generate random Fourier features for a GP approximation.

        Parameters
        ----------
        X : NDArray, shape (n, d)
        n_features : int
        rng : Generator

        Returns
        -------
        NDArray, shape (n, n_features)
        """
        d = X.shape[1]
        omega = rng.standard_normal((d, n_features))
        bias = rng.uniform(0, 2 * np.pi, size=n_features)
        return np.cos(X @ omega + bias) * np.sqrt(2.0 / n_features)


# ===================================================================
# NonlinearSCMGenerator
# ===================================================================


class NonlinearSCMGenerator:
    """Generate data from a nonlinear SCM.

    Parameters
    ----------
    dag : NDArray
        Adjacency matrix of shape ``(p, p)``.  ``dag[i,j] != 0``
        means i → j.
    mechanism_type : MechanismType
        Default mechanism type applied to all non-root nodes.
    noise_scale : float
        Standard deviation of additive Gaussian noise.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        dag: NDArray,
        mechanism_type: MechanismType = MechanismType.GP,
        noise_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self._dag = np.asarray(dag, dtype=np.float64)
        self._p = self._dag.shape[0]
        self._default_type = mechanism_type
        self._noise_scale = noise_scale
        self._rng = np.random.default_rng(seed)
        self._custom_mechanisms: Dict[int, Callable[..., NDArray]] = {}
        self._mechanism_fns: Dict[int, MechanismFunction] = {}
        self._noise_type = "gaussian"

        self._init_mechanisms()

    def _init_mechanisms(self) -> None:
        """Initialise mechanism functions for each node."""
        order = _topological_sort(self._dag)
        for j in order:
            parents = list(np.where(self._dag[:, j] != 0)[0])
            n_parents = len(parents)
            if n_parents == 0:
                continue
            self._mechanism_fns[j] = MechanismFunction(
                self._default_type,
                n_parents,
                seed=int(self._rng.integers(2**31)),
            )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def generate(self, n_samples: int, seed: Optional[int] = None) -> NDArray:
        """Sample observational data from the nonlinear SCM.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        seed : int or None
            Override random seed.

        Returns
        -------
        NDArray, shape (n_samples, p)
        """
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        return self._sample(n_samples, rng=rng)

    def set_mechanism(
        self,
        node: int,
        mechanism_fn: Callable[..., NDArray],
    ) -> None:
        """Override the mechanism function for a specific node.

        Parameters
        ----------
        node : int
        mechanism_fn : callable
            Signature: ``(parents_data: NDArray) -> NDArray``.
        """
        if node < 0 or node >= self._p:
            raise ValueError(f"Node index {node} out of range [0, {self._p})")
        self._custom_mechanisms[node] = mechanism_fn

    def intervene(
        self,
        targets: Dict[int, float],
        values: Optional[Dict[int, float]] = None,
        n_samples: int = 1000,
    ) -> NDArray:
        """Generate data under hard (do) intervention.

        Parameters
        ----------
        targets : dict
            Mapping from variable index to intervention value.
        values : dict or None
            Alias for *targets*.
        n_samples : int
            Number of samples.

        Returns
        -------
        NDArray, shape (n_samples, p)
        """
        interventions = dict(targets)
        if values is not None:
            interventions.update(values)
        return self._sample(n_samples, interventions=interventions, rng=self._rng)

    def counterfactual(
        self,
        evidence: NDArray,
        intervention: Dict[int, float],
        target: int,
    ) -> NDArray:
        """Counterfactual via abduction-action-prediction.

        Parameters
        ----------
        evidence : NDArray, shape (p,) or (n, p)
            Factual observation(s).
        intervention : dict
            do-values.
        target : int
            Variable whose counterfactual value is requested.

        Returns
        -------
        NDArray
            Counterfactual value(s).
        """
        if isinstance(evidence, dict):
            p = self._dag.shape[0]
            ev_array = np.zeros((1, p))
            for k, v in evidence.items():
                ev_array[0, k] = v
            evidence = ev_array
        evidence = np.atleast_2d(evidence)
        n = evidence.shape[0]

        # Step 1: Abduction – infer noise terms
        noises = self._abduct(evidence)

        # Step 2: Action – apply intervention
        # Step 3: Prediction – forward pass with intervened parents
        order = _topological_sort(self._dag)
        cf_data = np.zeros_like(evidence)

        for j in order:
            if j in intervention:
                cf_data[:, j] = intervention[j]
            else:
                parents = list(np.where(self._dag[:, j] != 0)[0])
                if len(parents) == 0:
                    cf_data[:, j] = noises[:, j]
                else:
                    parent_data = cf_data[:, parents]
                    if j in self._custom_mechanisms:
                        cm = self._custom_mechanisms[j]; mech_out = cm.evaluate(parent_data) if hasattr(cm, "evaluate") else cm(parent_data)
                    elif j in self._mechanism_fns:
                        mech_out = self._mechanism_fns[j].evaluate(parent_data)
                    else:
                        mech_out = np.zeros(n)
                    cf_data[:, j] = mech_out + noises[:, j]

        return cf_data[:, target]

    # -----------------------------------------------------------------
    # Internal sampling
    # -----------------------------------------------------------------

    def _sample(
        self,
        n: int,
        interventions: Optional[Dict[int, float]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray:
        """Core forward sampling."""
        rng = rng or self._rng
        order = _topological_sort(self._dag)
        data = np.zeros((n, self._p), dtype=np.float64)

        for j in order:
            noise = self._generate_noise(n, rng)

            if interventions and j in interventions:
                data[:, j] = interventions[j]
                continue

            parents = list(np.where(self._dag[:, j] != 0)[0])
            if len(parents) == 0:
                data[:, j] = noise
            else:
                parent_data = data[:, parents]
                if j in self._custom_mechanisms:
                    cm = self._custom_mechanisms[j]; mech_out = cm.evaluate(parent_data) if hasattr(cm, "evaluate") else cm(parent_data)
                elif j in self._mechanism_fns:
                    mech_out = self._mechanism_fns[j].evaluate(parent_data)
                else:
                    mech_out = np.zeros(n)
                data[:, j] = mech_out + noise

        return data

    def _generate_noise(self, n: int, rng: np.random.Generator) -> NDArray:
        """Generate noise samples."""
        if self._noise_type == "gaussian":
            return rng.normal(0, self._noise_scale, size=n)
        elif self._noise_type == "laplace":
            return rng.laplace(0, self._noise_scale / np.sqrt(2), size=n)
        elif self._noise_type == "uniform":
            half = self._noise_scale * np.sqrt(3)
            return rng.uniform(-half, half, size=n)
        return rng.normal(0, self._noise_scale, size=n)

    def _abduct(self, evidence: NDArray) -> NDArray:
        """Infer noise terms from factual evidence.

        Uses the structural equations in topological order to
        back-out the noise: U_j = X_j - f_j(Pa_j).
        """
        evidence = np.atleast_2d(evidence)
        n = evidence.shape[0]
        noises = np.zeros_like(evidence)
        order = _topological_sort(self._dag)

        for j in order:
            parents = list(np.where(self._dag[:, j] != 0)[0])
            if len(parents) == 0:
                noises[:, j] = evidence[:, j]
            else:
                parent_data = evidence[:, parents]
                if j in self._custom_mechanisms:
                    cm = self._custom_mechanisms[j]; mech_out = cm.evaluate(parent_data) if hasattr(cm, "evaluate") else cm(parent_data)
                elif j in self._mechanism_fns:
                    mech_out = self._mechanism_fns[j].evaluate(parent_data)
                else:
                    mech_out = np.zeros(n)
                noises[:, j] = evidence[:, j] - mech_out

        return noises


# ===================================================================
# Topological sort helper
# ===================================================================


def _topological_sort(adj: NDArray) -> List[int]:
    """Topological sort via Kahn's algorithm."""
    p = adj.shape[0]
    in_deg = np.sum(adj != 0, axis=0).astype(int)
    queue = [i for i in range(p) if in_deg[i] == 0]
    order: List[int] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for j in range(p):
            if adj[node, j] != 0:
                in_deg[j] -= 1
                if in_deg[j] == 0:
                    queue.append(j)
    if len(order) != p:
        raise ValueError("Graph contains a cycle")
    return order
