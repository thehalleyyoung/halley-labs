"""
Noise model implementations for structural equation models.

Provides pluggable noise generators that conform to the
:class:`~causalcert.simulation.protocols.NoiseModel` protocol.  Each model
supports sampling, log-density evaluation, and a factory function
:func:`create_noise` that instantiates models from short string specs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from causalcert.types import VariableType


# ---------------------------------------------------------------------------
# Base helpers
# ---------------------------------------------------------------------------


def _ensure_rng(rng: np.random.Generator | None) -> np.random.Generator:
    """Return *rng* or create a default generator."""
    return rng if rng is not None else np.random.default_rng()


# ---------------------------------------------------------------------------
# GaussianNoise
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GaussianNoise:
    r"""Isotropic Gaussian noise  :math:`\varepsilon \sim N(0, \sigma^2)`.

    Parameters
    ----------
    sigma : float
        Standard deviation (default 1.0).
    per_variable_sigma : tuple[float, ...] | None
        If given, each variable gets its own σ.  Length must match
        *n_variables* passed to :meth:`sample`.
    """

    sigma: float = 1.0
    per_variable_sigma: tuple[float, ...] | None = None

    @property
    def name(self) -> str:  # noqa: D401
        return "gaussian"

    def sample(
        self,
        n_samples: int,
        n_variables: int,
        *,
        scale: float = 1.0,
        variable_types: tuple[VariableType, ...] = (),
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        rng = _ensure_rng(rng)
        if self.per_variable_sigma is not None:
            sigmas = np.array(self.per_variable_sigma[:n_variables])
        else:
            sigmas = np.full(n_variables, self.sigma)
        return rng.normal(0.0, sigmas * scale, size=(n_samples, n_variables))

    def log_density(
        self,
        noise: NDArray[np.float64],
        *,
        scale: float = 1.0,
    ) -> NDArray[np.float64]:
        sigma = self.sigma * scale
        return np.sum(
            sp_stats.norm.logpdf(noise, 0.0, sigma), axis=1
        ).astype(np.float64)


# ---------------------------------------------------------------------------
# StudentTNoise
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StudentTNoise:
    r"""Heavy-tailed Student-*t* noise.

    Parameters
    ----------
    df : float
        Degrees of freedom (lower → heavier tails).
    sigma : float
        Scale parameter.
    """

    df: float = 5.0
    sigma: float = 1.0

    @property
    def name(self) -> str:  # noqa: D401
        return "student_t"

    def sample(
        self,
        n_samples: int,
        n_variables: int,
        *,
        scale: float = 1.0,
        variable_types: tuple[VariableType, ...] = (),
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        rng = _ensure_rng(rng)
        raw = rng.standard_t(self.df, size=(n_samples, n_variables))
        return raw * self.sigma * scale

    def log_density(
        self,
        noise: NDArray[np.float64],
        *,
        scale: float = 1.0,
    ) -> NDArray[np.float64]:
        sigma = self.sigma * scale
        return np.sum(
            sp_stats.t.logpdf(noise, self.df, 0.0, sigma), axis=1
        ).astype(np.float64)


# ---------------------------------------------------------------------------
# MixtureNoise
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MixtureNoise:
    """Gaussian mixture noise for multimodal error distributions.

    Parameters
    ----------
    means : tuple[float, ...]
        Component means.
    sigmas : tuple[float, ...]
        Component standard deviations.
    weights : tuple[float, ...] | None
        Mixing weights (uniform if ``None``).
    """

    means: tuple[float, ...] = (-1.0, 1.0)
    sigmas: tuple[float, ...] = (0.5, 0.5)
    weights: tuple[float, ...] | None = None

    @property
    def name(self) -> str:  # noqa: D401
        return "mixture"

    def _normalised_weights(self) -> NDArray[np.float64]:
        if self.weights is None:
            k = len(self.means)
            return np.full(k, 1.0 / k)
        w = np.asarray(self.weights, dtype=np.float64)
        return w / w.sum()

    def sample(
        self,
        n_samples: int,
        n_variables: int,
        *,
        scale: float = 1.0,
        variable_types: tuple[VariableType, ...] = (),
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        rng = _ensure_rng(rng)
        w = self._normalised_weights()
        k = len(self.means)
        out = np.empty((n_samples, n_variables), dtype=np.float64)
        for j in range(n_variables):
            comp = rng.choice(k, size=n_samples, p=w)
            for c in range(k):
                mask = comp == c
                n_c = int(mask.sum())
                if n_c > 0:
                    out[mask, j] = rng.normal(
                        self.means[c], self.sigmas[c] * scale, size=n_c
                    )
        return out

    def log_density(
        self,
        noise: NDArray[np.float64],
        *,
        scale: float = 1.0,
    ) -> NDArray[np.float64]:
        w = self._normalised_weights()
        n_samples, n_vars = noise.shape
        log_p = np.zeros(n_samples, dtype=np.float64)
        for j in range(n_vars):
            comp_log = np.stack(
                [
                    np.log(w[c]) + sp_stats.norm.logpdf(
                        noise[:, j], self.means[c], self.sigmas[c] * scale
                    )
                    for c in range(len(self.means))
                ],
                axis=0,
            )
            log_p += np.logaddexp.reduce(comp_log, axis=0)
        return log_p


# ---------------------------------------------------------------------------
# HeteroskedasticNoise
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HeteroskedasticNoise:
    r"""Noise whose variance depends on parent values.

    The standard deviation for observation *i* at variable *j* is

    .. math::
        \sigma_{ij} = \sigma_0 \cdot (1 + \gamma \,|\text{parent\_mean}_i|)

    Parameters
    ----------
    base_sigma : float
        Baseline standard deviation.
    gamma : float
        Strength of heteroskedasticity.
    """

    base_sigma: float = 1.0
    gamma: float = 0.5

    @property
    def name(self) -> str:  # noqa: D401
        return "heteroskedastic"

    def sample(
        self,
        n_samples: int,
        n_variables: int,
        *,
        scale: float = 1.0,
        variable_types: tuple[VariableType, ...] = (),
        rng: np.random.Generator | None = None,
        parent_values: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Sample noise; *parent_values* shape ``(n_samples,)`` or ``None``."""
        rng = _ensure_rng(rng)
        out = np.empty((n_samples, n_variables), dtype=np.float64)
        for j in range(n_variables):
            if parent_values is not None and parent_values.ndim == 2:
                pv = np.abs(parent_values[:, j]) if j < parent_values.shape[1] else 0.0
            elif parent_values is not None:
                pv = np.abs(parent_values)
            else:
                pv = 0.0
            sigma_ij = self.base_sigma * scale * (1.0 + self.gamma * pv)
            out[:, j] = rng.normal(0.0, sigma_ij)
        return out

    def log_density(
        self,
        noise: NDArray[np.float64],
        *,
        scale: float = 1.0,
    ) -> NDArray[np.float64]:
        sigma = self.base_sigma * scale
        return np.sum(
            sp_stats.norm.logpdf(noise, 0.0, sigma), axis=1
        ).astype(np.float64)


# ---------------------------------------------------------------------------
# NonAdditiveNoise
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class NonAdditiveNoise:
    r"""Multiplicative / interaction noise model.

    For each variable *j*, the noise enters multiplicatively:

    .. math::
        X_j = f_j(\text{pa}_j) \cdot (1 + \alpha\,\varepsilon_j) + \beta\,\varepsilon_j

    With ``alpha=1, beta=0`` the model is purely multiplicative;
    with ``alpha=0, beta=1`` it reduces to additive Gaussian.

    Parameters
    ----------
    sigma : float
        Raw noise standard deviation.
    alpha : float
        Multiplicative mixing coefficient.
    beta : float
        Additive mixing coefficient.
    """

    sigma: float = 1.0
    alpha: float = 0.5
    beta: float = 0.5

    @property
    def name(self) -> str:  # noqa: D401
        return "non_additive"

    def sample(
        self,
        n_samples: int,
        n_variables: int,
        *,
        scale: float = 1.0,
        variable_types: tuple[VariableType, ...] = (),
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        rng = _ensure_rng(rng)
        return rng.normal(0.0, self.sigma * scale, size=(n_samples, n_variables))

    def apply(
        self,
        signal: NDArray[np.float64],
        raw_noise: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Combine signal and noise non-additively.

        Parameters
        ----------
        signal : NDArray
            Deterministic part *f(pa)* of shape ``(n_samples,)``.
        raw_noise : NDArray
            Raw noise vector of shape ``(n_samples,)``.

        Returns
        -------
        NDArray
            Combined value.
        """
        return signal * (1.0 + self.alpha * raw_noise) + self.beta * raw_noise

    def log_density(
        self,
        noise: NDArray[np.float64],
        *,
        scale: float = 1.0,
    ) -> NDArray[np.float64]:
        sigma = self.sigma * scale
        return np.sum(
            sp_stats.norm.logpdf(noise, 0.0, sigma), axis=1
        ).astype(np.float64)


# ---------------------------------------------------------------------------
# DiscreteNoise
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DiscreteNoise:
    """Noise model for discrete / categorical variables.

    Parameters
    ----------
    n_categories : int
        Number of categories (default 2 for binary).
    base_probs : tuple[float, ...] | None
        Category probabilities.  Uniform if ``None``.
    temperature : float
        Softmax temperature applied to parent-driven logits.
    """

    n_categories: int = 2
    base_probs: tuple[float, ...] | None = None
    temperature: float = 1.0

    @property
    def name(self) -> str:  # noqa: D401
        return "discrete"

    def _probs(self) -> NDArray[np.float64]:
        if self.base_probs is not None:
            p = np.asarray(self.base_probs, dtype=np.float64)
        else:
            p = np.ones(self.n_categories, dtype=np.float64) / self.n_categories
        return p / p.sum()

    def sample(
        self,
        n_samples: int,
        n_variables: int,
        *,
        scale: float = 1.0,
        variable_types: tuple[VariableType, ...] = (),
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        rng = _ensure_rng(rng)
        p = self._probs()
        out = np.empty((n_samples, n_variables), dtype=np.float64)
        for j in range(n_variables):
            out[:, j] = rng.choice(self.n_categories, size=n_samples, p=p)
        return out

    def sample_from_logits(
        self,
        logits: NDArray[np.float64],
        *,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Sample categories given parent-driven logits.

        Parameters
        ----------
        logits : NDArray
            Shape ``(n_samples, n_categories)`` logit values.
        rng : Generator | None
            Random state.

        Returns
        -------
        NDArray
            Sampled category indices, shape ``(n_samples,)``.
        """
        rng = _ensure_rng(rng)
        scaled = logits / self.temperature
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= probs.sum(axis=1, keepdims=True)
        n = logits.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = rng.choice(self.n_categories, p=probs[i])
        return out

    def log_density(
        self,
        noise: NDArray[np.float64],
        *,
        scale: float = 1.0,
    ) -> NDArray[np.float64]:
        p = self._probs()
        n_samples = noise.shape[0]
        ld = np.zeros(n_samples, dtype=np.float64)
        for j in range(noise.shape[1]):
            cats = noise[:, j].astype(int)
            cats = np.clip(cats, 0, len(p) - 1)
            ld += np.log(p[cats] + 1e-300)
        return ld


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {
    "gaussian": GaussianNoise,
    "student_t": StudentTNoise,
    "t": StudentTNoise,
    "mixture": MixtureNoise,
    "heteroskedastic": HeteroskedasticNoise,
    "non_additive": NonAdditiveNoise,
    "multiplicative": NonAdditiveNoise,
    "discrete": DiscreteNoise,
}


def create_noise(spec: str, **kwargs: Any) -> (
    GaussianNoise | StudentTNoise | MixtureNoise
    | HeteroskedasticNoise | NonAdditiveNoise | DiscreteNoise
):
    """Instantiate a noise model from a short string identifier.

    Parameters
    ----------
    spec : str
        One of ``"gaussian"``, ``"student_t"`` / ``"t"``, ``"mixture"``,
        ``"heteroskedastic"``, ``"non_additive"`` / ``"multiplicative"``,
        ``"discrete"``.
    **kwargs
        Forwarded to the noise model constructor.

    Returns
    -------
    NoiseModel
        An instance satisfying the :class:`NoiseModel` protocol.

    Raises
    ------
    ValueError
        If *spec* is not recognised.
    """
    key = spec.strip().lower()
    if key not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown noise model {spec!r}; choose from {valid}")
    return _REGISTRY[key](**kwargs)
