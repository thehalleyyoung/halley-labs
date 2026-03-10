"""
usability_oracle.algebra.models — Core data models for the cost algebra.

Cost Element Tuple
------------------
A :class:`CostElement` is a four-dimensional descriptor of cognitive cost:

    ``(μ, σ², κ, λ)``

where:

* **μ** (mu) — expected cognitive cost (seconds or bits, depending on model)
* **σ²** (sigma_sq) — variance of the cost distribution
* **κ** (kappa) — skewness parameter controlling asymmetry of the cost tail
* **λ** (lambda_) — tail-risk parameter: probability mass beyond a critical
  threshold, capturing catastrophic failure modes

Cost Expression Tree
--------------------
A :class:`CostExpression` is a recursive algebraic expression tree whose
leaves are :class:`CostElement` instances and whose internal nodes are
composition operators (⊕, ⊗, Δ).

Evaluation proceeds bottom-up using the rules in :mod:`.sequential`,
:mod:`.parallel`, and :mod:`.context`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# CostElement — the fundamental four-tuple
# ---------------------------------------------------------------------------


@dataclass
class CostElement:
    r"""A cognitive cost tuple ``(μ, σ², κ, λ)``.

    Parameters
    ----------
    mu : float
        Mean (expected) cost.  Must be ≥ 0.
    sigma_sq : float
        Variance of the cost distribution.  Must be ≥ 0.
    kappa : float
        Skewness parameter.  Positive values indicate a right-skewed
        (long right tail) distribution — common for error-recovery costs.
    lambda_ : float
        Tail-risk parameter ∈ [0, 1].  Represents the probability mass
        beyond a catastrophic threshold.

    Interpretation
    --------------
    * A *degenerate* element has ``σ² = 0`` (deterministic cost).
    * The 95 % confidence interval of cost is approximately
      ``[μ - 1.96·σ, μ + 1.96·σ]`` for a Gaussian model, but
      ``κ`` and ``λ`` encode departures from Gaussianity.
    """

    mu: float = 0.0
    sigma_sq: float = 0.0
    kappa: float = 0.0
    lambda_: float = 0.0

    def __post_init__(self) -> None:
        self.mu = float(self.mu)
        self.sigma_sq = float(self.sigma_sq)
        self.kappa = float(self.kappa)
        self.lambda_ = float(self.lambda_)

    # -- arithmetic operators ------------------------------------------------

    def __add__(self, other: "CostElement") -> "CostElement":
        """Element-wise addition (a simple, coupling-free sum).

        For proper sequential composition with coupling, use
        :meth:`SequentialComposer.compose`.
        """
        if not isinstance(other, CostElement):
            return NotImplemented
        return CostElement(
            mu=self.mu + other.mu,
            sigma_sq=self.sigma_sq + other.sigma_sq,
            kappa=self._combine_kappa(other),
            lambda_=max(self.lambda_, other.lambda_),
        )

    def __mul__(self, scalar: float) -> "CostElement":
        """Scalar multiplication.

        .. math::

            c · (μ, σ², κ, λ) = (c·μ,  c²·σ²,  κ,  λ)

        Skewness and tail risk are scale-invariant.
        """
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        c = float(scalar)
        return CostElement(
            mu=c * self.mu,
            sigma_sq=c * c * self.sigma_sq,
            kappa=self.kappa,
            lambda_=self.lambda_,
        )

    def __rmul__(self, scalar: float) -> "CostElement":
        return self.__mul__(scalar)

    def __neg__(self) -> "CostElement":
        return CostElement(
            mu=-self.mu,
            sigma_sq=self.sigma_sq,
            kappa=-self.kappa,
            lambda_=self.lambda_,
        )

    def __sub__(self, other: "CostElement") -> "CostElement":
        if not isinstance(other, CostElement):
            return NotImplemented
        return self + (-other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CostElement):
            return NotImplemented
        return (
            math.isclose(self.mu, other.mu, abs_tol=1e-12)
            and math.isclose(self.sigma_sq, other.sigma_sq, abs_tol=1e-12)
            and math.isclose(self.kappa, other.kappa, abs_tol=1e-12)
            and math.isclose(self.lambda_, other.lambda_, abs_tol=1e-12)
        )

    def __hash__(self) -> int:
        return hash((round(self.mu, 10), round(self.sigma_sq, 10),
                      round(self.kappa, 10), round(self.lambda_, 10)))

    # -- derived quantities --------------------------------------------------

    def expected_cost(self) -> float:
        """Return the mean cost μ."""
        return self.mu

    def std_dev(self) -> float:
        """Return the standard deviation √σ²."""
        return math.sqrt(max(0.0, self.sigma_sq))

    def coefficient_of_variation(self) -> float:
        """Return CV = σ / μ.  Returns ``inf`` if μ = 0."""
        if self.mu == 0.0:
            return float("inf") if self.sigma_sq > 0 else 0.0
        return self.std_dev() / abs(self.mu)

    def to_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Compute a symmetric confidence interval for the cost.

        Uses a Cornish-Fisher expansion to account for skewness ``κ``::

            z_α = Φ⁻¹((1+confidence)/2)
            w = z_α + (z_α² - 1)·κ/6
            CI = [μ - w·σ,  μ + w·σ]

        Parameters
        ----------
        confidence : float
            Confidence level ∈ (0, 1).

        Returns
        -------
        (lower, upper) : tuple[float, float]
        """
        from scipy.stats import norm

        alpha = (1.0 + confidence) / 2.0
        z = float(norm.ppf(alpha))
        # Cornish-Fisher adjustment for skewness
        w = z + (z * z - 1.0) * self.kappa / 6.0
        sigma = self.std_dev()
        return (self.mu - w * sigma, self.mu + w * sigma)

    def tail_probability(self, threshold: float) -> float:
        """Estimate P(cost > threshold) using a shifted log-normal model.

        For elements with ``κ ≈ 0`` this reduces to Gaussian tail probability.
        """
        if self.sigma_sq <= 0:
            return 0.0 if self.mu <= threshold else 1.0

        sigma = self.std_dev()
        z = (threshold - self.mu) / sigma

        if abs(self.kappa) < 1e-8:
            # Gaussian
            from scipy.stats import norm
            return float(1.0 - norm.cdf(z))

        # Log-normal approximation for skewed distributions
        # Map (μ, σ², κ) to log-normal parameters
        try:
            cv = sigma / max(abs(self.mu), 1e-12)
            sigma_ln = math.sqrt(math.log(1 + cv * cv))
            mu_ln = math.log(max(self.mu, 1e-12)) - 0.5 * sigma_ln * sigma_ln
            if threshold <= 0:
                return 1.0
            from scipy.stats import lognorm
            return float(1.0 - lognorm.cdf(threshold, s=sigma_ln, scale=math.exp(mu_ln)))
        except (ValueError, ZeroDivisionError):
            from scipy.stats import norm
            return float(1.0 - norm.cdf(z))

    # -- properties ----------------------------------------------------------

    @property
    def is_degenerate(self) -> bool:
        """True if the cost element has zero variance (deterministic)."""
        return self.sigma_sq < 1e-15

    @property
    def is_valid(self) -> bool:
        """True if the cost element satisfies basic sanity constraints."""
        return (
            math.isfinite(self.mu)
            and math.isfinite(self.sigma_sq)
            and self.sigma_sq >= 0
            and math.isfinite(self.kappa)
            and math.isfinite(self.lambda_)
            and 0.0 <= self.lambda_ <= 1.0
        )

    @property
    def as_array(self) -> np.ndarray:
        """Return as a numpy array ``[μ, σ², κ, λ]``."""
        return np.array([self.mu, self.sigma_sq, self.kappa, self.lambda_])

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, float]:
        return {
            "mu": self.mu,
            "sigma_sq": self.sigma_sq,
            "kappa": self.kappa,
            "lambda": self.lambda_,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostElement":
        return cls(
            mu=float(data.get("mu", 0)),
            sigma_sq=float(data.get("sigma_sq", 0)),
            kappa=float(data.get("kappa", 0)),
            lambda_=float(data.get("lambda", data.get("lambda_", 0))),
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "CostElement":
        """Construct from a numpy array ``[μ, σ², κ, λ]``."""
        return cls(mu=float(arr[0]), sigma_sq=float(arr[1]),
                   kappa=float(arr[2]), lambda_=float(arr[3]))

    @classmethod
    def zero(cls) -> "CostElement":
        """The additive identity element."""
        return cls(mu=0.0, sigma_sq=0.0, kappa=0.0, lambda_=0.0)

    # -- internal helpers ----------------------------------------------------

    def _combine_kappa(self, other: "CostElement") -> float:
        """Combine skewness for independent addition.

        .. math::

            κ_{a+b} = \\frac{κ_a · σ_a^3 + κ_b · σ_b^3}{(σ_a^2 + σ_b^2)^{3/2}}

        """
        total_var = self.sigma_sq + other.sigma_sq
        if total_var < 1e-15:
            return 0.0
        num = (
            self.kappa * self.sigma_sq ** 1.5
            + other.kappa * other.sigma_sq ** 1.5
        )
        return num / (total_var ** 1.5)

    def __repr__(self) -> str:
        return (
            f"CostElement(μ={self.mu:.4f}, σ²={self.sigma_sq:.4f}, "
            f"κ={self.kappa:.4f}, λ={self.lambda_:.4f})"
        )


# ---------------------------------------------------------------------------
# CostExpression — algebraic expression tree
# ---------------------------------------------------------------------------


class CostExpression(ABC):
    """Abstract base class for cost algebra expression nodes.

    An expression tree whose leaves are :class:`CostElement` instances and
    internal nodes are composition operators (⊕, ⊗, Δ).  Call
    :meth:`evaluate` to recursively reduce the tree to a single
    :class:`CostElement`.
    """

    @abstractmethod
    def evaluate(self) -> CostElement:
        """Recursively evaluate this expression to a :class:`CostElement`."""
        ...

    @abstractmethod
    def children(self) -> List["CostExpression"]:
        """Return child sub-expressions."""
        ...

    @abstractmethod
    def depth(self) -> int:
        """Return the depth of this expression tree."""
        ...

    @abstractmethod
    def node_count(self) -> int:
        """Return the total number of nodes in the expression tree."""
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialise the expression tree."""
        ...


@dataclass
class Leaf(CostExpression):
    """A leaf node wrapping a single :class:`CostElement`."""

    element: CostElement

    def evaluate(self) -> CostElement:
        return self.element

    def children(self) -> List[CostExpression]:
        return []

    def depth(self) -> int:
        return 0

    def node_count(self) -> int:
        return 1

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "leaf", "element": self.element.to_dict()}

    def __repr__(self) -> str:
        return f"Leaf({self.element!r})"


@dataclass
class Sequential(CostExpression):
    """Sequential composition ⊕ of two sub-expressions.

    Semantics: the left expression completes before the right begins.
    """

    left: CostExpression
    right: CostExpression
    coupling: float = 0.0

    def evaluate(self) -> CostElement:
        from usability_oracle.algebra.sequential import SequentialComposer

        a = self.left.evaluate()
        b = self.right.evaluate()
        return SequentialComposer().compose(a, b, coupling=self.coupling)

    def children(self) -> List[CostExpression]:
        return [self.left, self.right]

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def node_count(self) -> int:
        return 1 + self.left.node_count() + self.right.node_count()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "sequential",
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "coupling": self.coupling,
        }

    def __repr__(self) -> str:
        return f"({self.left!r} ⊕ {self.right!r})"


@dataclass
class Parallel(CostExpression):
    """Parallel composition ⊗ of two sub-expressions.

    Semantics: both expressions execute concurrently; total cost is
    determined by the slower channel plus interference.
    """

    left: CostExpression
    right: CostExpression
    interference: float = 0.0

    def evaluate(self) -> CostElement:
        from usability_oracle.algebra.parallel import ParallelComposer

        a = self.left.evaluate()
        b = self.right.evaluate()
        return ParallelComposer().compose(a, b, interference=self.interference)

    def children(self) -> List[CostExpression]:
        return [self.left, self.right]

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def node_count(self) -> int:
        return 1 + self.left.node_count() + self.right.node_count()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "parallel",
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "interference": self.interference,
        }

    def __repr__(self) -> str:
        return f"({self.left!r} ⊗ {self.right!r})"


@dataclass
class ContextMod(CostExpression):
    """Context modulation Δ of an expression.

    Adjusts a cost expression by cognitive context factors such as fatigue,
    working-memory load, practice, and stress.
    """

    expr: CostExpression
    context: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self) -> CostElement:
        from usability_oracle.algebra.context import ContextModulator, CognitiveContext

        base = self.expr.evaluate()
        ctx = CognitiveContext(**self.context)
        return ContextModulator().modulate(base, ctx)

    def children(self) -> List[CostExpression]:
        return [self.expr]

    def depth(self) -> int:
        return 1 + self.expr.depth()

    def node_count(self) -> int:
        return 1 + self.expr.node_count()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "context_mod",
            "expr": self.expr.to_dict(),
            "context": dict(self.context),
        }

    def __repr__(self) -> str:
        return f"Δ({self.expr!r}, {self.context})"


# ---------------------------------------------------------------------------
# Expression tree deserialisation
# ---------------------------------------------------------------------------

def expression_from_dict(data: Dict[str, Any]) -> CostExpression:
    """Reconstruct a :class:`CostExpression` from its serialised form."""
    expr_type = data.get("type", "leaf")
    if expr_type == "leaf":
        return Leaf(CostElement.from_dict(data["element"]))
    elif expr_type == "sequential":
        return Sequential(
            left=expression_from_dict(data["left"]),
            right=expression_from_dict(data["right"]),
            coupling=data.get("coupling", 0.0),
        )
    elif expr_type == "parallel":
        return Parallel(
            left=expression_from_dict(data["left"]),
            right=expression_from_dict(data["right"]),
            interference=data.get("interference", 0.0),
        )
    elif expr_type == "context_mod":
        return ContextMod(
            expr=expression_from_dict(data["expr"]),
            context=data.get("context", {}),
        )
    else:
        raise ValueError(f"Unknown CostExpression type: {expr_type!r}")
