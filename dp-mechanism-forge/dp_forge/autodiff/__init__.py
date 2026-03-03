"""
Automatic differentiation for privacy loss functions.

This package provides forward-mode and reverse-mode automatic differentiation
tailored to differential privacy loss computation. It enables gradient-based
optimisation of mechanism parameters by computing exact derivatives of
privacy loss functions (hockey-stick divergence, Rényi divergence, etc.)
with respect to mechanism probabilities.

Architecture:
    1. **DualNumber** — Forward-mode AD primitive carrying value + derivative.
    2. **TapeNode / ComputationTape** — Reverse-mode AD graph for efficient
       backpropagation through privacy loss computations.
    3. **PrivacyLossGrad** — Specialised gradient computation for standard
       privacy loss functions (hockey-stick, Rényi, KL, max-divergence).
    4. **MechanismOptimizer** — Gradient-based mechanism optimisation using
       projected gradient descent on the probability simplex.
    5. **HessianComputer** — Second-order information for Newton-type methods.

Example::

    from dp_forge.autodiff import PrivacyLossGrad, DiffMode

    grad_computer = PrivacyLossGrad(mode=DiffMode.REVERSE)
    grad_info = grad_computer.hockey_stick_gradient(
        mechanism=mech, epsilon=1.0, adjacent_pair=(0, 1)
    )
    print(f"Loss: {grad_info.value}, |gradient|: {grad_info.gradient_norm}")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

from dp_forge.types import (
    GradientInfo,
    PrivacyBudget,
)
from dp_forge.autodiff.dual_numbers import DualNumber


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DiffMode(Enum):
    """Automatic differentiation mode."""

    FORWARD = auto()
    REVERSE = auto()
    SYMBOLIC = auto()

    def __repr__(self) -> str:
        return f"DiffMode.{self.name}"


class DivergenceType(Enum):
    """Types of divergence functions that can be differentiated."""

    HOCKEY_STICK = auto()
    RENYI = auto()
    KL = auto()
    MAX_DIVERGENCE = auto()
    TOTAL_VARIATION = auto()
    CHI_SQUARED = auto()

    def __repr__(self) -> str:
        return f"DivergenceType.{self.name}"


class OpType(Enum):
    """Operations tracked in the computation tape."""

    ADD = auto()
    MUL = auto()
    DIV = auto()
    LOG = auto()
    EXP = auto()
    POW = auto()
    ABS = auto()
    MAX = auto()
    SUM = auto()
    NEG = auto()

    def __repr__(self) -> str:
        return f"OpType.{self.name}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AutodiffConfig:
    """Configuration for automatic differentiation.

    Attributes:
        mode: Differentiation mode (forward, reverse, symbolic).
        compute_hessian: Whether to compute second-order derivatives.
        numerical_eps: Perturbation size for numerical gradient checks.
        gradient_clip: Maximum gradient norm (None for no clipping).
        dtype: Numpy dtype for computations.
    """

    mode: DiffMode = DiffMode.REVERSE
    compute_hessian: bool = False
    numerical_eps: float = 1e-7
    gradient_clip: Optional[float] = None
    dtype: np.dtype = np.float64

    def __post_init__(self) -> None:
        if self.numerical_eps <= 0:
            raise ValueError(f"numerical_eps must be > 0, got {self.numerical_eps}")
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError(f"gradient_clip must be > 0, got {self.gradient_clip}")

    def __repr__(self) -> str:
        return f"AutodiffConfig(mode={self.mode.name}, hessian={self.compute_hessian})"


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class TapeNode:
    """A node in the reverse-mode computation tape.

    Attributes:
        node_id: Unique identifier for this node.
        op: The operation that produced this node.
        parents: Indices of parent nodes in the tape.
        value: The computed value at this node.
        local_gradients: Partial derivatives w.r.t. each parent.
    """

    node_id: int
    op: OpType
    parents: Tuple[int, ...]
    value: float
    local_gradients: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.parents) != len(self.local_gradients):
            raise ValueError(
                f"parents ({len(self.parents)}) and local_gradients "
                f"({len(self.local_gradients)}) must have same length"
            )

    def __repr__(self) -> str:
        return f"TapeNode(id={self.node_id}, op={self.op.name}, val={self.value:.4f})"


@dataclass
class ComputationTape:
    """Reverse-mode computation tape for backpropagation.

    Attributes:
        nodes: List of TapeNode entries in topological order.
        input_indices: Indices of input (leaf) nodes.
        output_index: Index of the output (loss) node.
    """

    nodes: List[TapeNode] = field(default_factory=list)
    input_indices: List[int] = field(default_factory=list)
    output_index: int = -1

    @property
    def size(self) -> int:
        """Number of nodes in the tape."""
        return len(self.nodes)

    def backward(self) -> npt.NDArray[np.float64]:
        """Run reverse-mode AD to compute gradients w.r.t. inputs.

        Returns:
            Gradient array with one entry per input node.
        """
        if self.output_index < 0:
            raise ValueError("output_index not set")
        n = len(self.nodes)
        adjoints = np.zeros(n, dtype=np.float64)
        adjoints[self.output_index] = 1.0

        for node in reversed(self.nodes):
            adj = adjoints[node.node_id]
            for parent_id, local_grad in zip(node.parents, node.local_gradients):
                adjoints[parent_id] += adj * local_grad

        return np.array(
            [adjoints[i] for i in self.input_indices], dtype=np.float64
        )

    def __repr__(self) -> str:
        return f"ComputationTape(nodes={self.size}, inputs={len(self.input_indices)})"


@dataclass
class DivergenceGradient:
    """Gradient of a specific divergence function.

    Attributes:
        divergence_type: Which divergence was computed.
        info: The gradient information (value, gradient, optional hessian).
        adjacent_pair: The (i, i') pair this gradient is computed for.
        alpha: Rényi order (only for Rényi divergence).
    """

    divergence_type: DivergenceType
    info: GradientInfo
    adjacent_pair: Tuple[int, int]
    alpha: Optional[float] = None

    def __repr__(self) -> str:
        return (
            f"DivergenceGradient(type={self.divergence_type.name}, "
            f"pair={self.adjacent_pair}, value={self.info.value:.6f})"
        )


# ---------------------------------------------------------------------------
# Protocols (interfaces)
# ---------------------------------------------------------------------------


@runtime_checkable
class Differentiable(Protocol):
    """Protocol for functions that support automatic differentiation."""

    def evaluate(self, x: npt.NDArray[np.float64]) -> float:
        """Evaluate the function at x."""
        ...

    def gradient(self, x: npt.NDArray[np.float64]) -> GradientInfo:
        """Compute the gradient at x."""
        ...


@runtime_checkable
class PrivacyLossFunction(Protocol):
    """Protocol for differentiable privacy loss functions."""

    def loss(
        self,
        mechanism: npt.NDArray[np.float64],
        i: int,
        i_prime: int,
    ) -> float:
        """Compute privacy loss for adjacent pair (i, i')."""
        ...

    def loss_gradient(
        self,
        mechanism: npt.NDArray[np.float64],
        i: int,
        i_prime: int,
    ) -> GradientInfo:
        """Compute gradient of privacy loss w.r.t. mechanism probabilities."""
        ...


# ---------------------------------------------------------------------------
# Public API classes
# ---------------------------------------------------------------------------


class PrivacyLossGrad:
    """Compute gradients of privacy loss functions via automatic differentiation.

    Supports hockey-stick divergence, Rényi divergence, KL divergence,
    and max-divergence with both forward and reverse mode AD.
    """

    def __init__(self, config: Optional[AutodiffConfig] = None) -> None:
        self.config = config or AutodiffConfig()

    def hockey_stick_gradient(
        self,
        mechanism: npt.NDArray[np.float64],
        epsilon: float,
        adjacent_pair: Tuple[int, int],
    ) -> DivergenceGradient:
        """Compute gradient of hockey-stick divergence.

        Args:
            mechanism: The n × k probability table.
            epsilon: Privacy parameter ε.
            adjacent_pair: The (i, i') pair.

        Returns:
            DivergenceGradient with value and gradient.
        """
        from dp_forge.autodiff.privacy_functions import hockey_stick_gradient as _hsg

        i, ip = adjacent_pair
        p, q = mechanism[i], mechanism[ip]
        val, gp, gq = _hsg(p, q, epsilon)
        n, k = mechanism.shape
        grad = np.zeros(n * k, dtype=np.float64)
        grad[i * k:(i + 1) * k] = gp
        grad[ip * k:(ip + 1) * k] = gq
        if self.config.gradient_clip is not None:
            norm = float(np.linalg.norm(grad))
            if norm > self.config.gradient_clip:
                grad *= self.config.gradient_clip / norm
        info = GradientInfo(value=val, gradient=grad)
        return DivergenceGradient(DivergenceType.HOCKEY_STICK, info, adjacent_pair)

    def renyi_gradient(
        self,
        mechanism: npt.NDArray[np.float64],
        alpha: float,
        adjacent_pair: Tuple[int, int],
    ) -> DivergenceGradient:
        """Compute gradient of Rényi divergence of order α.

        Args:
            mechanism: The n × k probability table.
            alpha: Rényi order α > 1.
            adjacent_pair: The (i, i') pair.

        Returns:
            DivergenceGradient with value and gradient.
        """
        from dp_forge.autodiff.privacy_functions import renyi_divergence_gradient as _rdg

        i, ip = adjacent_pair
        p, q = mechanism[i], mechanism[ip]
        val, gp, gq = _rdg(p, q, alpha)
        n, k = mechanism.shape
        grad = np.zeros(n * k, dtype=np.float64)
        grad[i * k:(i + 1) * k] = gp
        grad[ip * k:(ip + 1) * k] = gq
        if self.config.gradient_clip is not None:
            norm = float(np.linalg.norm(grad))
            if norm > self.config.gradient_clip:
                grad *= self.config.gradient_clip / norm
        info = GradientInfo(value=val, gradient=grad)
        return DivergenceGradient(DivergenceType.RENYI, info, adjacent_pair, alpha=alpha)

    def kl_gradient(
        self,
        mechanism: npt.NDArray[np.float64],
        adjacent_pair: Tuple[int, int],
    ) -> DivergenceGradient:
        """Compute gradient of KL divergence.

        Args:
            mechanism: The n × k probability table.
            adjacent_pair: The (i, i') pair.

        Returns:
            DivergenceGradient with value and gradient.
        """
        from dp_forge.autodiff.privacy_functions import kl_divergence_gradient as _klg

        i, ip = adjacent_pair
        p, q = mechanism[i], mechanism[ip]
        val, gp, gq = _klg(p, q)
        n, k = mechanism.shape
        grad = np.zeros(n * k, dtype=np.float64)
        grad[i * k:(i + 1) * k] = gp
        grad[ip * k:(ip + 1) * k] = gq
        if self.config.gradient_clip is not None:
            norm = float(np.linalg.norm(grad))
            if norm > self.config.gradient_clip:
                grad *= self.config.gradient_clip / norm
        info = GradientInfo(value=val, gradient=grad)
        return DivergenceGradient(DivergenceType.KL, info, adjacent_pair)

    def full_gradient(
        self,
        mechanism: npt.NDArray[np.float64],
        budget: PrivacyBudget,
        adjacent_pairs: List[Tuple[int, int]],
    ) -> GradientInfo:
        """Compute gradient of worst-case privacy loss over all pairs.

        Args:
            mechanism: The n × k probability table.
            budget: Privacy budget.
            adjacent_pairs: List of adjacent pairs to consider.

        Returns:
            GradientInfo aggregated over all pairs.
        """
        worst_val = -np.inf
        worst_grad = np.zeros(mechanism.size, dtype=np.float64)
        for pair in adjacent_pairs:
            dg = self.hockey_stick_gradient(mechanism, budget.epsilon, pair)
            if dg.info.value > worst_val:
                worst_val = dg.info.value
                worst_grad = dg.info.gradient.copy()
        return GradientInfo(value=float(worst_val), gradient=worst_grad)


class MechanismOptimizer:
    """Gradient-based mechanism optimisation on the probability simplex.

    Uses projected gradient descent with privacy constraints to optimise
    mechanism utility while maintaining DP guarantees.
    """

    def __init__(
        self,
        config: Optional[AutodiffConfig] = None,
        *,
        learning_rate: float = 0.01,
        max_steps: int = 1000,
        convergence_tol: float = 1e-8,
    ) -> None:
        self.config = config or AutodiffConfig()
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.convergence_tol = convergence_tol

    def optimize(
        self,
        initial_mechanism: npt.NDArray[np.float64],
        budget: PrivacyBudget,
        loss_fn: Callable[[npt.NDArray[np.float64]], float],
        adjacent_pairs: List[Tuple[int, int]],
    ) -> Tuple[npt.NDArray[np.float64], List[float]]:
        """Optimise a mechanism using projected gradient descent.

        Args:
            initial_mechanism: Starting n × k probability table.
            budget: Privacy budget constraint.
            loss_fn: Utility loss function to minimise.
            adjacent_pairs: Adjacent pairs defining DP constraints.

        Returns:
            Tuple of (optimised_mechanism, loss_history).
        """
        from dp_forge.autodiff.optimizer import (
            ProjectedGradientDescent,
            project_simplex_rows,
        )

        def obj(M: npt.NDArray[np.float64]) -> float:
            return loss_fn(M)

        def grad(M: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            n, k = M.shape
            g = np.zeros_like(M)
            h = 1e-7
            for i in range(n):
                for j in range(k):
                    Mp = M.copy()
                    Mm = M.copy()
                    Mp[i, j] += h
                    Mm[i, j] -= h
                    g[i, j] = (obj(Mp) - obj(Mm)) / (2 * h)
            return g

        pgd = ProjectedGradientDescent(
            learning_rate=self.learning_rate,
            max_iter=self.max_steps,
            tol=self.convergence_tol,
        )
        return pgd.optimize(initial_mechanism, obj, grad, project_simplex_rows)


class HessianComputer:
    """Compute second-order Hessian information for Newton-type methods."""

    def __init__(self, config: Optional[AutodiffConfig] = None) -> None:
        self.config = config or AutodiffConfig(compute_hessian=True)

    def hessian(
        self,
        mechanism: npt.NDArray[np.float64],
        divergence_type: DivergenceType,
        adjacent_pair: Tuple[int, int],
    ) -> npt.NDArray[np.float64]:
        """Compute the Hessian matrix of a divergence function.

        Args:
            mechanism: The n × k probability table.
            divergence_type: Which divergence to differentiate.
            adjacent_pair: The (i, i') pair.

        Returns:
            Hessian matrix of shape (n*k, n*k).
        """
        from dp_forge.autodiff.tape import reverse_hessian

        plg = PrivacyLossGrad(self.config)
        n, k = mechanism.shape

        def scalar_fn(flat: npt.NDArray[np.float64]) -> float:
            M = flat.reshape(n, k)
            if divergence_type == DivergenceType.HOCKEY_STICK:
                return plg.hockey_stick_gradient(M, 1.0, adjacent_pair).info.value
            elif divergence_type == DivergenceType.RENYI:
                return plg.renyi_gradient(M, 2.0, adjacent_pair).info.value
            elif divergence_type == DivergenceType.KL:
                return plg.kl_gradient(M, adjacent_pair).info.value
            else:
                return plg.hockey_stick_gradient(M, 1.0, adjacent_pair).info.value

        return reverse_hessian(scalar_fn, mechanism.ravel())

    def hessian_vector_product(
        self,
        mechanism: npt.NDArray[np.float64],
        divergence_type: DivergenceType,
        adjacent_pair: Tuple[int, int],
        vector: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute Hessian-vector product without forming the full Hessian.

        Args:
            mechanism: The n × k probability table.
            divergence_type: Which divergence to differentiate.
            adjacent_pair: The (i, i') pair.
            vector: Direction vector for the HVP.

        Returns:
            Hessian-vector product.
        """
        n, k = mechanism.shape
        plg = PrivacyLossGrad(self.config)
        h = self.config.numerical_eps

        def grad_fn(flat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            M = flat.reshape(n, k)
            if divergence_type == DivergenceType.HOCKEY_STICK:
                return plg.hockey_stick_gradient(M, 1.0, adjacent_pair).info.gradient
            elif divergence_type == DivergenceType.RENYI:
                return plg.renyi_gradient(M, 2.0, adjacent_pair).info.gradient
            elif divergence_type == DivergenceType.KL:
                return plg.kl_gradient(M, adjacent_pair).info.gradient
            else:
                return plg.hockey_stick_gradient(M, 1.0, adjacent_pair).info.gradient

        x = mechanism.ravel()
        v = vector.ravel()
        gp = grad_fn(x + h * v)
        gm = grad_fn(x - h * v)
        return (gp - gm) / (2.0 * h)


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def numerical_gradient_check(
    fn: Callable[[npt.NDArray[np.float64]], float],
    x: npt.NDArray[np.float64],
    analytic_gradient: npt.NDArray[np.float64],
    *,
    eps: float = 1e-5,
    rtol: float = 1e-3,
) -> Tuple[bool, float]:
    """Check analytic gradient against numerical finite differences.

    Args:
        fn: Scalar function to check.
        x: Point at which to check.
        analytic_gradient: Analytic gradient to verify.
        eps: Finite difference step size.
        rtol: Relative tolerance for agreement.

    Returns:
        Tuple of (passed, max_relative_error).
    """
    n = len(x)
    numerical = np.zeros(n, dtype=np.float64)
    for i in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        numerical[i] = (fn(xp) - fn(xm)) / (2.0 * eps)

    diffs = np.abs(analytic_gradient - numerical)
    scales = np.maximum(np.abs(analytic_gradient), np.abs(numerical)) + 1e-30
    rel_errors = diffs / scales
    max_err = float(np.max(rel_errors))
    return max_err < rtol, max_err


def compute_privacy_gradient(
    mechanism: npt.NDArray[np.float64],
    epsilon: float,
    delta: float = 0.0,
    *,
    divergence: DivergenceType = DivergenceType.HOCKEY_STICK,
    mode: DiffMode = DiffMode.REVERSE,
) -> GradientInfo:
    """Compute gradient of worst-case privacy loss.

    Convenience function that wraps PrivacyLossGrad for common use cases.

    Args:
        mechanism: The n × k probability table.
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        divergence: Type of divergence to use.
        mode: AD mode.

    Returns:
        GradientInfo with value and gradient.
    """
    config = AutodiffConfig(mode=mode)
    plg = PrivacyLossGrad(config)
    n = mechanism.shape[0]
    # Generate all adjacent pairs
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    if not pairs:
        return GradientInfo(value=0.0, gradient=np.zeros(mechanism.size))

    budget = PrivacyBudget(epsilon=epsilon, delta=delta)
    return plg.full_gradient(mechanism, budget, pairs)


__all__ = [
    # Enums
    "DiffMode",
    "DivergenceType",
    "OpType",
    # Config
    "AutodiffConfig",
    # Data types
    "DualNumber",
    "TapeNode",
    "ComputationTape",
    "DivergenceGradient",
    # Protocols
    "Differentiable",
    "PrivacyLossFunction",
    # Classes
    "PrivacyLossGrad",
    "MechanismOptimizer",
    "HessianComputer",
    # Functions
    "numerical_gradient_check",
    "compute_privacy_gradient",
]
