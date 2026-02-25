"""Architecture specification types for the finite-width phase diagram system.

Defines enumerations, dataclasses, and type structures for representing
neural network architectures in a form suitable for NTK analysis and
phase diagram computation.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Layer type enumeration
# ---------------------------------------------------------------------------

class LayerType(Enum):
    """Enumeration of supported neural-network layer types."""

    Dense = auto()
    Conv1D = auto()
    Conv2D = auto()
    BatchNorm = auto()
    LayerNorm = auto()
    ReLU = auto()
    GELU = auto()
    Sigmoid = auto()
    Tanh = auto()
    Softmax = auto()
    Residual = auto()
    Attention = auto()
    Pooling = auto()
    Flatten = auto()
    Dropout = auto()

    # ---- helpers --------------------------------------------------------

    @classmethod
    def linear_layers(cls) -> List["LayerType"]:
        """Return layer types that are linear (affine) transformations."""
        return [cls.Dense, cls.Conv1D, cls.Conv2D]

    @classmethod
    def activation_layers(cls) -> List["LayerType"]:
        """Return layer types that are element-wise activations."""
        return [cls.ReLU, cls.GELU, cls.Sigmoid, cls.Tanh, cls.Softmax]

    @classmethod
    def normalization_layers(cls) -> List["LayerType"]:
        """Return layer types that are normalization layers."""
        return [cls.BatchNorm, cls.LayerNorm]

    @classmethod
    def structural_layers(cls) -> List["LayerType"]:
        """Return layer types that alter tensor topology."""
        return [cls.Residual, cls.Attention, cls.Pooling, cls.Flatten, cls.Dropout]

    @property
    def has_parameters(self) -> bool:
        """Whether this layer type has trainable parameters."""
        return self in (
            LayerType.Dense,
            LayerType.Conv1D,
            LayerType.Conv2D,
            LayerType.BatchNorm,
            LayerType.LayerNorm,
            LayerType.Attention,
        )

    @property
    def is_activation(self) -> bool:
        return self in self.activation_layers()

    @property
    def is_linear(self) -> bool:
        return self in self.linear_layers()

    @property
    def is_normalization(self) -> bool:
        return self in self.normalization_layers()


# ---------------------------------------------------------------------------
# Activation type
# ---------------------------------------------------------------------------

class ActivationType(Enum):
    """Fine-grained activation function classification."""

    ReLU = "relu"
    LeakyReLU = "leaky_relu"
    GELU = "gelu"
    Sigmoid = "sigmoid"
    Tanh = "tanh"
    Softmax = "softmax"
    SiLU = "silu"
    Mish = "mish"
    ELU = "elu"
    SELU = "selu"
    Softplus = "softplus"
    Identity = "identity"

    # ---- kernel properties -----------------------------------------------

    @property
    def is_homogeneous(self) -> bool:
        """Whether σ(αx) = α^k σ(x) for some k."""
        return self in (ActivationType.ReLU, ActivationType.LeakyReLU, ActivationType.Identity)

    @property
    def homogeneity_degree(self) -> Optional[int]:
        if self in (ActivationType.ReLU, ActivationType.LeakyReLU, ActivationType.Identity):
            return 1
        return None

    @property
    def dual_activation_closed_form(self) -> bool:
        """Whether the dual activation has a known closed form (for NTK recursion)."""
        return self in (
            ActivationType.ReLU,
            ActivationType.LeakyReLU,
            ActivationType.GELU,
            ActivationType.Sigmoid,
            ActivationType.Tanh,
            ActivationType.Identity,
        )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the activation function element-wise."""
        if self == ActivationType.ReLU:
            return np.maximum(x, 0.0)
        elif self == ActivationType.LeakyReLU:
            return np.where(x > 0, x, 0.01 * x)
        elif self == ActivationType.GELU:
            return 0.5 * x * (1.0 + _erf_approx(x / math.sqrt(2.0)))
        elif self == ActivationType.Sigmoid:
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif self == ActivationType.Tanh:
            return np.tanh(x)
        elif self == ActivationType.SiLU:
            sig = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
            return x * sig
        elif self == ActivationType.Mish:
            return x * np.tanh(np.log1p(np.exp(np.clip(x, -500, 500))))
        elif self == ActivationType.ELU:
            return np.where(x > 0, x, np.exp(x) - 1.0)
        elif self == ActivationType.SELU:
            alpha = 1.6732632423543772
            scale = 1.0507009873554805
            return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1.0))
        elif self == ActivationType.Softplus:
            return np.log1p(np.exp(np.clip(x, -500, 500)))
        elif self == ActivationType.Softmax:
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / e_x.sum(axis=-1, keepdims=True)
        elif self == ActivationType.Identity:
            return x.copy()
        raise ValueError(f"Unknown activation: {self}")

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the derivative of the activation function."""
        if self == ActivationType.ReLU:
            return (x > 0).astype(x.dtype)
        elif self == ActivationType.LeakyReLU:
            return np.where(x > 0, 1.0, 0.01)
        elif self == ActivationType.GELU:
            cdf = 0.5 * (1.0 + _erf_approx(x / math.sqrt(2.0)))
            pdf = np.exp(-0.5 * x ** 2) / math.sqrt(2.0 * math.pi)
            return cdf + x * pdf
        elif self == ActivationType.Sigmoid:
            s = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
            return s * (1.0 - s)
        elif self == ActivationType.Tanh:
            return 1.0 - np.tanh(x) ** 2
        elif self == ActivationType.SiLU:
            s = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
            return s + x * s * (1.0 - s)
        elif self == ActivationType.ELU:
            return np.where(x > 0, 1.0, np.exp(x))
        elif self == ActivationType.SELU:
            alpha = 1.6732632423543772
            scale = 1.0507009873554805
            return scale * np.where(x > 0, 1.0, alpha * np.exp(x))
        elif self == ActivationType.Softplus:
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif self == ActivationType.Identity:
            return np.ones_like(x)
        raise ValueError(f"No derivative for {self}")

    def expected_sq_norm(self) -> float:
        """E[σ(z)^2] for z ~ N(0,1).  Used in kernel recursion base."""
        if self == ActivationType.ReLU:
            return 0.5
        elif self == ActivationType.LeakyReLU:
            return 0.5 + 0.5 * 0.01 ** 2
        elif self == ActivationType.Sigmoid:
            return 0.270  # numerical approximation
        elif self == ActivationType.Tanh:
            return 0.393  # numerical approximation
        elif self == ActivationType.Identity:
            return 1.0
        # Fallback: Monte-Carlo estimate
        rng = np.random.default_rng(42)
        z = rng.standard_normal(100_000)
        return float(np.mean(self.evaluate(z) ** 2))


def _erf_approx(x: np.ndarray) -> np.ndarray:
    """Vectorised erf using scipy when available, else Abramowitz & Stegun."""
    try:
        from scipy.special import erf
        return erf(x)
    except ImportError:
        # Abramowitz & Stegun approximation (max error ~1.5e-7)
        a = np.abs(x)
        t = 1.0 / (1.0 + 0.3275911 * a)
        poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        result = 1.0 - poly * np.exp(-a * a)
        return np.where(x >= 0, result, -result)


# ---------------------------------------------------------------------------
# Normalization type
# ---------------------------------------------------------------------------

class NormalizationType(Enum):
    """Supported normalization schemes."""

    BatchNorm = "batch_norm"
    LayerNorm = "layer_norm"
    GroupNorm = "group_norm"
    InstanceNorm = "instance_norm"
    RMSNorm = "rms_norm"
    NoNorm = "none"

    @property
    def has_affine_params(self) -> bool:
        return self not in (NormalizationType.NoNorm,)

    @property
    def normalizes_over_batch(self) -> bool:
        return self in (NormalizationType.BatchNorm,)

    def param_count(self, num_features: int) -> int:
        """Number of trainable parameters (gamma + beta)."""
        if self == NormalizationType.NoNorm:
            return 0
        if self == NormalizationType.RMSNorm:
            return num_features  # only gamma
        return 2 * num_features


# ---------------------------------------------------------------------------
# Initialization type
# ---------------------------------------------------------------------------

class InitializationType(Enum):
    """Weight initialization schemes with associated fan computation."""

    Xavier = "xavier"
    He = "he"
    LeCun = "lecun"
    Orthogonal = "orthogonal"
    Normal = "normal"
    Uniform = "uniform"
    Zeros = "zeros"
    Ones = "ones"
    Custom = "custom"

    def variance(self, fan_in: int, fan_out: int) -> float:
        """Compute initialization variance given fan dimensions."""
        if self == InitializationType.Xavier:
            return 2.0 / (fan_in + fan_out)
        elif self == InitializationType.He:
            return 2.0 / fan_in
        elif self == InitializationType.LeCun:
            return 1.0 / fan_in
        elif self == InitializationType.Normal:
            return 1.0
        elif self == InitializationType.Uniform:
            return 1.0 / 3.0
        elif self in (InitializationType.Zeros, InitializationType.Ones):
            return 0.0
        elif self == InitializationType.Orthogonal:
            return 1.0 / fan_in
        elif self == InitializationType.Custom:
            return 1.0  # caller must override
        return 1.0

    def sample(
        self,
        shape: Tuple[int, ...],
        fan_in: int,
        fan_out: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Generate an initialised weight tensor."""
        if rng is None:
            rng = np.random.default_rng()
        var = self.variance(fan_in, fan_out)

        if self == InitializationType.Orthogonal:
            flat = rng.standard_normal((fan_out, fan_in))
            u, _, vt = np.linalg.svd(flat, full_matrices=False)
            q = u if u.shape == (fan_out, fan_in) else vt
            return q.reshape(shape).astype(np.float64)

        if self == InitializationType.Zeros:
            return np.zeros(shape, dtype=np.float64)
        if self == InitializationType.Ones:
            return np.ones(shape, dtype=np.float64)
        if self == InitializationType.Uniform:
            bound = math.sqrt(3.0 * var)
            return rng.uniform(-bound, bound, size=shape).astype(np.float64)

        return (rng.standard_normal(shape) * math.sqrt(var)).astype(np.float64)

    def mu_p_scaling(self, width: int) -> float:
        """Return the µP width-scaling factor for this init."""
        if self == InitializationType.He:
            return 1.0 / math.sqrt(width)
        elif self == InitializationType.Xavier:
            return 1.0 / math.sqrt(width)
        elif self == InitializationType.LeCun:
            return 1.0 / math.sqrt(width)
        elif self == InitializationType.Orthogonal:
            return 1.0 / math.sqrt(width)
        return 1.0


# ---------------------------------------------------------------------------
# TensorShape
# ---------------------------------------------------------------------------

@dataclass
class TensorShape:
    """Track tensor shapes through a computation graph.

    Supports named dimensions and symbolic sizes for width-parametric analysis.
    """

    dims: Tuple[Optional[int], ...]
    dim_names: Optional[Tuple[str, ...]] = None

    # ----- construction helpers -------------------------------------------

    @classmethod
    def vector(cls, n: int) -> "TensorShape":
        return cls(dims=(n,), dim_names=("features",))

    @classmethod
    def matrix(cls, rows: int, cols: int) -> "TensorShape":
        return cls(dims=(rows, cols), dim_names=("batch", "features"))

    @classmethod
    def batched(cls, batch: Optional[int], *rest: int) -> "TensorShape":
        dims: Tuple[Optional[int], ...] = (batch,) + tuple(rest)
        return cls(dims=dims)

    @classmethod
    def image(
        cls,
        batch: Optional[int],
        channels: int,
        height: int,
        width: int,
    ) -> "TensorShape":
        return cls(
            dims=(batch, channels, height, width),
            dim_names=("batch", "channels", "height", "width"),
        )

    @classmethod
    def sequence(
        cls,
        batch: Optional[int],
        length: int,
        features: int,
    ) -> "TensorShape":
        return cls(
            dims=(batch, length, features),
            dim_names=("batch", "length", "features"),
        )

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "TensorShape":
        return cls(dims=tuple(arr.shape))

    # ----- properties -----------------------------------------------------

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def numel(self) -> Optional[int]:
        """Total number of elements (None if any dim is symbolic/None)."""
        if any(d is None for d in self.dims):
            return None
        result = 1
        for d in self.dims:
            result *= d  # type: ignore[operator]
        return result

    @property
    def batch_size(self) -> Optional[int]:
        if self.ndim == 0:
            return None
        return self.dims[0]

    @property
    def feature_dims(self) -> Tuple[Optional[int], ...]:
        if self.ndim <= 1:
            return self.dims
        return self.dims[1:]

    @property
    def num_features(self) -> Optional[int]:
        fd = self.feature_dims
        if any(d is None for d in fd):
            return None
        result = 1
        for d in fd:
            result *= d  # type: ignore[operator]
        return result

    # ----- shape arithmetic -----------------------------------------------

    def broadcast_with(self, other: "TensorShape") -> "TensorShape":
        """Compute the broadcast shape."""
        max_ndim = max(self.ndim, other.ndim)
        a = (None,) * (max_ndim - self.ndim) + self.dims
        b = (None,) * (max_ndim - other.ndim) + other.dims
        result: List[Optional[int]] = []
        for da, db in zip(a, b):
            if da is None and db is None:
                result.append(None)
            elif da is None:
                result.append(db)
            elif db is None:
                result.append(da)
            elif da == 1:
                result.append(db)
            elif db == 1:
                result.append(da)
            elif da == db:
                result.append(da)
            else:
                raise ValueError(f"Shapes {self.dims} and {other.dims} not broadcastable")
        return TensorShape(dims=tuple(result))

    def flatten(self, start_dim: int = 1, end_dim: int = -1) -> "TensorShape":
        """Flatten dimensions [start_dim, end_dim]."""
        if end_dim < 0:
            end_dim = self.ndim + end_dim
        flat_dims = self.dims[start_dim: end_dim + 1]
        if any(d is None for d in flat_dims):
            flat_size: Optional[int] = None
        else:
            flat_size = 1
            for d in flat_dims:
                flat_size *= d  # type: ignore[operator]
        new_dims = self.dims[:start_dim] + (flat_size,) + self.dims[end_dim + 1:]
        return TensorShape(dims=new_dims)

    def replace_dim(self, axis: int, new_size: Optional[int]) -> "TensorShape":
        dims = list(self.dims)
        if axis < 0:
            axis = self.ndim + axis
        dims[axis] = new_size
        return TensorShape(dims=tuple(dims))

    def append_dim(self, size: int) -> "TensorShape":
        return TensorShape(dims=self.dims + (size,))

    def remove_dim(self, axis: int) -> "TensorShape":
        dims = list(self.dims)
        if axis < 0:
            axis = self.ndim + axis
        dims.pop(axis)
        return TensorShape(dims=tuple(dims))

    def transpose(self, dim0: int, dim1: int) -> "TensorShape":
        dims = list(self.dims)
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        names = None
        if self.dim_names is not None:
            name_list = list(self.dim_names)
            name_list[dim0], name_list[dim1] = name_list[dim1], name_list[dim0]
            names = tuple(name_list)
        return TensorShape(dims=tuple(dims), dim_names=names)

    def conv_output_size(
        self,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> int:
        """Compute output spatial dimension after convolution."""
        if self.ndim < 3:
            raise ValueError("conv_output_size requires spatial dimensions")
        spatial = self.dims[-1]
        if spatial is None:
            raise ValueError("Cannot compute conv output for symbolic spatial dim")
        return (spatial + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def pool_output_size(
        self,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
    ) -> int:
        if stride is None:
            stride = kernel_size
        spatial = self.dims[-1]
        if spatial is None:
            raise ValueError("Cannot compute pool output for symbolic spatial dim")
        return (spatial + 2 * padding - kernel_size) // stride + 1

    def num_elements(self) -> int:
        """Total number of elements (product of all dimensions). Scalars return 1."""
        result = 1
        for d in self.dims:
            if d is None:
                raise ValueError("Cannot compute num_elements with symbolic dims")
            result *= d
        return result

    # ----- comparison / hashing -------------------------------------------

    def is_compatible(self, other: "TensorShape") -> bool:
        """Check if shapes are compatible (equal where both are concrete)."""
        if self.ndim != other.ndim:
            return False
        for a, b in zip(self.dims, other.dims):
            if a is not None and b is not None and a != b:
                return False
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorShape):
            return NotImplemented
        return self.dims == other.dims

    def __hash__(self) -> int:
        return hash(self.dims)

    def __repr__(self) -> str:
        if self.dim_names:
            parts = [f"{n}={d}" for n, d in zip(self.dim_names, self.dims)]
            return f"TensorShape({', '.join(parts)})"
        return f"TensorShape({self.dims})"

    # ----- serialization --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"dims": list(self.dims)}
        if self.dim_names:
            d["dim_names"] = list(self.dim_names)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TensorShape":
        dims = tuple(d["dims"])
        names = tuple(d["dim_names"]) if "dim_names" in d else None
        return cls(dims=dims, dim_names=names)


# ---------------------------------------------------------------------------
# ScalingExponents (µP parameterization)
# ---------------------------------------------------------------------------

@dataclass
class ScalingExponents:
    r"""Maximal-update parameterization (µP) scaling exponents.

    For a layer of width *n*, the three exponents control:
    * **a** – initialization scale:  σ_init ∝ n^{-a}
    * **b** – forward-pass multiplier:  layer output ∝ n^{-b}
    * **c** – learning-rate scaling:  η_layer ∝ n^{-c}

    Standard parameterization (SP):  a = 0.5, b = 0, c = 0
    Maximal-update parameterization (µP):  a = 0.5, b = 0.5, c = 1  (for hidden)
    """

    a: float = 0.5
    b: float = 0.0
    c: float = 0.0

    # ----- presets --------------------------------------------------------

    @classmethod
    def standard(cls) -> "ScalingExponents":
        return cls(a=0.5, b=0.0, c=0.0)

    @classmethod
    def mu_p_hidden(cls) -> "ScalingExponents":
        return cls(a=0.5, b=0.5, c=1.0)

    @classmethod
    def mu_p_output(cls) -> "ScalingExponents":
        return cls(a=0.5, b=1.0, c=1.0)

    @classmethod
    def mu_p_input(cls) -> "ScalingExponents":
        return cls(a=0.5, b=0.0, c=0.5)

    @classmethod
    def ntk(cls) -> "ScalingExponents":
        """NTK parameterization: a=0.5, b=0.5, c=0."""
        return cls(a=0.5, b=0.5, c=0.0)

    @classmethod
    def mean_field(cls) -> "ScalingExponents":
        return cls(a=0.5, b=0.5, c=1.0)

    # ----- derived quantities ---------------------------------------------

    @property
    def alpha_w(self) -> float:
        """Alias for init scale exponent (a)."""
        return self.a

    @property
    def alpha_b(self) -> float:
        """Alias for forward scale exponent (b)."""
        return self.b

    def init_scale(self, width: int) -> float:
        return width ** (-self.a)

    def forward_scale(self, width: int) -> float:
        return width ** (-self.b)

    def lr_scale(self, width: int) -> float:
        return width ** (-self.c)

    def effective_lr(self, base_lr: float, width: int) -> float:
        return base_lr * self.lr_scale(width)

    def kernel_scaling(self, width: int) -> float:
        """Scaling of the NTK contribution from this layer at given width."""
        return width ** (1.0 - 2.0 * self.a - 2.0 * self.b + self.c)

    def is_feature_learning(self) -> bool:
        """Whether µP exponents allow feature learning (rich regime)."""
        return self.c > 0.0

    def is_lazy(self) -> bool:
        """Whether the layer stays in the kernel/lazy regime."""
        return self.c == 0.0

    def regime_descriptor(self) -> str:
        if self.is_lazy():
            return "lazy/kernel"
        if self.c == 1.0 and self.b == 0.5:
            return "maximal-update (µP)"
        return f"intermediate (c={self.c})"

    def interpolate(self, other: "ScalingExponents", t: float) -> "ScalingExponents":
        """Linearly interpolate between two scaling exponent sets."""
        return ScalingExponents(
            a=self.a + t * (other.a - self.a),
            b=self.b + t * (other.b - self.b),
            c=self.c + t * (other.c - self.c),
        )

    # ----- validation -----------------------------------------------------

    def validate(self) -> List[str]:
        """Return a list of warnings/errors about the exponent choice."""
        issues: List[str] = []
        if self.a < 0:
            issues.append(f"Init exponent a={self.a} is negative (diverging init).")
        if self.b < 0:
            issues.append(f"Forward exponent b={self.b} is negative (diverging activations).")
        if self.c < 0:
            issues.append(f"LR exponent c={self.c} is negative (diverging updates).")
        if self.a > 1:
            issues.append(f"Init exponent a={self.a} > 1 (vanishing init).")
        return issues

    # ----- serialization --------------------------------------------------

    def to_dict(self) -> Dict[str, float]:
        return {"a": self.a, "b": self.b, "c": self.c}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "ScalingExponents":
        return cls(a=d["a"], b=d["b"], c=d["c"])

    def __repr__(self) -> str:
        return f"ScalingExponents(a={self.a}, b={self.b}, c={self.c}) [{self.regime_descriptor()}]"


# ---------------------------------------------------------------------------
# KernelRecursionType
# ---------------------------------------------------------------------------

class KernelRecursionType(Enum):
    """Classification of a layer for NTK kernel recursion.

    Each layer in an architecture contributes to the NTK recursion differently.
    This enum classifies how a layer transforms the kernel matrix.
    """

    Linear = "linear"
    DualActivation = "dual_activation"
    Normalization = "normalization"
    Residual = "residual"
    Pooling = "pooling"
    Identity = "identity"
    Attention = "attention"
    Convolution = "convolution"

    # ----- semantics ------------------------------------------------------

    @property
    def modifies_kernel(self) -> bool:
        """Whether this recursion type changes the kernel matrix."""
        return self != KernelRecursionType.Identity

    @property
    def requires_eigendecomposition(self) -> bool:
        """Whether updating kernel at this step requires eigendecomposition."""
        return self in (KernelRecursionType.Normalization, KernelRecursionType.Attention)

    @property
    def has_closed_form(self) -> bool:
        """Whether the kernel update has a known closed-form expression."""
        return self in (
            KernelRecursionType.Linear,
            KernelRecursionType.DualActivation,
            KernelRecursionType.Identity,
            KernelRecursionType.Residual,
        )

    def describe(self) -> str:
        descriptions = {
            KernelRecursionType.Linear: (
                "K^{l+1}(x,x') = σ_w^2 / n_l · K^l(x,x') + σ_b^2"
            ),
            KernelRecursionType.DualActivation: (
                "K^{l+1}(x,x') = E_{(u,v)~N(0,Λ)}[σ(u)σ(v)] where "
                "Λ = [[K^l(x,x), K^l(x,x')], [K^l(x',x), K^l(x',x')]]"
            ),
            KernelRecursionType.Normalization: (
                "Kernel normalisation: K^{l+1} = K^l / sqrt(diag(K^l) diag(K^l)^T)"
            ),
            KernelRecursionType.Residual: (
                "K^{l+1}(x,x') = K^{skip}(x,x') + K^{branch}(x,x')"
            ),
            KernelRecursionType.Pooling: "Spatial averaging of kernel entries",
            KernelRecursionType.Identity: "K^{l+1} = K^l (pass-through)",
            KernelRecursionType.Attention: (
                "Softmax-weighted kernel composition (data-dependent)"
            ),
            KernelRecursionType.Convolution: (
                "K^{l+1}_{ij}(x,x') = Σ_δ w_δ^2 K^l_{i+δ,j+δ}(x,x') / fan_in"
            ),
        }
        return descriptions.get(self, "Unknown recursion type")


# ---------------------------------------------------------------------------
# Composite helper: ArchConfig
# ---------------------------------------------------------------------------

@dataclass
class ArchConfig:
    """High-level architecture configuration for phase-diagram analysis."""

    name: str = "unnamed"
    depth: int = 1
    width: int = 128
    input_dim: int = 784
    output_dim: int = 10
    activation: ActivationType = ActivationType.ReLU
    init: InitializationType = InitializationType.He
    norm: NormalizationType = NormalizationType.NoNorm
    scaling: ScalingExponents = field(default_factory=ScalingExponents.standard)
    bias: bool = True
    dropout_rate: float = 0.0
    residual: bool = False

    def total_params(self) -> int:
        """Rough parameter count for a simple MLP with these settings."""
        count = 0
        # first layer
        count += self.input_dim * self.width + (self.width if self.bias else 0)
        # hidden layers
        for _ in range(self.depth - 1):
            count += self.width * self.width + (self.width if self.bias else 0)
            count += self.norm.param_count(self.width)
        # output
        count += self.width * self.output_dim + (self.output_dim if self.bias else 0)
        return count

    def param_density(self) -> float:
        """Parameters per unit width squared (normalised complexity)."""
        return self.total_params() / (self.width ** 2) if self.width > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "depth": self.depth,
            "width": self.width,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "activation": self.activation.value,
            "init": self.init.value,
            "norm": self.norm.value,
            "scaling": self.scaling.to_dict(),
            "bias": self.bias,
            "dropout_rate": self.dropout_rate,
            "residual": self.residual,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArchConfig":
        return cls(
            name=d.get("name", "unnamed"),
            depth=d["depth"],
            width=d["width"],
            input_dim=d["input_dim"],
            output_dim=d["output_dim"],
            activation=ActivationType(d["activation"]),
            init=InitializationType(d["init"]),
            norm=NormalizationType(d.get("norm", "none")),
            scaling=ScalingExponents.from_dict(d["scaling"]) if "scaling" in d else ScalingExponents.standard(),
            bias=d.get("bias", True),
            dropout_rate=d.get("dropout_rate", 0.0),
            residual=d.get("residual", False),
        )
