"""Computation graph nodes for the architecture IR.

Each node represents a single operation in a neural network computation graph,
carrying shape information, parameter counts, kernel recursion metadata,
and µP scaling exponents required for NTK and phase-diagram analysis.
"""

from __future__ import annotations

import copy
import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from .types import (
    ActivationType,
    InitializationType,
    KernelRecursionType,
    LayerType,
    NormalizationType,
    ScalingExponents,
    TensorShape,
)


# ---------------------------------------------------------------------------
# Abstract base node
# ---------------------------------------------------------------------------

class AbstractNode(ABC):
    """Base class for all computation-graph nodes."""

    def __init__(
        self,
        name: str,
        layer_type: LayerType,
        *,
        input_shape: Optional[TensorShape] = None,
        output_shape: Optional[TensorShape] = None,
        scaling: Optional[ScalingExponents] = None,
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.node_id: str = node_id or uuid.uuid4().hex[:12]
        self.name = name
        self.layer_type = layer_type
        self._input_shape = input_shape
        self._output_shape = output_shape
        self.scaling = scaling or ScalingExponents.standard()
        self.metadata: Dict[str, Any] = metadata or {}
        # Graph connectivity (managed externally by ComputationGraph)
        self.predecessors: List[str] = []
        self.successors: List[str] = []

    # ----- abstract interface ---------------------------------------------

    @abstractmethod
    def parameter_count(self) -> int:
        """Return the number of trainable parameters in this node."""

    @abstractmethod
    def kernel_recursion_type(self) -> KernelRecursionType:
        """Classify how this layer contributes to kernel recursion."""

    @abstractmethod
    def infer_output_shape(self, input_shape: TensorShape) -> TensorShape:
        """Compute the output shape given an input shape."""

    # ----- shape helpers --------------------------------------------------

    @property
    def input_shape(self) -> Optional[TensorShape]:
        return self._input_shape

    @input_shape.setter
    def input_shape(self, shape: TensorShape) -> None:
        self._input_shape = shape
        self._output_shape = self.infer_output_shape(shape)

    @property
    def output_shape(self) -> Optional[TensorShape]:
        return self._output_shape

    @output_shape.setter
    def output_shape(self, shape: TensorShape) -> None:
        self._output_shape = shape

    def resolve_shapes(self, input_shape: TensorShape) -> TensorShape:
        """Set input shape and infer output shape; return output."""
        self._input_shape = input_shape
        self._output_shape = self.infer_output_shape(input_shape)
        return self._output_shape

    # ----- fan dimensions -------------------------------------------------

    def fan_in(self) -> int:
        if self._input_shape is None:
            return 1
        nf = self._input_shape.num_features
        return nf if nf is not None else 1

    def fan_out(self) -> int:
        if self._output_shape is None:
            return 1
        nf = self._output_shape.num_features
        return nf if nf is not None else 1

    # ----- µP helpers ----------------------------------------------------

    def init_scale(self, width: int) -> float:
        return self.scaling.init_scale(width)

    def forward_scale(self, width: int) -> float:
        return self.scaling.forward_scale(width)

    def lr_scale(self, width: int) -> float:
        return self.scaling.lr_scale(width)

    # ----- weight sharing -------------------------------------------------

    def weight_sharing_key(self) -> Optional[str]:
        """Return a key for weight-sharing equivalence; None = no sharing."""
        return self.metadata.get("weight_sharing_key")

    def set_weight_sharing(self, key: str) -> None:
        self.metadata["weight_sharing_key"] = key

    # ----- skip-connection topology ----------------------------------------

    @property
    def is_skip_target(self) -> bool:
        return self.metadata.get("skip_target", False)

    @property
    def skip_source_ids(self) -> List[str]:
        return self.metadata.get("skip_sources", [])

    def add_skip_source(self, source_id: str) -> None:
        sources = self.metadata.setdefault("skip_sources", [])
        if source_id not in sources:
            sources.append(source_id)
        self.metadata["skip_target"] = True

    # ----- serialization --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "node_id": self.node_id,
            "name": self.name,
            "layer_type": self.layer_type.name,
            "scaling": self.scaling.to_dict(),
            "metadata": self.metadata,
            "predecessors": self.predecessors,
            "successors": self.successors,
        }
        if self._input_shape is not None:
            d["input_shape"] = self._input_shape.to_dict()
        if self._output_shape is not None:
            d["output_shape"] = self._output_shape.to_dict()
        d.update(self._extra_dict())
        return d

    def _extra_dict(self) -> Dict[str, Any]:
        """Subclass-specific serialisation fields."""
        return {}

    def __repr__(self) -> str:
        ish = self._input_shape.dims if self._input_shape else "?"
        osh = self._output_shape.dims if self._output_shape else "?"
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"in={ish}, out={osh}, params={self.parameter_count()})"
        )

    def summary_line(self) -> str:
        params = self.parameter_count()
        osh = self._output_shape.dims if self._output_shape else "?"
        return f"{self.name:<25s} {str(osh):<20s} {params:>10,d}"


# ---------------------------------------------------------------------------
# InputNode
# ---------------------------------------------------------------------------

class InputNode(AbstractNode):
    """Graph entry point."""

    def __init__(
        self,
        name: str = "input",
        shape: Optional[TensorShape] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, LayerType.Flatten, input_shape=shape, output_shape=shape, **kwargs)
        self._shape = shape

    def parameter_count(self) -> int:
        return 0

    def kernel_recursion_type(self) -> KernelRecursionType:
        return KernelRecursionType.Identity

    def infer_output_shape(self, input_shape: TensorShape) -> TensorShape:
        return input_shape


# ---------------------------------------------------------------------------
# OutputNode
# ---------------------------------------------------------------------------

class OutputNode(AbstractNode):
    """Graph exit point."""

    def __init__(self, name: str = "output", **kwargs: Any) -> None:
        super().__init__(name, LayerType.Flatten, **kwargs)

    def parameter_count(self) -> int:
        return 0

    def kernel_recursion_type(self) -> KernelRecursionType:
        return KernelRecursionType.Identity

    def infer_output_shape(self, input_shape: TensorShape) -> TensorShape:
        return input_shape


# ---------------------------------------------------------------------------
# DenseNode
# ---------------------------------------------------------------------------

class DenseNode(AbstractNode):
    """Fully-connected (linear) layer."""

    def __init__(
        self,
        name: str,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: InitializationType = InitializationType.He,
        scaling: Optional[ScalingExponents] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, LayerType.Dense, scaling=scaling, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.init = init

    def parameter_count(self) -> int:
        count = self.in_features * self.out_features
        if self.bias:
            count += self.out_features
        return count

    def kernel_recursion_type(self) -> KernelRecursionType:
        return KernelRecursionType.Linear

    def infer_output_shape(self, input_shape: TensorShape) -> TensorShape:
        if input_shape.ndim == 1:
            return TensorShape(dims=(self.out_features,))
        new_dims = input_shape.dims[:-1] + (self.out_features,)
        return TensorShape(dims=new_dims)

    def weight_variance(self) -> float:
        return self.init.variance(self.in_features, self.out_features)

    def bias_variance(self) -> float:
        return 0.0  # typical default

    def kernel_update_params(self) -> Dict[str, float]:
        """Parameters for the NTK kernel recursion at this layer."""
        sigma_w_sq = self.weight_variance()
        sigma_b_sq = self.bias_variance()
        return {
            "sigma_w_sq": sigma_w_sq,
            "sigma_b_sq": sigma_b_sq,
            "fan_in": float(self.in_features),
        }

    def _extra_dict(self) -> Dict[str, Any]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "init": self.init.value,
        }


# ---------------------------------------------------------------------------
# Conv1DNode
# ---------------------------------------------------------------------------

class Conv1DNode(AbstractNode):
    """1-D convolutional layer."""

    def __init__(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        init: InitializationType = InitializationType.He,
        scaling: Optional[ScalingExponents] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, LayerType.Conv1D, scaling=scaling, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.init = init

    @property
    def fan_in_per_group(self) -> int:
        return (self.in_channels // self.groups) * self.kernel_size

    def parameter_count(self) -> int:
        count = self.out_channels * (self.in_channels // self.groups) * self.kernel_size
        if self.bias:
            count += self.out_channels
        return count

    def kernel_recursion_type(self) -> KernelRecursionType:
        return KernelRecursionType.Convolution

    def infer_output_shape(self, input_shape: TensorShape) -> TensorShape:
        # input: (batch, channels, length)
        if input_shape.ndim < 2:
            raise ValueError(f"Conv1D requires >= 2 dims, got {input_shape.ndim}")
        length = input_shape.dims[-1]
        if length is not None:
            out_length = (length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        else:
            out_length = None
        if input_shape.ndim == 2:
            return TensorShape(dims=(self.out_channels, out_length))
        new_dims = input_shape.dims[:-2] + (self.out_channels, out_length)
        return TensorShape(dims=new_dims)

    def weight_variance(self) -> float:
        return self.init.variance(self.fan_in_per_group, self.out_channels)

    def kernel_update_params(self) -> Dict[str, float]:
        return {
            "sigma_w_sq": self.weight_variance(),
            "sigma_b_sq": 0.0,
            "fan_in": float(self.fan_in_per_group),
            "kernel_size": float(self.kernel_size),
        }

    def _extra_dict(self) -> Dict[str, Any]:
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "groups": self.groups,
            "bias": self.bias,
            "init": self.init.value,
        }


# ---------------------------------------------------------------------------
# Conv2DNode
# ---------------------------------------------------------------------------

class Conv2DNode(AbstractNode):
    """2-D convolutional layer."""

    def __init__(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        init: InitializationType = InitializationType.He,
        scaling: Optional[ScalingExponents] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, LayerType.Conv2D, scaling=scaling, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.bias = bias
        self.init = init

    @property
    def fan_in_per_group(self) -> int:
        return (self.in_channels // self.groups) * self.kernel_size[0] * self.kernel_size[1]

    def parameter_count(self) -> int:
        count = self.out_channels * (self.in_channels // self.groups) * self.kernel_size[0] * self.kernel_size[1]
        if self.bias:
            count += self.out_channels
        return count

    def kernel_recursion_type(self) -> KernelRecursionType:
        return KernelRecursionType.Convolution

    def _conv_out_dim(self, in_dim: Optional[int], axis: int) -> Optional[int]:
        if in_dim is None:
            return None
        return (
            in_dim
            + 2 * self.padding[axis]
            - self.dilation[axis] * (self.kernel_size[axis] - 1)
            - 1
        ) // self.stride[axis] + 1

    def infer_output_shape(self, input_shape: TensorShape) -> TensorShape:
        # input: (batch, channels, H, W)
        if input_shape.ndim < 3:
            raise ValueError(f"Conv2D requires >= 3 dims, got {input_shape.ndim}")
        h_in = input_shape.dims[-2]
        w_in = input_shape.dims[-1]
        h_out = self._conv_out_dim(h_in, 0)
        w_out = self._conv_out_dim(w_in, 1)
        if input_shape.ndim == 3:
            return TensorShape(dims=(self.out_channels, h_out, w_out))
        new_dims = input_shape.dims[:-3] + (self.out_channels, h_out, w_out)
        return TensorShape(dims=new_dims)

    def weight_variance(self) -> float:
        return self.init.variance(self.fan_in_per_group, self.out_channels)

    def _extra_dict(self) -> Dict[str, Any]:
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": list(self.kernel_size),
            "stride": list(self.stride),
            "padding": list(self.padding),
            "dilation": list(self.dilation),
            "groups": self.groups,
            "bias": self.bias,
            "init": self.init.value,
        }


# ---------------------------------------------------------------------------
# ActivationNode
# ---------------------------------------------------------------------------

class ActivationNode(AbstractNode):
    """Element-wise activation function."""

    def __init__(
        self,
        name: str,
        activation: ActivationType = ActivationType.ReLU,
        negative_slope: float = 0.0,
        **kwargs: Any,
    ) -> None:
        lt = {
            ActivationType.ReLU: LayerType.ReLU,
            ActivationType.LeakyReLU: LayerType.ReLU,
            ActivationType.GELU: LayerType.GELU,
            ActivationType.Sigmoid: LayerType.Sigmoid,
            ActivationType.Tanh: LayerType.Tanh,
            ActivationType.Softmax: LayerType.Softmax,
        }.get(activation, LayerType.ReLU)
        super().__init__(name, lt, **kwargs)
        self.activation = activation
        self.negative_slope = negative_slope

    def parameter_count(self) -> int:
        return 0

    def kernel_recursion_type(self) -> KernelRecursionType:
        if self.activation == ActivationType.Identity:
            return KernelRecursionType.Identity
        return KernelRecursionType.DualActivation

    def infer_output_shape(self, input_shape: TensorShape) -> TensorShape:
        return copy.deepcopy(input_shape)

    def dual_activation_value(self, k_xx: float, k_yy: float, k_xy: float) -> float:
        """Compute E[σ(u)σ(v)] for (u,v) ~ N(0, [[k_xx,k_xy],[k_xy,k_yy]])."""
        if self.activation == ActivationType.ReLU:
            return _relu_dual(k_xx, k_yy, k_xy)
        elif self.activation == ActivationType.Identity:
            return k_xy
        # Numerical integration fallback
        return _mc_dual(self.activation, k_xx, k_yy, k_xy)

    def dot_dual_activation_value(self, k_xx: float, k_yy: float, k_xy: float) -> float:
        """Compute E[σ'(u)σ'(v)] (derivative dual)."""
        if self.activation == ActivationType.ReLU:
            return _relu_dot_dual(k_xx, k_yy, k_xy)
        elif self.activation == ActivationType.Identity:
            return 1.0
        return _mc_dot_dual(self.activation, k_xx, k_yy, k_xy)

    def _extra_dict(self) -> Dict[str, Any]:
        return {
            "activation": self.activation.value,
            "negative_slope": self.negative_slope,
        }


def _relu_dual(k_xx: float, k_yy: float, k_xy: float) -> float:
    """Closed-form ReLU dual activation (arc-cosine kernel of order 1)."""
    denom = math.sqrt(max(k_xx * k_yy, 1e-30))
    cos_theta = np.clip(k_xy / denom, -1.0, 1.0)
    theta = math.acos(float(cos_theta))
    return (1.0 / (2.0 * math.pi)) * denom * (math.sin(theta) + (math.pi - theta) * cos_theta)


def _relu_dot_dual(k_xx: float, k_yy: float, k_xy: float) -> float:
    """Closed-form derivative dual for ReLU (arc-cosine kernel of order 0)."""
    denom = math.sqrt(max(k_xx * k_yy, 1e-30))
    cos_theta = np.clip(k_xy / denom, -1.0, 1.0)
    theta = math.acos(float(cos_theta))
    return (math.pi - theta) / (2.0 * math.pi)


def _mc_dual(
    act: ActivationType, k_xx: float, k_yy: float, k_xy: float, n_samples: int = 50_000
) -> float:
    """Monte-Carlo estimate of E[σ(u)σ(v)]."""
    rng = np.random.default_rng(0)
    cov = np.array([[k_xx, k_xy], [k_xy, k_yy]])
    cov = (cov + cov.T) / 2.0
    eigvals = np.linalg.eigvalsh(cov)
    if np.min(eigvals) < -1e-8:
        cov += (abs(np.min(eigvals)) + 1e-8) * np.eye(2)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov += 1e-6 * np.eye(2)
        L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_samples, 2))
    uv = z @ L.T
    su = act.evaluate(uv[:, 0])
    sv = act.evaluate(uv[:, 1])
    return float(np.mean(su * sv))


def _mc_dot_dual(
    act: ActivationType, k_xx: float, k_yy: float, k_xy: float, n_samples: int = 50_000
) -> float:
    """Monte-Carlo estimate of E[σ'(u)σ'(v)]."""
    rng = np.random.default_rng(0)
    cov = np.array([[k_xx, k_xy], [k_xy, k_yy]])
    cov = (cov + cov.T) / 2.0
    eigvals = np.linalg.eigvalsh(cov)
    if np.min(eigvals) < -1e-8:
        cov += (abs(np.min(eigvals)) + 1e-8) * np.eye(2)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov += 1e-6 * np.eye(2)
        L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_samples, 2))
    uv = z @ L.T
    du = act.derivative(uv[:, 0])
    dv = act.derivative(uv[:, 1])
    return float(np.mean(du * dv))


# ---------------------------------------------------------------------------
# NormNode
# ---------------------------------------------------------------------------

class NormNode(AbstractNode):
    """Normalization layer (BatchNorm, LayerNorm, etc.)."""

    def __init__(
        self,
        name: str,
        norm_type: NormalizationType,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        num_groups: int = 1,
        **kwargs: Any,
    ) -> None:
        lt = LayerType.BatchNorm if norm_type == NormalizationType.BatchNorm else LayerType.LayerNorm
        super().__init__(name, lt, **kwargs)
        self.norm_type = norm_type
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.num_groups = num_groups

    def parameter_count(self) -> int:
        if not self.affine:
            return 0
        return self.norm_type.param_count(self.num_features)

    def kernel_recursion_type(self) -> KernelRecursionType:
        return KernelRecursionType.Normalization

    def infer_output_shape(self, input_shape: TensorShape) -> TensorShape:
        return copy.deepcopy(input_shape)

    def kernel_normalise(self, K: np.ndarray) -> np.ndarray:
        """Normalise kernel matrix as layer-norm would."""
        diag = np.sqrt(np.maximum(np.diag(K), self.eps))
        outer = np.outer(diag, diag)
        return K / outer

    def _extra_dict(self) -> Dict[str, Any]:
        return {
            "norm_type": self.norm_type.value,
            "num_features": self.num_features,
            "eps": self.eps,
            "affine": self.affine,
            "num_groups": self.num_groups,
        }


# ---------------------------------------------------------------------------
# ResidualNode
# ---------------------------------------------------------------------------

class ResidualNode(AbstractNode):
    """Residual (skip-connection) merge node.

    Combines the branch output with the skip input: out = skip + branch.
    """

    def __init__(
        self,
        name: str,
        alpha: float = 1.0,
        beta: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, LayerType.Residual, **kwargs)
        self.alpha = alpha  # skip weight
        self.beta = beta    # branch weight

    def parameter_count(self) -> int:
        return 0

    def kernel_recursion_type(self) -> KernelRecursionType:
        return KernelRecursionType.Residual

    def infer_output_shape(self, input_shape: TensorShape) -> TensorShape:
        return copy.deepcopy(input_shape)

    def combine_kernels(self, K_skip: np.ndarray, K_branch: np.ndarray) -> np.ndarray:
        """K_res = α² K_skip + 2αβ K_cross + β² K_branch.
        When skip and branch are independent, K_cross ≈ 0."""
        return self.alpha ** 2 * K_skip + self.beta ** 2 * K_branch

    def combine_kernels_with_cross(
        self, K_skip: np.ndarray, K_branch: np.ndarray, K_cross: np.ndarray
    ) -> np.ndarray:
        return (
            self.alpha ** 2 * K_skip
            + 2.0 * self.alpha * self.beta * K_cross
            + self.beta ** 2 * K_branch
        )

    def _extra_dict(self) -> Dict[str, Any]:
        return {"alpha": self.alpha, "beta": self.beta}


# ---------------------------------------------------------------------------
# PoolingNode
# ---------------------------------------------------------------------------

class PoolingNode(AbstractNode):
    """Spatial pooling layer (avg, max, adaptive)."""

    def __init__(
        self,
        name: str,
        pool_type: str = "avg",
        kernel_size: Union[int, Tuple[int, ...]] = 2,
        stride: Optional[Union[int, Tuple[int, ...]]] = None,
        padding: Union[int, Tuple[int, ...]] = 0,
        adaptive_output_size: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, LayerType.Pooling, **kwargs)
        self.pool_type = pool_type
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.stride = stride
        self.padding = padding if isinstance(padding, tuple) else (padding,)
        self.adaptive_output_size = adaptive_output_size

    def parameter_count(self) -> int:
        return 0

    def kernel_recursion_type(self) -> KernelRecursionType:
        return KernelRecursionType.Pooling

    def infer_output_shape(self, input_shape: TensorShape) -> TensorShape:
        if self.adaptive_output_size is not None:
            osz = self.adaptive_output_size
            if isinstance(osz, int):
                osz = (osz,)
            new_dims = input_shape.dims[: input_shape.ndim - len(osz)] + tuple(osz)
            return TensorShape(dims=new_dims)

        # Non-adaptive pooling
        ndim_spatial = len(self.kernel_size)
        spatial_dims = input_shape.dims[-ndim_spatial:]
        stride = self.stride or self.kernel_size
        if isinstance(stride, int):
            stride = (stride,) * ndim_spatial
        out_spatial: List[Optional[int]] = []
        for i in range(ndim_spatial):
            sd = spatial_dims[i]
            if sd is None:
                out_spatial.append(None)
            else:
                pad = self.padding[i] if i < len(self.padding) else 0
                out_spatial.append((sd + 2 * pad - self.kernel_size[i]) // stride[i] + 1)
        new_dims = input_shape.dims[:-ndim_spatial] + tuple(out_spatial)
        return TensorShape(dims=new_dims)

    def _extra_dict(self) -> Dict[str, Any]:
        return {
            "pool_type": self.pool_type,
            "kernel_size": list(self.kernel_size),
            "stride": self.stride,
            "padding": list(self.padding),
            "adaptive_output_size": self.adaptive_output_size,
        }


# ---------------------------------------------------------------------------
# FlattenNode
# ---------------------------------------------------------------------------

class FlattenNode(AbstractNode):
    """Flatten spatial/channel dimensions into a single feature dimension."""

    def __init__(
        self,
        name: str,
        start_dim: int = 1,
        end_dim: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, LayerType.Flatten, **kwargs)
        self.start_dim = start_dim
        self.end_dim = end_dim

    def parameter_count(self) -> int:
        return 0

    def kernel_recursion_type(self) -> KernelRecursionType:
        return KernelRecursionType.Identity

    def infer_output_shape(self, input_shape: TensorShape) -> TensorShape:
        return input_shape.flatten(self.start_dim, self.end_dim)

    def _extra_dict(self) -> Dict[str, Any]:
        return {"start_dim": self.start_dim, "end_dim": self.end_dim}


# ---------------------------------------------------------------------------
# DropoutNode
# ---------------------------------------------------------------------------

class DropoutNode(AbstractNode):
    """Dropout layer (no-op at inference; scaling factor for kernel analysis)."""

    def __init__(
        self,
        name: str,
        p: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, LayerType.Dropout, **kwargs)
        self.p = p

    def parameter_count(self) -> int:
        return 0

    def kernel_recursion_type(self) -> KernelRecursionType:
        return KernelRecursionType.Identity

    def infer_output_shape(self, input_shape: TensorShape) -> TensorShape:
        return copy.deepcopy(input_shape)

    def kernel_scale_factor(self) -> float:
        """At inference, dropout scales by (1-p); kernel scales by (1-p)²."""
        return (1.0 - self.p) ** 2

    def _extra_dict(self) -> Dict[str, Any]:
        return {"p": self.p}


# ---------------------------------------------------------------------------
# AttentionNode (simplified single-head self-attention)
# ---------------------------------------------------------------------------

class AttentionNode(AbstractNode):
    """Multi-head self-attention layer."""

    def __init__(
        self,
        name: str,
        embed_dim: int,
        num_heads: int = 1,
        bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, LayerType.Attention, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // max(num_heads, 1)
        self.bias = bias

    def parameter_count(self) -> int:
        # Q, K, V projections + output projection
        qkv = 3 * self.embed_dim * self.embed_dim
        out = self.embed_dim * self.embed_dim
        b = 0
        if self.bias:
            b = 3 * self.embed_dim + self.embed_dim
        return qkv + out + b

    def kernel_recursion_type(self) -> KernelRecursionType:
        return KernelRecursionType.Attention

    def infer_output_shape(self, input_shape: TensorShape) -> TensorShape:
        # (batch, seq, embed) -> (batch, seq, embed)
        return copy.deepcopy(input_shape)

    def _extra_dict(self) -> Dict[str, Any]:
        return {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "bias": self.bias,
        }


# ---------------------------------------------------------------------------
# Factory: create node from dict spec
# ---------------------------------------------------------------------------

_NODE_REGISTRY: Dict[str, type] = {
    "dense": DenseNode,
    "conv1d": Conv1DNode,
    "conv2d": Conv2DNode,
    "activation": ActivationNode,
    "norm": NormNode,
    "residual": ResidualNode,
    "pooling": PoolingNode,
    "flatten": FlattenNode,
    "dropout": DropoutNode,
    "input": InputNode,
    "output": OutputNode,
    "attention": AttentionNode,
}


def create_node(spec: Dict[str, Any]) -> AbstractNode:
    """Create a node from a dictionary specification.

    The dict must contain a ``"type"`` key matching one of the registered
    node types. Remaining keys are forwarded as constructor kwargs.
    """
    spec = dict(spec)
    node_type = spec.pop("type").lower()
    if node_type not in _NODE_REGISTRY:
        raise ValueError(f"Unknown node type: {node_type!r}. Available: {list(_NODE_REGISTRY)}")
    cls = _NODE_REGISTRY[node_type]

    # Handle activation shorthand
    if node_type == "activation" and "activation" in spec and isinstance(spec["activation"], str):
        spec["activation"] = ActivationType(spec["activation"])
    if node_type == "norm" and "norm_type" in spec and isinstance(spec["norm_type"], str):
        spec["norm_type"] = NormalizationType(spec["norm_type"])
    if "init" in spec and isinstance(spec["init"], str):
        spec["init"] = InitializationType(spec["init"])
    if "scaling" in spec and isinstance(spec["scaling"], dict):
        spec["scaling"] = ScalingExponents.from_dict(spec["scaling"])

    # Handle shape
    if "shape" in spec and isinstance(spec["shape"], (list, tuple)):
        spec["shape"] = TensorShape(dims=tuple(spec["shape"]))
    if "input_shape" in spec and isinstance(spec["input_shape"], (list, tuple)):
        spec["input_shape"] = TensorShape(dims=tuple(spec["input_shape"]))

    return cls(**spec)
