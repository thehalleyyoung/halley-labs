"""
Compositional mean-field engine for arbitrary PyTorch computation graphs.

Generalizes PhaseKit's mean-field analysis beyond hardcoded MLP/Transformer
paths by defining composable variance propagation rules for each layer type.
The engine walks a module tree, builds a sequence of rules, detects residual
connections, and propagates predicted variance alongside empirical forward-hook
measurements.

Key idea: every rule is a **pure function** of the input variance (and layer
parameters), so rules compose freely across arbitrary architectures—Sequential
chains, custom modules, residual blocks, multi-branch networks, etc.

References:
    Poole et al., "Exponential expressivity in deep neural networks through
        transient chaos", NeurIPS 2016
    Schoenholz et al., "Deep Information Propagation", ICLR 2017
    Xiao et al., "Dynamical Isometry and a Mean Field Theory of CNNs",
        ICML 2018
    Noci et al., "Signal Propagation in Transformers", ICML 2022
    Yang & Schoenholz, "Mean Field Residual Networks", 2017
"""

from __future__ import annotations

import abc
import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch import Tensor

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from mean_field_theory import ActivationVarianceMaps, MeanFieldAnalyzer
except ImportError:
    from .mean_field_theory import ActivationVarianceMaps, MeanFieldAnalyzer

try:
    from graph_analyzer import (
        LayerInfo,
        GraphAnalysisResult,
        VarianceTracer,
        _classify_module,
    )
except ImportError:
    from .graph_analyzer import (
        LayerInfo,
        GraphAnalysisResult,
        VarianceTracer,
        _classify_module,
    )


def _require_torch() -> None:
    """Raise if PyTorch is unavailable."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for compositional_mf. "
            "Install with: pip install torch"
        )


# ======================================================================
# Activation helpers
# ======================================================================

_VARIANCE_MAPS: Dict[str, Callable[[float], float]] = {}
_CHI_MAPS: Dict[str, Callable[[float], float]] = {}


def _populate_maps() -> None:
    """Lazily populate activation variance/chi look-ups."""
    if _VARIANCE_MAPS:
        return
    avm = ActivationVarianceMaps
    _VARIANCE_MAPS.update({
        "relu": avm.relu_variance,
        "tanh": avm.tanh_variance,
        "sigmoid": avm.sigmoid_variance,
        "gelu": avm.gelu_variance,
        "silu": avm.silu_variance,
        "swish": avm.silu_variance,
        "elu": avm.elu_variance,
        "leaky_relu": avm.leaky_relu_variance,
        "mish": avm.mish_variance,
        "linear": avm.linear_variance,
    })
    _CHI_MAPS.update({
        "relu": avm.relu_chi,
        "tanh": avm.tanh_chi,
        "sigmoid": avm.sigmoid_chi,
        "gelu": avm.gelu_chi,
        "silu": avm.silu_chi,
        "swish": avm.silu_chi,
        "elu": avm.elu_chi,
        "leaky_relu": avm.leaky_relu_chi,
        "mish": avm.mish_chi,
        "linear": avm.linear_chi,
    })


def _get_V(activation: str) -> Callable[[float], float]:
    """Return the variance map V(q) for *activation*."""
    _populate_maps()
    return _VARIANCE_MAPS.get(activation, _VARIANCE_MAPS["relu"])


def _get_chi(activation: str) -> Callable[[float], float]:
    """Return the chi map for *activation*."""
    _populate_maps()
    return _CHI_MAPS.get(activation, _CHI_MAPS["relu"])


# ======================================================================
# Module-type string constants (mirror graph_analyzer)
# ======================================================================

_ACT_MAP: Dict[str, str] = {
    "ReLU": "relu",
    "LeakyReLU": "leaky_relu",
    "GELU": "gelu",
    "SiLU": "silu",
    "Tanh": "tanh",
    "Sigmoid": "sigmoid",
    "Mish": "mish",
    "ELU": "elu",
}

_NORM_LAYERS = frozenset({
    "LayerNorm", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
    "GroupNorm", "InstanceNorm1d", "InstanceNorm2d", "RMSNorm",
})

_ATTENTION_TYPES = frozenset({
    "MultiheadAttention", "MultiHeadAttention",
    "SelfAttention", "CausalSelfAttention",
    "ScaledDotProductAttention",
})

_RESIDUAL_CONTAINERS = frozenset({
    "BasicBlock", "Bottleneck", "ResidualBlock", "ResBlock",
    "TransformerEncoderLayer", "TransformerDecoderLayer",
})


# ======================================================================
# Section 1 — Variance propagation rules
# ======================================================================

@dataclass
class PropagationResult:
    """Output of a single variance-propagation rule.

    Attributes
    ----------
    q_out : float
        Output pre-activation variance.
    chi_1 : float
        Local susceptibility d(q_out)/d(q_in) for this rule.
    name : str
        Human-readable label (e.g. ``"Linear(768→3072)"``).
    """
    q_out: float
    chi_1: float
    name: str = ""


class VarianceRule(abc.ABC):
    """Abstract base for a composable variance-propagation rule.

    Every concrete rule must implement :meth:`propagate`, a **pure function**
    of the input variance and the rule's own parameters.  This guarantees
    composability: rules can be chained in any order, stored in lists, and
    applied to multi-branch graphs without side effects.
    """

    @abc.abstractmethod
    def propagate(self, q_in: float) -> PropagationResult:
        """Map input variance *q_in* to output variance.

        Parameters
        ----------
        q_in : float
            Input pre-activation variance.

        Returns
        -------
        PropagationResult
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ------------------------------------------------------------------
# LinearRule
# ------------------------------------------------------------------

class LinearRule(VarianceRule):
    r"""Standard mean-field rule for a fully-connected layer.

    .. math::
        q_\text{out} = \sigma_w^2 \, V_\varphi(q_\text{in}) + \sigma_b^2

    where :math:`V_\varphi` is the variance map for the *preceding* activation
    (defaults to identity / linear when no activation precedes this layer).

    Parameters
    ----------
    sigma_w : float
        Weight standard deviation scaled by :math:`\sqrt{\text{fan\_in}}`.
    sigma_b : float
        Bias standard deviation.
    fan_in : int
        Number of input features.
    fan_out : int
        Number of output features.
    label : str
        Human-readable label.
    """

    def __init__(
        self,
        sigma_w: float = 1.0,
        sigma_b: float = 0.0,
        fan_in: int = 1,
        fan_out: int = 1,
        label: str = "Linear",
    ) -> None:
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.label = label

    def propagate(self, q_in: float) -> PropagationResult:
        q_out = self.sigma_w ** 2 * q_in + self.sigma_b ** 2
        chi_1 = self.sigma_w ** 2
        return PropagationResult(q_out=q_out, chi_1=chi_1, name=self.label)

    def __repr__(self) -> str:
        return f"LinearRule(fan_in={self.fan_in}, fan_out={self.fan_out}, σ_w={self.sigma_w:.4f})"


# ------------------------------------------------------------------
# ConvRule
# ------------------------------------------------------------------

class ConvRule(VarianceRule):
    r"""Variance propagation for convolutional layers (Xiao et al. 2018).

    Same recursion as :class:`LinearRule` but with
    :math:`\text{fan\_in} = k_h \times k_w \times C_\text{in}`.

    Parameters
    ----------
    sigma_w : float
        Weight std scaled by :math:`\sqrt{\text{fan\_in}}`.
    sigma_b : float
        Bias std.
    kernel_size : tuple of int
        Spatial kernel dimensions ``(k_h, k_w)`` (or ``(k,)`` for 1-D).
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels / filters.
    label : str
        Human-readable label.
    """

    def __init__(
        self,
        sigma_w: float = 1.0,
        sigma_b: float = 0.0,
        kernel_size: Tuple[int, ...] = (3, 3),
        in_channels: int = 3,
        out_channels: int = 64,
        label: str = "Conv",
    ) -> None:
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fan_in = int(np.prod(kernel_size)) * in_channels
        self.label = label

    def propagate(self, q_in: float) -> PropagationResult:
        q_out = self.sigma_w ** 2 * q_in + self.sigma_b ** 2
        chi_1 = self.sigma_w ** 2
        return PropagationResult(q_out=q_out, chi_1=chi_1, name=self.label)

    def __repr__(self) -> str:
        return (
            f"ConvRule(kernel={self.kernel_size}, C_in={self.in_channels}, "
            f"C_out={self.out_channels}, σ_w={self.sigma_w:.4f})"
        )


# ------------------------------------------------------------------
# ActivationRule
# ------------------------------------------------------------------

class ActivationRule(VarianceRule):
    r"""Activation-specific variance map.

    .. math::
        q_\text{out} = V_\varphi(q_\text{in})

    where :math:`V_\varphi(q) = \mathbb{E}_{z \sim \mathcal{N}(0,q)}[\varphi(z)^2]`.

    Parameters
    ----------
    activation : str
        Activation name (``"relu"``, ``"gelu"``, ``"tanh"``, etc.).
    label : str
        Human-readable label.
    """

    def __init__(self, activation: str = "relu", label: str = "Activation") -> None:
        self.activation = activation
        self._V = _get_V(activation)
        self._chi = _get_chi(activation)
        self.label = label

    def propagate(self, q_in: float) -> PropagationResult:
        q_out = self._V(max(q_in, 0.0))
        chi_1 = self._chi(max(q_in, 1e-30))
        return PropagationResult(q_out=q_out, chi_1=chi_1, name=self.label)

    def __repr__(self) -> str:
        return f"ActivationRule(activation={self.activation!r})"


# ------------------------------------------------------------------
# NormRule
# ------------------------------------------------------------------

class NormRule(VarianceRule):
    r"""Normalization layer (LayerNorm / BatchNorm / GroupNorm / RMSNorm).

    After normalization the pre-activation variance resets to the learned
    affine scale:

    .. math::
        q_\text{out} = \gamma^2

    Parameters
    ----------
    gamma : float
        Learned scale parameter (default 1.0 for standard init).
    label : str
        Human-readable label.
    """

    def __init__(self, gamma: float = 1.0, label: str = "Norm") -> None:
        self.gamma = gamma
        self.label = label

    def propagate(self, q_in: float) -> PropagationResult:
        q_out = self.gamma ** 2
        # Susceptibility through norm: d(q_out)/d(q_in) → 0 in the
        # infinite-width limit (norm erases input scale), but we report
        # the ratio for finite networks.
        chi_1 = self.gamma ** 2 / max(q_in, 1e-30) if q_in > 1e-30 else 1.0
        return PropagationResult(q_out=q_out, chi_1=chi_1, name=self.label)

    def __repr__(self) -> str:
        return f"NormRule(γ={self.gamma:.4f})"


# ------------------------------------------------------------------
# ResidualRule
# ------------------------------------------------------------------

class ResidualRule(VarianceRule):
    r"""Additive residual / skip connection.

    .. math::
        q_\text{out} = q_\text{skip} + q_\text{branch}

    The rule stores the skip variance when the residual block opens and
    adds it when the block closes.

    Parameters
    ----------
    alpha : float
        Scaling factor on the branch (``q_out = q_skip + alpha^2 * q_branch``).
        Default 1.0 (standard ResNet pre-activation).
    label : str
        Human-readable label.
    """

    def __init__(self, alpha: float = 1.0, label: str = "Residual") -> None:
        self.alpha = alpha
        self.q_skip: float = 0.0
        self.label = label

    def save_skip(self, q: float) -> None:
        """Record the skip-connection variance before the branch."""
        self.q_skip = q

    def propagate(self, q_in: float) -> PropagationResult:
        q_out = self.q_skip + self.alpha ** 2 * q_in
        # chi_1 w.r.t. branch variance only
        chi_1 = self.alpha ** 2
        return PropagationResult(q_out=q_out, chi_1=chi_1, name=self.label)

    def __repr__(self) -> str:
        return f"ResidualRule(α={self.alpha:.4f})"


# ------------------------------------------------------------------
# AttentionRule
# ------------------------------------------------------------------

class AttentionRule(VarianceRule):
    r"""Softmax self-attention variance propagation (Noci et al. 2022).

    Under mean-field assumptions with uniform attention weights:

    .. math::
        q_\text{out} \approx q_v \left(1 + \frac{1}{T}\right)

    where :math:`q_v` is the value-projection variance and :math:`T` is the
    sequence length.  The :math:`1/T` term captures the concentration effect
    of softmax averaging.

    Parameters
    ----------
    seq_len : int
        Sequence length *T*.
    label : str
        Human-readable label.
    """

    def __init__(self, seq_len: int = 128, label: str = "Attention") -> None:
        self.seq_len = max(seq_len, 1)
        self.label = label

    def propagate(self, q_in: float) -> PropagationResult:
        q_out = q_in * (1.0 + 1.0 / self.seq_len)
        chi_1 = 1.0 + 1.0 / self.seq_len
        return PropagationResult(q_out=q_out, chi_1=chi_1, name=self.label)

    def __repr__(self) -> str:
        return f"AttentionRule(T={self.seq_len})"


# ------------------------------------------------------------------
# DropoutRule
# ------------------------------------------------------------------

class DropoutRule(VarianceRule):
    r"""Dropout variance scaling.

    During **training** (inverted dropout):

    .. math::
        q_\text{out} = \frac{q_\text{in}}{1 - p}

    During **eval**: :math:`q_\text{out} = q_\text{in}`.

    Parameters
    ----------
    p : float
        Drop probability.
    training : bool
        Whether the model is in training mode.
    label : str
        Human-readable label.
    """

    def __init__(
        self, p: float = 0.0, training: bool = True, label: str = "Dropout"
    ) -> None:
        self.p = min(max(p, 0.0), 1.0 - 1e-7)
        self.training = training
        self.label = label

    def propagate(self, q_in: float) -> PropagationResult:
        if self.training and self.p > 0.0:
            q_out = q_in / (1.0 - self.p)
        else:
            q_out = q_in
        chi_1 = 1.0 / (1.0 - self.p) if (self.training and self.p > 0.0) else 1.0
        return PropagationResult(q_out=q_out, chi_1=chi_1, name=self.label)

    def __repr__(self) -> str:
        return f"DropoutRule(p={self.p:.3f}, training={self.training})"


# ------------------------------------------------------------------
# PoolingRule
# ------------------------------------------------------------------

class PoolingRule(VarianceRule):
    r"""Spatial pooling variance propagation.

    * **Average pooling**: :math:`q_\text{out} = q_\text{in} / k` where
      :math:`k` is the pool size (averaging reduces variance).
    * **Max pooling**: :math:`q_\text{out} \approx q_\text{in}(1 + 2\log(k)/\pi)`
      using the expected maximum of *k* i.i.d. Gaussians.

    Parameters
    ----------
    pool_size : int
        Number of elements in the pooling window.
    mode : str
        ``"avg"`` or ``"max"``.
    label : str
        Human-readable label.
    """

    def __init__(
        self, pool_size: int = 2, mode: str = "avg", label: str = "Pool"
    ) -> None:
        self.pool_size = max(pool_size, 1)
        self.mode = mode
        self.label = label

    def propagate(self, q_in: float) -> PropagationResult:
        if self.mode == "avg":
            q_out = q_in / self.pool_size
            chi_1 = 1.0 / self.pool_size
        else:  # max
            correction = 1.0 + 2.0 * math.log(max(self.pool_size, 1)) / math.pi
            q_out = q_in * correction
            chi_1 = correction
        return PropagationResult(q_out=q_out, chi_1=chi_1, name=self.label)

    def __repr__(self) -> str:
        return f"PoolingRule(size={self.pool_size}, mode={self.mode!r})"


# ------------------------------------------------------------------
# ConcatRule
# ------------------------------------------------------------------

class ConcatRule(VarianceRule):
    r"""Channel concatenation of multiple branches.

    The output variance is the weighted average of input variances,
    weighted by the relative channel counts.

    Parameters
    ----------
    weights : list of float
        Relative weights (will be normalised internally).
    label : str
        Human-readable label.
    """

    def __init__(
        self, weights: Optional[List[float]] = None, label: str = "Concat"
    ) -> None:
        if weights is None:
            weights = [1.0]
        total = sum(weights) or 1.0
        self.weights = [w / total for w in weights]
        self.branch_variances: List[float] = []
        self.label = label

    def set_branch_variances(self, variances: List[float]) -> None:
        """Store per-branch variances before calling :meth:`propagate`."""
        self.branch_variances = list(variances)

    def propagate(self, q_in: float) -> PropagationResult:
        if self.branch_variances:
            # Pad weights if needed
            w = self.weights
            v = self.branch_variances
            if len(w) < len(v):
                extra = (1.0 / len(v),) * (len(v) - len(w))
                w = list(w) + list(extra)
                total = sum(w)
                w = [x / total for x in w]
            q_out = sum(wi * vi for wi, vi in zip(w, v))
        else:
            q_out = q_in
        chi_1 = 1.0  # weighted average preserves order of magnitude
        return PropagationResult(q_out=q_out, chi_1=chi_1, name=self.label)

    def __repr__(self) -> str:
        return f"ConcatRule(weights={self.weights})"


# ------------------------------------------------------------------
# MultiplyRule
# ------------------------------------------------------------------

class MultiplyRule(VarianceRule):
    r"""Element-wise multiplication (gating, attention logits, etc.).

    For independent inputs:

    .. math::
        q_\text{out} = q_a \cdot q_b

    Parameters
    ----------
    q_other : float
        Variance of the second operand.
    label : str
        Human-readable label.
    """

    def __init__(self, q_other: float = 1.0, label: str = "Multiply") -> None:
        self.q_other = q_other
        self.label = label

    def propagate(self, q_in: float) -> PropagationResult:
        q_out = q_in * self.q_other
        chi_1 = self.q_other
        return PropagationResult(q_out=q_out, chi_1=chi_1, name=self.label)

    def __repr__(self) -> str:
        return f"MultiplyRule(q_other={self.q_other:.4f})"


# ------------------------------------------------------------------
# IdentityRule (pass-through)
# ------------------------------------------------------------------

class IdentityRule(VarianceRule):
    """Identity / pass-through (containers, flatten, reshape, etc.)."""

    def __init__(self, label: str = "Identity") -> None:
        self.label = label

    def propagate(self, q_in: float) -> PropagationResult:
        return PropagationResult(q_out=q_in, chi_1=1.0, name=self.label)


# ======================================================================
# Section 2 — ComputationGraphMF
# ======================================================================

@dataclass
class _ResidualSpan:
    """Book-keeping for a detected residual block."""
    start_idx: int
    end_idx: int
    rule: ResidualRule


class ComputationGraphMF:
    """Compositional mean-field engine for arbitrary ``nn.Module`` graphs.

    The engine:

    1. Walks the module tree and creates a sequence of :class:`VarianceRule`
       objects.
    2. Detects residual connections—first via ``torch.fx.symbolic_trace``
       (looks for ``add`` call nodes whose inputs include a skip), then via
       module-structure heuristics (known residual container names,
       ``"residual"`` / ``"shortcut"`` in submodule names).
    3. Propagates variance through the rule sequence, inserting skip-save /
       skip-add operations at residual boundaries.
    4. Computes per-layer :math:`\\chi_1` and an overall predicted phase.
    5. Optionally applies finite-width corrections using
       :class:`ActivationVarianceMaps` moment methods.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model.
    seq_len : int
        Default sequence length for attention rules (only used when the
        model contains attention layers and no explicit length is given).
    training : bool
        Whether to model training-mode dropout scaling.
    """

    def __init__(
        self,
        model: "nn.Module",
        seq_len: int = 128,
        training: bool = False,
    ) -> None:
        _require_torch()
        self.model = model
        self.seq_len = seq_len
        self.training = training

        self.rules: List[VarianceRule] = []
        self.residual_spans: List[_ResidualSpan] = []
        self._layer_infos: List[LayerInfo] = []

        self._build_rules()

    # ------------------------------------------------------------------
    # Rule construction
    # ------------------------------------------------------------------

    def _build_rules(self) -> None:
        """Walk the module tree and emit a flat rule sequence."""
        self.rules.clear()
        self.residual_spans.clear()
        self._layer_infos.clear()

        # Collect leaf-like modules in traversal order
        leaves: List[Tuple[str, "nn.Module"]] = []
        for name, module in self.model.named_modules():
            children = list(module.children())
            # Keep leaves *and* known atomic containers (attention, residual)
            if (len(children) == 0
                    or isinstance(module, nn.MultiheadAttention)
                    or type(module).__name__ in _RESIDUAL_CONTAINERS
                    or type(module).__name__ in _ATTENTION_TYPES):
                if len(children) > 0 and type(module).__name__ in _RESIDUAL_CONTAINERS:
                    continue  # handle children individually, mark residual
                leaves.append((name, module))

        # Detect residual containers for span marking
        residual_children: Dict[str, str] = {}  # child_name → container_name
        for name, module in self.model.named_modules():
            if type(module).__name__ in _RESIDUAL_CONTAINERS:
                for cname, _ in module.named_modules():
                    if cname:
                        full = f"{name}.{cname}" if name else cname
                        residual_children[full] = name

        # Also detect by naming convention
        for name, module in self.model.named_modules():
            lower = name.lower()
            if any(tok in lower for tok in ("residual", "shortcut", "skip", "downsample")):
                residual_children[name] = name

        # Try FX-based residual detection
        fx_residual_pairs = self._detect_residual_fx()

        # Build rules from leaves
        current_residual_container: Optional[str] = None
        residual_start_idx: Optional[int] = None

        for leaf_name, module in leaves:
            cls_name = type(module).__name__
            info = _classify_module(leaf_name, module)
            self._layer_infos.append(info)

            # --- Check if we enter / exit a residual block ---
            container = residual_children.get(leaf_name)
            if container is not None and container != current_residual_container:
                # Close previous residual span if open
                if current_residual_container is not None and residual_start_idx is not None:
                    self._close_residual_span(residual_start_idx)
                # Open new span
                current_residual_container = container
                residual_start_idx = len(self.rules)

            # --- Emit rule ---
            rule = self._module_to_rule(leaf_name, module, info)
            if rule is not None:
                self.rules.append(rule)

        # Close any trailing residual span
        if current_residual_container is not None and residual_start_idx is not None:
            self._close_residual_span(residual_start_idx)

        # Overlay FX-detected residual spans (may add spans missed above)
        self._apply_fx_residuals(fx_residual_pairs)

    def _module_to_rule(
        self, name: str, module: "nn.Module", info: LayerInfo
    ) -> Optional[VarianceRule]:
        """Convert a single module to its corresponding variance rule."""
        cls_name = type(module).__name__

        # Linear
        if isinstance(module, nn.Linear):
            sigma_w = info.sigma_w
            sigma_b = 0.0
            if module.bias is not None:
                sigma_b = float(module.bias.data.std().item()) if module.bias.numel() > 1 else 0.0
            return LinearRule(
                sigma_w=sigma_w, sigma_b=sigma_b,
                fan_in=module.in_features, fan_out=module.out_features,
                label=f"Linear({module.in_features}→{module.out_features}) [{name}]",
            )

        # Conv
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            sigma_w = info.sigma_w
            sigma_b = 0.0
            if module.bias is not None and module.bias.numel() > 1:
                sigma_b = float(module.bias.data.std().item())
            ks = module.kernel_size
            if isinstance(ks, int):
                ks = (ks,)
            return ConvRule(
                sigma_w=sigma_w, sigma_b=sigma_b,
                kernel_size=ks,
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                label=f"Conv{len(ks)}d({module.in_channels}→{module.out_channels}) [{name}]",
            )

        # Activation
        if cls_name in _ACT_MAP:
            act_name = _ACT_MAP[cls_name]
            return ActivationRule(activation=act_name, label=f"{cls_name} [{name}]")

        # Normalization
        if cls_name in _NORM_LAYERS or isinstance(
            module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)
        ):
            gamma = 1.0
            if hasattr(module, "weight") and module.weight is not None:
                gamma = float(module.weight.data.mean().item())
                gamma = max(abs(gamma), 1e-6)
            return NormRule(gamma=gamma, label=f"{cls_name} [{name}]")

        # Attention
        if cls_name in _ATTENTION_TYPES or isinstance(module, nn.MultiheadAttention):
            return AttentionRule(seq_len=self.seq_len, label=f"Attention [{name}]")

        # Dropout
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            p = getattr(module, "p", 0.0)
            return DropoutRule(p=p, training=self.training, label=f"Dropout(p={p}) [{name}]")

        # Pooling
        if isinstance(module, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.AdaptiveAvgPool2d)):
            ks = getattr(module, "kernel_size", 2)
            if isinstance(ks, tuple):
                pool_size = int(np.prod(ks))
            elif isinstance(ks, int):
                pool_size = ks
            else:
                pool_size = 2
            pool_size = max(pool_size, 1)
            return PoolingRule(pool_size=pool_size, mode="avg", label=f"AvgPool [{name}]")

        if isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            ks = getattr(module, "kernel_size", 2)
            if isinstance(ks, tuple):
                pool_size = int(np.prod(ks))
            elif isinstance(ks, int):
                pool_size = ks
            else:
                pool_size = 2
            pool_size = max(pool_size, 1)
            return PoolingRule(pool_size=pool_size, mode="max", label=f"MaxPool [{name}]")

        # Flatten, Identity, etc. → pass-through
        if isinstance(module, (nn.Flatten, nn.Identity)):
            return IdentityRule(label=f"{cls_name} [{name}]")

        # Embedding → treat as setting variance from weight stats
        if isinstance(module, nn.Embedding):
            w_var = float(module.weight.data.var().item())
            return NormRule(gamma=math.sqrt(max(w_var, 1e-30)), label=f"Embedding [{name}]")

        # Unknown leaf → identity (best-effort)
        return IdentityRule(label=f"{cls_name} [{name}]")

    def _close_residual_span(self, start_idx: int) -> None:
        """Register a residual span ending at the current tail of self.rules."""
        end_idx = len(self.rules) - 1
        if end_idx <= start_idx:
            return
        rule = ResidualRule(alpha=1.0, label=f"ResidualAdd @{start_idx}-{end_idx}")
        self.residual_spans.append(
            _ResidualSpan(start_idx=start_idx, end_idx=end_idx, rule=rule)
        )

    # ------------------------------------------------------------------
    # FX-based residual detection
    # ------------------------------------------------------------------

    def _detect_residual_fx(self) -> List[Tuple[int, int]]:
        """Try ``torch.fx.symbolic_trace`` to find add-based skip connections.

        Returns a list of ``(producer_rule_idx, consumer_rule_idx)`` pairs
        representing skip connections.  Falls back silently to ``[]`` if FX
        tracing fails (common for models with data-dependent control flow).
        """
        if not HAS_TORCH:
            return []
        try:
            import torch.fx as fx  # type: ignore[import]
        except ImportError:
            return []

        try:
            traced = fx.symbolic_trace(self.model)
        except Exception:
            return []

        pairs: List[Tuple[int, int]] = []
        node_list = list(traced.graph.nodes)

        # Map module target names to rule indices
        module_to_rule_idx: Dict[str, int] = {}
        rule_idx = 0
        for node in node_list:
            if node.op == "call_module" and rule_idx < len(self.rules):
                module_to_rule_idx[node.target] = rule_idx
                rule_idx += 1

        # Look for add / iadd call_function nodes
        for node in node_list:
            if node.op == "call_function":
                import operator
                if node.target in (operator.add, operator.iadd, torch.add):
                    args = node.args
                    if len(args) >= 2:
                        a_idx = self._node_to_rule_idx(args[0], module_to_rule_idx)
                        b_idx = self._node_to_rule_idx(args[1], module_to_rule_idx)
                        if a_idx is not None and b_idx is not None and a_idx != b_idx:
                            pairs.append((min(a_idx, b_idx), max(a_idx, b_idx)))
        return pairs

    @staticmethod
    def _node_to_rule_idx(
        node: Any, mapping: Dict[str, int]
    ) -> Optional[int]:
        """Resolve an FX node to its corresponding rule index."""
        if not hasattr(node, "target"):
            return None
        target = getattr(node, "target", None)
        if isinstance(target, str) and target in mapping:
            return mapping[target]
        return None

    def _apply_fx_residuals(self, pairs: List[Tuple[int, int]]) -> None:
        """Convert FX-detected ``(start, end)`` pairs into residual spans."""
        existing = {(s.start_idx, s.end_idx) for s in self.residual_spans}
        for start, end in pairs:
            if (start, end) not in existing and 0 <= start < end < len(self.rules):
                rule = ResidualRule(
                    alpha=1.0, label=f"ResidualAdd(fx) @{start}-{end}"
                )
                self.residual_spans.append(
                    _ResidualSpan(start_idx=start, end_idx=end, rule=rule)
                )

    # ------------------------------------------------------------------
    # Variance propagation
    # ------------------------------------------------------------------

    def propagate(
        self,
        q0: float = 1.0,
        width: Optional[int] = None,
        apply_finite_width: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """Propagate variance through the rule sequence.

        Parameters
        ----------
        q0 : float
            Input variance.
        width : int, optional
            Network width for finite-width corrections (ignored if
            *apply_finite_width* is False).
        apply_finite_width : bool
            Whether to add O(1/N) corrections at linear/conv layers.

        Returns
        -------
        variance_trajectory : list of float
            Predicted variance at each rule boundary (length = ``len(rules)+1``).
        chi_trajectory : list of float
            Per-rule susceptibility (length = ``len(rules)``).
        """
        q = q0
        var_traj: List[float] = [q]
        chi_traj: List[float] = []

        # Pre-compute residual span lookup: rule_idx → list of spans starting here
        span_starts: Dict[int, List[_ResidualSpan]] = {}
        span_ends: Dict[int, List[_ResidualSpan]] = {}
        for span in self.residual_spans:
            span_starts.setdefault(span.start_idx, []).append(span)
            span_ends.setdefault(span.end_idx, []).append(span)

        for i, rule in enumerate(self.rules):
            # Save skip variances for any spans opening at this index
            for span in span_starts.get(i, []):
                span.rule.save_skip(q)

            result = rule.propagate(q)
            q_new = result.q_out
            chi_local = result.chi_1

            # Apply finite-width correction at weight layers
            if apply_finite_width and width is not None and width > 0:
                if isinstance(rule, (LinearRule, ConvRule)):
                    q_new = self._apply_fw_correction(
                        rule, q, q_new, width
                    )

            # Close residual spans ending at this index
            for span in span_ends.get(i, []):
                res = span.rule.propagate(q_new)
                q_new = res.q_out
                chi_local *= res.chi_1

            q = max(q_new, 1e-30)
            var_traj.append(q)
            chi_traj.append(chi_local)

        return var_traj, chi_traj

    @staticmethod
    def _apply_fw_correction(
        rule: Union[LinearRule, ConvRule],
        q_in: float,
        q_mf: float,
        width: int,
    ) -> float:
        """Add O(1/N) finite-width correction at a weight layer.

        Uses the preceding activation's kurtosis to estimate the correction
        term ``sigma_w^4 * kappa * V(q)^2 / N``.
        """
        N = max(width, 1)
        sigma_w = rule.sigma_w
        # Use q_in as V(q) approximation (already post-activation)
        kappa = 0.5  # ReLU default; conservative
        c1 = sigma_w ** 4 * kappa * q_in ** 2 / N
        correction_ratio = abs(c1) / max(abs(q_mf), 1e-30)
        if correction_ratio > 0.3:
            c1 = math.copysign(0.3 * abs(q_mf), c1)
        return max(q_mf + c1, 1e-30)

    # ------------------------------------------------------------------
    # Per-layer chi_1 and phase
    # ------------------------------------------------------------------

    def compute_chi_total(self, chi_trajectory: List[float]) -> float:
        """Product of per-layer susceptibilities for weight layers only.

        For architectures with normalization, uses per-block analysis:
        norm layers reset variance, so the total chi is the product of
        per-block chi values (each block bounded by norm layers).
        """
        has_norm = any(isinstance(r, NormRule) for r in self.rules)
        has_residual = len(self.residual_spans) > 0

        if has_norm:
            return self._compute_chi_block_aware(chi_trajectory)
        if has_residual:
            return self._compute_chi_residual_aware(chi_trajectory)

        product = 1.0
        for i, rule in enumerate(self.rules):
            if isinstance(rule, (LinearRule, ConvRule)) and i < len(chi_trajectory):
                product *= chi_trajectory[i]
        return product

    def _compute_chi_block_aware(self, chi_trajectory: List[float]) -> float:
        """For normalized architectures, compute per-block chi and average.

        Norm layers reset variance, so the relevant metric is the
        per-block variance growth (between successive norm layers).
        """
        block_chis: List[float] = []
        current_block_chi = 1.0

        for i, rule in enumerate(self.rules):
            if i >= len(chi_trajectory):
                break
            if isinstance(rule, NormRule):
                if current_block_chi != 1.0:
                    block_chis.append(current_block_chi)
                current_block_chi = 1.0
            elif isinstance(rule, (LinearRule, ConvRule, ActivationRule)):
                current_block_chi *= chi_trajectory[i]

        if current_block_chi != 1.0:
            block_chis.append(current_block_chi)

        if not block_chis:
            return 1.0

        # For normalized architectures, use geometric mean of per-block chi
        # raised to the number of blocks (captures total depth effect)
        geo_mean = float(np.exp(np.mean(np.log(np.maximum(block_chis, 1e-30)))))
        n_blocks = len(block_chis)
        # Per-block chi near 1 is critical; report the per-block value
        # so phase classification uses per-block stability
        return geo_mean

    def _compute_chi_residual_aware(self, chi_trajectory: List[float]) -> float:
        """For residual architectures without norm, account for skip connections.

        The total chi for residual nets is typically > 1 (skip ensures
        chi >= 1), so we compute the excess-per-block as the stability metric.
        """
        if not self.residual_spans:
            product = 1.0
            for i, rule in enumerate(self.rules):
                if isinstance(rule, (LinearRule, ConvRule)) and i < len(chi_trajectory):
                    product *= chi_trajectory[i]
            return product

        # Per-block chi: product of chi within each residual span
        block_chis: List[float] = []
        covered = set()
        for span in self.residual_spans:
            block_chi = 1.0
            for i in range(span.start_idx, span.end_idx + 1):
                if i < len(self.rules) and i < len(chi_trajectory):
                    if isinstance(self.rules[i], (LinearRule, ConvRule, ActivationRule)):
                        block_chi *= chi_trajectory[i]
                    covered.add(i)
            # Residual adds 1 + alpha^2 * branch_chi
            block_chi_with_skip = 1.0 + span.rule.alpha ** 2 * block_chi
            block_chis.append(block_chi_with_skip)

        # Non-residual layers
        non_res_chi = 1.0
        for i, rule in enumerate(self.rules):
            if i not in covered and isinstance(rule, (LinearRule, ConvRule)):
                if i < len(chi_trajectory):
                    non_res_chi *= chi_trajectory[i]

        if block_chis:
            total = non_res_chi * float(np.prod(block_chis))
            n_blocks = len(block_chis)
            # Report per-block geometric mean for phase classification
            return total ** (1.0 / max(n_blocks, 1))
        return non_res_chi

    def classify_phase(self, chi_total: float, has_norm: bool = False) -> str:
        """Classify into ordered / critical / chaotic from total chi."""
        has_residual = len(self.residual_spans) > 0
        if has_norm:
            # Norm layers reset variance; per-block chi near 1 is critical
            if abs(chi_total - 1.0) < 0.3:
                return "critical"
            return "ordered" if chi_total < 1.0 else "chaotic"
        if has_residual:
            # Residual nets have chi >= 1; slightly above 1 is critical
            if chi_total < 0.9:
                return "ordered"
            elif chi_total > 2.0:
                return "chaotic"
            return "critical"
        if abs(chi_total - 1.0) < 0.1:
            return "critical"
        return "ordered" if chi_total < 1.0 else "chaotic"

    # ------------------------------------------------------------------
    # Optimal sigma_w recommendation
    # ------------------------------------------------------------------

    def recommend_sigma_w(
        self,
        chi_trajectory: List[float],
        target_chi: float = 1.0,
    ) -> Dict[str, float]:
        """Recommend per-layer sigma_w for criticality.

        Returns a dict mapping rule labels to recommended sigma_w values.
        The recommendation adjusts each weight layer's sigma_w so that its
        local chi_1 equals ``target_chi``.
        """
        recommendations: Dict[str, float] = {}
        for i, rule in enumerate(self.rules):
            if isinstance(rule, (LinearRule, ConvRule)) and i < len(chi_trajectory):
                current_chi = chi_trajectory[i]
                if current_chi > 1e-12:
                    # chi_1 ∝ sigma_w^2  →  new_sigma = old * sqrt(target / current)
                    scale = math.sqrt(target_chi / current_chi)
                    recommendations[rule.label] = rule.sigma_w * scale
                else:
                    recommendations[rule.label] = rule.sigma_w
        return recommendations

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of the rule sequence."""
        lines = [f"ComputationGraphMF — {len(self.rules)} rules, "
                 f"{len(self.residual_spans)} residual spans"]
        for i, rule in enumerate(self.rules):
            prefix = ""
            for span in self.residual_spans:
                if i == span.start_idx:
                    prefix += "┌skip "
                if i == span.end_idx:
                    prefix += "└+skip "
            lines.append(f"  [{i:3d}] {prefix}{rule!r}")
        return "\n".join(lines)


# ======================================================================
# Section 3 — Public integration function
# ======================================================================

def analyze_arbitrary_graph(
    model: "nn.Module",
    input_shape: Tuple[int, ...],
    n_samples: int = 256,
    seed: int = 42,
    input_variance: float = 1.0,
    seq_len: int = 128,
    training: bool = False,
    apply_finite_width: bool = True,
) -> GraphAnalysisResult:
    """Analyse any PyTorch model with both predicted and empirical trajectories.

    This is the primary entry point for the compositional mean-field engine.
    It combines:

    * **Predicted trajectory** — compositional rule-based mean-field
      propagation (the new engine in this module).
    * **Empirical trajectory** — forward-hook variance tracing via
      :class:`VarianceTracer` from ``graph_analyzer``.

    The result populates *both* ``predicted_variance_trajectory`` and
    ``empirical_variance_trajectory`` in the returned
    :class:`GraphAnalysisResult`, filling the gap left by the original
    ``analyze_graph`` (which always returned an empty predicted trajectory).

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model.
    input_shape : tuple of int
        Shape of a single input (excluding batch dim).
    n_samples : int
        Batch size for empirical variance estimation.
    seed : int
        Random seed for reproducibility.
    input_variance : float
        Variance of the Gaussian input distribution.
    seq_len : int
        Default sequence length for attention variance rules.
    training : bool
        Whether to model training-mode dropout scaling.
    apply_finite_width : bool
        Whether to apply O(1/N) finite-width corrections to the predicted
        trajectory.

    Returns
    -------
    GraphAnalysisResult
        Result with both predicted and empirical trajectories populated,
        per-layer chi_1, phase classification, and initialization
        recommendations.
    """
    _require_torch()
    torch.manual_seed(seed)

    # ---- 1. Build compositional rules and propagate predicted variance ----
    engine = ComputationGraphMF(model, seq_len=seq_len, training=training)

    # Estimate width from model parameters
    width = _estimate_width(model)

    predicted_var, predicted_chi = engine.propagate(
        q0=input_variance,
        width=width if apply_finite_width else None,
        apply_finite_width=apply_finite_width,
    )

    # ---- 2. Empirical variance tracing via forward hooks ----
    x = torch.randn(n_samples, *input_shape) * math.sqrt(input_variance)

    tracer = VarianceTracer()
    empirical_variances = tracer.trace(model, x)
    empirical_var_traj = [input_variance] + list(empirical_variances.values())

    # Gradient norms (best-effort)
    try:
        grad_norms = tracer.trace_gradients(model, x)
        grad_traj = list(grad_norms.values())
    except Exception:
        grad_traj = []

    # ---- 3. Classify all modules for LayerInfo ----
    layer_infos: List[LayerInfo] = []
    for name, module in model.named_modules():
        info = _classify_module(name, module)
        layer_infos.append(info)

    has_attention = any(li.is_attention for li in layer_infos)
    has_layernorm = any(li.is_norm for li in layer_infos)
    has_residual = len(engine.residual_spans) > 0 or _detect_residual_heuristic(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    depth = sum(
        1 for li in layer_infos
        if li.module_type in ("Linear", "Conv1d", "Conv2d", "Conv3d")
    )

    # ---- 4. Compute per-layer and total chi ----
    chi_total = engine.compute_chi_total(predicted_chi)

    # For normalized / residual architectures, also compute empirical chi
    # from forward-hook variance ratios and use as cross-check
    if has_layernorm or has_residual:
        emp_vals = empirical_var_traj[1:]
        if len(emp_vals) >= 2:
            emp_ratio = emp_vals[-1] / max(emp_vals[0], 1e-12)
            per_step = emp_ratio ** (1.0 / max(len(emp_vals) - 1, 1))
            # Use empirical per-step ratio for phase (more reliable for
            # normalized architectures where compositional chi breaks down)
            if per_step < 0.85:
                phase = "ordered"
            elif per_step > 1.3:
                phase = "chaotic"
            else:
                phase = "critical"
            chi_total = per_step
        else:
            phase = engine.classify_phase(chi_total, has_norm=has_layernorm)
    else:
        phase = engine.classify_phase(chi_total, has_norm=has_layernorm)

    # ---- 5. Initialization recommendations ----
    per_layer_rec = engine.recommend_sigma_w(predicted_chi)
    # Overall recommended sigma_w is the geometric mean of per-layer values
    if per_layer_rec:
        rec_values = list(per_layer_rec.values())
        recommended_sw = float(np.exp(np.mean(np.log(np.maximum(rec_values, 1e-30)))))
    else:
        recommended_sw = 1.0

    # ---- 6. Architecture summary ----
    arch_type = "Transformer" if has_attention else "Non-Transformer"
    norm_tag = "LN" if has_layernorm else "no-norm"
    skip_tag = "residual" if has_residual else "no-skip"
    arch_summary = (
        f"{arch_type} ({depth} weight layers, {n_params:,} params, "
        f"{norm_tag}, {skip_tag})"
    )

    explanation = (
        f"{arch_summary}. "
        f"Predicted total χ₁={chi_total:.4f} → phase={phase}. "
        f"Mean per-layer χ₁: "
        f"{np.mean(predicted_chi):.4f} ± {np.std(predicted_chi):.4f}. "
        f"Recommended σ_w={recommended_sw:.4f}."
    )

    return GraphAnalysisResult(
        layer_info=layer_infos,
        empirical_variance_trajectory=empirical_var_traj,
        predicted_variance_trajectory=predicted_var,
        chi_1_per_layer=predicted_chi,
        chi_1_total=chi_total,
        phase=phase,
        gradient_norm_trajectory=grad_traj,
        architecture_summary=arch_summary,
        has_attention=has_attention,
        has_layernorm=has_layernorm,
        has_residual=has_residual,
        n_params=n_params,
        depth=depth,
        recommended_sigma_w=recommended_sw,
        explanation=explanation,
    )


# ======================================================================
# Helpers
# ======================================================================

def _estimate_width(model: "nn.Module") -> int:
    """Heuristic: return the median hidden dimension across linear layers."""
    _require_torch()
    dims: List[int] = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            dims.append(m.in_features)
            dims.append(m.out_features)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            dims.append(m.out_channels)
    if not dims:
        return 512
    return int(np.median(dims))


def _detect_residual_heuristic(model: "nn.Module") -> bool:
    """Detect residual connections from module-structure heuristics."""
    _require_torch()
    module_names = {type(m).__name__ for m in model.modules()}
    if _RESIDUAL_CONTAINERS & module_names:
        return True
    cls_name = type(model).__name__.lower()
    return "resnet" in cls_name or "transformer" in cls_name
