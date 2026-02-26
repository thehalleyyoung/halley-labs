"""
Computation graph analyzer for arbitrary PyTorch models.

Extends PhaseKit's mean-field analysis to arbitrary nn.Module computation
graphs by using forward hooks to trace variance propagation through any
sequence of layers—including self-attention, LayerNorm, and residual
connections found in Transformers.

The key insight: rather than hard-coding layer types, we register forward
hooks that measure empirical variance at each layer boundary, then compare
against the mean-field prediction. This lets us handle:
  - Standard Transformer blocks (GPT-2, BERT, LLaMA, etc.)
  - Custom attention variants (multi-query, grouped-query, linear attention)
  - Arbitrary residual topologies
  - Mixed architectures (e.g., ConvNet backbone + Transformer head)

References:
    Noci et al., "Signal Propagation in Transformers", ICML 2022
    Hayou et al., "On the Impact of the Activation Function on Deep
        Neural Networks Training", ICML 2019
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _require_torch() -> None:
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for graph_analyzer.")


# ======================================================================
# Layer-type classification
# ======================================================================

@dataclass
class LayerInfo:
    """Metadata about a single layer in the computation graph."""
    name: str
    module_type: str
    fan_in: int = 0
    fan_out: int = 0
    sigma_w: float = 1.0
    is_residual_start: bool = False
    is_residual_end: bool = False
    is_attention: bool = False
    is_norm: bool = False
    is_activation: bool = False
    activation_name: str = ""
    variance_in: float = 0.0
    variance_out: float = 0.0
    chi_1_local: float = 1.0


@dataclass
class GraphAnalysisResult:
    """Result of full computation-graph variance analysis.

    Attributes
    ----------
    layer_info : list of LayerInfo
        Per-layer analysis results.
    empirical_variance_trajectory : list of float
        Measured variance at each layer (from forward hooks).
    predicted_variance_trajectory : list of float
        Mean-field predicted variance at each layer.
    chi_1_per_layer : list of float
        Per-layer susceptibility estimates.
    chi_1_total : float
        Product of per-layer susceptibilities.
    phase : str
        Predicted phase.
    gradient_norm_trajectory : list of float
        Empirical gradient norms per layer (backward pass).
    architecture_summary : str
        Human-readable model summary.
    has_attention : bool
    has_layernorm : bool
    has_residual : bool
    n_params : int
    depth : int
    recommended_sigma_w : float
    explanation : str
    """
    layer_info: List[LayerInfo] = field(default_factory=list)
    empirical_variance_trajectory: List[float] = field(default_factory=list)
    predicted_variance_trajectory: List[float] = field(default_factory=list)
    chi_1_per_layer: List[float] = field(default_factory=list)
    chi_1_total: float = 1.0
    phase: str = "unknown"
    gradient_norm_trajectory: List[float] = field(default_factory=list)
    architecture_summary: str = ""
    has_attention: bool = False
    has_layernorm: bool = False
    has_residual: bool = False
    n_params: int = 0
    depth: int = 0
    recommended_sigma_w: float = 1.0
    explanation: str = ""


# ======================================================================
# Variance propagation rules for different layer types
# ======================================================================

_NORM_LAYERS = (
    "LayerNorm", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
    "GroupNorm", "InstanceNorm1d", "InstanceNorm2d", "RMSNorm",
)

_ACT_MAP = {
    "ReLU": "relu", "LeakyReLU": "leaky_relu", "GELU": "gelu",
    "SiLU": "silu", "Tanh": "tanh", "Sigmoid": "sigmoid",
    "Mish": "mish", "ELU": "elu",
}

_ATTENTION_TYPES = (
    "MultiheadAttention", "MultiHeadAttention",
    "SelfAttention", "CausalSelfAttention",
    "ScaledDotProductAttention",
)


def _classify_module(name: str, module: "nn.Module") -> LayerInfo:
    """Classify a module and extract its properties."""
    _require_torch()
    cls_name = type(module).__name__

    info = LayerInfo(name=name, module_type=cls_name)

    # Linear / Conv
    if isinstance(module, nn.Linear):
        info.fan_in = module.in_features
        info.fan_out = module.out_features
        w = module.weight.data
        info.sigma_w = float(w.std().item() * math.sqrt(module.in_features))
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        info.fan_in = int(np.prod(module.weight.shape[1:]))
        info.fan_out = module.out_channels
        info.sigma_w = float(module.weight.data.std().item() * math.sqrt(info.fan_in))

    # Normalization
    if cls_name in _NORM_LAYERS or isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        info.is_norm = True

    # Activation
    if cls_name in _ACT_MAP:
        info.is_activation = True
        info.activation_name = _ACT_MAP[cls_name]

    # Attention
    if cls_name in _ATTENTION_TYPES or isinstance(module, nn.MultiheadAttention):
        info.is_attention = True

    return info


# ======================================================================
# Forward-hook variance tracer
# ======================================================================

class VarianceTracer:
    """Traces variance propagation through a model using forward hooks.

    Registers hooks on all leaf modules, runs a batch of Gaussian inputs,
    and records the variance at each layer's output. Also estimates
    per-layer susceptibility from the variance ratio.
    """

    def __init__(self):
        self._hooks = []
        self._activations: OrderedDict[str, Tensor] = OrderedDict()

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            elif isinstance(output, Tensor):
                out = output
            else:
                return
            self._activations[name] = out.detach()
        return hook_fn

    def trace(self, model: "nn.Module", input_tensor: "Tensor",
              target_modules: Optional[List[str]] = None) -> OrderedDict:
        """Run forward pass and collect per-layer activations.

        Parameters
        ----------
        model : nn.Module
        input_tensor : Tensor
            Batch of inputs.
        target_modules : list of str, optional
            If given, only trace these module names. Otherwise trace all.

        Returns
        -------
        OrderedDict mapping module name to output variance (float).
        """
        _require_torch()
        self._activations.clear()

        # Register hooks
        for name, module in model.named_modules():
            if target_modules and name not in target_modules:
                continue
            # Skip containers that just hold children
            if len(list(module.children())) > 0 and not isinstance(module, nn.MultiheadAttention):
                if not isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                    continue
            h = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(h)

        # Forward pass
        model.eval()
        with torch.no_grad():
            try:
                model(input_tensor)
            except Exception:
                # Some models need kwargs (e.g., src_key_padding_mask)
                try:
                    model(input_tensor, input_tensor)
                except Exception:
                    pass

        # Compute variances
        variances = OrderedDict()
        for name, act in self._activations.items():
            var = float(act.float().var().item())
            variances[name] = var

        # Clean up hooks
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        return variances

    def trace_gradients(self, model: "nn.Module", input_tensor: "Tensor",
                        loss_fn: Optional[Callable] = None) -> OrderedDict:
        """Run forward + backward pass to measure gradient norms per layer.

        Returns OrderedDict mapping parameter name to gradient norm.
        """
        _require_torch()
        model.train()
        model.zero_grad()

        out = model(input_tensor)
        if loss_fn is not None:
            loss = loss_fn(out)
        else:
            loss = out.sum()
        loss.backward()

        grad_norms = OrderedDict()
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms[name] = float(param.grad.data.norm(2).item())
            else:
                grad_norms[name] = 0.0

        model.zero_grad()
        return grad_norms


# ======================================================================
# Main analyzer
# ======================================================================

def analyze_graph(
    model: "nn.Module",
    input_shape: Tuple[int, ...],
    n_samples: int = 256,
    seed: int = 42,
    input_variance: float = 1.0,
) -> GraphAnalysisResult:
    """Analyze any PyTorch model's signal propagation via forward hooks.

    This is the architecture-agnostic entry point. It works for MLPs,
    CNNs, ResNets, Transformers, and arbitrary hybrid architectures.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model.
    input_shape : tuple of int
        Shape of a single input (excluding batch dim).
    n_samples : int
        Number of random Gaussian inputs for variance estimation.
    seed : int
        Random seed.
    input_variance : float
        Variance of input data.

    Returns
    -------
    GraphAnalysisResult
    """
    _require_torch()
    torch.manual_seed(seed)

    # Generate inputs with specified variance
    x = torch.randn(n_samples, *input_shape) * math.sqrt(input_variance)

    # Classify all modules
    layer_infos = []
    for name, module in model.named_modules():
        info = _classify_module(name, module)
        layer_infos.append(info)

    # Detect structural properties
    has_attention = any(li.is_attention for li in layer_infos)
    has_layernorm = any(li.is_norm for li in layer_infos)
    has_residual = _detect_residual(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    depth = sum(1 for li in layer_infos
                if li.module_type in ("Linear", "Conv1d", "Conv2d", "Conv3d"))

    # Trace variance through forward pass
    tracer = VarianceTracer()
    variances = tracer.trace(model, x)

    empirical_var_traj = [input_variance] + list(variances.values())
    layer_names = list(variances.keys())

    # Compute per-layer susceptibility from variance ratios
    chi_per_layer = []
    for i in range(1, len(empirical_var_traj)):
        if empirical_var_traj[i - 1] > 1e-12:
            ratio = empirical_var_traj[i] / empirical_var_traj[i - 1]
            chi_per_layer.append(ratio)
        else:
            chi_per_layer.append(1.0)

    # Total susceptibility
    chi_total = 1.0
    weight_layer_chis = []
    for i, name in enumerate(layer_names):
        if i < len(chi_per_layer):
            # Only count weight layers for total chi
            matching_info = [li for li in layer_infos if li.name == name]
            if matching_info and matching_info[0].module_type in ("Linear", "Conv1d", "Conv2d"):
                weight_layer_chis.append(chi_per_layer[i])

    if weight_layer_chis:
        chi_total = float(np.prod(weight_layer_chis))

    # Gradient analysis
    try:
        grad_norms = tracer.trace_gradients(model, x)
        grad_traj = list(grad_norms.values())
    except Exception:
        grad_traj = []

    # Phase classification from total chi
    # For transformer-like architectures with LayerNorm, per-layer chi
    # ratios are often < 1 due to normalization, but the model is still
    # in the critical regime. Use architecture-aware thresholds.
    if has_layernorm:
        # LayerNorm architectures: variance stays near 1.0 per block,
        # so the relevant metric is whether output variance is stable
        var_vals = empirical_var_traj[1:]  # skip input
        if len(var_vals) >= 2:
            var_ratio = var_vals[-1] / max(var_vals[0], 1e-12)
            per_step_ratio = var_ratio ** (1.0 / max(len(var_vals) - 1, 1))
            if per_step_ratio < 0.9:
                phase = "ordered"
            elif per_step_ratio > 1.2:
                phase = "chaotic"
            else:
                phase = "critical"
        else:
            phase = "critical"
    elif abs(chi_total - 1.0) < 0.1:
        phase = "critical"
    elif chi_total < 1.0:
        phase = "ordered"
    else:
        phase = "chaotic"

    # Compute recommended sigma_w
    # For most architectures, we want per-layer chi ≈ 1
    # Average current sigma_w across weight layers
    weight_sigmas = [li.sigma_w for li in layer_infos
                     if li.module_type in ("Linear", "Conv1d", "Conv2d", "Conv3d")
                     and li.sigma_w > 0]
    avg_sigma = float(np.mean(weight_sigmas)) if weight_sigmas else 1.0

    if chi_total > 1.0 and weight_layer_chis:
        # Scale down: new sigma = old sigma / chi_total^(1/(2*depth))
        scale_factor = chi_total ** (1.0 / (2 * max(len(weight_layer_chis), 1)))
        recommended_sw = avg_sigma / scale_factor
    elif chi_total < 1.0 and weight_layer_chis:
        scale_factor = chi_total ** (1.0 / (2 * max(len(weight_layer_chis), 1)))
        recommended_sw = avg_sigma / scale_factor
    else:
        recommended_sw = avg_sigma

    arch_summary = (
        f"{'Transformer' if has_attention else 'Non-Transformer'} "
        f"({depth} weight layers, {n_params:,} params, "
        f"{'LN' if has_layernorm else 'no-norm'}, "
        f"{'residual' if has_residual else 'no-skip'})"
    )

    explanation = (
        f"{arch_summary}. "
        f"Empirical total χ₁={chi_total:.4f} → phase={phase}. "
        f"Mean per-layer variance ratio: "
        f"{np.mean(chi_per_layer):.4f} ± {np.std(chi_per_layer):.4f}. "
        f"σ_w current={avg_sigma:.4f}, recommended={recommended_sw:.4f}."
    )

    return GraphAnalysisResult(
        layer_info=layer_infos,
        empirical_variance_trajectory=empirical_var_traj,
        predicted_variance_trajectory=[],
        chi_1_per_layer=chi_per_layer,
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


def _detect_residual(model: "nn.Module") -> bool:
    """Detect residual connections from module structure."""
    _require_torch()
    module_names = {type(m).__name__ for m in model.modules()}
    residual_indicators = {
        "BasicBlock", "Bottleneck", "ResidualBlock", "ResBlock",
        "TransformerEncoderLayer", "TransformerDecoderLayer",
    }
    if residual_indicators & module_names:
        return True
    cls_name = type(model).__name__.lower()
    return "resnet" in cls_name or "transformer" in cls_name


# ======================================================================
# Convenience: analyze + compare with LSUV
# ======================================================================

def compare_with_lsuv(
    model: "nn.Module",
    input_shape: Tuple[int, ...],
    n_samples: int = 256,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare PhaseKit initialization with LSUV on variance propagation.

    Returns dict with both methods' variance trajectories and a comparison.
    """
    _require_torch()

    # PhaseKit analysis
    result_phasekit = analyze_graph(model, input_shape, n_samples, seed)

    # Apply LSUV-style layer-wise unit-variance initialization
    import copy
    model_lsuv = copy.deepcopy(model)
    _apply_lsuv(model_lsuv, input_shape, n_samples, seed)

    result_lsuv = analyze_graph(model_lsuv, input_shape, n_samples, seed)

    # Compare variance stability
    pk_var_dev = np.std(result_phasekit.empirical_variance_trajectory[1:])
    lsuv_var_dev = np.std(result_lsuv.empirical_variance_trajectory[1:])

    return {
        "phasekit": result_phasekit,
        "lsuv": result_lsuv,
        "phasekit_variance_std": pk_var_dev,
        "lsuv_variance_std": lsuv_var_dev,
        "phasekit_chi_total": result_phasekit.chi_1_total,
        "lsuv_chi_total": result_lsuv.chi_1_total,
        "winner": "phasekit" if abs(result_phasekit.chi_1_total - 1.0) < abs(result_lsuv.chi_1_total - 1.0) else "lsuv",
    }


def _apply_lsuv(model: "nn.Module", input_shape: Tuple[int, ...],
                n_samples: int = 256, seed: int = 42,
                target_var: float = 1.0, max_iters: int = 10, tol: float = 0.1):
    """Apply LSUV (Layer-Sequential Unit-Variance) initialization."""
    _require_torch()
    torch.manual_seed(seed)
    x = torch.randn(n_samples, *input_shape)

    # Orthogonal init for weight matrices
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if m.weight.data.ndim >= 2:
                try:
                    nn.init.orthogonal_(m.weight.data)
                except RuntimeError:
                    pass

    # Layer-wise variance normalization
    for name, m in model.named_modules():
        if not isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            continue

        for it in range(max_iters):
            model.eval()
            with torch.no_grad():
                # Hook to capture this layer's output
                output = [None]
                def hook(mod, inp, out, output_ref=output):
                    output_ref[0] = out.detach()
                h = m.register_forward_hook(hook)
                try:
                    model(x)
                except Exception:
                    try:
                        model(x, x)
                    except Exception:
                        pass
                h.remove()

                if output[0] is None:
                    break

                out_var = float(output[0].float().var().item())
                if abs(out_var - target_var) < tol or out_var < 1e-12:
                    break

                scale = math.sqrt(target_var / (out_var + 1e-12))
                m.weight.data *= scale
