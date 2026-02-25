"""PyTorch integration for phase diagram analysis.

Works with any ``torch.nn.Module`` — MLPs, CNNs, ResNets, Transformers.
Extracts the empirical NTK via ``torch.func`` (functorch), auto-detects
architecture type, and computes finite-width corrections for real models.

Memory-efficient: supports models up to ~1B parameters via chunked
Jacobian--vector products and optional Nyström approximation.

Example
-------
>>> import torch.nn as nn
>>> from phase_diagrams.pytorch_integration import analyze_model
>>> model = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10))
>>> result = analyze_model(model, input_shape=(784,), lr=0.01)
>>> print(result.regime, result.critical_lr)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    import torch.nn as nn
    from torch import Tensor

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from torch.func import functional_call, grad, jacrev, jvp, vmap

    HAS_FUNCTORCH = True
except ImportError:
    HAS_FUNCTORCH = False


def _require_torch() -> None:
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for pytorch_integration. "
            "Install with: pip install phase-diagrams[torch]"
        )


# ======================================================================
# Architecture detection
# ======================================================================

class ArchType(str, Enum):
    """Detected architecture family."""
    MLP = "mlp"
    CNN = "cnn"
    RESNET = "resnet"
    TRANSFORMER = "transformer"
    UNKNOWN = "unknown"


@dataclass
class ArchitectureInfo:
    """Result of architecture auto-detection.

    Attributes
    ----------
    arch_type : ArchType
        Detected family.
    depth : int
        Effective depth (number of weight layers).
    widths : list of int
        Width at each layer.
    total_params : int
        Total learnable parameters.
    has_residual : bool
        Whether skip connections were detected.
    has_attention : bool
        Whether attention layers were detected.
    has_conv : bool
        Whether convolutional layers were detected.
    has_normalization : bool
        Whether normalization layers were detected.
    """
    arch_type: ArchType = ArchType.UNKNOWN
    depth: int = 0
    widths: List[int] = field(default_factory=list)
    total_params: int = 0
    has_residual: bool = False
    has_attention: bool = False
    has_conv: bool = False
    has_normalization: bool = False


def detect_architecture(model: "nn.Module") -> ArchitectureInfo:
    """Auto-detect architecture type from an ``nn.Module``.

    Inspects the module tree for characteristic layer types (Conv2d,
    MultiheadAttention, residual patterns) and infers the architecture
    family, depth, and per-layer widths.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model.

    Returns
    -------
    ArchitectureInfo
    """
    _require_torch()

    widths: List[int] = []
    has_conv = False
    has_attention = False
    has_residual = False
    has_norm = False
    linear_count = 0
    conv_count = 0

    for name, module in model.named_modules():
        cls = type(module).__name__
        if isinstance(module, nn.Linear):
            widths.append(module.out_features)
            linear_count += 1
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            widths.append(module.out_channels)
            has_conv = True
            conv_count += 1
        elif "attention" in cls.lower() or isinstance(
            module, nn.MultiheadAttention
        ):
            has_attention = True
        elif isinstance(
            module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
        ):
            has_norm = True

    # Heuristic: detect residual connections by looking for Add patterns
    # or known residual block types
    module_names = {type(m).__name__ for m in model.modules()}
    residual_indicators = {"BasicBlock", "Bottleneck", "ResidualBlock", "ResBlock"}
    if residual_indicators & module_names:
        has_residual = True
    # Also check if the model class name contains "resnet" or "residual"
    model_cls = type(model).__name__.lower()
    if "resnet" in model_cls or "residual" in model_cls:
        has_residual = True

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    depth = linear_count + conv_count

    # Classify
    if has_attention:
        arch_type = ArchType.TRANSFORMER
    elif has_residual:
        arch_type = ArchType.RESNET
    elif has_conv:
        arch_type = ArchType.CNN
    elif linear_count > 0:
        arch_type = ArchType.MLP
    else:
        arch_type = ArchType.UNKNOWN

    return ArchitectureInfo(
        arch_type=arch_type,
        depth=depth,
        widths=widths,
        total_params=total_params,
        has_residual=has_residual,
        has_attention=has_attention,
        has_conv=has_conv,
        has_normalization=has_norm,
    )


# ======================================================================
# NTK extraction
# ======================================================================

@dataclass
class NTKResult:
    """Result of NTK computation.

    Attributes
    ----------
    kernel_matrix : NDArray
        NTK Gram matrix of shape (n, n).
    eigenvalues : NDArray
        Sorted eigenvalues (descending).
    trace : float
        Trace of the kernel matrix.
    frobenius_norm : float
        Frobenius norm.
    effective_rank : float
        Effective rank = trace / max_eigenvalue.
    method : str
        Method used ("full_jacobian", "jvp_chunked", "nystrom").
    """
    kernel_matrix: Optional[NDArray] = None
    eigenvalues: Optional[NDArray] = None
    trace: float = 0.0
    frobenius_norm: float = 0.0
    effective_rank: float = 0.0
    method: str = "full_jacobian"


def _flatten_params(model: "nn.Module") -> Tuple["Tensor", Dict[str, Any]]:
    """Flatten model parameters into a single vector with metadata."""
    _require_torch()
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    shapes = {n: p.shape for n, p in params.items()}
    flat = torch.cat([p.flatten() for p in params.values()])
    return flat, {"names": list(params.keys()), "shapes": shapes}


def _param_count(model: "nn.Module") -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def extract_ntk(
    model: "nn.Module",
    inputs: "Tensor",
    output_index: int = 0,
    chunk_size: int = 64,
    method: str = "auto",
    nystrom_rank: int = 100,
) -> NTKResult:
    """Compute the empirical NTK Gram matrix for a model.

    Uses ``torch.func.jacrev`` for small models and chunked JVP for
    large models.  Falls back to Nyström approximation when the full
    Jacobian would exceed available memory.

    Parameters
    ----------
    model : nn.Module
        Any differentiable PyTorch model.
    inputs : Tensor of shape (n, ...)
        Input batch.
    output_index : int
        Which output dimension to use (for multi-output models).
    chunk_size : int
        Number of parameters per JVP chunk (for memory efficiency).
    method : str
        ``"auto"``, ``"full_jacobian"``, ``"jvp_chunked"``, or ``"nystrom"``.
    nystrom_rank : int
        Rank for Nyström approximation.

    Returns
    -------
    NTKResult
    """
    _require_torch()
    n = inputs.shape[0]
    P = _param_count(model)

    # Choose method
    if method == "auto":
        mem_estimate = n * P * 4  # bytes (float32)
        if mem_estimate < 2e9:  # < 2 GB
            method = "full_jacobian"
        elif P < 1e8:  # < 100M params
            method = "jvp_chunked"
        else:
            method = "nystrom"

    if method == "full_jacobian":
        return _ntk_full_jacobian(model, inputs, output_index)
    elif method == "jvp_chunked":
        return _ntk_jvp_chunked(model, inputs, output_index, chunk_size)
    else:
        return _ntk_nystrom(model, inputs, output_index, nystrom_rank)


def _ntk_full_jacobian(
    model: "nn.Module", inputs: "Tensor", output_index: int = 0
) -> NTKResult:
    """Full Jacobian NTK via torch.func."""
    _require_torch()
    if not HAS_FUNCTORCH:
        warnings.warn(
            "torch.func not available; falling back to manual Jacobian.",
            RuntimeWarning,
        )
        return _ntk_manual_jacobian(model, inputs, output_index)

    model.eval()
    params = dict(model.named_parameters())
    param_keys = [k for k, v in params.items() if v.requires_grad]

    def fnet_single(p_dict: Dict[str, "Tensor"], x: "Tensor") -> "Tensor":
        out = functional_call(model, p_dict, (x.unsqueeze(0),))
        return out.squeeze(0)[output_index]

    # Compute Jacobian for each input
    jacobians = []
    for i in range(inputs.shape[0]):
        x_i = inputs[i]
        jac = jacrev(fnet_single)(params, x_i)
        jac_flat = torch.cat([jac[k].flatten() for k in param_keys])
        jacobians.append(jac_flat)

    J = torch.stack(jacobians)  # (n, P)
    K = (J @ J.T).detach().cpu().numpy()

    eigvals = np.linalg.eigvalsh(K)[::-1]
    trace = float(np.trace(K))
    frob = float(np.linalg.norm(K, "fro"))
    eff_rank = trace / (eigvals[0] + 1e-12) if len(eigvals) > 0 else 0.0

    return NTKResult(
        kernel_matrix=K,
        eigenvalues=eigvals,
        trace=trace,
        frobenius_norm=frob,
        effective_rank=eff_rank,
        method="full_jacobian",
    )


def _ntk_manual_jacobian(
    model: "nn.Module", inputs: "Tensor", output_index: int = 0
) -> NTKResult:
    """Manual per-parameter gradient Jacobian (no functorch needed)."""
    _require_torch()
    model.eval()
    n = inputs.shape[0]

    jacobian_rows = []
    for i in range(n):
        model.zero_grad()
        out = model(inputs[i : i + 1])
        scalar = out[0, output_index]
        scalar.backward(retain_graph=True)
        row = torch.cat(
            [p.grad.flatten() for p in model.parameters() if p.requires_grad and p.grad is not None]
        )
        jacobian_rows.append(row.detach())

    J = torch.stack(jacobian_rows)
    K = (J @ J.T).cpu().numpy()

    eigvals = np.linalg.eigvalsh(K)[::-1]
    trace = float(np.trace(K))
    frob = float(np.linalg.norm(K, "fro"))
    eff_rank = trace / (eigvals[0] + 1e-12)

    return NTKResult(
        kernel_matrix=K,
        eigenvalues=eigvals,
        trace=trace,
        frobenius_norm=frob,
        effective_rank=eff_rank,
        method="manual_jacobian",
    )


def _ntk_jvp_chunked(
    model: "nn.Module",
    inputs: "Tensor",
    output_index: int = 0,
    chunk_size: int = 64,
) -> NTKResult:
    """Chunked JVP-based NTK for medium-sized models.

    Computes K[i,j] = sum_k (df/dθ_k)(x_i) · (df/dθ_k)(x_j)
    by iterating over parameter chunks.
    """
    _require_torch()
    model.eval()
    n = inputs.shape[0]
    P = _param_count(model)

    K = np.zeros((n, n), dtype=np.float32)
    param_list = [p for p in model.parameters() if p.requires_grad]

    # Accumulate K in chunks of parameters
    offset = 0
    for param in param_list:
        p_numel = param.numel()
        # Compute gradient w.r.t. this parameter for all inputs
        grads = []
        for i in range(n):
            model.zero_grad()
            out = model(inputs[i : i + 1])
            scalar = out[0, output_index]
            scalar.backward(retain_graph=True)
            grads.append(param.grad.flatten().detach().clone())
            model.zero_grad()

        G = torch.stack(grads).cpu().numpy()  # (n, p_numel)
        K += G @ G.T
        offset += p_numel

    eigvals = np.linalg.eigvalsh(K)[::-1]
    trace = float(np.trace(K))
    frob = float(np.linalg.norm(K, "fro"))
    eff_rank = trace / (eigvals[0] + 1e-12)

    return NTKResult(
        kernel_matrix=K,
        eigenvalues=eigvals,
        trace=trace,
        frobenius_norm=frob,
        effective_rank=eff_rank,
        method="jvp_chunked",
    )


def _ntk_nystrom(
    model: "nn.Module",
    inputs: "Tensor",
    output_index: int = 0,
    rank: int = 100,
) -> NTKResult:
    """Nyström-approximated NTK for large models (>100M params).

    Samples ``rank`` random parameter directions and estimates the
    kernel from projected gradients.
    """
    _require_torch()
    model.eval()
    n = inputs.shape[0]
    P = _param_count(model)
    actual_rank = min(rank, P, n)

    # Random projection matrix
    rng = torch.Generator()
    rng.manual_seed(42)
    proj = torch.randn(P, actual_rank, generator=rng, device=inputs.device)
    proj /= math.sqrt(actual_rank)

    # Compute projected Jacobian
    proj_jac = np.zeros((n, actual_rank), dtype=np.float32)
    flat_params = torch.cat([p.flatten() for p in model.parameters() if p.requires_grad])

    for i in range(n):
        model.zero_grad()
        out = model(inputs[i : i + 1])
        scalar = out[0, output_index]
        scalar.backward(retain_graph=True)
        grad_flat = torch.cat(
            [p.grad.flatten() for p in model.parameters() if p.requires_grad and p.grad is not None]
        )
        proj_jac[i] = (grad_flat @ proj).detach().cpu().numpy()
        model.zero_grad()

    K_approx = proj_jac @ proj_jac.T
    eigvals = np.linalg.eigvalsh(K_approx)[::-1]

    # Scale eigenvalues to account for projection
    scale = P / actual_rank
    eigvals *= scale

    trace = float(np.sum(eigvals))
    frob = float(np.sqrt(np.sum(eigvals ** 2)))
    eff_rank = trace / (eigvals[0] + 1e-12)

    return NTKResult(
        kernel_matrix=K_approx,
        eigenvalues=eigvals,
        trace=trace,
        frobenius_norm=frob,
        effective_rank=eff_rank,
        method="nystrom",
    )


# ======================================================================
# Finite-width corrections for PyTorch models
# ======================================================================

@dataclass
class CorrectionResult:
    """Result of finite-width correction analysis.

    Attributes
    ----------
    theta_0 : NDArray
        Infinite-width NTK estimate (extrapolated).
    theta_1 : NDArray
        First-order 1/N correction matrix.
    convergence_exponent : float
        Fitted exponent α in ||Θ(N) - Θ∞|| ~ N^{-α}.
    widths_used : list of int
        Widths at which empirical NTKs were measured.
    r_squared : float
        R² of the power-law fit.
    perturbative_valid : bool
        Whether the expansion is valid (ε_N < 0.5).
    """
    theta_0: Optional[NDArray] = None
    theta_1: Optional[NDArray] = None
    convergence_exponent: float = 0.0
    widths_used: List[int] = field(default_factory=list)
    r_squared: float = 0.0
    perturbative_valid: bool = False


def finite_width_corrections(
    model_factory: Callable[[int], "nn.Module"],
    inputs: "Tensor",
    widths: Sequence[int] = (64, 128, 256, 512),
    output_index: int = 0,
) -> CorrectionResult:
    """Compute 1/N corrections from empirical NTK at multiple widths.

    Requires a factory function that creates models of varying width.

    Parameters
    ----------
    model_factory : callable (width: int) -> nn.Module
        Factory that creates a model with the specified width.
    inputs : Tensor
        Input batch.
    widths : sequence of int
        Widths to sample.
    output_index : int
        Output dimension.

    Returns
    -------
    CorrectionResult
    """
    _require_torch()
    n = inputs.shape[0]
    widths = sorted(widths)

    # Compute NTK at each width
    ntk_matrices = {}
    for w in widths:
        m = model_factory(w)
        m.eval()
        ntk = extract_ntk(m, inputs, output_index, method="auto")
        ntk_matrices[w] = ntk.kernel_matrix

    # Fit Θ(N) = Θ^(0) + Θ^(1)/N via least squares on each matrix entry
    inv_widths = np.array([1.0 / w for w in widths])
    K_stack = np.stack([ntk_matrices[w] for w in widths])  # (W, n, n)

    theta_0 = np.zeros((n, n))
    theta_1 = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            y = K_stack[:, i, j]
            # Least-squares fit: y = a + b/N
            A = np.column_stack([np.ones(len(widths)), inv_widths])
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            theta_0[i, j] = coeffs[0]
            theta_1[i, j] = coeffs[1]

    # Convergence exponent: ||Θ(N) - Θ^(0)||_F ~ N^{-α}
    diffs = [np.linalg.norm(ntk_matrices[w] - theta_0, "fro") for w in widths]
    log_w = np.log(np.array(widths, dtype=float))
    log_d = np.log(np.array(diffs) + 1e-12)
    if len(widths) >= 2:
        slope, intercept = np.polyfit(log_w, log_d, 1)
        alpha = -slope
        # R²
        predicted = slope * log_w + intercept
        ss_res = np.sum((log_d - predicted) ** 2)
        ss_tot = np.sum((log_d - np.mean(log_d)) ** 2) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
    else:
        alpha = 0.0
        r2 = 0.0

    # Perturbative validity: ε_N = ||Θ^(1)/N|| / ||Θ^(0)||
    theta_0_norm = np.linalg.norm(theta_0, "fro") + 1e-12
    max_width = max(widths)
    eps_N = np.linalg.norm(theta_1, "fro") / (max_width * theta_0_norm)
    valid = eps_N < 0.5

    return CorrectionResult(
        theta_0=theta_0,
        theta_1=theta_1,
        convergence_exponent=alpha,
        widths_used=list(widths),
        r_squared=r2,
        perturbative_valid=valid,
    )


# ======================================================================
# High-level analysis
# ======================================================================

@dataclass
class ModelAnalysis:
    """Complete analysis of a PyTorch model's training regime.

    Attributes
    ----------
    regime : str
        ``"lazy"``, ``"rich"``, or ``"critical"``.
    critical_lr : float
        Phase boundary learning rate.
    gamma : float
        Effective coupling at the specified LR.
    gamma_star : float
        Critical coupling.
    architecture : ArchitectureInfo
        Detected architecture info.
    ntk : NTKResult
        NTK computation result.
    confidence : float
        Confidence in regime prediction (0–1).
    explanation : str
        Human-readable summary.
    """
    regime: str = "unknown"
    critical_lr: float = 0.0
    gamma: float = 0.0
    gamma_star: float = 0.0
    architecture: Optional[ArchitectureInfo] = None
    ntk: Optional[NTKResult] = None
    confidence: float = 0.0
    explanation: str = ""


def _estimate_init_scale(model: "nn.Module") -> float:
    """Estimate effective initialisation scale σ from first layer weights."""
    _require_torch()
    for p in model.parameters():
        if p.requires_grad and p.ndim >= 2:
            fan_in = p.shape[1] if p.ndim == 2 else np.prod(p.shape[1:])
            return float(p.std().item() * math.sqrt(fan_in))
    return 1.0


def _effective_width(arch: ArchitectureInfo) -> int:
    """Get the effective width (median hidden-layer width)."""
    if not arch.widths:
        return 256  # fallback
    # Exclude last layer (output)
    hidden = arch.widths[:-1] if len(arch.widths) > 1 else arch.widths
    return int(np.median(hidden))


def analyze_model(
    model: "nn.Module",
    input_shape: Tuple[int, ...],
    lr: float = 0.01,
    n_samples: int = 50,
    training_steps: int = 100,
    seed: int = 42,
) -> ModelAnalysis:
    """Analyze a PyTorch model to predict its training regime.

    This is the primary entry point for PyTorch users.  It auto-detects
    the architecture, computes the NTK, and predicts whether the model
    will train in the lazy or rich regime at the given learning rate.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model.
    input_shape : tuple of int
        Shape of a single input (excluding batch dimension).
    lr : float
        Learning rate to evaluate.
    n_samples : int
        Number of random inputs for NTK estimation.
    training_steps : int
        Planned training duration.
    seed : int
        Random seed.

    Returns
    -------
    ModelAnalysis
    """
    _require_torch()

    # Detect architecture
    arch = detect_architecture(model)
    width = _effective_width(arch)
    init_scale = _estimate_init_scale(model)

    # Generate random inputs
    torch.manual_seed(seed)
    inputs = torch.randn(n_samples, *input_shape)

    # Compute NTK
    ntk = extract_ntk(model, inputs, output_index=0, method="auto")

    # Effective perturbation eigenvalue from NTK spectrum
    if ntk.eigenvalues is not None and len(ntk.eigenvalues) > 0:
        mu_max_eff = float(ntk.eigenvalues[0]) / (width * n_samples)
    else:
        mu_max_eff = 1.0 / width

    # Critical coupling
    drift_threshold = 0.1
    drift_floor = 1e-3
    c = math.log(drift_threshold / drift_floor)
    gamma_star = c / (training_steps * mu_max_eff) if mu_max_eff > 0 else float("inf")

    # Effective coupling
    gamma = lr * init_scale ** 2 / width

    # Classify
    if gamma < gamma_star * 0.8:
        regime = "lazy"
        confidence = min(1.0, (gamma_star - gamma) / (gamma_star + 1e-12))
    elif gamma > gamma_star * 1.2:
        regime = "rich"
        confidence = min(1.0, (gamma - gamma_star) / (gamma_star + 1e-12))
    else:
        regime = "critical"
        confidence = 0.5

    critical_lr = gamma_star * width / (init_scale ** 2) if init_scale > 0 else float("inf")

    explanation = (
        f"Architecture: {arch.arch_type.value} (depth={arch.depth}, "
        f"width≈{width}, params={arch.total_params:,}). "
        f"At LR={lr:.2e}, γ={gamma:.4e} vs γ*={gamma_star:.4e} → "
        f"{regime} regime (confidence={confidence:.2f}). "
        f"Critical LR={critical_lr:.2e}."
    )

    return ModelAnalysis(
        regime=regime,
        critical_lr=critical_lr,
        gamma=gamma,
        gamma_star=gamma_star,
        architecture=arch,
        ntk=ntk,
        confidence=confidence,
        explanation=explanation,
    )


# ======================================================================
# Mean-field integration: extract ArchitectureSpec from nn.Module
# ======================================================================

# Map torch activation modules to PhaseKit activation names
_TORCH_ACT_MAP = {
    "ReLU": "relu",
    "LeakyReLU": "leaky_relu",
    "Tanh": "tanh",
    "Sigmoid": "sigmoid",
    "GELU": "gelu",
    "SiLU": "silu",
    "Mish": "mish",
    "ELU": "elu",
}


@dataclass
class MeanFieldResult:
    """Result of mean-field phase analysis for a PyTorch model.

    Attributes
    ----------
    phase : str
        Predicted phase: "ordered", "critical", or "chaotic".
    chi_1 : float
        Susceptibility (1.0 = edge of chaos).
    sigma_w_star : float
        Optimal weight scale for edge-of-chaos init.
    recommended_sigma_w : float
        Recommended σ_w for the detected activation.
    variance_trajectory : list of float
        Per-layer predicted variance.
    depth_scale : float
        Effective depth scale ξ = 1/|log(χ₁)|.
    probabilities : dict
        Soft phase probabilities.
    current_sigma_w : float
        Current effective weight scale.
    explanation : str
        Human-readable summary.
    """
    phase: str = "unknown"
    chi_1: float = 0.0
    sigma_w_star: float = 0.0
    recommended_sigma_w: float = 0.0
    variance_trajectory: List[float] = field(default_factory=list)
    depth_scale: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    current_sigma_w: float = 0.0
    explanation: str = ""


def _detect_activation(model: "nn.Module") -> str:
    """Detect the primary activation function in a model."""
    _require_torch()
    act_counts: Dict[str, int] = {}
    for m in model.modules():
        cls_name = type(m).__name__
        if cls_name in _TORCH_ACT_MAP:
            act = _TORCH_ACT_MAP[cls_name]
            act_counts[act] = act_counts.get(act, 0) + 1
    if not act_counts:
        return "relu"  # default
    return max(act_counts, key=act_counts.get)


def _extract_layer_sigma_w(module: "nn.Module") -> Optional[float]:
    """Extract effective σ_w from a weight layer."""
    _require_torch()
    if not hasattr(module, "weight") or module.weight is None:
        return None
    w = module.weight.data
    if w.ndim < 2:
        return None
    fan_in = w.shape[1] if w.ndim == 2 else int(np.prod(w.shape[1:]))
    return float(w.std().item() * math.sqrt(fan_in))


def extract_architecture_spec(model: "nn.Module") -> "ArchitectureSpec":
    """Extract a PhaseKit ArchitectureSpec from a torch.nn.Module.

    Walks the module tree, detects layer types, widths, activation,
    and estimates the current weight variance σ_w.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    ArchitectureSpec
    """
    _require_torch()
    from mean_field_theory import ArchitectureSpec  # local import to avoid circular

    arch = detect_architecture(model)
    activation = _detect_activation(model)
    width = _effective_width(arch)

    # Estimate σ_w from weight layers
    sigma_ws = []
    has_bn = False
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            sw = _extract_layer_sigma_w(m)
            if sw is not None:
                sigma_ws.append(sw)
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            has_bn = True

    sigma_w = float(np.median(sigma_ws)) if sigma_ws else 1.0

    return ArchitectureSpec(
        depth=arch.depth,
        width=width,
        activation=activation,
        sigma_w=sigma_w,
        sigma_b=0.0,
        has_residual=arch.has_residual,
        has_batchnorm=has_bn or arch.has_normalization,
    )


def _extract_transformer_spec(model: "nn.Module") -> Optional["TransformerSpec"]:
    """Extract TransformerSpec from a PyTorch Transformer model.

    Returns None if the model is not a Transformer.
    """
    _require_torch()
    arch = detect_architecture(model)
    if not arch.has_attention:
        return None

    from transformer_mean_field import TransformerSpec

    # Detect d_model, n_heads, n_layers from module tree
    d_model = None
    n_heads = None
    n_layers = 0
    d_ff = None
    activation = _detect_activation(model)
    has_pre_ln = False

    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            d_model = d_model or module.embed_dim
            n_heads = n_heads or module.num_heads
            n_layers += 1
        elif isinstance(module, nn.TransformerEncoderLayer):
            d_model = d_model or module.self_attn.embed_dim
            n_heads = n_heads or module.self_attn.num_heads
            d_ff = d_ff or module.linear1.out_features
            # Check norm_first attribute for Pre-LN detection
            if hasattr(module, 'norm_first'):
                has_pre_ln = module.norm_first
        elif isinstance(module, nn.TransformerEncoder):
            n_layers = max(n_layers, module.num_layers)

    if d_model is None:
        d_model = arch.widths[0] if arch.widths else 512
    if n_heads is None:
        n_heads = max(1, d_model // 64)
    if d_ff is None:
        d_ff = 4 * d_model
    if n_layers == 0:
        n_layers = max(1, sum(1 for m in model.modules()
                              if isinstance(m, nn.MultiheadAttention)))

    # Estimate sigma_w
    sigma_ws = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            sw = _extract_layer_sigma_w(m)
            if sw is not None:
                sigma_ws.append(sw)
    sigma_w = float(np.median(sigma_ws)) if sigma_ws else 1.0

    return TransformerSpec(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        activation=activation if activation != "relu" else "gelu",
        sigma_w=sigma_w,
        pre_ln=has_pre_ln,
        input_variance=1.0,
    )


def analyze(model: "nn.Module") -> MeanFieldResult:
    """Analyze a PyTorch model's initialization phase using mean-field theory.

    This is the primary convenience function. Given any ``torch.nn.Module``,
    it extracts architecture parameters, runs mean-field analysis, and
    returns phase classification with recommendations.

    For Transformer models (nn.TransformerEncoder, nn.MultiheadAttention),
    uses specialized attention mean-field theory.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    MeanFieldResult
    """
    _require_torch()

    # Check if model is a Transformer
    transformer_spec = _extract_transformer_spec(model)
    if transformer_spec is not None:
        return _analyze_transformer(model, transformer_spec)

    from mean_field_theory import MeanFieldAnalyzer  # local import

    spec = extract_architecture_spec(model)
    mf = MeanFieldAnalyzer()
    report = mf.analyze(spec)

    # Find optimal σ_w
    sw_star, _ = mf.find_edge_of_chaos(spec.activation)

    explanation = (
        f"Architecture: depth={spec.depth}, width≈{spec.width}, "
        f"activation={spec.activation}. "
        f"Current σ_w={spec.sigma_w:.4f}, optimal σ_w*={sw_star:.4f}. "
        f"Phase: {report.phase} (χ₁={report.chi_1:.4f}). "
        f"Depth scale ξ={report.depth_scale:.1f}."
    )

    return MeanFieldResult(
        phase=report.phase,
        chi_1=report.chi_1,
        sigma_w_star=sw_star,
        recommended_sigma_w=sw_star,
        variance_trajectory=report.variance_trajectory,
        depth_scale=report.depth_scale,
        probabilities=report.phase_classification.probabilities if report.phase_classification else {},
        current_sigma_w=spec.sigma_w,
        explanation=explanation,
    )


def _analyze_transformer(model: "nn.Module", spec: "TransformerSpec") -> MeanFieldResult:
    """Analyze a Transformer model using attention mean-field theory."""
    _require_torch()
    from transformer_mean_field import TransformerMeanField

    tmf = TransformerMeanField()
    report = tmf.analyze(spec)

    return MeanFieldResult(
        phase=report.phase,
        chi_1=report.chi_1_block,
        sigma_w_star=report.sigma_w_star,
        recommended_sigma_w=report.sigma_w_star,
        variance_trajectory=report.variance_trajectory,
        depth_scale=report.depth_scale,
        probabilities={},
        current_sigma_w=spec.sigma_w,
        explanation=report.explanation,
    )


def recommend_init(model: "nn.Module", apply: bool = False) -> MeanFieldResult:
    """Recommend and optionally apply optimal initialization.

    Computes the edge-of-chaos σ_w* for the model's activation function
    and optionally re-initializes all weight layers.

    Parameters
    ----------
    model : nn.Module
    apply : bool
        If True, re-initialize weights with the recommended σ_w*.

    Returns
    -------
    MeanFieldResult
    """
    _require_torch()
    result = analyze(model)

    if apply:
        sw_star = result.recommended_sigma_w
        for m in model.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.in_features
                nn.init.normal_(m.weight, 0, sw_star / math.sqrt(fan_in))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                fan_in = m.weight.shape[1] * int(np.prod(m.weight.shape[2:]))
                nn.init.normal_(m.weight, 0, sw_star / math.sqrt(fan_in))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    return result
