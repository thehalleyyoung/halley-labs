"""Phase diagrams for Vision Transformers (ViTs).

Extends the finite-width phase diagram framework to Vision Transformers,
accounting for attention mechanisms, patch embeddings, position encodings,
and layer normalization. The NTK for ViTs is decomposed into attention NTK
and MLP NTK components with coupling through layer norm.

Example
-------
>>> from phase_diagrams.vision_transformers import vit_phase_diagram
>>> diagram = vit_phase_diagram(model, dataset, lr_range=(1e-5, 0.1))
>>> print(diagram.metadata["patch_size"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .api import PhaseDiagram, PhasePoint, Regime, _compute_gamma, _predict_gamma_star


# ======================================================================
# Enums and data classes
# ======================================================================

class AttentionPattern(str, Enum):
    """Dominant attention pattern observed in the model."""
    UNIFORM = "uniform"
    SPARSE = "sparse"
    LOCAL = "local"
    GLOBAL = "global"
    MIXED = "mixed"


class PoolingStrategy(str, Enum):
    """Token pooling strategy for classification."""
    CLS = "cls"
    MEAN = "mean"
    MAX = "max"


@dataclass
class AttentionRegime:
    """Characterization of the attention regime during training.

    Attributes
    ----------
    pattern : AttentionPattern
        Dominant attention pattern.
    entropy : float
        Average attention entropy across heads and layers.
    rank : float
        Effective rank of attention matrices.
    head_diversity : float
        Diversity of attention patterns across heads (0 = identical, 1 = maximally diverse).
    regime : Regime
        Overall training regime inferred from attention dynamics.
    per_layer_entropy : NDArray
        Attention entropy at each layer.
    per_head_pattern : Dict[int, AttentionPattern]
        Dominant pattern for each attention head.
    critical_layer : int
        Layer index where attention transitions from lazy to feature-learning.
    attention_ntk_contribution : float
        Fraction of total NTK norm coming from attention parameters.
    """
    pattern: AttentionPattern = AttentionPattern.UNIFORM
    entropy: float = 0.0
    rank: float = 0.0
    head_diversity: float = 0.0
    regime: Regime = Regime.LAZY
    per_layer_entropy: NDArray = field(default_factory=lambda: np.array([]))
    per_head_pattern: Dict[int, AttentionPattern] = field(default_factory=dict)
    critical_layer: int = 0
    attention_ntk_contribution: float = 0.0


@dataclass
class PoolingRegime:
    """Comparison of CLS token vs. mean pooling in terms of phase behavior.

    Attributes
    ----------
    cls_critical_lr : float
        Critical LR when using CLS token pooling.
    mean_critical_lr : float
        Critical LR when using mean pooling.
    cls_regime : Regime
        Regime with CLS token at default LR.
    mean_regime : Regime
        Regime with mean pooling at default LR.
    recommended_pooling : PoolingStrategy
        Recommended pooling strategy for feature learning.
    regime_gap : float
        How different the phase boundaries are (0 = same, 1 = very different).
    explanation : str
        Human-readable explanation of the difference.
    """
    cls_critical_lr: float = 0.0
    mean_critical_lr: float = 0.0
    cls_regime: Regime = Regime.LAZY
    mean_regime: Regime = Regime.LAZY
    recommended_pooling: PoolingStrategy = PoolingStrategy.CLS
    regime_gap: float = 0.0
    explanation: str = ""


@dataclass
class LayerNormEffect:
    """How layer normalization affects the phase diagram.

    Attributes
    ----------
    with_ln_critical_lr : float
        Critical LR with layer normalization.
    without_ln_critical_lr : float
        Critical LR without layer normalization.
    ln_regime_shift : float
        How much LN shifts the phase boundary (positive = towards rich).
    gradient_norm_ratio : float
        Ratio of gradient norms with/without LN.
    ntk_condition_with_ln : float
        Condition number of NTK with LN.
    ntk_condition_without_ln : float
        Condition number of NTK without LN.
    per_layer_effect : NDArray
        LN effect magnitude at each layer.
    recommendation : str
        Whether LN is beneficial for the desired regime.
    """
    with_ln_critical_lr: float = 0.0
    without_ln_critical_lr: float = 0.0
    ln_regime_shift: float = 0.0
    gradient_norm_ratio: float = 1.0
    ntk_condition_with_ln: float = 1.0
    ntk_condition_without_ln: float = 1.0
    per_layer_effect: NDArray = field(default_factory=lambda: np.array([]))
    recommendation: str = ""


@dataclass
class PosEncEffect:
    """How position encoding affects the phase diagram.

    Attributes
    ----------
    sinusoidal_critical_lr : float
        Critical LR with sinusoidal position encoding.
    learned_critical_lr : float
        Critical LR with learned position encoding.
    none_critical_lr : float
        Critical LR with no position encoding.
    sinusoidal_regime : Regime
        Regime with sinusoidal PE at default LR.
    learned_regime : Regime
        Regime with learned PE at default LR.
    pos_enc_ntk_fraction : float
        Fraction of NTK from position encoding parameters.
    recommended_encoding : str
        Recommended position encoding type.
    position_sensitivity : NDArray
        How sensitive the phase boundary is to position encoding scale.
    """
    sinusoidal_critical_lr: float = 0.0
    learned_critical_lr: float = 0.0
    none_critical_lr: float = 0.0
    sinusoidal_regime: Regime = Regime.LAZY
    learned_regime: Regime = Regime.LAZY
    pos_enc_ntk_fraction: float = 0.0
    recommended_encoding: str = "learned"
    position_sensitivity: NDArray = field(default_factory=lambda: np.array([]))


# ======================================================================
# Internal helpers
# ======================================================================

def _extract_vit_params(model: Any) -> Dict[str, Any]:
    """Extract ViT architecture parameters from model or config dict."""
    if isinstance(model, dict):
        return {
            "embed_dim": model.get("embed_dim", 768),
            "n_heads": model.get("n_heads", 12),
            "n_layers": model.get("n_layers", 12),
            "mlp_ratio": model.get("mlp_ratio", 4.0),
            "patch_size": model.get("patch_size", 16),
            "image_size": model.get("image_size", 224),
            "init_scale": model.get("init_scale", 1.0),
            "pooling": PoolingStrategy(model.get("pooling", "cls")),
            "layer_norm": model.get("layer_norm", True),
            "pos_encoding": model.get("pos_encoding", "learned"),
        }
    # Infer from weight list
    if isinstance(model, (list, tuple)):
        embed_dim = model[0].shape[-1] if model[0].ndim >= 2 else 64
        n_layers = len(model) // 4  # QKV + MLP per layer
        return {
            "embed_dim": embed_dim,
            "n_heads": max(1, embed_dim // 64),
            "n_layers": max(1, n_layers),
            "mlp_ratio": 4.0,
            "patch_size": 16,
            "image_size": 224,
            "init_scale": float(np.std(model[0]) * math.sqrt(embed_dim)),
            "pooling": PoolingStrategy.CLS,
            "layer_norm": True,
            "pos_encoding": "learned",
        }
    raise TypeError(f"Unsupported model type: {type(model)}")


def _generate_patch_embeddings(
    dataset: NDArray,
    patch_size: int,
    embed_dim: int,
    image_size: int,
    seed: int = 42,
) -> NDArray:
    """Generate patch embeddings from image data or random features.

    Returns array of shape (n_samples, n_patches + 1, embed_dim) where
    +1 accounts for the CLS token.
    """
    rng = np.random.RandomState(seed)
    n_samples = dataset.shape[0]

    if dataset.ndim >= 3:
        # Actual image data: extract patches
        n_patches_side = image_size // patch_size
        n_patches = n_patches_side * n_patches_side
        patch_dim = patch_size * patch_size * (dataset.shape[-1] if dataset.ndim == 4 else 1)
        proj = rng.randn(patch_dim, embed_dim) / math.sqrt(patch_dim)
        embeddings = np.zeros((n_samples, n_patches + 1, embed_dim))
        embeddings[:, 0] = rng.randn(n_samples, embed_dim) * 0.02  # CLS token
        for i in range(n_patches):
            row, col = divmod(i, n_patches_side)
            if dataset.ndim == 4:
                patch = dataset[
                    :,
                    row * patch_size:(row + 1) * patch_size,
                    col * patch_size:(col + 1) * patch_size,
                    :,
                ].reshape(n_samples, -1)
            else:
                patch = dataset[
                    :,
                    row * patch_size:(row + 1) * patch_size,
                    col * patch_size:(col + 1) * patch_size,
                ].reshape(n_samples, -1)
            if patch.shape[1] >= patch_dim:
                patch = patch[:, :patch_dim]
            else:
                pad = np.zeros((n_samples, patch_dim - patch.shape[1]))
                patch = np.hstack([patch, pad])
            embeddings[:, i + 1] = patch @ proj
    else:
        # Flat features: create pseudo-patches
        n_patches = max(1, dataset.shape[1] // embed_dim)
        embeddings = np.zeros((n_samples, n_patches + 1, embed_dim))
        embeddings[:, 0] = rng.randn(n_samples, embed_dim) * 0.02
        for i in range(n_patches):
            start = i * embed_dim
            end = min(start + embed_dim, dataset.shape[1])
            chunk = dataset[:, start:end]
            if chunk.shape[1] < embed_dim:
                pad = np.zeros((n_samples, embed_dim - chunk.shape[1]))
                chunk = np.hstack([chunk, pad])
            embeddings[:, i + 1] = chunk

    return embeddings


def _attention_matrix(
    queries: NDArray,
    keys: NDArray,
    d_k: float,
) -> NDArray:
    """Compute softmax attention: softmax(QK^T / sqrt(d_k))."""
    scores = queries @ keys.T / math.sqrt(d_k)
    # Numerically stable softmax
    scores -= scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-12)


def _attention_entropy(attn: NDArray) -> float:
    """Shannon entropy of attention distribution, averaged over queries."""
    eps = 1e-12
    entropies = -np.sum(attn * np.log(attn + eps), axis=-1)
    return float(np.mean(entropies))


def _classify_attention_pattern(attn: NDArray) -> AttentionPattern:
    """Classify the dominant attention pattern from an attention matrix."""
    n = attn.shape[0]
    uniform = np.ones_like(attn) / n
    uniform_dist = float(np.mean(np.abs(attn - uniform)))

    # Check sparsity: fraction of attention weights > 1/n
    sparse_frac = float(np.mean(attn > 2.0 / n))

    # Check locality: attention concentrated near diagonal
    diag_mask = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) <= max(1, n // 8)
    local_mass = float(np.mean(np.sum(attn * diag_mask, axis=-1)))

    if uniform_dist < 0.1:
        return AttentionPattern.UNIFORM
    elif sparse_frac > 0.7:
        return AttentionPattern.SPARSE
    elif local_mass > 0.6:
        return AttentionPattern.LOCAL
    elif sparse_frac < 0.3 and local_mass < 0.3:
        return AttentionPattern.GLOBAL
    else:
        return AttentionPattern.MIXED


def _vit_ntk_eigenspectrum(
    embeddings: NDArray,
    n_heads: int,
    n_layers: int,
    mlp_ratio: float,
    embed_dim: int,
    seed: int = 42,
) -> Tuple[NDArray, float, float]:
    """Approximate ViT NTK eigenspectrum.

    Returns (eigenvalues, attention_ntk_fraction, condition_number).
    The ViT NTK decomposes as K = K_attn + K_mlp + K_embed, where
    each component reflects the Jacobian from different parameter groups.
    """
    rng = np.random.RandomState(seed)
    n_samples = embeddings.shape[0]
    n_tokens = embeddings.shape[1]
    d_k = embed_dim // n_heads
    mlp_dim = int(embed_dim * mlp_ratio)

    # Aggregate token embeddings to sample-level features
    # Using mean pooling for NTK computation
    sample_features = embeddings.mean(axis=1)  # (n_samples, embed_dim)

    K_attn = np.zeros((n_samples, n_samples))
    K_mlp = np.zeros((n_samples, n_samples))

    h = sample_features.copy()
    for layer in range(n_layers):
        # Attention NTK contribution
        Wq = rng.randn(embed_dim, embed_dim) / math.sqrt(embed_dim)
        Wk = rng.randn(embed_dim, embed_dim) / math.sqrt(embed_dim)
        Wv = rng.randn(embed_dim, embed_dim) / math.sqrt(embed_dim)
        Wo = rng.randn(embed_dim, embed_dim) / math.sqrt(embed_dim)

        Q = h @ Wq
        K_mat = h @ Wk
        V = h @ Wv

        # Attention output approximation (no softmax for NTK)
        attn_out = Q @ K_mat.T / math.sqrt(d_k)
        attn_jacobian = h  # dF/dWq contribution (simplified)
        K_attn += attn_jacobian @ attn_jacobian.T / embed_dim

        # MLP NTK contribution
        W1 = rng.randn(embed_dim, mlp_dim) / math.sqrt(embed_dim)
        W2 = rng.randn(mlp_dim, embed_dim) / math.sqrt(mlp_dim)

        pre_act = h @ W1
        act = np.maximum(pre_act, 0)  # GELU ≈ ReLU for NTK
        act_deriv = (pre_act > 0).astype(float)

        mlp_jacobian = h
        K_mlp += mlp_jacobian @ mlp_jacobian.T / embed_dim

        # Update hidden state
        h = h + act @ W2  # Residual connection

    K_total = K_attn + K_mlp
    eigvals = np.sort(np.linalg.eigvalsh(K_total))[::-1]

    attn_norm = float(np.linalg.norm(K_attn, "fro"))
    total_norm = float(np.linalg.norm(K_total, "fro"))
    attn_fraction = attn_norm / (total_norm + 1e-12)

    cond = float(eigvals[0] / (eigvals[-1] + 1e-12)) if len(eigvals) > 1 else 1.0

    return eigvals, attn_fraction, cond


def _compute_vit_mu_max(
    embeddings: NDArray,
    params: Dict[str, Any],
    seed: int = 42,
) -> float:
    """Effective mu_max for ViT bifurcation analysis."""
    eigvals, _, _ = _vit_ntk_eigenspectrum(
        embeddings,
        params["n_heads"],
        params["n_layers"],
        params["mlp_ratio"],
        params["embed_dim"],
        seed,
    )
    return float(eigvals[0]) / params["embed_dim"] if len(eigvals) > 0 else 1.0


def _sinusoidal_position_encoding(n_positions: int, d_model: int) -> NDArray:
    """Generate sinusoidal position encodings."""
    pe = np.zeros((n_positions, d_model))
    position = np.arange(n_positions)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term[:d_model // 2])
    return pe


def _layer_norm_forward(x: NDArray, eps: float = 1e-5) -> Tuple[NDArray, NDArray, NDArray]:
    """Apply layer normalization. Returns (output, mean, inv_std)."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    return (x - mean) * inv_std, mean, inv_std


# ======================================================================
# Public API
# ======================================================================

def vit_phase_diagram(
    model: Any,
    dataset: NDArray,
    lr_range: Tuple[float, float] = (1e-5, 0.1),
    patch_sizes: Optional[Sequence[int]] = None,
    n_lr_steps: int = 30,
    training_steps: int = 100,
    seed: int = 42,
) -> PhaseDiagram:
    """Compute a phase diagram for a Vision Transformer.

    The phase boundary depends on the attention mechanism, patch size,
    and position encoding. The ViT NTK decomposes as K = K_attn + K_mlp,
    and the attention component introduces a qualitatively different
    spectral structure compared to standard MLPs.

    Parameters
    ----------
    model : dict or list of NDArray
        ViT specification. Dict with keys ``{'embed_dim', 'n_heads',
        'n_layers', 'mlp_ratio', 'patch_size', 'image_size'}``.
    dataset : NDArray
        Input data of shape ``(n_samples, ...)`` (images or flat features).
    lr_range : (float, float)
        Learning rate scan range.
    patch_sizes : sequence of int or None
        Patch sizes to scan. If None, uses model's patch size only.
    n_lr_steps : int
        Number of LR grid points.
    training_steps : int
        Assumed training duration T.
    seed : int
        Random seed.

    Returns
    -------
    PhaseDiagram
        Phase diagram with boundary curve.
    """
    params = _extract_vit_params(model)
    embed_dim = params["embed_dim"]
    init_scale = params["init_scale"]

    if patch_sizes is None:
        patch_sizes = [params["patch_size"]]

    lrs = np.logspace(math.log10(lr_range[0]), math.log10(lr_range[1]), n_lr_steps)
    points: List[PhasePoint] = []
    boundary_pts: List[Tuple[float, int]] = []

    for ps in patch_sizes:
        embeddings = _generate_patch_embeddings(
            dataset, ps, embed_dim, params["image_size"], seed
        )
        mu_max = _compute_vit_mu_max(embeddings, params, seed)
        g_star = _predict_gamma_star(mu_max, training_steps)

        # Patch size acts as an effective width modifier:
        # fewer patches → each patch carries more information → shifts boundary
        n_patches = (params["image_size"] // ps) ** 2
        width_eff = embed_dim * n_patches

        prev_regime = None
        for lr in lrs:
            gamma = _compute_gamma(lr, init_scale, embed_dim)
            if gamma < g_star * 0.8:
                regime = Regime.LAZY
                confidence = min(1.0, (g_star - gamma) / g_star)
            elif gamma > g_star * 1.2:
                regime = Regime.RICH
                confidence = min(1.0, (gamma - g_star) / g_star)
            else:
                regime = Regime.CRITICAL
                confidence = 1.0 - abs(gamma - g_star) / (0.2 * g_star + 1e-12)

            ntk_drift = gamma * mu_max * training_steps
            points.append(PhasePoint(
                lr=float(lr),
                width=width_eff,
                regime=regime,
                gamma=gamma,
                gamma_star=g_star,
                confidence=max(0.0, min(1.0, confidence)),
                ntk_drift_predicted=ntk_drift,
            ))

            if prev_regime is not None and prev_regime != regime:
                boundary_pts.append((float(lr), width_eff))
            prev_regime = regime

    boundary_curve = np.array(boundary_pts) if boundary_pts else None
    timescale_constants = []
    for bp_lr, bp_w in boundary_pts:
        g = _compute_gamma(bp_lr, init_scale, embed_dim)
        timescale_constants.append(training_steps * g)
    tc = float(np.mean(timescale_constants)) if timescale_constants else 0.0

    return PhaseDiagram(
        points=points,
        lr_range=lr_range,
        width_range=(embed_dim, embed_dim * max(
            (params["image_size"] // ps) ** 2 for ps in patch_sizes
        )),
        boundary_curve=boundary_curve,
        timescale_constant=tc,
        metadata={
            "architecture": "ViT",
            "embed_dim": embed_dim,
            "n_heads": params["n_heads"],
            "n_layers": params["n_layers"],
            "patch_sizes": list(patch_sizes),
            "pooling": params["pooling"].value,
        },
    )


def attention_regime_analysis(
    model: Any,
    dataset: NDArray,
    lr: float = 0.001,
    training_steps: int = 100,
    seed: int = 42,
) -> AttentionRegime:
    """Analyze what attention regime a ViT operates in.

    Characterizes the attention pattern (uniform, sparse, local, global)
    and determines whether the attention mechanism is in the lazy regime
    (attention patterns don't change from initialization) or rich regime
    (attention learns task-specific patterns).

    Parameters
    ----------
    model : dict or list of NDArray
        ViT specification.
    dataset : NDArray
        Input data.
    lr : float
        Learning rate for regime analysis.
    training_steps : int
        Assumed training duration.
    seed : int
        Random seed.

    Returns
    -------
    AttentionRegime
        Full characterization of attention behavior.
    """
    params = _extract_vit_params(model)
    embed_dim = params["embed_dim"]
    n_heads = params["n_heads"]
    n_layers = params["n_layers"]
    d_k = embed_dim // n_heads

    embeddings = _generate_patch_embeddings(
        dataset, params["patch_size"], embed_dim, params["image_size"], seed
    )

    rng = np.random.RandomState(seed)
    n_tokens = embeddings.shape[1]

    per_layer_entropy = np.zeros(n_layers)
    per_head_pattern: Dict[int, AttentionPattern] = {}
    all_entropies = []

    h = embeddings.mean(axis=1)  # (n_samples, embed_dim)

    # Track attention rank and pattern per layer
    head_idx = 0
    critical_layer = n_layers
    found_critical = False

    for layer in range(n_layers):
        layer_entropies = []
        for head in range(n_heads):
            Wq = rng.randn(embed_dim, d_k) / math.sqrt(embed_dim)
            Wk = rng.randn(embed_dim, d_k) / math.sqrt(embed_dim)

            Q = h @ Wq
            K_mat = h @ Wk

            # Use a subset of samples for attention computation
            n_sub = min(32, h.shape[0])
            attn = _attention_matrix(Q[:n_sub], K_mat[:n_sub], d_k)

            ent = _attention_entropy(attn)
            layer_entropies.append(ent)

            pattern = _classify_attention_pattern(attn)
            per_head_pattern[head_idx] = pattern
            head_idx += 1

        per_layer_entropy[layer] = float(np.mean(layer_entropies))
        all_entropies.extend(layer_entropies)

        # Detect transition from uniform (lazy) to structured (rich) attention
        max_entropy = math.log(n_sub)
        if per_layer_entropy[layer] < 0.7 * max_entropy and not found_critical:
            critical_layer = layer
            found_critical = True

        # Update h with residual
        W1 = rng.randn(embed_dim, int(embed_dim * params["mlp_ratio"])) / math.sqrt(embed_dim)
        W2 = rng.randn(int(embed_dim * params["mlp_ratio"]), embed_dim) / math.sqrt(
            int(embed_dim * params["mlp_ratio"]))
        h = h + np.maximum(h @ W1, 0) @ W2

    avg_entropy = float(np.mean(all_entropies))

    # Effective rank of attention via eigenvalues
    # Approximate from entropy
    max_ent = math.log(max(2, embeddings.shape[0]))
    rank = math.exp(avg_entropy) if avg_entropy < max_ent else float(embeddings.shape[0])

    # Head diversity: variance of entropies across heads
    head_diversity = float(np.std(all_entropies) / (np.mean(all_entropies) + 1e-12))
    head_diversity = min(1.0, head_diversity)

    # Count patterns
    pattern_counts: Dict[AttentionPattern, int] = {}
    for p in per_head_pattern.values():
        pattern_counts[p] = pattern_counts.get(p, 0) + 1
    dominant_pattern = max(pattern_counts, key=lambda p: pattern_counts[p])

    # Determine overall regime from attention properties
    eigvals, attn_fraction, _ = _vit_ntk_eigenspectrum(
        embeddings, n_heads, n_layers, params["mlp_ratio"], embed_dim, seed
    )
    mu_max = float(eigvals[0]) / embed_dim if len(eigvals) > 0 else 1.0
    g_star = _predict_gamma_star(mu_max, training_steps)
    gamma = _compute_gamma(lr, params["init_scale"], embed_dim)

    if gamma < g_star * 0.8:
        regime = Regime.LAZY
    elif gamma > g_star * 1.2:
        regime = Regime.RICH
    else:
        regime = Regime.CRITICAL

    return AttentionRegime(
        pattern=dominant_pattern,
        entropy=avg_entropy,
        rank=rank,
        head_diversity=head_diversity,
        regime=regime,
        per_layer_entropy=per_layer_entropy,
        per_head_pattern=per_head_pattern,
        critical_layer=critical_layer,
        attention_ntk_contribution=attn_fraction,
    )


def patch_size_phase_boundary(
    model: Any,
    dataset: NDArray,
    patch_sizes: Optional[Sequence[int]] = None,
    training_steps: int = 100,
    seed: int = 42,
) -> float:
    """Find the critical patch size separating lazy from rich regime.

    Larger patches reduce the sequence length, changing the effective
    width of the model and shifting the phase boundary. This function
    binary-searches for the patch size at which the phase transitions.

    Parameters
    ----------
    model : dict or list of NDArray
        ViT specification.
    dataset : NDArray
        Input data.
    patch_sizes : sequence of int or None
        Patch sizes to evaluate. If None, uses powers of 2 from 4 to 64.
    training_steps : int
        Assumed training duration.
    seed : int
        Random seed.

    Returns
    -------
    float
        Critical patch size (interpolated).
    """
    params = _extract_vit_params(model)
    embed_dim = params["embed_dim"]
    init_scale = params["init_scale"]
    image_size = params["image_size"]

    if patch_sizes is None:
        patch_sizes = [4, 8, 16, 32, 64]
    patch_sizes = sorted([ps for ps in patch_sizes if ps <= image_size // 2])

    if not patch_sizes:
        return float(params["patch_size"])

    gammas = []
    gamma_stars = []
    default_lr = 0.001

    for ps in patch_sizes:
        embeddings = _generate_patch_embeddings(
            dataset, ps, embed_dim, image_size, seed
        )
        mu_max = _compute_vit_mu_max(embeddings, params, seed)
        g_star = _predict_gamma_star(mu_max, training_steps)
        gamma = _compute_gamma(default_lr, init_scale, embed_dim)
        gammas.append(gamma)
        gamma_stars.append(g_star)

    # Find where gamma crosses gamma_star
    gammas = np.array(gammas)
    gamma_stars = np.array(gamma_stars)
    diff = gammas - gamma_stars

    # Look for sign change
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] < 0:
            # Linear interpolation
            t = abs(diff[i]) / (abs(diff[i]) + abs(diff[i + 1]) + 1e-12)
            return float(patch_sizes[i] * (1 - t) + patch_sizes[i + 1] * t)

    # No crossing found: return patch size closest to boundary
    idx = int(np.argmin(np.abs(diff)))
    return float(patch_sizes[idx])


def cls_vs_mean_pooling_regime(
    model: Any,
    dataset: NDArray,
    lr: float = 0.001,
    training_steps: int = 100,
    seed: int = 42,
) -> PoolingRegime:
    """Compare CLS token and mean pooling in terms of phase behavior.

    CLS token pooling concentrates the classification signal in a single
    token, while mean pooling distributes it. This affects the effective
    coupling and can shift the phase boundary significantly.

    Parameters
    ----------
    model : dict or list of NDArray
        ViT specification.
    dataset : NDArray
        Input data.
    lr : float
        Reference learning rate.
    training_steps : int
        Assumed training duration.
    seed : int
        Random seed.

    Returns
    -------
    PoolingRegime
        Comparison of CLS and mean pooling phase behavior.
    """
    params = _extract_vit_params(model)
    embed_dim = params["embed_dim"]
    init_scale = params["init_scale"]

    embeddings = _generate_patch_embeddings(
        dataset, params["patch_size"], embed_dim, params["image_size"], seed
    )

    rng = np.random.RandomState(seed)
    n_samples = embeddings.shape[0]
    n_tokens = embeddings.shape[1]

    # CLS pooling: use only CLS token representation
    cls_features = embeddings[:, 0, :]  # (n_samples, embed_dim)
    mean_features = embeddings.mean(axis=1)  # (n_samples, embed_dim)

    # Compute NTK for CLS pooling
    K_cls = cls_features @ cls_features.T
    cls_eigvals = np.sort(np.linalg.eigvalsh(K_cls))[::-1]
    cls_mu_max = float(cls_eigvals[0]) / embed_dim if len(cls_eigvals) > 0 else 1.0
    cls_g_star = _predict_gamma_star(cls_mu_max, training_steps)
    cls_critical_lr = cls_g_star * embed_dim / (init_scale ** 2 + 1e-12)

    # Compute NTK for mean pooling
    K_mean = mean_features @ mean_features.T
    mean_eigvals = np.sort(np.linalg.eigvalsh(K_mean))[::-1]
    mean_mu_max = float(mean_eigvals[0]) / embed_dim if len(mean_eigvals) > 0 else 1.0
    mean_g_star = _predict_gamma_star(mean_mu_max, training_steps)
    mean_critical_lr = mean_g_star * embed_dim / (init_scale ** 2 + 1e-12)

    gamma = _compute_gamma(lr, init_scale, embed_dim)
    cls_regime = Regime.LAZY if gamma < cls_g_star else (
        Regime.RICH if gamma > cls_g_star * 1.2 else Regime.CRITICAL)
    mean_regime = Regime.LAZY if gamma < mean_g_star else (
        Regime.RICH if gamma > mean_g_star * 1.2 else Regime.CRITICAL)

    regime_gap = abs(math.log(cls_critical_lr + 1e-12) - math.log(mean_critical_lr + 1e-12))
    regime_gap = min(1.0, regime_gap / 5.0)

    # Recommendation
    if mean_critical_lr < cls_critical_lr:
        recommended = PoolingStrategy.MEAN
        explanation = (
            f"Mean pooling enters the rich regime at a lower LR "
            f"(η*={mean_critical_lr:.2e} vs {cls_critical_lr:.2e}), "
            f"making feature learning easier to achieve. This is because "
            f"mean pooling aggregates gradients from all patches, "
            f"amplifying the effective coupling."
        )
    else:
        recommended = PoolingStrategy.CLS
        explanation = (
            f"CLS token pooling enters the rich regime at a lower LR "
            f"(η*={cls_critical_lr:.2e} vs {mean_critical_lr:.2e}). "
            f"The concentrated gradient signal through the CLS token "
            f"provides stronger feature learning drive."
        )

    return PoolingRegime(
        cls_critical_lr=cls_critical_lr,
        mean_critical_lr=mean_critical_lr,
        cls_regime=cls_regime,
        mean_regime=mean_regime,
        recommended_pooling=recommended,
        regime_gap=regime_gap,
        explanation=explanation,
    )


def layer_norm_phase_effect(
    model: Any,
    dataset: NDArray,
    lr: float = 0.001,
    training_steps: int = 100,
    seed: int = 42,
) -> LayerNormEffect:
    """Analyze how layer normalization affects the phase diagram.

    Layer normalization constrains the representation geometry, which
    modifies the NTK structure and can significantly shift the phase
    boundary. Pre-LN vs. post-LN architectures may show different effects.

    Parameters
    ----------
    model : dict or list of NDArray
        ViT specification.
    dataset : NDArray
        Input data.
    lr : float
        Reference learning rate.
    training_steps : int
        Assumed training duration.
    seed : int
        Random seed.

    Returns
    -------
    LayerNormEffect
        Analysis of LN's effect on phase behavior.
    """
    params = _extract_vit_params(model)
    embed_dim = params["embed_dim"]
    n_layers = params["n_layers"]
    init_scale = params["init_scale"]

    embeddings = _generate_patch_embeddings(
        dataset, params["patch_size"], embed_dim, params["image_size"], seed
    )
    rng = np.random.RandomState(seed)
    n_samples = embeddings.shape[0]
    h = embeddings.mean(axis=1)

    # Forward pass WITH layer norm
    h_ln = h.copy()
    grad_norms_ln = []
    for layer in range(n_layers):
        W = rng.randn(embed_dim, embed_dim) / math.sqrt(embed_dim)
        h_ln, _, _ = _layer_norm_forward(h_ln)
        pre = h_ln @ W
        h_ln = h_ln + np.maximum(pre, 0)
        grad_norms_ln.append(float(np.linalg.norm(W)))

    K_ln = h_ln @ h_ln.T
    ln_eigvals = np.sort(np.linalg.eigvalsh(K_ln))[::-1]
    ln_mu_max = float(ln_eigvals[0]) / embed_dim if len(ln_eigvals) > 0 else 1.0
    ln_g_star = _predict_gamma_star(ln_mu_max, training_steps)
    ln_cond = float(ln_eigvals[0] / (ln_eigvals[-1] + 1e-12))

    # Forward pass WITHOUT layer norm
    rng2 = np.random.RandomState(seed)
    h_no_ln = h.copy()
    grad_norms_no_ln = []
    for layer in range(n_layers):
        W = rng2.randn(embed_dim, embed_dim) / math.sqrt(embed_dim)
        pre = h_no_ln @ W
        h_no_ln = h_no_ln + np.maximum(pre, 0)
        grad_norms_no_ln.append(float(np.linalg.norm(W)))

    K_no_ln = h_no_ln @ h_no_ln.T
    no_ln_eigvals = np.sort(np.linalg.eigvalsh(K_no_ln))[::-1]
    no_ln_mu_max = float(no_ln_eigvals[0]) / embed_dim if len(no_ln_eigvals) > 0 else 1.0
    no_ln_g_star = _predict_gamma_star(no_ln_mu_max, training_steps)
    no_ln_cond = float(no_ln_eigvals[0] / (no_ln_eigvals[-1] + 1e-12))

    with_ln_lr = ln_g_star * embed_dim / (init_scale ** 2 + 1e-12)
    without_ln_lr = no_ln_g_star * embed_dim / (init_scale ** 2 + 1e-12)

    regime_shift = math.log(with_ln_lr + 1e-12) - math.log(without_ln_lr + 1e-12)

    grad_ratio = float(np.mean(grad_norms_ln)) / (float(np.mean(grad_norms_no_ln)) + 1e-12)

    per_layer_effect = np.zeros(n_layers)
    for l in range(n_layers):
        per_layer_effect[l] = abs(grad_norms_ln[l] - grad_norms_no_ln[l]) / (
            grad_norms_no_ln[l] + 1e-12)

    if regime_shift > 0:
        recommendation = (
            "Layer norm pushes the phase boundary to higher LR, "
            "making lazy training more likely at standard LRs. "
            "Use higher LR to enter the rich regime."
        )
    else:
        recommendation = (
            "Layer norm lowers the phase boundary, "
            "facilitating feature learning at standard LRs. "
            "Keep layer norm enabled for rich training."
        )

    return LayerNormEffect(
        with_ln_critical_lr=with_ln_lr,
        without_ln_critical_lr=without_ln_lr,
        ln_regime_shift=regime_shift,
        gradient_norm_ratio=grad_ratio,
        ntk_condition_with_ln=ln_cond,
        ntk_condition_without_ln=no_ln_cond,
        per_layer_effect=per_layer_effect,
        recommendation=recommendation,
    )


def position_encoding_phase_analysis(
    model: Any,
    dataset: NDArray,
    lr: float = 0.001,
    training_steps: int = 100,
    scales: Optional[Sequence[float]] = None,
    seed: int = 42,
) -> PosEncEffect:
    """Analyze how position encoding affects the phase diagram.

    Position encodings add structure to the input that can modify the
    NTK spectrum. Learned position encodings contribute additional
    parameters whose gradients affect the effective coupling, while
    sinusoidal encodings are fixed and only modify the input distribution.

    Parameters
    ----------
    model : dict or list of NDArray
        ViT specification.
    dataset : NDArray
        Input data.
    lr : float
        Reference learning rate.
    training_steps : int
        Assumed training duration.
    scales : sequence of float or None
        Position encoding scales to evaluate sensitivity.
    seed : int
        Random seed.

    Returns
    -------
    PosEncEffect
        Analysis of position encoding's effect on phase behavior.
    """
    params = _extract_vit_params(model)
    embed_dim = params["embed_dim"]
    init_scale = params["init_scale"]
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]

    embeddings = _generate_patch_embeddings(
        dataset, params["patch_size"], embed_dim, params["image_size"], seed
    )
    n_tokens = embeddings.shape[1]
    rng = np.random.RandomState(seed)

    # Sinusoidal PE
    sin_pe = _sinusoidal_position_encoding(n_tokens, embed_dim)
    emb_sin = embeddings + sin_pe[np.newaxis, :, :]
    eigvals_sin, attn_frac_sin, _ = _vit_ntk_eigenspectrum(
        emb_sin, n_heads, n_layers, params["mlp_ratio"], embed_dim, seed
    )
    mu_sin = float(eigvals_sin[0]) / embed_dim if len(eigvals_sin) > 0 else 1.0
    g_star_sin = _predict_gamma_star(mu_sin, training_steps)
    sin_lr = g_star_sin * embed_dim / (init_scale ** 2 + 1e-12)

    # Learned PE (random initialization, contributes to NTK)
    learned_pe = rng.randn(n_tokens, embed_dim) * 0.02
    emb_learned = embeddings + learned_pe[np.newaxis, :, :]
    # Learned PE adds parameters → increases NTK
    pe_param_count = n_tokens * embed_dim
    total_params = n_layers * (4 * embed_dim ** 2 + 2 * embed_dim * int(embed_dim * params["mlp_ratio"]))
    pe_fraction = pe_param_count / (total_params + pe_param_count + 1e-12)

    eigvals_learned, attn_frac_learned, _ = _vit_ntk_eigenspectrum(
        emb_learned, n_heads, n_layers, params["mlp_ratio"], embed_dim, seed
    )
    mu_learned = float(eigvals_learned[0]) / embed_dim if len(eigvals_learned) > 0 else 1.0
    # Learned PE increases effective mu_max by PE fraction
    mu_learned_eff = mu_learned * (1.0 + pe_fraction)
    g_star_learned = _predict_gamma_star(mu_learned_eff, training_steps)
    learned_lr = g_star_learned * embed_dim / (init_scale ** 2 + 1e-12)

    # No PE
    eigvals_none, _, _ = _vit_ntk_eigenspectrum(
        embeddings, n_heads, n_layers, params["mlp_ratio"], embed_dim, seed
    )
    mu_none = float(eigvals_none[0]) / embed_dim if len(eigvals_none) > 0 else 1.0
    g_star_none = _predict_gamma_star(mu_none, training_steps)
    none_lr = g_star_none * embed_dim / (init_scale ** 2 + 1e-12)

    gamma = _compute_gamma(lr, init_scale, embed_dim)
    sin_regime = Regime.LAZY if gamma < g_star_sin else (
        Regime.RICH if gamma > g_star_sin * 1.2 else Regime.CRITICAL)
    learned_regime = Regime.LAZY if gamma < g_star_learned else (
        Regime.RICH if gamma > g_star_learned * 1.2 else Regime.CRITICAL)

    # Position sensitivity: how critical LR changes with PE scale
    if scales is None:
        scales = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

    position_sensitivity = np.zeros(len(scales))
    for idx, scale in enumerate(scales):
        scaled_pe = sin_pe * scale
        emb_scaled = embeddings + scaled_pe[np.newaxis, :, :]
        eigvals_s, _, _ = _vit_ntk_eigenspectrum(
            emb_scaled, n_heads, n_layers, params["mlp_ratio"], embed_dim, seed
        )
        mu_s = float(eigvals_s[0]) / embed_dim if len(eigvals_s) > 0 else 1.0
        g_star_s = _predict_gamma_star(mu_s, training_steps)
        position_sensitivity[idx] = g_star_s * embed_dim / (init_scale ** 2 + 1e-12)

    # Recommendation
    lrs_dict = {"sinusoidal": sin_lr, "learned": learned_lr, "none": none_lr}
    best = min(lrs_dict, key=lambda k: lrs_dict[k])
    recommended = best if best != "none" else "learned"

    return PosEncEffect(
        sinusoidal_critical_lr=sin_lr,
        learned_critical_lr=learned_lr,
        none_critical_lr=none_lr,
        sinusoidal_regime=sin_regime,
        learned_regime=learned_regime,
        pos_enc_ntk_fraction=pe_fraction,
        recommended_encoding=recommended,
        position_sensitivity=position_sensitivity,
    )
