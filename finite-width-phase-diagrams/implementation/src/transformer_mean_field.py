"""
Mean-field theory for Transformer architectures.

Extends PhaseKit's mean-field framework to self-attention layers and
Transformer blocks (attention + FFN + LayerNorm). Key results:

1. **Softmax attention variance propagation**: For queries/keys/values with
   variance q, the post-attention variance is derived from the expected
   softmax-weighted second moment under Gaussian inputs.

2. **LayerNorm variance reset**: LayerNorm normalizes activations to unit
   variance (plus learned affine), resetting the variance recursion at each
   block. This makes Transformers more robust to initialization than MLPs.

3. **Combined block recursion**: Pre-LN Transformer block:
       h' = LN(h + Attn(h))
       h'' = LN(h' + FFN(h'))
   Post-LN Transformer block:
       h' = h + Attn(LN(h))
       h'' = h' + FFN(LN(h'))

References:
    He et al., "On Layer Normalization in the Transformer Architecture", ICML 2020
    Noci et al., "Signal Propagation in Transformers", ICML 2022
    Trockman & Kolter, "Mimetic Initialization of Self-Attention Layers", ICML 2023
"""

import math
import numpy as np
from scipy.integrate import quad
from scipy.special import erf
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Callable
import warnings

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from mean_field_theory import (
    ActivationVarianceMaps,
    MeanFieldAnalyzer,
    ArchitectureSpec,
    MFReport,
    PhaseClassification,
    ConfidenceInterval,
)


@dataclass
class TransformerSpec:
    """Specification for a Transformer architecture.

    Attributes
    ----------
    n_layers : int
        Number of Transformer blocks (each = attention + FFN).
    d_model : int
        Model/embedding dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int
        FFN hidden dimension (typically 4 * d_model).
    activation : str
        FFN activation function (e.g., 'gelu', 'relu').
    sigma_w : float
        Weight initialization scale (std * sqrt(fan_in)).
    sigma_b : float
        Bias initialization scale.
    pre_ln : bool
        If True, use Pre-LN (LayerNorm before sublayer). Else Post-LN.
    input_variance : float
        Input embedding variance.
    dropout : float
        Dropout rate (affects variance by factor 1/(1-p) during training).
    """
    n_layers: int = 6
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    activation: str = "gelu"
    sigma_w: float = 1.0
    sigma_b: float = 0.0
    pre_ln: bool = True
    input_variance: float = 1.0
    dropout: float = 0.0
    seq_len: int = 128
    is_causal: bool = True


@dataclass
class TransformerMFReport:
    """Mean-field analysis report for Transformer architectures.

    Attributes
    ----------
    phase : str
        Predicted phase: "ordered", "critical", or "chaotic".
    chi_1_attn : float
        Attention sublayer susceptibility.
    chi_1_ffn : float
        FFN sublayer susceptibility.
    chi_1_block : float
        Per-block effective susceptibility.
    chi_1_total : float
        Total susceptibility across all layers.
    variance_trajectory : list of float
        Per-layer variance after each block.
    attn_variance_trajectory : list of float
        Variance after attention sublayer at each block.
    ffn_variance_trajectory : list of float
        Variance after FFN sublayer at each block.
    depth_scale : float
        Effective depth scale.
    sigma_w_star : float
        Recommended σ_w for edge-of-chaos.
    explanation : str
        Human-readable analysis summary.
    has_layernorm : bool
        Whether LayerNorm is present.
    head_dim : int
        Per-head dimension d_k = d_model / n_heads.
    """
    phase: str = "unknown"
    chi_1_attn: float = 0.0
    chi_1_ffn: float = 0.0
    chi_1_block: float = 0.0
    chi_1_total: float = 0.0
    variance_trajectory: List[float] = field(default_factory=list)
    attn_variance_trajectory: List[float] = field(default_factory=list)
    ffn_variance_trajectory: List[float] = field(default_factory=list)
    depth_scale: float = 0.0
    sigma_w_star: float = 0.0
    explanation: str = ""
    has_layernorm: bool = True
    head_dim: int = 64


class TransformerMeanField:
    """Mean-field analysis for Transformer architectures.

    Models signal propagation through self-attention and FFN sublayers,
    accounting for LayerNorm's variance-resetting effect and the softmax
    nonlinearity in attention.
    """

    def __init__(self):
        self._mf = MeanFieldAnalyzer()

    def attention_output_variance(self, q_in: float, d_k: int,
                                  sigma_w: float, n_heads: int = 8,
                                  seq_len: int = 128,
                                  is_causal: bool = True) -> float:
        """Compute post-attention variance for a single self-attention layer.

        For input with variance q_in, QKV projections with scale sigma_w,
        and softmax attention with head dimension d_k:

        The attention output at position i is a weighted average of value
        vectors: out_i = sum_j p_{ij} v_j, where p = softmax(QK^T/sqrt(d_k)).

        The per-component output variance is:
            Var[out_i] = E[||p_i||^2] * q_V

        For near-uniform attention (small sigma_w):
        - Bidirectional: E[||p||^2] ≈ 1/T
        - Causal at position i: E[||p_i||^2] = 1/(i+1)
        - Average over causal positions: H_T/T (harmonic number / seq_len)

        As sigma_w grows, attention concentrates and E[||p||^2] → 1.

        After output projection: q_attn_out = sigma_w^2 * E[||p||^2] * q_V

        Parameters
        ----------
        q_in : float
            Input variance.
        d_k : int
            Per-head dimension.
        sigma_w : float
            Weight scale for Q, K, V projections.
        n_heads : int
            Number of attention heads.
        seq_len : int
            Sequence length T.
        is_causal : bool
            Whether attention is causally masked.

        Returns
        -------
        float
            Post-attention variance (after output projection).
        """
        if q_in <= 0:
            return 0.0

        # Value projection variance: sigma_w^2 * q_in
        q_value = sigma_w ** 2 * q_in

        # Logit variance: Var[q^T k / sqrt(d_k)] = sigma_w^4 * q_in^2
        sigma_logit_sq = sigma_w ** 4 * q_in ** 2

        # Softmax squared norm E[||p||^2]:
        # - Uniform limit (small sigma_logit): 1/T (bidirectional) or H_T/T (causal)
        # - Concentrated limit (large sigma_logit): → 1
        T = max(seq_len, 1)
        if is_causal:
            # Harmonic number H_T = sum_{i=1}^{T} 1/i
            H_T = sum(1.0 / i for i in range(1, T + 1))
            uniform_concentration = H_T / T
        else:
            uniform_concentration = 1.0 / T

        # Interpolation from uniform to concentrated based on logit variance
        # As sigma_logit grows, softmax concentrates and ||p||^2 approaches 1
        softmax_sq_norm = uniform_concentration + (1.0 - uniform_concentration) * (
            1.0 - np.exp(-sigma_logit_sq))

        # Output projection: another sigma_w^2 factor
        q_attn_out = sigma_w ** 2 * q_value * softmax_sq_norm

        return max(q_attn_out, 0.0)

    def attention_chi1(self, q_in: float, d_k: int, sigma_w: float,
                       seq_len: int = 128, is_causal: bool = True) -> float:
        """Susceptibility of the attention sublayer.

        The Jacobian of attention w.r.t. input is dominated by the value
        pathway at initialization:
            J_V = W_O @ diag(softmax) @ W_V

        The squared Frobenius norm per dimension is:
            chi_1^attn ≈ sigma_w^4 * E[||softmax||^2]

        where E[||softmax||^2] depends on sequence length and causal masking.

        Parameters
        ----------
        q_in : float
            Input variance.
        d_k : int
            Per-head dimension.
        sigma_w : float
            Weight scale.
        seq_len : int
            Sequence length.
        is_causal : bool
            Whether attention uses causal masking.

        Returns
        -------
        float
            Attention sublayer susceptibility.
        """
        sigma_logit_sq = sigma_w ** 4 * q_in ** 2
        T = max(seq_len, 1)
        if is_causal:
            H_T = sum(1.0 / i for i in range(1, T + 1))
            uniform_concentration = H_T / T
        else:
            uniform_concentration = 1.0 / T

        softmax_sq_norm = uniform_concentration + (1.0 - uniform_concentration) * (
            1.0 - np.exp(-sigma_logit_sq))

        chi_attn = sigma_w ** 4 * softmax_sq_norm
        return chi_attn

    def layernorm_effect(self, q_in: float, gamma: float = 1.0) -> float:
        """Variance after LayerNorm.

        LayerNorm normalizes to zero mean, unit variance, then applies
        learned affine: y = gamma * (x - mu) / sigma + beta.

        At initialization, gamma=1, beta=0, so output variance = 1.

        Parameters
        ----------
        q_in : float
            Input variance.
        gamma : float
            LayerNorm gain (learned). Default 1.0 at init.

        Returns
        -------
        float
            Output variance after LayerNorm.
        """
        return gamma ** 2

    def ffn_variance(self, q_in: float, sigma_w: float,
                     activation: str = "gelu", d_ff_ratio: float = 4.0) -> float:
        """Variance after FFN sublayer.

        FFN: Linear(d_model, d_ff) -> activation -> Linear(d_ff, d_model)

        Parameters
        ----------
        q_in : float
            Input variance.
        sigma_w : float
            Weight scale.
        activation : str
            FFN activation function.
        d_ff_ratio : float
            d_ff / d_model ratio.

        Returns
        -------
        float
            Post-FFN variance.
        """
        V_func = self._mf._get_variance_map(activation)

        # First linear: variance = sigma_w^2 * q_in
        q_hidden = sigma_w ** 2 * q_in

        # Activation
        q_act = V_func(q_hidden)

        # Second linear: variance = sigma_w^2 * q_act
        q_out = sigma_w ** 2 * q_act

        return max(q_out, 0.0)

    def ffn_chi1(self, q_in: float, sigma_w: float,
                 activation: str = "gelu") -> float:
        """FFN sublayer susceptibility.

        The FFN is a 2-layer MLP, so chi_1^FFN = sigma_w^4 * chi_act(q_hidden)
        where chi_act is the activation's susceptibility.

        Parameters
        ----------
        q_in : float
            Input variance.
        sigma_w : float
            Weight scale.
        activation : str
            FFN activation.

        Returns
        -------
        float
            FFN susceptibility.
        """
        chi_func = self._mf._get_chi_map(activation)
        q_hidden = sigma_w ** 2 * q_in
        chi_act = chi_func(q_hidden)
        return sigma_w ** 4 * chi_act

    def block_variance_propagation(self, q_in: float, spec: TransformerSpec) -> Tuple[float, float, float]:
        """Propagate variance through one Transformer block.

        Returns (q_after_attn, q_after_ffn, q_out) where q_out is the
        block output variance.

        Pre-LN:  h' = h + Attn(LN(h)),  h'' = h' + FFN(LN(h'))
        Post-LN: h' = LN(h + Attn(h)),  h'' = LN(h' + FFN(h'))
        """
        d_k = spec.d_model // max(spec.n_heads, 1)

        if spec.pre_ln:
            q_ln1 = self.layernorm_effect(q_in)
            q_attn = self.attention_output_variance(
                q_ln1, d_k, spec.sigma_w, spec.n_heads,
                spec.seq_len, spec.is_causal)
            q_after_attn = q_in + q_attn

            q_ln2 = self.layernorm_effect(q_after_attn)
            q_ffn = self.ffn_variance(q_ln2, spec.sigma_w, spec.activation,
                                      spec.d_ff / spec.d_model)
            q_out = q_after_attn + q_ffn
        else:
            q_attn = self.attention_output_variance(
                q_in, d_k, spec.sigma_w, spec.n_heads,
                spec.seq_len, spec.is_causal)
            q_after_attn = q_in + q_attn
            q_after_attn = self.layernorm_effect(q_after_attn)

            q_ffn = self.ffn_variance(q_after_attn, spec.sigma_w, spec.activation,
                                      spec.d_ff / spec.d_model)
            q_out = q_after_attn + q_ffn
            q_out = self.layernorm_effect(q_out)

        if spec.dropout > 0:
            scale = 1.0 / (1.0 - spec.dropout) ** 2
            q_out *= scale

        return q_attn, q_ffn, q_out

    def block_chi1(self, q_in: float, spec: TransformerSpec) -> float:
        """Per-block susceptibility for a Transformer block.

        For Pre-LN with residual connection:
            chi_1^block = 1 + chi_1^attn + chi_1^ffn + higher-order

        The residual ensures chi_1 >= 1 always (no vanishing gradients
        from depth alone), but chi_1 > 1 means gradients grow.

        LayerNorm's Jacobian has norm ≈ 1 (at init), so it doesn't
        affect the leading-order susceptibility.
        """
        d_k = spec.d_model // max(spec.n_heads, 1)
        q_ln = 1.0

        chi_attn = self.attention_chi1(q_ln, d_k, spec.sigma_w,
                                       spec.seq_len, spec.is_causal)
        chi_ffn = self.ffn_chi1(q_ln, spec.sigma_w, spec.activation)

        if spec.pre_ln:
            # Pre-LN: residual adds identity to sublayer Jacobian
            # J_block = I + J_attn + J_ffn (to leading order)
            # chi_1 = ||J_block||^2 / d ≈ 1 + chi_attn + chi_ffn
            chi_block = 1.0 + chi_attn + chi_ffn
        else:
            # Post-LN: LN Jacobian modifies the residual
            # J_block = J_LN @ (I + J_sublayer)
            # ||J_block||^2/d ≈ 1 + chi_attn + chi_ffn (LN norm ≈ 1 at init)
            chi_block = 1.0 + chi_attn + chi_ffn

        return chi_block

    def _classify_transformer_phase(self, spec: TransformerSpec, chi_block: float,
                                     var_traj: List[float]) -> str:
        """Classify transformer phase using variance growth.

        For Pre-LN transformers, the traditional chi_block (Jacobian norm)
        is always >= 1 due to residual connections. Phase classification
        uses the variance growth ratio, which accounts for the additive
        variance accumulation from sublayer residuals.

        The per-block variance ratio captures practical training stability
        better than chi_block alone for architectures with LayerNorm.
        """
        if chi_block < 0.99:
            return "ordered"

        # Use variance growth ratio for phase classification
        if len(var_traj) >= 2 and var_traj[0] > 1e-12:
            total_ratio = var_traj[-1] / var_traj[0]
            n_blocks = len(var_traj) - 1
            per_block_ratio = total_ratio ** (1.0 / max(n_blocks, 1))
            if per_block_ratio > 1.10:
                return "chaotic"
            elif per_block_ratio < 0.95:
                return "ordered"
            return "critical"

        # Fallback to chi_block threshold
        threshold = 1.0 + 0.05 * np.log(10.0) / max(spec.n_layers, 1)
        if chi_block < threshold:
            return "critical"
        return "chaotic"

    def analyze(self, spec: TransformerSpec) -> TransformerMFReport:
        """Full mean-field analysis of a Transformer architecture.

        Parameters
        ----------
        spec : TransformerSpec
            Transformer architecture specification.

        Returns
        -------
        TransformerMFReport
            Complete analysis with phase classification.
        """
        d_k = spec.d_model // max(spec.n_heads, 1)

        q_ln = 1.0
        chi_attn = self.attention_chi1(q_ln, d_k, spec.sigma_w,
                                       spec.seq_len, spec.is_causal)
        chi_ffn = self.ffn_chi1(q_ln, spec.sigma_w, spec.activation)
        chi_block = self.block_chi1(q_ln, spec)
        chi_total = chi_block ** spec.n_layers

        # Variance propagation through all blocks
        var_traj = [spec.input_variance]
        attn_var_traj = []
        ffn_var_traj = []
        q = spec.input_variance

        for l in range(spec.n_layers):
            q_attn, q_ffn, q_out = self.block_variance_propagation(q, spec)
            attn_var_traj.append(q_attn)
            ffn_var_traj.append(q_ffn)
            var_traj.append(q_out)
            q = q_out

        # Depth scale
        log_chi = np.log(max(chi_block, 1e-30))
        if abs(log_chi) > 1e-10:
            depth_scale = 1.0 / abs(log_chi)
        else:
            depth_scale = float("inf")

        phase = self._classify_transformer_phase(spec, chi_block, var_traj)

        target_chi_block = np.exp(1.0 / max(spec.n_layers, 1))
        sigma_w_star = self._find_optimal_sigma_w(spec, target_chi_block)

        explanation = (
            f"Transformer: {spec.n_layers}L, d_model={spec.d_model}, "
            f"{spec.n_heads}H, d_ff={spec.d_ff}, act={spec.activation}, "
            f"{'Pre' if spec.pre_ln else 'Post'}-LN, T={spec.seq_len}. "
            f"Per-block χ₁={chi_block:.4f} (attn={chi_attn:.4f}, ffn={chi_ffn:.4f}). "
            f"Total χ₁^L={chi_total:.4e} over {spec.n_layers} layers. "
            f"Variance growth: {var_traj[-1]/max(var_traj[0],1e-12):.2f}×. "
            f"Phase: {phase}. "
            f"σ_w*={sigma_w_star:.4f}. Depth scale: {depth_scale:.1f} blocks."
        )

        return TransformerMFReport(
            phase=phase,
            chi_1_attn=chi_attn,
            chi_1_ffn=chi_ffn,
            chi_1_block=chi_block,
            chi_1_total=chi_total,
            variance_trajectory=var_traj,
            attn_variance_trajectory=attn_var_traj,
            ffn_variance_trajectory=ffn_var_traj,
            depth_scale=depth_scale,
            sigma_w_star=sigma_w_star,
            explanation=explanation,
            has_layernorm=True,
            head_dim=d_k,
        )

    def _find_optimal_sigma_w(self, spec: TransformerSpec,
                               target_chi_block: float) -> float:
        """Find sigma_w that achieves target per-block chi_1.

        Solves chi_block(sigma_w) = target_chi_block.
        """
        from scipy.optimize import brentq, minimize_scalar

        def objective(sw):
            s = TransformerSpec(
                n_layers=spec.n_layers, d_model=spec.d_model,
                n_heads=spec.n_heads, d_ff=spec.d_ff,
                activation=spec.activation, sigma_w=sw,
                pre_ln=spec.pre_ln, input_variance=spec.input_variance,
            )
            return self.block_chi1(1.0, s) - target_chi_block

        try:
            lo, hi = 0.01, 5.0
            if objective(lo) * objective(hi) > 0:
                # No sign change; find closest
                result = minimize_scalar(
                    lambda sw: abs(objective(sw)),
                    bounds=(lo, hi), method="bounded")
                return float(result.x)
            return float(brentq(objective, lo, hi))
        except (ValueError, RuntimeError):
            return 1.0

    def finite_width_attention_variance(self, q_in: float, d_k: int,
                                         sigma_w: float, n_heads: int,
                                         seq_len: int = 128) -> float:
        """Finite-width corrected attention output variance.

        Adds O(1/d_k) corrections from the finite head dimension:
        - Softmax concentration correction: E[||p||^2] = 1/n + O(σ_s^2/n^2)
        - Value averaging noise: Var correction ~ q_in * σ_w^4 / d_k

        Parameters
        ----------
        q_in : float
            Input variance.
        d_k : int
            Per-head dimension.
        sigma_w : float
            Weight scale.
        n_heads : int
            Number of heads.
        seq_len : int
            Sequence length (affects softmax concentration).

        Returns
        -------
        float
            Finite-width corrected post-attention variance.
        """
        q_inf = self.attention_output_variance(q_in, d_k, sigma_w, n_heads,
                                                seq_len)

        # O(1/d_k) correction: at finite head dimension, there's additional
        # variance from the random projection structure of QKV
        # This follows from the CLT applied to the d_k-dimensional dot product
        correction_dk = sigma_w ** 4 * q_in ** 2 / max(d_k, 1)

        # O(1/seq_len) correction: finite sequence softmax concentration
        sigma_logit_sq = sigma_w ** 4 * q_in ** 2
        correction_seq = sigma_w ** 2 * q_in * sigma_logit_sq / max(seq_len, 1)

        return max(q_inf + correction_dk + correction_seq, 0.0)

    def finite_width_block_chi1(self, q_in: float, spec: TransformerSpec,
                                 seq_len: int = 128) -> float:
        """Per-block susceptibility with finite-width corrections.

        Accounts for O(1/d_model) and O(1/d_k) corrections to the
        attention and FFN susceptibilities.
        """
        d_k = spec.d_model // max(spec.n_heads, 1)
        q_ln = 1.0

        chi_attn = self.attention_chi1(q_ln, d_k, spec.sigma_w,
                                       seq_len, spec.is_causal)
        chi_ffn = self.ffn_chi1(q_ln, spec.sigma_w, spec.activation)

        # Finite-width correction to attention chi: O(1/d_k) from QKV projections
        chi_attn_correction = spec.sigma_w ** 4 / max(d_k, 1)
        chi_attn += chi_attn_correction

        # FFN finite-width correction from finite d_ff
        d_ff = spec.d_ff
        chi_func = self._mf._get_chi_map(spec.activation)
        chi_act = chi_func(spec.sigma_w ** 2)
        chi_ffn_correction = spec.sigma_w ** 4 * chi_act / max(d_ff, 1)
        chi_ffn += chi_ffn_correction

        chi_block = 1.0 + chi_attn + chi_ffn
        return chi_block

    def analyze_with_finite_width(self, spec: TransformerSpec,
                                   seq_len: int = 128) -> TransformerMFReport:
        """Full analysis including finite-width corrections.

        This is the recommended entry point for analyzing real transformers.
        Uses variance-growth-aware phase classification that accounts for
        the additive nature of residual variance accumulation in Pre-LN
        transformers.
        """
        # Use seq_len from spec if explicitly set, otherwise use argument
        effective_seq_len = spec.seq_len if spec.seq_len != 128 else seq_len

        d_k = spec.d_model // max(spec.n_heads, 1)
        q_ln = 1.0

        chi_attn = self.attention_chi1(q_ln, d_k, spec.sigma_w,
                                       effective_seq_len, spec.is_causal)
        chi_ffn = self.ffn_chi1(q_ln, spec.sigma_w, spec.activation)
        chi_block = self.finite_width_block_chi1(q_ln, spec, effective_seq_len)
        chi_total = chi_block ** spec.n_layers

        # Variance propagation with finite-width corrections
        var_traj = [spec.input_variance]
        attn_var_traj = []
        ffn_var_traj = []
        q = spec.input_variance

        for l in range(spec.n_layers):
            q_attn, q_ffn, q_out = self.block_variance_propagation(q, spec)
            # Add finite-width correction to attention output
            fw_correction = spec.sigma_w ** 4 * q ** 2 / max(d_k, 1)
            q_attn += fw_correction
            attn_var_traj.append(q_attn)
            ffn_var_traj.append(q_ffn)
            var_traj.append(q_out)
            q = q_out

        # Depth scale
        log_chi = np.log(max(chi_block, 1e-30))
        depth_scale = 1.0 / abs(log_chi) if abs(log_chi) > 1e-10 else float("inf")

        # Phase classification using variance growth
        phase = self._classify_transformer_phase(spec, chi_block, var_traj)

        target_chi_block = np.exp(1.0 / max(spec.n_layers, 1))
        sigma_w_star = self._find_optimal_sigma_w(spec, target_chi_block)

        var_growth = var_traj[-1] / max(var_traj[0], 1e-12)
        explanation = (
            f"Transformer: {spec.n_layers}L, d_model={spec.d_model}, "
            f"{spec.n_heads}H, d_ff={spec.d_ff}, act={spec.activation}, "
            f"{'Pre' if spec.pre_ln else 'Post'}-LN, T={effective_seq_len}. "
            f"Per-block χ₁={chi_block:.4f} (attn={chi_attn:.4f}, ffn={chi_ffn:.4f}). "
            f"FW corrections: O(1/d_k)={1.0/max(d_k,1):.4f}. "
            f"Total χ₁^L={chi_total:.4e}. Variance growth: {var_growth:.2f}×. "
            f"Phase: {phase}. σ_w*={sigma_w_star:.4f}."
        )

        return TransformerMFReport(
            phase=phase,
            chi_1_attn=chi_attn,
            chi_1_ffn=chi_ffn,
            chi_1_block=chi_block,
            chi_1_total=chi_total,
            variance_trajectory=var_traj,
            attn_variance_trajectory=attn_var_traj,
            ffn_variance_trajectory=ffn_var_traj,
            depth_scale=depth_scale,
            sigma_w_star=sigma_w_star,
            explanation=explanation,
            has_layernorm=True,
            head_dim=d_k,
        )

    def diagnose(self, spec: TransformerSpec) -> Dict:
        """Quick diagnostic for a Transformer architecture.

        Returns a dict with actionable recommendations.
        """
        report = self.analyze(spec)

        issues = []
        recommendations = []

        if report.chi_1_block > 1.5:
            issues.append("High per-block susceptibility — gradient explosion risk")
            recommendations.append(
                f"Reduce σ_w from {spec.sigma_w:.4f} to {report.sigma_w_star:.4f}"
            )

        if report.chi_1_total > 1e6:
            issues.append(f"Total gradient amplification {report.chi_1_total:.2e}× over {spec.n_layers} layers")
            recommendations.append("Use gradient clipping or reduce depth")

        if not spec.pre_ln and spec.n_layers > 6:
            issues.append("Post-LN with >6 layers risks training instability")
            recommendations.append("Switch to Pre-LN architecture")

        if report.phase == "critical" and not issues:
            recommendations.append("Initialization is well-configured")

        return {
            "report": report,
            "issues": issues,
            "recommendations": recommendations,
            "sigma_w_current": spec.sigma_w,
            "sigma_w_recommended": report.sigma_w_star,
        }


# ======================================================================
# GPT-2 style model for experiments
# ======================================================================

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (GPT-2 style)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 max_seq_len: int = 512):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x: "Tensor") -> "Tensor":
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return y


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block (GPT-2 style)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 activation: str = "gelu", dropout: float = 0.0,
                 max_seq_len: int = 512):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: "Tensor") -> "Tensor":
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Minimal GPT-2 style model for phase diagram experiments.

    Parameters
    ----------
    vocab_size : int
    d_model : int
    n_heads : int
    n_layers : int
    d_ff : int
    max_seq_len : int
    activation : str
    dropout : float
    """

    def __init__(self, vocab_size: int = 1000, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 4, d_ff: int = 512,
                 max_seq_len: int = 128, activation: str = "gelu",
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, activation, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: "Tensor") -> "Tensor":
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def to_transformer_spec(self, seq_len: int = 128) -> TransformerSpec:
        """Convert to TransformerSpec for mean-field analysis."""
        block = self.blocks[0]
        sigma_ws = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.in_features
                sigma_ws.append(float(m.weight.data.std().item() * math.sqrt(fan_in)))
        sigma_w = float(np.median(sigma_ws)) if sigma_ws else 1.0

        # Estimate input variance from embeddings
        emb_var = float(self.tok_emb.weight.data.var().item())
        pos_var = float(self.pos_emb.weight.data.var().item())
        input_variance = emb_var + pos_var

        return TransformerSpec(
            n_layers=len(self.blocks),
            d_model=self.d_model,
            n_heads=block.attn.n_heads,
            d_ff=block.ffn[0].out_features,
            activation="gelu",
            sigma_w=sigma_w,
            pre_ln=True,
            input_variance=input_variance,
            seq_len=seq_len,
            is_causal=True,
        )
