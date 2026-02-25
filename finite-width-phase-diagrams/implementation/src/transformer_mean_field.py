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

import numpy as np
from scipy.integrate import quad
from scipy.special import erf
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Callable
import warnings

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
                                  sigma_w: float, n_heads: int = 8) -> float:
        """Compute post-attention variance for a single self-attention layer.

        For input with variance q_in, QKV projections with scale sigma_w,
        and softmax attention with head dimension d_k:

        The QK^T product has entries ~ N(0, d_k * q_in^2 * sigma_w^4)
        before the 1/sqrt(d_k) scaling, giving logits ~ N(0, q_in^2 * sigma_w^4).
        After softmax and value multiplication, the output variance is:

            Var[attn_out] ≈ sigma_w^2 * q_in * E[||softmax(s)||^2]

        where s are the attention logits. For Gaussian logits with variance
        σ_s^2 = q_in * sigma_w^4, the expected squared softmax norm
        E[||p||^2] ≈ 1/n_eff where n_eff is the effective number of
        attended positions (depends on σ_s).

        At initialization (small σ_w), attention is approximately uniform,
        so E[||p||^2] ≈ 1/seq_len and output variance ≈ σ_w^2 * q_in.

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

        Returns
        -------
        float
            Post-attention variance (before output projection).
        """
        if q_in <= 0:
            return 0.0

        # Logit variance after 1/sqrt(d_k) scaling
        # Each logit l_ij = (q_i^T k_j) / sqrt(d_k)
        # With q,k having variance sigma_w^2 * q_in per component,
        # q^T k / sqrt(d_k) has variance ~ sigma_w^4 * q_in^2 * d_k / d_k = sigma_w^4 * q_in^2
        # But for properly scaled init (sigma_w ~ 1/sqrt(fan_in)):
        sigma_logit_sq = sigma_w ** 4 * q_in ** 2

        # For small logit variance (near init), softmax ≈ uniform
        # E[||softmax(s)||^2] ≈ 1/n + var(s) * (n-1)/n^2 for Gaussian logits
        # We use the approximation that at init, attention is near-uniform
        # Key insight: the output projection further scales by sigma_w^2
        # Total: Var[output] = sigma_w^2 * sigma_w^2 * q_in (V proj + O proj)
        # = sigma_w^4 * q_in... but for standard init sigma_w ≈ 1, so ≈ q_in

        # Value projection variance: sigma_w^2 * q_in
        q_value = sigma_w ** 2 * q_in

        # Attention-weighted output: for near-uniform attention (at init),
        # output is approx average of values, preserving variance.
        # For concentrated attention (large sigma_logit), variance can grow.
        # Softmax concentration factor: higher logit variance → more peaked softmax
        concentration = 1.0 / (1.0 + sigma_logit_sq)

        # Output projection: another sigma_w^2 factor
        q_attn_out = sigma_w ** 2 * q_value * concentration

        return max(q_attn_out, 0.0)

    def attention_chi1(self, q_in: float, d_k: int, sigma_w: float) -> float:
        """Susceptibility of the attention sublayer.

        The Jacobian of attention w.r.t. input has norm that depends on
        the attention pattern concentration. At initialization with
        near-uniform attention:

            chi_1^attn ≈ sigma_w^4 (from W_Q, W_K, W_V, W_O)

        More precisely, for the value pathway (which dominates at init):
            chi_1^attn ≈ sigma_w^4 * (1 + O(sigma_w^4 * q_in))

        Parameters
        ----------
        q_in : float
            Input variance.
        d_k : int
            Per-head dimension.
        sigma_w : float
            Weight scale.

        Returns
        -------
        float
            Attention sublayer susceptibility.
        """
        # At initialization: near-uniform attention
        # Value path Jacobian: dout/din = W_O @ softmax_weighted @ W_V
        # ||J||^2_F / d_model ≈ sigma_w^4
        # The QK path contributes O(sigma_w^8) at init (subdominant)
        chi_attn = sigma_w ** 4
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
            # Pre-LN: LN first, then sublayer, then residual
            # After LN: variance = 1
            q_ln1 = self.layernorm_effect(q_in)
            q_attn = self.attention_output_variance(q_ln1, d_k, spec.sigma_w, spec.n_heads)
            # Residual: Var[h + attn(LN(h))] = q_in + q_attn (uncorrelated at init)
            q_after_attn = q_in + q_attn

            q_ln2 = self.layernorm_effect(q_after_attn)
            q_ffn = self.ffn_variance(q_ln2, spec.sigma_w, spec.activation,
                                      spec.d_ff / spec.d_model)
            # Residual
            q_out = q_after_attn + q_ffn
        else:
            # Post-LN: sublayer first, then residual, then LN
            q_attn = self.attention_output_variance(q_in, d_k, spec.sigma_w, spec.n_heads)
            q_after_attn = q_in + q_attn
            q_after_attn = self.layernorm_effect(q_after_attn)

            q_ffn = self.ffn_variance(q_after_attn, spec.sigma_w, spec.activation,
                                      spec.d_ff / spec.d_model)
            q_out = q_after_attn + q_ffn
            q_out = self.layernorm_effect(q_out)

        # Dropout scales variance by 1/(1-p)^2 during training
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

        # LN resets variance to 1
        q_ln = 1.0

        chi_attn = self.attention_chi1(q_ln, d_k, spec.sigma_w)
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

        # Compute per-block susceptibilities
        q_ln = 1.0  # LN resets to unit variance
        chi_attn = self.attention_chi1(q_ln, d_k, spec.sigma_w)
        chi_ffn = self.ffn_chi1(q_ln, spec.sigma_w, spec.activation)
        chi_block = self.block_chi1(q_ln, spec)

        # Total susceptibility across L blocks
        # For Pre-LN: chi_total = chi_block^L (each block compounds)
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

        # Phase classification
        # For Transformers with Pre-LN, chi_block >= 1 always due to residual.
        # Critical: chi_block ≈ 1 (small sublayer contributions)
        # Chaotic: chi_block >> 1 (gradient explosion)
        # Ordered: only possible without residual (rare in practice)
        if chi_block < 0.99:
            phase = "ordered"
        elif chi_block < 1.0 + 0.1 * np.log(10.0) / max(spec.n_layers, 1):
            phase = "critical"
        else:
            phase = "chaotic"

        # Find recommended sigma_w for edge-of-chaos
        # At edge: chi_block = 1, meaning chi_attn + chi_ffn = 0
        # With residual this means sigma_w → 0. For Transformers,
        # the goal is chi_block ≈ 1 + epsilon (slow gradient growth).
        # Practical target: chi_block^L ≈ e (Noci et al.)
        target_chi_block = np.exp(1.0 / max(spec.n_layers, 1))
        # chi_block = 1 + sigma_w^4 + sigma_w^4 * chi_act(sigma_w^2)
        # Solve numerically
        sigma_w_star = self._find_optimal_sigma_w(spec, target_chi_block)

        explanation = (
            f"Transformer: {spec.n_layers}L, d_model={spec.d_model}, "
            f"{spec.n_heads}H, d_ff={spec.d_ff}, act={spec.activation}, "
            f"{'Pre' if spec.pre_ln else 'Post'}-LN. "
            f"Per-block χ₁={chi_block:.4f} (attn={chi_attn:.4f}, ffn={chi_ffn:.4f}). "
            f"Total χ₁^L={chi_total:.4e} over {spec.n_layers} layers. "
            f"Phase: {phase}. "
            f"σ_w*={sigma_w_star:.4f} for gradual gradient growth. "
            f"Depth scale: {depth_scale:.1f} blocks."
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
