"""
Residual network NTK computation with finite-width corrections and
depth-dependent phase transitions.

Implements:
  - ResNetNTKConfig: architecture specification
  - SkipConnectionKernel: NTK recursion through skip connections
  - BatchNormResNetKernel: batch-norm effects in residual blocks
  - PreActivationResNet: pre-act vs post-act block kernels
  - SignalPropagationResNet: mean-field signal propagation and phase diagrams
  - ResNetFiniteWidthCorrections: O(1/N) corrections for finite width
  - WidthDepthPhaseDiagram: full (width, depth) phase diagram
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import linalg as sp_linalg
from scipy.optimize import brentq, fixed_point, minimize_scalar
from scipy.special import erf


# ======================================================================
# Helpers
# ======================================================================

def _ensure_symmetric(K: np.ndarray) -> np.ndarray:
    return 0.5 * (K + K.T)


def _stable_inv_sqrt(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return 1.0 / np.sqrt(np.maximum(x, eps))


def _centering_matrix(n: int) -> np.ndarray:
    """H = I - (1/n) 11^T."""
    return np.eye(n) - np.ones((n, n)) / n


def _cosine_from_kernel(K: np.ndarray, i: int, j: int) -> float:
    denom = np.sqrt(np.abs(K[i, i] * K[j, j])) + 1e-12
    return K[i, j] / denom


# ------------------------------------------------------------------
# Activation helpers (scalar and expectation forms)
# ------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _relu_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1.0, 0.0)


def _tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def _gelu_derivative(x: np.ndarray) -> np.ndarray:
    phi = 0.5 * (1.0 + erf(x / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * x ** 2) / math.sqrt(2.0 * math.pi)
    return phi + x * pdf


_ACTIVATIONS: Dict[str, Callable] = {
    "relu": _relu,
    "tanh": np.tanh,
    "gelu": _gelu,
}

_ACTIVATION_DERIVATIVES: Dict[str, Callable] = {
    "relu": _relu_derivative,
    "tanh": _tanh_derivative,
    "gelu": _gelu_derivative,
}


def _dual_activation_kernel(
    q11: float, q12: float, q22: float, activation: str,
    n_mc: int = 10_000,
) -> Tuple[float, float]:
    """Compute E[φ(u)φ(v)] and E[φ'(u)φ'(v)] for correlated Gaussians.

    (u, v) ~ N(0, [[q11, q12], [q12, q22]]).

    Returns
    -------
    kappa : E[φ(u)φ(v)]
    dot_kappa : E[φ'(u)φ'(v)]
    """
    act_fn = _ACTIVATIONS[activation]
    act_deriv = _ACTIVATION_DERIVATIVES[activation]

    # Cholesky for sampling correlated Gaussians
    eps = 1e-12
    L11 = math.sqrt(max(q11, eps))
    rho = q12 / (L11 * math.sqrt(max(q22, eps)) + eps)
    rho = np.clip(rho, -1.0 + eps, 1.0 - eps)
    L22 = math.sqrt(max(q22 * (1.0 - rho ** 2), eps))

    rng = np.random.RandomState(42)
    z1 = rng.randn(n_mc)
    z2 = rng.randn(n_mc)
    u = L11 * z1
    v = math.sqrt(max(q22, eps)) * (rho * z1 + math.sqrt(max(1.0 - rho ** 2, eps)) * z2)

    kappa = float(np.mean(act_fn(u) * act_fn(v)))
    dot_kappa = float(np.mean(act_deriv(u) * act_deriv(v)))
    return kappa, dot_kappa


def _relu_kappa_analytic(q11: float, q12: float, q22: float) -> Tuple[float, float]:
    """Analytic dual-activation kernel for ReLU.

    Uses the arc-cosine kernel formula:
        kappa_1(θ) = (1/(2π)) * (sin θ + (π - θ) cos θ) * √(q11 q22)
        dot_kappa  = (π - θ) / (2π)
    where cos θ = q12 / √(q11 q22).
    """
    eps = 1e-12
    denom = math.sqrt(max(q11, eps) * max(q22, eps))
    cos_theta = np.clip(q12 / (denom + eps), -1.0, 1.0)
    theta = math.acos(cos_theta)

    kappa = (denom / (2.0 * math.pi)) * (
        math.sin(theta) + (math.pi - theta) * cos_theta
    )
    dot_kappa = (math.pi - theta) / (2.0 * math.pi)
    return kappa, dot_kappa


def _activation_kernel(
    q11: float, q12: float, q22: float, activation: str,
) -> Tuple[float, float]:
    """Dispatch to analytic formula when available, else MC."""
    if activation == "relu":
        return _relu_kappa_analytic(q11, q12, q22)
    return _dual_activation_kernel(q11, q12, q22, activation)


# ======================================================================
# Config
# ======================================================================

@dataclass
class ResNetNTKConfig:
    """Configuration for a residual network NTK computation.

    Parameters
    ----------
    width : int
        Hidden layer width N.
    depth : int
        Number of residual blocks L.
    activation : str
        Activation function: 'relu', 'tanh', or 'gelu'.
    use_batchnorm : bool
        Whether batch normalization is applied inside blocks.
    use_preactivation : bool
        If True use pre-activation residual blocks (BN-Act-Conv);
        if False use post-activation (Conv-BN-Act).
    skip_weight : float
        Scaling factor α for the residual branch:
        y = x + α · F(x).
    """

    width: int = 512
    depth: int = 20
    activation: str = "relu"
    use_batchnorm: bool = False
    use_preactivation: bool = True
    skip_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{self.activation}'; "
                f"choose from {list(_ACTIVATIONS.keys())}"
            )
        if self.width < 1:
            raise ValueError("width must be >= 1")
        if self.depth < 1:
            raise ValueError("depth must be >= 1")


# ======================================================================
# 1. SkipConnectionKernel
# ======================================================================

class SkipConnectionKernel:
    """NTK computation for networks with skip (residual) connections.

    A residual block computes  y = x + α F(x)  where F is the residual
    branch (typically two convolution/dense layers with activation).  The
    kernel of the full block mixes identity and residual-path kernels.

    The NTK of the full depth-L residual network is built by recursively
    composing block kernels, accumulating gradient contributions from each
    layer.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """
        Parameters
        ----------
        alpha : float
            Residual branch weight.  y = x + alpha * F(x).
        """
        self.alpha = alpha

    # -----------------------------------------------------------------
    # Block-level kernel
    # -----------------------------------------------------------------

    def residual_block_kernel(
        self,
        K_identity: np.ndarray,
        K_residual: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Kernel of a single residual block y = x + α F(x).

        K_block = K_identity + α² K_residual + α K_cross

        where K_cross ≈ symmetrised product of identity and residual
        paths (upper-bound approximation when exact cross term is
        unavailable).

        Parameters
        ----------
        K_identity : (n, n) kernel matrix of the identity path.
        K_residual : (n, n) kernel matrix of the residual branch.
        alpha : residual scaling factor.

        Returns
        -------
        K_block : (n, n) combined block kernel.
        """
        K_cross = self.cross_term_kernel_approx(K_identity, K_residual)
        K_block = K_identity + alpha ** 2 * K_residual + alpha * K_cross
        return _ensure_symmetric(K_block)

    def cross_term_kernel(
        self,
        X: np.ndarray,
        W: np.ndarray,
        activation_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Exact cross-kernel between the skip path and the residual path.

        K_cross(x, x') = E_W[ <x, φ(Wx)> <x', φ(Wx')> ] (unnormalised).

        Estimated via Monte-Carlo over weight matrices W.

        Parameters
        ----------
        X : (n, d) input data matrix.
        W : (d_out, d) weight matrix sample (or list thereof).
        activation_fn : element-wise activation.

        Returns
        -------
        K_cross : (n, n)
        """
        n, d = X.shape
        # single sample estimate: K_cross[i,j] = <x_i, φ(W x_j)> + transpose
        preact = X @ W.T  # (n, d_out)
        postact = activation_fn(preact)

        # Cross term: identity * residual inner products
        # <x_i, φ(W x_i)> style — average over output dims
        K_cross = (X @ postact.T) / max(W.shape[0], 1)
        K_cross = 0.5 * (K_cross + K_cross.T)
        return K_cross

    def cross_term_kernel_approx(
        self,
        K_identity: np.ndarray,
        K_residual: np.ndarray,
    ) -> np.ndarray:
        """Approximate cross-kernel via geometric mean of the two paths.

        K_cross ≈ 2 * (K_id ⊙ K_res)^{1/2}    (element-wise)

        This is the Cauchy–Schwarz-saturated bound and works well when
        identity and residual features are moderately correlated.

        Parameters
        ----------
        K_identity, K_residual : (n, n) kernel matrices.

        Returns
        -------
        K_cross : (n, n)
        """
        # Element-wise geometric mean (safe for positive semi-definite)
        product = np.abs(K_identity) * np.abs(K_residual)
        K_cross = 2.0 * np.sqrt(product) * np.sign(K_identity * K_residual)
        return _ensure_symmetric(K_cross)

    # -----------------------------------------------------------------
    # Composing blocks
    # -----------------------------------------------------------------

    def compose_blocks(
        self,
        block_kernels: List[np.ndarray],
        alphas: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Compose L blocks' NNGP kernels sequentially.

        K^{l+1} = residual_block_kernel(K^l, F_kernel(K^l), α_l)

        Parameters
        ----------
        block_kernels : list of (n, n) residual-branch kernels per block.
        alphas : per-block residual weights; defaults to [self.alpha]*L.

        Returns
        -------
        K_final : (n, n) composed kernel after all blocks.
        """
        L = len(block_kernels)
        if alphas is None:
            alphas = [self.alpha] * L
        if len(alphas) != L:
            raise ValueError("len(alphas) must equal len(block_kernels)")

        K = block_kernels[0].copy()
        for l in range(L):
            K_res = block_kernels[l]
            K = self.residual_block_kernel(K, K_res, alphas[l])
        return K

    # -----------------------------------------------------------------
    # Recursive NTK through depth
    # -----------------------------------------------------------------

    def ntk_residual_recursion(
        self,
        K0: np.ndarray,
        sigma_w: float,
        sigma_b: float,
        depth: int,
        alpha: float,
        activation: str = "relu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Recursively compute the NTK of a depth-L residual network.

        At each block l:
            q^{l+1}_{ij} = q^l_{ij} + α² (σ_w² κ(q^l) + σ_b²)
            Θ^{l+1} = Θ^l · (1 + α² σ_w² κ̇(q^l)) + (σ_w² κ(q^l) + σ_b²)

        Parameters
        ----------
        K0 : (n, n) initial kernel (input Gram matrix, normalised).
        sigma_w, sigma_b : weight and bias variance scales.
        depth : number of residual blocks L.
        alpha : residual weight.
        activation : activation function name.

        Returns
        -------
        K_nngp : (n, n) NNGP kernel after depth L.
        K_ntk  : (n, n) NTK after depth L.
        """
        n = K0.shape[0]
        K = K0.copy()
        Theta = sigma_w ** 2 * K + sigma_b ** 2 * np.ones((n, n))

        for _ in range(depth):
            K_new = np.zeros_like(K)
            Theta_new = np.zeros_like(Theta)
            for i in range(n):
                for j in range(i, n):
                    q11, q12, q22 = K[i, i], K[i, j], K[j, j]
                    kappa, dot_kappa = _activation_kernel(
                        q11, q12, q22, activation
                    )
                    # NNGP recursion: K^{l+1} = K^l + α²(σ_w² κ + σ_b²)
                    K_res = sigma_w ** 2 * kappa + sigma_b ** 2
                    K_new[i, j] = K[i, j] + alpha ** 2 * K_res
                    K_new[j, i] = K_new[i, j]

                    # NTK recursion
                    Theta_new[i, j] = (
                        Theta[i, j] * (1.0 + alpha ** 2 * sigma_w ** 2 * dot_kappa)
                        + K_res
                    )
                    Theta_new[j, i] = Theta_new[i, j]

            K = K_new
            Theta = Theta_new

        return _ensure_symmetric(K), _ensure_symmetric(Theta)

    # -----------------------------------------------------------------
    # Gradient flow analysis
    # -----------------------------------------------------------------

    def skip_gradient_contribution(self, depth: int) -> np.ndarray:
        r"""Gradient flow factor through skip connections at each layer.

        For a residual network, the gradient at layer l receives a
        multiplicative factor from each subsequent skip:

            ∏_{k=l}^{L-1} (1 + α ∂F_k/∂h_k)

        In the infinite-width mean-field limit, ∂F_k/∂h_k → 0 so the
        product ≈ 1 for all l (gradient "highway").  For finite width
        there are fluctuations; we return the mean-field baseline here.

        Parameters
        ----------
        depth : number of blocks L.

        Returns
        -------
        contributions : array of shape (depth,), mean-field gradient norm
            scaling at each block.
        """
        # Mean-field: each skip contributes factor 1 → product = 1 for all l
        # Fluctuations scale as O(α² / N) per block
        contributions = np.ones(depth)
        return contributions

    def effective_depth(self, alpha: float, nominal_depth: int) -> float:
        r"""Effective depth of a residual network as a function of skip weight.

        The effective depth quantifies how many layers actually contribute
        to feature transformation.  When α → 0 the network is essentially
        the identity (effective depth → 0); when α → ∞ the skip path is
        negligible and effective depth → nominal depth.

        A common heuristic is:

            L_eff = L · α² / (1 + α²)

        Parameters
        ----------
        alpha : residual branch weight.
        nominal_depth : total number of blocks L.

        Returns
        -------
        L_eff : float, effective depth.
        """
        return nominal_depth * alpha ** 2 / (1.0 + alpha ** 2)


# ======================================================================
# 2. BatchNormResNetKernel
# ======================================================================

class BatchNormResNetKernel:
    """Batch normalisation effects on the NTK inside residual blocks.

    Batch-norm centres and rescales pre-activations, which modifies both
    the NNGP and NTK kernels and introduces data-dependent Jacobian
    terms into the NTK.
    """

    def __init__(self, momentum: float = 0.1, eps: float = 1e-5) -> None:
        self.momentum = momentum
        self.eps = eps

    # -----------------------------------------------------------------
    def batchnorm_kernel_transform(
        self,
        K_pre: np.ndarray,
        batch_stats: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Transform the kernel K through a batch-norm layer.

        BN centres and rescales:
            x̂ = (x − μ_B) / √(σ²_B + ε)

        In kernel space this is:
            K_bn = D^{-1/2} H K H D^{-1/2}

        where H is the centering matrix and D = diag(σ²_B + ε).

        Parameters
        ----------
        K_pre : (n, n) kernel matrix before BN.
        batch_stats : dict with keys 'mean' (n,) and 'var' (n,).

        Returns
        -------
        K_bn : (n, n) kernel after BN.
        """
        n = K_pre.shape[0]
        var = batch_stats["var"]
        inv_std = _stable_inv_sqrt(var + self.eps)

        H = _centering_matrix(n)
        K_centered = H @ K_pre @ H

        D_inv_sqrt = np.diag(inv_std)
        K_bn = D_inv_sqrt @ K_centered @ D_inv_sqrt
        return _ensure_symmetric(K_bn)

    # -----------------------------------------------------------------
    def running_stats_effect(
        self,
        K: np.ndarray,
        training_steps: int,
    ) -> np.ndarray:
        """Effect of exponential-moving-average running statistics on the kernel.

        During training BN uses batch statistics; at test time it uses
        running estimates  μ_run = (1-m) μ_run + m μ_batch.  After T
        steps the running variance converges as:

            σ²_run(T) ≈ σ²_true + O((1-m)^T)

        We model the kernel as interpolating between train-time (centred)
        and converged (identity-scaled) behaviour.

        Parameters
        ----------
        K : (n, n) kernel matrix (train-time, centred).
        training_steps : number of training updates seen.

        Returns
        -------
        K_adjusted : (n, n) kernel reflecting running-stat convergence.
        """
        # Convergence factor: how close running stats are to true stats
        gamma = 1.0 - (1.0 - self.momentum) ** training_steps
        # At convergence (γ→1) the centering effect is fully absorbed
        # into the affine parameters; kernel → rescaled version of K
        n = K.shape[0]
        K_identity = np.diag(np.diag(K))
        K_adjusted = gamma * K + (1.0 - gamma) * K_identity
        return _ensure_symmetric(K_adjusted)

    # -----------------------------------------------------------------
    def batchnorm_gradient_jacobian(
        self,
        X: np.ndarray,
        mean: np.ndarray,
        var: np.ndarray,
    ) -> np.ndarray:
        """Jacobian of BN output with respect to its input for NTK.

        ∂x̂_i/∂x_j = (δ_{ij} − 1/n) / √(σ² + ε)
                     − x̂_i (x_j − μ) / (n (σ² + ε))

        Parameters
        ----------
        X : (n, d) input batch.
        mean : (d,) batch mean.
        var : (d,) batch variance.

        Returns
        -------
        J : (n, n) Jacobian of the BN mapping (per-feature averaged).
        """
        n, d = X.shape
        inv_std = _stable_inv_sqrt(var + self.eps)  # (d,)
        X_centered = X - mean[np.newaxis, :]  # (n, d)
        X_hat = X_centered * inv_std[np.newaxis, :]  # (n, d)

        # Average the Jacobian over features
        # J_{ij} = (1/d) Σ_k [ (δ_{ij} - 1/n) / σ_k
        #           - x̂_{ik} (x_{jk} - μ_k) / (n σ_k²) ]
        eye = np.eye(n)
        term1 = (eye - 1.0 / n) * np.mean(inv_std)
        # Second term: (1/d) Σ_k x̂_{ik} (x_{jk} - μ_k) / (n σ_k²)
        weights = inv_std ** 2 / (n * d)
        term2 = (X_hat * inv_std[np.newaxis, :]) @ X_centered.T / (n * d)
        # Correct scaling
        term2 = X_hat @ (X_centered * (inv_std ** 2)[np.newaxis, :]).T / (n * d)
        J = term1 - term2
        return J

    # -----------------------------------------------------------------
    def bn_ntk_correction(
        self,
        K_ntk: np.ndarray,
        X: np.ndarray,
        bn_params: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Correct the NTK for the presence of batch normalisation.

        BN introduces additional parameter-dependence via batch statistics.
        The corrected NTK is:

            Θ_bn = J_bn Θ J_bn^T + Θ_affine

        where J_bn is the Jacobian of BN and Θ_affine captures the
        learnable scale/shift parameters (γ, β).

        Parameters
        ----------
        K_ntk : (n, n) NTK without BN correction.
        X : (n, d) input data.
        bn_params : dict with 'mean' (d,), 'var' (d,), 'gamma' (d,),
                     'beta' (d,).

        Returns
        -------
        K_ntk_corrected : (n, n)
        """
        mean = bn_params["mean"]
        var = bn_params["var"]
        gamma = bn_params.get("gamma", np.ones_like(mean))

        J = self.batchnorm_gradient_jacobian(X, mean, var)

        # Transform NTK through BN Jacobian
        K_corrected = J @ K_ntk @ J.T

        # Affine parameters (γ, β) contribute an additive rank-2n kernel
        n, d = X.shape
        inv_std = _stable_inv_sqrt(var + self.eps)
        X_hat = (X - mean[np.newaxis, :]) * inv_std[np.newaxis, :]

        # ∂(γ x̂ + β)/∂γ = x̂,  ∂/∂β = 1
        # Θ_affine = x̂ x̂^T / d  +  11^T / d
        K_gamma = (X_hat @ X_hat.T) / d
        K_beta = np.ones((n, n)) / d
        K_affine = K_gamma + K_beta

        K_ntk_corrected = K_corrected + K_affine
        return _ensure_symmetric(K_ntk_corrected)


# ======================================================================
# 3. PreActivationResNet
# ======================================================================

class PreActivationResNet:
    """Compare pre-activation and post-activation residual block kernels.

    Pre-activation block:  BN → Act → Conv → BN → Act → Conv  (+ skip)
    Post-activation block: Conv → BN → Act → Conv → BN → Act  (+ skip)

    These orderings change the kernel recursion because the activation
    function sees normalised vs un-normalised inputs.
    """

    def __init__(self, config: ResNetNTKConfig) -> None:
        self.config = config
        self._act = config.activation

    # -----------------------------------------------------------------
    def preact_block_kernel(
        self,
        K_in: np.ndarray,
        sigma_w: float,
        sigma_b: float,
    ) -> np.ndarray:
        """Kernel of a pre-activation residual block.

        BN → Act → Conv → BN → Act → Conv:
            1. Normalise K_in  (BN centering / rescaling).
            2. Apply activation kernel κ.
            3. Linear map: K ← σ_w² κ(K) + σ_b².
            4. Repeat steps 1-3 for the second sub-layer.

        Parameters
        ----------
        K_in : (n, n) input kernel.
        sigma_w, sigma_b : weight / bias variance.

        Returns
        -------
        K_block : (n, n) residual-branch kernel (before adding skip).
        """
        n = K_in.shape[0]
        K = K_in.copy()

        # --- first sub-layer: BN → Act → Conv ---
        # BN: normalise diagonal to 1
        diag = np.sqrt(np.diag(K)) + 1e-12
        K = K / np.outer(diag, diag)

        # Activation kernel
        K_act = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                kappa, _ = _activation_kernel(K[i, i], K[i, j], K[j, j], self._act)
                K_act[i, j] = kappa
                K_act[j, i] = kappa
        # Linear
        K = sigma_w ** 2 * K_act + sigma_b ** 2

        # --- second sub-layer: BN → Act → Conv ---
        diag = np.sqrt(np.diag(K)) + 1e-12
        K = K / np.outer(diag, diag)

        K_act2 = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                kappa, _ = _activation_kernel(K[i, i], K[i, j], K[j, j], self._act)
                K_act2[i, j] = kappa
                K_act2[j, i] = kappa
        K_block = sigma_w ** 2 * K_act2 + sigma_b ** 2

        return _ensure_symmetric(K_block)

    # -----------------------------------------------------------------
    def postact_block_kernel(
        self,
        K_in: np.ndarray,
        sigma_w: float,
        sigma_b: float,
    ) -> np.ndarray:
        """Kernel of a post-activation residual block.

        Conv → BN → Act → Conv → BN → Act:
            1. Linear map.
            2. Normalise.
            3. Activation kernel.
            4. Repeat for second sub-layer.

        Parameters
        ----------
        K_in : (n, n) input kernel.
        sigma_w, sigma_b : weight / bias variance.

        Returns
        -------
        K_block : (n, n) residual-branch kernel.
        """
        n = K_in.shape[0]

        # --- first sub-layer: Conv → BN → Act ---
        K = sigma_w ** 2 * K_in + sigma_b ** 2

        diag = np.sqrt(np.diag(K)) + 1e-12
        K = K / np.outer(diag, diag)

        K_act = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                kappa, _ = _activation_kernel(K[i, i], K[i, j], K[j, j], self._act)
                K_act[i, j] = kappa
                K_act[j, i] = kappa

        # --- second sub-layer: Conv → BN → Act ---
        K = sigma_w ** 2 * K_act + sigma_b ** 2

        diag = np.sqrt(np.diag(K)) + 1e-12
        K = K / np.outer(diag, diag)

        K_act2 = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                kappa, _ = _activation_kernel(K[i, i], K[i, j], K[j, j], self._act)
                K_act2[i, j] = kappa
                K_act2[j, i] = kappa

        K_block = K_act2
        return _ensure_symmetric(K_block)

    # -----------------------------------------------------------------
    def compare_prepost(
        self,
        K_in: np.ndarray,
        sigma_w: float,
        sigma_b: float,
    ) -> Dict[str, Any]:
        """Compare pre-activation and post-activation block kernels.

        Returns kernel matrices and summary statistics (Frobenius distance,
        spectral differences).

        Parameters
        ----------
        K_in : (n, n) input kernel.
        sigma_w, sigma_b : weight / bias variance.

        Returns
        -------
        result : dict with keys 'pre', 'post', 'frob_diff',
                 'max_eigval_pre', 'max_eigval_post', 'spectral_gap'.
        """
        K_pre = self.preact_block_kernel(K_in, sigma_w, sigma_b)
        K_post = self.postact_block_kernel(K_in, sigma_w, sigma_b)

        frob_diff = float(np.linalg.norm(K_pre - K_post, "fro"))
        eig_pre = float(np.max(np.linalg.eigvalsh(K_pre)))
        eig_post = float(np.max(np.linalg.eigvalsh(K_post)))

        return {
            "pre": K_pre,
            "post": K_post,
            "frob_diff": frob_diff,
            "max_eigval_pre": eig_pre,
            "max_eigval_post": eig_post,
            "spectral_gap": abs(eig_pre - eig_post),
        }


# ======================================================================
# 4. SignalPropagationResNet
# ======================================================================

class SignalPropagationResNet:
    """Mean-field signal propagation through residual networks.

    Tracks the evolution of the variance (diagonal kernel element) and
    the correlation (off-diagonal) through depth, determining the
    ordered / chaotic / edge-of-chaos phases.

    Key quantities:
        q^l     : pre-activation variance at layer l
        c^l     : correlation between two inputs at layer l
        χ       : Lyapunov exponent (derivative of the variance map)
        ξ       : correlation length (depth scale)
    """

    def __init__(self, config: ResNetNTKConfig) -> None:
        self.config = config
        self._act = config.activation
        self._alpha = config.skip_weight

    # -----------------------------------------------------------------
    # Forward propagation of mean / variance
    # -----------------------------------------------------------------

    def forward_mean_variance(
        self,
        q_init: float,
        sigma_w: float,
        sigma_b: float,
        depth: int,
    ) -> np.ndarray:
        r"""Propagate the pre-activation variance q^l through depth.

        For a plain network:
            q^{l+1} = σ_w² E_{z~N(0,q^l)}[φ(z)²] + σ_b²

        For a residual network with skip weight α:
            q^{l+1} = q^l + α² (σ_w² E[φ(z)²] + σ_b²)

        Parameters
        ----------
        q_init : initial variance q^0 (typically the input norm²).
        sigma_w, sigma_b : weight / bias variance.
        depth : number of layers L.

        Returns
        -------
        q_trajectory : array of shape (depth + 1,) giving q^0 … q^L.
        """
        q_traj = np.zeros(depth + 1)
        q_traj[0] = q_init
        alpha = self._alpha

        for l in range(depth):
            q = q_traj[l]
            # E[φ(z)²] for z ~ N(0, q) = κ(q, q, q)  (diagonal)
            kappa_diag, _ = _activation_kernel(q, q, q, self._act)
            q_res = sigma_w ** 2 * kappa_diag + sigma_b ** 2
            q_traj[l + 1] = q + alpha ** 2 * q_res

        return q_traj

    # -----------------------------------------------------------------
    # Backward gradient norm propagation
    # -----------------------------------------------------------------

    def backward_gradient_norm(
        self,
        g_init: float,
        sigma_w: float,
        sigma_b: float,
        depth: int,
    ) -> np.ndarray:
        r"""Propagate gradient norm backward through the network.

        At each layer the gradient norm is multiplied by:
            χ_l = 1 + α² σ_w² E[φ'(z)²]      (residual)

        In the plain network χ_l = σ_w² E[φ'(z)²].  For gradients to
        neither vanish nor explode we need χ ≈ 1.

        We first do a forward pass to get q^l, then compute χ_l at each
        layer.

        Parameters
        ----------
        g_init : gradient norm at the output layer.
        sigma_w, sigma_b : weight / bias variance.
        depth : number of blocks.

        Returns
        -------
        g_trajectory : array of shape (depth + 1,) giving gradient norms
            from layer L down to layer 0.
        """
        # Forward pass to get variances
        q_init = 1.0  # canonical initialisation
        q_traj = self.forward_mean_variance(q_init, sigma_w, sigma_b, depth)

        g_traj = np.zeros(depth + 1)
        g_traj[0] = g_init  # layer L
        alpha = self._alpha

        for l in range(depth):
            q = q_traj[depth - l]
            _, dot_kappa = _activation_kernel(q, q, q, self._act)
            chi_l = 1.0 + alpha ** 2 * sigma_w ** 2 * dot_kappa
            g_traj[l + 1] = g_traj[l] * chi_l

        return g_traj

    # -----------------------------------------------------------------
    # Fixed point
    # -----------------------------------------------------------------

    def fixed_point_qstar(
        self,
        sigma_w: float,
        sigma_b: float,
        activation: str,
    ) -> float:
        r"""Find the fixed point q* of the variance map.

        For a plain network: q* satisfies  q* = σ_w² κ(q*, q*, q*) + σ_b².
        For a residual network the variance diverges linearly with depth
        (no bounded fixed point unless α = 0 or the activation is
        contractive).  In the plain-network limit (α → ∞, rescaled) the
        fixed point exists.

        We solve the plain-network fixed-point equation.

        Parameters
        ----------
        sigma_w, sigma_b : weight / bias variance.
        activation : activation name.

        Returns
        -------
        q_star : float
        """
        def _map(q: float) -> float:
            q = max(q, 1e-12)
            kappa, _ = _activation_kernel(q, q, q, activation)
            return sigma_w ** 2 * kappa + sigma_b ** 2

        # Iterate to convergence
        q = 1.0
        for _ in range(500):
            q_new = _map(q)
            if abs(q_new - q) < 1e-10:
                break
            q = q_new
        return q

    # -----------------------------------------------------------------
    # Lyapunov exponent
    # -----------------------------------------------------------------

    def lyapunov_exponent(
        self,
        q_star: float,
        sigma_w: float,
        activation: str,
    ) -> float:
        r"""Compute the Lyapunov exponent χ = σ_w² E[φ'(h)²] at q*.

        χ < 1 → ordered phase (correlations converge to fixed point)
        χ = 1 → edge of chaos
        χ > 1 → chaotic phase (nearby inputs decorrelate)

        Parameters
        ----------
        q_star : fixed-point variance.
        sigma_w : weight variance.
        activation : activation name.

        Returns
        -------
        chi : float, the Lyapunov exponent.
        """
        _, dot_kappa = _activation_kernel(q_star, q_star, q_star, activation)
        chi = sigma_w ** 2 * dot_kappa
        return chi

    # -----------------------------------------------------------------
    # Edge of chaos
    # -----------------------------------------------------------------

    def edge_of_chaos_boundary(
        self,
        sigma_w_range: np.ndarray,
        sigma_b_range: np.ndarray,
    ) -> np.ndarray:
        r"""Find the χ = 1 boundary in (σ_w, σ_b) space.

        For each σ_b, find the σ_w such that χ(q*(σ_w, σ_b)) = 1.

        Parameters
        ----------
        sigma_w_range : 1-D array of σ_w values to search over.
        sigma_b_range : 1-D array of σ_b values.

        Returns
        -------
        boundary : array of shape (len(sigma_b_range), 2) with columns
            [σ_b, σ_w*] where σ_w* is the critical weight variance.
            Rows where no solution is found contain NaN.
        """
        boundary = np.full((len(sigma_b_range), 2), np.nan)
        act = self._act

        for idx, sb in enumerate(sigma_b_range):
            boundary[idx, 0] = sb

            # χ as a function of σ_w
            def _chi_minus_1(sw: float) -> float:
                q_star = self.fixed_point_qstar(sw, sb, act)
                chi = self.lyapunov_exponent(q_star, sw, act)
                return chi - 1.0

            # Search for root in sigma_w_range
            sw_lo, sw_hi = sigma_w_range[0], sigma_w_range[-1]
            try:
                f_lo = _chi_minus_1(sw_lo)
                f_hi = _chi_minus_1(sw_hi)
                if f_lo * f_hi < 0:
                    sw_star = brentq(_chi_minus_1, sw_lo, sw_hi, xtol=1e-6)
                    boundary[idx, 1] = sw_star
            except (ValueError, RuntimeError):
                pass

        return boundary

    # -----------------------------------------------------------------
    def critical_initialization(
        self,
        activation: str,
    ) -> Tuple[float, float]:
        r"""Find (σ_w*, σ_b*) that places the network at the edge of chaos.

        The critical point satisfies:
            q* = σ_w² κ(q*) + σ_b²
            σ_w² κ̇(q*) = 1

        For ReLU: σ_w* = √2, σ_b* = 0  (He initialisation).
        For tanh: solved numerically.

        Parameters
        ----------
        activation : activation name.

        Returns
        -------
        (sigma_w_star, sigma_b_star) : critical initialisation pair.
        """
        if activation == "relu":
            return (math.sqrt(2.0), 0.0)

        # Numerical solution: fix σ_b = 0, solve for σ_w
        sigma_b = 0.0

        def _chi_minus_1(sw: float) -> float:
            q_star = self.fixed_point_qstar(sw, sigma_b, activation)
            chi = self.lyapunov_exponent(q_star, sw, activation)
            return chi - 1.0

        try:
            sw_star = brentq(_chi_minus_1, 0.1, 5.0, xtol=1e-8)
        except ValueError:
            sw_star = 1.0  # fallback
        return (sw_star, sigma_b)

    # -----------------------------------------------------------------
    def depth_scale(
        self,
        sigma_w: float,
        sigma_b: float,
        activation: str,
    ) -> float:
        r"""Correlation length ξ = −1 / ln(χ).

        ξ characterises the depth scale over which correlations between
        two inputs evolve.  At the edge of chaos ξ → ∞ (correlations
        persist forever); deep in the ordered phase ξ → 0.

        Parameters
        ----------
        sigma_w, sigma_b : initialisation variances.
        activation : activation name.

        Returns
        -------
        xi : float, correlation length in units of layers.  Returns
            np.inf when χ ≥ 1.
        """
        q_star = self.fixed_point_qstar(sigma_w, sigma_b, activation)
        chi = self.lyapunov_exponent(q_star, sigma_w, activation)

        if chi >= 1.0:
            return np.inf
        return -1.0 / math.log(chi)

    # -----------------------------------------------------------------
    def signal_propagation_diagram(
        self,
        sigma_w_range: np.ndarray,
        sigma_b_range: np.ndarray,
        depth: int,
    ) -> Dict[str, np.ndarray]:
        """Compute the full signal-propagation phase diagram.

        For each (σ_w, σ_b) pair, compute the Lyapunov exponent χ, the
        fixed-point variance q*, and the depth scale ξ.

        Parameters
        ----------
        sigma_w_range : 1-D array of σ_w values.
        sigma_b_range : 1-D array of σ_b values.
        depth : network depth (used for forward propagation trajectory).

        Returns
        -------
        diagram : dict with keys
            'chi'   : (len(sigma_b_range), len(sigma_w_range))
            'q_star': same shape
            'xi'    : same shape
            'q_final': same shape (q^L for given depth)
            'sigma_w': the σ_w grid
            'sigma_b': the σ_b grid
        """
        n_sb = len(sigma_b_range)
        n_sw = len(sigma_w_range)
        act = self._act

        chi_map = np.zeros((n_sb, n_sw))
        q_star_map = np.zeros((n_sb, n_sw))
        xi_map = np.zeros((n_sb, n_sw))
        q_final_map = np.zeros((n_sb, n_sw))

        for i, sb in enumerate(sigma_b_range):
            for j, sw in enumerate(sigma_w_range):
                q_star = self.fixed_point_qstar(sw, sb, act)
                chi = self.lyapunov_exponent(q_star, sw, act)

                chi_map[i, j] = chi
                q_star_map[i, j] = q_star

                if chi >= 1.0:
                    xi_map[i, j] = np.inf
                else:
                    xi_map[i, j] = -1.0 / math.log(max(chi, 1e-30))

                q_traj = self.forward_mean_variance(1.0, sw, sb, depth)
                q_final_map[i, j] = q_traj[-1]

        return {
            "chi": chi_map,
            "q_star": q_star_map,
            "xi": xi_map,
            "q_final": q_final_map,
            "sigma_w": sigma_w_range,
            "sigma_b": sigma_b_range,
        }


# ======================================================================
# 5. ResNetFiniteWidthCorrections
# ======================================================================

class ResNetFiniteWidthCorrections:
    """O(1/N) finite-width corrections to the infinite-width NTK of ResNets.

    At finite width N, the NTK is random and fluctuates around its
    infinite-width limit.  The leading correction is O(1/N) and can be
    computed via a fourth-order tensor (the "H-tensor"):

        Var[Θ_{ij}] = (1/N) H_{iijj} + O(1/N²)

    Skip connections and batch normalisation modify the structure of H.
    """

    def __init__(self, config: ResNetNTKConfig) -> None:
        self.config = config
        self._act = config.activation
        self._alpha = config.skip_weight

    # -----------------------------------------------------------------
    def h_tensor_residual_block(
        self,
        X: np.ndarray,
        block_params: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute the H-tensor for a single residual block.

        H_{ijkl} captures the fourth-order correlations needed for the
        O(1/N) NTK variance.  For a dense layer with weights W:

            H_{ijkl} = E[ (∂f_i/∂W)^T (∂f_j/∂W) (∂f_k/∂W)^T (∂f_l/∂W) ]
                       − Θ_{ij} Θ_{kl}

        For a residual block y = x + α F(x), the H-tensor mixes
        identity and residual contributions.

        Parameters
        ----------
        X : (n, d) input data.
        block_params : dict with 'W1' (d_mid, d), 'W2' (d, d_mid) weight
                       matrices for the two sub-layers in the block.

        Returns
        -------
        H : (n, n, n, n) fourth-order tensor.
        """
        n, d = X.shape
        W1 = block_params["W1"]
        W2 = block_params["W2"]
        alpha = self._alpha
        act_fn = _ACTIVATIONS[self._act]
        act_deriv = _ACTIVATION_DERIVATIVES[self._act]

        # Forward through residual branch
        h1 = X @ W1.T  # (n, d_mid)
        a1 = act_fn(h1)
        h2 = a1 @ W2.T  # (n, d)

        # Jacobians of the residual branch w.r.t. W1 (simplified: per-sample)
        # J1_i = ∂F(x_i)/∂vec(W1) = (φ'(W1 x_i) ⊙ W2^T) ⊗ x_i
        da1 = act_deriv(h1)  # (n, d_mid)

        # Approximate H via sample outer products of Jacobian norms
        # H_{ijkl} ≈ (α^4 / d_mid) Σ_m (da1_{im} da1_{jm} da1_{km} da1_{lm})
        #            × (x_i^T x_j)(x_k^T x_l)  −  Θ_{ij} Θ_{kl}
        G = X @ X.T / d  # (n, n) normalised Gram
        M = da1 @ da1.T / W1.shape[0]  # (n, n) activation correlation

        # NTK estimate for this block (mean-field)
        Theta_block = alpha ** 2 * (G * M) + np.eye(n)  # skip + residual

        H = np.zeros((n, n, n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        fourth = float(
                            alpha ** 4 * G[i, j] * G[k, l] * M[i, k] * M[j, l]
                        )
                        H[i, j, k, l] = fourth - Theta_block[i, j] * Theta_block[k, l]

        return H

    # -----------------------------------------------------------------
    def skip_correction_interaction(
        self,
        corrections: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """How skip connections modify the O(1/N) correction tensor.

        The skip path provides an identity contribution to the kernel
        whose variance is zero (deterministic).  The cross term between
        skip and residual is O(α/N).  Therefore:

            H_resnet = α⁴ H_residual + O(α³/N)

        The skip connection *suppresses* finite-width fluctuations by
        the factor α⁴ < 1 when α < 1.

        Parameters
        ----------
        corrections : (n, n, n, n) H-tensor from the residual branch.
        alpha : skip weight.

        Returns
        -------
        H_modified : (n, n, n, n) corrected H-tensor.
        """
        return alpha ** 4 * corrections

    # -----------------------------------------------------------------
    def depth_accumulated_correction(
        self,
        X: np.ndarray,
        all_params: List[Dict[str, np.ndarray]],
        width: int,
    ) -> Dict[str, Any]:
        """Accumulated O(1/N) correction over the full depth.

        The variance of the NTK grows with depth:
            Var[Θ^L] ≈ (1/N) Σ_{l=1}^L  α^{4} ||H^l||

        This gives the scale at which the infinite-width limit breaks
        down as a function of depth.

        Parameters
        ----------
        X : (n, d) input data.
        all_params : list of per-block param dicts (length = depth).
        width : network width N.

        Returns
        -------
        result : dict with
            'per_layer_var'  : (L,) array of per-layer variance contributions
            'cumulative_var' : (L,) cumulative variance
            'breakdown_depth': depth at which Var[Θ] ≈ ||Θ||  (signal ≈ noise)
        """
        L = len(all_params)
        alpha = self._alpha
        n = X.shape[0]

        per_layer_var = np.zeros(L)
        for l in range(L):
            H = self.h_tensor_residual_block(X, all_params[l])
            H_mod = self.skip_correction_interaction(H, alpha)
            # Variance contribution: trace-norm of H / N
            # Use Frobenius norm of the (n²×n²) reshaping
            per_layer_var[l] = np.sum(H_mod ** 2) / (width * n ** 2)

        cumulative_var = np.cumsum(per_layer_var)

        # Breakdown depth: where cumulative var exceeds 1 (normalised)
        threshold = 1.0
        exceeded = np.where(cumulative_var > threshold)[0]
        breakdown_depth = int(exceeded[0]) if len(exceeded) > 0 else L

        return {
            "per_layer_var": per_layer_var,
            "cumulative_var": cumulative_var,
            "breakdown_depth": breakdown_depth,
        }

    # -----------------------------------------------------------------
    def batchnorm_correction(
        self,
        X: np.ndarray,
        bn_params: Dict[str, np.ndarray],
        width: int,
    ) -> np.ndarray:
        """Batch-norm's effect on the finite-width NTK correction.

        BN introduces sample-sample coupling into the Jacobian, which
        changes the covariance structure of the NTK fluctuations.  The
        dominant effect is that BN *reduces* the NTK variance by a
        factor ≈ (1 − 1/n) due to centering.

        Parameters
        ----------
        X : (n, d) input data.
        bn_params : dict with 'mean' (d,), 'var' (d,).
        width : network width N.

        Returns
        -------
        correction : (n, n) matrix of O(1/N) NTK variance reduction
            (subtract from diagonal of Var[Θ]).
        """
        n, d = X.shape
        var = bn_params["var"]
        inv_std = _stable_inv_sqrt(var + 1e-5)

        # Centering reduces fluctuations by removing the mean mode
        H = _centering_matrix(n)
        X_centered = H @ X
        G = X_centered @ (X_centered * (inv_std ** 2)[np.newaxis, :]).T / d

        # The BN-induced variance reduction is proportional to G² / N
        correction = G ** 2 / width
        return correction


# ======================================================================
# 6. WidthDepthPhaseDiagram
# ======================================================================

class WidthDepthPhaseDiagram:
    """Full (width N, depth L) phase diagram for residual networks.

    Classifies each (N, L) point into one of several regimes based on
    the behaviour of the NTK:

        'kernel'   : infinite-width (lazy) regime, NTK ≈ constant
        'feature'  : finite-width feature-learning regime
        'chaotic'  : NTK fluctuations dominate, training unstable
        'dead'     : signal has vanished (ordered phase, too deep)

    The boundaries depend on initialisation (σ_w, σ_b), activation,
    and the skip weight α.
    """

    def __init__(self, config: ResNetNTKConfig) -> None:
        self.config = config
        self._signal = SignalPropagationResNet(config)
        self._corrections = ResNetFiniteWidthCorrections(config)
        self._skip = SkipConnectionKernel(config.skip_weight)

    # -----------------------------------------------------------------
    def compute_phase_point(
        self,
        width: int,
        depth: int,
        sigma_w: float,
        sigma_b: float,
        X: np.ndarray,
    ) -> Dict[str, Any]:
        """Classify a single (width, depth) point.

        Computes the Lyapunov exponent χ, the NTK variance scaling, and
        the effective depth to assign a phase label.

        Parameters
        ----------
        width : network width N.
        depth : number of blocks L.
        sigma_w, sigma_b : initialisation variances.
        X : (n, d) representative data.

        Returns
        -------
        result : dict with keys
            'phase'       : str label
            'chi'         : Lyapunov exponent
            'ntk_var'     : estimated NTK variance (relative to mean)
            'q_final'     : final pre-activation variance
            'eff_depth'   : effective depth
        """
        act = self.config.activation
        alpha = self.config.skip_weight

        # Signal propagation
        q_star = self._signal.fixed_point_qstar(sigma_w, sigma_b, act)
        chi = self._signal.lyapunov_exponent(q_star, sigma_w, act)
        q_traj = self._signal.forward_mean_variance(1.0, sigma_w, sigma_b, depth)
        q_final = float(q_traj[-1])

        # Effective depth
        eff_depth = self._skip.effective_depth(alpha, depth)

        # NTK variance scaling: Var[Θ]/E[Θ]² ∝ L α⁴ / N
        ntk_var_ratio = depth * alpha ** 4 / max(width, 1)

        # Phase classification
        if chi > 1.0 + 0.05:
            phase = "chaotic"
        elif q_final < 1e-6:
            phase = "dead"
        elif ntk_var_ratio > 1.0:
            phase = "chaotic"
        elif ntk_var_ratio < 0.01:
            phase = "kernel"
        else:
            phase = "feature"

        return {
            "phase": phase,
            "chi": chi,
            "ntk_var": ntk_var_ratio,
            "q_final": q_final,
            "eff_depth": eff_depth,
        }

    # -----------------------------------------------------------------
    def sweep_width_depth(
        self,
        widths: np.ndarray,
        depths: np.ndarray,
        sigma_w: float,
        sigma_b: float,
        X: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Sweep a grid of (width, depth) and classify each point.

        Parameters
        ----------
        widths : 1-D array of widths.
        depths : 1-D array of depths.
        sigma_w, sigma_b : initialisation variances.
        X : (n, d) representative data.

        Returns
        -------
        result : dict with
            'phases'  : (len(depths), len(widths)) array of str labels
            'chi'     : same shape, float
            'ntk_var' : same shape, float
            'q_final' : same shape, float
        """
        nd, nw = len(depths), len(widths)
        phases = np.empty((nd, nw), dtype=object)
        chi_map = np.zeros((nd, nw))
        var_map = np.zeros((nd, nw))
        q_map = np.zeros((nd, nw))

        for i, d in enumerate(depths):
            for j, w in enumerate(widths):
                res = self.compute_phase_point(int(w), int(d), sigma_w, sigma_b, X)
                phases[i, j] = res["phase"]
                chi_map[i, j] = res["chi"]
                var_map[i, j] = res["ntk_var"]
                q_map[i, j] = res["q_final"]

        return {
            "phases": phases,
            "chi": chi_map,
            "ntk_var": var_map,
            "q_final": q_map,
        }

    # -----------------------------------------------------------------
    def find_boundary_curves(
        self,
        phase_map: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """Extract phase-boundary pixels from a 2-D phase label array.

        A pixel (i, j) is on a boundary if any of its 4-neighbours has
        a different phase label.

        Parameters
        ----------
        phase_map : (H, W) array of phase labels (strings).

        Returns
        -------
        boundary : list of (row, col) tuples on phase boundaries.
        """
        H, W = phase_map.shape
        boundary: List[Tuple[int, int]] = []

        for i in range(H):
            for j in range(W):
                label = phase_map[i, j]
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        if phase_map[ni, nj] != label:
                            boundary.append((i, j))
                            break
        return boundary

    # -----------------------------------------------------------------
    def critical_depth_at_width(
        self,
        width: int,
        sigma_w: float,
        sigma_b: float,
        X: np.ndarray,
    ) -> int:
        """Depth at which signal propagation degrades for a given width.

        Searches for the depth L* where the NTK variance ratio crosses 1
        (finite-width fluctuations overwhelm the signal) or where the
        correlation collapses, whichever comes first.

        Parameters
        ----------
        width : network width N.
        sigma_w, sigma_b : initialisation.
        X : (n, d) data.

        Returns
        -------
        L_star : int, critical depth.
        """
        act = self.config.activation
        alpha = self.config.skip_weight
        max_depth = 10000

        q_star = self._signal.fixed_point_qstar(sigma_w, sigma_b, act)
        chi = self._signal.lyapunov_exponent(q_star, sigma_w, act)

        # From NTK variance ratio:  L α⁴ / N ≈ 1  →  L* ≈ N / α⁴
        L_var = max(int(width / max(alpha ** 4, 1e-12)), 1)

        # From signal propagation: depth scale ξ
        if chi < 1.0:
            xi = -1.0 / math.log(max(chi, 1e-30))
            L_signal = max(int(3.0 * xi), 1)  # 3 correlation lengths
        else:
            L_signal = max_depth

        return min(L_var, L_signal, max_depth)

    # -----------------------------------------------------------------
    def optimal_depth_width_ratio(
        self,
        sigma_w: float,
        sigma_b: float,
        X: np.ndarray,
    ) -> float:
        """Optimal depth-to-width ratio L/N for the feature-learning regime.

        The feature-learning (rich / mean-field) regime exists when:
            1/N ≪ L α⁴ / N ≪ 1

        This gives the optimal ratio L*/N ≈ c / α⁴ where c is an
        O(1) constant depending on the activation and initialisation.

        Parameters
        ----------
        sigma_w, sigma_b : initialisation.
        X : (n, d) data.

        Returns
        -------
        ratio : float, optimal L/N.
        """
        alpha = self.config.skip_weight
        act = self.config.activation

        q_star = self._signal.fixed_point_qstar(sigma_w, sigma_b, act)
        chi = self._signal.lyapunov_exponent(q_star, sigma_w, act)

        # The feature-learning window is maximised when χ ≈ 1.
        # The ratio scales as  c / α⁴  where c depends on distance to EOC.
        if chi >= 1.0:
            c = 0.5  # chaotic side: smaller ratio to avoid blowup
        else:
            c = 1.0 / max(1.0 - chi, 0.01)  # wider window near EOC

        ratio = c / max(alpha ** 4, 1e-12)
        # Clamp to reasonable range
        ratio = min(ratio, 1000.0)
        return ratio

    # -----------------------------------------------------------------
    def phase_diagram_full(
        self,
        widths: np.ndarray,
        depths: np.ndarray,
        sigma_w: float,
        sigma_b: float,
        X: np.ndarray,
    ) -> Dict[str, Any]:
        """Full phase diagram with regime classification and boundaries.

        Combines sweep_width_depth and find_boundary_curves with
        additional diagnostic quantities.

        Parameters
        ----------
        widths : 1-D array of widths to scan.
        depths : 1-D array of depths to scan.
        sigma_w, sigma_b : initialisation variances.
        X : (n, d) representative input data.

        Returns
        -------
        diagram : dict with
            'phases'         : (len(depths), len(widths)) label array
            'chi'            : same shape, Lyapunov exponent
            'ntk_var'        : same shape, NTK variance ratio
            'q_final'        : same shape, final variance
            'boundaries'     : list of (row, col) boundary pixels
            'critical_depths': (len(widths),) critical depth per width
            'optimal_ratio'  : float, optimal L/N
            'widths'         : the width grid
            'depths'         : the depth grid
        """
        sweep = self.sweep_width_depth(widths, depths, sigma_w, sigma_b, X)
        boundaries = self.find_boundary_curves(sweep["phases"])

        critical_depths = np.array([
            self.critical_depth_at_width(int(w), sigma_w, sigma_b, X)
            for w in widths
        ])

        opt_ratio = self.optimal_depth_width_ratio(sigma_w, sigma_b, X)

        return {
            "phases": sweep["phases"],
            "chi": sweep["chi"],
            "ntk_var": sweep["ntk_var"],
            "q_final": sweep["q_final"],
            "boundaries": boundaries,
            "critical_depths": critical_depths,
            "optimal_ratio": opt_ratio,
            "widths": widths,
            "depths": depths,
        }
