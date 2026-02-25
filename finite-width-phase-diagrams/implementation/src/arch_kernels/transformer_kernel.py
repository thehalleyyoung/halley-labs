"""
Transformer NTK computation with finite-width corrections and phase boundary
analysis.

Implements Neural Tangent Kernel computations for full transformer architectures
including self-attention, multi-head attention, layer normalization, positional
encodings, finite-width perturbative corrections, and phase diagram construction
for lazy-to-rich transitions in width/depth/learning-rate space.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from scipy import linalg
from scipy.special import softmax as scipy_softmax


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TransformerNTKConfig:
    """Configuration for transformer NTK computation.

    Parameters
    ----------
    d_model : int
        Model / residual stream dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Hidden dimension of the feed-forward sub-layer.
    n_layers : int
        Number of transformer blocks.
    seq_len : int
        Sequence length.
    activation : str
        Activation function used in the FFN ('relu', 'gelu', 'swish').
    use_layernorm : bool
        Whether LayerNorm is applied (pre-norm style).
    positional_encoding_type : str
        Type of positional encoding ('sinusoidal', 'learned', 'rotary').
    """

    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_layers: int = 6
    seq_len: int = 128
    activation: str = "relu"
    use_layernorm: bool = True
    positional_encoding_type: str = "sinusoidal"

    @property
    def d_k(self) -> int:
        """Per-head key/query dimension."""
        return self.d_model // self.n_heads

    @property
    def d_v(self) -> int:
        """Per-head value dimension."""
        return self.d_model // self.n_heads


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def _activation_fn(x: np.ndarray, name: str) -> np.ndarray:
    """Element-wise activation."""
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "gelu":
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
    if name == "swish":
        return x / (1.0 + np.exp(-x))
    raise ValueError(f"Unknown activation: {name}")


def _activation_derivative(x: np.ndarray, name: str) -> np.ndarray:
    """Element-wise derivative of activation."""
    if name == "relu":
        return (x > 0).astype(x.dtype)
    if name == "gelu":
        t = np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3))
        dt = (1.0 - t ** 2) * np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x ** 2)
        return 0.5 * (1.0 + t) + 0.5 * x * dt
    if name == "swish":
        sig = 1.0 / (1.0 + np.exp(-x))
        return sig + x * sig * (1.0 - sig)
    raise ValueError(f"Unknown activation: {name}")


def _stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax."""
    shifted = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Self-Attention NTK
# ---------------------------------------------------------------------------

class SelfAttentionNTK:
    """NTK computation for a single self-attention head.

    Computes the Neural Tangent Kernel contribution from the attention
    mechanism by differentiating the attention output with respect to the
    weight matrices W_Q, W_K, W_V, W_O and combining the per-parameter
    Jacobians.

    Parameters
    ----------
    d_model : int
        Input / output dimension.
    n_heads : int
        Number of heads (used to infer per-head dimension).
    temperature : float
        Softmax temperature scaling factor.
    """

    def __init__(self, d_model: int, n_heads: int, temperature: float = 1.0):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.temperature = temperature
        self.scale = 1.0 / (np.sqrt(self.d_k) * temperature)

    # ---- kernel from Q, K, V projections ----

    def compute_qkv_kernel(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        W_Q: np.ndarray,
        W_K: np.ndarray,
        W_V: np.ndarray,
    ) -> np.ndarray:
        """NTK block arising from Q, K, V projection weights.

        Computes Θ_QKV(x, x') = Σ_{θ ∈ {W_Q, W_K, W_V}} ⟨∂f/∂θ(x), ∂f/∂θ(x')⟩.

        Parameters
        ----------
        X1 : ndarray, shape (n1, d_model)
        X2 : ndarray, shape (n2, d_model)
        W_Q : ndarray, shape (d_model, d_k)
        W_K : ndarray, shape (d_model, d_k)
        W_V : ndarray, shape (d_model, d_k)

        Returns
        -------
        K_qkv : ndarray, shape (n1, n2)
        """
        n1 = X1.shape[0]
        n2 = X2.shape[0]

        # Attention weights for both inputs
        Q1, K1, V1 = X1 @ W_Q, X1 @ W_K, X1 @ W_V
        Q2, K2, V2 = X2 @ W_Q, X2 @ W_K, X2 @ W_V

        A1 = _stable_softmax(Q1 @ K1.T * self.scale)
        A2 = _stable_softmax(Q2 @ K2.T * self.scale)

        # Gradient w.r.t. W_V is simplest: ∂(AV)/∂W_V = A ⊗ X
        # Inner product in parameter space gives kernel contribution
        # K_V(i, j) = tr(A1[i,:] ⊗ X1 . A2[j,:] ⊗ X2)
        #           = (A1 @ A2^T) ⊙ (X1 @ X2^T) summed over seq dims
        K_V = np.zeros((n1, n2))
        gram_X = X1 @ X2.T  # (n1, n2)
        # For single-token (non-sequence) inputs the formula simplifies
        for i in range(n1):
            for j in range(n2):
                K_V[i, j] = gram_X[i, j] * np.dot(A1[i], A2[j])

        # Gradient w.r.t. W_Q involves softmax Jacobian
        K_Q = self._param_kernel_Q(X1, X2, Q1, K1, V1, Q2, K2, V2, A1, A2)

        # Gradient w.r.t. W_K (analogous, transposed roles)
        K_K = self._param_kernel_K(X1, X2, Q1, K1, V1, Q2, K2, V2, A1, A2)

        return K_Q + K_K + K_V

    # ---- full attention output kernel ----

    def attention_output_kernel(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        W_Q: np.ndarray,
        W_K: np.ndarray,
        W_V: np.ndarray,
        W_O: np.ndarray,
    ) -> np.ndarray:
        """Full single-head attention NTK including output projection.

        The output y = W_O @ Attn(X; W_Q, W_K, W_V), so the NTK includes an
        additional term from ∂y/∂W_O plus W_O acting on the Q, K, V kernel.

        Parameters
        ----------
        X1, X2 : ndarray, shape (n, d_model)
        W_Q, W_K, W_V : ndarray, shape (d_model, d_k)
        W_O : ndarray, shape (d_k, d_model)

        Returns
        -------
        K_attn : ndarray, shape (n1, n2)
        """
        n1, n2 = X1.shape[0], X2.shape[0]

        # Intermediate attention outputs
        Q1, K1, V1 = X1 @ W_Q, X1 @ W_K, X1 @ W_V
        Q2, K2, V2 = X2 @ W_Q, X2 @ W_K, X2 @ W_V
        A1 = _stable_softmax(Q1 @ K1.T * self.scale)
        A2 = _stable_softmax(Q2 @ K2.T * self.scale)
        Z1 = A1 @ V1  # (n1, d_k)
        Z2 = A2 @ V2  # (n2, d_k)

        # Kernel from W_O: ∂(W_O z)/∂W_O = z ⊗ I ⟹ K_O = Z1 Z2^T
        K_O = Z1 @ Z2.T  # (n1, n2)

        # Kernel from Q, K, V propagated through W_O
        K_qkv = self.compute_qkv_kernel(X1, X2, W_Q, W_K, W_V)

        # W_O acts on both sides of the inner product: Θ → W_O Θ W_O^T
        # For simplicity (and correctness in the infinite-width limit),
        # we use the trace formulation which collapses to a scalar factor.
        wo_scale = np.trace(W_O.T @ W_O) / self.d_k

        return K_O + wo_scale * K_qkv

    # ---- softmax Jacobian ----

    def softmax_jacobian(self, attention_weights: np.ndarray) -> np.ndarray:
        """Jacobian of row-wise softmax.

        For a single row a = softmax(z), the Jacobian is
            J_{ij} = a_i (δ_{ij} - a_j).

        Parameters
        ----------
        attention_weights : ndarray, shape (seq_len, seq_len)
            Each row sums to 1.

        Returns
        -------
        J : ndarray, shape (seq_len, seq_len, seq_len)
            J[t, i, j] = ∂A[t,i] / ∂logit[t,j].
        """
        T = attention_weights.shape[0]
        S = attention_weights.shape[1]
        J = np.zeros((T, S, S))
        for t in range(T):
            a = attention_weights[t]  # (S,)
            J[t] = np.diag(a) - np.outer(a, a)
        return J

    # ---- gradients w.r.t. individual weight matrices ----

    def _gradient_wrt_queries(
        self,
        X: np.ndarray,
        W_Q: np.ndarray,
        W_K: np.ndarray,
        W_V: np.ndarray,
        A: np.ndarray,
    ) -> np.ndarray:
        """Gradient of attention output w.r.t. W_Q.

        y_t = Σ_s A_{ts} V_s,   A = softmax(Q K^T / sqrt(d))
        ∂y_t/∂W_Q = Σ_s (∂A_{ts}/∂Q) (∂Q/∂W_Q) V_s^T

        Parameters
        ----------
        X : ndarray, shape (n, d_model)
        W_Q, W_K, W_V : ndarray, shape (d_model, d_k)
        A : ndarray, shape (n, n)

        Returns
        -------
        grad : ndarray, shape (n, d_model, d_k)
            grad[t] is ∂y_t / ∂vec(W_Q) reshaped to (d_model, d_k).
        """
        n = X.shape[0]
        V = X @ W_V  # (n, d_k)
        K = X @ W_K  # (n, d_k)
        J = self.softmax_jacobian(A)  # (n, n, n)

        grad = np.zeros((n, self.d_model, self.d_k))
        for t in range(n):
            # ∂A[t,:]/∂logit[t,:] @ (V) gives derivative of attention output
            # w.r.t. logits, shape (n, d_k)
            dA_dlogit = J[t]  # (n, n)
            # logit = Q_t . K_s / sqrt(d), so ∂logit/∂W_Q = X_t^T K_s / sqrt(d)
            # Chain: Σ_s dA_{ts}/dlogit_{tu} * (K_u / sqrt(d)) * x_t
            # → outer(x_t, Σ_s J[t,:,s] * K[s] * scale) ... contracted with V
            dout_dlogit = dA_dlogit @ V  # (n, d_k)
            # dlogit_{tu}/dW_Q = scale * x_t ⊗ K_u
            # So ∂y_t/∂W_Q = scale * Σ_u dout_dlogit[u] ⊗ (x_t, K_u)
            # Summing over u: scale * x_t ⊗ (K^T @ dout_dlogit[·, :])
            effective_k = K.T @ dout_dlogit[:, :]  # (d_k, d_k) -- contract over seq
            # But we only need the vectorised form for kernel computation.
            # The gradient tensor is scale * np.outer(X[t], K^T @ dout_dlogit)
            # flattened to (d_model, d_k) via x_t (d_model,) and col (d_k,).
            col = self.scale * (dout_dlogit.T @ K).sum(axis=0)  # (d_k,)
            grad[t] = np.outer(X[t], col)
        return grad

    def _gradient_wrt_keys(
        self,
        X: np.ndarray,
        W_Q: np.ndarray,
        W_K: np.ndarray,
        W_V: np.ndarray,
        A: np.ndarray,
    ) -> np.ndarray:
        """Gradient of attention output w.r.t. W_K.

        Symmetric to the query gradient with Q and K roles swapped.

        Parameters
        ----------
        X : ndarray, shape (n, d_model)
        W_Q, W_K, W_V : ndarray, shape (d_model, d_k)
        A : ndarray, shape (n, n)

        Returns
        -------
        grad : ndarray, shape (n, d_model, d_k)
        """
        n = X.shape[0]
        V = X @ W_V
        Q = X @ W_Q
        J = self.softmax_jacobian(A)

        grad = np.zeros((n, self.d_model, self.d_k))
        for t in range(n):
            dA_dlogit = J[t]  # (n, n)
            dout_dlogit = dA_dlogit @ V  # (n, d_k)
            # logit_{ts} = scale * Q_t . K_s → ∂logit/∂W_K involves X_s
            # Σ_s ∂y_t/∂logit_{ts} * ∂logit_{ts}/∂W_K
            # = scale * Σ_s dout_dlogit[s,:] ⊗ (Q_t . ) — contracted with X_s
            col = self.scale * (dout_dlogit.T @ Q[t]).reshape(-1)  # (d_k,)
            # sum contributions from all source positions
            x_sum = np.zeros(self.d_model)
            for s in range(n):
                x_sum += dout_dlogit[s].sum() * X[s]
            grad[t] = np.outer(x_sum / n, col)
        return grad

    def _gradient_wrt_values(
        self,
        X: np.ndarray,
        W_Q: np.ndarray,
        W_K: np.ndarray,
        W_V: np.ndarray,
        A: np.ndarray,
    ) -> np.ndarray:
        """Gradient of attention output w.r.t. W_V.

        y_t = A[t,:] @ V = A[t,:] @ X @ W_V
        ∂y_t/∂W_V = (A[t,:] @ X)^T — straightforward linear dependence.

        Parameters
        ----------
        X : ndarray, shape (n, d_model)
        W_Q, W_K, W_V : ndarray, shape (d_model, d_k)
        A : ndarray, shape (n, n)

        Returns
        -------
        grad : ndarray, shape (n, d_model, d_k)
        """
        n = X.shape[0]
        grad = np.zeros((n, self.d_model, self.d_k))
        for t in range(n):
            # ∂y_t/∂W_V = X^T diag(A[t,:]) → sum_s A[t,s] x_s x_s^T applied
            context = A[t] @ X  # (d_model,)
            grad[t] = np.outer(context, np.ones(self.d_k))
        return grad

    # ---- single-head NTK ----

    def ntk_single_head(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        params_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute the NTK for a single attention head.

        Θ(x, x') = Σ_θ ⟨∂f/∂θ(x), ∂f/∂θ(x')⟩

        Parameters
        ----------
        X1 : ndarray, shape (n1, d_model)
        X2 : ndarray, shape (n2, d_model)
        params_dict : dict
            Keys 'W_Q', 'W_K', 'W_V', 'W_O' each mapping to ndarrays.

        Returns
        -------
        Theta : ndarray, shape (n1, n2)
        """
        W_Q = params_dict["W_Q"]
        W_K = params_dict["W_K"]
        W_V = params_dict["W_V"]
        W_O = params_dict["W_O"]

        return self.attention_output_kernel(X1, X2, W_Q, W_K, W_V, W_O)

    # ---- internal helpers for parameter-space kernels ----

    def _param_kernel_Q(
        self, X1, X2, Q1, K1, V1, Q2, K2, V2, A1, A2
    ) -> np.ndarray:
        """Kernel contribution from W_Q parameters.

        K_Q(i,j) = ⟨∂f/∂W_Q(x_i), ∂f/∂W_Q(x_j)⟩
        """
        n1, n2 = X1.shape[0], X2.shape[0]
        J1 = self.softmax_jacobian(A1)
        J2 = self.softmax_jacobian(A2)
        K_Q = np.zeros((n1, n2))

        for i in range(n1):
            dA1 = J1[i] @ V1  # (n1, d_k)
            eff1 = self.scale * (K1.T @ dA1)  # (d_k, d_k)
            g1 = np.outer(X1[i], eff1.ravel())  # flattened gradient

            for j in range(n2):
                dA2 = J2[j] @ V2  # (n2, d_k)
                eff2 = self.scale * (K2.T @ dA2)
                g2 = np.outer(X2[j], eff2.ravel())
                K_Q[i, j] = np.sum(g1 * g2)
        return K_Q

    def _param_kernel_K(
        self, X1, X2, Q1, K1, V1, Q2, K2, V2, A1, A2
    ) -> np.ndarray:
        """Kernel contribution from W_K parameters."""
        n1, n2 = X1.shape[0], X2.shape[0]
        J1 = self.softmax_jacobian(A1)
        J2 = self.softmax_jacobian(A2)
        K_K = np.zeros((n1, n2))

        for i in range(n1):
            dA1 = J1[i] @ V1
            eff1 = self.scale * (Q1.T @ dA1)
            g1 = np.outer(X1[i], eff1.ravel())

            for j in range(n2):
                dA2 = J2[j] @ V2
                eff2 = self.scale * (Q2.T @ dA2)
                g2 = np.outer(X2[j], eff2.ravel())
                K_K[i, j] = np.sum(g1 * g2)
        return K_K


# ---------------------------------------------------------------------------
# Multi-Head Attention NTK
# ---------------------------------------------------------------------------

class MultiHeadAttentionNTK:
    """NTK for multi-head attention with inter-head coupling.

    In the infinite-width limit the per-head kernels simply add.  At finite
    width the heads share the residual stream, inducing coupling terms of
    order O(1/d_model).

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    """

    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self._head_ntk = SelfAttentionNTK(d_model, n_heads)

    def compose_head_kernels(
        self, head_kernels: List[np.ndarray]
    ) -> np.ndarray:
        """Combine per-head NTK matrices.

        In the infinite-width (GP) limit the heads contribute independently
        and the total kernel is the sum.

        Parameters
        ----------
        head_kernels : list of ndarray, each shape (n, n)

        Returns
        -------
        K_mha : ndarray, shape (n, n)
        """
        K = np.zeros_like(head_kernels[0])
        for Kh in head_kernels:
            K += Kh
        return K / self.n_heads

    def cross_head_coupling(
        self,
        X: np.ndarray,
        params_list: List[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """Finite-width coupling between heads sharing the residual stream.

        At finite d_model the heads are not independent because the output
        projections W_O^{(h)} all map into the same d_model-dimensional
        residual stream.  The coupling kernel is

            C(x, x') = (1/d_model) Σ_{h≠h'} tr(W_O^h^T W_O^{h'})
                        · ⟨z^h(x), z^{h'}(x')⟩

        Parameters
        ----------
        X : ndarray, shape (n, d_model)
        params_list : list of dicts, one per head.

        Returns
        -------
        C : ndarray, shape (n, n)
        """
        n = X.shape[0]
        H = len(params_list)
        # Pre-compute per-head attention outputs
        Z = []
        for h in range(H):
            p = params_list[h]
            Q = X @ p["W_Q"]
            K = X @ p["W_K"]
            V = X @ p["W_V"]
            A = _stable_softmax(Q @ K.T * self._head_ntk.scale)
            Z.append(A @ V)  # (n, d_k)

        C = np.zeros((n, n))
        for h1 in range(H):
            for h2 in range(H):
                if h1 == h2:
                    continue
                wo_coupling = np.trace(
                    params_list[h1]["W_O"].T @ params_list[h2]["W_O"]
                )
                C += wo_coupling * (Z[h1] @ Z[h2].T)
        return C / self.d_model

    def compute_full_mha_ntk(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        all_params: List[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """Full multi-head attention NTK including cross-head coupling.

        Parameters
        ----------
        X1 : ndarray, shape (n1, d_model)
        X2 : ndarray, shape (n2, d_model)
        all_params : list of dicts with keys 'W_Q', 'W_K', 'W_V', 'W_O'.

        Returns
        -------
        Theta : ndarray, shape (n1, n2)
        """
        head_kernels = []
        for p in all_params:
            Kh = self._head_ntk.ntk_single_head(X1, X2, p)
            head_kernels.append(Kh)

        K_sum = self.compose_head_kernels(head_kernels)

        # Cross-head coupling (use mean of X1, X2 for the coupling estimate)
        X_combined = np.vstack([X1, X2])
        C_full = self.cross_head_coupling(X_combined, all_params)
        n1, n2 = X1.shape[0], X2.shape[0]
        C_block = C_full[:n1, n1:]

        return K_sum + C_block

    def head_diversity_metric(
        self, head_kernels: List[np.ndarray]
    ) -> float:
        """Measure diversity across head kernels.

        Computes the average pairwise cosine distance between normalised
        head kernel matrices (treated as vectors).

        Parameters
        ----------
        head_kernels : list of ndarray, each shape (n, n)

        Returns
        -------
        diversity : float in [0, 1]
            0 = all heads identical, 1 = maximally diverse.
        """
        H = len(head_kernels)
        if H < 2:
            return 0.0
        vecs = [K.ravel() for K in head_kernels]
        norms = [np.linalg.norm(v) + 1e-12 for v in vecs]
        vecs_normed = [v / n for v, n in zip(vecs, norms)]

        total = 0.0
        count = 0
        for i in range(H):
            for j in range(i + 1, H):
                cos_sim = np.dot(vecs_normed[i], vecs_normed[j])
                total += 1.0 - cos_sim
                count += 1
        return total / count

    def effective_rank_per_head(
        self, head_kernels: List[np.ndarray]
    ) -> List[float]:
        """Effective rank of each head kernel via Shannon entropy of singular
        values.

        erank(K) = exp(- Σ p_i log p_i),  p_i = σ_i / Σ σ_j

        Parameters
        ----------
        head_kernels : list of ndarray, each shape (n, n)

        Returns
        -------
        eranks : list of float
        """
        eranks = []
        for K in head_kernels:
            sv = np.linalg.svd(K, compute_uv=False)
            sv = sv[sv > 1e-12]
            p = sv / sv.sum()
            entropy = -np.sum(p * np.log(p))
            eranks.append(np.exp(entropy))
        return eranks


# ---------------------------------------------------------------------------
# Layer-Norm kernel effect
# ---------------------------------------------------------------------------

class LayerNormKernelEffect:
    """How LayerNorm modifies the NTK.

    LayerNorm is a non-linear map LN(x) = γ ⊙ (x - μ) / σ + β.  Its
    Jacobian contracts certain directions in activation space and therefore
    modifies the kernel.

    Parameters
    ----------
    d_model : int
        Feature dimension.
    eps : float
        LayerNorm epsilon for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps

    def layernorm_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Jacobian of LayerNorm for each sample.

        For x ∈ ℝ^d with μ = mean(x), σ² = var(x):
            J_{ij} = (1/σ) [δ_{ij} - 1/d - (x_i - μ)(x_j - μ)/(d σ²)]

        Parameters
        ----------
        X : ndarray, shape (n, d)

        Returns
        -------
        J : ndarray, shape (n, d, d)
        """
        n, d = X.shape
        mu = X.mean(axis=1, keepdims=True)  # (n, 1)
        var = X.var(axis=1, keepdims=True) + self.eps  # (n, 1)
        sigma = np.sqrt(var)
        X_centered = X - mu  # (n, d)

        J = np.zeros((n, d, d))
        for t in range(n):
            xc = X_centered[t]  # (d,)
            s = sigma[t, 0]
            v = var[t, 0]
            J[t] = (np.eye(d) - 1.0 / d - np.outer(xc, xc) / (d * v)) / s
        return J

    def modify_kernel_with_layernorm(
        self, K_pre: np.ndarray, X: np.ndarray
    ) -> np.ndarray:
        """Transform a kernel K through LayerNorm.

        If f_post = LN(f_pre), then
            Θ_post ≈ J(x) Θ_pre J(x')^T

        evaluated at the expected pre-activation.

        Parameters
        ----------
        K_pre : ndarray, shape (n, n)
            Pre-LayerNorm NTK.
        X : ndarray, shape (n, d_model)
            Input at which to evaluate the LayerNorm Jacobian.

        Returns
        -------
        K_post : ndarray, shape (n, n)
        """
        J = self.layernorm_jacobian(X)  # (n, d, d)
        n = X.shape[0]
        K_post = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # Θ_post(i,j) = tr(J_i Θ_{ij} J_j^T)
                # Using scalar kernel entry as a proxy for the diagonal
                # contribution in the (d x d) kernel tensor
                K_post[i, j] = K_pre[i, j] * np.trace(J[i] @ J[j].T) / self.d_model
        return K_post

    def layernorm_fixed_point(
        self, sigma_w: float, sigma_b: float
    ) -> Tuple[float, float]:
        """Fixed point of the kernel recursion under LayerNorm.

        LayerNorm normalises the pre-activation variance to 1 (up to
        learned γ), so the fixed-point diagonal kernel entry is

            q* = σ_w² + σ_b²

        and the off-diagonal correlation c* satisfies a self-consistent
        equation derived from the normalisation constraint.

        Parameters
        ----------
        sigma_w : float
            Weight variance scale.
        sigma_b : float
            Bias variance scale.

        Returns
        -------
        q_star : float
            Fixed-point diagonal kernel value.
        c_star : float
            Fixed-point off-diagonal correlation.
        """
        q_star = sigma_w ** 2 + sigma_b ** 2
        # Under LN the off-diagonal correlation converges to 0 when the
        # inputs are drawn i.i.d.  For correlated inputs it converges to a
        # value determined by the input correlation.
        # Approximate fixed point from mean-field analysis:
        c_star = sigma_b ** 2 / q_star if q_star > 0 else 0.0
        return q_star, c_star

    def gradient_norm_preservation(
        self, X: np.ndarray, depth: int
    ) -> np.ndarray:
        """Check whether LayerNorm preserves gradient norms across depth.

        Computes ‖J^L‖_F for L = 1..depth to see if gradient norms remain
        O(1) or grow/shrink.

        Parameters
        ----------
        X : ndarray, shape (n, d_model)
        depth : int

        Returns
        -------
        norms : ndarray, shape (depth,)
            Frobenius norm of cumulative Jacobian at each depth.
        """
        J = self.layernorm_jacobian(X)
        # Use the average Jacobian over samples
        J_avg = J.mean(axis=0)  # (d, d)
        norms = np.zeros(depth)
        J_cum = np.eye(self.d_model)
        for l in range(depth):
            J_cum = J_avg @ J_cum
            norms[l] = np.linalg.norm(J_cum, "fro")
        return norms


# ---------------------------------------------------------------------------
# Positional Encoding Kernel
# ---------------------------------------------------------------------------

class PositionalEncodingKernel:
    """Kernel contributions from positional encodings.

    Different positional encoding schemes induce different additive or
    multiplicative modifications to the NTK.

    Parameters
    ----------
    d_model : int
        Encoding dimension.
    max_len : int
        Maximum sequence length supported.
    encoding_type : str
        One of 'sinusoidal', 'learned', 'rotary'.
    """

    def __init__(
        self, d_model: int, max_len: int, encoding_type: str = "sinusoidal"
    ):
        self.d_model = d_model
        self.max_len = max_len
        self.encoding_type = encoding_type

        # Pre-compute sinusoidal table
        self._sin_table = self._build_sinusoidal_table()

    def _build_sinusoidal_table(self) -> np.ndarray:
        """Standard sinusoidal positional encoding table.

        PE(pos, 2i)   = sin(pos / 10000^{2i/d})
        PE(pos, 2i+1) = cos(pos / 10000^{2i/d})

        Returns
        -------
        table : ndarray, shape (max_len, d_model)
        """
        positions = np.arange(self.max_len)[:, None]
        dim_indices = np.arange(self.d_model)[None, :]
        angles = positions / np.power(
            10000.0, (2.0 * (dim_indices // 2)) / self.d_model
        )
        table = np.zeros((self.max_len, self.d_model))
        table[:, 0::2] = np.sin(angles[:, 0::2])
        table[:, 1::2] = np.cos(angles[:, 1::2])
        return table

    def sinusoidal_kernel(self, pos1: int, pos2: int) -> float:
        """Kernel entry from sinusoidal positional encoding.

        K_PE(p, p') = ⟨PE(p), PE(p')⟩ / d_model

        Parameters
        ----------
        pos1, pos2 : int
            Position indices.

        Returns
        -------
        k : float
        """
        pe1 = self._sin_table[pos1]
        pe2 = self._sin_table[pos2]
        return np.dot(pe1, pe2) / self.d_model

    def learned_pe_kernel(
        self, pos1: int, pos2: int, W_pe: np.ndarray
    ) -> float:
        """Kernel entry from learned positional embedding.

        W_pe is the embedding matrix, shape (max_len, d_model).  The NTK
        contribution is the inner product of the relevant rows *plus* the
        derivative contribution w.r.t. the embedding parameters.

        Parameters
        ----------
        pos1, pos2 : int
        W_pe : ndarray, shape (max_len, d_model)

        Returns
        -------
        k : float
        """
        pe1 = W_pe[pos1]
        pe2 = W_pe[pos2]
        # Inner product of embeddings (forward kernel)
        k_fwd = np.dot(pe1, pe2) / self.d_model
        # Derivative kernel: ∂f/∂W_pe[p] = indicator(pos==p) · downstream Jac
        # In the NTK the parameter-space inner product is non-zero only when
        # pos1 == pos2, in which case it contributes ‖downstream Jac‖².
        # At initialisation the downstream Jacobian is O(1), so we approximate:
        k_param = 1.0 if pos1 == pos2 else 0.0
        return k_fwd + k_param / self.d_model

    def rotary_kernel(
        self, pos1: int, pos2: int, freqs: np.ndarray
    ) -> float:
        """Kernel contribution from Rotary Position Embedding (RoPE).

        RoPE applies a position-dependent rotation R(p) to queries and keys.
        The inner product Q_p · K_{p'} becomes x^T R(p-p') x, so the kernel
        contribution is determined by the rotation angle difference.

        K_RoPE(p, p') = (1/d) Σ_i cos(freq_i · (p - p'))

        Parameters
        ----------
        pos1, pos2 : int
        freqs : ndarray, shape (d_model // 2,)
            Frequency vector θ_i = 1 / 10000^{2i/d}.

        Returns
        -------
        k : float
        """
        delta = pos1 - pos2
        return np.mean(np.cos(freqs * delta))

    def position_kernel_matrix(self, seq_len: int) -> np.ndarray:
        """Full position-position kernel matrix.

        Parameters
        ----------
        seq_len : int
            Actual sequence length (≤ max_len).

        Returns
        -------
        K_pos : ndarray, shape (seq_len, seq_len)
        """
        if self.encoding_type == "sinusoidal":
            PE = self._sin_table[:seq_len]  # (seq_len, d_model)
            return PE @ PE.T / self.d_model
        elif self.encoding_type == "rotary":
            freqs = 1.0 / np.power(
                10000.0,
                2.0 * np.arange(self.d_model // 2) / self.d_model,
            )
            K = np.zeros((seq_len, seq_len))
            for i in range(seq_len):
                for j in range(seq_len):
                    K[i, j] = self.rotary_kernel(i, j, freqs)
            return K
        else:
            # Learned: identity at init (embeddings are i.i.d.)
            return np.eye(seq_len)


# ---------------------------------------------------------------------------
# Finite-Width Corrections
# ---------------------------------------------------------------------------

class TransformerFiniteWidthCorrections:
    """Perturbative finite-width corrections to the transformer NTK.

    At finite width d_model the NTK deviates from its infinite-width limit.
    The leading corrections are O(1/d_model) and arise from the non-linear
    interactions (softmax, LayerNorm) that couple different feature
    dimensions.  We compute the correction via the *H-tensor* formalism of
    Dyer & Gur-Ari (2020) extended to attention layers.

    Parameters
    ----------
    config : TransformerNTKConfig
    """

    def __init__(self, config: TransformerNTKConfig):
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_k

    # ---- H-tensor for attention ----

    def attention_h_tensor(
        self,
        X: np.ndarray,
        W_Q: np.ndarray,
        W_K: np.ndarray,
        W_V: np.ndarray,
    ) -> np.ndarray:
        """Third-order H-tensor for a single attention layer.

        H_{ijk}(x) captures the leading finite-width fluctuation of the NTK
        and is defined as

            H_{ijk} = Σ_α ∂²f_i / (∂θ_α ∂θ_β) · ∂f_j / ∂θ_β · ∂f_k / ∂θ_α

        For the attention layer the second derivative comes from the softmax
        non-linearity: ∂²A/∂logit² is the third moment of the attention
        distribution.

        Parameters
        ----------
        X : ndarray, shape (n, d_model)
        W_Q, W_K, W_V : ndarray, shape (d_model, d_k)

        Returns
        -------
        H : ndarray, shape (n, n, n)
        """
        n = X.shape[0]
        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V
        scale = 1.0 / np.sqrt(self.d_k)
        A = _stable_softmax(Q @ K.T * scale)  # (n, n)

        # Second derivative of softmax: ∂²A_i/∂z_j∂z_k
        # = A_i(δ_{ij} - A_j)(δ_{ik} - A_k) - A_i·A_j(δ_{ik} - A_k)
        H = np.zeros((n, n, n))
        for t in range(n):
            a = A[t]  # (n,)
            v = V  # (n, d_k)
            for j in range(n):
                for k in range(n):
                    # Second derivative contribution from softmax Hessian
                    d2A = a * ((np.eye(n)[j] - a[j]) * (np.eye(n)[k] - a[k])
                              - a[j] * (np.eye(n)[k] - a[k]))
                    # Contract with value vectors
                    H[t, j, k] = scale ** 2 * np.dot(d2A, V @ V.T @ X[t])
        return H

    # ---- H-tensor for FFN ----

    def ffn_h_tensor(
        self, X: np.ndarray, W1: np.ndarray, W2: np.ndarray
    ) -> np.ndarray:
        """H-tensor for a feed-forward sub-layer.

        FFN(x) = W2 · σ(W1 · x).  The H-tensor arises from the second
        derivative of the activation function σ.

        Parameters
        ----------
        X : ndarray, shape (n, d_model)
        W1 : ndarray, shape (d_ff, d_model)
        W2 : ndarray, shape (d_model, d_ff)

        Returns
        -------
        H : ndarray, shape (n, n, n)
        """
        n = X.shape[0]
        pre_act = X @ W1.T  # (n, d_ff)
        sigma_prime = _activation_derivative(pre_act, self.config.activation)
        # Second derivative (approximate for ReLU as 0, use finite diff for others)
        eps_fd = 1e-5
        sigma_pp = (
            _activation_derivative(pre_act + eps_fd, self.config.activation)
            - _activation_derivative(pre_act - eps_fd, self.config.activation)
        ) / (2.0 * eps_fd)  # (n, d_ff)

        H = np.zeros((n, n, n))
        # H_{ijk} ∝ Σ_α x_i^T W1_α σ''(W1_α x_j) x_k^T W1_α · W2_α
        # Simplified contraction using the dominant term
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    # Dominant contribution from σ'' contraction
                    g_j = sigma_pp[j]  # (d_ff,)
                    a_i = pre_act[i] * sigma_prime[i]  # (d_ff,)
                    a_k = pre_act[k] * sigma_prime[k]  # (d_ff,)
                    w2_diag = np.sum(W2 ** 2, axis=0)  # (d_ff,)
                    H[i, j, k] = np.dot(a_i * g_j * a_k, w2_diag) / self.d_model
        return H

    # ---- perturbative corrections ----

    def first_order_correction(
        self,
        X: np.ndarray,
        params: Dict[str, np.ndarray],
        width: int,
    ) -> np.ndarray:
        """O(1/d_model) correction to the NTK.

        ΔΘ^{(1)} = (1/width) · Σ_{ijk} H_{ijk} H_{ijk}

        Parameters
        ----------
        X : ndarray, shape (n, d_model)
        params : dict with 'W_Q', 'W_K', 'W_V', 'W1', 'W2'.
        width : int

        Returns
        -------
        delta_K : ndarray, shape (n, n)
        """
        H_attn = self.attention_h_tensor(
            X, params["W_Q"], params["W_K"], params["W_V"]
        )
        H_ffn = self.ffn_h_tensor(X, params["W1"], params["W2"])
        H_total = H_attn + H_ffn

        n = X.shape[0]
        delta_K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                delta_K[i, j] = np.dot(H_total[i].ravel(), H_total[j].ravel())
        return delta_K / width

    def second_order_correction(
        self,
        X: np.ndarray,
        params: Dict[str, np.ndarray],
        width: int,
    ) -> np.ndarray:
        """O(1/d_model²) correction to the NTK.

        Includes the quartic fluctuation term:
            ΔΘ^{(2)} = (1/width²) · [H⊗H contraction - (ΔΘ^{(1)})²]

        Parameters
        ----------
        X : ndarray, shape (n, d_model)
        params : dict
        width : int

        Returns
        -------
        delta_K2 : ndarray, shape (n, n)
        """
        delta1 = self.first_order_correction(X, params, width)

        H_attn = self.attention_h_tensor(
            X, params["W_Q"], params["W_K"], params["W_V"]
        )
        H_ffn = self.ffn_h_tensor(X, params["W1"], params["W2"])
        H = H_attn + H_ffn

        n = X.shape[0]
        delta_K2 = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # Fourth-order cumulant contribution
                hi = H[i].ravel()
                hj = H[j].ravel()
                quartic = np.dot(hi ** 2, hj ** 2) - (np.dot(hi, hj)) ** 2
                delta_K2[i, j] = quartic
        return delta_K2 / (width ** 2) - delta1 ** 2 / width

    def layer_wise_corrections(
        self,
        X: np.ndarray,
        all_layer_params: List[Dict[str, np.ndarray]],
    ) -> List[np.ndarray]:
        """Compute finite-width corrections for each transformer layer.

        Parameters
        ----------
        X : ndarray, shape (n, d_model)
        all_layer_params : list of dicts, one per layer.

        Returns
        -------
        corrections : list of ndarray, each shape (n, n)
        """
        corrections = []
        for layer_params in all_layer_params:
            delta = self.first_order_correction(X, layer_params, self.d_model)
            corrections.append(delta)
        return corrections

    def residual_stream_correction(
        self, corrections_per_layer: List[np.ndarray]
    ) -> np.ndarray:
        """How skip connections modulate finite-width corrections.

        With residual connections x_{l+1} = x_l + f_l(x_l), the total
        correction is *not* the simple sum because the skip connection
        preserves the identity component.  The corrected kernel is

            ΔΘ_total = Σ_l ΔΘ_l + Σ_{l<l'} ΔΘ_l · ΔΘ_{l'} / d_model

        Parameters
        ----------
        corrections_per_layer : list of ndarray, each shape (n, n)

        Returns
        -------
        delta_total : ndarray, shape (n, n)
        """
        L = len(corrections_per_layer)
        delta_total = np.zeros_like(corrections_per_layer[0])
        for l in range(L):
            delta_total += corrections_per_layer[l]
        # Cross-layer residual coupling
        for l1 in range(L):
            for l2 in range(l1 + 1, L):
                delta_total += (
                    corrections_per_layer[l1] * corrections_per_layer[l2]
                    / self.d_model
                )
        return delta_total

    def total_transformer_correction(
        self,
        X: np.ndarray,
        model_params: List[Dict[str, np.ndarray]],
        d_model: int,
    ) -> np.ndarray:
        """Total finite-width correction for a full transformer.

        Combines layer-wise first-order corrections with residual stream
        coupling.

        Parameters
        ----------
        X : ndarray, shape (n, d_model)
        model_params : list of dicts, one per layer.
        d_model : int
            Width parameter for the perturbative expansion.

        Returns
        -------
        delta : ndarray, shape (n, n)
        """
        per_layer = []
        for lp in model_params:
            delta_l = self.first_order_correction(X, lp, d_model)
            per_layer.append(delta_l)
        return self.residual_stream_correction(per_layer)


# ---------------------------------------------------------------------------
# Phase Boundary Analysis
# ---------------------------------------------------------------------------

class TransformerPhaseBoundary:
    """Phase diagram construction for transformer NTK.

    Identifies the boundaries between the *lazy* regime (NTK stays close to
    initialisation), the *rich* / feature-learning regime (NTK evolves
    significantly), and potential *chaotic* regimes where training diverges.

    The order parameter is the relative change in the NTK after one gradient
    step:
        η = ‖Θ_1 - Θ_0‖ / ‖Θ_0‖

    Parameters
    ----------
    config : TransformerNTKConfig
    """

    def __init__(self, config: TransformerNTKConfig):
        self.config = config
        self._sa_ntk = SelfAttentionNTK(
            config.d_model, config.n_heads
        )
        self._corrections = TransformerFiniteWidthCorrections(config)

    # ---- helpers ----

    @staticmethod
    def _random_params(d_model, d_k, d_ff, activation="relu"):
        """Draw random parameters at NTK initialisation scale."""
        scale_qkv = 1.0 / np.sqrt(d_model)
        scale_o = 1.0 / np.sqrt(d_k)
        scale_w1 = 1.0 / np.sqrt(d_model)
        scale_w2 = 1.0 / np.sqrt(d_ff)
        return {
            "W_Q": np.random.randn(d_model, d_k) * scale_qkv,
            "W_K": np.random.randn(d_model, d_k) * scale_qkv,
            "W_V": np.random.randn(d_model, d_k) * scale_qkv,
            "W_O": np.random.randn(d_k, d_model) * scale_o,
            "W1": np.random.randn(d_ff, d_model) * scale_w1,
            "W2": np.random.randn(d_model, d_ff) * scale_w2,
        }

    def _build_ntk(self, X, params, width):
        """Compute NTK + leading finite-width correction."""
        K0 = self._sa_ntk.ntk_single_head(
            X, X,
            {"W_Q": params["W_Q"], "W_K": params["W_K"],
             "W_V": params["W_V"], "W_O": params["W_O"]},
        )
        delta = self._corrections.first_order_correction(X, params, width)
        return K0 + delta

    def _simulate_one_step(self, X, params, lr, width, Y=None):
        """Simulate one gradient-descent step and return the updated NTK.

        Uses a synthetic regression target Y (or random if not provided)
        and computes the NTK change after a single linearised update.
        """
        n = X.shape[0]
        if Y is None:
            Y = np.random.randn(n)
        K0 = self._build_ntk(X, params, width)
        K0_reg = K0 + 1e-6 * np.eye(n)
        alpha = linalg.solve(K0_reg, Y, assume_a="pos")

        # Linearised NTK update: Θ_1 ≈ Θ_0 - lr · dΘ/dt
        # dΘ/dt ∝ H · α, where H is the H-tensor
        H = self._corrections.attention_h_tensor(
            X, params["W_Q"], params["W_K"], params["W_V"]
        )
        dK = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dK[i, j] = np.einsum("k,k->", H[i, j, :], alpha)
        K1 = K0 - lr * dK
        return K0, K1

    # ---- public analysis methods ----

    def lazy_to_rich_transition(
        self,
        X: np.ndarray,
        widths: np.ndarray,
        learning_rates: np.ndarray,
    ) -> np.ndarray:
        """Find the lazy-to-rich phase boundary in (width, lr) space.

        For each (width, lr) pair, computes the order parameter η and
        returns the full grid.

        Parameters
        ----------
        X : ndarray, shape (n, d_model)
        widths : ndarray, shape (W,)
        learning_rates : ndarray, shape (L,)

        Returns
        -------
        eta_grid : ndarray, shape (W, L)
            Order parameter at each grid point.
        """
        d_k = self.config.d_k
        d_ff = self.config.d_ff
        W, L = len(widths), len(learning_rates)
        eta_grid = np.zeros((W, L))

        for wi, width in enumerate(widths):
            d_model_eff = int(width)
            d_k_eff = max(d_model_eff // self.config.n_heads, 1)
            d_ff_eff = 4 * d_model_eff
            # Project X to effective width
            if d_model_eff < X.shape[1]:
                X_eff = X[:, :d_model_eff]
            else:
                X_eff = np.pad(X, ((0, 0), (0, d_model_eff - X.shape[1])))
            params = self._random_params(d_model_eff, d_k_eff, d_ff_eff)
            for li, lr in enumerate(learning_rates):
                eta_grid[wi, li] = self._compute_order_parameter(
                    X_eff, params, d_model_eff, lr
                )
        return eta_grid

    def feature_learning_onset(
        self,
        X: np.ndarray,
        widths: np.ndarray,
        n_steps: int = 5,
    ) -> Dict[int, np.ndarray]:
        """Track NTK evolution over multiple steps to find onset of feature
        learning.

        Feature learning begins when ‖Θ_t - Θ_0‖ / ‖Θ_0‖ exceeds a
        threshold (conventionally 0.1).

        Parameters
        ----------
        X : ndarray, shape (n, d_model)
        widths : ndarray of int
        n_steps : int

        Returns
        -------
        evolution : dict mapping width → ndarray of shape (n_steps,)
            Relative NTK change at each step.
        """
        evolution = {}
        lr = 0.01
        for width in widths:
            d_model_eff = int(width)
            d_k_eff = max(d_model_eff // self.config.n_heads, 1)
            d_ff_eff = 4 * d_model_eff
            if d_model_eff < X.shape[1]:
                X_eff = X[:, :d_model_eff]
            else:
                X_eff = np.pad(X, ((0, 0), (0, d_model_eff - X.shape[1])))
            params = self._random_params(d_model_eff, d_k_eff, d_ff_eff)
            K0 = self._build_ntk(X_eff, params, d_model_eff)
            norm0 = np.linalg.norm(K0) + 1e-12

            deltas = np.zeros(n_steps)
            K_prev = K0.copy()
            for step in range(n_steps):
                _, K_next = self._simulate_one_step(
                    X_eff, params, lr, d_model_eff
                )
                deltas[step] = np.linalg.norm(K_next - K0) / norm0
                K_prev = K_next
            evolution[int(width)] = deltas
        return evolution

    def critical_depth_analysis(
        self,
        X: np.ndarray,
        params: Dict[str, np.ndarray],
        depths: np.ndarray,
    ) -> np.ndarray:
        """Find the critical depth at which the NTK degenerates.

        Stacks L identical transformer layers and computes the condition
        number of the resulting NTK.  A blow-up indicates the *edge of
        chaos*.

        Parameters
        ----------
        X : ndarray, shape (n, d_model)
        params : dict
        depths : ndarray of int

        Returns
        -------
        cond_numbers : ndarray, shape (len(depths),)
        """
        n = X.shape[0]
        cond_numbers = np.zeros(len(depths))
        K_base = self._sa_ntk.ntk_single_head(
            X, X,
            {"W_Q": params["W_Q"], "W_K": params["W_K"],
             "W_V": params["W_V"], "W_O": params["W_O"]},
        )
        for di, depth in enumerate(depths):
            # Residual composition: K_L = K_base + (L-1)·I + cross terms
            # In the mean-field limit with residual connections:
            K_L = depth * K_base + np.eye(n) * (depth - 1)
            # Finite-width correction grows with depth
            delta_l = self._corrections.first_order_correction(
                X, params, self.config.d_model
            )
            K_L += depth * (depth - 1) / 2.0 * delta_l / self.config.d_model
            sv = np.linalg.svd(K_L, compute_uv=False)
            cond_numbers[di] = sv[0] / (sv[-1] + 1e-12)
        return cond_numbers

    def attention_collapse_boundary(
        self,
        X: np.ndarray,
        params: Dict[str, np.ndarray],
        widths: np.ndarray,
    ) -> np.ndarray:
        """Find widths at which attention patterns collapse to uniform.

        Attention collapse is measured by the entropy of the attention
        distribution: H(A) → log(n) indicates uniform (collapsed) attention.

        Parameters
        ----------
        X : ndarray, shape (n, d_model)
        params : dict
        widths : ndarray

        Returns
        -------
        entropy_ratio : ndarray, shape (len(widths),)
            H(A) / log(n) for each width.  Values close to 1 indicate
            collapse.
        """
        n = X.shape[0]
        max_entropy = np.log(n + 1e-12)
        entropy_ratio = np.zeros(len(widths))

        for wi, width in enumerate(widths):
            d_model_eff = int(width)
            d_k_eff = max(d_model_eff // self.config.n_heads, 1)
            if d_model_eff < X.shape[1]:
                X_eff = X[:, :d_model_eff]
            else:
                X_eff = np.pad(X, ((0, 0), (0, d_model_eff - X.shape[1])))
            scale = 1.0 / np.sqrt(d_k_eff)
            W_Q = np.random.randn(d_model_eff, d_k_eff) / np.sqrt(d_model_eff)
            W_K = np.random.randn(d_model_eff, d_k_eff) / np.sqrt(d_model_eff)
            Q = X_eff @ W_Q
            K = X_eff @ W_K
            A = _stable_softmax(Q @ K.T * scale)
            # Average entropy over rows
            row_entropy = -np.sum(A * np.log(A + 1e-12), axis=1)
            entropy_ratio[wi] = np.mean(row_entropy) / max_entropy
        return entropy_ratio

    def phase_diagram_width_depth(
        self,
        X: np.ndarray,
        width_range: np.ndarray,
        depth_range: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Full (width, depth) phase diagram.

        Returns a grid of order parameters and regime labels.

        Parameters
        ----------
        X : ndarray, shape (n, d_model)
        width_range : ndarray of int, shape (W,)
        depth_range : ndarray of int, shape (D,)

        Returns
        -------
        order_params : ndarray, shape (W, D)
        regimes : ndarray of str, shape (W, D)
        """
        W, D = len(width_range), len(depth_range)
        order_params = np.zeros((W, D))
        regimes = np.empty((W, D), dtype=object)
        lr = 0.01

        for wi, width in enumerate(width_range):
            d_model_eff = int(width)
            d_k_eff = max(d_model_eff // self.config.n_heads, 1)
            d_ff_eff = 4 * d_model_eff
            if d_model_eff < X.shape[1]:
                X_eff = X[:, :d_model_eff]
            else:
                X_eff = np.pad(X, ((0, 0), (0, d_model_eff - X.shape[1])))
            params = self._random_params(d_model_eff, d_k_eff, d_ff_eff)

            for di, depth in enumerate(depth_range):
                depth = int(depth)
                # Scale corrections with depth
                eta = self._compute_order_parameter(
                    X_eff, params, d_model_eff, lr * depth
                )
                order_params[wi, di] = eta
                regimes[wi, di] = self._classify_regime(eta)
        return order_params, regimes

    def phase_diagram_width_lr(
        self,
        X: np.ndarray,
        width_range: np.ndarray,
        lr_range: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Full (width, learning_rate) phase diagram.

        Parameters
        ----------
        X : ndarray, shape (n, d_model)
        width_range : ndarray, shape (W,)
        lr_range : ndarray, shape (L,)

        Returns
        -------
        order_params : ndarray, shape (W, L)
        regimes : ndarray of str, shape (W, L)
        """
        eta_grid = self.lazy_to_rich_transition(X, width_range, lr_range)
        W, L = eta_grid.shape
        regimes = np.empty((W, L), dtype=object)
        for wi in range(W):
            for li in range(L):
                regimes[wi, li] = self._classify_regime(eta_grid[wi, li])
        return eta_grid, regimes

    def rank_collapse_transition(
        self,
        X: np.ndarray,
        params: Dict[str, np.ndarray],
        widths: np.ndarray,
    ) -> np.ndarray:
        """Find the width at which the NTK effective rank collapses.

        Effective rank is defined via the Shannon entropy of the normalised
        singular values.

        Parameters
        ----------
        X : ndarray, shape (n, d_model)
        params : dict
        widths : ndarray

        Returns
        -------
        eranks : ndarray, shape (len(widths),)
        """
        n = X.shape[0]
        eranks = np.zeros(len(widths))

        for wi, width in enumerate(widths):
            d_model_eff = int(width)
            d_k_eff = max(d_model_eff // self.config.n_heads, 1)
            d_ff_eff = 4 * d_model_eff
            if d_model_eff < X.shape[1]:
                X_eff = X[:, :d_model_eff]
            else:
                X_eff = np.pad(X, ((0, 0), (0, d_model_eff - X.shape[1])))
            p = self._random_params(d_model_eff, d_k_eff, d_ff_eff)
            K = self._build_ntk(X_eff, p, d_model_eff)
            sv = np.linalg.svd(K, compute_uv=False)
            sv = sv[sv > 1e-12]
            if len(sv) == 0:
                eranks[wi] = 0.0
                continue
            p_sv = sv / sv.sum()
            entropy = -np.sum(p_sv * np.log(p_sv))
            eranks[wi] = np.exp(entropy)
        return eranks

    # ---- internal order-parameter computation ----

    def _compute_order_parameter(
        self,
        X: np.ndarray,
        params: Dict[str, np.ndarray],
        width: int,
        lr: float,
    ) -> float:
        """Order parameter η = ‖Θ_1 - Θ_0‖ / ‖Θ_0‖.

        Parameters
        ----------
        X : ndarray, shape (n, d_model)
        params : dict
        width : int
        lr : float

        Returns
        -------
        eta : float
        """
        K0, K1 = self._simulate_one_step(X, params, lr, width)
        norm0 = np.linalg.norm(K0)
        if norm0 < 1e-12:
            return 0.0
        return np.linalg.norm(K1 - K0) / norm0

    def _classify_regime(self, order_param: float) -> str:
        """Classify a point in parameter space into a dynamical regime.

        Parameters
        ----------
        order_param : float
            Relative NTK change η.

        Returns
        -------
        regime : str
            One of 'lazy', 'rich', 'chaotic'.
        """
        if order_param < 0.1:
            return "lazy"
        elif order_param < 1.0:
            return "rich"
        else:
            return "chaotic"
