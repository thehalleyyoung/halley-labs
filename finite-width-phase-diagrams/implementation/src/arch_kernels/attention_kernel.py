"""
Attention mechanism kernels for Neural Tangent Kernel computation.

Implements NTK-based kernel computations for softmax attention, multi-head
attention, attention pattern analysis, self-attention recursion through
transformer layers, and positional encoding effects.
"""

import numpy as np
from scipy import linalg
from scipy.special import softmax as scipy_softmax
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist


class SoftmaxAttentionKernel:
    """Softmax attention NTK computation."""

    def __init__(self, d_model, temperature=1.0):
        self.d_model = d_model
        self.temperature = temperature
        self.scale = temperature / np.sqrt(d_model)

    def query_key_kernel(self, Q1, K1, Q2, K2):
        """K_QK(x,x') from query-key interaction.

        Computes the kernel arising from the bilinear query-key interaction:
        K_QK = (Q1 @ K1^T) ⊙ (Q2 @ K2^T) averaged over feature dimension.

        Parameters
        ----------
        Q1, K1 : ndarray, shape (seq_len1, d_k) and (seq_len1, d_k)
        Q2, K2 : ndarray, shape (seq_len2, d_k) and (seq_len2, d_k)

        Returns
        -------
        K_QK : ndarray, shape (seq_len1, seq_len2)
        """
        logits1 = Q1 @ K1.T * self.scale
        logits2 = Q2 @ K2.T * self.scale

        n1, s1 = logits1.shape
        n2, s2 = logits2.shape
        K_QK = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K_QK[i, j] = np.dot(logits1[i], logits2[j]) / s1
        return K_QK

    def attention_weights(self, Q, K, mask=None):
        """A = softmax(QK^T / sqrt(d)).

        Parameters
        ----------
        Q : ndarray, shape (seq_len, d_k)
        K : ndarray, shape (seq_len, d_k)
        mask : ndarray or None, shape (seq_len, seq_len)
            If provided, positions with False/0 are masked to -inf before softmax.

        Returns
        -------
        A : ndarray, shape (seq_len, seq_len)
        """
        d_k = Q.shape[-1]
        logits = Q @ K.T / (np.sqrt(d_k) * self.temperature)
        if mask is not None:
            logits = np.where(mask, logits, -1e9)
        # Numerically stable softmax
        logits_max = logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        A = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        return A

    def attention_kernel_single_head(self, X1, X2, W_Q, W_K, W_V):
        """Single-head attention kernel.

        Computes the NTK for a single attention head by combining gradients
        w.r.t. W_Q, W_K, and W_V.

        Parameters
        ----------
        X1, X2 : ndarray, shape (seq_len, d_model)
        W_Q, W_K, W_V : ndarray, shape (d_model, d_k)

        Returns
        -------
        K_head : ndarray, shape (seq_len, seq_len)
        """
        Q1, K1, V1 = X1 @ W_Q, X1 @ W_K, X1 @ W_V
        Q2, K2, V2 = X2 @ W_Q, X2 @ W_K, X2 @ W_V

        A1 = self.attention_weights(Q1, K1)
        A2 = self.attention_weights(Q2, K2)

        # Value contribution: kernel from V path
        attended1 = A1 @ V1
        attended2 = A2 @ V2
        K_value = attended1 @ attended2.T / V1.shape[1]

        # Gradient contribution from attention weights (Q, K paths)
        J1 = self.softmax_jacobian(A1)
        J2 = self.softmax_jacobian(A2)

        seq_len = X1.shape[0]
        d_k = W_Q.shape[1]
        K_grad = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                # Gradient through Q: ∂A/∂Q * V
                dA_dQ1 = J1[i] @ K1 / (np.sqrt(d_k) * self.temperature)
                dA_dQ2 = J2[j] @ K2 / (np.sqrt(d_k) * self.temperature)
                grad_Q1 = np.outer(dA_dQ1 @ V1, X1[i]).ravel()
                grad_Q2 = np.outer(dA_dQ2 @ V2, X2[j]).ravel()
                K_grad[i, j] += np.dot(grad_Q1, grad_Q2) / (d_k * self.d_model)

        return K_value + K_grad

    def softmax_jacobian(self, attention_weights):
        """∂softmax/∂logits Jacobian.

        For each row of attention_weights, computes the Jacobian matrix
        J_ij = a_i (δ_ij - a_j).

        Parameters
        ----------
        attention_weights : ndarray, shape (seq_len, seq_len)

        Returns
        -------
        J : ndarray, shape (seq_len, seq_len, seq_len)
            J[i] is the Jacobian for the i-th row.
        """
        seq_len = attention_weights.shape[0]
        J = np.zeros((seq_len, seq_len, seq_len))
        for i in range(seq_len):
            a = attention_weights[i]
            J[i] = np.diag(a) - np.outer(a, a)
        return J

    def attention_gradient_kernel(self, X1, X2, params):
        """NTK contribution from attention gradients.

        Parameters
        ----------
        X1, X2 : ndarray, shape (seq_len, d_model)
        params : dict with keys 'W_Q', 'W_K', 'W_V'

        Returns
        -------
        K_grad : ndarray, shape (seq_len, seq_len)
        """
        W_Q, W_K, W_V = params['W_Q'], params['W_K'], params['W_V']
        Q1, K1, V1 = X1 @ W_Q, X1 @ W_K, X1 @ W_V
        Q2, K2, V2 = X2 @ W_Q, X2 @ W_K, X2 @ W_V

        A1 = self.attention_weights(Q1, K1)
        A2 = self.attention_weights(Q2, K2)
        J1 = self.softmax_jacobian(A1)
        J2 = self.softmax_jacobian(A2)

        seq_len = X1.shape[0]
        d_k = W_Q.shape[1]
        scale = 1.0 / (np.sqrt(d_k) * self.temperature)
        K_grad = np.zeros((seq_len, seq_len))

        for i in range(seq_len):
            for j in range(seq_len):
                # Gradient w.r.t. W_Q
                dout_dlogits1 = J1[i] @ np.outer(np.ones(seq_len), V1[i])
                dout_dlogits2 = J2[j] @ np.outer(np.ones(seq_len), V2[j])
                dlogits_dWQ1 = scale * K1
                dlogits_dWQ2 = scale * K2
                g1 = (dout_dlogits1.T @ dlogits_dWQ1).ravel()
                g2 = (dout_dlogits2.T @ dlogits_dWQ2).ravel()
                K_grad[i, j] += np.dot(g1, g2) / (d_k * self.d_model)

                # Gradient w.r.t. W_K
                dlogits_dWK1 = scale * Q1
                dlogits_dWK2 = scale * Q2
                g1k = (dout_dlogits1.T @ dlogits_dWK1).ravel()
                g2k = (dout_dlogits2.T @ dlogits_dWK2).ravel()
                K_grad[i, j] += np.dot(g1k, g2k) / (d_k * self.d_model)

        return K_grad

    def value_kernel(self, V1, V2):
        """Kernel from value computation: K_V = V1 @ V2^T / d_v.

        Parameters
        ----------
        V1, V2 : ndarray, shape (seq_len, d_v)

        Returns
        -------
        K_V : ndarray, shape (seq_len1, seq_len2)
        """
        d_v = V1.shape[1]
        return V1 @ V2.T / d_v

    def output_projection_kernel(self, attended1, attended2, W_O):
        """Output projection kernel.

        Parameters
        ----------
        attended1, attended2 : ndarray, shape (seq_len, d_v)
        W_O : ndarray, shape (d_v, d_model)

        Returns
        -------
        K_O : ndarray, shape (seq_len1, seq_len2)
        """
        out1 = attended1 @ W_O
        out2 = attended2 @ W_O
        return out1 @ out2.T / W_O.shape[1]

    def infinite_width_attention_kernel(self, X1, X2, d_model_inf):
        """d_model → ∞ limit of the attention kernel.

        In the infinite-width limit, the kernel converges to a deterministic
        form governed by the covariance structure Σ = X^T X / d.

        Parameters
        ----------
        X1, X2 : ndarray, shape (seq_len, d_model)
        d_model_inf : int
            Width to use for normalization (conceptually → ∞).

        Returns
        -------
        K_inf : ndarray, shape (seq_len1, seq_len2)
        """
        Sigma11 = X1 @ X1.T / d_model_inf
        Sigma22 = X2 @ X2.T / d_model_inf
        Sigma12 = X1 @ X2.T / d_model_inf

        diag1 = np.sqrt(np.diag(Sigma11))
        diag2 = np.sqrt(np.diag(Sigma22))
        norm = np.outer(diag1, diag2)
        norm = np.where(norm > 1e-12, norm, 1e-12)
        cos_sim = Sigma12 / norm

        # In the infinite-width GP limit, the attention kernel reduces to
        # a function of the normalized Gram matrix
        exp_sim = np.exp(cos_sim / self.temperature)
        row_sums = exp_sim.sum(axis=-1, keepdims=True)
        row_sums = np.where(row_sums > 1e-12, row_sums, 1e-12)
        K_inf = exp_sim / row_sums

        # Combine with value kernel (identity in the limit)
        K_inf = K_inf @ Sigma12
        return K_inf

    def finite_width_correction(self, X1, X2, d_model):
        """O(1/d) correction term to infinite-width kernel.

        Parameters
        ----------
        X1, X2 : ndarray, shape (seq_len, d_model)
        d_model : int

        Returns
        -------
        correction : ndarray, shape (seq_len1, seq_len2)
        """
        K_inf = self.infinite_width_attention_kernel(X1, X2, d_model)

        Sigma12 = X1 @ X2.T / d_model
        # Fourth moment correction: involves kurtosis of the feature distribution
        X1_sq = (X1 ** 2).sum(axis=1)
        X2_sq = (X2 ** 2).sum(axis=1)
        fourth_moment = np.outer(X1_sq, X2_sq) / (d_model ** 2)
        second_moment_sq = Sigma12 ** 2

        # Correction is proportional to excess kurtosis / d
        excess = fourth_moment - second_moment_sq
        correction = excess / d_model * K_inf
        return correction

    def temperature_scaling_effect(self, kernel, temperatures):
        """Effect of 1/√d scaling temperature on kernel.

        Parameters
        ----------
        kernel : ndarray, shape (n, n)
            Base kernel matrix.
        temperatures : array-like
            Temperature values to evaluate.

        Returns
        -------
        scaled_kernels : list of ndarray
            Kernel at each temperature.
        """
        results = []
        for tau in temperatures:
            # Temperature rescales off-diagonal entries
            diag = np.diag(np.diag(kernel))
            off_diag = kernel - diag
            scaled = diag + off_diag * (self.temperature / tau)
            results.append(scaled)
        return results


class MultiHeadAttentionKernel:
    """Multi-head attention kernel composition."""

    def __init__(self, d_model, n_heads, d_k=None, d_v=None):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else d_model // n_heads
        self.d_v = d_v if d_v is not None else d_model // n_heads

    def per_head_kernel(self, X1, X2, head_idx, params):
        """Kernel for individual head.

        Parameters
        ----------
        X1, X2 : ndarray, shape (seq_len, d_model)
        head_idx : int
        params : dict with 'W_Q', 'W_K', 'W_V' each shape (n_heads, d_model, d_k/d_v)

        Returns
        -------
        K_h : ndarray, shape (seq_len1, seq_len2)
        """
        W_Q_h = params['W_Q'][head_idx]
        W_K_h = params['W_K'][head_idx]
        W_V_h = params['W_V'][head_idx]

        kernel = SoftmaxAttentionKernel(self.d_model)
        return kernel.attention_kernel_single_head(X1, X2, W_Q_h, W_K_h, W_V_h)

    def combined_kernel(self, X1, X2, params):
        """Sum of per-head kernels: K_MHA = (1/H) Σ_h K_h.

        Parameters
        ----------
        X1, X2 : ndarray, shape (seq_len, d_model)
        params : dict

        Returns
        -------
        K_combined : ndarray, shape (seq_len1, seq_len2)
        """
        seq_len1 = X1.shape[0]
        seq_len2 = X2.shape[0]
        K_combined = np.zeros((seq_len1, seq_len2))
        for h in range(self.n_heads):
            K_combined += self.per_head_kernel(X1, X2, h, params)
        return K_combined / self.n_heads

    def head_interaction_kernel(self, X1, X2, params):
        """Cross-head interaction terms in the NTK.

        Captures correlations between heads through the output projection.

        Parameters
        ----------
        X1, X2 : ndarray, shape (seq_len, d_model)
        params : dict with 'W_O' shape (d_model, n_heads * d_v)

        Returns
        -------
        K_inter : ndarray, shape (seq_len1, seq_len2)
        """
        head_outputs1 = []
        head_outputs2 = []
        for h in range(self.n_heads):
            W_Q_h = params['W_Q'][h]
            W_K_h = params['W_K'][h]
            W_V_h = params['W_V'][h]
            kernel = SoftmaxAttentionKernel(self.d_model)
            Q1, K1, V1 = X1 @ W_Q_h, X1 @ W_K_h, X1 @ W_V_h
            Q2, K2, V2 = X2 @ W_Q_h, X2 @ W_K_h, X2 @ W_V_h
            A1 = kernel.attention_weights(Q1, K1)
            A2 = kernel.attention_weights(Q2, K2)
            head_outputs1.append(A1 @ V1)
            head_outputs2.append(A2 @ V2)

        # Concatenated outputs
        concat1 = np.concatenate(head_outputs1, axis=-1)
        concat2 = np.concatenate(head_outputs2, axis=-1)

        W_O = params['W_O']
        out1 = concat1 @ W_O
        out2 = concat2 @ W_O

        # Interaction = full kernel minus sum of per-head kernels
        K_full = out1 @ out2.T / self.d_model
        K_sum = np.zeros_like(K_full)
        for h in range(self.n_heads):
            o1 = head_outputs1[h] @ W_O[h * self.d_v:(h + 1) * self.d_v, :]
            o2 = head_outputs2[h] @ W_O[h * self.d_v:(h + 1) * self.d_v, :]
            K_sum += o1 @ o2.T / self.d_model
        return K_full - K_sum

    def head_diversity_measure(self, per_head_kernels):
        """Measure diversity across head kernels using average pairwise distance.

        Parameters
        ----------
        per_head_kernels : list of ndarray, each shape (n, n)

        Returns
        -------
        diversity : float
            Average Frobenius distance between pairs of head kernels.
        """
        n_heads = len(per_head_kernels)
        if n_heads < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                diff = per_head_kernels[i] - per_head_kernels[j]
                total += np.sqrt(np.sum(diff ** 2))
                count += 1
        return total / count

    def head_redundancy(self, per_head_kernels):
        """Redundancy between heads measured by average cosine similarity.

        Parameters
        ----------
        per_head_kernels : list of ndarray

        Returns
        -------
        redundancy : float in [0, 1]
            1 means all heads are identical, 0 means fully orthogonal.
        """
        n_heads = len(per_head_kernels)
        if n_heads < 2:
            return 0.0
        flat = np.array([K.ravel() for K in per_head_kernels])
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1e-12)
        normalized = flat / norms
        sim_matrix = normalized @ normalized.T

        # Average off-diagonal cosine similarity
        mask = ~np.eye(n_heads, dtype=bool)
        return float(np.mean(sim_matrix[mask]))

    def effective_number_of_heads(self, per_head_kernels):
        """Effective rank of head ensemble via participation ratio.

        Parameters
        ----------
        per_head_kernels : list of ndarray

        Returns
        -------
        n_eff : float
            Effective number of distinct heads.
        """
        flat = np.array([K.ravel() for K in per_head_kernels])
        # SVD of the (n_heads x n^2) matrix
        _, s, _ = np.linalg.svd(flat, full_matrices=False)
        s = s[s > 1e-12]
        if len(s) == 0:
            return 0.0
        # Participation ratio: (Σ s_i)^2 / Σ s_i^2
        p = s ** 2
        return float((p.sum()) ** 2 / (p ** 2).sum()) if (p ** 2).sum() > 0 else 0.0

    def head_pruning_effect(self, per_head_kernels, n_prune):
        """Effect of pruning heads on the combined kernel.

        Prunes heads with smallest Frobenius norm.

        Parameters
        ----------
        per_head_kernels : list of ndarray
        n_prune : int
            Number of heads to remove.

        Returns
        -------
        pruned_kernel : ndarray
            Combined kernel after removing n_prune heads.
        pruned_indices : list of int
            Indices of pruned heads.
        """
        n_heads = len(per_head_kernels)
        n_keep = max(1, n_heads - n_prune)
        norms = [np.linalg.norm(K, 'fro') for K in per_head_kernels]
        sorted_idx = np.argsort(norms)
        pruned_indices = sorted_idx[:n_prune].tolist()
        kept_indices = sorted_idx[n_prune:]

        pruned_kernel = np.zeros_like(per_head_kernels[0])
        for idx in kept_indices:
            pruned_kernel += per_head_kernels[idx]
        pruned_kernel /= n_keep
        return pruned_kernel, pruned_indices

    def infinite_heads_limit(self, X1, X2, params, n_heads_range):
        """n_heads → ∞ limit: kernel convergence as head count increases.

        Simulates the kernel for different numbers of heads by subsampling
        and averaging, then extrapolates.

        Parameters
        ----------
        X1, X2 : ndarray, shape (seq_len, d_model)
        params : dict
        n_heads_range : array-like of int

        Returns
        -------
        kernels : dict mapping n_heads -> kernel matrix
        extrapolated : ndarray
            Estimated infinite-heads limit.
        """
        kernels = {}
        all_head_kernels = []
        max_heads = max(n_heads_range)

        # Compute kernels for max_heads with random projections
        rng = np.random.RandomState(42)
        for h in range(max_heads):
            W_Q_h = rng.randn(self.d_model, self.d_k) / np.sqrt(self.d_model)
            W_K_h = rng.randn(self.d_model, self.d_k) / np.sqrt(self.d_model)
            W_V_h = rng.randn(self.d_model, self.d_v) / np.sqrt(self.d_model)
            sk = SoftmaxAttentionKernel(self.d_model)
            K_h = sk.attention_kernel_single_head(X1, X2, W_Q_h, W_K_h, W_V_h)
            all_head_kernels.append(K_h)

        for nh in n_heads_range:
            avg = np.mean(all_head_kernels[:nh], axis=0)
            kernels[nh] = avg

        # Richardson extrapolation: K_∞ ≈ K_n + (K_n - K_{n/2}) * n / (n - n/2)
        sorted_nh = sorted(n_heads_range)
        if len(sorted_nh) >= 2:
            K_n = kernels[sorted_nh[-1]]
            K_n2 = kernels[sorted_nh[-2]]
            extrapolated = 2 * K_n - K_n2
        else:
            extrapolated = kernels[sorted_nh[-1]]

        return kernels, extrapolated


class AttentionPatternAnalyzer:
    """Analyze attention patterns in kernel framework."""

    def __init__(self, seq_length, n_heads):
        self.seq_length = seq_length
        self.n_heads = n_heads

    def attention_entropy(self, attention_weights):
        """H = -Σ a_ij log a_ij per head.

        Parameters
        ----------
        attention_weights : ndarray, shape (n_heads, seq_len, seq_len)

        Returns
        -------
        entropy : ndarray, shape (n_heads, seq_len)
            Entropy per query position per head.
        """
        a = np.clip(attention_weights, 1e-12, 1.0)
        return -np.sum(a * np.log(a), axis=-1)

    def attention_sparsity(self, attention_weights, threshold=0.1):
        """Fraction of attention weights above threshold.

        Parameters
        ----------
        attention_weights : ndarray, shape (n_heads, seq_len, seq_len)
        threshold : float

        Returns
        -------
        sparsity : ndarray, shape (n_heads,)
            Fraction of entries above threshold per head.
        """
        above = (attention_weights > threshold).astype(float)
        total = attention_weights.shape[-1] * attention_weights.shape[-2]
        return above.reshape(attention_weights.shape[0], -1).sum(axis=-1) / total

    def attention_distance(self, attention_weights, positions):
        """Average attention distance weighted by attention weights.

        Parameters
        ----------
        attention_weights : ndarray, shape (n_heads, seq_len, seq_len)
        positions : ndarray, shape (seq_len,)
            Position indices.

        Returns
        -------
        avg_distance : ndarray, shape (n_heads,)
        """
        pos_diff = np.abs(positions[:, None] - positions[None, :])
        weighted = attention_weights * pos_diff[None, :, :]
        return weighted.sum(axis=(-2, -1)) / attention_weights.sum(axis=(-2, -1))

    def attention_pattern_kernel(self, attn_pattern1, attn_pattern2):
        """Kernel between attention patterns using normalized Frobenius inner product.

        Parameters
        ----------
        attn_pattern1, attn_pattern2 : ndarray, shape (seq_len, seq_len)

        Returns
        -------
        k : float
        """
        norm1 = np.linalg.norm(attn_pattern1, 'fro')
        norm2 = np.linalg.norm(attn_pattern2, 'fro')
        if norm1 < 1e-12 or norm2 < 1e-12:
            return 0.0
        return float(np.sum(attn_pattern1 * attn_pattern2) / (norm1 * norm2))

    def pattern_clustering(self, attention_patterns, n_clusters=5):
        """Cluster attention patterns using k-means.

        Parameters
        ----------
        attention_patterns : ndarray, shape (n_patterns, seq_len, seq_len)
        n_clusters : int

        Returns
        -------
        labels : ndarray, shape (n_patterns,)
        centroids : ndarray, shape (n_clusters, seq_len * seq_len)
        """
        n_patterns = attention_patterns.shape[0]
        flat = attention_patterns.reshape(n_patterns, -1).astype(np.float64)
        n_clusters = min(n_clusters, n_patterns)
        centroids, labels = kmeans2(flat, n_clusters, minit='points', seed=42)
        return labels, centroids

    def positional_bias(self, attention_weights, seq_length):
        """Measure bias toward local vs. global attention.

        Returns ratio of attention weight on nearby tokens (|i-j| <= seq_length/4)
        vs. total weight.

        Parameters
        ----------
        attention_weights : ndarray, shape (n_heads, seq_len, seq_len)
        seq_length : int

        Returns
        -------
        local_ratio : ndarray, shape (n_heads,)
            Fraction of attention on local tokens per head.
        """
        positions = np.arange(seq_length)
        dist = np.abs(positions[:, None] - positions[None, :])
        local_mask = dist <= seq_length // 4
        local_weight = (attention_weights * local_mask[None]).sum(axis=(-2, -1))
        total_weight = attention_weights.sum(axis=(-2, -1))
        total_weight = np.where(total_weight > 1e-12, total_weight, 1e-12)
        return local_weight / total_weight

    def attention_rank(self, attention_weights):
        """Effective rank of attention matrix via participation ratio of singular values.

        Parameters
        ----------
        attention_weights : ndarray, shape (n_heads, seq_len, seq_len)

        Returns
        -------
        ranks : ndarray, shape (n_heads,)
        """
        ranks = np.zeros(attention_weights.shape[0])
        for h in range(attention_weights.shape[0]):
            s = np.linalg.svd(attention_weights[h], compute_uv=False)
            s = s[s > 1e-12]
            if len(s) == 0:
                ranks[h] = 0.0
                continue
            p = s ** 2 / (s ** 2).sum()
            ranks[h] = np.exp(-np.sum(p * np.log(p + 1e-12)))
        return ranks

    def pattern_evolution(self, attention_patterns_over_training):
        """Track how attention patterns change during training.

        Parameters
        ----------
        attention_patterns_over_training : list of ndarray
            Each entry shape (n_heads, seq_len, seq_len) at a training step.

        Returns
        -------
        evolution : dict with keys:
            'entropy_trajectory': (n_steps, n_heads, seq_len)
            'rank_trajectory': (n_steps, n_heads)
            'pattern_change_rate': (n_steps - 1,)
        """
        n_steps = len(attention_patterns_over_training)
        first = attention_patterns_over_training[0]
        n_heads = first.shape[0]
        seq_len = first.shape[1]

        entropy_traj = np.zeros((n_steps, n_heads, seq_len))
        rank_traj = np.zeros((n_steps, n_heads))
        change_rate = np.zeros(max(n_steps - 1, 0))

        for t, patterns in enumerate(attention_patterns_over_training):
            entropy_traj[t] = self.attention_entropy(patterns)
            rank_traj[t] = self.attention_rank(patterns)
            if t > 0:
                prev = attention_patterns_over_training[t - 1]
                diff = np.linalg.norm(patterns - prev) / (np.linalg.norm(prev) + 1e-12)
                change_rate[t - 1] = diff

        return {
            'entropy_trajectory': entropy_traj,
            'rank_trajectory': rank_traj,
            'pattern_change_rate': change_rate,
        }


class SelfAttentionRecursion:
    """Self-attention kernel recursion through transformer layers."""

    def __init__(self, n_layers, d_model, n_heads):
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads

    def single_layer_map(self, K_in, layer_params):
        """K^{l+1} from K^l through self-attention kernel map.

        The attention kernel map applies softmax nonlinearity to the
        normalized kernel, then recombines with value kernel.

        Parameters
        ----------
        K_in : ndarray, shape (n, n)
            Input kernel (Gram matrix).
        layer_params : dict with optional keys:
            'sigma_q', 'sigma_k', 'sigma_v': variance scales

        Returns
        -------
        K_out : ndarray, shape (n, n)
        """
        sigma_q = layer_params.get('sigma_q', 1.0)
        sigma_k = layer_params.get('sigma_k', 1.0)
        sigma_v = layer_params.get('sigma_v', 1.0)

        # Normalize kernel to get cosine similarities
        diag = np.sqrt(np.diag(K_in))
        diag = np.where(diag > 1e-12, diag, 1e-12)
        K_norm = K_in / np.outer(diag, diag)

        # Query-key interaction kernel: exponential of scaled similarity
        qk_scale = sigma_q * sigma_k / np.sqrt(self.d_model)
        K_qk = np.exp(qk_scale * K_norm)

        # Normalize rows (softmax-like)
        row_sums = K_qk.sum(axis=-1, keepdims=True)
        row_sums = np.where(row_sums > 1e-12, row_sums, 1e-12)
        A_kernel = K_qk / row_sums

        # Value kernel
        K_v = sigma_v ** 2 * K_in

        # Output: A_kernel applied to value kernel
        K_out = A_kernel @ K_v @ A_kernel.T
        return K_out

    def propagate_through_layers(self, K_input, layer_params_list):
        """Full propagation of kernel through all layers.

        Parameters
        ----------
        K_input : ndarray, shape (n, n)
        layer_params_list : list of dict, length n_layers

        Returns
        -------
        K_final : ndarray, shape (n, n)
        trajectory : list of ndarray
            Kernel at each layer.
        """
        trajectory = [K_input.copy()]
        K = K_input.copy()
        for l in range(self.n_layers):
            params = layer_params_list[l] if l < len(layer_params_list) else {}
            K = self.single_layer_map(K, params)
            trajectory.append(K.copy())
        return K, trajectory

    def residual_connection(self, K_attn, K_input, alpha=1.0):
        """K_out = K_input + α·K_attn + α·cross_terms.

        For residual connection x + α·f(x), the kernel becomes:
        K_res = K_input + α² K_attn + α (K_cross + K_cross^T)

        We approximate the cross terms from the geometric mean.

        Parameters
        ----------
        K_attn : ndarray, shape (n, n)
        K_input : ndarray, shape (n, n)
        alpha : float

        Returns
        -------
        K_res : ndarray, shape (n, n)
        """
        # Cross-term approximation: sqrt(K_input ⊙ K_attn)
        prod = np.abs(K_input * K_attn)
        K_cross = np.sqrt(prod) * np.sign(K_input * K_attn)
        return K_input + alpha ** 2 * K_attn + alpha * (K_cross + K_cross.T)

    def layer_norm_effect(self, K_in):
        """Layer normalization modification to kernel.

        LayerNorm projects onto the unit sphere, modifying the kernel to
        the normalized version.

        Parameters
        ----------
        K_in : ndarray, shape (n, n)

        Returns
        -------
        K_ln : ndarray, shape (n, n)
        """
        diag = np.sqrt(np.diag(K_in))
        diag = np.where(diag > 1e-12, diag, 1e-12)
        K_ln = K_in / np.outer(diag, diag)

        # LayerNorm also introduces a Jacobian correction
        n = K_in.shape[0]
        d = self.d_model
        # Jacobian of LN: (I - 11^T/d) / ||x||, contributes (1 - 1/d) factor
        correction = (1.0 - 1.0 / d)
        K_ln *= correction
        return K_ln

    def ffn_kernel(self, K_in, hidden_dim, activation='relu'):
        """Feed-forward network block kernel.

        FFN: x -> W2 * activation(W1 * x + b1) + b2
        In the NTK, this becomes a dual-activation kernel.

        Parameters
        ----------
        K_in : ndarray, shape (n, n)
        hidden_dim : int
        activation : str, one of 'relu', 'gelu'

        Returns
        -------
        K_ffn : ndarray, shape (n, n)
        """
        diag = np.sqrt(np.abs(np.diag(K_in)))
        diag = np.where(diag > 1e-12, diag, 1e-12)
        K_norm = K_in / np.outer(diag, diag)
        K_norm = np.clip(K_norm, -1.0, 1.0)

        n = K_in.shape[0]
        if activation == 'relu':
            # ReLU dual kernel: κ1(θ) = (sin θ + (π - θ) cos θ) / (2π)
            theta = np.arccos(K_norm)
            kappa1 = (np.sin(theta) + (np.pi - theta) * np.cos(theta)) / (2 * np.pi)
        elif activation == 'gelu':
            # GELU approximation: similar to ReLU with smooth correction
            theta = np.arccos(K_norm)
            kappa1 = (np.sin(theta) + (np.pi - theta) * np.cos(theta)) / (2 * np.pi)
            # Smooth correction for GELU
            kappa1 += 0.5 * K_norm * (1.0 + np.erf(K_norm / np.sqrt(2))) / np.pi
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Scale by variances and hidden dim ratio
        scale = np.outer(diag, diag)
        K_ffn = kappa1 * scale * (hidden_dim / self.d_model)
        return K_ffn

    def transformer_block_kernel(self, K_in, block_params):
        """Full transformer block: LN -> Attention -> Residual -> LN -> FFN -> Residual.

        Parameters
        ----------
        K_in : ndarray, shape (n, n)
        block_params : dict with keys:
            'attention': dict of attention params
            'ffn_hidden_dim': int
            'ffn_activation': str
            'alpha_attn': float (residual scaling for attention)
            'alpha_ffn': float (residual scaling for FFN)

        Returns
        -------
        K_out : ndarray, shape (n, n)
        """
        alpha_attn = block_params.get('alpha_attn', 1.0)
        alpha_ffn = block_params.get('alpha_ffn', 1.0)
        ffn_hidden = block_params.get('ffn_hidden_dim', 4 * self.d_model)
        ffn_act = block_params.get('ffn_activation', 'relu')

        # Pre-norm attention
        K_ln1 = self.layer_norm_effect(K_in)
        attn_params = block_params.get('attention', {})
        K_attn = self.single_layer_map(K_ln1, attn_params)
        K_mid = self.residual_connection(K_attn, K_in, alpha_attn)

        # Pre-norm FFN
        K_ln2 = self.layer_norm_effect(K_mid)
        K_ffn = self.ffn_kernel(K_ln2, ffn_hidden, ffn_act)
        K_out = self.residual_connection(K_ffn, K_mid, alpha_ffn)

        return K_out

    def depth_kernel_trajectory(self, K_input, block_params_list):
        """Kernel trajectory through all transformer blocks.

        Parameters
        ----------
        K_input : ndarray, shape (n, n)
        block_params_list : list of dict

        Returns
        -------
        K_final : ndarray, shape (n, n)
        trajectory : list of ndarray
            Kernel after each block.
        norms : list of float
            Frobenius norm at each layer.
        """
        trajectory = [K_input.copy()]
        norms = [float(np.linalg.norm(K_input, 'fro'))]
        K = K_input.copy()

        for block_params in block_params_list:
            K = self.transformer_block_kernel(K, block_params)
            trajectory.append(K.copy())
            norms.append(float(np.linalg.norm(K, 'fro')))

        return K, trajectory, norms

    def fixed_point_analysis(self, block_params):
        """Fixed point of transformer kernel recursion.

        Iterates the kernel map from a random initialization until convergence
        or max iterations.

        Parameters
        ----------
        block_params : dict

        Returns
        -------
        result : dict with keys:
            'fixed_point': ndarray or None
            'converged': bool
            'iterations': int
            'trajectory_norms': list of float
        """
        n = 8  # small size for fixed-point search
        rng = np.random.RandomState(0)
        X = rng.randn(n, self.d_model)
        K = X @ X.T / self.d_model

        max_iter = 200
        tol = 1e-6
        trajectory_norms = [float(np.linalg.norm(K, 'fro'))]

        for it in range(max_iter):
            K_new = self.transformer_block_kernel(K, block_params)
            diff = np.linalg.norm(K_new - K, 'fro')
            trajectory_norms.append(float(np.linalg.norm(K_new, 'fro')))
            if diff < tol:
                return {
                    'fixed_point': K_new,
                    'converged': True,
                    'iterations': it + 1,
                    'trajectory_norms': trajectory_norms,
                }
            K = K_new

        return {
            'fixed_point': K,
            'converged': False,
            'iterations': max_iter,
            'trajectory_norms': trajectory_norms,
        }


class PositionEncodingKernel:
    """Position encoding effects on kernel."""

    def __init__(self, d_model, max_seq_length=512):
        self.d_model = d_model
        self.max_seq_length = max_seq_length

    def sinusoidal_encoding(self, positions):
        """Standard sinusoidal positional encoding.

        PE(pos, 2i) = sin(pos / 10000^{2i/d})
        PE(pos, 2i+1) = cos(pos / 10000^{2i/d})

        Parameters
        ----------
        positions : ndarray, shape (seq_len,)

        Returns
        -------
        PE : ndarray, shape (seq_len, d_model)
        """
        positions = np.asarray(positions, dtype=np.float64)
        pe = np.zeros((len(positions), self.d_model))
        div_term = np.exp(
            np.arange(0, self.d_model, 2, dtype=np.float64)
            * (-np.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = np.sin(positions[:, None] * div_term[None, :])
        pe[:, 1::2] = np.cos(positions[:, None] * div_term[:pe[:, 1::2].shape[1], None].T)
        return pe

    def sinusoidal_kernel(self, pos1, pos2):
        """Kernel induced by sinusoidal PE: K_PE(i,j) = PE(i) · PE(j) / d.

        For sinusoidal encodings, this has a known closed form that depends
        only on position difference.

        Parameters
        ----------
        pos1, pos2 : ndarray, shape (n1,) and (n2,)

        Returns
        -------
        K_PE : ndarray, shape (n1, n2)
        """
        pe1 = self.sinusoidal_encoding(pos1)
        pe2 = self.sinusoidal_encoding(pos2)
        return pe1 @ pe2.T / self.d_model

    def learned_pe_kernel(self, pe_matrix):
        """Kernel from learned positional embedding matrix.

        Parameters
        ----------
        pe_matrix : ndarray, shape (max_seq_length, d_model)

        Returns
        -------
        K_PE : ndarray, shape (max_seq_length, max_seq_length)
        """
        return pe_matrix @ pe_matrix.T / self.d_model

    def rotary_encoding(self, positions, dim):
        """Rotary Position Embedding (RoPE).

        Computes rotation matrices for each position.

        Parameters
        ----------
        positions : ndarray, shape (seq_len,)
        dim : int
            Dimension of the embedding (must be even).

        Returns
        -------
        cos_enc : ndarray, shape (seq_len, dim // 2)
        sin_enc : ndarray, shape (seq_len, dim // 2)
        """
        positions = np.asarray(positions, dtype=np.float64)
        dim_half = dim // 2
        freqs = 1.0 / (10000.0 ** (np.arange(dim_half, dtype=np.float64) / dim_half))
        angles = positions[:, None] * freqs[None, :]
        return np.cos(angles), np.sin(angles)

    def rotary_kernel_modification(self, K_base, positions1, positions2):
        """RoPE effect on kernel.

        RoPE modifies the QK^T product to depend on relative position:
        (R_m q)^T (R_n k) = q^T R_{n-m} k

        The kernel modification introduces a position-dependent factor.

        Parameters
        ----------
        K_base : ndarray, shape (n1, n2)
        positions1, positions2 : ndarray

        Returns
        -------
        K_rope : ndarray, shape (n1, n2)
        """
        dim = self.d_model
        cos1, sin1 = self.rotary_encoding(positions1, dim)
        cos2, sin2 = self.rotary_encoding(positions2, dim)

        # RoPE factor: sum_i cos(θ_i (m - n)) averaged over dimensions
        # = sum_i [cos_m_i * cos_n_i + sin_m_i * sin_n_i]
        rope_factor = (cos1 @ cos2.T + sin1 @ sin2.T) / (dim // 2)

        return K_base * rope_factor

    def alibi_kernel(self, positions1, positions2, slopes):
        """ALiBi (Attention with Linear Biases) modification.

        Adds a linear bias -m * |i - j| to attention logits, which modifies
        the kernel through the softmax.

        Parameters
        ----------
        positions1, positions2 : ndarray, shape (n1,) and (n2,)
        slopes : ndarray, shape (n_heads,)

        Returns
        -------
        alibi_biases : ndarray, shape (n_heads, n1, n2)
            Bias terms to add to attention logits.
        """
        pos_diff = np.abs(
            np.asarray(positions1)[:, None] - np.asarray(positions2)[None, :]
        )
        slopes = np.asarray(slopes)
        biases = -slopes[:, None, None] * pos_diff[None, :, :]
        return biases

    def relative_pe_kernel(self, K_base, relative_positions):
        """Relative position encoding effect on kernel.

        Adds a learned bias based on relative position to the kernel.

        Parameters
        ----------
        K_base : ndarray, shape (n, n)
        relative_positions : ndarray, shape (n, n)
            Matrix of relative position indices.

        Returns
        -------
        K_rel : ndarray, shape (n, n)
        """
        max_rel = int(np.max(np.abs(relative_positions)))
        # Create a decaying bias based on relative distance
        bias_values = np.exp(-np.abs(np.arange(-max_rel, max_rel + 1)) / max(max_rel, 1))

        n = K_base.shape[0]
        bias_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                rel = int(relative_positions[i, j])
                rel = np.clip(rel, -max_rel, max_rel)
                bias_matrix[i, j] = bias_values[rel + max_rel]

        return K_base + bias_matrix

    def pe_effect_on_ntk(self, ntk_without_pe, pe_kernel):
        """Combined NTK with positional encoding contribution.

        The full NTK for inputs with PE is:
        K_NTK(x+pe, x'+pe') = K_NTK(x,x') + K_PE(pe,pe') + K_cross(x,pe')

        We approximate K_cross from the geometric mean of content and position
        kernels.

        Parameters
        ----------
        ntk_without_pe : ndarray, shape (n, n)
        pe_kernel : ndarray, shape (n, n)

        Returns
        -------
        ntk_combined : ndarray, shape (n, n)
        """
        # Cross-term: approximate interaction between content and position
        prod = np.abs(ntk_without_pe * pe_kernel)
        cross = np.sqrt(prod) * np.sign(ntk_without_pe)

        ntk_combined = ntk_without_pe + pe_kernel + cross + cross.T
        return ntk_combined
