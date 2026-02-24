"""
Pooling layer kernels for neural network kernel theory.

Implements exact and approximate kernel transformations through
max pooling, average pooling, global average pooling, and adaptive pooling.
"""

import numpy as np
from scipy import linalg, special, stats


class MaxPoolingKernel:
    """Max pooling kernel approximation using Gaussian integral techniques."""

    def __init__(self, pool_size=2, stride=2, padding=0):
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding

    def max_pool_expected_kernel(self, K_in, spatial_dim):
        """Expected kernel after max pooling under Gaussian approximation.

        For correlated Gaussians with covariance K_in, computes E[max(x_i) max(x_j)]
        over pooling windows.
        """
        n = K_in.shape[0]
        p = self.pool_size
        out_dim = (spatial_dim + 2 * self.padding - p) // self.stride + 1
        total_out = out_dim * out_dim

        K_out = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                kii = K_in[i, i]
                kjj = K_in[j, j]
                kij = K_in[i, j]
                rho = kij / (np.sqrt(kii * kjj) + 1e-12)
                # E[max] for standard Gaussians scales with sqrt(2 log p^2)
                scale_i = np.sqrt(kii)
                scale_j = np.sqrt(kjj)
                p2 = p * p
                e_max_scale = np.sqrt(2.0 * np.log(max(p2, 2)))
                # Correlation of maxima approximated via extreme value theory
                rho_max = rho ** p2
                K_out[i, j] = scale_i * scale_j * e_max_scale**2 * rho_max
                K_out[j, i] = K_out[i, j]

        return K_out

    def max_of_gaussians_correlation(self, means, covariance, pool_size):
        """Correlation of max of correlated Gaussians via Monte Carlo.

        Draws samples from N(means, covariance) and computes
        correlation structure of the max over groups of pool_size.
        """
        n = len(means)
        n_samples = 5000
        samples = np.random.multivariate_normal(means, covariance, size=n_samples)

        n_groups = n // pool_size
        max_samples = np.zeros((n_samples, n_groups))
        for g in range(n_groups):
            start = g * pool_size
            end = start + pool_size
            max_samples[:, g] = np.max(samples[:, start:end], axis=1)

        corr = np.corrcoef(max_samples.T)
        return corr

    def max_pool_jacobian(self, pre_pool_activations, pool_indices):
        """Jacobian of max pooling: binary mask at argmax positions.

        Returns sparse Jacobian ∂maxpool/∂input where entries are 1
        at the index achieving the max in each window, 0 elsewhere.
        """
        n_out = len(pool_indices)
        n_in = pre_pool_activations.shape[-1]
        jacobian = np.zeros((n_out, n_in))

        for i, indices in enumerate(pool_indices):
            window = pre_pool_activations[indices]
            winner = indices[np.argmax(window)]
            jacobian[i, winner] = 1.0

        return jacobian

    def max_pool_ntk_contribution(self, K_in, jacobian):
        """NTK contribution through max pooling layer.

        NTK_out = J @ K_in @ J^T where J is the max pool Jacobian.
        """
        return jacobian @ K_in @ jacobian.T

    def smooth_max_approximation(self, K_in, beta=10.0):
        """Smooth max (log-sum-exp) kernel approximation.

        Uses softmax(beta * x) as differentiable proxy for max.
        K_smooth[i,j] ≈ E[softmax_i softmax_j] under Gaussian assumption.
        """
        n = K_in.shape[0]
        K_smooth = np.zeros_like(K_in)

        for i in range(n):
            for j in range(i, n):
                kii = K_in[i, i]
                kjj = K_in[j, j]
                kij = K_in[i, j]
                # For softmax at temperature 1/beta, effective kernel scales
                var_i = beta**2 * kii
                var_j = beta**2 * kjj
                cov_ij = beta**2 * kij
                # LogSumExp mean ≈ max + gamma/beta for iid; use moment matching
                # Second moment: E[LSE_i LSE_j] via cumulant expansion
                rho = cov_ij / (np.sqrt(var_i * var_j) + 1e-12)
                K_smooth[i, j] = np.sqrt(kii * kjj) * (1.0 + rho) / 2.0
                K_smooth[j, i] = K_smooth[i, j]

        return K_smooth

    def max_pool_variance(self, K_in, spatial_dim, n_samples=1000):
        """Monte Carlo estimate of kernel variance after max pooling."""
        n = K_in.shape[0]
        p = self.pool_size
        out_dim = (spatial_dim + 2 * self.padding - p) // self.stride + 1

        L = linalg.cholesky(K_in + 1e-8 * np.eye(n), lower=True)
        kernel_samples = np.zeros((n_samples, n, n))

        for s in range(n_samples):
            z = L @ np.random.randn(n, spatial_dim * spatial_dim)
            pooled = np.zeros((n, out_dim * out_dim))
            idx = 0
            for row in range(out_dim):
                for col in range(out_dim):
                    r0 = row * self.stride
                    c0 = col * self.stride
                    window_indices = []
                    for dr in range(p):
                        for dc in range(p):
                            si = (r0 + dr) * spatial_dim + (c0 + dc)
                            if si < z.shape[1]:
                                window_indices.append(si)
                    if window_indices:
                        pooled[:, idx] = np.max(z[:, window_indices], axis=1)
                    idx += 1

            kernel_samples[s] = pooled @ pooled.T / max(pooled.shape[1], 1)

        mean_kernel = np.mean(kernel_samples, axis=0)
        var_kernel = np.var(kernel_samples, axis=0)
        return mean_kernel, var_kernel

    def spatial_kernel_after_maxpool(self, K_spatial, pool_size):
        """Transform spatial kernel through max pooling.

        K_spatial is (S, S) kernel over spatial locations.
        Returns (S', S') kernel after pooling.
        """
        S = K_spatial.shape[0]
        S_out = S // pool_size
        K_out = np.zeros((S_out, S_out))

        for i in range(S_out):
            for j in range(S_out):
                block_i = slice(i * pool_size, (i + 1) * pool_size)
                block_j = slice(j * pool_size, (j + 1) * pool_size)
                sub_block = K_spatial[block_i, block_j]
                diag_i = np.diag(K_spatial)[block_i]
                diag_j = np.diag(K_spatial)[block_j]
                # Approximate E[max_a max_b] using dominant correlation
                max_corr = np.max(sub_block / (
                    np.sqrt(np.outer(diag_i, diag_j)) + 1e-12))
                scale = np.sqrt(np.max(diag_i) * np.max(diag_j))
                K_out[i, j] = scale * max_corr

        return K_out

    def max_pool_depth_propagation(self, K_input, depth, pool_layers):
        """Propagate kernel through multiple conv layers with pooling.

        pool_layers: set of layer indices where max pooling is applied.
        Alternates ReLU kernel map and max pool.
        """
        K = K_input.copy()
        for d in range(depth):
            # ReLU kernel transform: K' = (1/π)(√(1-ρ²) + ρ(π - arccos(ρ)))
            diag = np.sqrt(np.diag(K))
            outer = np.outer(diag, diag) + 1e-12
            rho = np.clip(K / outer, -1, 1)
            K = outer * (1.0 / np.pi) * (
                np.sqrt(1.0 - rho**2) + rho * (np.pi - np.arccos(rho)))

            if d in pool_layers:
                # Max pool shrinks effective dimensionality
                n = K.shape[0]
                scale = np.sqrt(2.0 * np.log(max(self.pool_size**2, 2)))
                rho_new = np.clip(K / (np.sqrt(
                    np.outer(np.diag(K), np.diag(K))) + 1e-12), -1, 1)
                K = np.diag(K)[:, None] * np.ones((1, n)) * scale**2
                K = np.sqrt(np.outer(np.diag(K).copy(), np.diag(K).copy())) * rho_new

        return K

    def max_pool_information_loss(self, K_before, K_after):
        """Information loss from max pooling measured by log-det ratio.

        Uses log det K_before - log det K_after as information measure.
        """
        eps = 1e-8
        n1 = K_before.shape[0]
        n2 = K_after.shape[0]
        sign1, logdet1 = np.linalg.slogdet(K_before + eps * np.eye(n1))
        sign2, logdet2 = np.linalg.slogdet(K_after + eps * np.eye(n2))
        # Normalize by dimension
        info_before = logdet1 / n1
        info_after = logdet2 / n2
        return {
            'info_before': info_before,
            'info_after': info_after,
            'info_loss': info_before - info_after,
            'relative_loss': 1.0 - info_after / (info_before + 1e-12),
        }


class AveragePoolingKernel:
    """Average pooling exact kernel computations."""

    def __init__(self, pool_size=2, stride=2, padding=0):
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding

    def avg_pool_kernel(self, K_in, spatial_dim):
        """Exact kernel after average pooling.

        K_out[a,b] = (1/p^2)^2 Σ_{i in window_a} Σ_{j in window_b} K_in_spatial[i,j]
        For sample kernel K_in of shape (n, n), averages over spatial blocks.
        """
        n = K_in.shape[0]
        p = self.pool_size
        out_dim = (spatial_dim + 2 * self.padding - p) // self.stride + 1

        P = self.avg_pool_matrix(spatial_dim, p, self.stride)
        # K_in represents the kernel between n data points
        # Average pooling acts on spatial dims; for data kernel, it's a contraction
        K_out = K_in.copy()
        # Scale by spatial averaging factor
        spatial_factor = 1.0 / (p * p)
        # Average pooling preserves kernel structure, contracts spatial variance
        K_out = K_in * spatial_factor
        # Add back diagonal boost from averaging correlated features
        K_out += np.diag(np.diag(K_in)) * spatial_factor * (p * p - 1) / (p * p)
        return K_out

    def avg_pool_matrix(self, spatial_dim, pool_size, stride):
        """Construct the average pooling matrix P such that output = P @ input.

        P is (out_dim^2, spatial_dim^2) with entries 1/pool_size^2.
        """
        out_dim = (spatial_dim - pool_size) // stride + 1
        n_in = spatial_dim * spatial_dim
        n_out = out_dim * out_dim
        P = np.zeros((n_out, n_in))

        out_idx = 0
        for row in range(out_dim):
            for col in range(out_dim):
                r0 = row * stride
                c0 = col * stride
                for dr in range(pool_size):
                    for dc in range(pool_size):
                        in_idx = (r0 + dr) * spatial_dim + (c0 + dc)
                        P[out_idx, in_idx] = 1.0 / (pool_size * pool_size)
                out_idx += 1

        return P

    def avg_pool_ntk(self, K_in, pool_matrix):
        """NTK through average pooling.

        Since avg pool is linear: NTK_out = P @ NTK_in @ P^T.
        """
        return pool_matrix @ K_in @ pool_matrix.T

    def spatial_averaging_effect(self, K_spatial, pool_size):
        """Effect of average pooling on spatial correlation structure.

        Given spatial kernel K_spatial of shape (S, S), compute
        the block-averaged kernel of shape (S/p, S/p).
        """
        S = K_spatial.shape[0]
        S_out = S // pool_size
        K_out = np.zeros((S_out, S_out))

        for i in range(S_out):
            for j in range(S_out):
                block = K_spatial[
                    i * pool_size:(i + 1) * pool_size,
                    j * pool_size:(j + 1) * pool_size
                ]
                K_out[i, j] = np.mean(block)

        return K_out

    def strided_avg_pool(self, K_in, spatial_dim, stride):
        """Average pooling with arbitrary stride (may differ from pool_size)."""
        p = self.pool_size
        out_dim = (spatial_dim - p) // stride + 1
        P = np.zeros((out_dim * out_dim, spatial_dim * spatial_dim))

        out_idx = 0
        for row in range(out_dim):
            for col in range(out_dim):
                r0, c0 = row * stride, col * stride
                for dr in range(p):
                    for dc in range(p):
                        in_idx = (r0 + dr) * spatial_dim + (c0 + dc)
                        P[out_idx, in_idx] = 1.0 / (p * p)
                out_idx += 1

        return P @ K_in @ P.T

    def avg_pool_eigenspectrum(self, K_in, pool_matrix):
        """Eigenspectrum of kernel after average pooling.

        Computes eigenvalues of P K P^T and compares to K.
        """
        K_pooled = pool_matrix @ K_in @ pool_matrix.T
        eigs_before = np.sort(np.linalg.eigvalsh(K_in))[::-1]
        eigs_after = np.sort(np.linalg.eigvalsh(K_pooled))[::-1]

        # Effective rank via participation ratio
        def participation_ratio(eigs):
            eigs_pos = eigs[eigs > 1e-10]
            if len(eigs_pos) == 0:
                return 0.0
            p = eigs_pos / np.sum(eigs_pos)
            return 1.0 / np.sum(p**2)

        return {
            'eigenvalues_before': eigs_before,
            'eigenvalues_after': eigs_after,
            'rank_before': participation_ratio(eigs_before),
            'rank_after': participation_ratio(eigs_after),
            'trace_ratio': np.sum(eigs_after) / (np.sum(eigs_before) + 1e-12),
        }

    def multi_scale_avg_pool(self, K_in, pool_sizes):
        """Average pooling at multiple scales, returning kernels at each.

        Useful for multi-scale kernel analysis.
        """
        results = {}
        S = K_in.shape[0]
        for ps in pool_sizes:
            if S % ps != 0:
                continue
            K_pooled = self.spatial_averaging_effect(K_in, ps)
            eigs = np.linalg.eigvalsh(K_pooled)
            results[ps] = {
                'kernel': K_pooled,
                'output_size': S // ps,
                'trace': np.trace(K_pooled),
                'top_eigenvalue': np.max(eigs),
                'effective_rank': np.sum(eigs > 1e-8 * np.max(eigs)),
            }
        return results

    def avg_pool_vs_max_pool(self, K_in, spatial_dim, pool_size):
        """Compare average pooling and max pooling kernels."""
        K_avg = self.spatial_averaging_effect(K_in, pool_size)

        mp = MaxPoolingKernel(pool_size=pool_size)
        K_max = mp.spatial_kernel_after_maxpool(K_in, pool_size)

        diff = K_avg - K_max
        return {
            'avg_kernel': K_avg,
            'max_kernel': K_max,
            'frobenius_diff': np.linalg.norm(diff, 'fro'),
            'max_entry_diff': np.max(np.abs(diff)),
            'trace_avg': np.trace(K_avg),
            'trace_max': np.trace(K_max),
            'correlation': np.sum(K_avg * K_max) / (
                np.linalg.norm(K_avg, 'fro') * np.linalg.norm(K_max, 'fro') + 1e-12),
        }


class GlobalAveragePoolingKernel:
    """Global average pooling kernel computations."""

    def __init__(self):
        pass

    def gap_kernel(self, K_spatial, spatial_dim):
        """Kernel after global average pooling.

        K_GAP = (1/S^2) Σ_{i,j} K_spatial[i,j]
        Collapses spatial dimensions into a scalar kernel value per pair.
        """
        S2 = spatial_dim * spatial_dim
        if K_spatial.ndim == 2 and K_spatial.shape[0] == K_spatial.shape[1]:
            # K_spatial is (S, S) spatial kernel
            return np.sum(K_spatial) / (S2)
        elif K_spatial.ndim == 4:
            # K_spatial is (n, n, S, S) -> (n, n)
            n = K_spatial.shape[0]
            K_gap = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K_gap[i, j] = np.sum(K_spatial[i, j]) / S2
            return K_gap
        return np.mean(K_spatial)

    def gap_ntk(self, K_in, spatial_dim):
        """NTK through global average pooling.

        GAP is linear: NTK_GAP = (1/S^2) Σ_{i,j} NTK_spatial[i,j].
        For data kernel K_in of shape (n, n), applies spatial contraction.
        """
        S2 = spatial_dim * spatial_dim
        if K_in.ndim == 4:
            n = K_in.shape[0]
            ntk_gap = np.zeros((n, n))
            for a in range(n):
                for b in range(n):
                    ntk_gap[a, b] = np.sum(K_in[a, b]) / S2
            return ntk_gap
        # If K_in is already (n, n), GAP just scales
        return K_in / S2

    def gap_feature_kernel(self, feature_kernels, spatial_dim):
        """Kernel after GAP applied per channel.

        feature_kernels: list of (S, S) spatial kernels, one per channel.
        Returns the combined kernel after GAP across channels.
        """
        n_channels = len(feature_kernels)
        S2 = spatial_dim * spatial_dim

        # Each channel contributes 1/S^2 * sum of its spatial kernel
        channel_contribs = np.zeros(n_channels)
        for c in range(n_channels):
            channel_contribs[c] = np.sum(feature_kernels[c]) / S2

        # Total GAP kernel is sum over channels (assuming independent channels)
        return np.sum(channel_contribs) / n_channels

    def spatial_to_channel_kernel(self, K_full, spatial_dim, n_channels):
        """Restructure (C*S, C*S) kernel into per-channel spatial kernels.

        K_full has spatial and channel dims flattened.
        Returns array of shape (C, C, S, S).
        """
        S = spatial_dim
        K_structured = np.zeros((n_channels, n_channels, S, S))

        for ci in range(n_channels):
            for cj in range(n_channels):
                for si in range(S):
                    for sj in range(S):
                        idx_i = ci * S + si
                        idx_j = cj * S + sj
                        if idx_i < K_full.shape[0] and idx_j < K_full.shape[1]:
                            K_structured[ci, cj, si, sj] = K_full[idx_i, idx_j]

        return K_structured

    def gap_information_content(self, K_before_gap, K_after_gap):
        """Information retained by GAP via mutual information proxy.

        Compares rank and trace of kernels before and after GAP.
        """
        eps = 1e-8
        n1 = K_before_gap.shape[0]
        eigs_before = np.linalg.eigvalsh(K_before_gap + eps * np.eye(n1))
        eigs_before = eigs_before[eigs_before > eps]

        if K_after_gap.ndim == 0 or np.isscalar(K_after_gap):
            # Scalar kernel after GAP
            return {
                'rank_before': len(eigs_before),
                'rank_after': 1,
                'entropy_before': -np.sum(
                    eigs_before / np.sum(eigs_before) *
                    np.log(eigs_before / np.sum(eigs_before) + eps)),
                'entropy_after': 0.0,
                'retention_ratio': 0.0,
            }

        n2 = K_after_gap.shape[0]
        eigs_after = np.linalg.eigvalsh(K_after_gap + eps * np.eye(n2))
        eigs_after = eigs_after[eigs_after > eps]

        def entropy(eigs):
            p = eigs / np.sum(eigs)
            return -np.sum(p * np.log(p + eps))

        h_before = entropy(eigs_before)
        h_after = entropy(eigs_after)

        return {
            'rank_before': len(eigs_before),
            'rank_after': len(eigs_after),
            'entropy_before': h_before,
            'entropy_after': h_after,
            'retention_ratio': h_after / (h_before + eps),
        }

    def gap_vs_flatten(self, K_spatial, spatial_dim):
        """Compare GAP to flattening (keeping all spatial features).

        GAP averages spatial dims; flatten preserves them.
        """
        S = spatial_dim
        K_gap_val = self.gap_kernel(K_spatial, spatial_dim)

        # Flatten kernel is just K_spatial itself (all spatial positions kept)
        K_flat = K_spatial

        eigs_flat = np.sort(np.linalg.eigvalsh(K_flat))[::-1]
        gap_scalar = K_gap_val if np.isscalar(K_gap_val) else np.mean(K_gap_val)

        return {
            'gap_kernel_value': gap_scalar,
            'flatten_kernel': K_flat,
            'flatten_trace': np.trace(K_flat),
            'flatten_rank': np.sum(eigs_flat > 1e-8 * eigs_flat[0]),
            'gap_dimensionality_reduction': S * S,
            'spatial_variance_lost': np.var(K_spatial) * (S * S - 1) / (S * S),
        }

    def gap_gradient_flow(self, K_spatial, spatial_dim):
        """Gradient flow through GAP layer.

        GAP distributes gradients uniformly: ∂L/∂x_i = (1/S^2) ∂L/∂x_gap.
        Effect on NTK: uniform contribution from each spatial position.
        """
        S2 = spatial_dim * spatial_dim
        gradient_scale = 1.0 / S2

        # Gradient covariance after GAP
        grad_cov = K_spatial * gradient_scale**2

        # Effective gradient magnitude
        grad_magnitude = np.sqrt(np.trace(K_spatial)) * gradient_scale

        # Gradient alignment across spatial positions
        diag = np.diag(K_spatial)
        off_diag = K_spatial - np.diag(diag)
        alignment = np.sum(off_diag) / (np.sum(np.abs(off_diag)) + 1e-12)

        return {
            'gradient_scale': gradient_scale,
            'gradient_covariance': grad_cov,
            'effective_gradient_magnitude': grad_magnitude,
            'spatial_gradient_alignment': alignment,
            'gradient_uniformity': 1.0 - np.std(diag) / (np.mean(diag) + 1e-12),
        }

    def gap_depth_effect(self, K_inputs, depths, spatial_dims):
        """Analyze GAP applied at different depths.

        K_inputs: list of spatial kernels at different depths.
        Returns GAP kernel value and information content at each depth.
        """
        results = []
        for K, depth, sdim in zip(K_inputs, depths, spatial_dims):
            gap_val = self.gap_kernel(K, sdim)
            trace = np.trace(K)
            eigs = np.linalg.eigvalsh(K)
            eigs_pos = eigs[eigs > 1e-10]
            p = eigs_pos / np.sum(eigs_pos)
            eff_rank = 1.0 / np.sum(p**2) if len(eigs_pos) > 0 else 0

            results.append({
                'depth': depth,
                'spatial_dim': sdim,
                'gap_kernel': gap_val,
                'pre_gap_trace': trace,
                'pre_gap_effective_rank': eff_rank,
                'spatial_correlation_mean': np.mean(K),
            })
        return results


class AdaptivePoolingKernel:
    """Adaptive pooling kernel for variable input/output spatial sizes."""

    def __init__(self, output_size):
        self.output_size = output_size

    def adaptive_avg_pool_kernel(self, K_in, input_spatial, output_spatial):
        """Adaptive average pooling kernel.

        Maps input_spatial -> output_spatial by choosing window sizes
        that evenly cover the input.
        """
        P = self.pooling_matrix_adaptive(input_spatial, output_spatial)
        return P @ K_in @ P.T

    def adaptive_max_pool_kernel(self, K_in, input_spatial, output_spatial):
        """Adaptive max pooling kernel (Gaussian approximation).

        Uses smooth max approximation for differentiability.
        """
        P_regions = self._adaptive_regions(input_spatial, output_spatial)
        n_out = output_spatial * output_spatial
        K_out = np.zeros((n_out, n_out))

        for i in range(n_out):
            for j in range(n_out):
                ri, rj = P_regions[i], P_regions[j]
                sub = K_in[np.ix_(ri, rj)]
                diag_i = np.array([K_in[a, a] for a in ri])
                diag_j = np.array([K_in[b, b] for b in rj])
                # Approximate max correlation
                if len(ri) > 0 and len(rj) > 0:
                    max_var_i = np.max(diag_i)
                    max_var_j = np.max(diag_j)
                    max_cov = np.max(sub)
                    rho = max_cov / (np.sqrt(max_var_i * max_var_j) + 1e-12)
                    scale = np.sqrt(2 * np.log(max(len(ri), 2)))
                    K_out[i, j] = np.sqrt(max_var_i * max_var_j) * rho * scale**2
                    
        return K_out

    def pooling_matrix_adaptive(self, input_size, output_size):
        """Compute adaptive average pooling matrix.

        Divides input into output_size bins and averages within each.
        """
        n_in = input_size * input_size
        n_out = output_size * output_size
        P = np.zeros((n_out, n_in))

        for out_r in range(output_size):
            for out_c in range(output_size):
                r_start = (out_r * input_size) // output_size
                r_end = ((out_r + 1) * input_size) // output_size
                c_start = (out_c * input_size) // output_size
                c_end = ((out_c + 1) * input_size) // output_size

                out_idx = out_r * output_size + out_c
                count = 0
                for r in range(r_start, r_end):
                    for c in range(c_start, c_end):
                        in_idx = r * input_size + c
                        P[out_idx, in_idx] = 1.0
                        count += 1
                if count > 0:
                    P[out_idx] /= count

        return P

    def _adaptive_regions(self, input_size, output_size):
        """Compute index regions for adaptive pooling."""
        regions = []
        for out_r in range(output_size):
            for out_c in range(output_size):
                r_start = (out_r * input_size) // output_size
                r_end = ((out_r + 1) * input_size) // output_size
                c_start = (out_c * input_size) // output_size
                c_end = ((out_c + 1) * input_size) // output_size
                indices = []
                for r in range(r_start, r_end):
                    for c in range(c_start, c_end):
                        indices.append(r * input_size + c)
                regions.append(indices)
        return regions

    def spatial_resolution_effect(self, K_in, output_sizes):
        """Effect of different output resolutions on kernel.

        Computes adaptive avg pool kernel for each output size.
        """
        input_size = int(np.sqrt(K_in.shape[0]))
        results = {}
        for os in output_sizes:
            if os > input_size:
                continue
            P = self.pooling_matrix_adaptive(input_size, os)
            K_out = P @ K_in @ P.T
            eigs = np.linalg.eigvalsh(K_out)
            results[os] = {
                'kernel': K_out,
                'trace': np.trace(K_out),
                'top_eigenvalue': np.max(eigs),
                'compression_ratio': (input_size / os) ** 2,
            }
        return results

    def adaptive_vs_fixed(self, K_in, spatial_dim, output_size):
        """Compare adaptive pooling to fixed-size pooling."""
        P_adaptive = self.pooling_matrix_adaptive(spatial_dim, output_size)
        K_adaptive = P_adaptive @ K_in @ P_adaptive.T

        pool_size = spatial_dim // output_size
        if pool_size > 0 and spatial_dim % output_size == 0:
            ap = AveragePoolingKernel(pool_size=pool_size, stride=pool_size)
            P_fixed = ap.avg_pool_matrix(spatial_dim, pool_size, pool_size)
            K_fixed = P_fixed @ K_in @ P_fixed.T
        else:
            K_fixed = K_adaptive  # fallback

        diff = K_adaptive - K_fixed
        return {
            'adaptive_kernel': K_adaptive,
            'fixed_kernel': K_fixed,
            'frobenius_diff': np.linalg.norm(diff, 'fro'),
            'are_equivalent': np.allclose(K_adaptive, K_fixed, atol=1e-10),
        }

    def multi_resolution_kernel(self, K_in, resolutions):
        """Compute kernel at multiple spatial resolutions.

        Returns dict mapping resolution -> pooled kernel.
        """
        input_size = int(np.sqrt(K_in.shape[0]))
        kernels = {}
        for res in resolutions:
            if res > input_size:
                continue
            P = self.pooling_matrix_adaptive(input_size, res)
            kernels[res] = P @ K_in @ P.T
        return kernels


class PoolingSpatialAnalyzer:
    """Analyze how pooling affects spatial structure of kernels."""

    def __init__(self):
        pass

    def spatial_correlation_before_after(self, K_spatial, pool_fn):
        """Measure change in spatial correlation structure from pooling.

        pool_fn: callable that takes K_spatial and returns K_pooled.
        """
        K_after = pool_fn(K_spatial)

        # Normalize to correlation matrices
        def to_corr(K):
            d = np.sqrt(np.diag(K) + 1e-12)
            return K / np.outer(d, d)

        C_before = to_corr(K_spatial)
        C_after = to_corr(K_after)

        # Statistics
        off_diag_before = C_before[np.triu_indices_from(C_before, k=1)]
        off_diag_after = C_after[np.triu_indices_from(C_after, k=1)]

        return {
            'corr_before': C_before,
            'corr_after': C_after,
            'mean_corr_before': np.mean(off_diag_before) if len(off_diag_before) > 0 else 0,
            'mean_corr_after': np.mean(off_diag_after) if len(off_diag_after) > 0 else 0,
            'std_corr_before': np.std(off_diag_before) if len(off_diag_before) > 0 else 0,
            'std_corr_after': np.std(off_diag_after) if len(off_diag_after) > 0 else 0,
        }

    def translation_invariance_measure(self, K_spatial):
        """Measure how translation-invariant a spatial kernel is.

        A Toeplitz (circulant) kernel is perfectly translation-invariant.
        Measures deviation from Toeplitz structure.
        """
        S = K_spatial.shape[0]
        # Group entries by displacement
        displacement_values = {}
        for i in range(S):
            for j in range(S):
                d = abs(i - j)
                if d not in displacement_values:
                    displacement_values[d] = []
                displacement_values[d].append(K_spatial[i, j])

        # Variance within each displacement group (should be 0 for Toeplitz)
        total_var = 0.0
        total_count = 0
        for d, vals in displacement_values.items():
            if len(vals) > 1:
                total_var += np.var(vals) * len(vals)
                total_count += len(vals)

        within_var = total_var / max(total_count, 1)
        total_kernel_var = np.var(K_spatial)

        invariance = 1.0 - within_var / (total_kernel_var + 1e-12)
        return {
            'invariance_score': np.clip(invariance, 0, 1),
            'within_displacement_var': within_var,
            'total_var': total_kernel_var,
            'n_displacements': len(displacement_values),
        }

    def pooling_induced_invariance(self, K_before, K_after):
        """Invariance gained from pooling operation."""
        inv_before = self.translation_invariance_measure(K_before)
        inv_after = self.translation_invariance_measure(K_after)

        return {
            'invariance_before': inv_before['invariance_score'],
            'invariance_after': inv_after['invariance_score'],
            'invariance_gain': (inv_after['invariance_score'] -
                                inv_before['invariance_score']),
        }

    def receptive_field_growth(self, pool_sizes, strides, depths):
        """Compute receptive field growth through pooling layers.

        Assuming 3x3 convolutions between pool layers.
        RF grows as: rf = rf + (kernel_size - 1) * stride_product
        """
        rf = 1  # initial receptive field
        stride_product = 1
        rf_history = [rf]

        for d in range(depths):
            # Conv layer with 3x3 kernel
            rf += (3 - 1) * stride_product

            # Pool layer if within pool schedule
            if d < len(pool_sizes):
                ps = pool_sizes[d]
                st = strides[d]
                rf += (ps - 1) * stride_product
                stride_product *= st

            rf_history.append(rf)

        return {
            'receptive_fields': rf_history,
            'final_rf': rf,
            'total_stride': stride_product,
            'growth_rate': rf / max(depths, 1),
        }

    def spatial_hierarchy(self, K_inputs_at_layers):
        """Analyze spatial kernel structure at each layer.

        K_inputs_at_layers: list of spatial kernels at successive layers.
        """
        hierarchy = []
        for layer_idx, K in enumerate(K_inputs_at_layers):
            eigs = np.linalg.eigvalsh(K)
            eigs_pos = eigs[eigs > 1e-10]
            p = eigs_pos / np.sum(eigs_pos) if len(eigs_pos) > 0 else np.array([1.0])
            eff_rank = 1.0 / np.sum(p**2) if len(eigs_pos) > 0 else 0

            inv = self.translation_invariance_measure(K)
            diag_mean = np.mean(np.diag(K))
            off_diag_mean = (np.sum(K) - np.trace(K)) / max(K.size - K.shape[0], 1)

            hierarchy.append({
                'layer': layer_idx,
                'spatial_dim': K.shape[0],
                'effective_rank': eff_rank,
                'trace': np.trace(K),
                'diagonal_mean': diag_mean,
                'off_diagonal_mean': off_diag_mean,
                'invariance': inv['invariance_score'],
                'condition_number': eigs[-1] / (eigs[0] + 1e-12) if len(eigs) > 1 else 1.0,
            })
        return hierarchy

    def effective_spatial_resolution(self, K_spatial):
        """Effective resolution: how many independent spatial positions.

        Based on eigenspectrum of the spatial kernel.
        """
        eigs = np.linalg.eigvalsh(K_spatial)
        eigs = eigs[eigs > 0]
        if len(eigs) == 0:
            return {'effective_resolution': 0, 'total_positions': K_spatial.shape[0]}

        p = eigs / np.sum(eigs)
        entropy = -np.sum(p * np.log(p + 1e-12))
        eff_res = np.exp(entropy)

        return {
            'effective_resolution': eff_res,
            'total_positions': K_spatial.shape[0],
            'compression_factor': K_spatial.shape[0] / (eff_res + 1e-12),
            'spectral_entropy': entropy,
        }

    def deformation_sensitivity(self, K_spatial, deformations):
        """Sensitivity of kernel to spatial deformations.

        deformations: list of permutation arrays representing deformations.
        Measures how much the kernel changes under each deformation.
        """
        K_norm = np.linalg.norm(K_spatial, 'fro')
        sensitivities = []

        for perm in deformations:
            K_deformed = K_spatial[np.ix_(perm, perm)]
            diff = np.linalg.norm(K_spatial - K_deformed, 'fro')
            sensitivities.append(diff / (K_norm + 1e-12))

        return {
            'sensitivities': np.array(sensitivities),
            'mean_sensitivity': np.mean(sensitivities),
            'max_sensitivity': np.max(sensitivities) if sensitivities else 0.0,
            'min_sensitivity': np.min(sensitivities) if sensitivities else 0.0,
        }

    def pooling_hierarchy_kernel(self, K_input, pool_schedule):
        """Full kernel through a hierarchy of pooling operations.

        pool_schedule: list of dicts with 'type' ('avg' or 'max') and 'size'.
        Applies each pooling operation sequentially.
        """
        K = K_input.copy()

        for layer in pool_schedule:
            pool_type = layer.get('type', 'avg')
            pool_size = layer.get('size', 2)

            S = K.shape[0]
            if S < pool_size:
                break

            if pool_type == 'avg':
                ap = AveragePoolingKernel(pool_size=pool_size, stride=pool_size)
                K = ap.spatial_averaging_effect(K, pool_size)
            elif pool_type == 'max':
                mp = MaxPoolingKernel(pool_size=pool_size, stride=pool_size)
                K = mp.spatial_kernel_after_maxpool(K, pool_size)

        return K
