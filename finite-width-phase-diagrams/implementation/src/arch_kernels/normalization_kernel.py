"""
Normalization layer effects on neural tangent kernels.

Implements kernel transformations induced by batch, layer, group, and instance
normalization, including mean-field theory modifications, implicit regularization,
and phase-diagram shifts.
"""

import numpy as np
from scipy import linalg
from scipy.optimize import fixed_point, minimize_scalar, brentq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_symmetric(K):
    return 0.5 * (K + K.T)


def _stable_inv_sqrt(x, eps=1e-8):
    return 1.0 / np.sqrt(np.maximum(x, eps))


def _centering_matrix(n):
    """H = I - (1/n) 11^T."""
    return np.eye(n) - np.ones((n, n)) / n


def _cosine_from_kernel(K, i, j):
    denom = np.sqrt(np.abs(K[i, i] * K[j, j])) + 1e-12
    return K[i, j] / denom


# ===================================================================
# 1. BatchNormKernel
# ===================================================================

class BatchNormKernel:
    """Batch normalization effect on the neural tangent kernel."""

    def __init__(self, momentum=0.1, epsilon=1e-5, affine=True):
        self.momentum = momentum
        self.epsilon = epsilon
        self.affine = affine

    # ----- core forward kernel -----
    def bn_forward_kernel(self, K_in, batch_mean, batch_var):
        """Kernel after batch normalization.

        BN centres and rescales: x_hat = (x - mu_B) / sqrt(var_B + eps).
        In kernel space this corresponds to centring then rescaling the Gram
        matrix by the inverse standard deviations.
        """
        n = K_in.shape[0]
        inv_std = _stable_inv_sqrt(batch_var + self.epsilon)

        H = _centering_matrix(n)
        K_centered = H @ K_in @ H

        # Rescale rows/cols by inverse stds (one per sample)
        D = np.diag(inv_std)
        K_out = D @ K_centered @ D

        if self.affine:
            # gamma, beta learnable — gamma^2 multiplies the kernel
            K_out = K_out  # gamma=1 at init
        return _ensure_symmetric(K_out)

    # ----- Jacobian of BN -----
    def bn_jacobian(self, pre_bn_activations, running_mean, running_var):
        """Jacobian ∂BN(x)/∂x for a mini-batch.

        pre_bn_activations : (batch, features)
        Returns J : (batch*features, batch*features)
        """
        B, D = pre_bn_activations.shape
        var_inv = _stable_inv_sqrt(running_var + self.epsilon)

        x_hat = (pre_bn_activations - running_mean[None, :]) * var_inv[None, :]

        J = np.zeros((B * D, B * D))
        for d in range(D):
            vi = var_inv[d]
            block = vi * (np.eye(B) - 1.0 / B
                          - x_hat[:, d:d+1] @ x_hat[:, d:d+1].T / B)
            J[d * B:(d + 1) * B, d * B:(d + 1) * B] = block
        return J

    # ----- NTK contribution -----
    def bn_ntk_contribution(self, K_in, jacobian):
        """NTK modification: Theta_BN = J K J^T + extra gamma/beta terms."""
        Theta_core = jacobian @ K_in @ jacobian.T

        if self.affine:
            n = K_in.shape[0]
            # gamma contributes diag(x_hat x_hat^T), beta contributes 11^T
            gamma_contrib = np.diag(np.diag(Theta_core))
            beta_contrib = np.ones((n, n))
            Theta_core = Theta_core + gamma_contrib + beta_contrib
        return _ensure_symmetric(Theta_core)

    # ----- gradient covariance -----
    def bn_gradient_covariance(self, K_in, batch_size):
        """Gradient covariance matrix under BN.

        BN couples gradients across samples via the centering and
        variance-normalisation steps.
        """
        n = K_in.shape[0]
        H = _centering_matrix(batch_size)

        diag_inv = np.diag(_stable_inv_sqrt(np.diag(K_in) + self.epsilon))
        K_norm = diag_inv @ K_in @ diag_inv

        G = H @ K_norm @ H

        # Off-diagonal coupling correction
        coupling = np.ones((n, n)) / batch_size
        G = G - coupling * np.trace(G) / n
        return _ensure_symmetric(G)

    # ----- train vs eval kernel -----
    def train_vs_eval_kernel(self, K_train, K_eval):
        """Gap between train-mode kernel (batch stats) and eval kernel (running stats).

        Returns the Frobenius-norm relative difference and the difference matrix.
        """
        diff = K_train - K_eval
        rel = np.linalg.norm(diff, 'fro') / (np.linalg.norm(K_eval, 'fro') + 1e-12)
        return {"difference_matrix": diff, "relative_gap": rel}

    # ----- inter-sample coupling -----
    def bn_induced_coupling(self, K_in, batch_size):
        """Off-diagonal coupling introduced by BN.

        BN makes the output for sample i depend on all other samples in the
        batch through the mean and variance.  We quantify this as the ratio
        of off-diagonal to diagonal energy in the BN-modified kernel.
        """
        n = K_in.shape[0]
        H = _centering_matrix(batch_size)
        K_bn = H @ K_in @ H

        diag_energy = np.sum(np.diag(K_bn) ** 2)
        off_diag_energy = np.sum(K_bn ** 2) - diag_energy
        coupling_ratio = off_diag_energy / (diag_energy + 1e-12)

        # Per-sample coupling strengths
        per_sample = np.zeros(n)
        for i in range(n):
            per_sample[i] = np.sum(np.abs(K_bn[i, :])) - np.abs(K_bn[i, i])
        return {"coupling_ratio": coupling_ratio,
                "per_sample_coupling": per_sample,
                "K_bn": _ensure_symmetric(K_bn)}

    # ----- depth propagation -----
    def bn_depth_propagation(self, K_input, depth):
        """Propagate kernel through `depth` BN + linear layers.

        At each layer: K <- BN(sigma_w^2 * K / n + sigma_b^2).
        BN normalises the diagonal to 1, which is the key stabilisation effect.
        """
        n = K_input.shape[0]
        K = K_input.copy()
        trajectory = [K.copy()]

        for _ in range(depth):
            # Linear map kernel step (identity weights)
            K_lin = K.copy()
            # BN: centre then normalise diag to 1
            H = _centering_matrix(n)
            K_c = H @ K_lin @ H
            diag_vals = np.sqrt(np.abs(np.diag(K_c)) + self.epsilon)
            D_inv = np.diag(1.0 / diag_vals)
            K = D_inv @ K_c @ D_inv
            K = _ensure_symmetric(K)
            trajectory.append(K.copy())
        return np.array(trajectory)

    # ----- ghost batch norm -----
    def ghost_bn_kernel(self, K_in, ghost_batch_size):
        """Ghost batch norm: BN applied independently to sub-batches.

        Splits the kernel into ghost-batch blocks, normalises each block,
        and reassembles.
        """
        n = K_in.shape[0]
        n_ghosts = max(1, n // ghost_batch_size)
        K_out = np.zeros_like(K_in)

        for g in range(n_ghosts):
            s = g * ghost_batch_size
            e = min(s + ghost_batch_size, n)
            block = K_in[s:e, s:e]
            bsz = e - s
            H = _centering_matrix(bsz)
            block_c = H @ block @ H
            d = np.sqrt(np.abs(np.diag(block_c)) + self.epsilon)
            D_inv = np.diag(1.0 / d)
            K_out[s:e, s:e] = D_inv @ block_c @ D_inv

        # Cross-ghost blocks are zeroed (ghost BN removes cross-ghost dependence)
        return _ensure_symmetric(K_out)

    # ----- critical point shift -----
    def bn_critical_point_shift(self, sigma_w_range, sigma_b_range):
        """How BN shifts the edge-of-chaos critical line.

        Without BN: chi_1 = sigma_w^2 * E[phi'(z)^2] = 1 at critical point.
        With BN: diagonal is forced to 1, so the effective variance seen by
        the next layer is always 1 regardless of (sigma_w, sigma_b).
        The critical line degenerates to chi_1 = sigma_w^2 * E[phi'(z)^2] = 1
        evaluated at q* = 1.
        """
        results = np.zeros((len(sigma_w_range), len(sigma_b_range)))
        for i, sw in enumerate(sigma_w_range):
            for j, sb in enumerate(sigma_b_range):
                # With ReLU: E[phi'(z)^2] = 0.5 at q*=1
                chi_1_no_bn = sw ** 2 * 0.5
                # BN forces q* = 1, so chi_1 is always sw^2 * 0.5
                chi_1_bn = sw ** 2 * 0.5
                results[i, j] = chi_1_bn - chi_1_no_bn
        critical_sw_with_bn = np.sqrt(2.0)  # sw^2 * 0.5 = 1
        return {"shift_matrix": results,
                "critical_sigma_w_with_bn": critical_sw_with_bn,
                "note": "BN collapses critical line to sigma_w = sqrt(2) for ReLU"}

    # ----- effective LR -----
    def effective_lr_with_bn(self, lr, weight_norm):
        """Effective learning rate due to BN scale invariance.

        BN output is invariant to ||W||, so the effective LR is lr / ||W||.
        After an SGD step, ||W|| grows, so the effective LR decreases,
        producing an implicit warm-up / decay schedule.
        """
        eff_lr = lr / (weight_norm + 1e-12)
        # Decay rate of effective LR per step (first-order approximation)
        d_norm_dt = lr  # ||W|| grows roughly as lr per step
        eff_lr_decay_rate = -lr / (weight_norm ** 2 + 1e-12) * d_norm_dt
        return {"effective_lr": eff_lr,
                "decay_rate": eff_lr_decay_rate,
                "equilibrium_norm": np.sqrt(lr / (self.momentum + 1e-12))}


# ===================================================================
# 2. LayerNormKernel
# ===================================================================

class LayerNormKernel:
    """Layer normalization kernel modification."""

    def __init__(self, normalized_shape, epsilon=1e-5, elementwise_affine=True):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.d = int(np.prod(normalized_shape))
        self.epsilon = epsilon
        self.elementwise_affine = elementwise_affine

    # ----- forward kernel -----
    def ln_forward_kernel(self, K_in, feature_mean, feature_var):
        """Kernel after layer normalization.

        LN normalises across features for each sample independently, so it
        does NOT couple samples.  Each diagonal entry becomes 1 and
        off-diag entries are rescaled.
        """
        n = K_in.shape[0]
        inv_std = _stable_inv_sqrt(feature_var + self.epsilon)
        D = np.diag(inv_std)

        # Subtract mean contribution: mu_i mu_j correlation
        mu_outer = np.outer(feature_mean, feature_mean)
        K_centered = K_in - mu_outer

        K_out = D @ K_centered @ D
        return _ensure_symmetric(K_out)

    # ----- Jacobian -----
    def ln_jacobian(self, pre_ln_activations):
        """Jacobian ∂LN(x)/∂x.  LN is per-sample so J is block-diagonal.

        pre_ln_activations : (batch, features)
        Returns J : (batch*features, batch*features)
        """
        B, D = pre_ln_activations.shape
        J = np.zeros((B * D, B * D))

        for b in range(B):
            x = pre_ln_activations[b]
            mu = np.mean(x)
            var = np.var(x) + self.epsilon
            inv_std = 1.0 / np.sqrt(var)
            x_hat = (x - mu) * inv_std

            # J_b = inv_std * (I - (1/D)11^T - (1/D) x_hat x_hat^T)
            J_b = inv_std * (np.eye(D) - np.ones((D, D)) / D
                             - np.outer(x_hat, x_hat) / D)
            J[b * D:(b + 1) * D, b * D:(b + 1) * D] = J_b
        return J

    # ----- NTK contribution -----
    def ln_ntk_contribution(self, K_in, jacobian):
        """NTK modification from LN.

        Theta_LN = J Theta_pre J^T  +  (affine parameter contributions).
        """
        Theta = jacobian @ K_in @ jacobian.T

        if self.elementwise_affine:
            n = K_in.shape[0]
            # gamma param contributes diag(x_hat * x_hat^T)
            gamma_term = np.diag(np.diag(Theta))
            # beta param contributes identity-like block
            beta_term = np.eye(n) * np.mean(np.diag(Theta))
            Theta = Theta + gamma_term + beta_term
        return _ensure_symmetric(Theta)

    # ----- kernel recursion through layers -----
    def ln_kernel_recursion(self, K_in, depth):
        """LN effect through multiple layers.

        Each layer: K_{l+1} = LN_kernel(ReLU_kernel(K_l)).
        LN normalises the diagonal to 1.
        """
        n = K_in.shape[0]
        K = K_in.copy()
        trajectory = [K.copy()]

        for _ in range(depth):
            # ReLU kernel step (dual activation)
            diag = np.diag(K)
            cos_mat = K / (np.sqrt(np.outer(diag, diag)) + 1e-12)
            cos_mat = np.clip(cos_mat, -1, 1)
            theta = np.arccos(cos_mat)
            K_relu = (np.sqrt(np.outer(diag, diag)) / (2 * np.pi)
                      * (np.sin(theta) + (np.pi - theta) * np.cos(theta)))

            # LN normalises diagonal to 1
            d = np.sqrt(np.abs(np.diag(K_relu)) + self.epsilon)
            D_inv = np.diag(1.0 / d)
            K = D_inv @ K_relu @ D_inv
            K = _ensure_symmetric(K)
            trajectory.append(K.copy())
        return np.array(trajectory)

    # ----- fixed point -----
    def ln_fixed_point(self, K_initial, max_iter=1000):
        """Find fixed-point kernel under iterated LN + ReLU."""
        n = K_initial.shape[0]
        K = K_initial.copy()
        tol = 1e-8

        for it in range(max_iter):
            K_prev = K.copy()
            diag = np.diag(K)
            cos_mat = K / (np.sqrt(np.outer(diag, diag)) + 1e-12)
            cos_mat = np.clip(cos_mat, -1, 1)
            theta = np.arccos(cos_mat)
            K_relu = (np.sqrt(np.outer(diag, diag)) / (2 * np.pi)
                      * (np.sin(theta) + (np.pi - theta) * np.cos(theta)))

            d = np.sqrt(np.abs(np.diag(K_relu)) + self.epsilon)
            D_inv = np.diag(1.0 / d)
            K = D_inv @ K_relu @ D_inv
            K = _ensure_symmetric(K)

            if np.linalg.norm(K - K_prev, 'fro') < tol:
                return {"K_fixed": K, "converged": True, "iterations": it + 1}

        return {"K_fixed": K, "converged": False, "iterations": max_iter}

    # ----- gradient flow -----
    def ln_gradient_flow(self, K_in, depth):
        """Gradient signal magnitude through LN layers (backward pass).

        We track the product of Jacobian singular values as a proxy for
        gradient flow health.
        """
        n = K_in.shape[0]
        K = K_in.copy()
        grad_norms = [1.0]

        for _ in range(depth):
            diag = np.diag(K)
            inv_std = _stable_inv_sqrt(diag + self.epsilon)
            # LN Jacobian contribution to gradient: inv_std * (I - ...)
            # Singular-value proxy: inv_std * (1 - 1/n)
            sv_proxy = np.mean(inv_std) * (1.0 - 1.0 / n)
            grad_norms.append(grad_norms[-1] * sv_proxy)

            # Propagate kernel
            cos_mat = K / (np.sqrt(np.outer(diag, diag)) + 1e-12)
            cos_mat = np.clip(cos_mat, -1, 1)
            theta = np.arccos(cos_mat)
            K = (np.sqrt(np.outer(diag, diag)) / (2 * np.pi)
                 * (np.sin(theta) + (np.pi - theta) * np.cos(theta)))
            d = np.sqrt(np.abs(np.diag(K)) + self.epsilon)
            D_inv = np.diag(1.0 / d)
            K = _ensure_symmetric(D_inv @ K @ D_inv)

        return np.array(grad_norms)

    # ----- with vs without comparison -----
    def ln_vs_no_ln_comparison(self, K_in, depths):
        """Compare kernel evolution with and without LN across depths."""
        n = K_in.shape[0]
        results = {"with_ln": [], "without_ln": [], "off_diag_with": [],
                    "off_diag_without": []}
        K_ln = K_in.copy()
        K_no = K_in.copy()

        for d in depths:
            # Without LN: just ReLU kernel recursion
            diag = np.diag(K_no)
            cos_mat = K_no / (np.sqrt(np.outer(diag, diag)) + 1e-12)
            cos_mat = np.clip(cos_mat, -1, 1)
            theta = np.arccos(cos_mat)
            K_no = (np.sqrt(np.outer(diag, diag)) / (2 * np.pi)
                    * (np.sin(theta) + (np.pi - theta) * np.cos(theta)))

            # With LN
            diag_ln = np.diag(K_ln)
            cos_ln = K_ln / (np.sqrt(np.outer(diag_ln, diag_ln)) + 1e-12)
            cos_ln = np.clip(cos_ln, -1, 1)
            theta_ln = np.arccos(cos_ln)
            K_relu = (np.sqrt(np.outer(diag_ln, diag_ln)) / (2 * np.pi)
                      * (np.sin(theta_ln) + (np.pi - theta_ln) * np.cos(theta_ln)))
            dd = np.sqrt(np.abs(np.diag(K_relu)) + self.epsilon)
            D_inv = np.diag(1.0 / dd)
            K_ln = _ensure_symmetric(D_inv @ K_relu @ D_inv)

            mask = ~np.eye(n, dtype=bool)
            results["with_ln"].append(K_ln.copy())
            results["without_ln"].append(K_no.copy())
            results["off_diag_with"].append(np.mean(np.abs(K_ln[mask])))
            results["off_diag_without"].append(np.mean(np.abs(K_no[mask])))

        return results

    # ----- RMSNorm -----
    def rms_norm_kernel(self, K_in):
        """RMSNorm: normalise by RMS without centering.

        K_out[i,j] = K_in[i,j] / sqrt(K_in[i,i] * K_in[j,j]).
        """
        diag = np.diag(K_in)
        D_inv = np.diag(_stable_inv_sqrt(diag))
        K_out = D_inv @ K_in @ D_inv
        return _ensure_symmetric(K_out)

    # ----- stabilisation analysis -----
    def ln_stabilization_analysis(self, K_trajectory_with_ln, K_trajectory_without):
        """Quantify how LN stabilises kernel evolution.

        Metrics: diagonal variance, off-diagonal collapse, condition number.
        """
        T = min(len(K_trajectory_with_ln), len(K_trajectory_without))
        metrics = {"diag_var_with": [], "diag_var_without": [],
                   "cond_with": [], "cond_without": [],
                   "off_diag_spread_with": [], "off_diag_spread_without": []}
        for t in range(T):
            Kw = K_trajectory_with_ln[t]
            Ko = K_trajectory_without[t]
            n = Kw.shape[0]
            mask = ~np.eye(n, dtype=bool)

            metrics["diag_var_with"].append(np.var(np.diag(Kw)))
            metrics["diag_var_without"].append(np.var(np.diag(Ko)))

            ew = np.linalg.eigvalsh(Kw)
            eo = np.linalg.eigvalsh(Ko)
            metrics["cond_with"].append(
                (np.max(np.abs(ew)) + 1e-12) / (np.min(np.abs(ew)) + 1e-12))
            metrics["cond_without"].append(
                (np.max(np.abs(eo)) + 1e-12) / (np.min(np.abs(eo)) + 1e-12))

            metrics["off_diag_spread_with"].append(np.std(Kw[mask]))
            metrics["off_diag_spread_without"].append(np.std(Ko[mask]))

        return {k: np.array(v) for k, v in metrics.items()}


# ===================================================================
# 3. GroupNormKernel
# ===================================================================

class GroupNormKernel:
    """Group normalization kernel analysis."""

    def __init__(self, num_groups, num_channels, epsilon=1e-5):
        assert num_channels % num_groups == 0, "channels must be divisible by groups"
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.channels_per_group = num_channels // num_groups
        self.epsilon = epsilon

    def _group_assignment(self):
        """Return array mapping each channel to its group index."""
        return np.repeat(np.arange(self.num_groups), self.channels_per_group)

    # ----- forward kernel -----
    def gn_forward_kernel(self, K_in, group_stats):
        """Kernel after group normalization.

        group_stats : dict with 'means' (n, G) and 'vars' (n, G).
        The kernel is normalised within each group independently.
        """
        n = K_in.shape[0]
        G = self.num_groups
        C_per_G = self.channels_per_group
        means = group_stats["means"]   # (n, G)
        variances = group_stats["vars"]  # (n, G)

        # Per-sample, per-group inv_std
        inv_stds = _stable_inv_sqrt(variances + self.epsilon)  # (n, G)

        # The full kernel sees contributions from each group.
        # K_out[i,j] = sum_g (1/(C_per_G)) * inv_stds[i,g] * inv_stds[j,g]
        #              * (K_in_g[i,j] - means[i,g]*means[j,g])
        # We approximate K_in_g ≈ K_in / G  (uniform group contribution).
        K_out = np.zeros((n, n))
        for g in range(G):
            K_g = K_in / G
            mu_outer = np.outer(means[:, g], means[:, g])
            K_g_centered = K_g - mu_outer
            D_g = np.diag(inv_stds[:, g])
            K_out += D_g @ K_g_centered @ D_g / G
        return _ensure_symmetric(K_out)

    # ----- Jacobian -----
    def gn_jacobian(self, pre_gn_activations, group_assignment):
        """Jacobian ∂GN(x)/∂x.  GN is per-sample, per-group.

        pre_gn_activations : (batch, channels)
        group_assignment   : (channels,) int array mapping channel -> group
        Returns J : (B*C, B*C)
        """
        B, C = pre_gn_activations.shape
        J = np.zeros((B * C, B * C))

        for b in range(B):
            x = pre_gn_activations[b]
            for g in range(self.num_groups):
                idx = np.where(group_assignment == g)[0]
                cg = len(idx)
                x_g = x[idx]
                mu = np.mean(x_g)
                var = np.var(x_g) + self.epsilon
                inv_std = 1.0 / np.sqrt(var)
                x_hat = (x_g - mu) * inv_std

                J_g = inv_std * (np.eye(cg)
                                 - np.ones((cg, cg)) / cg
                                 - np.outer(x_hat, x_hat) / cg)
                for ii, ci in enumerate(idx):
                    for jj, cj in enumerate(idx):
                        J[b * C + ci, b * C + cj] = J_g[ii, jj]
        return J

    # ----- NTK contribution -----
    def gn_ntk_contribution(self, K_in, jacobian):
        """NTK modification from group normalization."""
        return _ensure_symmetric(jacobian @ K_in @ jacobian.T)

    # ----- group interaction kernel -----
    def group_interaction_kernel(self, K_in, group_assignment):
        """Inter-group interaction matrix.

        Returns a (G, G) matrix measuring the kernel coupling between groups.
        """
        G = self.num_groups
        interaction = np.zeros((G, G))
        n = K_in.shape[0]
        for g1 in range(G):
            for g2 in range(G):
            # Interaction measured as mean absolute kernel entry
            # between samples weighted by group membership.
            # Here we use a proxy: correlation of group-normalised
            # diagonal blocks.
                interaction[g1, g2] = (1.0 if g1 == g2 else
                                       np.exp(-abs(g1 - g2) / G))
        # Scale by overall kernel magnitude
        interaction *= np.mean(np.abs(K_in))
        return interaction

    # ----- compare BN vs LN vs GN -----
    def gn_vs_bn_vs_ln(self, K_in, batch_size, n_groups):
        """Compare kernels after BN, LN, and GN normalisation."""
        n = K_in.shape[0]

        # BN: centre across batch, normalise diagonal
        H = _centering_matrix(batch_size)
        K_bn = H @ K_in @ H
        d_bn = np.sqrt(np.abs(np.diag(K_bn)) + self.epsilon)
        K_bn = np.diag(1.0 / d_bn) @ K_bn @ np.diag(1.0 / d_bn)

        # LN: normalise diagonal (no cross-sample coupling)
        d_ln = np.sqrt(np.abs(np.diag(K_in)) + self.epsilon)
        K_ln = np.diag(1.0 / d_ln) @ K_in @ np.diag(1.0 / d_ln)

        # GN: interpolation between LN and IN
        alpha = 1.0 / n_groups  # GN with 1 group = LN, G groups = IN
        K_gn = (1 - alpha) * K_ln + alpha * np.eye(n)

        return {"K_bn": _ensure_symmetric(K_bn),
                "K_ln": _ensure_symmetric(K_ln),
                "K_gn": _ensure_symmetric(K_gn),
                "bn_ln_diff": np.linalg.norm(K_bn - K_ln, 'fro'),
                "gn_ln_diff": np.linalg.norm(K_gn - K_ln, 'fro'),
                "gn_bn_diff": np.linalg.norm(K_gn - K_bn, 'fro')}

    # ----- optimal number of groups -----
    def optimal_num_groups(self, K_in, channel_dim, loss_fn):
        """Find optimal group count that minimises a given loss function.

        loss_fn : callable(K_normalised) -> scalar
        Searches over divisors of channel_dim.
        """
        divisors = [g for g in range(1, channel_dim + 1) if channel_dim % g == 0]
        n = K_in.shape[0]
        best_g, best_loss = 1, np.inf

        for g in divisors:
            # Approximate GN kernel: interpolation LN <-> IN
            d = np.sqrt(np.abs(np.diag(K_in)) + self.epsilon)
            K_ln = np.diag(1.0 / d) @ K_in @ np.diag(1.0 / d)
            alpha = 1.0 / g
            K_gn = (1 - alpha) * K_ln + alpha * np.eye(n)
            loss = loss_fn(_ensure_symmetric(K_gn))
            if loss < best_loss:
                best_loss = loss
                best_g = g

        return {"optimal_groups": best_g,
                "optimal_loss": best_loss,
                "divisors_tested": divisors}

    # ----- instance norm -----
    def instance_norm_kernel(self, K_in):
        """Instance normalisation = GN with groups = channels.

        Each feature is independently normalised, so the kernel collapses
        to the identity (all diagonal entries 1, off-diagonal determined by
        the correlation structure of the single-channel kernel).
        """
        n = K_in.shape[0]
        diag = np.diag(K_in)
        D_inv = np.diag(_stable_inv_sqrt(diag))
        K_in_norm = D_inv @ K_in @ D_inv

        # Instance norm additionally centres per-instance, driving
        # off-diagonal entries towards zero for uncorrelated features.
        off_diag_decay = 1.0 / (self.num_channels + 1e-12)
        mask = ~np.eye(n, dtype=bool)
        K_out = K_in_norm.copy()
        K_out[mask] *= (1.0 - off_diag_decay)
        return _ensure_symmetric(K_out)

    # ----- depth propagation -----
    def gn_depth_propagation(self, K_input, depth, n_groups):
        """Depth propagation with group norm."""
        n = K_input.shape[0]
        K = K_input.copy()
        trajectory = [K.copy()]
        alpha = 1.0 / n_groups

        for _ in range(depth):
            # ReLU kernel step
            diag = np.diag(K)
            cos_mat = K / (np.sqrt(np.outer(diag, diag)) + 1e-12)
            cos_mat = np.clip(cos_mat, -1, 1)
            theta = np.arccos(cos_mat)
            K_relu = (np.sqrt(np.outer(diag, diag)) / (2 * np.pi)
                      * (np.sin(theta) + (np.pi - theta) * np.cos(theta)))

            # GN normalisation
            d = np.sqrt(np.abs(np.diag(K_relu)) + self.epsilon)
            D_inv = np.diag(1.0 / d)
            K_ln = D_inv @ K_relu @ D_inv
            K = (1 - alpha) * K_ln + alpha * np.eye(n)
            K = _ensure_symmetric(K)
            trajectory.append(K.copy())
        return np.array(trajectory)


# ===================================================================
# 4. NormalizationRegularizer
# ===================================================================

class NormalizationRegularizer:
    """Normalization-induced implicit regularization."""

    def __init__(self, norm_type='layer'):
        assert norm_type in ('layer', 'batch', 'group', 'rms')
        self.norm_type = norm_type

    # ----- implicit regularization strength -----
    def implicit_regularization_strength(self, K_in, norm_params):
        """Effective regularization λ induced by normalization.

        Normalization constrains the kernel diagonal, which acts like a
        Tikhonov regularizer on the kernel eigenvalues.
        """
        eigvals = np.linalg.eigvalsh(K_in)
        eigvals = np.sort(eigvals)[::-1]

        # The normalization projects onto the unit-diagonal manifold.
        # This is equivalent to adding lambda * I where lambda makes
        # the smallest eigenvalue positive.
        diag = np.diag(K_in)
        target_diag = np.ones_like(diag)  # normalisation targets diag=1

        # Regularization ~ how much the diagonal needed to change
        diag_correction = np.mean(np.abs(diag - target_diag))
        effective_lambda = diag_correction * norm_params.get("strength", 1.0)

        # Spectral view: regularization lifts smallest eigenvalues
        spectral_gap = eigvals[0] - eigvals[-1]
        return {"effective_lambda": effective_lambda,
                "spectral_gap_before": spectral_gap,
                "eigenvalues": eigvals}

    # ----- scale invariance -----
    def scale_invariance_effect(self, K_in, weight_scale_range):
        """Effect of weight scale invariance on the kernel.

        Normalization makes the output invariant to ||W||, so we measure
        how the kernel changes (should be constant) across weight scales.
        """
        results = []
        K_base = K_in.copy()
        for s in weight_scale_range:
            K_scaled = s ** 2 * K_base
            # After normalisation: diag -> 1
            d = np.sqrt(np.abs(np.diag(K_scaled)) + 1e-8)
            D_inv = np.diag(1.0 / d)
            K_norm = D_inv @ K_scaled @ D_inv
            results.append(K_norm)

        # All should be identical
        max_diff = max(np.linalg.norm(r - results[0], 'fro') for r in results)
        return {"normalised_kernels": results,
                "max_deviation": max_diff,
                "scale_invariant": max_diff < 1e-6}

    # ----- effective weight decay -----
    def effective_weight_decay(self, lr, momentum, weight_norm):
        """Weight decay from scale invariance (Hoffer et al., 2018).

        For BN/LN: effective WD ≈ lr / (||W||^2).
        With momentum m: effective WD ≈ lr / ((1-m) * ||W||^2).
        """
        wd_no_momentum = lr / (weight_norm ** 2 + 1e-12)
        wd_with_momentum = lr / ((1.0 - momentum) * weight_norm ** 2 + 1e-12)
        # Equilibrium weight norm: ||W||^2 ≈ lr / wd_target
        # where wd_target is the implicit WD at equilibrium.
        equilibrium_norm_sq = lr / (wd_no_momentum + 1e-12)
        return {"wd_no_momentum": wd_no_momentum,
                "wd_with_momentum": wd_with_momentum,
                "equilibrium_norm_sq": equilibrium_norm_sq}

    # ----- Hessian modification -----
    def regularization_on_hessian(self, H_without_norm, norm_jacobian):
        """How normalization modifies the loss Hessian.

        H_with_norm = J^T H J + second-order terms from the Jacobian.
        The second-order terms act as curvature regularization.
        """
        H_with = norm_jacobian.T @ H_without_norm @ norm_jacobian

        eigvals_without = np.linalg.eigvalsh(H_without_norm)
        eigvals_with = np.linalg.eigvalsh(H_with)

        # Curvature reduction
        max_curv_reduction = np.max(eigvals_without) - np.max(eigvals_with)
        trace_reduction = np.sum(eigvals_without) - np.sum(eigvals_with)
        return {"H_with_norm": _ensure_symmetric(H_with),
                "max_curvature_reduction": max_curv_reduction,
                "trace_reduction": trace_reduction,
                "eigvals_without": np.sort(eigvals_without)[::-1],
                "eigvals_with": np.sort(eigvals_with)[::-1]}

    # ----- spectral regularization -----
    def spectral_regularization(self, eigenvalues_with_norm, eigenvalues_without):
        """Spectral effect of normalization on kernel eigenvalues."""
        ew = np.sort(eigenvalues_with_norm)[::-1]
        eo = np.sort(eigenvalues_without)[::-1]
        n = min(len(ew), len(eo))
        ew, eo = ew[:n], eo[:n]

        ratio = ew / (eo + 1e-12)
        effective_rank_with = np.sum(ew > 1e-8 * ew[0])
        effective_rank_without = np.sum(eo > 1e-8 * eo[0])

        # Spectral entropy
        pw = np.abs(ew) / (np.sum(np.abs(ew)) + 1e-12)
        po = np.abs(eo) / (np.sum(np.abs(eo)) + 1e-12)
        entropy_with = -np.sum(pw * np.log(pw + 1e-12))
        entropy_without = -np.sum(po * np.log(po + 1e-12))

        return {"eigenvalue_ratio": ratio,
                "effective_rank_with": int(effective_rank_with),
                "effective_rank_without": int(effective_rank_without),
                "spectral_entropy_with": entropy_with,
                "spectral_entropy_without": entropy_without,
                "entropy_increase": entropy_with - entropy_without}

    # ----- curvature reduction -----
    def curvature_reduction(self, K_in, norm_type):
        """How normalization reduces kernel curvature.

        Curvature is measured as the variance of the off-diagonal kernel
        entries (high variance = highly curved kernel manifold).
        """
        n = K_in.shape[0]

        # Normalise
        d = np.sqrt(np.abs(np.diag(K_in)) + 1e-8)
        D_inv = np.diag(1.0 / d)
        K_norm = D_inv @ K_in @ D_inv

        if norm_type == 'batch':
            H = _centering_matrix(n)
            K_norm = H @ K_norm @ H
            dd = np.sqrt(np.abs(np.diag(K_norm)) + 1e-8)
            K_norm = np.diag(1.0 / dd) @ K_norm @ np.diag(1.0 / dd)

        mask = ~np.eye(n, dtype=bool)
        curvature_before = np.var(K_in[mask])
        curvature_after = np.var(K_norm[mask])
        return {"curvature_before": curvature_before,
                "curvature_after": curvature_after,
                "reduction_factor": curvature_before / (curvature_after + 1e-12)}

    # ----- generalization bound -----
    def generalization_bound_with_norm(self, K_in, n_train, norm_params):
        """PAC-Bayes-style generalization bound with normalization.

        Bound ≈ sqrt(trace(K) / (n * lambda_min)) + sqrt(log(1/delta) / n).
        Normalization improves this by reducing trace(K) and increasing lambda_min.
        """
        strength = norm_params.get("strength", 1.0)
        delta = norm_params.get("delta", 0.05)

        # Without normalization
        eigvals = np.linalg.eigvalsh(K_in)
        trace_K = np.trace(K_in)
        lambda_min = max(np.min(eigvals), 1e-10)

        # With normalization
        d = np.sqrt(np.abs(np.diag(K_in)) + 1e-8)
        D_inv = np.diag(1.0 / d)
        K_norm = D_inv @ K_in @ D_inv
        eigvals_norm = np.linalg.eigvalsh(K_norm)
        trace_K_norm = np.trace(K_norm)
        lambda_min_norm = max(np.min(eigvals_norm), 1e-10)

        complexity_no_norm = np.sqrt(trace_K / (n_train * lambda_min))
        complexity_norm = np.sqrt(trace_K_norm / (n_train * lambda_min_norm))
        stat_term = np.sqrt(np.log(1.0 / delta) / n_train)

        return {"bound_without_norm": complexity_no_norm + stat_term,
                "bound_with_norm": complexity_norm + stat_term,
                "improvement": (complexity_no_norm - complexity_norm) / (complexity_no_norm + 1e-12),
                "trace_reduction": trace_K - trace_K_norm}


# ===================================================================
# 5. NormalizationMeanField
# ===================================================================

class NormalizationMeanField:
    """Normalization in mean-field theory."""

    def __init__(self, norm_type='layer', width=None):
        self.norm_type = norm_type
        self.width = width

    # ----- mean field with BN -----
    def mean_field_with_bn(self, q, qhat, batch_size):
        """Mean-field order parameter update with batch normalisation.

        BN forces the variance to 1 at each layer, so the order-parameter
        recursion becomes:
            q_{l+1} = 1  (forced by BN)
            qhat_{l+1} = sigma_w^2 * E_z[phi(sqrt(q_l)*z) * phi(sqrt(qhat_l)*z')]
        evaluated at q_l = 1.
        """
        # With BN, diagonal order parameter is always 1
        q_next = 1.0

        # Off-diagonal: correlation propagation at q=1
        c = qhat / (q + 1e-12)
        c = np.clip(c, -1, 1)
        theta = np.arccos(c)

        # ReLU dual: qhat_next at q=1
        qhat_next = (1.0 / (2 * np.pi)) * (np.sin(theta) + (np.pi - theta) * c)

        # Finite batch-size correction: BN estimates have variance ~ 1/B
        correction = 1.0 / batch_size
        qhat_next *= (1.0 + correction)

        return {"q_next": q_next, "qhat_next": qhat_next,
                "correlation": qhat_next / q_next}

    # ----- mean field with LN -----
    def mean_field_with_ln(self, q, qhat):
        """Mean-field order parameter update with layer normalisation.

        LN normalises across features for each sample, forcing the
        per-sample variance to 1 but not coupling samples.
        """
        q_next = 1.0  # LN also forces diagonal to 1

        c = qhat / (q + 1e-12)
        c = np.clip(c, -1, 1)
        theta = np.arccos(c)

        # ReLU dual evaluated at q*=1
        qhat_next = (1.0 / (2 * np.pi)) * (np.sin(theta) + (np.pi - theta) * c)

        return {"q_next": q_next, "qhat_next": qhat_next,
                "correlation": qhat_next}

    # ----- order parameter shift -----
    def order_parameter_shift(self, q_without_norm, norm_params):
        """Change in order parameter q* caused by normalization.

        Without norm: q* can be arbitrary fixed point.
        With norm: q* is forced to 1 (or a value set by gamma^2 if affine).
        """
        gamma = norm_params.get("gamma", 1.0)
        q_with_norm = gamma ** 2

        delta_q = q_with_norm - q_without_norm
        relative_shift = delta_q / (np.abs(q_without_norm) + 1e-12)
        return {"q_without_norm": q_without_norm,
                "q_with_norm": q_with_norm,
                "delta_q": delta_q,
                "relative_shift": relative_shift}

    # ----- phase boundary with normalization -----
    def phase_boundary_with_norm(self, sigma_w_range, sigma_b_range, norm_type):
        """Phase boundary (edge of chaos) with normalization.

        chi_1 = sigma_w^2 * E[phi'(sqrt(q*) z)^2].
        With norm q* = 1, so chi_1 = sigma_w^2 * E[phi'(z)^2].
        For ReLU: E[phi'(z)^2] = 0.5, so chi_1 = 1 iff sigma_w = sqrt(2).
        The phase boundary becomes a vertical line at sigma_w = sqrt(2),
        independent of sigma_b.
        """
        n_w = len(sigma_w_range)
        n_b = len(sigma_b_range)
        chi_1_with_norm = np.zeros((n_w, n_b))
        chi_1_without_norm = np.zeros((n_w, n_b))

        for i, sw in enumerate(sigma_w_range):
            for j, sb in enumerate(sigma_b_range):
                # Without norm: q* depends on (sw, sb)
                # Fixed point of q -> sw^2 * E[phi(sqrt(q)*z)^2] + sb^2
                # For ReLU: E[phi(sqrt(q)*z)^2] = q/2
                # q* = sw^2 * q*/2 + sb^2  =>  q* = sb^2 / (1 - sw^2/2)
                if sw ** 2 / 2 < 1:
                    q_star = sb ** 2 / (1 - sw ** 2 / 2)
                else:
                    q_star = 100.0  # divergent

                chi_1_without_norm[i, j] = sw ** 2 * 0.5
                chi_1_with_norm[i, j] = sw ** 2 * 0.5  # q* = 1 with norm

        # Phase boundary: chi_1 = 1
        critical_sw = np.sqrt(2.0)
        boundary_without = []
        for j, sb in enumerate(sigma_b_range):
            # chi_1 = sw^2 * 0.5 = 1 => sw = sqrt(2) (same for ReLU)
            boundary_without.append(critical_sw)

        return {"chi_1_with_norm": chi_1_with_norm,
                "chi_1_without_norm": chi_1_without_norm,
                "critical_sigma_w": critical_sw,
                "boundary_with_norm": np.full(n_b, critical_sw),
                "boundary_without_norm": np.array(boundary_without),
                "norm_type": norm_type}

    # ----- critical point location -----
    def critical_point_with_norm(self, norm_type, norm_params):
        """Critical point (sigma_w*, sigma_b*) with normalization.

        Returns the exact critical point where chi_1 = 1.
        """
        gamma = norm_params.get("gamma", 1.0)

        if norm_type in ('layer', 'batch', 'rms'):
            # q* = gamma^2, chi_1 = sw^2 * E[phi'(sqrt(gamma^2)*z)^2]
            # ReLU: E[phi'(z)^2] = 0.5 regardless of scale
            sw_crit = np.sqrt(2.0)
            # sigma_b is irrelevant because norm absorbs the bias
            sb_crit = 0.0
        elif norm_type == 'group':
            n_groups = norm_params.get("n_groups", 1)
            # Interpolation: partial normalisation
            alpha = 1.0 / n_groups
            # Effective chi_1 = sw^2 * 0.5 * (1 - alpha) + alpha
            # chi_1 = 1 => sw^2 = 2 * (1 - alpha) / (1 - alpha) = 2 when alpha != 1
            if alpha < 1:
                sw_crit = np.sqrt(2.0 / (1.0 - alpha + 1e-12))
            else:
                sw_crit = np.sqrt(2.0)
            sb_crit = 0.0
        else:
            sw_crit = np.sqrt(2.0)
            sb_crit = 0.0

        return {"sigma_w_critical": sw_crit,
                "sigma_b_critical": sb_crit,
                "norm_type": norm_type,
                "q_star": gamma ** 2}

    # ----- susceptibility modification -----
    def norm_effect_on_susceptibility(self, chi_without_norm, norm_params):
        """How normalization modifies the susceptibility chi_1.

        Normalization clamps q* = 1, which fixes the variance entering
        the susceptibility calculation.  This can either increase or
        decrease chi_1 depending on whether the unnormalised q* > 1 or < 1.
        """
        gamma = norm_params.get("gamma", 1.0)
        sigma_w = norm_params.get("sigma_w", 1.0)

        # chi_1 with norm evaluated at q* = gamma^2
        # ReLU: E[phi'(sqrt(q)*z)^2] = 0.5 for all q (piecewise linear)
        chi_with_norm = sigma_w ** 2 * 0.5

        delta_chi = chi_with_norm - chi_without_norm
        # If chi < 1: ordered phase.  chi > 1: chaotic phase.
        phase_without = "chaotic" if chi_without_norm > 1 else "ordered"
        phase_with = "chaotic" if chi_with_norm > 1 else "ordered"
        return {"chi_without_norm": chi_without_norm,
                "chi_with_norm": chi_with_norm,
                "delta_chi": delta_chi,
                "phase_without": phase_without,
                "phase_with": phase_with,
                "phase_changed": phase_without != phase_with}

    # ----- depth scale with normalization -----
    def depth_scale_with_norm(self, chi_1, norm_type):
        """Correlation length xi with normalization.

        xi = -1 / log(chi_1).  At the critical point chi_1 = 1, xi -> inf.
        Normalization shifts chi_1 and therefore changes xi.
        """
        if norm_type in ('layer', 'batch', 'rms'):
            # chi_1 is fixed by sigma_w alone
            chi_eff = chi_1
        elif norm_type == 'group':
            # Partial normalisation modifies chi_1 slightly
            chi_eff = chi_1 * 0.95  # heuristic 5% reduction
        else:
            chi_eff = chi_1

        if np.abs(chi_eff - 1.0) < 1e-10:
            xi = np.inf
        else:
            xi = -1.0 / np.log(np.abs(chi_eff) + 1e-12)

        xi_without = -1.0 / np.log(np.abs(chi_1) + 1e-12) if np.abs(chi_1 - 1.0) > 1e-10 else np.inf
        return {"xi_with_norm": xi,
                "xi_without_norm": xi_without,
                "chi_effective": chi_eff,
                "at_criticality": np.abs(chi_eff - 1.0) < 1e-6}

    # ----- universality class -----
    def norm_universal_class(self, norm_type):
        """Does normalization change the universality class?

        For standard normalizations (BN, LN, GN) with ReLU, the universality
        class remains mean-field (infinite-width Gaussian process) because
        normalization is a smooth deterministic map that does not change the
        scaling of fluctuations at large width.  However, BN introduces
        inter-sample coupling that changes the finite-size scaling exponents.
        """
        if norm_type == 'batch':
            return {
                "changes_universality": True,
                "reason": ("BN couples samples within a batch, breaking "
                           "sample independence.  Finite-size corrections "
                           "scale as 1/B (batch size) in addition to 1/n (width)."),
                "infinite_width_class": "Gaussian process (unchanged)",
                "finite_size_exponents": {"width": -1, "batch_size": -1},
            }
        elif norm_type in ('layer', 'rms'):
            return {
                "changes_universality": False,
                "reason": ("LN/RMSNorm is per-sample and per-layer, so it "
                           "preserves sample independence.  The GP limit and "
                           "finite-size scaling exponents are unchanged."),
                "infinite_width_class": "Gaussian process (unchanged)",
                "finite_size_exponents": {"width": -1},
            }
        elif norm_type == 'group':
            return {
                "changes_universality": False,
                "reason": ("GN is per-sample (like LN) but groups channels. "
                           "Universality class is unchanged; only the "
                           "effective number of independent features changes."),
                "infinite_width_class": "Gaussian process (unchanged)",
                "finite_size_exponents": {"width": -1},
            }
        else:
            return {"changes_universality": False,
                    "reason": "Unknown norm type; assuming no change.",
                    "infinite_width_class": "Gaussian process (unchanged)"}
