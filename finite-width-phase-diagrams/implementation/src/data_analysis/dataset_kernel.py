"""
Dataset-dependent kernel analysis for neural tangent kernel eigenspectrum,
kernel-target alignment, generalization bounds, effective dimension,
and spectral bias.
"""

import numpy as np
from scipy import linalg
from scipy.stats import gaussian_kde, pearsonr
from scipy.optimize import minimize_scalar, brentq
from scipy.special import entr


class DataDependentNTK:
    """Data-dependent NTK eigenspectrum analysis."""

    def __init__(self, kernel_fn=None):
        self.kernel_fn = kernel_fn

    def compute_empirical_ntk(self, params, data_x, jacobian_fn):
        """Compute empirical NTK: Theta = J @ J^T from the Jacobian.

        Parameters
        ----------
        params : array-like
            Network parameters (flattened or structured).
        data_x : ndarray of shape (n, d)
            Input data points.
        jacobian_fn : callable
            Function (params, x) -> Jacobian matrix of shape (n_outputs, n_params).

        Returns
        -------
        ntk : ndarray of shape (n, n)
            Empirical neural tangent kernel matrix.
        """
        n = data_x.shape[0]
        jacobians = []
        for i in range(n):
            x_i = data_x[i:i+1]
            jac_i = jacobian_fn(params, x_i)
            jacobians.append(jac_i.reshape(1, -1))
        J = np.vstack(jacobians)  # (n, n_params)
        ntk = J @ J.T
        return ntk

    def ntk_eigenspectrum(self, kernel_matrix, n_eigenvalues=None):
        """Compute eigenvalues of the NTK matrix.

        Parameters
        ----------
        kernel_matrix : ndarray of shape (n, n)
            Symmetric positive semi-definite kernel matrix.
        n_eigenvalues : int or None
            Number of top eigenvalues to return. None returns all.

        Returns
        -------
        eigenvalues : ndarray
            Eigenvalues in descending order.
        eigenvectors : ndarray
            Corresponding eigenvectors as columns.
        """
        K_sym = 0.5 * (kernel_matrix + kernel_matrix.T)
        if n_eigenvalues is not None and n_eigenvalues < K_sym.shape[0]:
            n = K_sym.shape[0]
            eigenvalues, eigenvectors = linalg.eigh(
                K_sym, subset_by_index=[n - n_eigenvalues, n - 1]
            )
        else:
            eigenvalues, eigenvectors = linalg.eigh(K_sym)
        idx = np.argsort(eigenvalues)[::-1]
        return eigenvalues[idx], eigenvectors[:, idx]

    def spectral_density(self, eigenvalues, n_bins=100):
        """Compute empirical spectral density of eigenvalues.

        Parameters
        ----------
        eigenvalues : ndarray
            Array of eigenvalues.
        n_bins : int
            Number of bins for histogram / KDE evaluation grid.

        Returns
        -------
        grid : ndarray
            Evaluation points.
        density : ndarray
            Estimated density at each grid point.
        """
        eigenvalues = np.asarray(eigenvalues, dtype=float)
        pos = eigenvalues[eigenvalues > 0]
        if len(pos) < 2:
            grid = np.linspace(0, 1, n_bins)
            return grid, np.zeros(n_bins)
        kde = gaussian_kde(pos, bw_method='silverman')
        lo = max(pos.min() * 0.8, 1e-15)
        hi = pos.max() * 1.2
        grid = np.linspace(lo, hi, n_bins)
        density = kde(grid)
        return grid, density

    def marcenko_pastur_fit(self, eigenvalues, gamma):
        """Fit eigenvalue distribution to Marchenko-Pastur law.

        Parameters
        ----------
        eigenvalues : ndarray
            Empirical eigenvalues.
        gamma : float
            Aspect ratio p/n (number of parameters / number of samples).

        Returns
        -------
        result : dict
            'sigma_sq': fitted variance,
            'lambda_minus', 'lambda_plus': MP bulk edges,
            'ks_statistic': KS distance between empirical and MP CDF.
        """
        eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))

        # Estimate sigma^2 from median of bulk eigenvalues
        def mp_edges(sigma_sq):
            lm = sigma_sq * (1 - np.sqrt(gamma)) ** 2
            lp = sigma_sq * (1 + np.sqrt(gamma)) ** 2
            return lm, lp

        def mp_density(x, sigma_sq):
            lm, lp = mp_edges(sigma_sq)
            density = np.zeros_like(x)
            mask = (x >= lm) & (x <= lp)
            density[mask] = (
                np.sqrt((lp - x[mask]) * (x[mask] - lm))
                / (2 * np.pi * gamma * sigma_sq * x[mask])
            )
            return density

        # Fit sigma^2 by matching mean eigenvalue to sigma^2
        sigma_sq_est = np.mean(eigenvalues)

        lm, lp = mp_edges(sigma_sq_est)

        # KS statistic: compare empirical CDF with MP CDF
        grid = np.linspace(max(lm, 1e-15), lp, 500)
        mp_pdf = mp_density(grid, sigma_sq_est)
        dx = grid[1] - grid[0]
        mp_cdf = np.cumsum(mp_pdf) * dx
        mp_cdf = mp_cdf / mp_cdf[-1] if mp_cdf[-1] > 0 else mp_cdf

        empirical_cdf = np.searchsorted(eigenvalues, grid) / len(eigenvalues)
        ks_stat = np.max(np.abs(empirical_cdf - mp_cdf))

        return {
            'sigma_sq': sigma_sq_est,
            'lambda_minus': lm,
            'lambda_plus': lp,
            'ks_statistic': ks_stat,
        }

    def bulk_eigenvalues(self, eigenvalues, threshold='auto'):
        """Return bulk eigenvalues (non-outlier part of spectrum).

        Parameters
        ----------
        eigenvalues : ndarray
        threshold : float or 'auto'
            If 'auto', use median + 3 * MAD to separate bulk from outliers.

        Returns
        -------
        bulk : ndarray
        """
        eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
        if threshold == 'auto':
            med = np.median(eigenvalues)
            mad = np.median(np.abs(eigenvalues - med))
            threshold = med + 3.0 * max(mad, 1e-12)
        return eigenvalues[eigenvalues <= threshold]

    def outlier_eigenvalues(self, eigenvalues, threshold='auto'):
        """Return outlier eigenvalues above threshold.

        Parameters
        ----------
        eigenvalues : ndarray
        threshold : float or 'auto'

        Returns
        -------
        outliers : ndarray
        """
        eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
        if threshold == 'auto':
            med = np.median(eigenvalues)
            mad = np.median(np.abs(eigenvalues - med))
            threshold = med + 3.0 * max(mad, 1e-12)
        return eigenvalues[eigenvalues > threshold]

    def spiked_covariance_model(self, eigenvalues, gamma):
        """Analyze eigenvalues under the spiked covariance model.

        Identifies spikes that emerge above the Marchenko-Pastur bulk edge
        and computes the BBP (Baik-Ben Arous-Peche) phase transition threshold.

        Parameters
        ----------
        eigenvalues : ndarray
        gamma : float
            Aspect ratio p/n.

        Returns
        -------
        result : dict
            'bbp_threshold': critical spike value gamma^{1/2},
            'detected_spikes': eigenvalues above MP+ edge,
            'spike_locations': predicted locations of sample eigenvalues
                               for each population spike via BBP formula,
            'bulk_edge': lambda_+ of MP.
        """
        eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]

        sigma_sq = np.mean(eigenvalues)
        lambda_plus = sigma_sq * (1 + np.sqrt(gamma)) ** 2
        bbp_threshold = sigma_sq * np.sqrt(gamma)

        detected = eigenvalues[eigenvalues > lambda_plus]

        # BBP formula: for population spike l, sample spike -> l + gamma*sigma^4/l
        spike_locations = []
        for l_hat in detected:
            # Invert: estimate population spike from sample spike
            # sample = pop + gamma*sigma^4 / pop  => quadratic in pop
            # pop^2 - l_hat*pop + gamma*sigma^4 = 0
            disc = l_hat ** 2 - 4 * gamma * sigma_sq ** 2
            if disc > 0:
                pop_spike = 0.5 * (l_hat + np.sqrt(disc))
                predicted_sample = pop_spike + gamma * sigma_sq ** 2 / pop_spike
                spike_locations.append(predicted_sample)
            else:
                spike_locations.append(l_hat)

        return {
            'bbp_threshold': bbp_threshold,
            'detected_spikes': detected,
            'spike_locations': np.array(spike_locations),
            'bulk_edge': lambda_plus,
        }

    def eigenvalue_spacing(self, eigenvalues):
        """Compute nearest-neighbor level spacing statistics.

        Parameters
        ----------
        eigenvalues : ndarray

        Returns
        -------
        result : dict
            'spacings': sorted spacings,
            'mean_spacing': mean spacing,
            'spacing_ratio': mean ratio of consecutive spacings (r-statistic),
            'wigner_surmise_fit': KS distance to Wigner surmise (GOE).
        """
        eigs = np.sort(np.asarray(eigenvalues, dtype=float))
        spacings = np.diff(eigs)
        spacings = spacings[spacings > 0]
        if len(spacings) == 0:
            return {'spacings': spacings, 'mean_spacing': 0.0,
                    'spacing_ratio': 0.0, 'wigner_surmise_fit': 1.0}

        mean_s = np.mean(spacings)
        normalized = spacings / mean_s if mean_s > 0 else spacings

        # Spacing ratio (r-statistic)
        if len(spacings) > 1:
            ratios = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(
                spacings[:-1], spacings[1:]
            )
            spacing_ratio = np.mean(ratios)
        else:
            spacing_ratio = 0.0

        # KS distance to Wigner surmise P(s) = (pi*s/2) exp(-pi*s^2/4)
        s_sorted = np.sort(normalized)
        n = len(s_sorted)
        ecdf = np.arange(1, n + 1) / n
        wigner_cdf = 1.0 - np.exp(-np.pi * s_sorted ** 2 / 4.0)
        ks = np.max(np.abs(ecdf - wigner_cdf))

        return {
            'spacings': spacings,
            'mean_spacing': mean_s,
            'spacing_ratio': spacing_ratio,
            'wigner_surmise_fit': ks,
        }

    def kernel_condition_number(self, kernel_matrix):
        """Compute condition number kappa = lambda_max / lambda_min.

        Parameters
        ----------
        kernel_matrix : ndarray of shape (n, n)

        Returns
        -------
        kappa : float
        """
        K_sym = 0.5 * (kernel_matrix + kernel_matrix.T)
        eigs = linalg.eigvalsh(K_sym)
        eigs_pos = eigs[eigs > 0]
        if len(eigs_pos) == 0:
            return np.inf
        return float(eigs_pos[-1] / eigs_pos[0])

    def effective_rank(self, kernel_matrix):
        """Compute effective rank: r_eff = (sum lambda)^2 / sum(lambda^2).

        Parameters
        ----------
        kernel_matrix : ndarray of shape (n, n)

        Returns
        -------
        r_eff : float
        """
        K_sym = 0.5 * (kernel_matrix + kernel_matrix.T)
        eigs = linalg.eigvalsh(K_sym)
        eigs = eigs[eigs > 0]
        if len(eigs) == 0:
            return 0.0
        s1 = np.sum(eigs)
        s2 = np.sum(eigs ** 2)
        if s2 == 0:
            return 0.0
        return float(s1 ** 2 / s2)

    def data_dependent_spectrum_shift(self, ntk_data1, ntk_data2):
        """Quantify the spectrum shift between two NTK matrices on different data.

        Parameters
        ----------
        ntk_data1 : ndarray of shape (n1, n1)
        ntk_data2 : ndarray of shape (n2, n2)

        Returns
        -------
        result : dict
            'eigenvalue_shift': L2 distance between normalized spectra,
            'effective_rank_change': change in effective rank,
            'condition_number_ratio': ratio of condition numbers,
            'spectral_divergence': KL divergence between spectral densities.
        """
        eigs1 = np.sort(linalg.eigvalsh(0.5 * (ntk_data1 + ntk_data1.T)))[::-1]
        eigs2 = np.sort(linalg.eigvalsh(0.5 * (ntk_data2 + ntk_data2.T)))[::-1]

        eigs1_pos = eigs1[eigs1 > 0]
        eigs2_pos = eigs2[eigs2 > 0]

        # Normalize to distributions
        p1 = eigs1_pos / np.sum(eigs1_pos) if np.sum(eigs1_pos) > 0 else eigs1_pos
        p2 = eigs2_pos / np.sum(eigs2_pos) if np.sum(eigs2_pos) > 0 else eigs2_pos

        # Align lengths via interpolation
        n_common = max(len(p1), len(p2))
        p1_interp = np.interp(
            np.linspace(0, 1, n_common), np.linspace(0, 1, len(p1)), p1
        )
        p2_interp = np.interp(
            np.linspace(0, 1, n_common), np.linspace(0, 1, len(p2)), p2
        )

        eigenvalue_shift = float(np.sqrt(np.sum((p1_interp - p2_interp) ** 2)))

        r1 = self.effective_rank(ntk_data1)
        r2 = self.effective_rank(ntk_data2)

        k1 = self.kernel_condition_number(ntk_data1)
        k2 = self.kernel_condition_number(ntk_data2)

        # KL divergence between spectral densities (add epsilon for stability)
        eps = 1e-12
        q1 = p1_interp + eps
        q2 = p2_interp + eps
        q1 /= q1.sum()
        q2 /= q2.sum()
        kl = float(np.sum(q1 * np.log(q1 / q2)))

        return {
            'eigenvalue_shift': eigenvalue_shift,
            'effective_rank_change': r2 - r1,
            'condition_number_ratio': k2 / k1 if k1 > 0 else np.inf,
            'spectral_divergence': kl,
        }


class KernelTargetAlignment:
    """Kernel-target alignment analysis."""

    def __init__(self):
        pass

    def compute_alignment(self, kernel, targets):
        """Compute kernel-target alignment A(K, y) = <K, yy^T> / (||K||_F ||yy^T||_F).

        Parameters
        ----------
        kernel : ndarray of shape (n, n)
        targets : ndarray of shape (n,) or (n, 1)

        Returns
        -------
        alignment : float in [-1, 1]
        """
        y = np.asarray(targets, dtype=float).ravel()
        yyt = np.outer(y, y)
        num = np.sum(kernel * yyt)
        denom = linalg.norm(kernel, 'fro') * linalg.norm(yyt, 'fro')
        if denom == 0:
            return 0.0
        return float(num / denom)

    def centered_alignment(self, kernel, targets):
        """Compute centered kernel-target alignment (HSIC-normalized).

        Centers both the kernel and target kernel in feature space using
        the centering matrix H = I - (1/n) 11^T.

        Parameters
        ----------
        kernel : ndarray of shape (n, n)
        targets : ndarray of shape (n,)

        Returns
        -------
        alignment : float
        """
        n = kernel.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        K_c = H @ kernel @ H
        y = np.asarray(targets, dtype=float).ravel()
        yyt = np.outer(y, y)
        L_c = H @ yyt @ H

        num = np.sum(K_c * L_c)
        denom = linalg.norm(K_c, 'fro') * linalg.norm(L_c, 'fro')
        if denom == 0:
            return 0.0
        return float(num / denom)

    def class_alignment(self, kernel, labels, n_classes):
        """Compute per-class kernel-target alignment.

        Parameters
        ----------
        kernel : ndarray of shape (n, n)
        labels : ndarray of shape (n,), integer class labels
        n_classes : int

        Returns
        -------
        alignments : dict mapping class index to alignment value
        """
        labels = np.asarray(labels).ravel()
        alignments = {}
        for c in range(n_classes):
            binary_targets = (labels == c).astype(float)
            binary_targets = 2.0 * binary_targets - 1.0  # map to {-1, 1}
            alignments[c] = self.compute_alignment(kernel, binary_targets)
        return alignments

    def alignment_vs_depth(self, kernels_at_depths, targets):
        """Compute alignment at each network depth.

        Parameters
        ----------
        kernels_at_depths : list of ndarray
            Kernel matrices computed at each depth.
        targets : ndarray of shape (n,)

        Returns
        -------
        depths : ndarray
        alignments : ndarray
        centered_alignments : ndarray
        """
        depths = np.arange(len(kernels_at_depths))
        alignments = np.array([self.compute_alignment(K, targets)
                               for K in kernels_at_depths])
        centered = np.array([self.centered_alignment(K, targets)
                             for K in kernels_at_depths])
        return depths, alignments, centered

    def alignment_vs_width(self, kernels_at_widths, targets):
        """Compute alignment at each network width.

        Parameters
        ----------
        kernels_at_widths : list of (width, kernel_matrix) tuples
        targets : ndarray of shape (n,)

        Returns
        -------
        widths : ndarray
        alignments : ndarray
        centered_alignments : ndarray
        """
        widths = np.array([w for w, _ in kernels_at_widths])
        alignments = np.array([self.compute_alignment(K, targets)
                               for _, K in kernels_at_widths])
        centered = np.array([self.centered_alignment(K, targets)
                             for _, K in kernels_at_widths])
        return widths, alignments, centered

    def optimal_kernel_for_targets(self, targets):
        """Compute the optimal kernel K* = yy^T for given targets.

        Parameters
        ----------
        targets : ndarray of shape (n,)

        Returns
        -------
        K_star : ndarray of shape (n, n)
        """
        y = np.asarray(targets, dtype=float).ravel()
        return np.outer(y, y)

    def alignment_gap(self, kernel, targets):
        """Compute gap between current and optimal alignment.

        Parameters
        ----------
        kernel : ndarray of shape (n, n)
        targets : ndarray of shape (n,)

        Returns
        -------
        gap : float
            1 - A(K, y), where A is the centered alignment.
        """
        return 1.0 - self.centered_alignment(kernel, targets)

    def alignment_decomposition(self, kernel, targets, eigenvalues, eigenvectors):
        """Decompose alignment into contributions from each eigenmode.

        Parameters
        ----------
        kernel : ndarray of shape (n, n)
        targets : ndarray of shape (n,)
        eigenvalues : ndarray of shape (k,)
        eigenvectors : ndarray of shape (n, k)

        Returns
        -------
        mode_alignments : ndarray of shape (k,)
            Contribution of each eigenmode to alignment.
        cumulative : ndarray of shape (k,)
            Cumulative alignment from top modes.
        """
        y = np.asarray(targets, dtype=float).ravel()
        coeffs = eigenvectors.T @ y  # projection of y onto each eigenvector
        yyt_norm = linalg.norm(np.outer(y, y), 'fro')
        k_norm = linalg.norm(kernel, 'fro')
        denom = k_norm * yyt_norm
        if denom == 0:
            return np.zeros(len(eigenvalues)), np.zeros(len(eigenvalues))

        # Each mode contributes lambda_i * c_i^2 to <K, yy^T>
        mode_contributions = eigenvalues * coeffs ** 2
        mode_alignments = mode_contributions / denom
        cumulative = np.cumsum(mode_alignments)
        return mode_alignments, cumulative

    def alignment_during_training(self, kernel_snapshots, targets):
        """Track kernel-target alignment over training.

        Parameters
        ----------
        kernel_snapshots : list of (step, kernel_matrix) tuples
        targets : ndarray of shape (n,)

        Returns
        -------
        steps : ndarray
        alignments : ndarray
        centered_alignments : ndarray
        """
        steps = np.array([s for s, _ in kernel_snapshots])
        alignments = np.array([self.compute_alignment(K, targets)
                               for _, K in kernel_snapshots])
        centered = np.array([self.centered_alignment(K, targets)
                             for _, K in kernel_snapshots])
        return steps, alignments, centered

    def alignment_significance(self, kernel, targets, n_permutations=1000):
        """Permutation test for kernel-target alignment significance.

        Parameters
        ----------
        kernel : ndarray of shape (n, n)
        targets : ndarray of shape (n,)
        n_permutations : int

        Returns
        -------
        result : dict
            'observed': observed alignment,
            'p_value': empirical p-value,
            'null_mean': mean alignment under null,
            'null_std': std of null distribution,
            'z_score': z-score of observed alignment.
        """
        observed = self.compute_alignment(kernel, targets)
        y = np.asarray(targets, dtype=float).ravel()
        rng = np.random.RandomState(42)
        null_alignments = np.empty(n_permutations)
        for i in range(n_permutations):
            perm = rng.permutation(y)
            null_alignments[i] = self.compute_alignment(kernel, perm)

        p_value = float(np.mean(null_alignments >= observed))
        null_mean = float(np.mean(null_alignments))
        null_std = float(np.std(null_alignments))
        z_score = (observed - null_mean) / null_std if null_std > 0 else 0.0

        return {
            'observed': observed,
            'p_value': p_value,
            'null_mean': null_mean,
            'null_std': null_std,
            'z_score': z_score,
        }


class GeneralizationBound:
    """Generalization bounds from kernel-data interaction."""

    def __init__(self, kernel_matrix, targets, n_train):
        self.kernel_matrix = np.asarray(kernel_matrix, dtype=float)
        self.targets = np.asarray(targets, dtype=float).ravel()
        self.n_train = n_train

    def rademacher_bound(self, kernel_matrix, n_train):
        """Rademacher complexity bound for kernel class.

        R_n(F) <= sqrt(tr(K) / n^2)

        Parameters
        ----------
        kernel_matrix : ndarray of shape (n, n)
        n_train : int

        Returns
        -------
        bound : float
            Rademacher complexity bound on generalization gap.
        """
        trace_K = np.trace(kernel_matrix)
        rademacher = np.sqrt(trace_K) / n_train
        # Generalization bound: 2 * R_n + sqrt(log(2/delta)/(2n)) for delta=0.05
        delta = 0.05
        bound = 2.0 * rademacher + np.sqrt(np.log(2.0 / delta) / (2.0 * n_train))
        return float(bound)

    def kernel_ridge_bound(self, kernel, targets, regularization):
        """Generalization bound for kernel ridge regression.

        Uses the standard bound: E[excess risk] <= lambda ||f*||_H^2 + sigma^2 d_eff / n

        Parameters
        ----------
        kernel : ndarray of shape (n, n)
        targets : ndarray of shape (n,)
        regularization : float

        Returns
        -------
        bound : float
        """
        n = kernel.shape[0]
        y = np.asarray(targets, dtype=float).ravel()

        eigs = linalg.eigvalsh(kernel)
        eigs = np.maximum(eigs, 0)

        # Effective dimension
        d_eff = np.sum(eigs / (eigs + regularization))

        # RKHS norm estimate: ||f*||^2 ~ y^T (K + lambda I)^{-1} y
        K_reg = kernel + regularization * np.eye(n)
        alpha = linalg.solve(K_reg, y, assume_a='pos')
        rkhs_norm_sq = float(y @ alpha)

        # Noise variance estimate from residuals
        f_hat = kernel @ alpha
        residuals = y - f_hat
        sigma_sq = float(np.mean(residuals ** 2))

        bias = regularization * rkhs_norm_sq
        variance = sigma_sq * d_eff / n
        return float(bias + variance)

    def spectral_bound(self, eigenvalues, targets_in_eigenbasis, regularization):
        """Spectral generalization bound decomposed per eigenmode.

        Parameters
        ----------
        eigenvalues : ndarray of shape (k,)
        targets_in_eigenbasis : ndarray of shape (k,)
            Coefficients of targets in kernel eigenbasis.
        regularization : float

        Returns
        -------
        result : dict
            'total_bound': total bound,
            'bias_per_mode': bias contribution from each mode,
            'variance_per_mode': variance contribution from each mode.
        """
        lam = np.asarray(eigenvalues, dtype=float)
        c = np.asarray(targets_in_eigenbasis, dtype=float)
        r = regularization

        # Bias per mode: (r / (lam_k + r))^2 * c_k^2
        bias_per_mode = (r / (lam + r)) ** 2 * c ** 2

        # Variance per mode: lam_k / (lam_k + r)^2  (times sigma^2/n, but we set sigma=1)
        variance_per_mode = lam / (lam + r) ** 2

        return {
            'total_bound': float(np.sum(bias_per_mode) + np.sum(variance_per_mode)),
            'bias_per_mode': bias_per_mode,
            'variance_per_mode': variance_per_mode,
        }

    def loo_bound(self, kernel, targets, regularization):
        """Leave-one-out cross-validation bound (exact for KRR).

        LOO_error = (1/n) sum_i (y_i - f_{-i}(x_i))^2
                  = (1/n) sum_i (alpha_i / G_{ii})^2

        Parameters
        ----------
        kernel : ndarray of shape (n, n)
        targets : ndarray of shape (n,)
        regularization : float

        Returns
        -------
        loo_error : float
        """
        n = kernel.shape[0]
        y = np.asarray(targets, dtype=float).ravel()
        K_reg = kernel + regularization * np.eye(n)
        K_inv = linalg.inv(K_reg)
        alpha = K_inv @ y

        # LOO formula: e_i = alpha_i / K_inv_{ii}
        diag_inv = np.diag(K_inv)
        loo_residuals = alpha / diag_inv
        loo_error = float(np.mean(loo_residuals ** 2))
        return loo_error

    def pac_bayes_bound(self, kernel, targets, prior_variance):
        """PAC-Bayes generalization bound for kernel predictor.

        PAC-Bayes: E[L] <= E_hat[L] + sqrt(KL(Q||P) + log(2n/delta)) / (2(n-1)))

        Parameters
        ----------
        kernel : ndarray of shape (n, n)
        targets : ndarray of shape (n,)
        prior_variance : float
            Prior variance for the Gaussian prior.

        Returns
        -------
        bound : float
        """
        n = kernel.shape[0]
        y = np.asarray(targets, dtype=float).ravel()
        delta = 0.05

        # Posterior: KRR solution with regularization = 1/prior_variance
        reg = 1.0 / prior_variance
        K_reg = kernel + reg * np.eye(n)
        alpha = linalg.solve(K_reg, y, assume_a='pos')
        f_hat = kernel @ alpha

        # Empirical loss (squared)
        empirical_loss = float(np.mean((y - f_hat) ** 2))

        # KL divergence: KL(posterior || prior) for Gaussian
        # posterior covariance ≈ (K + reg I)^{-1}, prior covariance = prior_variance * I
        # Simplified: KL ≈ 0.5 * (||alpha||^2 / prior_variance + d_eff - log_det_ratio)
        eigs = linalg.eigvalsh(kernel)
        eigs = np.maximum(eigs, 0)
        d_eff = np.sum(eigs / (eigs + reg))

        w_norm_sq = float(alpha @ kernel @ alpha)
        kl = 0.5 * (w_norm_sq / prior_variance + d_eff)

        complexity = np.sqrt((kl + np.log(2.0 * n / delta)) / (2.0 * (n - 1)))
        bound = empirical_loss + float(complexity)
        return bound

    def effective_dimension_bound(self, kernel, regularization, n_train):
        """Generalization bound based on effective dimension.

        Parameters
        ----------
        kernel : ndarray of shape (n, n)
        regularization : float
        n_train : int

        Returns
        -------
        bound : float
        """
        eigs = linalg.eigvalsh(kernel)
        eigs = np.maximum(eigs, 0)
        d_eff = float(np.sum(eigs / (eigs + regularization)))
        # Standard rate: O(d_eff / n)
        bound = d_eff / n_train
        return bound

    def bias_variance_decomposition(self, kernel, targets, regularization, n_train):
        """Decompose expected risk into bias and variance.

        Parameters
        ----------
        kernel : ndarray of shape (n, n)
        targets : ndarray of shape (n,)
        regularization : float
        n_train : int

        Returns
        -------
        result : dict
            'bias': bias term,
            'variance': variance term,
            'total': bias + variance.
        """
        n = kernel.shape[0]
        y = np.asarray(targets, dtype=float).ravel()
        eigs, V = linalg.eigh(kernel)
        eigs = np.maximum(eigs, 0)

        # Project targets
        c = V.T @ y

        # Bias: sum_k (reg / (lam_k + reg))^2 * c_k^2
        filter_bias = (regularization / (eigs + regularization)) ** 2
        bias = float(np.sum(filter_bias * c ** 2))

        # Variance: (sigma^2/n) * sum_k lam_k^2 / (lam_k + reg)^2
        # Estimate sigma from residuals
        K_reg = kernel + regularization * np.eye(n)
        alpha = linalg.solve(K_reg, y, assume_a='pos')
        residuals = y - kernel @ alpha
        sigma_sq = float(np.mean(residuals ** 2))

        filter_var = eigs ** 2 / (eigs + regularization) ** 2
        variance = sigma_sq * float(np.sum(filter_var)) / n_train

        return {
            'bias': bias,
            'variance': variance,
            'total': bias + variance,
        }

    def learning_curve_prediction(self, kernel, targets, n_train_range):
        """Predict learning curve from kernel eigenstructure.

        For each n in n_train_range, estimates the expected test error
        using the bias-variance decomposition.

        Parameters
        ----------
        kernel : ndarray of shape (n, n)
        targets : ndarray of shape (n,)
        n_train_range : array-like of int

        Returns
        -------
        n_values : ndarray
        predicted_errors : ndarray
        """
        n_full = kernel.shape[0]
        y = np.asarray(targets, dtype=float).ravel()
        eigs, V = linalg.eigh(kernel)
        eigs = np.maximum(eigs, 0)
        c = V.T @ y

        # Use a default regularization of 1e-3 * trace(K) / n
        reg = 1e-3 * np.sum(eigs) / n_full

        predicted_errors = []
        for n in n_train_range:
            n = int(n)
            # Bias: sum (reg/(lam+reg))^2 c_k^2
            bias = np.sum((reg / (eigs + reg)) ** 2 * c ** 2)
            # Variance scales as d_eff / n
            d_eff = np.sum(eigs / (eigs + reg))
            variance = d_eff / n
            predicted_errors.append(bias + variance)

        return np.asarray(n_train_range), np.asarray(predicted_errors)

    def minimax_rate(self, eigenvalues, smoothness):
        """Compute minimax rate for the kernel class with given smoothness.

        For Sobolev-type classes with eigenvalue decay lambda_k ~ k^{-2s},
        the minimax rate is n^{-2s/(2s+1)}.

        Parameters
        ----------
        eigenvalues : ndarray
            Kernel eigenvalues in descending order.
        smoothness : float
            Sobolev smoothness parameter s.

        Returns
        -------
        result : dict
            'minimax_rate_exponent': -2s/(2s+1),
            'estimated_decay_rate': fitted eigenvalue decay rate,
            'optimal_regularization': lambda ~ n^{-2s/(2s+1)}.
        """
        eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
        eigenvalues = eigenvalues[eigenvalues > 0]

        # Fit power law decay: log(lambda_k) = -alpha * log(k) + const
        k_vals = np.arange(1, len(eigenvalues) + 1, dtype=float)
        log_k = np.log(k_vals)
        log_eig = np.log(eigenvalues)
        # Linear regression
        A = np.vstack([log_k, np.ones_like(log_k)]).T
        slope, _ = np.linalg.lstsq(A, log_eig, rcond=None)[0]
        estimated_decay = -slope

        rate_exponent = -2.0 * smoothness / (2.0 * smoothness + 1.0)
        opt_reg_exponent = rate_exponent  # optimal reg ~ n^{rate_exponent}

        return {
            'minimax_rate_exponent': rate_exponent,
            'estimated_decay_rate': float(estimated_decay),
            'optimal_regularization_exponent': opt_reg_exponent,
        }


class EffectiveDimension:
    """Effective dimension computation for kernel methods."""

    def __init__(self):
        pass

    def compute_effective_dim(self, kernel, regularization):
        """Compute effective dimension d_eff = tr(K (K + lambda I)^{-1}).

        Parameters
        ----------
        kernel : ndarray of shape (n, n)
        regularization : float

        Returns
        -------
        d_eff : float
        """
        eigs = linalg.eigvalsh(kernel)
        eigs = np.maximum(eigs, 0)
        d_eff = float(np.sum(eigs / (eigs + regularization)))
        return d_eff

    def effective_dim_vs_regularization(self, kernel, reg_range):
        """Compute effective dimension as a function of regularization.

        Parameters
        ----------
        kernel : ndarray of shape (n, n)
        reg_range : array-like of float

        Returns
        -------
        reg_values : ndarray
        d_eff_values : ndarray
        """
        eigs = linalg.eigvalsh(kernel)
        eigs = np.maximum(eigs, 0)
        reg_values = np.asarray(reg_range, dtype=float)
        d_eff_values = np.array([float(np.sum(eigs / (eigs + r)))
                                 for r in reg_values])
        return reg_values, d_eff_values

    def degrees_of_freedom(self, kernel, regularization):
        """Compute degrees of freedom (same as effective dimension).

        Parameters
        ----------
        kernel : ndarray of shape (n, n)
        regularization : float

        Returns
        -------
        dof : float
        """
        return self.compute_effective_dim(kernel, regularization)

    def information_dimension(self, eigenvalues):
        """Compute information dimension from eigenvalue entropy.

        d_info = exp(H(p)) where p_k = lambda_k / sum(lambda) and H is Shannon entropy.

        Parameters
        ----------
        eigenvalues : ndarray

        Returns
        -------
        d_info : float
        """
        eigs = np.asarray(eigenvalues, dtype=float)
        eigs = eigs[eigs > 0]
        if len(eigs) == 0:
            return 0.0
        p = eigs / np.sum(eigs)
        entropy = -np.sum(p * np.log(p + 1e-30))
        return float(np.exp(entropy))

    def participation_ratio(self, eigenvalues):
        """Compute participation ratio PR = (sum lambda)^2 / sum(lambda^2).

        Parameters
        ----------
        eigenvalues : ndarray

        Returns
        -------
        pr : float
        """
        eigs = np.asarray(eigenvalues, dtype=float)
        eigs = eigs[eigs > 0]
        if len(eigs) == 0:
            return 0.0
        s1 = np.sum(eigs)
        s2 = np.sum(eigs ** 2)
        if s2 == 0:
            return 0.0
        return float(s1 ** 2 / s2)

    def intrinsic_dimension(self, data, method='mle'):
        """Estimate intrinsic dimension of the data manifold.

        Parameters
        ----------
        data : ndarray of shape (n, d)
        method : str
            'mle': Maximum likelihood estimator (Levina-Bickel).
            'pca': PCA-based (fraction of variance explained).

        Returns
        -------
        dim : float
        """
        data = np.asarray(data, dtype=float)
        n, d = data.shape

        if method == 'mle':
            # Levina-Bickel MLE for intrinsic dimension
            from scipy.spatial.distance import cdist
            dists = cdist(data, data)
            k = min(10, n - 1)
            dims = []
            for i in range(n):
                sorted_d = np.sort(dists[i])[1:k+1]  # exclude self
                sorted_d = sorted_d[sorted_d > 0]
                if len(sorted_d) < 2:
                    continue
                T_k = sorted_d[-1]
                if T_k <= 0:
                    continue
                log_ratios = np.log(T_k / sorted_d[:-1])
                if np.sum(log_ratios) > 0:
                    dims.append((len(log_ratios)) / np.sum(log_ratios))
            return float(np.mean(dims)) if dims else 0.0

        elif method == 'pca':
            # PCA-based: number of components for 90% variance
            data_centered = data - np.mean(data, axis=0)
            _, s, _ = linalg.svd(data_centered, full_matrices=False)
            variance_explained = np.cumsum(s ** 2) / np.sum(s ** 2)
            dim = int(np.searchsorted(variance_explained, 0.9) + 1)
            return float(min(dim, d))
        else:
            raise ValueError(f"Unknown method: {method}")

    def spectral_dimension(self, eigenvalues, threshold=0.99):
        """Number of eigenvalues needed to capture threshold fraction of total variance.

        Parameters
        ----------
        eigenvalues : ndarray
        threshold : float
            Fraction of total variance to capture (default 0.99).

        Returns
        -------
        d_spectral : int
        """
        eigs = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
        eigs = eigs[eigs > 0]
        if len(eigs) == 0:
            return 0
        cumvar = np.cumsum(eigs) / np.sum(eigs)
        return int(np.searchsorted(cumvar, threshold) + 1)

    def effective_dim_scaling(self, kernels_at_sizes, regularization):
        """Analyze how effective dimension scales with dataset size.

        Parameters
        ----------
        kernels_at_sizes : list of (n, kernel_matrix) tuples
        regularization : float

        Returns
        -------
        result : dict
            'sizes': dataset sizes,
            'd_eff': effective dimensions,
            'scaling_exponent': fitted power-law exponent d_eff ~ n^alpha.
        """
        sizes = []
        d_effs = []
        for n, K in kernels_at_sizes:
            sizes.append(n)
            d_effs.append(self.compute_effective_dim(K, regularization))

        sizes = np.array(sizes, dtype=float)
        d_effs = np.array(d_effs)

        # Fit power law: log(d_eff) = alpha * log(n) + const
        valid = (sizes > 0) & (d_effs > 0)
        if np.sum(valid) >= 2:
            log_n = np.log(sizes[valid])
            log_d = np.log(d_effs[valid])
            A = np.vstack([log_n, np.ones_like(log_n)]).T
            coeffs = np.linalg.lstsq(A, log_d, rcond=None)[0]
            alpha = coeffs[0]
        else:
            alpha = 0.0

        return {
            'sizes': sizes,
            'd_eff': d_effs,
            'scaling_exponent': float(alpha),
        }


class SpectralBiasAnalyzer:
    """Spectral bias analysis for neural networks."""

    def __init__(self):
        pass

    def target_in_eigenbasis(self, eigenvectors, targets):
        """Project targets onto kernel eigenbasis.

        Parameters
        ----------
        eigenvectors : ndarray of shape (n, k)
            Eigenvectors as columns.
        targets : ndarray of shape (n,)

        Returns
        -------
        coefficients : ndarray of shape (k,)
            Projection coefficients c_k = v_k^T y.
        """
        y = np.asarray(targets, dtype=float).ravel()
        return eigenvectors.T @ y

    def learning_speed_per_mode(self, eigenvalues, learning_rate):
        """Compute learning speed for each eigenmode.

        For gradient descent on the NTK model, mode k converges at rate
        1 - (1 - eta * lambda_k)^t, so the per-step convergence factor is
        eta * lambda_k.

        Parameters
        ----------
        eigenvalues : ndarray of shape (k,)
        learning_rate : float

        Returns
        -------
        speeds : ndarray of shape (k,)
            Learning speed eta * lambda_k per mode.
        convergence_time : ndarray of shape (k,)
            Approximate time to convergence 1 / (eta * lambda_k) per mode.
        """
        eigs = np.asarray(eigenvalues, dtype=float)
        speeds = learning_rate * eigs
        with np.errstate(divide='ignore', invalid='ignore'):
            convergence_time = np.where(
                speeds > 0, 1.0 / speeds, np.inf
            )
        return speeds, convergence_time

    def bias_profile(self, eigenvalues, target_coefficients):
        """Analyze which eigenmodes dominate the target function.

        Parameters
        ----------
        eigenvalues : ndarray of shape (k,)
        target_coefficients : ndarray of shape (k,)

        Returns
        -------
        result : dict
            'energy_per_mode': |c_k|^2, energy in each mode,
            'fractional_energy': normalized energy per mode,
            'cumulative_energy': cumulative energy fraction,
            'effective_target_dimension': participation ratio of energy distribution,
            'spectral_decay_rate': fitted power-law decay of coefficients.
        """
        c = np.asarray(target_coefficients, dtype=float)
        energy = c ** 2
        total_energy = np.sum(energy)
        if total_energy == 0:
            k = len(c)
            return {
                'energy_per_mode': energy,
                'fractional_energy': np.zeros(k),
                'cumulative_energy': np.zeros(k),
                'effective_target_dimension': 0.0,
                'spectral_decay_rate': 0.0,
            }

        frac = energy / total_energy
        cumulative = np.cumsum(frac)
        pr = total_energy ** 2 / np.sum(energy ** 2) if np.sum(energy ** 2) > 0 else 0.0

        # Fit power-law decay to energy spectrum
        pos_energy = energy[energy > 0]
        if len(pos_energy) >= 2:
            k_vals = np.arange(1, len(pos_energy) + 1, dtype=float)
            log_k = np.log(k_vals)
            log_e = np.log(pos_energy)
            A = np.vstack([log_k, np.ones_like(log_k)]).T
            slope = np.linalg.lstsq(A, log_e, rcond=None)[0][0]
            decay_rate = -slope
        else:
            decay_rate = 0.0

        return {
            'energy_per_mode': energy,
            'fractional_energy': frac,
            'cumulative_energy': cumulative,
            'effective_target_dimension': float(pr),
            'spectral_decay_rate': float(decay_rate),
        }

    def frequency_bias(self, eigenvectors, spatial_frequencies):
        """Map kernel eigenmodes to spatial frequencies.

        Computes the overlap between each eigenmode and each spatial frequency
        component to determine which frequencies the network learns first.

        Parameters
        ----------
        eigenvectors : ndarray of shape (n, k)
        spatial_frequencies : ndarray of shape (n, m)
            Frequency basis functions evaluated at data points.

        Returns
        -------
        frequency_map : ndarray of shape (k, m)
            |<v_k, phi_j>|^2 overlap between mode k and frequency j.
        dominant_frequency : ndarray of shape (k,)
            Index of dominant frequency for each mode.
        """
        V = np.asarray(eigenvectors, dtype=float)
        F = np.asarray(spatial_frequencies, dtype=float)
        # Overlap: |V^T @ F|^2, shape (k, m)
        overlap = (V.T @ F) ** 2
        dominant = np.argmax(overlap, axis=1)
        return overlap, dominant

    def time_to_learn_mode(self, eigenvalue, target_coefficient, tolerance):
        """Compute time steps needed to learn mode k to within tolerance.

        Under gradient flow on NTK: residual_k(t) = c_k * exp(-lambda_k * t)
        Time to |residual| < tol: t = -log(tol / |c_k|) / lambda_k

        Parameters
        ----------
        eigenvalue : float
        target_coefficient : float
        tolerance : float

        Returns
        -------
        t : float
            Number of time steps (continuous time).
        """
        if eigenvalue <= 0 or tolerance <= 0:
            return np.inf
        c = abs(target_coefficient)
        if c <= tolerance:
            return 0.0
        return float(-np.log(tolerance / c) / eigenvalue)

    def spectral_bias_curve(self, eigenvalues, target_coefficients, times):
        """Compute per-mode residual over time under gradient flow.

        residual_k(t) = c_k * exp(-lambda_k * t)

        Parameters
        ----------
        eigenvalues : ndarray of shape (k,)
        target_coefficients : ndarray of shape (k,)
        times : ndarray of shape (T,)

        Returns
        -------
        residuals : ndarray of shape (k, T)
            Per-mode residual at each time.
        total_residual : ndarray of shape (T,)
            Total squared residual sum_k residual_k(t)^2.
        """
        eigs = np.asarray(eigenvalues, dtype=float)[:, None]
        c = np.asarray(target_coefficients, dtype=float)[:, None]
        t = np.asarray(times, dtype=float)[None, :]

        residuals = c * np.exp(-eigs * t)
        total_residual = np.sum(residuals ** 2, axis=0)
        return residuals, total_residual

    def anti_bias_regularization(self, eigenvalues, target_coefficients):
        """Compute regularization that counteracts spectral bias.

        The idea is to equalize learning speed across modes by choosing
        per-mode learning rates proportional to 1/lambda_k, which is
        equivalent to preconditioning with K^{-1}.

        Parameters
        ----------
        eigenvalues : ndarray of shape (k,)
        target_coefficients : ndarray of shape (k,)

        Returns
        -------
        result : dict
            'preconditioner_eigenvalues': 1/lambda_k (capped for stability),
            'equalized_speeds': speeds after preconditioning,
            'optimal_regularization': regularization that minimizes
                max-mode convergence time disparity.
        """
        eigs = np.asarray(eigenvalues, dtype=float)
        c = np.asarray(target_coefficients, dtype=float)

        # Preconditioner eigenvalues: 1/lambda_k capped at 1/eps
        eps = max(eigs.max() * 1e-10, 1e-15)
        precond = 1.0 / np.maximum(eigs, eps)

        # Equalized speeds: lambda_k * precond_k = 1 (ideally)
        equalized = eigs * precond

        # Find regularization that minimizes the ratio of max/min convergence times
        # among modes with significant target energy
        energy = c ** 2
        threshold = 0.01 * np.max(energy) if np.max(energy) > 0 else 0
        significant = energy > threshold
        if np.sum(significant) < 2:
            opt_reg = 0.0
        else:
            sig_eigs = eigs[significant]

            def disparity(log_reg):
                reg = np.exp(log_reg)
                speeds = sig_eigs / (sig_eigs + reg)
                ratio = speeds.max() / max(speeds.min(), 1e-30)
                return ratio

            if sig_eigs.min() > 0:
                result = minimize_scalar(
                    disparity,
                    bounds=(np.log(sig_eigs.min() * 0.01),
                            np.log(sig_eigs.max() * 10)),
                    method='bounded',
                )
                opt_reg = np.exp(result.x)
            else:
                opt_reg = 0.0

        return {
            'preconditioner_eigenvalues': precond,
            'equalized_speeds': equalized,
            'optimal_regularization': float(opt_reg),
        }

    def implicit_bias_analysis(self, eigenvalues, target_coefficients, n_train):
        """Analyze implicit bias from finite training data.

        With n training samples, the kernel is n x n and only the top n
        eigenmodes can be learned. This creates an implicit bias toward
        the top eigenmodes.

        Parameters
        ----------
        eigenvalues : ndarray of shape (k,)
            Full (population) eigenvalues.
        target_coefficients : ndarray of shape (k,)
        n_train : int

        Returns
        -------
        result : dict
            'learnable_energy': fraction of target energy in top n modes,
            'truncation_error': energy lost by truncating to n modes,
            'mode_cutoff': effective number of modes that can be learned,
            'bias_ratio': ratio of top-mode to bottom-mode learning speed,
            'approximation_error': ||f* - f*_n||^2 from truncation.
        """
        eigs = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
        c = np.asarray(target_coefficients, dtype=float)

        # Reorder c to match descending eigenvalue order if needed
        if len(c) > len(eigs):
            c = c[:len(eigs)]
        elif len(eigs) > len(c):
            eigs = eigs[:len(c)]

        total_energy = np.sum(c ** 2)
        n_eff = min(n_train, len(eigs))

        learnable_energy = np.sum(c[:n_eff] ** 2) / total_energy if total_energy > 0 else 0.0
        truncation_error = np.sum(c[n_eff:] ** 2) if n_eff < len(c) else 0.0

        # Effective number of learnable modes (using effective dimension idea)
        if len(eigs) > 0 and eigs[0] > 0:
            # With n samples, effective regularization ~ trace(K)/n
            reg_implicit = np.sum(eigs) / n_train
            d_eff = float(np.sum(eigs / (eigs + reg_implicit)))
        else:
            d_eff = 0.0

        # Bias ratio
        if n_eff >= 2 and eigs[0] > 0 and eigs[n_eff - 1] > 0:
            bias_ratio = float(eigs[0] / eigs[n_eff - 1])
        else:
            bias_ratio = np.inf

        return {
            'learnable_energy': float(learnable_energy),
            'truncation_error': float(truncation_error),
            'mode_cutoff': float(d_eff),
            'bias_ratio': bias_ratio,
            'approximation_error': float(truncation_error),
        }
