"""
Rich/feature-learning regime analysis for neural networks.

Implements metrics and trackers for quantifying feature learning,
representation change, feature alignment, and neural collapse.
"""

import numpy as np
from scipy import linalg, stats, spatial
from scipy.optimize import linear_sum_assignment


class RichRegimeAnalyzer:
    """Analyze whether training is in the rich (feature-learning) regime."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def is_rich_regime(self, kernel_trajectory, threshold=0.2):
        """Check if features change significantly during training.

        Compares initial and final NTK; large change indicates rich regime.

        Parameters
        ----------
        kernel_trajectory : list of ndarray, shape (n, n)
            NTK matrices at different training times.
        threshold : float
            Minimum relative Frobenius-norm change to classify as rich.

        Returns
        -------
        bool
        """
        K0 = kernel_trajectory[0]
        Kf = kernel_trajectory[-1]
        norm_K0 = np.linalg.norm(K0, 'fro')
        if norm_K0 < 1e-12:
            return True
        relative_change = np.linalg.norm(Kf - K0, 'fro') / norm_K0
        return relative_change > threshold

    def feature_learning_strength(self, kernel_trajectory, times):
        """Quantify degree of feature learning over training.

        Returns a scalar summarising how much the kernel drifted,
        normalised by duration and initial kernel scale.

        Parameters
        ----------
        kernel_trajectory : list of ndarray
        times : array-like, shape (T,)

        Returns
        -------
        float
            Integrated relative kernel velocity.
        """
        times = np.asarray(times, dtype=float)
        strengths = np.zeros(len(times))
        norm_K0 = np.linalg.norm(kernel_trajectory[0], 'fro')
        if norm_K0 < 1e-12:
            norm_K0 = 1.0
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            if dt < 1e-15:
                continue
            dK = kernel_trajectory[i] - kernel_trajectory[i - 1]
            strengths[i] = np.linalg.norm(dK, 'fro') / (norm_K0 * dt)
        # Trapezoidal integration of velocity curve
        integrated = np.trapz(strengths, times)
        return float(integrated)

    def representation_similarity(self, features_t1, features_t2):
        """Centered Kernel Alignment between two feature snapshots.

        Parameters
        ----------
        features_t1, features_t2 : ndarray, shape (n, d)

        Returns
        -------
        float in [0, 1]
        """
        K1 = features_t1 @ features_t1.T
        K2 = features_t2 @ features_t2.T
        metric = RepresentationChangeMetric()
        return metric.centered_kernel_alignment(K1, K2)

    def effective_rank_evolution(self, feature_matrices):
        """Track effective rank (exp of spectral entropy) over time.

        Parameters
        ----------
        feature_matrices : list of ndarray, shape (n, d)

        Returns
        -------
        ndarray, shape (T,)
        """
        ranks = np.empty(len(feature_matrices))
        for i, F in enumerate(feature_matrices):
            s = np.linalg.svd(F, compute_uv=False)
            s = s[s > 1e-12]
            if len(s) == 0:
                ranks[i] = 0.0
                continue
            p = s / s.sum()
            entropy = -np.sum(p * np.log(p))
            ranks[i] = np.exp(entropy)
        return ranks

    def feature_covariance_evolution(self, feature_matrices):
        """Track Φ^T Φ covariance matrices over time.

        Parameters
        ----------
        feature_matrices : list of ndarray, shape (n, d)

        Returns
        -------
        list of ndarray, shape (d, d)
            Covariance matrices at each time step.
        """
        covs = []
        for F in feature_matrices:
            F_centered = F - F.mean(axis=0, keepdims=True)
            n = F_centered.shape[0]
            covs.append(F_centered.T @ F_centered / max(n - 1, 1))
        return covs

    def alignment_with_target(self, features, target_function, inputs):
        """How well features align with a target function.

        Computes fraction of target variance explained by a linear
        readout from features.

        Parameters
        ----------
        features : ndarray, shape (n, d)
        target_function : callable  inputs -> (n,) or (n, k)
        inputs : ndarray, shape (n, p)

        Returns
        -------
        float in [0, 1]
        """
        y = np.atleast_2d(target_function(inputs))
        if y.shape[0] == 1 and y.shape[1] != 1:
            y = y.T  # ensure (n, k)
        # Least-squares fit
        F = features
        coeff, _, _, _ = np.linalg.lstsq(F, y, rcond=None)
        y_pred = F @ coeff
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean(axis=0)) ** 2)
        if ss_tot < 1e-12:
            return 1.0
        return float(1.0 - ss_res / ss_tot)


class FeatureEvolutionTracker:
    """Track feature evolution during training."""

    def __init__(self, n_checkpoints=100):
        self.n_checkpoints = n_checkpoints
        self.snapshots = []
        self.snapshot_times = []

    def track_features(self, get_features_fn, training_steps):
        """Record features at evenly-spaced checkpoints.

        Parameters
        ----------
        get_features_fn : callable  step -> ndarray (n, d)
        training_steps : int

        Returns
        -------
        list of ndarray
            Feature snapshots.
        """
        checkpoint_indices = np.linspace(
            0, training_steps - 1, self.n_checkpoints, dtype=int
        )
        checkpoint_indices = np.unique(checkpoint_indices)
        self.snapshots = []
        self.snapshot_times = []
        for step in checkpoint_indices:
            self.snapshots.append(get_features_fn(int(step)))
            self.snapshot_times.append(int(step))
        return self.snapshots

    def feature_velocity(self, feature_snapshots, times):
        """Compute dΦ/dt via finite differences.

        Parameters
        ----------
        feature_snapshots : list of ndarray, shape (n, d)
        times : array-like, shape (T,)

        Returns
        -------
        list of ndarray, shape (n, d)
            Velocity matrices (length T-1).
        """
        times = np.asarray(times, dtype=float)
        velocities = []
        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            if dt < 1e-15:
                velocities.append(np.zeros_like(feature_snapshots[i]))
            else:
                velocities.append(
                    (feature_snapshots[i] - feature_snapshots[i - 1]) / dt
                )
        return velocities

    def feature_acceleration(self, feature_snapshots, times):
        """Compute d²Φ/dt² via second-order finite differences.

        Parameters
        ----------
        feature_snapshots : list of ndarray, shape (n, d)
        times : array-like, shape (T,)

        Returns
        -------
        list of ndarray, shape (n, d)
            Acceleration matrices (length T-2).
        """
        velocities = self.feature_velocity(feature_snapshots, times)
        times = np.asarray(times, dtype=float)
        mid_times = 0.5 * (times[:-1] + times[1:])
        accelerations = []
        for i in range(1, len(velocities)):
            dt = mid_times[i] - mid_times[i - 1]
            if dt < 1e-15:
                accelerations.append(np.zeros_like(velocities[i]))
            else:
                accelerations.append((velocities[i] - velocities[i - 1]) / dt)
        return accelerations

    def principal_feature_dynamics(self, feature_snapshots):
        """PCA of the stacked feature evolution trajectory.

        Flattens each snapshot, stacks into (T, n*d), runs PCA.

        Parameters
        ----------
        feature_snapshots : list of ndarray, shape (n, d)

        Returns
        -------
        dict with keys:
            components : ndarray (k, n*d)  – principal directions
            explained_variance_ratio : ndarray (k,)
            projections : ndarray (T, k) – trajectory in PC space
        """
        flat = np.array([F.ravel() for F in feature_snapshots])
        flat_centered = flat - flat.mean(axis=0, keepdims=True)
        cov = flat_centered.T @ flat_centered / max(flat.shape[0] - 1, 1)
        k = min(flat.shape[0], 20)
        # Use truncated eigendecomposition for efficiency
        if cov.shape[0] <= k:
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
        else:
            # Use covariance in sample space for efficiency when n*d >> T
            gram = flat_centered @ flat_centered.T / max(flat.shape[0] - 1, 1)
            eigvals, alpha = np.linalg.eigh(gram)
            idx = np.argsort(eigvals)[::-1][:k]
            eigvals = eigvals[idx]
            alpha = alpha[:, idx]
            eigvecs = flat_centered.T @ alpha
            for j in range(eigvecs.shape[1]):
                norm = np.linalg.norm(eigvecs[:, j])
                if norm > 1e-12:
                    eigvecs[:, j] /= norm

        total_var = eigvals.sum()
        evr = eigvals / total_var if total_var > 1e-12 else eigvals * 0
        projections = flat_centered @ eigvecs[:, :k]
        return {
            'components': eigvecs[:, :k].T,
            'explained_variance_ratio': evr[:k],
            'projections': projections,
        }

    def feature_turnover(self, feature_snapshots):
        """Frobenius-norm change between successive snapshots.

        Parameters
        ----------
        feature_snapshots : list of ndarray

        Returns
        -------
        ndarray, shape (T-1,)
        """
        turnovers = np.empty(len(feature_snapshots) - 1)
        for i in range(1, len(feature_snapshots)):
            turnovers[i - 1] = np.linalg.norm(
                feature_snapshots[i] - feature_snapshots[i - 1], 'fro'
            )
        return turnovers

    def convergence_of_features(self, feature_snapshots, window=20):
        """Check if features are converging using trailing-window velocity.

        Parameters
        ----------
        feature_snapshots : list of ndarray
        window : int

        Returns
        -------
        dict with keys:
            converged : bool
            residual_velocity : float
            velocity_trajectory : ndarray
        """
        turnovers = self.feature_turnover(feature_snapshots)
        if len(turnovers) == 0:
            return {'converged': True, 'residual_velocity': 0.0,
                    'velocity_trajectory': np.array([])}
        w = min(window, len(turnovers))
        tail_mean = turnovers[-w:].mean()
        global_mean = turnovers.mean()
        converged = tail_mean < 0.1 * global_mean if global_mean > 1e-12 else True
        return {
            'converged': converged,
            'residual_velocity': float(tail_mean),
            'velocity_trajectory': turnovers,
        }

    def feature_rank_trajectory(self, feature_snapshots):
        """Effective rank at each snapshot.

        Parameters
        ----------
        feature_snapshots : list of ndarray

        Returns
        -------
        ndarray, shape (T,)
        """
        analyzer = RichRegimeAnalyzer(0, 0, 0)
        return analyzer.effective_rank_evolution(feature_snapshots)


class RepresentationChangeMetric:
    """Quantify changes between two representations."""

    def __init__(self):
        pass

    @staticmethod
    def _center_kernel(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    def centered_kernel_alignment(self, K1, K2):
        """CKA between two kernel matrices.

        CKA = HSIC(K1,K2) / sqrt(HSIC(K1,K1) * HSIC(K2,K2))

        Parameters
        ----------
        K1, K2 : ndarray, shape (n, n)

        Returns
        -------
        float in [0, 1]
        """
        Kc1 = self._center_kernel(K1)
        Kc2 = self._center_kernel(K2)
        hsic_12 = np.sum(Kc1 * Kc2)
        hsic_11 = np.sum(Kc1 * Kc1)
        hsic_22 = np.sum(Kc2 * Kc2)
        denom = np.sqrt(hsic_11 * hsic_22)
        if denom < 1e-12:
            return 0.0
        return float(np.clip(hsic_12 / denom, 0.0, 1.0))

    def procrustes_distance(self, features1, features2):
        """Procrustes analysis distance after optimal rotation/scaling.

        Parameters
        ----------
        features1, features2 : ndarray, shape (n, d)

        Returns
        -------
        float >= 0
        """
        F1 = features1 - features1.mean(axis=0)
        F2 = features2 - features2.mean(axis=0)
        norm1 = np.linalg.norm(F1, 'fro')
        norm2 = np.linalg.norm(F2, 'fro')
        if norm1 < 1e-12 or norm2 < 1e-12:
            return float(norm1 + norm2)
        F1 /= norm1
        F2 /= norm2
        U, _, Vt = np.linalg.svd(F2.T @ F1, full_matrices=False)
        R = U @ Vt
        return float(np.linalg.norm(F1 - F2 @ R, 'fro'))

    def mutual_information_estimate(self, features1, features2):
        """Estimate mutual information via k-NN (KSG estimator).

        Uses 1-NN distances for a simple nonparametric estimate.

        Parameters
        ----------
        features1, features2 : ndarray, shape (n, d1), (n, d2)

        Returns
        -------
        float
        """
        n = features1.shape[0]
        if n < 5:
            return 0.0
        joint = np.hstack([features1, features2])
        k = max(1, min(5, n // 5))
        # Joint distances
        from scipy.spatial import cKDTree
        tree_joint = cKDTree(joint)
        tree_x = cKDTree(features1)
        tree_y = cKDTree(features2)
        dd_joint, _ = tree_joint.query(joint, k=k + 1)
        eps = dd_joint[:, -1]  # distance to k-th neighbour
        eps = np.maximum(eps, 1e-12)
        # Count neighbours within eps in marginals
        n_x = np.array([
            len(tree_x.query_ball_point(features1[i], eps[i])) - 1
            for i in range(n)
        ])
        n_y = np.array([
            len(tree_y.query_ball_point(features2[i], eps[i])) - 1
            for i in range(n)
        ])
        n_x = np.maximum(n_x, 1)
        n_y = np.maximum(n_y, 1)
        from scipy.special import digamma
        mi = digamma(k) - np.mean(digamma(n_x) + digamma(n_y)) + digamma(n)
        return float(max(mi, 0.0))

    def representation_dissimilarity(self, rdm1, rdm2):
        """Compare two representational dissimilarity matrices.

        Uses Spearman correlation of upper-triangular entries.

        Parameters
        ----------
        rdm1, rdm2 : ndarray, shape (n, n)

        Returns
        -------
        float in [-1, 1]  (1 = identical structure)
        """
        idx = np.triu_indices_from(rdm1, k=1)
        v1 = rdm1[idx]
        v2 = rdm2[idx]
        if len(v1) < 2:
            return 0.0
        rho, _ = stats.spearmanr(v1, v2)
        return float(rho) if np.isfinite(rho) else 0.0

    def subspace_angle(self, features1, features2, k=10):
        """Principal angles between top-k subspaces of two feature sets.

        Parameters
        ----------
        features1, features2 : ndarray, shape (n, d)
        k : int

        Returns
        -------
        ndarray, shape (k,)
            Principal angles in radians.
        """
        k = min(k, features1.shape[1], features2.shape[1])
        U1, _, _ = np.linalg.svd(features1, full_matrices=False)
        U2, _, _ = np.linalg.svd(features2, full_matrices=False)
        U1 = U1[:, :k]
        U2 = U2[:, :k]
        M = U1.T @ U2
        sigmas = np.linalg.svd(M, compute_uv=False)
        sigmas = np.clip(sigmas, -1.0, 1.0)
        return np.arccos(sigmas[:k])

    def cca_similarity(self, features1, features2):
        """Mean canonical correlation between two feature sets.

        Parameters
        ----------
        features1, features2 : ndarray, shape (n, d1), (n, d2)

        Returns
        -------
        float in [0, 1]
        """
        F1 = features1 - features1.mean(axis=0)
        F2 = features2 - features2.mean(axis=0)
        n = F1.shape[0]
        d1, d2 = F1.shape[1], F2.shape[1]
        k = min(d1, d2, n)
        # QR for numerical stability
        Q1, _ = np.linalg.qr(F1, mode='reduced')
        Q2, _ = np.linalg.qr(F2, mode='reduced')
        svs = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
        svs = np.clip(svs[:k], 0.0, 1.0)
        return float(svs.mean())

    def linear_probe_accuracy_change(self, features_before, features_after, labels):
        """Change in linear probe accuracy before vs after.

        Parameters
        ----------
        features_before, features_after : ndarray, shape (n, d)
        labels : ndarray, shape (n,)

        Returns
        -------
        dict with keys accuracy_before, accuracy_after, change
        """
        def _linear_probe_acc(F, y):
            classes = np.unique(y)
            if len(classes) <= 1:
                return 1.0
            # One-hot encode
            Y = np.zeros((len(y), len(classes)))
            label_map = {c: i for i, c in enumerate(classes)}
            for idx, c in enumerate(y):
                Y[idx, label_map[c]] = 1.0
            # Ridge regression probe
            lam = 1e-3
            FtF = F.T @ F + lam * np.eye(F.shape[1])
            W = np.linalg.solve(FtF, F.T @ Y)
            preds = np.argmax(F @ W, axis=1)
            true_idx = np.array([label_map[c] for c in y])
            return float(np.mean(preds == true_idx))

        acc_before = _linear_probe_acc(features_before, labels)
        acc_after = _linear_probe_acc(features_after, labels)
        return {
            'accuracy_before': acc_before,
            'accuracy_after': acc_after,
            'change': acc_after - acc_before,
        }


class FeatureAlignmentAnalyzer:
    """Analyze alignment between learned features and task structure."""

    def __init__(self, target_dim=None):
        self.target_dim = target_dim

    def kernel_target_alignment(self, kernel, targets):
        """KTA = ⟨K, yy^T⟩ / (||K||_F ||yy^T||_F).

        Parameters
        ----------
        kernel : ndarray, shape (n, n)
        targets : ndarray, shape (n,) or (n, k)

        Returns
        -------
        float
        """
        y = np.atleast_2d(targets)
        if y.shape[0] == 1:
            y = y.T
        Ky = y @ y.T
        num = np.sum(kernel * Ky)
        denom = np.linalg.norm(kernel, 'fro') * np.linalg.norm(Ky, 'fro')
        if denom < 1e-12:
            return 0.0
        return float(num / denom)

    def feature_label_correlation(self, features, labels):
        """Mean absolute correlation between each feature and one-hot labels.

        Parameters
        ----------
        features : ndarray, shape (n, d)
        labels : ndarray, shape (n,)

        Returns
        -------
        ndarray, shape (d,)
        """
        classes = np.unique(labels)
        Y = np.zeros((len(labels), len(classes)))
        for i, c in enumerate(classes):
            Y[labels == c, i] = 1.0
        # Correlation of each feature with each one-hot column, take max
        d = features.shape[1]
        corrs = np.zeros(d)
        for j in range(d):
            f = features[:, j]
            f_std = f.std()
            if f_std < 1e-12:
                continue
            max_corr = 0.0
            for col in range(Y.shape[1]):
                y = Y[:, col]
                y_std = y.std()
                if y_std < 1e-12:
                    continue
                r = np.abs(np.corrcoef(f, y)[0, 1])
                if r > max_corr:
                    max_corr = r
            corrs[j] = max_corr
        return corrs

    def class_separability(self, features, labels):
        """Fisher discriminant ratio: tr(Σ_B) / tr(Σ_W).

        Parameters
        ----------
        features : ndarray, shape (n, d)
        labels : ndarray, shape (n,)

        Returns
        -------
        float
        """
        detector = NeuralCollapseDetector(len(np.unique(labels)))
        sw = detector.within_class_variance(features, labels)
        sb = detector.between_class_variance(features, labels)
        tr_w = np.trace(sw)
        tr_b = np.trace(sb)
        if tr_w < 1e-12:
            return float('inf') if tr_b > 1e-12 else 0.0
        return float(tr_b / tr_w)

    def feature_informativeness(self, features, labels, method='mutual_info'):
        """Estimate informativeness of features about labels.

        Parameters
        ----------
        features : ndarray, shape (n, d)
        labels : ndarray, shape (n,)
        method : str, 'mutual_info' or 'linear_probe'

        Returns
        -------
        float
        """
        if method == 'linear_probe':
            metric = RepresentationChangeMetric()
            result = metric.linear_probe_accuracy_change(
                np.zeros_like(features), features, labels
            )
            return result['accuracy_after']

        # Mutual information via binning per feature dimension
        n, d = features.shape
        classes = np.unique(labels)
        n_bins = max(5, int(np.sqrt(n)))
        mi_total = 0.0
        for j in range(d):
            f = features[:, j]
            bins = np.linspace(f.min() - 1e-10, f.max() + 1e-10, n_bins + 1)
            digitized = np.digitize(f, bins) - 1
            # Joint counts
            joint = np.zeros((n_bins, len(classes)))
            for i in range(n):
                b = min(digitized[i], n_bins - 1)
                c = np.searchsorted(classes, labels[i])
                joint[b, c] += 1
            joint /= n
            p_f = joint.sum(axis=1)
            p_c = joint.sum(axis=0)
            for bi in range(n_bins):
                for ci in range(len(classes)):
                    if joint[bi, ci] > 1e-15 and p_f[bi] > 1e-15 and p_c[ci] > 1e-15:
                        mi_total += joint[bi, ci] * np.log(
                            joint[bi, ci] / (p_f[bi] * p_c[ci])
                        )
        return float(mi_total / d)

    def alignment_rate(self, alignment_trajectory, times):
        """Compute d(alignment)/dt.

        Parameters
        ----------
        alignment_trajectory : array-like, shape (T,)
        times : array-like, shape (T,)

        Returns
        -------
        ndarray, shape (T-1,)
        """
        a = np.asarray(alignment_trajectory, dtype=float)
        t = np.asarray(times, dtype=float)
        dt = np.diff(t)
        da = np.diff(a)
        rates = np.zeros_like(da)
        mask = dt > 1e-15
        rates[mask] = da[mask] / dt[mask]
        return rates

    def optimal_feature_direction(self, targets, input_data):
        """Compute directions features should learn (top singular vectors of y x^T).

        Parameters
        ----------
        targets : ndarray, shape (n,) or (n, k)
        input_data : ndarray, shape (n, p)

        Returns
        -------
        dict with keys:
            directions : ndarray (k, p)
            singular_values : ndarray (k,)
        """
        y = np.atleast_2d(targets)
        if y.shape[0] == 1:
            y = y.T
        M = y.T @ input_data  # (k, p)
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        return {'directions': Vt, 'singular_values': s}

    def feature_redundancy(self, features):
        """Quantify redundancy in features via ratio of nuclear to Frobenius norm.

        Returns 1 when all features are collinear, 1/sqrt(rank) at minimum.

        Parameters
        ----------
        features : ndarray, shape (n, d)

        Returns
        -------
        float in (0, 1]
        """
        s = np.linalg.svd(features, compute_uv=False)
        nuc = s.sum()
        fro = np.sqrt(np.sum(s ** 2))
        if fro < 1e-12:
            return 0.0
        # Normalise so 1 = maximally redundant, 0 = maximally diverse
        d = min(features.shape)
        ratio = nuc / fro  # in [1, sqrt(d)]
        return float((ratio - 1.0) / (np.sqrt(d) - 1.0)) if d > 1 else 0.0


class NeuralCollapseDetector:
    """Detect neural collapse (NC1–NC4) in last-layer features."""

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def class_means(self, features, labels):
        """Per-class mean features.

        Parameters
        ----------
        features : ndarray, shape (n, d)
        labels : ndarray, shape (n,)

        Returns
        -------
        ndarray, shape (C, d)
        """
        classes = np.unique(labels)
        means = np.zeros((len(classes), features.shape[1]))
        for i, c in enumerate(classes):
            means[i] = features[labels == c].mean(axis=0)
        return means

    def within_class_variance(self, features, labels):
        """Within-class covariance Σ_W.

        Parameters
        ----------
        features : ndarray, shape (n, d)
        labels : ndarray, shape (n,)

        Returns
        -------
        ndarray, shape (d, d)
        """
        classes = np.unique(labels)
        d = features.shape[1]
        Sw = np.zeros((d, d))
        means = self.class_means(features, labels)
        for i, c in enumerate(classes):
            Fc = features[labels == c] - means[i]
            Sw += Fc.T @ Fc
        Sw /= features.shape[0]
        return Sw

    def between_class_variance(self, features, labels):
        """Between-class covariance Σ_B.

        Parameters
        ----------
        features : ndarray, shape (n, d)
        labels : ndarray, shape (n,)

        Returns
        -------
        ndarray, shape (d, d)
        """
        classes = np.unique(labels)
        global_mean = features.mean(axis=0)
        d = features.shape[1]
        Sb = np.zeros((d, d))
        for i, c in enumerate(classes):
            nc = np.sum(labels == c)
            diff = (self.class_means(features, labels)[i] - global_mean).reshape(-1, 1)
            Sb += nc * (diff @ diff.T)
        Sb /= features.shape[0]
        return Sb

    def nc1_variability_collapse(self, features, labels):
        """NC1: tr(Σ_W Σ_B^+) → 0 indicates collapse.

        Parameters
        ----------
        features : ndarray, shape (n, d)
        labels : ndarray, shape (n,)

        Returns
        -------
        float  (lower = more collapsed)
        """
        Sw = self.within_class_variance(features, labels)
        Sb = self.between_class_variance(features, labels)
        Sb_pinv = np.linalg.pinv(Sb)
        return float(np.trace(Sw @ Sb_pinv) / self.n_classes)

    def nc2_convergence_to_simplex_etf(self, class_means_matrix):
        """NC2: Distance of centred class means to simplex ETF.

        The simplex ETF in C classes has M^T M ∝ I - 11^T/C.

        Parameters
        ----------
        class_means_matrix : ndarray, shape (C, d)

        Returns
        -------
        float  (0 = perfect simplex ETF)
        """
        H = class_means_matrix - class_means_matrix.mean(axis=0)
        norms = np.linalg.norm(H, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        H_normed = H / norms
        G = H_normed @ H_normed.T
        C = G.shape[0]
        target = np.eye(C) - np.ones((C, C)) / C
        # Normalise target to have same scale
        target *= (C / (C - 1)) if C > 1 else 1.0
        # Actually the ideal gram for simplex ETF normalised rows is:
        # G_ideal[i,j] = -1/(C-1) for i!=j, 1 for i==j
        G_ideal = np.eye(C) - np.ones((C, C)) / (C - 1) + np.eye(C) / (C - 1)
        # Simplify: G_ideal = C/(C-1) * I - 1/(C-1) * 11^T
        G_ideal = (C / (C - 1)) * np.eye(C) - (1 / (C - 1)) * np.ones((C, C))
        return float(np.linalg.norm(G - G_ideal, 'fro') / np.linalg.norm(G_ideal, 'fro'))

    def nc3_self_duality(self, class_means_matrix, classifier_weights):
        """NC3: Classifier weights W should be proportional to class means H.

        Measures cosine similarity between normalised W and normalised H.

        Parameters
        ----------
        class_means_matrix : ndarray, shape (C, d)
        classifier_weights : ndarray, shape (C, d)

        Returns
        -------
        float in [-1, 1]  (1 = perfect self-duality)
        """
        H = class_means_matrix - class_means_matrix.mean(axis=0)
        W = classifier_weights - classifier_weights.mean(axis=0)
        H_flat = H.ravel()
        W_flat = W.ravel()
        nH = np.linalg.norm(H_flat)
        nW = np.linalg.norm(W_flat)
        if nH < 1e-12 or nW < 1e-12:
            return 0.0
        return float(H_flat @ W_flat / (nH * nW))

    def nc4_simplification_to_nearest_class(self, features, class_means_matrix, labels):
        """NC4: Nearest-class-center classification accuracy.

        Parameters
        ----------
        features : ndarray, shape (n, d)
        class_means_matrix : ndarray, shape (C, d)
        labels : ndarray, shape (n,)

        Returns
        -------
        float in [0, 1]
        """
        classes = np.unique(labels)
        dists = spatial.distance.cdist(features, class_means_matrix)
        preds = classes[np.argmin(dists, axis=1)]
        return float(np.mean(preds == labels))

    def neural_collapse_metrics(self, features, labels, classifier_weights):
        """Compute all four NC metrics.

        Parameters
        ----------
        features : ndarray, shape (n, d)
        labels : ndarray, shape (n,)
        classifier_weights : ndarray, shape (C, d)

        Returns
        -------
        dict with keys nc1, nc2, nc3, nc4
        """
        means = self.class_means(features, labels)
        return {
            'nc1': self.nc1_variability_collapse(features, labels),
            'nc2': self.nc2_convergence_to_simplex_etf(means),
            'nc3': self.nc3_self_duality(means, classifier_weights),
            'nc4': self.nc4_simplification_to_nearest_class(features, means, labels),
        }

    def collapse_trajectory(self, feature_snapshots, labels):
        """Compute NC1 metric over training snapshots.

        Parameters
        ----------
        feature_snapshots : list of ndarray, shape (n, d)
        labels : ndarray, shape (n,)

        Returns
        -------
        dict with keys:
            nc1_trajectory : ndarray (T,)
            nc2_trajectory : ndarray (T,)
        """
        nc1_vals = np.empty(len(feature_snapshots))
        nc2_vals = np.empty(len(feature_snapshots))
        for i, F in enumerate(feature_snapshots):
            nc1_vals[i] = self.nc1_variability_collapse(F, labels)
            means = self.class_means(F, labels)
            nc2_vals[i] = self.nc2_convergence_to_simplex_etf(means)
        return {
            'nc1_trajectory': nc1_vals,
            'nc2_trajectory': nc2_vals,
        }
