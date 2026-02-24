"""Random feature analysis for kernel approximation and regression."""

import numpy as np
from scipy import linalg, stats, optimize
from scipy.spatial.distance import cdist


class RandomFeatureApproximation:
    """Random feature approximation to NTK."""

    def __init__(self, n_features, input_dim, activation='relu'):
        self.n_features = n_features
        self.input_dim = input_dim
        self.activation = activation
        self._rng = np.random.default_rng()
        self._activation_fn = self._get_activation(activation)

    def _get_activation(self, name):
        activations = {
            'relu': lambda x: np.maximum(x, 0),
            'tanh': np.tanh,
            'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
            'cos': np.cos,
            'erf': lambda x: np.vectorize(np.math.erf)(x),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]

    def generate_features(self, X, W, b=None):
        """Φ(X) = σ(X W^T + b). X: (n, d), W: (D, d), b: (D,)."""
        X = np.asarray(X, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        pre_activation = X @ W.T
        if b is not None:
            pre_activation += np.asarray(b, dtype=np.float64)[np.newaxis, :]
        return self._activation_fn(pre_activation)

    def random_feature_kernel(self, X1, X2, n_features):
        """K_RF = (1/D) Φ(X1) Φ(X2)^T using random weights."""
        W = self._rng.standard_normal((n_features, self.input_dim))
        b = self._rng.uniform(0, 2 * np.pi, size=n_features)
        phi1 = self.generate_features(X1, W, b)
        phi2 = self.generate_features(X2, W, b)
        return (1.0 / n_features) * (phi1 @ phi2.T)

    def kernel_approximation_error(self, K_exact, K_approx):
        """Relative Frobenius error: ||K - K̃||_F / ||K||_F."""
        K_exact = np.asarray(K_exact, dtype=np.float64)
        K_approx = np.asarray(K_approx, dtype=np.float64)
        norm_exact = linalg.norm(K_exact, 'fro')
        if norm_exact < 1e-15:
            return linalg.norm(K_approx, 'fro')
        return linalg.norm(K_exact - K_approx, 'fro') / norm_exact

    def convergence_with_features(self, X, n_features_range, K_exact):
        """Compute approximation error as a function of feature count D."""
        X = np.asarray(X, dtype=np.float64)
        K_exact = np.asarray(K_exact, dtype=np.float64)
        errors = []
        for D in n_features_range:
            K_approx = self.random_feature_kernel(X, X, D)
            err = self.kernel_approximation_error(K_exact, K_approx)
            errors.append(err)
        return np.array(errors)

    def optimal_n_features(self, X, K_exact, tolerance=0.01):
        """Find minimum D such that approximation error < tolerance via doubling."""
        X = np.asarray(X, dtype=np.float64)
        K_exact = np.asarray(K_exact, dtype=np.float64)
        D = max(1, self.input_dim)
        max_D = 100000
        while D <= max_D:
            K_approx = self.random_feature_kernel(X, X, D)
            err = self.kernel_approximation_error(K_exact, K_approx)
            if err < tolerance:
                return D
            D *= 2
        return max_D

    def orthogonal_features(self, X, n_features):
        """Orthogonal Random Features (ORF) via QR of Gaussian matrices."""
        X = np.asarray(X, dtype=np.float64)
        d = X.shape[1]
        n_blocks = int(np.ceil(n_features / d))
        W_blocks = []
        for _ in range(n_blocks):
            G = self._rng.standard_normal((d, d))
            Q, _ = linalg.qr(G, mode='economic')
            S = np.sqrt(self._rng.chisquare(d, size=d))
            W_blocks.append(S[:, np.newaxis] * Q)
        W = np.vstack(W_blocks)[:n_features]
        b = self._rng.uniform(0, 2 * np.pi, size=n_features)
        features = self.generate_features(X, W, b)
        return features / np.sqrt(n_features)

    def structured_features(self, X, n_features):
        """Structured random features using Hadamard-diagonal-Hadamard."""
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        # Pad to next power of 2
        d_pad = 1
        while d_pad < d:
            d_pad *= 2
        X_pad = np.zeros((n, d_pad))
        X_pad[:, :d] = X

        n_blocks = int(np.ceil(n_features / d_pad))
        all_features = []
        for _ in range(n_blocks):
            diag1 = self._rng.choice([-1, 1], size=d_pad).astype(np.float64)
            diag2 = self._rng.choice([-1, 1], size=d_pad).astype(np.float64)
            diag3 = self._rng.choice([-1, 1], size=d_pad).astype(np.float64)
            # Apply HD₁ H D₂ H D₃
            Z = X_pad * diag3[np.newaxis, :]
            Z = np.fft.fft(Z, axis=1).real / np.sqrt(d_pad)
            Z = Z * diag2[np.newaxis, :]
            Z = np.fft.fft(Z, axis=1).real / np.sqrt(d_pad)
            Z = Z * diag1[np.newaxis, :]
            all_features.append(Z)
        features = np.hstack(all_features)[:, :n_features]
        return self._activation_fn(features) / np.sqrt(n_features)

    def feature_importance(self, features, targets):
        """Feature importance via correlation magnitude with target."""
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64).ravel()
        D = features.shape[1]
        importances = np.zeros(D)
        t_centered = targets - targets.mean()
        t_norm = linalg.norm(t_centered)
        if t_norm < 1e-15:
            return importances
        for j in range(D):
            f_centered = features[:, j] - features[:, j].mean()
            f_norm = linalg.norm(f_centered)
            if f_norm > 1e-15:
                importances[j] = np.abs(f_centered @ t_centered) / (f_norm * t_norm)
        return importances

    def feature_selection(self, features, targets, k):
        """Select top-k features by importance."""
        importances = self.feature_importance(features, targets)
        top_k_idx = np.argsort(importances)[-k:][::-1]
        return features[:, top_k_idx], top_k_idx, importances[top_k_idx]

    def ntk_random_features(self, X, n_features, depth=2):
        """Random features approximating the NTK for a depth-L ReLU network.

        For depth L, NTK decomposes as sum of per-layer contributions.
        Each layer l contributes features from the derivative of activation
        composed with inner features.
        """
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        features_per_layer = n_features // depth
        all_features = []

        for l in range(depth):
            W = self._rng.standard_normal((features_per_layer, d)) / np.sqrt(d)
            b = self._rng.uniform(0, 2 * np.pi, size=features_per_layer)
            pre = X @ W.T + b[np.newaxis, :]
            # Layer l value features
            val_feat = self._activation_fn(pre)
            # Layer l derivative features (for gradient)
            if self.activation == 'relu':
                grad_feat = (pre > 0).astype(np.float64)
            elif self.activation == 'tanh':
                grad_feat = 1.0 - np.tanh(pre) ** 2
            else:
                grad_feat = (pre > 0).astype(np.float64)

            # NTK feature for layer l: elementwise product of gradient and input norm
            input_norms = linalg.norm(X, axis=1, keepdims=True)
            layer_feat = grad_feat * input_norms
            if l < depth - 1:
                layer_feat = layer_feat * np.sqrt(val_feat.shape[1])
            all_features.append(layer_feat / np.sqrt(n_features))

        return np.hstack(all_features)

    def gradient_features(self, X, params, network_fn):
        """Gradient-based features: Jacobian J(x) = ∂f(x;θ)/∂θ.

        Args:
            X: Input data (n, d)
            params: Flat parameter vector
            network_fn: Callable(x, params) -> scalar output
        Returns:
            Jacobian matrix (n, p) where p = len(params)
        """
        X = np.asarray(X, dtype=np.float64)
        params = np.asarray(params, dtype=np.float64)
        n = X.shape[0]
        p = len(params)
        eps = 1e-5
        jacobian = np.zeros((n, p))
        for j in range(p):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[j] += eps
            params_minus[j] -= eps
            f_plus = np.array([network_fn(X[i], params_plus) for i in range(n)])
            f_minus = np.array([network_fn(X[i], params_minus) for i in range(n)])
            jacobian[:, j] = (f_plus - f_minus) / (2 * eps)
        return jacobian


class FeatureDimensionEstimator:
    """Estimate required feature dimension for kernel approximation."""

    def __init__(self, target_error=0.01):
        self.target_error = target_error

    def estimate_dimension(self, kernel_eigenvalues, target_error=None):
        """Estimate D needed for ε-approximation based on kernel spectrum.

        Uses the result that error ~ sqrt(sum_{i>D} λ_i^2) / sum(λ_i).
        """
        target_error = target_error or self.target_error
        eigs = np.sort(np.asarray(kernel_eigenvalues, dtype=np.float64))[::-1]
        eigs = eigs[eigs > 0]
        total_trace = np.sum(eigs)
        if total_trace < 1e-15:
            return 1
        cumulative = np.cumsum(eigs)
        residuals = total_trace - cumulative
        relative_residuals = residuals / total_trace
        candidates = np.where(relative_residuals < target_error)[0]
        if len(candidates) == 0:
            return len(eigs)
        return int(candidates[0]) + 1

    def johnson_lindenstrauss_bound(self, n_points, epsilon):
        """JL lemma: D >= C * log(n) / ε^2."""
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError("epsilon must be in (0, 1)")
        C = 8.0  # constant from JL lemma
        return int(np.ceil(C * np.log(n_points) / (epsilon ** 2)))

    def effective_dimension_bound(self, kernel, regularization):
        """d_eff = tr(K (K + λI)^{-1}), and D >= O(d_eff log(d_eff))."""
        K = np.asarray(kernel, dtype=np.float64)
        n = K.shape[0]
        eigenvalues = linalg.eigvalsh(K)
        eigenvalues = np.maximum(eigenvalues, 0)
        d_eff = np.sum(eigenvalues / (eigenvalues + regularization))
        # Bound: D >= c * d_eff * log(d_eff / delta) for high probability
        D_bound = int(np.ceil(4.0 * d_eff * np.log(max(d_eff, 2))))
        return {'d_eff': d_eff, 'D_bound': D_bound, 'regularization': regularization}

    def empirical_estimation(self, X, n_features_range, K_exact):
        """Empirically estimate required D by fitting error decay curve."""
        X = np.asarray(X, dtype=np.float64)
        K_exact = np.asarray(K_exact, dtype=np.float64)
        rfa = RandomFeatureApproximation(1, X.shape[1])
        errors = rfa.convergence_with_features(X, n_features_range, K_exact)

        # Fit power law: error ~ a * D^(-b)
        log_D = np.log(np.array(n_features_range, dtype=np.float64))
        log_err = np.log(np.maximum(errors, 1e-15))
        mask = np.isfinite(log_err)
        if mask.sum() < 2:
            return {'estimated_D': n_features_range[-1], 'errors': errors}
        coeffs = np.polyfit(log_D[mask], log_err[mask], 1)
        rate = coeffs[0]
        intercept = coeffs[1]
        # Solve a * D^rate = target_error
        if rate >= 0:
            estimated_D = n_features_range[-1]
        else:
            estimated_D = int(np.ceil(np.exp(
                (np.log(self.target_error) - intercept) / rate
            )))
        return {
            'estimated_D': estimated_D,
            'decay_rate': rate,
            'errors': errors,
            'n_features_range': np.array(n_features_range),
        }

    def dimension_vs_accuracy_curve(self, X, K_exact, D_range):
        """Compute accuracy (1 - relative error) vs feature dimension."""
        X = np.asarray(X, dtype=np.float64)
        K_exact = np.asarray(K_exact, dtype=np.float64)
        rfa = RandomFeatureApproximation(1, X.shape[1])
        errors = rfa.convergence_with_features(X, D_range, K_exact)
        accuracies = 1.0 - errors
        return {'D_range': np.array(D_range), 'accuracies': accuracies, 'errors': errors}

    def spectral_decay_rate(self, eigenvalues):
        """Estimate eigenvalue decay rate: λ_k ~ k^{-α} or λ_k ~ exp(-αk)."""
        eigs = np.sort(np.asarray(eigenvalues, dtype=np.float64))[::-1]
        eigs = eigs[eigs > 1e-15]
        if len(eigs) < 3:
            return {'polynomial_rate': np.nan, 'exponential_rate': np.nan, 'best_fit': 'unknown'}

        k = np.arange(1, len(eigs) + 1, dtype=np.float64)
        log_k = np.log(k)
        log_eigs = np.log(eigs)

        # Polynomial fit: log(λ) = -α log(k) + c
        poly_coeffs = np.polyfit(log_k, log_eigs, 1)
        poly_rate = -poly_coeffs[0]
        poly_residual = np.sum((log_eigs - np.polyval(poly_coeffs, log_k)) ** 2)

        # Exponential fit: log(λ) = -α k + c
        exp_coeffs = np.polyfit(k, log_eigs, 1)
        exp_rate = -exp_coeffs[0]
        exp_residual = np.sum((log_eigs - np.polyval(exp_coeffs, k)) ** 2)

        best_fit = 'polynomial' if poly_residual < exp_residual else 'exponential'
        return {
            'polynomial_rate': poly_rate,
            'exponential_rate': exp_rate,
            'polynomial_residual': poly_residual,
            'exponential_residual': exp_residual,
            'best_fit': best_fit,
        }

    def required_features_for_task(self, kernel_eigenvalues, target_coefficients):
        """Task-dependent D: features needed depend on how target aligns with spectrum.

        If target f* = Σ c_k φ_k, error depends on tail Σ_{k>D} c_k^2 λ_k.
        """
        eigs = np.sort(np.asarray(kernel_eigenvalues, dtype=np.float64))[::-1]
        coeffs = np.asarray(target_coefficients, dtype=np.float64)
        min_len = min(len(eigs), len(coeffs))
        eigs = eigs[:min_len]
        coeffs = coeffs[:min_len]

        weighted = coeffs ** 2 * eigs
        total = np.sum(weighted)
        if total < 1e-15:
            return 1
        cumulative = np.cumsum(weighted)
        residuals = total - cumulative
        relative = residuals / total
        candidates = np.where(relative < self.target_error)[0]
        if len(candidates) == 0:
            return min_len
        return int(candidates[0]) + 1

    def overparameterization_threshold(self, n_train, eigenvalue_decay):
        """Estimate D* where interpolation occurs.

        At D* ≈ n, the system transitions from under- to over-parameterized.
        The eigenvalue decay influences the sharpness of this transition.
        """
        eigs = np.sort(np.asarray(eigenvalue_decay, dtype=np.float64))[::-1]
        decay_info = self.spectral_decay_rate(eigs)
        # Threshold is at D* ≈ n, modulated by spectral decay
        D_star = n_train
        if decay_info['best_fit'] == 'polynomial':
            alpha = decay_info['polynomial_rate']
            # Slower decay -> sharper transition
            sharpness = 1.0 / max(alpha, 0.1)
        else:
            sharpness = 1.0
        return {
            'D_star': D_star,
            'sharpness': sharpness,
            'spectral_decay': decay_info,
        }


class FeatureQualityMetric:
    """Metrics for evaluating quality of random features."""

    def __init__(self):
        pass

    def kernel_approximation_quality(self, K_exact, features):
        """Assess how well features approximate the kernel.

        K_approx = (1/D) Φ Φ^T, compared to K_exact.
        """
        features = np.asarray(features, dtype=np.float64)
        K_exact = np.asarray(K_exact, dtype=np.float64)
        D = features.shape[1]
        K_approx = (1.0 / D) * (features @ features.T)
        rel_error = linalg.norm(K_exact - K_approx, 'fro') / max(linalg.norm(K_exact, 'fro'), 1e-15)

        # Per-entry statistics
        diffs = (K_exact - K_approx).ravel()
        return {
            'relative_frobenius_error': rel_error,
            'max_entry_error': np.max(np.abs(diffs)),
            'mean_entry_error': np.mean(np.abs(diffs)),
            'entry_error_std': np.std(diffs),
            'spectral_norm_error': linalg.norm(K_exact - K_approx, 2),
        }

    def feature_coherence(self, features):
        """Maximum coherence: max_{i≠j} |⟨φᵢ, φⱼ⟩| / (||φᵢ|| ||φⱼ||)."""
        features = np.asarray(features, dtype=np.float64)
        D = features.shape[1]
        if D < 2:
            return 0.0
        norms = linalg.norm(features, axis=0)
        norms = np.maximum(norms, 1e-15)
        normalized = features / norms[np.newaxis, :]
        gram = normalized.T @ normalized
        np.fill_diagonal(gram, 0)
        return np.max(np.abs(gram))

    def feature_diversity(self, features):
        """Feature diversity measured by volume of feature parallelotope.

        Uses log-determinant of normalized Gram matrix.
        """
        features = np.asarray(features, dtype=np.float64)
        n, D = features.shape
        if D > n:
            G = features @ features.T
        else:
            G = features.T @ features
        G /= max(linalg.norm(G, 'fro'), 1e-15)
        eigvals = linalg.eigvalsh(G)
        eigvals = np.maximum(eigvals, 1e-15)
        log_det = np.sum(np.log(eigvals))
        # Normalize by dimension
        return log_det / len(eigvals)

    def feature_informativeness(self, features, targets):
        """Mutual information estimate I(Φ; y) via explained variance.

        Uses R^2 of optimal linear predictor as proxy.
        """
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64).ravel()
        n, D = features.shape

        # Compute R^2 using ridge regression with small regularization
        reg = 1e-6 * n
        FTF = features.T @ features + reg * np.eye(D)
        FTy = features.T @ targets
        try:
            w = linalg.solve(FTF, FTy, assume_a='pos')
        except linalg.LinAlgError:
            w = linalg.lstsq(features, targets)[0]
        predictions = features @ w
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - targets.mean()) ** 2)
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-15)
        r_squared = np.clip(r_squared, 0, 1)

        # Convert to mutual information estimate (Gaussian assumption)
        # I(X;Y) = -0.5 log(1 - R^2) for jointly Gaussian
        mi_estimate = -0.5 * np.log(max(1.0 - r_squared, 1e-15))
        return {'mutual_information': mi_estimate, 'r_squared': r_squared}

    def feature_stability(self, features_run1, features_run2):
        """Stability: how similar are features across two random seeds.

        Measures subspace angle between column spans.
        """
        F1 = np.asarray(features_run1, dtype=np.float64)
        F2 = np.asarray(features_run2, dtype=np.float64)
        k = min(F1.shape[1], F2.shape[1], F1.shape[0])

        U1, _, _ = linalg.svd(F1, full_matrices=False)
        U2, _, _ = linalg.svd(F2, full_matrices=False)
        U1 = U1[:, :k]
        U2 = U2[:, :k]

        # Canonical correlations = singular values of U1^T U2
        cos_angles = linalg.svdvals(U1.T @ U2)
        cos_angles = np.clip(cos_angles, 0, 1)
        mean_cos = np.mean(cos_angles)
        principal_angles = np.arccos(cos_angles)
        return {
            'mean_cosine_similarity': mean_cos,
            'principal_angles': principal_angles,
            'grassmann_distance': linalg.norm(principal_angles),
            'stability_score': mean_cos,
        }

    def restricted_isometry(self, features, s, n_trials=100):
        """Estimate RIP constant δ_s for the feature matrix.

        For all s-sparse x: (1-δ_s)||x||^2 <= ||Φx||^2 <= (1+δ_s)||x||^2.
        Estimated by random s-sparse vectors.
        """
        features = np.asarray(features, dtype=np.float64)
        n, D = features.shape
        # Normalize columns
        col_norms = linalg.norm(features, axis=0)
        col_norms = np.maximum(col_norms, 1e-15)
        Phi = features / col_norms[np.newaxis, :]

        rng = np.random.default_rng(42)
        max_distortion = 0.0
        for _ in range(n_trials):
            support = rng.choice(D, size=min(s, D), replace=False)
            x = np.zeros(D)
            x[support] = rng.standard_normal(len(support))
            x /= linalg.norm(x)
            Phi_x = Phi @ x
            energy = np.sum(Phi_x ** 2)
            # Ideally energy ≈ n/D (due to normalization)
            expected_energy = n / D
            distortion = np.abs(energy / max(expected_energy, 1e-15) - 1.0)
            max_distortion = max(max_distortion, distortion)
        return {'rip_constant': max_distortion, 'sparsity': s, 'n_trials': n_trials}

    def feature_condition_number(self, features):
        """Condition number of Φ^T Φ."""
        features = np.asarray(features, dtype=np.float64)
        sv = linalg.svdvals(features)
        sv = sv[sv > 1e-15]
        if len(sv) == 0:
            return np.inf
        return sv[0] / sv[-1]

    def leverage_scores(self, features):
        """Statistical leverage scores: h_i = [Φ (Φ^T Φ)^{-1} Φ^T]_{ii}."""
        features = np.asarray(features, dtype=np.float64)
        n, D = features.shape
        try:
            U, _, _ = linalg.svd(features, full_matrices=False)
            scores = np.sum(U ** 2, axis=1)
        except linalg.LinAlgError:
            # Fallback
            FTF_inv = linalg.pinv(features.T @ features)
            H = features @ FTF_inv @ features.T
            scores = np.diag(H)
        return scores

    def effective_feature_dimension(self, features):
        """Effective dimensionality: d_eff = (Σ σ_i)^2 / Σ σ_i^2."""
        features = np.asarray(features, dtype=np.float64)
        sv = linalg.svdvals(features)
        sv = sv[sv > 1e-15]
        if len(sv) == 0:
            return 0.0
        return (np.sum(sv)) ** 2 / np.sum(sv ** 2)

    def feature_rank(self, features, threshold=1e-10):
        """Numerical rank: number of singular values above threshold."""
        features = np.asarray(features, dtype=np.float64)
        sv = linalg.svdvals(features)
        return int(np.sum(sv > threshold))


class FeatureAlignmentAnalyzer:
    """Analyze alignment between features and target function."""

    def __init__(self):
        pass

    def feature_target_correlation(self, features, targets):
        """Pearson correlation of each feature column with the target."""
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64).ravel()
        D = features.shape[1]
        correlations = np.zeros(D)
        t_centered = targets - targets.mean()
        t_std = np.std(targets)
        if t_std < 1e-15:
            return correlations
        for j in range(D):
            f_centered = features[:, j] - features[:, j].mean()
            f_std = np.std(features[:, j])
            if f_std > 1e-15:
                correlations[j] = np.mean(f_centered * t_centered) / (f_std * t_std)
        return correlations

    def alignment_score(self, features, targets):
        """Overall alignment: ||Φ^T y||^2 / (||Φ||_F^2 ||y||^2).

        Measures how much of the target lies in the feature column space.
        """
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64).ravel()
        Phi_t_y = features.T @ targets
        numerator = np.sum(Phi_t_y ** 2)
        denominator = linalg.norm(features, 'fro') ** 2 * linalg.norm(targets) ** 2
        if denominator < 1e-15:
            return 0.0
        return numerator / denominator

    def principal_alignment(self, features, targets, k=10):
        """Alignment with top-k right singular vectors of features."""
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64).ravel()
        U, S, Vt = linalg.svd(features, full_matrices=False)
        k = min(k, len(S))
        U_k = U[:, :k]
        # Project target onto top-k left singular directions
        projections = U_k.T @ targets
        projection_norms_sq = projections ** 2
        target_norm_sq = linalg.norm(targets) ** 2
        if target_norm_sq < 1e-15:
            return {'alignments': np.zeros(k), 'cumulative': np.zeros(k), 'singular_values': S[:k]}
        alignments = projection_norms_sq / target_norm_sq
        cumulative = np.cumsum(alignments)
        return {
            'alignments': alignments,
            'cumulative': cumulative,
            'singular_values': S[:k],
            'total_explained': cumulative[-1] if k > 0 else 0.0,
        }

    def alignment_in_kernel_eigenbasis(self, features, kernel_eigenvectors, targets):
        """Alignment of target with kernel eigenvectors.

        Compute c_k = <φ_k, y> / ||y|| for each kernel eigenvector φ_k.
        """
        V = np.asarray(kernel_eigenvectors, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64).ravel()
        t_norm = linalg.norm(targets)
        if t_norm < 1e-15:
            return np.zeros(V.shape[1])
        coefficients = (V.T @ targets) / t_norm
        return coefficients

    def alignment_evolution(self, feature_snapshots, targets):
        """Track alignment score over a sequence of feature snapshots."""
        targets = np.asarray(targets, dtype=np.float64).ravel()
        scores = []
        for features in feature_snapshots:
            score = self.alignment_score(features, targets)
            scores.append(score)
        return np.array(scores)

    def optimal_feature_rotation(self, features, targets):
        """Procrustes alignment: find orthogonal R minimizing ||ΦR - Y||.

        Y is reshaped from targets to match feature dimension if needed.
        """
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if targets.ndim == 1:
            # Project target into feature space via pseudoinverse for Procrustes
            # Instead, find rotation maximizing alignment
            U, S, Vt = linalg.svd(features, full_matrices=False)
            # Coefficients of target in left singular vector basis
            coeffs = U.T @ targets
            # Optimal is to align largest singular direction with target
            target_direction = targets / max(linalg.norm(targets), 1e-15)
            feature_directions = U
            alignment_per_direction = feature_directions.T @ target_direction
            return {
                'alignment_per_sv_direction': alignment_per_direction,
                'singular_values': S,
                'max_alignment': np.max(np.abs(alignment_per_direction)),
            }
        # Multi-output case: standard Procrustes
        M = targets.T @ features
        U, _, Vt = linalg.svd(M)
        R = Vt.T @ U.T
        aligned = features @ R
        residual = linalg.norm(targets - aligned, 'fro')
        return {'rotation': R, 'aligned_features': aligned, 'residual': residual}

    def subspace_alignment(self, feature_subspace, target_subspace):
        """Alignment between two subspaces via principal angles.

        Inputs are matrices whose columns span the subspaces.
        """
        F = np.asarray(feature_subspace, dtype=np.float64)
        T = np.asarray(target_subspace, dtype=np.float64)
        # Orthonormalize
        QF, _ = linalg.qr(F, mode='economic')
        QT, _ = linalg.qr(T, mode='economic')
        cos_angles = linalg.svdvals(QF.T @ QT)
        cos_angles = np.clip(cos_angles, 0, 1)
        angles = np.arccos(cos_angles)
        # Alignment score: average cosine of principal angles
        score = np.mean(cos_angles)
        return {'principal_angles': angles, 'cosines': cos_angles, 'alignment_score': score}

    def alignment_vs_generalization(self, alignment_scores, test_errors):
        """Correlation between alignment scores and test errors."""
        a = np.asarray(alignment_scores, dtype=np.float64)
        e = np.asarray(test_errors, dtype=np.float64)
        if len(a) < 3:
            return {'pearson_r': np.nan, 'p_value': np.nan, 'spearman_r': np.nan}
        pearson_r, p_val = stats.pearsonr(a, e)
        spearman_r, sp_p = stats.spearmanr(a, e)
        return {
            'pearson_r': pearson_r,
            'pearson_p_value': p_val,
            'spearman_r': spearman_r,
            'spearman_p_value': sp_p,
        }

    def misalignment_directions(self, features, targets, k=5):
        """Find k directions of worst feature-target alignment.

        These are directions in feature space where target has little energy
        but features have large energy.
        """
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64).ravel()
        U, S, Vt = linalg.svd(features, full_matrices=False)
        # Target projection onto each left singular direction
        target_projections = np.abs(U.T @ targets)
        t_norm = max(linalg.norm(targets), 1e-15)
        # Misalignment: high singular value but low target projection
        misalignment = S / max(S[0], 1e-15) - target_projections / t_norm
        k = min(k, len(misalignment))
        worst_idx = np.argsort(misalignment)[-k:][::-1]
        return {
            'directions': Vt[worst_idx],
            'misalignment_scores': misalignment[worst_idx],
            'singular_values': S[worst_idx],
            'target_projections': target_projections[worst_idx],
        }

    def alignment_improvement_rate(self, alignment_trajectory):
        """Compute d(alignment)/dt from a trajectory of alignment scores."""
        traj = np.asarray(alignment_trajectory, dtype=np.float64)
        if len(traj) < 2:
            return {'rates': np.array([]), 'mean_rate': 0.0, 'acceleration': 0.0}
        rates = np.diff(traj)
        mean_rate = np.mean(rates)
        acceleration = 0.0
        if len(rates) >= 2:
            acceleration = np.mean(np.diff(rates))
        return {'rates': rates, 'mean_rate': mean_rate, 'acceleration': acceleration}


class RandomFeatureRegression:
    """Regression using random features."""

    def __init__(self, n_features, regularization=1e-6):
        self.n_features = n_features
        self.regularization = regularization
        self._rng = np.random.default_rng()
        self.W_ = None
        self.b_ = None
        self.weights_ = None
        self.input_dim_ = None

    def _generate_random_params(self, input_dim):
        """Generate and store random projection parameters."""
        self.input_dim_ = input_dim
        self.W_ = self._rng.standard_normal((self.n_features, input_dim)) / np.sqrt(input_dim)
        self.b_ = self._rng.uniform(0, 2 * np.pi, size=self.n_features)

    def _transform(self, X):
        """Apply random feature transform: cos(XW^T + b) * sqrt(2/D)."""
        Z = X @ self.W_.T + self.b_[np.newaxis, :]
        return np.cos(Z) * np.sqrt(2.0 / self.n_features)

    def fit(self, X_train, y_train):
        """Fit random feature ridge regression."""
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).ravel()
        self._generate_random_params(X_train.shape[1])
        Phi = self._transform(X_train)
        n, D = Phi.shape
        # Solve (Φ^TΦ + λI)w = Φ^Ty
        A = Phi.T @ Phi + self.regularization * np.eye(D)
        rhs = Phi.T @ y_train
        self.weights_ = linalg.solve(A, rhs, assume_a='pos')
        return self

    def predict(self, X_test):
        """Predict using fitted model."""
        X_test = np.asarray(X_test, dtype=np.float64)
        Phi = self._transform(X_test)
        return Phi @ self.weights_

    def ridgeless_fit(self, X_train, y_train):
        """Min-norm interpolation (λ → 0).

        If D > n: w = Φ^T (Φ Φ^T)^{-1} y (min-norm solution)
        If D <= n: w = (Φ^T Φ)^{-1} Φ^T y (least squares)
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).ravel()
        self._generate_random_params(X_train.shape[1])
        Phi = self._transform(X_train)
        n, D = Phi.shape

        if D >= n:
            # Over-parameterized: min-norm interpolation
            K = Phi @ Phi.T
            K += 1e-12 * np.eye(n)  # tiny regularization for stability
            alpha = linalg.solve(K, y_train, assume_a='pos')
            self.weights_ = Phi.T @ alpha
        else:
            # Under-parameterized: least squares
            self.weights_ = linalg.lstsq(Phi, y_train)[0]
        return self

    def double_descent_curve(self, X_train, y_train, X_test, y_test, n_features_range):
        """Compute test error vs number of features, showing double descent."""
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).ravel()
        X_test = np.asarray(X_test, dtype=np.float64)
        y_test = np.asarray(y_test, dtype=np.float64).ravel()
        n_train = X_train.shape[0]

        test_errors = []
        train_errors = []
        for D in n_features_range:
            self.n_features = D
            try:
                self.ridgeless_fit(X_train, y_train)
                pred_train = self.predict(X_train)
                pred_test = self.predict(X_test)
                train_err = np.mean((y_train - pred_train) ** 2)
                test_err = np.mean((y_test - pred_test) ** 2)
            except (linalg.LinAlgError, np.linalg.LinAlgError):
                train_err = np.nan
                test_err = np.nan
            train_errors.append(train_err)
            test_errors.append(test_err)

        return {
            'n_features_range': np.array(n_features_range),
            'test_errors': np.array(test_errors),
            'train_errors': np.array(train_errors),
            'interpolation_threshold': n_train,
        }

    def interpolation_threshold(self, n_train):
        """D* = n: the interpolation threshold."""
        return n_train

    def bias_variance_tradeoff(self, X_train, y_train, X_test, y_test,
                               n_features_range, n_trials=20):
        """Decompose test error into bias^2 + variance over random seeds."""
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).ravel()
        X_test = np.asarray(X_test, dtype=np.float64)
        y_test = np.asarray(y_test, dtype=np.float64).ravel()
        n_test = len(y_test)

        biases = []
        variances = []
        total_errors = []

        for D in n_features_range:
            predictions = np.zeros((n_trials, n_test))
            for trial in range(n_trials):
                self.n_features = D
                self._rng = np.random.default_rng(trial)
                try:
                    self.fit(X_train, y_train)
                    predictions[trial] = self.predict(X_test)
                except (linalg.LinAlgError, np.linalg.LinAlgError):
                    predictions[trial] = np.nan

            mean_pred = np.nanmean(predictions, axis=0)
            bias_sq = np.mean((mean_pred - y_test) ** 2)
            variance = np.mean(np.nanvar(predictions, axis=0))
            total = bias_sq + variance

            biases.append(bias_sq)
            variances.append(variance)
            total_errors.append(total)

        return {
            'n_features_range': np.array(n_features_range),
            'bias_squared': np.array(biases),
            'variance': np.array(variances),
            'total_error': np.array(total_errors),
        }

    def optimal_regularization(self, X_train, y_train, reg_range):
        """Find optimal regularization via Generalized Cross-Validation (GCV).

        GCV(λ) = (1/n) ||y - Φ w_λ||^2 / (1 - tr(H_λ)/n)^2
        where H_λ = Φ (Φ^TΦ + λI)^{-1} Φ^T is the hat matrix.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).ravel()
        n = X_train.shape[0]

        if self.W_ is None:
            self._generate_random_params(X_train.shape[1])
        Phi = self._transform(X_train)
        D = Phi.shape[1]

        # SVD of Phi for efficient computation over multiple λ
        U, S, Vt = linalg.svd(Phi, full_matrices=False)
        S2 = S ** 2

        gcv_scores = []
        for lam in reg_range:
            # Coefficients in SVD basis
            d_lambda = S2 / (S2 + lam)
            # Fitted values: y_hat = U diag(d_λ) U^T y
            Uty = U.T @ y_train
            y_hat = U @ (d_lambda * Uty)
            residual = y_train - y_hat
            # tr(H_λ) = sum(d_λ)
            tr_H = np.sum(d_lambda)
            denom = (1.0 - tr_H / n) ** 2
            if denom < 1e-15:
                gcv_scores.append(np.inf)
            else:
                gcv = np.mean(residual ** 2) / denom
                gcv_scores.append(gcv)

        gcv_scores = np.array(gcv_scores)
        best_idx = np.nanargmin(gcv_scores)
        return {
            'optimal_reg': reg_range[best_idx],
            'gcv_scores': gcv_scores,
            'reg_range': np.array(reg_range),
        }

    def learning_curve(self, X, y, n_train_range, n_features):
        """Test error as a function of training set size."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_total = X.shape[0]
        self.n_features = n_features

        test_errors = []
        train_errors = []
        # Use last 20% as test set
        n_test = max(int(0.2 * n_total), 1)
        idx = np.arange(n_total)
        self._rng.shuffle(idx)
        test_idx = idx[:n_test]
        remaining_idx = idx[n_test:]
        X_test, y_test = X[test_idx], y[test_idx]

        for n_train in n_train_range:
            n_train = min(n_train, len(remaining_idx))
            train_idx = remaining_idx[:n_train]
            X_train, y_train = X[train_idx], y[train_idx]
            self.fit(X_train, y_train)
            pred_test = self.predict(X_test)
            pred_train = self.predict(X_train)
            test_errors.append(np.mean((y_test - pred_test) ** 2))
            train_errors.append(np.mean((y_train - pred_train) ** 2))

        return {
            'n_train_range': np.array(n_train_range),
            'test_errors': np.array(test_errors),
            'train_errors': np.array(train_errors),
        }

    def generalization_bound(self, n_features, n_train, reg):
        """Theoretical generalization bound for random feature ridge regression.

        Bound ~ D/n + λ * ||f*||^2_H + noise terms.
        Returns approximate bound components.
        """
        complexity_term = n_features / n_train
        bias_term = reg  # proportional to regularization
        # Rademacher complexity bound
        rademacher = np.sqrt(2.0 * np.log(2.0 * n_features) / n_train)
        bound = complexity_term * (1.0 / reg if reg > 0 else 1.0) / n_train + bias_term
        return {
            'complexity_term': complexity_term,
            'bias_term': bias_term,
            'rademacher_complexity': rademacher,
            'total_bound': bound + rademacher,
        }

    def spectral_prediction(self, kernel_eigenvalues, target_coefficients,
                            n_features, n_train, reg):
        """Predict test error from kernel spectrum and target alignment.

        E[error] ≈ Σ_k c_k^2 λ_k (λ_k + γ)^{-2} [γ^2 + σ^2 λ_k / n]
        where γ = n * reg.
        """
        eigs = np.asarray(kernel_eigenvalues, dtype=np.float64)
        coeffs = np.asarray(target_coefficients, dtype=np.float64)
        min_len = min(len(eigs), len(coeffs))
        eigs = eigs[:min_len]
        coeffs = coeffs[:min_len]

        gamma = n_train * reg

        # Bias: Σ c_k^2 γ^2 / (λ_k + γ)^2
        bias = np.sum(coeffs ** 2 * gamma ** 2 / (eigs + gamma) ** 2)

        # Variance: (σ^2 / n) Σ λ_k / (λ_k + γ)^2
        # Assume unit noise variance
        sigma_sq = 1.0
        d_eff = np.sum(eigs / (eigs + gamma))
        variance = sigma_sq * d_eff / n_train

        # Approximation error from finite features
        # ~ Σ_{k > D} c_k^2 λ_k
        if n_features < min_len:
            approx_error = np.sum(coeffs[n_features:] ** 2 * eigs[n_features:])
        else:
            approx_error = 0.0

        total = bias + variance + approx_error
        return {
            'bias': bias,
            'variance': variance,
            'approximation_error': approx_error,
            'total_predicted_error': total,
            'effective_dimension': d_eff,
        }
