"""
Lazy (kernel) regime analysis for neural network training dynamics.

Implements NTK-based linearized dynamics, stability checking, kernel regression,
and detection of the lazy-to-rich regime transition.
"""

import numpy as np
from scipy import linalg
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, brentq


class LazyRegimeAnalyzer:
    """Analyze whether training dynamics remain in the lazy (kernel) regime."""

    def __init__(self, network_width, learning_rate=0.01):
        self.network_width = network_width
        self.learning_rate = learning_rate

    def is_lazy_regime(self, kernel_trajectory, tolerance=0.05):
        """Check if NTK stays approximately constant: ||Θ(t)-Θ(0)||_F / ||Θ(0)||_F < tol."""
        kernel_0 = kernel_trajectory[0]
        norm_0 = np.linalg.norm(kernel_0, 'fro')
        if norm_0 < 1e-15:
            return False
        for kernel_t in kernel_trajectory[1:]:
            rel_change = np.linalg.norm(kernel_t - kernel_0, 'fro') / norm_0
            if rel_change > tolerance:
                return False
        return True

    def lazy_regime_duration(self, kernel_trajectory, times):
        """Return the time at which the kernel first departs from its initial value.

        Uses linear interpolation between snapshots to estimate exact departure time.
        """
        kernel_0 = kernel_trajectory[0]
        norm_0 = np.linalg.norm(kernel_0, 'fro')
        if norm_0 < 1e-15:
            return times[0]

        threshold = 0.05
        prev_change = 0.0
        for i in range(1, len(kernel_trajectory)):
            rel_change = np.linalg.norm(kernel_trajectory[i] - kernel_0, 'fro') / norm_0
            if rel_change > threshold:
                # Interpolate between previous and current snapshot
                if rel_change - prev_change > 1e-15:
                    frac = (threshold - prev_change) / (rel_change - prev_change)
                else:
                    frac = 0.0
                return times[i - 1] + frac * (times[i] - times[i - 1])
            prev_change = rel_change
        return times[-1]

    def kernel_relative_change(self, kernel_t, kernel_0):
        """Compute ||Θ(t) - Θ(0)||_F / ||Θ(0)||_F."""
        norm_0 = np.linalg.norm(kernel_0, 'fro')
        if norm_0 < 1e-15:
            return np.inf
        return np.linalg.norm(kernel_t - kernel_0, 'fro') / norm_0

    def lazy_regime_conditions(self, width, learning_rate, init_scale):
        """Check sufficient conditions for lazy regime.

        Conditions (from NTK theory):
        1. Width m >> n (number of samples) — approximated by width > 100
        2. Learning rate η ~ 1/m
        3. Initialization scale ~ 1/√m
        Returns dict with each condition's status and an overall flag.
        """
        effective_lr = learning_rate * width
        expected_init_scale = 1.0 / np.sqrt(width)
        scale_ratio = init_scale / expected_init_scale if expected_init_scale > 0 else np.inf

        cond_width = width >= 100
        cond_lr = effective_lr < 10.0
        cond_init = 0.1 < scale_ratio < 10.0

        return {
            'width_sufficient': cond_width,
            'lr_compatible': cond_lr,
            'init_scale_compatible': cond_init,
            'all_satisfied': cond_width and cond_lr and cond_init,
            'effective_lr': effective_lr,
            'scale_ratio': scale_ratio,
        }

    def effective_ridge(self, kernel, regularization):
        """Compute effective ridge regularization: eigenvalues of K + λI.

        Returns the eigenvalues of the regularized kernel and the effective
        degrees of freedom tr(K(K + λI)^{-1}).
        """
        eigvals = np.linalg.eigvalsh(kernel)
        reg_eigvals = eigvals + regularization
        dof = np.sum(eigvals / reg_eigvals)
        return {
            'regularized_eigenvalues': reg_eigvals,
            'effective_dof': dof,
            'condition_number': reg_eigvals[-1] / max(reg_eigvals[0], 1e-15),
        }

    def generalization_in_lazy(self, train_kernel, test_kernel, targets, reg=1e-6):
        """Kernel ridge regression prediction in the lazy regime.

        f_test = K_test_train (K_train + λI)^{-1} y
        """
        n = train_kernel.shape[0]
        K_reg = train_kernel + reg * np.eye(n)
        alpha = linalg.solve(K_reg, targets, assume_a='pos')
        predictions = test_kernel @ alpha

        # Training residual
        train_pred = train_kernel @ alpha
        train_mse = np.mean((train_pred - targets) ** 2)

        return {
            'predictions': predictions,
            'alpha': alpha,
            'train_mse': train_mse,
        }


class NTKStabilityChecker:
    """Verify that the neural tangent kernel remains stable during training."""

    def __init__(self, stability_threshold=0.1):
        self.stability_threshold = stability_threshold

    def check_stability(self, kernel_snapshots, times):
        """Monitor NTK stability over time.

        Returns per-snapshot relative changes and flags instability events.
        """
        kernel_0 = kernel_snapshots[0]
        norm_0 = np.linalg.norm(kernel_0, 'fro')
        if norm_0 < 1e-15:
            return {
                'stable': False,
                'relative_changes': np.full(len(times), np.inf),
                'instability_time': times[0],
            }

        rel_changes = np.zeros(len(kernel_snapshots))
        instability_time = None
        for i in range(1, len(kernel_snapshots)):
            rel_changes[i] = np.linalg.norm(kernel_snapshots[i] - kernel_0, 'fro') / norm_0
            if rel_changes[i] > self.stability_threshold and instability_time is None:
                instability_time = times[i]

        return {
            'stable': instability_time is None,
            'relative_changes': rel_changes,
            'instability_time': instability_time,
            'max_change': np.max(rel_changes),
        }

    def spectral_stability(self, kernel_snapshots):
        """Check stability of eigenvalue spectrum across snapshots.

        Computes max relative eigenvalue change for each snapshot.
        """
        eigvals_0 = np.sort(np.linalg.eigvalsh(kernel_snapshots[0]))[::-1]
        max_eigval_0 = np.max(np.abs(eigvals_0))
        if max_eigval_0 < 1e-15:
            return {'stable': False, 'eigenvalue_changes': [], 'eigenvalues_initial': eigvals_0}

        eigval_changes = []
        for snap in kernel_snapshots[1:]:
            eigvals_t = np.sort(np.linalg.eigvalsh(snap))[::-1]
            change = np.max(np.abs(eigvals_t - eigvals_0)) / max_eigval_0
            eigval_changes.append(change)

        eigval_changes = np.array(eigval_changes)
        return {
            'stable': np.all(eigval_changes < self.stability_threshold),
            'eigenvalue_changes': eigval_changes,
            'eigenvalues_initial': eigvals_0,
            'max_spectral_change': np.max(eigval_changes) if len(eigval_changes) > 0 else 0.0,
        }

    def trace_stability(self, kernel_snapshots):
        """Check tr(Θ(t)) / tr(Θ(0)) stays near 1."""
        trace_0 = np.trace(kernel_snapshots[0])
        if np.abs(trace_0) < 1e-15:
            return {'stable': False, 'trace_ratios': np.array([]), 'trace_initial': trace_0}

        trace_ratios = np.array([np.trace(k) / trace_0 for k in kernel_snapshots])
        deviations = np.abs(trace_ratios - 1.0)

        return {
            'stable': np.all(deviations < self.stability_threshold),
            'trace_ratios': trace_ratios,
            'max_deviation': np.max(deviations),
            'trace_initial': trace_0,
        }

    def operator_norm_change(self, kernel_t, kernel_0):
        """Compute ||Θ(t) - Θ(0)||_op (largest singular value of the difference)."""
        diff = kernel_t - kernel_0
        return np.linalg.norm(diff, 2)

    def perturbation_bound(self, width, depth, learning_rate):
        """Theoretical upper bound on NTK change from perturbation theory.

        For networks with width m, depth L, learning rate η:
        ||Θ(t) - Θ(0)|| ≤ C · L² · η · t / √m
        where C is a universal constant (set to 1 for normalized bound).
        """
        C = 1.0
        bound_per_unit_time = C * depth ** 2 * learning_rate / np.sqrt(width)
        return {
            'bound_per_unit_time': bound_per_unit_time,
            'critical_time': self.stability_threshold / max(bound_per_unit_time, 1e-15),
            'width_scaling': 1.0 / np.sqrt(width),
            'depth_scaling': depth ** 2,
        }

    def stability_margin(self, kernel_trajectory):
        """Compute distance from the instability boundary.

        margin = threshold - max_relative_change. Positive = still stable.
        """
        kernel_0 = kernel_trajectory[0]
        norm_0 = np.linalg.norm(kernel_0, 'fro')
        if norm_0 < 1e-15:
            return -np.inf

        max_change = 0.0
        for kernel_t in kernel_trajectory[1:]:
            change = np.linalg.norm(kernel_t - kernel_0, 'fro') / norm_0
            max_change = max(max_change, change)

        return self.stability_threshold - max_change


class LinearizedDynamicsSolver:
    """Solve the linearized (NTK) training dynamics.

    In the lazy regime, f(t) evolves as:
        df/dt = -Θ (f(t) - y)
    with solution f(t) = y - exp(-Θt)(y - f(0)).
    """

    def __init__(self, kernel_matrix, targets, regularization=0.0):
        self.kernel = kernel_matrix
        self.targets = targets
        self.regularization = regularization
        n = kernel_matrix.shape[0]
        self.K_reg = kernel_matrix + regularization * np.eye(n)

        # Eigendecompose for efficient dynamics
        self.eigvals, self.eigvecs = np.linalg.eigh(self.K_reg)
        # Clamp small eigenvalues for numerical stability
        self.eigvals = np.maximum(self.eigvals, 1e-15)

    def solve_continuous(self, t_span, n_points=1000):
        """Solve continuous-time linearized dynamics.

        f(t) = y - exp(-Θt)(y - f₀)  where f₀ = 0 (zero initialization).
        """
        times = np.linspace(t_span[0], t_span[1], n_points)
        n = len(self.targets)

        # Project targets into eigenbasis
        y_proj = self.eigvecs.T @ self.targets  # (n,)

        # f(t) in eigenbasis: y_proj_k * (1 - exp(-λ_k * t))
        predictions = np.zeros((n_points, n))
        residuals = np.zeros((n_points, n))
        for i, t in enumerate(times):
            decay = np.exp(-self.eigvals * t)
            f_proj = y_proj * (1.0 - decay)
            predictions[i] = self.eigvecs @ f_proj
            r_proj = y_proj * decay
            residuals[i] = self.eigvecs @ r_proj

        losses = np.array([np.mean(r ** 2) for r in residuals])

        return {
            'times': times,
            'predictions': predictions,
            'residuals': residuals,
            'losses': losses,
        }

    def solve_discrete(self, learning_rate, n_steps):
        """Solve discrete gradient descent dynamics.

        f_{t+1} = f_t - η Θ (f_t - y)
        In eigenbasis: r_k(t+1) = (1 - η λ_k) r_k(t)
        """
        y_proj = self.eigvecs.T @ self.targets
        n = len(self.targets)

        predictions = np.zeros((n_steps + 1, n))
        residuals = np.zeros((n_steps + 1, n))
        losses = np.zeros(n_steps + 1)

        # Initial residual r(0) = -y (since f(0)=0)
        r_proj = y_proj.copy()
        residuals[0] = self.eigvecs @ r_proj
        losses[0] = np.mean(residuals[0] ** 2)

        step_factors = 1.0 - learning_rate * self.eigvals

        # Check stability: |1 - η λ_k| < 1 for all k
        stable = np.all(np.abs(step_factors) < 1.0)

        for t in range(1, n_steps + 1):
            r_proj = r_proj * step_factors
            residuals[t] = self.eigvecs @ r_proj
            predictions[t] = self.targets - residuals[t]
            losses[t] = np.mean(residuals[t] ** 2)

        return {
            'predictions': predictions,
            'residuals': residuals,
            'losses': losses,
            'stable': stable,
            'step_factors': step_factors,
        }

    def residual_dynamics(self, t):
        """Compute residual r(t) = exp(-Θt) r(0), where r(0) = y (assuming f₀=0)."""
        y_proj = self.eigvecs.T @ self.targets
        decay = np.exp(-self.eigvals * t)
        r_proj = y_proj * decay
        return self.eigvecs @ r_proj

    def mode_decomposition(self, residual_0):
        """Decompose residual into kernel eigenmodes.

        residual_0 = Σ_k c_k v_k where v_k are eigenvectors of Θ.
        Returns coefficients, eigenvalues, and fraction of variance per mode.
        """
        coeffs = self.eigvecs.T @ residual_0
        mode_variances = coeffs ** 2
        total_var = np.sum(mode_variances)
        fractions = mode_variances / max(total_var, 1e-15)

        return {
            'coefficients': coeffs,
            'eigenvalues': self.eigvals,
            'mode_variances': mode_variances,
            'variance_fractions': fractions,
            'cumulative_fraction': np.cumsum(np.sort(fractions)[::-1]),
        }

    def convergence_per_mode(self, eigenvalues, t):
        """Compute convergence factor exp(-λ_k t) for each eigenmode."""
        return np.exp(-eigenvalues * t)

    def total_convergence_time(self, eigenvalues, tolerance=1e-6):
        """Time for slowest mode to reach tolerance: t* = -log(tol) / λ_min."""
        lambda_min = np.min(eigenvalues[eigenvalues > 1e-15])
        if lambda_min <= 0:
            return np.inf
        return -np.log(tolerance) / lambda_min

    def prediction_at_test(self, test_kernel, t):
        """Out-of-sample prediction at time t.

        f_test(t) = K_test_train Θ^{-1} (I - exp(-Θt)) y
        Using eigendecomposition for stability.
        """
        y_proj = self.eigvecs.T @ self.targets
        decay = np.exp(-self.eigvals * t)
        # Θ^{-1}(I - exp(-Θt)) in eigenbasis: (1 - exp(-λt))/λ
        filter_factors = (1.0 - decay) / self.eigvals
        filtered = self.eigvecs @ (filter_factors * y_proj)
        return test_kernel @ filtered

    def bias_variance_decomposition(self, test_kernel, noise_var, t):
        """Bias-variance decomposition at time t for noisy targets.

        bias²  = ||K_*Θ^{-1}exp(-Θt) y_true||² (signal not yet learned)
        variance = σ² tr(K_* Θ^{-1}(I-exp(-Θt)) ... )
        Simplified using kernel eigenstructure.
        """
        n = self.kernel.shape[0]
        decay = np.exp(-self.eigvals * t)
        learned = 1.0 - decay

        # Bias: signal in modes not yet learned
        y_proj = self.eigvecs.T @ self.targets
        bias_per_mode = (y_proj * decay) ** 2
        bias_sq = np.sum(bias_per_mode) / n

        # Variance: noise amplified through learned modes
        # var = σ² Σ_k (learned_k / λ_k)² λ_k  (simplified for diagonal structure)
        var_per_mode = noise_var * (learned / self.eigvals) ** 2 * self.eigvals
        variance = np.sum(var_per_mode) / n

        return {
            'bias_squared': bias_sq,
            'variance': variance,
            'total_risk': bias_sq + variance,
            'bias_per_mode': bias_per_mode,
            'variance_per_mode': var_per_mode,
            'optimal_time': self._optimal_stopping_time(noise_var),
        }

    def _optimal_stopping_time(self, noise_var):
        """Find optimal early stopping time minimizing bias² + variance."""
        y_proj = self.eigvecs.T @ self.targets
        n = len(self.targets)

        def risk(log_t):
            t = np.exp(log_t)
            decay = np.exp(-self.eigvals * t)
            learned = 1.0 - decay
            bias_sq = np.sum((y_proj * decay) ** 2) / n
            var = noise_var * np.sum((learned / self.eigvals) ** 2 * self.eigvals) / n
            return bias_sq + var

        result = minimize_scalar(risk, bounds=(-5, 10), method='bounded')
        return np.exp(result.x)


class KernelRegressionPredictor:
    """Kernel regression predictor for the lazy regime infinite-time limit."""

    def __init__(self, regularization=1e-6):
        self.regularization = regularization
        self.alpha_ = None
        self.kernel_train_ = None

    def fit(self, kernel_train, targets):
        """Solve α = (K + λI)^{-1} y via Cholesky decomposition."""
        n = kernel_train.shape[0]
        K_reg = kernel_train + self.regularization * np.eye(n)
        try:
            L = linalg.cholesky(K_reg, lower=True)
            self.alpha_ = linalg.cho_solve((L, True), targets)
        except linalg.LinAlgError:
            # Fall back to least-squares if not positive definite
            self.alpha_ = linalg.lstsq(K_reg, targets)[0]
        self.kernel_train_ = kernel_train
        return self

    def predict(self, kernel_test_train):
        """Predict f(x) = K(x, X) α."""
        if self.alpha_ is None:
            raise RuntimeError("Must call fit() before predict().")
        return kernel_test_train @ self.alpha_

    def leave_one_out_error(self, kernel_train, targets):
        """Efficient LOO error using the identity:

        LOO_i = (α_i / [C^{-1}]_{ii})²
        where C = K + λI, α = C^{-1} y.
        """
        n = kernel_train.shape[0]
        K_reg = kernel_train + self.regularization * np.eye(n)
        C_inv = np.linalg.inv(K_reg)
        alpha = C_inv @ targets

        diag_C_inv = np.diag(C_inv)
        diag_C_inv = np.where(np.abs(diag_C_inv) < 1e-15, 1e-15, diag_C_inv)
        loo_residuals = alpha / diag_C_inv
        loo_mse = np.mean(loo_residuals ** 2)

        return {
            'loo_mse': loo_mse,
            'loo_residuals': loo_residuals,
            'loo_predictions': targets - loo_residuals,
        }

    def generalization_bound(self, kernel_train, targets, n_test):
        """Rademacher-complexity-based generalization bound for kernel regression.

        bound = train_error + 2 R_n(H) / √n + √(log(1/δ)/(2n))
        where R_n ≤ √(tr(K)/n) · ||α||.
        """
        n = kernel_train.shape[0]
        self.fit(kernel_train, targets)
        train_pred = kernel_train @ self.alpha_
        train_error = np.mean((train_pred - targets) ** 2)

        trace_K = np.trace(kernel_train)
        alpha_norm = np.linalg.norm(self.alpha_)
        rademacher = np.sqrt(trace_K / n) * alpha_norm / np.sqrt(n)

        delta = 0.05
        concentration = np.sqrt(np.log(1.0 / delta) / (2 * n))

        return {
            'train_error': train_error,
            'rademacher_complexity': rademacher,
            'generalization_bound': train_error + 2 * rademacher + concentration,
            'alpha_norm': alpha_norm,
            'effective_complexity': trace_K / n,
        }

    def effective_dimension(self, kernel_train, regularization):
        """Effective dimension d_eff = tr(K (K + λI)^{-1})."""
        eigvals = np.linalg.eigvalsh(kernel_train)
        eigvals = np.maximum(eigvals, 0.0)
        d_eff = np.sum(eigvals / (eigvals + regularization))
        return d_eff

    def optimal_regularization(self, kernel_train, targets):
        """Find optimal λ via generalized cross-validation (GCV).

        GCV(λ) = (1/n) ||y - Ŝ_λ y||² / (1 - tr(Ŝ_λ)/n)²
        where Ŝ_λ = K(K + λI)^{-1}.
        """
        n = kernel_train.shape[0]
        eigvals, eigvecs = np.linalg.eigh(kernel_train)
        eigvals = np.maximum(eigvals, 0.0)
        y_proj = eigvecs.T @ targets

        def gcv(log_lam):
            lam = np.exp(log_lam)
            smoother_eigvals = eigvals / (eigvals + lam)
            # Residual in eigenbasis
            resid_proj = y_proj * (1.0 - smoother_eigvals)
            resid_norm_sq = np.sum(resid_proj ** 2)
            trace_S = np.sum(smoother_eigvals)
            denom = (1.0 - trace_S / n) ** 2
            if denom < 1e-15:
                return np.inf
            return resid_norm_sq / (n * denom)

        result = minimize_scalar(gcv, bounds=(-15, 5), method='bounded')
        optimal_lam = np.exp(result.x)

        return {
            'optimal_lambda': optimal_lam,
            'gcv_score': result.fun,
            'effective_dof': self.effective_dimension(kernel_train, optimal_lam),
        }

    def spectral_learning_curve(self, eigenvalues, n_train_range):
        """Predicted learning curve from kernel eigenspectrum.

        For kernel with eigenvalues λ_1 ≥ ... ≥ λ_p, the expected test error
        at training size n scales as:
        E[error] ≈ σ² d_eff(n)/n + Σ_{k: λ_k < threshold} c_k²
        Simplified model assuming uniform target in eigenspace.
        """
        eigenvalues = np.sort(eigenvalues)[::-1]
        p = len(eigenvalues)
        total_signal = np.sum(eigenvalues)

        errors = []
        for n in n_train_range:
            # Effective regularization from finite sample: λ_eff ≈ total_signal / n
            lam_eff = total_signal / max(n, 1)
            d_eff = np.sum(eigenvalues / (eigenvalues + lam_eff))
            # Variance term
            variance = d_eff / max(n, 1)
            # Bias term: modes with eigenvalue < lam_eff are not learned
            bias = np.sum(np.minimum(eigenvalues, lam_eff)) / total_signal if total_signal > 0 else 0
            errors.append(bias + variance)

        return {
            'n_train': np.array(n_train_range),
            'predicted_errors': np.array(errors),
            'eigenvalues': eigenvalues,
        }


class LazyToRichTransitionDetector:
    """Detect and characterize the transition from lazy to rich/feature-learning regime."""

    def __init__(self, detection_threshold=0.2):
        self.detection_threshold = detection_threshold

    def detect_transition_time(self, kernel_trajectory, times):
        """Find transition time t* where kernel change exceeds threshold.

        Uses linear interpolation for sub-snapshot resolution.
        """
        kernel_0 = kernel_trajectory[0]
        norm_0 = np.linalg.norm(kernel_0, 'fro')
        if norm_0 < 1e-15:
            return {'transition_time': times[0], 'detected': True, 'index': 0}

        prev_change = 0.0
        for i in range(1, len(kernel_trajectory)):
            change = np.linalg.norm(kernel_trajectory[i] - kernel_0, 'fro') / norm_0
            if change > self.detection_threshold:
                if change - prev_change > 1e-15:
                    frac = (self.detection_threshold - prev_change) / (change - prev_change)
                else:
                    frac = 0.0
                t_star = times[i - 1] + frac * (times[i] - times[i - 1])
                return {'transition_time': t_star, 'detected': True, 'index': i}
            prev_change = change

        return {'transition_time': None, 'detected': False, 'index': None}

    def transition_sharpness(self, kernel_trajectory, times):
        """Measure sharpness of the lazy→rich transition.

        Computes the derivative of kernel change at the transition point.
        Sharp transition = large derivative; smooth crossover = small derivative.
        """
        kernel_0 = kernel_trajectory[0]
        norm_0 = np.linalg.norm(kernel_0, 'fro')
        if norm_0 < 1e-15:
            return {'sharpness': np.inf, 'change_rate': np.array([])}

        changes = np.array([
            np.linalg.norm(k - kernel_0, 'fro') / norm_0 for k in kernel_trajectory
        ])

        # Numerical derivative of the change curve
        dt = np.diff(times)
        dt = np.where(dt < 1e-15, 1e-15, dt)
        change_rate = np.diff(changes) / dt

        # Find maximum rate of change (sharpness)
        if len(change_rate) == 0:
            return {'sharpness': 0.0, 'change_rate': change_rate, 'changes': changes}

        peak_idx = np.argmax(change_rate)
        sharpness = change_rate[peak_idx]

        return {
            'sharpness': sharpness,
            'peak_time': times[peak_idx],
            'change_rate': change_rate,
            'changes': changes,
        }

    def control_parameter_for_transition(self, widths, depths, learning_rates):
        """Estimate the control parameter that governs the lazy-rich transition.

        The control parameter is α = η · L² / m, where large α → rich, small α → lazy.
        Returns α for each configuration.
        """
        widths = np.asarray(widths, dtype=float)
        depths = np.asarray(depths, dtype=float)
        learning_rates = np.asarray(learning_rates, dtype=float)

        alpha = learning_rates * depths ** 2 / widths

        return {
            'control_parameter': alpha,
            'lazy_regime': alpha < 0.1,
            'rich_regime': alpha > 1.0,
            'transition_zone': (alpha >= 0.1) & (alpha <= 1.0),
        }

    def phase_boundary(self, width_range, lr_range, metric_fn):
        """Compute the lazy-rich phase boundary in (width, lr) space.

        For each width, finds the critical learning rate where
        metric_fn(width, lr) crosses the detection threshold.

        metric_fn(width, lr) -> float: should return the kernel change metric.
        """
        boundary_lrs = []
        valid_widths = []

        for w in width_range:
            # Evaluate metric across learning rates
            metrics = np.array([metric_fn(w, lr) for lr in lr_range])

            # Find crossing of threshold
            above = metrics > self.detection_threshold
            if not np.any(above):
                # Always lazy — boundary is beyond lr_range
                boundary_lrs.append(lr_range[-1])
                valid_widths.append(w)
                continue
            if np.all(above):
                # Always rich — boundary is below lr_range
                boundary_lrs.append(lr_range[0])
                valid_widths.append(w)
                continue

            # Find first crossing
            cross_idx = np.argmax(above)
            if cross_idx == 0:
                boundary_lrs.append(lr_range[0])
            else:
                # Linear interpolation
                m0 = metrics[cross_idx - 1]
                m1 = metrics[cross_idx]
                frac = (self.detection_threshold - m0) / max(m1 - m0, 1e-15)
                lr_star = lr_range[cross_idx - 1] + frac * (lr_range[cross_idx] - lr_range[cross_idx - 1])
                boundary_lrs.append(lr_star)
            valid_widths.append(w)

        return {
            'widths': np.array(valid_widths),
            'critical_learning_rates': np.array(boundary_lrs),
        }

    def order_parameter(self, kernel_trajectory):
        """Compute order parameter for the lazy-rich transition.

        Uses alignment between initial and final kernel principal eigenvectors.
        Order parameter ψ = |⟨v₁(0), v₁(T)⟩|.
        ψ ≈ 1: lazy regime (eigenvectors unchanged).
        ψ << 1: rich regime (feature learning rotated kernel structure).
        """
        kernel_0 = kernel_trajectory[0]
        kernel_T = kernel_trajectory[-1]

        eigvals_0, eigvecs_0 = np.linalg.eigh(kernel_0)
        eigvals_T, eigvecs_T = np.linalg.eigh(kernel_T)

        # Top eigenvector alignment
        v1_0 = eigvecs_0[:, -1]
        v1_T = eigvecs_T[:, -1]
        alignment_top = np.abs(np.dot(v1_0, v1_T))

        # Full subspace alignment (top-k eigenvectors)
        k = min(5, kernel_0.shape[0])
        V0 = eigvecs_0[:, -k:]
        VT = eigvecs_T[:, -k:]
        # Subspace alignment = ||V0^T VT||_F / √k
        subspace_alignment = np.linalg.norm(V0.T @ VT, 'fro') / np.sqrt(k)

        # Spectral shape change
        ev0 = eigvals_0 / max(np.sum(eigvals_0), 1e-15)
        evT = eigvals_T / max(np.sum(eigvals_T), 1e-15)
        spectral_distance = np.linalg.norm(ev0 - evT)

        return {
            'top_eigenvector_alignment': alignment_top,
            'subspace_alignment': subspace_alignment,
            'spectral_distance': spectral_distance,
            'is_lazy': alignment_top > 0.95,
            'is_rich': alignment_top < 0.5,
        }

    def finite_width_corrections(self, width, kernel_change):
        """Compute finite-width corrections to the lazy regime prediction.

        In the 1/m expansion:
        Θ(t) = Θ₀ + (1/√m) Θ₁(t) + (1/m) Θ₂(t) + ...

        Returns estimated correction terms and their relative magnitudes.
        """
        # Leading correction scales as 1/√m
        correction_1 = kernel_change * np.sqrt(width)
        # Next-order correction estimate
        correction_2 = kernel_change * width

        # Estimated residual after first-order correction
        residual_after_1st = kernel_change - correction_1 / np.sqrt(width)
        # This should be O(1/m)
        residual_estimate_2nd = correction_2 / width

        return {
            'first_order_coefficient': correction_1,
            'second_order_coefficient': correction_2,
            'relative_first_order': 1.0 / np.sqrt(width),
            'relative_second_order': 1.0 / width,
            'kernel_change': kernel_change,
            'predicted_change_at_2x_width': kernel_change / np.sqrt(2),
            'convergence_rate_to_ntk': 0.5,  # exponent in m^{-1/2} scaling
        }
