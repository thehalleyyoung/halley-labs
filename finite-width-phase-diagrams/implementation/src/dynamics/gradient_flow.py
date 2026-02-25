"""
Gradient flow ODE analysis for neural network training dynamics.

Implements continuous-time gradient descent, NTK dynamics, feature learning
beyond the kernel regime, learning rate scheduling, and SGD-as-SDE analysis.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, expm, solve, cholesky, norm
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d


class GradientFlowSolver:
    """Continuous-time gradient descent ODE solver.

    Solves dθ/dt = -∇L(θ) using adaptive ODE integration.
    """

    def __init__(self, loss_fn, param_dim, dt=1e-3):
        self.loss_fn = loss_fn
        self.param_dim = param_dim
        self.dt = dt
        self._grad_eps = 1e-6

    def solve(self, initial_params, t_span, method='RK45'):
        """Solve dθ/dt = -∇L(θ) using scipy.integrate.solve_ivp.

        Parameters
        ----------
        initial_params : ndarray, shape (param_dim,)
        t_span : tuple (t0, tf)
        method : str, one of 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'

        Returns
        -------
        solution : OdeResult with .t (times) and .y (params over time)
        """
        initial_params = np.asarray(initial_params, dtype=np.float64).ravel()

        def rhs(t, theta):
            grad = self._numerical_gradient(theta)
            return -grad

        sol = solve_ivp(
            rhs, t_span, initial_params,
            method=method, dense_output=True,
            rtol=1e-8, atol=1e-10,
            max_step=self.dt * 10,
        )
        return sol

    def solve_with_callbacks(self, initial_params, t_span, callbacks):
        """Solve with monitoring callbacks invoked at each internal step.

        Parameters
        ----------
        initial_params : ndarray
        t_span : tuple
        callbacks : list of callable(t, theta, loss) -> None

        Returns
        -------
        solution : OdeResult
        """
        initial_params = np.asarray(initial_params, dtype=np.float64).ravel()
        callback_log = []

        def rhs(t, theta):
            grad = self._numerical_gradient(theta)
            loss_val = self.loss_fn(theta)
            for cb in callbacks:
                cb(t, theta.copy(), loss_val)
            callback_log.append((t, loss_val))
            return -grad

        sol = solve_ivp(
            rhs, t_span, initial_params,
            method='RK45', dense_output=True,
            rtol=1e-8, atol=1e-10,
            max_step=self.dt * 10,
        )
        sol.callback_log = callback_log
        return sol

    def compute_gradient(self, params, data_x, data_y):
        """Compute ∇L numerically via central differences.

        Parameters
        ----------
        params : ndarray, shape (param_dim,)
        data_x : ndarray, shape (n_samples, input_dim)
        data_y : ndarray, shape (n_samples,) or (n_samples, output_dim)

        Returns
        -------
        grad : ndarray, shape (param_dim,)
        """
        params = np.asarray(params, dtype=np.float64).ravel()
        grad = np.zeros_like(params)
        eps = self._grad_eps
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            loss_plus = self.loss_fn(params_plus, data_x, data_y)
            loss_minus = self.loss_fn(params_minus, data_x, data_y)
            grad[i] = (loss_plus - loss_minus) / (2.0 * eps)
        return grad

    def trajectory_energy(self, trajectory):
        """Compute E(t) = L(θ(t)) along a trajectory.

        Parameters
        ----------
        trajectory : ndarray, shape (param_dim, n_times) or (n_times, param_dim)

        Returns
        -------
        energies : ndarray, shape (n_times,)
        """
        if trajectory.shape[0] == self.param_dim and trajectory.ndim == 2:
            traj = trajectory.T
        else:
            traj = np.asarray(trajectory)
        n_times = traj.shape[0]
        energies = np.zeros(n_times)
        for i in range(n_times):
            energies[i] = self.loss_fn(traj[i])
        return energies

    def convergence_rate(self, trajectory):
        """Estimate convergence rate from trajectory.

        Fits an exponential decay to the loss curve: L(t) ~ L* + C exp(-αt).

        Parameters
        ----------
        trajectory : ndarray, shape (param_dim, n_times)

        Returns
        -------
        rate : float, estimated exponential convergence rate α
        residual_loss : float, estimated L*
        """
        energies = self.trajectory_energy(trajectory)
        n = len(energies)
        if n < 3:
            return 0.0, energies[-1]

        final_loss = energies[-1]
        shifted = energies - final_loss
        shifted = np.maximum(shifted, 1e-30)

        log_shifted = np.log(shifted)
        valid = np.isfinite(log_shifted)
        if valid.sum() < 2:
            return 0.0, final_loss

        times = np.arange(n, dtype=np.float64)
        t_valid = times[valid]
        y_valid = log_shifted[valid]

        coeffs = np.polyfit(t_valid, y_valid, 1)
        rate = -coeffs[0]
        return float(rate), float(final_loss)

    def lyapunov_function(self, trajectory, loss_values):
        """Verify L is a Lyapunov function: dL/dt ≤ 0 along trajectory.

        Parameters
        ----------
        trajectory : ndarray, shape (param_dim, n_times)
        loss_values : ndarray, shape (n_times,)

        Returns
        -------
        is_lyapunov : bool
        dL_dt : ndarray of dL/dt estimates
        max_violation : float, max positive dL/dt (0 if none)
        """
        loss_values = np.asarray(loss_values, dtype=np.float64)
        dL_dt = np.diff(loss_values)
        is_lyapunov = bool(np.all(dL_dt <= 1e-10))
        max_violation = float(max(0.0, np.max(dL_dt)))
        return is_lyapunov, dL_dt, max_violation

    def phase_portrait(self, param_ranges, resolution=50):
        """Compute 2D phase portrait of gradient flow.

        Parameters
        ----------
        param_ranges : list of two tuples [(lo1, hi1), (lo2, hi2)]
        resolution : int, grid resolution per axis

        Returns
        -------
        grid_x : ndarray, shape (resolution,)
        grid_y : ndarray, shape (resolution,)
        U : ndarray, shape (resolution, resolution), negative gradient x-component
        V : ndarray, shape (resolution, resolution), negative gradient y-component
        loss_landscape : ndarray, shape (resolution, resolution)
        """
        (x_lo, x_hi) = param_ranges[0]
        (y_lo, y_hi) = param_ranges[1]
        grid_x = np.linspace(x_lo, x_hi, resolution)
        grid_y = np.linspace(y_lo, y_hi, resolution)

        U = np.zeros((resolution, resolution))
        V = np.zeros((resolution, resolution))
        loss_landscape = np.zeros((resolution, resolution))

        base_params = np.zeros(self.param_dim)
        eps = self._grad_eps

        for i, x in enumerate(grid_x):
            for j, y in enumerate(grid_y):
                base_params[0] = x
                base_params[1] = y
                loss_landscape[j, i] = self.loss_fn(base_params)

                p_xp = base_params.copy(); p_xp[0] += eps
                p_xm = base_params.copy(); p_xm[0] -= eps
                p_yp = base_params.copy(); p_yp[1] += eps
                p_ym = base_params.copy(); p_ym[1] -= eps

                gx = (self.loss_fn(p_xp) - self.loss_fn(p_xm)) / (2.0 * eps)
                gy = (self.loss_fn(p_yp) - self.loss_fn(p_ym)) / (2.0 * eps)
                U[j, i] = -gx
                V[j, i] = -gy

        return grid_x, grid_y, U, V, loss_landscape

    # ---- internal helpers ----

    def _numerical_gradient(self, params):
        """Central-difference gradient of self.loss_fn(params)."""
        params = np.asarray(params, dtype=np.float64)
        grad = np.zeros_like(params)
        eps = self._grad_eps
        for i in range(len(params)):
            e = np.zeros_like(params)
            e[i] = eps
            grad[i] = (self.loss_fn(params + e) - self.loss_fn(params - e)) / (2.0 * eps)
        return grad


class NTKDynamics:
    """Neural tangent kernel parameterized dynamics.

    Solves the linearized training dynamics: df/dt = -Θ(f - y).
    """

    def __init__(self, kernel_matrix, target_values):
        """
        Parameters
        ----------
        kernel_matrix : ndarray, shape (n, n), the NTK Θ
        target_values : ndarray, shape (n,) or (n, d)
        """
        self.kernel = np.asarray(kernel_matrix, dtype=np.float64)
        self.target = np.asarray(target_values, dtype=np.float64)
        self.n = self.kernel.shape[0]
        self._eigenvalues = None
        self._eigenvectors = None

    def solve_ntk_ode(self, initial_predictions, t_span):
        """Solve df/dt = -Θ(f - y) via ODE integration.

        Parameters
        ----------
        initial_predictions : ndarray, shape (n,)
        t_span : tuple (t0, tf)

        Returns
        -------
        solution : OdeResult
        """
        f0 = np.asarray(initial_predictions, dtype=np.float64).ravel()
        y = self.target.ravel()
        K = self.kernel

        def rhs(t, f):
            return -K @ (f - y)

        sol = solve_ivp(rhs, t_span, f0, method='RK45', dense_output=True,
                        rtol=1e-10, atol=1e-12)
        return sol

    def analytical_solution(self, t):
        """Closed-form solution f(t) = y - exp(-Θt)(y - f₀).

        Requires initial predictions stored from last solve_ntk_ode or
        defaults to zeros.

        Parameters
        ----------
        t : float or ndarray of times

        Returns
        -------
        predictions : ndarray, shape (n,) if scalar t, else (len(t), n)
        """
        y = self.target.ravel()
        evals, evecs = self._get_eigen()

        scalar_input = np.isscalar(t)
        t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64))

        # Project residual onto eigenbasis
        residual_0 = y.copy()  # f0 = 0 assumed
        coeffs = evecs.T @ residual_0

        results = np.zeros((len(t_arr), self.n))
        for k, tk in enumerate(t_arr):
            decay = np.exp(-evals * tk)
            evolved_coeffs = decay * coeffs
            results[k] = y - evecs @ evolved_coeffs

        if scalar_input:
            return results[0]
        return results

    def kernel_eigendecomposition(self, kernel):
        """Eigendecompose Θ for spectral analysis.

        Parameters
        ----------
        kernel : ndarray, shape (n, n)

        Returns
        -------
        eigenvalues : ndarray, shape (n,), sorted descending
        eigenvectors : ndarray, shape (n, n), columns are eigenvectors
        """
        kernel = np.asarray(kernel, dtype=np.float64)
        kernel_sym = 0.5 * (kernel + kernel.T)
        evals, evecs = eigh(kernel_sym)
        idx = np.argsort(evals)[::-1]
        return evals[idx], evecs[:, idx]

    def mode_dynamics(self, eigenvalues, eigenvectors, t):
        """Individual eigenmode evolution.

        Parameters
        ----------
        eigenvalues : ndarray, shape (n,)
        eigenvectors : ndarray, shape (n, n)
        t : float or ndarray

        Returns
        -------
        mode_amplitudes : ndarray, shape (n,) or (len(t), n)
        """
        y = self.target.ravel()
        coeffs = eigenvectors.T @ y

        scalar_input = np.isscalar(t)
        t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64))

        amps = np.zeros((len(t_arr), len(eigenvalues)))
        for k, tk in enumerate(t_arr):
            amps[k] = coeffs * (1.0 - np.exp(-eigenvalues * tk))

        if scalar_input:
            return amps[0]
        return amps

    def convergence_time_per_mode(self, eigenvalues):
        """Convergence time scale τ_k = 1/λ_k per eigenmode.

        Parameters
        ----------
        eigenvalues : ndarray

        Returns
        -------
        timescales : ndarray, same shape; inf for zero eigenvalues
        """
        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
        timescales = np.full_like(eigenvalues, np.inf)
        pos = eigenvalues > 1e-14
        timescales[pos] = 1.0 / eigenvalues[pos]
        return timescales

    def spectral_bias_prediction(self, eigenvalues, eigenvectors, target):
        """Predict which eigenmodes are learned first (spectral bias).

        Modes with larger eigenvalues converge faster, leading to spectral bias
        toward the top eigenspace of the NTK.

        Parameters
        ----------
        eigenvalues : ndarray, shape (n,)
        eigenvectors : ndarray, shape (n, n)
        target : ndarray, shape (n,)

        Returns
        -------
        mode_order : ndarray of int, modes ordered by convergence speed
        target_projections : ndarray, |<v_k, y>|^2 per mode
        learning_speeds : ndarray, λ_k * |<v_k, y>|^2
        """
        target = np.asarray(target, dtype=np.float64).ravel()
        projections = eigenvectors.T @ target
        target_proj_sq = projections ** 2

        learning_speeds = eigenvalues * target_proj_sq
        mode_order = np.argsort(learning_speeds)[::-1]

        return mode_order, target_proj_sq, learning_speeds

    def ntk_prediction_at_time(self, t, test_kernel):
        """Prediction at test points at time t using NTK dynamics.

        f_test(t) = K_test_train @ K_train^{-1} (I - exp(-Θt)) y

        Parameters
        ----------
        t : float
        test_kernel : ndarray, shape (n_test, n_train)

        Returns
        -------
        predictions : ndarray, shape (n_test,)
        """
        evals, evecs = self._get_eigen()
        y = self.target.ravel()
        decay = np.exp(-evals * t)
        learned_coeffs = (1.0 - decay) * (evecs.T @ y)
        train_pred = evecs @ learned_coeffs

        K_reg = self.kernel + 1e-10 * np.eye(self.n)
        alpha = solve(K_reg, train_pred, assume_a='pos')
        test_kernel = np.asarray(test_kernel, dtype=np.float64)
        return test_kernel @ alpha

    # ---- internal ----

    def _get_eigen(self):
        if self._eigenvalues is None:
            self._eigenvalues, self._eigenvectors = self.kernel_eigendecomposition(self.kernel)
        return self._eigenvalues, self._eigenvectors


class FeatureLearningDynamics:
    """Beyond-NTK feature learning dynamics.

    Tracks how learned features evolve during training, departing from the
    frozen-feature (kernel) regime.
    """

    def __init__(self, network_width, input_dim, output_dim):
        self.width = network_width
        self.input_dim = input_dim
        self.output_dim = output_dim

    def feature_kernel_evolution(self, params_trajectory):
        """Track the empirical feature kernel Φ(x)ᵀΦ(x) over training.

        Approximates features as the first-layer representation. The kernel is
        K_ij(t) = <φ(x_i; θ(t)), φ(x_j; θ(t))> where φ is the penultimate
        layer representation.

        Parameters
        ----------
        params_trajectory : list of ndarray, each shape (param_dim,)
            Parameters at successive training steps.

        Returns
        -------
        kernel_trajectory : list of ndarray, each shape (width, width)
            Gram matrices over hidden units at each time step.
        times : ndarray
        """
        n_steps = len(params_trajectory)
        kernel_trajectory = []
        times = np.arange(n_steps, dtype=np.float64)

        for step_params in params_trajectory:
            params = np.asarray(step_params, dtype=np.float64)
            n_weights = self.width * self.input_dim
            if len(params) >= n_weights:
                W = params[:n_weights].reshape(self.width, self.input_dim)
            else:
                W = params.reshape(-1, self.input_dim)[:self.width]
            gram = W @ W.T / self.input_dim
            kernel_trajectory.append(gram)

        return kernel_trajectory, times

    def kernel_change_rate(self, kernel_trajectory, times):
        """Compute dΘ/dt rate from a trajectory of kernels.

        Parameters
        ----------
        kernel_trajectory : list of ndarray
        times : ndarray

        Returns
        -------
        rates : ndarray, ||K(t+1) - K(t)|| / Δt at each step
        cumulative_change : ndarray, cumulative Frobenius change
        """
        n = len(kernel_trajectory)
        if n < 2:
            return np.array([0.0]), np.array([0.0])

        times = np.asarray(times, dtype=np.float64)
        rates = np.zeros(n - 1)
        cumulative_change = np.zeros(n)

        for i in range(n - 1):
            diff = kernel_trajectory[i + 1] - kernel_trajectory[i]
            dt = times[i + 1] - times[i]
            if dt < 1e-15:
                dt = 1.0
            rates[i] = norm(diff, 'fro') / dt
            cumulative_change[i + 1] = cumulative_change[i] + norm(diff, 'fro')

        return rates, cumulative_change

    def mean_field_feature_ode(self, initial_features, coupling):
        """Mean-field ODE for feature dynamics.

        dW/dt = -coupling @ W models the mean-field evolution of weight matrices
        under gradient flow.

        Parameters
        ----------
        initial_features : ndarray, shape (width, input_dim)
        coupling : ndarray, shape (width, width)

        Returns
        -------
        solution : OdeResult
        """
        W0 = np.asarray(initial_features, dtype=np.float64)
        shape = W0.shape
        w0_flat = W0.ravel()
        C = np.asarray(coupling, dtype=np.float64)

        def rhs(t, w_flat):
            W = w_flat.reshape(shape)
            dW = -C @ W
            return dW.ravel()

        sol = solve_ivp(rhs, (0, 10.0), w0_flat, method='RK45',
                        dense_output=True, rtol=1e-8, atol=1e-10)
        return sol

    def feature_alignment_dynamics(self, features_t, target_features):
        """Track alignment between learned features and target features over time.

        Parameters
        ----------
        features_t : list of ndarray, each shape (width, input_dim)
        target_features : ndarray, shape (width, input_dim) or (k, input_dim)

        Returns
        -------
        alignments : ndarray, shape (n_steps,), average cosine similarity
        principal_angles : list of ndarray, principal angles at each step
        """
        target = np.asarray(target_features, dtype=np.float64)
        n_steps = len(features_t)
        alignments = np.zeros(n_steps)
        principal_angles_list = []

        # Compute target subspace (via SVD)
        U_target, _, _ = np.linalg.svd(target, full_matrices=False)

        for i, feat in enumerate(features_t):
            feat = np.asarray(feat, dtype=np.float64)
            U_feat, _, _ = np.linalg.svd(feat, full_matrices=False)

            k = min(U_feat.shape[1], U_target.shape[1])
            M = U_feat[:, :k].T @ U_target[:, :k]
            svals = np.linalg.svd(M, compute_uv=False)
            svals = np.clip(svals, -1.0, 1.0)
            angles = np.arccos(svals)
            principal_angles_list.append(angles)
            alignments[i] = np.mean(svals)

        return alignments, principal_angles_list

    def ntk_deviation(self, empirical_kernel_t, initial_kernel):
        """Compute relative NTK deviation ||Θ(t) - Θ(0)||/||Θ(0)||.

        Parameters
        ----------
        empirical_kernel_t : ndarray or list of ndarray
        initial_kernel : ndarray

        Returns
        -------
        deviations : ndarray, relative Frobenius norm deviation at each time
        """
        K0 = np.asarray(initial_kernel, dtype=np.float64)
        norm_K0 = norm(K0, 'fro')
        if norm_K0 < 1e-15:
            norm_K0 = 1.0

        if isinstance(empirical_kernel_t, list):
            deviations = np.zeros(len(empirical_kernel_t))
            for i, Kt in enumerate(empirical_kernel_t):
                Kt = np.asarray(Kt, dtype=np.float64)
                deviations[i] = norm(Kt - K0, 'fro') / norm_K0
            return deviations
        else:
            Kt = np.asarray(empirical_kernel_t, dtype=np.float64)
            return float(norm(Kt - K0, 'fro') / norm_K0)

    def feature_learning_timescale(self, kernel_trajectory, times):
        """Estimate t* where features change significantly.

        Finds the time at which ||Θ(t) - Θ(0)|| first exceeds a threshold
        fraction of ||Θ(0)||.

        Parameters
        ----------
        kernel_trajectory : list of ndarray
        times : ndarray

        Returns
        -------
        t_star : float, estimated feature learning onset time
        relative_changes : ndarray, relative kernel change at each time
        threshold : float, the threshold used (0.1 = 10%)
        """
        times = np.asarray(times, dtype=np.float64)
        K0 = kernel_trajectory[0]
        norm_K0 = norm(K0, 'fro')
        if norm_K0 < 1e-15:
            norm_K0 = 1.0

        threshold = 0.1
        n = len(kernel_trajectory)
        relative_changes = np.zeros(n)

        for i in range(n):
            relative_changes[i] = norm(kernel_trajectory[i] - K0, 'fro') / norm_K0

        crossed = np.where(relative_changes > threshold)[0]
        if len(crossed) > 0:
            idx = crossed[0]
            if idx > 0 and idx < n:
                # Linear interpolation for more precise estimate
                frac = ((threshold - relative_changes[idx - 1]) /
                        (relative_changes[idx] - relative_changes[idx - 1] + 1e-15))
                t_star = times[idx - 1] + frac * (times[idx] - times[idx - 1])
            else:
                t_star = times[idx]
        else:
            t_star = times[-1]

        return float(t_star), relative_changes, threshold

    def catapult_phase_detection(self, loss_trajectory):
        """Detect catapult phase and progressive sharpening in loss trajectory.

        The catapult phase is characterized by an initial increase in loss
        before convergence. Progressive sharpening shows the loss curvature
        increasing during training.

        Parameters
        ----------
        loss_trajectory : ndarray, shape (n_steps,)

        Returns
        -------
        has_catapult : bool
        catapult_start : int or None, index where loss starts increasing
        catapult_end : int or None, index where loss returns below initial
        sharpening_rate : float, rate of curvature increase
        """
        losses = np.asarray(loss_trajectory, dtype=np.float64)
        n = len(losses)

        # Detect catapult: loss goes up before coming down
        initial_loss = losses[0]
        has_catapult = False
        catapult_start = None
        catapult_end = None

        # Find first point where loss exceeds initial
        for i in range(1, n):
            if losses[i] > initial_loss * 1.05:
                catapult_start = i
                has_catapult = True
                break

        if has_catapult and catapult_start is not None:
            # Find where loss drops back below initial
            for i in range(catapult_start, n):
                if losses[i] < initial_loss:
                    catapult_end = i
                    break

        # Progressive sharpening: estimate local curvature over time
        if n >= 5:
            window = max(3, n // 20)
            curvatures = []
            for i in range(window, n - window):
                local = losses[max(0, i - window):i + window + 1]
                if len(local) >= 3:
                    t_local = np.arange(len(local), dtype=np.float64)
                    coeffs = np.polyfit(t_local, local, 2)
                    curvatures.append(2.0 * coeffs[0])
            if len(curvatures) >= 2:
                curvatures = np.array(curvatures)
                t_curv = np.arange(len(curvatures), dtype=np.float64)
                slope = np.polyfit(t_curv, curvatures, 1)[0]
                sharpening_rate = float(slope)
            else:
                sharpening_rate = 0.0
        else:
            sharpening_rate = 0.0

        return has_catapult, catapult_start, catapult_end, sharpening_rate


class LearningRateScheduler:
    """Learning rate schedule effects on gradient flow dynamics."""

    def __init__(self, base_lr=0.01):
        self.base_lr = base_lr

    def constant(self, t):
        """Constant learning rate η(t) = η₀."""
        return self.base_lr

    def exponential_decay(self, t, decay_rate=0.99):
        """Exponential decay: η(t) = η₀ exp(-γt).

        Parameters
        ----------
        t : float or ndarray
        decay_rate : float, γ in the exponent

        Returns
        -------
        lr : float or ndarray
        """
        t = np.asarray(t, dtype=np.float64)
        return self.base_lr * np.exp(-decay_rate * t)

    def cosine_annealing(self, t, t_max, eta_min=0):
        """Cosine annealing schedule.

        η(t) = η_min + 0.5(η₀ - η_min)(1 + cos(πt/T_max))

        Parameters
        ----------
        t : float or ndarray
        t_max : float
        eta_min : float

        Returns
        -------
        lr : float or ndarray
        """
        t = np.asarray(t, dtype=np.float64)
        return eta_min + 0.5 * (self.base_lr - eta_min) * (
            1.0 + np.cos(np.pi * t / t_max)
        )

    def warmup_cosine(self, t, warmup_steps, total_steps):
        """Linear warmup followed by cosine decay.

        Parameters
        ----------
        t : float or ndarray
        warmup_steps : float
        total_steps : float

        Returns
        -------
        lr : float or ndarray
        """
        t = np.asarray(t, dtype=np.float64)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)
        lr = np.zeros_like(t)

        warmup_mask = t < warmup_steps
        cosine_mask = ~warmup_mask

        lr[warmup_mask] = self.base_lr * t[warmup_mask] / warmup_steps
        if np.any(cosine_mask):
            progress = (t[cosine_mask] - warmup_steps) / (total_steps - warmup_steps)
            progress = np.clip(progress, 0.0, 1.0)
            lr[cosine_mask] = self.base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))

        if scalar:
            return float(lr[0])
        return lr

    def cyclical(self, t, cycle_length, min_lr, max_lr):
        """Cyclical learning rate (triangular).

        Parameters
        ----------
        t : float or ndarray
        cycle_length : float
        min_lr : float
        max_lr : float

        Returns
        -------
        lr : float or ndarray
        """
        scalar_input = np.isscalar(t)
        t = np.asarray(t, dtype=np.float64)
        cycle_pos = np.mod(t, cycle_length) / cycle_length
        # Triangular: go up first half, down second half
        lr = np.where(
            cycle_pos < 0.5,
            min_lr + (max_lr - min_lr) * 2.0 * cycle_pos,
            max_lr - (max_lr - min_lr) * 2.0 * (cycle_pos - 0.5),
        )
        if scalar_input:
            return float(lr)
        return lr

    def one_over_t(self, t, t0=1.0):
        """Inverse time decay: η(t) = η₀ / (1 + t/t₀).

        Parameters
        ----------
        t : float or ndarray
        t0 : float

        Returns
        -------
        lr : float or ndarray
        """
        t = np.asarray(t, dtype=np.float64)
        return self.base_lr / (1.0 + t / t0)

    def critical_learning_rate(self, hessian_eigenvalues):
        """Critical learning rate η_c = 2/λ_max.

        Above this, gradient descent diverges.

        Parameters
        ----------
        hessian_eigenvalues : ndarray

        Returns
        -------
        eta_c : float
        """
        eigenvalues = np.asarray(hessian_eigenvalues, dtype=np.float64)
        lambda_max = np.max(np.abs(eigenvalues))
        if lambda_max < 1e-15:
            return np.inf
        return 2.0 / lambda_max

    def stability_boundary(self, lr, hessian_eigenvalues):
        """Check if learning rate is in the stable region.

        For gradient descent, stability requires |1 - η λ_k| < 1 for all k,
        i.e., 0 < η < 2/λ_k.

        Parameters
        ----------
        lr : float
        hessian_eigenvalues : ndarray

        Returns
        -------
        is_stable : bool
        max_amplification : float, max |1 - η λ_k|
        unstable_modes : ndarray of int, indices of unstable modes
        """
        eigenvalues = np.asarray(hessian_eigenvalues, dtype=np.float64)
        amplification = np.abs(1.0 - lr * eigenvalues)
        is_stable = bool(np.all(amplification < 1.0))
        max_amp = float(np.max(amplification))
        unstable = np.where(amplification >= 1.0)[0]
        return is_stable, max_amp, unstable

    def edge_of_stability_lr(self, loss_trajectory, lr):
        """Detect edge of stability phenomenon.

        At the edge of stability, the sharpness (max Hessian eigenvalue)
        hovers around 2/η. We detect this by checking if the loss oscillates
        while still decreasing on average.

        Parameters
        ----------
        loss_trajectory : ndarray
        lr : float

        Returns
        -------
        is_eos : bool, whether edge of stability is detected
        oscillation_amplitude : float
        trend_slope : float, overall loss trend
        predicted_sharpness : float, 2/η
        """
        losses = np.asarray(loss_trajectory, dtype=np.float64)
        n = len(losses)
        if n < 10:
            return False, 0.0, 0.0, 2.0 / lr

        # Overall trend
        t = np.arange(n, dtype=np.float64)
        trend_coeffs = np.polyfit(t, losses, 1)
        trend_slope = float(trend_coeffs[0])
        trend = np.polyval(trend_coeffs, t)

        # Detrended oscillation
        detrended = losses - trend
        oscillation_amplitude = float(np.std(detrended))

        # EOS: loss decreasing on average but oscillating
        decreasing = trend_slope < 0
        oscillating = oscillation_amplitude > 0.01 * np.mean(np.abs(losses))

        # Check for non-monotone behavior
        diffs = np.diff(losses)
        frac_increasing = np.mean(diffs > 0)
        non_monotone = frac_increasing > 0.2

        is_eos = decreasing and oscillating and non_monotone
        predicted_sharpness = 2.0 / lr

        return is_eos, float(oscillation_amplitude), float(trend_slope), predicted_sharpness


class GradientNoiseSDE:
    """SGD as stochastic differential equation.

    Models minibatch SGD noise: dθ = -∇L(θ)dt + √(η/B) Σ^{1/2} dW
    where Σ is the per-sample gradient covariance.
    """

    def __init__(self, param_dim, batch_size, dataset_size, learning_rate):
        self.param_dim = param_dim
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.learning_rate = learning_rate
        self._grad_eps = 1e-6

    def noise_covariance(self, params, loss_fn, data_x, data_y):
        """Estimate noise covariance Σ = Cov(∇L_i) from per-sample gradients.

        Parameters
        ----------
        params : ndarray, shape (param_dim,)
        loss_fn : callable(params, x_i, y_i) -> scalar loss for one sample
        data_x : ndarray, shape (n, input_dim)
        data_y : ndarray, shape (n,) or (n, output_dim)

        Returns
        -------
        sigma : ndarray, shape (param_dim, param_dim), gradient covariance
        mean_grad : ndarray, shape (param_dim,), mean gradient
        """
        params = np.asarray(params, dtype=np.float64).ravel()
        n = data_x.shape[0]
        d = self.param_dim
        eps = self._grad_eps

        grads = np.zeros((n, d))
        for i in range(n):
            xi = data_x[i:i + 1]
            yi = data_y[i:i + 1]
            for j in range(d):
                e = np.zeros(d)
                e[j] = eps
                loss_plus = loss_fn(params + e, xi, yi)
                loss_minus = loss_fn(params - e, xi, yi)
                grads[i, j] = (loss_plus - loss_minus) / (2.0 * eps)

        mean_grad = np.mean(grads, axis=0)
        centered = grads - mean_grad
        sigma = (centered.T @ centered) / max(n - 1, 1)
        return sigma, mean_grad

    def effective_temperature(self, learning_rate, batch_size):
        """Compute effective SGD temperature T_eff = η/(2B) * tr(Σ).

        The temperature controls exploration vs exploitation: higher T_eff
        leads to wider minima.

        Parameters
        ----------
        learning_rate : float
        batch_size : int

        Returns
        -------
        temperature : float (requires noise_covariance to have been computed,
                      returns the formula coefficient η/(2B))
        ratio : float, η/B ratio controlling noise level
        """
        ratio = learning_rate / batch_size
        temperature_coeff = ratio / 2.0
        return temperature_coeff, ratio

    def sde_euler_maruyama(self, initial_params, drift, diffusion, dt, n_steps):
        """Euler-Maruyama integration of SDE.

        dθ = drift(θ) dt + diffusion(θ) dW

        Parameters
        ----------
        initial_params : ndarray, shape (param_dim,)
        drift : callable(theta) -> ndarray, shape (param_dim,)
        diffusion : callable(theta) -> ndarray, shape (param_dim, param_dim)
        dt : float
        n_steps : int

        Returns
        -------
        trajectory : ndarray, shape (n_steps + 1, param_dim)
        times : ndarray, shape (n_steps + 1,)
        """
        d = self.param_dim
        theta = np.asarray(initial_params, dtype=np.float64).ravel().copy()
        trajectory = np.zeros((n_steps + 1, d))
        times = np.zeros(n_steps + 1)
        trajectory[0] = theta
        sqrt_dt = np.sqrt(dt)

        for i in range(n_steps):
            mu = drift(theta)
            sigma = diffusion(theta)
            dW = np.random.randn(d) * sqrt_dt
            theta = theta + mu * dt + sigma @ dW
            trajectory[i + 1] = theta
            times[i + 1] = (i + 1) * dt

        return trajectory, times

    def sde_milstein(self, initial_params, drift, diffusion, dt, n_steps):
        """Milstein scheme for SDE integration (higher order than Euler-Maruyama).

        Includes the correction term for multiplicative noise:
        θ_{n+1} = θ_n + a dt + b dW + 0.5 b b' (dW^2 - dt)

        Parameters
        ----------
        initial_params : ndarray, shape (param_dim,)
        drift : callable(theta) -> ndarray, shape (param_dim,)
        diffusion : callable(theta) -> ndarray, shape (param_dim, param_dim)
        dt : float
        n_steps : int

        Returns
        -------
        trajectory : ndarray, shape (n_steps + 1, param_dim)
        times : ndarray, shape (n_steps + 1,)
        """
        d = self.param_dim
        theta = np.asarray(initial_params, dtype=np.float64).ravel().copy()
        trajectory = np.zeros((n_steps + 1, d))
        times = np.zeros(n_steps + 1)
        trajectory[0] = theta
        sqrt_dt = np.sqrt(dt)
        eps = 1e-6

        for i in range(n_steps):
            mu = drift(theta)
            sigma = diffusion(theta)
            dW = np.random.randn(d) * sqrt_dt

            # Euler-Maruyama base step
            theta_new = theta + mu * dt + sigma @ dW

            # Milstein correction: approximate d(sigma)/d(theta) numerically
            correction = np.zeros(d)
            for j in range(d):
                e_j = np.zeros(d)
                e_j[j] = eps
                sigma_plus = diffusion(theta + e_j)
                dsigma_dtheta_j = (sigma_plus - sigma) / eps
                # Correction for each noise component
                for k in range(d):
                    correction += 0.5 * dsigma_dtheta_j[:, k] * sigma[j, k] * (
                        dW[j] * dW[k] - (dt if j == k else 0.0)
                    )

            theta_new += correction
            theta = theta_new
            trajectory[i + 1] = theta
            times[i + 1] = (i + 1) * dt

        return trajectory, times

    def fokker_planck_stationary(self, drift, diffusion, param_range):
        """Compute stationary distribution of the Fokker-Planck equation.

        For 1D: P_ss(θ) ∝ (1/D(θ)) exp(∫ a(θ')/D(θ') dθ')

        Parameters
        ----------
        drift : callable(theta_scalar) -> float
        diffusion : callable(theta_scalar) -> float, diffusion coefficient D(θ)
        param_range : tuple (lo, hi)

        Returns
        -------
        theta_grid : ndarray, shape (n_grid,)
        density : ndarray, shape (n_grid,), normalized probability density
        potential : ndarray, shape (n_grid,), effective potential -ln(P_ss)
        """
        lo, hi = param_range
        n_grid = 500
        theta_grid = np.linspace(lo, hi, n_grid)
        dtheta = theta_grid[1] - theta_grid[0]

        # Compute drift and diffusion on grid
        a_vals = np.array([drift(th) for th in theta_grid])
        D_vals = np.array([diffusion(th) for th in theta_grid])
        D_vals = np.maximum(D_vals, 1e-15)

        # Integrate a(θ)/D(θ) to get the exponent
        integrand = a_vals / D_vals
        exponent = np.cumsum(integrand) * dtheta
        exponent -= np.max(exponent)  # numerical stability

        log_density = exponent - np.log(D_vals)
        log_density -= np.max(log_density)
        density = np.exp(log_density)

        # Normalize
        Z = np.trapz(density, theta_grid)
        if Z > 1e-15:
            density /= Z

        potential = -np.log(density + 1e-30)
        potential -= np.min(potential)

        return theta_grid, density, potential

    def escape_rate(self, barrier_height, temperature):
        """Kramers escape rate for escaping a local minimum.

        rate ~ exp(-ΔE / T_eff), the Arrhenius-Kramers formula.

        Parameters
        ----------
        barrier_height : float, energy barrier ΔE
        temperature : float, effective temperature T_eff

        Returns
        -------
        rate : float, escape rate
        mean_escape_time : float, 1/rate
        """
        if temperature < 1e-15:
            return 0.0, np.inf

        exponent = -barrier_height / temperature
        exponent = max(exponent, -500.0)  # avoid underflow
        rate = np.exp(exponent)
        mean_escape_time = 1.0 / rate if rate > 1e-30 else np.inf
        return float(rate), float(mean_escape_time)

    def noise_induced_transition(self, minima, barrier, temperature, n_samples):
        """Sample noise-induced transitions between minima.

        Simulates an Ornstein-Uhlenbeck-like process in a double-well potential
        and counts transitions between basins.

        Parameters
        ----------
        minima : tuple of float, (x_left, x_right) positions of minima
        barrier : float, barrier height between minima
        temperature : float, noise temperature
        n_samples : int, number of transition events to attempt

        Returns
        -------
        transition_times : ndarray, times at which transitions occurred
        n_transitions : int, total number of transitions observed
        transition_rate : float, empirical transition rate
        basin_occupancy : ndarray, shape (2,), fraction of time in each basin
        """
        x_left, x_right = minima
        midpoint = 0.5 * (x_left + x_right)

        # Double-well potential: V(x) = barrier * ((x - midpoint)^2 - (half_width)^2)^2 / half_width^4
        half_width = 0.5 * (x_right - x_left)
        if half_width < 1e-10:
            half_width = 1.0

        dt = 0.01
        n_steps_per_sample = 1000
        total_steps = n_samples * n_steps_per_sample
        sigma = np.sqrt(2.0 * temperature)

        x = x_left  # start in left basin
        transition_times = []
        basin_counts = np.zeros(2)
        current_basin = 0  # 0 = left, 1 = right

        for step in range(total_steps):
            # Drift from double-well: -dV/dx
            normalized = (x - midpoint) / half_width
            drift_val = -4.0 * barrier * normalized * (normalized ** 2 - 1.0) / half_width
            dW = np.random.randn() * np.sqrt(dt)
            x = x + drift_val * dt + sigma * dW

            # Determine basin
            new_basin = 0 if x < midpoint else 1
            if new_basin != current_basin:
                transition_times.append(step * dt)
                current_basin = new_basin
            basin_counts[current_basin] += 1

        transition_times = np.array(transition_times)
        n_transitions = len(transition_times)
        total_time = total_steps * dt
        transition_rate = n_transitions / total_time if total_time > 0 else 0.0
        basin_occupancy = basin_counts / basin_counts.sum() if basin_counts.sum() > 0 else np.array([0.5, 0.5])

        return transition_times, n_transitions, float(transition_rate), basin_occupancy
