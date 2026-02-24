"""SGD dynamics simulation for finite-width neural network phase diagrams.

Implements discrete-time SGD, learning rate / batch size phase analysis,
momentum dynamics, noise covariance estimation, and SGD-to-SDE conversion.
"""

import numpy as np
from scipy import linalg, optimize, integrate, sparse


class SGDSimulator:
    """Discrete-time SGD simulation with mini-batch support."""

    def __init__(self, param_dim, learning_rate=0.01, batch_size=32):
        self.param_dim = param_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.rng = np.random.default_rng()

    def step(self, params, gradient, lr=None):
        """Single SGD step: θ_{t+1} = θ_t - η∇L."""
        if lr is None:
            lr = self.learning_rate
        return params - lr * gradient

    def run(self, initial_params, loss_fn, grad_fn, data_x, data_y, n_steps=1000):
        """Full SGD run returning final parameters and loss history."""
        params = np.array(initial_params, dtype=np.float64)
        n_data = len(data_x)
        losses = np.empty(n_steps)
        for t in range(n_steps):
            batch_x, batch_y = self.sample_batch(data_x, data_y, self.batch_size)
            grad = grad_fn(params, batch_x, batch_y)
            params = self.step(params, grad)
            if t % max(1, n_steps // 100) == 0 or t == n_steps - 1:
                losses[t] = loss_fn(params, data_x, data_y)
            else:
                losses[t] = np.nan
        # Forward-fill NaNs for sparse evaluation
        mask = np.isnan(losses)
        if mask.any():
            idx = np.where(~mask, np.arange(len(losses)), 0)
            np.maximum.accumulate(idx, out=idx)
            losses = losses[idx]
        return params, losses

    def run_with_recording(self, initial_params, loss_fn, grad_fn, data_x, data_y,
                           n_steps, record_every=10):
        """Run SGD recording parameter trajectory at specified intervals."""
        params = np.array(initial_params, dtype=np.float64)
        n_records = n_steps // record_every + 1
        trajectory = np.empty((n_records, len(params)))
        losses = np.empty(n_records)
        grad_norms = np.empty(n_records)
        rec_idx = 0

        trajectory[0] = params.copy()
        losses[0] = loss_fn(params, data_x, data_y)
        grad_norms[0] = np.linalg.norm(self.full_gradient(params, grad_fn, data_x, data_y))
        rec_idx = 1

        for t in range(1, n_steps + 1):
            batch_x, batch_y = self.sample_batch(data_x, data_y, self.batch_size)
            grad = grad_fn(params, batch_x, batch_y)
            params = self.step(params, grad)
            if t % record_every == 0 and rec_idx < n_records:
                trajectory[rec_idx] = params.copy()
                losses[rec_idx] = loss_fn(params, data_x, data_y)
                fg = self.full_gradient(params, grad_fn, data_x, data_y)
                grad_norms[rec_idx] = np.linalg.norm(fg)
                rec_idx += 1

        return {
            'trajectory': trajectory[:rec_idx],
            'losses': losses[:rec_idx],
            'grad_norms': grad_norms[:rec_idx],
            'steps': np.arange(0, rec_idx) * record_every,
        }

    def sample_batch(self, data_x, data_y, batch_size):
        """Random mini-batch sampling without replacement."""
        n = len(data_x)
        bs = min(batch_size, n)
        indices = self.rng.choice(n, size=bs, replace=False)
        return data_x[indices], data_y[indices]

    def full_gradient(self, params, grad_fn, data_x, data_y):
        """Full-batch gradient over entire dataset."""
        return grad_fn(params, data_x, data_y)

    def stochastic_gradient(self, params, grad_fn, data_x, data_y, batch_size):
        """Mini-batch stochastic gradient."""
        batch_x, batch_y = self.sample_batch(data_x, data_y, batch_size)
        return grad_fn(params, batch_x, batch_y)

    def gradient_variance(self, params, grad_fn, data_x, data_y, n_samples=50):
        """Estimate Var(∇L_batch) via repeated mini-batch sampling."""
        grads = np.array([
            self.stochastic_gradient(params, grad_fn, data_x, data_y, self.batch_size)
            for _ in range(n_samples)
        ])
        mean_grad = grads.mean(axis=0)
        # Per-coordinate variance and total trace
        coord_var = np.var(grads, axis=0, ddof=1)
        return {
            'per_coordinate': coord_var,
            'trace': np.sum(coord_var),
            'mean_gradient': mean_grad,
            'grad_norm_sq': np.dot(mean_grad, mean_grad),
        }

    def loss_trajectory(self, param_trajectory, loss_fn, data_x, data_y):
        """Compute loss along a recorded parameter trajectory."""
        return np.array([loss_fn(p, data_x, data_y) for p in param_trajectory])


class LearningRatePhaseAnalyzer:
    """Learning rate effects on phase transitions in SGD training."""

    def __init__(self, lr_range=(1e-4, 10.0), n_lr_points=50):
        self.lr_range = lr_range
        self.n_lr_points = n_lr_points
        self.lr_values = np.geomspace(lr_range[0], lr_range[1], n_lr_points)

    def scan_learning_rates(self, loss_fn, grad_fn, data, initial_params,
                            n_steps=2000, batch_size=32):
        """Train at each learning rate and collect results."""
        data_x, data_y = data
        results = {
            'lr_values': self.lr_values.copy(),
            'final_losses': np.full(self.n_lr_points, np.inf),
            'converged': np.zeros(self.n_lr_points, dtype=bool),
            'diverged': np.zeros(self.n_lr_points, dtype=bool),
            'convergence_steps': np.full(self.n_lr_points, n_steps),
            'loss_histories': [],
        }
        for i, lr in enumerate(self.lr_values):
            sim = SGDSimulator(len(initial_params), learning_rate=lr, batch_size=batch_size)
            try:
                params, losses = sim.run(initial_params.copy(), loss_fn, grad_fn,
                                         data_x, data_y, n_steps)
                final_loss = loss_fn(params, data_x, data_y)
                results['final_losses'][i] = final_loss
                results['loss_histories'].append(losses)
                if np.isfinite(final_loss) and final_loss < 1e6:
                    results['converged'][i] = True
                    # Estimate convergence step: first time loss < 1.1 * final
                    threshold = 1.1 * final_loss + 1e-8
                    conv_idx = np.where(losses <= threshold)[0]
                    if len(conv_idx) > 0:
                        results['convergence_steps'][i] = conv_idx[0]
                else:
                    results['diverged'][i] = True
            except (FloatingPointError, OverflowError):
                results['diverged'][i] = True
                results['loss_histories'].append(np.full(n_steps, np.inf))
        return results

    def find_critical_lr(self, lr_scan_results):
        """Find the learning rate at the convergence/divergence phase transition."""
        converged = lr_scan_results['converged']
        lr_vals = lr_scan_results['lr_values']
        if not np.any(converged):
            return lr_vals[0]
        if np.all(converged):
            return lr_vals[-1]
        # Critical LR is the largest converging LR
        max_converged_idx = np.where(converged)[0][-1]
        if max_converged_idx + 1 < len(lr_vals):
            lr_crit = np.sqrt(lr_vals[max_converged_idx] * lr_vals[max_converged_idx + 1])
        else:
            lr_crit = lr_vals[max_converged_idx]
        return lr_crit

    def convergence_vs_lr(self, lr_scan_results):
        """Convergence rate as a function of learning rate."""
        converged = lr_scan_results['converged']
        lr_vals = lr_scan_results['lr_values'][converged]
        conv_steps = lr_scan_results['convergence_steps'][converged]
        rates = np.zeros_like(lr_vals)
        for i, hist in enumerate(
            [lr_scan_results['loss_histories'][j]
             for j in np.where(converged)[0]]
        ):
            valid = hist[hist > 0]
            if len(valid) > 10:
                log_loss = np.log(valid[:len(valid)//2] + 1e-30)
                t = np.arange(len(log_loss))
                if len(t) > 1:
                    slope, _ = np.polyfit(t, log_loss, 1)
                    rates[i] = -slope
        return {'lr_values': lr_vals, 'convergence_rates': rates,
                'convergence_steps': conv_steps}

    def final_loss_vs_lr(self, lr_scan_results):
        """Final loss as a function of learning rate."""
        return {
            'lr_values': lr_scan_results['lr_values'],
            'final_losses': lr_scan_results['final_losses'],
            'converged': lr_scan_results['converged'],
        }

    def edge_of_stability_analysis(self, lr, hessian_fn, trajectory):
        """Detect sharpness ≈ 2/η (edge of stability phenomenon).

        Tracks the top Hessian eigenvalue along the trajectory and checks
        whether it converges to the progressive sharpening limit 2/η.
        """
        n_points = len(trajectory)
        sharpness = np.empty(n_points)
        eos_threshold = 2.0 / lr
        for i, params in enumerate(trajectory):
            H = hessian_fn(params)
            eig_max = np.max(np.real(np.linalg.eigvalsh(H)))
            sharpness[i] = eig_max

        # Detect EoS: sharpness stabilizes near 2/η
        tail = sharpness[n_points // 2:]
        mean_sharpness = np.mean(tail)
        at_eos = abs(mean_sharpness - eos_threshold) / eos_threshold < 0.15
        onset_idx = 0
        if at_eos:
            near_eos = np.abs(sharpness - eos_threshold) / eos_threshold < 0.2
            onset_candidates = np.where(near_eos)[0]
            if len(onset_candidates) > 0:
                onset_idx = onset_candidates[0]

        return {
            'sharpness': sharpness,
            'eos_threshold': eos_threshold,
            'at_eos': at_eos,
            'mean_tail_sharpness': mean_sharpness,
            'onset_step': onset_idx,
        }

    def lr_warmup_effect(self, warmup_schedules, loss_fn, grad_fn, data, initial_params,
                         n_steps=2000, batch_size=32):
        """Compare different learning rate warmup strategies."""
        data_x, data_y = data
        results = {}
        for name, schedule_fn in warmup_schedules.items():
            params = np.array(initial_params, dtype=np.float64)
            sim = SGDSimulator(len(params), batch_size=batch_size)
            losses = np.empty(n_steps)
            for t in range(n_steps):
                lr_t = schedule_fn(t, n_steps)
                batch_x, batch_y = sim.sample_batch(data_x, data_y, batch_size)
                grad = grad_fn(params, batch_x, batch_y)
                params = sim.step(params, grad, lr=lr_t)
                losses[t] = loss_fn(params, data_x, data_y)
            results[name] = {
                'final_params': params,
                'final_loss': losses[-1],
                'loss_history': losses,
            }
        return results

    def lr_phase_diagram(self, lr_range, width_range, loss_fn, grad_fn, data_fn,
                         n_lr=30, n_width=20, n_steps=1000, batch_size=32):
        """2D phase diagram over (learning rate, network width)."""
        lr_vals = np.geomspace(lr_range[0], lr_range[1], n_lr)
        width_vals = np.linspace(width_range[0], width_range[1], n_width, dtype=int)
        phase_map = np.full((n_lr, n_width), np.nan)
        for i, lr in enumerate(lr_vals):
            for j, width in enumerate(width_vals):
                data_x, data_y, init_params = data_fn(width)
                sim = SGDSimulator(len(init_params), learning_rate=lr, batch_size=batch_size)
                try:
                    params, losses = sim.run(init_params, loss_fn, grad_fn,
                                             data_x, data_y, n_steps)
                    fl = loss_fn(params, data_x, data_y)
                    phase_map[i, j] = fl if np.isfinite(fl) else np.inf
                except (FloatingPointError, OverflowError):
                    phase_map[i, j] = np.inf
        return {
            'lr_values': lr_vals,
            'width_values': width_vals,
            'phase_map': phase_map,
        }


class BatchSizeEffectAnalyzer:
    """Batch size effects on SGD dynamics and phase transitions."""

    def __init__(self, batch_sizes=None):
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.batch_sizes = np.array(batch_sizes)

    def scan_batch_sizes(self, loss_fn, grad_fn, data, initial_params, lr,
                         n_steps=2000):
        """Train at each batch size and collect results."""
        data_x, data_y = data
        n_data = len(data_x)
        results = {
            'batch_sizes': self.batch_sizes.copy(),
            'final_losses': np.full(len(self.batch_sizes), np.inf),
            'convergence_steps': np.full(len(self.batch_sizes), n_steps),
            'loss_histories': [],
            'grad_variances': np.zeros(len(self.batch_sizes)),
        }
        for i, bs in enumerate(self.batch_sizes):
            bs = min(int(bs), n_data)
            sim = SGDSimulator(len(initial_params), learning_rate=lr, batch_size=bs)
            try:
                params, losses = sim.run(initial_params.copy(), loss_fn, grad_fn,
                                         data_x, data_y, n_steps)
                fl = loss_fn(params, data_x, data_y)
                results['final_losses'][i] = fl
                results['loss_histories'].append(losses)
                var_info = sim.gradient_variance(params, grad_fn, data_x, data_y, n_samples=30)
                results['grad_variances'][i] = var_info['trace']
                if np.isfinite(fl):
                    threshold = 1.1 * fl + 1e-8
                    conv_idx = np.where(losses <= threshold)[0]
                    if len(conv_idx) > 0:
                        results['convergence_steps'][i] = conv_idx[0]
            except (FloatingPointError, OverflowError):
                results['loss_histories'].append(np.full(n_steps, np.inf))
        return results

    def critical_batch_size(self, batch_scan_results):
        """Estimate B* where linear scaling of throughput breaks down.

        Below B*, doubling batch size halves required steps (linear scaling).
        Above B*, doubling batch size gives diminishing returns.
        """
        bs = batch_scan_results['batch_sizes'].astype(float)
        steps = batch_scan_results['convergence_steps'].astype(float)
        finite = np.isfinite(steps) & (steps > 0)
        bs, steps = bs[finite], steps[finite]
        if len(bs) < 3:
            return bs[0] if len(bs) > 0 else 32
        # In linear scaling regime: steps ∝ 1/B, so B*steps = const.
        # B* is where B*steps starts increasing.
        throughput = bs * steps  # should be ~constant below B*
        # Find where throughput starts increasing significantly
        ratios = throughput[1:] / throughput[:-1]
        deviation_idx = np.where(ratios > 1.3)[0]
        if len(deviation_idx) > 0:
            return bs[deviation_idx[0]]
        return bs[-1]

    def noise_scale(self, batch_size, dataset_size, gradient_variance):
        """Compute noise scale B_noise = (tr(Σ) · B) / (N · ||∇L||²).

        This is the effective noise temperature of SGD.
        """
        grad_var_trace = gradient_variance['trace']
        grad_norm_sq = gradient_variance['grad_norm_sq']
        if grad_norm_sq < 1e-30:
            return np.inf
        # Noise scale: η * tr(Σ) / (2B) is the effective temperature
        # Simple noise scale: tr(Σ) / ||∇L||²
        return grad_var_trace / grad_norm_sq

    def linear_scaling_rule(self, base_lr, base_batch, new_batch):
        """Linear scaling rule: η_new = η_base × B_new / B_base."""
        return base_lr * new_batch / base_batch

    def gradient_noise_ratio(self, params, grad_fn, data_x, data_y, batch_size):
        """Signal-to-noise ratio of mini-batch gradients."""
        sim = SGDSimulator(len(params), batch_size=batch_size)
        var_info = sim.gradient_variance(params, grad_fn, data_x, data_y, n_samples=50)
        signal = var_info['grad_norm_sq']
        noise = var_info['trace']
        if noise < 1e-30:
            return np.inf
        return np.sqrt(signal / noise)

    def optimal_batch_size(self, compute_budget, per_sample_cost):
        """Optimal batch size for fixed compute budget.

        Balances gradient quality (larger B) vs number of steps (smaller B).
        Optimal B* ≈ B_noise (the noise scale).
        """
        max_batch = int(compute_budget / per_sample_cost)
        # Without noise scale info, return geometric mean heuristic
        return min(max_batch, max(1, int(np.sqrt(max_batch))))

    def batch_size_phase_diagram(self, batch_range, lr_range, loss_fn, grad_fn, data,
                                 n_batch=15, n_lr=20, n_steps=1000):
        """Phase diagram over (batch size, learning rate)."""
        data_x, data_y = data
        bs_vals = np.geomspace(batch_range[0], batch_range[1], n_batch).astype(int)
        bs_vals = np.unique(bs_vals)
        lr_vals = np.geomspace(lr_range[0], lr_range[1], n_lr)
        phase_map = np.full((len(bs_vals), len(lr_vals)), np.nan)
        n_data = len(data_x)
        param_dim = None

        for i, bs in enumerate(bs_vals):
            for j, lr in enumerate(lr_vals):
                # Need initial params - infer from data dim
                if param_dim is None:
                    # Run with a dummy to get param dim
                    continue
                sim = SGDSimulator(param_dim, learning_rate=lr, batch_size=min(int(bs), n_data))
                try:
                    params_init = np.zeros(param_dim)
                    params, losses = sim.run(params_init, loss_fn, grad_fn,
                                             data_x, data_y, n_steps)
                    fl = loss_fn(params, data_x, data_y)
                    phase_map[i, j] = fl if np.isfinite(fl) else np.inf
                except (FloatingPointError, OverflowError):
                    phase_map[i, j] = np.inf
        return {
            'batch_sizes': bs_vals,
            'lr_values': lr_vals,
            'phase_map': phase_map,
        }


class MomentumDynamics:
    """Momentum effects on SGD dynamics and convergence."""

    def __init__(self, momentum=0.9, nesterov=False):
        self.momentum = momentum
        self.nesterov = nesterov

    def sgd_momentum_step(self, params, gradient, velocity, lr, momentum=None):
        """Heavy ball momentum update:
            v_{t+1} = β v_t + ∇L(θ_t)
            θ_{t+1} = θ_t - η v_{t+1}
        """
        if momentum is None:
            momentum = self.momentum
        velocity_new = momentum * velocity + gradient
        params_new = params - lr * velocity_new
        return params_new, velocity_new

    def nesterov_step(self, params, gradient, velocity, lr, momentum=None):
        """Nesterov accelerated gradient:
            v_{t+1} = β v_t + ∇L(θ_t - η β v_t)  [gradient at lookahead]
            θ_{t+1} = θ_t - η v_{t+1}
        Here gradient is assumed already evaluated at the lookahead point.
        """
        if momentum is None:
            momentum = self.momentum
        velocity_new = momentum * velocity + gradient
        params_new = params - lr * (momentum * velocity_new + gradient)
        return params_new, velocity_new

    def run_with_momentum(self, initial_params, loss_fn, grad_fn, data_x, data_y,
                          n_steps, lr=0.01, batch_size=32, record_every=10):
        """Full SGD+momentum run with trajectory recording."""
        params = np.array(initial_params, dtype=np.float64)
        velocity = np.zeros_like(params)
        sim = SGDSimulator(len(params), learning_rate=lr, batch_size=batch_size)
        n_records = n_steps // record_every + 1
        trajectory = np.empty((n_records, len(params)))
        losses = np.empty(n_records)
        rec_idx = 0

        trajectory[0] = params.copy()
        losses[0] = loss_fn(params, data_x, data_y)
        rec_idx = 1

        for t in range(1, n_steps + 1):
            batch_x, batch_y = sim.sample_batch(data_x, data_y, batch_size)
            if self.nesterov:
                # Evaluate gradient at lookahead position
                lookahead = params - lr * self.momentum * velocity
                grad = grad_fn(lookahead, batch_x, batch_y)
                params, velocity = self.nesterov_step(params, grad, velocity, lr)
            else:
                grad = grad_fn(params, batch_x, batch_y)
                params, velocity = self.sgd_momentum_step(params, grad, velocity, lr)

            if t % record_every == 0 and rec_idx < n_records:
                trajectory[rec_idx] = params.copy()
                losses[rec_idx] = loss_fn(params, data_x, data_y)
                rec_idx += 1

        return {
            'trajectory': trajectory[:rec_idx],
            'losses': losses[:rec_idx],
            'final_params': params,
            'final_velocity': velocity,
        }

    def effective_learning_rate(self, lr, momentum=None):
        """Effective learning rate with momentum: η_eff = η / (1 - β)."""
        if momentum is None:
            momentum = self.momentum
        if momentum >= 1.0:
            return np.inf
        return lr / (1.0 - momentum)

    def momentum_eigenvalue_analysis(self, hessian, lr, momentum=None):
        """Stability analysis of momentum SGD in terms of Hessian eigenvalues.

        The momentum update forms a 2D linear system per eigendirection:
            [θ; v]_{t+1} = M [θ; v]_t where M = [[1-ηλ, -ηβ λ?], ...]
        Stability requires spectral radius of M < 1.
        """
        if momentum is None:
            momentum = self.momentum
        eigenvalues = np.linalg.eigvalsh(hessian)
        eigenvalues = np.sort(np.real(eigenvalues))

        # For each Hessian eigenvalue λ, the momentum iteration matrix is:
        #   M = [[1 - η λ, -η β], [λ, β]]  (in the (θ, v) basis per mode)
        # Wait, standard form: v' = βv + g, θ' = θ - η v'
        # Per-mode: v' = βv + λθ, θ' = θ - η(βv + λθ) = (1-ηλ)θ - ηβv
        # M = [[(1-ηλ), -ηβ], [λ, β]]  -- but this mixes. Let's use:
        # v_{t+1} = β v_t + λ θ_t
        # θ_{t+1} = (1 - ηλ) θ_t - ηβ v_t
        # So M = [[(1-ηλ), -ηβ], [λ, β]]

        stable = np.ones(len(eigenvalues), dtype=bool)
        spectral_radii = np.zeros(len(eigenvalues))
        for i, lam in enumerate(eigenvalues):
            M = np.array([[1 - lr * lam, -lr * momentum],
                          [lam, momentum]])
            eigs_M = np.linalg.eigvals(M)
            sr = np.max(np.abs(eigs_M))
            spectral_radii[i] = sr
            stable[i] = sr < 1.0

        max_stable_lr = np.inf
        if len(eigenvalues) > 0 and eigenvalues[-1] > 0:
            # Critical lr for largest eigenvalue
            lam_max = eigenvalues[-1]
            # For heavy ball: stability requires η < 2(1+β)/(λ_max)
            max_stable_lr = 2 * (1 + momentum) / lam_max

        return {
            'hessian_eigenvalues': eigenvalues,
            'spectral_radii': spectral_radii,
            'stable': stable,
            'all_stable': np.all(stable),
            'max_stable_lr': max_stable_lr,
        }

    def optimal_momentum(self, hessian_eigenvalues, lr):
        """Optimal momentum β for given Hessian spectrum and learning rate.

        For quadratic loss with condition number κ = λ_max/λ_min,
        optimal β = ((√κ - 1) / (√κ + 1))².
        """
        eigenvalues = np.sort(np.abs(hessian_eigenvalues))
        pos_eigs = eigenvalues[eigenvalues > 1e-10]
        if len(pos_eigs) < 2:
            return 0.0
        kappa = pos_eigs[-1] / pos_eigs[0]
        sqrt_kappa = np.sqrt(kappa)
        beta_opt = ((sqrt_kappa - 1) / (sqrt_kappa + 1)) ** 2
        return float(np.clip(beta_opt, 0.0, 0.999))

    def oscillation_detection(self, trajectory, window=50):
        """Detect oscillatory behavior in parameter trajectory."""
        n_points = len(trajectory)
        if n_points < 2 * window:
            return {'oscillating': False, 'frequency': 0.0, 'amplitude': 0.0}

        # Compute displacement norms
        displacements = np.diff(trajectory, axis=0)
        disp_norms = np.linalg.norm(displacements, axis=1)

        # Detect sign changes in displacement along principal direction
        if trajectory.shape[1] > 1:
            # Project onto top PCA direction
            centered = trajectory - trajectory.mean(axis=0)
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ Vt[0]
        else:
            proj = trajectory[:, 0]

        # Count zero crossings of the detrended projection
        detrended = proj - np.linspace(proj[0], proj[-1], len(proj))
        tail = detrended[-window:]
        crossings = np.sum(np.diff(np.sign(tail)) != 0)
        frequency = crossings / (2.0 * window)
        amplitude = np.std(tail)

        return {
            'oscillating': frequency > 0.05,
            'frequency': frequency,
            'amplitude': amplitude,
            'displacement_norms': disp_norms,
        }

    def damping_analysis(self, trajectory, eigenvalues):
        """Relate trajectory damping to momentum and eigenvalue spectrum."""
        n = len(trajectory)
        if n < 10:
            return {'damping_rates': np.array([]), 'underdamped': False}

        norms = np.linalg.norm(trajectory - trajectory[-1], axis=1)
        norms = norms[norms > 1e-15]
        if len(norms) < 5:
            return {'damping_rates': np.array([0.0]), 'underdamped': False}

        log_norms = np.log(norms + 1e-30)
        t = np.arange(len(log_norms))
        slope, intercept = np.polyfit(t, log_norms, 1)
        damping_rate = -slope

        # Check for underdamped oscillations
        residuals = log_norms - (slope * t + intercept)
        osc = MomentumDynamics.oscillation_detection(self,
            residuals.reshape(-1, 1), window=min(50, len(residuals) // 2))

        # Theoretical critical damping for quadratic
        pos_eigs = eigenvalues[eigenvalues > 1e-10]
        if len(pos_eigs) > 0:
            critical_momentum = (1 - np.sqrt(self.momentum)) ** 2  # placeholder
        else:
            critical_momentum = 0.0

        return {
            'damping_rate': damping_rate,
            'underdamped': osc['oscillating'],
            'oscillation_frequency': osc['frequency'],
            'critical_momentum': critical_momentum,
        }


class SGDNoiseCovarianceEstimator:
    """Estimate the noise covariance of stochastic gradients."""

    def __init__(self, n_samples=100):
        self.n_samples = n_samples
        self.rng = np.random.default_rng()

    def estimate_covariance(self, params, grad_fn, data_x, data_y, batch_size):
        """Estimate Σ = Cov(g_B) via repeated mini-batch sampling."""
        n_data = len(data_x)
        bs = min(batch_size, n_data)
        grads = []
        for _ in range(self.n_samples):
            idx = self.rng.choice(n_data, size=bs, replace=False)
            g = grad_fn(params, data_x[idx], data_y[idx])
            grads.append(g)
        grads = np.array(grads)
        mean_grad = grads.mean(axis=0)
        centered = grads - mean_grad
        cov = (centered.T @ centered) / (self.n_samples - 1)
        return cov

    def low_rank_approximation(self, covariance, rank=10):
        """Low-rank approximation Σ ≈ U Λ U^T via truncated eigendecomposition."""
        d = covariance.shape[0]
        k = min(rank, d)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        # Take top-k (largest eigenvalues are at the end for eigh)
        idx = np.argsort(eigenvalues)[::-1][:k]
        top_eigenvalues = eigenvalues[idx]
        top_eigenvectors = eigenvectors[:, idx]
        # Clamp negative eigenvalues (numerical)
        top_eigenvalues = np.maximum(top_eigenvalues, 0)
        approx = top_eigenvectors @ np.diag(top_eigenvalues) @ top_eigenvectors.T
        reconstruction_error = np.linalg.norm(covariance - approx, 'fro')
        explained_variance = np.sum(top_eigenvalues) / max(np.trace(covariance), 1e-30)
        return {
            'eigenvalues': top_eigenvalues,
            'eigenvectors': top_eigenvectors,
            'approximation': approx,
            'reconstruction_error': reconstruction_error,
            'explained_variance_ratio': explained_variance,
        }

    def trace_estimation(self, params, grad_fn, data_x, data_y, batch_size,
                         n_hutchinson=20):
        """Hutchinson trace estimator: tr(Σ) ≈ (1/m) Σ_i z_i^T Σ z_i.

        Uses Rademacher random vectors to avoid forming Σ explicitly.
        """
        n_data = len(data_x)
        bs = min(batch_size, n_data)
        # First estimate mean gradient
        mean_grads = []
        for _ in range(min(self.n_samples, 50)):
            idx = self.rng.choice(n_data, size=bs, replace=False)
            mean_grads.append(grad_fn(params, data_x[idx], data_y[idx]))
        mean_grad = np.mean(mean_grads, axis=0)
        d = len(mean_grad)

        trace_estimates = []
        for _ in range(n_hutchinson):
            z = self.rng.choice([-1.0, 1.0], size=d)
            # Estimate z^T Σ z = E[(z^T (g - μ))²]
            quad_samples = []
            for _ in range(self.n_samples):
                idx = self.rng.choice(n_data, size=bs, replace=False)
                g = grad_fn(params, data_x[idx], data_y[idx])
                val = np.dot(z, g - mean_grad)
                quad_samples.append(val ** 2)
            trace_estimates.append(np.mean(quad_samples))

        tr_est = np.mean(trace_estimates)
        tr_std = np.std(trace_estimates) / np.sqrt(n_hutchinson)
        return {'trace': tr_est, 'std': tr_std, 'estimates': np.array(trace_estimates)}

    def top_eigenvalues(self, covariance, k=10):
        """Top-k eigenvalues and eigenvectors of the noise covariance."""
        d = covariance.shape[0]
        k = min(k, d)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        idx = np.argsort(eigenvalues)[::-1][:k]
        return {
            'eigenvalues': eigenvalues[idx],
            'eigenvectors': eigenvectors[:, idx],
        }

    def noise_alignment(self, covariance, hessian):
        """Measure alignment between noise covariance and loss Hessian.

        Computes tr(Σ H) / (||Σ||_F ||H||_F) as an alignment metric.
        """
        alignment_trace = np.trace(covariance @ hessian)
        norm_product = np.linalg.norm(covariance, 'fro') * np.linalg.norm(hessian, 'fro')
        if norm_product < 1e-30:
            normalized = 0.0
        else:
            normalized = alignment_trace / norm_product

        # Also compute eigenspace overlap
        eig_cov = np.linalg.eigh(covariance)
        eig_hess = np.linalg.eigh(hessian)
        # Top eigenvector overlap
        v_cov = eig_cov[1][:, -1]
        v_hess = eig_hess[1][:, -1]
        top_overlap = np.abs(np.dot(v_cov, v_hess))

        return {
            'alignment_trace': alignment_trace,
            'normalized_alignment': normalized,
            'top_eigenvector_overlap': top_overlap,
        }

    def effective_noise_dimension(self, covariance):
        """Effective dimension d_eff = tr(Σ)² / tr(Σ²)."""
        tr = np.trace(covariance)
        tr_sq = np.trace(covariance @ covariance)
        if tr_sq < 1e-30:
            return 0.0
        return tr ** 2 / tr_sq

    def noise_anisotropy(self, covariance):
        """Anisotropy ratio λ_max / λ_min of the noise covariance."""
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.sort(np.abs(eigenvalues))
        pos = eigenvalues[eigenvalues > 1e-15]
        if len(pos) < 2:
            return {'ratio': 1.0, 'eigenvalues': eigenvalues}
        return {
            'ratio': pos[-1] / pos[0],
            'max_eigenvalue': pos[-1],
            'min_eigenvalue': pos[0],
            'eigenvalues': eigenvalues,
        }


class SGDtoSDEConverter:
    """Convert discrete SGD dynamics to continuous-time SDE approximation.

    The SGD update θ_{t+1} = θ_t - η g_B(θ_t) can be approximated by
    dθ = -∇L(θ) dt + √(η Σ(θ)/B) dW  in the small-η limit.
    """

    def __init__(self, learning_rate, batch_size, dataset_size):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.temperature = learning_rate / batch_size
        self.rng = np.random.default_rng()

    def drift_coefficient(self, params, grad_fn, data_x, data_y):
        """Drift = -∇L(θ), the negative full-batch gradient."""
        return -grad_fn(params, data_x, data_y)

    def diffusion_coefficient(self, params, grad_fn, data_x, data_y, batch_size=None):
        """Diffusion matrix √(η/B · Σ(θ)).

        Returns the matrix square root of (η/B) Σ where Σ is the gradient
        noise covariance.
        """
        if batch_size is None:
            batch_size = self.batch_size
        estimator = SGDNoiseCovarianceEstimator(n_samples=50)
        cov = estimator.estimate_covariance(params, grad_fn, data_x, data_y, batch_size)
        scaled_cov = (self.learning_rate / batch_size) * cov
        # Matrix square root via eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(scaled_cov)
        eigenvalues = np.maximum(eigenvalues, 0)
        sqrt_cov = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
        return sqrt_cov

    def sde_simulation(self, initial_params, drift_fn, diffusion_fn, dt, n_steps):
        """Euler-Maruyama simulation of the SDE:
            dθ = a(θ) dt + B(θ) dW

        Args:
            drift_fn: θ -> drift vector a(θ)
            diffusion_fn: θ -> diffusion matrix B(θ)
            dt: time step
            n_steps: number of integration steps
        """
        d = len(initial_params)
        trajectory = np.empty((n_steps + 1, d))
        trajectory[0] = initial_params.copy()
        params = np.array(initial_params, dtype=np.float64)

        sqrt_dt = np.sqrt(dt)
        for t in range(n_steps):
            drift = drift_fn(params)
            diffusion = diffusion_fn(params)
            dW = self.rng.standard_normal(d) * sqrt_dt
            params = params + drift * dt + diffusion @ dW
            trajectory[t + 1] = params.copy()

        return trajectory

    def stationary_distribution(self, drift, diffusion, param_range, n_grid=200):
        """Compute stationary distribution for 1D SDE.

        P_ss(x) ∝ exp(-2 ∫₀ˣ a(y)/D(y)² dy) / D(x)²

        where dX = a(X)dt + D(X)dW.
        """
        x = np.linspace(param_range[0], param_range[1], n_grid)
        dx = x[1] - x[0]
        a = np.array([drift(xi) if callable(drift) else drift for xi in x])
        D = np.array([diffusion(xi) if callable(diffusion) else diffusion for xi in x])
        D = np.maximum(np.abs(D), 1e-15)

        # Compute potential: φ(x) = -2 ∫₀ˣ a(y)/D(y)² dy
        integrand = 2.0 * a / (D ** 2)
        potential = np.cumsum(integrand) * dx
        potential -= potential.min()

        log_prob = -potential - 2 * np.log(D)
        log_prob -= np.max(log_prob)  # numerical stability
        prob = np.exp(log_prob)
        prob /= np.trapz(prob, x)

        return {'x': x, 'density': prob, 'potential': potential}

    def mean_first_passage_time(self, drift, diffusion, start, barrier, n_grid=500):
        """Mean first passage time from start to barrier for 1D SDE.

        Uses the formula:
        T(x) = 2 ∫_x^b (1/D(y)²) exp(φ(y)) ∫_a^y (1/D(z)²) exp(-φ(z)) dz dy
        where φ(x) = -2 ∫ a(x)/D(x)² dx.
        """
        if start > barrier:
            start, barrier = barrier, start

        x = np.linspace(start, barrier, n_grid)
        dx = x[1] - x[0]
        a = np.array([drift(xi) if callable(drift) else drift for xi in x])
        D = np.array([diffusion(xi) if callable(diffusion) else diffusion for xi in x])
        D = np.maximum(np.abs(D), 1e-15)

        # Potential φ(x) = -2 ∫ a(y)/D²(y) dy
        integrand = 2.0 * a / (D ** 2)
        phi = np.cumsum(integrand) * dx

        # Scale function s(x) = exp(φ(x))
        # Speed density m(x) = 1 / (D² s(x))
        s = np.exp(phi - phi[0])  # normalize for stability
        m = 1.0 / (D ** 2 * s)

        # MFPT: T(x_0) = 2 ∫_{x_0}^{barrier} s(y) [∫_{start}^{y} m(z) dz] dy
        cum_m = np.cumsum(m) * dx
        integrand_outer = s * cum_m
        mfpt = 2.0 * np.trapz(integrand_outer, x)

        return max(mfpt, 0.0)

    def kramers_rate(self, barrier_height, temperature=None):
        """Kramers escape rate: k ∝ exp(-ΔF / T).

        For SGD, T = η/(2B) · tr(Σ) is the effective temperature.
        """
        if temperature is None:
            temperature = self.temperature
        if temperature < 1e-30:
            return 0.0
        prefactor = 1.0  # depends on curvature at minimum and saddle
        rate = prefactor * np.exp(-barrier_height / temperature)
        return rate

    def sde_error_bound(self, lr, batch_size, lipschitz_constant):
        """Bound on the weak error of the SGD-SDE approximation.

        The weak error is O(η) where η is the learning rate,
        under smoothness assumptions on the loss.

        For Lipschitz constant L of ∇²f:
            E[|θ_SGD - θ_SDE|²] ≤ C · η · L · T
        where T is the time horizon.
        """
        # Leading order weak error bound
        error = lr * lipschitz_constant
        # Strong error bound (higher order)
        strong_error = np.sqrt(lr) * lipschitz_constant
        return {
            'weak_error': error,
            'strong_error': strong_error,
            'temperature': lr / batch_size,
            'noise_strength': lr / batch_size,
            'valid_regime': lr * lipschitz_constant < 0.1,
        }
