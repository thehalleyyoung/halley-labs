"""
Depth scaling analysis for neural network phase diagrams.

Implements kernel propagation through depth, signal propagation analysis
(edge of chaos), depth-dependent phase boundaries, optimal depth prediction,
and depth-width interaction effects.
"""

import numpy as np
from scipy import optimize, integrate, special, interpolate


class KernelDepthPropagation:
    """Kernel propagation through network depth via mean-field recursion."""

    def __init__(self, activation='relu', weight_variance=2.0, bias_variance=0.0):
        self.activation = activation
        self.sigma_w2 = weight_variance
        self.sigma_b2 = bias_variance

    def single_layer_kernel_map(self, K_in):
        """
        Compute K^{l+1} = sigma_w^2 * E[sigma(h) sigma(h')] + sigma_b^2.

        K_in can be a scalar (diagonal) or a 2x2 matrix [[K_11, K_12], [K_21, K_22]].
        """
        K_in = np.asarray(K_in, dtype=np.float64)
        if K_in.ndim == 0:
            # Scalar: diagonal element, compute E[sigma(h)^2] with h ~ N(0, K_in)
            expectation = self.dual_activation_function(
                np.array([[K_in, K_in], [K_in, K_in]]), self.activation
            )
            return self.sigma_w2 * expectation + self.sigma_b2
        elif K_in.shape == (2, 2):
            K_out = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    K_sub = np.array([[K_in[i, i], K_in[i, j]],
                                      [K_in[j, i], K_in[j, j]]])
                    K_out[i, j] = self.sigma_w2 * self.dual_activation_function(
                        K_sub, self.activation
                    ) + self.sigma_b2
            return K_out
        else:
            raise ValueError(f"K_in must be scalar or 2x2, got shape {K_in.shape}")

    def propagate_kernel(self, K_input, depth):
        """Iterate kernel through L layers, returning final kernel."""
        K = np.asarray(K_input, dtype=np.float64).copy()
        for _ in range(depth):
            K = self.single_layer_kernel_map(K)
        return K

    def kernel_fixed_point(self, initial_diagonal=1.0, tol=1e-12):
        """Find fixed point q* of the diagonal kernel recursion q -> F(q)."""
        def F(q):
            return float(self.single_layer_kernel_map(np.float64(q)))

        q = initial_diagonal
        for _ in range(10000):
            q_new = F(q)
            if abs(q_new - q) < tol:
                return q_new
            q = q_new
        return q

    def approach_to_fixed_point(self, K_input, depth, fixed_point):
        """
        Compute the rate of approach to q* as a function of layer.

        Returns array of |K^l - q*| for each layer.
        """
        trajectory = self.kernel_trajectory(K_input, max_depth=depth)
        distances = np.abs(trajectory - fixed_point)
        return distances

    def dual_activation_function(self, K_in, activation='relu'):
        """
        Compute E[sigma(h1) sigma(h2)] analytically for joint Gaussian (h1,h2).

        K_in is 2x2: [[K_11, K_12], [K_21, K_22]].
        """
        K_in = np.asarray(K_in, dtype=np.float64)
        K_11 = K_in[0, 0]
        K_22 = K_in[1, 1]
        K_12 = K_in[0, 1]

        denom = np.sqrt(K_11 * K_22)
        if denom < 1e-30:
            return 0.0
        rho = np.clip(K_12 / denom, -1.0, 1.0)

        if activation == 'relu':
            # Analytical formula: E[ReLU(h1) ReLU(h2)]
            # = (sqrt(K11 K22) / (2pi)) * (sin(theta) + (pi - theta) cos(theta))
            # where theta = arccos(rho)
            theta = np.arccos(rho)
            result = (np.sqrt(K_11 * K_22) / (2.0 * np.pi)) * (
                np.sin(theta) + (np.pi - theta) * np.cos(theta)
            )
            return float(result)
        elif activation == 'erf':
            # E[erf(h1) erf(h2)] = (2/pi) arcsin(2 K12 / sqrt((1+2K11)(1+2K22)))
            arg = 2.0 * K_12 / np.sqrt((1.0 + 2.0 * K_11) * (1.0 + 2.0 * K_22))
            arg = np.clip(arg, -1.0, 1.0)
            return float((2.0 / np.pi) * np.arcsin(arg))
        elif activation == 'tanh':
            # Numerical integration for tanh
            def integrand(u1, u2):
                z1 = np.sqrt(K_11) * u1
                z2 = np.sqrt(K_22) * (rho * u1 + np.sqrt(1 - rho**2) * u2)
                gauss = np.exp(-0.5 * (u1**2 + u2**2)) / (2.0 * np.pi)
                return np.tanh(z1) * np.tanh(z2) * gauss

            result, _ = integrate.dblquad(
                integrand, -6, 6, -6, 6, epsabs=1e-10
            )
            return float(result)
        elif activation == 'linear':
            return float(K_12)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def correlation_propagation(self, rho_input, depth):
        """
        Propagate correlation rho = K_12 / sqrt(K_11 * K_22) through depth.

        Returns array of rho values at each layer.
        """
        q_star = self.kernel_fixed_point()
        rhos = np.zeros(depth + 1)
        rhos[0] = rho_input

        rho = rho_input
        for l in range(1, depth + 1):
            K_in = np.array([[q_star, rho * q_star],
                              [rho * q_star, q_star]])
            K_out = self.single_layer_kernel_map(K_in)
            q_out = K_out[0, 0]
            if q_out < 1e-30:
                rho = 0.0
            else:
                rho = np.clip(K_out[0, 1] / q_out, -1.0, 1.0)
            rhos[l] = rho
        return rhos

    def correlation_fixed_point(self):
        """Find fixed point rho* of the correlation propagation map."""
        q_star = self.kernel_fixed_point()

        def rho_map(rho):
            K_in = np.array([[q_star, rho * q_star],
                              [rho * q_star, q_star]])
            K_out = self.single_layer_kernel_map(K_in)
            q_out = K_out[0, 0]
            if q_out < 1e-30:
                return 0.0
            return np.clip(K_out[0, 1] / q_out, -1.0, 1.0)

        # rho=1 is always a fixed point; look for non-trivial one
        try:
            result = optimize.brentq(lambda r: rho_map(r) - r, -0.99, 0.99)
            return result
        except ValueError:
            return 1.0

    def jacobian_of_kernel_map(self, K, activation='relu'):
        """
        Compute dK^{l+1}/dK^l for stability analysis.

        For 2x2 kernel, returns 4x4 Jacobian via finite differences.
        """
        K = np.asarray(K, dtype=np.float64)
        eps = 1e-7

        if K.ndim == 0:
            K_plus = self.single_layer_kernel_map(K + eps)
            K_minus = self.single_layer_kernel_map(K - eps)
            return float((K_plus - K_minus) / (2 * eps))

        K_flat = K.flatten()
        n = len(K_flat)
        J = np.zeros((n, n))
        for i in range(n):
            K_p = K_flat.copy()
            K_m = K_flat.copy()
            K_p[i] += eps
            K_m[i] -= eps
            out_p = self.single_layer_kernel_map(K_p.reshape(K.shape)).flatten()
            out_m = self.single_layer_kernel_map(K_m.reshape(K.shape)).flatten()
            J[:, i] = (out_p - out_m) / (2 * eps)
        return J

    def kernel_trajectory(self, K_input, max_depth=100):
        """
        Compute full trajectory K^0, K^1, ..., K^L.

        For scalar input, returns 1D array. For 2x2, returns list of matrices.
        """
        K = np.asarray(K_input, dtype=np.float64).copy()
        if K.ndim == 0:
            trajectory = np.zeros(max_depth + 1)
            trajectory[0] = float(K)
            for l in range(1, max_depth + 1):
                K = self.single_layer_kernel_map(K)
                trajectory[l] = float(K)
            return trajectory
        else:
            trajectory = [K.copy()]
            for l in range(max_depth):
                K = self.single_layer_kernel_map(K)
                trajectory.append(K.copy())
            return trajectory


class SignalPropagationAnalyzer:
    """Signal propagation analysis at the edge of chaos."""

    def __init__(self, activation='relu'):
        self.activation = activation

    def chi_1(self, weight_variance, q_star):
        """
        Compute chi_1 = sigma_w^2 * E[sigma'(h)^2] where h ~ N(0, q*).

        This is the order-to-chaos transition parameter.
        """
        sigma_w2 = weight_variance
        if q_star < 1e-30:
            return 0.0

        if self.activation == 'relu':
            # E[ReLU'(h)^2] = E[1_{h>0}] = 1/2
            return sigma_w2 * 0.5
        elif self.activation == 'erf':
            # E[erf'(h)^2] = E[(2/sqrt(pi))^2 exp(-2h^2)]
            # = (4/pi) * 1/sqrt(1 + 4*q_star) (Gaussian integral)
            return sigma_w2 * (4.0 / np.pi) / np.sqrt(1.0 + 4.0 * q_star)
        elif self.activation == 'tanh':
            # E[sech^4(h)] via numerical integration, h ~ N(0, q*)
            def integrand(h):
                gauss = np.exp(-0.5 * h**2 / q_star) / np.sqrt(2 * np.pi * q_star)
                return (1.0 / np.cosh(h))**4 * gauss

            result, _ = integrate.quad(integrand, -10 * np.sqrt(q_star),
                                        10 * np.sqrt(q_star))
            return sigma_w2 * result
        elif self.activation == 'linear':
            return sigma_w2
        else:
            # Numerical: E[sigma'(h)^2] via finite difference
            def sigma(h):
                if self.activation == 'softplus':
                    return np.log1p(np.exp(h))
                return h

            def integrand(h):
                eps = 1e-5
                sp = (sigma(h + eps) - sigma(h - eps)) / (2 * eps)
                gauss = np.exp(-0.5 * h**2 / q_star) / np.sqrt(2 * np.pi * q_star)
                return sp**2 * gauss

            result, _ = integrate.quad(integrand, -10 * np.sqrt(q_star),
                                        10 * np.sqrt(q_star))
            return sigma_w2 * result

    def ordered_phase(self, chi_1_val):
        """
        Characterize the ordered phase (chi_1 < 1).

        Returns dict with phase properties: signals collapse exponentially.
        """
        if chi_1_val >= 1.0:
            return {'phase': 'not_ordered', 'chi_1': chi_1_val}

        depth_scale = self.depth_scale(chi_1_val)
        return {
            'phase': 'ordered',
            'chi_1': chi_1_val,
            'depth_scale': depth_scale,
            'convergence_rate': chi_1_val,
            'correlation_collapse': True,
            'gradient_vanishing': True,
            'lyapunov_exponent': np.log(chi_1_val),
        }

    def chaotic_phase(self, chi_1_val):
        """
        Characterize the chaotic phase (chi_1 > 1).

        Returns dict with phase properties: signals diverge exponentially.
        """
        if chi_1_val <= 1.0:
            return {'phase': 'not_chaotic', 'chi_1': chi_1_val}

        depth_scale = self.depth_scale(chi_1_val)
        return {
            'phase': 'chaotic',
            'chi_1': chi_1_val,
            'depth_scale': depth_scale,
            'divergence_rate': chi_1_val,
            'correlation_divergence': True,
            'gradient_exploding': True,
            'lyapunov_exponent': np.log(chi_1_val),
        }

    def edge_of_chaos(self, weight_variance_range):
        """
        Find sigma_w^2 where chi_1 = 1 (the critical point).

        Returns the critical weight variance.
        """
        sw_range = np.asarray(weight_variance_range, dtype=np.float64)

        def chi_minus_one(sw2):
            prop = KernelDepthPropagation(
                activation=self.activation, weight_variance=sw2, bias_variance=0.0
            )
            q_star = prop.kernel_fixed_point()
            return self.chi_1(sw2, q_star) - 1.0

        # Bracket the root
        vals = [chi_minus_one(sw2) for sw2 in sw_range]
        for i in range(len(vals) - 1):
            if vals[i] * vals[i + 1] < 0:
                result = optimize.brentq(chi_minus_one, sw_range[i], sw_range[i + 1])
                return result

        # If no sign change, return the value closest to chi_1=1
        idx = np.argmin(np.abs(vals))
        return float(sw_range[idx])

    def depth_scale(self, chi_1_val):
        """
        Compute characteristic depth scale xi = 1 / |ln(chi_1)|.

        This is the number of layers over which signals decay/grow by factor e.
        """
        if abs(chi_1_val) < 1e-30 or chi_1_val < 0:
            return 0.0
        log_chi = np.log(chi_1_val)
        if abs(log_chi) < 1e-15:
            return np.inf
        return 1.0 / abs(log_chi)

    def gradient_norm_propagation(self, depth, chi_1_val):
        """
        Compute E[||grad||^2] ~ chi_1^L at each layer.

        Returns array of gradient norm scaling for layers 0..depth.
        """
        layers = np.arange(depth + 1)
        grad_norms = chi_1_val ** layers
        return grad_norms

    def mean_field_phase_diagram(self, sigma_w_range, sigma_b_range):
        """
        Compute phase diagram in (sigma_w, sigma_b) plane.

        Returns 2D array of chi_1 values and the phase boundary.
        """
        sw_arr = np.asarray(sigma_w_range, dtype=np.float64)
        sb_arr = np.asarray(sigma_b_range, dtype=np.float64)
        chi_1_grid = np.zeros((len(sb_arr), len(sw_arr)))

        for i, sb2 in enumerate(sb_arr):
            for j, sw2 in enumerate(sw_arr):
                prop = KernelDepthPropagation(
                    activation=self.activation, weight_variance=sw2, bias_variance=sb2
                )
                q_star = prop.kernel_fixed_point()
                chi_1_grid[i, j] = self.chi_1(sw2, q_star)

        # Extract phase boundary (chi_1 = 1 contour)
        boundary_sw = []
        for i in range(len(sb_arr)):
            row = chi_1_grid[i, :]
            sign_changes = np.where(np.diff(np.sign(row - 1.0)))[0]
            for idx in sign_changes:
                # Linear interpolation
                x0, x1 = sw_arr[idx], sw_arr[idx + 1]
                y0, y1 = row[idx] - 1.0, row[idx + 1] - 1.0
                if abs(y1 - y0) > 1e-15:
                    sw_crit = x0 - y0 * (x1 - x0) / (y1 - y0)
                    boundary_sw.append((sb_arr[i], sw_crit))

        return {
            'chi_1_grid': chi_1_grid,
            'sigma_w_range': sw_arr,
            'sigma_b_range': sb_arr,
            'phase_boundary': np.array(boundary_sw) if boundary_sw else np.array([]),
        }

    def correlation_length(self, chi_1_val, depth):
        """
        Compute correlation length in deep networks.

        The correlation function decays as exp(-l/xi) where xi = depth_scale.
        Returns the correlation function at each layer.
        """
        xi = self.depth_scale(chi_1_val)
        layers = np.arange(depth + 1)
        if np.isinf(xi):
            return np.ones(depth + 1)
        if xi < 1e-15:
            corr = np.zeros(depth + 1)
            corr[0] = 1.0
            return corr
        return np.exp(-layers / xi)

    def information_propagation(self, mutual_info_input, depth, chi_1_val):
        """
        Model mutual information propagation through depth.

        MI decays exponentially in ordered phase, bounded in chaotic phase.
        Returns MI at each layer.
        """
        mi = np.zeros(depth + 1)
        mi[0] = mutual_info_input
        xi = self.depth_scale(chi_1_val)

        for l in range(1, depth + 1):
            if chi_1_val < 1.0:
                # Ordered: MI decays exponentially
                mi[l] = mutual_info_input * np.exp(-l / xi) if xi > 1e-15 else 0.0
            elif chi_1_val > 1.0:
                # Chaotic: MI saturates at a floor (sensitivity to noise)
                mi_floor = mutual_info_input * 0.01
                mi[l] = max(
                    mi_floor,
                    mutual_info_input * np.exp(-l / (xi * 0.5))
                )
            else:
                # Critical: power law decay
                mi[l] = mutual_info_input / (1.0 + l)
        return mi

    def trainability_analysis(self, depths, chi_1_val):
        """
        Analyze trainability as a function of depth.

        Returns dict with gradient norms, convergence rates, and trainability scores.
        """
        depths = np.asarray(depths, dtype=np.float64)
        grad_norms = chi_1_val ** depths

        # Trainability score: high when gradients are O(1)
        trainability = np.exp(-np.abs(np.log(grad_norms + 1e-30)))

        # Convergence rate estimate: faster when chi_1 ~ 1
        conv_rates = np.where(
            grad_norms > 1e-10,
            np.minimum(1.0, 1.0 / grad_norms),
            0.0
        )

        return {
            'depths': depths,
            'gradient_norms': grad_norms,
            'trainability_scores': trainability,
            'convergence_rates': conv_rates,
            'effective_depth': self.depth_scale(chi_1_val),
            'chi_1': chi_1_val,
        }


class DepthPhaseBoundary:
    """Depth-dependent phase boundaries in neural network parameter space."""

    def __init__(self):
        pass

    def critical_depth(self, chi_1, threshold=0.01):
        """
        Compute depth at which signals lose information below threshold.

        L_crit = ln(threshold) / ln(chi_1) for chi_1 < 1.
        """
        if chi_1 >= 1.0:
            return np.inf
        if chi_1 <= 0.0:
            return 0.0
        log_chi = np.log(chi_1)
        if abs(log_chi) < 1e-15:
            return np.inf
        return np.log(threshold) / log_chi

    def phase_boundary_vs_depth(self, param_range, depths):
        """
        How phase boundary shifts with depth.

        For each depth, find parameter value where signal drops to 1/e.
        Returns boundary values for each depth.
        """
        param_range = np.asarray(param_range, dtype=np.float64)
        depths = np.asarray(depths, dtype=np.float64)
        boundaries = np.zeros(len(depths))

        for i, L in enumerate(depths):
            # For each depth, find chi_1 such that chi_1^L = 1/e
            # chi_1 = exp(-1/L)
            target_chi = np.exp(-1.0 / L) if L > 0 else 0.0

            # Map chi_1 back to parameter space via interpolation
            analyzer = SignalPropagationAnalyzer(activation='relu')
            chi_vals = []
            for p in param_range:
                prop = KernelDepthPropagation(
                    activation='relu', weight_variance=p, bias_variance=0.0
                )
                q_star = prop.kernel_fixed_point()
                chi_vals.append(analyzer.chi_1(p, q_star))
            chi_vals = np.array(chi_vals)

            # Interpolate to find parameter at target chi_1
            sign_changes = np.where(np.diff(np.sign(chi_vals - target_chi)))[0]
            if len(sign_changes) > 0:
                idx = sign_changes[0]
                x0, x1 = param_range[idx], param_range[idx + 1]
                y0, y1 = chi_vals[idx] - target_chi, chi_vals[idx + 1] - target_chi
                if abs(y1 - y0) > 1e-15:
                    boundaries[i] = x0 - y0 * (x1 - x0) / (y1 - y0)
                else:
                    boundaries[i] = x0
            else:
                boundaries[i] = np.nan

        return boundaries

    def depth_width_phase_diagram(self, depth_range, width_range, observable_fn):
        """
        Compute 2D phase diagram in (depth, width) plane.

        observable_fn(depth, width) returns a scalar order parameter.
        Returns 2D grid of order parameter values.
        """
        depth_range = np.asarray(depth_range)
        width_range = np.asarray(width_range)
        diagram = np.zeros((len(depth_range), len(width_range)))

        for i, L in enumerate(depth_range):
            for j, n in enumerate(width_range):
                diagram[i, j] = observable_fn(int(L), int(n))

        return {
            'diagram': diagram,
            'depth_range': depth_range,
            'width_range': width_range,
        }

    def depth_scaling_of_order_parameter(self, depths, order_param_fn):
        """
        Compute order parameter as a function of depth.

        Returns array of order parameter values and fitted scaling exponent.
        """
        depths = np.asarray(depths, dtype=np.float64)
        values = np.array([order_param_fn(int(L)) for L in depths])

        # Fit power law: value ~ depth^alpha
        valid = (depths > 0) & (values > 0) & np.isfinite(values)
        if np.sum(valid) >= 2:
            log_d = np.log(depths[valid])
            log_v = np.log(values[valid])
            coeffs = np.polyfit(log_d, log_v, 1)
            alpha = coeffs[0]
            prefactor = np.exp(coeffs[1])
        else:
            alpha = np.nan
            prefactor = np.nan

        return {
            'depths': depths,
            'values': values,
            'scaling_exponent': alpha,
            'prefactor': prefactor,
        }

    def maximal_useful_depth(self, chi_1, signal_threshold=0.01):
        """
        Compute maximum depth before signal degrades below threshold.

        For ordered phase: L_max = ln(threshold) / ln(chi_1).
        For critical: L_max = infinity.
        For chaotic: limited by gradient explosion.
        """
        if abs(chi_1 - 1.0) < 1e-12:
            return np.inf

        if chi_1 < 1.0:
            return self.critical_depth(chi_1, signal_threshold)
        else:
            # Chaotic phase: gradients explode; max depth where grad norm < 1/threshold
            log_chi = np.log(chi_1)
            return np.log(1.0 / signal_threshold) / log_chi

    def residual_connection_effect(self, depth, skip_weight):
        """
        Compute how skip connections modify the critical depth.

        With residual connections: x^{l+1} = alpha * F(x^l) + (1-alpha) * x^l.
        The effective chi_1 becomes alpha * chi_1 + (1 - alpha).
        Returns effective depth scale and modified chi_1.
        """
        alpha = skip_weight
        # For ReLU with sigma_w^2 = 2: chi_1 = 1 (critical)
        # With residual: effective chi_1 = alpha * chi_1_base + (1 - alpha)
        # This keeps chi_1_eff closer to 1 for any alpha in (0, 1)
        chi_1_base = 1.0  # Assume critical initialization

        chi_1_eff = alpha * chi_1_base + (1 - alpha)

        # Modified propagation: signal at depth L
        signal_retention = chi_1_eff ** depth

        # Without residual, the Jacobian eigenvalues spread; with residual they contract
        jacobian_spectral_radius = abs(alpha * chi_1_base + (1 - alpha))

        log_sr = np.log(jacobian_spectral_radius) if jacobian_spectral_radius > 0 else -np.inf
        effective_depth_scale = 1.0 / abs(log_sr) if abs(log_sr) > 1e-15 else np.inf

        return {
            'effective_chi_1': chi_1_eff,
            'signal_retention': signal_retention,
            'effective_depth_scale': effective_depth_scale,
            'jacobian_spectral_radius': jacobian_spectral_radius,
            'depth': depth,
            'skip_weight': skip_weight,
        }

    def normalization_effect(self, depth, norm_type='layer'):
        """
        Compute how normalization layers modify the depth limit.

        Layer norm / batch norm enforce q^l = 1, preventing collapse.
        Returns effective signal propagation properties.
        """
        if norm_type == 'layer':
            # Layer norm pins q^l = 1 at each layer
            # Correlation still evolves but diagonal is fixed
            # Effective chi_1 for correlation becomes chi_1_perp (transverse)
            q_fixed = 1.0
            # chi_1 for ReLU at q*=1 with sigma_w^2=2: chi_1 = 1
            chi_1_eff = 1.0
            # With layer norm, signals can propagate much deeper
            effective_max_depth = np.inf
            signal_retention = np.ones(depth + 1)
        elif norm_type == 'batch':
            # Batch norm: similar to layer norm but across batch dimension
            q_fixed = 1.0
            chi_1_eff = 1.0
            effective_max_depth = np.inf
            signal_retention = np.ones(depth + 1)
        elif norm_type == 'none':
            # Without normalization, use standard propagation
            analyzer = SignalPropagationAnalyzer(activation='relu')
            chi_1_eff = analyzer.chi_1(2.0, 1.0)
            xi = analyzer.depth_scale(chi_1_eff)
            layers = np.arange(depth + 1)
            if np.isinf(xi):
                signal_retention = np.ones(depth + 1)
            else:
                signal_retention = np.exp(-layers / xi)
            effective_max_depth = xi * 10
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        return {
            'norm_type': norm_type,
            'effective_chi_1': chi_1_eff,
            'effective_max_depth': effective_max_depth,
            'signal_retention': signal_retention,
            'depth': depth,
        }


class OptimalDepthPredictor:
    """Predict optimal network depth for given width and dataset size."""

    def __init__(self, width, dataset_size):
        self.width = width
        self.dataset_size = dataset_size

    def compute_optimal_depth(self, width, dataset_size, task_complexity):
        """
        Predict optimal depth L* based on width, data, and task complexity.

        Uses bias-variance tradeoff: deeper = lower bias, higher variance.
        L* ~ (n * N / C)^{1/3} where n=width, N=data, C=complexity.
        """
        n = width
        N = dataset_size
        C = task_complexity

        # Bias decreases as ~1/L, variance increases as ~L/N
        # Optimal at L* = (n * N / C)^{1/3}
        L_star = (n * N / max(C, 1e-10)) ** (1.0 / 3.0)

        # Compute bias and variance at L*
        bias_at_opt = task_complexity / max(L_star, 1e-10)
        variance_at_opt = L_star / max(N, 1)

        return {
            'optimal_depth': L_star,
            'bias': bias_at_opt,
            'variance': variance_at_opt,
            'total_risk': bias_at_opt + variance_at_opt,
            'width': n,
            'dataset_size': N,
            'task_complexity': C,
        }

    def depth_vs_width_tradeoff(self, total_params, depth_range):
        """
        For fixed parameter budget, find optimal depth.

        total_params ~ n^2 * L, so n = sqrt(total_params / L).
        """
        depth_range = np.asarray(depth_range, dtype=np.float64)
        widths = np.sqrt(total_params / np.maximum(depth_range, 1))
        widths = np.maximum(widths, 1)

        # Effective capacity: deeper networks can represent more complex functions
        # but are harder to train. Model: performance ~ min(capacity, trainability)
        capacity = depth_range * np.log(widths + 1)
        trainability = np.exp(-depth_range / (10 * widths))
        effective_performance = capacity * trainability

        optimal_idx = np.argmax(effective_performance)

        return {
            'depth_range': depth_range,
            'widths': widths,
            'capacity': capacity,
            'trainability': trainability,
            'effective_performance': effective_performance,
            'optimal_depth': depth_range[optimal_idx],
            'optimal_width': widths[optimal_idx],
            'total_params': total_params,
        }

    def excess_risk_vs_depth(self, depths, width, dataset_size):
        """
        Compute test error (excess risk) as a function of depth.

        Model: R(L) = bias(L) + variance(L) = C/L + L/(n*N).
        """
        depths = np.asarray(depths, dtype=np.float64)
        n = width
        N = dataset_size

        # Bias: approximation error decreases with depth
        bias = 1.0 / np.maximum(depths, 1e-10)

        # Variance: estimation error increases with depth (more params)
        variance = depths / max(n * N, 1)

        # Total risk
        risk = bias + variance

        # Optimal depth
        L_opt = np.sqrt(max(n * N, 1))
        min_risk = 2.0 / np.sqrt(max(n * N, 1))

        return {
            'depths': depths,
            'bias': bias,
            'variance': variance,
            'excess_risk': risk,
            'optimal_depth': L_opt,
            'minimum_risk': min_risk,
        }

    def underfitting_depth(self, depths, train_errors):
        """
        Find depth below which underfitting occurs.

        Underfitting: training error is still significantly decreasing.
        Uses second derivative to detect knee point.
        """
        depths = np.asarray(depths, dtype=np.float64)
        train_errors = np.asarray(train_errors, dtype=np.float64)

        if len(depths) < 3:
            return float(depths[0])

        # Smooth the training error curve
        if len(depths) >= 4:
            spl = interpolate.UnivariateSpline(depths, train_errors, s=len(depths) * 0.01)
            smooth_errors = spl(depths)
            # First derivative
            d1 = np.gradient(smooth_errors, depths)
        else:
            d1 = np.gradient(train_errors, depths)

        # Underfitting boundary: where |d(train_error)/dL| drops below threshold
        threshold = 0.1 * np.abs(d1[0]) if abs(d1[0]) > 1e-10 else 0.01
        underfitting_mask = np.abs(d1) > threshold

        # Last depth where still underfitting
        indices = np.where(underfitting_mask)[0]
        if len(indices) == 0:
            return float(depths[0])
        return float(depths[indices[-1]])

    def overfitting_depth(self, depths, train_errors, test_errors):
        """
        Find depth above which overfitting occurs.

        Overfitting: test error starts increasing while train error decreases.
        """
        depths = np.asarray(depths, dtype=np.float64)
        train_errors = np.asarray(train_errors, dtype=np.float64)
        test_errors = np.asarray(test_errors, dtype=np.float64)

        gap = test_errors - train_errors

        # Find where generalization gap starts increasing significantly
        d_gap = np.gradient(gap, depths)

        # Overfitting starts where d(gap)/dL > 0 consistently
        overfit_mask = d_gap > 0.01 * np.max(np.abs(d_gap))
        indices = np.where(overfit_mask)[0]

        if len(indices) == 0:
            return float(depths[-1])

        # First depth where overfitting begins
        return float(depths[indices[0]])

    def depth_doubling_experiment(self, depth_range, perf_fn):
        """
        Compare performance at depth L and 2L.

        Returns performance ratio perf(2L)/perf(L) indicating returns from depth.
        """
        depth_range = np.asarray(depth_range, dtype=np.float64)
        perfs = np.array([perf_fn(int(L)) for L in depth_range])
        doubled_perfs = np.array([perf_fn(int(2 * L)) for L in depth_range])

        ratios = np.where(
            np.abs(perfs) > 1e-15,
            doubled_perfs / perfs,
            np.nan
        )

        # Diminishing returns when ratio approaches 1
        diminishing_mask = ratios < 1.05
        dim_indices = np.where(diminishing_mask)[0]
        diminishing_depth = float(depth_range[dim_indices[0]]) if len(dim_indices) > 0 else np.inf

        return {
            'depth_range': depth_range,
            'performance': perfs,
            'doubled_performance': doubled_perfs,
            'doubling_ratio': ratios,
            'diminishing_returns_depth': diminishing_depth,
        }

    def scaling_law_depth(self, depths, performances):
        """
        Fit performance ~ depth^alpha scaling law.

        Returns fitted exponent alpha and goodness of fit.
        """
        depths = np.asarray(depths, dtype=np.float64)
        performances = np.asarray(performances, dtype=np.float64)

        valid = (depths > 0) & (performances > 0) & np.isfinite(performances)
        if np.sum(valid) < 2:
            return {
                'exponent': np.nan,
                'prefactor': np.nan,
                'r_squared': np.nan,
                'depths': depths,
                'performances': performances,
            }

        log_d = np.log(depths[valid])
        log_p = np.log(performances[valid])

        coeffs = np.polyfit(log_d, log_p, 1)
        alpha = coeffs[0]
        prefactor = np.exp(coeffs[1])

        # R^2
        predicted = np.polyval(coeffs, log_d)
        ss_res = np.sum((log_p - predicted) ** 2)
        ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-15)

        return {
            'exponent': alpha,
            'prefactor': prefactor,
            'r_squared': r_squared,
            'depths': depths,
            'performances': performances,
            'fitted_curve': prefactor * depths ** alpha,
        }


class DepthWidthInteraction:
    """Analyze interactions between depth and width in neural networks."""

    def __init__(self):
        pass

    def compute_depth_width_grid(self, depth_range, width_range, metric_fn):
        """
        Compute a grid of metric values over (depth, width) space.

        metric_fn(depth, width) -> scalar.
        """
        depth_range = np.asarray(depth_range)
        width_range = np.asarray(width_range)
        grid = np.zeros((len(depth_range), len(width_range)))

        for i, L in enumerate(depth_range):
            for j, n in enumerate(width_range):
                grid[i, j] = metric_fn(int(L), int(n))

        return {
            'grid': grid,
            'depth_range': depth_range,
            'width_range': width_range,
        }

    def iso_performance_curves(self, depth_range, width_range, performance_fn, levels):
        """
        Find curves of constant performance in (depth, width) space.

        Returns contour data: for each level, the (depth, width) pairs.
        """
        depth_range = np.asarray(depth_range, dtype=np.float64)
        width_range = np.asarray(width_range, dtype=np.float64)
        levels = np.asarray(levels, dtype=np.float64)

        # Compute performance grid
        grid = np.zeros((len(depth_range), len(width_range)))
        for i, L in enumerate(depth_range):
            for j, n in enumerate(width_range):
                grid[i, j] = performance_fn(int(L), int(n))

        # Extract iso-curves via linear interpolation
        contours = {}
        for level in levels:
            curve_points = []
            # Search rows for crossings
            for i in range(len(depth_range)):
                row = grid[i, :]
                crossings = np.where(np.diff(np.sign(row - level)))[0]
                for idx in crossings:
                    # Interpolate width at crossing
                    w0, w1 = width_range[idx], width_range[idx + 1]
                    v0, v1 = row[idx] - level, row[idx + 1] - level
                    if abs(v1 - v0) > 1e-15:
                        w_cross = w0 - v0 * (w1 - w0) / (v1 - v0)
                        curve_points.append((depth_range[i], w_cross))

            # Search columns for crossings
            for j in range(len(width_range)):
                col = grid[:, j]
                crossings = np.where(np.diff(np.sign(col - level)))[0]
                for idx in crossings:
                    d0, d1 = depth_range[idx], depth_range[idx + 1]
                    v0, v1 = col[idx] - level, col[idx + 1] - level
                    if abs(v1 - v0) > 1e-15:
                        d_cross = d0 - v0 * (d1 - d0) / (v1 - v0)
                        curve_points.append((d_cross, width_range[j]))

            contours[float(level)] = np.array(curve_points) if curve_points else np.array([])

        return {
            'contours': contours,
            'grid': grid,
            'depth_range': depth_range,
            'width_range': width_range,
        }

    def optimal_aspect_ratio(self, total_params_range, performance_fn):
        """
        Find optimal L/n ratio as a function of total parameter count.

        For each param budget P, sweep L and set n = sqrt(P/L).
        """
        total_params_range = np.asarray(total_params_range, dtype=np.float64)
        optimal_ratios = np.zeros(len(total_params_range))
        optimal_depths = np.zeros(len(total_params_range))
        optimal_widths = np.zeros(len(total_params_range))
        optimal_perfs = np.zeros(len(total_params_range))

        for k, P in enumerate(total_params_range):
            best_perf = -np.inf
            best_L = 1
            best_n = int(np.sqrt(P))

            # Sweep depth from 1 to sqrt(P)
            max_L = max(2, int(np.sqrt(P)))
            for L in range(1, max_L + 1):
                n = int(np.sqrt(P / L))
                if n < 1:
                    continue
                perf = performance_fn(L, n)
                if perf > best_perf:
                    best_perf = perf
                    best_L = L
                    best_n = n

            optimal_depths[k] = best_L
            optimal_widths[k] = best_n
            optimal_ratios[k] = best_L / max(best_n, 1)
            optimal_perfs[k] = best_perf

        return {
            'total_params': total_params_range,
            'optimal_depths': optimal_depths,
            'optimal_widths': optimal_widths,
            'optimal_ratios': optimal_ratios,
            'optimal_performance': optimal_perfs,
        }

    def phase_diagram_depth_width(self, depth_range, width_range, order_param_fn):
        """
        Compute phase diagram in (depth, width) plane using an order parameter.

        Identifies phase boundaries as level sets of the order parameter.
        """
        depth_range = np.asarray(depth_range)
        width_range = np.asarray(width_range)
        grid = np.zeros((len(depth_range), len(width_range)))

        for i, L in enumerate(depth_range):
            for j, n in enumerate(width_range):
                grid[i, j] = order_param_fn(int(L), int(n))

        # Classify phases based on order parameter value
        # Phase 1: order_param < 0.33 (ordered/convergent)
        # Phase 2: 0.33 <= order_param < 0.67 (critical)
        # Phase 3: order_param >= 0.67 (chaotic/divergent)
        phase_map = np.zeros_like(grid, dtype=int)
        phase_map[grid < 0.33] = 0
        phase_map[(grid >= 0.33) & (grid < 0.67)] = 1
        phase_map[grid >= 0.67] = 2

        # Find boundaries between phases
        boundaries = []
        for i in range(len(depth_range) - 1):
            for j in range(len(width_range) - 1):
                if phase_map[i, j] != phase_map[i + 1, j]:
                    boundaries.append((
                        0.5 * (depth_range[i] + depth_range[i + 1]),
                        width_range[j],
                        phase_map[i, j],
                        phase_map[i + 1, j]
                    ))
                if phase_map[i, j] != phase_map[i, j + 1]:
                    boundaries.append((
                        depth_range[i],
                        0.5 * (width_range[j] + width_range[j + 1]),
                        phase_map[i, j],
                        phase_map[i, j + 1]
                    ))

        return {
            'order_parameter_grid': grid,
            'phase_map': phase_map,
            'boundaries': boundaries,
            'depth_range': depth_range,
            'width_range': width_range,
        }

    def interaction_strength(self, depths, widths, metric_values):
        """
        Quantify depth-width interaction strength.

        Tests whether metric = f(L) + g(n) (additive) by measuring residuals
        from additive decomposition.
        """
        depths = np.asarray(depths, dtype=np.float64)
        widths = np.asarray(widths, dtype=np.float64)
        metric_values = np.asarray(metric_values, dtype=np.float64)

        n_d, n_w = metric_values.shape
        assert n_d == len(depths) and n_w == len(widths)

        # Additive decomposition: M(i,j) ≈ mu + alpha_i + beta_j
        grand_mean = np.mean(metric_values)
        row_means = np.mean(metric_values, axis=1)
        col_means = np.mean(metric_values, axis=0)

        # Additive model
        additive = (row_means[:, None] + col_means[None, :] - grand_mean)

        # Interaction = residual from additive model
        interaction = metric_values - additive

        # Strength metrics
        ss_total = np.sum((metric_values - grand_mean) ** 2)
        ss_interaction = np.sum(interaction ** 2)
        ss_additive = np.sum((additive - grand_mean) ** 2)

        interaction_fraction = ss_interaction / max(ss_total, 1e-15)

        return {
            'interaction_residuals': interaction,
            'additive_model': additive,
            'ss_total': ss_total,
            'ss_interaction': ss_interaction,
            'ss_additive': ss_additive,
            'interaction_fraction': interaction_fraction,
            'interaction_is_weak': interaction_fraction < 0.05,
        }

    def separability_test(self, depths, widths, metric_values):
        """
        Test if metric can be written as f(L) * g(n) (multiplicative separability).

        Uses SVD: if rank-1 approximation captures most variance, it's separable.
        """
        depths = np.asarray(depths, dtype=np.float64)
        widths = np.asarray(widths, dtype=np.float64)
        M = np.asarray(metric_values, dtype=np.float64)

        U, S, Vt = np.linalg.svd(M, full_matrices=False)

        # Fraction of variance explained by rank-1
        total_variance = np.sum(S ** 2)
        rank1_variance = S[0] ** 2 if len(S) > 0 else 0
        rank1_fraction = rank1_variance / max(total_variance, 1e-15)

        # Rank-1 approximation: M ≈ s_0 * u_0 * v_0^T = f(L) * g(n)
        if len(S) > 0:
            f_depth = U[:, 0] * np.sqrt(S[0])
            g_width = Vt[0, :] * np.sqrt(S[0])
            rank1_approx = np.outer(f_depth, g_width)
        else:
            f_depth = np.zeros(len(depths))
            g_width = np.zeros(len(widths))
            rank1_approx = np.zeros_like(M)

        residual = M - rank1_approx
        residual_norm = np.linalg.norm(residual) / max(np.linalg.norm(M), 1e-15)

        return {
            'is_separable': rank1_fraction > 0.95,
            'rank1_fraction': rank1_fraction,
            'singular_values': S,
            'f_depth': f_depth,
            'g_width': g_width,
            'rank1_approximation': rank1_approx,
            'residual_norm': residual_norm,
        }

    def scaling_exponent_dependence(self, widths, depths, metric_fn):
        """
        How scaling exponents depend on the other variable.

        For each width, fit metric ~ depth^alpha and see how alpha depends on width.
        For each depth, fit metric ~ width^beta and see how beta depends on depth.
        """
        widths = np.asarray(widths, dtype=np.float64)
        depths = np.asarray(depths, dtype=np.float64)

        # Compute metric grid
        grid = np.zeros((len(depths), len(widths)))
        for i, L in enumerate(depths):
            for j, n in enumerate(widths):
                grid[i, j] = metric_fn(int(L), int(n))

        # For each width, fit depth exponent
        alpha_vs_width = np.zeros(len(widths))
        for j, n in enumerate(widths):
            col = grid[:, j]
            valid = (depths > 0) & (col > 0) & np.isfinite(col)
            if np.sum(valid) >= 2:
                coeffs = np.polyfit(np.log(depths[valid]), np.log(col[valid]), 1)
                alpha_vs_width[j] = coeffs[0]
            else:
                alpha_vs_width[j] = np.nan

        # For each depth, fit width exponent
        beta_vs_depth = np.zeros(len(depths))
        for i, L in enumerate(depths):
            row = grid[i, :]
            valid = (widths > 0) & (row > 0) & np.isfinite(row)
            if np.sum(valid) >= 2:
                coeffs = np.polyfit(np.log(widths[valid]), np.log(row[valid]), 1)
                beta_vs_depth[i] = coeffs[0]
            else:
                beta_vs_depth[i] = np.nan

        # Check if exponents are constant (universal scaling)
        alpha_std = np.nanstd(alpha_vs_width)
        beta_std = np.nanstd(beta_vs_depth)

        return {
            'alpha_vs_width': alpha_vs_width,
            'beta_vs_depth': beta_vs_depth,
            'widths': widths,
            'depths': depths,
            'alpha_mean': np.nanmean(alpha_vs_width),
            'alpha_std': alpha_std,
            'beta_mean': np.nanmean(beta_vs_depth),
            'beta_std': beta_std,
            'universal_depth_scaling': alpha_std < 0.1,
            'universal_width_scaling': beta_std < 0.1,
            'metric_grid': grid,
        }
