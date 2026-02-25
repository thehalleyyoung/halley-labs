"""Width scaling analysis for neural networks.

Analyzes how neural network properties (NTK, observables, critical exponents)
scale with network width, including finite-width corrections and scaling collapse.
"""

import numpy as np
from scipy import optimize, stats, special
from scipy.interpolate import UnivariateSpline


class NTKWidthScaling:
    """Analyzes how the Neural Tangent Kernel scales with network width."""

    def __init__(self, base_width=64):
        self.base_width = base_width

    def ntk_at_width(self, width, input_data, depth=2, activation='relu'):
        """Compute NTK at given width via random features approximation.

        Uses the recursive NTK formula with random weight matrices.
        """
        n_samples = input_data.shape[0]
        d_in = input_data.shape[1]
        act_fn = self._get_activation(activation)
        act_deriv = self._get_activation_deriv(activation)

        np.random.seed(None)
        W = [np.random.randn(d_in, width) / np.sqrt(d_in)]
        for l in range(1, depth):
            W.append(np.random.randn(width, width) / np.sqrt(width))

        # Forward pass storing pre- and post-activations
        h = input_data  # (n, d_in)
        pre_acts = []
        post_acts = [h]
        for l in range(depth):
            z = h @ W[l]
            pre_acts.append(z)
            h = act_fn(z)
            post_acts.append(h)

        # NTK = sum over layers of (dout/dW_l)(dout/dW_l)^T contributions
        # For the last layer output we use identity (single output neuron weights)
        v = np.random.randn(width, 1) / np.sqrt(width)
        kernel = np.zeros((n_samples, n_samples))

        for l in range(depth):
            # Jacobian w.r.t. W[l] via backprop
            if l == depth - 1:
                delta = v.T * act_deriv(pre_acts[l])  # (n, width)
            else:
                delta = np.ones_like(pre_acts[-1])
                for m in range(depth - 1, l, -1):
                    delta = (delta * act_deriv(pre_acts[m])) @ W[m].T
                delta = delta * act_deriv(pre_acts[l])

            # Contribution: J_l @ J_l^T where J_l = delta ⊗ post_acts[l]
            # Trace over weight indices gives delta @ delta^T * post_acts @ post_acts^T
            kernel += (delta @ delta.T) * (post_acts[l] @ post_acts[l].T)

        return kernel / width

    def ntk_trace_vs_width(self, widths, input_data, depth=2):
        """Compute tr(Θ) as a function of width n."""
        traces = []
        for w in widths:
            K = self.ntk_at_width(w, input_data, depth)
            traces.append(np.trace(K))
        return np.array(widths), np.array(traces)

    def ntk_frobenius_vs_width(self, widths, input_data, depth=2):
        """Compute ||Θ||_F as a function of width n."""
        frob_norms = []
        for w in widths:
            K = self.ntk_at_width(w, input_data, depth)
            frob_norms.append(np.sqrt(np.sum(K ** 2)))
        return np.array(widths), np.array(frob_norms)

    def eigenvalue_scaling(self, widths, input_data, k=10):
        """Track top-k eigenvalues of the NTK as a function of width."""
        all_eigs = {}
        for w in widths:
            K = self.ntk_at_width(w, input_data)
            eigs = np.sort(np.linalg.eigvalsh(K))[::-1]
            all_eigs[w] = eigs[:min(k, len(eigs))]
        return all_eigs

    def kernel_convergence_rate(self, widths, input_data):
        """Measure rate of convergence to the infinite-width kernel.

        Estimates convergence by comparing successive width kernels and
        fitting ||K_n - K_{n'}||_F ~ C / n^alpha.
        """
        widths = sorted(widths)
        K_ref = self.ntk_at_width(widths[-1], input_data)
        diffs = []
        for w in widths[:-1]:
            K_w = self.ntk_at_width(w, input_data)
            diffs.append(np.sqrt(np.sum((K_w - K_ref) ** 2)) / K_ref.size)

        ws = np.array(widths[:-1], dtype=float)
        ds = np.array(diffs)
        mask = ds > 0
        if np.sum(mask) < 2:
            return {'rate': np.nan, 'coefficient': np.nan}

        log_w = np.log(ws[mask])
        log_d = np.log(ds[mask])
        slope, intercept, r, _, stderr = stats.linregress(log_w, log_d)
        return {
            'rate': -slope,
            'coefficient': np.exp(intercept),
            'r_squared': r ** 2,
            'stderr': stderr,
        }

    def finite_width_kernel_correction(self, width, infinite_width_kernel):
        """Compute finite-width correction: Θ_n = Θ_∞ + δΘ/n + O(1/n²).

        Returns estimated correction matrix δΘ/n.
        """
        K_n = self.ntk_at_width(width, np.eye(infinite_width_kernel.shape[0]))
        correction = K_n - infinite_width_kernel
        return {
            'correction_matrix': correction,
            'correction_norm': np.linalg.norm(correction, 'fro'),
            'relative_correction': np.linalg.norm(correction, 'fro') /
                                   max(np.linalg.norm(infinite_width_kernel, 'fro'), 1e-12),
            'estimated_delta': correction * width,
        }

    def width_for_convergence(self, tolerance, input_data):
        """Find minimum width for ε-convergence to infinite-width kernel.

        Binary search for the smallest width where successive kernel
        differences are below tolerance.
        """
        lo, hi = self.base_width, self.base_width * 64
        K_large = self.ntk_at_width(hi, input_data)
        norm_ref = max(np.linalg.norm(K_large, 'fro'), 1e-12)

        while lo < hi - 1:
            mid = (lo + hi) // 2
            K_mid = self.ntk_at_width(mid, input_data)
            rel_diff = np.linalg.norm(K_mid - K_large, 'fro') / norm_ref
            if rel_diff < tolerance:
                hi = mid
            else:
                lo = mid
        return hi

    def monte_carlo_ntk(self, width, input_data, n_samples=100):
        """Monte Carlo estimate of NTK at width n with error bars."""
        kernels = []
        for _ in range(n_samples):
            K = self.ntk_at_width(width, input_data)
            kernels.append(K)
        kernels = np.array(kernels)
        return {
            'mean': np.mean(kernels, axis=0),
            'std': np.std(kernels, axis=0),
            'stderr': np.std(kernels, axis=0) / np.sqrt(n_samples),
            'n_samples': n_samples,
        }

    @staticmethod
    def _get_activation(name):
        activations = {
            'relu': lambda x: np.maximum(0, x),
            'tanh': np.tanh,
            'sigmoid': special.expit,
            'gelu': lambda x: x * 0.5 * (1 + special.erf(x / np.sqrt(2))),
        }
        return activations.get(name, activations['relu'])

    @staticmethod
    def _get_activation_deriv(name):
        derivs = {
            'relu': lambda x: (x > 0).astype(float),
            'tanh': lambda x: 1 - np.tanh(x) ** 2,
            'sigmoid': lambda x: special.expit(x) * (1 - special.expit(x)),
            'gelu': lambda x: 0.5 * (1 + special.erf(x / np.sqrt(2))) +
                               x * np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi),
        }
        return derivs.get(name, derivs['relu'])


class FiniteWidthCorrectionScaling:
    """Finite-width corrections scaling analysis.

    Extracts 1/n corrections to infinite-width observables and performs
    finite-size scaling analysis.
    """

    def __init__(self):
        pass

    def first_order_correction(self, widths, observable_values):
        """Extract O(1/n) correction coefficient.

        Fits observable(n) = a + b/n and returns (a, b).
        """
        widths = np.asarray(widths, dtype=float)
        values = np.asarray(observable_values, dtype=float)
        X = np.column_stack([np.ones_like(widths), 1.0 / widths])
        coeffs, residuals, _, _ = np.linalg.lstsq(X, values, rcond=None)
        y_pred = X @ coeffs
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_sq = 1 - ss_res / max(ss_tot, 1e-12)
        return {
            'infinite_width_value': coeffs[0],
            'first_order_coeff': coeffs[1],
            'r_squared': r_sq,
            'residuals': values - y_pred,
        }

    def second_order_correction(self, widths, observable_values):
        """Extract O(1/n²) correction: observable = a + b/n + c/n²."""
        widths = np.asarray(widths, dtype=float)
        values = np.asarray(observable_values, dtype=float)
        inv_w = 1.0 / widths
        X = np.column_stack([np.ones_like(inv_w), inv_w, inv_w ** 2])
        coeffs, residuals, _, _ = np.linalg.lstsq(X, values, rcond=None)
        y_pred = X @ coeffs
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_sq = 1 - ss_res / max(ss_tot, 1e-12)
        n_params = 3
        n = len(values)
        aic = n * np.log(ss_res / max(n, 1)) + 2 * n_params
        bic = n * np.log(ss_res / max(n, 1)) + n_params * np.log(max(n, 1))
        return {
            'infinite_width_value': coeffs[0],
            'first_order_coeff': coeffs[1],
            'second_order_coeff': coeffs[2],
            'r_squared': r_sq,
            'aic': aic,
            'bic': bic,
        }

    def correction_exponent(self, widths, observable_values):
        """Fit observable ~ a + b/n^α, extracting the correction exponent α."""
        widths = np.asarray(widths, dtype=float)
        values = np.asarray(observable_values, dtype=float)

        def model(n, a, b, alpha):
            return a + b * n ** (-alpha)

        try:
            p0_a = values[-1]
            p0_b = (values[0] - values[-1]) * widths[0]
            popt, pcov = optimize.curve_fit(
                model, widths, values,
                p0=[p0_a, p0_b, 1.0],
                bounds=([-np.inf, -np.inf, 0.01], [np.inf, np.inf, 10.0]),
                maxfev=10000,
            )
            perr = np.sqrt(np.diag(pcov))
            y_pred = model(widths, *popt)
            ss_res = np.sum((values - y_pred) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            return {
                'infinite_width_value': popt[0],
                'coefficient': popt[1],
                'exponent': popt[2],
                'uncertainties': {'a': perr[0], 'b': perr[1], 'alpha': perr[2]},
                'r_squared': 1 - ss_res / max(ss_tot, 1e-12),
            }
        except (RuntimeError, ValueError):
            return {'exponent': np.nan, 'r_squared': 0.0}

    def extrapolate_to_infinite(self, widths, values):
        """Extrapolate observable to n→∞ using Richardson extrapolation.

        Uses successive elimination of 1/n, 1/n² terms.
        """
        widths = np.asarray(widths, dtype=float)
        values = np.asarray(values, dtype=float)
        idx = np.argsort(widths)[::-1]
        widths, values = widths[idx], values[idx]

        estimates = [values.copy()]
        for order in range(1, min(4, len(values))):
            prev = estimates[-1]
            new_est = []
            for i in range(len(prev) - 1):
                r = (widths[i] / widths[i + order]) ** order
                extrapolated = (r * prev[i] - prev[i + 1]) / (r - 1)
                new_est.append(extrapolated)
            estimates.append(np.array(new_est))
            if len(new_est) == 1:
                break

        final_estimates = [est[0] for est in estimates if len(est) > 0]
        return {
            'estimate': final_estimates[-1],
            'all_orders': final_estimates,
            'convergence': np.diff(final_estimates) if len(final_estimates) > 1 else [],
        }

    def correction_confidence_interval(self, widths, values, n_bootstrap=1000):
        """Bootstrap confidence interval for finite-width correction parameters."""
        widths = np.asarray(widths, dtype=float)
        values = np.asarray(values, dtype=float)
        n = len(widths)
        boot_a, boot_b = [], []

        for _ in range(n_bootstrap):
            idx = np.random.randint(0, n, size=n)
            w_boot = widths[idx]
            v_boot = values[idx]
            X = np.column_stack([np.ones(n), 1.0 / w_boot])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, v_boot, rcond=None)
                boot_a.append(coeffs[0])
                boot_b.append(coeffs[1])
            except np.linalg.LinAlgError:
                continue

        boot_a, boot_b = np.array(boot_a), np.array(boot_b)
        return {
            'a_mean': np.mean(boot_a),
            'a_ci_95': (np.percentile(boot_a, 2.5), np.percentile(boot_a, 97.5)),
            'b_mean': np.mean(boot_b),
            'b_ci_95': (np.percentile(boot_b, 2.5), np.percentile(boot_b, 97.5)),
            'a_std': np.std(boot_a),
            'b_std': np.std(boot_b),
        }

    def finite_size_scaling(self, widths, values_at_widths, control_param):
        """Finite-width scaling collapse: f(g, n) = n^{-a} F(n^b (g - gc)).

        widths: array of widths
        values_at_widths: dict {width: array of observable vs control_param}
        control_param: array of control parameter values
        """
        widths = np.asarray(widths, dtype=float)
        control_param = np.asarray(control_param, dtype=float)

        def collapse_cost(params):
            gc, a, b = params
            all_x, all_y = [], []
            for w in widths:
                vals = np.asarray(values_at_widths[w])
                scaled_x = w ** b * (control_param - gc)
                scaled_y = w ** a * vals
                all_x.extend(scaled_x)
                all_y.extend(scaled_y)
            all_x, all_y = np.array(all_x), np.array(all_y)
            order = np.argsort(all_x)
            all_x, all_y = all_x[order], all_y[order]
            if len(all_x) < 3:
                return 1e10
            try:
                spl = UnivariateSpline(all_x, all_y, s=len(all_x))
                return np.mean((all_y - spl(all_x)) ** 2)
            except Exception:
                return 1e10

        gc0 = np.mean(control_param)
        result = optimize.minimize(
            collapse_cost, [gc0, 0.5, 0.5],
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-6},
        )
        gc, a, b = result.x
        collapsed = {}
        for w in widths:
            vals = np.asarray(values_at_widths[w])
            collapsed[w] = {
                'x': w ** b * (control_param - gc),
                'y': w ** a * vals,
            }
        return {
            'critical_point': gc,
            'exponent_a': a,
            'exponent_b': b,
            'collapsed_data': collapsed,
            'cost': result.fun,
        }

    def optimal_width_selection(self, widths, values, target_accuracy):
        """Find minimum width achieving target accuracy relative to extrapolated value."""
        extrap = self.extrapolate_to_infinite(widths, values)
        inf_val = extrap['estimate']
        widths = np.asarray(widths, dtype=float)
        values = np.asarray(values, dtype=float)
        rel_errors = np.abs(values - inf_val) / max(abs(inf_val), 1e-12)

        for w, err in sorted(zip(widths, rel_errors)):
            if err <= target_accuracy:
                return {'min_width': int(w), 'achieved_accuracy': err}
        return {'min_width': None, 'message': 'No width meets target accuracy'}


class CriticalExponentExtractor:
    """Extract critical exponents from width scaling data."""

    def __init__(self, min_points=5):
        self.min_points = min_points

    def extract_exponent(self, x_values, y_values, critical_point=None):
        """Fit y ~ |x - xc|^α near a critical point.

        If critical_point is None, it is estimated from the data.
        """
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)

        if critical_point is None:
            critical_point = self._estimate_critical_point(x, y)

        dist = np.abs(x - critical_point)
        mask = dist > 1e-12
        if np.sum(mask) < self.min_points:
            return {'exponent': np.nan, 'critical_point': critical_point}

        result = self.log_log_fit(dist[mask], np.abs(y[mask]))
        result['critical_point'] = critical_point
        return result

    def log_log_fit(self, x, y):
        """Linear fit in log-log space: log(y) = α·log(x) + log(a)."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        mask = (x > 0) & (y > 0)
        if np.sum(mask) < 2:
            return {'exponent': np.nan, 'amplitude': np.nan, 'r_squared': 0.0}

        lx, ly = np.log(x[mask]), np.log(y[mask])
        slope, intercept, r, p, stderr = stats.linregress(lx, ly)
        return {
            'exponent': slope,
            'amplitude': np.exp(intercept),
            'r_squared': r ** 2,
            'p_value': p,
            'stderr': stderr,
        }

    def weighted_log_log_fit(self, x, y, weights=None):
        """Weighted linear fit in log-log space."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        mask = (x > 0) & (y > 0)
        lx, ly = np.log(x[mask]), np.log(y[mask])
        if weights is None:
            w = np.ones(len(lx))
        else:
            w = np.asarray(weights, dtype=float)[mask]

        W = np.diag(w)
        A = np.column_stack([lx, np.ones(len(lx))])
        AW = A.T @ W
        coeffs = np.linalg.solve(AW @ A, AW @ ly)
        y_pred = A @ coeffs
        ss_res = np.sum(w * (ly - y_pred) ** 2)
        ss_tot = np.sum(w * (ly - np.average(ly, weights=w)) ** 2)
        return {
            'exponent': coeffs[0],
            'amplitude': np.exp(coeffs[1]),
            'r_squared': 1 - ss_res / max(ss_tot, 1e-12),
        }

    def crossover_detection(self, x, y):
        """Detect crossover between scaling regimes.

        Fits piecewise power laws and finds the crossover point.
        """
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        order = np.argsort(x)
        x, y = x[order], y[order]
        mask = (x > 0) & (y > 0)
        x, y = x[mask], y[mask]
        if len(x) < 2 * self.min_points:
            return {'crossover_point': np.nan}

        best_cost = np.inf
        best_idx = len(x) // 2
        lx, ly = np.log(x), np.log(y)

        for i in range(self.min_points, len(x) - self.min_points):
            s1, i1, _, _, _ = stats.linregress(lx[:i], ly[:i])
            s2, i2, _, _, _ = stats.linregress(lx[i:], ly[i:])
            pred = np.concatenate([s1 * lx[:i] + i1, s2 * lx[i:] + i2])
            cost = np.sum((ly - pred) ** 2)
            if cost < best_cost:
                best_cost = cost
                best_idx = i

        fit_lo = self.log_log_fit(x[:best_idx], y[:best_idx])
        fit_hi = self.log_log_fit(x[best_idx:], y[best_idx:])
        # Crossover where the two power laws intersect
        if abs(fit_lo['exponent'] - fit_hi['exponent']) > 1e-10:
            log_xc = (np.log(fit_hi['amplitude']) - np.log(fit_lo['amplitude'])) / \
                      (fit_lo['exponent'] - fit_hi['exponent'])
            xc = np.exp(log_xc)
        else:
            xc = x[best_idx]

        return {
            'crossover_point': xc,
            'exponent_below': fit_lo['exponent'],
            'exponent_above': fit_hi['exponent'],
            'fit_below': fit_lo,
            'fit_above': fit_hi,
        }

    def universal_ratio(self, exponents_system1, exponents_system2):
        """Compare universality: check if exponent ratios match between systems."""
        e1 = np.asarray(exponents_system1, dtype=float)
        e2 = np.asarray(exponents_system2, dtype=float)
        n = min(len(e1), len(e2))
        e1, e2 = e1[:n], e2[:n]
        mask2 = np.abs(e2) > 1e-12
        ratios = np.full(n, np.nan)
        ratios[mask2] = e1[mask2] / e2[mask2]
        ratio_std = np.nanstd(ratios)
        return {
            'ratios': ratios,
            'mean_ratio': np.nanmean(ratios),
            'ratio_std': ratio_std,
            'consistent': ratio_std < 0.1 * abs(np.nanmean(ratios)) if np.isfinite(ratio_std) else False,
        }

    def confidence_interval(self, x, y, critical_point, n_bootstrap=1000):
        """Bootstrap confidence interval for the critical exponent."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        n = len(x)
        exponents = []
        for _ in range(n_bootstrap):
            idx = np.random.randint(0, n, size=n)
            res = self.extract_exponent(x[idx], y[idx], critical_point)
            if np.isfinite(res.get('exponent', np.nan)):
                exponents.append(res['exponent'])

        if len(exponents) < 10:
            return {'exponent_mean': np.nan, 'ci_95': (np.nan, np.nan)}

        exponents = np.array(exponents)
        return {
            'exponent_mean': np.mean(exponents),
            'exponent_std': np.std(exponents),
            'ci_95': (np.percentile(exponents, 2.5), np.percentile(exponents, 97.5)),
            'ci_68': (np.percentile(exponents, 16), np.percentile(exponents, 84)),
            'n_successful': len(exponents),
        }

    def systematic_error(self, x, y, x_ranges):
        """Check exponent stability across different fitting ranges."""
        results = []
        for xmin, xmax in x_ranges:
            mask = (np.asarray(x) >= xmin) & (np.asarray(x) <= xmax)
            if np.sum(mask) >= self.min_points:
                fit = self.log_log_fit(np.asarray(x)[mask], np.asarray(y)[mask])
                fit['x_range'] = (xmin, xmax)
                fit['n_points'] = int(np.sum(mask))
                results.append(fit)

        exps = [r['exponent'] for r in results if np.isfinite(r['exponent'])]
        return {
            'fits': results,
            'exponent_spread': max(exps) - min(exps) if len(exps) > 1 else 0.0,
            'exponent_mean': np.mean(exps) if exps else np.nan,
            'exponent_std': np.std(exps) if exps else np.nan,
        }

    def corrections_to_scaling(self, x, y, critical_point):
        """Fit with leading correction: y ~ A|x-xc|^α (1 + B|x-xc|^ω)."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        dist = np.abs(x - critical_point)
        mask = dist > 1e-12
        dist, yp = dist[mask], np.abs(y[mask])

        def model(d, A, alpha, B, omega):
            return A * d ** alpha * (1 + B * d ** omega)

        # Initial guess from simple power-law
        simple = self.log_log_fit(dist, yp)
        try:
            popt, pcov = optimize.curve_fit(
                model, dist, yp,
                p0=[simple.get('amplitude', 1.0), simple.get('exponent', 1.0), 0.0, 0.5],
                bounds=([0, -10, -np.inf, 0.01], [np.inf, 10, np.inf, 5.0]),
                maxfev=10000,
            )
            perr = np.sqrt(np.diag(pcov))
            return {
                'amplitude': popt[0],
                'exponent': popt[1],
                'correction_amplitude': popt[2],
                'correction_exponent': popt[3],
                'uncertainties': perr,
            }
        except (RuntimeError, ValueError):
            return {'exponent': simple.get('exponent', np.nan)}

    def _estimate_critical_point(self, x, y):
        """Estimate critical point as location of maximum |dy/dx|."""
        order = np.argsort(x)
        x, y = x[order], y[order]
        dy = np.abs(np.diff(y) / np.diff(x))
        idx = np.argmax(dy)
        return 0.5 * (x[idx] + x[idx + 1])


class ScalingCollapseAnalyzer:
    """Perform scaling collapse to extract universal behavior."""

    def __init__(self):
        pass

    def scaling_collapse(self, data_sets, sizes, control_params, nu, eta):
        """Collapse data onto a universal curve.

        data_sets: list of arrays, one per system size
        sizes: corresponding widths/sizes
        control_params: control parameter values (same for all sizes)
        nu, eta: scaling exponents
        """
        collapsed = []
        for data, L in zip(data_sets, sizes):
            data = np.asarray(data, dtype=float)
            cp = np.asarray(control_params, dtype=float)
            x_scaled = L ** (1.0 / nu) * cp
            y_scaled = L ** (eta / nu) * data
            collapsed.append({'x': x_scaled, 'y': y_scaled, 'size': L})
        return collapsed

    def collapse_quality(self, collapsed_data):
        """Measure quality of scaling collapse via chi-squared statistic.

        Interpolates each dataset and measures scatter around the mean curve.
        """
        all_x = np.concatenate([c['x'] for c in collapsed_data])
        all_y = np.concatenate([c['y'] for c in collapsed_data])
        order = np.argsort(all_x)
        all_x, all_y = all_x[order], all_y[order]

        if len(all_x) < 4:
            return {'chi_squared': np.inf, 'quality': 0.0}

        try:
            spl = UnivariateSpline(all_x, all_y, s=len(all_x) * 0.5)
        except Exception:
            return {'chi_squared': np.inf, 'quality': 0.0}

        chi2 = 0.0
        n_pts = 0
        for c in collapsed_data:
            pred = spl(c['x'])
            chi2 += np.sum((c['y'] - pred) ** 2)
            n_pts += len(c['y'])

        var = np.var(all_y) if np.var(all_y) > 0 else 1.0
        chi2_norm = chi2 / (n_pts * var)
        return {
            'chi_squared': chi2,
            'chi_squared_normalized': chi2_norm,
            'quality': np.exp(-chi2_norm),
            'n_points': n_pts,
        }

    def optimize_exponents(self, data_sets, sizes, control_params, nu_range, eta_range):
        """Optimize ν, η for best scaling collapse."""
        def cost(params):
            nu, eta = params
            if nu < 1e-6:
                return 1e10
            collapsed = self.scaling_collapse(data_sets, sizes, control_params, nu, eta)
            q = self.collapse_quality(collapsed)
            return q['chi_squared_normalized']

        nu_grid = np.linspace(nu_range[0], nu_range[1], 20)
        eta_grid = np.linspace(eta_range[0], eta_range[1], 20)
        best_cost, best_params = np.inf, (1.0, 0.0)
        for nu in nu_grid:
            for eta in eta_grid:
                c = cost([nu, eta])
                if c < best_cost:
                    best_cost = c
                    best_params = (nu, eta)

        result = optimize.minimize(cost, best_params, method='Nelder-Mead',
                                   options={'maxiter': 2000, 'xatol': 1e-5})
        nu_opt, eta_opt = result.x
        collapsed = self.scaling_collapse(data_sets, sizes, control_params, nu_opt, eta_opt)
        quality = self.collapse_quality(collapsed)
        return {
            'nu': nu_opt,
            'eta': eta_opt,
            'cost': result.fun,
            'quality': quality,
            'collapsed_data': collapsed,
        }

    def universal_function(self, collapsed_data, method='polynomial'):
        """Fit the universal scaling function from collapsed data."""
        all_x = np.concatenate([c['x'] for c in collapsed_data])
        all_y = np.concatenate([c['y'] for c in collapsed_data])
        order = np.argsort(all_x)
        all_x, all_y = all_x[order], all_y[order]

        if method == 'polynomial':
            for deg in range(2, 8):
                coeffs = np.polyfit(all_x, all_y, deg)
                pred = np.polyval(coeffs, all_x)
                ss_res = np.sum((all_y - pred) ** 2)
                ss_tot = np.sum((all_y - np.mean(all_y)) ** 2)
                r2 = 1 - ss_res / max(ss_tot, 1e-12)
                if r2 > 0.99:
                    break
            return {
                'method': 'polynomial',
                'degree': deg,
                'coefficients': coeffs,
                'r_squared': r2,
                'predict': lambda x, c=coeffs: np.polyval(c, x),
            }
        elif method == 'spline':
            spl = UnivariateSpline(all_x, all_y, s=len(all_x) * 0.1)
            pred = spl(all_x)
            ss_res = np.sum((all_y - pred) ** 2)
            ss_tot = np.sum((all_y - np.mean(all_y)) ** 2)
            return {
                'method': 'spline',
                'spline': spl,
                'r_squared': 1 - ss_res / max(ss_tot, 1e-12),
                'predict': spl,
            }
        else:
            raise ValueError(f"Unknown method: {method}")

    def residual_analysis(self, collapsed_data, universal_func):
        """Analyze residuals from the scaling collapse."""
        residuals_by_size = {}
        all_res = []
        for c in collapsed_data:
            pred = universal_func(c['x'])
            res = c['y'] - pred
            residuals_by_size[c['size']] = res
            all_res.extend(res)

        all_res = np.array(all_res)
        _, sw_p = stats.shapiro(all_res[:min(5000, len(all_res))])
        return {
            'residuals_by_size': residuals_by_size,
            'overall_std': np.std(all_res),
            'overall_mean': np.mean(all_res),
            'shapiro_p_value': sw_p,
            'normally_distributed': sw_p > 0.05,
            'max_residual': np.max(np.abs(all_res)),
        }

    def bootstrap_exponent_errors(self, data_sets, sizes, control_params, n_bootstrap=500):
        """Bootstrap error bars on optimized exponents."""
        n_sizes = len(sizes)
        nu_samples, eta_samples = [], []

        for _ in range(n_bootstrap):
            boot_data = []
            for ds in data_sets:
                ds = np.asarray(ds, dtype=float)
                idx = np.random.randint(0, len(ds), size=len(ds))
                boot_data.append(ds[idx])
                boot_cp = np.asarray(control_params, dtype=float)[idx]

            try:
                result = self.optimize_exponents(
                    boot_data, sizes, boot_cp,
                    nu_range=(0.1, 3.0), eta_range=(-2.0, 2.0),
                )
                nu_samples.append(result['nu'])
                eta_samples.append(result['eta'])
            except Exception:
                continue

        nu_samples = np.array(nu_samples)
        eta_samples = np.array(eta_samples)
        return {
            'nu_mean': np.mean(nu_samples),
            'nu_std': np.std(nu_samples),
            'nu_ci_95': (np.percentile(nu_samples, 2.5), np.percentile(nu_samples, 97.5)),
            'eta_mean': np.mean(eta_samples),
            'eta_std': np.std(eta_samples),
            'eta_ci_95': (np.percentile(eta_samples, 2.5), np.percentile(eta_samples, 97.5)),
            'n_successful': len(nu_samples),
        }


class PowerLawFitter:
    """Power-law fitting with uncertainty quantification."""

    def __init__(self, method='mle'):
        self.method = method

    def fit_power_law(self, x, y):
        """Fit y = a·x^b using log-linear regression or MLE."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if self.method == 'mle':
            return self.mle_fit(x, y)

        mask = (x > 0) & (y > 0)
        lx, ly = np.log(x[mask]), np.log(y[mask])
        slope, intercept, r, p, stderr = stats.linregress(lx, ly)
        return {
            'a': np.exp(intercept),
            'b': slope,
            'r_squared': r ** 2,
            'p_value': p,
            'stderr_b': stderr,
        }

    def fit_with_constant(self, x, y):
        """Fit y = a·x^b + c (power law with constant offset)."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

        def model(xv, a, b, c):
            return a * np.power(xv, b) + c

        simple = self.fit_power_law(x, y)
        a0 = simple.get('a', 1.0)
        b0 = simple.get('b', -1.0)
        try:
            popt, pcov = optimize.curve_fit(
                model, x, y, p0=[a0, b0, 0.0], maxfev=10000,
            )
            perr = np.sqrt(np.diag(pcov))
            y_pred = model(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return {
                'a': popt[0], 'b': popt[1], 'c': popt[2],
                'uncertainties': {'a': perr[0], 'b': perr[1], 'c': perr[2]},
                'r_squared': 1 - ss_res / max(ss_tot, 1e-12),
            }
        except (RuntimeError, ValueError):
            return {'a': np.nan, 'b': np.nan, 'c': np.nan, 'r_squared': 0.0}

    def fit_broken_power_law(self, x, y):
        """Fit broken power law: two power laws joined at a breakpoint."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        order = np.argsort(x)
        x, y = x[order], y[order]

        def model(xv, a1, b1, a2, b2, x_break):
            result = np.empty_like(xv)
            lo = xv <= x_break
            hi = ~lo
            result[lo] = a1 * xv[lo] ** b1
            # Ensure continuity at break: a2 * x_break^b2 = a1 * x_break^b1
            a2_cont = a1 * x_break ** (b1 - b2)
            result[hi] = a2_cont * xv[hi] ** b2
            return result

        x_mid = np.exp(0.5 * (np.log(x[0] + 1e-12) + np.log(x[-1] + 1e-12)))
        fit_lo = self.fit_power_law(x[:len(x) // 2], y[:len(y) // 2])
        fit_hi = self.fit_power_law(x[len(x) // 2:], y[len(y) // 2:])

        try:
            popt, pcov = optimize.curve_fit(
                model, x, y,
                p0=[fit_lo.get('a', 1), fit_lo.get('b', -1),
                    fit_hi.get('a', 1), fit_hi.get('b', -1), x_mid],
                maxfev=20000,
            )
            y_pred = model(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return {
                'a1': popt[0], 'b1': popt[1],
                'a2': popt[2], 'b2': popt[3],
                'x_break': popt[4],
                'r_squared': 1 - ss_res / max(ss_tot, 1e-12),
            }
        except (RuntimeError, ValueError):
            return {'x_break': np.nan, 'r_squared': 0.0}

    def goodness_of_fit(self, x, y, fit_params):
        """Compute R², adjusted R², AIC, BIC for a power-law fit."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        a, b = fit_params.get('a', 1.0), fit_params.get('b', 1.0)
        c = fit_params.get('c', 0.0)
        y_pred = a * x ** b + c
        n = len(y)
        k = 3 if 'c' in fit_params else 2
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        r2_adj = 1 - (1 - r2) * (n - 1) / max(n - k - 1, 1)
        log_likelihood = -0.5 * n * np.log(2 * np.pi * ss_res / max(n, 1)) - n / 2
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(max(n, 1)) - 2 * log_likelihood
        return {
            'r_squared': r2,
            'r_squared_adjusted': r2_adj,
            'aic': aic,
            'bic': bic,
            'rmse': np.sqrt(ss_res / max(n, 1)),
        }

    def power_law_vs_exponential(self, x, y):
        """Compare power-law vs exponential fit using AIC/BIC."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        mask = (x > 0) & (y > 0)
        x, y = x[mask], y[mask]
        n = len(x)

        # Power-law fit
        pl = self.fit_power_law(x, y)
        y_pl = pl['a'] * x ** pl['b']
        ss_pl = np.sum((y - y_pl) ** 2)

        # Exponential fit: y = A * exp(B * x)
        try:
            def exp_model(xv, A, B):
                return A * np.exp(B * xv)
            popt, _ = optimize.curve_fit(exp_model, x, y, p0=[y[0], -0.01], maxfev=5000)
            y_exp = exp_model(x, *popt)
            ss_exp = np.sum((y - y_exp) ** 2)
        except (RuntimeError, ValueError):
            ss_exp = np.inf
            popt = [np.nan, np.nan]

        aic_pl = n * np.log(ss_pl / max(n, 1)) + 2 * 2
        aic_exp = n * np.log(ss_exp / max(n, 1)) + 2 * 2 if np.isfinite(ss_exp) else np.inf
        delta_aic = aic_pl - aic_exp  # negative favors power law

        return {
            'power_law_params': {'a': pl['a'], 'b': pl['b']},
            'exponential_params': {'A': popt[0], 'B': popt[1]},
            'aic_power_law': aic_pl,
            'aic_exponential': aic_exp,
            'delta_aic': delta_aic,
            'preferred': 'power_law' if delta_aic < 0 else 'exponential',
        }

    def prediction_interval(self, x_new, fit_params, x_data, y_data):
        """Compute prediction intervals for new x values."""
        x_new = np.asarray(x_new, dtype=float)
        x_data = np.asarray(x_data, dtype=float)
        y_data = np.asarray(y_data, dtype=float)
        a, b = fit_params['a'], fit_params['b']
        c = fit_params.get('c', 0.0)

        y_pred_data = a * x_data ** b + c
        residuals = y_data - y_pred_data
        sigma = np.std(residuals)
        n = len(x_data)

        # Prediction in log-space for heteroscedastic errors
        mask = (x_data > 0) & (y_data > 0)
        if np.sum(mask) > 2:
            log_x = np.log(x_data[mask])
            log_x_mean = np.mean(log_x)
            s_xx = np.sum((log_x - log_x_mean) ** 2)
            log_x_new = np.log(np.maximum(x_new, 1e-12))
            leverage = 1.0 / n + (log_x_new - log_x_mean) ** 2 / max(s_xx, 1e-12)
        else:
            leverage = 1.0 / max(n, 1) * np.ones_like(x_new)

        t_val = stats.t.ppf(0.975, max(n - 2, 1))
        y_pred = a * x_new ** b + c
        margin = t_val * sigma * np.sqrt(1 + leverage)
        return {
            'prediction': y_pred,
            'lower_95': y_pred - margin,
            'upper_95': y_pred + margin,
            'sigma': sigma,
        }

    def mle_fit(self, x, y):
        """Maximum likelihood estimation for power-law y = a·x^b + noise."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

        def neg_log_likelihood(params):
            a, b, log_sigma = params
            sigma = np.exp(log_sigma)
            y_pred = a * np.power(np.maximum(x, 1e-12), b)
            return 0.5 * len(x) * np.log(2 * np.pi * sigma ** 2) + \
                   np.sum((y - y_pred) ** 2) / (2 * sigma ** 2)

        # Initialize with OLS in log-space
        mask = (x > 0) & (y > 0)
        if np.sum(mask) >= 2:
            slope, intercept, _, _, _ = stats.linregress(np.log(x[mask]), np.log(y[mask]))
            p0 = [np.exp(intercept), slope, np.log(0.1)]
        else:
            p0 = [1.0, -1.0, np.log(0.1)]

        result = optimize.minimize(neg_log_likelihood, p0, method='Nelder-Mead',
                                   options={'maxiter': 5000})
        a, b, log_sigma = result.x

        # Approximate Hessian for uncertainties
        try:
            hess_inv = result.hess_inv if hasattr(result, 'hess_inv') else None
            if hess_inv is None:
                from scipy.optimize import approx_fprime
                def grad(p):
                    return optimize.approx_fprime(p, neg_log_likelihood, 1e-8)
                H = np.zeros((3, 3))
                for i in range(3):
                    e = np.zeros(3)
                    e[i] = 1e-5
                    H[i] = (grad(result.x + e) - grad(result.x - e)) / (2e-5)
                try:
                    cov = np.linalg.inv(H)
                    perr = np.sqrt(np.abs(np.diag(cov)))
                except np.linalg.LinAlgError:
                    perr = np.full(3, np.nan)
            else:
                perr = np.sqrt(np.abs(np.diag(hess_inv)))
        except Exception:
            perr = np.full(3, np.nan)

        y_pred = a * np.power(np.maximum(x, 1e-12), b)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return {
            'a': a, 'b': b, 'sigma': np.exp(log_sigma),
            'neg_log_likelihood': result.fun,
            'r_squared': 1 - ss_res / max(ss_tot, 1e-12),
            'uncertainties': {'a': perr[0], 'b': perr[1]},
        }

    def bayesian_fit(self, x, y, prior_params=None):
        """Bayesian power-law fitting using Metropolis-Hastings MCMC.

        prior_params: dict with 'a_mean', 'a_std', 'b_mean', 'b_std'
                      for Gaussian priors on log(a) and b.
        """
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if prior_params is None:
            prior_params = {'a_mean': 0.0, 'a_std': 10.0,
                            'b_mean': -1.0, 'b_std': 5.0}

        def log_posterior(params):
            log_a, b, log_sigma = params
            a = np.exp(log_a)
            sigma = np.exp(log_sigma)
            if sigma < 1e-10:
                return -np.inf
            y_pred = a * np.power(np.maximum(x, 1e-12), b)
            ll = -0.5 * np.sum(((y - y_pred) / sigma) ** 2) - \
                 len(x) * np.log(sigma)
            # Gaussian priors
            lp = -0.5 * ((log_a - prior_params['a_mean']) / prior_params['a_std']) ** 2
            lp += -0.5 * ((b - prior_params['b_mean']) / prior_params['b_std']) ** 2
            lp += -0.5 * (log_sigma / 5.0) ** 2  # broad prior on log(sigma)
            return ll + lp

        # Initialize from MLE
        mle = self.mle_fit(x, y)
        current = np.array([np.log(max(mle['a'], 1e-12)), mle['b'],
                            np.log(max(mle.get('sigma', 0.1), 1e-12))])
        current_lp = log_posterior(current)

        n_samples = 5000
        n_burn = 1000
        proposal_scale = np.array([0.1, 0.1, 0.1])
        samples = []
        n_accept = 0

        for i in range(n_samples + n_burn):
            proposal = current + proposal_scale * np.random.randn(3)
            proposal_lp = log_posterior(proposal)

            if np.log(np.random.rand()) < proposal_lp - current_lp:
                current = proposal
                current_lp = proposal_lp
                n_accept += 1

            if i >= n_burn:
                samples.append(current.copy())

        samples = np.array(samples)
        a_samples = np.exp(samples[:, 0])
        b_samples = samples[:, 1]
        sigma_samples = np.exp(samples[:, 2])

        return {
            'a_mean': np.mean(a_samples),
            'a_std': np.std(a_samples),
            'a_ci_95': (np.percentile(a_samples, 2.5), np.percentile(a_samples, 97.5)),
            'b_mean': np.mean(b_samples),
            'b_std': np.std(b_samples),
            'b_ci_95': (np.percentile(b_samples, 2.5), np.percentile(b_samples, 97.5)),
            'sigma_mean': np.mean(sigma_samples),
            'acceptance_rate': n_accept / (n_samples + n_burn),
            'samples': {'a': a_samples, 'b': b_samples, 'sigma': sigma_samples},
        }
