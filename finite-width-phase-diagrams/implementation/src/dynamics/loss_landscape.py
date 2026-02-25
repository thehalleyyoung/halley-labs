"""Loss landscape analysis: Hessian spectra, saddle detection, barriers, and trajectory analysis."""

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.interpolate import CubicSpline


class HessianAnalyzer:
    """Hessian computation and eigenspectrum analysis."""

    def __init__(self, param_dim, loss_fn=None):
        self.param_dim = param_dim
        self.loss_fn = loss_fn

    def compute_hessian(self, params, loss_fn, data_x, data_y, epsilon=1e-5):
        """Full Hessian via second-order finite differences.

        H_ij = (L(p + e_i*eps + e_j*eps) - L(p + e_i*eps - e_j*eps)
              - L(p - e_i*eps + e_j*eps) + L(p - e_i*eps - e_j*eps)) / (4*eps^2)
        """
        d = len(params)
        hessian = np.zeros((d, d))
        params = np.array(params, dtype=np.float64)

        for i in range(d):
            for j in range(i, d):
                e_i = np.zeros(d)
                e_j = np.zeros(d)
                e_i[i] = epsilon
                e_j[j] = epsilon

                fpp = loss_fn(params + e_i + e_j, data_x, data_y)
                fpm = loss_fn(params + e_i - e_j, data_x, data_y)
                fmp = loss_fn(params - e_i + e_j, data_x, data_y)
                fmm = loss_fn(params - e_i - e_j, data_x, data_y)

                hessian[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * epsilon * epsilon)
                hessian[j, i] = hessian[i, j]

        return hessian

    def hessian_vector_product(self, params, vector, loss_fn, data_x, data_y, epsilon=1e-5):
        """Hessian-vector product via finite differences: Hv ≈ (∇L(p+εv) - ∇L(p-εv)) / (2ε).

        The gradient itself is computed via finite differences.
        """
        params = np.array(params, dtype=np.float64)
        vector = np.array(vector, dtype=np.float64)
        d = len(params)

        def _grad(p):
            g = np.zeros(d)
            for k in range(d):
                e_k = np.zeros(d)
                e_k[k] = epsilon
                g[k] = (loss_fn(p + e_k, data_x, data_y) -
                         loss_fn(p - e_k, data_x, data_y)) / (2.0 * epsilon)
            return g

        grad_plus = _grad(params + epsilon * vector)
        grad_minus = _grad(params - epsilon * vector)
        return (grad_plus - grad_minus) / (2.0 * epsilon)

    def top_eigenvalues(self, hessian, k=20):
        """Top-k eigenvalues via scipy.sparse.linalg.eigsh (Lanczos)."""
        n = hessian.shape[0]
        k = min(k, n - 1) if n > 1 else 1
        if k < 1:
            return np.array([]), np.array([[]])
        eigenvalues, eigenvectors = eigsh(hessian, k=k, which='LM')
        idx = np.argsort(-eigenvalues)
        return eigenvalues[idx], eigenvectors[:, idx]

    def full_spectrum(self, hessian):
        """Full eigenvalue decomposition of symmetric Hessian."""
        eigenvalues, eigenvectors = linalg.eigh(hessian)
        idx = np.argsort(-eigenvalues)
        return eigenvalues[idx], eigenvectors[:, idx]

    def spectral_density(self, eigenvalues, n_bins=100):
        """Density of states ρ(λ) via histogram of eigenvalues."""
        eigenvalues = np.array(eigenvalues)
        if len(eigenvalues) == 0:
            return np.array([]), np.array([])

        hist, bin_edges = np.histogram(eigenvalues, bins=n_bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return bin_centers, hist

    def lanczos_spectrum(self, hessian_vector_product_fn, dim, n_lanczos=100):
        """Lanczos approximation to eigenvalue spectrum.

        Builds tridiagonal matrix T via Lanczos iteration, then diagonalizes T.
        """
        n_lanczos = min(n_lanczos, dim)
        alpha = np.zeros(n_lanczos)
        beta = np.zeros(n_lanczos)

        q = np.random.randn(dim)
        q = q / np.linalg.norm(q)
        q_prev = np.zeros(dim)

        Q = np.zeros((dim, n_lanczos))
        Q[:, 0] = q

        for j in range(n_lanczos):
            w = hessian_vector_product_fn(q)
            alpha[j] = np.dot(q, w)
            if j < n_lanczos - 1:
                w = w - alpha[j] * q - beta[j] * q_prev
                # Re-orthogonalize against all previous Lanczos vectors
                for k in range(j + 1):
                    w -= np.dot(Q[:, k], w) * Q[:, k]
                beta[j + 1] = np.linalg.norm(w)
                if beta[j + 1] < 1e-12:
                    n_lanczos = j + 1
                    alpha = alpha[:n_lanczos]
                    beta = beta[:n_lanczos]
                    break
                q_prev = q
                q = w / beta[j + 1]
                Q[:, j + 1] = q

        T = np.diag(alpha[:n_lanczos])
        for j in range(n_lanczos - 1):
            T[j, j + 1] = beta[j + 1]
            T[j + 1, j] = beta[j + 1]

        ritz_values = linalg.eigvalsh(T)
        return np.sort(ritz_values)[::-1]

    def trace_estimation(self, hvp_fn, dim, n_samples=30):
        """Hutchinson's stochastic trace estimator: Tr(H) ≈ (1/m) Σ z^T H z."""
        trace_est = 0.0
        for _ in range(n_samples):
            z = np.random.choice([-1.0, 1.0], size=dim)
            hz = hvp_fn(z)
            trace_est += np.dot(z, hz)
        return trace_est / n_samples

    def bulk_edge_ratio(self, eigenvalues):
        """Ratio of outlier eigenvalues to bulk edge.

        Uses median + 1.5*IQR to define bulk edge, returns λ_max / bulk_edge.
        """
        eigenvalues = np.sort(np.array(eigenvalues))[::-1]
        if len(eigenvalues) < 4:
            return float('inf') if len(eigenvalues) > 0 else 0.0

        q75, q25 = np.percentile(eigenvalues, [75, 25])
        iqr = q75 - q25
        bulk_edge = q75 + 1.5 * iqr
        if abs(bulk_edge) < 1e-15:
            return float('inf')
        return eigenvalues[0] / bulk_edge

    def condition_number(self, eigenvalues):
        """κ = λ_max / λ_min over positive eigenvalues."""
        pos = eigenvalues[eigenvalues > 0]
        if len(pos) < 2:
            return float('inf')
        return np.max(pos) / np.min(pos)

    def negative_curvature_directions(self, eigenvalues, eigenvectors):
        """Return eigenvectors corresponding to negative eigenvalues."""
        mask = eigenvalues < 0
        neg_vals = eigenvalues[mask]
        neg_vecs = eigenvectors[:, mask]
        idx = np.argsort(neg_vals)
        return neg_vals[idx], neg_vecs[:, idx]

    def sharpness(self, eigenvalues):
        """Various sharpness measures from the Hessian spectrum."""
        eigenvalues = np.array(eigenvalues)
        pos = eigenvalues[eigenvalues > 0]
        return {
            'trace': float(np.sum(eigenvalues)),
            'max_eigenvalue': float(np.max(eigenvalues)) if len(eigenvalues) > 0 else 0.0,
            'spectral_norm': float(np.max(np.abs(eigenvalues))) if len(eigenvalues) > 0 else 0.0,
            'frobenius_approx': float(np.sqrt(np.sum(eigenvalues ** 2))),
            'log_sum_pos': float(np.sum(np.log(pos))) if len(pos) > 0 else 0.0,
            'n_negative': int(np.sum(eigenvalues < 0)),
            'mean_positive': float(np.mean(pos)) if len(pos) > 0 else 0.0,
        }

    def spectral_gap(self, eigenvalues):
        """Gap between consecutive leading eigenvalues."""
        eigenvalues = np.sort(np.array(eigenvalues))[::-1]
        if len(eigenvalues) < 2:
            return np.array([])
        gaps = eigenvalues[:-1] - eigenvalues[1:]
        return gaps


class LossSurfaceVisualizer:
    """Loss surface visualization data generation."""

    def __init__(self, loss_fn, data_x, data_y):
        self.loss_fn = loss_fn
        self.data_x = data_x
        self.data_y = data_y

    def compute_1d_slice(self, params, direction, range_val=(-1, 1), n_points=200):
        """1D loss slice: L(θ + α·d) for α ∈ [range_val[0], range_val[1]]."""
        params = np.array(params, dtype=np.float64)
        direction = np.array(direction, dtype=np.float64)
        alphas = np.linspace(range_val[0], range_val[1], n_points)
        losses = np.zeros(n_points)
        for i, alpha in enumerate(alphas):
            losses[i] = self.loss_fn(params + alpha * direction, self.data_x, self.data_y)
        return alphas, losses

    def compute_2d_slice(self, params, dir1, dir2, range1=(-1, 1), range2=(-1, 1),
                         resolution=50):
        """2D loss slice: L(θ + α·d1 + β·d2)."""
        params = np.array(params, dtype=np.float64)
        dir1 = np.array(dir1, dtype=np.float64)
        dir2 = np.array(dir2, dtype=np.float64)
        alphas = np.linspace(range1[0], range1[1], resolution)
        betas = np.linspace(range2[0], range2[1], resolution)
        loss_grid = np.zeros((resolution, resolution))

        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                p = params + alpha * dir1 + beta * dir2
                loss_grid[i, j] = self.loss_fn(p, self.data_x, self.data_y)

        return alphas, betas, loss_grid

    def random_direction(self, param_dim, normalize=True):
        """Random unit direction in parameter space (Gaussian)."""
        d = np.random.randn(param_dim)
        if normalize:
            norm = np.linalg.norm(d)
            if norm > 1e-15:
                d = d / norm
        return d

    def filter_normalized_direction(self, direction, params):
        """Filter normalization: scale direction components by parameter norms.

        Each component of direction is scaled by ||θ_i|| / ||d_i|| to produce
        directions that account for different parameter scales.
        """
        direction = np.array(direction, dtype=np.float64)
        params = np.array(params, dtype=np.float64)
        param_norm = np.linalg.norm(params)
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-15 or param_norm < 1e-15:
            return direction
        return direction * (param_norm / dir_norm)

    def pca_directions(self, trajectory, n_components=2):
        """Top PCA directions of optimization trajectory."""
        trajectory = np.array(trajectory)
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(1, -1)
        mean = np.mean(trajectory, axis=0)
        centered = trajectory - mean
        cov = np.dot(centered.T, centered) / max(len(trajectory) - 1, 1)
        eigenvalues, eigenvectors = linalg.eigh(cov)
        idx = np.argsort(-eigenvalues)[:n_components]
        return eigenvectors[:, idx], eigenvalues[idx], mean

    def loss_along_trajectory(self, trajectory):
        """Evaluate loss at each point along optimization trajectory."""
        trajectory = np.array(trajectory)
        losses = np.zeros(len(trajectory))
        for i, params in enumerate(trajectory):
            losses[i] = self.loss_fn(params, self.data_x, self.data_y)
        return losses

    def interpolation_loss(self, params1, params2, n_points=100):
        """Loss along linear interpolation: θ(t) = (1-t)θ₁ + tθ₂."""
        params1 = np.array(params1, dtype=np.float64)
        params2 = np.array(params2, dtype=np.float64)
        ts = np.linspace(0.0, 1.0, n_points)
        losses = np.zeros(n_points)
        for i, t in enumerate(ts):
            p = (1.0 - t) * params1 + t * params2
            losses[i] = self.loss_fn(p, self.data_x, self.data_y)
        return ts, losses

    def loss_contour_data(self, params, dir1, dir2, levels=20):
        """Generate data for contour plots of loss surface."""
        alphas, betas, loss_grid = self.compute_2d_slice(params, dir1, dir2)
        vmin, vmax = np.min(loss_grid), np.max(loss_grid)
        contour_levels = np.linspace(vmin, vmax, levels)
        return {
            'alphas': alphas,
            'betas': betas,
            'loss_grid': loss_grid,
            'contour_levels': contour_levels,
            'vmin': float(vmin),
            'vmax': float(vmax),
        }


class SaddlePointDetector:
    """Detect and classify saddle points in the loss landscape."""

    def __init__(self, gradient_threshold=1e-6):
        self.gradient_threshold = gradient_threshold

    def _numerical_gradient(self, params, loss_fn, data_x, data_y, epsilon=1e-5):
        """Central-difference gradient."""
        params = np.array(params, dtype=np.float64)
        d = len(params)
        grad = np.zeros(d)
        for k in range(d):
            e_k = np.zeros(d)
            e_k[k] = epsilon
            grad[k] = (loss_fn(params + e_k, data_x, data_y) -
                        loss_fn(params - e_k, data_x, data_y)) / (2.0 * epsilon)
        return grad

    def find_saddle_points(self, params, loss_fn, grad_fn, data_x, data_y,
                           max_iter=1000, lr=0.01, noise_scale=0.01):
        """Find stationary points via gradient descent with noise injection.

        Uses gradient descent with occasional noise to escape minima, collecting
        points where ||∇L|| < threshold.
        """
        params = np.array(params, dtype=np.float64).copy()
        stationary_points = []
        grad_norms = []

        for step in range(max_iter):
            if grad_fn is not None:
                grad = grad_fn(params, data_x, data_y)
            else:
                grad = self._numerical_gradient(params, loss_fn, data_x, data_y)

            g_norm = np.linalg.norm(grad)
            grad_norms.append(g_norm)

            if g_norm < self.gradient_threshold:
                stationary_points.append({
                    'params': params.copy(),
                    'grad_norm': float(g_norm),
                    'loss': float(loss_fn(params, data_x, data_y)),
                    'step': step,
                })

            params -= lr * grad
            if step % 50 == 0 and step > 0:
                params += noise_scale * np.random.randn(len(params))

        return stationary_points, np.array(grad_norms)

    def classify_stationary_point(self, hessian):
        """Classify stationary point as minimum, maximum, or saddle via Hessian eigenvalues."""
        eigenvalues = linalg.eigvalsh(hessian)
        n_pos = int(np.sum(eigenvalues > 0))
        n_neg = int(np.sum(eigenvalues < 0))
        n_zero = int(np.sum(np.abs(eigenvalues) < 1e-10))

        if n_neg == 0 and n_zero == 0:
            kind = 'strict_minimum'
        elif n_neg == 0:
            kind = 'minimum'  # may be degenerate
        elif n_pos == 0 and n_zero == 0:
            kind = 'strict_maximum'
        elif n_pos == 0:
            kind = 'maximum'
        else:
            kind = 'saddle'

        return {
            'type': kind,
            'n_positive': n_pos,
            'n_negative': n_neg,
            'n_zero': n_zero,
            'eigenvalues': eigenvalues,
        }

    def saddle_index(self, hessian):
        """Saddle index: number of negative eigenvalues of the Hessian."""
        eigenvalues = linalg.eigvalsh(hessian)
        return int(np.sum(eigenvalues < 0))

    def escape_direction(self, hessian):
        """Direction of most negative curvature (eigenvector of smallest eigenvalue)."""
        eigenvalues, eigenvectors = linalg.eigh(hessian)
        idx = np.argmin(eigenvalues)
        return eigenvectors[:, idx], float(eigenvalues[idx])

    def saddle_connectivity(self, saddle_points, loss_fn, data_x, data_y,
                            step_size=0.01, max_steps=500):
        """Determine which minima are connected through each saddle point.

        From each saddle, follow positive and negative escape directions to
        find the minima on either side.
        """
        connections = []
        for sp_info in saddle_points:
            sp = sp_info['params']
            hessian_analyzer = HessianAnalyzer(len(sp))
            hessian = hessian_analyzer.compute_hessian(sp, loss_fn, data_x, data_y)
            esc_dir, esc_val = self.escape_direction(hessian)

            if esc_val >= 0:
                continue

            # Follow escape direction in both senses
            endpoints = []
            for sign in [1.0, -1.0]:
                p = sp.copy() + sign * step_size * esc_dir
                for _ in range(max_steps):
                    grad = self._numerical_gradient(p, loss_fn, data_x, data_y)
                    g_norm = np.linalg.norm(grad)
                    if g_norm < self.gradient_threshold:
                        break
                    p -= step_size * grad
                endpoints.append(p.copy())

            connections.append({
                'saddle': sp.copy(),
                'saddle_loss': float(loss_fn(sp, data_x, data_y)),
                'minimum_a': endpoints[0],
                'minimum_b': endpoints[1],
                'loss_a': float(loss_fn(endpoints[0], data_x, data_y)),
                'loss_b': float(loss_fn(endpoints[1], data_x, data_y)),
                'escape_eigenvalue': float(esc_val),
            })

        return connections

    def gradient_norm_landscape(self, params_range, loss_fn, grad_fn, data_x, data_y,
                                resolution=50):
        """Compute ||∇L|| over a 2D parameter grid.

        params_range: dict with keys 'dim1_range', 'dim2_range', 'base_params',
                      'dim1_idx', 'dim2_idx'.
        """
        base = np.array(params_range['base_params'], dtype=np.float64)
        d1_range = params_range['dim1_range']
        d2_range = params_range['dim2_range']
        d1_idx = params_range['dim1_idx']
        d2_idx = params_range['dim2_idx']

        vals1 = np.linspace(d1_range[0], d1_range[1], resolution)
        vals2 = np.linspace(d2_range[0], d2_range[1], resolution)
        grad_norms = np.zeros((resolution, resolution))

        for i, v1 in enumerate(vals1):
            for j, v2 in enumerate(vals2):
                p = base.copy()
                p[d1_idx] = v1
                p[d2_idx] = v2
                if grad_fn is not None:
                    g = grad_fn(p, data_x, data_y)
                else:
                    g = self._numerical_gradient(p, loss_fn, data_x, data_y)
                grad_norms[i, j] = np.linalg.norm(g)

        return vals1, vals2, grad_norms

    def newton_step_to_saddle(self, params, grad_fn, hessian_fn, data_x, data_y,
                              max_iter=100, tol=1e-8, damping=1e-4):
        """Newton's method to converge to a nearby stationary point.

        Uses damped Newton: Δθ = -(H + λI)⁻¹ ∇L.
        """
        params = np.array(params, dtype=np.float64).copy()
        trajectory = [params.copy()]

        for step in range(max_iter):
            grad = grad_fn(params, data_x, data_y)
            g_norm = np.linalg.norm(grad)
            if g_norm < tol:
                break

            H = hessian_fn(params, data_x, data_y)
            d = H.shape[0]
            H_damped = H + damping * np.eye(d)

            try:
                delta = linalg.solve(H_damped, -grad, assume_a='sym')
            except linalg.LinAlgError:
                delta = -grad * 0.01

            params += delta
            trajectory.append(params.copy())

        return params, np.array(trajectory)


class LossBarrierEstimator:
    """Estimate loss barriers between minima."""

    def __init__(self, n_interpolation_points=50):
        self.n_interpolation_points = n_interpolation_points

    def linear_barrier(self, params1, params2, loss_fn, data_x, data_y):
        """Maximum loss along linear interpolation minus endpoint losses."""
        params1 = np.array(params1, dtype=np.float64)
        params2 = np.array(params2, dtype=np.float64)
        ts = np.linspace(0.0, 1.0, self.n_interpolation_points)
        losses = np.zeros(self.n_interpolation_points)
        for i, t in enumerate(ts):
            p = (1.0 - t) * params1 + t * params2
            losses[i] = loss_fn(p, data_x, data_y)

        endpoint_loss = max(losses[0], losses[-1])
        max_loss = np.max(losses)
        return {
            'barrier_height': float(max_loss - endpoint_loss),
            'max_loss': float(max_loss),
            'max_loss_t': float(ts[np.argmax(losses)]),
            'ts': ts,
            'losses': losses,
        }

    def neb_barrier(self, params1, params2, loss_fn, grad_fn, data_x, data_y,
                    n_images=20, spring_constant=1.0, max_iter=200, step_size=0.01):
        """Nudged elastic band method for minimum energy path.

        Optimizes a chain of images between two endpoints with spring forces
        along the band and true forces perpendicular to it.
        """
        params1 = np.array(params1, dtype=np.float64)
        params2 = np.array(params2, dtype=np.float64)
        d = len(params1)

        # Initialize images by linear interpolation
        images = np.zeros((n_images, d))
        for i in range(n_images):
            t = i / (n_images - 1)
            images[i] = (1.0 - t) * params1 + t * params2

        def _numerical_grad(p):
            if grad_fn is not None:
                return grad_fn(p, data_x, data_y)
            eps = 1e-5
            g = np.zeros(d)
            for k in range(d):
                e_k = np.zeros(d)
                e_k[k] = eps
                g[k] = (loss_fn(p + e_k, data_x, data_y) -
                         loss_fn(p - e_k, data_x, data_y)) / (2.0 * eps)
            return g

        for iteration in range(max_iter):
            # Only optimize interior images (endpoints are fixed)
            for i in range(1, n_images - 1):
                # Tangent along the band
                tau = images[i + 1] - images[i - 1]
                tau_norm = np.linalg.norm(tau)
                if tau_norm > 1e-15:
                    tau_hat = tau / tau_norm
                else:
                    tau_hat = np.zeros(d)

                # True gradient (perpendicular component)
                grad = _numerical_grad(images[i])
                grad_perp = grad - np.dot(grad, tau_hat) * tau_hat

                # Spring force (parallel component)
                spring_force = (spring_constant *
                                (np.linalg.norm(images[i + 1] - images[i]) -
                                 np.linalg.norm(images[i] - images[i - 1])) * tau_hat)

                # NEB force = -grad_perp + spring_parallel
                neb_force = -grad_perp + spring_force
                images[i] += step_size * neb_force

        # Evaluate losses along converged path
        losses = np.array([loss_fn(images[i], data_x, data_y) for i in range(n_images)])
        endpoint_loss = max(losses[0], losses[-1])

        return {
            'barrier_height': float(np.max(losses) - endpoint_loss),
            'max_loss': float(np.max(losses)),
            'images': images.copy(),
            'losses': losses,
            'path_length': float(np.sum(np.linalg.norm(np.diff(images, axis=0), axis=1))),
        }

    def string_method_barrier(self, params1, params2, loss_fn, grad_fn, data_x, data_y,
                              n_images=30, max_iter=200, step_size=0.01):
        """String method: evolve images by gradient then reparametrize by arc length."""
        params1 = np.array(params1, dtype=np.float64)
        params2 = np.array(params2, dtype=np.float64)
        d = len(params1)

        images = np.zeros((n_images, d))
        for i in range(n_images):
            t = i / (n_images - 1)
            images[i] = (1.0 - t) * params1 + t * params2

        def _numerical_grad(p):
            if grad_fn is not None:
                return grad_fn(p, data_x, data_y)
            eps = 1e-5
            g = np.zeros(d)
            for k in range(d):
                e_k = np.zeros(d)
                e_k[k] = eps
                g[k] = (loss_fn(p + e_k, data_x, data_y) -
                         loss_fn(p - e_k, data_x, data_y)) / (2.0 * eps)
            return g

        for iteration in range(max_iter):
            # Evolve interior images by negative gradient
            for i in range(1, n_images - 1):
                grad = _numerical_grad(images[i])
                images[i] -= step_size * grad

            # Reparametrize by arc length using cubic spline interpolation
            seg_lengths = np.linalg.norm(np.diff(images, axis=0), axis=1)
            cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
            total_length = cumulative[-1]
            if total_length < 1e-15:
                continue

            cumulative_normalized = cumulative / total_length
            target_s = np.linspace(0.0, 1.0, n_images)

            new_images = np.zeros_like(images)
            new_images[0] = images[0]
            new_images[-1] = images[-1]
            for dim_k in range(d):
                cs = CubicSpline(cumulative_normalized, images[:, dim_k])
                new_images[:, dim_k] = cs(target_s)

            # Keep endpoints fixed
            new_images[0] = params1
            new_images[-1] = params2
            images = new_images

        losses = np.array([loss_fn(images[i], data_x, data_y) for i in range(n_images)])
        endpoint_loss = max(losses[0], losses[-1])

        return {
            'barrier_height': float(np.max(losses) - endpoint_loss),
            'max_loss': float(np.max(losses)),
            'images': images.copy(),
            'losses': losses,
            'path_length': float(np.sum(np.linalg.norm(np.diff(images, axis=0), axis=1))),
        }

    def mode_connectivity(self, params1, params2, loss_fn, data_x, data_y,
                          threshold_factor=1.1):
        """Test if two minima are connected by a low-loss path.

        Minima are 'mode connected' if the barrier is small relative to
        the endpoint loss values. Uses Bezier curve as the test path.
        """
        params1 = np.array(params1, dtype=np.float64)
        params2 = np.array(params2, dtype=np.float64)

        loss1 = loss_fn(params1, data_x, data_y)
        loss2 = loss_fn(params2, data_x, data_y)
        ref_loss = max(loss1, loss2)

        # Linear interpolation test
        linear_result = self.linear_barrier(params1, params2, loss_fn, data_x, data_y)

        # Bezier curve test
        bezier_result = self.bezier_curve_barrier(params1, params2, loss_fn, data_x, data_y)

        best_barrier = min(linear_result['barrier_height'], bezier_result['barrier_height'])
        connected = best_barrier < threshold_factor * ref_loss if ref_loss > 1e-15 else best_barrier < 1e-6

        return {
            'connected': bool(connected),
            'linear_barrier': float(linear_result['barrier_height']),
            'bezier_barrier': float(bezier_result['barrier_height']),
            'best_barrier': float(best_barrier),
            'loss1': float(loss1),
            'loss2': float(loss2),
        }

    def bezier_curve_barrier(self, params1, params2, loss_fn, data_x, data_y,
                             n_bend=5, n_eval=100, optimize_iter=50, lr=0.01):
        """Bezier curve path between two points with optimizable control points.

        Optimizes control points to minimize the maximum loss along the curve.
        """
        params1 = np.array(params1, dtype=np.float64)
        params2 = np.array(params2, dtype=np.float64)
        d = len(params1)

        # Initialize control points by linear interpolation with small perturbation
        control_points = np.zeros((n_bend, d))
        for i in range(n_bend):
            t = (i + 1) / (n_bend + 1)
            control_points[i] = (1.0 - t) * params1 + t * params2
            control_points[i] += 0.01 * np.random.randn(d)

        def _eval_bezier(ctrl_pts, n_eval_pts):
            """Evaluate de Casteljau Bezier curve with given control points."""
            all_pts = np.vstack([params1.reshape(1, -1), ctrl_pts, params2.reshape(1, -1)])
            n_ctrl = len(all_pts)
            ts = np.linspace(0.0, 1.0, n_eval_pts)
            curve_pts = np.zeros((n_eval_pts, d))
            for idx, t in enumerate(ts):
                # De Casteljau
                working = all_pts.copy()
                for level in range(n_ctrl - 1):
                    new_working = np.zeros((n_ctrl - 1 - level, d))
                    for k in range(n_ctrl - 1 - level):
                        new_working[k] = (1.0 - t) * working[k] + t * working[k + 1]
                    working = new_working
                curve_pts[idx] = working[0]
            return curve_pts

        # Optimize control points to lower the barrier
        for opt_step in range(optimize_iter):
            curve_pts = _eval_bezier(control_points, n_eval)
            losses = np.array([loss_fn(curve_pts[i], data_x, data_y) for i in range(n_eval)])
            worst_idx = np.argmax(losses)

            # Numerical gradient of max loss w.r.t. control points
            for cp_idx in range(n_bend):
                for dim_k in range(d):
                    eps = 1e-4
                    control_points[cp_idx, dim_k] += eps
                    pts_p = _eval_bezier(control_points, n_eval)
                    loss_p = loss_fn(pts_p[worst_idx], data_x, data_y)

                    control_points[cp_idx, dim_k] -= 2.0 * eps
                    pts_m = _eval_bezier(control_points, n_eval)
                    loss_m = loss_fn(pts_m[worst_idx], data_x, data_y)

                    control_points[cp_idx, dim_k] += eps  # restore
                    grad_k = (loss_p - loss_m) / (2.0 * eps)
                    control_points[cp_idx, dim_k] -= lr * grad_k

        # Final evaluation
        curve_pts = _eval_bezier(control_points, n_eval)
        losses = np.array([loss_fn(curve_pts[i], data_x, data_y) for i in range(n_eval)])
        endpoint_loss = max(losses[0], losses[-1])

        return {
            'barrier_height': float(np.max(losses) - endpoint_loss),
            'max_loss': float(np.max(losses)),
            'control_points': control_points.copy(),
            'losses': losses,
            'curve_points': curve_pts,
        }

    def barrier_vs_width(self, params1_fn, params2_fn, loss_fn, data_fn, width_range):
        """Barrier height as a function of network width.

        params1_fn(width) -> params1, params2_fn(width) -> params2,
        data_fn(width) -> (data_x, data_y), loss_fn(params, data_x, data_y) -> scalar.
        """
        widths = np.array(width_range)
        barriers = np.zeros(len(widths))
        barrier_data = []

        for i, width in enumerate(widths):
            p1 = params1_fn(width)
            p2 = params2_fn(width)
            dx, dy = data_fn(width)
            result = self.linear_barrier(p1, p2, loss_fn, dx, dy)
            barriers[i] = result['barrier_height']
            barrier_data.append(result)

        # Fit power law: barrier ~ width^alpha
        log_w = np.log(widths[widths > 0])
        log_b = np.log(barriers[widths > 0] + 1e-15)
        if len(log_w) >= 2:
            coeffs = np.polyfit(log_w, log_b, 1)
            scaling_exponent = coeffs[0]
        else:
            scaling_exponent = float('nan')

        return {
            'widths': widths,
            'barriers': barriers,
            'scaling_exponent': float(scaling_exponent),
            'barrier_data': barrier_data,
        }


class TrajectoryAnalyzer:
    """Analyze loss landscape curvature along training trajectories."""

    def __init__(self):
        self._hessian_analyzer = HessianAnalyzer(param_dim=0)

    def curvature_along_trajectory(self, trajectory, loss_fn, data_x, data_y,
                                   use_trace=True, n_hutchinson=30):
        """Hessian trace (or top eigenvalue) at each point along trajectory."""
        trajectory = np.array(trajectory)
        n_steps = len(trajectory)
        curvatures = np.zeros(n_steps)

        for i in range(n_steps):
            params = trajectory[i]
            d = len(params)

            if use_trace:
                def hvp_fn(v, p=params):
                    return self._hessian_analyzer.hessian_vector_product(
                        p, v, loss_fn, data_x, data_y)
                curvatures[i] = self._hessian_analyzer.trace_estimation(hvp_fn, d, n_hutchinson)
            else:
                H = self._hessian_analyzer.compute_hessian(params, loss_fn, data_x, data_y)
                eigs = linalg.eigvalsh(H)
                curvatures[i] = np.max(eigs)

        return curvatures

    def sharpness_trajectory(self, trajectory, loss_fn, data_x, data_y):
        """Compute maximum eigenvalue of Hessian at each point of trajectory."""
        trajectory = np.array(trajectory)
        n_steps = len(trajectory)
        sharpness_vals = np.zeros(n_steps)

        for i in range(n_steps):
            params = trajectory[i]
            H = self._hessian_analyzer.compute_hessian(params, loss_fn, data_x, data_y)
            eigs = linalg.eigvalsh(H)
            sharpness_vals[i] = np.max(eigs)

        return sharpness_vals

    def edge_of_stability_detection(self, sharpness_trajectory, learning_rate):
        """Detect edge of stability: λ_max ≈ 2/η.

        Returns indices where sharpness crosses or oscillates near the threshold.
        """
        threshold = 2.0 / learning_rate
        sharpness = np.array(sharpness_trajectory)

        # Detect crossings of the threshold
        above = sharpness > threshold
        crossings = np.where(np.diff(above.astype(int)) != 0)[0]

        # Detect sustained oscillation near threshold
        near_threshold = np.abs(sharpness - threshold) < 0.1 * threshold
        oscillation_regions = []
        in_region = False
        start = 0
        for i in range(len(near_threshold)):
            if near_threshold[i] and not in_region:
                in_region = True
                start = i
            elif not near_threshold[i] and in_region:
                in_region = False
                if i - start >= 5:
                    oscillation_regions.append((start, i))
        if in_region and len(near_threshold) - start >= 5:
            oscillation_regions.append((start, len(near_threshold)))

        # Estimate onset: first time sharpness reaches threshold
        onset_idx = -1
        for i in range(len(sharpness)):
            if sharpness[i] >= threshold * 0.95:
                onset_idx = i
                break

        return {
            'threshold': float(threshold),
            'crossings': crossings.tolist(),
            'oscillation_regions': oscillation_regions,
            'onset_index': onset_idx,
            'mean_sharpness_post_onset': float(np.mean(sharpness[onset_idx:])) if onset_idx >= 0 else float('nan'),
            'at_edge': bool(len(oscillation_regions) > 0),
        }

    def progressive_sharpening(self, sharpness_trajectory, window=10):
        """Test if sharpness is monotonically increasing (progressive sharpening).

        Uses smoothed derivative to be robust to fluctuations.
        """
        sharpness = np.array(sharpness_trajectory)
        if len(sharpness) < 2 * window:
            return {
                'is_progressive': False,
                'increasing_fraction': 0.0,
                'smoothed_sharpness': sharpness,
            }

        # Smooth with moving average
        kernel = np.ones(window) / window
        smoothed = np.convolve(sharpness, kernel, mode='valid')

        diffs = np.diff(smoothed)
        increasing_fraction = float(np.mean(diffs > 0))

        return {
            'is_progressive': increasing_fraction > 0.8,
            'increasing_fraction': increasing_fraction,
            'smoothed_sharpness': smoothed,
            'derivative': diffs,
        }

    def gradient_norm_trajectory(self, trajectory, loss_fn, grad_fn, data_x, data_y):
        """||∇L|| at each point of the optimization trajectory."""
        trajectory = np.array(trajectory)
        n_steps = len(trajectory)
        norms = np.zeros(n_steps)

        for i in range(n_steps):
            params = trajectory[i]
            if grad_fn is not None:
                grad = grad_fn(params, data_x, data_y)
            else:
                d = len(params)
                grad = np.zeros(d)
                eps = 1e-5
                for k in range(d):
                    e_k = np.zeros(d)
                    e_k[k] = eps
                    grad[k] = (loss_fn(params + e_k, data_x, data_y) -
                                loss_fn(params - e_k, data_x, data_y)) / (2.0 * eps)
            norms[i] = np.linalg.norm(grad)

        return norms

    def loss_smoothness_estimation(self, trajectory, loss_values):
        """Estimate local Lipschitz constant of the gradient from trajectory data.

        L_local(t) ≈ |L(t+1) - L(t)| / ||θ(t+1) - θ(t)||
        """
        trajectory = np.array(trajectory)
        loss_values = np.array(loss_values)
        n = len(trajectory)
        if n < 2:
            return np.array([])

        smoothness = np.zeros(n - 1)
        for i in range(n - 1):
            delta_theta = np.linalg.norm(trajectory[i + 1] - trajectory[i])
            delta_loss = abs(loss_values[i + 1] - loss_values[i])
            if delta_theta > 1e-15:
                smoothness[i] = delta_loss / delta_theta
            else:
                smoothness[i] = 0.0

        return smoothness

    def trajectory_length(self, trajectory):
        """Total path length in parameter space: Σ ||θ(t+1) - θ(t)||."""
        trajectory = np.array(trajectory)
        if len(trajectory) < 2:
            return 0.0
        diffs = np.diff(trajectory, axis=0)
        step_lengths = np.linalg.norm(diffs, axis=1)
        return float(np.sum(step_lengths))

    def trajectory_efficiency(self, trajectory, loss_values):
        """Loss decrease per unit path length.

        Efficiency = (L_initial - L_final) / trajectory_length.
        """
        trajectory = np.array(trajectory)
        loss_values = np.array(loss_values)

        total_length = self.trajectory_length(trajectory)
        if total_length < 1e-15:
            return {
                'efficiency': 0.0,
                'total_loss_decrease': 0.0,
                'total_length': 0.0,
                'displacement': 0.0,
                'straightness': 0.0,
            }

        total_decrease = float(loss_values[0] - loss_values[-1])
        displacement = float(np.linalg.norm(trajectory[-1] - trajectory[0]))

        return {
            'efficiency': total_decrease / total_length,
            'total_loss_decrease': total_decrease,
            'total_length': total_length,
            'displacement': displacement,
            'straightness': displacement / total_length if total_length > 0 else 0.0,
        }
