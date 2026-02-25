"""Task complexity measurement for kernel learning theory.

Provides tools to quantify how hard a learning task is relative to a given
kernel / architecture, including smoothness estimation, RKHS norm computation,
curriculum learning analysis, and task-architecture compatibility scoring.
"""

import numpy as np
from scipy import linalg, spatial, stats, optimize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rbf_kernel(x1, x2, length_scale=1.0):
    """Default RBF kernel."""
    sq = spatial.distance.cdist(x1, x2, metric="sqeuclidean")
    return np.exp(-sq / (2.0 * length_scale ** 2))


def _safe_solve(K, y, reg=1e-10):
    """Solve (K + reg I) x = y robustly."""
    K_reg = K + reg * np.eye(K.shape[0])
    try:
        L = linalg.cholesky(K_reg, lower=True)
        return linalg.cho_solve((L, True), y)
    except linalg.LinAlgError:
        return linalg.lstsq(K_reg, y)[0]


def _eigen_decompose(K, top_k=None):
    """Return (eigenvalues, eigenvectors) sorted descending."""
    vals, vecs = linalg.eigh(K)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    if top_k is not None:
        vals, vecs = vals[:top_k], vecs[:, :top_k]
    return np.maximum(vals, 0.0), vecs


def _finite_differences_gradient(target_values, input_data, epsilon=1e-5):
    """Estimate gradient norms via finite differences on nearest neighbors."""
    n, d = input_data.shape
    tree = spatial.cKDTree(input_data)
    grad_norms = np.zeros(n)
    for i in range(n):
        dists, idxs = tree.query(input_data[i], k=min(d + 2, n))
        grad = np.zeros(d)
        for j in idxs[1:]:
            delta = input_data[j] - input_data[i]
            dist = np.linalg.norm(delta)
            if dist < epsilon:
                continue
            grad += (target_values[j] - target_values[i]) / (dist ** 2) * delta
        grad_norms[i] = np.linalg.norm(grad)
    return grad_norms


# ---------------------------------------------------------------------------
# 1. TargetSmoothnessEstimator
# ---------------------------------------------------------------------------

class TargetSmoothnessEstimator:
    """Estimate smoothness of a target function from samples."""

    def __init__(self, kernel_fn=None):
        self.kernel_fn = kernel_fn or _rbf_kernel

    # ---- core estimators --------------------------------------------------

    def sobolev_smoothness(self, target_values, input_data, order=1):
        """Estimate Sobolev smoothness index s of the target.

        Uses spectral decay of the empirical kernel matrix to infer the
        effective smoothness.  A function in H^s(R^d) has Fourier coefficients
        decaying as |k|^{-s-d/2}; we estimate s from the kernel-projected
        coefficient decay.

        Returns dict with 'smoothness_index', 'residuals', 'decay_rate'.
        """
        K = self.kernel_fn(input_data, input_data)
        eigenvalues, eigenvectors = _eigen_decompose(K)
        n = len(eigenvalues)

        # Project target onto eigenbasis
        coeffs = eigenvectors.T @ target_values
        coeffs_sq = coeffs ** 2

        # Fit log-linear decay: log|c_k|^2 ~ -2s * log(k)
        keep = eigenvalues > eigenvalues[0] * 1e-12
        k_idx = np.arange(1, keep.sum() + 1, dtype=float)
        log_k = np.log(k_idx)
        log_c = np.log(coeffs_sq[keep] + 1e-30)

        slope, intercept, r, _, _ = stats.linregress(log_k, log_c)
        d = input_data.shape[1]
        s_est = max(-slope / 2.0 - d / 4.0, 0.0) * order

        return {
            "smoothness_index": s_est,
            "decay_rate": -slope,
            "r_squared": r ** 2,
            "residuals": log_c - (slope * log_k + intercept),
        }

    def lipschitz_constant(self, target_values, input_data):
        """Estimate the Lipschitz constant L = max |f(x)-f(y)| / ||x-y||.

        Uses a random subset of pairs for scalability (O(n^2) otherwise).
        Returns dict with 'lipschitz', 'ratios_percentiles'.
        """
        n = len(target_values)
        max_pairs = min(n * (n - 1) // 2, 50000)

        if n <= 500:
            dists = spatial.distance.pdist(input_data)
            diffs = spatial.distance.pdist(target_values.reshape(-1, 1))
        else:
            rng = np.random.default_rng(42)
            idx_a = rng.integers(0, n, size=max_pairs)
            idx_b = rng.integers(0, n, size=max_pairs)
            mask = idx_a != idx_b
            idx_a, idx_b = idx_a[mask], idx_b[mask]
            dists = np.linalg.norm(input_data[idx_a] - input_data[idx_b], axis=1)
            diffs = np.abs(target_values[idx_a] - target_values[idx_b])

        nonzero = dists > 1e-15
        ratios = diffs[nonzero] / dists[nonzero]
        L = float(np.max(ratios)) if ratios.size > 0 else 0.0

        return {
            "lipschitz": L,
            "ratios_percentiles": {
                "50": float(np.percentile(ratios, 50)),
                "90": float(np.percentile(ratios, 90)),
                "99": float(np.percentile(ratios, 99)),
                "100": float(np.max(ratios)),
            },
        }

    def holder_exponent(self, target_values, input_data, n_pairs=1000):
        r"""Estimate Hölder exponent α: |f(x)-f(y)| ≤ C ||x-y||^α.

        Fits α via log-log regression on sampled pairs.
        """
        rng = np.random.default_rng(0)
        n = len(target_values)
        ia = rng.integers(0, n, size=n_pairs)
        ib = rng.integers(0, n, size=n_pairs)
        mask = ia != ib
        ia, ib = ia[mask], ib[mask]

        dists = np.linalg.norm(input_data[ia] - input_data[ib], axis=1)
        diffs = np.abs(target_values[ia] - target_values[ib])

        pos = (dists > 1e-15) & (diffs > 1e-15)
        log_d = np.log(dists[pos])
        log_f = np.log(diffs[pos])

        slope, intercept, r, _, _ = stats.linregress(log_d, log_f)
        alpha = np.clip(slope, 0.0, 2.0)

        return {
            "holder_exponent": float(alpha),
            "holder_constant": float(np.exp(intercept)),
            "r_squared": float(r ** 2),
        }

    def spectral_smoothness(self, target_values, kernel_eigenvectors,
                            kernel_eigenvalues):
        """Smoothness measured by how fast spectral coefficients decay.

        Returns effective smoothness β such that |<f, φ_k>|² ~ k^{-2β}.
        """
        coeffs = kernel_eigenvectors.T @ target_values
        coeffs_sq = coeffs ** 2
        keep = kernel_eigenvalues > kernel_eigenvalues[0] * 1e-12
        m = keep.sum()
        log_k = np.log(np.arange(1, m + 1, dtype=float))
        log_c = np.log(coeffs_sq[:m] + 1e-30)

        slope, intercept, r, _, _ = stats.linregress(log_k, log_c)
        beta = -slope / 2.0

        cumulative_energy = np.cumsum(coeffs_sq[:m]) / (np.sum(coeffs_sq[:m]) + 1e-30)
        effective_dim = int(np.searchsorted(cumulative_energy, 0.95)) + 1

        return {
            "spectral_smoothness": float(beta),
            "decay_slope": float(slope),
            "r_squared": float(r ** 2),
            "effective_dimension": effective_dim,
            "cumulative_energy": cumulative_energy,
        }

    def local_smoothness(self, target_values, input_data, neighborhoods):
        """Measure how smoothness varies across regions of input space.

        Parameters
        ----------
        neighborhoods : list of array-like
            Each element is an array of indices defining a local region.

        Returns dict mapping region index to local Lipschitz constant and
        an overall summary.
        """
        results = {}
        for idx, nbr in enumerate(neighborhoods):
            nbr = np.asarray(nbr)
            if len(nbr) < 2:
                results[idx] = {"lipschitz": 0.0, "variation": 0.0}
                continue
            local_x = input_data[nbr]
            local_y = target_values[nbr]
            dists = spatial.distance.pdist(local_x)
            diffs = spatial.distance.pdist(local_y.reshape(-1, 1))
            nonzero = dists > 1e-15
            if nonzero.any():
                ratios = diffs[nonzero] / dists[nonzero]
                results[idx] = {
                    "lipschitz": float(np.max(ratios)),
                    "variation": float(np.std(ratios)),
                }
            else:
                results[idx] = {"lipschitz": 0.0, "variation": 0.0}

        lips = [r["lipschitz"] for r in results.values()]
        return {
            "local_results": results,
            "mean_lipschitz": float(np.mean(lips)),
            "std_lipschitz": float(np.std(lips)),
            "max_lipschitz": float(np.max(lips)),
            "smoothness_uniformity": 1.0 - float(np.std(lips)) / (float(np.mean(lips)) + 1e-15),
        }

    def gradient_norm_distribution(self, target_values, input_data,
                                   epsilon=1e-5):
        """Distribution of ||∇f|| estimated via finite differences."""
        grad_norms = _finite_differences_gradient(target_values, input_data,
                                                  epsilon)
        return {
            "mean": float(np.mean(grad_norms)),
            "std": float(np.std(grad_norms)),
            "median": float(np.median(grad_norms)),
            "max": float(np.max(grad_norms)),
            "percentiles": {
                q: float(np.percentile(grad_norms, q))
                for q in [10, 25, 50, 75, 90, 95, 99]
            },
            "gradient_norms": grad_norms,
        }

    def smoothness_vs_input_dimension(self, target_fn, dim_range,
                                      n_samples=500):
        """How smoothness scales with input dimension d.

        Parameters
        ----------
        target_fn : callable(X) -> y
            Function to evaluate.
        dim_range : iterable of int
            Dimensions to probe.
        """
        rng = np.random.default_rng(1)
        results = {}
        for d in dim_range:
            X = rng.standard_normal((n_samples, d))
            y = target_fn(X)
            sob = self.sobolev_smoothness(y, X)
            lip = self.lipschitz_constant(y, X)
            results[d] = {
                "sobolev_index": sob["smoothness_index"],
                "lipschitz": lip["lipschitz"],
            }
        dims = sorted(results.keys())
        sob_vals = [results[d]["sobolev_index"] for d in dims]
        lip_vals = [results[d]["lipschitz"] for d in dims]
        log_d = np.log(np.array(dims, dtype=float))

        sob_slope = 0.0
        if len(dims) > 1:
            sob_slope = float(stats.linregress(log_d, np.log(np.array(sob_vals) + 1e-30)).slope)

        return {
            "per_dim": results,
            "sobolev_scaling_exponent": sob_slope,
            "dimensions": dims,
            "sobolev_values": sob_vals,
            "lipschitz_values": lip_vals,
        }

    def effective_smoothness(self, target_coefficients, eigenvalues):
        r"""Effective Sobolev index s* = argmin_s Σ c_k² λ_k^{-s} < ∞.

        Uses bisection to find the largest s for which the sum converges.
        """
        coeffs_sq = np.asarray(target_coefficients) ** 2
        eigs = np.maximum(np.asarray(eigenvalues), 1e-30)

        def _sum_at_s(s):
            return np.sum(coeffs_sq * eigs ** (-s))

        lo, hi = 0.0, 10.0
        # Expand hi if needed
        while _sum_at_s(hi) < 1e15 and hi < 100:
            hi *= 2

        # Bisect to find critical s
        for _ in range(64):
            mid = (lo + hi) / 2.0
            if np.isfinite(_sum_at_s(mid)) and _sum_at_s(mid) < 1e12:
                lo = mid
            else:
                hi = mid

        return {
            "effective_smoothness": float(lo),
            "sum_at_s": float(_sum_at_s(lo)),
        }

    def smoothness_comparison(self, targets_dict, input_data):
        """Compare smoothness across multiple target functions.

        Parameters
        ----------
        targets_dict : dict[str, array]
            Mapping name -> target values.
        """
        results = {}
        for name, y in targets_dict.items():
            sob = self.sobolev_smoothness(y, input_data)
            lip = self.lipschitz_constant(y, input_data)
            hol = self.holder_exponent(y, input_data)
            results[name] = {
                "sobolev_index": sob["smoothness_index"],
                "lipschitz": lip["lipschitz"],
                "holder_exponent": hol["holder_exponent"],
            }
        # Rank by difficulty (higher Lipschitz = harder)
        ranked = sorted(results.items(),
                        key=lambda kv: kv[1]["lipschitz"], reverse=True)
        return {
            "per_target": results,
            "difficulty_ranking": [name for name, _ in ranked],
        }

    def anisotropic_smoothness(self, target_values, input_data):
        """Directional smoothness analysis along principal axes.

        Returns per-axis Lipschitz estimates after PCA rotation.
        """
        n, d = input_data.shape
        mean = input_data.mean(axis=0)
        X_c = input_data - mean
        _, _, Vt = linalg.svd(X_c, full_matrices=False)

        X_rot = X_c @ Vt.T  # rotated to principal axes

        per_axis = {}
        for j in range(d):
            order = np.argsort(X_rot[:, j])
            sorted_y = target_values[order]
            sorted_x = X_rot[order, j]
            diffs_y = np.abs(np.diff(sorted_y))
            diffs_x = np.abs(np.diff(sorted_x))
            nonzero = diffs_x > 1e-15
            if nonzero.any():
                ratios = diffs_y[nonzero] / diffs_x[nonzero]
                per_axis[j] = {
                    "lipschitz": float(np.max(ratios)),
                    "median_ratio": float(np.median(ratios)),
                }
            else:
                per_axis[j] = {"lipschitz": 0.0, "median_ratio": 0.0}

        lips = [v["lipschitz"] for v in per_axis.values()]
        anisotropy = float(np.max(lips)) / (float(np.min(lips)) + 1e-15)

        return {
            "per_axis": per_axis,
            "anisotropy_ratio": anisotropy,
            "principal_directions": Vt,
        }


# ---------------------------------------------------------------------------
# 2. RKHSNormComputer
# ---------------------------------------------------------------------------

class RKHSNormComputer:
    """Compute RKHS norm of a target function given kernel information."""

    def __init__(self, kernel_fn=None, regularization=1e-10):
        self.kernel_fn = kernel_fn or _rbf_kernel
        self.regularization = regularization

    def compute_rkhs_norm(self, target_values, kernel_matrix):
        """||f||_K² = y^T K^{-1} y."""
        alpha = _safe_solve(kernel_matrix, target_values, reg=0.0)
        norm_sq = float(target_values @ alpha)
        return {"rkhs_norm_squared": max(norm_sq, 0.0),
                "rkhs_norm": np.sqrt(max(norm_sq, 0.0))}

    def regularized_rkhs_norm(self, target_values, kernel_matrix, reg):
        """||f||_K² with Tikhonov regularization."""
        alpha = _safe_solve(kernel_matrix, target_values, reg=reg)
        norm_sq = float(target_values @ alpha)
        residual = np.linalg.norm(kernel_matrix @ alpha - target_values)
        return {
            "rkhs_norm_squared": max(norm_sq, 0.0),
            "rkhs_norm": np.sqrt(max(norm_sq, 0.0)),
            "regularization": reg,
            "residual_norm": float(residual),
        }

    def rkhs_norm_vs_kernel(self, target_values, kernel_matrices, kernel_names):
        """Compare RKHS norms across different kernels."""
        results = {}
        for name, K in zip(kernel_names, kernel_matrices):
            res = self.regularized_rkhs_norm(target_values, K,
                                             self.regularization)
            results[name] = res
        ranked = sorted(results.items(),
                        key=lambda kv: kv[1]["rkhs_norm"])
        return {
            "per_kernel": results,
            "best_kernel": ranked[0][0],
            "ranking": [name for name, _ in ranked],
        }

    def rkhs_norm_decomposition(self, target_values, eigenvalues, eigenvectors):
        """Per-eigenmode contribution to ||f||_K².

        ||f||_K² = Σ_k c_k² / λ_k  where c_k = <f, φ_k>.
        """
        coeffs = eigenvectors.T @ target_values
        eigs = np.maximum(eigenvalues, 1e-30)
        per_mode = coeffs ** 2 / eigs
        cumulative = np.cumsum(per_mode)
        total = cumulative[-1] if len(cumulative) else 0.0

        return {
            "per_mode_contribution": per_mode,
            "cumulative_norm": cumulative,
            "total_rkhs_norm_squared": float(total),
            "coefficients": coeffs,
            "dominant_mode": int(np.argmax(per_mode)),
        }

    def is_in_rkhs(self, target_values, kernel_matrix, threshold=None):
        """Check whether f lies in H_K by examining convergence of ||f||_K².

        If threshold is None, use adaptive threshold based on eigenvalue decay.
        """
        eigenvalues, eigenvectors = _eigen_decompose(kernel_matrix)
        coeffs = eigenvectors.T @ target_values
        eigs = np.maximum(eigenvalues, 1e-30)
        per_mode = coeffs ** 2 / eigs

        partial_sums = np.cumsum(per_mode)
        # Check if partial sums stabilize
        if len(partial_sums) < 5:
            converged = True
        else:
            tail = partial_sums[-5:]
            relative_growth = np.diff(tail) / (tail[:-1] + 1e-30)
            converged = bool(np.all(np.abs(relative_growth) < 0.01))

        if threshold is None:
            threshold = 10.0 * np.trace(kernel_matrix) / len(target_values)

        total = float(partial_sums[-1])
        in_rkhs = converged and total < threshold

        return {
            "in_rkhs": in_rkhs,
            "rkhs_norm_squared": total,
            "converged": converged,
            "threshold": float(threshold),
            "partial_sums": partial_sums,
        }

    def source_condition(self, target_coefficients, eigenvalues, beta_range):
        r"""Find source condition exponent β: f = L_K^β g, ||g|| bounded.

        Checks Σ c_k² / λ_k^{2β} < ∞ for each β ∈ beta_range.
        """
        coeffs_sq = np.asarray(target_coefficients) ** 2
        eigs = np.maximum(np.asarray(eigenvalues), 1e-30)

        sums = {}
        for beta in beta_range:
            s = float(np.sum(coeffs_sq / eigs ** (2 * beta)))
            sums[float(beta)] = s

        # Find largest β with finite sum
        finite_betas = [b for b, s in sums.items()
                        if np.isfinite(s) and s < 1e15]
        best_beta = max(finite_betas) if finite_betas else 0.0

        return {
            "source_sums": sums,
            "best_beta": best_beta,
            "finite_betas": finite_betas,
        }

    def approximation_error_vs_n(self, target_values, kernel_matrix, n_range):
        """How approximation error decreases as training set size grows.

        For each n, use the first n data points and measure reconstruction
        error on the remaining points.
        """
        N = len(target_values)
        results = {}
        for n in n_range:
            if n >= N:
                continue
            K_train = kernel_matrix[:n, :n]
            K_test_train = kernel_matrix[n:, :n]
            y_train = target_values[:n]
            y_test = target_values[n:]

            alpha = _safe_solve(K_train, y_train, reg=self.regularization)
            y_pred = K_test_train @ alpha
            mse = float(np.mean((y_test - y_pred) ** 2))
            results[n] = mse

        # Fit power law: error ~ n^{-rate}
        ns = np.array(sorted(results.keys()), dtype=float)
        errs = np.array([results[int(ni)] for ni in ns])
        pos = errs > 0
        rate = 0.0
        if pos.sum() > 2:
            slope, _, _, _, _ = stats.linregress(np.log(ns[pos]),
                                                 np.log(errs[pos]))
            rate = -slope

        return {
            "errors": results,
            "convergence_rate": float(rate),
        }

    def minimax_optimal_rate(self, eigenvalues, smoothness_index):
        r"""Minimax optimal learning rate n^{-2s/(2s+1)} for Sobolev-s.

        Also computes effective sample size from eigenvalue decay.
        """
        s = smoothness_index
        rate_exponent = 2 * s / (2 * s + 1) if s > 0 else 0.0

        eigs = np.asarray(eigenvalues)
        effective_n = float(np.sum(eigs) ** 2 / (np.sum(eigs ** 2) + 1e-30))

        return {
            "rate_exponent": rate_exponent,
            "rate_formula": f"n^{{-{rate_exponent:.4f}}}",
            "effective_sample_size": effective_n,
            "smoothness_used": s,
        }

    def rkhs_norm_as_complexity(self, target_values, kernel_matrices):
        """Task complexity as minimum RKHS norm over a set of kernels."""
        norms = []
        for K in kernel_matrices:
            res = self.regularized_rkhs_norm(target_values, K,
                                             self.regularization)
            norms.append(res["rkhs_norm"])
        best_idx = int(np.argmin(norms))
        return {
            "complexity": float(norms[best_idx]),
            "all_norms": [float(v) for v in norms],
            "best_kernel_index": best_idx,
        }

    def effective_rkhs_dimension(self, target_coefficients, eigenvalues,
                                 threshold=0.99):
        """Number of eigenmodes needed to capture `threshold` of ||f||_K²."""
        coeffs_sq = np.asarray(target_coefficients) ** 2
        eigs = np.maximum(np.asarray(eigenvalues), 1e-30)
        per_mode = coeffs_sq / eigs
        total = np.sum(per_mode)
        if total < 1e-30:
            return {"effective_dimension": 0, "threshold": threshold,
                    "total_norm_squared": 0.0}

        cumulative = np.cumsum(per_mode) / total
        dim = int(np.searchsorted(cumulative, threshold)) + 1

        return {
            "effective_dimension": dim,
            "threshold": threshold,
            "total_norm_squared": float(total),
            "cumulative_fraction": cumulative,
        }


# ---------------------------------------------------------------------------
# 3. CurriculumLearningAnalyzer
# ---------------------------------------------------------------------------

class CurriculumLearningAnalyzer:
    """Analyse curriculum-learning implications from task complexity."""

    def __init__(self, kernel_fn=None):
        self.kernel_fn = kernel_fn or _rbf_kernel

    def sort_by_difficulty(self, samples_x, samples_y, kernel_matrix):
        """Rank samples by learning difficulty (leverage score + residual).

        Difficulty = leverage_i * |y_i - ŷ_{-i}| (LOO residual weighted).
        """
        n = len(samples_y)
        K_reg = kernel_matrix + 1e-8 * np.eye(n)
        try:
            K_inv = linalg.inv(K_reg)
        except linalg.LinAlgError:
            K_inv = linalg.pinv(K_reg)

        # LOO residual: (K^{-1} y)_i / (K^{-1})_{ii}
        alpha = K_inv @ samples_y
        diag_inv = np.diag(K_inv)
        diag_inv = np.where(np.abs(diag_inv) < 1e-15, 1e-15, diag_inv)
        loo_residuals = np.abs(alpha / diag_inv)

        # Leverage scores
        hat = kernel_matrix @ K_inv
        leverage = np.diag(hat)

        difficulty = leverage * loo_residuals
        order = np.argsort(difficulty)

        return {
            "order_easy_to_hard": order,
            "difficulties": difficulty,
            "loo_residuals": loo_residuals,
            "leverage_scores": leverage,
        }

    def difficulty_score(self, sample_x, sample_y, kernel_matrix,
                         training_data_x, training_data_y):
        """Difficulty of a single sample relative to a training set.

        Uses kernel ridge regression prediction error as difficulty proxy.
        """
        sample_x = np.atleast_2d(sample_x)
        n = len(training_data_y)
        K_train = kernel_matrix[:n, :n]
        k_star = self.kernel_fn(sample_x, training_data_x).ravel()

        alpha = _safe_solve(K_train, training_data_y)
        y_pred = float(k_star @ alpha)
        error = abs(sample_y - y_pred)

        # Posterior variance (GP interpretation)
        v = _safe_solve(K_train, k_star)
        k_ss = float(self.kernel_fn(sample_x, sample_x))
        variance = max(k_ss - float(k_star @ v), 0.0)

        return {
            "difficulty": float(error),
            "prediction": y_pred,
            "posterior_variance": variance,
            "combined_score": float(error + np.sqrt(variance)),
        }

    def curriculum_schedule(self, difficulties, n_stages=5):
        """Design an easy-to-hard curriculum with `n_stages` stages."""
        difficulties = np.asarray(difficulties)
        order = np.argsort(difficulties)
        n = len(difficulties)
        stage_size = max(1, n // n_stages)

        stages = []
        for s in range(n_stages):
            start = s * stage_size
            end = n if s == n_stages - 1 else (s + 1) * stage_size
            indices = order[start:end]
            stages.append({
                "stage": s,
                "indices": indices,
                "difficulty_range": (float(difficulties[indices[0]]),
                                     float(difficulties[indices[-1]])),
                "size": len(indices),
            })

        return {"stages": stages, "n_stages": n_stages, "total_samples": n}

    def anti_curriculum_schedule(self, difficulties, n_stages=5):
        """Hard-to-easy curriculum (anti-curriculum)."""
        difficulties = np.asarray(difficulties)
        order = np.argsort(difficulties)[::-1]  # hard first
        n = len(difficulties)
        stage_size = max(1, n // n_stages)

        stages = []
        for s in range(n_stages):
            start = s * stage_size
            end = n if s == n_stages - 1 else (s + 1) * stage_size
            indices = order[start:end]
            stages.append({
                "stage": s,
                "indices": indices,
                "difficulty_range": (float(difficulties[indices[-1]]),
                                     float(difficulties[indices[0]])),
                "size": len(indices),
            })
        return {"stages": stages, "n_stages": n_stages, "total_samples": n}

    def self_paced_threshold(self, losses, epoch, pace_fn=None):
        """Compute self-paced learning threshold for given epoch.

        Parameters
        ----------
        pace_fn : callable(epoch) -> float, optional
            Returns the loss threshold at the given epoch.  Defaults to a
            linearly increasing schedule.
        """
        losses = np.asarray(losses)
        if pace_fn is None:
            max_loss = float(np.max(losses))
            pace_fn = lambda t: max_loss * min(1.0, (t + 1) / 50.0)

        threshold = pace_fn(epoch)
        selected = np.where(losses <= threshold)[0]

        return {
            "threshold": float(threshold),
            "selected_indices": selected,
            "n_selected": len(selected),
            "fraction_selected": len(selected) / max(len(losses), 1),
        }

    def curriculum_effect_prediction(self, difficulties, kernel_eigenvalues):
        """Predict whether curriculum will help based on spectral alignment.

        If difficult samples align with small eigenvalues, curriculum is
        expected to help because it avoids early contamination of the
        well-learned subspace.
        """
        difficulties = np.asarray(difficulties)
        eigs = np.asarray(kernel_eigenvalues)
        m = min(len(difficulties), len(eigs))
        difficulties_sorted = np.sort(difficulties)[:m]
        eigs_sorted = np.sort(eigs)[::-1][:m]

        # Correlation between difficulty rank and eigenvalue rank
        corr, p_val = stats.spearmanr(difficulties_sorted, eigs_sorted)

        # Effective curriculum gain: fraction of energy in easy samples
        easy_half = difficulties_sorted[:m // 2]
        total_energy = np.sum(eigs_sorted)
        easy_energy = np.sum(eigs_sorted[:m // 2])
        energy_ratio = easy_energy / (total_energy + 1e-30)

        predicted_benefit = energy_ratio * (1 - abs(corr))

        return {
            "difficulty_eigenvalue_correlation": float(corr),
            "p_value": float(p_val),
            "energy_in_easy_half": float(energy_ratio),
            "predicted_benefit": float(predicted_benefit),
            "recommendation": "curriculum" if predicted_benefit > 0.3 else "random",
        }

    def optimal_data_ordering(self, kernel_matrix, targets, method="greedy"):
        """Find an ordering that greedily minimises cumulative LOO error.

        'greedy' selects the next sample that maximally reduces prediction
        error on the remaining samples.
        """
        n = len(targets)
        remaining = list(range(n))
        order = []
        K_reg = kernel_matrix + 1e-8 * np.eye(n)

        if method == "greedy":
            for _ in range(n):
                best_idx = None
                best_score = np.inf
                for r in remaining:
                    subset = order + [r]
                    K_sub = K_reg[np.ix_(subset, subset)]
                    y_sub = targets[np.array(subset)]
                    alpha = _safe_solve(K_sub, y_sub)
                    # Score: prediction error on remaining
                    rem = [x for x in remaining if x != r]
                    if not rem:
                        score = 0.0
                    else:
                        k_rem = kernel_matrix[np.array(rem)][:, np.array(subset)]
                        y_pred = k_rem @ alpha
                        score = float(np.mean((targets[np.array(rem)] - y_pred) ** 2))
                    if score < best_score:
                        best_score = score
                        best_idx = r
                order.append(best_idx)
                remaining.remove(best_idx)
                # Limit greedy search for large n
                if len(order) > 50:
                    order.extend(remaining)
                    break
        else:
            order = list(range(n))

        return {"order": np.array(order), "method": method}

    def curriculum_speedup(self, curriculum_losses, random_losses):
        """Compute speedup factor of curriculum over random ordering.

        Speedup = area under random curve / area under curriculum curve.
        """
        curriculum_losses = np.asarray(curriculum_losses, dtype=float)
        random_losses = np.asarray(random_losses, dtype=float)
        m = min(len(curriculum_losses), len(random_losses))
        auc_curr = float(np.trapz(curriculum_losses[:m]))
        auc_rand = float(np.trapz(random_losses[:m]))

        speedup = auc_rand / (auc_curr + 1e-30)

        # Epochs to reach threshold (e.g. 10% of initial loss)
        threshold = random_losses[0] * 0.1
        curr_epoch = np.searchsorted(-curriculum_losses[:m], -threshold)
        rand_epoch = np.searchsorted(-random_losses[:m], -threshold)

        return {
            "speedup_factor": float(speedup),
            "auc_curriculum": auc_curr,
            "auc_random": auc_rand,
            "epochs_to_threshold_curriculum": int(curr_epoch),
            "epochs_to_threshold_random": int(rand_epoch),
        }

    def competence_function(self, t, c0=0.01, p=2):
        """Competence-based curriculum: c(t) = min(1, c0 + (1-c0)*(t/T)^p).

        Here T is normalised so t ∈ [0, 1].
        """
        t = np.asarray(t, dtype=float)
        c = np.minimum(1.0, c0 + (1.0 - c0) * t ** p)
        return {"competence": c, "c0": c0, "p": p}

    def difficulty_distribution(self, difficulties, n_bins=50):
        """Histogram / distribution statistics of sample difficulties."""
        difficulties = np.asarray(difficulties)
        hist, bin_edges = np.histogram(difficulties, bins=n_bins, density=True)
        skew = float(stats.skew(difficulties))
        kurt = float(stats.kurtosis(difficulties))

        return {
            "histogram": hist,
            "bin_edges": bin_edges,
            "mean": float(np.mean(difficulties)),
            "std": float(np.std(difficulties)),
            "skewness": skew,
            "kurtosis": kurt,
            "min": float(np.min(difficulties)),
            "max": float(np.max(difficulties)),
        }


# ---------------------------------------------------------------------------
# 4. TaskArchitectureCompatibility
# ---------------------------------------------------------------------------

class TaskArchitectureCompatibility:
    """Score how well an architecture (kernel) matches a target task."""

    def __init__(self):
        pass

    def compatibility_score(self, kernel, target_values, n_train):
        """Overall compatibility = statistical + spectral, normalised to [0,1].

        Parameters
        ----------
        kernel : ndarray (n, n)
            Kernel matrix evaluated on training + test data.
        """
        eigenvalues, eigenvectors = _eigen_decompose(kernel)
        coeffs = eigenvectors.T @ target_values

        spec = self.spectral_compatibility(eigenvalues, coeffs)
        stat = self.statistical_compatibility(kernel, n_train,
                                              float(np.sqrt(np.sum(coeffs ** 2 / (eigenvalues + 1e-30)))))
        expr = self.kernel_expressivity(eigenvalues)

        combined = (0.4 * spec["spectral_score"]
                    + 0.3 * stat["statistical_score"]
                    + 0.3 * expr["expressivity_score"])

        return {
            "compatibility": float(np.clip(combined, 0, 1)),
            "spectral": spec,
            "statistical": stat,
            "expressivity": expr,
        }

    def spectral_compatibility(self, kernel_eigenvalues, target_coefficients):
        """How well target spectrum aligns with kernel spectrum.

        Score is high when large target coefficients align with large eigenvalues.
        """
        eigs = np.asarray(kernel_eigenvalues)
        coeffs = np.asarray(target_coefficients)
        m = min(len(eigs), len(coeffs))
        eigs, coeffs = eigs[:m], coeffs[:m]

        # Normalised alignment
        eig_norm = eigs / (np.sum(eigs) + 1e-30)
        coeff_energy = coeffs ** 2
        coeff_norm = coeff_energy / (np.sum(coeff_energy) + 1e-30)

        alignment = float(np.sum(np.sqrt(eig_norm * coeff_norm)))  # Bhattacharyya
        # Cumulative energy captured by top-k eigenvalues
        cum_eig = np.cumsum(eig_norm)
        cum_coeff = np.cumsum(coeff_norm)
        auc_diff = float(np.mean(np.abs(cum_eig - cum_coeff)))

        score = alignment * (1.0 - auc_diff)

        return {
            "spectral_score": float(np.clip(score, 0, 1)),
            "bhattacharyya_alignment": alignment,
            "cumulative_gap": auc_diff,
        }

    def depth_compatibility(self, target_smoothness, depth_range, width):
        """Optimal depth for a target with given smoothness.

        Deeper networks approximate smoother functions more efficiently but
        waste capacity on rough functions.
        """
        results = {}
        for depth in depth_range:
            # Approximation rate: deeper = higher-order polynomial approx
            approx_power = min(depth * target_smoothness, 20.0)
            # Effective capacity ~ width * depth
            capacity = width * depth
            # Generalisation cost grows with capacity
            gen_cost = np.sqrt(capacity / 1000.0)
            # Net score: approx power minus generalisation cost
            score = approx_power / (gen_cost + 1e-5)
            results[depth] = {
                "score": float(score),
                "approx_power": float(approx_power),
                "gen_cost": float(gen_cost),
            }

        best_depth = max(results, key=lambda d: results[d]["score"])
        return {
            "per_depth": results,
            "optimal_depth": best_depth,
            "target_smoothness": target_smoothness,
        }

    def width_compatibility(self, target_complexity, width_range, depth):
        """Optimal width for a target with given complexity (RKHS norm)."""
        results = {}
        for width in width_range:
            capacity = width * depth
            # Approximation error ~ complexity / sqrt(width)
            approx_err = target_complexity / (np.sqrt(width) + 1e-5)
            # Estimation error ~ sqrt(capacity / n), approximate with capacity
            est_err = np.sqrt(capacity) / 100.0
            total_err = approx_err + est_err
            results[width] = {
                "total_error": float(total_err),
                "approx_error": float(approx_err),
                "estimation_error": float(est_err),
            }

        best_width = min(results, key=lambda w: results[w]["total_error"])
        return {
            "per_width": results,
            "optimal_width": best_width,
            "target_complexity": target_complexity,
        }

    def architecture_recommendation(self, target_properties, arch_candidates):
        """Rank architecture candidates for a given target.

        Parameters
        ----------
        target_properties : dict
            Must contain 'smoothness', 'rkhs_norm', 'eigenvalues', 'coefficients'.
        arch_candidates : list of dict
            Each dict has 'name', 'kernel_eigenvalues', 'depth', 'width'.
        """
        scores = []
        for arch in arch_candidates:
            spec = self.spectral_compatibility(
                arch["kernel_eigenvalues"],
                target_properties["coefficients"],
            )
            depth_res = self.depth_compatibility(
                target_properties["smoothness"],
                [arch["depth"]], arch["width"],
            )
            width_res = self.width_compatibility(
                target_properties["rkhs_norm"],
                [arch["width"]], arch["depth"],
            )
            score = (0.5 * spec["spectral_score"]
                     + 0.25 * depth_res["per_depth"][arch["depth"]]["score"] / 100.0
                     + 0.25 * (1.0 / (width_res["per_width"][arch["width"]]["total_error"] + 1e-5)))
            score = float(np.clip(score, 0, 1))
            scores.append({
                "name": arch["name"],
                "score": score,
                "spectral": spec,
            })

        scores.sort(key=lambda x: x["score"], reverse=True)
        return {"ranking": scores, "best": scores[0]["name"] if scores else None}

    def kernel_expressivity(self, kernel_eigenvalues):
        """Expressivity of a kernel RKHS from eigenvalue spectrum.

        Higher effective dimensionality => more expressive.
        """
        eigs = np.maximum(np.asarray(kernel_eigenvalues), 0.0)
        total = np.sum(eigs) + 1e-30
        p = eigs / total

        # Effective dimension (participation ratio)
        eff_dim = total ** 2 / (np.sum(eigs ** 2) + 1e-30)
        # Spectral entropy
        p_pos = p[p > 1e-30]
        entropy = -float(np.sum(p_pos * np.log(p_pos)))
        max_entropy = np.log(len(eigs)) if len(eigs) > 0 else 1.0

        score = float(entropy / (max_entropy + 1e-30))

        return {
            "expressivity_score": np.clip(score, 0, 1),
            "effective_dimension": float(eff_dim),
            "spectral_entropy": entropy,
            "max_entropy": float(max_entropy),
        }

    def approximation_theoretic_score(self, kernel, target_fn, input_data):
        """Approximation-theory score: how well the kernel basis approximates
        the target as a function of basis size.
        """
        y = target_fn(input_data) if callable(target_fn) else np.asarray(target_fn)
        K = kernel(input_data, input_data) if callable(kernel) else np.asarray(kernel)
        eigenvalues, eigenvectors = _eigen_decompose(K)

        coeffs = eigenvectors.T @ y
        total_energy = float(np.sum(coeffs ** 2))

        fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        cum = np.cumsum(coeffs ** 2) / (total_energy + 1e-30)
        dims_needed = {}
        for frac in fractions:
            idx = int(np.searchsorted(cum, frac)) + 1
            dims_needed[frac] = idx

        # Score: fewer dimensions needed => better approximation
        n = len(y)
        score = 1.0 - dims_needed[0.95] / n

        return {
            "score": float(np.clip(score, 0, 1)),
            "dims_needed": dims_needed,
            "total_energy": total_energy,
        }

    def statistical_compatibility(self, kernel, n_train, target_rkhs_norm):
        """Statistical learning bound: excess risk ~ ||f||_K² * N(λ) / n.

        N(λ) = Σ λ_k/(λ_k+λ) is the effective dimensionality at
        regularization level λ ~ 1/n.
        """
        if callable(kernel):
            raise ValueError("Provide kernel matrix, not function")
        K = np.asarray(kernel)
        eigenvalues = np.maximum(linalg.eigvalsh(K), 0.0)[::-1]

        lam = 1.0 / n_train
        effective_dim = float(np.sum(eigenvalues / (eigenvalues + lam)))
        excess_risk = target_rkhs_norm ** 2 * effective_dim / n_train

        # Normalise to [0,1] score (lower risk = higher score)
        score = 1.0 / (1.0 + excess_risk)

        return {
            "statistical_score": float(score),
            "excess_risk_bound": float(excess_risk),
            "effective_dimension": effective_dim,
            "regularization": lam,
        }

    def computational_compatibility(self, kernel_cost, target_accuracy,
                                    compute_budget):
        """Computational efficiency: can we reach target accuracy within budget?

        Parameters
        ----------
        kernel_cost : float
            Cost of one kernel evaluation (seconds).
        target_accuracy : float
            Desired accuracy (e.g. MSE).
        compute_budget : float
            Total compute budget in seconds.
        """
        max_evals = compute_budget / (kernel_cost + 1e-30)
        # GP training is O(n^3); maximum n we can afford
        max_n = int(max_evals ** (1.0 / 3.0))

        # Rough estimate: error ~ 1/sqrt(n)
        estimated_accuracy = 1.0 / (np.sqrt(max_n) + 1e-5)
        feasible = estimated_accuracy <= target_accuracy

        score = min(1.0, target_accuracy / (estimated_accuracy + 1e-30))

        return {
            "computational_score": float(score),
            "feasible": bool(feasible),
            "max_training_size": max_n,
            "estimated_accuracy": float(estimated_accuracy),
            "target_accuracy": target_accuracy,
        }

    def compatibility_matrix(self, kernels, targets):
        """Build a (n_kernels × n_targets) compatibility matrix.

        Parameters
        ----------
        kernels : list of ndarray
            Kernel matrices.
        targets : list of ndarray
            Target value vectors.

        Returns n_kernels × n_targets matrix of compatibility scores.
        """
        n_k = len(kernels)
        n_t = len(targets)
        matrix = np.zeros((n_k, n_t))

        for i, K in enumerate(kernels):
            eigenvalues, eigenvectors = _eigen_decompose(K)
            for j, y in enumerate(targets):
                coeffs = eigenvectors.T @ y
                spec = self.spectral_compatibility(eigenvalues, coeffs)
                matrix[i, j] = spec["spectral_score"]

        # Best kernel per target
        best_kernels = np.argmax(matrix, axis=0)
        # Best target per kernel
        best_targets = np.argmax(matrix, axis=1)

        return {
            "matrix": matrix,
            "best_kernel_per_target": best_kernels,
            "best_target_per_kernel": best_targets,
        }
