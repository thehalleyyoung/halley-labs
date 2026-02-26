"""
GP-based surrogate for adaptive phase-boundary exploration.

The GP surrogate is strictly ADVISORY — it guides refinement priority
but never contaminates certified guarantees. All regime claims are
independently verified by the tiered certification pipeline.

Kernel: Matérn-5/2 (twice differentiable, appropriate for bifurcation surfaces).
Acquisition: Expected Improvement (EI) for boundary-focused exploration.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class GPPrediction:
    """GP posterior prediction at a query point."""
    mean: float
    variance: float
    std: float

    @property
    def lower_95(self) -> float:
        return self.mean - 1.96 * self.std

    @property
    def upper_95(self) -> float:
        return self.mean + 1.96 * self.std


def matern52_kernel(x1: np.ndarray, x2: np.ndarray,
                    length_scale: float, signal_var: float) -> float:
    """Matérn-5/2 kernel: k(r) = σ² (1 + √5r/l + 5r²/3l²) exp(-√5r/l)."""
    r = np.linalg.norm(x1 - x2)
    s = np.sqrt(5.0) * r / length_scale
    return signal_var * (1.0 + s + s * s / 3.0) * np.exp(-s)


def matern52_kernel_ard(x1: np.ndarray, x2: np.ndarray,
                        length_scales: np.ndarray, signal_var: float) -> float:
    """
    ARD Matérn-5/2 kernel with per-dimension length scales.
    
    k(x1, x2) = σ² (1 + √5·r + 5r²/3) exp(-√5·r)
    where r = √(Σ_d ((x1_d - x2_d) / l_d)²)
    
    ARD (Automatic Relevance Determination) allows the kernel to learn
    which parameter dimensions are most relevant for regime classification.
    """
    diff = (x1 - x2) / length_scales
    r = np.linalg.norm(diff)
    s = np.sqrt(5.0) * r
    return signal_var * (1.0 + s + s * s / 3.0) * np.exp(-s)


def matern52_kernel_matrix(X: np.ndarray, length_scale: float,
                           signal_var: float) -> np.ndarray:
    """Compute kernel matrix K[i,j] = k(X[i], X[j])."""
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i, j] = matern52_kernel(X[i], X[j], length_scale, signal_var)
            K[j, i] = K[i, j]
    return K


def matern52_kernel_matrix_ard(X: np.ndarray, length_scales: np.ndarray,
                                signal_var: float) -> np.ndarray:
    """Compute ARD kernel matrix K[i,j] = k_ARD(X[i], X[j])."""
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i, j] = matern52_kernel_ard(X[i], X[j], length_scales, signal_var)
            K[j, i] = K[i, j]
    return K


class GPSurrogate:
    """
    Gaussian process surrogate for regime-boundary prediction.

    Trains on (parameter_midpoint, regime_label_int) pairs from certified cells.
    Predicts regime probability at uncertified parameter points to prioritize
    refinement near predicted phase boundaries.

    Kernel: Matérn-5/2 with automatic length-scale selection.
    Noise: σ²_n = 1e-6 (near-noiseless since inputs are certified).
    """

    def __init__(self, length_scale: float = 1.0, signal_var: float = 1.0,
                 noise_var: float = 1e-6, use_ard: bool = False,
                 length_scales: Optional[np.ndarray] = None):
        self.length_scale = length_scale
        self.signal_var = signal_var
        self.noise_var = noise_var
        self.use_ard = use_ard
        self.length_scales = length_scales  # per-dimension (ARD)
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self._fitted = False

    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute kernel matrix using isotropic or ARD kernel."""
        if self.use_ard and self.length_scales is not None:
            return matern52_kernel_matrix_ard(X, self.length_scales, self.signal_var)
        return matern52_kernel_matrix(X, self.length_scale, self.signal_var)

    def _compute_kernel_vector(self, x_query: np.ndarray) -> np.ndarray:
        """Compute kernel vector k(x_query, X_train)."""
        if self.use_ard and self.length_scales is not None:
            return np.array([
                matern52_kernel_ard(x_query, self.X_train[i],
                                    self.length_scales, self.signal_var)
                for i in range(self.X_train.shape[0])
            ])
        return np.array([
            matern52_kernel(x_query, self.X_train[i],
                            self.length_scale, self.signal_var)
            for i in range(self.X_train.shape[0])
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit GP to training data.

        Args:
            X: (n, d) array of parameter midpoints from certified cells.
            y: (n,) array of regime labels (encoded as integers).
        """
        self.X_train = X.copy()
        self.y_train = y.copy().astype(float)
        n = X.shape[0]

        # Initialize ARD length scales if not set
        if self.use_ard and self.length_scales is None:
            self.length_scales = np.full(X.shape[1], self.length_scale)

        K = self._compute_kernel_matrix(X)
        K += self.noise_var * np.eye(n)

        try:
            L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
            self.K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))
        except np.linalg.LinAlgError:
            K += 1e-4 * np.eye(n)
            L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
            self.K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))

        self._fitted = True

    def predict(self, x_query: np.ndarray) -> GPPrediction:
        """Predict at a single query point."""
        if not self._fitted:
            return GPPrediction(mean=0.0, variance=1.0, std=1.0)

        k_star = self._compute_kernel_vector(x_query)

        mean = float(k_star @ self.alpha)
        if self.use_ard and self.length_scales is not None:
            k_ss = matern52_kernel_ard(x_query, x_query,
                                       self.length_scales, self.signal_var)
        else:
            k_ss = matern52_kernel(x_query, x_query,
                                   self.length_scale, self.signal_var)
        var = max(0.0, k_ss - float(k_star @ self.K_inv @ k_star))
        return GPPrediction(mean=mean, variance=var, std=np.sqrt(var))

    def predict_batch(self, X_query: np.ndarray) -> List[GPPrediction]:
        """Predict at multiple query points."""
        return [self.predict(X_query[i]) for i in range(X_query.shape[0])]

    def optimize_length_scale(self, X: np.ndarray, y: np.ndarray,
                              candidates: Optional[List[float]] = None) -> float:
        """Select length scale by maximizing marginal log-likelihood."""
        if candidates is None:
            dists = []
            for i in range(min(X.shape[0], 100)):
                for j in range(i + 1, min(X.shape[0], 100)):
                    dists.append(np.linalg.norm(X[i] - X[j]))
            if dists:
                median_dist = float(np.median(dists))
                candidates = [median_dist * f for f in [0.1, 0.3, 1.0, 3.0, 10.0]]
            else:
                candidates = [0.1, 0.5, 1.0, 5.0]

        best_ll = -np.inf
        best_ls = self.length_scale
        n = X.shape[0]

        for ls in candidates:
            K = matern52_kernel_matrix(X, ls, self.signal_var)
            K += self.noise_var * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y.astype(float)))
                ll = -0.5 * float(y.astype(float) @ alpha) - \
                     np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
                if ll > best_ll:
                    best_ll = ll
                    best_ls = ls
            except np.linalg.LinAlgError:
                continue

        self.length_scale = best_ls
        return best_ls

    def optimize_ard_length_scales(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Optimize per-dimension ARD length scales via marginal log-likelihood.
        
        Tests each dimension independently with a coarse grid, then refines.
        """
        d = X.shape[1]
        best_ls = np.full(d, self.length_scale)
        
        # Compute per-dimension distance statistics
        for dim in range(d):
            dists = []
            for i in range(min(X.shape[0], 50)):
                for j in range(i + 1, min(X.shape[0], 50)):
                    dists.append(abs(X[i, dim] - X[j, dim]))
            if dists:
                median_dist = float(np.median(dists))
                if median_dist > 1e-10:
                    best_ls[dim] = median_dist
        
        # Grid search around median distances
        candidates = [0.1, 0.3, 1.0, 3.0, 10.0]
        best_ll = -np.inf
        best_scales = best_ls.copy()
        n = X.shape[0]
        
        for scale_factor in candidates:
            ls_test = best_ls * scale_factor
            K = matern52_kernel_matrix_ard(X, ls_test, self.signal_var)
            K += self.noise_var * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y.astype(float)))
                ll = -0.5 * float(y.astype(float) @ alpha) - \
                     np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
                if ll > best_ll:
                    best_ll = ll
                    best_scales = ls_test.copy()
            except np.linalg.LinAlgError:
                continue
        
        self.length_scales = best_scales
        self.use_ard = True
        return best_scales

    def loo_cross_validation(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Leave-one-out cross-validation error.
        
        Uses the efficient LOO formula:
          LOO_i = (α_i / [K^{-1}]_{ii})²
        where α = K^{-1} y.
        
        Returns mean squared LOO error.
        """
        self.fit(X, y)
        if self.K_inv is None:
            return float('inf')
        
        n = X.shape[0]
        loo_errors = np.zeros(n)
        for i in range(n):
            k_inv_ii = self.K_inv[i, i]
            if abs(k_inv_ii) > 1e-15:
                loo_errors[i] = (self.alpha[i] / k_inv_ii) ** 2
            else:
                loo_errors[i] = float('inf')
        
        return float(np.mean(loo_errors))

    def calibration_error(self, X_test: np.ndarray, y_test: np.ndarray,
                          n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).

        For each confidence bin, compares predicted coverage with observed
        coverage. ECE = Σ |bin_count/n| × |observed_coverage - predicted_coverage|.
        """
        if not self._fitted or len(X_test) == 0:
            return 1.0

        preds = self.predict_batch(X_test)
        residuals = np.array([y_test[i] - preds[i].mean for i in range(len(y_test))])
        stds = np.array([max(preds[i].std, 1e-10) for i in range(len(y_test))])
        z_scores = np.abs(residuals / stds)

        confidence_levels = np.linspace(0.1, 0.95, n_bins)
        ece = 0.0
        for conf in confidence_levels:
            from scipy.stats import norm
            z_crit = norm.ppf((1 + conf) / 2)
            observed = np.mean(z_scores <= z_crit)
            ece += abs(observed - conf) / n_bins

        return float(ece)

    # ------------------------------------------------------------------
    # Atlas-aware helpers
    # ------------------------------------------------------------------

    @classmethod
    def train_from_atlas(cls, atlas, use_ard: bool = False,
                         noise_var: float = 1e-6) -> 'GPSurrogate':
        """Train a GPSurrogate from a :class:`PhaseAtlas`.

        Extracts (parameter-box midpoint, integer regime label) pairs from
        every certified cell in *atlas* and fits a new GP to the data.

        Args:
            atlas: A ``PhaseAtlas`` instance with at least one cell.
            use_ard: Whether to use ARD length scales.
            noise_var: Observation noise variance.

        Returns:
            A fitted ``GPSurrogate``.
        """
        if not atlas.cells:
            return cls(noise_var=noise_var, use_ard=use_ard)

        from ..tiered.certificate import RegimeType

        _regime_map = {
            RegimeType.MONOSTABLE: 0,
            RegimeType.BISTABLE: 1,
            RegimeType.MULTISTABLE: 2,
            RegimeType.OSCILLATORY: 3,
            RegimeType.EXCITABLE: 4,
            RegimeType.INCONCLUSIVE: 5,
        }

        midpoints = []
        labels = []
        for cell in atlas.cells:
            mid = np.array([(lo + hi) / 2.0 for lo, hi in cell.parameter_box])
            midpoints.append(mid)
            labels.append(_regime_map.get(cell.regime, 5))

        X = np.array(midpoints)
        y = np.array(labels, dtype=float)

        gp = cls(noise_var=noise_var, use_ard=use_ard)
        gp.optimize_length_scale(X, y)
        gp.fit(X, y)
        return gp

    def predict_regime_boundary(self, x_query: np.ndarray) -> float:
        """Predict probability of *x_query* being near a phase boundary.

        The score is high when the GP posterior mean is near an integer
        boundary (regime transition) and uncertainty is large.

        Returns a value in [0, ∞); higher means more likely near a boundary.
        """
        pred = self.predict(x_query)
        nearest_int = round(pred.mean)
        boundary_proximity = 1.0 - min(1.0, abs(pred.mean - nearest_int))
        return pred.std * boundary_proximity
