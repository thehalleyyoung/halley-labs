"""
Information theory for neural networks.

Implements mutual information estimation, information bottleneck,
data processing inequality verification, compression-generalization tradeoff,
information plane analysis, minimum description length, Fisher information,
and channel capacity estimation.
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import digamma, gamma as gamma_fn
from scipy.spatial import KDTree
from scipy.linalg import eigvalsh, det, inv
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
import warnings


@dataclass
class InfoReport:
    """Report from information-theoretic analysis."""
    mutual_info_xt: float = 0.0  # I(X; T) - input to representation
    mutual_info_ty: float = 0.0  # I(T; Y) - representation to output
    mutual_info_xy: float = 0.0  # I(X; Y) - input to output
    dpi_satisfied: bool = True   # Data processing inequality
    compression_ratio: float = 0.0
    generalization_bound: float = 0.0
    info_plane_trajectory: List[Tuple[float, float]] = field(default_factory=list)
    mdl_code_length: float = 0.0
    fisher_info_trace: float = 0.0
    channel_capacity: float = 0.0
    bottleneck_beta: float = 0.0
    entropy_x: float = 0.0
    entropy_y: float = 0.0
    entropy_t: float = 0.0
    estimation_method: str = "binning"


@dataclass
class ModelSpec:
    """Neural network specification for info analysis."""
    depth: int = 5
    width: int = 100
    sigma_w: float = 1.0
    sigma_b: float = 0.0
    activation: str = "relu"
    input_dim: int = 10
    output_dim: int = 2
    n_params: int = 0


@dataclass
class DataSpec:
    """Data specification for info analysis."""
    n_samples: int = 1000
    input_dim: int = 10
    n_classes: int = 2
    noise_level: float = 0.1
    X: Optional[np.ndarray] = None
    Y: Optional[np.ndarray] = None


class BinningMIEstimator:
    """Mutual information estimation via binning (histogram method)."""

    def __init__(self, n_bins: int = 30):
        self.n_bins = n_bins

    def estimate_entropy(self, X: np.ndarray) -> float:
        """Estimate entropy of X using binning."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n, d = X.shape
        if d == 1:
            hist, _ = np.histogram(X[:, 0], bins=self.n_bins, density=True)
            bin_width = (np.max(X[:, 0]) - np.min(X[:, 0])) / self.n_bins
            hist = hist * bin_width
            hist = hist[hist > 0]
            return float(-np.sum(hist * np.log(hist + 1e-30)) + np.log(bin_width + 1e-30))

        if d <= 3:
            bins_per_dim = max(5, self.n_bins // d)
            ranges = [(np.min(X[:, i]) - 1e-6, np.max(X[:, i]) + 1e-6) for i in range(d)]
            hist, _ = np.histogramdd(X, bins=bins_per_dim, range=ranges, density=True)
            bin_volume = np.prod([(r[1] - r[0]) / bins_per_dim for r in ranges])
            hist = hist.ravel() * bin_volume
            hist = hist[hist > 0]
            return float(-np.sum(hist * np.log(hist + 1e-30)) + np.log(bin_volume + 1e-30))

        cov = np.cov(X, rowvar=False)
        sign, logdet = np.linalg.slogdet(cov + 1e-6 * np.eye(d))
        return float(0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * logdet)

    def estimate_mi(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Estimate mutual information I(X; Y) using binning."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        H_X = self.estimate_entropy(X)
        H_Y = self.estimate_entropy(Y)
        H_XY = self.estimate_entropy(np.hstack([X, Y]))

        mi = H_X + H_Y - H_XY
        return float(max(0, mi))

    def estimate_conditional_entropy(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Estimate H(X|Y) = H(X,Y) - H(Y)."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        H_XY = self.estimate_entropy(np.hstack([X, Y]))
        H_Y = self.estimate_entropy(Y)
        return float(max(0, H_XY - H_Y))


class KNNMIEstimator:
    """KNN-based mutual information estimation (Kraskov et al.)."""

    def __init__(self, k: int = 5):
        self.k = k

    def estimate_mi(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Estimate MI using KSG estimator (Kraskov, Stögbauer, Grassberger)."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n = len(X)
        if n < self.k + 1:
            return 0.0

        XY = np.hstack([X, Y])
        tree_xy = KDTree(XY)
        tree_x = KDTree(X)
        tree_y = KDTree(Y)

        mi = digamma(self.k) + digamma(n)

        for i in range(n):
            distances_xy, _ = tree_xy.query(XY[i], k=self.k + 1)
            eps = distances_xy[-1]

            if eps < 1e-10:
                eps = 1e-10

            n_x = max(1, tree_x.query_ball_point(X[i], eps, return_length=True) - 1)
            n_y = max(1, tree_y.query_ball_point(Y[i], eps, return_length=True) - 1)

            mi -= (digamma(n_x) + digamma(n_y)) / n

        return float(max(0, mi))

    def estimate_entropy(self, X: np.ndarray) -> float:
        """Estimate differential entropy using KNN."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n, d = X.shape
        if n < self.k + 1:
            return 0.0

        tree = KDTree(X)
        log_distances = np.zeros(n)

        for i in range(n):
            distances, _ = tree.query(X[i], k=self.k + 1)
            eps = distances[-1]
            if eps < 1e-10:
                eps = 1e-10
            log_distances[i] = np.log(eps)

        volume_unit_ball = np.pi ** (d / 2.0) / gamma_fn(d / 2.0 + 1)
        entropy = d * np.mean(log_distances) + np.log(volume_unit_ball) + \
                  digamma(n) - digamma(self.k)

        return float(entropy)


class MINEEstimator:
    """MINE-like mutual information estimator using gradient ascent."""

    def __init__(self, hidden_dim: int = 64, n_iterations: int = 500,
                 learning_rate: float = 0.01, batch_size: int = 256):
        self.hidden_dim = hidden_dim
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def _init_network(self, input_dim: int) -> Dict[str, np.ndarray]:
        """Initialize a simple statistics network."""
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / self.hidden_dim)
        return {
            "W1": np.random.randn(input_dim, self.hidden_dim) * scale1,
            "b1": np.zeros(self.hidden_dim),
            "W2": np.random.randn(self.hidden_dim, self.hidden_dim) * scale2,
            "b2": np.zeros(self.hidden_dim),
            "W3": np.random.randn(self.hidden_dim, 1) * scale2,
            "b3": np.zeros(1),
        }

    def _forward(self, x: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
        """Forward pass through statistics network."""
        h1 = np.maximum(0, x @ params["W1"] + params["b1"])
        h2 = np.maximum(0, h1 @ params["W2"] + params["b2"])
        out = h2 @ params["W3"] + params["b3"]
        return out.ravel()

    def _compute_gradients(self, x_joint: np.ndarray, x_marginal: np.ndarray,
                           params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute gradients of MINE objective via finite differences."""
        grads = {}
        eps = 1e-4

        current_loss = self._mine_objective(x_joint, x_marginal, params)

        for key in params:
            grad = np.zeros_like(params[key])
            flat = params[key].ravel()
            n_params = len(flat)
            sample_indices = np.random.choice(n_params, size=min(50, n_params), replace=False)

            for idx in sample_indices:
                flat_copy = flat.copy()
                flat_copy[idx] += eps
                params_plus = {k: v.copy() for k, v in params.items()}
                params_plus[key] = flat_copy.reshape(params[key].shape)
                loss_plus = self._mine_objective(x_joint, x_marginal, params_plus)
                grad.ravel()[idx] = (loss_plus - current_loss) / eps

            grads[key] = grad
        return grads

    def _mine_objective(self, x_joint: np.ndarray, x_marginal: np.ndarray,
                        params: Dict[str, np.ndarray]) -> float:
        """Compute MINE objective (Donsker-Varadhan bound)."""
        t_joint = self._forward(x_joint, params)
        t_marginal = self._forward(x_marginal, params)
        t_marginal_shifted = t_marginal - np.max(t_marginal)
        return float(np.mean(t_joint) - np.log(np.mean(np.exp(t_marginal_shifted)) + 1e-30)
                     - np.max(t_marginal))

    def estimate_mi(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Estimate MI using MINE."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n = len(X)
        input_dim = X.shape[1] + Y.shape[1]
        params = self._init_network(input_dim)

        best_mi = 0.0
        for iteration in range(self.n_iterations):
            idx = np.random.choice(n, size=min(self.batch_size, n), replace=False)
            x_batch = X[idx]
            y_batch = Y[idx]

            joint = np.hstack([x_batch, y_batch])
            perm = np.random.permutation(len(y_batch))
            marginal = np.hstack([x_batch, y_batch[perm]])

            mi = self._mine_objective(joint, marginal, params)
            best_mi = max(best_mi, mi)

            grads = self._compute_gradients(joint, marginal, params)
            for key in params:
                params[key] += self.learning_rate * grads[key]

        return float(max(0, best_mi))


class InformationBottleneck:
    """Information bottleneck analysis."""

    def __init__(self, n_bins: int = 30):
        self.n_bins = n_bins
        self.mi_estimator = BinningMIEstimator(n_bins)

    def compute_ib_curve(
        self, X: np.ndarray, Y: np.ndarray, T: np.ndarray,
        beta_values: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute information bottleneck curve: I(X;T) vs I(T;Y) for different betas."""
        I_XT = self.mi_estimator.estimate_mi(X, T)
        I_TY = self.mi_estimator.estimate_mi(T, Y)
        I_XY = self.mi_estimator.estimate_mi(X, Y)

        if beta_values is None:
            beta_values = np.logspace(-1, 2, 20)

        curve = []
        for beta in beta_values:
            noise_scale = 1.0 / (beta + 1e-12)
            T_noisy = T + np.random.randn(*T.shape) * noise_scale
            I_XT_beta = self.mi_estimator.estimate_mi(X, T_noisy)
            I_TY_beta = self.mi_estimator.estimate_mi(T_noisy, Y)
            curve.append({
                "beta": float(beta),
                "I_XT": float(I_XT_beta),
                "I_TY": float(I_TY_beta),
            })

        return {
            "I_XT": float(I_XT),
            "I_TY": float(I_TY),
            "I_XY": float(I_XY),
            "curve": curve,
        }

    def optimal_representation(
        self, X: np.ndarray, Y: np.ndarray, n_clusters: int = 10, beta: float = 1.0
    ) -> Dict[str, Any]:
        """Find optimal representation via deterministic IB (simplified)."""
        n = len(X)
        assignments = np.random.randint(0, n_clusters, size=n)

        for iteration in range(50):
            old_assignments = assignments.copy()
            centroids = []
            for c in range(n_clusters):
                mask = assignments == c
                if np.sum(mask) > 0:
                    centroids.append(np.mean(X[mask], axis=0))
                else:
                    centroids.append(X[np.random.randint(n)])
            centroids = np.array(centroids)

            distances = np.zeros((n, n_clusters))
            for c in range(n_clusters):
                distances[:, c] = np.sum((X - centroids[c]) ** 2, axis=1)

            if Y.ndim == 1:
                for c in range(n_clusters):
                    mask = assignments == c
                    if np.sum(mask) > 0:
                        y_dist = np.bincount(Y[mask].astype(int),
                                             minlength=int(np.max(Y)) + 1)
                        y_dist = y_dist / (np.sum(y_dist) + 1e-12)
                        for i in range(n):
                            yi = int(Y[i])
                            distances[i, c] -= beta * np.log(y_dist[yi] + 1e-12)

            assignments = np.argmin(distances, axis=1)
            if np.array_equal(assignments, old_assignments):
                break

        T = np.zeros((n, n_clusters))
        T[np.arange(n), assignments] = 1.0
        I_XT = self.mi_estimator.estimate_mi(X, T)
        I_TY = self.mi_estimator.estimate_mi(T, Y.reshape(-1, 1))

        return {
            "assignments": assignments,
            "I_XT": float(I_XT),
            "I_TY": float(I_TY),
            "n_clusters_used": int(len(np.unique(assignments))),
        }


class DataProcessingInequality:
    """Verify data processing inequality: I(X;Y) >= I(f(X);Y)."""

    def __init__(self, n_bins: int = 30):
        self.mi_estimator = BinningMIEstimator(n_bins)
        self.knn_estimator = KNNMIEstimator(k=5)

    def verify(self, X: np.ndarray, Y: np.ndarray,
               transform: Callable) -> Dict[str, Any]:
        """Verify DPI: I(X;Y) >= I(f(X);Y)."""
        fX = transform(X)
        if fX.ndim == 1:
            fX = fX.reshape(-1, 1)
        if Y.ndim == 1:
            Y_r = Y.reshape(-1, 1)
        else:
            Y_r = Y

        I_XY_bin = self.mi_estimator.estimate_mi(X, Y_r)
        I_fXY_bin = self.mi_estimator.estimate_mi(fX, Y_r)

        I_XY_knn = self.knn_estimator.estimate_mi(X, Y_r)
        I_fXY_knn = self.knn_estimator.estimate_mi(fX, Y_r)

        satisfied_bin = I_XY_bin >= I_fXY_bin - 0.05
        satisfied_knn = I_XY_knn >= I_fXY_knn - 0.05

        info_loss = max(0, I_XY_bin - I_fXY_bin)

        return {
            "I_XY_binning": float(I_XY_bin),
            "I_fXY_binning": float(I_fXY_bin),
            "I_XY_knn": float(I_XY_knn),
            "I_fXY_knn": float(I_fXY_knn),
            "satisfied_binning": bool(satisfied_bin),
            "satisfied_knn": bool(satisfied_knn),
            "information_loss": float(info_loss),
            "compression_ratio": float(I_fXY_bin / (I_XY_bin + 1e-12)),
        }

    def verify_chain(self, X: np.ndarray, Y: np.ndarray,
                     transforms: List[Callable]) -> Dict[str, Any]:
        """Verify DPI along a chain of transformations X -> T1 -> T2 -> ... -> Y."""
        mis = []
        current = X.copy()
        if current.ndim == 1:
            current = current.reshape(-1, 1)
        Y_r = Y.reshape(-1, 1) if Y.ndim == 1 else Y

        I_start = self.mi_estimator.estimate_mi(current, Y_r)
        mis.append({"layer": 0, "I_TY": float(I_start)})

        all_satisfied = True
        for i, transform in enumerate(transforms):
            current = transform(current)
            if current.ndim == 1:
                current = current.reshape(-1, 1)
            I_current = self.mi_estimator.estimate_mi(current, Y_r)
            mis.append({"layer": i + 1, "I_TY": float(I_current)})

            if I_current > mis[-2]["I_TY"] + 0.1:
                all_satisfied = False

        return {
            "layer_mis": mis,
            "all_satisfied": all_satisfied,
            "total_info_loss": float(mis[0]["I_TY"] - mis[-1]["I_TY"]),
        }


class CompressionGeneralizationTradeoff:
    """Relate compression of representation to generalization."""

    def __init__(self, n_bins: int = 30):
        self.mi_estimator = BinningMIEstimator(n_bins)

    def compute_tradeoff(
        self, X: np.ndarray, Y: np.ndarray,
        representations: List[np.ndarray],
        test_errors: List[float]
    ) -> Dict[str, Any]:
        """Compute compression vs generalization tradeoff."""
        compressions = []
        relevances = []

        Y_r = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        I_XY = self.mi_estimator.estimate_mi(X, Y_r)

        for T in representations:
            if T.ndim == 1:
                T = T.reshape(-1, 1)
            I_XT = self.mi_estimator.estimate_mi(X, T)
            I_TY = self.mi_estimator.estimate_mi(T, Y_r)

            H_X = self.mi_estimator.estimate_entropy(X)
            compression = 1.0 - I_XT / (H_X + 1e-12)
            relevance = I_TY / (I_XY + 1e-12)

            compressions.append(float(compression))
            relevances.append(float(relevance))

        if len(compressions) > 2:
            coeffs = np.polyfit(compressions, test_errors, 1)
            slope = coeffs[0]
        else:
            slope = 0.0

        return {
            "compressions": compressions,
            "relevances": relevances,
            "test_errors": test_errors,
            "compression_error_slope": float(slope),
            "I_XY": float(I_XY),
        }

    def predict_generalization(
        self, I_XT: float, I_TY: float, n_samples: int, n_params: int
    ) -> Dict[str, float]:
        """Predict generalization gap from information quantities."""
        complexity_term = I_XT / n_samples
        relevance_efficiency = I_TY / (I_XT + 1e-12)
        param_complexity = n_params * np.log(n_samples) / (2 * n_samples)

        predicted_gap = complexity_term + param_complexity * (1 - relevance_efficiency)
        predicted_gap = max(0, predicted_gap)

        return {
            "predicted_gen_gap": float(predicted_gap),
            "complexity_term": float(complexity_term),
            "relevance_efficiency": float(relevance_efficiency),
            "param_complexity": float(param_complexity),
        }


class InformationPlaneAnalyzer:
    """Analyze trajectories in the information plane I(X;T) vs I(T;Y)."""

    def __init__(self, n_bins: int = 30):
        self.mi_estimator = BinningMIEstimator(n_bins)

    def compute_trajectory(
        self, X: np.ndarray, Y: np.ndarray,
        representation_snapshots: List[np.ndarray]
    ) -> List[Tuple[float, float]]:
        """Compute information plane trajectory from representation snapshots."""
        trajectory = []
        Y_r = Y.reshape(-1, 1) if Y.ndim == 1 else Y

        for T in representation_snapshots:
            if T.ndim == 1:
                T = T.reshape(-1, 1)
            I_XT = self.mi_estimator.estimate_mi(X, T)
            I_TY = self.mi_estimator.estimate_mi(T, Y_r)
            trajectory.append((float(I_XT), float(I_TY)))

        return trajectory

    def analyze_phases(self, trajectory: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Analyze phases of learning from information plane trajectory."""
        if len(trajectory) < 3:
            return {"phases": ["unknown"], "transition_points": []}

        I_XTs = np.array([t[0] for t in trajectory])
        I_TYs = np.array([t[1] for t in trajectory])

        dI_XT = np.diff(I_XTs)
        dI_TY = np.diff(I_TYs)

        phases = []
        transition_points = []

        for i in range(len(dI_XT)):
            if dI_XT[i] > 0.01 and dI_TY[i] > 0.01:
                phase = "fitting"
            elif dI_XT[i] < -0.01 and dI_TY[i] > -0.01:
                phase = "compression"
            elif abs(dI_XT[i]) < 0.01 and abs(dI_TY[i]) < 0.01:
                phase = "converged"
            else:
                phase = "other"

            if phases and phase != phases[-1]:
                transition_points.append(i)
            phases.append(phase)

        has_fitting = "fitting" in phases
        has_compression = "compression" in phases

        return {
            "phases": phases,
            "transition_points": transition_points,
            "has_fitting_phase": has_fitting,
            "has_compression_phase": has_compression,
            "final_I_XT": float(I_XTs[-1]),
            "final_I_TY": float(I_TYs[-1]),
            "max_I_XT": float(np.max(I_XTs)),
            "max_I_TY": float(np.max(I_TYs)),
        }

    def compute_layer_wise(
        self, X: np.ndarray, Y: np.ndarray,
        layer_activations: List[np.ndarray]
    ) -> List[Dict[str, float]]:
        """Compute information quantities for each layer."""
        results = []
        Y_r = Y.reshape(-1, 1) if Y.ndim == 1 else Y

        for layer_idx, T in enumerate(layer_activations):
            if T.ndim == 1:
                T = T.reshape(-1, 1)
            I_XT = self.mi_estimator.estimate_mi(X, T)
            I_TY = self.mi_estimator.estimate_mi(T, Y_r)
            H_T = self.mi_estimator.estimate_entropy(T)

            results.append({
                "layer": layer_idx,
                "I_XT": float(I_XT),
                "I_TY": float(I_TY),
                "H_T": float(H_T),
                "compression": float(1.0 - I_XT / (H_T + 1e-12)),
                "sufficiency": float(I_TY / (self.mi_estimator.estimate_mi(X, Y_r) + 1e-12)),
            })

        return results


class MDLAnalyzer:
    """Minimum Description Length analysis for neural networks."""

    def __init__(self):
        pass

    def compute_model_code_length(
        self, n_params: int, param_precision: int = 32,
        n_quantization_levels: int = 256
    ) -> float:
        """Compute code length of model parameters."""
        bits_per_param = np.log2(n_quantization_levels)
        structural_bits = n_params * np.log2(n_params + 1)
        return float(n_params * bits_per_param + structural_bits)

    def compute_data_code_length(
        self, predictions: np.ndarray, targets: np.ndarray,
        n_classes: int = 2
    ) -> float:
        """Compute code length of data given model."""
        n = len(targets)
        if n_classes == 2:
            probs = np.clip(predictions, 1e-10, 1 - 1e-10)
            if targets.ndim == 1:
                nll = -np.sum(targets * np.log(probs) + (1 - targets) * np.log(1 - probs))
            else:
                nll = -np.sum(targets * np.log(probs))
            return float(nll / np.log(2))
        else:
            if predictions.ndim == 1:
                predictions = np.eye(n_classes)[predictions.astype(int)]
            probs = np.clip(predictions, 1e-10, 1.0)
            probs = probs / probs.sum(axis=1, keepdims=True)
            nll = -np.sum(np.log(probs[np.arange(n), targets.astype(int)]))
            return float(nll / np.log(2))

    def compute_mdl(
        self, n_params: int, predictions: np.ndarray, targets: np.ndarray,
        n_classes: int = 2, param_precision: int = 32
    ) -> Dict[str, float]:
        """Compute total MDL = model code length + data code length."""
        model_cl = self.compute_model_code_length(n_params, param_precision)
        data_cl = self.compute_data_code_length(predictions, targets, n_classes)
        total = model_cl + data_cl

        return {
            "model_code_length": float(model_cl),
            "data_code_length": float(data_cl),
            "total_mdl": float(total),
            "model_fraction": float(model_cl / (total + 1e-12)),
            "normalized_mdl": float(total / len(targets)),
        }

    def compare_models(
        self, model_specs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare models using MDL criterion."""
        mdl_values = []
        for spec in model_specs:
            mdl = self.compute_mdl(
                spec["n_params"], spec["predictions"],
                spec["targets"], spec.get("n_classes", 2)
            )
            mdl_values.append(mdl)

        total_mdls = [m["total_mdl"] for m in mdl_values]
        best_idx = int(np.argmin(total_mdls))

        return {
            "mdl_values": mdl_values,
            "best_model_index": best_idx,
            "best_mdl": total_mdls[best_idx],
            "mdl_ranking": np.argsort(total_mdls).tolist(),
        }


class FisherInformationComputer:
    """Compute Fisher information for neural networks."""

    def __init__(self, n_samples: int = 5000):
        self.n_samples = n_samples

    def compute_empirical_fisher(
        self, X: np.ndarray, weights: np.ndarray, sigma_w: float
    ) -> np.ndarray:
        """Compute empirical Fisher information matrix."""
        n, d = X.shape
        n_w = weights.size

        eps = 1e-4
        grad_log_probs = np.zeros((min(n, self.n_samples), n_w))

        for i in range(min(n, self.n_samples)):
            x = X[i]
            base_output = x @ weights.reshape(d, -1) if weights.ndim == 1 else x @ weights
            base_log_prob = -0.5 * np.sum(base_output ** 2)

            for j in range(min(n_w, 100)):
                w_plus = weights.ravel().copy()
                w_plus[j] += eps
                output_plus = x @ w_plus.reshape(d, -1) if weights.ndim == 1 else x @ w_plus.reshape(weights.shape)
                log_prob_plus = -0.5 * np.sum(output_plus ** 2)
                grad_log_probs[i, j] = (log_prob_plus - base_log_prob) / eps

        fisher = grad_log_probs.T @ grad_log_probs / len(grad_log_probs)
        return fisher

    def compute_fisher_trace(self, X: np.ndarray, weights: np.ndarray,
                              sigma_w: float) -> float:
        """Compute trace of Fisher information matrix."""
        fisher = self.compute_empirical_fisher(X, weights, sigma_w)
        return float(np.trace(fisher))

    def compute_fisher_eigenspectrum(
        self, X: np.ndarray, weights: np.ndarray, sigma_w: float
    ) -> Dict[str, Any]:
        """Compute eigenspectrum of Fisher information matrix."""
        fisher = self.compute_empirical_fisher(X, weights, sigma_w)
        eigenvalues = eigvalsh(fisher)
        eigenvalues = eigenvalues[::-1]

        return {
            "eigenvalues": eigenvalues.tolist(),
            "trace": float(np.sum(eigenvalues)),
            "max_eigenvalue": float(eigenvalues[0]) if len(eigenvalues) > 0 else 0.0,
            "effective_dimensionality": float(np.sum(eigenvalues) ** 2 /
                                               (np.sum(eigenvalues ** 2) + 1e-12)),
            "condition_number": float(eigenvalues[0] / (eigenvalues[-1] + 1e-12))
                if len(eigenvalues) > 1 else 1.0,
        }

    def natural_gradient_scaling(
        self, X: np.ndarray, weights: np.ndarray, gradient: np.ndarray,
        sigma_w: float, damping: float = 1e-3
    ) -> np.ndarray:
        """Compute natural gradient: F^{-1} * gradient."""
        fisher = self.compute_empirical_fisher(X, weights, sigma_w)
        n = fisher.shape[0]
        fisher_reg = fisher + damping * np.eye(n)
        try:
            natural_grad = np.linalg.solve(fisher_reg, gradient.ravel()[:n])
            return natural_grad
        except np.linalg.LinAlgError:
            return gradient.ravel()[:n]


class ChannelCapacityEstimator:
    """Estimate channel capacity of neural network layers."""

    def __init__(self, n_samples: int = 10000, n_bins: int = 30):
        self.n_samples = n_samples
        self.mi_estimator = BinningMIEstimator(n_bins)

    def estimate_capacity_gaussian(
        self, weight_matrix: np.ndarray, noise_variance: float = 0.1
    ) -> Dict[str, float]:
        """Estimate channel capacity for Gaussian channel: C = 0.5 log(1 + SNR)."""
        singular_values = np.linalg.svd(weight_matrix, compute_uv=False)
        snr_per_mode = singular_values ** 2 / (noise_variance + 1e-12)

        capacity = 0.5 * np.sum(np.log2(1 + snr_per_mode))
        water_filling_capacity = self._water_filling(snr_per_mode)

        return {
            "capacity_sum": float(capacity),
            "water_filling_capacity": float(water_filling_capacity),
            "n_effective_channels": int(np.sum(snr_per_mode > 1)),
            "max_snr": float(np.max(snr_per_mode)),
            "mean_snr": float(np.mean(snr_per_mode)),
        }

    def _water_filling(self, snr_per_mode: np.ndarray,
                       total_power: Optional[float] = None) -> float:
        """Water-filling algorithm for capacity."""
        if total_power is None:
            total_power = float(np.sum(snr_per_mode))

        noise_levels = 1.0 / (snr_per_mode + 1e-12)
        sorted_noise = np.sort(noise_levels)

        n = len(sorted_noise)
        for k in range(n, 0, -1):
            water_level = (total_power + np.sum(sorted_noise[:k])) / k
            if water_level > sorted_noise[k - 1]:
                powers = np.maximum(0, water_level - noise_levels)
                capacity = 0.5 * np.sum(np.log2(1 + powers / noise_levels))
                return float(capacity)

        return 0.0

    def estimate_capacity_empirical(
        self, layer_fn: Callable, input_dim: int,
        noise_variance: float = 0.1, n_trials: int = 20
    ) -> Dict[str, float]:
        """Estimate capacity empirically by maximizing MI over input distributions."""
        best_mi = 0.0
        best_input_std = 1.0

        for input_std in np.logspace(-1, 1, n_trials):
            X = np.random.randn(self.n_samples, input_dim) * input_std
            Y = layer_fn(X) + np.random.randn(self.n_samples, input_dim) * np.sqrt(noise_variance)
            mi = self.mi_estimator.estimate_mi(X, Y)
            if mi > best_mi:
                best_mi = mi
                best_input_std = input_std

        return {
            "estimated_capacity": float(best_mi),
            "optimal_input_std": float(best_input_std),
        }


class NNInfoAnalyzer:
    """Main information-theoretic analyzer for neural networks."""

    def __init__(self, n_bins: int = 30, k_knn: int = 5, n_samples: int = 5000):
        self.binning_estimator = BinningMIEstimator(n_bins)
        self.knn_estimator = KNNMIEstimator(k_knn)
        self.ib = InformationBottleneck(n_bins)
        self.dpi = DataProcessingInequality(n_bins)
        self.cg_tradeoff = CompressionGeneralizationTradeoff(n_bins)
        self.ip_analyzer = InformationPlaneAnalyzer(n_bins)
        self.mdl = MDLAnalyzer()
        self.fisher = FisherInformationComputer(n_samples)
        self.capacity = ChannelCapacityEstimator(n_samples, n_bins)

    def analyze(self, model_spec: ModelSpec, data_spec: DataSpec) -> InfoReport:
        """Full information-theoretic analysis."""
        report = InfoReport()

        if data_spec.X is None or data_spec.Y is None:
            X, Y = self._generate_synthetic_data(data_spec)
        else:
            X, Y = data_spec.X, data_spec.Y

        Y_r = Y.reshape(-1, 1) if Y.ndim == 1 else Y

        report.entropy_x = self.binning_estimator.estimate_entropy(X)
        report.entropy_y = self.binning_estimator.estimate_entropy(Y_r)
        report.mutual_info_xy = self.binning_estimator.estimate_mi(X, Y_r)

        W = np.random.randn(model_spec.input_dim, model_spec.width) * \
            model_spec.sigma_w / np.sqrt(model_spec.input_dim)
        T = X @ W
        if model_spec.activation == "relu":
            T = np.maximum(0, T)
        elif model_spec.activation == "tanh":
            T = np.tanh(T)

        report.mutual_info_xt = self.binning_estimator.estimate_mi(X, T)
        report.mutual_info_ty = self.binning_estimator.estimate_mi(T, Y_r)
        report.entropy_t = self.binning_estimator.estimate_entropy(T)

        report.dpi_satisfied = report.mutual_info_xy >= report.mutual_info_ty - 0.05

        report.compression_ratio = 1.0 - report.mutual_info_xt / (report.entropy_x + 1e-12)

        n_params = model_spec.n_params if model_spec.n_params > 0 else \
            model_spec.input_dim * model_spec.width * model_spec.depth
        gen_pred = self.cg_tradeoff.predict_generalization(
            report.mutual_info_xt, report.mutual_info_ty,
            data_spec.n_samples, n_params
        )
        report.generalization_bound = gen_pred["predicted_gen_gap"]

        report.info_plane_trajectory = [(report.mutual_info_xt, report.mutual_info_ty)]

        predictions = np.random.rand(len(Y))
        mdl_result = self.mdl.compute_mdl(n_params, predictions, Y, data_spec.n_classes)
        report.mdl_code_length = mdl_result["total_mdl"]

        weights = np.random.randn(model_spec.input_dim, model_spec.width) * model_spec.sigma_w
        report.fisher_info_trace = self.fisher.compute_fisher_trace(
            X[:min(200, len(X))], weights, model_spec.sigma_w
        )

        cap_result = self.capacity.estimate_capacity_gaussian(W)
        report.channel_capacity = cap_result["capacity_sum"]

        report.estimation_method = "binning"

        return report

    def _generate_synthetic_data(self, data_spec: DataSpec) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data for analysis."""
        n = data_spec.n_samples
        d = data_spec.input_dim
        k = data_spec.n_classes

        X = np.random.randn(n, d)

        w_true = np.random.randn(d)
        w_true /= np.linalg.norm(w_true)
        projections = X @ w_true + np.random.randn(n) * data_spec.noise_level

        if k == 2:
            Y = (projections > 0).astype(float)
        else:
            boundaries = np.percentile(projections, np.linspace(0, 100, k + 1)[1:-1])
            Y = np.digitize(projections, boundaries).astype(float)

        return X, Y

    def compare_estimators(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
        """Compare different MI estimation methods."""
        if Y.ndim == 1:
            Y_r = Y.reshape(-1, 1)
        else:
            Y_r = Y

        mi_bin = self.binning_estimator.estimate_mi(X, Y_r)
        mi_knn = self.knn_estimator.estimate_mi(X, Y_r)

        return {
            "mi_binning": float(mi_bin),
            "mi_knn": float(mi_knn),
            "relative_difference": float(abs(mi_bin - mi_knn) / (max(mi_bin, mi_knn) + 1e-12)),
        }
