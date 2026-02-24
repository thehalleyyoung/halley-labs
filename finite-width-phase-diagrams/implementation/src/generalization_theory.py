"""
Neural network generalization theory.

Implements VC dimension bounds, Rademacher complexity, PAC-Bayes bounds,
compression bounds, stability bounds, margin-based bounds,
flatness-based bounds, and double descent analysis.
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.special import comb, gammaln
from scipy.linalg import eigvalsh, svdvals, norm
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
import warnings


@dataclass
class GenReport:
    """Report from generalization analysis."""
    vc_bound: float = 0.0
    rademacher_bound: float = 0.0
    pac_bayes_bound: float = 0.0
    compression_bound: float = 0.0
    stability_bound: float = 0.0
    margin_bound: float = 0.0
    flatness_bound: float = 0.0
    predicted_gap: float = 0.0
    double_descent_regime: str = "unknown"
    effective_dimensionality: float = 0.0
    n_params: int = 0
    n_samples: int = 0
    train_error: float = 0.0
    test_error: float = 0.0
    bounds_summary: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelSpec:
    """Model specification for generalization analysis."""
    depth: int = 5
    width: int = 256
    n_params: int = 0
    sigma_w: float = 1.0
    sigma_b: float = 0.0
    activation: str = "relu"
    weight_norm: float = 0.0
    spectral_norm: float = 0.0
    margin: float = 0.0
    flatness: float = 0.0  # sharpness measure
    weights: Optional[List[np.ndarray]] = None


@dataclass
class DataSpec:
    """Data specification for generalization analysis."""
    n_samples: int = 1000
    input_dim: int = 10
    n_classes: int = 2
    noise_level: float = 0.0
    X: Optional[np.ndarray] = None
    Y: Optional[np.ndarray] = None


class VCDimensionBound:
    """VC dimension bounds for neural networks."""

    def __init__(self):
        pass

    def compute_vc_dim(self, depth: int, width: int, n_params: int) -> int:
        """Compute VC dimension upper bound for neural networks.

        VC-dim <= O(W * L * log(W)) where W = n_params, L = depth
        """
        if n_params == 0:
            n_params = width * width * depth
        vc_dim = int(n_params * depth * np.log2(n_params + 2))
        return vc_dim

    def compute_vc_bound(self, vc_dim: int, n_samples: int,
                          delta: float = 0.05) -> float:
        """Compute VC generalization bound.

        With probability >= 1 - delta:
        |train_error - test_error| <= sqrt((vc_dim * (1 + log(2*n/vc_dim)) + log(4/delta)) / n)
        """
        if n_samples <= 0 or vc_dim <= 0:
            return 1.0

        if vc_dim >= n_samples:
            return 1.0

        numerator = vc_dim * (1 + np.log(2 * n_samples / vc_dim)) + np.log(4 / delta)
        bound = np.sqrt(numerator / n_samples)
        return float(min(1.0, bound))

    def growth_function_bound(self, vc_dim: int, n: int) -> float:
        """Compute growth function bound Pi_H(n) <= sum_{i=0}^{d} C(n, i)."""
        if vc_dim >= n:
            return 2 ** n

        total = 0
        for i in range(vc_dim + 1):
            total += comb(n, i, exact=True)
        return float(total)

    def effective_vc_dim(self, weights: List[np.ndarray]) -> float:
        """Estimate effective VC dimension from weight matrices."""
        total_params = sum(w.size for w in weights)
        spectral_norms = [float(svdvals(w)[0]) for w in weights if w.ndim == 2]

        if spectral_norms:
            spectral_product = np.prod(spectral_norms)
            frobenius_product = np.prod([float(norm(w, 'fro')) for w in weights if w.ndim == 2])
            vc_spectral = (spectral_product ** 2 * frobenius_product ** 2) / \
                          (np.prod(spectral_norms) ** 2 + 1e-12)
            return float(min(total_params, vc_spectral))

        return float(total_params)


class RademacherComplexity:
    """Rademacher complexity bounds."""

    def __init__(self, n_trials: int = 100):
        self.n_trials = n_trials

    def empirical_rademacher(self, X: np.ndarray, hypothesis_class_fn: Callable,
                              n_hypotheses: int = 100) -> float:
        """Estimate empirical Rademacher complexity.

        R_n(H) = E_sigma[sup_{h in H} (1/n) sum_i sigma_i * h(x_i)]
        """
        n = len(X)
        max_correlations = []

        for _ in range(self.n_trials):
            sigma = np.random.choice([-1, 1], size=n)
            best_correlation = -np.inf

            for _ in range(n_hypotheses):
                h_values = hypothesis_class_fn(X)
                correlation = np.mean(sigma * h_values)
                best_correlation = max(best_correlation, correlation)

            max_correlations.append(best_correlation)

        return float(np.mean(max_correlations))

    def linear_rademacher(self, X: np.ndarray, weight_bound: float = 1.0) -> float:
        """Rademacher complexity for linear classifiers with bounded weights.

        R_n(H) <= weight_bound * sqrt(trace(X^T X / n^2))
        """
        n = len(X)
        gram = X.T @ X / n
        trace = np.trace(gram)
        return float(weight_bound * np.sqrt(trace) / n)

    def neural_network_rademacher(self, X: np.ndarray, depth: int,
                                   spectral_norms: List[float],
                                   frobenius_norms: List[float]) -> float:
        """Rademacher complexity bound for deep neural networks.

        Golowich et al. (2018) bound:
        R_n <= (prod spectral_norms) * (sum frobenius/spectral)^{1/2} * ||X||_F / n
        """
        n = len(X)
        spectral_product = np.prod(spectral_norms)
        ratio_sum = sum(f / (s + 1e-12) for f, s in zip(frobenius_norms, spectral_norms))
        x_norm = norm(X, 'fro')

        bound = spectral_product * np.sqrt(ratio_sum) * x_norm / n
        return float(min(1.0, bound))

    def compute_bound(self, rademacher: float, n_samples: int,
                      delta: float = 0.05) -> float:
        """Convert Rademacher complexity to generalization bound."""
        bound = 2 * rademacher + np.sqrt(np.log(2 / delta) / (2 * n_samples))
        return float(min(1.0, bound))


class PACBayesBound:
    """PAC-Bayes generalization bounds."""

    def __init__(self):
        pass

    def mcallester_bound(self, train_error: float, kl_divergence: float,
                          n_samples: int, delta: float = 0.05) -> float:
        """McAllester's PAC-Bayes bound.

        With probability >= 1 - delta:
        test_error <= train_error + sqrt((KL(Q||P) + log(2*sqrt(n)/delta)) / (2*n))
        """
        numerator = kl_divergence + np.log(2 * np.sqrt(n_samples) / delta)
        bound = train_error + np.sqrt(numerator / (2 * n_samples))
        return float(min(1.0, bound))

    def catoni_bound(self, train_error: float, kl_divergence: float,
                     n_samples: int, delta: float = 0.05) -> float:
        """Catoni's tighter PAC-Bayes bound."""
        C = kl_divergence + np.log(2 * np.sqrt(n_samples) / delta)
        if n_samples * train_error <= 0:
            return float(min(1.0, C / n_samples))

        bound = 1 - np.exp(-train_error - C / n_samples)
        return float(min(1.0, max(0, bound)))

    def compute_kl_divergence_gaussian(self, posterior_mean: np.ndarray,
                                        posterior_var: float,
                                        prior_var: float) -> float:
        """Compute KL(N(mu, sigma_q^2) || N(0, sigma_p^2))."""
        d = len(posterior_mean)
        kl = 0.5 * (d * (posterior_var / prior_var - 1 + np.log(prior_var / posterior_var)) +
                     np.sum(posterior_mean ** 2) / prior_var)
        return float(max(0, kl))

    def compute_kl_from_weights(self, weights: List[np.ndarray],
                                 prior_var: float = 1.0) -> float:
        """Compute KL divergence from weight matrices."""
        all_weights = np.concatenate([w.ravel() for w in weights])
        posterior_mean = all_weights
        posterior_var = np.var(all_weights - np.mean(all_weights)) + 1e-6
        return self.compute_kl_divergence_gaussian(posterior_mean, posterior_var, prior_var)

    def optimize_prior(self, weights: List[np.ndarray], train_error: float,
                       n_samples: int, delta: float = 0.05) -> Dict[str, float]:
        """Optimize prior variance to get tightest bound."""
        all_weights = np.concatenate([w.ravel() for w in weights])

        def bound_fn(log_prior_var):
            prior_var = np.exp(log_prior_var)
            kl = self.compute_kl_divergence_gaussian(
                all_weights, np.var(all_weights) + 1e-6, prior_var
            )
            return self.mcallester_bound(train_error, kl, n_samples, delta)

        result = minimize_scalar(bound_fn, bounds=(-5, 5), method="bounded")
        optimal_prior_var = np.exp(result.x)
        kl = self.compute_kl_divergence_gaussian(
            all_weights, np.var(all_weights) + 1e-6, optimal_prior_var
        )

        return {
            "optimal_prior_var": float(optimal_prior_var),
            "kl_divergence": float(kl),
            "bound": float(result.fun),
        }


class CompressionBound:
    """Generalization from compressibility."""

    def __init__(self):
        pass

    def compute_compression_bound(self, n_params: int, effective_params: int,
                                   n_samples: int, train_error: float = 0.0,
                                   delta: float = 0.05) -> float:
        """Compression-based generalization bound.

        If model can be compressed from n_params to effective_params:
        gap <= sqrt((effective_params * log(n_params) + log(1/delta)) / n_samples)
        """
        if effective_params >= n_params:
            effective_params = n_params

        numerator = effective_params * np.log(n_params + 1) + np.log(1 / delta)
        bound = train_error + np.sqrt(numerator / (n_samples + 1e-12))
        return float(min(1.0, bound))

    def estimate_effective_params_svd(self, weights: List[np.ndarray],
                                       threshold: float = 0.01) -> int:
        """Estimate effective parameters via SVD truncation."""
        total_effective = 0
        for W in weights:
            if W.ndim < 2:
                total_effective += W.size
                continue
            sv = svdvals(W)
            sv_normalized = sv / (sv[0] + 1e-12)
            n_significant = int(np.sum(sv_normalized > threshold))
            total_effective += n_significant * (W.shape[0] + W.shape[1])
        return total_effective

    def estimate_effective_params_fisher(self, fisher_eigenvalues: np.ndarray,
                                          threshold: float = 0.01) -> int:
        """Estimate effective parameters from Fisher information spectrum."""
        normalized = fisher_eigenvalues / (np.max(fisher_eigenvalues) + 1e-12)
        return int(np.sum(normalized > threshold))

    def compute_description_length_bound(self, weights: List[np.ndarray],
                                          n_samples: int,
                                          quantization_bits: int = 8,
                                          delta: float = 0.05) -> float:
        """Bound based on description length of quantized network."""
        n_params = sum(w.size for w in weights)
        code_length = n_params * quantization_bits
        bound = np.sqrt((code_length * np.log(2) + np.log(1 / delta)) / (2 * n_samples))
        return float(min(1.0, bound))


class StabilityBound:
    """Generalization from algorithmic stability."""

    def __init__(self):
        pass

    def uniform_stability_bound(self, stability_constant: float,
                                 n_samples: int, delta: float = 0.05) -> float:
        """Bousquet-Elisseeff uniform stability bound.

        If algorithm has uniform stability beta:
        |gen_error| <= 2*beta + (4*n*beta + 1) * sqrt(log(1/delta) / (2*n))
        """
        beta = stability_constant
        n = n_samples
        bound = 2 * beta + (4 * n * beta + 1) * np.sqrt(np.log(1 / delta) / (2 * n))
        return float(min(1.0, bound))

    def sgd_stability(self, learning_rate: float, n_steps: int,
                       lipschitz: float = 1.0, n_samples: int = 1000) -> float:
        """Stability of SGD.

        beta_SGD <= 2 * L^2 * eta * T / n for convex losses
        """
        beta = 2 * lipschitz ** 2 * learning_rate * n_steps / n_samples
        return float(beta)

    def regularized_stability(self, learning_rate: float, n_steps: int,
                               reg_strength: float, lipschitz: float = 1.0,
                               n_samples: int = 1000) -> float:
        """Stability of regularized ERM.

        beta <= L^2 / (2 * lambda * n)
        """
        beta = lipschitz ** 2 / (2 * reg_strength * n_samples + 1e-12)
        return float(beta)

    def compute_empirical_stability(self, X: np.ndarray, Y: np.ndarray,
                                     train_fn: Callable,
                                     n_loo_samples: int = 20) -> float:
        """Estimate stability via leave-one-out perturbation."""
        n = len(X)
        n_loo = min(n_loo_samples, n)
        indices = np.random.choice(n, size=n_loo, replace=False)

        full_prediction = train_fn(X, Y)
        stability_values = []

        for idx in indices:
            X_loo = np.delete(X, idx, axis=0)
            Y_loo = np.delete(Y, idx, axis=0)
            loo_prediction = train_fn(X_loo, Y_loo)
            diff = np.abs(full_prediction - loo_prediction)
            stability_values.append(float(np.mean(diff)))

        return float(np.mean(stability_values))


class MarginBound:
    """Margin-based generalization bounds."""

    def __init__(self):
        pass

    def compute_margin(self, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute margins for classification."""
        if scores.ndim == 1:
            margins = scores * (2 * labels - 1)
        else:
            n_classes = scores.shape[1]
            correct_scores = scores[np.arange(len(labels)), labels.astype(int)]
            max_other = np.max(
                scores - np.eye(n_classes)[labels.astype(int)] * 1e10, axis=1
            )
            margins = correct_scores - max_other
        return margins

    def margin_bound(self, margins: np.ndarray, spectral_norms: List[float],
                     n_samples: int, delta: float = 0.05) -> float:
        """Bartlett's margin-based bound.

        gap <= O(prod spectral_norms / (gamma * sqrt(n)))
        """
        if len(margins) == 0:
            return 1.0

        gamma = float(np.percentile(margins, 10))
        if gamma <= 0:
            return 1.0

        spectral_product = np.prod(spectral_norms)
        bound = spectral_product / (gamma * np.sqrt(n_samples))
        bound += np.sqrt(np.log(1 / delta) / n_samples)
        return float(min(1.0, bound))

    def normalized_margin_bound(self, margins: np.ndarray, weight_norms: List[float],
                                 n_samples: int, depth: int) -> float:
        """Normalized margin bound from Neyshabur et al."""
        gamma = float(np.percentile(margins, 10)) if len(margins) > 0 else 1e-6
        if gamma <= 0:
            gamma = 1e-6

        product_norms = np.prod(weight_norms)
        sum_ratios = sum(w ** 2 for w in weight_norms) / (product_norms ** (2 / depth) + 1e-12)

        bound = product_norms * np.sqrt(sum_ratios) / (gamma * np.sqrt(n_samples))
        return float(min(1.0, bound))

    def compute_margin_distribution(self, scores: np.ndarray,
                                     labels: np.ndarray) -> Dict[str, float]:
        """Compute margin distribution statistics."""
        margins = self.compute_margin(scores, labels)
        return {
            "mean_margin": float(np.mean(margins)),
            "median_margin": float(np.median(margins)),
            "min_margin": float(np.min(margins)),
            "margin_10th_percentile": float(np.percentile(margins, 10)),
            "fraction_positive": float(np.mean(margins > 0)),
            "margin_std": float(np.std(margins)),
        }


class FlatnessBound:
    """Flatness-based generalization bounds."""

    def __init__(self):
        pass

    def compute_sharpness(self, loss_fn: Callable, weights: np.ndarray,
                           epsilon: float = 0.01, n_directions: int = 50) -> float:
        """Compute sharpness: max loss increase in epsilon-ball."""
        base_loss = loss_fn(weights)
        max_increase = 0.0

        for _ in range(n_directions):
            direction = np.random.randn(*weights.shape)
            direction = direction / (np.linalg.norm(direction) + 1e-12) * epsilon
            perturbed_loss = loss_fn(weights + direction)
            increase = perturbed_loss - base_loss
            max_increase = max(max_increase, increase)

        return float(max_increase)

    def compute_sharpness_aware(self, loss_fn: Callable, weights: np.ndarray,
                                 rho: float = 0.05) -> float:
        """Compute SAM-style sharpness."""
        base_loss = loss_fn(weights)
        eps = 1e-4
        grad = np.zeros_like(weights)
        flat_w = weights.ravel()

        n_params = len(flat_w)
        sample_indices = np.random.choice(n_params, size=min(100, n_params), replace=False)

        for idx in sample_indices:
            w_plus = flat_w.copy()
            w_plus[idx] += eps
            grad.ravel()[idx] = (loss_fn(w_plus.reshape(weights.shape)) - base_loss) / eps

        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            perturbation = rho * grad / grad_norm
            perturbed_loss = loss_fn(weights + perturbation)
            return float(perturbed_loss - base_loss)
        return 0.0

    def flatness_bound(self, sharpness: float, n_params: int,
                        n_samples: int, delta: float = 0.05) -> float:
        """Compute generalization bound from flatness.

        Keskar et al. / Dziugaite-Roy style:
        gap <= O(sharpness * n_params / n_samples)
        """
        bound = sharpness * np.sqrt(n_params / (n_samples + 1e-12))
        bound += np.sqrt(np.log(1 / delta) / (2 * n_samples))
        return float(min(1.0, bound))

    def pac_bayes_flatness_bound(self, sharpness: float, weight_norm: float,
                                  n_params: int, n_samples: int,
                                  train_error: float = 0.0,
                                  delta: float = 0.05) -> float:
        """PAC-Bayes bound using flatness to set prior/posterior."""
        if sharpness <= 0:
            sigma = weight_norm
        else:
            sigma = weight_norm / np.sqrt(sharpness * n_params + 1e-12)

        kl = n_params * (np.log(weight_norm / (sigma + 1e-12)) + 0.5)
        kl = max(0, kl)

        bound = train_error + np.sqrt((kl + np.log(2 * np.sqrt(n_samples) / delta)) /
                                       (2 * n_samples))
        return float(min(1.0, bound))


class DoublDescentAnalyzer:
    """Analyze double descent phenomenon."""

    def __init__(self, n_trials: int = 5):
        self.n_trials = n_trials

    def compute_interpolation_threshold(self, n_samples: int, input_dim: int) -> int:
        """Compute interpolation threshold: n_params ≈ n_samples."""
        return n_samples

    def predict_double_descent_curve(
        self, n_samples: int, input_dim: int,
        param_range: Optional[Tuple[int, int]] = None,
        noise_level: float = 0.1
    ) -> Dict[str, Any]:
        """Predict test error curve showing double descent."""
        if param_range is None:
            param_range = (input_dim, 5 * n_samples)

        n_params_values = np.unique(np.logspace(
            np.log10(param_range[0]), np.log10(param_range[1]), 50
        ).astype(int))

        test_errors = []
        train_errors = []
        regimes = []

        interpolation_threshold = n_samples

        for n_params in n_params_values:
            ratio = n_params / n_samples

            if ratio < 0.5:
                regime = "classical_underparameterized"
                train_err = max(0, noise_level ** 2 * (1 - ratio) + 0.01 / ratio)
                test_err = train_err + 0.5 / np.sqrt(n_samples)
            elif ratio < 1.0:
                regime = "approaching_interpolation"
                train_err = max(0, noise_level ** 2 * (1 - ratio) ** 2)
                test_err = noise_level ** 2 / (1 - ratio + 1e-6) + 0.1
            elif abs(ratio - 1.0) < 0.1:
                regime = "interpolation_peak"
                train_err = 0.0
                test_err = noise_level ** 2 * n_samples + 1.0
            elif ratio < 3.0:
                regime = "overparameterized_descent"
                train_err = 0.0
                test_err = noise_level ** 2 * n_samples / (n_params - n_samples + 1)
            else:
                regime = "highly_overparameterized"
                train_err = 0.0
                test_err = noise_level ** 2 / (ratio - 1 + 1e-6)

            test_errors.append(float(test_err))
            train_errors.append(float(train_err))
            regimes.append(regime)

        return {
            "n_params_values": n_params_values.tolist(),
            "test_errors": test_errors,
            "train_errors": train_errors,
            "regimes": regimes,
            "interpolation_threshold": interpolation_threshold,
            "peak_test_error": float(max(test_errors)),
            "peak_location": int(n_params_values[np.argmax(test_errors)]),
        }

    def detect_regime(self, n_params: int, n_samples: int) -> str:
        """Detect which regime of double descent we're in."""
        ratio = n_params / (n_samples + 1e-12)
        if ratio < 0.8:
            return "classical_underparameterized"
        elif ratio < 1.2:
            return "interpolation_threshold"
        elif ratio < 3.0:
            return "modern_overparameterized"
        else:
            return "highly_overparameterized"

    def epoch_wise_double_descent(
        self, n_epochs: int, n_params: int, n_samples: int,
        noise_level: float = 0.1
    ) -> Dict[str, Any]:
        """Predict epoch-wise double descent."""
        ratio = n_params / (n_samples + 1e-12)
        epochs = np.arange(1, n_epochs + 1)

        train_errors = []
        test_errors = []

        for epoch in epochs:
            effective_steps = epoch * n_samples
            convergence = 1 - np.exp(-effective_steps / (n_params + 1e-12))
            train_err = max(0, (1 - convergence) * 0.5 + noise_level ** 2 * max(0, 1 - ratio))
            test_err = train_err + noise_level ** 2 * convergence / (abs(ratio - 1) + 0.1)

            if epoch > n_epochs * 0.7 and ratio > 1.2:
                test_err *= np.exp(-0.01 * (epoch - n_epochs * 0.7))

            train_errors.append(float(train_err))
            test_errors.append(float(test_err))

        return {
            "epochs": epochs.tolist(),
            "train_errors": train_errors,
            "test_errors": test_errors,
            "has_epoch_dd": bool(ratio > 0.8 and ratio < 2.0),
        }


class GeneralizationAnalyzer:
    """Main generalization theory analyzer."""

    def __init__(self, n_samples_est: int = 10000):
        self.n_samples_est = n_samples_est
        self.vc = VCDimensionBound()
        self.rademacher = RademacherComplexity()
        self.pac_bayes = PACBayesBound()
        self.compression = CompressionBound()
        self.stability = StabilityBound()
        self.margin_analyzer = MarginBound()
        self.flatness_analyzer = FlatnessBound()
        self.dd_analyzer = DoublDescentAnalyzer()

    def analyze(self, model_spec: ModelSpec, data_spec: DataSpec) -> GenReport:
        """Full generalization analysis."""
        report = GenReport()

        if data_spec.X is None or data_spec.Y is None:
            X, Y = self._generate_synthetic_data(data_spec)
        else:
            X, Y = data_spec.X, data_spec.Y

        n = len(X)
        report.n_samples = n

        n_params = model_spec.n_params if model_spec.n_params > 0 else \
            model_spec.width ** 2 * model_spec.depth
        report.n_params = n_params

        vc_dim = self.vc.compute_vc_dim(model_spec.depth, model_spec.width, n_params)
        report.vc_bound = self.vc.compute_vc_bound(vc_dim, n)

        if model_spec.weights:
            spectral_norms = [float(svdvals(w)[0]) for w in model_spec.weights if w.ndim == 2]
            frobenius_norms = [float(norm(w, 'fro')) for w in model_spec.weights if w.ndim == 2]
        else:
            spectral_norms = [model_spec.spectral_norm if model_spec.spectral_norm > 0
                              else model_spec.sigma_w] * model_spec.depth
            frobenius_norms = [model_spec.sigma_w * np.sqrt(model_spec.width)] * model_spec.depth

        rad = self.rademacher.neural_network_rademacher(X, model_spec.depth,
                                                         spectral_norms, frobenius_norms)
        report.rademacher_bound = self.rademacher.compute_bound(rad, n)

        if model_spec.weights:
            kl = self.pac_bayes.compute_kl_from_weights(model_spec.weights)
        else:
            kl = n_params * 0.5 * np.log(model_spec.sigma_w ** 2 + 1)

        train_error = model_spec.margin if model_spec.margin > 0 else 0.01
        report.train_error = train_error
        report.pac_bayes_bound = self.pac_bayes.mcallester_bound(train_error, kl, n)

        if model_spec.weights:
            effective_params = self.compression.estimate_effective_params_svd(model_spec.weights)
        else:
            effective_params = max(1, n_params // 10)
        report.effective_dimensionality = float(effective_params)
        report.compression_bound = self.compression.compute_compression_bound(
            n_params, effective_params, n, train_error
        )

        lr = 0.01
        n_steps = 1000
        beta = self.stability.sgd_stability(lr, n_steps, 1.0, n)
        report.stability_bound = self.stability.uniform_stability_bound(beta, n)

        if model_spec.margin > 0:
            report.margin_bound = model_spec.margin
        else:
            dummy_scores = X @ np.random.randn(X.shape[1]) * model_spec.sigma_w
            margins = np.abs(dummy_scores)
            report.margin_bound = self.margin_analyzer.margin_bound(
                margins, spectral_norms, n
            )

        sharpness = model_spec.flatness if model_spec.flatness > 0 else 0.1
        weight_norm = float(np.sqrt(n_params) * model_spec.sigma_w)
        report.flatness_bound = self.flatness_analyzer.pac_bayes_flatness_bound(
            sharpness, weight_norm, n_params, n, train_error
        )

        bounds = {
            "vc": report.vc_bound,
            "rademacher": report.rademacher_bound,
            "pac_bayes": report.pac_bayes_bound,
            "compression": report.compression_bound,
            "stability": report.stability_bound,
            "margin": report.margin_bound,
            "flatness": report.flatness_bound,
        }
        report.bounds_summary = bounds

        valid_bounds = [v for v in bounds.values() if 0 < v < 1.0]
        report.predicted_gap = float(min(valid_bounds)) if valid_bounds else 1.0
        report.test_error = report.train_error + report.predicted_gap

        report.double_descent_regime = self.dd_analyzer.detect_regime(n_params, n)

        return report

    def _generate_synthetic_data(self, data_spec: DataSpec) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data."""
        n = data_spec.n_samples
        d = data_spec.input_dim
        X = np.random.randn(n, d) / np.sqrt(d)
        w = np.random.randn(d)
        Y = (X @ w + np.random.randn(n) * data_spec.noise_level > 0).astype(float)
        return X, Y

    def compare_bounds(self, model_spec: ModelSpec, data_spec: DataSpec) -> Dict[str, Any]:
        """Compare all generalization bounds."""
        report = self.analyze(model_spec, data_spec)

        sorted_bounds = sorted(report.bounds_summary.items(), key=lambda x: x[1])

        return {
            "bounds": report.bounds_summary,
            "tightest_bound": sorted_bounds[0][0] if sorted_bounds else "none",
            "tightest_value": sorted_bounds[0][1] if sorted_bounds else 1.0,
            "loosest_bound": sorted_bounds[-1][0] if sorted_bounds else "none",
            "loosest_value": sorted_bounds[-1][1] if sorted_bounds else 1.0,
            "bound_range": float(sorted_bounds[-1][1] - sorted_bounds[0][1])
                if sorted_bounds else 0.0,
        }

    def scaling_analysis(self, model_spec: ModelSpec,
                          sample_sizes: List[int]) -> Dict[str, Any]:
        """Analyze how bounds scale with sample size."""
        results = {"n_samples": [], "bounds": {}}
        for bound_name in ["vc", "rademacher", "pac_bayes", "compression",
                           "stability", "margin", "flatness"]:
            results["bounds"][bound_name] = []

        for n in sample_sizes:
            data_spec = DataSpec(n_samples=n, input_dim=10)
            report = self.analyze(model_spec, data_spec)
            results["n_samples"].append(n)
            for bound_name, value in report.bounds_summary.items():
                if bound_name in results["bounds"]:
                    results["bounds"][bound_name].append(value)

        return results
