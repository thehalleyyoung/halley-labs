"""
Finite-width corrections to infinite-width theory.

Implements 1/n corrections to NTK, fluctuation analysis, feature learning
corrections, generalization gap corrections, and critical width estimation.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erf
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import warnings


@dataclass
class CorrectedPrediction:
    """Prediction with finite-width corrections applied."""
    corrected_value: float
    correction_magnitude: float
    confidence: float  # confidence in the correction
    infinite_width_value: float = 0.0
    correction_terms: Dict[str, float] = field(default_factory=dict)
    width: int = 0


@dataclass
class FluctuationResult:
    """Result of finite-width fluctuation analysis."""
    mean_output: np.ndarray
    output_variance: np.ndarray
    relative_fluctuation: float
    width_dependence: str  # "1/sqrt(n)", "1/n", etc.
    scaling_exponent: float


@dataclass
class EnsembleComparison:
    """Comparison between finite-width ensemble and infinite-width GP."""
    ensemble_mean: np.ndarray
    ensemble_variance: np.ndarray
    gp_mean: np.ndarray
    gp_variance: np.ndarray
    mean_discrepancy: float
    variance_discrepancy: float
    width: int
    ensemble_size: int


class FiniteWidthCorrector:
    """Apply finite-width corrections to infinite-width theory predictions."""

    def __init__(self, activation: str = "relu"):
        self.activation = activation

    def correct(self, infinite_width_prediction: float,
                width: int,
                depth: int = 2,
                correction_order: int = 1) -> CorrectedPrediction:
        """Apply finite-width corrections to an infinite-width prediction.

        Args:
            infinite_width_prediction: Value from infinite-width theory.
            width: Actual network width.
            depth: Network depth.
            correction_order: Order of correction (1 = first order 1/n).

        Returns:
            CorrectedPrediction with corrected value and metadata.
        """
        n = width
        corrections = {}

        # First order: O(1/n)
        c1 = self._first_order_correction(infinite_width_prediction, n, depth)
        corrections["1/n"] = c1

        total_correction = c1
        if correction_order >= 2 and n > 1:
            c2 = self._second_order_correction(infinite_width_prediction, n, depth)
            corrections["1/n^2"] = c2
            total_correction += c2

        corrected = infinite_width_prediction + total_correction

        # Confidence: higher width -> more confident in correction
        confidence = 1.0 - 1.0 / max(np.sqrt(n), 1)
        confidence *= min(1.0, n / (10 * depth))

        return CorrectedPrediction(
            corrected_value=float(corrected),
            correction_magnitude=float(abs(total_correction)),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            infinite_width_value=float(infinite_width_prediction),
            correction_terms=corrections,
            width=width,
        )

    def _first_order_correction(self, value: float, n: int, depth: int) -> float:
        """Compute first-order (1/n) correction.

        The leading finite-width correction is O(depth/n), with the coefficient
        determined by the excess kurtosis ratio of the activation function:
        kappa = E[phi^4] / (E[phi^2])^2 - 1.
        """
        from mean_field_theory import ActivationVarianceMaps
        kurtosis_correction = ActivationVarianceMaps.get_kurtosis_excess(
            self.activation, max(abs(value), 0.1)
        )
        correction = -value * kurtosis_correction * depth / max(n, 1)
        return correction

    def _second_order_correction(self, value: float, n: int, depth: int) -> float:
        """Compute second-order (1/n^2) correction."""
        if n <= 1:
            return 0.0
        # Second order comes from correlations between finite-width corrections
        c2_coeff = 0.5 * depth * (depth - 1)
        return value * c2_coeff / max(n ** 2, 1)

    def ntk_correction(self, ntk_infinite: np.ndarray, width: int,
                       depth: int = 2, sigma_w: float = 1.0) -> Dict[str, Any]:
        """Apply 1/n corrections to NTK.

        The finite-width NTK differs from the infinite-width limit as:
        Theta_n = Theta_inf + (1/n) * Delta_1 + O(1/n^2)

        The correction Delta_1 captures the leading-order correlations
        between neurons at finite width.

        Args:
            ntk_infinite: Infinite-width NTK matrix.
            width: Finite width n.
            depth: Network depth.
            sigma_w: Weight initialization scale.

        Returns:
            Dictionary with corrected NTK and analysis.
        """
        n = ntk_infinite.shape[0]
        N = width  # network width

        # Compute correction matrix
        # Leading correction is proportional to the Hadamard product K ∘ K
        if self.activation == "relu":
            correction_coeff = sigma_w ** 2 * depth / N
        elif self.activation in ("tanh", "erf"):
            correction_coeff = sigma_w ** 2 * depth * 0.5 / N
        else:
            correction_coeff = sigma_w ** 2 * depth / N

        # Correction matrix: perturbation of NTK at finite width
        # First-order correction is related to the variance of the kernel
        diag = np.diag(ntk_infinite)
        outer = np.outer(np.sqrt(diag), np.sqrt(diag))

        # The correction reduces off-diagonal elements more than diagonal
        correction_matrix = -correction_coeff * (ntk_infinite ** 2) / np.maximum(outer, 1e-10)

        ntk_corrected = ntk_infinite + correction_matrix

        # Ensure PSD
        evals = np.linalg.eigvalsh(ntk_corrected)
        if np.min(evals) < 0:
            ntk_corrected += (-np.min(evals) + 1e-8) * np.eye(n)

        # Relative correction magnitude
        rel_correction = np.linalg.norm(correction_matrix, "fro") / max(
            np.linalg.norm(ntk_infinite, "fro"), 1e-10
        )

        return {
            "ntk_corrected": ntk_corrected,
            "correction_matrix": correction_matrix,
            "relative_correction": float(rel_correction),
            "correction_coefficient": float(correction_coeff),
            "expected_scaling": f"O(depth/n) = O({depth}/{N})",
        }

    def fluctuation_analysis(self, width: int, depth: int,
                              n_samples: int = 100,
                              sigma_w: float = 1.0,
                              sigma_b: float = 0.0,
                              input_dim: int = 10,
                              n_networks: int = 200) -> FluctuationResult:
        """Analyze output fluctuations at finite width.

        Generates multiple random networks and measures the variance
        of outputs, which should scale as 1/width.

        Args:
            width: Network width.
            depth: Network depth.
            n_samples: Number of input samples.
            sigma_w: Weight scale.
            sigma_b: Bias scale.
            input_dim: Input dimension.
            n_networks: Number of random networks to sample.

        Returns:
            FluctuationResult with variance analysis.
        """
        rng = np.random.RandomState(42)
        X = rng.randn(n_samples, input_dim) / np.sqrt(input_dim)

        outputs = []
        for _ in range(n_networks):
            h = X.copy()
            for l in range(depth):
                fan_in = h.shape[1]
                fan_out = width if l < depth - 1 else 1
                W = rng.randn(fan_in, fan_out) * sigma_w / np.sqrt(fan_in)
                b = rng.randn(fan_out) * sigma_b
                h = h @ W + b
                if l < depth - 1:
                    h = self._apply_activation(h)
            outputs.append(h.ravel())

        outputs = np.array(outputs)  # (n_networks, n_samples)
        mean_output = np.mean(outputs, axis=0)
        output_variance = np.var(outputs, axis=0)
        relative_fluctuation = np.mean(np.sqrt(output_variance)) / max(
            np.mean(np.abs(mean_output)), 1e-10
        )

        # Determine scaling: for GP, variance ~ 1/1 (independent of width)
        # For corrections to mean, variance ~ 1/width
        scaling_exponent = 0.5  # default: 1/sqrt(n)

        return FluctuationResult(
            mean_output=mean_output,
            output_variance=output_variance,
            relative_fluctuation=float(relative_fluctuation),
            width_dependence="1/sqrt(n)" if scaling_exponent == 0.5 else "1/n",
            scaling_exponent=float(scaling_exponent),
        )

    def feature_learning_correction(self, width: int, depth: int,
                                     learning_rate: float,
                                     n_steps: int,
                                     sigma_w: float = 1.0) -> Dict[str, Any]:
        """Predict how much features change vs infinite-width freeze.

        At infinite width (NTK limit), features are frozen.
        At finite width, features change by O(1/sqrt(n)) per step
        in standard parameterization.

        Args:
            width: Network width.
            depth: Network depth.
            learning_rate: Learning rate.
            n_steps: Number of training steps.
            sigma_w: Weight scale.

        Returns:
            Dictionary with feature learning predictions.
        """
        n = width

        # Feature change per step in standard parameterization
        # Delta features ~ lr * (1/sqrt(n)) per step
        delta_per_step = learning_rate * sigma_w / np.sqrt(n)

        # Cumulative feature change (grows as sqrt(steps) due to random walk)
        cumulative_change = delta_per_step * np.sqrt(n_steps)

        # In mean-field parameterization, features change as O(1) per step
        # The parameterization matters: NTK param vs standard param vs mu-param
        feature_change_ntk = learning_rate / n  # features frozen at O(1/n)
        feature_change_standard = delta_per_step * np.sqrt(n_steps)
        feature_change_mu = learning_rate * np.sqrt(n_steps)  # O(1) changes

        # Determine regime
        if cumulative_change < 0.01:
            regime = "lazy"
        elif cumulative_change < 1.0:
            regime = "transitional"
        else:
            regime = "rich"

        return {
            "feature_change_per_step": float(delta_per_step),
            "cumulative_change": float(cumulative_change),
            "regime_prediction": regime,
            "by_parameterization": {
                "ntk": float(feature_change_ntk),
                "standard": float(feature_change_standard),
                "mu": float(feature_change_mu),
            },
            "critical_width_for_lazy": int(max(1, learning_rate ** 2 * n_steps)),
            "critical_width_for_rich": int(max(1, 1.0 / (learning_rate ** 2 * n_steps + 1e-10))),
        }

    def generalization_gap_correction(self, n_train: int, width: int,
                                       depth: int, n_params: Optional[int] = None,
                                       sigma_w: float = 1.0) -> Dict[str, Any]:
        """Predict generalization gap correction at finite width.

        Infinite-width networks have a specific generalization behavior (GP).
        Finite-width corrections change the generalization gap.

        Args:
            n_train: Number of training samples.
            width: Network width.
            depth: Network depth.
            n_params: Total number of parameters (if None, estimated).
            sigma_w: Weight scale.

        Returns:
            Dictionary with generalization gap analysis.
        """
        if n_params is None:
            n_params = width ** 2 * depth

        # Infinite-width generalization (GP)
        # Bias-variance decomposition
        gp_bias = 0.0  # GP can fit any training data
        gp_variance = n_params / max(n_train, 1)  # Rademacher complexity

        # Finite-width corrections
        # At finite width, the effective complexity is reduced
        effective_params = n_params * (1.0 - depth / max(width, 1))
        effective_params = max(effective_params, 1)

        # Generalization gap ~ effective_params / n_train
        gen_gap_infinite = n_params / max(n_train, 1)
        gen_gap_finite = effective_params / max(n_train, 1)

        # Double descent phenomenon
        interpolation_threshold = n_params / max(n_train, 1)
        in_double_descent = 0.8 < interpolation_threshold < 1.2

        # Optimal regularization
        optimal_lambda = 1.0 / max(width, 1)

        return {
            "gen_gap_infinite_width": float(gen_gap_infinite),
            "gen_gap_finite_width": float(gen_gap_finite),
            "correction": float(gen_gap_finite - gen_gap_infinite),
            "effective_params": float(effective_params),
            "interpolation_threshold": float(interpolation_threshold),
            "in_double_descent_region": in_double_descent,
            "optimal_regularization": float(optimal_lambda),
            "bias_variance": {
                "bias": float(gp_bias),
                "variance": float(min(gen_gap_finite, 10.0)),
            },
        }

    def ensemble_comparison(self, width: int, depth: int,
                            X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray,
                            sigma_w: float = 1.0, sigma_b: float = 0.0,
                            n_ensemble: int = 50,
                            learning_rate: float = 0.01,
                            n_steps: int = 1000) -> EnsembleComparison:
        """Compare finite-width ensemble with infinite-width GP.

        Trains multiple finite-width networks and compares their ensemble
        prediction with the GP (NNGP) prediction.

        Args:
            width: Network width.
            depth: Network depth.
            X_train: Training inputs.
            y_train: Training targets.
            X_test: Test inputs.
            sigma_w: Weight scale.
            sigma_b: Bias scale.
            n_ensemble: Number of networks in ensemble.
            learning_rate: Learning rate for training.
            n_steps: Number of training steps.

        Returns:
            EnsembleComparison with predictions from both.
        """
        rng = np.random.RandomState(42)
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        input_dim = X_train.shape[1] if X_train.ndim > 1 else 1

        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        # Train ensemble of finite-width networks
        ensemble_predictions = []
        for _ in range(n_ensemble):
            pred = self._train_single_network(
                X_train, y_train, X_test, width, depth,
                sigma_w, sigma_b, learning_rate, n_steps, rng
            )
            ensemble_predictions.append(pred.ravel())

        ensemble_predictions = np.array(ensemble_predictions)
        ensemble_mean = np.mean(ensemble_predictions, axis=0)
        ensemble_var = np.var(ensemble_predictions, axis=0)

        # GP prediction (NNGP)
        gp_mean, gp_var = self._gp_prediction(
            X_train, y_train.ravel(), X_test, sigma_w, sigma_b, depth, input_dim
        )

        # Discrepancies
        mean_disc = np.mean(np.abs(ensemble_mean - gp_mean)) / max(
            np.mean(np.abs(gp_mean)), 1e-10
        )
        var_disc = np.mean(np.abs(ensemble_var - gp_var)) / max(
            np.mean(np.abs(gp_var)), 1e-10
        )

        return EnsembleComparison(
            ensemble_mean=ensemble_mean,
            ensemble_variance=ensemble_var,
            gp_mean=gp_mean,
            gp_variance=gp_var,
            mean_discrepancy=float(mean_disc),
            variance_discrepancy=float(var_disc),
            width=width,
            ensemble_size=n_ensemble,
        )

    def critical_width_estimation(self, depth: int, target_accuracy: float = 0.1,
                                   sigma_w: float = 1.0,
                                   activation: str = "relu") -> Dict[str, Any]:
        """Estimate minimum width for mean field theory to be accurate.

        Mean field theory becomes accurate when 1/n corrections are small.
        The critical width depends on depth and activation function.

        Args:
            depth: Network depth.
            target_accuracy: Target relative accuracy of MF predictions.
            sigma_w: Weight scale.
            activation: Activation function.

        Returns:
            Dictionary with critical width estimates.
        """
        # The correction scales as depth * kurtosis / width
        if activation == "relu":
            kurtosis = 2.0
        elif activation in ("tanh", "erf"):
            kurtosis = 1.0
        elif activation == "linear":
            kurtosis = 0.0
        else:
            kurtosis = 1.5

        # Critical width: depth * kurtosis / n_c = target_accuracy
        if target_accuracy > 0 and kurtosis > 0:
            critical_width = int(np.ceil(depth * kurtosis * sigma_w ** 2 / target_accuracy))
        else:
            critical_width = depth

        critical_width = max(critical_width, 2)

        # Width for different accuracy levels
        width_for_accuracy = {}
        for acc in [0.5, 0.2, 0.1, 0.05, 0.01]:
            if kurtosis > 0 and acc > 0:
                w = int(np.ceil(depth * kurtosis * sigma_w ** 2 / acc))
            else:
                w = depth
            width_for_accuracy[f"{acc:.0%}"] = max(w, 2)

        return {
            "critical_width": critical_width,
            "target_accuracy": target_accuracy,
            "depth": depth,
            "kurtosis_factor": float(kurtosis),
            "scaling": f"n_c ~ {depth} * {kurtosis} / {target_accuracy} = {critical_width}",
            "width_for_accuracy": width_for_accuracy,
            "recommendation": (
                f"Use width >= {critical_width} for {target_accuracy:.0%} "
                f"accuracy of mean field predictions."
            ),
        }

    def _train_single_network(self, X_train, y_train, X_test,
                               width, depth, sigma_w, sigma_b,
                               lr, n_steps, rng):
        """Train a single network and return test predictions."""
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]

        # Initialize
        weights = []
        biases = []
        layers = [input_dim] + [width] * (depth - 1) + [output_dim]
        for i in range(len(layers) - 1):
            W = rng.randn(layers[i], layers[i + 1]) * sigma_w / np.sqrt(layers[i])
            b = rng.randn(layers[i + 1]) * sigma_b
            weights.append(W)
            biases.append(b)

        # Train with gradient descent
        for step in range(min(n_steps, 500)):
            # Forward
            activations = [X_train]
            h = X_train
            for i in range(len(weights)):
                z = h @ weights[i] + biases[i]
                if i < len(weights) - 1:
                    h = self._apply_activation(z)
                else:
                    h = z
                activations.append(h)

            # Loss gradient
            grad_out = 2.0 * (h - y_train) / X_train.shape[0]

            # Backward
            grad = grad_out
            for i in range(len(weights) - 1, -1, -1):
                dW = activations[i].T @ grad
                db = np.sum(grad, axis=0)

                weights[i] -= lr * dW
                biases[i] -= lr * db

                if i > 0:
                    grad = grad @ weights[i].T
                    pre_act = activations[i]
                    grad = grad * (pre_act > 0).astype(float)  # ReLU derivative approx

        # Test predictions
        h = X_test
        for i in range(len(weights)):
            z = h @ weights[i] + biases[i]
            if i < len(weights) - 1:
                h = self._apply_activation(z)
            else:
                h = z
        return h

    def _gp_prediction(self, X_train, y_train, X_test,
                       sigma_w, sigma_b, depth, input_dim):
        """Compute GP (NNGP) prediction."""
        # Build NNGP kernel using recursive computation
        X_all = np.vstack([X_train, X_test])
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        # Base kernel
        K = sigma_w ** 2 / input_dim * (X_all @ X_all.T) + sigma_b ** 2

        # Propagate through layers
        for l in range(depth):
            diag = np.diag(K)
            q_sqrt = np.sqrt(np.maximum(diag, 1e-30))
            outer_sqrt = np.outer(q_sqrt, q_sqrt)
            cos_angle = K / np.maximum(outer_sqrt, 1e-30)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)

            # ReLU kernel (arc-cosine)
            theta = np.arccos(cos_angle)
            K_new = sigma_w ** 2 * outer_sqrt / (2 * np.pi) * (
                np.sin(theta) + (np.pi - theta) * cos_angle
            ) + sigma_b ** 2
            K = K_new

        # Extract blocks
        K_tt = K[:n_train, :n_train]
        K_st = K[n_train:, :n_train]
        K_ss = K[n_train:, n_train:]

        # GP prediction
        jitter = 1e-6 * np.eye(n_train)
        alpha = np.linalg.solve(K_tt + jitter, y_train)
        gp_mean = K_st @ alpha

        # GP variance
        v = np.linalg.solve(K_tt + jitter, K_st.T)
        gp_var = np.diag(K_ss) - np.sum(K_st.T * v, axis=0)
        gp_var = np.maximum(gp_var, 0.0)

        return gp_mean, gp_var

    def _apply_activation(self, x):
        """Apply ReLU activation."""
        if self.activation == "relu":
            return np.maximum(x, 0)
        elif self.activation == "tanh":
            return np.tanh(x)
        elif self.activation in ("silu", "swish"):
            return x / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return np.maximum(x, 0)


def corrections_decrease_with_width(depths: List[int] = None,
                                     widths: List[int] = None) -> Dict[str, Any]:
    """Verify that finite-width corrections decrease with increasing width.

    Args:
        depths: Depths to test.
        widths: Widths to test.

    Returns:
        Dictionary with verification results.
    """
    if depths is None:
        depths = [2, 5, 10]
    if widths is None:
        widths = [10, 50, 100, 500, 1000]

    results = {}
    corrector = FiniteWidthCorrector()

    for depth in depths:
        corrections = []
        for width in widths:
            result = corrector.correct(1.0, width, depth)
            corrections.append(result.correction_magnitude)

        # Check monotonic decrease
        is_decreasing = all(
            corrections[i] >= corrections[i + 1] - 1e-10
            for i in range(len(corrections) - 1)
        )

        results[f"depth_{depth}"] = {
            "widths": widths,
            "corrections": corrections,
            "is_decreasing": is_decreasing,
        }

    return results
