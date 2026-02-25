"""
Regime detector for neural networks.

Detects lazy (NTK), rich (feature learning), catapult, and grokking regimes
from network architecture, initialization, and training dynamics.
"""

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from scipy.integrate import quad
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import warnings


@dataclass
class Regime:
    """Detected training regime."""
    type: str  # "lazy", "rich", "catapult", "grokking", "chaotic", "unknown"
    confidence: float  # 0.0 to 1.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    training_recommendations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpecForDetection:
    """Model specification for regime detection."""
    layer_widths: List[int]
    activation: str = "relu"
    sigma_w: float = 1.0
    sigma_b: float = 0.0
    learning_rate: float = 0.01
    batch_size: int = 32
    parameterization: str = "standard"  # "standard" or "ntk"


@dataclass
class TrainingTrace:
    """Training dynamics trace for regime detection."""
    train_losses: List[float] = field(default_factory=list)
    test_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    test_accuracies: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)
    weight_norms: List[float] = field(default_factory=list)
    feature_changes: List[float] = field(default_factory=list)
    ntk_changes: List[float] = field(default_factory=list)


class CKAComputer:
    """Centered Kernel Alignment (CKA) computation.

    CKA measures similarity between representations (feature maps).
    CKA = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    where HSIC is the Hilbert-Schmidt Independence Criterion.
    """

    @staticmethod
    def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
        """Compute linear CKA between two feature matrices.

        Args:
            X: Features from model 1, shape (n_samples, d1).
            Y: Features from model 2, shape (n_samples, d2).

        Returns:
            CKA similarity in [0, 1].
        """
        n = X.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n

        # Centering
        HX = H @ X
        HY = H @ Y

        # HSIC estimates
        hsic_xy = np.trace(HX @ HX.T @ HY @ HY.T) / (n - 1) ** 2
        hsic_xx = np.trace(HX @ HX.T @ HX @ HX.T) / (n - 1) ** 2
        hsic_yy = np.trace(HY @ HY.T @ HY @ HY.T) / (n - 1) ** 2

        denom = np.sqrt(max(hsic_xx * hsic_yy, 1e-30))
        return float(np.clip(hsic_xy / denom, 0.0, 1.0))

    @staticmethod
    def rbf_cka(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
        """Compute RBF (kernel) CKA between two feature matrices.

        Args:
            X: Features from model 1, shape (n_samples, d1).
            Y: Features from model 2, shape (n_samples, d2).
            sigma: RBF kernel bandwidth.

        Returns:
            CKA similarity in [0, 1].
        """
        n = X.shape[0]

        # RBF kernels
        dx = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
        dy = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)
        K = np.exp(-dx / (2 * sigma ** 2))
        L = np.exp(-dy / (2 * sigma ** 2))

        # Center
        H = np.eye(n) - np.ones((n, n)) / n
        K_c = H @ K @ H
        L_c = H @ L @ H

        hsic_kl = np.sum(K_c * L_c) / (n - 1) ** 2
        hsic_kk = np.sum(K_c * K_c) / (n - 1) ** 2
        hsic_ll = np.sum(L_c * L_c) / (n - 1) ** 2

        denom = np.sqrt(max(hsic_kk * hsic_ll, 1e-30))
        return float(np.clip(hsic_kl / denom, 0.0, 1.0))


class RegimeDetector:
    """Detect training regime of a neural network."""

    def __init__(self):
        self.cka = CKAComputer()

    def detect(self, model_spec: ModelSpecForDetection,
               training_trace: Optional[TrainingTrace] = None) -> Regime:
        """Detect the training regime.

        Can detect from:
        1. Architecture + initialization alone (prediction mode)
        2. Training trace data (detection mode)

        Args:
            model_spec: Model specification.
            training_trace: Optional training dynamics data.

        Returns:
            Regime with type, confidence, evidence, and recommendations.
        """
        if training_trace is not None and len(training_trace.train_losses) > 10:
            return self._detect_from_trace(model_spec, training_trace)
        else:
            return self._predict_from_architecture(model_spec)

    def _predict_from_architecture(self, spec: ModelSpecForDetection) -> Regime:
        """Predict regime from architecture and initialization without training.

        Uses theoretical indicators:
        - Width / depth ratio
        - Learning rate * width product
        - Mean field chi_1
        """
        evidence = {}

        # Width analysis
        min_width = min(spec.layer_widths[1:-1]) if len(spec.layer_widths) > 2 else spec.layer_widths[-1]
        max_width = max(spec.layer_widths[1:-1]) if len(spec.layer_widths) > 2 else spec.layer_widths[-1]
        depth = len(spec.layer_widths) - 1

        evidence["min_width"] = min_width
        evidence["depth"] = depth
        evidence["width_depth_ratio"] = min_width / max(depth, 1)

        # Learning rate * width product (lazy regime indicator)
        lr_width_product = spec.learning_rate * min_width
        evidence["lr_width_product"] = lr_width_product

        # Parametrization scaling
        if spec.parameterization == "ntk":
            # NTK parametrization: features frozen for any width
            feature_change_estimate = 1.0 / max(min_width, 1)
        else:
            # Standard parametrization: features change as ~lr * sqrt(width)
            feature_change_estimate = spec.learning_rate * np.sqrt(min_width) * spec.sigma_w ** 2

        evidence["feature_change_estimate"] = feature_change_estimate

        # Mean field chi_1 estimate
        chi_1 = self._estimate_chi1(spec)
        evidence["chi_1_estimate"] = chi_1

        # Temperature (noise level)
        temperature = spec.learning_rate / max(spec.batch_size, 1)
        evidence["temperature"] = temperature

        # Classify
        scores = {
            "lazy": 0.0,
            "rich": 0.0,
            "chaotic": 0.0,
        }

        # Lazy regime indicators
        if min_width > 1000:
            scores["lazy"] += 0.3
        if lr_width_product > 10:
            scores["lazy"] += 0.2
        if spec.parameterization == "ntk":
            scores["lazy"] += 0.4
        if feature_change_estimate > 5.0:
            scores["lazy"] += 0.1

        # Rich regime indicators
        if 10 < min_width < 1000:
            scores["rich"] += 0.3
        if 0.01 < lr_width_product < 10:
            scores["rich"] += 0.2
        if abs(chi_1 - 1.0) < 0.1:
            scores["rich"] += 0.3
        if spec.parameterization == "standard" and feature_change_estimate < 5.0:
            scores["rich"] += 0.2

        # Chaotic regime indicators
        if chi_1 > 1.5:
            scores["chaotic"] += 0.4
        if spec.learning_rate > 1.0:
            scores["chaotic"] += 0.3
        if temperature > 0.01:
            scores["chaotic"] += 0.2

        # Pick best
        best_regime = max(scores, key=scores.get)
        confidence = scores[best_regime] / max(sum(scores.values()), 1e-10)

        # Training recommendations
        recommendations = self._get_recommendations(best_regime, spec, evidence)

        return Regime(
            type=best_regime,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            evidence=evidence,
            training_recommendations=recommendations,
        )

    def _detect_from_trace(self, spec: ModelSpecForDetection,
                           trace: TrainingTrace) -> Regime:
        """Detect regime from training dynamics."""
        evidence = {}
        detectors = [
            self._detect_lazy(spec, trace, evidence),
            self._detect_rich(spec, trace, evidence),
            self._detect_catapult(spec, trace, evidence),
            self._detect_grokking(spec, trace, evidence),
        ]

        # Find highest confidence detection
        best = max(detectors, key=lambda r: r.confidence)
        best.evidence.update(evidence)
        return best

    def _detect_lazy(self, spec: ModelSpecForDetection,
                     trace: TrainingTrace, evidence: Dict) -> Regime:
        """Detect lazy (NTK) regime: NTK changes < threshold during training."""
        confidence = 0.0

        # Check NTK drift
        if trace.ntk_changes:
            max_ntk_change = max(trace.ntk_changes)
            evidence["max_ntk_change"] = max_ntk_change
            if max_ntk_change < 0.1:
                confidence += 0.4
            elif max_ntk_change < 0.3:
                confidence += 0.2

        # Check feature changes
        if trace.feature_changes:
            max_feature_change = max(trace.feature_changes)
            evidence["max_feature_change"] = max_feature_change
            if max_feature_change < 0.1:
                confidence += 0.3
            elif max_feature_change < 0.3:
                confidence += 0.15

        # Check loss curve: lazy regime has exponential decay
        if len(trace.train_losses) > 5:
            losses = np.array(trace.train_losses)
            # Check if log-loss is approximately linear (exponential decay)
            log_losses = np.log(np.maximum(losses, 1e-30))
            steps = np.arange(len(log_losses))
            if len(steps) > 2:
                A = np.vstack([steps, np.ones(len(steps))]).T
                slope, _ = np.linalg.lstsq(A, log_losses, rcond=None)[0]
                residuals = log_losses - (slope * steps + _)
                r_squared = 1.0 - np.var(residuals) / max(np.var(log_losses), 1e-30)
                evidence["exponential_decay_r2"] = float(r_squared)
                if r_squared > 0.95:
                    confidence += 0.3
                elif r_squared > 0.8:
                    confidence += 0.15

        recommendations = self._get_recommendations("lazy", spec, evidence)

        return Regime(
            type="lazy",
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            evidence=evidence.copy(),
            training_recommendations=recommendations,
        )

    def _detect_rich(self, spec: ModelSpecForDetection,
                     trace: TrainingTrace, evidence: Dict) -> Regime:
        """Detect rich (feature learning) regime."""
        confidence = 0.0

        # Check feature changes
        if trace.feature_changes:
            avg_change = np.mean(trace.feature_changes)
            evidence["avg_feature_change"] = float(avg_change)
            if avg_change > 0.3:
                confidence += 0.3
            elif avg_change > 0.1:
                confidence += 0.15

        # Check NTK changes (should be significant)
        if trace.ntk_changes:
            avg_ntk_change = np.mean(trace.ntk_changes)
            evidence["avg_ntk_change"] = float(avg_ntk_change)
            if avg_ntk_change > 0.3:
                confidence += 0.3
            elif avg_ntk_change > 0.1:
                confidence += 0.15

        # Check if loss decreases but not exponentially
        if len(trace.train_losses) > 10:
            losses = np.array(trace.train_losses)
            # Check for non-exponential decrease
            first_half = losses[:len(losses) // 2]
            second_half = losses[len(losses) // 2:]
            if np.mean(second_half) < np.mean(first_half):
                confidence += 0.2

            # Rich regime often shows faster-than-exponential convergence
            log_losses = np.log(np.maximum(losses, 1e-30))
            second_derivative = np.diff(np.diff(log_losses))
            if len(second_derivative) > 0 and np.mean(second_derivative) < 0:
                confidence += 0.2

        recommendations = self._get_recommendations("rich", spec, evidence)

        return Regime(
            type="rich",
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            evidence=evidence.copy(),
            training_recommendations=recommendations,
        )

    def _detect_catapult(self, spec: ModelSpecForDetection,
                         trace: TrainingTrace, evidence: Dict) -> Regime:
        """Detect catapult phase: loss spike followed by faster convergence.

        The catapult phase occurs when the learning rate is large enough
        to cause an initial loss spike, but the network recovers and
        converges faster than with smaller learning rate.
        """
        confidence = 0.0

        if len(trace.train_losses) < 20:
            return Regime(type="catapult", confidence=0.0, evidence=evidence.copy())

        losses = np.array(trace.train_losses)
        n = len(losses)

        # Look for spike in early training
        early_window = max(n // 10, 5)
        early_losses = losses[:early_window]
        max_early_idx = np.argmax(early_losses)
        initial_loss = losses[0]
        peak_loss = early_losses[max_early_idx]

        spike_ratio = peak_loss / max(initial_loss, 1e-10)
        evidence["spike_ratio"] = float(spike_ratio)
        evidence["spike_location"] = int(max_early_idx)

        if spike_ratio > 1.5 and max_early_idx > 0:
            confidence += 0.3

            # Check if loss recovers and drops below initial
            post_spike = losses[max_early_idx:]
            if len(post_spike) > 5:
                min_post_spike = np.min(post_spike)
                if min_post_spike < initial_loss * 0.5:
                    confidence += 0.3

                    # Check convergence speed after spike
                    recovery_steps = np.argmax(post_spike < initial_loss) if np.any(post_spike < initial_loss) else len(post_spike)
                    evidence["recovery_steps"] = int(recovery_steps)
                    if recovery_steps < len(post_spike) // 2:
                        confidence += 0.2

        # Gradient norm spike
        if trace.gradient_norms and len(trace.gradient_norms) > 10:
            gnorms = np.array(trace.gradient_norms)
            early_gnorms = gnorms[:max(len(gnorms) // 10, 5)]
            if np.max(early_gnorms) > 3 * np.median(gnorms):
                confidence += 0.2
                evidence["gradient_spike"] = True

        recommendations = self._get_recommendations("catapult", spec, evidence)

        return Regime(
            type="catapult",
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            evidence=evidence.copy(),
            training_recommendations=recommendations,
        )

    def _detect_grokking(self, spec: ModelSpecForDetection,
                         trace: TrainingTrace, evidence: Dict) -> Regime:
        """Detect grokking: delayed generalization.

        Grokking occurs when training accuracy reaches ~100% long before
        test accuracy starts improving.
        """
        confidence = 0.0

        if not trace.train_accuracies or not trace.test_accuracies:
            if trace.train_losses and trace.test_losses:
                # Use losses instead
                return self._detect_grokking_from_losses(spec, trace, evidence)
            return Regime(type="grokking", confidence=0.0, evidence=evidence.copy())

        train_acc = np.array(trace.train_accuracies)
        test_acc = np.array(trace.test_accuracies)
        n = len(train_acc)

        if n < 20:
            return Regime(type="grokking", confidence=0.0, evidence=evidence.copy())

        # Find when train accuracy is high
        high_train_threshold = 0.95
        train_high_idx = np.argmax(train_acc > high_train_threshold)
        if train_acc[train_high_idx] <= high_train_threshold:
            train_high_idx = n  # never reached

        evidence["train_high_epoch"] = int(train_high_idx)

        # Find when test accuracy starts improving significantly
        test_improve_threshold = 0.7
        test_improve_idx = np.argmax(test_acc > test_improve_threshold)
        if test_acc[test_improve_idx] <= test_improve_threshold:
            test_improve_idx = n

        evidence["test_improve_epoch"] = int(test_improve_idx)

        # Grokking: large gap between train_high and test_improve
        if train_high_idx < n and test_improve_idx < n:
            gap = test_improve_idx - train_high_idx
            gap_ratio = gap / max(n, 1)
            evidence["generalization_delay"] = int(gap)
            evidence["delay_ratio"] = float(gap_ratio)

            if gap > n * 0.3:
                confidence += 0.5
            elif gap > n * 0.1:
                confidence += 0.3

            # Check if test eventually catches up
            final_test = test_acc[-1]
            final_train = train_acc[-1]
            if final_test > 0.9 and final_train > 0.95:
                confidence += 0.3

        # Check weight norm growth (characteristic of grokking)
        if trace.weight_norms and len(trace.weight_norms) > 10:
            wnorms = np.array(trace.weight_norms)
            if wnorms[-1] < wnorms[len(wnorms) // 2] and len(wnorms) > 20:
                # Weight norm peak then decrease (regularization kicks in)
                confidence += 0.2

        recommendations = self._get_recommendations("grokking", spec, evidence)

        return Regime(
            type="grokking",
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            evidence=evidence.copy(),
            training_recommendations=recommendations,
        )

    def _detect_grokking_from_losses(self, spec: ModelSpecForDetection,
                                      trace: TrainingTrace,
                                      evidence: Dict) -> Regime:
        """Detect grokking using loss curves instead of accuracy."""
        confidence = 0.0
        train_losses = np.array(trace.train_losses)
        test_losses = np.array(trace.test_losses)
        n = len(train_losses)

        if n < 20:
            return Regime(type="grokking", confidence=0.0, evidence=evidence.copy())

        # Check for generalization gap
        gap = test_losses - train_losses
        max_gap_idx = np.argmax(gap)
        max_gap = gap[max_gap_idx]
        evidence["max_generalization_gap"] = float(max_gap)
        evidence["gap_at_epoch"] = int(max_gap_idx)

        # Grokking: gap increases then decreases
        if max_gap_idx > n * 0.2 and max_gap_idx < n * 0.8:
            late_gap = np.mean(gap[int(n * 0.8):])
            early_gap = np.mean(gap[:int(n * 0.2)])
            if max_gap > 2 * max(late_gap, early_gap, 0.01):
                confidence += 0.4

        if train_losses[-1] < 0.1 * train_losses[0] and test_losses[-1] < 0.5 * test_losses[0]:
            confidence += 0.2

        recommendations = self._get_recommendations("grokking", spec, evidence)

        return Regime(
            type="grokking",
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            evidence=evidence.copy(),
            training_recommendations=recommendations,
        )

    def lazy_regime_test(self, ntk_matrices: List[np.ndarray],
                         threshold: float = 0.1) -> Dict[str, Any]:
        """Test if network is in lazy regime by checking NTK stability.

        Args:
            ntk_matrices: NTK matrices at different training steps.
            threshold: Relative change threshold for lazy classification.

        Returns:
            Dictionary with lazy regime test results.
        """
        if len(ntk_matrices) < 2:
            return {"is_lazy": None, "error": "Need at least 2 NTK snapshots"}

        K0 = ntk_matrices[0]
        K0_norm = np.linalg.norm(K0, "fro")

        changes = []
        for K in ntk_matrices[1:]:
            rel_change = np.linalg.norm(K - K0, "fro") / max(K0_norm, 1e-30)
            changes.append(float(rel_change))

        max_change = max(changes)
        avg_change = np.mean(changes)

        return {
            "is_lazy": max_change < threshold,
            "max_relative_change": float(max_change),
            "avg_relative_change": float(avg_change),
            "changes_per_step": changes,
            "threshold": threshold,
        }

    def kernel_alignment_test(self, ntk_matrix: np.ndarray,
                               y_train: np.ndarray) -> Dict[str, Any]:
        """Test if NTK aligns with the task-relevant kernel y @ y^T.

        Good alignment indicates the NTK can efficiently solve the task.

        Args:
            ntk_matrix: NTK matrix.
            y_train: Training targets.

        Returns:
            Dictionary with alignment metrics.
        """
        n = ntk_matrix.shape[0]
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        # Target kernel
        target_kernel = y_train @ y_train.T

        # Center both kernels
        H = np.eye(n) - np.ones((n, n)) / n
        K_c = H @ ntk_matrix @ H
        T_c = H @ target_kernel @ H

        # Alignment
        numer = np.sum(K_c * T_c)
        denom = np.sqrt(np.sum(K_c ** 2) * np.sum(T_c ** 2))
        alignment = numer / max(denom, 1e-30)

        # Fraction of target variance captured by top-k eigenmodes of NTK
        evals, evecs = np.linalg.eigh(ntk_matrix + 1e-8 * np.eye(n))
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        target_projections = (evecs.T @ y_train) ** 2
        total_target_norm = np.sum(target_projections)
        cumulative = np.cumsum(target_projections.sum(axis=1)) / max(total_target_norm, 1e-30)

        # Find effective dimensionality: how many modes to capture 90% of target
        eff_dim = np.searchsorted(cumulative, 0.9) + 1

        return {
            "alignment": float(alignment),
            "effective_target_dim": int(eff_dim),
            "target_variance_in_top_modes": cumulative[:min(10, n)].tolist(),
            "is_well_aligned": alignment > 0.3,
        }

    def feature_learning_metric(self, features_init: np.ndarray,
                                 features_trained: np.ndarray) -> Dict[str, Any]:
        """Compute feature learning metric using CKA.

        Args:
            features_init: Features at initialization, shape (n_samples, d).
            features_trained: Features after training, shape (n_samples, d).

        Returns:
            Dictionary with feature learning metrics.
        """
        cka_linear = self.cka.linear_cka(features_init, features_trained)

        # Feature change magnitude
        diff = features_trained - features_init
        relative_change = np.linalg.norm(diff) / max(np.linalg.norm(features_init), 1e-30)

        # Per-sample change
        per_sample_change = np.linalg.norm(diff, axis=1) / np.maximum(
            np.linalg.norm(features_init, axis=1), 1e-30
        )

        return {
            "cka_similarity": float(cka_linear),
            "feature_change": float(1.0 - cka_linear),
            "relative_norm_change": float(relative_change),
            "mean_per_sample_change": float(np.mean(per_sample_change)),
            "max_per_sample_change": float(np.max(per_sample_change)),
            "is_feature_learning": cka_linear < 0.8,
        }

    def _estimate_chi1(self, spec: ModelSpecForDetection) -> float:
        """Estimate chi_1 from model spec."""
        sw = spec.sigma_w
        if spec.activation == "relu":
            chi_1 = sw ** 2 * 0.5  # ReLU: E[relu'(z)^2] = 1/2
        elif spec.activation in ("tanh", "erf"):
            # Find fixed point first
            q = 1.0
            for _ in range(100):
                def integrand(z):
                    return np.tanh(np.sqrt(q) * z) ** 2 * np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
                V, _ = quad(integrand, -6, 6)
                q = sw ** 2 * V + spec.sigma_b ** 2
                if q > 100:
                    break

            def chi_integrand(z):
                t = np.tanh(np.sqrt(q) * z)
                return (1 - t ** 2) ** 2 * np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
            chi_val, _ = quad(chi_integrand, -6, 6)
            chi_1 = sw ** 2 * chi_val
        elif spec.activation == "linear":
            chi_1 = sw ** 2
        else:
            chi_1 = sw ** 2 * 0.5
        return chi_1

    def _get_recommendations(self, regime: str, spec: ModelSpecForDetection,
                              evidence: Dict) -> Dict[str, Any]:
        """Get training recommendations based on detected regime."""
        recs = {}

        if regime == "lazy":
            recs["summary"] = "Network in lazy/NTK regime. Features frozen."
            recs["width_advice"] = "Reduce width to enable feature learning."
            recs["lr_advice"] = "Current LR is fine for convergence."
            recs["depth_advice"] = "Depth has diminishing returns in lazy regime."
            if spec.parameterization == "ntk":
                recs["parameterization"] = "Switch to standard parameterization for feature learning."

        elif regime == "rich":
            recs["summary"] = "Network in rich/feature learning regime."
            recs["lr_advice"] = "Use learning rate warmup for stability."
            recs["width_advice"] = "Current width is appropriate."
            recs["regularization"] = "Consider weight decay or dropout."

        elif regime == "catapult":
            recs["summary"] = "Catapult phase detected. Loss spike is normal."
            recs["lr_advice"] = "LR may be slightly too large. Consider reducing by 2x."
            recs["patience"] = "Wait for loss to recover — convergence may be faster."

        elif regime == "grokking":
            recs["summary"] = "Grokking detected. Delayed generalization."
            recs["training_time"] = "Train longer — generalization may still improve."
            recs["regularization"] = "Weight decay is critical for grokking."
            recs["data_advice"] = "Consider data augmentation."

        elif regime == "chaotic":
            recs["summary"] = "Network in chaotic regime. Training may be unstable."
            recs["initialization"] = "Use edge-of-chaos initialization."
            recs["lr_advice"] = "Reduce learning rate significantly."
            recs["architecture"] = "Add residual connections or batch normalization."

        else:
            recs["summary"] = "Regime unclear."
            recs["advice"] = "Monitor training dynamics for more information."

        return recs

    def predict_regime(self, spec: ModelSpecForDetection) -> Regime:
        """Predict regime from architecture + initialization without training.

        This is a convenience wrapper around detect() without a trace.
        """
        return self.detect(spec, training_trace=None)
