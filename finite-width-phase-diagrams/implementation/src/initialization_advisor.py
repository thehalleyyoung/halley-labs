"""
Initialization advisor for neural networks.

Recommends initialization strategies (Kaiming, Xavier, orthogonal, LSUV, Fixup)
and verifies gradient flow through the network.
"""

import numpy as np
from scipy.linalg import qr, svd
from scipy.integrate import quad
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import warnings


@dataclass
class InitRecommendation:
    """Recommendation for network initialization."""
    method: str  # "kaiming", "xavier", "orthogonal", "lsuv", "fixup"
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_gradient_magnitude: float = 1.0
    stability_score: float = 0.0  # 0.0 to 1.0
    explanation: str = ""
    per_layer_config: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class NetworkArchitecture:
    """Architecture specification for initialization."""
    layer_widths: List[int]
    activation: str = "relu"
    has_residual: bool = False
    has_batchnorm: bool = False
    bias: bool = True
    output_activation: str = "linear"


@dataclass
class GradientFlowReport:
    """Report from gradient flow verification."""
    layer_activation_stds: List[float]
    layer_gradient_stds: List[float]
    activation_variance_ratio: float  # last/first layer std ratio
    gradient_variance_ratio: float  # first/last layer gradient std ratio
    is_healthy: bool
    issues: List[str] = field(default_factory=list)


class ActivationGains:
    """Compute correct initialization gains for each activation function.

    The gain is chosen so that Var[activation(W @ x)] = Var[x]
    given weights W initialized appropriately.
    """

    # Gains derived from edge-of-chaos analysis: σ_w* such that χ₁ = 1.
    # For ReLU: σ_w* = √2 (closed-form). For others: Brent + quadrature.
    GAINS = {
        "relu": np.sqrt(2.0),
        "leaky_relu": lambda negative_slope=0.01: np.sqrt(2.0 / (1 + negative_slope ** 2)),
        "tanh": 1.006,  # edge-of-chaos via Brent + quadrature
        "sigmoid": 1.0,
        "linear": 1.0,
        "gelu": 1.534,  # edge-of-chaos via Brent + quadrature
        "silu": 1.677,  # edge-of-chaos via Brent + quadrature
        "swish": 1.677,  # same as SiLU
        "elu": np.sqrt(1.55),
        "selu": 1.0,  # SELU has its own init scheme
    }

    @classmethod
    def get_gain(cls, activation: str, **kwargs) -> float:
        """Get initialization gain for an activation function.

        Args:
            activation: Activation function name.
            **kwargs: Additional parameters (e.g., negative_slope for leaky_relu).

        Returns:
            Gain value.
        """
        gain = cls.GAINS.get(activation, 1.0)
        if callable(gain):
            return float(gain(**kwargs))
        return float(gain)

    @classmethod
    def compute_gain_numerically(cls, activation: str, n_samples: int = 100000) -> float:
        """Compute gain numerically by measuring variance preservation.

        Args:
            activation: Activation function name.
            n_samples: Number of samples for Monte Carlo estimation.

        Returns:
            Computed gain.
        """
        rng = np.random.RandomState(42)
        z = rng.randn(n_samples)

        if activation == "relu":
            out = np.maximum(z, 0)
        elif activation == "tanh":
            out = np.tanh(z)
        elif activation == "sigmoid":
            out = 1.0 / (1.0 + np.exp(-z))
        elif activation == "gelu":
            from scipy.special import erf
            out = 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))
        elif activation in ("silu", "swish"):
            out = z / (1.0 + np.exp(-z))
        elif activation == "elu":
            out = np.where(z > 0, z, np.exp(z) - 1)
        elif activation == "linear":
            out = z
        else:
            out = np.maximum(z, 0)  # default to ReLU

        var_out = np.var(out)
        if var_out < 1e-10:
            return 1.0
        return float(np.sqrt(1.0 / var_out))


class InitAdvisor:
    """Recommend initialization strategies for neural networks."""

    def __init__(self):
        self.gains = ActivationGains()

    def recommend(self, architecture: NetworkArchitecture) -> InitRecommendation:
        """Recommend initialization strategy for the given architecture.

        Args:
            architecture: Network architecture specification.

        Returns:
            InitRecommendation with method and parameters.
        """
        if architecture.has_residual:
            return self._recommend_fixup(architecture)
        elif architecture.has_batchnorm:
            return self._recommend_kaiming(architecture)
        elif len(architecture.layer_widths) > 10:
            return self._recommend_orthogonal(architecture)
        elif architecture.activation in ("tanh", "sigmoid", "linear"):
            return self._recommend_xavier(architecture)
        else:
            return self._recommend_kaiming(architecture)

    def _recommend_kaiming(self, arch: NetworkArchitecture) -> InitRecommendation:
        """Kaiming/He initialization for ReLU-like activations.

        W ~ N(0, 2 / fan_in) for "fan_in" mode (default).
        This preserves variance in forward pass for ReLU.
        """
        gain = self.gains.get_gain(arch.activation)
        per_layer = []

        for i in range(len(arch.layer_widths) - 1):
            fan_in = arch.layer_widths[i]
            fan_out = arch.layer_widths[i + 1]

            std = gain / np.sqrt(fan_in)

            layer_config = {
                "layer": i,
                "fan_in": fan_in,
                "fan_out": fan_out,
                "weight_std": float(std),
                "weight_distribution": "normal",
                "bias_init": "zeros",
            }
            per_layer.append(layer_config)

        # Verify gradient flow
        grad_report = self._simulate_gradient_flow(arch, per_layer)

        return InitRecommendation(
            method="kaiming",
            parameters={
                "mode": "fan_in",
                "nonlinearity": arch.activation,
                "gain": float(gain),
            },
            expected_gradient_magnitude=float(
                np.mean(grad_report.layer_gradient_stds) if grad_report.layer_gradient_stds else 1.0
            ),
            stability_score=float(1.0 if grad_report.is_healthy else 0.5),
            explanation=f"Kaiming init with gain={gain:.3f} for {arch.activation}. "
                        f"Preserves variance in forward pass.",
            per_layer_config=per_layer,
        )

    def _recommend_xavier(self, arch: NetworkArchitecture) -> InitRecommendation:
        """Xavier/Glorot initialization for symmetric activations.

        W ~ N(0, 2 / (fan_in + fan_out)).
        Preserves variance in both forward and backward pass.
        """
        per_layer = []

        for i in range(len(arch.layer_widths) - 1):
            fan_in = arch.layer_widths[i]
            fan_out = arch.layer_widths[i + 1]

            std = np.sqrt(2.0 / (fan_in + fan_out))

            layer_config = {
                "layer": i,
                "fan_in": fan_in,
                "fan_out": fan_out,
                "weight_std": float(std),
                "weight_distribution": "normal",
                "bias_init": "zeros",
            }
            per_layer.append(layer_config)

        grad_report = self._simulate_gradient_flow(arch, per_layer)

        return InitRecommendation(
            method="xavier",
            parameters={
                "mode": "fan_avg",
                "distribution": "normal",
            },
            expected_gradient_magnitude=float(
                np.mean(grad_report.layer_gradient_stds) if grad_report.layer_gradient_stds else 1.0
            ),
            stability_score=float(1.0 if grad_report.is_healthy else 0.5),
            explanation="Xavier/Glorot init. Balances forward and backward variance.",
            per_layer_config=per_layer,
        )

    def _recommend_orthogonal(self, arch: NetworkArchitecture) -> InitRecommendation:
        """Orthogonal initialization for deep networks.

        Weight matrices are initialized as (scaled) orthogonal matrices.
        Preserves norm of activations exactly at initialization.
        """
        gain = self.gains.get_gain(arch.activation)
        per_layer = []

        for i in range(len(arch.layer_widths) - 1):
            fan_in = arch.layer_widths[i]
            fan_out = arch.layer_widths[i + 1]

            layer_config = {
                "layer": i,
                "fan_in": fan_in,
                "fan_out": fan_out,
                "weight_std": float(gain),
                "weight_distribution": "orthogonal",
                "gain": float(gain),
                "bias_init": "zeros",
            }
            per_layer.append(layer_config)

        grad_report = self._simulate_gradient_flow(arch, per_layer)

        return InitRecommendation(
            method="orthogonal",
            parameters={
                "gain": float(gain),
                "nonlinearity": arch.activation,
            },
            expected_gradient_magnitude=float(
                np.mean(grad_report.layer_gradient_stds) if grad_report.layer_gradient_stds else 1.0
            ),
            stability_score=float(1.0 if grad_report.is_healthy else 0.6),
            explanation=f"Orthogonal init with gain={gain:.3f}. "
                        f"Preserves norm exactly for linear networks.",
            per_layer_config=per_layer,
        )

    def _recommend_fixup(self, arch: NetworkArchitecture) -> InitRecommendation:
        """Fixup initialization for residual networks.

        Scales residual branch weights by 1/sqrt(num_residual_blocks)
        to prevent variance explosion in deep residual networks.
        """
        depth = len(arch.layer_widths) - 1
        num_blocks = max(depth // 2, 1)
        gain = self.gains.get_gain(arch.activation)
        residual_scale = 1.0 / np.sqrt(num_blocks)

        per_layer = []
        for i in range(len(arch.layer_widths) - 1):
            fan_in = arch.layer_widths[i]
            fan_out = arch.layer_widths[i + 1]

            # For residual blocks, scale by 1/sqrt(L)
            if i > 0 and i < depth - 1:
                std = gain / np.sqrt(fan_in) * residual_scale
            else:
                std = gain / np.sqrt(fan_in)

            layer_config = {
                "layer": i,
                "fan_in": fan_in,
                "fan_out": fan_out,
                "weight_std": float(std),
                "weight_distribution": "normal",
                "residual_scale": float(residual_scale) if 0 < i < depth - 1 else 1.0,
                "bias_init": "zeros",
            }
            per_layer.append(layer_config)

        grad_report = self._simulate_gradient_flow(arch, per_layer)

        return InitRecommendation(
            method="fixup",
            parameters={
                "num_blocks": num_blocks,
                "residual_scale": float(residual_scale),
                "gain": float(gain),
            },
            expected_gradient_magnitude=float(
                np.mean(grad_report.layer_gradient_stds) if grad_report.layer_gradient_stds else 1.0
            ),
            stability_score=float(1.0 if grad_report.is_healthy else 0.6),
            explanation=f"Fixup init for residual network with {num_blocks} blocks. "
                        f"Residual branch scaled by {residual_scale:.4f}.",
            per_layer_config=per_layer,
        )

    def generate_orthogonal_matrix(self, shape: Tuple[int, int],
                                    gain: float = 1.0,
                                    rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Generate an orthogonal matrix for weight initialization.

        Args:
            shape: (fan_out, fan_in) shape of weight matrix.
            gain: Scaling factor.
            rng: Random state for reproducibility.

        Returns:
            Orthogonal weight matrix.
        """
        if rng is None:
            rng = np.random.RandomState()

        rows, cols = shape
        if rows < cols:
            flat = rng.randn(cols, rows)
        else:
            flat = rng.randn(rows, cols)

        # QR decomposition
        q, r = qr(flat, mode="economic")

        # Make Q deterministic by fixing sign of diagonal of R
        d = np.diag(r)
        ph = np.sign(d)
        ph[ph == 0] = 1
        q *= ph

        if rows < cols:
            q = q.T

        return gain * q[:rows, :cols]

    def lsuv_initialization(self, weights: List[np.ndarray],
                            biases: List[np.ndarray],
                            X: np.ndarray,
                            activation: str = "relu",
                            target_var: float = 1.0,
                            max_iter: int = 20,
                            tol: float = 0.01) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Layer-Sequential Unit-Variance (LSUV) initialization.

        Iteratively adjusts weight scales layer-by-layer so that
        each layer's output has unit variance on the data.

        Args:
            weights: Initial weight matrices.
            biases: Initial bias vectors.
            X: Input data for calibration.
            activation: Activation function.
            target_var: Target output variance per layer.
            max_iter: Maximum iterations per layer.
            tol: Tolerance for variance matching.

        Returns:
            Adjusted (weights, biases).
        """
        weights = [w.copy() for w in weights]
        biases = [b.copy() for b in biases]

        h = X.copy()

        for layer_idx in range(len(weights)):
            W = weights[layer_idx]
            b = biases[layer_idx]

            for iteration in range(max_iter):
                # Forward through this layer
                pre_act = h @ W + b

                # Apply activation (except last layer)
                if layer_idx < len(weights) - 1:
                    post_act = self._apply_activation(pre_act, activation)
                else:
                    post_act = pre_act

                # Compute variance
                current_var = np.var(post_act)

                if current_var < 1e-10:
                    # Reinitialize with larger scale
                    rng = np.random.RandomState(42 + layer_idx + iteration)
                    W = rng.randn(*W.shape) * 0.1
                    weights[layer_idx] = W
                    continue

                if abs(current_var - target_var) < tol:
                    break

                # Scale weights
                scale = np.sqrt(target_var / current_var)
                W *= scale
                weights[layer_idx] = W

            # Compute output of this layer for next layer's input
            pre_act = h @ weights[layer_idx] + biases[layer_idx]
            if layer_idx < len(weights) - 1:
                h = self._apply_activation(pre_act, activation)
            else:
                h = pre_act

        return weights, biases

    def data_dependent_init(self, X: np.ndarray, layer_widths: List[int],
                            activation: str = "relu") -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Data-dependent initialization: set biases from data statistics.

        Initializes weights with Kaiming scheme and sets biases to center
        the pre-activation distribution based on data statistics.

        Args:
            X: Input data for calibration.
            layer_widths: List of layer widths including input and output.
            activation: Activation function.

        Returns:
            (weights, biases) initialized from data.
        """
        rng = np.random.RandomState(42)
        gain = self.gains.get_gain(activation)

        weights = []
        biases = []
        h = X.copy()

        for i in range(len(layer_widths) - 1):
            fan_in = layer_widths[i]
            fan_out = layer_widths[i + 1]

            # Kaiming weights
            std = gain / np.sqrt(fan_in)
            W = rng.randn(fan_in, fan_out) * std

            # Compute pre-activations
            pre_act = h @ W

            # Set bias to center the distribution
            bias = -np.mean(pre_act, axis=0)

            weights.append(W)
            biases.append(bias)

            # Forward pass for next layer
            post_act = pre_act + bias
            if i < len(layer_widths) - 2:
                h = self._apply_activation(post_act, activation)
            else:
                h = post_act

        return weights, biases

    def verify_gradient_flow(self, weights: List[np.ndarray],
                              biases: List[np.ndarray],
                              X: np.ndarray,
                              activation: str = "relu") -> GradientFlowReport:
        """Verify gradient flow through the network.

        Simulates forward and backward pass, checking gradient magnitudes
        at each layer.

        Args:
            weights: Weight matrices.
            biases: Bias vectors.
            X: Input data.
            activation: Activation function.

        Returns:
            GradientFlowReport with gradient magnitudes.
        """
        n_layers = len(weights)

        # Forward pass: store activations and pre-activations
        activations = [X]
        pre_activations = []

        h = X.copy()
        for i in range(n_layers):
            z = h @ weights[i] + biases[i]
            pre_activations.append(z)
            if i < n_layers - 1:
                h = self._apply_activation(z, activation)
            else:
                h = z
            activations.append(h)

        # Activation standard deviations
        act_stds = [float(np.std(a)) for a in activations]

        # Backward pass: compute gradient magnitudes
        # Start with gradient of loss w.r.t. output = 1
        grad = np.ones_like(activations[-1])
        gradient_stds = []

        for i in range(n_layers - 1, -1, -1):
            # Gradient through linear layer
            grad_w = activations[i].T @ grad / X.shape[0]
            gradient_stds.append(float(np.std(grad_w)))

            # Gradient through activation (except first layer in backward)
            if i > 0:
                grad = grad @ weights[i].T
                # Apply activation derivative
                grad = grad * self._activation_derivative(pre_activations[i - 1], activation)

        gradient_stds.reverse()

        # Analyze health
        issues = []
        if len(act_stds) > 2:
            act_ratio = act_stds[-2] / max(act_stds[1], 1e-10)
            if act_ratio > 10:
                issues.append(f"Activation variance exploding: ratio={act_ratio:.2f}")
            elif act_ratio < 0.1:
                issues.append(f"Activation variance vanishing: ratio={act_ratio:.2f}")
        else:
            act_ratio = 1.0

        if len(gradient_stds) > 1:
            grad_ratio = gradient_stds[0] / max(gradient_stds[-1], 1e-10)
            if grad_ratio > 100:
                issues.append(f"Gradient exploding: ratio={grad_ratio:.2f}")
            elif grad_ratio < 0.01:
                issues.append(f"Gradient vanishing: ratio={grad_ratio:.2f}")
        else:
            grad_ratio = 1.0

        is_healthy = len(issues) == 0

        return GradientFlowReport(
            layer_activation_stds=act_stds,
            layer_gradient_stds=gradient_stds,
            activation_variance_ratio=float(act_ratio),
            gradient_variance_ratio=float(grad_ratio),
            is_healthy=is_healthy,
            issues=issues,
        )

    def _simulate_gradient_flow(self, arch: NetworkArchitecture,
                                 per_layer_config: List[Dict[str, Any]]) -> GradientFlowReport:
        """Simulate gradient flow using the given initialization config."""
        rng = np.random.RandomState(42)
        n_samples = 100
        X = rng.randn(n_samples, arch.layer_widths[0])

        weights = []
        biases = []
        for config in per_layer_config:
            fan_in = config["fan_in"]
            fan_out = config["fan_out"]
            std = config["weight_std"]

            if config.get("weight_distribution") == "orthogonal":
                W = self.generate_orthogonal_matrix(
                    (fan_out, fan_in), gain=config.get("gain", 1.0), rng=rng
                ).T
                # Ensure shape is (fan_in, fan_out) for X @ W
                if W.shape != (fan_in, fan_out):
                    W = rng.randn(fan_in, fan_out) * std
            else:
                W = rng.randn(fan_in, fan_out) * std

            b = np.zeros(fan_out)
            weights.append(W)
            biases.append(b)

        return self.verify_gradient_flow(weights, biases, X, arch.activation)

    def _apply_activation(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Apply activation function."""
        if activation == "relu":
            return np.maximum(x, 0)
        elif activation == "tanh":
            return np.tanh(x)
        elif activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif activation == "gelu":
            from scipy.special import erf
            return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
        elif activation in ("silu", "swish"):
            return x / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif activation == "elu":
            return np.where(x > 0, x, np.exp(np.clip(x, -500, 500)) - 1)
        elif activation == "linear":
            return x
        return np.maximum(x, 0)

    def _activation_derivative(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Compute derivative of activation function."""
        if activation == "relu":
            return (x > 0).astype(float)
        elif activation == "tanh":
            return 1.0 - np.tanh(x) ** 2
        elif activation == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
            return s * (1.0 - s)
        elif activation == "gelu":
            from scipy.special import erf
            phi = 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
            pdf = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
            return phi + x * pdf
        elif activation in ("silu", "swish"):
            s = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
            return s + x * s * (1.0 - s)
        elif activation == "elu":
            return np.where(x > 0, 1.0, np.exp(np.clip(x, -500, 500)))
        elif activation == "linear":
            return np.ones_like(x)
        return (x > 0).astype(float)
