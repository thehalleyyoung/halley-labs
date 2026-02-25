"""
Baseline initialization methods for head-to-head comparison with PhaseKit.

Implements LSUV, data-dependent init, gradient-norm-checking, Kaiming, and
Xavier initialization as independent baselines for evaluating PhaseKit's
σ_w* recommendations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


def apply_activation(h: np.ndarray, activation: str) -> np.ndarray:
    """Apply activation function element-wise."""
    if activation == "relu":
        return np.maximum(h, 0)
    elif activation == "tanh":
        return np.tanh(h)
    elif activation == "gelu":
        from scipy.special import erf
        return 0.5 * h * (1.0 + erf(h / np.sqrt(2.0)))
    elif activation in ("silu", "swish"):
        return h / (1.0 + np.exp(-np.clip(h, -500, 500)))
    elif activation == "leaky_relu":
        return np.where(h > 0, h, 0.01 * h)
    return np.maximum(h, 0)


def activation_derivative(h: np.ndarray, activation: str) -> np.ndarray:
    """Compute activation derivative element-wise."""
    if activation == "relu":
        return (h > 0).astype(float)
    elif activation == "tanh":
        return 1.0 - np.tanh(h) ** 2
    elif activation == "gelu":
        from scipy.special import erf
        phi = 0.5 * (1.0 + erf(h / np.sqrt(2.0)))
        pdf = np.exp(-h ** 2 / 2.0) / np.sqrt(2.0 * np.pi)
        return phi + h * pdf
    elif activation in ("silu", "swish"):
        sig = 1.0 / (1.0 + np.exp(-np.clip(h, -500, 500)))
        return sig + h * sig * (1.0 - sig)
    elif activation == "leaky_relu":
        return np.where(h > 0, 1.0, 0.01)
    return (h > 0).astype(float)


@dataclass
class InitResult:
    """Result of an initialization method."""
    method: str
    weights: List[np.ndarray]
    biases: List[np.ndarray]
    sigma_w_per_layer: List[float]
    description: str = ""


@dataclass
class GradientDiagnostic:
    """Result of gradient-norm diagnostic."""
    layer_grad_norms: List[float]
    layer_activation_norms: List[float]
    vanishing: bool = False
    exploding: bool = False
    diagnosis: str = ""
    recommended_action: str = ""


def xavier_init(dims: List[int], seed: int = 42) -> InitResult:
    """Xavier/Glorot initialization: W ~ N(0, 2/(fan_in + fan_out))."""
    rng = np.random.RandomState(seed)
    weights, biases, sigmas = [], [], []
    for i in range(len(dims) - 1):
        std = np.sqrt(2.0 / (dims[i] + dims[i + 1]))
        W = rng.randn(dims[i], dims[i + 1]) * std
        weights.append(W)
        biases.append(np.zeros(dims[i + 1]))
        sigmas.append(std * np.sqrt(dims[i]))
    return InitResult("xavier", weights, biases, sigmas,
                      "Xavier/Glorot: σ = √(2/(fan_in+fan_out))")


def kaiming_init(dims: List[int], activation: str = "relu",
                 seed: int = 42) -> InitResult:
    """Kaiming/He initialization: W ~ N(0, 2/fan_in) for ReLU."""
    rng = np.random.RandomState(seed)
    gain_map = {"relu": np.sqrt(2.0), "tanh": 1.0, "gelu": 1.0,
                "silu": 1.0, "leaky_relu": np.sqrt(2.0 / 1.0001)}
    gain = gain_map.get(activation, np.sqrt(2.0))
    weights, biases, sigmas = [], [], []
    for i in range(len(dims) - 1):
        std = gain / np.sqrt(dims[i])
        W = rng.randn(dims[i], dims[i + 1]) * std
        weights.append(W)
        biases.append(np.zeros(dims[i + 1]))
        sigmas.append(gain)
    return InitResult("kaiming", weights, biases, sigmas,
                      f"Kaiming/He: gain={gain:.3f} for {activation}")


def phasekit_init(dims: List[int], activation: str = "relu",
                  seed: int = 42) -> InitResult:
    """PhaseKit edge-of-chaos initialization from mean-field theory.

    For each activation, finds the gain σ_w that balances forward variance
    preservation (σ_w² V(q) = q) with backward gradient stability (σ_w² χ(q) ≤ 1).
    For ReLU, both conditions give σ_w = √2 (Kaiming). For non-ReLU activations,
    PhaseKit uses the backward-stable gain with a depth-dependent safety margin.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from mean_field_theory import ActivationVarianceMaps as AVM

    chi_fns = {
        "relu": AVM.relu_chi, "tanh": AVM.tanh_chi,
        "gelu": AVM.gelu_chi, "silu": AVM.silu_chi,
        "swish": AVM.silu_chi, "leaky_relu": AVM.leaky_relu_chi,
        "elu": AVM.elu_chi, "mish": AVM.mish_chi,
    }
    V_fns = {
        "relu": AVM.relu_variance, "tanh": AVM.tanh_variance,
        "gelu": AVM.gelu_variance, "silu": AVM.silu_variance,
        "swish": AVM.silu_variance, "leaky_relu": AVM.leaky_relu_variance,
        "elu": AVM.elu_variance, "mish": AVM.mish_variance,
    }
    chi_fn = chi_fns.get(activation, AVM.relu_chi)
    V_fn = V_fns.get(activation, AVM.relu_variance)

    n_hidden = len(dims) - 2
    q_op = 1.0  # operating variance (normalized inputs)

    # Gain from forward variance preservation: σ_w = √(q / V(q))
    V_q = V_fn(q_op)
    gain_fwd = np.sqrt(q_op / max(V_q, 1e-10))

    # Gain from backward gradient stability: σ_w = 1/√χ(q)
    chi_q = chi_fn(q_op)
    gain_bwd = 1.0 / np.sqrt(max(chi_q, 1e-10))

    # Use the minimum (more conservative) to ensure stability
    gain = min(gain_fwd, gain_bwd)

    # Depth-dependent safety margin: deeper networks need more conservative gain
    # O(1/√N) fluctuations accumulate over L layers
    if n_hidden > 5:
        width = dims[1] if len(dims) > 2 else 128
        safety = 1.0 - 0.5 / np.sqrt(width)
        gain *= safety

    rng = np.random.RandomState(seed)
    weights, biases, sigmas = [], [], []
    for i in range(len(dims) - 1):
        std = gain / np.sqrt(dims[i])
        W = rng.randn(dims[i], dims[i + 1]) * std
        weights.append(W)
        biases.append(np.zeros(dims[i + 1]))
        sigmas.append(gain)
    return InitResult("phasekit", weights, biases, sigmas,
                      f"PhaseKit MF gain={gain:.4f} for {activation}")


def lsuv_init(dims: List[int], activation: str = "relu",
              X_calib: Optional[np.ndarray] = None,
              seed: int = 42, max_iter: int = 20,
              target_var: float = 1.0, tol: float = 0.1) -> InitResult:
    """LSUV (Layer-Sequential Unit-Variance) initialization.

    Mishkin & Matas (2016): iteratively rescale each layer's weights
    so the output variance of that layer ≈ target_var on calibration data.
    """
    rng = np.random.RandomState(seed)
    if X_calib is None:
        X_calib = rng.randn(256, dims[0])

    # Start from orthogonal init (properly scaled)
    weights, biases = [], []
    for i in range(len(dims) - 1):
        m, n = dims[i], dims[i + 1]
        A = rng.randn(max(m, n), max(m, n))
        Q, _ = np.linalg.qr(A)
        W = Q[:m, :n].copy()
        weights.append(W)
        biases.append(np.zeros(n))

    def forward_to_layer(X, layer_idx, apply_act_on_last=True):
        """Forward pass through layers 0..layer_idx."""
        h = X.copy()
        for l in range(layer_idx + 1):
            h = h @ weights[l] + biases[l]
            is_last_hidden = (l < len(weights) - 1)
            if is_last_hidden and (l < layer_idx or apply_act_on_last):
                h = apply_activation(h, activation)
        return h

    # Layer-sequential variance normalization (hidden layers only)
    sigmas = []
    for layer_idx in range(len(weights) - 1):
        for _ in range(max_iter):
            # Forward through layers up to current, get pre-activation output
            h = forward_to_layer(X_calib, layer_idx, apply_act_on_last=False)
            var = float(np.var(h))
            if not np.isfinite(var) or var < 1e-20:
                weights[layer_idx] *= 0.5
                continue
            if abs(var - target_var) / max(target_var, 1e-10) < tol:
                break
            scale = np.sqrt(target_var / var)
            scale = np.clip(scale, 0.01, 100.0)
            weights[layer_idx] *= scale

        sigmas.append(float(np.std(weights[layer_idx]) * np.sqrt(dims[layer_idx])))

    sigmas.append(float(np.std(weights[-1]) * np.sqrt(dims[-2])))

    return InitResult("lsuv", weights, biases, sigmas,
                      "LSUV: layer-sequential unit-variance")


def gradient_norm_diagnostic(weights: List[np.ndarray],
                             biases: List[np.ndarray],
                             activation: str,
                             X: np.ndarray,
                             y: np.ndarray) -> GradientDiagnostic:
    """One forward-backward pass gradient-norm diagnostic.

    Flags layers with vanishing (norm < 1e-5) or exploding (norm > 100) gradients.
    """
    n_layers = len(weights)
    h = X.copy()
    acts = [h]
    pre_acts = [None]

    # Forward pass
    for l in range(n_layers):
        z = h @ weights[l] + biases[l]
        pre_acts.append(z)
        if l < n_layers - 1:
            h = apply_activation(z, activation)
        else:
            h = z
        acts.append(h)

    # Backward pass
    grad = 2.0 * (h.ravel() - y.ravel()).reshape(-1, 1) / len(y)
    layer_grad_norms = []

    for l in range(n_layers - 1, -1, -1):
        dW = acts[l].T @ grad
        layer_grad_norms.append(float(np.linalg.norm(dW)))
        if l > 0:
            grad = (grad @ weights[l].T) * activation_derivative(
                pre_acts[l], activation)

    layer_grad_norms.reverse()

    # Activation norms (forward pass)
    layer_act_norms = [float(np.std(a)) for a in acts[1:]]

    # Diagnosis
    vanishing = any(g < 1e-5 for g in layer_grad_norms[:-1])
    exploding = any(g > 100.0 for g in layer_grad_norms)
    ratio = (layer_grad_norms[0] / max(layer_grad_norms[-1], 1e-30)
             if layer_grad_norms else 1.0)

    if vanishing:
        diagnosis = "Vanishing gradients detected"
        action = "Increase σ_w or use critical initialization"
    elif exploding:
        diagnosis = "Exploding gradients detected"
        action = "Decrease σ_w or use gradient clipping"
    else:
        diagnosis = "Healthy gradient flow"
        action = "No action needed"

    return GradientDiagnostic(
        layer_grad_norms=layer_grad_norms,
        layer_activation_norms=layer_act_norms,
        vanishing=vanishing,
        exploding=exploding,
        diagnosis=diagnosis,
        recommended_action=action,
    )


def train_mlp(weights: List[np.ndarray], biases: List[np.ndarray],
              X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              activation: str = "relu", n_steps: int = 500,
              lr: float = 0.01, clip_grad: float = 5.0) -> Dict[str, Any]:
    """Train MLP and return metrics including loss curve."""
    W = [w.copy() for w in weights]
    b = [bi.copy() for bi in biases]
    n_layers = len(W)
    loss_curve = []

    for step in range(n_steps):
        h = X_train
        acts = [h]
        pre_acts = [None]

        for l in range(n_layers):
            z = h @ W[l] + b[l]
            pre_acts.append(z)
            if l < n_layers - 1:
                h = apply_activation(z, activation)
            else:
                h = z
            acts.append(h)

        loss = float(np.mean((h.ravel() - y_train.ravel()) ** 2))
        if np.isnan(loss) or loss > 1e10:
            loss_curve.extend([float('inf')] * (n_steps - step))
            break
        loss_curve.append(loss)

        grad = 2.0 * (h.ravel() - y_train.ravel()).reshape(-1, 1) / len(y_train)
        for l in range(n_layers - 1, -1, -1):
            dW = acts[l].T @ grad
            db = np.sum(grad, axis=0)
            # Gradient clipping
            gn = np.linalg.norm(dW)
            if gn > clip_grad:
                dW = dW * (clip_grad / gn)
                db = db * (clip_grad / (np.linalg.norm(db) + 1e-10))
            W[l] -= lr * dW
            b[l] -= lr * db
            if l > 0:
                grad = (grad @ W[l].T) * activation_derivative(
                    pre_acts[l], activation)

    # Compute gradient norms at final state
    h = X_train
    acts_final = [h]
    pre_acts_final = [None]
    for l in range(n_layers):
        z = h @ W[l] + b[l]
        pre_acts_final.append(z)
        if l < n_layers - 1:
            h = apply_activation(z, activation)
        else:
            h = z
        acts_final.append(h)

    # Test loss
    h = X_test
    for l in range(n_layers):
        h = h @ W[l] + b[l]
        if l < n_layers - 1:
            h = apply_activation(h, activation)
    test_loss = float(np.mean((h.ravel() - y_test.ravel()) ** 2))
    if np.isnan(test_loss):
        test_loss = float('inf')

    final_loss = loss_curve[-1] if loss_curve else float('inf')
    init_loss = loss_curve[0] if loss_curve else float('inf')

    return {
        "init_loss": init_loss,
        "final_loss": final_loss,
        "test_loss": test_loss,
        "loss_curve": loss_curve,
        "converged": final_loss < 0.1 * init_loss if init_loss < float('inf') else False,
        "exploded": final_loss == float('inf'),
        "loss_ratio": final_loss / max(init_loss, 1e-10) if init_loss < float('inf') else float('inf'),
    }


def data_dependent_init(dims: List[int], activation: str = "relu",
                        X_calib: Optional[np.ndarray] = None,
                        seed: int = 42, target_var: float = 1.0) -> InitResult:
    """Data-dependent initialization (Kraehenbuehl & Doersch, 2016).

    Initializes weights using Kaiming, then adjusts per-layer scale and bias
    so that each layer's output has zero mean and target variance on
    calibration data. Unlike LSUV which iterates, this uses a single
    forward pass with analytic correction.
    """
    rng = np.random.RandomState(seed)
    if X_calib is None:
        X_calib = rng.randn(256, dims[0])

    gain_map = {"relu": np.sqrt(2.0), "tanh": 1.0, "gelu": 1.0,
                "silu": 1.0, "leaky_relu": np.sqrt(2.0 / 1.0001)}
    gain = gain_map.get(activation, np.sqrt(2.0))

    weights, biases = [], []
    for i in range(len(dims) - 1):
        std = gain / np.sqrt(dims[i])
        W = rng.randn(dims[i], dims[i + 1]) * std
        weights.append(W)
        biases.append(np.zeros(dims[i + 1]))

    # Single forward pass: adjust each layer's scale and bias
    h = X_calib.copy()
    sigmas = []
    for l in range(len(weights)):
        z = h @ weights[l] + biases[l]

        # Adjust bias for zero mean
        mean_z = np.mean(z, axis=0)
        biases[l] = biases[l] - mean_z

        # Adjust weight scale for target variance
        var_z = np.var(z - mean_z)
        if np.isfinite(var_z) and var_z > 1e-20:
            scale = np.sqrt(target_var / var_z)
            scale = np.clip(scale, 0.01, 100.0)
            weights[l] *= scale
            biases[l] *= scale

        # Recompute for next layer
        z = h @ weights[l] + biases[l]
        if l < len(weights) - 1:
            h = apply_activation(z, activation)
        else:
            h = z

        sigmas.append(float(np.std(weights[l]) * np.sqrt(dims[l])))

    return InitResult("data_dependent", weights, biases, sigmas,
                      "Data-dependent: single-pass mean/var correction")


def gradnorm_checking_init(dims: List[int], activation: str = "relu",
                           X_calib: Optional[np.ndarray] = None,
                           seed: int = 42, target_grad_ratio: float = 1.0,
                           max_iter: int = 10) -> InitResult:
    """Gradient-norm checking initialization.

    Starts from Kaiming init, then iteratively adjusts the global weight
    scale so that the ratio of gradient norms (first layer / last layer)
    is close to target_grad_ratio (1.0 = uniform gradient flow).

    This is a simple baseline that directly optimizes what we care about:
    healthy gradient flow.
    """
    rng = np.random.RandomState(seed)
    if X_calib is None:
        X_calib = rng.randn(256, dims[0])

    y_calib = rng.randn(X_calib.shape[0])

    gain_map = {"relu": np.sqrt(2.0), "tanh": 1.0, "gelu": 1.0,
                "silu": 1.0, "leaky_relu": np.sqrt(2.0 / 1.0001)}
    gain = gain_map.get(activation, np.sqrt(2.0))

    best_scale = 1.0
    best_ratio_err = float('inf')

    for trial in range(max_iter):
        scale = gain * (0.5 + trial * 0.2)

        weights, biases = [], []
        for i in range(len(dims) - 1):
            std = scale / np.sqrt(dims[i])
            W = rng.randn(dims[i], dims[i + 1]) * std
            weights.append(W)
            biases.append(np.zeros(dims[i + 1]))

        diag = gradient_norm_diagnostic(weights, biases, activation,
                                        X_calib[:64], y_calib[:64])

        if (not diag.vanishing and not diag.exploding and
                len(diag.layer_grad_norms) >= 2):
            first = diag.layer_grad_norms[0]
            last = diag.layer_grad_norms[-1]
            ratio = first / max(last, 1e-30)
            ratio_err = abs(np.log(ratio / target_grad_ratio))
            if ratio_err < best_ratio_err:
                best_ratio_err = ratio_err
                best_scale = scale

    # Rebuild with best scale
    rng2 = np.random.RandomState(seed)
    weights, biases, sigmas = [], [], []
    for i in range(len(dims) - 1):
        std = best_scale / np.sqrt(dims[i])
        W = rng2.randn(dims[i], dims[i + 1]) * std
        weights.append(W)
        biases.append(np.zeros(dims[i + 1]))
        sigmas.append(best_scale)

    return InitResult("gradnorm_check", weights, biases, sigmas,
                      f"Gradient-norm checking: scale={best_scale:.4f}")
