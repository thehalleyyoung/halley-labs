"""CNN NTK computation and H_{ijk} tensor derivation for convolutional networks.

Computes the finite-width NTK for convolutional networks and derives the
third-order correction tensors H_{ijk} for conv layers, handling weight
sharing properly.

Supports simple CNN architectures (2-3 conv layers + FC head).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class CNNConfig:
    """Configuration for a simple CNN architecture.

    Attributes
    ----------
    input_channels : int
    input_height : int
    input_width : int
    conv_channels : list of int
        Output channels for each conv layer.
    kernel_sizes : list of int
        Kernel size for each conv layer (square kernels).
    fc_width : int
        Width of the fully connected hidden layer.
    output_dim : int
    """
    input_channels: int = 1
    input_height: int = 28
    input_width: int = 28
    conv_channels: List[int] = field(default_factory=lambda: [16, 32])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3])
    fc_width: int = 64
    output_dim: int = 1


class CNN:
    """Simple CNN with analytical Jacobian for NTK computation.

    Architecture: [Conv+ReLU]* -> Flatten -> FC+ReLU -> FC (output)
    Uses stride=1, no padding for simplicity.
    """

    def __init__(self, config: CNNConfig, seed: int = 42) -> None:
        self.config = config
        rng = np.random.RandomState(seed)

        self.conv_weights: List[NDArray] = []
        self.conv_biases: List[NDArray] = []

        in_c = config.input_channels
        h, w = config.input_height, config.input_width

        for i, (out_c, ks) in enumerate(
            zip(config.conv_channels, config.kernel_sizes)
        ):
            fan_in = in_c * ks * ks
            W = rng.randn(out_c, in_c, ks, ks) / np.sqrt(fan_in)
            b = np.zeros(out_c)
            self.conv_weights.append(W)
            self.conv_biases.append(b)
            h = h - ks + 1
            w = w - ks + 1
            in_c = out_c

        self.flat_dim = in_c * h * w
        # FC layers
        self.fc1_w = rng.randn(self.flat_dim, config.fc_width) / np.sqrt(self.flat_dim)
        self.fc1_b = np.zeros(config.fc_width)
        self.fc2_w = rng.randn(config.fc_width, config.output_dim) / np.sqrt(config.fc_width)
        self.fc2_b = np.zeros(config.output_dim)

        self._cache = {}

    def _conv2d(self, x: NDArray, W: NDArray, b: NDArray) -> NDArray:
        """Manual 2D convolution. x: (N, C_in, H, W), W: (C_out, C_in, kH, kW)."""
        N, C_in, H, W_in = x.shape
        C_out, _, kH, kW = W.shape
        H_out = H - kH + 1
        W_out = W_in - kW + 1
        out = np.zeros((N, C_out, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, :, i:i+kH, j:j+kW]  # (N, C_in, kH, kW)
                for c in range(C_out):
                    out[:, c, i, j] = np.sum(
                        patch * W[c:c+1], axis=(1, 2, 3)
                    ) + b[c]
        return out

    def forward(self, X: NDArray) -> NDArray:
        """Forward pass. X: (N, C, H, W).

        Returns
        -------
        NDArray, shape (N, output_dim)
        """
        h = X
        self._cache['input'] = X
        self._cache['conv_pre'] = []
        self._cache['conv_post'] = []

        for i, (W, b) in enumerate(zip(self.conv_weights, self.conv_biases)):
            z = self._conv2d(h, W, b)
            self._cache['conv_pre'].append(z)
            h = np.maximum(z, 0)  # ReLU
            self._cache['conv_post'].append(h)

        # Flatten
        N = h.shape[0]
        flat = h.reshape(N, -1)
        self._cache['flat'] = flat

        # FC1
        z1 = flat @ self.fc1_w + self.fc1_b
        self._cache['fc1_pre'] = z1
        h1 = np.maximum(z1, 0)
        self._cache['fc1_post'] = h1

        # FC2
        out = h1 @ self.fc2_w + self.fc2_b
        self._cache['output'] = out
        return out

    def get_params(self) -> NDArray:
        """Flatten all parameters into a 1D vector."""
        parts = []
        for W, b in zip(self.conv_weights, self.conv_biases):
            parts.append(W.ravel())
            parts.append(b.ravel())
        parts.append(self.fc1_w.ravel())
        parts.append(self.fc1_b.ravel())
        parts.append(self.fc2_w.ravel())
        parts.append(self.fc2_b.ravel())
        return np.concatenate(parts)

    def set_params(self, flat_params: NDArray) -> None:
        """Set parameters from a 1D vector."""
        idx = 0
        for i in range(len(self.conv_weights)):
            sz = self.conv_weights[i].size
            self.conv_weights[i] = flat_params[idx:idx+sz].reshape(self.conv_weights[i].shape)
            idx += sz
            sz = self.conv_biases[i].size
            self.conv_biases[i] = flat_params[idx:idx+sz]
            idx += sz

        sz = self.fc1_w.size
        self.fc1_w = flat_params[idx:idx+sz].reshape(self.fc1_w.shape)
        idx += sz
        sz = self.fc1_b.size
        self.fc1_b = flat_params[idx:idx+sz]
        idx += sz
        sz = self.fc2_w.size
        self.fc2_w = flat_params[idx:idx+sz].reshape(self.fc2_w.shape)
        idx += sz
        sz = self.fc2_b.size
        self.fc2_b = flat_params[idx:idx+sz]
        idx += sz

    def compute_ntk(self, X: NDArray, eps: float = 1e-4) -> NDArray:
        """Compute empirical NTK via finite differences on parameters.

        Parameters
        ----------
        X : NDArray, shape (N, C, H, W)
        eps : float

        Returns
        -------
        NDArray, shape (N, N)
        """
        params = self.get_params()
        f0 = self.forward(X).squeeze(-1)  # (N,)
        N = f0.shape[0]
        P = len(params)

        J = np.zeros((N, P))
        for i in range(P):
            params_p = params.copy()
            params_p[i] += eps
            self.set_params(params_p)
            f_p = self.forward(X).squeeze(-1)
            J[:, i] = (f_p - f0) / eps

        self.set_params(params)  # restore
        return J @ J.T

    def compute_ntk_fast(self, X: NDArray) -> NDArray:
        """Compute NTK using analytical Jacobian for FC layers + numerical for conv.

        For moderate-size networks this is faster than full finite differences.
        Uses the structure: NTK = J_conv @ J_conv^T + J_fc @ J_fc^T.
        """
        # For the FC part, compute analytically
        self.forward(X)
        N = X.shape[0]

        # FC2 Jacobian: ∂out/∂fc2_w = fc1_post ⊗ I
        h1 = self._cache['fc1_post']  # (N, fc_width)
        J_fc2_w = h1  # (N, fc_width) since output_dim=1
        J_fc2_b = np.ones((N, 1))

        # FC1 Jacobian (backprop through ReLU)
        relu_mask = (self._cache['fc1_pre'] > 0).astype(float)  # (N, fc_width)
        delta_fc1 = self.fc2_w.squeeze(-1) * relu_mask  # (N, fc_width)
        flat = self._cache['flat']  # (N, flat_dim)
        J_fc1_w = np.einsum('ni,nj->nij', delta_fc1, flat).reshape(N, -1)
        J_fc1_b = delta_fc1

        # For conv layers, use finite differences (more robust)
        params = self.get_params()
        conv_param_count = sum(W.size + b.size for W, b in
                               zip(self.conv_weights, self.conv_biases))

        f0 = self.forward(X).squeeze(-1)
        J_conv = np.zeros((N, conv_param_count))
        eps = 1e-4
        for i in range(conv_param_count):
            params_p = params.copy()
            params_p[i] += eps
            self.set_params(params_p)
            f_p = self.forward(X).squeeze(-1)
            J_conv[:, i] = (f_p - f0) / eps

        self.set_params(params)

        # Combine Jacobians
        J_full = np.concatenate([J_conv, J_fc1_w, J_fc1_b, J_fc2_w, J_fc2_b], axis=1)
        return J_full @ J_full.T

    def train_step(self, X: NDArray, y: NDArray, lr: float) -> float:
        """Single gradient descent step on MSE loss.

        Parameters
        ----------
        X : (N, C, H, W)
        y : (N,)
        lr : float

        Returns
        -------
        float : loss value
        """
        params = self.get_params()
        pred = self.forward(X).squeeze(-1)
        loss = 0.5 * np.mean((pred - y) ** 2)

        # Numerical gradient
        eps = 1e-4
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params[i] += eps
            self.set_params(params)
            lp = 0.5 * np.mean((self.forward(X).squeeze(-1) - y) ** 2)
            params[i] -= 2 * eps
            self.set_params(params)
            lm = 0.5 * np.mean((self.forward(X).squeeze(-1) - y) ** 2)
            params[i] += eps
            grad[i] = (lp - lm) / (2 * eps)

        params -= lr * grad
        self.set_params(params)
        return loss


def compute_cnn_h_tensor(
    cnn: CNN,
    X: NDArray,
    eps: float = 1e-4,
    max_params: int = 500,
) -> NDArray:
    """Compute H_{ijk} = ∂²f_i / ∂θ_j ∂θ_k for the CNN.

    For tractability, only computes for the first max_params parameters.

    Parameters
    ----------
    cnn : CNN
    X : (N, C, H, W)
    eps : float
    max_params : int

    Returns
    -------
    NDArray, shape (N, P, P) where P = min(n_params, max_params)
    """
    params = cnn.get_params()
    P = min(len(params), max_params)
    N = X.shape[0]

    f0 = cnn.forward(X).squeeze(-1)

    # First derivatives
    J = np.zeros((N, P))
    for j in range(P):
        p = params.copy()
        p[j] += eps
        cnn.set_params(p)
        J[:, j] = (cnn.forward(X).squeeze(-1) - f0) / eps

    # Second derivatives (Hessian of f w.r.t. params)
    H = np.zeros((N, P, P))
    for j in range(P):
        for k in range(j, P):
            p_pp = params.copy()
            p_pp[j] += eps
            p_pp[k] += eps
            cnn.set_params(p_pp)
            f_pp = cnn.forward(X).squeeze(-1)

            p_pm = params.copy()
            p_pm[j] += eps
            p_pm[k] -= eps
            cnn.set_params(p_pm)
            f_pm = cnn.forward(X).squeeze(-1)

            p_mp = params.copy()
            p_mp[j] -= eps
            p_mp[k] += eps
            cnn.set_params(p_mp)
            f_mp = cnn.forward(X).squeeze(-1)

            p_mm = params.copy()
            p_mm[j] -= eps
            p_mm[k] -= eps
            cnn.set_params(p_mm)
            f_mm = cnn.forward(X).squeeze(-1)

            H[:, j, k] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)
            H[:, k, j] = H[:, j, k]

    cnn.set_params(params)  # restore
    return H


def cnn_ntk_correction_from_h(
    J: NDArray,
    H: NDArray,
    width: int,
) -> NDArray:
    """Compute the O(1/N) NTK correction from J and H tensors.

    The correction is:
        δΘ_{ij} = (1/N) Σ_k H_{ika} H_{jka}

    Parameters
    ----------
    J : (N, P) Jacobian
    H : (N, P, P) Hessian tensor
    width : int

    Returns
    -------
    NDArray, shape (N, N) correction matrix
    """
    N, P = J.shape
    # Contract: Σ_{a,b} H_{iab} H_{jab}
    # This is tr(H_i @ H_j^T) for each pair (i,j)
    correction = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            correction[i, j] = np.sum(H[i] * H[j])
            correction[j, i] = correction[i, j]

    return correction / width
