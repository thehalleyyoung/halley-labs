"""
Neural Tangent Kernel (NTK) computation module.

Implements analytical and empirical NTK computation, eigenspectrum analysis,
drift measurement, convolutional NTK, and kernel regression predictions.
"""

import numpy as np
from scipy import linalg as sla
from scipy.special import erf
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import warnings


@dataclass
class ModelSpec:
    """Specification of a neural network architecture."""
    layer_widths: List[int]
    activation: str = "relu"
    sigma_w: float = 1.0
    sigma_b: float = 0.0
    architecture: str = "fc"  # "fc" or "conv"
    kernel_sizes: Optional[List[int]] = None
    strides: Optional[List[int]] = None
    input_dim: int = 1

    @property
    def depth(self) -> int:
        return len(self.layer_widths) - 1

    @property
    def num_params(self) -> int:
        total = 0
        for i in range(len(self.layer_widths) - 1):
            total += self.layer_widths[i] * self.layer_widths[i + 1]
            total += self.layer_widths[i + 1]
        return total


@dataclass
class NTKResult:
    """Result of NTK computation."""
    kernel_matrix: np.ndarray
    eigenvalues: Optional[np.ndarray] = None
    eigenvectors: Optional[np.ndarray] = None
    condition_number: Optional[float] = None
    spectral_decay_rate: Optional[float] = None
    trace: Optional[float] = None


@dataclass
class NTKDriftResult:
    """Result of NTK drift measurement."""
    initial_ntk: np.ndarray
    current_ntk: np.ndarray
    relative_change: float
    frobenius_distance: float
    spectral_distance: float
    is_lazy: bool
    drift_per_step: List[float] = field(default_factory=list)


@dataclass
class NTKAlignmentResult:
    """Result of NTK alignment analysis."""
    cosine_similarity: float
    target_alignment: float
    effective_rank: float
    alignment_spectrum: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class KernelRegressionResult:
    """Result of kernel regression prediction."""
    predictions: np.ndarray
    residuals: Optional[np.ndarray] = None
    effective_dimensionality: float = 0.0
    regularization_used: float = 0.0


class ActivationKernelMap:
    """Maps activation functions to their kernel transformations.

    For a zero-mean Gaussian with covariance [[q1, q12], [q12, q2]],
    the expected value E[sigma(u)*sigma(v)] defines the next-layer kernel.
    """

    @staticmethod
    def relu_kappa0(cos_angle: np.ndarray) -> np.ndarray:
        """Fraction of time both ReLU inputs are positive."""
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return (np.pi - np.arccos(cos_angle)) / (2.0 * np.pi)

    @staticmethod
    def relu_kappa1(cos_angle: np.ndarray) -> np.ndarray:
        """First-order arc-cosine kernel for ReLU."""
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        sin_angle = np.sqrt(np.maximum(1.0 - cos_angle ** 2, 0.0))
        theta = np.arccos(cos_angle)
        return (sin_angle + (np.pi - theta) * cos_angle) / (2.0 * np.pi)

    @staticmethod
    def relu_dot_kappa(cos_angle: np.ndarray) -> np.ndarray:
        """Derivative kernel for ReLU: E[sigma'(u)*sigma'(v)]."""
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return (np.pi - np.arccos(cos_angle)) / (2.0 * np.pi)

    @staticmethod
    def erf_kernel(q1: float, q2: float, q12: np.ndarray) -> np.ndarray:
        """Kernel for erf activation (related to tanh)."""
        denom = np.sqrt((1.0 + 2.0 * q1) * (1.0 + 2.0 * q2))
        arg = 2.0 * q12 / denom
        arg = np.clip(arg, -1.0, 1.0)
        return (2.0 / np.pi) * np.arcsin(arg)

    @staticmethod
    def erf_dot_kernel(q1: float, q2: float, q12: np.ndarray) -> np.ndarray:
        """Derivative kernel for erf activation."""
        denom_sq = (1.0 + 2.0 * q1) * (1.0 + 2.0 * q2)
        arg_sq = 4.0 * q12 ** 2 / denom_sq
        arg_sq = np.clip(arg_sq, 0.0, 1.0 - 1e-12)
        return (2.0 / np.pi) / np.sqrt(denom_sq * (1.0 - arg_sq))

    @staticmethod
    def linear_kernel(q1: float, q2: float, q12: np.ndarray) -> np.ndarray:
        """Kernel for linear activation (identity)."""
        return q12

    @staticmethod
    def linear_dot_kernel(q1: float, q2: float, q12: np.ndarray) -> np.ndarray:
        """Derivative kernel for linear activation."""
        return np.ones_like(q12)


class NTKComputer:
    """Compute Neural Tangent Kernel for various architectures.

    Supports both analytical (infinite-width) and empirical NTK computation.
    """

    def __init__(self, jitter: float = 1e-8):
        self.jitter = jitter
        self._kernel_map = ActivationKernelMap()

    def compute(self, model_spec: ModelSpec, inputs: np.ndarray) -> NTKResult:
        """Compute the NTK matrix for given model and inputs.

        Args:
            model_spec: Network architecture specification.
            inputs: Input data, shape (n_samples, input_dim).

        Returns:
            NTKResult with kernel matrix and spectral information.
        """
        if model_spec.architecture == "fc":
            K = self._analytical_fc_ntk(model_spec, inputs)
        elif model_spec.architecture == "conv":
            K = self._analytical_conv_ntk(model_spec, inputs)
        else:
            raise ValueError(f"Unknown architecture: {model_spec.architecture}")

        result = NTKResult(kernel_matrix=K)
        self._compute_spectral_info(result)
        return result

    def _analytical_fc_ntk(self, spec: ModelSpec, X: np.ndarray) -> np.ndarray:
        """Analytical NTK for fully-connected networks.

        Uses the recursive kernel computation through layers:
        K^0(x,x') = sigma_w^2 / d_in * x^T x' + sigma_b^2
        K^l(x,x') = sigma_w^2 * F(K^{l-1}) + sigma_b^2
        Theta^L = sum_{l=0}^{L} prod_{l'=l+1}^{L} dot_K^{l'} * K^l
        """
        n = X.shape[0]
        sigma_w = spec.sigma_w
        sigma_b = spec.sigma_b
        depth = spec.depth

        # Base kernel: K^0(x,x') = sigma_w^2 / d_in * <x, x'> + sigma_b^2
        d_in = X.shape[1] if X.ndim > 1 else 1
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        base_cov = (sigma_w ** 2 / d_in) * (X @ X.T) + sigma_b ** 2

        # Store kernels and derivative kernels at each layer
        kernels = [base_cov.copy()]
        dot_kernels = []

        # Forward pass: compute K^l and dot_K^l for each layer
        K_prev = base_cov.copy()
        for l in range(depth):
            # Diagonal entries (variances)
            diag = np.diag(K_prev)
            q_sqrt = np.sqrt(np.maximum(diag, 1e-30))
            # Cosine of angle between pairs
            outer_sqrt = np.outer(q_sqrt, q_sqrt)
            cos_angle = K_prev / np.maximum(outer_sqrt, 1e-30)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)

            if spec.activation == "relu":
                # Next layer kernel
                K_next = sigma_w ** 2 * outer_sqrt * self._kernel_map.relu_kappa1(cos_angle) + sigma_b ** 2
                # Derivative kernel
                dot_K = sigma_w ** 2 * self._kernel_map.relu_dot_kappa(cos_angle)
            elif spec.activation in ("tanh", "erf"):
                q1_arr = diag
                q2_arr = diag
                K_vals = np.zeros_like(K_prev)
                dot_K_vals = np.zeros_like(K_prev)
                for i in range(n):
                    for j in range(n):
                        K_vals[i, j] = self._kernel_map.erf_kernel(
                            q1_arr[i], q2_arr[j], K_prev[i, j]
                        )
                        dot_K_vals[i, j] = self._kernel_map.erf_dot_kernel(
                            q1_arr[i], q2_arr[j], K_prev[i, j]
                        )
                K_next = sigma_w ** 2 * K_vals + sigma_b ** 2
                dot_K = sigma_w ** 2 * dot_K_vals
            elif spec.activation == "linear":
                K_next = sigma_w ** 2 * K_prev + sigma_b ** 2
                dot_K = sigma_w ** 2 * np.ones((n, n))
            else:
                # Default: use ReLU
                K_next = sigma_w ** 2 * outer_sqrt * self._kernel_map.relu_kappa1(cos_angle) + sigma_b ** 2
                dot_K = sigma_w ** 2 * self._kernel_map.relu_dot_kappa(cos_angle)

            kernels.append(K_next.copy())
            dot_kernels.append(dot_K.copy())
            K_prev = K_next

        # NTK: Theta = sum_{l=0}^{L} (prod_{l'=l+1}^{L} dot_K^{l'}) * K^l
        ntk = np.zeros((n, n))
        for l in range(depth + 1):
            contrib = kernels[l].copy()
            for lp in range(l, depth):
                contrib = contrib * dot_kernels[lp]
            ntk += contrib

        # Symmetrize
        ntk = 0.5 * (ntk + ntk.T)
        return ntk

    def empirical_ntk(self, weights: List[np.ndarray], biases: List[np.ndarray],
                      X: np.ndarray, activation: str = "relu",
                      delta: float = 1e-5) -> np.ndarray:
        """Compute empirical NTK via Jacobian: Theta = J @ J^T.

        Args:
            weights: List of weight matrices for each layer.
            biases: List of bias vectors for each layer.
            X: Input data, shape (n_samples, input_dim).
            activation: Activation function name.
            delta: Finite difference step size.

        Returns:
            Empirical NTK matrix, shape (n_samples, n_samples).
        """
        n = X.shape[0]

        def forward(W_list, b_list, x):
            """Forward pass through the network."""
            h = x.copy()
            for i, (W, b) in enumerate(zip(W_list, b_list)):
                h = h @ W + b
                if i < len(W_list) - 1:
                    h = self._apply_activation(h, activation)
            return h

        # Flatten all parameters
        params = []
        for W in weights:
            params.append(W.ravel())
        for b in biases:
            params.append(b.ravel())
        theta = np.concatenate(params)
        p = len(theta)

        # Compute Jacobian via finite differences
        f0 = forward(weights, biases, X)
        out_dim = f0.shape[1] if f0.ndim > 1 else 1
        if f0.ndim == 1:
            f0 = f0.reshape(-1, 1)

        jacobian = np.zeros((n * out_dim, p))

        for k in range(p):
            theta_plus = theta.copy()
            theta_plus[k] += delta

            # Unflatten parameters
            W_plus, b_plus = self._unflatten_params(theta_plus, weights, biases)
            f_plus = forward(W_plus, b_plus, X)
            if f_plus.ndim == 1:
                f_plus = f_plus.reshape(-1, 1)

            jacobian[:, k] = ((f_plus - f0) / delta).ravel()

        # NTK = J @ J^T, then reshape to (n, n) by summing over output dims
        full_ntk = jacobian @ jacobian.T  # (n*out_dim, n*out_dim)

        # Sum over output dimensions to get (n, n) kernel
        ntk = np.zeros((n, n))
        for d in range(out_dim):
            block = full_ntk[d * n:(d + 1) * n, d * n:(d + 1) * n]
            ntk += block

        ntk = 0.5 * (ntk + ntk.T)
        return ntk

    def _unflatten_params(self, theta: np.ndarray,
                          weights_template: List[np.ndarray],
                          biases_template: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Unflatten parameter vector back into weight matrices and biases."""
        W_list = []
        b_list = []
        idx = 0
        for W in weights_template:
            size = W.size
            W_list.append(theta[idx:idx + size].reshape(W.shape))
            idx += size
        for b in biases_template:
            size = b.size
            b_list.append(theta[idx:idx + size].reshape(b.shape))
            idx += size
        return W_list, b_list

    def _apply_activation(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Apply activation function."""
        if activation == "relu":
            return np.maximum(x, 0)
        elif activation == "tanh":
            return np.tanh(x)
        elif activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif activation == "erf":
            return erf(x)
        elif activation == "gelu":
            return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
        elif activation == "silu":
            return x / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif activation == "linear":
            return x
        else:
            return np.maximum(x, 0)  # default to ReLU

    def eigenspectrum_analysis(self, ntk_matrix: np.ndarray,
                               top_k: Optional[int] = None) -> Dict[str, Any]:
        """Analyze eigenspectrum of the NTK.

        Args:
            ntk_matrix: NTK matrix, shape (n, n).
            top_k: Number of top eigenvalues to return. None = all.

        Returns:
            Dictionary with eigenvalues, spectral decay rate, effective rank, etc.
        """
        n = ntk_matrix.shape[0]
        if top_k is None:
            top_k = n

        # Add jitter for numerical stability
        K = ntk_matrix + self.jitter * np.eye(n)

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(K)
        eigenvalues = np.sort(eigenvalues)[::-1]  # descending order

        # Spectral decay rate: fit power law lambda_k ~ k^{-alpha}
        positive_evals = eigenvalues[eigenvalues > 0]
        if len(positive_evals) > 2:
            log_k = np.log(np.arange(1, len(positive_evals) + 1))
            log_lambda = np.log(positive_evals)
            # Linear regression in log-log space
            A = np.vstack([log_k, np.ones(len(log_k))]).T
            alpha, log_c = np.linalg.lstsq(A, log_lambda, rcond=None)[0]
            decay_rate = -alpha
        else:
            decay_rate = 0.0

        # Effective rank: exp(entropy of normalized eigenvalues)
        pos_evals = eigenvalues[eigenvalues > 0]
        if len(pos_evals) > 0:
            p = pos_evals / pos_evals.sum()
            entropy = -np.sum(p * np.log(p + 1e-30))
            effective_rank = np.exp(entropy)
        else:
            effective_rank = 0.0

        # Condition number
        if len(pos_evals) > 1:
            condition_number = pos_evals[0] / max(pos_evals[-1], 1e-30)
        else:
            condition_number = float("inf")

        # Spectral gap
        if len(pos_evals) > 1:
            spectral_gap = pos_evals[0] - pos_evals[1]
        else:
            spectral_gap = 0.0

        return {
            "eigenvalues": eigenvalues[:top_k],
            "decay_rate": decay_rate,
            "effective_rank": effective_rank,
            "condition_number": condition_number,
            "spectral_gap": spectral_gap,
            "trace": np.sum(eigenvalues),
            "top_eigenvalue": eigenvalues[0] if len(eigenvalues) > 0 else 0.0,
        }

    def ntk_drift(self, spec: ModelSpec, X: np.ndarray,
                  weights_trajectory: List[Tuple[List[np.ndarray], List[np.ndarray]]],
                  ) -> NTKDriftResult:
        """Measure NTK drift during training.

        Args:
            spec: Model specification.
            X: Input data.
            weights_trajectory: List of (weights, biases) at different training steps.

        Returns:
            NTKDriftResult with drift measurements.
        """
        if len(weights_trajectory) < 2:
            raise ValueError("Need at least 2 snapshots for drift measurement")

        # Compute NTK at initialization
        w0, b0 = weights_trajectory[0]
        ntk_init = self.empirical_ntk(w0, b0, X, spec.activation)

        # Compute NTK at each snapshot
        drift_per_step = []
        ntk_current = ntk_init
        for i in range(1, len(weights_trajectory)):
            w_i, b_i = weights_trajectory[i]
            ntk_i = self.empirical_ntk(w_i, b_i, X, spec.activation)

            frob_dist = np.linalg.norm(ntk_i - ntk_init, "fro")
            frob_init = np.linalg.norm(ntk_init, "fro")
            rel_change = frob_dist / max(frob_init, 1e-30)
            drift_per_step.append(rel_change)
            ntk_current = ntk_i

        # Final drift metrics
        frob_final = np.linalg.norm(ntk_current - ntk_init, "fro")
        frob_init = np.linalg.norm(ntk_init, "fro")
        relative_change = frob_final / max(frob_init, 1e-30)

        # Spectral distance: difference in top eigenvalues
        evals_init = np.sort(np.linalg.eigvalsh(ntk_init))[::-1]
        evals_curr = np.sort(np.linalg.eigvalsh(ntk_current))[::-1]
        spectral_distance = np.linalg.norm(evals_init - evals_curr) / max(
            np.linalg.norm(evals_init), 1e-30
        )

        lazy_threshold = 0.1
        is_lazy = relative_change < lazy_threshold

        return NTKDriftResult(
            initial_ntk=ntk_init,
            current_ntk=ntk_current,
            relative_change=relative_change,
            frobenius_distance=frob_final,
            spectral_distance=spectral_distance,
            is_lazy=is_lazy,
            drift_per_step=drift_per_step,
        )

    def _analytical_conv_ntk(self, spec: ModelSpec, X: np.ndarray) -> np.ndarray:
        """Analytical CNTK (Convolutional NTK).

        For convolutional architectures, the NTK has additional structure
        from weight sharing. We compute it by treating convolutions as
        structured matrix multiplications.

        Args:
            spec: Model specification with conv parameters.
            X: Input data, shape (n_samples, spatial_dim) or (n_samples, channels, spatial_dim).

        Returns:
            CNTK matrix, shape (n_samples, n_samples).
        """
        n = X.shape[0]
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        kernel_sizes = spec.kernel_sizes or [3] * spec.depth
        strides = spec.strides or [1] * spec.depth
        sigma_w = spec.sigma_w
        sigma_b = spec.sigma_b

        # For CNTK, we compute the kernel by averaging over spatial positions
        # Base kernel from input inner products
        if X.ndim == 2:
            spatial_dim = X.shape[1]
        else:
            spatial_dim = X.shape[-1]

        # Compute pairwise inner products normalized by spatial dimension
        if X.ndim == 2:
            base_cov = (sigma_w ** 2 / spatial_dim) * (X @ X.T) + sigma_b ** 2
        else:
            X_flat = X.reshape(n, -1)
            d = X_flat.shape[1]
            base_cov = (sigma_w ** 2 / d) * (X_flat @ X_flat.T) + sigma_b ** 2

        # Propagate through convolutional layers
        # For CNTK, each layer applies the activation kernel but with
        # averaging over receptive field positions
        K_prev = base_cov.copy()
        kernels_list = [K_prev.copy()]
        dot_kernels_list = []

        current_spatial = spatial_dim
        for l in range(spec.depth):
            ks = kernel_sizes[l] if l < len(kernel_sizes) else 3
            stride = strides[l] if l < len(strides) else 1

            # Receptive field averaging factor
            rf_factor = min(ks, current_spatial) / max(current_spatial, 1)

            diag = np.diag(K_prev)
            q_sqrt = np.sqrt(np.maximum(diag, 1e-30))
            outer_sqrt = np.outer(q_sqrt, q_sqrt)
            cos_angle = K_prev / np.maximum(outer_sqrt, 1e-30)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)

            if spec.activation == "relu":
                K_next = sigma_w ** 2 * outer_sqrt * self._kernel_map.relu_kappa1(cos_angle) + sigma_b ** 2
                dot_K = sigma_w ** 2 * self._kernel_map.relu_dot_kappa(cos_angle)
            else:
                K_next = sigma_w ** 2 * outer_sqrt * self._kernel_map.relu_kappa1(cos_angle) + sigma_b ** 2
                dot_K = sigma_w ** 2 * self._kernel_map.relu_dot_kappa(cos_angle)

            # Weight sharing correction: multiply by receptive field ratio
            K_next *= (1.0 + rf_factor) / 2.0
            K_next += sigma_b ** 2 * rf_factor

            kernels_list.append(K_next.copy())
            dot_kernels_list.append(dot_K.copy())
            K_prev = K_next

            # Update spatial dimension
            current_spatial = max(1, (current_spatial - ks) // stride + 1)

        # Accumulate NTK contributions from all layers
        ntk = np.zeros((n, n))
        for l in range(spec.depth + 1):
            contrib = kernels_list[l].copy()
            for lp in range(l, spec.depth):
                contrib = contrib * dot_kernels_list[lp]
            ntk += contrib

        ntk = 0.5 * (ntk + ntk.T)
        return ntk

    def hierarchical_ntk(self, spec: ModelSpec, X: np.ndarray) -> Dict[str, Any]:
        """Analyze block structure of NTK for deep networks.

        Decomposes the NTK into per-layer contributions and analyzes
        which layers dominate the kernel.

        Args:
            spec: Model specification.
            X: Input data.

        Returns:
            Dictionary with per-layer NTK contributions and dominance analysis.
        """
        n = X.shape[0]
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        sigma_w = spec.sigma_w
        sigma_b = spec.sigma_b
        depth = spec.depth
        d_in = X.shape[1]

        base_cov = (sigma_w ** 2 / d_in) * (X @ X.T) + sigma_b ** 2

        kernels = [base_cov.copy()]
        dot_kernels = []
        K_prev = base_cov.copy()

        for l in range(depth):
            diag = np.diag(K_prev)
            q_sqrt = np.sqrt(np.maximum(diag, 1e-30))
            outer_sqrt = np.outer(q_sqrt, q_sqrt)
            cos_angle = K_prev / np.maximum(outer_sqrt, 1e-30)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)

            K_next = sigma_w ** 2 * outer_sqrt * self._kernel_map.relu_kappa1(cos_angle) + sigma_b ** 2
            dot_K = sigma_w ** 2 * self._kernel_map.relu_dot_kappa(cos_angle)

            kernels.append(K_next.copy())
            dot_kernels.append(dot_K.copy())
            K_prev = K_next

        # Per-layer NTK contributions
        layer_contributions = []
        layer_norms = []
        for l in range(depth + 1):
            contrib = kernels[l].copy()
            for lp in range(l, depth):
                contrib = contrib * dot_kernels[lp]
            layer_contributions.append(contrib)
            layer_norms.append(np.linalg.norm(contrib, "fro"))

        total_norm = sum(layer_norms)
        layer_fractions = [ln / max(total_norm, 1e-30) for ln in layer_norms]

        # Find dominant layer
        dominant_layer = int(np.argmax(layer_norms))

        # Block diagonal approximation quality
        full_ntk = sum(layer_contributions)
        full_ntk = 0.5 * (full_ntk + full_ntk.T)

        return {
            "full_ntk": full_ntk,
            "layer_contributions": layer_contributions,
            "layer_norms": layer_norms,
            "layer_fractions": layer_fractions,
            "dominant_layer": dominant_layer,
            "num_layers": depth + 1,
        }

    def ntk_alignment(self, ntk_matrix: np.ndarray,
                      target_kernel: np.ndarray) -> NTKAlignmentResult:
        """Compute alignment between NTK and a target kernel.

        Args:
            ntk_matrix: The NTK matrix.
            target_kernel: Target kernel to compare against (e.g., y @ y^T).

        Returns:
            NTKAlignmentResult with alignment metrics.
        """
        # Center kernels
        n = ntk_matrix.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        K_centered = H @ ntk_matrix @ H
        T_centered = H @ target_kernel @ H

        # Cosine similarity (kernel alignment)
        numer = np.sum(K_centered * T_centered)
        denom = np.sqrt(np.sum(K_centered ** 2) * np.sum(T_centered ** 2))
        cosine_sim = numer / max(denom, 1e-30)

        # Target alignment: Frobenius inner product normalized
        target_alignment = np.trace(ntk_matrix @ target_kernel) / (
            np.sqrt(np.trace(ntk_matrix @ ntk_matrix) * np.trace(target_kernel @ target_kernel))
            + 1e-30
        )

        # Effective rank of NTK
        evals = np.linalg.eigvalsh(ntk_matrix + self.jitter * np.eye(n))
        pos_evals = evals[evals > 0]
        if len(pos_evals) > 0:
            p = pos_evals / pos_evals.sum()
            entropy = -np.sum(p * np.log(p + 1e-30))
            effective_rank = np.exp(entropy)
        else:
            effective_rank = 0.0

        # Alignment spectrum: eigenvalues of K^{-1} T
        try:
            K_inv = np.linalg.inv(ntk_matrix + self.jitter * np.eye(n))
            alignment_spectrum = np.sort(np.real(np.linalg.eigvals(K_inv @ target_kernel)))[::-1]
        except np.linalg.LinAlgError:
            alignment_spectrum = np.array([])

        return NTKAlignmentResult(
            cosine_similarity=float(cosine_sim),
            target_alignment=float(target_alignment),
            effective_rank=float(effective_rank),
            alignment_spectrum=alignment_spectrum,
        )

    def kernel_regression(self, K_train: np.ndarray, y_train: np.ndarray,
                          K_test_train: np.ndarray,
                          regularization: float = 1e-6,
                          y_test: Optional[np.ndarray] = None) -> KernelRegressionResult:
        """Perform kernel regression using NTK.

        Predicts y_test = K(X_test, X_train) @ (K(X_train, X_train) + lambda I)^{-1} @ y_train

        Args:
            K_train: Training kernel matrix, shape (n_train, n_train).
            y_train: Training targets, shape (n_train,) or (n_train, d_out).
            K_test_train: Test-train kernel matrix, shape (n_test, n_train).
            regularization: Regularization parameter lambda.
            y_test: Optional true test targets for computing residuals.

        Returns:
            KernelRegressionResult with predictions.
        """
        n_train = K_train.shape[0]

        # Regularized inversion: (K + lambda I)^{-1} y
        K_reg = K_train + regularization * np.eye(n_train)
        try:
            alpha = sla.solve(K_reg, y_train, assume_a="pos")
        except (sla.LinAlgError, np.linalg.LinAlgError):
            alpha = np.linalg.lstsq(K_reg, y_train, rcond=None)[0]

        # Predictions
        predictions = K_test_train @ alpha

        # Effective dimensionality
        evals = np.linalg.eigvalsh(K_train)
        eff_dim = np.sum(evals / (evals + regularization))

        # Residuals
        residuals = None
        if y_test is not None:
            residuals = y_test - predictions

        return KernelRegressionResult(
            predictions=predictions,
            residuals=residuals,
            effective_dimensionality=float(eff_dim),
            regularization_used=regularization,
        )

    def condition_number_analysis(self, ntk_matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze NTK condition number to predict training speed.

        The condition number kappa = lambda_max / lambda_min determines
        the convergence rate of gradient descent in the NTK regime:
        - Small kappa -> fast, uniform convergence
        - Large kappa -> slow convergence, some modes learn much faster

        Args:
            ntk_matrix: NTK matrix.

        Returns:
            Dictionary with condition number analysis.
        """
        n = ntk_matrix.shape[0]
        evals = np.sort(np.linalg.eigvalsh(ntk_matrix + self.jitter * np.eye(n)))[::-1]
        pos_evals = evals[evals > self.jitter * 10]

        if len(pos_evals) < 2:
            return {
                "condition_number": float("inf"),
                "max_eigenvalue": float(evals[0]) if len(evals) > 0 else 0.0,
                "min_eigenvalue": 0.0,
                "max_learning_rate": 0.0,
                "convergence_rate": 0.0,
                "estimated_steps_to_converge": float("inf"),
            }

        lambda_max = pos_evals[0]
        lambda_min = pos_evals[-1]
        kappa = lambda_max / lambda_min

        # Maximum stable learning rate: 2 / lambda_max
        max_lr = 2.0 / lambda_max

        # Convergence rate per step: 1 - 2*eta*lambda_min for optimal eta
        # Optimal eta = 2 / (lambda_max + lambda_min)
        optimal_eta = 2.0 / (lambda_max + lambda_min)
        convergence_rate = 1.0 - 2.0 * optimal_eta * lambda_min

        # Steps to reduce error by factor of 1/e
        if convergence_rate < 1.0:
            steps_to_converge = -1.0 / np.log(max(convergence_rate, 1e-30))
        else:
            steps_to_converge = float("inf")

        # Spectral gap ratio
        spectral_gap_ratio = (pos_evals[0] - pos_evals[1]) / pos_evals[0] if len(pos_evals) > 1 else 0.0

        return {
            "condition_number": float(kappa),
            "max_eigenvalue": float(lambda_max),
            "min_eigenvalue": float(lambda_min),
            "max_learning_rate": float(max_lr),
            "optimal_learning_rate": float(optimal_eta),
            "convergence_rate": float(convergence_rate),
            "estimated_steps_to_converge": float(steps_to_converge),
            "spectral_gap_ratio": float(spectral_gap_ratio),
            "eigenvalue_histogram": pos_evals.tolist(),
        }

    def _compute_spectral_info(self, result: NTKResult) -> None:
        """Fill in spectral information for an NTK result."""
        n = result.kernel_matrix.shape[0]
        K = result.kernel_matrix + self.jitter * np.eye(n)

        evals, evecs = np.linalg.eigh(K)
        idx = np.argsort(evals)[::-1]
        result.eigenvalues = evals[idx]
        result.eigenvectors = evecs[:, idx]
        result.trace = float(np.sum(evals))

        pos_evals = result.eigenvalues[result.eigenvalues > self.jitter * 10]
        if len(pos_evals) >= 2:
            result.condition_number = float(pos_evals[0] / pos_evals[-1])
        else:
            result.condition_number = float("inf")

        if len(pos_evals) > 2:
            log_k = np.log(np.arange(1, len(pos_evals) + 1))
            log_l = np.log(pos_evals)
            A = np.vstack([log_k, np.ones(len(log_k))]).T
            slope, _ = np.linalg.lstsq(A, log_l, rcond=None)[0]
            result.spectral_decay_rate = float(-slope)
        else:
            result.spectral_decay_rate = 0.0


def compute_ntk_simple(X: np.ndarray, depth: int = 2,
                       sigma_w: float = 1.0, sigma_b: float = 0.0,
                       activation: str = "relu") -> np.ndarray:
    """Convenience function to compute analytical NTK for a simple FC network.

    Args:
        X: Input data, shape (n_samples, input_dim).
        depth: Number of layers.
        sigma_w: Weight initialization scale.
        sigma_b: Bias initialization scale.
        activation: Activation function.

    Returns:
        NTK matrix, shape (n_samples, n_samples).
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    d_in = X.shape[1]
    widths = [d_in] + [100] * depth + [1]
    spec = ModelSpec(
        layer_widths=widths,
        activation=activation,
        sigma_w=sigma_w,
        sigma_b=sigma_b,
    )
    computer = NTKComputer()
    result = computer.compute(spec, X)
    return result.kernel_matrix


def verify_ntk_properties(K: np.ndarray, tol: float = 1e-6) -> Dict[str, bool]:
    """Verify that a matrix has valid NTK properties.

    Args:
        K: Matrix to verify.
        tol: Tolerance for numerical checks.

    Returns:
        Dictionary of property checks.
    """
    n = K.shape[0]
    is_square = K.shape[0] == K.shape[1]
    is_symmetric = np.allclose(K, K.T, atol=tol)

    evals = np.linalg.eigvalsh(K)
    is_psd = np.all(evals >= -tol)

    has_positive_diagonal = np.all(np.diag(K) > 0)

    return {
        "is_square": is_square,
        "is_symmetric": is_symmetric,
        "is_positive_semidefinite": is_psd,
        "has_positive_diagonal": has_positive_diagonal,
        "min_eigenvalue": float(np.min(evals)),
        "max_eigenvalue": float(np.max(evals)),
    }
