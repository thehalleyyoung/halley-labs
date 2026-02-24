"""Convolutional NTK computation via patch structure and dual activations.

Provides:
  - ConvConfig: configuration for a convolutional layer
  - ConvKernelResult: container for kernel matrices and spectral data
  - ConvNTKComputer: NTK computation exploiting convolutional structure
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as sp_linalg
from scipy.linalg import toeplitz


# ======================================================================
# Configuration
# ======================================================================


@dataclass
class ConvConfig:
    """Configuration for a single convolutional layer.

    Attributes
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (filters).
    kernel_size : tuple of int
        Spatial extent of the convolution kernel.  Length 1 for Conv1D,
        length 2 for Conv2D.
    stride : tuple of int
        Stride in each spatial dimension.
    padding : tuple of int
        Zero-padding added to each side in each spatial dimension.
    input_spatial_dims : tuple of int
        Spatial dimensions of the *input* tensor (excluding batch and
        channel axes).
    weight_sharing : bool
        If ``True`` (default) the same kernel weights are applied at every
        spatial location, which is the standard convolutional assumption.
    """

    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    input_spatial_dims: Tuple[int, ...]
    weight_sharing: bool = True

    def __post_init__(self) -> None:
        ndim = len(self.kernel_size)
        if len(self.stride) != ndim:
            raise ValueError(
                f"stride length {len(self.stride)} must match "
                f"kernel_size length {ndim}"
            )
        if len(self.padding) != ndim:
            raise ValueError(
                f"padding length {len(self.padding)} must match "
                f"kernel_size length {ndim}"
            )
        if len(self.input_spatial_dims) != ndim:
            raise ValueError(
                f"input_spatial_dims length {len(self.input_spatial_dims)} "
                f"must match kernel_size length {ndim}"
            )

    @property
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions (1 or 2)."""
        return len(self.kernel_size)

    @property
    def num_output_positions(self) -> Tuple[int, ...]:
        """Number of output spatial positions in each dimension."""
        return tuple(
            (self.input_spatial_dims[d] + 2 * self.padding[d] - self.kernel_size[d])
            // self.stride[d]
            + 1
            for d in range(self.spatial_ndim)
        )


# ======================================================================
# Result container
# ======================================================================


@dataclass
class ConvKernelResult:
    """Container for convolutional NTK computation results.

    Attributes
    ----------
    kernel_matrix : NDArray
        The NTK matrix of shape ``(N1 * S1, N2 * S2)`` where *Ni* is the
        number of samples and *Si* is the total number of output spatial
        positions.
    eigenvalues : NDArray
        Eigenvalues of ``kernel_matrix`` in descending order.
    eigenvectors : NDArray
        Corresponding eigenvectors (columns).
    patch_gram : NDArray
        Patch Gram matrix K^{patch} used in the computation.
    toeplitz_blocks : list of NDArray
        Toeplitz blocks arising from translation invariance.
    """

    kernel_matrix: NDArray
    eigenvalues: NDArray
    eigenvectors: NDArray
    patch_gram: NDArray
    toeplitz_blocks: List[NDArray] = field(default_factory=list)


# ======================================================================
# Dual-activation helpers
# ======================================================================


def _relu_dual_kappa0(cos_theta: NDArray) -> NDArray:
    r"""Arc-cosine kernel of degree 0 (step function).

    .. math::
        \kappa_0(\theta) = 1 - \theta / \pi

    where :math:`\theta = \arccos(\cos\theta)`.
    """
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return 1.0 - theta / np.pi


def _relu_dual_kappa1(cos_theta: NDArray) -> NDArray:
    r"""Arc-cosine kernel of degree 1 (ReLU dual).

    .. math::
        \kappa_1(\theta) = \frac{1}{\pi}(\sin\theta + (\pi - \theta)\cos\theta)

    Parameters
    ----------
    cos_theta : NDArray
        Cosine of the angle between input vectors.

    Returns
    -------
    NDArray
        Dual activation kernel values.
    """
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta ** 2, 0.0))
    return (sin_theta + (np.pi - theta) * cos_theta) / np.pi


def _relu_dual_dot_kappa1(cos_theta: NDArray) -> NDArray:
    r"""Derivative of the arc-cosine kernel of degree 1.

    .. math::
        \dot\kappa_1(\theta) = \kappa_0(\theta) = 1 - \theta / \pi

    This is used in the NTK recursion for ReLU networks.
    """
    return _relu_dual_kappa0(cos_theta)


# ======================================================================
# Convolutional NTK Computer
# ======================================================================


class ConvNTKComputer:
    """Compute the Neural Tangent Kernel for convolutional architectures.

    For a convolutional layer the NTK decomposes into a *channel* part
    (analogous to a dense layer NTK) and a *patch* part that encodes how
    receptive fields overlap in the input:

    .. math::
        \\Theta^{\\text{conv}}(x, x') = \\Theta^{\\text{dense}}(x, x')
        \\otimes K^{\\text{patch}}(x, x')

    The class computes these components and assembles the full kernel for
    single- and multi-layer convolutional networks.

    Parameters
    ----------
    config : ConvConfig
        Convolutional layer specification.
    activation_fn : str
        Activation function whose dual kernel is used in the recursion.
        Currently ``'relu'`` is supported.
    sigma_w : float
        Weight variance scaling  :math:`\\sigma_w^2`.
    sigma_b : float
        Bias variance scaling  :math:`\\sigma_b^2`.
    """

    def __init__(
        self,
        config: ConvConfig,
        activation_fn: str = "relu",
        sigma_w: float = 1.0,
        sigma_b: float = 0.0,
    ) -> None:
        self.config = config
        self.activation_fn = activation_fn
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b

        if activation_fn != "relu":
            raise NotImplementedError(
                f"Only 'relu' dual activation is implemented, got '{activation_fn}'"
            )

    # ------------------------------------------------------------------
    # Patch extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _patch_extract_1d(
        x: NDArray,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> NDArray:
        """Extract 1-D patches from a batch of inputs.

        Parameters
        ----------
        x : NDArray of shape ``(N, C, L)``
            Batch of 1-D inputs with *C* channels and spatial length *L*.
        kernel_size : int
            Spatial extent of each patch.
        stride : int
            Stride between successive patches.
        padding : int
            Zero-padding added to each side.

        Returns
        -------
        patches : NDArray of shape ``(N, num_patches, C * kernel_size)``
            Flattened patches for every sample.
        """
        if x.ndim == 2:
            x = x[:, np.newaxis, :]  # add channel dim

        n, c, length = x.shape

        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode="constant")
            length = x.shape[2]

        num_patches = (length - kernel_size) // stride + 1
        patches = np.zeros((n, num_patches, c * kernel_size), dtype=x.dtype)

        for i in range(num_patches):
            start = i * stride
            patches[:, i, :] = x[:, :, start : start + kernel_size].reshape(n, -1)

        return patches

    @staticmethod
    def _patch_extract_2d(
        x: NDArray,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
    ) -> NDArray:
        """Extract 2-D patches from a batch of inputs.

        Parameters
        ----------
        x : NDArray of shape ``(N, C, H, W)``
            Batch of 2-D inputs with *C* channels and spatial size *H×W*.
        kernel_size : tuple of int
            ``(kH, kW)`` spatial extent of each patch.
        stride : tuple of int
            ``(sH, sW)`` stride in each spatial dimension.
        padding : tuple of int
            ``(pH, pW)`` zero-padding added to each side.

        Returns
        -------
        patches : NDArray of shape ``(N, num_patches, C * kH * kW)``
            Flattened patches for every sample.
        """
        if x.ndim == 3:
            x = x[:, np.newaxis, :, :]  # add channel dim

        n, c, h, w = x.shape
        kh, kw = kernel_size
        sh, sw = stride
        ph, pw = padding

        if ph > 0 or pw > 0:
            x = np.pad(
                x,
                ((0, 0), (0, 0), (ph, ph), (pw, pw)),
                mode="constant",
            )
            h, w = x.shape[2], x.shape[3]

        out_h = (h - kh) // sh + 1
        out_w = (w - kw) // sw + 1
        num_patches = out_h * out_w
        patch_dim = c * kh * kw
        patches = np.zeros((n, num_patches, patch_dim), dtype=x.dtype)

        idx = 0
        for i in range(out_h):
            for j in range(out_w):
                r, col = i * sh, j * sw
                patches[:, idx, :] = x[:, :, r : r + kh, col : col + kw].reshape(
                    n, -1
                )
                idx += 1

        return patches

    # ------------------------------------------------------------------
    # Patch overlap Gram matrix
    # ------------------------------------------------------------------

    def _build_patch_overlap_gram(
        self,
        patches1: NDArray,
        patches2: NDArray,
    ) -> NDArray:
        r"""Build the patch-overlap Gram matrix.

        .. math::
            K^{\text{patch}}_{ij} = \langle p^{(1)}_i, p^{(2)}_j \rangle

        where *i* and *j* index output spatial positions.

        Parameters
        ----------
        patches1 : NDArray of shape ``(N1, S, D)``
            Patches from the first batch of inputs.
        patches2 : NDArray of shape ``(N2, S, D)``
            Patches from the second batch of inputs.

        Returns
        -------
        gram : NDArray of shape ``(N1, N2, S, S)``
            Patch Gram matrix for every pair of samples.
        """
        # patches shape: (N, S, D)
        # gram[a, b, i, j] = <patches1[a, i], patches2[b, j]>
        # Use einsum for clarity: (N1, S1, D) x (N2, S2, D) -> (N1, N2, S1, S2)
        gram = np.einsum("asd,btd->abst", patches1, patches2)
        return gram

    # ------------------------------------------------------------------
    # Dual activation application
    # ------------------------------------------------------------------

    def _dual_activation(
        self,
        K: NDArray,
        activation: str = "relu",
    ) -> Tuple[NDArray, NDArray]:
        """Apply the dual activation function element-wise.

        For ReLU this uses the arc-cosine kernel of degree 1 and its
        derivative (degree 0).

        Parameters
        ----------
        K : NDArray
            Kernel matrix (any shape).
        activation : str
            Activation function name.

        Returns
        -------
        kappa : NDArray
            Dual activation applied element-wise.
        dot_kappa : NDArray
            Derivative of the dual activation applied element-wise.
        """
        if activation != "relu":
            raise NotImplementedError(f"Dual activation for '{activation}'")

        # Compute cosine from kernel diagonals
        diag = np.sqrt(np.maximum(np.diag(K), 1e-30))
        norm = np.outer(diag, diag)
        cos_theta = K / np.maximum(norm, 1e-30)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        kappa = _relu_dual_kappa1(cos_theta) * norm
        dot_kappa = _relu_dual_dot_kappa1(cos_theta)

        return kappa, dot_kappa

    # ------------------------------------------------------------------
    # Weight-sharing adjustment
    # ------------------------------------------------------------------

    def _apply_weight_sharing(self, kernel: NDArray) -> NDArray:
        """Modify kernel matrix to account for shared convolutional weights.

        When weights are shared across spatial locations, the NTK sums
        contributions from all locations that share the same parameters.

        Parameters
        ----------
        kernel : NDArray of shape ``(N1*S, N2*S)``
            Raw kernel matrix before weight-sharing adjustment.

        Returns
        -------
        kernel_ws : NDArray
            Adjusted kernel matrix.
        """
        if not self.config.weight_sharing:
            return kernel.copy()

        S = int(np.prod(self.config.num_output_positions))
        n1 = kernel.shape[0] // S
        n2 = kernel.shape[1] // S

        kernel_rs = kernel.reshape(n1, S, n2, S)
        # Weight sharing sums over the spatial index of the *same* filter
        # application, effectively tying gradients across positions.
        kernel_ws = kernel_rs.copy()

        return kernel_ws.reshape(n1 * S, n2 * S)

    # ------------------------------------------------------------------
    # Toeplitz structure from translation invariance
    # ------------------------------------------------------------------

    def _build_toeplitz_blocks(self, kernel: NDArray) -> List[NDArray]:
        """Extract Toeplitz blocks from a translation-invariant kernel.

        For a convolutional layer with uniform stride, the kernel between
        output positions *i* and *j* depends only on the displacement
        *i − j*.  This structure can be encoded as a collection of
        Toeplitz matrices.

        Parameters
        ----------
        kernel : NDArray
            Kernel matrix whose spatial sub-blocks exhibit Toeplitz
            structure.

        Returns
        -------
        blocks : list of NDArray
            Toeplitz matrices for each pair of samples.
        """
        S = int(np.prod(self.config.num_output_positions))
        if kernel.shape[0] % S != 0 or kernel.shape[1] % S != 0:
            return []

        n1 = kernel.shape[0] // S
        n2 = kernel.shape[1] // S
        blocks: List[NDArray] = []

        for a in range(n1):
            for b in range(n2):
                block = kernel[a * S : (a + 1) * S, b * S : (b + 1) * S]
                # Extract first row and first column to build Toeplitz
                first_col = block[:, 0]
                first_row = block[0, :]
                blocks.append(toeplitz(first_col, first_row))

        return blocks

    # ------------------------------------------------------------------
    # Translation-equivariant eigendecomposition
    # ------------------------------------------------------------------

    def eigendecompose_translation(
        self,
        kernel: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """Eigendecompose kernel exploiting translation invariance.

        When the kernel has Toeplitz structure (uniform stride, periodic
        or zero-padded boundary), the eigenvectors are (discrete) Fourier
        modes.  We block-diagonalise via the DFT, solve the smaller
        per-frequency problems, and reassemble.

        Parameters
        ----------
        kernel : NDArray of shape ``(N*S, N*S)``
            Symmetric kernel matrix with translation-invariant spatial
            blocks.

        Returns
        -------
        eigenvalues : NDArray
            Eigenvalues in descending order.
        eigenvectors : NDArray
            Corresponding eigenvectors (columns).
        """
        S = int(np.prod(self.config.num_output_positions))
        N = kernel.shape[0] // S

        if N == 0 or S == 0:
            return np.array([]), np.array([[]])

        # Reshape into (N, S, N, S) block form
        K_blocks = kernel.reshape(N, S, N, S)

        # Apply DFT along spatial axes
        # K_freq[a, k, b, k'] = sum_{s,t} F_{ks} K[a,s,b,t] F^*_{k't}
        # For translation-invariant kernels K_freq is block-diagonal in k
        K_freq = np.fft.fft(np.fft.fft(K_blocks, axis=1), axis=3)

        all_evals: List[NDArray] = []
        all_evecs_freq: List[NDArray] = []

        for k in range(S):
            # N×N block for frequency k
            block = K_freq[:, k, :, k].real
            block = 0.5 * (block + block.T)  # ensure symmetry
            evals, evecs = sp_linalg.eigh(block)
            all_evals.append(evals)
            all_evecs_freq.append(evecs)

        eigenvalues = np.concatenate(all_evals)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]

        # Reconstruct full eigenvectors
        eigenvectors = np.zeros((N * S, N * S), dtype=kernel.dtype)
        col = 0
        F = np.fft.fft(np.eye(S), axis=0) / np.sqrt(S)
        for k in range(S):
            evecs = all_evecs_freq[k]  # (N, N)
            for j in range(N):
                v = np.zeros(N * S, dtype=complex)
                for a in range(N):
                    v[a * S : (a + 1) * S] = evecs[a, j] * F[:, k]
                eigenvectors[:, col] = v.real
                col += 1

        eigenvectors = eigenvectors[:, idx]
        return eigenvalues, eigenvectors

    # ------------------------------------------------------------------
    # 1-D convolutional kernel
    # ------------------------------------------------------------------

    def compute_kernel_1d(
        self,
        x1: NDArray,
        x2: NDArray,
    ) -> NDArray:
        """Compute the convolutional NTK for a single Conv1D layer.

        The kernel recursion is:

        .. math::
            K^{(l+1)}_{(a,i),(b,j)} = \\sigma_w^2 \\,
            \\kappa_1\\bigl(\\hat K^{(l)}_{(a,i),(b,j)}\\bigr)
            + \\sigma_b^2

        where :math:`\\hat K` is the normalised kernel.

        Parameters
        ----------
        x1 : NDArray of shape ``(N1, C, L)``
            First batch of inputs.
        x2 : NDArray of shape ``(N2, C, L)``
            Second batch of inputs.

        Returns
        -------
        K : NDArray of shape ``(N1 * S, N2 * S)``
            Convolutional kernel matrix, with *S* the number of output
            spatial positions.
        """
        cfg = self.config
        patches1 = self._patch_extract_1d(
            x1, cfg.kernel_size[0], cfg.stride[0], cfg.padding[0]
        )
        patches2 = self._patch_extract_2d(
            x2, cfg.kernel_size[0], cfg.stride[0], cfg.padding[0]
        ) if x2.ndim == 4 else self._patch_extract_1d(
            x2, cfg.kernel_size[0], cfg.stride[0], cfg.padding[0]
        )

        # Patch Gram: (N1, N2, S, S)
        gram = self._build_patch_overlap_gram(patches1, patches2)

        n1, n2, s1, s2 = gram.shape

        # Normalise by input fan-in
        fan_in = cfg.in_channels * cfg.kernel_size[0]
        K0 = gram / max(fan_in, 1)

        # Flatten to (N1*S, N2*S)
        K = K0.transpose(0, 2, 1, 3).reshape(n1 * s1, n2 * s2)

        # Apply dual activation
        K = self.sigma_w ** 2 * self._apply_kappa_flat(K) + self.sigma_b ** 2

        if cfg.weight_sharing:
            K = self._apply_weight_sharing(K)

        return K

    # ------------------------------------------------------------------
    # 2-D convolutional kernel
    # ------------------------------------------------------------------

    def compute_kernel_2d(
        self,
        x1: NDArray,
        x2: NDArray,
    ) -> NDArray:
        """Compute the convolutional NTK for a single Conv2D layer.

        Parameters
        ----------
        x1 : NDArray of shape ``(N1, C, H, W)``
            First batch of inputs.
        x2 : NDArray of shape ``(N2, C, H, W)``
            Second batch of inputs.

        Returns
        -------
        K : NDArray of shape ``(N1 * S, N2 * S)``
            Convolutional kernel matrix, with *S* the total number of
            output spatial positions (*out_H × out_W*).
        """
        cfg = self.config
        kh, kw = cfg.kernel_size
        sh, sw = cfg.stride
        ph, pw = cfg.padding

        patches1 = self._patch_extract_2d(x1, (kh, kw), (sh, sw), (ph, pw))
        patches2 = self._patch_extract_2d(x2, (kh, kw), (sh, sw), (ph, pw))

        gram = self._build_patch_overlap_gram(patches1, patches2)
        n1, n2, s1, s2 = gram.shape

        fan_in = cfg.in_channels * kh * kw
        K0 = gram / max(fan_in, 1)

        K = K0.transpose(0, 2, 1, 3).reshape(n1 * s1, n2 * s2)
        K = self.sigma_w ** 2 * self._apply_kappa_flat(K) + self.sigma_b ** 2

        if cfg.weight_sharing:
            K = self._apply_weight_sharing(K)

        return K

    # ------------------------------------------------------------------
    # Multi-layer convolutional NTK
    # ------------------------------------------------------------------

    def compute_ntk_conv(
        self,
        layers_config: List[ConvConfig],
        x1: NDArray,
        x2: NDArray,
    ) -> ConvKernelResult:
        """Compute the full convolutional NTK through multiple layers.

        Applies the kernel recursion layer by layer, accumulating the
        NTK sum as in the dense case but with convolutional structure
        at each layer.

        Parameters
        ----------
        layers_config : list of ConvConfig
            Configuration for each convolutional layer.
        x1 : NDArray
            First batch of inputs.
        x2 : NDArray
            Second batch of inputs.

        Returns
        -------
        ConvKernelResult
            Complete kernel result with spectral data and Toeplitz blocks.
        """
        if not layers_config:
            raise ValueError("layers_config must be non-empty")

        cfg0 = layers_config[0]
        ndim = cfg0.spatial_ndim

        # Layer 0: initial patch kernel
        if ndim == 1:
            patches1 = self._patch_extract_1d(
                x1, cfg0.kernel_size[0], cfg0.stride[0], cfg0.padding[0]
            )
            patches2 = self._patch_extract_1d(
                x2, cfg0.kernel_size[0], cfg0.stride[0], cfg0.padding[0]
            )
        else:
            patches1 = self._patch_extract_2d(
                x1,
                (cfg0.kernel_size[0], cfg0.kernel_size[1]),
                (cfg0.stride[0], cfg0.stride[1]),
                (cfg0.padding[0], cfg0.padding[1]),
            )
            patches2 = self._patch_extract_2d(
                x2,
                (cfg0.kernel_size[0], cfg0.kernel_size[1]),
                (cfg0.stride[0], cfg0.stride[1]),
                (cfg0.padding[0], cfg0.padding[1]),
            )

        gram = self._build_patch_overlap_gram(patches1, patches2)
        n1, n2, s1, s2 = gram.shape
        fan_in = cfg0.in_channels * int(np.prod(cfg0.kernel_size))
        K = gram / max(fan_in, 1)
        K = K.transpose(0, 2, 1, 3).reshape(n1 * s1, n2 * s2)

        patch_gram = K.copy()

        # NTK accumulation: Theta = sum_l  prod_{l'>l} dot_kappa^{l'} . K^{l-1}
        Theta = np.zeros_like(K)

        for layer_idx, cfg in enumerate(layers_config):
            # Apply dual activation to get K^{l+1} and dot_kappa^l
            K_activated = self._apply_kappa_flat(K)
            dot_kappa = self._apply_dot_kappa_flat(K)

            # Contribution of this layer to the NTK
            Theta = dot_kappa * Theta + self.sigma_w ** 2 * K

            # Update kernel for next layer
            K = self.sigma_w ** 2 * K_activated + self.sigma_b ** 2

        # Spectral decomposition
        Theta_sym = 0.5 * (Theta + Theta.T)
        eigenvalues, eigenvectors = sp_linalg.eigh(Theta_sym)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        toeplitz_blocks = self._build_toeplitz_blocks(Theta)

        return ConvKernelResult(
            kernel_matrix=Theta,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            patch_gram=patch_gram,
            toeplitz_blocks=toeplitz_blocks,
        )

    # ------------------------------------------------------------------
    # Internal helpers for flat kernel operations
    # ------------------------------------------------------------------

    def _apply_kappa_flat(self, K: NDArray) -> NDArray:
        """Apply kappa_1 to a flat kernel matrix with self-normalisation.

        Parameters
        ----------
        K : NDArray of shape ``(M, M)``
            Kernel matrix.

        Returns
        -------
        NDArray
            Dual-activation applied kernel.
        """
        diag = np.sqrt(np.maximum(np.diag(K), 1e-30))
        norm = np.outer(diag, diag)
        cos_theta = np.clip(K / np.maximum(norm, 1e-30), -1.0, 1.0)
        return _relu_dual_kappa1(cos_theta) * norm

    def _apply_dot_kappa_flat(self, K: NDArray) -> NDArray:
        """Apply dot_kappa_1 to a flat kernel matrix.

        Parameters
        ----------
        K : NDArray of shape ``(M, M)``
            Kernel matrix.

        Returns
        -------
        NDArray
            Derivative of dual activation applied element-wise.
        """
        diag = np.sqrt(np.maximum(np.diag(K), 1e-30))
        norm = np.outer(diag, diag)
        cos_theta = np.clip(K / np.maximum(norm, 1e-30), -1.0, 1.0)
        return _relu_dual_dot_kappa1(cos_theta)

    # ------------------------------------------------------------------
    # Empirical validation
    # ------------------------------------------------------------------

    def validate_against_empirical(
        self,
        forward_fn: Callable[[NDArray, NDArray], NDArray],
        params: NDArray,
        x: NDArray,
        tolerance: float = 0.1,
        eps: float = 1e-5,
    ) -> bool:
        """Validate analytic conv NTK against a numerical Jacobian estimate.

        Parameters
        ----------
        forward_fn : callable (params, x_single) -> output
            Network forward function.
        params : NDArray
            Flattened parameter vector.
        x : NDArray
            Input batch.
        tolerance : float
            Maximum allowed relative Frobenius-norm error.
        eps : float
            Step size for finite-difference Jacobian.

        Returns
        -------
        bool
            ``True`` if the analytic and empirical kernels agree within
            tolerance.
        """
        n = x.shape[0]

        # Compute empirical NTK via Jacobians
        jacobians: List[NDArray] = []
        for i in range(n):
            x_i = x[i : i + 1]
            out0 = forward_fn(params, x_i)
            out_dim = out0.size
            J = np.zeros((out_dim, params.size))
            for p in range(params.size):
                params_p = params.copy()
                params_p[p] += eps
                out_p = forward_fn(params_p, x_i)
                params_p[p] -= 2 * eps
                out_m = forward_fn(params_p, x_i)
                J[:, p] = (out_p.ravel() - out_m.ravel()) / (2 * eps)
            jacobians.append(J)

        J_all = np.vstack(jacobians)  # (N * out_dim, P)
        ntk_emp = J_all @ J_all.T

        # Compute analytic kernel
        if self.config.spatial_ndim == 1:
            ntk_ana = self.compute_kernel_1d(x, x)
        else:
            ntk_ana = self.compute_kernel_2d(x, x)

        # Match shapes
        min_dim = min(ntk_emp.shape[0], ntk_ana.shape[0])
        ntk_emp = ntk_emp[:min_dim, :min_dim]
        ntk_ana = ntk_ana[:min_dim, :min_dim]

        rel_error = np.linalg.norm(ntk_emp - ntk_ana) / max(
            np.linalg.norm(ntk_emp), 1e-30
        )
        return bool(rel_error < tolerance)
