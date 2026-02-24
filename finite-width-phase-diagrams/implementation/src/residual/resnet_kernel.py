"""ResNet-specific NTK computation.

Computes the infinite-width Neural Tangent Kernel for ResNet
architectures (pre- and post-activation, bottleneck variants) via
recursive kernel propagation through residual blocks.  Includes
depth-analysis, eigenvalue-evolution, and fixed-point (infinite-depth)
utilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as sp_linalg


# ======================================================================
# Configuration dataclasses
# ======================================================================


@dataclass
class ResNetBlockConfig:
    """Configuration for a single residual block.

    Parameters
    ----------
    num_layers_per_block : int
        Number of weight layers inside the block (default 2).
    activation : str
        Name of the activation function (e.g. ``"relu"``).
    pre_activation : bool
        If *True*, use pre-activation ordering (BN→Act→Conv).
    residual_scale : float
        Scaling factor ``alpha`` for the residual branch.
    bottleneck : bool
        Whether to use a bottleneck block.
    bottleneck_ratio : float
        Channel reduction ratio for bottleneck blocks.
    """

    num_layers_per_block: int = 2
    activation: str = "relu"
    pre_activation: bool = True
    residual_scale: float = 1.0
    bottleneck: bool = False
    bottleneck_ratio: float = 0.25


@dataclass
class ResNetConfig:
    """Top-level ResNet configuration.

    Parameters
    ----------
    num_blocks : int
        Number of residual blocks.
    block_config : ResNetBlockConfig
        Configuration shared by every block.
    width : int
        Hidden layer width.
    input_dim : int
        Dimensionality of the input.
    output_dim : int
        Dimensionality of the output.
    """

    num_blocks: int = 4
    block_config: ResNetBlockConfig = field(default_factory=ResNetBlockConfig)
    width: int = 128
    input_dim: int = 784
    output_dim: int = 10


# ======================================================================
# Result container
# ======================================================================


@dataclass
class ResNetKernelResult:
    """Container for ResNet NTK computation results.

    Parameters
    ----------
    ntk_matrix : ndarray of shape (n, n)
        Full NTK for the ResNet.
    per_block_kernels : list of ndarray
        Kernel at the output of each residual block.
    depth_kernel_curve : ndarray of shape (num_blocks,)
        Frobenius norm of the kernel at each block depth.
    eigenvalue_evolution : ndarray of shape (num_blocks, n)
        Eigenvalues of the kernel at each block depth.
    """

    ntk_matrix: NDArray[np.floating]
    per_block_kernels: List[NDArray[np.floating]] = field(default_factory=list)
    depth_kernel_curve: NDArray[np.floating] = field(
        default_factory=lambda: np.empty(0)
    )
    eigenvalue_evolution: NDArray[np.floating] = field(
        default_factory=lambda: np.empty(0)
    )


# ======================================================================
# Dual-activation helpers
# ======================================================================


def _relu_kappa(k_xx: float, k_yy: float, k_xy: float) -> float:
    """ReLU dual activation (expected inner product under Gaussian).

    Parameters
    ----------
    k_xx, k_yy, k_xy : float
        Kernel entries.

    Returns
    -------
    float
        Dual activation value.
    """
    norms = math.sqrt(max(k_xx * k_yy, 1e-30))
    cos_theta = np.clip(k_xy / norms, -1.0, 1.0)
    theta = math.acos(cos_theta)
    return norms / (2.0 * math.pi) * (math.sin(theta) + (math.pi - theta) * cos_theta)


def _relu_dot_kappa(k_xx: float, k_yy: float, k_xy: float) -> float:
    """Derivative of the ReLU dual activation w.r.t. ``k_xy``.

    Parameters
    ----------
    k_xx, k_yy, k_xy : float
        Kernel entries.

    Returns
    -------
    float
        Derivative value.
    """
    norms = math.sqrt(max(k_xx * k_yy, 1e-30))
    cos_theta = np.clip(k_xy / norms, -1.0, 1.0)
    theta = math.acos(cos_theta)
    return (math.pi - theta) / (2.0 * math.pi)


def _default_dual_activation(q: float) -> float:
    """Scalar dual activation for ReLU: ``kappa(q) = q / 2``.

    Parameters
    ----------
    q : float
        Input variance.

    Returns
    -------
    float
        Post-activation variance.
    """
    return q / 2.0


# ======================================================================
# Activation registry
# ======================================================================


_KAPPA_REGISTRY: Dict[str, Callable[[float, float, float], float]] = {
    "relu": _relu_kappa,
}

_DOT_KAPPA_REGISTRY: Dict[str, Callable[[float, float, float], float]] = {
    "relu": _relu_dot_kappa,
}

_SCALAR_DUAL_REGISTRY: Dict[str, Callable[[float], float]] = {
    "relu": _default_dual_activation,
}


def _get_kappa(activation: str) -> Callable[[float, float, float], float]:
    """Look up dual-activation function by name."""
    if activation in _KAPPA_REGISTRY:
        return _KAPPA_REGISTRY[activation]
    raise ValueError(f"No dual activation registered for '{activation}'")


def _get_dot_kappa(activation: str) -> Callable[[float, float, float], float]:
    """Look up derivative of dual-activation by name."""
    if activation in _DOT_KAPPA_REGISTRY:
        return _DOT_KAPPA_REGISTRY[activation]
    raise ValueError(f"No dot-kappa registered for '{activation}'")


def _get_scalar_dual(activation: str) -> Callable[[float], float]:
    """Look up scalar dual-activation by name."""
    if activation in _SCALAR_DUAL_REGISTRY:
        return _SCALAR_DUAL_REGISTRY[activation]
    raise ValueError(f"No scalar dual registered for '{activation}'")


# ======================================================================
# ResNet NTK computer
# ======================================================================


class ResNetNTKComputer:
    """Compute the infinite-width NTK for ResNet architectures.

    Parameters
    ----------
    config : ResNetConfig
        Architecture configuration.
    activation_dual_fn : callable or None
        Optional override for the dual activation.  If *None*, the
        built-in ReLU dual is used.
    """

    def __init__(
        self,
        config: ResNetConfig,
        activation_dual_fn: Optional[
            Callable[[float, float, float], float]
        ] = None,
    ) -> None:
        self.config = config
        bc = config.block_config
        self._kappa = activation_dual_fn or _get_kappa(bc.activation)
        self._dot_kappa = _get_dot_kappa(bc.activation)
        self._scalar_dual = _get_scalar_dual(bc.activation)

    # ------------------------------------------------------------------
    # Full NTK computation
    # ------------------------------------------------------------------

    def compute_ntk(
        self,
        x1: NDArray[np.floating],
        x2: Optional[NDArray[np.floating]] = None,
    ) -> ResNetKernelResult:
        """Compute the full NTK for the ResNet architecture.

        Parameters
        ----------
        x1 : ndarray of shape (n1, d)
            First set of inputs.
        x2 : ndarray of shape (n2, d) or None
            Second set of inputs.  If *None*, uses *x1*.

        Returns
        -------
        ResNetKernelResult
            Result container with ``ntk_matrix``, ``per_block_kernels``,
            ``depth_kernel_curve``, and ``eigenvalue_evolution``.
        """
        if x2 is None:
            x2 = x1

        n1 = x1.shape[0]
        n2 = x2.shape[0]
        d = x1.shape[1]

        # Initial kernel: K^0 = x1 @ x2.T / d
        K = x1 @ x2.T / d

        bc = self.config.block_config
        alpha = bc.residual_scale

        per_block: List[NDArray[np.floating]] = []
        depth_curve: List[float] = []
        eig_evolution: List[NDArray[np.floating]] = []

        # NTK accumulator (derivative contribution from each block)
        Theta = np.zeros((n1, n2))
        # Derivative chain product (∂K^L / ∂K^l)
        D = np.ones((n1, n2))

        for _block_idx in range(self.config.num_blocks):
            K_prev = K.copy()

            # Forward kernel through one block
            K = self._block_kernel_recursion(K, bc)

            # NTK contribution from this block's parameters
            # Θ += D * (∂block / ∂W) terms, approximated here as D * alpha^2 * dot_kappa
            dot_kappa_mat = self._elementwise_fn(
                K_prev, K_prev, K_prev, self._dot_kappa
            )
            Theta += D * alpha ** 2 * dot_kappa_mat

            # Update derivative chain (linearised recursion Jacobian)
            D = D * (1.0 + alpha ** 2 * dot_kappa_mat)

            per_block.append(K.copy())
            depth_curve.append(float(np.linalg.norm(K, "fro")))

            if n1 == n2:
                eigs = np.sort(np.linalg.eigvalsh(K))[::-1]
            else:
                eigs = np.array([])
            eig_evolution.append(eigs)

        # Final readout layer NTK contribution
        Theta += D

        depth_curve_arr = np.asarray(depth_curve)
        eig_arr = (
            np.vstack(eig_evolution) if eig_evolution and eig_evolution[0].size > 0
            else np.empty(0)
        )

        return ResNetKernelResult(
            ntk_matrix=Theta,
            per_block_kernels=per_block,
            depth_kernel_curve=depth_curve_arr,
            eigenvalue_evolution=eig_arr,
        )

    # ------------------------------------------------------------------
    # Block kernel recursion
    # ------------------------------------------------------------------

    def _block_kernel_recursion(
        self,
        K_in: NDArray[np.floating],
        block_config: ResNetBlockConfig,
    ) -> NDArray[np.floating]:
        """Propagate kernel through one residual block.

        Dispatches to pre-activation, post-activation, or bottleneck
        implementations.

        Parameters
        ----------
        K_in : ndarray of shape (n, n)
            Input kernel.
        block_config : ResNetBlockConfig
            Block configuration.

        Returns
        -------
        ndarray of shape (n, n)
            Output kernel after the block.
        """
        alpha = block_config.residual_scale

        if block_config.bottleneck:
            return self._bottleneck_block(K_in, block_config.bottleneck_ratio, alpha)

        if block_config.pre_activation:
            return self._pre_activation_block(K_in, alpha)

        return self._post_activation_block(K_in, alpha)

    # ------------------------------------------------------------------
    # Dual activation (element-wise)
    # ------------------------------------------------------------------

    def _dual_activation(
        self, K: NDArray[np.floating], activation: str
    ) -> NDArray[np.floating]:
        """Apply the dual activation element-wise on a kernel matrix.

        Parameters
        ----------
        K : ndarray of shape (n, n)
            Input kernel.
        activation : str
            Name of the activation.

        Returns
        -------
        ndarray of shape (n, n)
            Post-activation kernel.
        """
        kappa = _get_kappa(activation)
        return self._elementwise_fn(K, K, K, kappa)

    # ------------------------------------------------------------------
    # Block implementations
    # ------------------------------------------------------------------

    def _pre_activation_block(
        self,
        K: NDArray[np.floating],
        alpha: float,
    ) -> NDArray[np.floating]:
        """Pre-activation residual block kernel.

        ``K_out = K + alpha^2 * kappa(kappa(K))``

        Two layers of activation inside the block, matching the standard
        pre-activation ResNet layout.

        Parameters
        ----------
        K : ndarray of shape (n, n)
            Input kernel.
        alpha : float
            Residual scaling.

        Returns
        -------
        ndarray of shape (n, n)
            Output kernel.
        """
        bc = self.config.block_config
        K_inner = self._dual_activation(K, bc.activation)
        K_inner = self._dual_activation(K_inner, bc.activation)
        return K + alpha ** 2 * K_inner

    def _post_activation_block(
        self,
        K: NDArray[np.floating],
        alpha: float,
    ) -> NDArray[np.floating]:
        """Post-activation residual block kernel.

        ``K_out = kappa(K + alpha^2 * kappa(K))``

        Parameters
        ----------
        K : ndarray of shape (n, n)
            Input kernel.
        alpha : float
            Residual scaling.

        Returns
        -------
        ndarray of shape (n, n)
            Output kernel.
        """
        bc = self.config.block_config
        K_inner = self._dual_activation(K, bc.activation)
        K_sum = K + alpha ** 2 * K_inner
        return self._dual_activation(K_sum, bc.activation)

    def _bottleneck_block(
        self,
        K: NDArray[np.floating],
        ratio: float,
        alpha: float,
    ) -> NDArray[np.floating]:
        """Bottleneck residual block kernel.

        Three layers: 1×1 reduce → 3×3 → 1×1 expand, modelled in kernel
        space as three successive dual-activation applications scaled by
        the bottleneck ratio.

        Parameters
        ----------
        K : ndarray of shape (n, n)
            Input kernel.
        ratio : float
            Bottleneck reduction ratio.
        alpha : float
            Residual scaling.

        Returns
        -------
        ndarray of shape (n, n)
            Output kernel.
        """
        bc = self.config.block_config
        # 1×1 reduce
        K_inner = ratio * self._dual_activation(K, bc.activation)
        # 3×3
        K_inner = self._dual_activation(K_inner, bc.activation)
        # 1×1 expand
        K_inner = (1.0 / ratio) * self._dual_activation(K_inner, bc.activation)
        return K + alpha ** 2 * K_inner

    # ------------------------------------------------------------------
    # Depth analysis
    # ------------------------------------------------------------------

    def depth_analysis(
        self,
        x: NDArray[np.floating],
        max_depth: int,
    ) -> Dict[str, Any]:
        """Analyse how the kernel changes with depth.

        Parameters
        ----------
        x : ndarray of shape (n, d)
            Input data.
        max_depth : int
            Maximum number of blocks to iterate.

        Returns
        -------
        dict
            ``kernels`` : list of ndarray
                Kernel at each depth.
            ``convergence`` : ndarray of shape (max_depth,)
                Frobenius-norm change between successive kernels.
            ``rate`` : float
                Estimated convergence rate (geometric decay constant).
        """
        d = x.shape[1]
        K = x @ x.T / d
        bc = self.config.block_config

        kernels: List[NDArray[np.floating]] = [K.copy()]
        convergence: List[float] = []

        for _ in range(max_depth):
            K_prev = K.copy()
            K = self._block_kernel_recursion(K, bc)
            kernels.append(K.copy())
            convergence.append(float(np.linalg.norm(K - K_prev, "fro")))

        conv_arr = np.asarray(convergence)
        # Estimate geometric decay rate from last two changes
        if len(conv_arr) >= 2 and conv_arr[-2] > 1e-30:
            rate = float(conv_arr[-1] / conv_arr[-2])
        else:
            rate = 0.0

        return {
            "kernels": kernels,
            "convergence": conv_arr,
            "rate": rate,
        }

    # ------------------------------------------------------------------
    # Residual scaling analysis
    # ------------------------------------------------------------------

    def residual_scaling_analysis(
        self,
        x: NDArray[np.floating],
        alphas: Sequence[float],
    ) -> Dict[str, Any]:
        """Analyse the effect of residual scaling on the kernel.

        Parameters
        ----------
        x : ndarray of shape (n, d)
            Input data.
        alphas : sequence of float
            Residual scaling values to sweep.

        Returns
        -------
        dict
            ``alphas`` : ndarray
                Scaling values.
            ``kernel_norms`` : ndarray
                Frobenius norm of final kernel for each alpha.
            ``eigenvalue_gaps`` : ndarray
                Top eigenvalue gap for each alpha.
        """
        d = x.shape[1]
        K_init = x @ x.T / d
        n = K_init.shape[0]

        norms: List[float] = []
        gaps: List[float] = []

        original_scale = self.config.block_config.residual_scale
        try:
            for a in alphas:
                self.config.block_config.residual_scale = a
                K = K_init.copy()
                for _ in range(self.config.num_blocks):
                    K = self._block_kernel_recursion(K, self.config.block_config)
                norms.append(float(np.linalg.norm(K, "fro")))
                eigs = np.sort(np.linalg.eigvalsh(K))[::-1]
                gap = float(eigs[0] - eigs[1]) if len(eigs) > 1 else float(eigs[0])
                gaps.append(gap)
        finally:
            self.config.block_config.residual_scale = original_scale

        return {
            "alphas": np.asarray(alphas),
            "kernel_norms": np.asarray(norms),
            "eigenvalue_gaps": np.asarray(gaps),
        }

    # ------------------------------------------------------------------
    # Eigenvalue evolution
    # ------------------------------------------------------------------

    def eigenvalue_evolution(
        self,
        x: NDArray[np.floating],
        depths: Sequence[int],
    ) -> NDArray[np.floating]:
        """Track kernel eigenvalues across depth.

        Parameters
        ----------
        x : ndarray of shape (n, d)
            Input data.
        depths : sequence of int
            Block depths at which to record eigenvalues.

        Returns
        -------
        ndarray of shape (len(depths), n)
            Eigenvalues (descending) at each requested depth.
        """
        d = x.shape[1]
        n = x.shape[0]
        K = x @ x.T / d
        bc = self.config.block_config

        max_depth = max(depths)
        depth_set = set(depths)
        results: List[NDArray[np.floating]] = []

        for blk in range(1, max_depth + 1):
            K = self._block_kernel_recursion(K, bc)
            if blk in depth_set:
                eigs = np.sort(np.linalg.eigvalsh(K))[::-1]
                results.append(eigs)

        return np.vstack(results)

    # ------------------------------------------------------------------
    # Pre- vs post-activation comparison
    # ------------------------------------------------------------------

    def compare_pre_post_activation(
        self,
        x: NDArray[np.floating],
    ) -> Dict[str, Any]:
        """Compare pre-activation and post-activation ResNet kernels.

        Parameters
        ----------
        x : ndarray of shape (n, d)
            Input data.

        Returns
        -------
        dict
            ``pre_kernel`` : ndarray
                Final kernel with pre-activation blocks.
            ``post_kernel`` : ndarray
                Final kernel with post-activation blocks.
            ``frobenius_diff`` : float
                Frobenius-norm difference between the two.
            ``spectral_diff`` : ndarray
                Absolute difference of sorted eigenvalues.
        """
        d = x.shape[1]
        K_init = x @ x.T / d
        bc = self.config.block_config
        alpha = bc.residual_scale

        K_pre = K_init.copy()
        K_post = K_init.copy()

        for _ in range(self.config.num_blocks):
            K_pre = self._pre_activation_block(K_pre, alpha)
            K_post = self._post_activation_block(K_post, alpha)

        frob = float(np.linalg.norm(K_pre - K_post, "fro"))
        eigs_pre = np.sort(np.linalg.eigvalsh(K_pre))[::-1]
        eigs_post = np.sort(np.linalg.eigvalsh(K_post))[::-1]
        spectral_diff = np.abs(eigs_pre - eigs_post)

        return {
            "pre_kernel": K_pre,
            "post_kernel": K_post,
            "frobenius_diff": frob,
            "spectral_diff": spectral_diff,
        }

    # ------------------------------------------------------------------
    # Fixed-point kernel
    # ------------------------------------------------------------------

    def fixed_point_kernel(
        self,
        K_init: NDArray[np.floating],
        tol: float = 1e-8,
        max_iter: int = 10_000,
    ) -> NDArray[np.floating]:
        """Iterate the kernel recursion to its fixed point (infinite depth limit).

        Parameters
        ----------
        K_init : ndarray of shape (n, n)
            Initial kernel.
        tol : float
            Convergence tolerance (Frobenius norm of change).
        max_iter : int
            Maximum number of iterations.

        Returns
        -------
        ndarray of shape (n, n)
            Fixed-point kernel.
        """
        bc = self.config.block_config
        K = K_init.copy()
        for _ in range(max_iter):
            K_new = self._block_kernel_recursion(K, bc)
            if np.linalg.norm(K_new - K, "fro") < tol:
                return K_new
            K = K_new
        return K

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _elementwise_fn(
        K: NDArray[np.floating],
        K_diag_row: NDArray[np.floating],
        K_diag_col: NDArray[np.floating],
        fn: Callable[[float, float, float], float],
    ) -> NDArray[np.floating]:
        """Apply *fn(k_xx, k_yy, k_xy)* element-wise over a kernel.

        Parameters
        ----------
        K : ndarray of shape (n1, n2)
            Kernel matrix.
        K_diag_row : ndarray of shape (n1, n1)
            Kernel whose diagonal gives row variances.
        K_diag_col : ndarray of shape (n2, n2)
            Kernel whose diagonal gives column variances.
        fn : callable
            Scalar function ``(k_xx, k_yy, k_xy) -> float``.

        Returns
        -------
        ndarray of shape (n1, n2)
            Result matrix.
        """
        n1, n2 = K.shape
        out = np.empty((n1, n2))
        diag_r = np.diag(K_diag_row) if K_diag_row.ndim == 2 else K_diag_row
        diag_c = np.diag(K_diag_col) if K_diag_col.ndim == 2 else K_diag_col
        for i in range(n1):
            for j in range(n2):
                out[i, j] = fn(diag_r[i], diag_c[j], K[i, j])
        return out
