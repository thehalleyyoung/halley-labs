"""Skip connection handling for kernel computation.

Provides kernel propagation through various skip-connection topologies
including additive (ResNet), multiplicative (Hadamard), dense (DenseNet),
and highway (gated) connections.  Each topology transforms the NNGP / NTK
kernel matrices differently; this module implements the corresponding
kernel-space updates and gradient-flow diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as sp_linalg


# ======================================================================
# Skip connection types
# ======================================================================


class SkipType(Enum):
    """Enumeration of supported skip-connection topologies."""

    ADDITIVE = auto()
    MULTIPLICATIVE = auto()
    DENSE = auto()
    HIGHWAY = auto()


# ======================================================================
# Configuration
# ======================================================================


@dataclass
class SkipConfig:
    """Configuration for a skip connection.

    Parameters
    ----------
    skip_type : SkipType
        The topology of the skip connection.
    alpha : float
        Residual scaling factor.  For additive connections the output
        kernel is ``K_skip + 2*alpha*K_cross + alpha**2 * K_residual``.
    gate_fn : callable or None
        Gating function for HIGHWAY connections.  Receives a scalar
        correlation and returns a gate value in [0, 1].
    dense_growth_rate : int
        Channel growth rate for DENSE (DenseNet-style) connections.
    """

    skip_type: SkipType = SkipType.ADDITIVE
    alpha: float = 1.0
    gate_fn: Optional[Callable[[float], float]] = None
    dense_growth_rate: int = 12


# ======================================================================
# Skip connection handler
# ======================================================================


class SkipConnectionHandler:
    """Apply kernel-space updates for skip connections.

    Parameters
    ----------
    config : SkipConfig
        Skip connection configuration.
    """

    def __init__(self, config: SkipConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_skip_kernel(
        self,
        K_residual: NDArray[np.floating],
        K_skip: NDArray[np.floating],
        K_cross: Optional[NDArray[np.floating]] = None,
    ) -> NDArray[np.floating]:
        """Combine residual and skip kernels according to the configured topology.

        Parameters
        ----------
        K_residual : ndarray of shape (n, n)
            Kernel matrix from the residual branch.
        K_skip : ndarray of shape (n, n)
            Kernel matrix from the skip (identity) branch.
        K_cross : ndarray of shape (n, n) or None
            Cross-covariance between the two branches.  Required for
            ADDITIVE connections; ignored otherwise.

        Returns
        -------
        ndarray of shape (n, n)
            Combined output kernel.
        """
        stype = self.config.skip_type

        if stype is SkipType.ADDITIVE:
            if K_cross is None:
                K_cross = K_skip.copy()
            return self._additive_kernel(
                K_residual, K_skip, K_cross, self.config.alpha
            )

        if stype is SkipType.MULTIPLICATIVE:
            return self._multiplicative_kernel(K_residual, K_skip)

        if stype is SkipType.DENSE:
            return self._dense_kernel([K_skip, K_residual])

        if stype is SkipType.HIGHWAY:
            gate_vals = self._compute_gate_values(K_skip)
            return self._highway_kernel(K_residual, K_skip, gate_vals)

        raise ValueError(f"Unsupported skip type: {stype}")

    # ------------------------------------------------------------------
    # Kernel combination helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _additive_kernel(
        K_res: NDArray[np.floating],
        K_skip: NDArray[np.floating],
        K_cross: NDArray[np.floating],
        alpha: float,
    ) -> NDArray[np.floating]:
        """Additive (ResNet-style) kernel update.

        Parameters
        ----------
        K_res : ndarray of shape (n, n)
            Residual branch kernel.
        K_skip : ndarray of shape (n, n)
            Skip branch kernel.
        K_cross : ndarray of shape (n, n)
            Cross-covariance kernel.
        alpha : float
            Residual scaling factor.

        Returns
        -------
        ndarray of shape (n, n)
            ``K_skip + 2*alpha*K_cross + alpha**2 * K_res``.
        """
        return K_skip + 2.0 * alpha * K_cross + alpha ** 2 * K_res

    @staticmethod
    def _multiplicative_kernel(
        K_res: NDArray[np.floating],
        K_skip: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Multiplicative (Hadamard) kernel update.

        Parameters
        ----------
        K_res : ndarray of shape (n, n)
            Residual branch kernel.
        K_skip : ndarray of shape (n, n)
            Skip branch kernel.

        Returns
        -------
        ndarray of shape (n, n)
            Element-wise product ``K_skip * K_res``.
        """
        return K_skip * K_res

    @staticmethod
    def _dense_kernel(
        K_list: List[NDArray[np.floating]],
    ) -> NDArray[np.floating]:
        """DenseNet-style kernel from all preceding layer kernels.

        In a DenseNet, the output of layer *l* is the concatenation of
        all preceding feature maps.  In kernel space this translates to
        summing the kernels from every preceding layer (under the
        independence-at-initialisation assumption).

        Parameters
        ----------
        K_list : list of ndarray of shape (n, n)
            Kernel matrices from all preceding layers (including the
            current residual branch).

        Returns
        -------
        ndarray of shape (n, n)
            Sum of all kernel matrices in *K_list*.
        """
        K_out = np.zeros_like(K_list[0])
        for K in K_list:
            K_out = K_out + K
        return K_out

    @staticmethod
    def _highway_kernel(
        K_res: NDArray[np.floating],
        K_skip: NDArray[np.floating],
        gate_values: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Highway (gated) kernel combination.

        Parameters
        ----------
        K_res : ndarray of shape (n, n)
            Residual branch kernel.
        K_skip : ndarray of shape (n, n)
            Skip branch kernel.
        gate_values : ndarray of shape (n, n)
            Gate values in [0, 1].

        Returns
        -------
        ndarray of shape (n, n)
            ``gate_values * K_res + (1 - gate_values) * K_skip``.
        """
        return gate_values * K_res + (1.0 - gate_values) * K_skip

    # ------------------------------------------------------------------
    # Gate computation
    # ------------------------------------------------------------------

    def _compute_gate_values(
        self, K: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Compute element-wise gate values for highway connections.

        If no ``gate_fn`` is configured, falls back to a sigmoid with
        bias -1 (i.e. gate ≈ 0.27 at zero, favouring the skip path).

        Parameters
        ----------
        K : ndarray of shape (n, n)
            Input kernel used to derive correlations.

        Returns
        -------
        ndarray of shape (n, n)
            Gate values in [0, 1].
        """
        diag = np.sqrt(np.maximum(np.diag(K), 1e-30))
        corr = K / np.outer(diag, diag)
        corr = np.clip(corr, -1.0, 1.0)

        if self.config.gate_fn is not None:
            vfn = np.vectorize(self.config.gate_fn)
            return vfn(corr)

        # Default sigmoid gate with bias favouring the skip path.
        return 1.0 / (1.0 + np.exp(-(corr - 1.0)))

    # ------------------------------------------------------------------
    # Gradient flow analysis
    # ------------------------------------------------------------------

    def gradient_flow_analysis(
        self,
        depth: int,
        alpha: float,
        activation_dual: Callable[[float], float],
    ) -> Dict[str, Any]:
        """Analyse gradient flow through residual connections.

        Uses mean-field theory to estimate how gradient norms behave
        across *depth* residual blocks with scaling factor *alpha*.

        Parameters
        ----------
        depth : int
            Number of residual blocks.
        alpha : float
            Residual scaling factor.
        activation_dual : callable
            Dual activation function ``kappa(q)`` mapping a kernel
            diagonal entry to the post-activation value.

        Returns
        -------
        dict
            ``effective_depth`` : float
                Effective depth accounting for skip connections.
            ``gradient_norms`` : ndarray of shape (depth,)
                Estimated gradient norm at each layer.
            ``signal_propagation`` : dict
                Mean-field signal propagation statistics including
                ``q_star`` (fixed-point variance), ``chi`` (gradient
                multiplier), and ``convergence_rate``.
        """
        # -- Effective depth --
        eff_depth = self.effective_depth(depth, alpha)

        # -- Signal propagation (mean-field) --
        q = 1.0  # initial variance
        q_trajectory: List[float] = [q]
        for _ in range(depth):
            q_new = q + alpha ** 2 * activation_dual(q)
            q_trajectory.append(q_new)
            q = q_new

        q_star = q_trajectory[-1]

        # Gradient multiplier chi = d q^{l+1} / d q^l evaluated at q*
        eps = 1e-6
        chi = 1.0 + alpha ** 2 * (
            (activation_dual(q_star + eps) - activation_dual(q_star - eps))
            / (2.0 * eps)
        )

        # -- Gradient norms --
        grad_norms = np.empty(depth)
        gn = 1.0
        for l in range(depth - 1, -1, -1):
            gn *= chi
            grad_norms[l] = gn

        # Convergence rate (spectral radius of the linearised map)
        convergence_rate = abs(chi)

        signal_propagation = {
            "q_star": q_star,
            "chi": chi,
            "convergence_rate": convergence_rate,
            "q_trajectory": np.asarray(q_trajectory),
        }

        return {
            "effective_depth": eff_depth,
            "gradient_norms": grad_norms,
            "signal_propagation": signal_propagation,
        }

    # ------------------------------------------------------------------
    # Effective depth
    # ------------------------------------------------------------------

    @staticmethod
    def effective_depth(actual_depth: int, alpha: float) -> float:
        """Effective depth accounting for skip connections.

        With residual scaling *alpha*, the effective depth is reduced
        because skip connections shorten the gradient path.  A simple
        model gives ``actual_depth * alpha^2 / (1 + alpha^2)``.

        Parameters
        ----------
        actual_depth : int
            Number of layers / blocks.
        alpha : float
            Residual scaling factor.

        Returns
        -------
        float
            Effective depth.
        """
        return actual_depth * alpha ** 2 / (1.0 + alpha ** 2)

    # ------------------------------------------------------------------
    # Kernel through specific skip path
    # ------------------------------------------------------------------

    @staticmethod
    def kernel_through_skip_path(
        K_input: NDArray[np.floating],
        skip_indices: Sequence[int],
        layer_kernels: Sequence[NDArray[np.floating]],
    ) -> NDArray[np.floating]:
        """Compute kernel contribution from a specific skip path.

        Given a list of *layer_kernels* ``[K^0, K^1, …, K^L]`` and a
        set of *skip_indices* indicating which layers are bypassed, this
        method returns the kernel that results from following only the
        skip edges at the indicated layers.

        Parameters
        ----------
        K_input : ndarray of shape (n, n)
            Input kernel ``K^0``.
        skip_indices : sequence of int
            Layer indices where the skip connection is taken.
        layer_kernels : sequence of ndarray of shape (n, n)
            Kernel at the output of every layer ``[K^0, …, K^L]``.

        Returns
        -------
        ndarray of shape (n, n)
            Kernel resulting from the chosen path.
        """
        K = K_input.copy()
        skip_set = set(skip_indices)
        for idx, K_layer in enumerate(layer_kernels):
            if idx in skip_set:
                # Skip: keep current kernel unchanged (identity path).
                continue
            else:
                # Non-skip: advance through the layer kernel.
                K = K_layer.copy()
        return K
