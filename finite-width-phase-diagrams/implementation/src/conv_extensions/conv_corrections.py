"""Convolutional finite-width corrections via Kronecker factorisation.

Provides:
  - ConvCorrectionConfig: configuration tying the dense H tensor to the
    patch Gram matrix
  - ConvCorrectionResult: container for correction data
  - ConvFiniteWidthCorrector: compute and validate 1/N corrections for
    convolutional layers
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as sp_linalg


# ======================================================================
# Configuration & result containers
# ======================================================================


@dataclass
class ConvCorrectionConfig:
    """Configuration for convolutional finite-width corrections.

    Attributes
    ----------
    dense_h_tensor : NDArray
        Dense H tensor of shape ``(O, P_dense, P_dense)`` from the
        corrections module (:class:`HTensor`).  This is the channel-only
        Hessian tensor for a single spatial location.
    patch_gram : NDArray
        Patch Gram matrix of shape ``(S, S)`` encoding how receptive
        fields overlap.
    max_order : int
        Maximum order of the 1/N expansion to compute (default 2).
    factorize : bool
        If ``True`` (default) use the Kronecker factorisation
        :math:`H^{\\text{conv}} = H^{\\text{dense}} \\otimes K^{\\text{patch}}`
        instead of brute-force computation.
    """

    dense_h_tensor: NDArray
    patch_gram: NDArray
    max_order: int = 2
    factorize: bool = True


@dataclass
class ConvCorrectionResult:
    """Container for convolutional correction computation results.

    Attributes
    ----------
    correction_matrix : NDArray
        Leading-order (1/N) correction matrix of shape
        ``(N_out * S, N_out * S)``.
    factored_dense : NDArray
        Dense (channel) factor of the correction.
    factored_patch : NDArray
        Patch factor of the correction.
    factorization_error : float
        Relative Frobenius-norm error of the Kronecker factorisation
        versus the brute-force H tensor.  ``0.0`` if not validated.
    per_location_corrections : List[NDArray]
        Correction matrices broken out per spatial location.
    """

    correction_matrix: NDArray
    factored_dense: NDArray
    factored_patch: NDArray
    factorization_error: float = 0.0
    per_location_corrections: List[NDArray] = field(default_factory=list)


# ======================================================================
# Convolutional Finite-Width Corrector
# ======================================================================


class ConvFiniteWidthCorrector:
    r"""Compute finite-width corrections for convolutional layers.

    The central hypothesis is that the Hessian tensor for a convolutional
    layer factorises as a Kronecker product of the dense (channel)
    Hessian and the patch Gram matrix:

    .. math::
        H^{\text{conv}} \;=\; H^{\text{dense}} \;\otimes\; K^{\text{patch}}

    This factorisation is exact when the activation Hessian and the
    patch structure are independent, and provides a computationally
    efficient route to the 1/N correction.

    Parameters
    ----------
    config : ConvCorrectionConfig
        Configuration with the dense H tensor, patch Gram, and options.
    """

    def __init__(self, config: ConvCorrectionConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Kronecker product
    # ------------------------------------------------------------------

    @staticmethod
    def _kronecker_product(A: NDArray, B: NDArray) -> NDArray:
        r"""Efficient Kronecker product :math:`A \otimes B`.

        Parameters
        ----------
        A : NDArray of shape ``(m, n)``
        B : NDArray of shape ``(p, q)``

        Returns
        -------
        NDArray of shape ``(m*p, n*q)``
        """
        m, n = A.shape
        p, q = B.shape
        return np.einsum("ij,kl->ikjl", A, B).reshape(m * p, n * q)

    # ------------------------------------------------------------------
    # Kronecker-factored H tensor for convolutions
    # ------------------------------------------------------------------

    def compute_h_conv(
        self,
        dense_h: NDArray,
        patch_gram: NDArray,
    ) -> NDArray:
        r"""Compute the convolutional H tensor via Kronecker factorisation.

        .. math::
            H^{\text{conv}}_{(o),(ij),(kl)}
            = H^{\text{dense}}_{o,i,k} \; K^{\text{patch}}_{j,l}

        For a single output index *o*, the H tensor slice is
        :math:`H^{\text{dense}}_o \otimes K^{\text{patch}}`.

        Parameters
        ----------
        dense_h : NDArray of shape ``(O, P_d, P_d)``
            Dense H tensor.
        patch_gram : NDArray of shape ``(S, S)``
            Patch Gram matrix.

        Returns
        -------
        h_conv : NDArray of shape ``(O, P_d * S, P_d * S)``
            Convolutional H tensor.
        """
        O, Pd, _ = dense_h.shape
        S = patch_gram.shape[0]
        h_conv = np.zeros((O, Pd * S, Pd * S), dtype=dense_h.dtype)

        for o in range(O):
            h_conv[o] = self._kronecker_product(dense_h[o], patch_gram)

        return h_conv

    # ------------------------------------------------------------------
    # Full correction computation
    # ------------------------------------------------------------------

    def compute_correction(
        self,
        kernel_matrix: NDArray,
        width: int,
    ) -> ConvCorrectionResult:
        r"""Compute the 1/N correction for a convolutional layer.

        The leading-order correction to the NTK is:

        .. math::
            \Theta^{(1)} = -\frac{1}{N}\,\text{tr}_{\text{param}}
            \bigl(H^{\text{conv}}\bigr)

        Parameters
        ----------
        kernel_matrix : NDArray
            Infinite-width convolutional NTK.
        width : int
            Layer width (number of channels/filters) *N*.

        Returns
        -------
        ConvCorrectionResult
            Correction matrices and diagnostics.
        """
        cfg = self.config
        dense_h = cfg.dense_h_tensor
        patch_gram = cfg.patch_gram

        O, Pd, _ = dense_h.shape
        S = patch_gram.shape[0]

        if cfg.factorize:
            # Factored computation: trace over parameters
            # tr_param(H^dense_o) gives a scalar per output o
            dense_trace = np.array(
                [np.trace(dense_h[o]) for o in range(O)]
            )
            # The correction for each output is dense_trace[o] * K^patch / N
            correction_dense = np.diag(dense_trace)  # (O, O)
            correction_patch = patch_gram.copy()

            # Full correction: Kronecker product divided by width
            if O > 0 and S > 0:
                correction_matrix = (
                    self._kronecker_product(correction_dense, correction_patch)
                    / width
                )
            else:
                correction_matrix = np.zeros(
                    (O * S, O * S), dtype=dense_h.dtype
                )

            factored_dense = correction_dense
            factored_patch = correction_patch
        else:
            # Brute-force: build full H^conv then trace
            h_conv = self.compute_h_conv(dense_h, patch_gram)
            correction_matrix = np.zeros(
                (O * S, O * S), dtype=dense_h.dtype
            )
            for o in range(O):
                # parameter trace: sum over diagonal of H^conv_o
                correction_matrix += h_conv[o]
            correction_matrix /= width

            factored_dense = np.zeros((O, O))
            factored_patch = patch_gram.copy()

        per_loc = self.per_spatial_location(correction_matrix, O, S)

        return ConvCorrectionResult(
            correction_matrix=correction_matrix,
            factored_dense=factored_dense,
            factored_patch=factored_patch,
            factorization_error=0.0,
            per_location_corrections=per_loc,
        )

    # ------------------------------------------------------------------
    # Factorisation validation
    # ------------------------------------------------------------------

    def validate_factorization(
        self,
        h_conv_factored: NDArray,
        h_conv_brute: NDArray,
        tolerance: float = 1e-3,
    ) -> Dict[str, Any]:
        """Compare Kronecker-factored and brute-force H tensors.

        Parameters
        ----------
        h_conv_factored : NDArray of shape ``(O, P, P)``
            Factored convolutional H tensor.
        h_conv_brute : NDArray of shape ``(O, P, P)``
            Brute-force convolutional H tensor.
        tolerance : float
            Maximum allowed relative error for the factorisation to be
            considered valid.

        Returns
        -------
        dict
            ``'valid'`` : bool — whether the factorisation is within
            tolerance.
            ``'relative_error'`` : float — per-output-averaged relative
            Frobenius-norm error.
            ``'per_output_errors'`` : NDArray — error for each output
            index.
            ``'max_abs_diff'`` : float — maximum absolute element-wise
            difference.
        """
        O = h_conv_factored.shape[0]
        per_output_errors = np.zeros(O)

        for o in range(O):
            nrm = np.linalg.norm(h_conv_brute[o])
            if nrm < 1e-30:
                per_output_errors[o] = 0.0
            else:
                per_output_errors[o] = (
                    np.linalg.norm(h_conv_factored[o] - h_conv_brute[o]) / nrm
                )

        rel_error = float(np.mean(per_output_errors))
        max_abs = float(np.max(np.abs(h_conv_factored - h_conv_brute)))

        return {
            "valid": rel_error < tolerance,
            "relative_error": rel_error,
            "per_output_errors": per_output_errors,
            "max_abs_diff": max_abs,
        }

    # ------------------------------------------------------------------
    # Brute-force H tensor computation
    # ------------------------------------------------------------------

    def _brute_force_h_tensor(
        self,
        forward_fn: Callable[[NDArray, NDArray], NDArray],
        params: NDArray,
        x: NDArray,
        eps: float = 1e-4,
    ) -> NDArray:
        r"""Compute the H tensor directly via finite differences.

        .. math::
            H_{o,j,k} = \frac{\partial^2 f_o}{\partial \theta_j
            \partial \theta_k}

        Parameters
        ----------
        forward_fn : callable (params, x) -> output
            Network forward function.
        params : NDArray of shape ``(P,)``
            Flattened parameter vector.
        x : NDArray
            Single input (no batch axis).
        eps : float
            Step size for central finite differences.

        Returns
        -------
        H : NDArray of shape ``(O, P, P)``
            Full Hessian tensor.
        """
        f0 = forward_fn(params, x).ravel()
        O = f0.size
        P = params.size
        H = np.zeros((O, P, P), dtype=params.dtype)

        for j in range(P):
            for k in range(j, P):
                pp = params.copy()
                pp[j] += eps
                pp[k] += eps
                f_pp = forward_fn(pp, x).ravel()

                pm = params.copy()
                pm[j] += eps
                pm[k] -= eps
                f_pm = forward_fn(pm, x).ravel()

                mp = params.copy()
                mp[j] -= eps
                mp[k] += eps
                f_mp = forward_fn(mp, x).ravel()

                mm = params.copy()
                mm[j] -= eps
                mm[k] -= eps
                f_mm = forward_fn(mm, x).ravel()

                H[:, j, k] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)
                H[:, k, j] = H[:, j, k]

        return H

    def _compute_brute_force_conv_h(
        self,
        forward_fn: Callable[[NDArray, NDArray], NDArray],
        params: NDArray,
        x: NDArray,
        kernel_size: Tuple[int, ...],
        eps: float = 1e-4,
    ) -> NDArray:
        """Compute the H tensor directly for a convolutional layer.

        This is a wrapper around :meth:`_brute_force_h_tensor` that
        documents the convolution-specific context.

        Parameters
        ----------
        forward_fn : callable (params, x) -> output
            Network forward function.
        params : NDArray of shape ``(P,)``
            Flattened parameter vector.
        x : NDArray
            Single input.
        kernel_size : tuple of int
            Kernel size (for documentation; not used in computation as
            the forward function already encodes the architecture).
        eps : float
            Finite-difference step.

        Returns
        -------
        H : NDArray of shape ``(O, P, P)``
        """
        return self._brute_force_h_tensor(forward_fn, params, x, eps=eps)

    # ------------------------------------------------------------------
    # Per-spatial-location corrections
    # ------------------------------------------------------------------

    @staticmethod
    def per_spatial_location(
        correction: NDArray,
        num_outputs: int,
        num_spatial: int,
    ) -> List[NDArray]:
        """Break the correction matrix into per-location sub-matrices.

        Parameters
        ----------
        correction : NDArray of shape ``(O * S, O * S)``
            Full correction matrix.
        num_outputs : int
            Number of output channels *O*.
        num_spatial : int
            Number of spatial positions *S*.

        Returns
        -------
        list of NDArray
            *S* matrices of shape ``(O, O)``, one per spatial location.
        """
        if correction.size == 0:
            return []

        per_loc: List[NDArray] = []
        for s in range(num_spatial):
            # Extract the O×O block for location s
            indices = [o * num_spatial + s for o in range(num_outputs)]
            block = correction[np.ix_(indices, indices)]
            per_loc.append(block)

        return per_loc

    # ------------------------------------------------------------------
    # Correction magnitude analysis
    # ------------------------------------------------------------------

    @staticmethod
    def correction_magnitude_analysis(
        corrections: List[NDArray],
    ) -> Dict[str, Any]:
        """Compute statistics on correction sizes across spatial locations.

        Parameters
        ----------
        corrections : list of NDArray
            Per-location correction matrices, as returned by
            :meth:`per_spatial_location`.

        Returns
        -------
        dict
            ``'frobenius_norms'`` : NDArray — Frobenius norm of each
            location's correction.
            ``'mean_norm'`` : float
            ``'std_norm'`` : float
            ``'max_norm'`` : float
            ``'min_norm'`` : float
            ``'max_eigenvalue_per_location'`` : NDArray
            ``'spatial_uniformity'`` : float — coefficient of variation
            of the norms (0 = perfectly uniform).
        """
        if not corrections:
            return {
                "frobenius_norms": np.array([]),
                "mean_norm": 0.0,
                "std_norm": 0.0,
                "max_norm": 0.0,
                "min_norm": 0.0,
                "max_eigenvalue_per_location": np.array([]),
                "spatial_uniformity": 0.0,
            }

        norms = np.array([np.linalg.norm(c) for c in corrections])
        max_evals = np.array(
            [
                np.max(np.abs(sp_linalg.eigvalsh(0.5 * (c + c.T))))
                if c.size > 0
                else 0.0
                for c in corrections
            ]
        )
        mean_norm = float(np.mean(norms))
        std_norm = float(np.std(norms))

        return {
            "frobenius_norms": norms,
            "mean_norm": mean_norm,
            "std_norm": std_norm,
            "max_norm": float(np.max(norms)),
            "min_norm": float(np.min(norms)),
            "max_eigenvalue_per_location": max_evals,
            "spatial_uniformity": std_norm / mean_norm if mean_norm > 1e-30 else 0.0,
        }

    # ------------------------------------------------------------------
    # Spectrum comparison
    # ------------------------------------------------------------------

    @staticmethod
    def spectrum_comparison(
        factored: NDArray,
        brute_force: NDArray,
    ) -> Dict[str, Any]:
        """Compare eigenvalue spectra of factored and brute-force tensors.

        Parameters
        ----------
        factored : NDArray of shape ``(O, P, P)``
            Kronecker-factored convolutional H tensor.
        brute_force : NDArray of shape ``(O, P, P)``
            Brute-force convolutional H tensor.

        Returns
        -------
        dict
            ``'eigenvalue_errors'`` : NDArray — per-output relative
            eigenvalue difference.
            ``'mean_spectral_error'`` : float
            ``'max_spectral_error'`` : float
            ``'subspace_angles'`` : NDArray — principal angles between
            leading eigenspaces (per output).
        """
        O = factored.shape[0]
        eigenvalue_errors = np.zeros(O)
        subspace_angles = np.zeros(O)

        for o in range(O):
            f_sym = 0.5 * (factored[o] + factored[o].T)
            b_sym = 0.5 * (brute_force[o] + brute_force[o].T)

            evals_f = sp_linalg.eigvalsh(f_sym)[::-1]
            evals_b = sp_linalg.eigvalsh(b_sym)[::-1]

            nrm = np.linalg.norm(evals_b)
            if nrm < 1e-30:
                eigenvalue_errors[o] = 0.0
            else:
                eigenvalue_errors[o] = np.linalg.norm(evals_f - evals_b) / nrm

            # Leading subspace angle
            _, evecs_f = sp_linalg.eigh(f_sym)
            _, evecs_b = sp_linalg.eigh(b_sym)
            k = min(max(1, factored.shape[1] // 4), factored.shape[1])
            Vf = evecs_f[:, -k:]
            Vb = evecs_b[:, -k:]
            svs = np.linalg.svd(Vf.T @ Vb, compute_uv=False)
            svs = np.clip(svs, 0.0, 1.0)
            subspace_angles[o] = float(np.mean(np.arccos(svs)))

        return {
            "eigenvalue_errors": eigenvalue_errors,
            "mean_spectral_error": float(np.mean(eigenvalue_errors)),
            "max_spectral_error": float(np.max(eigenvalue_errors)),
            "subspace_angles": subspace_angles,
        }
