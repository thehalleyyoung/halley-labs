"""Patch Gram matrix construction and analysis for convolutional layers.

Provides:
  - PatchGramConfig: configuration for patch extraction
  - PatchGramMatrix: compute, decompose, and analyse the patch Gram matrix
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as sp_linalg
from scipy.linalg import toeplitz
from scipy import sparse


# ======================================================================
# Configuration
# ======================================================================


@dataclass
class PatchGramConfig:
    """Configuration for the patch Gram matrix.

    Attributes
    ----------
    kernel_size : tuple of int
        Spatial extent of the convolutional kernel (length 1 or 2).
    stride : tuple of int
        Stride in each spatial dimension.
    padding : tuple of int
        Zero-padding in each spatial dimension.
    input_shape : tuple of int
        Shape of a single input *excluding* the batch axis, e.g.
        ``(C, L)`` for 1-D or ``(C, H, W)`` for 2-D.
    max_rank : int or None
        Maximum rank for low-rank approximations.  ``None`` means full
        rank.
    """

    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    input_shape: Tuple[int, ...]
    max_rank: Optional[int] = None

    @property
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.kernel_size)

    @property
    def num_output_positions(self) -> Tuple[int, ...]:
        """Number of output positions in each spatial dimension."""
        spatial = self.input_shape[1:]  # skip channel axis
        return tuple(
            (spatial[d] + 2 * self.padding[d] - self.kernel_size[d])
            // self.stride[d]
            + 1
            for d in range(self.spatial_ndim)
        )

    @property
    def total_output_positions(self) -> int:
        """Total number of output spatial positions."""
        return int(np.prod(self.num_output_positions))

    @property
    def patch_dim(self) -> int:
        """Dimensionality of a single flattened patch."""
        return self.input_shape[0] * int(np.prod(self.kernel_size))


# ======================================================================
# Patch Gram Matrix
# ======================================================================


class PatchGramMatrix:
    """Compute and analyse the patch Gram (overlap) matrix.

    The patch Gram matrix encodes how convolutional receptive fields
    overlap in the input:

    .. math::
        K^{\\text{patch}}_{ij} = \\langle p_i, p_j \\rangle

    where :math:`p_i` is the flattened patch at output position *i*.

    Parameters
    ----------
    config : PatchGramConfig
        Patch extraction and analysis configuration.
    """

    def __init__(self, config: PatchGramConfig) -> None:
        self.config = config
        self._eigenvalues: Optional[NDArray] = None
        self._eigenvectors: Optional[NDArray] = None
        self._gram: Optional[NDArray] = None
        self._patches: Optional[NDArray] = None

    # ------------------------------------------------------------------
    # Patch extraction
    # ------------------------------------------------------------------

    def _extract_all_patches(self, x: NDArray) -> NDArray:
        """Extract all patches from a single input sample.

        Parameters
        ----------
        x : NDArray
            Single input of shape matching ``config.input_shape`` (no
            batch axis).

        Returns
        -------
        patches : NDArray of shape ``(num_patches, patch_dim)``
            Flattened patches.
        """
        cfg = self.config
        if cfg.spatial_ndim == 1:
            return self._extract_patches_1d(x)
        elif cfg.spatial_ndim == 2:
            return self._extract_patches_2d(x)
        else:
            raise NotImplementedError(
                f"Patch extraction for {cfg.spatial_ndim}-D not implemented"
            )

    def _extract_patches_1d(self, x: NDArray) -> NDArray:
        """Extract 1-D patches from a single input.

        Parameters
        ----------
        x : NDArray of shape ``(C, L)``

        Returns
        -------
        NDArray of shape ``(num_patches, C * kernel_size)``
        """
        cfg = self.config
        c = x.shape[0]
        k = cfg.kernel_size[0]
        s = cfg.stride[0]
        p = cfg.padding[0]

        if p > 0:
            x = np.pad(x, ((0, 0), (p, p)), mode="constant")

        length = x.shape[1]
        num_patches = (length - k) // s + 1
        patches = np.zeros((num_patches, c * k), dtype=x.dtype)

        for i in range(num_patches):
            start = i * s
            patches[i] = x[:, start : start + k].ravel()

        return patches

    def _extract_patches_2d(self, x: NDArray) -> NDArray:
        """Extract 2-D patches from a single input.

        Parameters
        ----------
        x : NDArray of shape ``(C, H, W)``

        Returns
        -------
        NDArray of shape ``(num_patches, C * kH * kW)``
        """
        cfg = self.config
        c = x.shape[0]
        kh, kw = cfg.kernel_size
        sh, sw = cfg.stride
        ph, pw = cfg.padding

        if ph > 0 or pw > 0:
            x = np.pad(x, ((0, 0), (ph, ph), (pw, pw)), mode="constant")

        h, w = x.shape[1], x.shape[2]
        out_h = (h - kh) // sh + 1
        out_w = (w - kw) // sw + 1
        num_patches = out_h * out_w
        patch_dim = c * kh * kw
        patches = np.zeros((num_patches, patch_dim), dtype=x.dtype)

        idx = 0
        for i in range(out_h):
            for j in range(out_w):
                r, col = i * sh, j * sw
                patches[idx] = x[:, r : r + kh, col : col + kw].ravel()
                idx += 1

        return patches

    # ------------------------------------------------------------------
    # Gram matrix computation
    # ------------------------------------------------------------------

    def compute(self, x: NDArray) -> NDArray:
        r"""Build the patch Gram matrix from input data.

        .. math::
            K^{\text{patch}}_{ij} = \langle p_i, p_j \rangle

        Parameters
        ----------
        x : NDArray
            Single input of shape ``config.input_shape`` (no batch axis).

        Returns
        -------
        gram : NDArray of shape ``(S, S)``
            Patch Gram matrix where *S* is the total number of output
            positions.
        """
        patches = self._extract_all_patches(x)
        self._patches = patches
        self._gram = patches @ patches.T
        self._eigenvalues = None  # invalidate cached decomposition
        self._eigenvectors = None
        return self._gram

    def compute_efficient(self, x: NDArray) -> NDArray:
        """Compute the patch Gram matrix exploiting Toeplitz structure.

        For uniform stride and no padding (or circular padding), the
        Gram matrix has Toeplitz structure.  In that case we compute the
        first row via correlation and construct the full matrix in
        :math:`O(S \\log S)` time using the FFT.

        Parameters
        ----------
        x : NDArray
            Single input of shape ``config.input_shape``.

        Returns
        -------
        gram : NDArray of shape ``(S, S)``
        """
        cfg = self.config
        patches = self._extract_all_patches(x)
        self._patches = patches
        S = patches.shape[0]

        # Check if Toeplitz structure holds (uniform stride, no padding)
        if all(p == 0 for p in cfg.padding) and cfg.spatial_ndim == 1:
            # Compute auto-correlation via FFT
            # First row: gram[0, j] = <p_0, p_j>
            first_row = patches[0] @ patches.T  # (S,)
            first_col = patches[:, :] @ patches[0]  # (S,)
            gram = toeplitz(first_col, first_row)
        else:
            # Fall back to direct computation
            gram = patches @ patches.T

        self._gram = gram
        self._eigenvalues = None
        self._eigenvectors = None
        return gram

    # ------------------------------------------------------------------
    # Spectral analysis
    # ------------------------------------------------------------------

    def eigendecompose(self) -> Tuple[NDArray, NDArray]:
        """Compute eigendecomposition of the patch Gram matrix.

        Returns
        -------
        eigenvalues : NDArray
            Eigenvalues in descending order.
        eigenvectors : NDArray
            Corresponding eigenvectors (columns).

        Raises
        ------
        RuntimeError
            If ``compute`` or ``compute_efficient`` has not been called.
        """
        if self._gram is None:
            raise RuntimeError(
                "Gram matrix not computed; call compute() first"
            )
        eigenvalues, eigenvectors = sp_linalg.eigh(self._gram)
        idx = np.argsort(eigenvalues)[::-1]
        self._eigenvalues = eigenvalues[idx]
        self._eigenvectors = eigenvectors[:, idx]
        return self._eigenvalues, self._eigenvectors

    def low_rank_approximation(self, rank: Optional[int] = None) -> NDArray:
        """Truncated eigendecomposition of the patch Gram matrix.

        Parameters
        ----------
        rank : int or None
            Number of leading eigencomponents to retain.  Defaults to
            ``config.max_rank``.

        Returns
        -------
        gram_lr : NDArray of shape ``(S, S)``
            Low-rank approximation  :math:`\\sum_{i=1}^r \\lambda_i v_i v_i^T`.

        Raises
        ------
        RuntimeError
            If ``compute`` has not been called.
        """
        if self._eigenvalues is None or self._eigenvectors is None:
            self.eigendecompose()

        r = rank or self.config.max_rank
        if r is None:
            r = len(self._eigenvalues)
        r = min(r, len(self._eigenvalues))

        evals = self._eigenvalues[:r]
        evecs = self._eigenvectors[:, :r]
        return (evecs * evals[np.newaxis, :]) @ evecs.T

    def effective_rank(self) -> float:
        """Participation ratio of the eigenvalue spectrum.

        .. math::
            r_{\\text{eff}} = \\frac{(\\sum_i \\lambda_i)^2}
                                    {\\sum_i \\lambda_i^2}

        Returns
        -------
        float
            Effective rank (1 ≤ r_eff ≤ S).
        """
        if self._eigenvalues is None:
            self.eigendecompose()

        evals = np.maximum(self._eigenvalues, 0.0)
        total = evals.sum()
        if total < 1e-30:
            return 0.0
        return float(total ** 2 / np.sum(evals ** 2))

    # ------------------------------------------------------------------
    # Overlap analysis
    # ------------------------------------------------------------------

    def overlap_analysis(self) -> Dict[str, Any]:
        """Analyse the patch overlap pattern.

        Returns
        -------
        dict
            ``'overlap_counts'`` : NDArray — how many patches share each
            input element.
            ``'mean_overlap'`` : float — average overlap count.
            ``'max_overlap'`` : int — maximum overlap count.
            ``'min_overlap'`` : int — minimum overlap count.
            ``'overlap_matrix'`` : NDArray — binary (S, D) indicator of
            which input elements belong to which patch.
        """
        cfg = self.config
        if cfg.spatial_ndim == 1:
            return self._overlap_analysis_1d()
        elif cfg.spatial_ndim == 2:
            return self._overlap_analysis_2d()
        else:
            raise NotImplementedError

    def _overlap_analysis_1d(self) -> Dict[str, Any]:
        """Overlap analysis for 1-D convolutions."""
        cfg = self.config
        c = cfg.input_shape[0]
        length = cfg.input_shape[1]
        k = cfg.kernel_size[0]
        s = cfg.stride[0]
        p = cfg.padding[0]

        padded_len = length + 2 * p
        num_patches = (padded_len - k) // s + 1
        input_dim = c * padded_len

        # Binary indicator: which elements belong to which patch
        indicator = np.zeros((num_patches, input_dim), dtype=np.int32)
        for i in range(num_patches):
            start = i * s
            for ch in range(c):
                for j in range(k):
                    indicator[i, ch * padded_len + start + j] = 1

        overlap_counts = indicator.sum(axis=0)

        return {
            "overlap_counts": overlap_counts,
            "mean_overlap": float(np.mean(overlap_counts)),
            "max_overlap": int(np.max(overlap_counts)),
            "min_overlap": int(np.min(overlap_counts)),
            "overlap_matrix": indicator,
        }

    def _overlap_analysis_2d(self) -> Dict[str, Any]:
        """Overlap analysis for 2-D convolutions."""
        cfg = self.config
        c = cfg.input_shape[0]
        h, w = cfg.input_shape[1], cfg.input_shape[2]
        kh, kw = cfg.kernel_size
        sh, sw = cfg.stride
        ph, pw = cfg.padding

        h_pad, w_pad = h + 2 * ph, w + 2 * pw
        out_h = (h_pad - kh) // sh + 1
        out_w = (w_pad - kw) // sw + 1
        num_patches = out_h * out_w
        input_dim = c * h_pad * w_pad

        indicator = np.zeros((num_patches, input_dim), dtype=np.int32)
        idx = 0
        for i in range(out_h):
            for j in range(out_w):
                r, col = i * sh, j * sw
                for ch in range(c):
                    for di in range(kh):
                        for dj in range(kw):
                            flat = ch * h_pad * w_pad + (r + di) * w_pad + (col + dj)
                            indicator[idx, flat] = 1
                idx += 1

        overlap_counts = indicator.sum(axis=0)

        return {
            "overlap_counts": overlap_counts,
            "mean_overlap": float(np.mean(overlap_counts)),
            "max_overlap": int(np.max(overlap_counts)),
            "min_overlap": int(np.min(overlap_counts)),
            "overlap_matrix": indicator,
        }

    # ------------------------------------------------------------------
    # Sparse and condition number
    # ------------------------------------------------------------------

    def to_sparse(self) -> sparse.csr_matrix:
        """Convert the patch Gram matrix to sparse CSR format.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse representation (useful when many patches do not
            overlap, yielding structural zeros).

        Raises
        ------
        RuntimeError
            If ``compute`` has not been called.
        """
        if self._gram is None:
            raise RuntimeError("Gram matrix not computed; call compute() first")
        return sparse.csr_matrix(self._gram)

    def condition_number(self) -> float:
        """Condition number of the patch Gram matrix.

        Returns
        -------
        float
            Ratio of largest to smallest eigenvalue
            :math:`\\kappa = \\lambda_{\\max} / \\lambda_{\\min}`.
        """
        if self._eigenvalues is None:
            self.eigendecompose()

        pos = self._eigenvalues[self._eigenvalues > 0]
        if len(pos) == 0:
            return float("inf")
        return float(pos[0] / pos[-1])

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize_overlap(self, ax: Optional[Any] = None) -> Any:
        """Plot the overlap pattern of patches on the input.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Axes to plot on.  If ``None`` a new figure is created.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the overlap plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualize_overlap")

        analysis = self.overlap_analysis()
        counts = analysis["overlap_counts"]

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 3))

        cfg = self.config
        if cfg.spatial_ndim == 1:
            ax.bar(range(len(counts)), counts)
            ax.set_xlabel("Input element index")
            ax.set_ylabel("Overlap count")
            ax.set_title("Patch overlap pattern (1-D)")
        else:
            c = cfg.input_shape[0]
            h_pad = cfg.input_shape[1] + 2 * cfg.padding[0]
            w_pad = cfg.input_shape[2] + 2 * cfg.padding[1]
            # Show overlap for first channel
            counts_2d = counts[: h_pad * w_pad].reshape(h_pad, w_pad)
            im = ax.imshow(counts_2d, cmap="viridis", aspect="auto")
            ax.set_title("Patch overlap pattern (2-D, channel 0)")
            ax.set_xlabel("Width")
            ax.set_ylabel("Height")
            plt.colorbar(im, ax=ax, label="Overlap count")

        return ax
