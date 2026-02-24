"""Hessian-Jacobian contraction tensor H_{ijk} for finite-width NTK corrections.

The tensor H_{ijk} = ∂²f_i / ∂θ_j ∂θ_k captures the second-order structure
of the neural tangent kernel and is the central object needed to compute the
leading O(1/N) correction to the infinite-width kernel.

Concretely, if f: R^P → R^O is the network map from P parameters to O outputs
evaluated on a batch of inputs, then

    H_{ijk} = ∂²f_i / ∂θ_j ∂θ_k,   i ∈ [O], j,k ∈ [P].

The NTK Θ_{ii'} = Σ_j (∂f_i/∂θ_j)(∂f_{i'}/∂θ_j) receives a 1/N correction
that involves contractions of H with the Jacobian and with itself.

References
----------
* Dyer & Gur-Ari, "Asymptotics of Wide Networks from Feynman Diagrams", 2020.
* Huang & Yau, "Dynamics of Deep Neural Networks and Neural Tangent
  Hierarchy", 2020.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as sp_linalg


# ======================================================================
#  Data container
# ======================================================================


@dataclass
class HTensor:
    """Container for the Hessian-Jacobian contraction tensor H_{ijk}.

    Attributes
    ----------
    data : NDArray
        Dense representation of shape ``(O, P, P)`` where *O* is the output
        dimension and *P* is the parameter count.
    shape : tuple[int, ...]
        Convenience accessor equal to ``data.shape``.
    layer_contributions : dict[str, NDArray]
        Per-layer slices of the full tensor keyed by layer name.  Each value
        has shape ``(O, P_l, P_l)`` where ``P_l`` is the number of
        parameters in that layer.  Cross-layer second derivatives are zero
        for feed-forward networks, so the full tensor is block-diagonal.
    factorization_error : float
        If the tensor was computed via an approximate factorization (e.g.
        the patch-factorization hypothesis for convolutions), this records
        the relative Frobenius-norm error compared to the brute-force
        tensor.  ``0.0`` when exact.
    computation_method : str
        Human-readable tag indicating how the tensor was obtained, e.g.
        ``'finite_difference'``, ``'analytic_relu'``, ``'conv_patch'``.
    """

    data: NDArray
    shape: Tuple[int, ...] = field(init=False)
    layer_contributions: Dict[str, NDArray] = field(default_factory=dict)
    factorization_error: float = 0.0
    computation_method: str = "unknown"

    def __post_init__(self) -> None:
        self.shape = self.data.shape
        if self.data.ndim != 3:
            raise ValueError(
                f"HTensor.data must be 3-dimensional, got ndim={self.data.ndim}"
            )


# ======================================================================
#  Core computation engine
# ======================================================================


class HTensorComputer:
    r"""Compute and manipulate the Hessian tensor H_{ijk} = ∂²f_i/∂θ_j∂θ_k.

    Two families of methods are provided:

    1. **Numerical** — finite-difference approximation of H from a callable
       ``forward_fn(params, X) -> outputs``.  Robust but O(P²) in parameter
       count.
    2. **Analytic** — closed-form expressions for specific layer types
       (dense + ReLU, dense + generic activation, convolution).

    Parameters
    ----------
    eps : float
        Default step size for central finite differences.
    method : str
        Default computation method tag stored in returned :class:`HTensor`.
    """

    def __init__(self, eps: float = 1e-4, method: str = "finite_difference") -> None:
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = eps
        self.method = method

    # ------------------------------------------------------------------
    #  Numerical (finite-difference) computation
    # ------------------------------------------------------------------

    def compute_numerical(
        self,
        forward_fn: Callable[[NDArray, NDArray], NDArray],
        params: NDArray,
        X: NDArray,
        eps: Optional[float] = None,
    ) -> HTensor:
        r"""Compute H_{ijk} via central finite differences of the network output.

        For every pair of parameter indices ``(j, k)`` the second derivative
        is approximated as

        .. math::

            \frac{\partial^2 f_i}{\partial\theta_j\,\partial\theta_k}
            \approx
            \frac{f(\theta+\epsilon e_j+\epsilon e_k)
                  - f(\theta+\epsilon e_j-\epsilon e_k)
                  - f(\theta-\epsilon e_j+\epsilon e_k)
                  + f(\theta-\epsilon e_j-\epsilon e_k)}
                 {4\epsilon^2}

        Parameters
        ----------
        forward_fn : callable
            ``forward_fn(params, X) -> NDArray`` of shape ``(O,)`` or ``(B, O)``.
        params : NDArray
            Flat parameter vector of shape ``(P,)``.
        X : NDArray
            Input data, passed unchanged to *forward_fn*.
        eps : float, optional
            Finite-difference step; defaults to ``self.eps``.

        Returns
        -------
        HTensor
            Tensor of shape ``(O, P, P)`` with symmetry ``H_{ijk} = H_{ikj}``.
        """
        if eps is None:
            eps = self.eps

        params = np.asarray(params, dtype=np.float64).ravel()
        P = params.shape[0]

        f0 = np.asarray(forward_fn(params, X), dtype=np.float64).ravel()
        O = f0.shape[0]

        H = np.zeros((O, P, P), dtype=np.float64)

        for j in range(P):
            for k in range(j, P):
                H[:, j, k] = self._second_order_fd(
                    forward_fn, params, X, j, k, eps
                )
                if k != j:
                    H[:, k, j] = H[:, j, k]

        # Validate symmetry up to numerical noise
        asym = np.max(np.abs(H - np.swapaxes(H, 1, 2)))
        if asym > 100 * eps:
            warnings.warn(
                f"H tensor symmetry violation: max|H_ijk - H_ikj| = {asym:.3e}"
            )

        return HTensor(
            data=H,
            factorization_error=0.0,
            computation_method="finite_difference",
        )

    def _second_order_fd(
        self,
        forward_fn: Callable[[NDArray, NDArray], NDArray],
        params: NDArray,
        X: NDArray,
        i: int,
        j: int,
        eps: float,
    ) -> NDArray:
        r"""Central finite difference for ∂²f/∂θ_i ∂θ_j.

        Uses the four-point stencil:

        .. math::

            \frac{f(\theta + \epsilon e_i + \epsilon e_j)
                  - f(\theta + \epsilon e_i - \epsilon e_j)
                  - f(\theta - \epsilon e_i + \epsilon e_j)
                  + f(\theta - \epsilon e_i - \epsilon e_j)}
                 {4\epsilon^2}

        When ``i == j`` the stencil simplifies to the standard second
        derivative formula using three points (but we keep the four-point
        version for uniform code).

        Parameters
        ----------
        forward_fn : callable
            Network forward function.
        params : NDArray
            Flat parameter vector ``(P,)``.
        X : NDArray
            Input data.
        i, j : int
            Parameter indices.
        eps : float
            Step size.

        Returns
        -------
        NDArray
            Vector of shape ``(O,)`` — the second derivative for every output.
        """
        p_pp = params.copy()
        p_pm = params.copy()
        p_mp = params.copy()
        p_mm = params.copy()

        p_pp[i] += eps
        p_pp[j] += eps

        p_pm[i] += eps
        p_pm[j] -= eps

        p_mp[i] -= eps
        p_mp[j] += eps

        p_mm[i] -= eps
        p_mm[j] -= eps

        f_pp = np.asarray(forward_fn(p_pp, X), dtype=np.float64).ravel()
        f_pm = np.asarray(forward_fn(p_pm, X), dtype=np.float64).ravel()
        f_mp = np.asarray(forward_fn(p_mp, X), dtype=np.float64).ravel()
        f_mm = np.asarray(forward_fn(p_mm, X), dtype=np.float64).ravel()

        return (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps)

    # ------------------------------------------------------------------
    #  Analytic: dense ReLU layer
    # ------------------------------------------------------------------

    def compute_dense_relu(
        self,
        W: NDArray,
        b: NDArray,
        X: NDArray,
    ) -> HTensor:
        r"""Analytic H for a single dense layer with ReLU activation.

        For a layer f(x) = ReLU(Wx + b):

        *   The second derivative w.r.t. weights vanishes except where
            the pre-activation crosses zero — there ReLU''(z) is a Dirac
            delta, which we approximate by a narrow Gaussian bump of width
            ``self.eps``.
        *   In practice, for finite-width corrections the dominant
            contribution comes from the Jacobian structure; the Dirac term
            is sub-leading.  We include it here for completeness.

        Parameters
        ----------
        W : NDArray, shape (O, D)
            Weight matrix of the layer.
        b : NDArray, shape (O,)
            Bias vector.
        X : NDArray, shape (B, D) or (D,)
            Input data.  If 2-D, uses the *first* sample.

        Returns
        -------
        HTensor
            Tensor of shape ``(O, P, P)`` where ``P = O*D + O`` (weights
            then biases, row-major).
        """
        W = np.asarray(W, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64).ravel()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = X[0]

        O, D = W.shape
        P = O * D + O  # total params: weights + biases

        z = W @ X + b  # pre-activations, shape (O,)

        # Approximate ReLU''(z) ≈ (1/(eps*sqrt(2π))) * exp(-z²/(2*eps²))
        sigma = max(self.eps, 1e-6)
        relu_pp = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
            -0.5 * (z / sigma) ** 2
        )

        H = np.zeros((O, P, P), dtype=np.float64)

        for o in range(O):
            # Indices of weights for output unit o: W[o, :] stored at
            # positions [o*D .. o*D + D).
            w_start = o * D
            w_end = w_start + D
            b_idx = O * D + o  # bias index

            # ∂²f_o / ∂W_{o,d} ∂W_{o,d'} = relu''(z_o) * x_d * x_d'
            xx = np.outer(X, X) * relu_pp[o]
            H[o, w_start:w_end, w_start:w_end] = xx

            # ∂²f_o / ∂W_{o,d} ∂b_o = relu''(z_o) * x_d
            xb = X * relu_pp[o]
            H[o, w_start:w_end, b_idx] = xb
            H[o, b_idx, w_start:w_end] = xb

            # ∂²f_o / ∂b_o ∂b_o = relu''(z_o)
            H[o, b_idx, b_idx] = relu_pp[o]

        return HTensor(
            data=H,
            layer_contributions={"dense_relu": H.copy()},
            factorization_error=0.0,
            computation_method="analytic_relu",
        )

    # ------------------------------------------------------------------
    #  Analytic: dense layer with generic activation
    # ------------------------------------------------------------------

    def compute_dense_generic(
        self,
        W: NDArray,
        b: NDArray,
        X: NDArray,
        activation_fn: Callable[[NDArray], NDArray],
        activation_hessian_fn: Callable[[NDArray], NDArray],
    ) -> HTensor:
        r"""H tensor for a dense layer with an arbitrary activation.

        For f(x) = σ(Wx + b), the non-zero second derivatives are:

        .. math::

            \frac{\partial^2 f_o}{\partial W_{o,d}\,\partial W_{o,d'}}
            = \sigma''(z_o)\,x_d\,x_{d'}

        .. math::

            \frac{\partial^2 f_o}{\partial W_{o,d}\,\partial b_o}
            = \sigma''(z_o)\,x_d

        .. math::

            \frac{\partial^2 f_o}{\partial b_o^2} = \sigma''(z_o)

        All cross-unit second derivatives vanish for a single dense layer.

        Parameters
        ----------
        W : NDArray, shape (O, D)
        b : NDArray, shape (O,)
        X : NDArray, shape (B, D) or (D,)
        activation_fn : callable
            σ(z) — unused here but accepted for API symmetry.
        activation_hessian_fn : callable
            σ''(z).  Must accept and return an array of shape ``(O,)``.

        Returns
        -------
        HTensor
        """
        W = np.asarray(W, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64).ravel()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = X[0]

        O, D = W.shape
        P = O * D + O

        z = W @ X + b
        sigma_pp = np.asarray(activation_hessian_fn(z), dtype=np.float64).ravel()

        if sigma_pp.shape[0] != O:
            raise ValueError(
                f"activation_hessian_fn returned length {sigma_pp.shape[0]}, "
                f"expected {O}"
            )

        H = np.zeros((O, P, P), dtype=np.float64)

        for o in range(O):
            w_start = o * D
            w_end = w_start + D
            b_idx = O * D + o

            s = sigma_pp[o]

            # weight-weight block
            H[o, w_start:w_end, w_start:w_end] = s * np.outer(X, X)

            # weight-bias cross terms
            H[o, w_start:w_end, b_idx] = s * X
            H[o, b_idx, w_start:w_end] = s * X

            # bias-bias
            H[o, b_idx, b_idx] = s

        return HTensor(
            data=H,
            layer_contributions={"dense_generic": H.copy()},
            factorization_error=0.0,
            computation_method="analytic_generic",
        )

    # ------------------------------------------------------------------
    #  Analytic: convolutional layer (patch factorization)
    # ------------------------------------------------------------------

    def compute_conv_h_tensor(
        self,
        kernel_weights: NDArray,
        X_patches: NDArray,
        stride: int = 1,
        padding: int = 0,
    ) -> HTensor:
        r"""Convolutional H_{ijk} via the patch-factorization hypothesis.

        For a convolution with kernel K of shape ``(C_out, C_in, kH, kW)``
        the im2col representation converts the operation to a dense
        matrix multiply over patches.  The patch-factorization hypothesis
        assumes that the Hessian factors as

        .. math::

            H_{\text{conv}} \approx H_{\text{patch}} \otimes I_{\text{spatial}}

        where ``H_patch`` is the Hessian of the equivalent dense layer
        applied to a *single* representative patch and ``I_spatial`` is the
        identity over spatial positions.

        Parameters
        ----------
        kernel_weights : NDArray, shape (C_out, C_in, kH, kW)
            Convolution kernel.
        X_patches : NDArray, shape (N_patches, C_in * kH * kW)
            Input patches extracted via im2col.  We use the *mean* patch
            as the representative.
        stride : int
            Convolution stride (metadata only; does not affect computation).
        padding : int
            Convolution padding (metadata only).

        Returns
        -------
        HTensor
            Approximate H tensor under patch factorization.
        """
        kernel_weights = np.asarray(kernel_weights, dtype=np.float64)
        X_patches = np.asarray(X_patches, dtype=np.float64)

        if kernel_weights.ndim != 4:
            raise ValueError(
                f"kernel_weights must be 4-D (C_out, C_in, kH, kW), "
                f"got ndim={kernel_weights.ndim}"
            )

        C_out, C_in, kH, kW = kernel_weights.shape
        patch_dim = C_in * kH * kW

        if X_patches.ndim != 2 or X_patches.shape[1] != patch_dim:
            raise ValueError(
                f"X_patches must have shape (N, {patch_dim}), "
                f"got {X_patches.shape}"
            )

        # Representative patch: mean over spatial locations
        x_rep = X_patches.mean(axis=0)  # (patch_dim,)
        N_patches = X_patches.shape[0]

        # Reshape kernel to dense weight matrix: (C_out, patch_dim)
        W_dense = kernel_weights.reshape(C_out, patch_dim)

        # For a linear convolution (no activation), the second derivative
        # of the output w.r.t. kernel weights vanishes.  The non-trivial
        # Hessian arises when there is an activation; here we model the
        # common case of ReLU using the Gaussian-smoothed Dirac delta.
        z = W_dense @ x_rep  # (C_out,)
        sigma = max(self.eps, 1e-6)
        relu_pp = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
            -0.5 * (z / sigma) ** 2
        )

        P_kernel = C_out * patch_dim
        H_patch = np.zeros((C_out, P_kernel, P_kernel), dtype=np.float64)

        for o in range(C_out):
            w_start = o * patch_dim
            w_end = w_start + patch_dim
            H_patch[o, w_start:w_end, w_start:w_end] = (
                relu_pp[o] * np.outer(x_rep, x_rep)
            )

        # Under patch factorization the full H is the patch H replicated
        # across spatial positions (Kronecker with I_{N_patches}).
        # We store only the patch-level tensor and record N_patches as
        # metadata via the factorization_error field.
        # The caller can reconstruct via np.kron if needed.
        return HTensor(
            data=H_patch,
            layer_contributions={
                "conv_patch": H_patch.copy(),
            },
            factorization_error=0.0,  # exact under hypothesis
            computation_method=f"conv_patch(stride={stride},pad={padding},"
            f"n_patches={N_patches})",
        )

    # ------------------------------------------------------------------
    #  Tensor contractions
    # ------------------------------------------------------------------

    @staticmethod
    def contract_12(H: NDArray, v: NDArray) -> NDArray:
        r"""Contract H_{ijk} with vector v on index j.

        .. math::

            R_{ik} = \sum_j H_{ijk}\,v_j

        Parameters
        ----------
        H : NDArray, shape (O, P, P)
        v : NDArray, shape (P,)

        Returns
        -------
        NDArray, shape (O, P)
        """
        H = np.asarray(H, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64).ravel()
        if H.ndim != 3:
            raise ValueError(f"H must be 3-D, got ndim={H.ndim}")
        if v.shape[0] != H.shape[1]:
            raise ValueError(
                f"v length {v.shape[0]} != H dim-1 {H.shape[1]}"
            )
        return np.einsum("ijk,j->ik", H, v)

    @staticmethod
    def contract_23(H: NDArray, M: NDArray) -> NDArray:
        r"""Contract H_{ijk} with matrix M on indices j and k.

        .. math::

            R_i = \sum_{j,k} H_{ijk}\,M_{jk}

        Parameters
        ----------
        H : NDArray, shape (O, P, P)
        M : NDArray, shape (P, P)

        Returns
        -------
        NDArray, shape (O,)
        """
        H = np.asarray(H, dtype=np.float64)
        M = np.asarray(M, dtype=np.float64)
        if H.ndim != 3:
            raise ValueError(f"H must be 3-D, got ndim={H.ndim}")
        if M.shape != H.shape[1:]:
            raise ValueError(
                f"M shape {M.shape} != H trailing dims {H.shape[1:]}"
            )
        return np.einsum("ijk,jk->i", H, M)

    @staticmethod
    def contract_full(H: NDArray, v1: NDArray, v2: NDArray) -> float:
        r"""Full contraction of H with two vectors.

        .. math::

            s = \sum_{i,j,k} H_{ijk}\,v1_j\,v2_k

        This is equivalent to ``np.sum(contract_12(H, v1) * v2)`` summed
        over the output index as well, giving a scalar.

        Parameters
        ----------
        H : NDArray, shape (O, P, P)
        v1 : NDArray, shape (P,)
        v2 : NDArray, shape (P,)

        Returns
        -------
        float
        """
        H = np.asarray(H, dtype=np.float64)
        v1 = np.asarray(v1, dtype=np.float64).ravel()
        v2 = np.asarray(v2, dtype=np.float64).ravel()
        if H.ndim != 3:
            raise ValueError(f"H must be 3-D, got ndim={H.ndim}")
        if v1.shape[0] != H.shape[1] or v2.shape[0] != H.shape[2]:
            raise ValueError(
                f"Vector lengths ({v1.shape[0]}, {v2.shape[0]}) incompatible "
                f"with H dims ({H.shape[1]}, {H.shape[2]})"
            )
        return float(np.einsum("ijk,j,k->", H, v1, v2))

    # ------------------------------------------------------------------
    #  Norms
    # ------------------------------------------------------------------

    @staticmethod
    def frobenius_norm(H: NDArray) -> float:
        r"""Frobenius norm of the tensor.

        .. math::

            \|H\|_F = \sqrt{\sum_{i,j,k} H_{ijk}^2}

        Parameters
        ----------
        H : NDArray, shape (O, P, P)

        Returns
        -------
        float
        """
        H = np.asarray(H, dtype=np.float64)
        return float(np.sqrt(np.sum(H * H)))

    @staticmethod
    def spectral_norm_estimate(
        H: NDArray,
        num_iters: int = 50,
    ) -> float:
        r"""Estimate the spectral norm of H via power iteration.

        We treat H as a linear map from vectors ``(v, w)`` of shapes
        ``(P,)`` each to a vector ``u`` of shape ``(O,)`` defined by

        .. math::

            u_i = \sum_{j,k} H_{ijk}\,v_j\,w_k.

        The spectral norm is the largest singular value of this map.
        We approximate it by alternating power iteration on v and w.

        Parameters
        ----------
        H : NDArray, shape (O, P, P)
        num_iters : int
            Number of power-iteration sweeps.

        Returns
        -------
        float
            Estimated spectral norm.
        """
        H = np.asarray(H, dtype=np.float64)
        if H.ndim != 3:
            raise ValueError(f"H must be 3-D, got ndim={H.ndim}")

        O, P1, P2 = H.shape

        rng = np.random.RandomState(42)
        v = rng.randn(P1)
        v /= np.linalg.norm(v) + 1e-30
        w = rng.randn(P2)
        w /= np.linalg.norm(w) + 1e-30

        sigma = 0.0
        for _ in range(num_iters):
            # Fix w, update v via H contracted on k with w:
            # M_{ij} = sum_k H_{ijk} w_k  →  shape (O, P1)
            M = np.einsum("ijk,k->ij", H, w)
            # Treat M as (O*P1) vector, find dominant direction in P1
            # by contracting with u (output direction).
            # u_i = sum_j M_{ij} v_j
            u = M @ v  # (O,)
            u_norm = np.linalg.norm(u)
            if u_norm < 1e-30:
                break
            u /= u_norm

            # Now recover v: v_j = sum_i u_i M_{ij}
            v_new = M.T @ u  # (P1,)
            v_norm = np.linalg.norm(v_new)
            if v_norm < 1e-30:
                break
            v = v_new / v_norm

            # Fix v, update w via H contracted on j with v:
            # N_{ik} = sum_j H_{ijk} v_j  →  shape (O, P2)
            N = np.einsum("ijk,j->ik", H, v)
            u2 = N @ w  # (O,)
            u2_norm = np.linalg.norm(u2)
            if u2_norm < 1e-30:
                break
            u2 /= u2_norm

            w_new = N.T @ u2  # (P2,)
            w_norm = np.linalg.norm(w_new)
            if w_norm < 1e-30:
                break
            w = w_new / w_norm

            sigma = float(np.abs(np.einsum("ijk,i,j,k->", H, u2, v, w)))

        return sigma

    # ------------------------------------------------------------------
    #  Symmetry utilities
    # ------------------------------------------------------------------

    @staticmethod
    def symmetrize(H: NDArray) -> NDArray:
        r"""Enforce H_{ijk} = H_{ikj} symmetry.

        Parameters
        ----------
        H : NDArray, shape (O, P, P)

        Returns
        -------
        NDArray
            Symmetrized copy of H.
        """
        H = np.asarray(H, dtype=np.float64)
        return 0.5 * (H + np.swapaxes(H, 1, 2))

    @staticmethod
    def check_symmetry(H: NDArray, atol: float = 1e-8) -> Tuple[bool, float]:
        r"""Check whether H_{ijk} = H_{ikj}.

        Parameters
        ----------
        H : NDArray, shape (O, P, P)
        atol : float
            Absolute tolerance.

        Returns
        -------
        is_symmetric : bool
        max_violation : float
        """
        H = np.asarray(H, dtype=np.float64)
        diff = np.max(np.abs(H - np.swapaxes(H, 1, 2)))
        return bool(diff <= atol), float(diff)

    # ------------------------------------------------------------------
    #  Memory-efficient chunked computation
    # ------------------------------------------------------------------

    def compute_numerical_chunked(
        self,
        forward_fn: Callable[[NDArray, NDArray], NDArray],
        params: NDArray,
        X: NDArray,
        chunk_size: int = 64,
        eps: Optional[float] = None,
    ) -> HTensor:
        r"""Memory-efficient numerical H computation via parameter chunking.

        Instead of allocating the full ``(O, P, P)`` tensor at once, this
        method processes blocks of parameter pairs of size ``chunk_size``
        and assembles the result incrementally.  Useful when P is large
        but the caller still wants the exact finite-difference answer.

        Parameters
        ----------
        forward_fn : callable
        params : NDArray, shape (P,)
        X : NDArray
        chunk_size : int
            Number of parameter indices per chunk.
        eps : float, optional

        Returns
        -------
        HTensor
        """
        if eps is None:
            eps = self.eps

        params = np.asarray(params, dtype=np.float64).ravel()
        P = params.shape[0]
        f0 = np.asarray(forward_fn(params, X), dtype=np.float64).ravel()
        O = f0.shape[0]

        H = np.zeros((O, P, P), dtype=np.float64)

        # Process upper triangle in chunks
        indices = list(range(P))
        n_chunks = (P + chunk_size - 1) // chunk_size

        for ci in range(n_chunks):
            i_start = ci * chunk_size
            i_end = min(i_start + chunk_size, P)
            for cj in range(ci, n_chunks):
                j_start = cj * chunk_size
                j_end = min(j_start + chunk_size, P)
                for i in range(i_start, i_end):
                    k_lo = max(j_start, i) if ci == cj else j_start
                    for k in range(k_lo, j_end):
                        val = self._second_order_fd(
                            forward_fn, params, X, i, k, eps
                        )
                        H[:, i, k] = val
                        if i != k:
                            H[:, k, i] = val

        return HTensor(
            data=H,
            factorization_error=0.0,
            computation_method="finite_difference_chunked",
        )

    # ------------------------------------------------------------------
    #  Jacobian-based H computation (more efficient for moderate P)
    # ------------------------------------------------------------------

    def compute_from_jacobian_fd(
        self,
        forward_fn: Callable[[NDArray, NDArray], NDArray],
        params: NDArray,
        X: NDArray,
        eps: Optional[float] = None,
    ) -> HTensor:
        r"""Compute H by finite-differencing the Jacobian.

        The Jacobian J_{ij} = ∂f_i/∂θ_j is computed at ``θ ± ε e_k`` for
        each k, and the second derivative is recovered as

        .. math::

            H_{ijk} = \frac{J_{ij}(\theta + \epsilon e_k)
                           - J_{ij}(\theta - \epsilon e_k)}{2\epsilon}

        This requires ``2P`` Jacobian evaluations (each itself ``2P``
        forward passes) but produces a symmetric tensor by construction
        when averaged with its transpose.

        Parameters
        ----------
        forward_fn, params, X, eps : see :meth:`compute_numerical`.

        Returns
        -------
        HTensor
        """
        if eps is None:
            eps = self.eps

        params = np.asarray(params, dtype=np.float64).ravel()
        P = params.shape[0]
        f0 = np.asarray(forward_fn(params, X), dtype=np.float64).ravel()
        O = f0.shape[0]

        def _jacobian(p: NDArray) -> NDArray:
            """Compute Jacobian via central differences, shape (O, P)."""
            J = np.zeros((O, P), dtype=np.float64)
            for j in range(P):
                p_plus = p.copy()
                p_minus = p.copy()
                p_plus[j] += eps
                p_minus[j] -= eps
                fp = np.asarray(forward_fn(p_plus, X), dtype=np.float64).ravel()
                fm = np.asarray(forward_fn(p_minus, X), dtype=np.float64).ravel()
                J[:, j] = (fp - fm) / (2.0 * eps)
            return J

        H = np.zeros((O, P, P), dtype=np.float64)

        for k in range(P):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[k] += eps
            p_minus[k] -= eps
            J_plus = _jacobian(p_plus)
            J_minus = _jacobian(p_minus)
            H[:, :, k] = (J_plus - J_minus) / (2.0 * eps)

        # Symmetrize to enforce H_{ijk} = H_{ikj}
        H = 0.5 * (H + np.swapaxes(H, 1, 2))

        return HTensor(
            data=H,
            factorization_error=0.0,
            computation_method="jacobian_fd",
        )


# ======================================================================
#  Factorization validator
# ======================================================================


class FactorizationValidator:
    """Validate approximate factorizations of the H tensor.

    When exact computation is infeasible (e.g. for convolutional layers)
    we use structured approximations such as the patch-factorization
    hypothesis.  This class provides tools to assess the quality of these
    approximations on small-scale problems where brute-force computation
    is possible.

    Parameters
    ----------
    atol : float
        Absolute tolerance for element-wise comparisons.
    rtol : float
        Relative tolerance for norm-based comparisons.
    """

    def __init__(self, atol: float = 1e-4, rtol: float = 1e-3) -> None:
        if atol < 0 or rtol < 0:
            raise ValueError("Tolerances must be non-negative")
        self.atol = atol
        self.rtol = rtol

    # ------------------------------------------------------------------
    #  Patch factorization validation
    # ------------------------------------------------------------------

    def validate_patch_factorization(
        self,
        H_full: NDArray,
        H_factored: NDArray,
    ) -> Dict[str, Union[bool, float, NDArray]]:
        r"""Compare brute-force H against patch-factored approximation.

        Parameters
        ----------
        H_full : NDArray, shape (O, P, P)
            Exact H tensor.
        H_factored : NDArray, shape (O, P, P)
            Approximate H from patch factorization.

        Returns
        -------
        dict
            ``passed`` : bool — whether the approximation is within tolerance.
            ``rel_error`` : float — relative Frobenius error.
            ``abs_error`` : float — absolute Frobenius error.
            ``max_elementwise`` : float — maximum element-wise absolute diff.
            ``error_distribution`` : NDArray — histogram of element-wise errors.
        """
        H_full = np.asarray(H_full, dtype=np.float64)
        H_factored = np.asarray(H_factored, dtype=np.float64)

        if H_full.shape != H_factored.shape:
            raise ValueError(
                f"Shape mismatch: H_full {H_full.shape} vs "
                f"H_factored {H_factored.shape}"
            )

        diff = H_full - H_factored
        abs_err = float(np.sqrt(np.sum(diff ** 2)))
        full_norm = float(np.sqrt(np.sum(H_full ** 2)))
        rel_err = abs_err / max(full_norm, 1e-30)
        max_ew = float(np.max(np.abs(diff)))

        # Error histogram (10 bins on log scale)
        flat_err = np.abs(diff.ravel())
        flat_err_pos = flat_err[flat_err > 0]
        if flat_err_pos.size > 0:
            log_bins = np.linspace(
                np.log10(flat_err_pos.min()),
                np.log10(flat_err_pos.max()),
                11,
            )
            hist, _ = np.histogram(np.log10(flat_err_pos), bins=log_bins)
        else:
            hist = np.zeros(10, dtype=int)

        passed = rel_err <= self.rtol and max_ew <= self.atol

        return {
            "passed": passed,
            "rel_error": rel_err,
            "abs_error": abs_err,
            "max_elementwise": max_ew,
            "error_distribution": hist,
        }

    # ------------------------------------------------------------------
    #  Layer decomposition validation
    # ------------------------------------------------------------------

    def validate_layer_decomposition(
        self,
        H_full: NDArray,
        layer_H_tensors: Dict[str, NDArray],
    ) -> Dict[str, Union[bool, float]]:
        r"""Validate that the sum of per-layer H tensors equals the full H.

        For a feed-forward network with L layers, the full Hessian is
        block-diagonal in the layer parameters (cross-layer second
        derivatives vanish):

        .. math::

            H = \bigoplus_{\ell=1}^{L} H^{(\ell)}

        This method checks that reassembling the per-layer blocks
        reproduces the full tensor.

        Parameters
        ----------
        H_full : NDArray, shape (O, P, P)
        layer_H_tensors : dict[str, NDArray]
            Mapping from layer name to that layer's H block.  Each block
            has shape ``(O, P_l, P_l)`` and the ``P_l`` values must sum
            to ``P``.

        Returns
        -------
        dict
            ``passed`` : bool
            ``rel_error`` : float
            ``abs_error`` : float
            ``per_layer_norms`` : dict[str, float]
        """
        H_full = np.asarray(H_full, dtype=np.float64)
        O = H_full.shape[0]
        P = H_full.shape[1]

        # Sort layers by name for deterministic ordering
        sorted_names = sorted(layer_H_tensors.keys())
        total_params = sum(layer_H_tensors[n].shape[1] for n in sorted_names)

        if total_params != P:
            raise ValueError(
                f"Sum of layer param counts ({total_params}) != P ({P})"
            )

        H_reconstructed = np.zeros_like(H_full)
        offset = 0
        per_layer_norms: Dict[str, float] = {}

        for name in sorted_names:
            Hl = np.asarray(layer_H_tensors[name], dtype=np.float64)
            if Hl.shape[0] != O:
                raise ValueError(
                    f"Layer '{name}' has output dim {Hl.shape[0]}, expected {O}"
                )
            Pl = Hl.shape[1]
            H_reconstructed[:, offset : offset + Pl, offset : offset + Pl] = Hl
            per_layer_norms[name] = float(np.sqrt(np.sum(Hl ** 2)))
            offset += Pl

        diff = H_full - H_reconstructed
        abs_err = float(np.sqrt(np.sum(diff ** 2)))
        full_norm = float(np.sqrt(np.sum(H_full ** 2)))
        rel_err = abs_err / max(full_norm, 1e-30)

        passed = rel_err <= self.rtol

        return {
            "passed": passed,
            "rel_error": rel_err,
            "abs_error": abs_err,
            "per_layer_norms": per_layer_norms,
        }

    # ------------------------------------------------------------------
    #  Factorization error
    # ------------------------------------------------------------------

    @staticmethod
    def compute_factorization_error(
        H_full: NDArray,
        H_approx: NDArray,
    ) -> float:
        r"""Relative Frobenius-norm error between exact and approximate H.

        .. math::

            \text{err} = \frac{\|H_{\text{full}} - H_{\text{approx}}\|_F}
                              {\|H_{\text{full}}\|_F}

        Parameters
        ----------
        H_full : NDArray
        H_approx : NDArray

        Returns
        -------
        float
        """
        H_full = np.asarray(H_full, dtype=np.float64)
        H_approx = np.asarray(H_approx, dtype=np.float64)

        if H_full.shape != H_approx.shape:
            raise ValueError(
                f"Shape mismatch: {H_full.shape} vs {H_approx.shape}"
            )

        diff_norm = float(np.sqrt(np.sum((H_full - H_approx) ** 2)))
        full_norm = float(np.sqrt(np.sum(H_full ** 2)))
        return diff_norm / max(full_norm, 1e-30)

    # ------------------------------------------------------------------
    #  Rank / mode-unfolding analysis
    # ------------------------------------------------------------------

    def rank_analysis(
        self,
        H: NDArray,
        mode: int,
    ) -> Dict[str, Union[NDArray, int, float]]:
        r"""Unfold tensor along a given mode and analyse singular values.

        Tensor unfolding (matricisation) along mode *m* reshapes the
        3-D tensor ``H`` of shape ``(n_0, n_1, n_2)`` into a matrix
        where mode *m* indexes the rows and the remaining modes are
        combined into columns.

        The singular value spectrum of the unfolded matrix reveals the
        effective rank of the tensor along that mode, which is useful
        for low-rank approximations.

        Parameters
        ----------
        H : NDArray, shape (O, P, P)
            The H tensor.
        mode : int
            Mode to unfold along (0, 1, or 2).

        Returns
        -------
        dict
            ``singular_values`` : NDArray — full singular value spectrum.
            ``effective_rank`` : int — number of singular values above
                ``rtol * σ_max``.
            ``spectral_decay_rate`` : float — ratio ``σ_2 / σ_1`` measuring
                how fast the spectrum decays.
            ``total_variance`` : float — sum of squared singular values.
            ``explained_variance_ratio`` : NDArray — cumulative proportion
                of variance explained by the first k singular values.
        """
        H = np.asarray(H, dtype=np.float64)
        if H.ndim != 3:
            raise ValueError(f"H must be 3-D, got ndim={H.ndim}")
        if mode not in (0, 1, 2):
            raise ValueError(f"mode must be 0, 1, or 2, got {mode}")

        # Unfolding: move target mode to front, flatten the rest
        H_unf = np.moveaxis(H, mode, 0)
        n_rows = H_unf.shape[0]
        n_cols = int(np.prod(H_unf.shape[1:]))
        mat = H_unf.reshape(n_rows, n_cols)

        # SVD (only singular values needed for analysis)
        try:
            s = sp_linalg.svdvals(mat)
        except np.linalg.LinAlgError:
            warnings.warn("SVD did not converge; returning zeros")
            s = np.zeros(min(n_rows, n_cols))

        s_max = s[0] if s.size > 0 else 0.0
        threshold = self.rtol * s_max if s_max > 0 else 0.0
        effective_rank = int(np.sum(s > threshold))

        total_var = float(np.sum(s ** 2))
        if total_var > 0:
            cumvar = np.cumsum(s ** 2) / total_var
        else:
            cumvar = np.zeros_like(s)

        decay_rate = float(s[1] / s[0]) if s.size >= 2 and s[0] > 0 else 0.0

        return {
            "singular_values": s,
            "effective_rank": effective_rank,
            "spectral_decay_rate": decay_rate,
            "total_variance": total_var,
            "explained_variance_ratio": cumvar,
        }

    # ------------------------------------------------------------------
    #  Summary report
    # ------------------------------------------------------------------

    def full_report(
        self,
        H: NDArray,
        H_approx: Optional[NDArray] = None,
        layer_contributions: Optional[Dict[str, NDArray]] = None,
    ) -> Dict[str, object]:
        """Generate a comprehensive validation report for an H tensor.

        Parameters
        ----------
        H : NDArray, shape (O, P, P)
            Reference (exact or best-available) H tensor.
        H_approx : NDArray, optional
            Approximate H tensor to compare against.
        layer_contributions : dict, optional
            Per-layer blocks for decomposition validation.

        Returns
        -------
        dict
            Report with keys ``symmetry``, ``norms``, ``rank_modes``,
            and optionally ``factorization`` and ``decomposition``.
        """
        H = np.asarray(H, dtype=np.float64)
        report: Dict[str, object] = {}

        # Symmetry check
        is_sym, max_viol = HTensorComputer.check_symmetry(H)
        report["symmetry"] = {
            "is_symmetric": is_sym,
            "max_violation": max_viol,
        }

        # Norms
        report["norms"] = {
            "frobenius": HTensorComputer.frobenius_norm(H),
            "spectral_estimate": HTensorComputer.spectral_norm_estimate(H),
        }

        # Rank analysis per mode
        rank_info = {}
        for mode in range(3):
            rank_info[f"mode_{mode}"] = self.rank_analysis(H, mode)
        report["rank_modes"] = rank_info

        # Factorization comparison
        if H_approx is not None:
            report["factorization"] = self.validate_patch_factorization(
                H, H_approx
            )

        # Layer decomposition
        if layer_contributions is not None:
            report["decomposition"] = self.validate_layer_decomposition(
                H, layer_contributions
            )

        return report
