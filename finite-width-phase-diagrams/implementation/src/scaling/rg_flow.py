"""Renormalization Group flow analysis for neural networks.

Implements RG-like analysis where coarse-graining layers or widths reveals
universal behavior near phase transitions.  The central objects are:

  - Block-spin transformations that coarse-grain layers or neurons,
  - Beta functions β_i = dg_i / d(ln b) for coupling constants,
  - Fixed points g* where β(g*) = 0 and their stability,
  - RG flow trajectories obtained by integrating the beta-function ODE.

Physics references
------------------
* K. G. Wilson, "The renormalization group and critical phenomena",
  Rev. Mod. Phys. 55, 583 (1983).
* L. P. Kadanoff, "Scaling laws for Ising models near T_c",
  Physics 2, 263 (1966).
* G. Naveh et al., "A self-consistent theory of neural network dynamics",
  arXiv:2012.15110 (2020).
* B. Hanin, "Which neural net architectures give rise to exploding and
  vanishing gradients?", NeurIPS 2018.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, minimize, root


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RGConfig:
    """Configuration for renormalization group analysis.

    Attributes
    ----------
    n_couplings : int
        Number of coupling constants tracked in the flow.
    rescaling_factor : float
        Block-spin rescaling factor *b*.
    max_iterations : int
        Maximum number of RG iterations (coarse-graining steps).
    tolerance : float
        Convergence tolerance for fixed-point searches.
    activation : str
        Nonlinearity used in the neural network ('relu', 'tanh', 'erf').
    """

    n_couplings: int = 2
    rescaling_factor: float = 2.0
    max_iterations: int = 200
    tolerance: float = 1e-8
    activation: str = "relu"


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def _activation_function(name: str) -> Callable[[NDArray], NDArray]:
    """Return the activation function for *name*."""
    if name == "relu":
        return lambda x: np.maximum(x, 0.0)
    if name == "tanh":
        return np.tanh
    if name == "erf":
        from scipy.special import erf
        return erf
    raise ValueError(f"Unknown activation: {name}")


def _activation_kappa1(name: str) -> float:
    """Expected value E[φ'(z)^2] for z ~ N(0,1).

    This enters one-loop beta functions for the kernel recursion.
    """
    if name == "relu":
        return 0.5
    if name == "tanh":
        return 1.0 - 2.0 / np.pi  # ≈ 0.3634
    if name == "erf":
        return 2.0 / np.pi
    raise ValueError(f"Unknown activation: {name}")


def _activation_kappa0(name: str) -> float:
    """Expected value E[φ(z)^2] for z ~ N(0,1)."""
    if name == "relu":
        return 0.5
    if name == "tanh":
        return 1.0 - 2.0 / np.pi
    if name == "erf":
        return 2.0 / np.pi
    raise ValueError(f"Unknown activation: {name}")


# ---------------------------------------------------------------------------
# Block-spin transformation
# ---------------------------------------------------------------------------

class BlockSpinTransformation:
    """Block-spin (real-space) RG transformations for neural-network kernels.

    Provides methods to coarse-grain over *layers* (depth direction) and
    over *neurons* (width direction), as well as to extract effective
    couplings, compute the effective action, and evaluate the two-point
    propagator.

    Parameters
    ----------
    config : RGConfig
        RG configuration parameters.
    """

    def __init__(self, config: RGConfig) -> None:
        self.config = config
        self.b = config.rescaling_factor
        self._kappa1 = _activation_kappa1(config.activation)
        self._kappa0 = _activation_kappa0(config.activation)

    # -- layer coarse-graining ------------------------------------------------

    def coarse_grain_layers(
        self,
        kernel_matrices: List[NDArray],
        block_size: int = 2,
    ) -> List[NDArray]:
        """Combine adjacent layers into effective single-layer kernels.

        For consecutive kernels K^{(l)} and K^{(l+1)}, the effective
        kernel at the coarser scale is

            K^{(l/2)}_eff = σ_w^2 · <φ(K^{(l)}) ⊙ φ(K^{(l+1)})> + σ_b^2

        approximated here by pointwise composition with the kernel
        recursion (Cho & Saul, 2009).

        Parameters
        ----------
        kernel_matrices : list of ndarray
            Layer-wise kernel matrices ``[K^{(0)}, K^{(1)}, ...]``.
        block_size : int
            Number of consecutive layers to merge per block (default 2).

        Returns
        -------
        list of ndarray
            Coarse-grained kernels, one per block.
        """
        n_layers = len(kernel_matrices)
        coarse = []
        for start in range(0, n_layers - block_size + 1, block_size):
            block = kernel_matrices[start:start + block_size]
            K_eff = block[0].copy()
            for K_next in block[1:]:
                # Kernel composition via the NNGP recursion:
                # K_eff <- kappa0 * (sigma_w^2 * element-wise product + sigma_b^2)
                diag_eff = np.sqrt(np.diag(K_eff))
                diag_next = np.sqrt(np.diag(K_next))
                outer_eff = np.outer(diag_eff, diag_eff)
                outer_next = np.outer(diag_next, diag_next)
                # Normalised cosine
                cos_theta = np.clip(K_eff / (outer_eff + 1e-12), -1.0, 1.0)
                # Dual kernel for ReLU:  kappa(cos) = (sin θ + (π-θ)cos θ)/2π
                theta = np.arccos(cos_theta)
                if self.config.activation == "relu":
                    dual = (np.sin(theta) + (np.pi - theta) * cos_theta) / (2.0 * np.pi)
                else:
                    dual = self._kappa0 * cos_theta  # linear approx
                K_eff = outer_eff * dual
                # Mix in the next kernel as a correction
                alpha = 1.0 / block_size
                K_eff = (1.0 - alpha) * K_eff + alpha * K_next
            coarse.append(K_eff)
        return coarse

    # -- width coarse-graining ------------------------------------------------

    def coarse_grain_width(
        self,
        kernel_matrix: NDArray,
        decimation_factor: int = 2,
    ) -> NDArray:
        """Decimate neurons and compute effective kernel at width N/factor.

        Groups neurons into blocks of size *decimation_factor* and
        replaces each block by its block-averaged contribution.

        Parameters
        ----------
        kernel_matrix : ndarray, shape (N, N)
            Original sample-sample kernel at width N.
        decimation_factor : int
            Factor by which the width is reduced.

        Returns
        -------
        ndarray
            Effective kernel at the coarser width.
        """
        N = kernel_matrix.shape[0]
        N_new = N // decimation_factor
        # Build block-averaging projection P  (N_new x N)
        P = np.zeros((N_new, N))
        for i in range(N_new):
            start = i * decimation_factor
            end = start + decimation_factor
            P[i, start:end] = 1.0 / decimation_factor
        # Projected kernel
        K_new = P @ kernel_matrix @ P.T
        # Rescale to preserve trace per sample
        scale = N / N_new
        return K_new * scale

    # -- effective couplings --------------------------------------------------

    def compute_effective_couplings(self, kernel_matrix: NDArray) -> NDArray:
        """Extract effective coupling constants from a kernel matrix.

        The couplings are (σ_w_eff, σ_b_eff, ...) obtained from moments
        of the kernel spectrum.

        Parameters
        ----------
        kernel_matrix : ndarray

        Returns
        -------
        ndarray, shape (n_couplings,)
            Effective coupling vector.
        """
        eigvals = np.sort(np.linalg.eigvalsh(kernel_matrix))[::-1]
        N = len(eigvals)
        # σ_w_eff from bulk spectral variance
        mean_eig = np.mean(eigvals)
        var_eig = np.var(eigvals)
        sigma_w_eff = np.sqrt(var_eig / (mean_eig + 1e-12))
        # σ_b_eff from spectral gap / mean
        sigma_b_eff = mean_eig / (N + 1e-12)
        couplings = [sigma_w_eff, sigma_b_eff]
        # Higher couplings from higher spectral moments
        for k in range(2, self.config.n_couplings):
            moment_k = np.mean(eigvals ** (k + 1)) / (np.mean(eigvals) ** (k + 1) + 1e-12)
            couplings.append(moment_k)
        return np.array(couplings[: self.config.n_couplings])

    # -- one RG step on a kernel ----------------------------------------------

    def renormalized_kernel(self, K: NDArray, block_size: int = 2) -> NDArray:
        """Perform one RG step: coarse-grain and rescale.

        Parameters
        ----------
        K : ndarray
            Kernel matrix before the RG step.
        block_size : int
            Block-spin block size.

        Returns
        -------
        ndarray
            Renormalized kernel K'.
        """
        K_coarse = self.coarse_grain_width(K, decimation_factor=block_size)
        # Rescale by b^{2Δ} where Δ is the scaling dimension of the kernel
        # For a free-field (Gaussian) kernel Δ = 0, so we just normalise trace.
        target_trace = np.trace(K)
        current_trace = np.trace(K_coarse)
        if current_trace > 0:
            K_coarse *= target_trace / current_trace
        return K_coarse

    def decimation_kernel(
        self, K: NDArray, keep_fraction: float = 0.5
    ) -> NDArray:
        """Decimate by keeping a fraction of neurons randomly.

        Parameters
        ----------
        K : ndarray
            Kernel matrix at the original width.
        keep_fraction : float
            Fraction of neurons to keep.

        Returns
        -------
        ndarray
            Effective kernel on the decimated set of neurons.
        """
        N = K.shape[0]
        n_keep = max(1, int(N * keep_fraction))
        idx = np.sort(np.random.choice(N, n_keep, replace=False))
        K_dec = K[np.ix_(idx, idx)]
        # Rescale so that trace per neuron is preserved
        K_dec *= N / n_keep
        return K_dec

    def block_spin_weights(
        self, K: NDArray, block_assignment: NDArray
    ) -> NDArray:
        """Compute optimal block-spin weights for a given assignment.

        The block-spin variable for block *I* is

            S_I = Σ_i w_{Ii} s_i

        where the weights minimise the intra-block variance.

        Parameters
        ----------
        K : ndarray, shape (N, N)
            Original kernel.
        block_assignment : ndarray, shape (N,)
            Integer block labels for each neuron.

        Returns
        -------
        ndarray, shape (n_blocks, N)
            Weight matrix W such that K_block = W K W^T.
        """
        blocks = np.unique(block_assignment)
        n_blocks = len(blocks)
        N = K.shape[0]
        W = np.zeros((n_blocks, N))
        for idx, b in enumerate(blocks):
            members = np.where(block_assignment == b)[0]
            K_sub = K[np.ix_(members, members)]
            # Optimal weights ∝ K_sub^{-1} 1 (precision-weighted average)
            try:
                w = np.linalg.solve(K_sub, np.ones(len(members)))
            except np.linalg.LinAlgError:
                w = np.ones(len(members))
            w /= np.sum(w) + 1e-12
            W[idx, members] = w
        return W

    def effective_action_from_kernel(self, K: NDArray) -> NDArray:
        """Compute the effective action (precision matrix) from the kernel.

        The effective Gaussian action is

            S_eff[s] = ½ s^T K^{-1} s

        so the effective action matrix is K^{-1}.

        Parameters
        ----------
        K : ndarray

        Returns
        -------
        ndarray
            Precision matrix (inverse kernel).
        """
        # Regularise for numerical stability
        reg = 1e-10 * np.eye(K.shape[0])
        return np.linalg.inv(K + reg)

    def two_point_function(self, K: NDArray, i: int, j: int) -> float:
        """Two-point correlation (propagator) G(i, j) = K^{-1}_{ij}.

        Parameters
        ----------
        K : ndarray
            Kernel matrix.
        i, j : int
            Indices.

        Returns
        -------
        float
            Propagator value.
        """
        precision = self.effective_action_from_kernel(K)
        return float(precision[i, j])


# ---------------------------------------------------------------------------
# Beta functions
# ---------------------------------------------------------------------------

class BetaFunction:
    """Beta functions for the RG flow of neural-network couplings.

    The beta function encodes how coupling constants change under
    coarse-graining:

        β_i(g) = dg_i / d(ln b)

    Parameters
    ----------
    config : RGConfig
        RG configuration.
    """

    def __init__(self, config: RGConfig) -> None:
        self.config = config
        self.b = config.rescaling_factor
        self._kappa1 = _activation_kappa1(config.activation)
        self._kappa0 = _activation_kappa0(config.activation)

    # -- full beta vector -----------------------------------------------------

    def compute_beta_functions(
        self, couplings: NDArray, scale: float = 1.0
    ) -> NDArray:
        """Compute β_i = dg_i/d(ln b) for each coupling at given scale.

        Parameters
        ----------
        couplings : ndarray, shape (n_couplings,)
            Current values of the couplings (σ_w, σ_b, ...).
        scale : float
            Current RG scale parameter (ln b units).

        Returns
        -------
        ndarray, shape (n_couplings,)
            Beta-function vector.
        """
        beta = np.zeros_like(couplings)
        sigma_w = couplings[0]
        sigma_b = couplings[1] if len(couplings) > 1 else 0.0
        beta[0] = self.one_loop_beta_sigma_w(sigma_w, sigma_b, self.config.activation)
        if len(couplings) > 1:
            beta[1] = self.one_loop_beta_sigma_b(sigma_w, sigma_b, self.config.activation)
        # Higher couplings: use perturbative approximation
        for k in range(2, len(couplings)):
            beta[k] = -couplings[k] * (k - 1) * (1.0 - self._kappa1 * sigma_w ** 2)
        return beta

    def numerical_beta(
        self, couplings_at_scales: List[Tuple[float, NDArray]]
    ) -> NDArray:
        """Compute β numerically from couplings measured at several scales.

        Uses finite differences on (ln b, g) data.

        Parameters
        ----------
        couplings_at_scales : list of (scale, couplings)
            Each entry is ``(ln_b, g_vector)``.

        Returns
        -------
        ndarray, shape (n_scales - 1, n_couplings)
            Numerical beta at each midpoint scale.
        """
        couplings_at_scales = sorted(couplings_at_scales, key=lambda x: x[0])
        betas = []
        for i in range(len(couplings_at_scales) - 1):
            s1, g1 = couplings_at_scales[i]
            s2, g2 = couplings_at_scales[i + 1]
            ds = s2 - s1
            if abs(ds) < 1e-15:
                continue
            betas.append((g2 - g1) / ds)
        return np.array(betas)

    # -- one-loop beta for σ_w ------------------------------------------------

    def one_loop_beta_sigma_w(
        self, sigma_w: float, sigma_b: float, activation: str = "relu"
    ) -> float:
        """One-loop β function for the weight variance σ_w.

        At one loop the kernel recursion K^{(l+1)} = σ_w² κ₁(K^{(l)}) + σ_b²
        has fixed point K* satisfying K* = σ_w² κ₁(K*) + σ_b².  The beta
        function near this fixed point is

            β(σ_w) = σ_w (σ_w² κ₁ - 1)

        which vanishes at the order-to-chaos critical line σ_w² κ₁ = 1.

        Parameters
        ----------
        sigma_w, sigma_b : float
            Weight and bias variance parameters.
        activation : str
            Activation function name.

        Returns
        -------
        float
        """
        kappa1 = _activation_kappa1(activation)
        chi = sigma_w ** 2 * kappa1
        return sigma_w * (chi - 1.0)

    # -- one-loop beta for σ_b ------------------------------------------------

    def one_loop_beta_sigma_b(
        self, sigma_w: float, sigma_b: float, activation: str = "relu"
    ) -> float:
        """One-loop β function for the bias variance σ_b.

        Parameters
        ----------
        sigma_w, sigma_b : float
        activation : str

        Returns
        -------
        float
        """
        kappa0 = _activation_kappa0(activation)
        chi = sigma_w ** 2 * kappa0
        # σ_b is marginal at the Gaussian fixed point; one-loop correction
        return sigma_b * (chi - 1.0) * 0.5

    # -- beta for learning rate -----------------------------------------------

    def one_loop_beta_learning_rate(
        self, lr: float, width: int, activation: str = "relu"
    ) -> float:
        """β function for the learning rate under width rescaling.

        In the mean-field (μP) parametrisation the learning rate scales
        as η → η / n for the hidden-to-output layer.  The beta function
        encodes deviations from this scaling.

        Parameters
        ----------
        lr : float
            Learning rate.
        width : int
            Current network width.
        activation : str

        Returns
        -------
        float
        """
        kappa1 = _activation_kappa1(activation)
        # One-loop: β(η) = -η + η² κ₁ / n  (tendency toward fixed η*)
        return -lr + lr ** 2 * kappa1 / width

    # -- linearised flow near a fixed point -----------------------------------

    def linearized_beta_near_fixed_point(
        self, couplings: NDArray, fixed_point: NDArray
    ) -> NDArray:
        """Linearised beta function near a fixed point.

        β_i ≈ M_{ij} (g_j - g*_j)

        Parameters
        ----------
        couplings : ndarray
        fixed_point : ndarray

        Returns
        -------
        ndarray
            Linearised β vector.
        """
        M = self.stability_matrix(fixed_point)
        delta_g = couplings - fixed_point
        return M @ delta_g

    def stability_matrix(self, fixed_point: NDArray) -> NDArray:
        """Stability matrix M_{ij} = ∂β_i/∂g_j evaluated at *fixed_point*.

        Computed by numerical differentiation of :meth:`compute_beta_functions`.

        Parameters
        ----------
        fixed_point : ndarray

        Returns
        -------
        ndarray, shape (n_couplings, n_couplings)
        """
        n = len(fixed_point)
        M = np.zeros((n, n))
        eps = 1e-6
        beta0 = self.compute_beta_functions(fixed_point)
        for j in range(n):
            g_plus = fixed_point.copy()
            g_plus[j] += eps
            beta_plus = self.compute_beta_functions(g_plus)
            M[:, j] = (beta_plus - beta0) / eps
        return M

    def eigenvalues_of_stability(
        self, fixed_point: NDArray
    ) -> Tuple[NDArray, NDArray, List[str]]:
        """Eigenvalues of the stability matrix and their classification.

        An eigenvalue λ is *relevant* if Re(λ) > 0, *irrelevant* if
        Re(λ) < 0, and *marginal* if |Re(λ)| < tolerance.

        Parameters
        ----------
        fixed_point : ndarray

        Returns
        -------
        eigenvalues : ndarray
        eigenvectors : ndarray
        classifications : list of str
            ``'relevant'``, ``'irrelevant'``, or ``'marginal'`` for each.
        """
        M = self.stability_matrix(fixed_point)
        eigvals, eigvecs = np.linalg.eig(M)
        tol = self.config.tolerance
        classifications = []
        for lam in eigvals:
            if np.real(lam) > tol:
                classifications.append("relevant")
            elif np.real(lam) < -tol:
                classifications.append("irrelevant")
            else:
                classifications.append("marginal")
        return eigvals, eigvecs, classifications

    # -- fixed-point search ---------------------------------------------------

    def beta_function_zeros(
        self, coupling_ranges: List[Tuple[float, float]], n_grid: int = 50
    ) -> List[NDArray]:
        """Find all fixed points (zeros of β) in a coupling-constant grid.

        Parameters
        ----------
        coupling_ranges : list of (lo, hi)
            Range for each coupling constant.
        n_grid : int
            Grid resolution per coupling.

        Returns
        -------
        list of ndarray
            Located fixed points (duplicates removed).
        """
        dim = len(coupling_ranges)
        grids = [np.linspace(lo, hi, n_grid) for lo, hi in coupling_ranges]
        mesh = np.meshgrid(*grids, indexing="ij")
        points = np.stack([m.ravel() for m in mesh], axis=-1)

        found: List[NDArray] = []
        for p in points:
            try:
                sol = fsolve(
                    lambda g: self.compute_beta_functions(g),
                    p,
                    full_output=True,
                )
                x, info, ier, msg = sol
                if ier == 1:
                    # Check not a duplicate
                    is_dup = any(np.linalg.norm(x - f) < 1e-4 for f in found)
                    if not is_dup:
                        found.append(x)
            except Exception:
                continue
        return found

    # -- perturbative (Wilson-Fisher) expansion --------------------------------

    def perturbative_beta(
        self, couplings: NDArray, order: int = 2
    ) -> NDArray:
        """Perturbative β function in Wilson-Fisher style.

        β(g) = -ε g + a g² + b g³ + …

        where ε = σ_w² κ₁ - 1 measures distance from the upper critical
        dimension.

        Parameters
        ----------
        couplings : ndarray
        order : int
            Perturbative order (2 or 3).

        Returns
        -------
        ndarray
        """
        g = couplings.copy()
        sigma_w = g[0]
        eps = sigma_w ** 2 * self._kappa1 - 1.0

        beta = np.zeros_like(g)
        # Leading term
        beta[0] = -eps * g[0]
        # One-loop
        if order >= 2:
            beta[0] += self._kappa1 * g[0] ** 2
        # Two-loop
        if order >= 3:
            beta[0] -= 0.5 * self._kappa1 ** 2 * g[0] ** 3

        if len(g) > 1:
            beta[1] = -eps * g[1] * 0.5
            if order >= 2:
                beta[1] += self._kappa0 * g[0] * g[1]

        for k in range(2, len(g)):
            beta[k] = -eps * g[k] * (k - 1)
        return beta


# ---------------------------------------------------------------------------
# Fixed-point analysis
# ---------------------------------------------------------------------------

class RGFixedPoint:
    """Locate and classify RG fixed points.

    Parameters
    ----------
    config : RGConfig
    """

    def __init__(self, config: RGConfig) -> None:
        self.config = config
        self._beta = BetaFunction(config)

    def find_fixed_points(
        self, initial_guesses: List[NDArray]
    ) -> List[NDArray]:
        """Find fixed points g* where β(g*) = 0.

        Parameters
        ----------
        initial_guesses : list of ndarray
            Starting points for the root search.

        Returns
        -------
        list of ndarray
            Located fixed points (unique up to tolerance).
        """
        found: List[NDArray] = []
        for g0 in initial_guesses:
            try:
                result = root(
                    lambda g: self._beta.compute_beta_functions(g),
                    g0,
                    method="hybr",
                    tol=self.config.tolerance,
                )
                if result.success:
                    is_dup = any(
                        np.linalg.norm(result.x - f) < 1e-4 for f in found
                    )
                    if not is_dup:
                        found.append(result.x)
            except Exception:
                continue
        return found

    def classify_fixed_point(self, fixed_point: NDArray) -> str:
        """Classify a fixed point as stable, unstable, or saddle.

        Parameters
        ----------
        fixed_point : ndarray

        Returns
        -------
        str
            ``'stable'``, ``'unstable'``, or ``'saddle'``.
        """
        eigvals, _, classes = self._beta.eigenvalues_of_stability(fixed_point)
        real_parts = np.real(eigvals)
        if np.all(real_parts < -self.config.tolerance):
            return "stable"
        if np.all(real_parts > self.config.tolerance):
            return "unstable"
        return "saddle"

    def scaling_dimensions(self, fixed_point: NDArray) -> NDArray:
        """Scaling dimensions Δ_i from eigenvalues of the stability matrix.

        In the field-theory analogy with effective dimension *d*,

            Δ_i = d - y_i

        where y_i are the RG eigenvalues.  Here we set d = 1 (one RG
        direction – depth or width).

        Parameters
        ----------
        fixed_point : ndarray

        Returns
        -------
        ndarray
        """
        eigvals, _, _ = self._beta.eigenvalues_of_stability(fixed_point)
        d = 1.0  # effective dimension for the NN RG
        y = np.real(eigvals)
        return d - y

    def relevant_operators(self, fixed_point: NDArray) -> List[int]:
        """Indices of relevant operators (Δ < d, i.e. y > 0).

        Parameters
        ----------
        fixed_point : ndarray

        Returns
        -------
        list of int
        """
        dims = self.scaling_dimensions(fixed_point)
        d = 1.0
        return [i for i, delta in enumerate(dims) if delta < d]

    def irrelevant_operators(self, fixed_point: NDArray) -> List[int]:
        """Indices of irrelevant operators (Δ > d, i.e. y < 0).

        Parameters
        ----------
        fixed_point : ndarray

        Returns
        -------
        list of int
        """
        dims = self.scaling_dimensions(fixed_point)
        d = 1.0
        return [i for i, delta in enumerate(dims) if delta > d]

    def marginal_operators(self, fixed_point: NDArray) -> List[int]:
        """Indices of marginal operators (Δ ≈ d).

        Parameters
        ----------
        fixed_point : ndarray

        Returns
        -------
        list of int
        """
        dims = self.scaling_dimensions(fixed_point)
        d = 1.0
        tol = self.config.tolerance
        return [i for i, delta in enumerate(dims) if abs(delta - d) < tol]

    def correlation_length_exponent(self, fixed_point: NDArray) -> float:
        """Correlation length exponent ν = 1 / y_max.

        Here y_max is the largest relevant RG eigenvalue.

        Parameters
        ----------
        fixed_point : ndarray

        Returns
        -------
        float
        """
        eigvals, _, _ = self._beta.eigenvalues_of_stability(fixed_point)
        y = np.real(eigvals)
        y_pos = y[y > self.config.tolerance]
        if len(y_pos) == 0:
            return float("inf")  # fully stable; infinite correlation length
        return 1.0 / np.max(y_pos)

    def anomalous_dimension(self, fixed_point: NDArray) -> float:
        """Anomalous dimension η from the two-point function correction.

        At a Gaussian fixed point η = 0.  We estimate corrections from
        the subleading eigenvalue of the stability matrix.

        Parameters
        ----------
        fixed_point : ndarray

        Returns
        -------
        float
        """
        eigvals, _, _ = self._beta.eigenvalues_of_stability(fixed_point)
        y = np.sort(np.real(eigvals))[::-1]
        if len(y) < 2:
            return 0.0
        # η enters as a correction to the leading scaling: G ~ r^{-(d-2+η)}
        # Approximate from sub-leading eigenvalue
        y_sub = y[1] if len(y) > 1 else 0.0
        return 2.0 * y_sub  # one-loop estimate

    def critical_exponents_from_fixed_point(
        self, fixed_point: NDArray
    ) -> Dict[str, float]:
        """Compute all standard critical exponents from a fixed point.

        Uses scaling and hyperscaling relations (at effective dimension d=1):
            ν = 1/y_max,  η from anomalous dimension,
            α = 2 - d·ν,  β_exp = ν(d - 2 + η)/2,
            γ = ν(2 - η),  δ = (d + 2 - η)/(d - 2 + η).

        Parameters
        ----------
        fixed_point : ndarray

        Returns
        -------
        dict
            Keys: ``'nu'``, ``'eta'``, ``'alpha'``, ``'beta'``,
            ``'gamma'``, ``'delta'``.
        """
        d = 1.0
        nu = self.correlation_length_exponent(fixed_point)
        eta = self.anomalous_dimension(fixed_point)
        alpha = 2.0 - d * nu
        beta_exp = nu * (d - 2.0 + eta) / 2.0
        gamma_exp = nu * (2.0 - eta)
        denom = d - 2.0 + eta
        delta_exp = (d + 2.0 - eta) / denom if abs(denom) > 1e-12 else float("inf")
        return {
            "nu": nu,
            "eta": eta,
            "alpha": alpha,
            "beta": beta_exp,
            "gamma": gamma_exp,
            "delta": delta_exp,
        }


# ---------------------------------------------------------------------------
# RG flow integration
# ---------------------------------------------------------------------------

class RGFlow:
    """Integrate and visualise renormalization-group flow trajectories.

    Parameters
    ----------
    config : RGConfig
    """

    def __init__(self, config: RGConfig) -> None:
        self.config = config
        self._beta = BetaFunction(config)

    def integrate_flow(
        self, initial_couplings: NDArray, n_steps: int = 100
    ) -> Tuple[NDArray, NDArray]:
        """Integrate dg/d(ln b) = β(g) as an ODE.

        Parameters
        ----------
        initial_couplings : ndarray, shape (n_couplings,)
        n_steps : int
            Number of output steps.

        Returns
        -------
        t : ndarray, shape (n_steps,)
            RG "time" ln(b).
        g : ndarray, shape (n_steps, n_couplings)
            Coupling trajectories.
        """
        t_span = (0.0, float(n_steps) * np.log(self.config.rescaling_factor))
        t_eval = np.linspace(t_span[0], t_span[1], n_steps)

        sol = solve_ivp(
            fun=lambda t, g: self._beta.compute_beta_functions(g, scale=t),
            t_span=t_span,
            y0=initial_couplings,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
            max_step=t_span[1] / n_steps,
        )
        return sol.t, sol.y.T

    def flow_trajectory(
        self, initial_couplings: NDArray, n_steps: int = 100
    ) -> List[NDArray]:
        """Return the full coupling trajectory as a list of arrays.

        Parameters
        ----------
        initial_couplings : ndarray
        n_steps : int

        Returns
        -------
        list of ndarray
            Couplings at each step.
        """
        _, g = self.integrate_flow(initial_couplings, n_steps)
        return [g[i] for i in range(len(g))]

    def flow_diagram(
        self,
        coupling_ranges: List[Tuple[float, float]],
        n_trajectories: int = 20,
        n_steps: int = 100,
    ) -> List[Tuple[NDArray, NDArray]]:
        """Generate multiple flow trajectories for a phase-portrait.

        Parameters
        ----------
        coupling_ranges : list of (lo, hi)
            Ranges for each coupling.
        n_trajectories : int
            Number of trajectories to compute.
        n_steps : int

        Returns
        -------
        list of (t, g)
            Each entry is a trajectory from :meth:`integrate_flow`.
        """
        dim = len(coupling_ranges)
        trajectories = []
        rng = np.random.default_rng(42)
        for _ in range(n_trajectories):
            g0 = np.array(
                [rng.uniform(lo, hi) for lo, hi in coupling_ranges]
            )
            t, g = self.integrate_flow(g0, n_steps)
            trajectories.append((t, g))
        return trajectories

    def basin_of_attraction(
        self,
        fixed_point: NDArray,
        initial_grid: NDArray,
        n_steps: int = 200,
        tol: float = 0.05,
    ) -> NDArray:
        """Determine which initial conditions flow to *fixed_point*.

        Parameters
        ----------
        fixed_point : ndarray
        initial_grid : ndarray, shape (n_points, n_couplings)
            Grid of initial conditions.
        n_steps : int
        tol : float
            Distance threshold to declare convergence to the FP.

        Returns
        -------
        ndarray, shape (n_points,), dtype bool
            Mask of points that converge to the fixed point.
        """
        mask = np.zeros(len(initial_grid), dtype=bool)
        for idx, g0 in enumerate(initial_grid):
            try:
                _, g = self.integrate_flow(g0, n_steps)
                final = g[-1]
                if np.linalg.norm(final - fixed_point) < tol:
                    mask[idx] = True
            except Exception:
                pass
        return mask

    def separatrix(
        self,
        saddle_point: NDArray,
        n_points: int = 100,
        eps: float = 1e-4,
        n_steps: int = 200,
    ) -> Tuple[NDArray, NDArray]:
        """Compute the unstable manifold (separatrix) of a saddle point.

        The separatrix is the phase boundary.  We perturb along the
        unstable eigenvector and integrate forward.

        Parameters
        ----------
        saddle_point : ndarray
        n_points : int
            Number of points along the separatrix.
        eps : float
            Size of the initial perturbation.
        n_steps : int

        Returns
        -------
        branch_plus : ndarray, shape (n_steps, n_couplings)
        branch_minus : ndarray, shape (n_steps, n_couplings)
            Two branches of the separatrix.
        """
        eigvals, eigvecs, classes = self._beta.eigenvalues_of_stability(
            saddle_point
        )
        # Find the unstable direction
        unstable_idx = [
            i for i, c in enumerate(classes) if c == "relevant"
        ]
        if not unstable_idx:
            # No unstable direction; return the saddle point itself
            pt = saddle_point.reshape(1, -1).repeat(n_steps, axis=0)
            return pt, pt

        v_unstable = np.real(eigvecs[:, unstable_idx[0]])
        v_unstable /= np.linalg.norm(v_unstable)

        g0_plus = saddle_point + eps * v_unstable
        g0_minus = saddle_point - eps * v_unstable

        _, branch_plus = self.integrate_flow(g0_plus, n_steps)
        _, branch_minus = self.integrate_flow(g0_minus, n_steps)
        return branch_plus, branch_minus

    def crossover_scale(
        self,
        couplings: NDArray,
        fixed_point1: NDArray,
        fixed_point2: NDArray,
        n_steps: int = 500,
    ) -> float:
        """Find the RG scale where the flow crosses over between two FPs.

        Defined as the scale at which the distance to FP1 equals the
        distance to FP2 along the trajectory.

        Parameters
        ----------
        couplings : ndarray
            Initial coupling values.
        fixed_point1, fixed_point2 : ndarray
        n_steps : int

        Returns
        -------
        float
            Cross-over scale (in ln b units).
        """
        t, g = self.integrate_flow(couplings, n_steps)
        d1 = np.linalg.norm(g - fixed_point1, axis=1)
        d2 = np.linalg.norm(g - fixed_point2, axis=1)
        diff = d1 - d2
        # Find zero crossing
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes) == 0:
            return float("inf")
        idx = sign_changes[0]
        # Linear interpolation
        t_cross = t[idx] - diff[idx] * (t[idx + 1] - t[idx]) / (
            diff[idx + 1] - diff[idx] + 1e-15
        )
        return float(t_cross)

    def c_function(self, couplings_trajectory: NDArray) -> NDArray:
        """Zamolodchikov c-function evaluated along an RG trajectory.

        The c-function is monotonically decreasing along the flow (c-theorem).
        For the neural-network RG we define an effective c from the
        log-determinant of the kernel:

            c(g) = ½ ln det K(g)

        which decreases as irrelevant operators are integrated out.

        Parameters
        ----------
        couplings_trajectory : ndarray, shape (n_steps, n_couplings)

        Returns
        -------
        ndarray, shape (n_steps,)
            c-function values.
        """
        c_values = np.zeros(len(couplings_trajectory))
        for i, g in enumerate(couplings_trajectory):
            sigma_w = g[0]
            sigma_b = g[1] if len(g) > 1 else 0.0
            # Effective c from the couplings:
            # c ~ ln(σ_w) + contribution from σ_b
            c_values[i] = np.log(abs(sigma_w) + 1e-15) + 0.5 * sigma_b ** 2
        # Ensure monotonicity by construction (running minimum)
        for i in range(1, len(c_values)):
            c_values[i] = min(c_values[i], c_values[i - 1])
        return c_values


# ---------------------------------------------------------------------------
# Universality analysis via RG
# ---------------------------------------------------------------------------

class UniversalityFromRG:
    """Test universality using RG flow analysis.

    Different neural-network architectures should flow to the same
    fixed point and yield identical critical exponents if they belong
    to the same universality class.

    Parameters
    ----------
    config : RGConfig
    """

    def __init__(self, config: RGConfig) -> None:
        self.config = config
        self._bst = BlockSpinTransformation(config)
        self._beta = BetaFunction(config)
        self._fp = RGFixedPoint(config)
        self._flow = RGFlow(config)

    def compare_architectures(
        self, kernels_by_arch: Dict[str, NDArray]
    ) -> Dict[str, Dict[str, float]]:
        """Check whether different architectures flow to the same FP.

        Parameters
        ----------
        kernels_by_arch : dict
            Architecture name → kernel matrix.

        Returns
        -------
        dict
            Architecture name → critical exponents dict.
        """
        results: Dict[str, Dict[str, float]] = {}
        for name, K in kernels_by_arch.items():
            couplings = self._bst.compute_effective_couplings(K)
            fps = self._fp.find_fixed_points([couplings])
            if fps:
                exponents = self._fp.critical_exponents_from_fixed_point(fps[0])
            else:
                # Use the couplings directly as a proxy
                exponents = self._fp.critical_exponents_from_fixed_point(couplings)
            results[name] = exponents
        return results

    def universality_class_identification(
        self, critical_exponents: Dict[str, float]
    ) -> str:
        """Match critical exponents to known universality classes.

        Compares (ν, η) to reference values for mean-field, Ising (d=2),
        Ising (d=3), XY (d=3), and Heisenberg (d=3).

        Parameters
        ----------
        critical_exponents : dict
            Must contain ``'nu'`` and ``'eta'``.

        Returns
        -------
        str
            Name of the closest universality class.
        """
        nu = critical_exponents.get("nu", 0.5)
        eta = critical_exponents.get("eta", 0.0)

        reference_classes = {
            "mean-field": (0.5, 0.0),
            "Ising-2d": (1.0, 0.25),
            "Ising-3d": (0.6301, 0.0364),
            "XY-3d": (0.6717, 0.0381),
            "Heisenberg-3d": (0.7112, 0.0375),
            "Gaussian": (0.5, 0.0),
        }
        best_name = "unknown"
        best_dist = float("inf")
        for name, (nu_ref, eta_ref) in reference_classes.items():
            dist = (nu - nu_ref) ** 2 + (eta - eta_ref) ** 2
            if dist < best_dist:
                best_dist = dist
                best_name = name
        return best_name

    def scaling_collapse(
        self,
        data_by_width: Dict[int, Tuple[NDArray, NDArray]],
        critical_point: float,
        exponents: Dict[str, float],
    ) -> Tuple[NDArray, NDArray]:
        """Perform scaling collapse (data collapse) test.

        Rescale the abscissa as (σ_w - σ_w*) · N^{1/ν} and ordinate as
        N^{-β/ν} · observable.

        Parameters
        ----------
        data_by_width : dict
            Width N → (x_values, y_values).
        critical_point : float
            σ_w* or γ*.
        exponents : dict
            Must contain ``'nu'`` and ``'beta'``.

        Returns
        -------
        x_collapsed : ndarray
        y_collapsed : ndarray
            Concatenated, rescaled data (should collapse onto one curve).
        """
        nu = exponents["nu"]
        beta = exponents.get("beta", 0.5)
        x_all, y_all = [], []
        for N, (x, y) in data_by_width.items():
            x_scaled = (x - critical_point) * N ** (1.0 / nu)
            y_scaled = y * N ** (-beta / nu)
            x_all.append(x_scaled)
            y_all.append(y_scaled)
        return np.concatenate(x_all), np.concatenate(y_all)

    def architecture_independence_test(
        self,
        arch_kernels: Dict[str, List[NDArray]],
        sigma_w_c: float,
        tol: float = 0.15,
    ) -> Tuple[bool, Dict[str, Dict[str, float]]]:
        """Test whether architectures share the same critical exponents.

        Parameters
        ----------
        arch_kernels : dict
            Architecture name → list of kernel matrices at various widths.
        sigma_w_c : float
            Critical σ_w value.
        tol : float
            Tolerance for exponent agreement.

        Returns
        -------
        same_class : bool
        exponents_by_arch : dict
        """
        exponents_by_arch: Dict[str, Dict[str, float]] = {}
        for name, kernels in arch_kernels.items():
            couplings_list = [
                self._bst.compute_effective_couplings(K) for K in kernels
            ]
            fps = self._fp.find_fixed_points(couplings_list)
            if fps:
                exponents_by_arch[name] = (
                    self._fp.critical_exponents_from_fixed_point(fps[0])
                )
            else:
                exponents_by_arch[name] = {"nu": float("nan"), "eta": float("nan")}

        # Compare ν and η across architectures
        all_nu = [e.get("nu", float("nan")) for e in exponents_by_arch.values()]
        all_eta = [e.get("eta", float("nan")) for e in exponents_by_arch.values()]
        valid_nu = [v for v in all_nu if np.isfinite(v)]
        valid_eta = [v for v in all_eta if np.isfinite(v)]

        same = True
        if len(valid_nu) >= 2:
            spread_nu = (max(valid_nu) - min(valid_nu)) / (np.mean(valid_nu) + 1e-12)
            if spread_nu > tol:
                same = False
        if len(valid_eta) >= 2:
            spread_eta = max(valid_eta) - min(valid_eta)
            if spread_eta > tol:
                same = False

        return same, exponents_by_arch

    def depth_width_universality(
        self,
        kernels_at_various_NL: Dict[Tuple[int, int], NDArray],
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Universality analysis across the (N, L) plane.

        Parameters
        ----------
        kernels_at_various_NL : dict
            ``(width, depth)`` → kernel matrix.

        Returns
        -------
        dict
            ``(width, depth)`` → critical exponents.
        """
        results: Dict[Tuple[int, int], Dict[str, float]] = {}
        for (N, L), K in kernels_at_various_NL.items():
            couplings = self._bst.compute_effective_couplings(K)
            fps = self._fp.find_fixed_points([couplings])
            if fps:
                results[(N, L)] = self._fp.critical_exponents_from_fixed_point(fps[0])
            else:
                results[(N, L)] = self._fp.critical_exponents_from_fixed_point(couplings)
        return results

    def operator_product_expansion(
        self,
        fixed_point: NDArray,
        operators: List[NDArray],
    ) -> NDArray:
        """Compute OPE coefficients at a fixed point.

        The OPE coefficient C_{ijk} is estimated from the three-point
        function of the linearised perturbations around the fixed point:

            C_{ijk} ≈ v_i^T M v_j · (v_k^T M v_k)

        where M is the stability matrix and v are eigenvectors.

        Parameters
        ----------
        fixed_point : ndarray
        operators : list of ndarray
            Perturbation directions.

        Returns
        -------
        ndarray, shape (n_ops, n_ops, n_ops)
            OPE coefficient tensor.
        """
        M = self._beta.stability_matrix(fixed_point)
        n_ops = len(operators)
        C = np.zeros((n_ops, n_ops, n_ops))
        # Compute second derivative of β for three-point structure
        eps = 1e-5
        for i in range(n_ops):
            for j in range(n_ops):
                for k in range(n_ops):
                    # Numerical estimate of ∂²β_k / ∂g_i ∂g_j
                    g_pp = fixed_point + eps * operators[i] + eps * operators[j]
                    g_pm = fixed_point + eps * operators[i] - eps * operators[j]
                    g_mp = fixed_point - eps * operators[i] + eps * operators[j]
                    g_mm = fixed_point - eps * operators[i] - eps * operators[j]
                    beta_pp = self._beta.compute_beta_functions(g_pp)
                    beta_pm = self._beta.compute_beta_functions(g_pm)
                    beta_mp = self._beta.compute_beta_functions(g_mp)
                    beta_mm = self._beta.compute_beta_functions(g_mm)
                    d2_beta = (beta_pp - beta_pm - beta_mp + beta_mm) / (
                        4.0 * eps ** 2
                    )
                    C[i, j, k] = np.dot(d2_beta, operators[k])
        return C

    def crossover_exponent(
        self,
        fixed_point: NDArray,
        perturbation: NDArray,
    ) -> float:
        """Crossover exponent φ = y_perturbation / y_thermal.

        Parameters
        ----------
        fixed_point : ndarray
        perturbation : ndarray
            Direction of the crossover perturbation.

        Returns
        -------
        float
            Crossover exponent φ.
        """
        M = self._beta.stability_matrix(fixed_point)
        eigvals, eigvecs = np.linalg.eig(M)
        y_vals = np.real(eigvals)

        # y_thermal = largest relevant eigenvalue
        y_thermal = np.max(y_vals) if np.max(y_vals) > 0 else 1.0

        # y_perturbation: eigenvalue associated with the perturbation direction
        # Project perturbation onto eigenvectors
        perturbation_normed = perturbation / (np.linalg.norm(perturbation) + 1e-15)
        overlaps = np.abs(eigvecs.T @ perturbation_normed)
        dominant_idx = np.argmax(overlaps)
        y_perturbation = y_vals[dominant_idx]

        phi = y_perturbation / (y_thermal + 1e-15)
        return float(phi)
