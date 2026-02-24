"""
Signal propagation in deep networks.

Implements mean-field theory for signal propagation following:
- Poole et al. (2016) "Exponential expressivity in deep neural networks through
  transient chaos"
- Schoenholz et al. (2017) "Deep Information Propagation"

The core idea: in wide networks, pre-activations become Gaussian by CLT, so
layer-to-layer propagation of the covariance kernel Q^(l) is deterministic
in the infinite-width limit:

    q^(l+1) = σ_w² ∫ Dz φ(√q^(l) z)² + σ_b²
    c^(l+1) = σ_w² ∫∫ Dz₁ Dz₂ φ(u₁)φ(u₂) + σ_b²

where u₁ = √q₁ z₁, u₂ = √q₂ (c z₁ + √(1-c²) z₂), and Dz is the standard
Gaussian measure.

The fixed point q* and its stability χ₁ = σ_w² E[φ'(h)²] determine the
phase (ordered vs chaotic) and whether gradients vanish or explode.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Callable
import numpy as np
from scipy import optimize, integrate, special
from scipy.stats import norm
import warnings


# ---------------------------------------------------------------------------
# Gauss-Hermite quadrature helpers
# ---------------------------------------------------------------------------

_GH_POINTS, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(64)
# Convert from physicist's convention to probabilist's: ∫ f(x) e^{-x²} dx
# → ∫ f(√2 t) (1/√π) dt  ↔  E_{z~N(0,1)}[f(z)]
_GH_Z = _GH_POINTS * np.sqrt(2.0)
_GH_W = _GH_WEIGHTS / np.sqrt(np.pi)


def _gauss_expectation(f: Callable, z: np.ndarray = _GH_Z,
                       w: np.ndarray = _GH_W) -> float:
    """Compute E_{z~N(0,1)}[f(z)] via Gauss-Hermite quadrature."""
    return float(np.sum(w * f(z)))


def _gauss_expectation_2d(f: Callable, n_points: int = 64) -> float:
    """Compute E_{z1,z2 ~ N(0,1)}[f(z1,z2)] via tensor-product GH quad."""
    pts, wts = np.polynomial.hermite.hermgauss(n_points)
    z = pts * np.sqrt(2.0)
    w = wts / np.sqrt(np.pi)
    z1, z2 = np.meshgrid(z, z)
    w1, w2 = np.meshgrid(w, w)
    return float(np.sum(w1 * w2 * f(z1, z2)))


# ---------------------------------------------------------------------------
# Activation function helpers
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _relu_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1.0, 0.0)


def _relu_second_derivative(x: np.ndarray) -> np.ndarray:
    # Dirac delta at 0 — approximated as narrow Gaussian for numerics
    eps = 1e-3
    return np.exp(-0.5 * (x / eps) ** 2) / (eps * np.sqrt(2 * np.pi))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


def _tanh_second_derivative(x: np.ndarray) -> np.ndarray:
    t = np.tanh(x)
    return -2.0 * t * (1.0 - t ** 2)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + special.erf(x / np.sqrt(2.0)))


def _gelu_derivative(x: np.ndarray) -> np.ndarray:
    cdf = 0.5 * (1.0 + special.erf(x / np.sqrt(2.0)))
    pdf = np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)
    return cdf + x * pdf


def _gelu_second_derivative(x: np.ndarray) -> np.ndarray:
    pdf = np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)
    return 2.0 * pdf - x ** 2 * pdf


def _erf_fn(x: np.ndarray) -> np.ndarray:
    return special.erf(x)


def _erf_derivative(x: np.ndarray) -> np.ndarray:
    return (2.0 / np.sqrt(np.pi)) * np.exp(-x ** 2)


def _erf_second_derivative(x: np.ndarray) -> np.ndarray:
    return -(4.0 / np.sqrt(np.pi)) * x * np.exp(-x ** 2)


_ACTIVATIONS = {
    "relu": (_relu, _relu_derivative, _relu_second_derivative),
    "tanh": (_tanh, _tanh_derivative, _tanh_second_derivative),
    "gelu": (_gelu, _gelu_derivative, _gelu_second_derivative),
    "erf": (_erf_fn, _erf_derivative, _erf_second_derivative),
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PropagationConfig:
    """Configuration for signal propagation analysis.

    Parameters
    ----------
    activation : str
        Activation function name: 'relu', 'tanh', 'gelu', or 'erf'.
    sigma_w : float
        Standard deviation of weight initialization.  Weights are drawn
        W_ij ~ N(0, σ_w² / n) so that the pre-activation variance is
        σ_w² times the second moment of the post-activation.
    sigma_b : float
        Standard deviation of bias initialization, b_i ~ N(0, σ_b²).
    depth : int
        Number of layers to propagate through.
    width : Optional[int]
        Network width (for finite-width corrections).  When ``None`` the
        infinite-width (mean-field) limit is used.
    """
    activation: str = "relu"
    sigma_w: float = 1.0
    sigma_b: float = 0.0
    depth: int = 100
    width: Optional[int] = None

    def __post_init__(self):
        if self.activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{self.activation}'. "
                f"Choose from {list(_ACTIVATIONS.keys())}."
            )
        if self.sigma_w <= 0:
            raise ValueError("sigma_w must be positive.")
        if self.sigma_b < 0:
            raise ValueError("sigma_b must be non-negative.")
        if self.depth < 1:
            raise ValueError("depth must be >= 1.")


# ---------------------------------------------------------------------------
# ActivationKernels
# ---------------------------------------------------------------------------

class ActivationKernels:
    """Pre-compute kernel functions for different activations.

    Given two pre-activations h₁, h₂ jointly Gaussian with covariance
    [[q₁, c√(q₁q₂)], [c√(q₁q₂), q₂]], this class computes expectations
    E[φ(h₁)φ(h₂)], E[φ'(h₁)φ'(h₂)], etc., which determine the
    layer-to-layer covariance recursion.

    For ReLU there are closed-form expressions (Cho & Saul 2009).
    For other activations we use Gauss-Hermite quadrature on the
    two-dimensional integral.
    """

    def __init__(self, n_quad: int = 64):
        """
        Parameters
        ----------
        n_quad : int
            Number of Gauss-Hermite quadrature points per dimension.
        """
        self.n_quad = n_quad
        pts, wts = np.polynomial.hermite.hermgauss(n_quad)
        self._z = pts * np.sqrt(2.0)
        self._w = wts / np.sqrt(np.pi)

    # ---- helpers ----------------------------------------------------------

    def _correlated_inputs(self, q1: float, q2: float, c: float):
        """Return grids (u1, u2) and weight matrix for 2-D GH quadrature.

        h₁ = √q₁ · z₁
        h₂ = √q₂ · (c·z₁ + √(1-c²)·z₂)
        """
        c_clip = np.clip(c, -1.0 + 1e-12, 1.0 - 1e-12)
        sq1 = np.sqrt(max(q1, 0.0))
        sq2 = np.sqrt(max(q2, 0.0))
        s = np.sqrt(1.0 - c_clip ** 2)
        z1, z2 = np.meshgrid(self._z, self._z)
        w1, w2 = np.meshgrid(self._w, self._w)
        u1 = sq1 * z1
        u2 = sq2 * (c_clip * z1 + s * z2)
        W = w1 * w2
        return u1, u2, W

    def _generic_kernel(self, phi, q1, q2, c):
        """E[φ(h₁)φ(h₂)] for arbitrary φ via 2-D quadrature."""
        u1, u2, W = self._correlated_inputs(q1, q2, c)
        return float(np.sum(W * phi(u1) * phi(u2)))

    # ---- ReLU kernel (closed form) ----------------------------------------

    def relu_kernel(self, q1: float, q2: float, c: float) -> float:
        """Compute E[relu(h₁)·relu(h₂)] for correlated Gaussians.

        Uses the closed-form result of Cho & Saul (2009):

            K(q₁, q₂, c) = (√(q₁q₂) / (2π)) · (sin θ + (π - θ) cos θ)

        where θ = arccos(c).

        Parameters
        ----------
        q1, q2 : float
            Variances of the two pre-activations.
        c : float
            Correlation coefficient, c = C₁₂ / √(q₁ q₂).

        Returns
        -------
        float
            The kernel value E[relu(h₁)relu(h₂)].
        """
        c_clip = np.clip(c, -1.0, 1.0)
        theta = np.arccos(c_clip)
        sq = np.sqrt(max(q1 * q2, 0.0))
        return sq / (2.0 * np.pi) * (np.sin(theta) + (np.pi - theta) * np.cos(theta))

    # ---- tanh kernel (quadrature) -----------------------------------------

    def tanh_kernel(self, q1: float, q2: float, c: float) -> float:
        """Compute E[tanh(h₁)·tanh(h₂)] for correlated Gaussians.

        No closed form exists; computed via 2-D Gauss-Hermite quadrature.

        Parameters
        ----------
        q1, q2 : float
            Pre-activation variances.
        c : float
            Correlation coefficient.

        Returns
        -------
        float
            Kernel value.
        """
        return self._generic_kernel(np.tanh, q1, q2, c)

    # ---- GELU kernel (quadrature) -----------------------------------------

    def gelu_kernel(self, q1: float, q2: float, c: float) -> float:
        """Compute E[GELU(h₁)·GELU(h₂)] for correlated Gaussians.

        GELU(x) = x · Φ(x) where Φ is the standard normal CDF.
        Computed via 2-D Gauss-Hermite quadrature.

        Parameters
        ----------
        q1, q2 : float
            Pre-activation variances.
        c : float
            Correlation coefficient.

        Returns
        -------
        float
            Kernel value.
        """
        return self._generic_kernel(_gelu, q1, q2, c)

    # ---- erf kernel (quadrature / analytic) --------------------------------

    def erf_kernel(self, q1: float, q2: float, c: float) -> float:
        """Compute E[erf(h₁)·erf(h₂)] for correlated Gaussians.

        Uses the known result (Williams 1997):

            K = (2/π) arcsin(2c√(q₁q₂) / √((1+2q₁)(1+2q₂)))

        Parameters
        ----------
        q1, q2 : float
            Pre-activation variances.
        c : float
            Correlation coefficient.

        Returns
        -------
        float
            Kernel value.
        """
        cov = c * np.sqrt(q1 * q2)
        denom = np.sqrt((1.0 + 2.0 * q1) * (1.0 + 2.0 * q2))
        arg = np.clip(2.0 * cov / denom, -1.0, 1.0)
        return (2.0 / np.pi) * np.arcsin(arg)

    # ---- derivative kernels ------------------------------------------------

    def derivative_kernel(self, q1: float, q2: float, c: float,
                          activation: str = "relu") -> float:
        """Compute E[φ'(h₁)·φ'(h₂)] — the first-derivative kernel.

        This quantity governs gradient propagation:
            χ₁ = σ_w² · E[φ'(h)²]

        is the key stability exponent at the diagonal c = 1, q₁ = q₂ = q*.

        Parameters
        ----------
        q1, q2 : float
            Pre-activation variances.
        c : float
            Correlation coefficient.
        activation : str
            Activation function name.

        Returns
        -------
        float
            The derivative kernel E[φ'(h₁)φ'(h₂)].
        """
        if activation == "relu":
            # E[H(h₁)H(h₂)] where H is Heaviside
            c_clip = np.clip(c, -1.0, 1.0)
            theta = np.arccos(c_clip)
            return (np.pi - theta) / (2.0 * np.pi)

        _, phi_prime, _ = _ACTIVATIONS[activation]
        return self._generic_kernel(phi_prime, q1, q2, c)

    def second_derivative_kernel(self, q1: float, q2: float, c: float,
                                 activation: str = "relu") -> float:
        """Compute E[φ''(h₁)·φ''(h₂)] — the second-derivative kernel.

        Relevant for curvature propagation and higher-order corrections
        in the mean-field expansion.

        Parameters
        ----------
        q1, q2 : float
            Pre-activation variances.
        c : float
            Correlation coefficient.
        activation : str
            Activation function name.

        Returns
        -------
        float
            The second-derivative kernel.
        """
        _, _, phi_pp = _ACTIVATIONS[activation]
        return self._generic_kernel(phi_pp, q1, q2, c)

    # ---- dispatch by name --------------------------------------------------

    def kernel(self, q1: float, q2: float, c: float,
               activation: str = "relu") -> float:
        """Compute E[φ(h₁)φ(h₂)] dispatching on activation name.

        Parameters
        ----------
        q1, q2 : float
            Pre-activation variances.
        c : float
            Correlation coefficient.
        activation : str
            One of 'relu', 'tanh', 'gelu', 'erf'.

        Returns
        -------
        float
            The kernel value.
        """
        dispatch = {
            "relu": self.relu_kernel,
            "tanh": self.tanh_kernel,
            "gelu": self.gelu_kernel,
            "erf": self.erf_kernel,
        }
        return dispatch[activation](q1, q2, c)


# ---------------------------------------------------------------------------
# ForwardPropagation
# ---------------------------------------------------------------------------

class ForwardPropagation:
    """Propagate signal statistics (variance and correlations) through depth.

    In the infinite-width limit, the pre-activation distribution at layer l
    is Gaussian with covariance determined by the recursion:

        q^(l+1) = σ_w² · E_{h~N(0,q^(l))}[φ(h)²] + σ_b²

    For two inputs x₁, x₂ with per-layer variances q₁^(l), q₂^(l) and
    correlation c^(l):

        c^(l+1) = (σ_w² · K(q₁, q₂, c) + σ_b²) / √(q₁^(l+1) q₂^(l+1))

    This class tracks these quantities through ``depth`` layers.
    """

    def __init__(self, config: PropagationConfig):
        """
        Parameters
        ----------
        config : PropagationConfig
            Propagation parameters.
        """
        self.config = config
        self.kernels = ActivationKernels()
        self._phi, self._phi_prime, _ = _ACTIVATIONS[config.activation]
        self._sw2 = config.sigma_w ** 2
        self._sb2 = config.sigma_b ** 2

    # ---- single-input variance recursion -----------------------------------

    def _variance_step(self, q: float) -> float:
        """One step of the variance recursion.

        q^(l+1) = σ_w² E[φ(√q · z)²] + σ_b²
        """
        phi = self._phi
        val = _gauss_expectation(lambda z: phi(np.sqrt(max(q, 0.0)) * z) ** 2)
        return self._sw2 * val + self._sb2

    def propagate_variance(self, q0: float, depth: Optional[int] = None
                           ) -> np.ndarray:
        """Propagate the pre-activation variance q through ``depth`` layers.

        Parameters
        ----------
        q0 : float
            Initial pre-activation variance (layer 0).
        depth : int, optional
            Number of layers.  Defaults to ``self.config.depth``.

        Returns
        -------
        np.ndarray, shape (depth+1,)
            Variance trajectory q^(0), q^(1), …, q^(depth).
        """
        depth = depth or self.config.depth
        q_traj = np.zeros(depth + 1)
        q_traj[0] = q0
        for l in range(depth):
            q_traj[l + 1] = self._variance_step(q_traj[l])
        return q_traj

    # ---- two-input correlation recursion -----------------------------------

    def _correlation_step(self, q1: float, q2: float, c: float
                          ) -> Tuple[float, float, float]:
        """One step of correlation propagation.

        Returns (q1_next, q2_next, c_next).
        """
        K_val = self.kernels.kernel(q1, q2, c, self.config.activation)
        q1_next = self._variance_step(q1)
        q2_next = self._variance_step(q2)
        cov_next = self._sw2 * K_val + self._sb2
        denom = np.sqrt(max(q1_next * q2_next, 1e-30))
        c_next = np.clip(cov_next / denom, -1.0, 1.0)
        return q1_next, q2_next, c_next

    def propagate_correlation(self, q1: float, q2: float, c0: float,
                              depth: Optional[int] = None
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Propagate correlations between two inputs through depth.

        Parameters
        ----------
        q1, q2 : float
            Initial pre-activation variances of the two inputs.
        c0 : float
            Initial correlation coefficient c^(0) ∈ [-1, 1].
        depth : int, optional
            Number of layers.

        Returns
        -------
        q1_traj, q2_traj, c_traj : np.ndarray
            Trajectories of shape (depth+1,) each.
        """
        depth = depth or self.config.depth
        q1_t = np.zeros(depth + 1)
        q2_t = np.zeros(depth + 1)
        c_t = np.zeros(depth + 1)
        q1_t[0], q2_t[0], c_t[0] = q1, q2, c0
        for l in range(depth):
            q1_t[l + 1], q2_t[l + 1], c_t[l + 1] = self._correlation_step(
                q1_t[l], q2_t[l], c_t[l]
            )
        return q1_t, q2_t, c_t

    def propagate_full_covariance(self, Q0: np.ndarray,
                                  depth: Optional[int] = None
                                  ) -> List[np.ndarray]:
        """Propagate a full covariance matrix Q^(l) through depth.

        Given an n×n covariance matrix Q^(0) (e.g. the Gram matrix of a
        mini-batch), compute Q^(1), …, Q^(depth) via the element-wise
        kernel recursion.

        Parameters
        ----------
        Q0 : np.ndarray, shape (n, n)
            Initial covariance (Gram) matrix.
        depth : int, optional
            Number of layers.

        Returns
        -------
        list of np.ndarray
            List of (depth+1) covariance matrices [Q^(0), Q^(1), …].
        """
        depth = depth or self.config.depth
        n = Q0.shape[0]
        trajectory: List[np.ndarray] = [Q0.copy()]
        Q = Q0.copy()

        for _ in range(depth):
            Q_new = np.zeros_like(Q)
            q_diag = np.diag(Q)
            for i in range(n):
                for j in range(i, n):
                    qi, qj = q_diag[i], q_diag[j]
                    denom = np.sqrt(max(qi * qj, 1e-30))
                    c_ij = Q[i, j] / denom
                    c_ij = np.clip(c_ij, -1.0, 1.0)
                    K_val = self.kernels.kernel(qi, qj, c_ij,
                                               self.config.activation)
                    Q_new[i, j] = self._sw2 * K_val + self._sb2
                    Q_new[j, i] = Q_new[i, j]
            Q = Q_new
            trajectory.append(Q.copy())

        return trajectory

    def convergence_to_fixed_point(self, q0: float,
                                   depth: Optional[int] = None
                                   ) -> Tuple[np.ndarray, float]:
        """Track convergence of variance to its fixed point q*.

        Parameters
        ----------
        q0 : float
            Initial variance.
        depth : int, optional
            Number of layers.

        Returns
        -------
        residuals : np.ndarray, shape (depth,)
            |q^(l) - q*| at each layer.
        q_star : float
            Estimated fixed point (value at last layer).
        """
        q_traj = self.propagate_variance(q0, depth)
        q_star = q_traj[-1]
        residuals = np.abs(q_traj[:-1] - q_star)
        return residuals, q_star

    def correlation_map(self, c_in: float, q_star: float) -> float:
        """Evaluate the correlation map c ↦ f(c) at the fixed point q*.

        f(c) = (σ_w² · K(q*, q*, c) + σ_b²) / q_next

        where q_next = σ_w² · E[φ(√q* z)²] + σ_b² ≈ q* at a fixed point.

        Parameters
        ----------
        c_in : float
            Input correlation.
        q_star : float
            Variance fixed point.

        Returns
        -------
        float
            Output correlation f(c_in).
        """
        K_val = self.kernels.kernel(q_star, q_star, c_in,
                                    self.config.activation)
        q_next = self._variance_step(q_star)
        return np.clip((self._sw2 * K_val + self._sb2) / max(q_next, 1e-30),
                       -1.0, 1.0)


# ---------------------------------------------------------------------------
# BackwardPropagation
# ---------------------------------------------------------------------------

class BackwardPropagation:
    """Back-propagate gradient statistics through depth.

    In the mean-field limit, gradient norms satisfy:

        E[||∇_{h^(l)} L||²] = σ_w² · E[φ'(h)²] · E[||∇_{h^(l+1)} L||²]

    so the gradient variance at layer l is multiplied by χ₁ = σ_w² E[φ'²]
    at each layer.  χ₁ < 1 → vanishing gradients, χ₁ > 1 → exploding.
    """

    def __init__(self, config: PropagationConfig):
        """
        Parameters
        ----------
        config : PropagationConfig
            Propagation parameters.
        """
        self.config = config
        self.kernels = ActivationKernels()
        self._phi, self._phi_prime, _ = _ACTIVATIONS[config.activation]
        self._sw2 = config.sigma_w ** 2
        self._sb2 = config.sigma_b ** 2

    def _chi1(self, q: float) -> float:
        """Compute χ₁(q) = σ_w² · E[φ'(√q z)²]."""
        phi_p = self._phi_prime
        val = _gauss_expectation(
            lambda z: phi_p(np.sqrt(max(q, 0.0)) * z) ** 2
        )
        return self._sw2 * val

    def gradient_variance(self, g_L: float, q_trajectory: np.ndarray,
                          depth: Optional[int] = None) -> np.ndarray:
        """Back-propagate gradient variance from the loss layer.

        E[||g^(l)||²] = χ₁(q^(l)) · E[||g^(l+1)||²]

        Parameters
        ----------
        g_L : float
            Gradient variance at the final (loss) layer.
        q_trajectory : np.ndarray, shape (depth+1,)
            Forward variance trajectory (needed to evaluate χ₁ at each layer).
        depth : int, optional
            Number of layers to back-propagate through.

        Returns
        -------
        np.ndarray, shape (depth+1,)
            Gradient variance at each layer, from layer ``depth`` back to 0.
            Index 0 → deepest layer (loss), index depth → shallowest (input).
        """
        depth = depth or self.config.depth
        depth = min(depth, len(q_trajectory) - 1)
        g_traj = np.zeros(depth + 1)
        g_traj[0] = g_L
        for l in range(depth):
            layer_idx = len(q_trajectory) - 1 - l
            chi = self._chi1(q_trajectory[max(layer_idx - 1, 0)])
            g_traj[l + 1] = chi * g_traj[l]
        return g_traj

    def gradient_correlation(self, g_L: float, q_traj: np.ndarray,
                             c_traj: np.ndarray,
                             depth: Optional[int] = None) -> np.ndarray:
        """Back-propagate gradient correlation between two input copies.

        Uses the derivative kernel to track how gradient correlations
        evolve backward through the network.

        Parameters
        ----------
        g_L : float
            Gradient variance at loss layer.
        q_traj : np.ndarray
            Forward variance trajectory (one input).
        c_traj : np.ndarray
            Forward correlation trajectory.
        depth : int, optional
            Number of layers.

        Returns
        -------
        np.ndarray
            Gradient correlation at each layer (from deep to shallow).
        """
        depth = depth or self.config.depth
        depth = min(depth, min(len(q_traj), len(c_traj)) - 1)
        gc = np.zeros(depth + 1)
        gc[0] = g_L
        for l in range(depth):
            idx = len(q_traj) - 1 - l
            q = q_traj[max(idx - 1, 0)]
            c = c_traj[max(idx - 1, 0)]
            dk = self.kernels.derivative_kernel(q, q, c,
                                                self.config.activation)
            gc[l + 1] = self._sw2 * dk * gc[l]
        return gc

    def jacobian_norm(self, q_traj: np.ndarray,
                      depth: Optional[int] = None) -> np.ndarray:
        """Compute ||J^(l)||² = ∏_{k=l}^{L-1} χ₁(q^(k)) through layers.

        Parameters
        ----------
        q_traj : np.ndarray, shape (depth+1,)
            Forward variance trajectory.
        depth : int, optional
            Number of layers.

        Returns
        -------
        np.ndarray, shape (depth+1,)
            Jacobian squared norm at each layer.  Index 0 = layer L (=1),
            index l = product of χ₁ from layer L-1 down to L-l.
        """
        depth = depth or self.config.depth
        depth = min(depth, len(q_traj) - 1)
        jac = np.zeros(depth + 1)
        jac[0] = 1.0
        for l in range(depth):
            idx = len(q_traj) - 1 - l
            chi = self._chi1(q_traj[max(idx - 1, 0)])
            jac[l + 1] = jac[l] * chi
        return jac

    def gradient_explosion_depth(self, q_star: float,
                                 threshold: float = 1e6) -> Optional[int]:
        """Estimate the depth at which gradients exceed ``threshold``.

        Under χ₁ > 1 the gradient variance grows as χ₁^l.  Returns the
        layer depth l such that χ₁^l ≥ threshold.

        Parameters
        ----------
        q_star : float
            Variance fixed point.
        threshold : float
            Explosion threshold.

        Returns
        -------
        int or None
            Depth at which explosion occurs, or None if χ₁ ≤ 1.
        """
        chi = self._chi1(q_star)
        if chi <= 1.0:
            return None
        return int(np.ceil(np.log(threshold) / np.log(chi)))

    def gradient_vanishing_depth(self, q_star: float,
                                 threshold: float = 1e-6) -> Optional[int]:
        """Estimate the depth at which gradients fall below ``threshold``.

        Under χ₁ < 1 the gradient variance decays as χ₁^l.

        Parameters
        ----------
        q_star : float
            Variance fixed point.
        threshold : float
            Vanishing threshold.

        Returns
        -------
        int or None
            Depth at which vanishing occurs, or None if χ₁ ≥ 1.
        """
        chi = self._chi1(q_star)
        if chi >= 1.0:
            return None
        if chi <= 0.0:
            return 1
        return int(np.ceil(np.log(threshold) / np.log(chi)))


# ---------------------------------------------------------------------------
# FixedPointAnalyzer
# ---------------------------------------------------------------------------

class FixedPointAnalyzer:
    """Analyze fixed points of the mean-field variance and correlation maps.

    The variance fixed point q* satisfies:

        q* = σ_w² · E[φ(√q* z)²] + σ_b²

    and its stability is governed by:

        χ₁ = σ_w² · E[φ'(√q* z)²]

    χ₁ < 1 ⇒ stable (ordered), χ₁ > 1 ⇒ unstable (chaotic).
    """

    def __init__(self, config: PropagationConfig):
        """
        Parameters
        ----------
        config : PropagationConfig
            Propagation parameters.
        """
        self.config = config
        self.kernels = ActivationKernels()
        self._phi, self._phi_prime, _ = _ACTIVATIONS[config.activation]
        self._sw2 = config.sigma_w ** 2
        self._sb2 = config.sigma_b ** 2

    def _variance_map(self, q: float) -> float:
        """f(q) = σ_w² E[φ(√q z)²] + σ_b²."""
        phi = self._phi
        val = _gauss_expectation(lambda z: phi(np.sqrt(max(q, 1e-30)) * z) ** 2)
        return self._sw2 * val + self._sb2

    def _variance_residual(self, q: float) -> float:
        """f(q) - q, root gives the fixed point."""
        return self._variance_map(q) - q

    def find_variance_fixed_point(self, q_init: float = 1.0) -> float:
        """Find the variance fixed point q* by solving f(q) = q.

        Uses Brent's method on the residual f(q) - q = 0 on a bracketed
        interval, falling back to fixed-point iteration.

        Parameters
        ----------
        q_init : float
            Starting guess for the fixed point.

        Returns
        -------
        float
            The fixed point q*.
        """
        # Try fixed-point iteration first (robust for monotone maps)
        q = q_init
        for _ in range(2000):
            q_new = self._variance_map(q)
            if abs(q_new - q) < 1e-12:
                return q_new
            q = q_new

        # Fall back to root finding
        try:
            sol = optimize.brentq(self._variance_residual, 1e-8, 1e4)
            return sol
        except ValueError:
            warnings.warn("Fixed-point search did not converge; "
                          "returning last iterate.")
            return q

    def find_correlation_fixed_points(self, q_star: float
                                      ) -> List[float]:
        """Find all fixed points c* of the correlation map at q*.

        The correlation map is:

            f(c) = [σ_w² K(q*, q*, c) + σ_b²] / q*_next

        We search for roots of f(c) - c = 0 on [-1, 1].

        Parameters
        ----------
        q_star : float
            Variance fixed point.

        Returns
        -------
        list of float
            All correlation fixed points found in [-1, 1].
        """
        fwd = ForwardPropagation(self.config)

        def residual(c):
            return fwd.correlation_map(c, q_star) - c

        # Scan for sign changes
        c_grid = np.linspace(-0.999, 0.999, 500)
        r_vals = np.array([residual(c) for c in c_grid])
        roots = []

        # Find sign changes
        for i in range(len(r_vals) - 1):
            if r_vals[i] * r_vals[i + 1] < 0:
                try:
                    root = optimize.brentq(residual, c_grid[i], c_grid[i + 1])
                    roots.append(float(root))
                except ValueError:
                    pass

        # Check if c=1 is always a fixed point (it typically is)
        try:
            r_at_1 = residual(0.9999)
            if abs(r_at_1) < 1e-4:
                roots.append(1.0)
        except Exception:
            pass

        # Deduplicate
        if not roots:
            return [1.0]
        unique = [roots[0]]
        for r in roots[1:]:
            if all(abs(r - u) > 1e-6 for u in unique):
                unique.append(r)
        return sorted(unique)

    def fixed_point_stability(self, q_star: float) -> float:
        """Compute χ₁ = σ_w² · E[φ'(√q* z)²] at the variance fixed point.

        χ₁ is the derivative of the variance map at q* and determines
        stability:
        - χ₁ < 1 → ordered phase (correlations converge to 1)
        - χ₁ > 1 → chaotic phase (correlations converge to c* < 1)
        - χ₁ = 1 → edge of chaos (critical initialization)

        Parameters
        ----------
        q_star : float
            Variance fixed point.

        Returns
        -------
        float
            χ₁ value.
        """
        phi_p = self._phi_prime
        val = _gauss_expectation(
            lambda z: phi_p(np.sqrt(max(q_star, 1e-30)) * z) ** 2
        )
        return self._sw2 * val

    def correlation_fixed_point_stability(self, c_star: float,
                                          q_star: float) -> float:
        """Compute ∂f/∂c at the correlation fixed point c*.

        This is the slope of the correlation map at the fixed point.
        |∂f/∂c| < 1 → stable fixed point, > 1 → unstable.

        Parameters
        ----------
        c_star : float
            Correlation fixed point.
        q_star : float
            Variance fixed point.

        Returns
        -------
        float
            Derivative ∂f(c)/∂c evaluated at c = c*.
        """
        fwd = ForwardPropagation(self.config)
        dc = 1e-6
        c_plus = min(c_star + dc, 0.99999)
        c_minus = max(c_star - dc, -0.99999)
        f_plus = fwd.correlation_map(c_plus, q_star)
        f_minus = fwd.correlation_map(c_minus, q_star)
        return (f_plus - f_minus) / (c_plus - c_minus)

    def bifurcation_analysis(self, sigma_w_range: np.ndarray
                             ) -> Dict[str, np.ndarray]:
        """Track fixed points and stability as σ_w varies.

        For each σ_w value, find q*, c*, and χ₁.  This traces out the
        bifurcation diagram.

        Parameters
        ----------
        sigma_w_range : np.ndarray
            Array of σ_w values to scan.

        Returns
        -------
        dict
            Keys: 'sigma_w', 'q_star', 'chi1', 'c_star_nontrivial'.
            Each is an array of the same length as sigma_w_range.
        """
        n = len(sigma_w_range)
        q_stars = np.zeros(n)
        chi1s = np.zeros(n)
        c_stars = np.zeros(n)

        original_sw = self.config.sigma_w
        for i, sw in enumerate(sigma_w_range):
            self.config.sigma_w = sw
            self._sw2 = sw ** 2
            q_star = self.find_variance_fixed_point()
            q_stars[i] = q_star
            chi1s[i] = self.fixed_point_stability(q_star)
            c_fps = self.find_correlation_fixed_points(q_star)
            # Take the non-trivial (c* < 1) fixed point if it exists
            non_trivial = [c for c in c_fps if c < 0.999]
            c_stars[i] = non_trivial[0] if non_trivial else 1.0

        self.config.sigma_w = original_sw
        self._sw2 = original_sw ** 2

        return {
            "sigma_w": sigma_w_range,
            "q_star": q_stars,
            "chi1": chi1s,
            "c_star_nontrivial": c_stars,
        }

    def number_of_fixed_points(self, sigma_w: float,
                               sigma_b: float) -> int:
        """Count the number of variance fixed points for given (σ_w, σ_b).

        Parameters
        ----------
        sigma_w : float
            Weight initialization scale.
        sigma_b : float
            Bias initialization scale.

        Returns
        -------
        int
            Number of fixed points of the variance map.
        """
        original_sw, original_sb = self.config.sigma_w, self.config.sigma_b
        self.config.sigma_w = sigma_w
        self.config.sigma_b = sigma_b
        self._sw2 = sigma_w ** 2
        self._sb2 = sigma_b ** 2

        q_grid = np.logspace(-4, 3, 1000)
        residuals = np.array([self._variance_residual(q) for q in q_grid])
        sign_changes = np.sum(np.diff(np.sign(residuals)) != 0)

        self.config.sigma_w = original_sw
        self.config.sigma_b = original_sb
        self._sw2 = original_sw ** 2
        self._sb2 = original_sb ** 2

        return max(int(sign_changes), 0)


# ---------------------------------------------------------------------------
# CriticalInitialization
# ---------------------------------------------------------------------------

class CriticalInitialization:
    """Find the critical initialization (edge of chaos) for a given activation.

    At criticality, χ₁ = σ_w² E[φ'(√q* z)²] = 1, which means gradients
    neither explode nor vanish and the correlation length ξ diverges.

    This class finds the critical line in (σ_w, σ_b) space and computes
    associated length scales.
    """

    def __init__(self, activation: str = "relu"):
        """
        Parameters
        ----------
        activation : str
            Activation function name.
        """
        self.activation = activation
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'.")

    def _make_config(self, sigma_w: float, sigma_b: float
                     ) -> PropagationConfig:
        return PropagationConfig(
            activation=self.activation,
            sigma_w=sigma_w,
            sigma_b=sigma_b,
            depth=10,
        )

    def find_critical_sigma_w(self, sigma_b: float = 0.0) -> float:
        """Find σ_w* such that χ₁ = 1 for given σ_b.

        Parameters
        ----------
        sigma_b : float
            Bias initialization scale.

        Returns
        -------
        float
            Critical weight scale σ_w*.
        """
        def chi1_minus_one(sw):
            if sw <= 0:
                return -1.0
            cfg = self._make_config(sw, sigma_b)
            analyzer = FixedPointAnalyzer(cfg)
            q_star = analyzer.find_variance_fixed_point()
            chi1 = analyzer.fixed_point_stability(q_star)
            return chi1 - 1.0

        # Bracket: at small σ_w, χ₁ < 1; at large σ_w, χ₁ > 1
        try:
            return optimize.brentq(chi1_minus_one, 0.1, 10.0, xtol=1e-8)
        except ValueError:
            # Expand bracket
            try:
                return optimize.brentq(chi1_minus_one, 0.01, 50.0, xtol=1e-8)
            except ValueError:
                warnings.warn("Could not bracket critical σ_w.")
                return float("nan")

    def find_critical_line(self, sigma_b_range: np.ndarray
                           ) -> np.ndarray:
        """Compute the full critical line σ_w*(σ_b) in (σ_w, σ_b) space.

        Parameters
        ----------
        sigma_b_range : np.ndarray
            Array of σ_b values.

        Returns
        -------
        np.ndarray
            Corresponding critical σ_w* values.
        """
        sw_critical = np.zeros_like(sigma_b_range)
        for i, sb in enumerate(sigma_b_range):
            sw_critical[i] = self.find_critical_sigma_w(sb)
        return sw_critical

    def correlation_length(self, sigma_w: float,
                           sigma_b: float) -> float:
        """Compute the correlation length ξ = -1 / ln|χ₁|.

        Near the critical line, ξ diverges as |σ_w - σ_w*|^{-1/2} for
        mean-field universality.

        Parameters
        ----------
        sigma_w, sigma_b : float
            Initialization parameters.

        Returns
        -------
        float
            Correlation length ξ (in units of layers).
        """
        cfg = self._make_config(sigma_w, sigma_b)
        analyzer = FixedPointAnalyzer(cfg)
        q_star = analyzer.find_variance_fixed_point()
        chi1 = analyzer.fixed_point_stability(q_star)
        if abs(chi1) < 1e-15:
            return 0.0
        log_chi = np.log(abs(chi1))
        if abs(log_chi) < 1e-15:
            return float("inf")
        return -1.0 / log_chi

    def depth_scale_at_criticality(self, sigma_w: float,
                                   sigma_b: float) -> float:
        """Compute the depth scale near criticality.

        At criticality, the effective depth scale is governed by higher-order
        corrections.  We estimate it from the second derivative of the
        correlation map.

        Parameters
        ----------
        sigma_w, sigma_b : float
            Initialization parameters.

        Returns
        -------
        float
            Depth scale (layers).
        """
        xi = self.correlation_length(sigma_w, sigma_b)
        if np.isinf(xi):
            # At exact criticality, need curvature correction
            cfg = self._make_config(sigma_w, sigma_b)
            analyzer = FixedPointAnalyzer(cfg)
            q_star = analyzer.find_variance_fixed_point()
            # Compute curvature of correlation map at c=1
            fwd = ForwardPropagation(cfg)
            dc = 1e-4
            f_p = fwd.correlation_map(1.0 - dc, q_star)
            f_m = fwd.correlation_map(1.0 - 2 * dc, q_star)
            f_0 = fwd.correlation_map(1.0, q_star)
            # Second derivative ≈ (f(c+dc) - 2f(c) + f(c-dc)) / dc²
            # evaluated near c = 1
            curvature = abs((f_0 - 2 * f_p + f_m) / dc ** 2)
            if curvature < 1e-15:
                return float("inf")
            return 1.0 / np.sqrt(curvature)
        return xi

    def optimal_init_for_depth(self, target_depth: int,
                               sigma_b: float = 0.0) -> float:
        """Find the σ_w value that supports signal propagation for a given depth.

        We want ξ ≈ target_depth, which means χ₁ = exp(-1/target_depth).

        Parameters
        ----------
        target_depth : int
            Target network depth.
        sigma_b : float
            Bias initialization scale.

        Returns
        -------
        float
            Recommended σ_w.
        """
        target_chi1 = np.exp(-1.0 / max(target_depth, 1))

        def obj(sw):
            if sw <= 0:
                return 10.0
            cfg = self._make_config(sw, sigma_b)
            analyzer = FixedPointAnalyzer(cfg)
            q_star = analyzer.find_variance_fixed_point()
            chi1 = analyzer.fixed_point_stability(q_star)
            return chi1 - target_chi1

        try:
            return optimize.brentq(obj, 0.1, 10.0, xtol=1e-8)
        except ValueError:
            # Fall back to critical value
            return self.find_critical_sigma_w(sigma_b)


# ---------------------------------------------------------------------------
# DepthPhaseAnalyzer
# ---------------------------------------------------------------------------

class DepthPhaseAnalyzer:
    """Analyze the ordered/chaotic phase structure as a function of depth.

    The phase is determined by χ₁ = σ_w² E[φ'(h)²]:
    - Ordered (χ₁ < 1): all inputs map to similar representations at depth.
    - Chaotic (χ₁ > 1): nearby inputs diverge exponentially.
    - Edge of chaos (χ₁ = 1): critical initialization.

    The Lyapunov exponent λ = ln(χ₁) quantifies the rate of divergence or
    convergence of nearby trajectories in function space.
    """

    def __init__(self, config: PropagationConfig):
        """
        Parameters
        ----------
        config : PropagationConfig
            Propagation parameters.
        """
        self.config = config
        self.kernels = ActivationKernels()
        self._phi, self._phi_prime, _ = _ACTIVATIONS[config.activation]
        self._sw2 = config.sigma_w ** 2
        self._sb2 = config.sigma_b ** 2

    def _get_chi1(self, sigma_w: float, sigma_b: float) -> float:
        """Compute χ₁ for given (σ_w, σ_b)."""
        cfg = PropagationConfig(
            activation=self.config.activation,
            sigma_w=sigma_w,
            sigma_b=sigma_b,
            depth=10,
        )
        analyzer = FixedPointAnalyzer(cfg)
        q_star = analyzer.find_variance_fixed_point()
        return analyzer.fixed_point_stability(q_star)

    def ordered_phase_check(self, sigma_w: float,
                            sigma_b: float) -> Tuple[bool, float]:
        """Check if (σ_w, σ_b) is in the ordered phase (χ₁ < 1).

        In the ordered phase, correlations between any two inputs converge
        to c* = 1 (complete loss of input information).

        Parameters
        ----------
        sigma_w, sigma_b : float
            Initialization parameters.

        Returns
        -------
        is_ordered : bool
            True if in the ordered phase.
        chi1 : float
            Value of χ₁.
        """
        chi1 = self._get_chi1(sigma_w, sigma_b)
        return chi1 < 1.0, chi1

    def chaotic_phase_check(self, sigma_w: float,
                            sigma_b: float) -> Tuple[bool, float]:
        """Check if (σ_w, σ_b) is in the chaotic phase (χ₁ > 1).

        In the chaotic phase, nearby inputs diverge and gradients explode.

        Parameters
        ----------
        sigma_w, sigma_b : float
            Initialization parameters.

        Returns
        -------
        is_chaotic : bool
            True if in the chaotic phase.
        chi1 : float
            Value of χ₁.
        """
        chi1 = self._get_chi1(sigma_w, sigma_b)
        return chi1 > 1.0, chi1

    def phase_map(self, sigma_w_range: np.ndarray,
                  sigma_b_range: np.ndarray) -> np.ndarray:
        """Classify each point in the (σ_w, σ_b) grid.

        Returns a 2-D array with values:
        - 0: ordered (χ₁ < 1)
        - 1: critical (|χ₁ - 1| < tolerance)
        - 2: chaotic (χ₁ > 1)

        Parameters
        ----------
        sigma_w_range : np.ndarray
            σ_w values (columns).
        sigma_b_range : np.ndarray
            σ_b values (rows).

        Returns
        -------
        np.ndarray, shape (len(sigma_b_range), len(sigma_w_range))
            Phase labels.
        """
        nw = len(sigma_w_range)
        nb = len(sigma_b_range)
        phases = np.zeros((nb, nw), dtype=int)
        tol = 0.01

        for i, sb in enumerate(sigma_b_range):
            for j, sw in enumerate(sigma_w_range):
                chi1 = self._get_chi1(sw, sb)
                if abs(chi1 - 1.0) < tol:
                    phases[i, j] = 1
                elif chi1 > 1.0:
                    phases[i, j] = 2
                else:
                    phases[i, j] = 0

        return phases

    def lyapunov_exponent_map(self, sigma_w_range: np.ndarray,
                              sigma_b_range: np.ndarray) -> np.ndarray:
        """Compute the Lyapunov exponent λ = ln(χ₁) over the grid.

        λ < 0 → ordered, λ > 0 → chaotic, λ = 0 → critical.

        Parameters
        ----------
        sigma_w_range : np.ndarray
            σ_w values.
        sigma_b_range : np.ndarray
            σ_b values.

        Returns
        -------
        np.ndarray, shape (len(sigma_b_range), len(sigma_w_range))
            Lyapunov exponent map.
        """
        nw = len(sigma_w_range)
        nb = len(sigma_b_range)
        lyap = np.zeros((nb, nw))

        for i, sb in enumerate(sigma_b_range):
            for j, sw in enumerate(sigma_w_range):
                chi1 = self._get_chi1(sw, sb)
                lyap[i, j] = np.log(max(chi1, 1e-30))

        return lyap

    def information_propagation_depth(self, sigma_w: float,
                                      sigma_b: float,
                                      epsilon: float = 0.01) -> int:
        """Estimate the maximum depth for information propagation.

        In the ordered phase, information about the input is lost at rate
        ξ = -1/ln(χ₁) layers.  We report the depth at which the correlation
        between two distinct inputs approaches 1 - ε.

        In the chaotic phase, nearby inputs diverge; we report the depth at
        which gradients exceed a practical threshold.

        Parameters
        ----------
        sigma_w, sigma_b : float
            Initialization parameters.
        epsilon : float
            Tolerance for correlation convergence (ordered phase) or
            gradient explosion (chaotic phase).

        Returns
        -------
        int
            Maximum useful depth for information propagation.
        """
        chi1 = self._get_chi1(sigma_w, sigma_b)

        if abs(chi1 - 1.0) < 1e-12:
            # At criticality, depth scale is very large
            return 10000

        if chi1 < 1.0:
            # Ordered: correlation length ξ = -1/ln(χ₁)
            xi = -1.0 / np.log(chi1)
            # Depth where (1 - c) ~ epsilon
            return max(int(xi * np.log(1.0 / epsilon)), 1)
        else:
            # Chaotic: gradients explode at rate χ₁^l
            # Depth where χ₁^l = 1/epsilon
            return max(int(np.log(1.0 / epsilon) / np.log(chi1)), 1)
