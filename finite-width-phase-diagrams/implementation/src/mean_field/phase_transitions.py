"""
Phase transition analysis for neural networks.

Implements detection and characterization of phase transitions in the
initialization parameter space (σ_w, σ_b) for deep neural networks,
building on the mean-field theory of signal propagation.

Key phase transitions analyzed:

1. **Order–disorder transition**: The variance fixed point q* undergoes a
   bifurcation as σ_w crosses a critical value.  For some activations this
   is a pitchfork bifurcation (second-order transition); for others the
   fixed point can jump discontinuously (first-order).

2. **Edge-of-chaos transition**: The correlation map fixed point c* goes
   from c* = 1 (ordered) to c* < 1 (chaotic) as χ₁ crosses 1.  This is a
   continuous transition with diverging correlation length ξ ~ |σ - σ_c|^{-ν}.

3. **Information propagation boundary**: The depth at which input
   information is preserved transitions from finite (ordered/chaotic) to
   infinite (critical) at the edge of chaos.

References:
- Poole et al. (2016) "Exponential expressivity in deep neural networks
  through transient chaos"
- Schoenholz et al. (2017) "Deep Information Propagation"
- Hayou et al. (2019) "On the Impact of the Activation Function on Deep
  Neural Networks Training"
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import numpy as np
from scipy import optimize, special
import warnings

from .signal_propagation import (
    PropagationConfig,
    ActivationKernels,
    ForwardPropagation,
    BackwardPropagation,
    FixedPointAnalyzer,
    CriticalInitialization,
    DepthPhaseAnalyzer,
    _ACTIVATIONS,
    _gauss_expectation,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PhaseTransitionConfig:
    """Configuration for phase transition analysis.

    Parameters
    ----------
    activation : str
        Activation function name: 'relu', 'tanh', 'gelu', or 'erf'.
    sigma_w_range : tuple of (float, float)
        Range of σ_w values to scan, as (min, max).
    sigma_b_range : tuple of (float, float)
        Range of σ_b values to scan, as (min, max).
    grid_resolution : int
        Number of grid points along each axis.
    depth : int
        Network depth for depth-dependent analyses.
    """
    activation: str = "relu"
    sigma_w_range: Tuple[float, float] = (0.5, 3.0)
    sigma_b_range: Tuple[float, float] = (0.0, 1.5)
    grid_resolution: int = 50
    depth: int = 100

    def __post_init__(self):
        if self.activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{self.activation}'. "
                f"Choose from {list(_ACTIVATIONS.keys())}."
            )

    @property
    def sigma_w_grid(self) -> np.ndarray:
        """1-D array of σ_w values."""
        return np.linspace(self.sigma_w_range[0], self.sigma_w_range[1],
                           self.grid_resolution)

    @property
    def sigma_b_grid(self) -> np.ndarray:
        """1-D array of σ_b values."""
        return np.linspace(self.sigma_b_range[0], self.sigma_b_range[1],
                           self.grid_resolution)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _prop_config(activation: str, sigma_w: float, sigma_b: float,
                 depth: int = 10) -> PropagationConfig:
    """Create a PropagationConfig for point analyses."""
    return PropagationConfig(
        activation=activation,
        sigma_w=sigma_w,
        sigma_b=sigma_b,
        depth=depth,
    )


def _get_q_star(activation: str, sigma_w: float, sigma_b: float) -> float:
    """Convenience: find variance fixed point for (activation, σ_w, σ_b)."""
    cfg = _prop_config(activation, sigma_w, sigma_b)
    return FixedPointAnalyzer(cfg).find_variance_fixed_point()


def _get_chi1(activation: str, sigma_w: float, sigma_b: float) -> float:
    """Convenience: compute χ₁ for (activation, σ_w, σ_b)."""
    cfg = _prop_config(activation, sigma_w, sigma_b)
    analyzer = FixedPointAnalyzer(cfg)
    q_star = analyzer.find_variance_fixed_point()
    return analyzer.fixed_point_stability(q_star)


# ---------------------------------------------------------------------------
# OrderDisorderTransition
# ---------------------------------------------------------------------------

class OrderDisorderTransition:
    """Analyze the order–disorder transition in the variance fixed point.

    The order parameter is q* - q_trivial, where q_trivial is the trivial
    (q = 0 or minimal) fixed point.  As σ_w crosses the critical value,
    q* either jumps (first-order) or grows continuously from zero
    (second-order, with q* ~ |σ - σ_c|^β).

    For ReLU networks the transition is continuous with β = 1 (linear onset).
    For bounded activations (tanh, erf) the transition can be first-order
    for large σ_b.
    """

    def __init__(self, config: PhaseTransitionConfig):
        """
        Parameters
        ----------
        config : PhaseTransitionConfig
            Phase transition analysis configuration.
        """
        self.config = config

    def compute_order_parameter(self, sigma_w: float,
                                sigma_b: float) -> float:
        """Compute the order parameter Δq = q* - q_trivial.

        The trivial fixed point is q = σ_b² (variance in the limit σ_w → 0).
        The order parameter measures how much the non-trivial fixed point
        departs from trivial.

        Parameters
        ----------
        sigma_w, sigma_b : float
            Initialization parameters.

        Returns
        -------
        float
            Order parameter q* - σ_b².
        """
        q_star = _get_q_star(self.config.activation, sigma_w, sigma_b)
        q_trivial = sigma_b ** 2
        return max(q_star - q_trivial, 0.0)

    def order_parameter_map(self) -> np.ndarray:
        """Compute the order parameter over the full (σ_w, σ_b) grid.

        Returns
        -------
        np.ndarray, shape (grid_resolution, grid_resolution)
            Order parameter at each grid point.
            Rows index σ_b, columns index σ_w.
        """
        sw_grid = self.config.sigma_w_grid
        sb_grid = self.config.sigma_b_grid
        result = np.zeros((len(sb_grid), len(sw_grid)))

        for i, sb in enumerate(sb_grid):
            for j, sw in enumerate(sw_grid):
                result[i, j] = self.compute_order_parameter(sw, sb)

        return result

    def find_transition_line(self) -> Tuple[np.ndarray, np.ndarray]:
        """Find the order–disorder transition line in (σ_w, σ_b) space.

        We define the transition as the contour where the order parameter
        first becomes non-zero (or jumps).

        Returns
        -------
        sigma_w_line, sigma_b_line : np.ndarray
            Points on the transition line.
        """
        sb_grid = self.config.sigma_b_grid
        sw_grid = self.config.sigma_w_grid
        sw_transition = []
        sb_transition = []

        for sb in sb_grid:
            # Find the σ_w where the order parameter first exceeds a threshold
            threshold = 0.01
            order_params = np.array([
                self.compute_order_parameter(sw, sb) for sw in sw_grid
            ])
            # Find first crossing
            crossings = np.where(order_params > threshold)[0]
            if len(crossings) > 0:
                idx = crossings[0]
                if idx > 0:
                    # Interpolate
                    sw_c = np.interp(
                        threshold,
                        [order_params[idx - 1], order_params[idx]],
                        [sw_grid[idx - 1], sw_grid[idx]],
                    )
                else:
                    sw_c = sw_grid[0]
                sw_transition.append(sw_c)
                sb_transition.append(sb)

        return np.array(sw_transition), np.array(sb_transition)

    def classify_transition_order(self, sigma_w: float,
                                  sigma_b: float) -> str:
        """Classify whether the transition is first-order, second-order, or crossover.

        Strategy: scan the order parameter as σ_w increases through the
        transition at fixed σ_b.  If it jumps discontinuously, first-order;
        if it grows continuously from zero, second-order; if there is no
        sharp feature, crossover.

        Parameters
        ----------
        sigma_w, sigma_b : float
            Point near the transition.

        Returns
        -------
        str
            One of 'first_order', 'second_order', 'crossover'.
        """
        sw_range = np.linspace(max(0.1, sigma_w - 0.5),
                               sigma_w + 0.5, 200)
        ops = np.array([
            self.compute_order_parameter(sw, sigma_b) for sw in sw_range
        ])

        # Compute gradient of order parameter
        d_op = np.gradient(ops, sw_range)
        max_gradient = np.max(np.abs(d_op))

        # Large jump → first order
        if max_gradient > 10.0:
            return "first_order"

        # Smooth onset from zero → second order
        zero_crossings = np.where(
            (ops[:-1] < 0.01) & (ops[1:] >= 0.01)
        )[0]
        if len(zero_crossings) > 0:
            return "second_order"

        return "crossover"

    def critical_exponent_beta(self, sigma_w_c: float,
                               sigma_b_c: float) -> float:
        """Estimate the critical exponent β for the order parameter.

        Near the transition, q* ~ |σ_w - σ_w_c|^β.  We fit the exponent
        from the slope on a log-log plot.

        Parameters
        ----------
        sigma_w_c, sigma_b_c : float
            Critical point.

        Returns
        -------
        float
            Estimated exponent β.
        """
        # Sample above the critical point
        deltas = np.logspace(-3, -0.5, 30)
        sw_vals = sigma_w_c + deltas
        ops = np.array([
            self.compute_order_parameter(sw, sigma_b_c) for sw in sw_vals
        ])

        # Filter out zero / near-zero values
        mask = ops > 1e-10
        if np.sum(mask) < 5:
            return float("nan")

        log_delta = np.log(deltas[mask])
        log_op = np.log(ops[mask])

        # Linear fit: log(op) = β * log(delta) + const
        coeffs = np.polyfit(log_delta, log_op, 1)
        return coeffs[0]

    def susceptibility_divergence(self, sigma_w_c: float,
                                  sigma_b_c: float) -> float:
        """Estimate the susceptibility exponent γ.

        The susceptibility χ = ∂q*/∂σ_w diverges as |σ - σ_c|^{-γ}
        near the critical point.

        Parameters
        ----------
        sigma_w_c, sigma_b_c : float
            Critical point.

        Returns
        -------
        float
            Estimated exponent γ.
        """
        deltas = np.logspace(-3, -0.5, 30)

        # Compute susceptibility via finite differences
        dsw = 1e-5
        susceptibilities = []
        for delta in deltas:
            sw = sigma_w_c + delta
            q_p = _get_q_star(self.config.activation, sw + dsw, sigma_b_c)
            q_m = _get_q_star(self.config.activation, sw - dsw, sigma_b_c)
            chi = abs((q_p - q_m) / (2 * dsw))
            susceptibilities.append(chi)

        susceptibilities = np.array(susceptibilities)
        mask = susceptibilities > 1e-10
        if np.sum(mask) < 5:
            return float("nan")

        log_delta = np.log(deltas[mask])
        log_chi = np.log(susceptibilities[mask])

        coeffs = np.polyfit(log_delta, log_chi, 1)
        # χ ~ delta^{-γ}, so slope = -γ
        return -coeffs[0]


# ---------------------------------------------------------------------------
# ChaosTransition
# ---------------------------------------------------------------------------

class ChaosTransition:
    """Analyze the edge-of-chaos transition.

    The transition occurs where χ₁ = σ_w² E[φ'(h)²] crosses 1.
    The Lyapunov exponent λ = ln(χ₁) changes sign, and the correlation
    length ξ = -1/ln(χ₁) diverges.

    In the chaotic phase (λ > 0), nearby inputs are mapped to increasingly
    distant representations — the "butterfly effect" in function space.
    """

    def __init__(self, config: PhaseTransitionConfig):
        """
        Parameters
        ----------
        config : PhaseTransitionConfig
            Phase transition analysis configuration.
        """
        self.config = config

    def lyapunov_exponent(self, sigma_w: float,
                          sigma_b: float) -> float:
        """Compute the Lyapunov exponent λ = ln(χ₁).

        λ < 0 → ordered (convergent), λ > 0 → chaotic (divergent).

        Parameters
        ----------
        sigma_w, sigma_b : float
            Initialization parameters.

        Returns
        -------
        float
            Lyapunov exponent.
        """
        chi1 = _get_chi1(self.config.activation, sigma_w, sigma_b)
        return np.log(max(chi1, 1e-30))

    def lyapunov_map(self) -> np.ndarray:
        """Compute the Lyapunov exponent λ = ln(χ₁) over the grid.

        Returns
        -------
        np.ndarray, shape (grid_resolution, grid_resolution)
            Lyapunov exponent map.  Rows index σ_b, columns index σ_w.
        """
        sw_grid = self.config.sigma_w_grid
        sb_grid = self.config.sigma_b_grid
        result = np.zeros((len(sb_grid), len(sw_grid)))

        for i, sb in enumerate(sb_grid):
            for j, sw in enumerate(sw_grid):
                result[i, j] = self.lyapunov_exponent(sw, sb)

        return result

    def chaos_boundary(self) -> Tuple[np.ndarray, np.ndarray]:
        """Find the edge-of-chaos boundary (λ = 0 contour).

        This is equivalent to the critical line σ_w*(σ_b) where χ₁ = 1.

        Returns
        -------
        sigma_w_boundary, sigma_b_boundary : np.ndarray
            Points on the chaos boundary.
        """
        ci = CriticalInitialization(self.config.activation)
        sb_grid = self.config.sigma_b_grid
        sw_boundary = ci.find_critical_line(sb_grid)

        # Filter out NaN values
        mask = ~np.isnan(sw_boundary)
        return sw_boundary[mask], sb_grid[mask]

    def correlation_decay_rate(self, sigma_w: float, sigma_b: float,
                               depth: int = 100) -> np.ndarray:
        """Measure how fast correlations between two inputs decay with depth.

        Starting from c^(0) = 0.99 (nearly identical inputs), propagate
        forward and return 1 - c^(l) as a function of depth.

        Parameters
        ----------
        sigma_w, sigma_b : float
            Initialization parameters.
        depth : int
            Number of layers.

        Returns
        -------
        np.ndarray, shape (depth+1,)
            1 - c^(l) at each layer.
        """
        cfg = _prop_config(self.config.activation, sigma_w, sigma_b, depth)
        fwd = ForwardPropagation(cfg)
        q_star = _get_q_star(self.config.activation, sigma_w, sigma_b)
        _, _, c_traj = fwd.propagate_correlation(q_star, q_star, 0.99, depth)
        return 1.0 - c_traj

    def butterfly_effect_distance(self, sigma_w: float, sigma_b: float,
                                  epsilon: float = 1e-4,
                                  depth: int = 100) -> np.ndarray:
        """Track the growth of perturbations through depth.

        Start with two inputs at correlation c^(0) = 1 - ε and track how
        the effective distance d^(l) = √(2q*(1 - c^(l))) grows.

        In the chaotic phase, d^(l) grows exponentially with rate λ.

        Parameters
        ----------
        sigma_w, sigma_b : float
            Initialization parameters.
        epsilon : float
            Initial perturbation size (1 - c₀).
        depth : int
            Number of layers.

        Returns
        -------
        np.ndarray, shape (depth+1,)
            Effective distance at each layer.
        """
        c0 = 1.0 - epsilon
        cfg = _prop_config(self.config.activation, sigma_w, sigma_b, depth)
        fwd = ForwardPropagation(cfg)
        q_star = _get_q_star(self.config.activation, sigma_w, sigma_b)
        _, _, c_traj = fwd.propagate_correlation(q_star, q_star, c0, depth)
        return np.sqrt(2.0 * max(q_star, 0.0) * (1.0 - c_traj))

    def transition_width(self, sigma_b: float) -> float:
        """Estimate the width of the transition region at fixed σ_b.

        The transition width is defined as the range of σ_w over which
        the Lyapunov exponent goes from -0.1 to +0.1 (i.e. the region
        where the system is near-critical).

        Parameters
        ----------
        sigma_b : float
            Bias initialization scale.

        Returns
        -------
        float
            Width of the transition region in σ_w.
        """
        sw_grid = self.config.sigma_w_grid
        lyap_vals = np.array([
            self.lyapunov_exponent(sw, sigma_b) for sw in sw_grid
        ])

        # Find where λ crosses -0.1 and +0.1
        threshold = 0.1
        below = np.where(lyap_vals < -threshold)[0]
        above = np.where(lyap_vals > threshold)[0]

        if len(below) == 0 or len(above) == 0:
            return float("nan")

        sw_low = sw_grid[below[-1]]
        sw_high = sw_grid[above[0]]
        return sw_high - sw_low


# ---------------------------------------------------------------------------
# InformationPropagation
# ---------------------------------------------------------------------------

class InformationPropagation:
    """Analyze information propagation through deep networks.

    In the ordered phase, mutual information between input and deep
    representations vanishes exponentially.  In the chaotic phase,
    information about noise is amplified.  Only at the edge of chaos
    can information be preserved to arbitrary depth.

    We compute bounds on mutual information, Fisher information, and
    information capacity per layer.
    """

    def __init__(self, config: PhaseTransitionConfig):
        """
        Parameters
        ----------
        config : PhaseTransitionConfig
            Phase transition analysis configuration.
        """
        self.config = config
        self.kernels = ActivationKernels()

    def mutual_information_bound(self, sigma_w: float, sigma_b: float,
                                 depth: int = 100) -> np.ndarray:
        """Upper bound on mutual information I(x; h^(l)) through depth.

        Uses the data processing inequality: I(x; h^(l)) ≤ I(x; h^(l-1)).
        Combined with the rate of correlation convergence, we bound:

            I(x; h^(l)) ≤ (n/2) log(1 / (1 - c^(l)²))

        where c^(l) is the typical correlation at layer l.

        Parameters
        ----------
        sigma_w, sigma_b : float
            Initialization parameters.
        depth : int
            Number of layers.

        Returns
        -------
        np.ndarray, shape (depth+1,)
            Upper bound on mutual information (nats) at each layer.
        """
        cfg = _prop_config(self.config.activation, sigma_w, sigma_b, depth)
        fwd = ForwardPropagation(cfg)
        q_star = _get_q_star(self.config.activation, sigma_w, sigma_b)

        # Propagate two inputs with moderate initial correlation
        _, _, c_traj = fwd.propagate_correlation(q_star, q_star, 0.5, depth)

        # MI bound: in the 1-D Gaussian case,
        # I = -0.5 * log(1 - ρ²) where ρ is correlation
        # We use |c| < 1 to avoid log(0)
        c_sq = np.clip(c_traj ** 2, 0.0, 1.0 - 1e-15)
        mi_bound = -0.5 * np.log(1.0 - c_sq)
        return mi_bound

    def fisher_information_depth(self, sigma_w: float, sigma_b: float,
                                 depth: int = 100) -> np.ndarray:
        """Compute Fisher information at each layer.

        The Fisher information about a parameter θ in the input
        is proportional to the gradient variance:

            F^(l) ∝ E[||∂h^(l)/∂θ||²] ∝ ∏_{k=0}^{l-1} χ₁(q^(k))

        Parameters
        ----------
        sigma_w, sigma_b : float
            Initialization parameters.
        depth : int
            Number of layers.

        Returns
        -------
        np.ndarray, shape (depth+1,)
            Fisher information (relative) at each layer.
        """
        cfg = _prop_config(self.config.activation, sigma_w, sigma_b, depth)
        fwd = ForwardPropagation(cfg)
        bwd = BackwardPropagation(cfg)
        q_traj = fwd.propagate_variance(1.0, depth)
        jac = bwd.jacobian_norm(q_traj, depth)
        # Fisher info is proportional to Jacobian norm (forward direction)
        # Reverse the Jacobian so index 0 = layer 0 (input)
        return jac[::-1]

    def information_bottleneck_depth(self, sigma_w: float,
                                     sigma_b: float,
                                     threshold: float = 0.1
                                     ) -> int:
        """Find the depth at which mutual information drops below a threshold.

        Parameters
        ----------
        sigma_w, sigma_b : float
            Initialization parameters.
        threshold : float
            MI threshold (nats).

        Returns
        -------
        int
            Layer index where MI first drops below the threshold.
            Returns ``depth`` if it never drops below.
        """
        depth = self.config.depth
        mi = self.mutual_information_bound(sigma_w, sigma_b, depth)

        below = np.where(mi < threshold)[0]
        if len(below) == 0:
            return depth
        return int(below[0])

    def information_phase_boundary(self) -> Tuple[np.ndarray, np.ndarray]:
        """Find the boundary in (σ_w, σ_b) space for information propagation.

        The boundary separates the region where information can propagate
        to depth D (the configured depth) from where it cannot.

        Returns
        -------
        sigma_w_boundary, sigma_b_boundary : np.ndarray
            Points on the information propagation boundary.
        """
        sb_grid = self.config.sigma_b_grid
        sw_grid = self.config.sigma_w_grid
        sw_boundary = []
        sb_boundary = []
        target_depth = self.config.depth

        for sb in sb_grid:
            # For each σ_b, find the σ_w where bottleneck depth = target_depth
            depths = []
            for sw in sw_grid:
                d = self.information_bottleneck_depth(sw, sb, threshold=0.1)
                depths.append(d)
            depths = np.array(depths)

            # Find crossing
            crossings = np.where(depths >= target_depth)[0]
            if len(crossings) > 0 and crossings[0] > 0:
                idx = crossings[0]
                # Interpolate
                sw_c = np.interp(
                    target_depth,
                    [depths[idx - 1], depths[idx]],
                    [sw_grid[idx - 1], sw_grid[idx]],
                )
                sw_boundary.append(sw_c)
                sb_boundary.append(sb)

        return np.array(sw_boundary), np.array(sb_boundary)

    def capacity_per_layer(self, sigma_w: float,
                           sigma_b: float) -> float:
        """Estimate the information capacity per layer.

        We use the rate of Fisher information decay as a proxy:

            C ≈ 0.5 · ln(χ₁)

        At criticality (χ₁ = 1), the capacity per layer approaches 0
        but information persists to infinite depth.

        Parameters
        ----------
        sigma_w, sigma_b : float
            Initialization parameters.

        Returns
        -------
        float
            Capacity per layer (nats).
        """
        chi1 = _get_chi1(self.config.activation, sigma_w, sigma_b)
        return 0.5 * np.log(max(chi1, 1e-30))


# ---------------------------------------------------------------------------
# PhaseTransitionAnalyzer
# ---------------------------------------------------------------------------

class PhaseTransitionAnalyzer:
    """Combined analysis of all phase transitions.

    Produces unified phase diagrams, identifies special points
    (triple points, coexistence regions), and determines universality classes.
    """

    def __init__(self, config: PhaseTransitionConfig):
        """
        Parameters
        ----------
        config : PhaseTransitionConfig
            Phase transition analysis configuration.
        """
        self.config = config
        self.order_disorder = OrderDisorderTransition(config)
        self.chaos = ChaosTransition(config)
        self.info = InformationPropagation(config)

    def full_phase_diagram(self) -> Dict[str, np.ndarray]:
        """Compute a combined phase diagram with all transitions.

        Produces three 2-D maps over the (σ_w, σ_b) grid:
        - 'phase_label': integer labels (0=ordered, 1=critical, 2=chaotic)
        - 'order_parameter': Δq = q* - σ_b²
        - 'lyapunov': λ = ln(χ₁)

        Returns
        -------
        dict
            Keys: 'sigma_w', 'sigma_b', 'phase_label', 'order_parameter',
            'lyapunov'.
        """
        sw_grid = self.config.sigma_w_grid
        sb_grid = self.config.sigma_b_grid
        nw, nb = len(sw_grid), len(sb_grid)

        phase_label = np.zeros((nb, nw), dtype=int)
        order_param = np.zeros((nb, nw))
        lyap = np.zeros((nb, nw))
        tol = 0.01

        for i, sb in enumerate(sb_grid):
            for j, sw in enumerate(sw_grid):
                chi1 = _get_chi1(self.config.activation, sw, sb)
                lam = np.log(max(chi1, 1e-30))
                lyap[i, j] = lam
                order_param[i, j] = self.order_disorder.compute_order_parameter(
                    sw, sb
                )
                if abs(chi1 - 1.0) < tol:
                    phase_label[i, j] = 1
                elif chi1 > 1.0:
                    phase_label[i, j] = 2
                else:
                    phase_label[i, j] = 0

        return {
            "sigma_w": sw_grid,
            "sigma_b": sb_grid,
            "phase_label": phase_label,
            "order_parameter": order_param,
            "lyapunov": lyap,
        }

    def triple_point(self) -> Optional[Tuple[float, float]]:
        """Find the triple point where three phases meet, if it exists.

        For standard activations (relu, tanh), the ordered–chaotic
        transition is a single line and no true triple point exists.
        However, for activations with multiple fixed points, three
        phases can coexist.

        Returns
        -------
        (sigma_w, sigma_b) or None
            Coordinates of the triple point, or None if none found.
        """
        sw_grid = self.config.sigma_w_grid
        sb_grid = self.config.sigma_b_grid

        # A triple point requires at least three distinct phases meeting.
        # We search for grid cells where all three labels are present
        # in a 2×2 neighborhood.
        phase_diag = self.full_phase_diagram()
        labels = phase_diag["phase_label"]

        for i in range(len(sb_grid) - 1):
            for j in range(len(sw_grid) - 1):
                neighborhood = {
                    labels[i, j], labels[i, j + 1],
                    labels[i + 1, j], labels[i + 1, j + 1],
                }
                if len(neighborhood) >= 3:
                    sw = 0.5 * (sw_grid[j] + sw_grid[j + 1])
                    sb = 0.5 * (sb_grid[i] + sb_grid[i + 1])
                    return (sw, sb)

        return None

    def phase_coexistence_region(self) -> np.ndarray:
        """Find the region where multiple variance fixed points coexist.

        In this region, the variance map f(q) has multiple stable fixed
        points, leading to hysteresis and metastability.

        Returns
        -------
        np.ndarray, shape (grid_resolution, grid_resolution)
            Number of fixed points at each grid point.
        """
        sw_grid = self.config.sigma_w_grid
        sb_grid = self.config.sigma_b_grid
        result = np.zeros((len(sb_grid), len(sw_grid)), dtype=int)

        for i, sb in enumerate(sb_grid):
            for j, sw in enumerate(sw_grid):
                cfg = _prop_config(self.config.activation, sw, sb)
                analyzer = FixedPointAnalyzer(cfg)
                result[i, j] = analyzer.number_of_fixed_points(sw, sb)

        return result

    def universality_class(self, sigma_w_c: float,
                           sigma_b_c: float) -> Dict[str, float]:
        """Determine the universality class at a critical point.

        Compute critical exponents β, γ, ν and compare with known
        universality classes (mean-field: β=1/2, γ=1, ν=1/2).

        Parameters
        ----------
        sigma_w_c, sigma_b_c : float
            Critical point coordinates.

        Returns
        -------
        dict
            Keys: 'beta', 'gamma', 'nu', 'class_name'.
        """
        beta = self.order_disorder.critical_exponent_beta(
            sigma_w_c, sigma_b_c
        )
        gamma = self.order_disorder.susceptibility_divergence(
            sigma_w_c, sigma_b_c
        )

        # Correlation length exponent ν from ξ ~ |σ - σ_c|^{-ν}
        ci = CriticalInitialization(self.config.activation)
        deltas = np.logspace(-3, -0.5, 30)
        xi_vals = []
        for d in deltas:
            xi = ci.correlation_length(sigma_w_c + d, sigma_b_c)
            if np.isfinite(xi) and xi > 0:
                xi_vals.append((d, xi))

        if len(xi_vals) >= 5:
            log_d = np.log([x[0] for x in xi_vals])
            log_xi = np.log([x[1] for x in xi_vals])
            nu = -np.polyfit(log_d, log_xi, 1)[0]
        else:
            nu = float("nan")

        # Classify
        class_name = "unknown"
        if all(np.isfinite([beta, gamma, nu])):
            # Mean-field: β ≈ 0.5, γ ≈ 1.0, ν ≈ 0.5
            if abs(beta - 0.5) < 0.15 and abs(gamma - 1.0) < 0.3:
                class_name = "mean_field"
            elif abs(beta - 1.0) < 0.2:
                class_name = "linear_onset"

        return {
            "beta": beta,
            "gamma": gamma,
            "nu": nu,
            "class_name": class_name,
        }

    def finite_size_scaling(self, sigma_w_c: float,
                            widths: np.ndarray) -> Dict[str, np.ndarray]:
        """Finite-size (finite-width) scaling analysis at the critical point.

        At the critical point, finite-width corrections scale as:

            χ₁(n) - 1 ~ n^{-1/ν_⊥}

        where n is the width and ν_⊥ is the finite-size scaling exponent.

        For the mean-field theory we expect corrections of order 1/n, so
        we model χ₁(n) ≈ 1 + a/n + b/n².

        Parameters
        ----------
        sigma_w_c : float
            Critical σ_w.
        widths : np.ndarray
            Array of network widths to test.

        Returns
        -------
        dict
            Keys: 'widths', 'chi1_correction', 'scaling_exponent'.
        """
        # In the infinite-width mean-field theory, χ₁ is width-independent.
        # Finite-width corrections come from the 1/n expansion.
        # We model: χ₁(n) = 1 + α/n where α is an activation-dependent
        # constant.
        chi1_inf = _get_chi1(self.config.activation, sigma_w_c, 0.0)

        # Approximate finite-width correction coefficient
        _, phi_prime, _ = _ACTIVATIONS[self.config.activation]
        q_star = _get_q_star(self.config.activation, sigma_w_c, 0.0)

        # Fourth moment correction: Var[φ'(h)²] contributes at O(1/n)
        sq = np.sqrt(max(q_star, 1e-30))
        fourth_moment = _gauss_expectation(
            lambda z: phi_prime(sq * z) ** 4
        )
        second_moment = _gauss_expectation(
            lambda z: phi_prime(sq * z) ** 2
        )
        correction_coeff = sigma_w_c ** 2 * (
            fourth_moment - second_moment ** 2
        )

        chi1_corrections = np.zeros_like(widths, dtype=float)
        for i, n in enumerate(widths):
            chi1_corrections[i] = chi1_inf + correction_coeff / max(n, 1)

        # Estimate scaling exponent from log-log
        if len(widths) >= 3:
            diffs = np.abs(chi1_corrections - chi1_inf)
            mask = diffs > 1e-15
            if np.sum(mask) >= 3:
                log_n = np.log(widths[mask].astype(float))
                log_d = np.log(diffs[mask])
                exponent = -np.polyfit(log_n, log_d, 1)[0]
            else:
                exponent = 1.0
        else:
            exponent = 1.0

        return {
            "widths": widths,
            "chi1_correction": chi1_corrections,
            "scaling_exponent": exponent,
        }

    def critical_slowing_analysis(self, sigma_w_range: np.ndarray,
                                  sigma_b: float) -> Dict[str, np.ndarray]:
        """Analyze critical slowing down near the phase boundary.

        Near the critical point, the number of iterations to reach the
        fixed point diverges.  We measure the convergence rate (spectral
        gap of the linearized map).

        Parameters
        ----------
        sigma_w_range : np.ndarray
            σ_w values to scan (should bracket the critical point).
        sigma_b : float
            Fixed σ_b.

        Returns
        -------
        dict
            Keys: 'sigma_w', 'convergence_rate', 'correlation_length'.
        """
        rates = np.zeros_like(sigma_w_range)
        xi_vals = np.zeros_like(sigma_w_range)
        ci = CriticalInitialization(self.config.activation)

        for i, sw in enumerate(sigma_w_range):
            chi1 = _get_chi1(self.config.activation, sw, sigma_b)
            # Convergence rate = |1 - χ₁| (spectral gap)
            rates[i] = abs(1.0 - chi1)
            xi_vals[i] = ci.correlation_length(sw, sigma_b)

        return {
            "sigma_w": sigma_w_range,
            "convergence_rate": rates,
            "correlation_length": xi_vals,
        }


# ---------------------------------------------------------------------------
# InitializationPhaseDiagram
# ---------------------------------------------------------------------------

class InitializationPhaseDiagram:
    """Produce publication-quality phase diagrams for initialization.

    Generates the (σ_w, σ_b) diagram showing ordered, chaotic, and
    edge-of-chaos regions, along with recommended initializations for
    different target depths.
    """

    def __init__(self, activation: str = "relu"):
        """
        Parameters
        ----------
        activation : str
            Activation function name.
        """
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'.")
        self.activation = activation

    def compute_diagram(self, sigma_w_range: np.ndarray,
                        sigma_b_range: np.ndarray,
                        depth: int = 100) -> Dict[str, np.ndarray]:
        """Compute the full (σ_w, σ_b) phase diagram.

        Parameters
        ----------
        sigma_w_range : np.ndarray
            σ_w values.
        sigma_b_range : np.ndarray
            σ_b values.
        depth : int
            Network depth for depth-dependent quantities.

        Returns
        -------
        dict
            Keys:
            - 'chi1': 2-D array of χ₁ values
            - 'q_star': 2-D array of variance fixed points
            - 'lyapunov': 2-D array of Lyapunov exponents
            - 'phase': 2-D int array (0=ordered, 1=critical, 2=chaotic)
            - 'max_depth': 2-D int array of max trainable depth
        """
        nw = len(sigma_w_range)
        nb = len(sigma_b_range)

        chi1_map = np.zeros((nb, nw))
        q_star_map = np.zeros((nb, nw))
        lyap_map = np.zeros((nb, nw))
        phase_map = np.zeros((nb, nw), dtype=int)
        depth_map = np.zeros((nb, nw), dtype=int)

        tol = 0.01

        for i, sb in enumerate(sigma_b_range):
            for j, sw in enumerate(sigma_w_range):
                q_star = _get_q_star(self.activation, sw, sb)
                chi1 = _get_chi1(self.activation, sw, sb)
                lam = np.log(max(chi1, 1e-30))

                chi1_map[i, j] = chi1
                q_star_map[i, j] = q_star
                lyap_map[i, j] = lam

                if abs(chi1 - 1.0) < tol:
                    phase_map[i, j] = 1
                    depth_map[i, j] = 10000
                elif chi1 > 1.0:
                    phase_map[i, j] = 2
                    max_d = int(np.log(1e6) / lam) if lam > 0 else depth
                    depth_map[i, j] = max_d
                else:
                    phase_map[i, j] = 0
                    max_d = int(-1.0 / lam) if lam < 0 else depth
                    depth_map[i, j] = max_d

        return {
            "chi1": chi1_map,
            "q_star": q_star_map,
            "lyapunov": lyap_map,
            "phase": phase_map,
            "max_depth": depth_map,
        }

    def annotate_regions(self, diagram: Dict[str, np.ndarray]
                         ) -> Dict[str, str]:
        """Label the regions in a computed phase diagram.

        Parameters
        ----------
        diagram : dict
            Output of ``compute_diagram``.

        Returns
        -------
        dict
            Mapping from region label to human-readable description.
        """
        phase = diagram["phase"]
        annotations = {}

        if np.any(phase == 0):
            annotations["ordered"] = (
                "Ordered phase (χ₁ < 1): correlations converge to 1, "
                "gradients vanish exponentially. All inputs map to the "
                "same representation at depth."
            )
        if np.any(phase == 1):
            annotations["critical"] = (
                "Edge of chaos (χ₁ ≈ 1): correlation length diverges, "
                "information propagates to arbitrary depth. Optimal for "
                "training deep networks."
            )
        if np.any(phase == 2):
            annotations["chaotic"] = (
                "Chaotic phase (χ₁ > 1): nearby inputs diverge exponentially, "
                "gradients explode. Network is sensitive to noise."
            )

        return annotations

    def gradient_flow_field(self, sigma_w_range: np.ndarray,
                            sigma_b_range: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the gradient flow field in (σ_w, σ_b) space.

        The flow field indicates how the effective initialization
        parameters evolve under one step of the variance recursion:

            Δσ_w ∝ ∂(χ₁ - 1)/∂σ_w
            Δσ_b ∝ ∂(χ₁ - 1)/∂σ_b

        This shows the "natural" dynamics pulling toward or away from
        the critical line.

        Parameters
        ----------
        sigma_w_range, sigma_b_range : np.ndarray
            Grid arrays.

        Returns
        -------
        dw, db : np.ndarray
            Flow field components, each shape (len(sb), len(sw)).
        """
        nw = len(sigma_w_range)
        nb = len(sigma_b_range)
        dw = np.zeros((nb, nw))
        db = np.zeros((nb, nw))
        eps = 1e-4

        for i, sb in enumerate(sigma_b_range):
            for j, sw in enumerate(sigma_w_range):
                chi_0 = _get_chi1(self.activation, sw, sb)

                # Gradient with respect to σ_w
                sw_p = sw + eps
                chi_wp = _get_chi1(self.activation, sw_p, sb)
                dw[i, j] = -(chi_wp - chi_0) / eps  # Flow toward χ₁=1

                # Gradient with respect to σ_b
                sb_p = sb + eps
                chi_bp = _get_chi1(self.activation, sw, sb_p)
                db[i, j] = -(chi_bp - chi_0) / eps

        # Normalize for visualization
        mag = np.sqrt(dw ** 2 + db ** 2)
        mag = np.maximum(mag, 1e-10)
        dw /= mag
        db /= mag

        return dw, db

    def basins_of_attraction(self, sigma_w_range: np.ndarray,
                             sigma_b_range: np.ndarray
                             ) -> np.ndarray:
        """Determine which fixed point each (σ_w, σ_b) point flows to.

        Starting from each grid point, iterate the variance map and
        determine which fixed point (if any) the trajectory converges to.

        Parameters
        ----------
        sigma_w_range, sigma_b_range : np.ndarray
            Grid arrays.

        Returns
        -------
        np.ndarray, shape (len(sb), len(sw))
            Fixed-point label at each grid point.  0 = trivial (q*≈σ_b²),
            1 = non-trivial, -1 = did not converge.
        """
        nw = len(sigma_w_range)
        nb = len(sigma_b_range)
        basins = np.zeros((nb, nw), dtype=int)

        for i, sb in enumerate(sigma_b_range):
            for j, sw in enumerate(sigma_w_range):
                cfg = _prop_config(self.activation, sw, sb, depth=500)
                fwd = ForwardPropagation(cfg)
                q_traj = fwd.propagate_variance(1.0, 500)
                q_final = q_traj[-1]
                q_trivial = sb ** 2

                if abs(q_final - q_trivial) < 0.01 * max(q_trivial, 0.01):
                    basins[i, j] = 0
                elif q_final > q_trivial + 0.01:
                    basins[i, j] = 1
                else:
                    # Check convergence
                    if abs(q_traj[-1] - q_traj[-2]) < 1e-8:
                        basins[i, j] = 1
                    else:
                        basins[i, j] = -1

        return basins

    def activation_comparison(self, activations: List[str],
                              sigma_w_range: np.ndarray,
                              sigma_b_range: np.ndarray
                              ) -> Dict[str, Dict[str, np.ndarray]]:
        """Compare phase diagrams across different activations.

        Parameters
        ----------
        activations : list of str
            Activation function names.
        sigma_w_range, sigma_b_range : np.ndarray
            Grid arrays.

        Returns
        -------
        dict
            Mapping from activation name to phase diagram dict
            (each as returned by ``compute_diagram``).
        """
        results = {}
        for act in activations:
            diag = InitializationPhaseDiagram(act)
            results[act] = diag.compute_diagram(
                sigma_w_range, sigma_b_range
            )
        return results

    def recommended_initialization(self, activation: str,
                                   target_depth: int) -> Dict[str, float]:
        """Produce a practical initialization recommendation.

        For a network of a given depth and activation, find the (σ_w, σ_b)
        that places the network as close to the edge of chaos as possible
        while keeping gradients well-behaved.

        Parameters
        ----------
        activation : str
            Activation function.
        target_depth : int
            Target network depth.

        Returns
        -------
        dict
            Keys: 'sigma_w', 'sigma_b', 'chi1', 'correlation_length',
            'gradient_norm_at_depth'.
        """
        ci = CriticalInitialization(activation)

        # For simplicity, recommend σ_b = 0 (no bias variance)
        sigma_b = 0.0
        sigma_w = ci.optimal_init_for_depth(target_depth, sigma_b)

        if np.isnan(sigma_w):
            # Fallback to critical
            sigma_w = ci.find_critical_sigma_w(sigma_b)

        chi1 = _get_chi1(activation, sigma_w, sigma_b)
        xi = ci.correlation_length(sigma_w, sigma_b)

        # Gradient norm at target depth
        grad_norm = chi1 ** target_depth

        return {
            "sigma_w": sigma_w,
            "sigma_b": sigma_b,
            "chi1": chi1,
            "correlation_length": xi,
            "gradient_norm_at_depth": grad_norm,
        }
