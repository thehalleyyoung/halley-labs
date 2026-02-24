"""
Renormalization group analysis for neural networks.

Implements block spin RG, Wilson RG, RG flow, fixed point detection,
critical exponents, universality class detection, scaling functions,
and crossover behavior analysis.
"""

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar, least_squares
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.linalg import eigvalsh, svdvals
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
import warnings


@dataclass
class RGReport:
    """Report from renormalization group analysis."""
    fixed_points: List[Dict[str, float]] = field(default_factory=list)
    rg_flow_trajectory: List[Dict[str, float]] = field(default_factory=list)
    critical_exponents: Dict[str, float] = field(default_factory=dict)
    universality_class: str = "unknown"
    scaling_function_params: Dict[str, float] = field(default_factory=dict)
    crossover_scales: List[float] = field(default_factory=list)
    effective_couplings: List[Dict[str, float]] = field(default_factory=list)
    is_at_criticality: bool = False
    depth_scale: float = 0.0
    phase: str = "unknown"
    correlation_length: float = 0.0
    block_rg_results: Dict[str, Any] = field(default_factory=dict)
    wilson_rg_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec:
    """Specification for RG analysis."""
    depth: int = 10
    width: int = 256
    sigma_w: float = 1.0
    sigma_b: float = 0.0
    activation: str = "relu"
    widths: Optional[List[int]] = None
    has_residual: bool = False
    residual_alpha: float = 1.0


class ActivationFunctions:
    """Library of activation functions and their statistical properties."""

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))

    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -500, 500)) - 1))

    @staticmethod
    def softplus(x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(np.clip(x, -500, 500)))

    @staticmethod
    def sin_act(x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    @staticmethod
    def get_activation(name: str) -> Callable:
        mapping = {
            "relu": ActivationFunctions.relu,
            "tanh": ActivationFunctions.tanh,
            "sigmoid": ActivationFunctions.sigmoid,
            "gelu": ActivationFunctions.gelu,
            "silu": ActivationFunctions.silu,
            "swish": ActivationFunctions.silu,
            "leaky_relu": ActivationFunctions.leaky_relu,
            "elu": ActivationFunctions.elu,
            "softplus": ActivationFunctions.softplus,
            "sin": ActivationFunctions.sin_act,
        }
        if name not in mapping:
            raise ValueError(f"Unknown activation: {name}")
        return mapping[name]


class VariancePropagator:
    """Propagate variance through layers using mean field theory."""

    def __init__(self, activation_fn: Callable, n_samples: int = 50000):
        self.activation_fn = activation_fn
        self.n_samples = n_samples

    def compute_variance_map(self, q_in: float, sigma_w: float, sigma_b: float) -> float:
        """Compute output variance given input variance q_in."""
        z = np.random.randn(self.n_samples) * np.sqrt(max(q_in, 1e-12))
        activated = self.activation_fn(z)
        return sigma_w ** 2 * np.mean(activated ** 2) + sigma_b ** 2

    def compute_chi(self, q: float, sigma_w: float) -> float:
        """Compute Jacobian norm factor chi_1 at variance q."""
        z = np.random.randn(self.n_samples) * np.sqrt(max(q, 1e-12))
        eps = 1e-6
        dphi = (self.activation_fn(z + eps) - self.activation_fn(z - eps)) / (2 * eps)
        return sigma_w ** 2 * np.mean(dphi ** 2)

    def find_fixed_point(self, sigma_w: float, sigma_b: float,
                         q_range: Tuple[float, float] = (0.01, 10.0),
                         n_grid: int = 100) -> float:
        """Find fixed point q* where variance_map(q*) = q*."""
        q_vals = np.linspace(q_range[0], q_range[1], n_grid)
        diffs = []
        for q in q_vals:
            mapped = self.compute_variance_map(q, sigma_w, sigma_b)
            diffs.append(mapped - q)
        diffs = np.array(diffs)

        sign_changes = np.where(np.diff(np.sign(diffs)))[0]
        if len(sign_changes) == 0:
            return self.compute_variance_map(1.0, sigma_w, sigma_b)

        idx = sign_changes[0]
        try:
            q_star = brentq(
                lambda q: self.compute_variance_map(q, sigma_w, sigma_b) - q,
                q_vals[idx], q_vals[idx + 1]
            )
            return q_star
        except (ValueError, RuntimeError):
            return q_vals[idx]

    def propagate(self, q_init: float, sigma_w: float, sigma_b: float,
                  n_layers: int) -> List[float]:
        """Propagate variance through n_layers."""
        trajectory = [q_init]
        q = q_init
        for _ in range(n_layers):
            q = self.compute_variance_map(q, sigma_w, sigma_b)
            trajectory.append(q)
        return trajectory


class BlockSpinRG:
    """Block spin renormalization group for neural networks.

    Coarse-grains layers by grouping adjacent neurons,
    computing effective couplings at each RG step.
    """

    def __init__(self, block_size: int = 2):
        self.block_size = block_size

    def coarse_grain_weights(self, W: np.ndarray) -> np.ndarray:
        """Coarse-grain weight matrix by averaging block_size x block_size blocks."""
        n_out, n_in = W.shape
        new_out = n_out // self.block_size
        new_in = n_in // self.block_size
        if new_out == 0 or new_in == 0:
            return W[:1, :1] if W.size > 0 else np.array([[0.0]])

        W_trimmed = W[:new_out * self.block_size, :new_in * self.block_size]
        W_reshaped = W_trimmed.reshape(new_out, self.block_size, new_in, self.block_size)
        W_coarse = W_reshaped.mean(axis=(1, 3)) * self.block_size
        return W_coarse

    def compute_effective_couplings(self, W: np.ndarray) -> Dict[str, float]:
        """Extract effective couplings from weight matrix."""
        singular_values = svdvals(W)
        n = min(W.shape)
        if n == 0:
            return {"weight_variance": 0.0, "spectral_norm": 0.0,
                    "effective_rank": 0.0, "frobenius_norm": 0.0}

        weight_var = np.var(W)
        spectral_norm = singular_values[0] if len(singular_values) > 0 else 0.0
        sv_normalized = singular_values / (np.sum(singular_values) + 1e-12)
        entropy = -np.sum(sv_normalized * np.log(sv_normalized + 1e-12))
        effective_rank = np.exp(entropy)
        frobenius = np.sqrt(np.sum(singular_values ** 2))

        return {
            "weight_variance": float(weight_var),
            "spectral_norm": float(spectral_norm),
            "effective_rank": float(effective_rank),
            "frobenius_norm": float(frobenius),
            "participation_ratio": float(np.sum(singular_values ** 2) ** 2 /
                                         (np.sum(singular_values ** 4) + 1e-12)),
        }

    def rg_step(self, W: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Perform one RG step: coarse-grain and compute new couplings."""
        W_coarse = self.coarse_grain_weights(W)
        couplings = self.compute_effective_couplings(W_coarse)
        return W_coarse, couplings

    def full_rg_flow(self, W: np.ndarray, n_steps: int = 5) -> List[Dict[str, float]]:
        """Perform multiple RG steps, tracking coupling flow."""
        flow = [self.compute_effective_couplings(W)]
        W_current = W.copy()
        for step in range(n_steps):
            if min(W_current.shape) < self.block_size:
                break
            W_current, couplings = self.rg_step(W_current)
            couplings["rg_step"] = step + 1
            couplings["matrix_shape"] = list(W_current.shape)
            flow.append(couplings)
        return flow

    def detect_fixed_point_from_flow(self, flow: List[Dict[str, float]],
                                     key: str = "weight_variance",
                                     tol: float = 0.05) -> Optional[Dict[str, float]]:
        """Check if coupling converges to fixed point."""
        if len(flow) < 3:
            return None
        values = [f[key] for f in flow if key in f]
        if len(values) < 3:
            return None
        ratios = [values[i + 1] / (values[i] + 1e-12) for i in range(len(values) - 1)]
        if len(ratios) >= 2 and abs(ratios[-1] - ratios[-2]) < tol:
            return {"fixed_point_value": values[-1], "scaling_ratio": ratios[-1],
                    "key": key, "converged": True}
        return {"fixed_point_value": values[-1], "scaling_ratio": ratios[-1] if ratios else 1.0,
                "key": key, "converged": False}


class WilsonRG:
    """Wilson renormalization group: integrate out high-frequency modes."""

    def __init__(self, cutoff_fraction: float = 0.5):
        self.cutoff_fraction = cutoff_fraction

    def decompose_modes(self, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decompose weight matrix into slow and fast modes via SVD."""
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        n_keep = max(1, int(len(s) * self.cutoff_fraction))
        s_slow = s[:n_keep]
        s_fast = s[n_keep:]
        W_slow = U[:, :n_keep] * s_slow @ Vt[:n_keep, :]
        W_fast = U[:, n_keep:] * s_fast @ Vt[n_keep:, :] if n_keep < len(s) else np.zeros_like(W)
        return W_slow, W_fast, s

    def integrate_out_fast(self, W: np.ndarray, sigma_w: float) -> Tuple[np.ndarray, Dict[str, float]]:
        """Integrate out high-frequency modes, renormalize slow modes."""
        W_slow, W_fast, singular_values = self.decompose_modes(W)

        fast_contribution = np.mean(W_fast ** 2)
        renorm_factor = 1.0 + fast_contribution / (sigma_w ** 2 + 1e-12)
        W_renormalized = W_slow * np.sqrt(renorm_factor)

        n_keep = max(1, int(len(singular_values) * self.cutoff_fraction))
        info = {
            "n_modes_total": len(singular_values),
            "n_modes_kept": n_keep,
            "n_modes_integrated": len(singular_values) - n_keep,
            "fast_mode_energy": float(np.sum(singular_values[n_keep:] ** 2)),
            "slow_mode_energy": float(np.sum(singular_values[:n_keep] ** 2)),
            "renorm_factor": float(renorm_factor),
            "effective_sigma_w": float(sigma_w * np.sqrt(renorm_factor)),
        }
        return W_renormalized, info

    def wilson_rg_flow(self, W: np.ndarray, sigma_w: float,
                       n_steps: int = 5) -> List[Dict[str, float]]:
        """Perform Wilson RG flow over multiple steps."""
        flow = []
        W_current = W.copy()
        current_sigma_w = sigma_w
        for step in range(n_steps):
            if min(W_current.shape) < 4:
                break
            W_current, info = self.integrate_out_fast(W_current, current_sigma_w)
            info["step"] = step
            current_sigma_w = info["effective_sigma_w"]
            flow.append(info)
        return flow

    def compute_beta_function(self, W: np.ndarray, sigma_w: float,
                              delta: float = 0.01) -> Dict[str, float]:
        """Compute beta function: how couplings change under RG."""
        var_before = np.var(W)
        W_after, info = self.integrate_out_fast(W, sigma_w)
        var_after = np.var(W_after)
        beta_var = (var_after - var_before) / (np.log(info["n_modes_total"]) -
                                                np.log(info["n_modes_kept"] + 1e-12) + 1e-12)
        spec_before = np.max(svdvals(W))
        spec_after = np.max(svdvals(W_after)) if W_after.size > 0 else 0.0
        beta_spec = (spec_after - spec_before) / (np.log(info["n_modes_total"]) -
                                                    np.log(info["n_modes_kept"] + 1e-12) + 1e-12)
        return {
            "beta_variance": float(beta_var),
            "beta_spectral": float(beta_spec),
            "is_relevant": bool(abs(beta_var) > delta),
            "flow_direction": "growing" if beta_var > 0 else "shrinking",
        }


class CriticalExponentComputer:
    """Compute critical exponents at phase transitions."""

    def __init__(self, n_samples: int = 30000):
        self.n_samples = n_samples

    def compute_correlation_length_exponent(
        self, activation_fn: Callable, sigma_b: float,
        sigma_w_range: Tuple[float, float] = (0.5, 3.0),
        n_points: int = 50
    ) -> Dict[str, float]:
        """Compute correlation length exponent nu from depth scale divergence."""
        sigma_w_values = np.linspace(sigma_w_range[0], sigma_w_range[1], n_points)
        prop = VariancePropagator(activation_fn, self.n_samples)

        sigma_w_c = self._find_critical_sigma_w(activation_fn, sigma_b, sigma_w_range)

        depth_scales = []
        distances = []
        for sw in sigma_w_values:
            if abs(sw - sigma_w_c) < 0.01:
                continue
            q_star = prop.find_fixed_point(sw, sigma_b)
            chi = prop.compute_chi(q_star, sw)
            if chi > 0 and chi != 1.0:
                xi = -1.0 / np.log(abs(chi) + 1e-12)
                depth_scales.append(max(xi, 0.1))
                distances.append(abs(sw - sigma_w_c))

        if len(depth_scales) < 5:
            return {"nu": 1.0, "sigma_w_c": sigma_w_c, "n_points_used": len(depth_scales)}

        depth_scales = np.array(depth_scales)
        distances = np.array(distances)
        mask = (distances > 0.01) & (distances < 1.0) & np.isfinite(depth_scales)
        if np.sum(mask) < 3:
            return {"nu": 1.0, "sigma_w_c": sigma_w_c, "n_points_used": 0}

        log_xi = np.log(depth_scales[mask])
        log_dist = np.log(distances[mask])
        coeffs = np.polyfit(log_dist, log_xi, 1)
        nu = -coeffs[0]

        return {
            "nu": float(np.clip(nu, 0.1, 10.0)),
            "sigma_w_c": float(sigma_w_c),
            "n_points_used": int(np.sum(mask)),
            "fit_intercept": float(coeffs[1]),
        }

    def _find_critical_sigma_w(self, activation_fn: Callable, sigma_b: float,
                                sigma_w_range: Tuple[float, float]) -> float:
        """Find critical sigma_w where chi_1 = 1."""
        prop = VariancePropagator(activation_fn, self.n_samples)

        def chi_minus_one(sw):
            q_star = prop.find_fixed_point(sw, sigma_b)
            chi = prop.compute_chi(q_star, sw)
            return chi - 1.0

        try:
            sw_vals = np.linspace(sigma_w_range[0], sigma_w_range[1], 30)
            chi_vals = [chi_minus_one(sw) for sw in sw_vals]
            sign_changes = np.where(np.diff(np.sign(chi_vals)))[0]
            if len(sign_changes) > 0:
                idx = sign_changes[0]
                return brentq(chi_minus_one, sw_vals[idx], sw_vals[idx + 1])
        except (ValueError, RuntimeError):
            pass
        return np.mean(sigma_w_range)

    def compute_order_parameter_exponent(
        self, activation_fn: Callable, sigma_b: float,
        sigma_w_c: float, n_points: int = 30
    ) -> Dict[str, float]:
        """Compute order parameter exponent beta near transition."""
        prop = VariancePropagator(activation_fn, self.n_samples)
        q_star_c = prop.find_fixed_point(sigma_w_c, sigma_b)

        deltas = np.linspace(0.05, 1.0, n_points)
        order_params = []
        valid_deltas = []
        for delta in deltas:
            sw = sigma_w_c + delta
            q_star = prop.find_fixed_point(sw, sigma_b)
            op = abs(q_star - q_star_c) / (q_star_c + 1e-12)
            if op > 1e-6 and np.isfinite(op):
                order_params.append(op)
                valid_deltas.append(delta)

        if len(order_params) < 3:
            return {"beta": 0.5, "n_points_used": len(order_params)}

        log_op = np.log(np.array(order_params))
        log_delta = np.log(np.array(valid_deltas))
        coeffs = np.polyfit(log_delta, log_op, 1)

        return {
            "beta": float(np.clip(coeffs[0], 0.1, 5.0)),
            "n_points_used": len(order_params),
        }

    def compute_susceptibility_exponent(
        self, activation_fn: Callable, sigma_b: float,
        sigma_w_c: float, n_points: int = 30
    ) -> Dict[str, float]:
        """Compute susceptibility exponent gamma from chi(sigma_w)."""
        prop = VariancePropagator(activation_fn, self.n_samples)
        deltas = np.linspace(0.05, 1.0, n_points)
        susceptibilities = []
        valid_deltas = []
        for delta in deltas:
            sw = sigma_w_c - delta
            if sw < 0.1:
                continue
            q_star = prop.find_fixed_point(sw, sigma_b)
            chi = prop.compute_chi(q_star, sw)
            susceptibility = 1.0 / (abs(1.0 - chi) + 1e-12)
            if np.isfinite(susceptibility) and susceptibility < 1e6:
                susceptibilities.append(susceptibility)
                valid_deltas.append(delta)

        if len(susceptibilities) < 3:
            return {"gamma": 1.0, "n_points_used": len(susceptibilities)}

        log_susc = np.log(np.array(susceptibilities))
        log_delta = np.log(np.array(valid_deltas))
        coeffs = np.polyfit(log_delta, log_susc, 1)

        return {
            "gamma": float(np.clip(-coeffs[0], 0.1, 10.0)),
            "n_points_used": len(susceptibilities),
        }


class UniversalityClassDetector:
    """Detect universality class of neural network architecture."""

    KNOWN_CLASSES = {
        "mean_field": {"nu": 0.5, "beta": 0.5, "gamma": 1.0},
        "ising_2d": {"nu": 1.0, "beta": 0.125, "gamma": 1.75},
        "ising_3d": {"nu": 0.63, "beta": 0.326, "gamma": 1.237},
        "percolation_2d": {"nu": 4.0 / 3.0, "beta": 5.0 / 36.0, "gamma": 43.0 / 18.0},
        "xy_2d": {"nu": 0.5, "beta": 0.23, "gamma": 1.0},
    }

    def classify(self, exponents: Dict[str, float]) -> Tuple[str, float]:
        """Classify universality class from measured exponents."""
        best_class = "unknown"
        best_distance = float("inf")

        for class_name, class_exponents in self.KNOWN_CLASSES.items():
            distance = 0.0
            n = 0
            for key in class_exponents:
                if key in exponents:
                    distance += (exponents[key] - class_exponents[key]) ** 2
                    n += 1
            if n > 0:
                distance = np.sqrt(distance / n)
                if distance < best_distance:
                    best_distance = distance
                    best_class = class_name

        return best_class, float(best_distance)

    def check_scaling_relations(self, exponents: Dict[str, float]) -> Dict[str, float]:
        """Check hyperscaling relations between exponents."""
        results = {}
        nu = exponents.get("nu", None)
        beta = exponents.get("beta", None)
        gamma = exponents.get("gamma", None)

        if nu and beta and gamma:
            rushbrooke = 2 * beta + gamma - 2.0
            results["rushbrooke_violation"] = float(abs(rushbrooke))
            results["rushbrooke_relation"] = f"2*beta + gamma = {2*beta + gamma:.4f} (should be 2)"

        if nu and gamma:
            fisher_ratio = gamma / nu
            results["fisher_ratio_gamma_nu"] = float(fisher_ratio)

        return results


class ScalingFunctionAnalyzer:
    """Analyze scaling functions and data collapse."""

    def __init__(self, n_samples: int = 20000):
        self.n_samples = n_samples

    def compute_observable_at_widths(
        self, activation_fn: Callable, sigma_w: float, sigma_b: float,
        widths: List[int], depth: int, observable: str = "variance"
    ) -> Dict[int, float]:
        """Compute an observable for different widths."""
        results = {}
        prop = VariancePropagator(activation_fn, self.n_samples)
        for width in widths:
            q_star = prop.find_fixed_point(sigma_w, sigma_b)
            if observable == "variance":
                trajectory = prop.propagate(1.0, sigma_w, sigma_b, depth)
                results[width] = trajectory[-1]
            elif observable == "chi":
                chi = prop.compute_chi(q_star, sigma_w)
                results[width] = chi
            elif observable == "depth_scale":
                chi = prop.compute_chi(q_star, sigma_w)
                xi = -1.0 / np.log(abs(chi) + 1e-12) if abs(chi) < 1 else float("inf")
                finite_size_correction = 1.0 + 1.0 / (width + 1e-12)
                results[width] = xi * finite_size_correction
            else:
                results[width] = q_star
        return results

    def attempt_data_collapse(
        self, data: Dict[int, List[Tuple[float, float]]],
        tc_range: Tuple[float, float] = (0.5, 3.0),
        nu_range: Tuple[float, float] = (0.3, 3.0)
    ) -> Dict[str, float]:
        """Attempt data collapse for finite-size scaling.

        data: {width: [(temperature, observable_value), ...]}
        """
        if not data or len(data) < 2:
            return {"tc": np.mean(tc_range), "nu": 1.0, "quality": 0.0}

        widths = sorted(data.keys())

        def collapse_quality(params):
            tc, nu = params
            all_x = []
            all_y = []
            for width in widths:
                for t, obs in data[width]:
                    x_scaled = (t - tc) * width ** (1.0 / (nu + 1e-6))
                    y_scaled = obs * width ** (-1.0 / (nu + 1e-6))
                    all_x.append(x_scaled)
                    all_y.append(y_scaled)

            if len(all_x) < 4:
                return 1e6

            all_x = np.array(all_x)
            all_y = np.array(all_y)
            sort_idx = np.argsort(all_x)
            all_x = all_x[sort_idx]
            all_y = all_y[sort_idx]

            residuals = 0.0
            count = 0
            for i in range(1, len(all_x)):
                dx = all_x[i] - all_x[i - 1]
                if dx > 1e-10:
                    dy = abs(all_y[i] - all_y[i - 1])
                    residuals += dy ** 2 / (dx + 1e-6)
                    count += 1
            return residuals / (count + 1) if count > 0 else 1e6

        result = minimize(
            collapse_quality,
            x0=[np.mean(tc_range), 1.0],
            bounds=[(tc_range[0], tc_range[1]), (nu_range[0], nu_range[1])],
            method="L-BFGS-B"
        )

        return {
            "tc": float(result.x[0]),
            "nu": float(result.x[1]),
            "quality": float(1.0 / (1.0 + result.fun)),
            "converged": bool(result.success),
        }

    def compute_scaling_function(
        self, activation_fn: Callable, sigma_b: float,
        sigma_w_values: np.ndarray, widths: List[int], depth: int,
        sigma_w_c: float, nu: float
    ) -> Dict[str, np.ndarray]:
        """Compute universal scaling function from data at multiple widths."""
        all_x_scaled = []
        all_y_scaled = []
        all_widths_label = []

        prop = VariancePropagator(activation_fn, self.n_samples)

        for width in widths:
            for sw in sigma_w_values:
                q_star = prop.find_fixed_point(sw, sigma_b)
                chi = prop.compute_chi(q_star, sw)
                xi = -1.0 / np.log(abs(chi) + 1e-12) if abs(chi) < 1 else width
                x_scaled = (sw - sigma_w_c) * width ** (1.0 / nu)
                y_scaled = xi / width
                all_x_scaled.append(x_scaled)
                all_y_scaled.append(y_scaled)
                all_widths_label.append(width)

        return {
            "x_scaled": np.array(all_x_scaled),
            "y_scaled": np.array(all_y_scaled),
            "widths": np.array(all_widths_label),
        }


class CrossoverAnalyzer:
    """Analyze crossover behavior between different scaling regimes."""

    def __init__(self, n_samples: int = 20000):
        self.n_samples = n_samples

    def detect_crossover_scale(
        self, activation_fn: Callable, sigma_w: float, sigma_b: float,
        depth_range: Tuple[int, int] = (2, 200)
    ) -> Dict[str, Any]:
        """Detect crossover depth scale where finite-width effects become important."""
        prop = VariancePropagator(activation_fn, self.n_samples)
        q_star = prop.find_fixed_point(sigma_w, sigma_b)
        chi = prop.compute_chi(q_star, sigma_w)

        trajectory = prop.propagate(1.0, sigma_w, sigma_b, depth_range[1])
        diffs = np.abs(np.diff(trajectory))

        crossover_depth = depth_range[1]
        threshold = 0.01 * abs(trajectory[0] - q_star + 1e-12)
        for i, d in enumerate(diffs):
            if d < threshold and i > 2:
                crossover_depth = i
                break

        return {
            "crossover_depth": int(crossover_depth),
            "chi_1": float(chi),
            "q_star": float(q_star),
            "depth_scale": float(-1.0 / np.log(abs(chi) + 1e-12)) if abs(chi) < 1.0 else float("inf"),
            "initial_variance": float(trajectory[0]),
            "final_variance": float(trajectory[-1]),
        }

    def characterize_regimes(
        self, activation_fn: Callable, sigma_w: float, sigma_b: float,
        depth: int
    ) -> Dict[str, Any]:
        """Characterize the scaling regimes present."""
        prop = VariancePropagator(activation_fn, self.n_samples)
        trajectory = prop.propagate(1.0, sigma_w, sigma_b, depth)
        trajectory = np.array(trajectory)

        regimes = []
        log_traj = np.log(np.abs(trajectory) + 1e-12)

        window = max(3, depth // 10)
        for i in range(0, len(log_traj) - window, window):
            segment = log_traj[i:i + window]
            slope = np.polyfit(range(len(segment)), segment, 1)[0]
            mean_var = np.mean(trajectory[i:i + window])
            regime_type = "exponential_growth" if slope > 0.1 else \
                          "exponential_decay" if slope < -0.1 else "stable"
            regimes.append({
                "start_layer": i,
                "end_layer": i + window,
                "type": regime_type,
                "growth_rate": float(slope),
                "mean_variance": float(mean_var),
            })

        return {
            "regimes": regimes,
            "n_regimes": len(regimes),
            "overall_behavior": regimes[-1]["type"] if regimes else "unknown",
        }

    def compute_crossover_function(
        self, activation_fn: Callable, sigma_b: float,
        sigma_w_values: np.ndarray, depth: int
    ) -> Dict[str, np.ndarray]:
        """Compute crossover function between ordered and chaotic phases."""
        prop = VariancePropagator(activation_fn, self.n_samples)
        chi_values = []
        depth_scales = []
        for sw in sigma_w_values:
            q_star = prop.find_fixed_point(sw, sigma_b)
            chi = prop.compute_chi(q_star, sw)
            chi_values.append(chi)
            if abs(chi) < 1 and chi > 0:
                depth_scales.append(-1.0 / np.log(chi))
            else:
                depth_scales.append(float(depth))

        return {
            "sigma_w_values": sigma_w_values,
            "chi_values": np.array(chi_values),
            "depth_scales": np.array(depth_scales),
        }


class RGFlowTracker:
    """Track RG flow of effective couplings as function of depth."""

    def __init__(self, n_samples: int = 20000):
        self.n_samples = n_samples

    def track_effective_couplings(
        self, activation_fn: Callable, sigma_w: float, sigma_b: float,
        depth: int, width: int
    ) -> List[Dict[str, float]]:
        """Track how effective couplings evolve through depth."""
        prop = VariancePropagator(activation_fn, self.n_samples)
        couplings = []

        q = 1.0
        for layer in range(depth):
            chi = prop.compute_chi(q, sigma_w)
            q_next = prop.compute_variance_map(q, sigma_w, sigma_b)
            effective_sigma_w = sigma_w * np.sqrt(q / (q_next + 1e-12))

            couplings.append({
                "layer": layer,
                "q": float(q),
                "chi_1": float(chi),
                "effective_sigma_w": float(effective_sigma_w),
                "effective_sigma_b": float(sigma_b),
                "delta_q": float(abs(q_next - q)),
                "lyapunov": float(np.log(abs(chi) + 1e-12)),
            })
            q = q_next

        return couplings

    def compute_flow_diagram(
        self, activation_fn: Callable, sigma_b: float,
        sigma_w_range: Tuple[float, float] = (0.5, 3.0),
        q_range: Tuple[float, float] = (0.1, 5.0),
        n_grid: int = 20
    ) -> Dict[str, np.ndarray]:
        """Compute RG flow diagram in (sigma_w, q) space."""
        prop = VariancePropagator(activation_fn, self.n_samples)
        sigma_w_vals = np.linspace(sigma_w_range[0], sigma_w_range[1], n_grid)
        q_vals = np.linspace(q_range[0], q_range[1], n_grid)

        dq = np.zeros((n_grid, n_grid))
        dsw = np.zeros((n_grid, n_grid))

        for i, sw in enumerate(sigma_w_vals):
            for j, q in enumerate(q_vals):
                q_next = prop.compute_variance_map(q, sw, sigma_b)
                chi = prop.compute_chi(q, sw)
                dq[i, j] = q_next - q
                dsw[i, j] = sw * (chi - 1.0) * 0.1

        return {
            "sigma_w_grid": sigma_w_vals,
            "q_grid": q_vals,
            "dq": dq,
            "dsw": dsw,
        }

    def find_flow_fixed_points(
        self, activation_fn: Callable, sigma_b: float,
        sigma_w_range: Tuple[float, float] = (0.5, 3.0),
        n_search: int = 20
    ) -> List[Dict[str, float]]:
        """Find fixed points of the RG flow."""
        prop = VariancePropagator(activation_fn, self.n_samples)
        fixed_points = []
        sigma_w_vals = np.linspace(sigma_w_range[0], sigma_w_range[1], n_search)

        for sw in sigma_w_vals:
            q_star = prop.find_fixed_point(sw, sigma_b)
            chi = prop.compute_chi(q_star, sw)
            if abs(chi - 1.0) < 0.1:
                stability = "marginal"
            elif chi < 1.0:
                stability = "stable"
            else:
                stability = "unstable"
            fixed_points.append({
                "sigma_w": float(sw),
                "q_star": float(q_star),
                "chi_1": float(chi),
                "stability": stability,
            })

        unique_fps = []
        for fp in fixed_points:
            is_duplicate = False
            for ufp in unique_fps:
                if abs(fp["q_star"] - ufp["q_star"]) < 0.1 and abs(fp["sigma_w"] - ufp["sigma_w"]) < 0.1:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_fps.append(fp)

        return unique_fps


class RenormalizationGroup:
    """Main RG analysis class for neural networks."""

    def __init__(self, n_samples: int = 20000, block_size: int = 2,
                 cutoff_fraction: float = 0.5):
        self.n_samples = n_samples
        self.block_rg = BlockSpinRG(block_size)
        self.wilson_rg = WilsonRG(cutoff_fraction)
        self.exponent_computer = CriticalExponentComputer(n_samples)
        self.universality_detector = UniversalityClassDetector()
        self.scaling_analyzer = ScalingFunctionAnalyzer(n_samples)
        self.crossover_analyzer = CrossoverAnalyzer(n_samples)
        self.flow_tracker = RGFlowTracker(n_samples)

    def analyze(self, model_spec: ModelSpec) -> RGReport:
        """Full RG analysis of a neural network architecture."""
        report = RGReport()
        activation_fn = ActivationFunctions.get_activation(model_spec.activation)
        prop = VariancePropagator(activation_fn, self.n_samples)

        q_star = prop.find_fixed_point(model_spec.sigma_w, model_spec.sigma_b)
        chi = prop.compute_chi(q_star, model_spec.sigma_w)

        if chi < 0.95:
            report.phase = "ordered"
        elif chi > 1.05:
            report.phase = "chaotic"
        else:
            report.phase = "critical"
            report.is_at_criticality = True

        if abs(chi) < 1.0 and chi > 0:
            report.depth_scale = -1.0 / np.log(chi)
        else:
            report.depth_scale = float(model_spec.depth)
        report.correlation_length = report.depth_scale

        couplings = self.flow_tracker.track_effective_couplings(
            activation_fn, model_spec.sigma_w, model_spec.sigma_b,
            model_spec.depth, model_spec.width
        )
        report.effective_couplings = couplings
        report.rg_flow_trajectory = couplings

        W_test = np.random.randn(model_spec.width, model_spec.width) * model_spec.sigma_w / np.sqrt(model_spec.width)
        block_flow = self.block_rg.full_rg_flow(W_test, n_steps=4)
        fp_result = self.block_rg.detect_fixed_point_from_flow(block_flow)
        report.block_rg_results = {
            "flow": block_flow,
            "fixed_point": fp_result,
        }

        wilson_flow = self.wilson_rg.wilson_rg_flow(W_test, model_spec.sigma_w, n_steps=4)
        beta = self.wilson_rg.compute_beta_function(W_test, model_spec.sigma_w)
        report.wilson_rg_results = {
            "flow": wilson_flow,
            "beta_function": beta,
        }

        report.fixed_points.append({
            "q_star": float(q_star),
            "chi_1": float(chi),
            "stability": "stable" if chi < 1.0 else "unstable",
            "sigma_w": float(model_spec.sigma_w),
        })

        try:
            nu_result = self.exponent_computer.compute_correlation_length_exponent(
                activation_fn, model_spec.sigma_b
            )
            sigma_w_c = nu_result["sigma_w_c"]
            beta_result = self.exponent_computer.compute_order_parameter_exponent(
                activation_fn, model_spec.sigma_b, sigma_w_c
            )
            gamma_result = self.exponent_computer.compute_susceptibility_exponent(
                activation_fn, model_spec.sigma_b, sigma_w_c
            )
            exponents = {
                "nu": nu_result["nu"],
                "beta": beta_result["beta"],
                "gamma": gamma_result["gamma"],
            }
            report.critical_exponents = exponents

            uclass, dist = self.universality_detector.classify(exponents)
            report.universality_class = uclass

            scaling_relations = self.universality_detector.check_scaling_relations(exponents)
            report.scaling_function_params = scaling_relations
        except Exception as e:
            report.critical_exponents = {"error": str(e)}
            report.universality_class = "undetermined"

        try:
            crossover = self.crossover_analyzer.detect_crossover_scale(
                activation_fn, model_spec.sigma_w, model_spec.sigma_b
            )
            report.crossover_scales = [float(crossover["crossover_depth"])]
        except Exception:
            report.crossover_scales = []

        return report

    def analyze_weight_matrix(self, W: np.ndarray, sigma_w: float = 1.0) -> Dict[str, Any]:
        """Analyze a specific weight matrix with RG methods."""
        block_flow = self.block_rg.full_rg_flow(W, n_steps=5)
        wilson_flow = self.wilson_rg.wilson_rg_flow(W, sigma_w, n_steps=5)
        beta = self.wilson_rg.compute_beta_function(W, sigma_w)
        fp = self.block_rg.detect_fixed_point_from_flow(block_flow)

        return {
            "block_rg_flow": block_flow,
            "wilson_rg_flow": wilson_flow,
            "beta_function": beta,
            "fixed_point": fp,
        }

    def compare_architectures(self, specs: List[ModelSpec]) -> List[RGReport]:
        """Compare RG analysis across multiple architectures."""
        return [self.analyze(spec) for spec in specs]

    def scan_parameter_space(
        self, base_spec: ModelSpec,
        sigma_w_range: Tuple[float, float] = (0.5, 3.0),
        n_points: int = 20
    ) -> Dict[str, Any]:
        """Scan parameter space to find phase boundaries."""
        sigma_w_vals = np.linspace(sigma_w_range[0], sigma_w_range[1], n_points)
        activation_fn = ActivationFunctions.get_activation(base_spec.activation)
        prop = VariancePropagator(activation_fn, self.n_samples)

        results = {"sigma_w": [], "chi_1": [], "q_star": [], "phase": []}
        for sw in sigma_w_vals:
            q_star = prop.find_fixed_point(sw, base_spec.sigma_b)
            chi = prop.compute_chi(q_star, sw)
            phase = "ordered" if chi < 0.95 else ("chaotic" if chi > 1.05 else "critical")
            results["sigma_w"].append(float(sw))
            results["chi_1"].append(float(chi))
            results["q_star"].append(float(q_star))
            results["phase"].append(phase)

        return results
