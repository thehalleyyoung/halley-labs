"""
Theory of activation functions for neural networks.

Implements variance propagation, mean field fixed point analysis, Jacobian analysis,
depth scale computation, curvature analysis, gradient flow analysis,
optimal activation selection, and activation function design.
"""

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.integrate import quad
from scipy.special import erf, erfc
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
import warnings


@dataclass
class ActivationReport:
    """Report from activation function analysis."""
    name: str = ""
    fixed_point_q: float = 0.0
    chi_1: float = 0.0
    phase: str = "unknown"
    depth_scale: float = 0.0
    max_trainable_depth: int = 0
    critical_sigma_w: float = 0.0
    variance_trajectory: List[float] = field(default_factory=list)
    gradient_trajectory: List[float] = field(default_factory=list)
    curvature_metrics: Dict[str, float] = field(default_factory=dict)
    optimal_sigma_w: float = 0.0
    optimal_sigma_b: float = 0.0
    edge_of_chaos_params: Dict[str, float] = field(default_factory=dict)
    comparison_score: float = 0.0


@dataclass
class ActivationSpec:
    """Specification for activation function analysis."""
    name: str = "relu"
    params: Dict[str, float] = field(default_factory=dict)
    sigma_w: float = 1.0
    sigma_b: float = 0.0
    depth: int = 100
    width: int = 1000


class ActivationLibrary:
    """Library of activation functions with analytic properties."""

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -500, 500)) - 1))

    @staticmethod
    def selu(x: np.ndarray) -> np.ndarray:
        alpha = 1.6732632423543772
        lam = 1.0507009873554805
        return lam * np.where(x > 0, x, alpha * (np.exp(np.clip(x, -500, 500)) - 1))

    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))

    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def mish(x: np.ndarray) -> np.ndarray:
        sp = np.log1p(np.exp(np.clip(x, -500, 500)))
        return x * np.tanh(sp)

    @staticmethod
    def softplus(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
        return np.log1p(np.exp(np.clip(beta * x, -500, 500))) / beta

    @staticmethod
    def tanh_act(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def sin_act(x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    @staticmethod
    def get(name: str, params: Optional[Dict[str, float]] = None) -> Callable:
        """Get activation function by name."""
        params = params or {}
        mapping = {
            "relu": ActivationLibrary.relu,
            "leaky_relu": lambda x: ActivationLibrary.leaky_relu(x, params.get("alpha", 0.01)),
            "elu": lambda x: ActivationLibrary.elu(x, params.get("alpha", 1.0)),
            "selu": ActivationLibrary.selu,
            "gelu": ActivationLibrary.gelu,
            "silu": ActivationLibrary.silu,
            "swish": ActivationLibrary.silu,
            "mish": ActivationLibrary.mish,
            "softplus": lambda x: ActivationLibrary.softplus(x, params.get("beta", 1.0)),
            "tanh": ActivationLibrary.tanh_act,
            "sigmoid": ActivationLibrary.sigmoid,
            "sin": ActivationLibrary.sin_act,
        }
        if name not in mapping:
            raise ValueError(f"Unknown activation: {name}. Available: {list(mapping.keys())}")
        return mapping[name]

    @staticmethod
    def all_names() -> List[str]:
        return ["relu", "leaky_relu", "elu", "selu", "gelu", "silu",
                "mish", "softplus", "tanh", "sigmoid", "sin"]


class VariancePropagationAnalyzer:
    """Analyze how variance propagates through layers for different activations."""

    def __init__(self, n_samples: int = 100000):
        self.n_samples = n_samples

    def compute_variance_map(self, activation_fn: Callable,
                              q_in: float, sigma_w: float, sigma_b: float) -> float:
        """Compute output variance given input variance."""
        z = np.random.randn(self.n_samples) * np.sqrt(max(q_in, 1e-12))
        activated = activation_fn(z)
        return float(sigma_w ** 2 * np.mean(activated ** 2) + sigma_b ** 2)

    def compute_variance_map_derivative(self, activation_fn: Callable,
                                         q: float, sigma_w: float, sigma_b: float,
                                         dq: float = 0.01) -> float:
        """Compute derivative of variance map dV(q)/dq."""
        v_plus = self.compute_variance_map(activation_fn, q + dq, sigma_w, sigma_b)
        v_minus = self.compute_variance_map(activation_fn, max(1e-6, q - dq), sigma_w, sigma_b)
        return (v_plus - v_minus) / (2 * dq)

    def propagate_variance(self, activation_fn: Callable,
                            q_init: float, sigma_w: float, sigma_b: float,
                            n_layers: int) -> List[float]:
        """Propagate variance through n_layers."""
        trajectory = [q_init]
        q = q_init
        for _ in range(n_layers):
            q = self.compute_variance_map(activation_fn, q, sigma_w, sigma_b)
            trajectory.append(q)
            if not np.isfinite(q) or q > 1e10:
                break
        return trajectory

    def compute_variance_map_analytic_relu(self, q: float, sigma_w: float,
                                           sigma_b: float) -> float:
        """Analytic variance map for ReLU: sigma_w^2 * q / 2 + sigma_b^2."""
        return sigma_w ** 2 * q / 2.0 + sigma_b ** 2

    def compute_variance_map_analytic_tanh(self, q: float, sigma_w: float,
                                           sigma_b: float) -> float:
        """Approximate variance map for tanh using integral."""
        if q < 0.01:
            return sigma_w ** 2 * q + sigma_b ** 2
        def integrand(z):
            return np.tanh(np.sqrt(q) * z) ** 2 * np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
        result, _ = quad(integrand, -6, 6)
        return sigma_w ** 2 * result + sigma_b ** 2


class MeanFieldFixedPointAnalyzer:
    """Find and analyze mean field fixed points for different activations."""

    def __init__(self, n_samples: int = 100000):
        self.n_samples = n_samples
        self.var_prop = VariancePropagationAnalyzer(n_samples)

    def find_fixed_point(self, activation_fn: Callable,
                          sigma_w: float, sigma_b: float,
                          q_range: Tuple[float, float] = (0.001, 50.0),
                          n_grid: int = 200) -> float:
        """Find fixed point q* where V(q*) = q*."""
        q_vals = np.logspace(np.log10(q_range[0]), np.log10(q_range[1]), n_grid)
        diffs = []
        for q in q_vals:
            v = self.var_prop.compute_variance_map(activation_fn, q, sigma_w, sigma_b)
            diffs.append(v - q)

        diffs = np.array(diffs)
        sign_changes = np.where(np.diff(np.sign(diffs)))[0]

        if len(sign_changes) == 0:
            return float(q_vals[np.argmin(np.abs(diffs))])

        q_star_candidates = []
        for idx in sign_changes:
            try:
                q_star = brentq(
                    lambda q: self.var_prop.compute_variance_map(activation_fn, q, sigma_w, sigma_b) - q,
                    q_vals[idx], q_vals[idx + 1]
                )
                q_star_candidates.append(q_star)
            except (ValueError, RuntimeError):
                continue

        if not q_star_candidates:
            return float(q_vals[np.argmin(np.abs(diffs))])

        return float(q_star_candidates[-1])

    def analyze_fixed_point_stability(self, activation_fn: Callable,
                                       q_star: float, sigma_w: float,
                                       sigma_b: float) -> Dict[str, float]:
        """Analyze stability of fixed point."""
        deriv = self.var_prop.compute_variance_map_derivative(
            activation_fn, q_star, sigma_w, sigma_b
        )
        is_stable = deriv < 1.0

        return {
            "q_star": float(q_star),
            "derivative": float(deriv),
            "is_stable": bool(is_stable),
            "convergence_rate": float(abs(1.0 - deriv)) if is_stable else float("inf"),
        }

    def find_all_fixed_points(self, activation_fn: Callable,
                               sigma_w: float, sigma_b: float,
                               q_range: Tuple[float, float] = (0.001, 50.0),
                               n_grid: int = 200) -> List[Dict[str, float]]:
        """Find all fixed points and classify stability."""
        q_vals = np.logspace(np.log10(q_range[0]), np.log10(q_range[1]), n_grid)
        diffs = []
        for q in q_vals:
            v = self.var_prop.compute_variance_map(activation_fn, q, sigma_w, sigma_b)
            diffs.append(v - q)

        diffs = np.array(diffs)
        sign_changes = np.where(np.diff(np.sign(diffs)))[0]

        fixed_points = []
        for idx in sign_changes:
            try:
                q_star = brentq(
                    lambda q: self.var_prop.compute_variance_map(activation_fn, q, sigma_w, sigma_b) - q,
                    q_vals[idx], q_vals[idx + 1]
                )
                stability = self.analyze_fixed_point_stability(
                    activation_fn, q_star, sigma_w, sigma_b
                )
                fixed_points.append(stability)
            except (ValueError, RuntimeError):
                continue

        return fixed_points


class JacobianAnalyzer:
    """Analyze Jacobian properties for different activations."""

    def __init__(self, n_samples: int = 100000):
        self.n_samples = n_samples

    def compute_chi1(self, activation_fn: Callable, q: float, sigma_w: float) -> float:
        """Compute chi_1 = sigma_w^2 * E[phi'(z)^2] at variance q."""
        z = np.random.randn(self.n_samples) * np.sqrt(max(q, 1e-12))
        eps = 1e-5
        dphi = (activation_fn(z + eps) - activation_fn(z - eps)) / (2 * eps)
        return float(sigma_w ** 2 * np.mean(dphi ** 2))

    def compute_chi1_analytic_relu(self, sigma_w: float) -> float:
        """Analytic chi_1 for ReLU: sigma_w^2 / 2."""
        return sigma_w ** 2 / 2.0

    def compute_chi1_analytic_tanh(self, q: float, sigma_w: float) -> float:
        """Compute chi_1 for tanh using integration."""
        def integrand(z):
            return (1 - np.tanh(np.sqrt(q) * z) ** 2) ** 2 * \
                   np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
        result, _ = quad(integrand, -6, 6)
        return float(sigma_w ** 2 * result)

    def compute_jacobian_spectrum(self, activation_fn: Callable,
                                   width: int, sigma_w: float, sigma_b: float,
                                   n_layers: int = 1) -> Dict[str, Any]:
        """Compute Jacobian spectrum for a random network layer."""
        W = np.random.randn(width, width) * sigma_w / np.sqrt(width)
        z = np.random.randn(width) * np.sqrt(sigma_w)

        eps = 1e-5
        dphi = (activation_fn(z + eps) - activation_fn(z - eps)) / (2 * eps)
        D = np.diag(dphi)
        J = D @ W

        eigenvalues = np.linalg.eigvals(J)
        magnitudes = np.abs(eigenvalues)
        spectral_radius = float(np.max(magnitudes))
        mean_magnitude = float(np.mean(magnitudes))

        return {
            "spectral_radius": spectral_radius,
            "mean_magnitude": mean_magnitude,
            "max_real_part": float(np.max(np.real(eigenvalues))),
            "fraction_growing": float(np.mean(magnitudes > 1.0)),
            "condition_number": float(np.max(magnitudes) / (np.min(magnitudes) + 1e-12)),
        }

    def compute_lyapunov_exponent(self, activation_fn: Callable,
                                    sigma_w: float, sigma_b: float,
                                    depth: int, width: int = 200) -> float:
        """Compute maximum Lyapunov exponent via Jacobian products."""
        z = np.random.randn(width)
        v = np.random.randn(width)
        v = v / np.linalg.norm(v)

        lyapunov_sum = 0.0
        for _ in range(depth):
            W = np.random.randn(width, width) * sigma_w / np.sqrt(width)
            b = np.random.randn(width) * sigma_b

            pre = W @ z + b
            eps = 1e-5
            dphi = (activation_fn(pre + eps) - activation_fn(pre - eps)) / (2 * eps)
            Jv = dphi * (W @ v)

            norm = np.linalg.norm(Jv)
            if norm > 0:
                lyapunov_sum += np.log(norm)
                v = Jv / norm
            z = activation_fn(pre)

        return float(lyapunov_sum / depth)


class DepthScaleComputer:
    """Compute maximum trainable depth for different activations."""

    def __init__(self, n_samples: int = 100000):
        self.n_samples = n_samples
        self.mf_analyzer = MeanFieldFixedPointAnalyzer(n_samples)
        self.jac_analyzer = JacobianAnalyzer(n_samples)

    def compute_depth_scale(self, activation_fn: Callable,
                             sigma_w: float, sigma_b: float) -> float:
        """Compute depth scale xi = -1/log|chi_1|."""
        q_star = self.mf_analyzer.find_fixed_point(activation_fn, sigma_w, sigma_b)
        chi = self.jac_analyzer.compute_chi1(activation_fn, q_star, sigma_w)

        if abs(chi) >= 1.0 or chi <= 0:
            return float("inf")
        return -1.0 / np.log(abs(chi))

    def compute_max_trainable_depth(self, activation_fn: Callable,
                                     sigma_w: float, sigma_b: float,
                                     gradient_threshold: float = 1e-6) -> int:
        """Estimate maximum trainable depth before gradient vanishing/exploding."""
        q_star = self.mf_analyzer.find_fixed_point(activation_fn, sigma_w, sigma_b)
        chi = self.jac_analyzer.compute_chi1(activation_fn, q_star, sigma_w)

        if abs(chi - 1.0) < 1e-4:
            return 10000

        if chi < 1.0 and chi > 0:
            max_depth = int(-np.log(gradient_threshold) / (-np.log(chi) + 1e-12))
            return min(max_depth, 10000)
        elif chi > 1.0:
            max_depth = int(np.log(1e6) / (np.log(chi) + 1e-12))
            return min(max_depth, 10000)
        else:
            return 1

    def find_critical_sigma_w(self, activation_fn: Callable,
                               sigma_b: float = 0.0,
                               sigma_w_range: Tuple[float, float] = (0.1, 5.0)) -> float:
        """Find sigma_w that puts network at edge of chaos (chi_1 = 1)."""
        def chi_minus_one(sw):
            q_star = self.mf_analyzer.find_fixed_point(activation_fn, sw, sigma_b)
            chi = self.jac_analyzer.compute_chi1(activation_fn, q_star, sw)
            return chi - 1.0

        sw_vals = np.linspace(sigma_w_range[0], sigma_w_range[1], 40)
        chi_vals = [chi_minus_one(sw) for sw in sw_vals]

        sign_changes = np.where(np.diff(np.sign(chi_vals)))[0]
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            try:
                return float(brentq(chi_minus_one, sw_vals[idx], sw_vals[idx + 1]))
            except (ValueError, RuntimeError):
                pass

        return float(sw_vals[np.argmin(np.abs(chi_vals))])

    def compare_depth_scales(self, activation_names: List[str],
                              sigma_w: float = 1.0, sigma_b: float = 0.0
                              ) -> Dict[str, Dict[str, float]]:
        """Compare depth scales across activations."""
        results = {}
        for name in activation_names:
            fn = ActivationLibrary.get(name)
            q_star = self.mf_analyzer.find_fixed_point(fn, sigma_w, sigma_b)
            chi = self.jac_analyzer.compute_chi1(fn, q_star, sigma_w)
            depth_scale = self.compute_depth_scale(fn, sigma_w, sigma_b)
            max_depth = self.compute_max_trainable_depth(fn, sigma_w, sigma_b)
            critical_sw = self.find_critical_sigma_w(fn, sigma_b)

            results[name] = {
                "q_star": float(q_star),
                "chi_1": float(chi),
                "depth_scale": float(depth_scale),
                "max_trainable_depth": max_depth,
                "critical_sigma_w": float(critical_sw),
                "phase": "ordered" if chi < 0.95 else ("chaotic" if chi > 1.05 else "critical"),
            }
        return results


class CurvatureAnalyzer:
    """Analyze activation function curvature and its effect on loss landscape."""

    def __init__(self, n_samples: int = 50000):
        self.n_samples = n_samples

    def compute_activation_curvature(self, activation_fn: Callable,
                                      q: float) -> Dict[str, float]:
        """Compute curvature statistics of activation at variance q."""
        z = np.random.randn(self.n_samples) * np.sqrt(max(q, 1e-12))
        eps = 1e-4

        phi = activation_fn(z)
        dphi = (activation_fn(z + eps) - activation_fn(z - eps)) / (2 * eps)
        d2phi = (activation_fn(z + eps) - 2 * phi + activation_fn(z - eps)) / (eps ** 2)

        return {
            "mean_curvature": float(np.mean(np.abs(d2phi))),
            "max_curvature": float(np.max(np.abs(d2phi))),
            "curvature_variance": float(np.var(d2phi)),
            "mean_slope": float(np.mean(np.abs(dphi))),
            "slope_variance": float(np.var(dphi)),
            "nonlinearity_index": float(np.mean(d2phi ** 2) / (np.mean(dphi ** 2) + 1e-12)),
            "smoothness": float(1.0 / (np.mean(d2phi ** 2) + 1e-6)),
        }

    def compute_loss_landscape_effect(self, activation_fn: Callable,
                                       width: int, sigma_w: float,
                                       depth: int = 5) -> Dict[str, float]:
        """Estimate effect of activation curvature on loss landscape."""
        z = np.random.randn(self.n_samples) * np.sqrt(sigma_w)
        eps = 1e-4

        dphi = (activation_fn(z + eps) - activation_fn(z - eps)) / (2 * eps)
        d2phi = (activation_fn(z + eps) - 2 * activation_fn(z) +
                 activation_fn(z - eps)) / (eps ** 2)

        mean_grad_sq = np.mean(dphi ** 2)
        mean_hess_sq = np.mean(d2phi ** 2)

        grad_flow_factor = mean_grad_sq ** depth
        hessian_factor = mean_hess_sq ** depth

        gradient_noise = float(np.var(dphi ** 2) * depth)
        curvature_ratio = mean_hess_sq / (mean_grad_sq ** 2 + 1e-12)

        return {
            "gradient_flow_factor": float(grad_flow_factor),
            "hessian_factor": float(hessian_factor),
            "gradient_noise": float(gradient_noise),
            "curvature_ratio": float(curvature_ratio),
            "expected_condition_number": float(np.exp(depth * np.log(
                mean_hess_sq / (mean_grad_sq + 1e-12) + 1e-12))),
        }

    def rank_activations_by_curvature(
        self, activation_names: List[str], q: float = 1.0
    ) -> List[Tuple[str, Dict[str, float]]]:
        """Rank activations by their curvature properties."""
        results = []
        for name in activation_names:
            fn = ActivationLibrary.get(name)
            curvature = self.compute_activation_curvature(fn, q)
            results.append((name, curvature))

        results.sort(key=lambda x: x[1]["smoothness"], reverse=True)
        return results


class GradientFlowAnalyzer:
    """Analyze gradient flow through networks with different activations."""

    def __init__(self, n_samples: int = 50000):
        self.n_samples = n_samples
        self.jac_analyzer = JacobianAnalyzer(n_samples)

    def compute_gradient_magnitude(self, activation_fn: Callable,
                                    sigma_w: float, sigma_b: float,
                                    n_layers: int) -> List[float]:
        """Compute expected gradient magnitude at each layer."""
        mf = MeanFieldFixedPointAnalyzer(self.n_samples)
        q_star = mf.find_fixed_point(activation_fn, sigma_w, sigma_b)
        chi = self.jac_analyzer.compute_chi1(activation_fn, q_star, sigma_w)

        magnitudes = [1.0]
        for layer in range(n_layers):
            magnitudes.append(magnitudes[-1] * chi)

        return magnitudes

    def compute_gradient_statistics(self, activation_fn: Callable,
                                     sigma_w: float, sigma_b: float,
                                     depth: int, width: int = 100
                                     ) -> Dict[str, Any]:
        """Compute detailed gradient statistics through a random network."""
        z = np.random.randn(width)
        grad_norms = []
        layer_outputs = [z.copy()]

        for layer in range(depth):
            W = np.random.randn(width, width) * sigma_w / np.sqrt(width)
            b = np.random.randn(width) * sigma_b
            z = activation_fn(W @ z + b)
            layer_outputs.append(z.copy())

        grad = np.ones(width)
        for layer in range(depth - 1, -1, -1):
            z = layer_outputs[layer]
            W = np.random.randn(width, width) * sigma_w / np.sqrt(width)
            pre = W @ z
            eps = 1e-5
            dphi = (activation_fn(pre + eps) - activation_fn(pre - eps)) / (2 * eps)
            grad = (W.T @ (dphi * grad))
            grad_norms.append(float(np.linalg.norm(grad)))

        grad_norms.reverse()

        return {
            "gradient_norms": grad_norms,
            "mean_gradient_norm": float(np.mean(grad_norms)),
            "gradient_ratio": float(grad_norms[-1] / (grad_norms[0] + 1e-12))
                if grad_norms else 0.0,
            "vanishing": bool(grad_norms[-1] < 1e-6 * grad_norms[0]) if grad_norms else False,
            "exploding": bool(grad_norms[-1] > 1e6 * grad_norms[0]) if grad_norms else False,
        }


class OptimalActivationSelector:
    """Select optimal activation function for given architecture and task."""

    def __init__(self, n_samples: int = 50000):
        self.n_samples = n_samples
        self.depth_computer = DepthScaleComputer(n_samples)
        self.curvature_analyzer = CurvatureAnalyzer(n_samples)
        self.gradient_analyzer = GradientFlowAnalyzer(n_samples)

    def score_activation(self, activation_name: str, depth: int,
                          width: int, sigma_w: float, sigma_b: float = 0.0
                          ) -> Dict[str, float]:
        """Score an activation function for given architecture."""
        fn = ActivationLibrary.get(activation_name)
        mf = MeanFieldFixedPointAnalyzer(self.n_samples)
        jac = JacobianAnalyzer(self.n_samples)

        q_star = mf.find_fixed_point(fn, sigma_w, sigma_b)
        chi = jac.compute_chi1(fn, q_star, sigma_w)

        criticality_score = 1.0 / (abs(chi - 1.0) + 0.01)
        depth_scale = self.depth_computer.compute_depth_scale(fn, sigma_w, sigma_b)
        depth_score = min(1.0, depth_scale / (depth + 1e-12))

        curvature = self.curvature_analyzer.compute_activation_curvature(fn, q_star)
        smoothness_score = curvature["smoothness"]

        gradient_mags = self.gradient_analyzer.compute_gradient_magnitude(
            fn, sigma_w, sigma_b, depth
        )
        gradient_range = max(gradient_mags) / (min(gradient_mags) + 1e-12)
        gradient_score = 1.0 / (1.0 + np.log(gradient_range + 1))

        overall = (0.3 * criticality_score + 0.3 * depth_score +
                   0.2 * smoothness_score + 0.2 * gradient_score)

        return {
            "overall_score": float(overall),
            "criticality_score": float(criticality_score),
            "depth_score": float(depth_score),
            "smoothness_score": float(smoothness_score),
            "gradient_score": float(gradient_score),
            "chi_1": float(chi),
            "depth_scale": float(depth_scale),
        }

    def recommend(self, depth: int, width: int, sigma_w: float = 1.0,
                   sigma_b: float = 0.0,
                   candidates: Optional[List[str]] = None) -> Dict[str, Any]:
        """Recommend best activation for given architecture."""
        if candidates is None:
            candidates = ActivationLibrary.all_names()

        scores = {}
        for name in candidates:
            try:
                score = self.score_activation(name, depth, width, sigma_w, sigma_b)
                scores[name] = score
            except Exception:
                continue

        if not scores:
            return {"recommendation": "relu", "scores": {}, "reason": "fallback"}

        best = max(scores, key=lambda k: scores[k]["overall_score"])

        return {
            "recommendation": best,
            "scores": scores,
            "reason": f"{best} has best overall score ({scores[best]['overall_score']:.3f})",
        }


class ActivationDesigner:
    """Design parameterized activation functions optimized for trainability."""

    def __init__(self, n_samples: int = 50000):
        self.n_samples = n_samples
        self.mf_analyzer = MeanFieldFixedPointAnalyzer(n_samples)
        self.jac_analyzer = JacobianAnalyzer(n_samples)

    def parameterized_activation(self, x: np.ndarray, alpha: float, beta: float,
                                  gamma: float) -> np.ndarray:
        """Parameterized activation: alpha * max(0, x) + beta * tanh(gamma * x)."""
        return alpha * np.maximum(0, x) + beta * np.tanh(gamma * x)

    def optimize_for_criticality(self, sigma_w: float, sigma_b: float = 0.0,
                                  target_chi: float = 1.0) -> Dict[str, float]:
        """Find activation parameters that achieve chi_1 = target_chi."""
        def objective(params):
            alpha, beta, gamma = params
            if alpha < 0 or beta < -2 or gamma < 0.01:
                return 1e10
            fn = lambda x: self.parameterized_activation(x, alpha, beta, gamma)
            try:
                q_star = self.mf_analyzer.find_fixed_point(fn, sigma_w, sigma_b)
                chi = self.jac_analyzer.compute_chi1(fn, q_star, sigma_w)
                return (chi - target_chi) ** 2
            except Exception:
                return 1e10

        best_result = None
        best_obj = float("inf")

        for alpha_init in [0.5, 1.0, 1.5]:
            for beta_init in [0.0, 0.3, 0.5]:
                for gamma_init in [0.5, 1.0, 2.0]:
                    try:
                        result = minimize(
                            objective,
                            x0=[alpha_init, beta_init, gamma_init],
                            bounds=[(0.01, 3.0), (-1.0, 2.0), (0.01, 5.0)],
                            method="L-BFGS-B"
                        )
                        if result.fun < best_obj:
                            best_obj = result.fun
                            best_result = result
                    except Exception:
                        continue

        if best_result is None:
            return {"alpha": 1.0, "beta": 0.0, "gamma": 1.0, "achieved_chi": 0.5}

        alpha, beta, gamma = best_result.x
        fn = lambda x: self.parameterized_activation(x, alpha, beta, gamma)
        q_star = self.mf_analyzer.find_fixed_point(fn, sigma_w, sigma_b)
        chi = self.jac_analyzer.compute_chi1(fn, q_star, sigma_w)

        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
            "achieved_chi": float(chi),
            "q_star": float(q_star),
            "optimization_success": bool(best_result.success),
        }

    def design_self_normalizing(self, sigma_w: float, sigma_b: float = 0.0
                                 ) -> Dict[str, float]:
        """Design activation that maintains unit variance and zero mean."""
        def objective(params):
            alpha, beta, gamma = params
            fn = lambda x: self.parameterized_activation(x, alpha, beta, gamma)
            z = np.random.randn(self.n_samples) * 1.0
            output = fn(z)
            mean_penalty = np.mean(output) ** 2
            var_penalty = (np.mean(output ** 2) - 1.0) ** 2
            q_star = self.mf_analyzer.find_fixed_point(fn, sigma_w, sigma_b)
            chi = self.jac_analyzer.compute_chi1(fn, q_star, sigma_w)
            chi_penalty = (chi - 1.0) ** 2
            return mean_penalty + var_penalty + chi_penalty

        result = minimize(
            objective,
            x0=[1.0, 0.3, 1.0],
            bounds=[(0.01, 3.0), (-1.0, 2.0), (0.01, 5.0)],
            method="L-BFGS-B"
        )

        alpha, beta, gamma = result.x
        fn = lambda x: self.parameterized_activation(x, alpha, beta, gamma)
        z = np.random.randn(self.n_samples)
        output = fn(z)

        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
            "output_mean": float(np.mean(output)),
            "output_var": float(np.var(output)),
            "optimization_success": bool(result.success),
        }


class ActivationAnalyzer:
    """Main activation function analyzer."""

    def __init__(self, n_samples: int = 50000):
        self.n_samples = n_samples
        self.var_prop = VariancePropagationAnalyzer(n_samples)
        self.mf_analyzer = MeanFieldFixedPointAnalyzer(n_samples)
        self.jac_analyzer = JacobianAnalyzer(n_samples)
        self.depth_computer = DepthScaleComputer(n_samples)
        self.curvature_analyzer = CurvatureAnalyzer(n_samples)
        self.gradient_analyzer = GradientFlowAnalyzer(n_samples)
        self.selector = OptimalActivationSelector(n_samples)
        self.designer = ActivationDesigner(n_samples)

    def analyze(self, activation: str, sigma_w: float = 1.0, sigma_b: float = 0.0,
                depth: int = 100, width: int = 1000,
                params: Optional[Dict[str, float]] = None) -> ActivationReport:
        """Full analysis of an activation function."""
        report = ActivationReport()
        report.name = activation

        fn = ActivationLibrary.get(activation, params)

        q_star = self.mf_analyzer.find_fixed_point(fn, sigma_w, sigma_b)
        report.fixed_point_q = q_star

        chi = self.jac_analyzer.compute_chi1(fn, q_star, sigma_w)
        report.chi_1 = chi

        if chi < 0.95:
            report.phase = "ordered"
        elif chi > 1.05:
            report.phase = "chaotic"
        else:
            report.phase = "critical"

        depth_scale = self.depth_computer.compute_depth_scale(fn, sigma_w, sigma_b)
        report.depth_scale = depth_scale

        max_depth = self.depth_computer.compute_max_trainable_depth(fn, sigma_w, sigma_b)
        report.max_trainable_depth = max_depth

        critical_sw = self.depth_computer.find_critical_sigma_w(fn, sigma_b)
        report.critical_sigma_w = critical_sw
        report.optimal_sigma_w = critical_sw
        report.optimal_sigma_b = sigma_b

        trajectory = self.var_prop.propagate_variance(fn, 1.0, sigma_w, sigma_b, min(depth, 200))
        report.variance_trajectory = trajectory

        gradient_mags = self.gradient_analyzer.compute_gradient_magnitude(
            fn, sigma_w, sigma_b, min(depth, 200)
        )
        report.gradient_trajectory = gradient_mags

        curvature = self.curvature_analyzer.compute_activation_curvature(fn, q_star)
        report.curvature_metrics = curvature

        report.edge_of_chaos_params = {
            "critical_sigma_w": critical_sw,
            "chi_at_critical": 1.0,
            "q_star_at_critical": float(
                self.mf_analyzer.find_fixed_point(fn, critical_sw, sigma_b)
            ),
        }

        score = self.selector.score_activation(activation, depth, width, sigma_w, sigma_b)
        report.comparison_score = score["overall_score"]

        return report

    def compare_activations(self, activation_names: Optional[List[str]] = None,
                             sigma_w: float = 1.0, sigma_b: float = 0.0,
                             depth: int = 100) -> Dict[str, ActivationReport]:
        """Compare multiple activation functions."""
        if activation_names is None:
            activation_names = ["relu", "tanh", "gelu", "silu", "elu", "selu"]

        reports = {}
        for name in activation_names:
            try:
                reports[name] = self.analyze(name, sigma_w, sigma_b, depth)
            except Exception as e:
                warnings.warn(f"Failed to analyze {name}: {e}")

        return reports

    def find_optimal_initialization(self, activation: str, depth: int,
                                     sigma_b: float = 0.0) -> Dict[str, float]:
        """Find optimal weight initialization for given activation and depth."""
        fn = ActivationLibrary.get(activation)
        critical_sw = self.depth_computer.find_critical_sigma_w(fn, sigma_b)

        return {
            "optimal_sigma_w": float(critical_sw),
            "optimal_sigma_b": float(sigma_b),
            "he_init_sigma_w": float(np.sqrt(2.0)),
            "lecun_init_sigma_w": 1.0,
            "recommended": "critical" if abs(critical_sw - np.sqrt(2)) > 0.1 else "he",
        }
