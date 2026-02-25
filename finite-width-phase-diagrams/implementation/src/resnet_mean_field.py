"""
ResNet mean field theory — variance recursion with skip connections.

For a pre-activation ResNet block:
    h^{l+1} = h^l + alpha * (W^l * phi(h^l) + b^l)

The variance recursion becomes:
    q^{l+1} = q^l + 2*alpha*sigma_w^2*C(q^l) + alpha^2*(sigma_w^2*V(q^l) + sigma_b^2)

where:
    V(q) = E[phi(z)^2],  z ~ N(0, q)
    C(q) = E[z * phi(z)],  z ~ N(0, q)   (cross-covariance term)

The Jacobian chi_1 for ResNets is:
    chi_1^res = 1 + 2*alpha*sigma_w^2*C'(q*) + alpha^2*sigma_w^2*E[phi'(z)^2]

where C'(q) = dC/dq.

References:
    Yang & Schoenholz, "Mean Field Residual Networks", 2017
    Hayou et al., "On the Impact of the Activation Function on Deep Neural Networks Training", ICML 2019
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erf
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Callable
import warnings

from mean_field_theory import (
    ActivationVarianceMaps,
    MeanFieldAnalyzer,
    ArchitectureSpec,
    MFReport,
    PhaseClassification,
    ConfidenceInterval,
)


@dataclass
class ResNetMFReport:
    """Mean field analysis report for ResNets."""
    # Architecture
    depth: int
    width: int
    activation: str
    sigma_w: float
    sigma_b: float
    alpha: float  # residual scaling

    # Fixed point analysis
    fixed_point_plain: float  # MLP fixed point
    fixed_point_resnet: float  # ResNet fixed point

    # Susceptibility
    chi_1_plain: float
    chi_1_resnet: float
    chi_1_resnet_effective: float  # effective per-block chi

    # Phase classification
    phase_plain: str
    phase_resnet: str

    # Trajectories
    variance_trajectory_plain: List[float] = field(default_factory=list)
    variance_trajectory_resnet: List[float] = field(default_factory=list)

    # Depth scale improvement
    depth_scale_plain: float = 0.0
    depth_scale_resnet: float = 0.0
    depth_improvement_factor: float = 1.0

    # Edge of chaos
    eoc_sigma_w_plain: float = 0.0
    eoc_sigma_w_resnet: float = 0.0

    # Finite-width corrected
    fw_variance_trajectory_resnet: Optional[List[float]] = None

    # Phase diagram data
    phase_diagram_data: Optional[List[Dict]] = None


class ResNetMeanField:
    """Mean field theory for ResNets with skip connections.

    Implements the modified variance recursion:
        q^{l+1} = q^l + 2*alpha*sigma_w^2*C(q^l) + alpha^2*(sigma_w^2*V(q^l) + sigma_b^2)

    where alpha is the residual scaling factor.
    """

    def __init__(self, tolerance: float = 1e-8, max_iterations: int = 10000):
        self.tol = tolerance
        self.max_iter = max_iterations
        self._var_maps = ActivationVarianceMaps()

    def cross_covariance(self, activation: str, q: float) -> float:
        """Compute C(q) = E[z * phi(z)] for z ~ N(0, q).

        This is the cross-covariance between the input and the activation output.
        """
        if q <= 0:
            return 0.0

        if activation == "relu":
            # E[z * ReLU(z)] = E[z^2 * 1_{z>0}] = q/2
            return q / 2.0

        if activation == "linear":
            return q

        act_funcs = {
            "tanh": lambda x: np.tanh(x),
            "gelu": lambda x: 0.5 * x * (1.0 + erf(x / np.sqrt(2.0))),
            "silu": lambda x: x / (1.0 + np.exp(-np.clip(x, -500, 500))),
            "swish": lambda x: x / (1.0 + np.exp(-np.clip(x, -500, 500))),
        }

        if activation not in act_funcs:
            return q / 2.0  # default

        phi = act_funcs[activation]

        def integrand(z):
            x = np.sqrt(q) * z
            return x * phi(x) * np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)

        result, _ = quad(integrand, -8, 8)
        return result

    def cross_covariance_derivative(self, activation: str, q: float) -> float:
        """Compute dC/dq numerically."""
        eps = max(abs(q) * 1e-6, 1e-8)
        c_plus = self.cross_covariance(activation, q + eps)
        c_minus = self.cross_covariance(activation, max(q - eps, 1e-30))
        return (c_plus - c_minus) / (2 * eps)

    def resnet_variance_map(self, q: float, sigma_w: float, sigma_b: float,
                             alpha: float, activation: str) -> float:
        """Single step of ResNet variance recursion.

        q^{l+1} = q^l + 2*alpha*sigma_w^2*C(q) + alpha^2*(sigma_w^2*V(q) + sigma_b^2)
        """
        V_func = self._get_variance_map(activation)
        V_q = V_func(q)
        C_q = self.cross_covariance(activation, q)
        return q + 2 * alpha * sigma_w**2 * C_q + alpha**2 * (sigma_w**2 * V_q + sigma_b**2)

    def resnet_chi1(self, q: float, sigma_w: float, alpha: float,
                     activation: str) -> float:
        """Compute effective chi_1 for one ResNet block.

        chi_1^res = 1 + 2*alpha*sigma_w^2*C'(q) + alpha^2*sigma_w^2*E[phi'(z)^2]
        """
        chi_func = self._get_chi_map(activation)
        E_dphi_sq = chi_func(q)  # E[phi'(z)^2]
        C_prime = self.cross_covariance_derivative(activation, q)
        return 1.0 + 2 * alpha * sigma_w**2 * C_prime + alpha**2 * sigma_w**2 * E_dphi_sq

    def find_resnet_fixed_point(self, sigma_w: float, sigma_b: float,
                                 alpha: float, activation: str,
                                 depth: int = 100) -> float:
        """Find the variance at large depth for a ResNet.

        Unlike plain MLPs, ResNets may not have a true fixed point —
        variance can grow linearly. We iterate and report the value
        at the target depth, or the growth rate.
        """
        V_func = self._get_variance_map(activation)
        q = 1.0

        for l in range(depth):
            q_next = self.resnet_variance_map(q, sigma_w, sigma_b, alpha, activation)
            if abs(q_next - q) < self.tol and abs(q_next - q) / max(abs(q), 1e-30) < self.tol:
                return q_next
            if q_next > 1e10:
                return q_next
            q = q_next

        return q

    def resnet_variance_propagation(self, depth: int, sigma_w: float,
                                     sigma_b: float, alpha: float,
                                     activation: str, q0: float = 1.0) -> List[float]:
        """Propagate variance through a ResNet."""
        trajectory = [q0]
        q = q0

        for l in range(depth):
            q_next = self.resnet_variance_map(q, sigma_w, sigma_b, alpha, activation)
            trajectory.append(max(min(q_next, 1e15), 1e-30))
            q = max(min(q_next, 1e15), 1e-30)

        return trajectory

    def resnet_fw_variance_propagation(self, depth: int, sigma_w: float,
                                        sigma_b: float, alpha: float,
                                        activation: str, width: int,
                                        q0: float = 1.0) -> List[float]:
        """Propagate variance through a ResNet with finite-width corrections."""
        trajectory = [q0]
        q = q0
        N = max(width, 1)
        cumulative_correction = 0.0

        for l in range(depth):
            V_func = self._get_variance_map(activation)
            V_q = V_func(q)
            C_q = self.cross_covariance(activation, q)

            q_mf = q + 2 * alpha * sigma_w**2 * C_q + alpha**2 * (sigma_w**2 * V_q + sigma_b**2)

            if q < 1e6 and np.isfinite(q):
                # O(1/N) correction
                kappa = ActivationVarianceMaps.get_kurtosis_excess(activation, q)
                c1 = alpha**2 * sigma_w**4 * kappa * V_q**2 / N if np.isfinite(V_q) else 0.0

                # O(1/N^2) correction
                eta = ActivationVarianceMaps.get_hyper_kurtosis(activation, q)
                c2 = alpha**2 * sigma_w**6 * abs(eta) * min(V_q, 1e10)**3 / (N**2)

                chi_l = self.resnet_chi1(q, sigma_w, alpha, activation)
                cumulative_correction = chi_l * cumulative_correction + c1
                effective_correction = cumulative_correction / max(l + 1, 1) + c2

                # Clamp: perturbative expansion validity
                max_correction = abs(q_mf) * 0.5
                effective_correction = np.clip(effective_correction, -max_correction, max_correction)
            else:
                effective_correction = 0.0

            q_next = q_mf + effective_correction
            trajectory.append(max(min(q_next, 1e15), 1e-30))
            q = max(min(q_next, 1e15), 1e-30)

        return trajectory

    def find_resnet_edge_of_chaos(self, activation: str, alpha: float,
                                   sigma_b: float = 0.0,
                                   sigma_w_range: Tuple[float, float] = (0.1, 5.0)) -> float:
        """Find sigma_w* where chi_1^res = 1 for a ResNet.

        For a ResNet block, chi_1^res = 1 is the criticality condition.
        Since chi_1^res = 1 + 2*alpha*C'*sigma_w^2 + alpha^2*sigma_w^2*E[phi'^2],
        the edge of chaos satisfies:
            2*alpha*C'*sigma_w^2 + alpha^2*sigma_w^2*E[phi'^2] = 0

        This only has solution sigma_w^2 > 0 if C' < 0, which doesn't happen
        for standard activations. For positive C' and chi > 0, the ResNet is
        always in the chaotic phase for sigma_w > 0. We find the sigma_w
        closest to criticality.
        """
        def chi_resnet_minus_1(sw):
            q = self.find_resnet_fixed_point(sw, sigma_b, alpha, activation)
            q = min(q, 1e6)
            return self.resnet_chi1(q, sw, alpha, activation) - 1.0

        try:
            lo_val = chi_resnet_minus_1(sigma_w_range[0])
            hi_val = chi_resnet_minus_1(sigma_w_range[1])

            if lo_val * hi_val < 0:
                return brentq(chi_resnet_minus_1, sigma_w_range[0], sigma_w_range[1])
            else:
                # Find minimum |chi - 1|
                from scipy.optimize import minimize_scalar
                result = minimize_scalar(
                    lambda sw: abs(chi_resnet_minus_1(sw)),
                    bounds=sigma_w_range,
                    method="bounded",
                )
                return result.x
        except (ValueError, RuntimeError):
            return 1.0

    def analyze(self, depth: int, width: int, activation: str,
                sigma_w: float, sigma_b: float = 0.0,
                alpha: float = 1.0) -> ResNetMFReport:
        """Full comparative analysis: plain MLP vs ResNet."""
        analyzer = MeanFieldAnalyzer()

        # Plain MLP analysis
        arch_plain = ArchitectureSpec(
            depth=depth, width=width, activation=activation,
            sigma_w=sigma_w, sigma_b=sigma_b, has_residual=False,
        )
        report_plain = analyzer.analyze(arch_plain)

        # ResNet variance propagation
        var_traj_resnet = self.resnet_variance_propagation(
            depth, sigma_w, sigma_b, alpha, activation
        )

        # ResNet finite-width corrected propagation
        fw_var_traj_resnet = self.resnet_fw_variance_propagation(
            depth, sigma_w, sigma_b, alpha, activation, width
        )

        # ResNet chi_1
        q_resnet = var_traj_resnet[-1] if var_traj_resnet else 1.0
        q_resnet = min(q_resnet, 1e6)
        chi_1_resnet = self.resnet_chi1(q_resnet, sigma_w, alpha, activation)

        # ResNet phase classification
        if abs(chi_1_resnet - 1.0) < 0.01:
            phase_resnet = "critical"
        elif chi_1_resnet < 1.0:
            phase_resnet = "ordered"
        else:
            phase_resnet = "chaotic"

        # Edge of chaos
        eoc_plain, _ = analyzer.find_edge_of_chaos(activation, sigma_b)
        eoc_resnet = self.find_resnet_edge_of_chaos(activation, alpha, sigma_b)

        # Depth scales
        def depth_scale(chi):
            if abs(chi - 1.0) < 1e-10:
                return float("inf")
            if chi <= 0:
                return 0.0
            log_chi = np.log(chi)
            return -1.0 / log_chi if abs(log_chi) > 1e-30 else float("inf")

        ds_plain = depth_scale(report_plain.chi_1)
        ds_resnet = depth_scale(chi_1_resnet)

        return ResNetMFReport(
            depth=depth, width=width, activation=activation,
            sigma_w=sigma_w, sigma_b=sigma_b, alpha=alpha,
            fixed_point_plain=report_plain.fixed_point,
            fixed_point_resnet=q_resnet,
            chi_1_plain=report_plain.chi_1,
            chi_1_resnet=chi_1_resnet,
            chi_1_resnet_effective=chi_1_resnet,
            phase_plain=report_plain.phase,
            phase_resnet=phase_resnet,
            variance_trajectory_plain=report_plain.variance_trajectory,
            variance_trajectory_resnet=var_traj_resnet,
            depth_scale_plain=ds_plain,
            depth_scale_resnet=ds_resnet,
            depth_improvement_factor=ds_resnet / max(ds_plain, 1e-10) if np.isfinite(ds_resnet) and np.isfinite(ds_plain) else 1.0,
            eoc_sigma_w_plain=eoc_plain,
            eoc_sigma_w_resnet=eoc_resnet,
            fw_variance_trajectory_resnet=fw_var_traj_resnet,
        )

    def resnet_phase_diagram(self, activation: str, alpha: float,
                              sigma_w_values: Optional[List[float]] = None,
                              depths: Optional[List[int]] = None,
                              width: int = 512) -> List[Dict]:
        """Generate phase diagram data for ResNet across sigma_w and depth."""
        if sigma_w_values is None:
            sigma_w_values = np.linspace(0.3, 3.0, 30).tolist()
        if depths is None:
            depths = [5, 10, 20, 50]

        results = []
        for depth in depths:
            for sw in sigma_w_values:
                report = self.analyze(depth, width, activation, sw, alpha=alpha)
                results.append({
                    "sigma_w": sw,
                    "depth": depth,
                    "alpha": alpha,
                    "activation": activation,
                    "chi_1_plain": report.chi_1_plain,
                    "chi_1_resnet": report.chi_1_resnet,
                    "phase_plain": report.phase_plain,
                    "phase_resnet": report.phase_resnet,
                    "var_final_plain": report.variance_trajectory_plain[-1],
                    "var_final_resnet": report.variance_trajectory_resnet[-1],
                    "depth_scale_plain": report.depth_scale_plain if np.isfinite(report.depth_scale_plain) else -1,
                    "depth_scale_resnet": report.depth_scale_resnet if np.isfinite(report.depth_scale_resnet) else -1,
                })
        return results

    def _get_variance_map(self, activation: str) -> Callable[[float], float]:
        maps = {
            "relu": self._var_maps.relu_variance,
            "tanh": self._var_maps.tanh_variance,
            "gelu": self._var_maps.gelu_variance,
            "silu": self._var_maps.silu_variance,
            "swish": self._var_maps.silu_variance,
            "linear": self._var_maps.linear_variance,
        }
        return maps.get(activation, self._var_maps.relu_variance)

    def _get_chi_map(self, activation: str) -> Callable[[float], float]:
        maps = {
            "relu": self._var_maps.relu_chi,
            "tanh": self._var_maps.tanh_chi,
            "gelu": self._var_maps.gelu_chi,
            "silu": self._var_maps.silu_chi,
            "swish": self._var_maps.silu_chi,
            "linear": self._var_maps.linear_chi,
        }
        return maps.get(activation, self._var_maps.relu_chi)
