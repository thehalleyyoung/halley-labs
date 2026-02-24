"""Mean-field theory for convolutional neural networks.

Extends the MLP mean-field variance recursion to Conv2d layers.  The key
insight (Xiao et al., 2018) is that for Conv2d with i.i.d. Gaussian weights
W_{c_out, c_in, i, j} ~ N(0, sigma_w^2 / (C_in * k * k)), the variance
recursion is identical to the dense case with effective fan-in = C_in * k * k:

    q^{l+1} = sigma_w^2 * V(q^l) + sigma_b^2

where V(q) = E[phi(sqrt(q) z)^2] is activation-dependent and identical to
the MLP case.  The chi_1 Jacobian is also unchanged:

    chi_1 = sigma_w^2 * E[phi'(sqrt(q*) z)^2]

Finite-width corrections differ: the effective width for a Conv2d layer is
C_out (not C_in * k * k), so corrections scale as O(1/C_out).

References:
    Xiao et al., "Dynamical Isometry and a Mean Field Theory of CNNs", ICML 2018
    Yang, "Tensor Programs II: Neural Tangent Kernel for Any Architecture", 2020
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from mean_field_theory import (
    ActivationVarianceMaps,
    MeanFieldAnalyzer,
    ArchitectureSpec,
    MFReport,
    PhaseClassification,
)


@dataclass
class ConvLayerSpec:
    """Specification for a single convolutional layer."""
    in_channels: int
    out_channels: int
    kernel_size: int  # assumes square kernels
    activation: str = "relu"
    stride: int = 1
    padding: int = 0


@dataclass
class ConvArchSpec:
    """Specification for a CNN architecture."""
    layers: List[ConvLayerSpec]
    sigma_w: float = 1.0
    sigma_b: float = 0.0
    input_channels: int = 3
    input_height: int = 32
    input_width: int = 32
    fc_widths: List[int] = field(default_factory=list)
    fc_activation: str = "relu"


@dataclass
class ConvMFReport:
    """Mean-field analysis report for a CNN."""
    # Per-layer analysis
    layer_variances: List[float]
    layer_chi1: List[float]
    layer_phases: List[str]
    layer_effective_widths: List[int]

    # Overall network
    overall_chi1: float  # geometric mean of per-layer chi_1
    overall_phase: str
    depth_scale: float
    max_trainable_depth: int

    # Fixed point
    fixed_point: float

    # Finite-width corrected
    fw_layer_variances: Optional[List[float]] = None

    # Edge of chaos
    eoc_sigma_w: float = 0.0

    # Spatial dimensions through layers
    spatial_dims: Optional[List[Tuple[int, int]]] = None


class ConvMeanField:
    """Mean-field theory for convolutional neural networks.

    For Conv2d with weight scaling sigma_w^2 / (C_in * k^2):
    - The variance map is identical to the MLP case
    - The effective width for finite-width corrections is C_out
    - Spatial dimensions change according to stride/padding
    """

    def __init__(self, tolerance: float = 1e-8, max_iterations: int = 10000):
        self.tol = tolerance
        self.max_iter = max_iterations
        self._var_maps = ActivationVarianceMaps()
        self._analyzer = MeanFieldAnalyzer(tolerance, max_iterations)

    def _get_variance_map(self, activation: str):
        maps = {
            "relu": self._var_maps.relu_variance,
            "tanh": self._var_maps.tanh_variance,
            "gelu": self._var_maps.gelu_variance,
            "silu": self._var_maps.silu_variance,
            "swish": self._var_maps.silu_variance,
            "linear": self._var_maps.linear_variance,
        }
        return maps.get(activation, self._var_maps.relu_variance)

    def _get_chi_map(self, activation: str):
        maps = {
            "relu": self._var_maps.relu_chi,
            "tanh": self._var_maps.tanh_chi,
            "gelu": self._var_maps.gelu_chi,
            "silu": self._var_maps.silu_chi,
            "swish": self._var_maps.silu_chi,
            "linear": self._var_maps.linear_chi,
        }
        return maps.get(activation, self._var_maps.relu_chi)

    def _compute_spatial_dims(self, h: int, w: int, layer: ConvLayerSpec) -> Tuple[int, int]:
        """Compute output spatial dimensions after a conv layer."""
        h_out = (h + 2 * layer.padding - layer.kernel_size) // layer.stride + 1
        w_out = (w + 2 * layer.padding - layer.kernel_size) // layer.stride + 1
        return max(h_out, 1), max(w_out, 1)

    def conv_variance_propagation(self, arch: ConvArchSpec,
                                   q0: float = 1.0) -> List[float]:
        """Propagate variance through a CNN.

        For Conv2d, the variance recursion is identical to the dense case:
            q^{l+1} = sigma_w^2 * V(q^l) + sigma_b^2

        This follows from the weight scaling W ~ N(0, sigma_w^2 / fan_in)
        where fan_in = C_in * k * k, and the sum over fan_in terms
        concentrates to sigma_w^2 * V(q^l) in the infinite-channel limit.
        """
        trajectory = [q0]
        q = q0

        for layer in arch.layers:
            V_func = self._get_variance_map(layer.activation)
            q_next = arch.sigma_w ** 2 * V_func(q) + arch.sigma_b ** 2
            trajectory.append(q_next)
            q = q_next

        # FC layers (if any)
        for _ in arch.fc_widths:
            V_func = self._get_variance_map(arch.fc_activation)
            q_next = arch.sigma_w ** 2 * V_func(q) + arch.sigma_b ** 2
            trajectory.append(q_next)
            q = q_next

        return trajectory

    def conv_fw_variance_propagation(self, arch: ConvArchSpec,
                                      q0: float = 1.0) -> List[float]:
        """Propagate variance with finite-width corrections.

        For Conv2d, the effective width is C_out (number of output channels).
        Corrections scale as O(1/C_out) per layer.
        """
        trajectory = [q0]
        q = q0

        for layer in arch.layers:
            V_func = self._get_variance_map(layer.activation)
            V_q = V_func(q)
            q_mf = arch.sigma_w ** 2 * V_q + arch.sigma_b ** 2

            N_eff = layer.out_channels
            if N_eff > 0 and q < 1e6 and np.isfinite(q):
                kappa = ActivationVarianceMaps.get_kurtosis_excess(
                    layer.activation, q)
                c1 = arch.sigma_w ** 4 * kappa * V_q ** 2 / N_eff
                eta = ActivationVarianceMaps.get_hyper_kurtosis(
                    layer.activation, q)
                c2 = arch.sigma_w ** 6 * abs(eta) * min(V_q, 1e10) ** 3 / (N_eff ** 2)
                correction = c1 + c2
                # Perturbative validity clamp
                if abs(correction) > 0.3 * abs(q_mf):
                    correction = np.sign(correction) * 0.3 * abs(q_mf)
            else:
                correction = 0.0

            q_next = q_mf + correction
            trajectory.append(max(min(q_next, 1e15), 1e-30))
            q = max(min(q_next, 1e15), 1e-30)

        # FC layers
        for fc_w in arch.fc_widths:
            V_func = self._get_variance_map(arch.fc_activation)
            V_q = V_func(q)
            q_mf = arch.sigma_w ** 2 * V_q + arch.sigma_b ** 2
            if fc_w > 0 and q < 1e6:
                kappa = ActivationVarianceMaps.get_kurtosis_excess(
                    arch.fc_activation, q)
                c1 = arch.sigma_w ** 4 * kappa * V_q ** 2 / fc_w
                correction = min(c1, 0.3 * abs(q_mf))
            else:
                correction = 0.0
            q_next = q_mf + correction
            trajectory.append(max(min(q_next, 1e15), 1e-30))
            q = max(min(q_next, 1e15), 1e-30)

        return trajectory

    def analyze(self, arch: ConvArchSpec) -> ConvMFReport:
        """Full mean-field analysis of a CNN architecture."""
        sigma_w = arch.sigma_w
        sigma_b = arch.sigma_b

        # Variance propagation
        var_traj = self.conv_variance_propagation(arch)
        fw_var_traj = self.conv_fw_variance_propagation(arch)

        # Per-layer chi_1 and phase
        layer_chi1 = []
        layer_phases = []
        effective_widths = []

        h, w = arch.input_height, arch.input_width
        spatial_dims = [(h, w)]

        for i, layer in enumerate(arch.layers):
            chi_func = self._get_chi_map(layer.activation)
            q = var_traj[i]
            chi1 = sigma_w ** 2 * chi_func(max(q, 1e-30))
            layer_chi1.append(chi1)

            if abs(chi1 - 1.0) < 0.01:
                layer_phases.append("critical")
            elif chi1 < 1.0:
                layer_phases.append("ordered")
            else:
                layer_phases.append("chaotic")

            effective_widths.append(layer.out_channels)
            h, w = self._compute_spatial_dims(h, w, layer)
            spatial_dims.append((h, w))

        # FC layers
        for fc_w in arch.fc_widths:
            chi_func = self._get_chi_map(arch.fc_activation)
            q = var_traj[len(arch.layers)]
            chi1 = sigma_w ** 2 * chi_func(max(q, 1e-30))
            layer_chi1.append(chi1)
            if abs(chi1 - 1.0) < 0.01:
                layer_phases.append("critical")
            elif chi1 < 1.0:
                layer_phases.append("ordered")
            else:
                layer_phases.append("chaotic")
            effective_widths.append(fc_w)

        # Overall chi_1: geometric mean
        if layer_chi1:
            log_chi = np.mean([np.log(max(c, 1e-30)) for c in layer_chi1])
            overall_chi1 = np.exp(log_chi)
        else:
            overall_chi1 = 1.0

        if abs(overall_chi1 - 1.0) < 0.01:
            overall_phase = "critical"
        elif overall_chi1 < 1.0:
            overall_phase = "ordered"
        else:
            overall_phase = "chaotic"

        # Depth scale
        if abs(overall_chi1 - 1.0) < 1e-10:
            depth_scale = float("inf")
        elif overall_chi1 <= 0:
            depth_scale = 0.0
        else:
            depth_scale = -1.0 / np.log(overall_chi1)

        # Fixed point (use first conv layer's activation)
        act = arch.layers[0].activation if arch.layers else "relu"
        V_func = self._get_variance_map(act)
        q_star = self._find_fixed_point(sigma_w, sigma_b, V_func)

        # Edge of chaos
        eoc, _ = self._analyzer.find_edge_of_chaos(act, sigma_b)

        total_depth = len(arch.layers) + len(arch.fc_widths)
        max_depth = max(1, int(abs(depth_scale) * 5)) if np.isfinite(depth_scale) else total_depth * 10

        return ConvMFReport(
            layer_variances=var_traj,
            layer_chi1=layer_chi1,
            layer_phases=layer_phases,
            layer_effective_widths=effective_widths,
            overall_chi1=overall_chi1,
            overall_phase=overall_phase,
            depth_scale=depth_scale if np.isfinite(depth_scale) else 1e6,
            max_trainable_depth=max_depth,
            fixed_point=q_star,
            fw_layer_variances=fw_var_traj,
            eoc_sigma_w=eoc,
            spatial_dims=spatial_dims,
        )

    def _find_fixed_point(self, sigma_w, sigma_b, V_func):
        """Find fixed point q* = sigma_w^2 * V(q*) + sigma_b^2."""
        q = 1.0
        for _ in range(self.max_iter):
            q_new = sigma_w ** 2 * V_func(q) + sigma_b ** 2
            if abs(q_new - q) < self.tol:
                return max(q_new, 1e-30)
            if q_new > 1e10:
                return q_new
            q = q_new
        return max(q, 1e-30)

    def conv_phase_diagram(self, activation: str = "relu",
                           channels: int = 64,
                           kernel_size: int = 3,
                           n_layers: int = 10,
                           sigma_w_values: Optional[List[float]] = None,
                           ) -> List[Dict[str, Any]]:
        """Generate phase diagram for a CNN across sigma_w values."""
        if sigma_w_values is None:
            sigma_w_values = np.linspace(0.3, 3.0, 30).tolist()

        results = []
        for sw in sigma_w_values:
            layers = [ConvLayerSpec(
                in_channels=channels if i > 0 else 3,
                out_channels=channels,
                kernel_size=kernel_size,
                activation=activation,
                padding=kernel_size // 2,  # same padding
            ) for i in range(n_layers)]

            arch = ConvArchSpec(
                layers=layers,
                sigma_w=sw,
                sigma_b=0.0,
                input_channels=3,
                input_height=32,
                input_width=32,
            )
            report = self.analyze(arch)
            results.append({
                "sigma_w": sw,
                "n_layers": n_layers,
                "channels": channels,
                "activation": activation,
                "overall_chi1": report.overall_chi1,
                "overall_phase": report.overall_phase,
                "depth_scale": report.depth_scale,
                "variance_final": report.layer_variances[-1],
                "fw_variance_final": report.fw_layer_variances[-1] if report.fw_layer_variances else None,
            })

        return results
