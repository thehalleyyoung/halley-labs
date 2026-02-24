"""
Phase diagram generator for neural network architectures.

Generates 2D and 3D phase diagrams showing ordered/critical/chaotic phases,
lazy/rich/untrainable regimes, and learning rate stability regions.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import json
import warnings


@dataclass
class PhaseDiagram:
    """Result of phase diagram generation."""
    grid_points: Dict[str, np.ndarray]  # axis name -> values
    phase_labels: np.ndarray  # integer labels at each grid point
    phase_names: Dict[int, str]  # label -> name mapping
    boundaries: List[List[Tuple[float, float]]]  # list of boundary curves
    critical_points: List[Tuple[float, ...]]  # special points
    recommended_region: Optional[Dict[str, Tuple[float, float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Export phase diagram data to JSON."""
        data = {
            "grid_points": {k: v.tolist() for k, v in self.grid_points.items()},
            "phase_labels": self.phase_labels.tolist(),
            "phase_names": {str(k): v for k, v in self.phase_names.items()},
            "boundaries": [
                [(float(p[0]), float(p[1])) for p in curve]
                for curve in self.boundaries
            ],
            "critical_points": [
                tuple(float(c) for c in pt) for pt in self.critical_points
            ],
            "recommended_region": self.recommended_region,
            "metadata": {
                k: v for k, v in self.metadata.items()
                if isinstance(v, (int, float, str, bool, list, dict))
            },
        }
        return json.dumps(data, indent=2, default=str)


@dataclass
class ArchConfig:
    """Architecture configuration for phase diagram generation."""
    activation: str = "relu"
    depth: int = 10
    width: int = 100
    sigma_w: float = 1.0
    sigma_b: float = 0.0
    has_residual: bool = False
    has_batchnorm: bool = False


class VarianceMapComputer:
    """Compute variance maps for activations used in phase diagram generation."""

    @staticmethod
    def relu_V(q: float) -> float:
        return max(q, 0.0) / 2.0

    @staticmethod
    def relu_chi(q: float) -> float:
        return 0.5

    @staticmethod
    def tanh_V(q: float) -> float:
        if q <= 0:
            return 0.0
        if q < 0.01:
            return q
        def integrand(z):
            return np.tanh(np.sqrt(q) * z) ** 2 * np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
        result, _ = quad(integrand, -8, 8)
        return result

    @staticmethod
    def tanh_chi(q: float) -> float:
        if q <= 0:
            return 1.0
        def integrand(z):
            t = np.tanh(np.sqrt(q) * z)
            return (1 - t ** 2) ** 2 * np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
        result, _ = quad(integrand, -8, 8)
        return result

    @staticmethod
    def sigmoid_V(q: float) -> float:
        if q <= 0:
            return 0.25
        def integrand(z):
            s = 1.0 / (1.0 + np.exp(-np.sqrt(q) * z))
            return s ** 2 * np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
        result, _ = quad(integrand, -8, 8)
        return result

    @staticmethod
    def sigmoid_chi(q: float) -> float:
        if q <= 0:
            return 0.0625
        def integrand(z):
            s = 1.0 / (1.0 + np.exp(-np.sqrt(q) * z))
            return (s * (1 - s)) ** 2 * np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
        result, _ = quad(integrand, -8, 8)
        return result

    def get_maps(self, activation: str):
        """Return (V_func, chi_func) for the given activation."""
        if activation == "relu":
            return self.relu_V, self.relu_chi
        elif activation in ("tanh", "erf"):
            return self.tanh_V, self.tanh_chi
        elif activation == "sigmoid":
            return self.sigmoid_V, self.sigmoid_chi
        else:
            return self.relu_V, self.relu_chi


class PhaseDiagramGenerator:
    """Generate phase diagrams for neural network architectures."""

    PHASE_ORDERED = 0
    PHASE_CRITICAL = 1
    PHASE_CHAOTIC = 2
    REGIME_LAZY = 3
    REGIME_RICH = 4
    REGIME_UNSTABLE = 5

    PHASE_NAMES = {
        0: "ordered",
        1: "critical",
        2: "chaotic",
        3: "lazy",
        4: "rich",
        5: "unstable",
    }

    def __init__(self, tolerance: float = 1e-8, max_fp_iter: int = 5000):
        self.tol = tolerance
        self.max_fp_iter = max_fp_iter
        self._vmc = VarianceMapComputer()

    def generate(self, architecture: ArchConfig,
                 param_ranges: Dict[str, Tuple[float, float]],
                 resolution: int = 50) -> PhaseDiagram:
        """Generate a 2D phase diagram.

        Args:
            architecture: Network architecture.
            param_ranges: Dict with "sigma_w" and "sigma_b" ranges.
            resolution: Grid resolution per axis.

        Returns:
            PhaseDiagram with phase classification at each grid point.
        """
        sw_range = param_ranges.get("sigma_w", (0.1, 3.0))
        sb_range = param_ranges.get("sigma_b", (0.0, 2.0))

        sw_vals = np.linspace(sw_range[0], sw_range[1], resolution)
        sb_vals = np.linspace(sb_range[0], sb_range[1], resolution)

        V_func, chi_func = self._vmc.get_maps(architecture.activation)

        phase_grid = np.zeros((resolution, resolution), dtype=int)
        chi_grid = np.zeros((resolution, resolution))

        for i, sw in enumerate(sw_vals):
            for j, sb in enumerate(sb_vals):
                q_star = self._find_fixed_point(sw, sb, V_func)
                chi_1 = sw ** 2 * chi_func(q_star)
                chi_grid[i, j] = chi_1

                if abs(chi_1 - 1.0) < 0.05:
                    phase_grid[i, j] = self.PHASE_CRITICAL
                elif chi_1 < 1.0:
                    phase_grid[i, j] = self.PHASE_ORDERED
                else:
                    phase_grid[i, j] = self.PHASE_CHAOTIC

        # Find boundaries
        boundaries = self._find_phase_boundaries(sw_vals, sb_vals, phase_grid)

        # Find critical points
        critical_points = self._find_critical_points(
            sw_vals, sb_vals, chi_grid, architecture
        )

        # Recommended region: near edge of chaos
        recommended = self._find_recommended_region(
            sw_vals, sb_vals, chi_grid
        )

        return PhaseDiagram(
            grid_points={"sigma_w": sw_vals, "sigma_b": sb_vals},
            phase_labels=phase_grid,
            phase_names=self.PHASE_NAMES,
            boundaries=boundaries,
            critical_points=critical_points,
            recommended_region=recommended,
            metadata={
                "activation": architecture.activation,
                "depth": architecture.depth,
                "resolution": resolution,
                "chi_grid": chi_grid.tolist(),
            },
        )

    def generate_3d(self, architecture: ArchConfig,
                    param_ranges: Dict[str, Tuple[float, float]],
                    depth_range: Tuple[int, int] = (2, 50),
                    resolution: int = 30,
                    depth_steps: int = 10) -> Dict[str, Any]:
        """Generate 3D phase diagram with depth as third axis.

        Args:
            architecture: Base architecture.
            param_ranges: Ranges for sigma_w and sigma_b.
            depth_range: Range of depths to sweep.
            resolution: Grid resolution for sigma_w x sigma_b.
            depth_steps: Number of depth values.

        Returns:
            Dictionary with 3D phase diagram data.
        """
        sw_range = param_ranges.get("sigma_w", (0.1, 3.0))
        sb_range = param_ranges.get("sigma_b", (0.0, 2.0))

        sw_vals = np.linspace(sw_range[0], sw_range[1], resolution)
        sb_vals = np.linspace(sb_range[0], sb_range[1], resolution)
        depths = np.linspace(depth_range[0], depth_range[1], depth_steps, dtype=int)

        V_func, chi_func = self._vmc.get_maps(architecture.activation)

        phase_volume = np.zeros((resolution, resolution, depth_steps), dtype=int)

        for k, depth in enumerate(depths):
            for i, sw in enumerate(sw_vals):
                for j, sb in enumerate(sb_vals):
                    q_star = self._find_fixed_point(sw, sb, V_func)
                    chi_1 = sw ** 2 * chi_func(q_star)

                    # Phase depends on chi_1 and depth
                    depth_scale = -1.0 / np.log(max(abs(chi_1), 1e-30)) if abs(chi_1 - 1.0) > 1e-10 else float("inf")

                    if chi_1 < 1.0 and depth > depth_scale * 5:
                        phase_volume[i, j, k] = self.PHASE_ORDERED
                    elif chi_1 > 1.0 and depth > depth_scale * 3:
                        phase_volume[i, j, k] = self.PHASE_CHAOTIC
                    elif abs(chi_1 - 1.0) < 0.05:
                        phase_volume[i, j, k] = self.PHASE_CRITICAL
                    elif chi_1 < 1.0:
                        phase_volume[i, j, k] = self.PHASE_ORDERED
                    else:
                        phase_volume[i, j, k] = self.PHASE_CHAOTIC

        return {
            "sigma_w": sw_vals.tolist(),
            "sigma_b": sb_vals.tolist(),
            "depths": depths.tolist(),
            "phase_volume": phase_volume.tolist(),
            "phase_names": self.PHASE_NAMES,
        }

    def generate_regime_diagram(self, architecture: ArchConfig,
                                 width_range: Tuple[int, int] = (10, 10000),
                                 lr_range: Tuple[float, float] = (1e-4, 1.0),
                                 resolution: int = 40) -> PhaseDiagram:
        """Generate regime diagram in (width, learning_rate) space.

        Classifies into lazy (NTK), rich (feature learning), and unstable regimes.

        Args:
            architecture: Network architecture.
            width_range: Range of widths.
            lr_range: Range of learning rates.
            resolution: Grid resolution.

        Returns:
            PhaseDiagram with regime classification.
        """
        widths = np.logspace(np.log10(width_range[0]), np.log10(width_range[1]), resolution)
        lrs = np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), resolution)

        regime_grid = np.zeros((resolution, resolution), dtype=int)

        for i, w in enumerate(widths):
            for j, lr in enumerate(lrs):
                regime = self._classify_regime(w, lr, architecture)
                regime_grid[i, j] = regime

        boundaries = self._find_phase_boundaries(widths, lrs, regime_grid)
        critical_pts = []

        # Find transitions
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                if regime_grid[i, j] != regime_grid[i + 1, j]:
                    critical_pts.append((widths[i], lrs[j]))
                if regime_grid[i, j] != regime_grid[i, j + 1]:
                    critical_pts.append((widths[i], lrs[j]))

        return PhaseDiagram(
            grid_points={"width": widths, "learning_rate": lrs},
            phase_labels=regime_grid,
            phase_names=self.PHASE_NAMES,
            boundaries=boundaries,
            critical_points=critical_pts[:20],
            metadata={
                "type": "regime_diagram",
                "activation": architecture.activation,
                "depth": architecture.depth,
            },
        )

    def generate_lr_stability_diagram(self, architecture: ArchConfig,
                                       lr_range: Tuple[float, float] = (1e-4, 10.0),
                                       width_range: Tuple[int, int] = (10, 5000),
                                       resolution: int = 40) -> PhaseDiagram:
        """Generate learning rate stability diagram.

        Sweeps (lr, width) and classifies each point as stable, oscillating,
        or divergent based on the maximum stable step size from NTK eigenvalues.

        Args:
            architecture: Network architecture.
            lr_range: Range of learning rates.
            width_range: Range of widths.
            resolution: Grid resolution.

        Returns:
            PhaseDiagram with stability regions.
        """
        lrs = np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), resolution)
        widths = np.logspace(np.log10(width_range[0]), np.log10(width_range[1]), resolution)

        STABLE = 0
        OSCILLATING = 1
        DIVERGENT = 2

        stability_names = {0: "stable", 1: "oscillating", 2: "divergent"}
        stability_grid = np.zeros((resolution, resolution), dtype=int)

        for i, lr in enumerate(lrs):
            for j, w in enumerate(widths):
                # Estimate max eigenvalue of NTK (scales as ~width for lazy regime)
                # In NTK parameterization, lambda_max ~ O(1)
                # In standard parameterization, lambda_max ~ O(n)
                lambda_max_est = architecture.sigma_w ** 2 * architecture.depth
                max_stable_lr = 2.0 / max(lambda_max_est, 1e-10)

                if lr < max_stable_lr * 0.9:
                    stability_grid[i, j] = STABLE
                elif lr < max_stable_lr * 2.0:
                    stability_grid[i, j] = OSCILLATING
                else:
                    stability_grid[i, j] = DIVERGENT

        boundaries = self._find_phase_boundaries(lrs, widths, stability_grid)

        return PhaseDiagram(
            grid_points={"learning_rate": lrs, "width": widths},
            phase_labels=stability_grid,
            phase_names=stability_names,
            boundaries=boundaries,
            critical_points=[],
            metadata={
                "type": "lr_stability",
                "activation": architecture.activation,
                "depth": architecture.depth,
            },
        )

    def generate_temperature_diagram(self, architecture: ArchConfig,
                                      batch_size_range: Tuple[int, int] = (1, 1024),
                                      lr_range: Tuple[float, float] = (1e-4, 1.0),
                                      resolution: int = 40) -> PhaseDiagram:
        """Generate temperature diagram using batch_size/lr ratio.

        The "temperature" T = lr / batch_size controls the noise level in SGD.
        High T: exploration, risk of instability.
        Low T: exploitation, risk of sharp minima.

        Args:
            architecture: Network architecture.
            batch_size_range: Range of batch sizes.
            lr_range: Range of learning rates.
            resolution: Grid resolution.

        Returns:
            PhaseDiagram with temperature-based regime classification.
        """
        batch_sizes = np.logspace(
            np.log10(batch_size_range[0]),
            np.log10(batch_size_range[1]),
            resolution,
        ).astype(int)
        lrs = np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), resolution)

        COLD = 0  # Low noise, sharp minima
        WARM = 1  # Balanced
        HOT = 2   # High noise

        temp_names = {0: "cold", 1: "warm", 2: "hot"}
        temp_grid = np.zeros((resolution, resolution), dtype=int)

        for i, bs in enumerate(batch_sizes):
            for j, lr in enumerate(lrs):
                temperature = lr / bs

                # Boundaries depend on architecture scale
                scale = architecture.sigma_w ** 2
                low_threshold = 1e-5 * scale
                high_threshold = 1e-2 * scale

                if temperature < low_threshold:
                    temp_grid[i, j] = COLD
                elif temperature < high_threshold:
                    temp_grid[i, j] = WARM
                else:
                    temp_grid[i, j] = HOT

        boundaries = self._find_phase_boundaries(
            batch_sizes.astype(float), lrs, temp_grid
        )

        return PhaseDiagram(
            grid_points={"batch_size": batch_sizes.astype(float), "learning_rate": lrs},
            phase_labels=temp_grid,
            phase_names=temp_names,
            boundaries=boundaries,
            critical_points=[],
            metadata={
                "type": "temperature",
                "activation": architecture.activation,
                "depth": architecture.depth,
            },
        )

    def _find_fixed_point(self, sigma_w: float, sigma_b: float,
                          V_func) -> float:
        """Find fixed point q* = sigma_w^2 * V(q*) + sigma_b^2."""
        q = 1.0
        for _ in range(self.max_fp_iter):
            q_new = sigma_w ** 2 * V_func(q) + sigma_b ** 2
            if abs(q_new - q) < self.tol:
                return max(q_new, 1e-30)
            q = q_new
            if q > 1e10 or np.isnan(q):
                return max(q, 1e-30)
        return max(q, 1e-30)

    def _classify_regime(self, width: float, lr: float,
                         arch: ArchConfig) -> int:
        """Classify training regime based on width and learning rate.

        Key insight: The ratio lr * width determines the regime.
        - lr * width >> 1: lazy (NTK) regime — features barely change
        - lr * width ~ 1: rich (feature learning) regime
        - lr > critical_lr: unstable regime
        """
        # Critical learning rate (stability boundary)
        lambda_max_est = arch.sigma_w ** 2 * arch.depth
        critical_lr = 2.0 / max(lambda_max_est, 1e-10)

        if lr > critical_lr:
            return self.REGIME_UNSTABLE

        # Lazy vs rich: parametrization scaling
        # In mean-field parameterization, features change as ~1/sqrt(width)
        # In NTK parameterization, features are frozen
        # Transition at lr * width ~ O(1)
        feature_change = lr * np.sqrt(width) * arch.sigma_w ** 2

        if feature_change > 10.0:
            return self.REGIME_LAZY
        elif feature_change < 0.1:
            return self.REGIME_RICH
        else:
            # Intermediate: classify based on depth
            V_func, chi_func = self._vmc.get_maps(arch.activation)
            q_star = self._find_fixed_point(arch.sigma_w, 0.0, V_func)
            chi_1 = arch.sigma_w ** 2 * chi_func(q_star)
            if chi_1 > 1.0:
                return self.REGIME_UNSTABLE
            return self.REGIME_RICH

    def _find_phase_boundaries(self, x_vals: np.ndarray, y_vals: np.ndarray,
                                phase_grid: np.ndarray) -> List[List[Tuple[float, float]]]:
        """Find boundaries between phase regions using contour detection."""
        boundaries = []
        nx, ny = phase_grid.shape

        # Find transitions along x-axis
        current_boundary = []
        for i in range(nx - 1):
            for j in range(ny):
                if phase_grid[i, j] != phase_grid[i + 1, j]:
                    x_mid = 0.5 * (x_vals[i] + x_vals[min(i + 1, len(x_vals) - 1)])
                    current_boundary.append((x_mid, y_vals[j]))

        if current_boundary:
            # Sort by y coordinate
            current_boundary.sort(key=lambda p: p[1])
            boundaries.append(current_boundary)

        # Find transitions along y-axis
        current_boundary = []
        for i in range(nx):
            for j in range(ny - 1):
                if phase_grid[i, j] != phase_grid[i, j + 1]:
                    y_mid = 0.5 * (y_vals[j] + y_vals[min(j + 1, len(y_vals) - 1)])
                    current_boundary.append((x_vals[i], y_mid))

        if current_boundary:
            current_boundary.sort(key=lambda p: p[0])
            boundaries.append(current_boundary)

        return boundaries

    def _find_critical_points(self, sw_vals: np.ndarray, sb_vals: np.ndarray,
                               chi_grid: np.ndarray,
                               arch: ArchConfig) -> List[Tuple[float, float]]:
        """Find critical points where chi_1 = 1."""
        critical_points = []
        nx, ny = chi_grid.shape

        for i in range(nx - 1):
            for j in range(ny - 1):
                # Check if chi=1 isoline crosses this cell
                vals = [chi_grid[i, j], chi_grid[i + 1, j],
                        chi_grid[i, j + 1], chi_grid[i + 1, j + 1]]
                if min(vals) <= 1.0 <= max(vals):
                    # Interpolate to find crossing point
                    sw_mid = 0.5 * (sw_vals[i] + sw_vals[i + 1])
                    sb_mid = 0.5 * (sb_vals[j] + sb_vals[j + 1])
                    critical_points.append((sw_mid, sb_mid))

        # Remove duplicates (keep subset)
        if len(critical_points) > 50:
            indices = np.linspace(0, len(critical_points) - 1, 50, dtype=int)
            critical_points = [critical_points[i] for i in indices]

        return critical_points

    def _find_recommended_region(self, sw_vals: np.ndarray, sb_vals: np.ndarray,
                                  chi_grid: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Find recommended (sigma_w, sigma_b) region near edge of chaos."""
        # Find points where chi is close to 1
        distance_to_critical = np.abs(chi_grid - 1.0)
        best_idx = np.unravel_index(np.argmin(distance_to_critical), chi_grid.shape)

        # Define region around best point
        margin = 2  # grid points
        sw_low = sw_vals[max(0, best_idx[0] - margin)]
        sw_high = sw_vals[min(len(sw_vals) - 1, best_idx[0] + margin)]
        sb_low = sb_vals[max(0, best_idx[1] - margin)]
        sb_high = sb_vals[min(len(sb_vals) - 1, best_idx[1] + margin)]

        return {
            "sigma_w": (float(sw_low), float(sw_high)),
            "sigma_b": (float(sb_low), float(sb_high)),
            "best_sigma_w": float(sw_vals[best_idx[0]]),
            "best_sigma_b": float(sb_vals[best_idx[1]]),
            "chi_at_best": float(chi_grid[best_idx]),
        }

    def phase_boundary_curve(self, activation: str = "relu",
                              sigma_b_range: Tuple[float, float] = (0.0, 2.0),
                              n_points: int = 100) -> List[Tuple[float, float]]:
        """Compute the chi_1 = 1 phase boundary curve analytically.

        For each sigma_b, finds sigma_w such that chi_1 = 1.

        Args:
            activation: Activation function.
            sigma_b_range: Range of sigma_b values.
            n_points: Number of boundary points.

        Returns:
            List of (sigma_w, sigma_b) on the boundary.
        """
        V_func, chi_func = self._vmc.get_maps(activation)
        sb_vals = np.linspace(sigma_b_range[0], sigma_b_range[1], n_points)
        boundary = []

        for sb in sb_vals:
            def objective(sw):
                q_star = self._find_fixed_point(sw, sb, V_func)
                return sw ** 2 * chi_func(q_star) - 1.0

            try:
                low = objective(0.1)
                high = objective(5.0)
                if low * high < 0:
                    sw_star = brentq(objective, 0.1, 5.0)
                    boundary.append((sw_star, sb))
                else:
                    result = minimize_scalar(
                        lambda sw: abs(objective(sw)),
                        bounds=(0.1, 5.0),
                        method="bounded",
                    )
                    if abs(objective(result.x)) < 0.01:
                        boundary.append((result.x, sb))
            except (ValueError, RuntimeError):
                continue

        return boundary

    def export_diagram_json(self, diagram: PhaseDiagram, filepath: str) -> None:
        """Export phase diagram to JSON file.

        Args:
            diagram: Phase diagram to export.
            filepath: Output file path.
        """
        json_str = diagram.to_json()
        with open(filepath, "w") as f:
            f.write(json_str)
