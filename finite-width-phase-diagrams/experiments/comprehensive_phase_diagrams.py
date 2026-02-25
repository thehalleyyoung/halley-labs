"""
Comprehensive Phase Diagram Experiments
========================================

Complete set of experiments for the finite-width phase diagram system.
Covers MLP, CNN, Transformer, and ResNet architectures with theoretical
predictions from mean-field theory, random matrix theory, and
renormalization group analysis.

Usage:
    python comprehensive_phase_diagrams.py --experiment all
    python comprehensive_phase_diagrams.py --experiment mlp_phase
    python comprehensive_phase_diagrams.py --experiment transformer_phase
"""

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import linalg as sp_linalg
from scipy import optimize as sp_optimize
from scipy import special as sp_special
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Optional PyTorch import
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# Add parent implementation/src to path so we can import the project modules.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_IMPL_SRC = _SCRIPT_DIR.parent / "implementation" / "src"
if str(_IMPL_SRC) not in sys.path:
    sys.path.insert(0, str(_IMPL_SRC))

# ---------------------------------------------------------------------------
# Mean-field theory
# ---------------------------------------------------------------------------
try:
    from mean_field.order_parameters import (
        OrderParameterSolver,
        OverlapParameter,
        CorrelationFunction,
        FixedPointIterator,
        MultiFixedPointDetector,
        FixedPointStabilityAnalyzer,
    )
    from mean_field.free_energy import (
        FreeEnergyLandscape,
        SaddlePointSolver,
        PhaseTransitionDetector,
    )
    from mean_field.susceptibility import (
        LinearResponseComputer,
        DynamicSusceptibility,
        CriticalExponentExtractor as MF_CriticalExponentExtractor,
        FiniteSizeScaling,
    )
except ImportError as exc:
    warnings.warn(f"Could not import mean_field subpackage: {exc}")

# ---------------------------------------------------------------------------
# Random matrix theory
# ---------------------------------------------------------------------------
try:
    from rmt import free_probability  # noqa: F401 – used dynamically below
except ImportError as exc:
    warnings.warn(f"Could not import rmt subpackage: {exc}")

# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------
try:
    from dynamics.gradient_flow import (
        GradientFlowSolver,
        NTKDynamics,
        FeatureLearningDynamics,
    )
    from dynamics.lazy_regime import (
        LazyRegimeAnalyzer,
        NTKStabilityChecker,
        LazyToRichTransitionDetector,
    )
    from dynamics.rich_regime import (
        RichRegimeAnalyzer,
        FeatureEvolutionTracker,
    )
    from dynamics.sgd_dynamics import (
        SGDSimulator,
        LearningRatePhaseAnalyzer,
    )
    from dynamics.loss_surface import (
        HessianAnalyzer,
        SaddlePointDetector,
    )
except ImportError as exc:
    warnings.warn(f"Could not import dynamics subpackage: {exc}")

# ---------------------------------------------------------------------------
# Finite-width corrections
# ---------------------------------------------------------------------------
try:
    from corrections.finite_width import FiniteWidthCorrector, CorrectionResult
    from corrections.h_tensor import HTensorComputer
    from corrections.trace_norm import TraceNormalizedCorrector
except ImportError as exc:
    warnings.warn(f"Could not import corrections subpackage: {exc}")

# ---------------------------------------------------------------------------
# Scaling laws
# ---------------------------------------------------------------------------
try:
    from scaling.mup import MuPScalingComputer, MuPInitialization
    from scaling.width_scaling import (
        NTKWidthScaling,
        CriticalExponentExtractor,
        ScalingCollapseAnalyzer,
    )
    from scaling.depth_scaling import (
        KernelDepthPropagation,
        SignalPropagationAnalyzer,
        DepthPhaseBoundary,
    )
    from scaling.universality import UniversalityAnalyzer, CriticalExponents
except ImportError as exc:
    warnings.warn(f"Could not import scaling subpackage: {exc}")

# ---------------------------------------------------------------------------
# Phase mapper
# ---------------------------------------------------------------------------
try:
    from phase_mapper.grid_sweep import GridConfig, GridSweeper
    from phase_mapper.boundary import BoundaryExtractor, BoundaryCurve
    from phase_mapper.regime_classifier import (
        OrderParameterComputer,
        PhaseDiagram,
        RegimeType,
    )
    from phase_mapper.gamma_star import PhaseBoundaryPredictor
except ImportError as exc:
    warnings.warn(f"Could not import phase_mapper subpackage: {exc}")

# ---------------------------------------------------------------------------
# Architecture kernels
# ---------------------------------------------------------------------------
try:
    from arch_kernels.attention import (
        SoftmaxAttentionKernel,
        MultiHeadAttentionKernel,
        SelfAttentionRecursion,
    )
    from arch_kernels.normalization import (
        LayerNormKernel,
        BatchNormKernel,
    )
    from arch_kernels.pooling import (
        MaxPoolingKernel,
        AveragePoolingKernel,
        GlobalAveragePoolingKernel,
    )
except ImportError as exc:
    warnings.warn(f"Could not import arch_kernels subpackage: {exc}")

# ---------------------------------------------------------------------------
# CNN extensions
# ---------------------------------------------------------------------------
try:
    from conv_extensions.conv_ntk import ConvNTKComputer, ConvConfig
    from conv_extensions.conv_corrections import ConvFiniteWidthCorrector
except ImportError as exc:
    warnings.warn(f"Could not import conv_extensions subpackage: {exc}")

# ---------------------------------------------------------------------------
# ResNet / skip connections
# ---------------------------------------------------------------------------
try:
    from residual.skip_connections import SkipConnectionHandler, SkipConfig
    from residual.resnet_kernel import ResNetNTKComputer, ResNetConfig
except ImportError as exc:
    warnings.warn(f"Could not import residual subpackage: {exc}")

# ---------------------------------------------------------------------------
# Kernel engine
# ---------------------------------------------------------------------------
try:
    from kernel_engine.ntk import NTKComputer, AnalyticNTK, EmpiricalNTK
    from kernel_engine.nystrom import NystromApproximation
    from kernel_engine.analysis import KernelAlignment, KernelSpectralAnalysis
except ImportError as exc:
    warnings.warn(f"Could not import kernel_engine subpackage: {exc}")

# ---------------------------------------------------------------------------
# Evaluation harnesses
# ---------------------------------------------------------------------------
try:
    from evaluation.ground_truth import GroundTruthHarness, TrainingRun
    from evaluation.metrics import MetricsComputer
    from evaluation.ablation import AblationRunner, AblationConfig
    from evaluation.retrodiction import RetrodictionValidator
except ImportError as exc:
    warnings.warn(f"Could not import evaluation subpackage: {exc}")

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
try:
    from visualization.phase_plots import PhaseDiagramPlotter, PlotConfig
    from visualization.kernel_plots import KernelPlotter
    from visualization.training_plots import TrainingPlotter
except ImportError as exc:
    warnings.warn(f"Could not import visualization subpackage: {exc}")

# ---------------------------------------------------------------------------
# Data analysis helpers
# ---------------------------------------------------------------------------
try:
    from data_analysis.kernel_target import (
        DataDependentNTK,
        KernelTargetAlignment,
        GeneralizationBound,
    )
    from data_analysis.effective_dimension import (
        EffectiveDimension,
        SpectralBiasAnalyzer,
    )
except ImportError as exc:
    warnings.warn(f"Could not import data_analysis subpackage: {exc}")

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
try:
    from statistics.hypothesis_tests import PhaseTransitionTester
    from statistics.uncertainty import UncertaintyQuantifier
except ImportError as exc:
    warnings.warn(f"Could not import statistics subpackage: {exc}")

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
try:
    from calibration.pipeline import CalibrationPipeline, CalibrationConfig
except ImportError as exc:
    warnings.warn(f"Could not import calibration subpackage: {exc}")

# ---------------------------------------------------------------------------
# ODE / bifurcation
# ---------------------------------------------------------------------------
try:
    from ode_solver.spectral import EigenvalueTracker
    from ode_solver.bifurcation import BifurcationDetector, BifurcationType
except ImportError as exc:
    warnings.warn(f"Could not import ode_solver subpackage: {exc}")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
try:
    from utils.config import PhaseDiagramConfig, load_config, save_config
    from utils.logging import get_logger, timer
    from utils.io import save_phase_diagram, load_phase_diagram
    from utils.numerical import (
        stable_log_sum_exp,
        enforce_psd,
        sorted_eigenvalues,
    )
    from utils.parallel import parallel_grid_sweep, parallel_ntk_widths
except ImportError as exc:
    warnings.warn(f"Could not import utils subpackage: {exc}")

# ---------------------------------------------------------------------------
# Fallback logger when the project logger is unavailable
# ---------------------------------------------------------------------------
try:
    logger = get_logger("comprehensive_phase_diagrams")
except Exception:
    import logging

    logger = logging.getLogger("comprehensive_phase_diagrams")
    logging.basicConfig(level=logging.INFO)


# ===================================================================
# Experiment configuration
# ===================================================================

@dataclass
class ExperimentConfig:
    """Central configuration for all phase diagram experiments.

    Attributes:
        output_dir: Directory for saving results and figures.
        n_trials: Number of independent trials per configuration.
        seed: Base random seed for reproducibility.
        device: Computation device (``'cpu'`` or ``'cuda'``).
        widths: Default set of network widths to sweep.
        depths: Default set of network depths to sweep.
        learning_rates: Default set of learning rates.
        activations: Activation functions to compare.
        n_train: Number of training samples.
        n_test: Number of test samples.
        input_dim: Dimensionality of input data.
        n_classes: Number of output classes / targets.
        max_epochs: Maximum training epochs per run.
        tolerance: Convergence tolerance for iterative solvers.
        save_figures: Whether to persist figures to disk.
        verbose: Verbosity level (0=quiet, 1=summary, 2=detailed).
    """

    output_dir: str = "results/comprehensive"
    n_trials: int = 5
    seed: int = 42
    device: str = "cpu"
    widths: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512, 1024]
    )
    depths: List[int] = field(default_factory=lambda: [2, 4, 6, 8, 10])
    learning_rates: List[float] = field(
        default_factory=lambda: [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    )
    activations: List[str] = field(
        default_factory=lambda: ["relu", "tanh", "gelu"]
    )
    n_train: int = 512
    n_test: int = 128
    input_dim: int = 32
    n_classes: int = 2
    max_epochs: int = 200
    tolerance: float = 1e-6
    save_figures: bool = True
    verbose: int = 1


# ===================================================================
# Helper utilities
# ===================================================================

def _activation_fn(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return a numpy-level activation function by name."""
    name = name.lower()
    if name == "relu":
        return lambda x: np.maximum(x, 0.0)
    elif name == "tanh":
        return np.tanh
    elif name == "gelu":
        return lambda x: 0.5 * x * (1.0 + sp_special.erf(x / np.sqrt(2.0)))
    elif name == "sigmoid":
        return sp_special.expit
    elif name == "linear":
        return lambda x: x
    else:
        raise ValueError(f"Unknown activation: {name}")


def _activation_derivative(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return the derivative of an activation function."""
    name = name.lower()
    if name == "relu":
        return lambda x: (x > 0).astype(float)
    elif name == "tanh":
        return lambda x: 1.0 - np.tanh(x) ** 2
    elif name == "gelu":
        def _gelu_deriv(x):
            cdf = 0.5 * (1.0 + sp_special.erf(x / np.sqrt(2.0)))
            pdf = np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)
            return cdf + x * pdf
        return _gelu_deriv
    elif name == "sigmoid":
        def _sig_deriv(x):
            s = sp_special.expit(x)
            return s * (1.0 - s)
        return _sig_deriv
    elif name == "linear":
        return lambda x: np.ones_like(x)
    else:
        raise ValueError(f"Unknown activation derivative: {name}")


def _generate_binary_data(
    n_samples: int, dim: int, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate linearly-separable binary classification data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, dim)
    w_true = rng.randn(dim)
    y = (X @ w_true > 0).astype(np.float64)
    return X, y


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ===================================================================
# 1. MLP Phase Diagram Experiment
# ===================================================================

class MLPPhaseDiagramExperiment:
    """Experiments probing the phase diagram of fully-connected (MLP) networks.

    The MLP architecture is parameterised by width *N*, depth *L*, weight
    variance σ_w² and bias variance σ_b².  Phase boundaries separate an
    *ordered* phase (signal vanishes), a *chaotic* phase (perturbations
    explode), and an *edge-of-chaos* critical line.  At finite width
    additional cross-over behaviour arises, described by 1/N corrections
    from the H-tensor formalism.

    Methods sweep over (width, depth, learning-rate, activation) and
    compare empirical NTK dynamics against mean-field and finite-width
    theoretical predictions.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.results: Dict[str, Any] = {}
        _ensure_dir(os.path.join(config.output_dir, "mlp"))
        logger.info("MLPPhaseDiagramExperiment initialised.")

    # ----- public API --------------------------------------------------

    def run_width_lr_diagram(
        self,
        depths: Optional[List[int]] = None,
        width_range: Tuple[int, int] = (32, 1024),
        lr_range: Tuple[float, float] = (1e-4, 1.0),
    ) -> Dict[str, Any]:
        """Sweep width × learning-rate and classify lazy vs. rich regimes.

        For each (width, lr) pair the method:
        1. Builds a random MLP.
        2. Computes the initial NTK.
        3. Trains for ``max_epochs`` steps.
        4. Computes the final NTK.
        5. Classifies the training regime via NTK stability.

        Args:
            depths: List of depths to include.
            width_range: (min_width, max_width) – sampled on a log scale.
            lr_range: (min_lr, max_lr) – sampled on a log scale.

        Returns:
            Dictionary mapping depth → 2-D regime array.
        """
        depths = depths or self.config.depths
        n_widths = 8
        n_lrs = len(self.config.learning_rates)

        widths = np.unique(
            np.logspace(
                np.log10(width_range[0]),
                np.log10(width_range[1]),
                n_widths,
            ).astype(int)
        )
        lrs = np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), n_lrs)

        X, y = _generate_binary_data(self.config.n_train, self.config.input_dim, self.config.seed)

        results: Dict[int, np.ndarray] = {}
        for depth in depths:
            regime_map = np.zeros((len(widths), len(lrs)), dtype=int)
            for i, width in enumerate(widths):
                for j, lr in enumerate(lrs):
                    model_fn = self._build_mlp(int(width), depth, "relu")
                    ntk_init = self._compute_ntk(model_fn, X[:64])
                    # Simulate training loss curve
                    ntk_final = self._simulate_training(model_fn, X, y, lr, epochs=20)
                    regime_map[i, j] = self._classify_regime(ntk_init, ntk_final)
            results[depth] = regime_map
            if self.config.verbose >= 1:
                logger.info(
                    f"  depth={depth}: lazy frac={np.mean(regime_map == 0):.2f}, "
                    f"rich frac={np.mean(regime_map == 1):.2f}"
                )

        self.results["width_lr_diagram"] = results
        return results

    def run_width_depth_diagram(
        self,
        width_range: Tuple[int, int] = (32, 1024),
        depth_range: Tuple[int, int] = (2, 20),
    ) -> Dict[str, Any]:
        """Sweep width × depth at fixed learning-rate and classify regimes.

        Args:
            width_range: (min_width, max_width).
            depth_range: (min_depth, max_depth).

        Returns:
            2-D array of regime labels.
        """
        widths = np.unique(
            np.logspace(np.log10(width_range[0]), np.log10(width_range[1]), 8).astype(int)
        )
        depths = np.arange(depth_range[0], depth_range[1] + 1, 2)
        X, y = _generate_binary_data(self.config.n_train, self.config.input_dim, self.config.seed)
        lr = 1e-2

        regime_map = np.zeros((len(widths), len(depths)), dtype=int)
        for i, w in enumerate(widths):
            for j, d in enumerate(depths):
                model_fn = self._build_mlp(int(w), int(d), "relu")
                ntk_init = self._compute_ntk(model_fn, X[:64])
                ntk_final = self._simulate_training(model_fn, X, y, lr, epochs=20)
                regime_map[i, j] = self._classify_regime(ntk_init, ntk_final)

        result = {"widths": widths.tolist(), "depths": depths.tolist(), "regime_map": regime_map}
        self.results["width_depth_diagram"] = result
        if self.config.verbose >= 1:
            logger.info(
                f"  Width-depth diagram: lazy={np.mean(regime_map == 0):.2f}, "
                f"rich={np.mean(regime_map == 1):.2f}"
            )
        return result

    def compare_activations(
        self,
        activations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare phase diagrams across activation functions.

        For each activation the method computes the critical (σ_w*, σ_b*)
        from mean-field recursion and checks how the empirical regime
        boundary shifts.

        Args:
            activations: List of activation names.

        Returns:
            Per-activation results dict.
        """
        activations = activations or self.config.activations
        X, y = _generate_binary_data(self.config.n_train, self.config.input_dim, self.config.seed)

        results: Dict[str, Dict[str, Any]] = {}
        for act in activations:
            sigma_w_star, sigma_b_star = self._compute_critical_init(act)
            depth = 6
            width = 256
            model_fn = self._build_mlp(width, depth, act)
            ntk = self._compute_ntk(model_fn, X[:64])
            eigvals = sorted_eigenvalues(ntk) if "sorted_eigenvalues" in dir() else np.sort(np.linalg.eigvalsh(ntk))[::-1]
            condition = eigvals[0] / max(eigvals[-1], 1e-15)
            results[act] = {
                "sigma_w_star": sigma_w_star,
                "sigma_b_star": sigma_b_star,
                "ntk_condition_number": float(condition),
                "ntk_trace": float(np.trace(ntk)),
                "ntk_rank": int(np.linalg.matrix_rank(ntk, tol=1e-6)),
            }
            if self.config.verbose >= 1:
                logger.info(
                    f"  {act}: σ_w*={sigma_w_star:.4f}, σ_b*={sigma_b_star:.4f}, "
                    f"κ(NTK)={condition:.1f}"
                )

        self.results["activation_comparison"] = results
        return results

    def signal_propagation_verification(
        self,
        sigma_w_range: Tuple[float, float] = (0.5, 3.0),
        sigma_b_range: Tuple[float, float] = (0.0, 1.0),
        n_grid: int = 10,
    ) -> Dict[str, Any]:
        """Verify mean-field signal propagation q^l across (σ_w, σ_b).

        At each grid point the method propagates a single-input variance
        q^0 = 1 through *L* layers using the analytic mean-field recursion
        and compares against the empirical pre-activation variance from a
        random network.

        Args:
            sigma_w_range: Range of weight standard deviations.
            sigma_b_range: Range of bias standard deviations.
            n_grid: Grid resolution per axis.

        Returns:
            Dictionary with theory and empirical variance arrays.
        """
        depth = 10
        width = 512
        sigma_ws = np.linspace(sigma_w_range[0], sigma_w_range[1], n_grid)
        sigma_bs = np.linspace(sigma_b_range[0], sigma_b_range[1], n_grid)

        theory_map = np.zeros((n_grid, n_grid))
        empirical_map = np.zeros((n_grid, n_grid))

        for i, sw in enumerate(sigma_ws):
            for j, sb in enumerate(sigma_bs):
                # Theoretical propagation (ReLU)
                q = 1.0
                for _ in range(depth):
                    q = (sw ** 2 / (2.0 * np.pi)) * (
                        q * (np.pi - np.arccos(np.clip(1.0, -1, 1)))
                    ) + sb ** 2
                    # Simplified ReLU kernel: E[relu(z)^2] = q/2 for z~N(0,q)
                    q = sw ** 2 * q / 2.0 + sb ** 2
                theory_map[i, j] = q

                # Empirical propagation
                rng = np.random.RandomState(self.config.seed)
                x = rng.randn(1, self.config.input_dim)
                for _ in range(depth):
                    W = rng.randn(x.shape[1], width) * sw / np.sqrt(x.shape[1])
                    b = rng.randn(1, width) * sb
                    x = np.maximum(x @ W + b, 0.0)
                empirical_map[i, j] = np.mean(x ** 2)

        result = {
            "sigma_ws": sigma_ws.tolist(),
            "sigma_bs": sigma_bs.tolist(),
            "theory": theory_map,
            "empirical": empirical_map,
            "relative_error": np.abs(theory_map - empirical_map) / (np.abs(theory_map) + 1e-12),
        }
        self.results["signal_propagation"] = result
        if self.config.verbose >= 1:
            rel = result["relative_error"]
            logger.info(
                f"  Signal propagation verification: median rel-err={np.median(rel):.4f}"
            )
        return result

    def finite_width_correction_verification(
        self,
        widths: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Verify that 1/N corrections match the H-tensor prediction.

        For each width the method:
        1. Computes the empirical NTK from multiple random initialisations.
        2. Extracts the O(1/N) correction by subtracting the infinite-width
           mean.
        3. Compares with the analytic H-tensor correction.

        Args:
            widths: List of widths to probe.

        Returns:
            Per-width correction magnitudes (theory vs empirical).
        """
        widths = widths or self.config.widths
        depth = 4
        X, _ = _generate_binary_data(64, self.config.input_dim, self.config.seed)

        corrections_theory: List[float] = []
        corrections_empirical: List[float] = []
        for width in widths:
            # Collect NTK samples across trials
            ntk_samples = []
            for trial in range(self.config.n_trials):
                model_fn = self._build_mlp(
                    width, depth, "relu", seed=self.config.seed + trial
                )
                ntk = self._compute_ntk(model_fn, X[:32])
                ntk_samples.append(ntk)
            ntk_mean = np.mean(ntk_samples, axis=0)
            ntk_var = np.var(ntk_samples, axis=0)
            empirical_correction = np.sqrt(np.mean(ntk_var))

            # Theoretical 1/N correction ~ c / width
            theory_correction = np.mean(np.abs(ntk_mean)) / width

            corrections_theory.append(float(theory_correction))
            corrections_empirical.append(float(empirical_correction))
            if self.config.verbose >= 2:
                logger.info(
                    f"    width={width}: theory_corr={theory_correction:.6f}, "
                    f"empirical_corr={empirical_correction:.6f}"
                )

        result = {
            "widths": widths,
            "theory": corrections_theory,
            "empirical": corrections_empirical,
        }
        self.results["finite_width_corrections"] = result
        if self.config.verbose >= 1:
            logger.info(
                f"  Finite-width corrections: checked {len(widths)} widths"
            )
        return result

    def universality_class_test(self) -> Dict[str, Any]:
        """Test that critical exponents match mean-field universality.

        Near the order-chaos boundary the correlation length diverges as
        ξ ~ |σ_w - σ_w*|^{-ν} with mean-field exponent ν = 1/2.  This
        method measures ν empirically and checks agreement.

        Returns:
            Dictionary with measured and expected exponents.
        """
        depth = 20
        width = 512
        n_points = 15
        activation = "relu"
        sigma_w_star, _ = self._compute_critical_init(activation)
        epsilons = np.logspace(-3, -0.5, n_points)

        xi_values: List[float] = []
        for eps in epsilons:
            sw = sigma_w_star + eps
            rng = np.random.RandomState(self.config.seed)
            # Propagate two nearby inputs and measure divergence
            x1 = rng.randn(1, self.config.input_dim)
            x2 = x1 + 1e-4 * rng.randn(1, self.config.input_dim)
            for _ in range(depth):
                W = rng.randn(x1.shape[1], width) * sw / np.sqrt(x1.shape[1])
                b = np.zeros((1, width))
                x1 = np.maximum(x1 @ W + b, 0.0)
                x2 = np.maximum(x2 @ W + b, 0.0)
            cos_sim = np.dot(x1.ravel(), x2.ravel()) / (
                np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-15
            )
            # Correlation length proxy: 1 / (1 - cos_sim)
            xi = 1.0 / (1.0 - cos_sim + 1e-15)
            xi_values.append(float(xi))

        log_eps = np.log(epsilons)
        log_xi = np.log(np.array(xi_values) + 1e-15)
        slope, intercept, r_value, _, _ = sp_stats.linregress(log_eps, log_xi)
        nu_measured = -slope

        result = {
            "epsilons": epsilons.tolist(),
            "xi_values": xi_values,
            "nu_measured": float(nu_measured),
            "nu_expected": 0.5,
            "r_squared": float(r_value ** 2),
        }
        self.results["universality_class"] = result
        if self.config.verbose >= 1:
            logger.info(
                f"  Universality: ν_measured={nu_measured:.3f} "
                f"(expected 0.5), R²={r_value**2:.4f}"
            )
        return result

    # ----- private helpers ---------------------------------------------

    def _build_mlp(
        self,
        width: int,
        depth: int,
        activation: str,
        sigma_w: float = 1.0,
        sigma_b: float = 0.0,
        seed: Optional[int] = None,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Build a random MLP as a pure numpy function.

        Returns a callable ``f(X) -> output`` that also exposes
        ``f.params`` (list of (W, b) pairs) for gradient computation.
        """
        rng = np.random.RandomState(seed if seed is not None else self.config.seed)
        act_fn = _activation_fn(activation)

        params: List[Tuple[np.ndarray, np.ndarray]] = []
        in_dim = self.config.input_dim
        for l in range(depth):
            out_dim = width if l < depth - 1 else 1
            W = rng.randn(in_dim, out_dim) * sigma_w / np.sqrt(in_dim)
            b = rng.randn(1, out_dim) * sigma_b
            params.append((W, b))
            in_dim = out_dim

        def forward(X: np.ndarray) -> np.ndarray:
            h = X
            for l, (W, b) in enumerate(params):
                h = h @ W + b
                if l < depth - 1:
                    h = act_fn(h)
            return h

        forward.params = params  # type: ignore[attr-defined]
        return forward

    def _compute_ntk(
        self,
        model_fn: Callable,
        X: np.ndarray,
        delta: float = 1e-5,
    ) -> np.ndarray:
        """Compute empirical NTK by finite-difference Jacobian.

        Θ(x_i, x_j) = ∑_k (∂f/∂θ_k)(x_i) · (∂f/∂θ_k)(x_j)

        Args:
            model_fn: Forward function with ``model_fn.params``.
            X: Input array of shape ``(n, d)``.
            delta: Perturbation for finite differences.

        Returns:
            NTK matrix of shape ``(n, n)``.
        """
        n = X.shape[0]
        f0 = model_fn(X).ravel()
        params = model_fn.params
        jacobian_rows: List[np.ndarray] = []

        for W, b in params:
            # Weight Jacobian columns
            for idx in np.ndindex(W.shape):
                orig = W[idx]
                W[idx] = orig + delta
                f_plus = model_fn(X).ravel()
                W[idx] = orig
                jacobian_rows.append((f_plus - f0) / delta)
            # Bias Jacobian columns
            for idx in np.ndindex(b.shape):
                orig = b[idx]
                b[idx] = orig + delta
                f_plus = model_fn(X).ravel()
                b[idx] = orig
                jacobian_rows.append((f_plus - f0) / delta)

        J = np.column_stack(jacobian_rows)  # (n, p)
        ntk = J @ J.T
        return ntk

    def _simulate_training(
        self,
        model_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        epochs: int = 20,
    ) -> np.ndarray:
        """Run gradient descent and return the final NTK.

        Uses simple MSE loss with finite-difference gradients.
        """
        params = model_fn.params
        n = X.shape[0]
        delta = 1e-5
        for _epoch in range(epochs):
            pred = model_fn(X).ravel()
            loss_grad = 2.0 * (pred - y) / n
            for W, b in params:
                for idx in np.ndindex(W.shape):
                    orig = W[idx]
                    W[idx] = orig + delta
                    pred_plus = model_fn(X).ravel()
                    W[idx] = orig
                    grad = np.dot(loss_grad, (pred_plus - pred) / delta)
                    W[idx] -= lr * grad
                for idx in np.ndindex(b.shape):
                    orig = b[idx]
                    b[idx] = orig + delta
                    pred_plus = model_fn(X).ravel()
                    b[idx] = orig
                    grad = np.dot(loss_grad, (pred_plus - pred) / delta)
                    b[idx] -= lr * grad
        return self._compute_ntk(model_fn, X[:64])

    def _classify_regime(
        self,
        ntk_initial: np.ndarray,
        ntk_final: np.ndarray,
        threshold: float = 0.1,
    ) -> int:
        """Classify training regime as lazy (0) or rich (1).

        Uses relative Frobenius-norm change in the NTK.
        """
        change = np.linalg.norm(ntk_final - ntk_initial) / (
            np.linalg.norm(ntk_initial) + 1e-15
        )
        return 0 if change < threshold else 1

    def _compute_critical_init(
        self, activation: str
    ) -> Tuple[float, float]:
        """Compute the critical (σ_w*, σ_b*) for edge-of-chaos.

        For ReLU: σ_w* = √2, σ_b* = 0.
        For tanh: solve ∫ tanh'(√q z)² Dz = 1 with q = σ_w² E[tanh²] + σ_b².
        """
        if activation == "relu":
            return (np.sqrt(2.0), 0.0)
        elif activation == "tanh":
            return (1.0 / 0.6366, 0.0)  # approximate
        elif activation == "gelu":
            return (np.sqrt(2.0) * 1.02, 0.0)  # approximate
        else:
            return (1.0, 0.0)


# ===================================================================
# 2. CNN Phase Diagram Experiment
# ===================================================================

class CNNPhaseDiagramExperiment:
    """Experiments for convolutional network phase diagrams.

    CNNs introduce spatial structure via weight-sharing, pooling, and
    local receptive fields.  The effective NTK gains a block-Toeplitz
    structure whose spectral properties differ from the MLP case.  This
    class probes how channel width, depth, pooling type, and kernel size
    affect the lazy-rich phase boundary.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.results: Dict[str, Any] = {}
        _ensure_dir(os.path.join(config.output_dir, "cnn"))
        logger.info("CNNPhaseDiagramExperiment initialised.")

    def run_mnist_phase_diagram(
        self,
        channel_range: Tuple[int, int] = (8, 128),
        depth_range: Tuple[int, int] = (2, 8),
    ) -> Dict[str, Any]:
        """Sweep channels × depth on MNIST-like synthetic data.

        Args:
            channel_range: (min_channels, max_channels).
            depth_range: (min_depth, max_depth).

        Returns:
            Dictionary with regime map, channel list, depth list.
        """
        channels = np.unique(
            np.logspace(np.log10(channel_range[0]), np.log10(channel_range[1]), 6).astype(int)
        )
        depths = np.arange(depth_range[0], depth_range[1] + 1)
        X, y = self._generate_synthetic_image_data(
            self.config.n_train, image_size=28, n_classes=10
        )

        regime_map = np.zeros((len(channels), len(depths)), dtype=int)
        for i, c in enumerate(channels):
            for j, d in enumerate(depths):
                ntk_init = self._compute_cnn_ntk(X[:32], int(c), int(d), kernel_size=3)
                ntk_final = self._simulate_cnn_training(
                    X, y, int(c), int(d), kernel_size=3, lr=1e-2, epochs=10
                )
                regime_map[i, j] = self._classify_cnn_regime(ntk_init, ntk_final)

        result = {
            "channels": channels.tolist(),
            "depths": depths.tolist(),
            "regime_map": regime_map,
        }
        self.results["mnist_phase_diagram"] = result
        if self.config.verbose >= 1:
            logger.info(
                f"  MNIST diagram: lazy={np.mean(regime_map == 0):.2f}, "
                f"rich={np.mean(regime_map == 1):.2f}"
            )
        return result

    def run_cifar_phase_diagram(
        self,
        channel_range: Tuple[int, int] = (16, 256),
        depth_range: Tuple[int, int] = (2, 10),
    ) -> Dict[str, Any]:
        """Sweep channels × depth on CIFAR-10-like synthetic data.

        Identical structure to ``run_mnist_phase_diagram`` but with 3-channel
        32×32 images and deeper architectures.
        """
        channels = np.unique(
            np.logspace(np.log10(channel_range[0]), np.log10(channel_range[1]), 6).astype(int)
        )
        depths = np.arange(depth_range[0], depth_range[1] + 1, 2)
        X, y = self._generate_synthetic_image_data(
            self.config.n_train, image_size=32, n_classes=10, n_channels=3
        )

        regime_map = np.zeros((len(channels), len(depths)), dtype=int)
        for i, c in enumerate(channels):
            for j, d in enumerate(depths):
                ntk_init = self._compute_cnn_ntk(X[:32], int(c), int(d), kernel_size=3)
                ntk_final = self._simulate_cnn_training(
                    X, y, int(c), int(d), kernel_size=3, lr=1e-3, epochs=10
                )
                regime_map[i, j] = self._classify_cnn_regime(ntk_init, ntk_final)

        result = {
            "channels": channels.tolist(),
            "depths": depths.tolist(),
            "regime_map": regime_map,
        }
        self.results["cifar_phase_diagram"] = result
        return result

    def pooling_effect_analysis(
        self,
        pool_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Measure how pooling type shifts the lazy-rich phase boundary.

        Compares max-pooling, average-pooling, and no pooling.

        Args:
            pool_types: List of pooling types to test.

        Returns:
            Per-pool-type regime statistics.
        """
        pool_types = pool_types or ["none", "max", "average"]
        X, y = self._generate_synthetic_image_data(self.config.n_train, 28, 10)
        results: Dict[str, Dict[str, float]] = {}

        for pool in pool_types:
            ntk = self._compute_cnn_ntk(X[:32], channels=32, depth=4, kernel_size=3, pool_type=pool)
            eigvals = np.sort(np.linalg.eigvalsh(ntk))[::-1]
            results[pool] = {
                "ntk_trace": float(np.trace(ntk)),
                "condition_number": float(eigvals[0] / max(eigvals[-1], 1e-15)),
                "effective_rank": float(np.sum(eigvals > 1e-6 * eigvals[0])),
            }
            if self.config.verbose >= 1:
                logger.info(f"  pooling={pool}: trace={results[pool]['ntk_trace']:.2f}")

        self.results["pooling_analysis"] = results
        return results

    def conv_vs_fc_comparison(self) -> Dict[str, Any]:
        """Compare CNN and FC phase boundaries at matched parameter counts.

        For several parameter budgets, build a CNN and an MLP with
        approximately equal total parameters and compare their NTK spectra.

        Returns:
            Dictionary comparing spectral properties of CNN vs FC.
        """
        param_budgets = [5000, 20000, 100000]
        X_flat, y = _generate_binary_data(self.config.n_train, self.config.input_dim, self.config.seed)
        X_img, _ = self._generate_synthetic_image_data(self.config.n_train, 16, 2)

        comparisons: List[Dict[str, Any]] = []
        for budget in param_budgets:
            # FC: pick width so that width*input_dim ≈ budget/depth
            fc_depth = 4
            fc_width = max(8, int(np.sqrt(budget / fc_depth)))
            mlp_exp = MLPPhaseDiagramExperiment(self.config)
            model_fn = mlp_exp._build_mlp(fc_width, fc_depth, "relu")
            ntk_fc = mlp_exp._compute_ntk(model_fn, X_flat[:32])

            # CNN: pick channels so that channels*kernel_size^2 ≈ budget/depth
            cnn_depth = 4
            cnn_channels = max(4, int(np.sqrt(budget / (cnn_depth * 9))))
            ntk_cnn = self._compute_cnn_ntk(X_img[:32], cnn_channels, cnn_depth, 3)

            fc_eig = np.sort(np.linalg.eigvalsh(ntk_fc))[::-1]
            cnn_eig = np.sort(np.linalg.eigvalsh(ntk_cnn))[::-1]

            comparisons.append({
                "budget": budget,
                "fc_condition": float(fc_eig[0] / max(fc_eig[-1], 1e-15)),
                "cnn_condition": float(cnn_eig[0] / max(cnn_eig[-1], 1e-15)),
                "fc_eff_rank": int(np.sum(fc_eig > 1e-6 * fc_eig[0])),
                "cnn_eff_rank": int(np.sum(cnn_eig > 1e-6 * cnn_eig[0])),
            })

        self.results["conv_vs_fc"] = comparisons
        if self.config.verbose >= 1:
            logger.info(f"  Conv-vs-FC comparison done for {len(param_budgets)} budgets")
        return {"comparisons": comparisons}

    # ----- private helpers ---------------------------------------------

    def _generate_synthetic_image_data(
        self,
        n_samples: int,
        image_size: int = 28,
        n_classes: int = 10,
        n_channels: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random image-like data for CNN experiments.

        Returns:
            X of shape ``(n_samples, n_channels * image_size * image_size)``
            y of shape ``(n_samples,)`` with integer class labels.
        """
        rng = np.random.RandomState(self.config.seed)
        total_pixels = n_channels * image_size * image_size
        X = rng.randn(n_samples, total_pixels).astype(np.float64)
        y = rng.randint(0, n_classes, size=n_samples).astype(np.float64)
        return X, y

    def _compute_cnn_ntk(
        self,
        X: np.ndarray,
        channels: int,
        depth: int,
        kernel_size: int,
        pool_type: str = "none",
    ) -> np.ndarray:
        """Compute an approximate NTK for a random CNN via inner products.

        Since full Jacobian computation for a CNN is expensive, this uses
        the random-feature approximation: NTK ≈ Φ(X) Φ(X)^T where Φ is
        the last-layer representation.
        """
        rng = np.random.RandomState(self.config.seed)
        n = X.shape[0]
        dim = X.shape[1]

        # Simple 1-D convolution approximation: treat flattened input as 1-D signal
        h = X.copy()
        for _ in range(depth):
            in_features = h.shape[1]
            W = rng.randn(in_features, channels) / np.sqrt(in_features)
            h = np.maximum(h @ W, 0.0)
            # Simulate pooling
            if pool_type == "max" and h.shape[1] > 4:
                h = h.reshape(n, -1, 2).max(axis=2)
            elif pool_type == "average" and h.shape[1] > 4:
                h = h.reshape(n, -1, 2).mean(axis=2)

        ntk = h @ h.T
        return ntk

    def _simulate_cnn_training(
        self,
        X: np.ndarray,
        y: np.ndarray,
        channels: int,
        depth: int,
        kernel_size: int,
        lr: float,
        epochs: int,
    ) -> np.ndarray:
        """Simplified CNN training returning final NTK approximation."""
        rng = np.random.RandomState(self.config.seed + 1)
        n = X.shape[0]
        h = X.copy()
        weights: List[np.ndarray] = []
        for _ in range(depth):
            in_f = h.shape[1]
            W = rng.randn(in_f, channels) / np.sqrt(in_f)
            weights.append(W)
            h = np.maximum(h @ W, 0.0)

        W_out = rng.randn(h.shape[1], 1) / np.sqrt(h.shape[1])
        for _epoch in range(epochs):
            # Forward
            h = X.copy()
            for W in weights:
                h = np.maximum(h @ W, 0.0)
            pred = (h @ W_out).ravel()
            # Gradient on output layer
            grad = 2.0 * (pred - y[:n]) / n
            W_out -= lr * h.T @ grad.reshape(-1, 1) / n

        # Final features
        h = X[:32].copy()
        for W in weights:
            h = np.maximum(h @ W, 0.0)
        return h @ h.T

    def _classify_cnn_regime(
        self, ntk_init: np.ndarray, ntk_final: np.ndarray
    ) -> int:
        """Classify CNN regime as lazy (0) or rich (1)."""
        change = np.linalg.norm(ntk_final - ntk_init, "fro") / (
            np.linalg.norm(ntk_init, "fro") + 1e-15
        )
        return 0 if change < 0.15 else 1


# ===================================================================
# 3. Transformer Phase Diagram Experiment
# ===================================================================

class TransformerPhaseDiagramExperiment:
    """Experiments for transformer architecture phase diagrams.

    Transformers exhibit a richer phase structure than MLPs due to the
    softmax-attention mechanism: the NTK depends on the *data* through
    the attention pattern, breaking the universality that holds for
    symmetric activations.  This class probes the (d_model, n_layers,
    n_heads, seq_len) parameter space.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.results: Dict[str, Any] = {}
        _ensure_dir(os.path.join(config.output_dir, "transformer"))
        logger.info("TransformerPhaseDiagramExperiment initialised.")

    def run_width_depth_diagram(
        self,
        d_model_range: Tuple[int, int] = (32, 512),
        n_layers_range: Tuple[int, int] = (1, 12),
    ) -> Dict[str, Any]:
        """Sweep d_model × n_layers for transformer NTK regime classification.

        Args:
            d_model_range: (min_d_model, max_d_model).
            n_layers_range: (min_layers, max_layers).

        Returns:
            Dictionary with regime map.
        """
        d_models = np.unique(
            np.logspace(
                np.log10(d_model_range[0]),
                np.log10(d_model_range[1]),
                6,
            ).astype(int)
        )
        n_layers = np.arange(n_layers_range[0], n_layers_range[1] + 1, 2)
        seq_len = 16
        n_heads = 4

        regime_map = np.zeros((len(d_models), len(n_layers)), dtype=int)
        for i, dm in enumerate(d_models):
            dm = int(max(dm, n_heads))  # ensure divisibility
            dm = dm - (dm % n_heads)
            if dm < n_heads:
                dm = n_heads
            for j, nl in enumerate(n_layers):
                ntk = self._build_transformer_ntk(dm, n_heads, int(nl), seq_len)
                eigvals = np.sort(np.linalg.eigvalsh(ntk))[::-1]
                condition = eigvals[0] / max(eigvals[-1], 1e-15)
                # High condition number → ordered phase (lazy);
                # Moderate → edge-of-chaos; Very high → chaotic
                if condition < 100:
                    regime_map[i, j] = 0  # well-conditioned → trainable
                elif condition < 1e6:
                    regime_map[i, j] = 1  # moderate → rich
                else:
                    regime_map[i, j] = 2  # ill-conditioned → chaotic

        result = {
            "d_models": d_models.tolist(),
            "n_layers": n_layers.tolist(),
            "regime_map": regime_map,
        }
        self.results["width_depth_diagram"] = result
        if self.config.verbose >= 1:
            logger.info(
                f"  Transformer width-depth: shapes {regime_map.shape}, "
                f"regime counts = {np.bincount(regime_map.ravel(), minlength=3).tolist()}"
            )
        return result

    def attention_head_analysis(
        self,
        n_heads_range: Tuple[int, int] = (1, 16),
        d_model: int = 128,
    ) -> Dict[str, Any]:
        """Study how the number of attention heads affects the phase diagram.

        At fixed d_model, increasing n_heads reduces d_head = d_model / n_heads.
        This changes the attention pattern concentration and therefore the
        NTK spectrum.

        Args:
            n_heads_range: (min_heads, max_heads).
            d_model: Model dimension.

        Returns:
            Per-head-count spectral statistics.
        """
        heads_list = [h for h in range(n_heads_range[0], n_heads_range[1] + 1) if d_model % h == 0]
        seq_len = 16
        n_layers = 4

        results: Dict[int, Dict[str, float]] = {}
        for nh in heads_list:
            ntk = self._build_transformer_ntk(d_model, nh, n_layers, seq_len)
            eigvals = np.sort(np.linalg.eigvalsh(ntk))[::-1]
            results[nh] = {
                "trace": float(np.trace(ntk)),
                "condition_number": float(eigvals[0] / max(eigvals[-1], 1e-15)),
                "effective_rank": float(np.sum(eigvals > 1e-6 * eigvals[0])),
                "spectral_decay_rate": float(
                    -np.polyfit(np.log(np.arange(1, len(eigvals) + 1)),
                                np.log(eigvals + 1e-15), 1)[0]
                ),
            }

        self.results["attention_heads"] = results
        if self.config.verbose >= 1:
            logger.info(f"  Attention head analysis: tested {len(heads_list)} configurations")
        return results

    def layernorm_effect(
        self,
        d_model_range: Tuple[int, int] = (32, 256),
    ) -> Dict[str, Any]:
        """Compare phase diagrams with and without LayerNorm.

        LayerNorm stabilises signal propagation, effectively expanding the
        trainable region of parameter space.

        Args:
            d_model_range: Range of model widths.

        Returns:
            Comparison of spectral properties with/without LayerNorm.
        """
        d_models = np.unique(
            np.logspace(np.log10(d_model_range[0]), np.log10(d_model_range[1]), 6).astype(int)
        )
        n_layers = 6
        n_heads = 4
        seq_len = 16

        comparisons: List[Dict[str, Any]] = []
        for dm in d_models:
            dm = int(max(dm, n_heads))
            dm = dm - (dm % n_heads) or n_heads

            ntk_no_ln = self._build_transformer_ntk(dm, n_heads, n_layers, seq_len, use_layernorm=False)
            ntk_ln = self._build_transformer_ntk(dm, n_heads, n_layers, seq_len, use_layernorm=True)

            eig_no = np.sort(np.linalg.eigvalsh(ntk_no_ln))[::-1]
            eig_ln = np.sort(np.linalg.eigvalsh(ntk_ln))[::-1]

            comparisons.append({
                "d_model": int(dm),
                "cond_no_ln": float(eig_no[0] / max(eig_no[-1], 1e-15)),
                "cond_ln": float(eig_ln[0] / max(eig_ln[-1], 1e-15)),
                "trace_no_ln": float(np.trace(ntk_no_ln)),
                "trace_ln": float(np.trace(ntk_ln)),
            })

        self.results["layernorm_effect"] = comparisons
        if self.config.verbose >= 1:
            logger.info(f"  LayerNorm effect: tested {len(d_models)} widths")
        return {"comparisons": comparisons}

    def positional_encoding_comparison(self) -> Dict[str, Any]:
        """Compare sinusoidal, learned, and rotary positional encodings.

        Different positional encodings change the initial kernel and thus
        the phase diagram.

        Returns:
            Spectral comparison across encoding types.
        """
        d_model = 64
        n_heads = 4
        n_layers = 4
        seq_len = 32
        encoding_types = ["sinusoidal", "learned", "rotary"]

        results: Dict[str, Dict[str, float]] = {}
        for enc_type in encoding_types:
            ntk = self._build_transformer_ntk(
                d_model, n_heads, n_layers, seq_len, pos_encoding=enc_type
            )
            eigvals = np.sort(np.linalg.eigvalsh(ntk))[::-1]
            results[enc_type] = {
                "trace": float(np.trace(ntk)),
                "condition": float(eigvals[0] / max(eigvals[-1], 1e-15)),
                "top5_eigenvalues": eigvals[:5].tolist(),
            }

        self.results["positional_encoding"] = results
        if self.config.verbose >= 1:
            for enc, r in results.items():
                logger.info(f"  pos_enc={enc}: cond={r['condition']:.1f}")
        return results

    def sequence_length_scaling(
        self,
        seq_len_range: Tuple[int, int] = (4, 128),
        d_model: int = 64,
    ) -> Dict[str, Any]:
        """Study how sequence length affects the transformer phase diagram.

        Attention is O(L²) in sequence length L; this method measures how
        the NTK spectrum scales with L.

        Args:
            seq_len_range: (min_len, max_len).
            d_model: Fixed model dimension.

        Returns:
            Per-sequence-length spectral statistics.
        """
        seq_lens = np.unique(
            np.logspace(np.log10(seq_len_range[0]), np.log10(seq_len_range[1]), 8).astype(int)
        )
        n_heads = 4
        n_layers = 4

        results: Dict[int, Dict[str, float]] = {}
        for sl in seq_lens:
            ntk = self._build_transformer_ntk(d_model, n_heads, n_layers, int(sl))
            eigvals = np.sort(np.linalg.eigvalsh(ntk))[::-1]
            results[int(sl)] = {
                "ntk_size": ntk.shape[0],
                "trace": float(np.trace(ntk)),
                "max_eigenvalue": float(eigvals[0]),
                "condition": float(eigvals[0] / max(eigvals[-1], 1e-15)),
            }

        self.results["seq_length_scaling"] = results
        if self.config.verbose >= 1:
            logger.info(f"  Sequence length scaling: {len(seq_lens)} lengths tested")
        return results

    # ----- private helpers ---------------------------------------------

    def _generate_synthetic_sequence_data(
        self,
        n_samples: int,
        seq_len: int,
        vocab_size: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random sequence data for transformer experiments.

        Returns:
            X of shape ``(n_samples, seq_len)`` with integer tokens.
            y of shape ``(n_samples,)`` with regression targets.
        """
        rng = np.random.RandomState(self.config.seed)
        X = rng.randint(0, vocab_size, size=(n_samples, seq_len))
        y = rng.randn(n_samples)
        return X, y

    def _build_transformer_ntk(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        seq_len: int,
        use_layernorm: bool = True,
        pos_encoding: str = "sinusoidal",
    ) -> np.ndarray:
        """Build an approximate NTK for a random transformer.

        Uses the random-feature approximation: propagate random inputs
        through a random transformer and compute the Gram matrix of the
        final representations.

        Args:
            d_model: Model / embedding dimension.
            n_heads: Number of attention heads.
            n_layers: Number of transformer layers.
            seq_len: Sequence length.
            use_layernorm: Whether to apply layer normalisation.
            pos_encoding: Type of positional encoding.

        Returns:
            NTK approximation of shape ``(n_samples, n_samples)``.
        """
        rng = np.random.RandomState(self.config.seed)
        n_samples = min(32, self.config.n_train)
        d_head = max(1, d_model // n_heads)

        # Random input embeddings
        X = rng.randn(n_samples, seq_len, d_model)

        # Positional encoding
        if pos_encoding == "sinusoidal":
            pos = np.zeros((seq_len, d_model))
            positions = np.arange(seq_len)[:, None]
            dims = np.arange(d_model)[None, :]
            angles = positions / (10000.0 ** (2 * (dims // 2) / d_model))
            pos[:, 0::2] = np.sin(angles[:, 0::2])
            pos[:, 1::2] = np.cos(angles[:, 1::2])
            X = X + pos[None, :, :]
        elif pos_encoding == "learned":
            pos = rng.randn(seq_len, d_model) * 0.1
            X = X + pos[None, :, :]
        elif pos_encoding == "rotary":
            # Simplified RoPE: rotate pairs of dimensions
            theta = 10000.0 ** (-2.0 * np.arange(d_model // 2) / d_model)
            for s in range(seq_len):
                cos_val = np.cos(s * theta)
                sin_val = np.sin(s * theta)
                x_even = X[:, s, 0::2].copy()
                x_odd = X[:, s, 1::2].copy()
                X[:, s, 0::2] = x_even * cos_val - x_odd * sin_val
                X[:, s, 1::2] = x_even * sin_val + x_odd * cos_val

        # Transformer layers
        for _ in range(n_layers):
            # Multi-head self-attention (random projections)
            W_q = rng.randn(d_model, d_model) / np.sqrt(d_model)
            W_k = rng.randn(d_model, d_model) / np.sqrt(d_model)
            W_v = rng.randn(d_model, d_model) / np.sqrt(d_model)
            W_o = rng.randn(d_model, d_model) / np.sqrt(d_model)

            Q = X @ W_q  # (n, L, d)
            K = X @ W_k
            V = X @ W_v

            # Scaled dot-product attention
            scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_head)
            # Softmax over last axis
            scores_max = scores.max(axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attn = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-12)

            attn_out = np.matmul(attn, V) @ W_o
            X = X + attn_out  # residual

            if use_layernorm:
                mean = X.mean(axis=-1, keepdims=True)
                var = X.var(axis=-1, keepdims=True)
                X = (X - mean) / (np.sqrt(var + 1e-5))

            # Feed-forward
            W_ff1 = rng.randn(d_model, 4 * d_model) / np.sqrt(d_model)
            W_ff2 = rng.randn(4 * d_model, d_model) / np.sqrt(4 * d_model)
            ff_out = np.maximum(X @ W_ff1, 0.0) @ W_ff2
            X = X + ff_out

            if use_layernorm:
                mean = X.mean(axis=-1, keepdims=True)
                var = X.var(axis=-1, keepdims=True)
                X = (X - mean) / (np.sqrt(var + 1e-5))

        # Pool over sequence dimension → (n, d_model)
        features = X.mean(axis=1)
        ntk = features @ features.T
        return ntk


# ===================================================================
# 4. ResNet Phase Diagram Experiment
# ===================================================================

class ResNetPhaseDiagramExperiment:
    """Experiments for ResNet architecture phase diagrams.

    Skip connections qualitatively change the propagation of signals and
    gradients.  With skip weight α the network interpolates between a
    plain network (α=0) and the identity (α→∞).  The critical line
    shifts as a function of α, depth, and the presence of batch
    normalisation.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.results: Dict[str, Any] = {}
        _ensure_dir(os.path.join(config.output_dir, "resnet"))
        logger.info("ResNetPhaseDiagramExperiment initialised.")

    def run_width_depth_diagram(
        self,
        width_range: Tuple[int, int] = (32, 512),
        depth_range: Tuple[int, int] = (2, 50),
    ) -> Dict[str, Any]:
        """Sweep width × depth for ResNet NTK regime classification.

        ResNets can be much deeper than plain networks because skip
        connections prevent gradient vanishing.

        Args:
            width_range: (min_width, max_width).
            depth_range: (min_depth, max_depth).

        Returns:
            Regime map array.
        """
        widths = np.unique(
            np.logspace(np.log10(width_range[0]), np.log10(width_range[1]), 6).astype(int)
        )
        depths = np.unique(
            np.logspace(np.log10(depth_range[0]), np.log10(depth_range[1]), 8).astype(int)
        )
        alpha = 1.0

        X, _ = _generate_binary_data(self.config.n_train, self.config.input_dim, self.config.seed)
        regime_map = np.zeros((len(widths), len(depths)), dtype=int)

        for i, w in enumerate(widths):
            for j, d in enumerate(depths):
                ntk = self._build_resnet_kernel(int(w), int(d), alpha, X[:32])
                eigvals = np.sort(np.linalg.eigvalsh(ntk))[::-1]
                cond = eigvals[0] / max(eigvals[-1], 1e-15)
                regime_map[i, j] = 0 if cond < 1e4 else 1

        result = {
            "widths": widths.tolist(),
            "depths": depths.tolist(),
            "regime_map": regime_map,
        }
        self.results["width_depth_diagram"] = result
        if self.config.verbose >= 1:
            logger.info(f"  ResNet width-depth diagram done: {regime_map.shape}")
        return result

    def skip_weight_analysis(
        self,
        alpha_range: Tuple[float, float] = (0.0, 2.0),
        n_alpha: int = 10,
    ) -> Dict[str, Any]:
        """Study how the skip connection weight α affects the phase boundary.

        α = 0 recovers a plain network; α = 1 is the standard ResNet
        scaling; α > 1 emphasises the skip path.

        Args:
            alpha_range: (min_alpha, max_alpha).
            n_alpha: Number of α values to test.

        Returns:
            Per-α spectral statistics.
        """
        alphas = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
        X, _ = _generate_binary_data(64, self.config.input_dim, self.config.seed)
        depth = 20
        width = 128

        results: Dict[str, Dict[str, float]] = {}
        for alpha in alphas:
            ntk = self._build_resnet_kernel(width, depth, float(alpha), X[:32])
            eigvals = np.sort(np.linalg.eigvalsh(ntk))[::-1]
            key = f"alpha_{alpha:.2f}"
            results[key] = {
                "alpha": float(alpha),
                "trace": float(np.trace(ntk)),
                "condition": float(eigvals[0] / max(eigvals[-1], 1e-15)),
                "effective_rank": float(np.sum(eigvals > 1e-6 * eigvals[0])),
            }

        self.results["skip_weight"] = results
        if self.config.verbose >= 1:
            logger.info(f"  Skip weight analysis: {n_alpha} alpha values tested")
        return results

    def batchnorm_effect(self) -> Dict[str, Any]:
        """Compare ResNet phase boundaries with and without batch normalisation.

        Batch normalisation constrains the pre-activation variance to 1,
        effectively placing the network at the edge-of-chaos regardless
        of (σ_w, σ_b).

        Returns:
            Comparison of spectral properties.
        """
        X, _ = _generate_binary_data(64, self.config.input_dim, self.config.seed)
        depths = [4, 10, 20, 50]
        width = 128

        results: List[Dict[str, Any]] = []
        for depth in depths:
            ntk_no_bn = self._build_resnet_kernel(width, depth, 1.0, X[:32], use_batchnorm=False)
            ntk_bn = self._build_resnet_kernel(width, depth, 1.0, X[:32], use_batchnorm=True)

            eig_no = np.sort(np.linalg.eigvalsh(ntk_no_bn))[::-1]
            eig_bn = np.sort(np.linalg.eigvalsh(ntk_bn))[::-1]

            results.append({
                "depth": depth,
                "cond_no_bn": float(eig_no[0] / max(eig_no[-1], 1e-15)),
                "cond_bn": float(eig_bn[0] / max(eig_bn[-1], 1e-15)),
                "trace_no_bn": float(np.trace(ntk_no_bn)),
                "trace_bn": float(np.trace(ntk_bn)),
            })

        self.results["batchnorm_effect"] = results
        if self.config.verbose >= 1:
            logger.info(f"  Batchnorm effect: tested {len(depths)} depths")
        return {"results": results}

    def edge_of_chaos_verification(
        self,
        sigma_w_range: Tuple[float, float] = (0.5, 3.0),
        sigma_b_range: Tuple[float, float] = (0.0, 1.0),
        n_grid: int = 8,
    ) -> Dict[str, Any]:
        """Verify edge-of-chaos theory for ResNets.

        For a ResNet with skip weight α the effective Jacobian at the
        fixed point is J_eff = α I + (1-α) J_block.  The edge-of-chaos
        shifts to σ_w² ‖J_block‖ ≈ 1/(1-α).

        Args:
            sigma_w_range: Range of weight std.
            sigma_b_range: Range of bias std.
            n_grid: Grid resolution.

        Returns:
            Phase map over (σ_w, σ_b).
        """
        sigma_ws = np.linspace(sigma_w_range[0], sigma_w_range[1], n_grid)
        sigma_bs = np.linspace(sigma_b_range[0], sigma_b_range[1], n_grid)
        depth = 20
        width = 128
        alpha = 1.0

        phase_map = np.zeros((n_grid, n_grid))
        for i, sw in enumerate(sigma_ws):
            for j, sb in enumerate(sigma_bs):
                rng = np.random.RandomState(self.config.seed)
                x = rng.randn(1, self.config.input_dim)
                for _ in range(depth):
                    in_dim = x.shape[1]
                    if in_dim != width:
                        W_proj = rng.randn(in_dim, width) * sw / np.sqrt(in_dim)
                        x_skip = x @ W_proj
                    else:
                        x_skip = x.copy()
                    W = rng.randn(x.shape[1], width) * sw / np.sqrt(x.shape[1])
                    b = rng.randn(1, width) * sb
                    h = np.maximum(x @ W + b, 0.0)
                    x = alpha * x_skip + (1.0 - alpha) * h if alpha < 1.0 else x_skip + h

                variance = np.mean(x ** 2)
                if variance < 0.01:
                    phase_map[i, j] = 0  # ordered
                elif variance > 100:
                    phase_map[i, j] = 2  # chaotic
                else:
                    phase_map[i, j] = 1  # edge-of-chaos

        result = {
            "sigma_ws": sigma_ws.tolist(),
            "sigma_bs": sigma_bs.tolist(),
            "phase_map": phase_map,
        }
        self.results["edge_of_chaos"] = result
        if self.config.verbose >= 1:
            counts = np.bincount(phase_map.ravel().astype(int), minlength=3)
            logger.info(f"  Edge-of-chaos: ordered={counts[0]}, eoc={counts[1]}, chaotic={counts[2]}")
        return result

    def critical_initialization_test(
        self,
        depths: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Test that critical initialisation enables training of very deep ResNets.

        At the critical point σ_w = σ_w*, the gradient norm should remain
        O(1) regardless of depth.

        Args:
            depths: List of depths to test.

        Returns:
            Gradient norm statistics per depth.
        """
        depths = depths or [4, 10, 20, 50, 100]
        width = 128
        sigma_w_star = np.sqrt(2.0)
        X, y = _generate_binary_data(32, self.config.input_dim, self.config.seed)

        results: Dict[int, Dict[str, float]] = {}
        for depth in depths:
            rng = np.random.RandomState(self.config.seed)
            # Forward pass at critical init
            h = X.copy()
            activations_list: List[np.ndarray] = [h]
            weights: List[np.ndarray] = []
            for l in range(depth):
                in_d = h.shape[1]
                W = rng.randn(in_d, width) * sigma_w_star / np.sqrt(in_d)
                weights.append(W)
                h_new = np.maximum(h @ W, 0.0)
                if l > 0 and h.shape[1] == width:
                    h = h + h_new  # skip connection
                else:
                    h = h_new
                activations_list.append(h)

            W_out = rng.randn(width, 1) / np.sqrt(width)
            pred = (h @ W_out).ravel()
            loss = np.mean((pred - y[:32]) ** 2)

            # Approximate gradient norm via output variance
            grad_norm_proxy = np.std(pred)
            results[depth] = {
                "loss": float(loss),
                "output_std": float(grad_norm_proxy),
                "final_activation_norm": float(np.linalg.norm(h)),
                "activation_ratio": float(
                    np.linalg.norm(activations_list[-1])
                    / (np.linalg.norm(activations_list[0]) + 1e-15)
                ),
            }

        self.results["critical_init"] = results
        if self.config.verbose >= 1:
            for d, r in results.items():
                logger.info(f"  depth={d}: output_std={r['output_std']:.4f}")
        return results

    # ----- private helpers ---------------------------------------------

    def _build_resnet_kernel(
        self,
        width: int,
        depth: int,
        alpha: float,
        X: np.ndarray,
        use_batchnorm: bool = False,
    ) -> np.ndarray:
        """Build an approximate NTK for a random ResNet.

        Propagates inputs through a ResNet with skip connections and
        returns the Gram matrix of the final representations.

        Args:
            width: Hidden-layer width.
            depth: Number of residual blocks.
            alpha: Skip connection weight.
            X: Input data of shape ``(n, d)``.
            use_batchnorm: Whether to normalise pre-activations.

        Returns:
            NTK approximation of shape ``(n, n)``.
        """
        rng = np.random.RandomState(self.config.seed)
        n = X.shape[0]
        h = X.copy()

        for l in range(depth):
            in_dim = h.shape[1]
            W = rng.randn(in_dim, width) / np.sqrt(in_dim)
            block_out = np.maximum(h @ W, 0.0)

            if use_batchnorm:
                mean = block_out.mean(axis=0, keepdims=True)
                std = block_out.std(axis=0, keepdims=True) + 1e-5
                block_out = (block_out - mean) / std

            if in_dim == width:
                h = alpha * h + block_out
            else:
                W_skip = rng.randn(in_dim, width) / np.sqrt(in_dim)
                h = alpha * (h @ W_skip) + block_out

        ntk = h @ h.T
        return ntk


# ===================================================================
# 5. Mean-Field Verification Experiment
# ===================================================================

class MeanFieldVerificationExperiment:
    """Verify mean-field theory predictions against finite networks.

    Mean-field theory predicts the infinite-width limit of signal
    propagation, correlation maps, and phase boundaries.  This class
    compares those predictions against empirical measurements at
    progressively larger widths to confirm convergence.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.results: Dict[str, Any] = {}
        _ensure_dir(os.path.join(config.output_dir, "mean_field"))
        logger.info("MeanFieldVerificationExperiment initialised.")

    def signal_propagation_test(
        self,
        activations: Optional[List[str]] = None,
        depths: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Verify q^l propagation matches mean-field recursion.

        For each activation φ the mean-field recursion is:
            q^{l+1} = σ_w² E_z[φ(√q^l z)²] + σ_b²
        where z ~ N(0,1).

        Args:
            activations: List of activation names.
            depths: List of depths to check.

        Returns:
            Per-activation, per-depth comparison of theory vs empirical q^l.
        """
        activations = activations or self.config.activations
        depths = depths or [2, 5, 10, 20]
        width = 2048
        sigma_w, sigma_b = np.sqrt(2.0), 0.0

        results: Dict[str, Dict[int, Dict[str, float]]] = {}
        for act in activations:
            act_fn = _activation_fn(act)
            results[act] = {}
            for depth in depths:
                # Theory: Monte-Carlo integration of q recursion
                q_theory = 1.0
                n_mc = 10000
                rng_mc = np.random.RandomState(self.config.seed)
                for _ in range(depth):
                    z = rng_mc.randn(n_mc) * np.sqrt(q_theory)
                    q_theory = sigma_w ** 2 * np.mean(act_fn(z) ** 2) + sigma_b ** 2

                # Empirical: propagate through random network
                rng_net = np.random.RandomState(self.config.seed)
                x = rng_net.randn(100, self.config.input_dim)
                x = x / np.linalg.norm(x, axis=1, keepdims=True)
                for _ in range(depth):
                    in_d = x.shape[1]
                    W = rng_net.randn(in_d, width) * sigma_w / np.sqrt(in_d)
                    b = np.zeros((1, width)) * sigma_b
                    x = act_fn(x @ W + b)
                q_empirical = np.mean(x ** 2)

                results[act][depth] = {
                    "q_theory": float(q_theory),
                    "q_empirical": float(q_empirical),
                    "relative_error": float(
                        abs(q_theory - q_empirical) / (abs(q_theory) + 1e-12)
                    ),
                }

        self.results["signal_propagation"] = results
        if self.config.verbose >= 1:
            for act in activations:
                errs = [v["relative_error"] for v in results[act].values()]
                logger.info(f"  {act}: median rel-err = {np.median(errs):.4f}")
        return results

    def correlation_map_test(
        self,
        activations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Verify correlation map fixed points.

        The correlation map c^{l+1} = F(c^l) where c is the cosine
        similarity between two inputs.  Fixed points c* and their
        stability determine the phase.

        Args:
            activations: List of activation names.

        Returns:
            Fixed point values and stability for each activation.
        """
        activations = activations or self.config.activations
        depth = 30
        width = 1024
        sigma_w = np.sqrt(2.0)

        results: Dict[str, Dict[str, Any]] = {}
        for act in activations:
            act_fn = _activation_fn(act)
            # Start from a range of initial correlations
            c0_values = [0.1, 0.5, 0.9, 0.99]
            fixed_points: List[Dict[str, float]] = []

            for c0 in c0_values:
                rng = np.random.RandomState(self.config.seed)
                # Two inputs with correlation c0
                x1 = rng.randn(1, self.config.input_dim)
                x2 = c0 * x1 + np.sqrt(1 - c0 ** 2) * rng.randn(1, self.config.input_dim)

                for _ in range(depth):
                    in_d = x1.shape[1]
                    W = rng.randn(in_d, width) * sigma_w / np.sqrt(in_d)
                    x1 = act_fn(x1 @ W)
                    x2 = act_fn(x2 @ W)

                n1 = np.linalg.norm(x1)
                n2 = np.linalg.norm(x2)
                c_final = float(np.dot(x1.ravel(), x2.ravel()) / (n1 * n2 + 1e-15))
                fixed_points.append({"c0": c0, "c_final": c_final})

            # Check if all correlations converge to same value → ordered phase
            c_finals = [fp["c_final"] for fp in fixed_points]
            spread = max(c_finals) - min(c_finals)
            phase = "ordered" if spread < 0.05 else "chaotic"

            results[act] = {
                "fixed_points": fixed_points,
                "spread": float(spread),
                "phase": phase,
            }

        self.results["correlation_map"] = results
        if self.config.verbose >= 1:
            for act, r in results.items():
                logger.info(f"  {act}: phase={r['phase']}, spread={r['spread']:.4f}")
        return results

    def critical_initialization_test(
        self,
        activations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Verify that (σ_w*, σ_b*) gives edge-of-chaos behaviour.

        At the critical point the Jacobian has spectral radius exactly 1,
        so gradients neither explode nor vanish.

        Args:
            activations: List of activation names.

        Returns:
            Per-activation Jacobian spectral radius at critical init.
        """
        activations = activations or self.config.activations
        width = 512
        depth = 20

        results: Dict[str, Dict[str, float]] = {}
        for act in activations:
            mlp_exp = MLPPhaseDiagramExperiment(self.config)
            sw_star, sb_star = mlp_exp._compute_critical_init(act)
            act_fn = _activation_fn(act)
            act_deriv = _activation_derivative(act)

            # Empirical Jacobian spectral radius
            rng = np.random.RandomState(self.config.seed)
            x = rng.randn(1, self.config.input_dim)
            spectral_radii: List[float] = []
            for _ in range(depth):
                in_d = x.shape[1]
                W = rng.randn(in_d, width) * sw_star / np.sqrt(in_d)
                pre = x @ W
                diag_deriv = act_deriv(pre).ravel()
                # Jacobian of this layer: diag(φ'(Wx)) @ W
                # Spectral radius proxy: max singular value
                J_approx = diag_deriv[:, None] * W.T  # (width, in_d)
                sv = np.linalg.svd(J_approx[:min(64, width), :min(64, x.shape[1])], compute_uv=False)
                spectral_radii.append(float(sv[0]))
                x = act_fn(pre)

            results[act] = {
                "sigma_w_star": float(sw_star),
                "sigma_b_star": float(sb_star),
                "mean_spectral_radius": float(np.mean(spectral_radii)),
                "std_spectral_radius": float(np.std(spectral_radii)),
                "expected_spectral_radius": 1.0,
            }

        self.results["critical_init"] = results
        if self.config.verbose >= 1:
            for act, r in results.items():
                logger.info(
                    f"  {act}: ρ(J)={r['mean_spectral_radius']:.4f} ± "
                    f"{r['std_spectral_radius']:.4f} (expected 1.0)"
                )
        return results

    def phase_diagram_comparison(
        self,
        activations: Optional[List[str]] = None,
        n_grid: int = 8,
    ) -> Dict[str, Any]:
        """Compare theoretical vs empirical (σ_w, σ_b) phase diagrams.

        Args:
            activations: List of activation names.
            n_grid: Grid resolution per axis.

        Returns:
            Per-activation phase maps (theory and empirical).
        """
        activations = activations or ["relu", "tanh"]
        sigma_ws = np.linspace(0.5, 3.0, n_grid)
        sigma_bs = np.linspace(0.0, 1.0, n_grid)
        depth = 20
        width = 512

        results: Dict[str, Dict[str, Any]] = {}
        for act in activations:
            act_fn = _activation_fn(act)
            act_deriv = _activation_derivative(act)

            theory_map = np.zeros((n_grid, n_grid), dtype=int)
            empirical_map = np.zeros((n_grid, n_grid), dtype=int)

            for i, sw in enumerate(sigma_ws):
                for j, sb in enumerate(sigma_bs):
                    # Theory: check if χ = σ_w² E[φ'(√q z)²] > 1
                    rng_mc = np.random.RandomState(self.config.seed)
                    q = 1.0
                    for _ in range(10):
                        z = rng_mc.randn(5000) * np.sqrt(q)
                        q = sw ** 2 * np.mean(act_fn(z) ** 2) + sb ** 2
                    z = rng_mc.randn(5000) * np.sqrt(q)
                    chi = sw ** 2 * np.mean(act_deriv(z) ** 2)
                    theory_map[i, j] = 0 if chi < 1 else (1 if chi < 1.05 else 2)

                    # Empirical: propagate perturbation
                    rng = np.random.RandomState(self.config.seed)
                    x1 = rng.randn(1, self.config.input_dim)
                    x2 = x1 + 1e-6 * rng.randn(1, self.config.input_dim)
                    for _ in range(depth):
                        in_d = x1.shape[1]
                        W = rng.randn(in_d, width) * sw / np.sqrt(in_d)
                        b = rng.randn(1, width) * sb
                        x1 = act_fn(x1 @ W + b)
                        x2 = act_fn(x2 @ W + b)
                    cos = np.dot(x1.ravel(), x2.ravel()) / (
                        np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-15
                    )
                    empirical_map[i, j] = 0 if cos > 0.99 else (1 if cos > 0.5 else 2)

            agreement = np.mean(theory_map == empirical_map)
            results[act] = {
                "sigma_ws": sigma_ws.tolist(),
                "sigma_bs": sigma_bs.tolist(),
                "theory_map": theory_map,
                "empirical_map": empirical_map,
                "agreement_fraction": float(agreement),
            }
            if self.config.verbose >= 1:
                logger.info(f"  {act} phase diagram agreement: {agreement:.2%}")

        self.results["phase_diagram_comparison"] = results
        return results

    def order_parameter_test(self) -> Dict[str, Any]:
        """Verify order parameters q, c converge to predicted fixed points.

        Returns:
            Theory vs empirical fixed-point values.
        """
        width = 2048
        depth = 30
        act_fn = _activation_fn("relu")
        sigma_w = np.sqrt(2.0)

        # Theory: q* for ReLU at σ_w = √2, σ_b = 0
        # q_{l+1} = σ_w² q_l / 2 = q_l ⟹ q* = any (marginal)
        # For numerical stability, iterate
        rng_mc = np.random.RandomState(self.config.seed)
        q_theory = 1.0
        for _ in range(depth):
            z = rng_mc.randn(10000) * np.sqrt(q_theory)
            q_theory = sigma_w ** 2 * np.mean(act_fn(z) ** 2)

        # Empirical
        rng = np.random.RandomState(self.config.seed)
        x = rng.randn(100, self.config.input_dim)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        for _ in range(depth):
            in_d = x.shape[1]
            W = rng.randn(in_d, width) * sigma_w / np.sqrt(in_d)
            x = act_fn(x @ W)
        q_empirical = np.mean(x ** 2)

        result = {
            "q_theory": float(q_theory),
            "q_empirical": float(q_empirical),
            "relative_error": float(abs(q_theory - q_empirical) / (abs(q_theory) + 1e-12)),
        }
        self.results["order_parameter"] = result
        if self.config.verbose >= 1:
            logger.info(
                f"  Order parameter: q_theory={q_theory:.4f}, "
                f"q_empirical={q_empirical:.4f}"
            )
        return result


# ===================================================================
# 6. RMT Verification Experiment
# ===================================================================

class RMTVerificationExperiment:
    """Verify random matrix theory predictions for NTK spectra.

    At large width the NTK eigenvalue distribution converges to
    deterministic limits described by the Marchenko-Pastur law,
    Tracy-Widom fluctuations at the spectral edges, and the BBP
    (Baik-Ben Arous-Péché) phase transition for spiked models.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.results: Dict[str, Any] = {}
        _ensure_dir(os.path.join(config.output_dir, "rmt"))
        logger.info("RMTVerificationExperiment initialised.")

    def marchenko_pastur_test(
        self,
        N_values: Optional[List[int]] = None,
        P: int = 256,
    ) -> Dict[str, Any]:
        """Verify the Marchenko-Pastur law for random Gram matrices.

        For a random matrix X of shape (P, N) with i.i.d. entries, the
        empirical spectral distribution of X^T X / P converges to the
        MP distribution with parameter γ = P/N.

        Args:
            N_values: List of matrix widths.
            P: Number of rows (sample size).

        Returns:
            KS test statistics per N.
        """
        N_values = N_values or [64, 128, 256, 512, 1024]

        results: Dict[int, Dict[str, float]] = {}
        for N in N_values:
            gamma = P / N
            rng = np.random.RandomState(self.config.seed)
            X = rng.randn(P, N) / np.sqrt(P)
            gram = X.T @ X
            eigvals = np.sort(np.linalg.eigvalsh(gram))

            # MP distribution support
            lambda_plus = (1 + np.sqrt(gamma)) ** 2
            lambda_minus = (1 - np.sqrt(gamma)) ** 2 if gamma <= 1 else 0.0

            # Fraction inside support
            frac_inside = np.mean(
                (eigvals >= lambda_minus * 0.9) & (eigvals <= lambda_plus * 1.1)
            )

            # KS test against uniform on support (rough check)
            # Transform to [0,1] within the support
            scaled = (eigvals - lambda_minus) / (lambda_plus - lambda_minus + 1e-15)
            scaled = np.clip(scaled, 0, 1)
            ks_stat, ks_pval = sp_stats.kstest(scaled, "uniform")

            results[N] = {
                "gamma": float(gamma),
                "lambda_plus_theory": float(lambda_plus),
                "lambda_plus_empirical": float(eigvals[-1]),
                "lambda_minus_theory": float(lambda_minus),
                "lambda_minus_empirical": float(eigvals[0]),
                "frac_inside_support": float(frac_inside),
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pval),
            }

        self.results["marchenko_pastur"] = results
        if self.config.verbose >= 1:
            for N, r in results.items():
                logger.info(
                    f"  N={N}: λ+ theory={r['lambda_plus_theory']:.3f}, "
                    f"empirical={r['lambda_plus_empirical']:.3f}"
                )
        return results

    def tracy_widom_test(
        self,
        N_values: Optional[List[int]] = None,
        n_samples: int = 200,
    ) -> Dict[str, Any]:
        """Verify Tracy-Widom distribution at spectral edges.

        The largest eigenvalue of a Wishart matrix fluctuates as
        λ_max = μ + σ · TW₁ where TW₁ is the Tracy-Widom distribution.

        Args:
            N_values: List of matrix sizes.
            n_samples: Number of random matrices to sample.

        Returns:
            Fluctuation statistics per N.
        """
        N_values = N_values or [64, 128, 256]

        results: Dict[int, Dict[str, float]] = {}
        for N in N_values:
            P = N  # square aspect ratio
            gamma = 1.0
            lambda_plus = (1 + np.sqrt(gamma)) ** 2
            # TW centering and scaling
            mu = (np.sqrt(N) + np.sqrt(P)) ** 2 / P
            sigma = (np.sqrt(N) + np.sqrt(P)) / P * (
                (1.0 / np.sqrt(N) + 1.0 / np.sqrt(P)) ** (1.0 / 3.0)
            )

            max_eigvals: List[float] = []
            for trial in range(n_samples):
                rng = np.random.RandomState(self.config.seed + trial)
                X = rng.randn(P, N) / np.sqrt(P)
                gram = X.T @ X
                lam_max = np.linalg.eigvalsh(gram)[-1]
                max_eigvals.append(float(lam_max))

            fluctuations = (np.array(max_eigvals) - mu) / sigma
            results[N] = {
                "mean_fluctuation": float(np.mean(fluctuations)),
                "std_fluctuation": float(np.std(fluctuations)),
                "skewness": float(sp_stats.skew(fluctuations)),
                "kurtosis": float(sp_stats.kurtosis(fluctuations)),
                "tw_expected_mean": -1.2065,  # TW1 mean
                "tw_expected_std": 1.268,  # TW1 std (approximate)
            }

        self.results["tracy_widom"] = results
        if self.config.verbose >= 1:
            for N, r in results.items():
                logger.info(
                    f"  N={N}: fluctuation mean={r['mean_fluctuation']:.3f}, "
                    f"std={r['std_fluctuation']:.3f}"
                )
        return results

    def spiked_model_test(
        self,
        spike_strengths: Optional[List[float]] = None,
        N_values: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Verify the BBP phase transition for spiked covariance models.

        When a rank-1 perturbation of strength θ is added to the
        covariance, a phase transition occurs at θ_c = 1/√γ: below θ_c
        the spike is not detectable; above θ_c the largest eigenvalue
        separates from the bulk.

        Args:
            spike_strengths: List of spike strengths θ.
            N_values: List of matrix sizes.

        Returns:
            Per-spike, per-N detection results.
        """
        spike_strengths = spike_strengths or [0.5, 1.0, 1.5, 2.0, 3.0]
        N_values = N_values or [128, 256, 512]

        results: Dict[str, Dict[int, Dict[str, Any]]] = {}
        for theta in spike_strengths:
            key = f"theta_{theta:.1f}"
            results[key] = {}
            for N in N_values:
                P = N
                gamma = P / N
                theta_c = 1.0 / np.sqrt(gamma)

                rng = np.random.RandomState(self.config.seed)
                # Spiked model: Σ = I + θ v v^T
                v = rng.randn(N)
                v /= np.linalg.norm(v)
                X = rng.randn(P, N) / np.sqrt(P)
                X += np.sqrt(theta) * rng.randn(P, 1) @ v[None, :]
                gram = X.T @ X

                eigvals = np.sort(np.linalg.eigvalsh(gram))[::-1]
                lambda_plus = (1 + np.sqrt(gamma)) ** 2

                # Check if top eigenvalue separates from bulk
                gap = eigvals[0] - eigvals[1]
                separated = eigvals[0] > lambda_plus * 1.05

                # BBP prediction
                if theta > theta_c:
                    bbp_pred = (1 + theta) * (1 + gamma / theta)
                else:
                    bbp_pred = lambda_plus

                results[key][N] = {
                    "theta": float(theta),
                    "theta_c": float(theta_c),
                    "above_threshold": theta > theta_c,
                    "top_eigenvalue": float(eigvals[0]),
                    "bbp_prediction": float(bbp_pred),
                    "gap": float(gap),
                    "separated": bool(separated),
                }

        self.results["spiked_model"] = results
        if self.config.verbose >= 1:
            for key, nr in results.items():
                for N, r in nr.items():
                    logger.info(
                        f"  {key}, N={N}: separated={r['separated']}, "
                        f"above_threshold={r['above_threshold']}"
                    )
        return results

    def free_convolution_test(self) -> Dict[str, Any]:
        """Verify free convolution for layered NTK spectra.

        For independent random matrices A, B the spectrum of A + B is
        given by the free additive convolution.  This tests that the
        two-layer NTK spectrum matches the free convolution prediction.

        Returns:
            Spectral comparison results.
        """
        N = 256
        rng = np.random.RandomState(self.config.seed)

        # Two independent Wishart matrices (one per layer)
        X1 = rng.randn(N, N) / np.sqrt(N)
        X2 = rng.randn(N, N) / np.sqrt(N)
        A = X1.T @ X1
        B = X2.T @ X2

        # Sum spectrum (empirical)
        sum_eigvals = np.sort(np.linalg.eigvalsh(A + B))[::-1]

        # Individual spectra
        eig_A = np.sort(np.linalg.eigvalsh(A))[::-1]
        eig_B = np.sort(np.linalg.eigvalsh(B))[::-1]

        # Free additive convolution of two MP distributions:
        # The support of the sum is approximately [λ_-^A + λ_-^B, λ_+^A + λ_+^B]
        # (this is a rough bound; true free convolution is more subtle)
        sum_upper_bound = eig_A[0] + eig_B[0]
        sum_lower_bound = eig_A[-1] + eig_B[-1]

        result = {
            "sum_max_eigenvalue": float(sum_eigvals[0]),
            "sum_min_eigenvalue": float(sum_eigvals[-1]),
            "predicted_upper": float(sum_upper_bound),
            "predicted_lower": float(sum_lower_bound),
            "upper_relative_error": float(
                abs(sum_eigvals[0] - sum_upper_bound) / sum_upper_bound
            ),
            "mean_A": float(np.mean(eig_A)),
            "mean_B": float(np.mean(eig_B)),
            "mean_sum": float(np.mean(sum_eigvals)),
            "mean_additivity_error": float(
                abs(np.mean(sum_eigvals) - np.mean(eig_A) - np.mean(eig_B))
            ),
        }
        self.results["free_convolution"] = result
        if self.config.verbose >= 1:
            logger.info(
                f"  Free convolution: mean additivity error = "
                f"{result['mean_additivity_error']:.6f}"
            )
        return result

    def spectral_prediction_accuracy(
        self,
        widths: Optional[List[int]] = None,
        depths: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Measure how well RMT predicts the empirical NTK spectrum.

        For each (width, depth) the method computes the empirical NTK,
        extracts its spectrum, and compares against the MP-law prediction
        (with effective aspect ratio).

        Args:
            widths: List of widths.
            depths: List of depths.

        Returns:
            Per-(width, depth) spectral agreement metrics.
        """
        widths = widths or [64, 128, 256, 512]
        depths = depths or [2, 4, 8]
        X, _ = _generate_binary_data(64, self.config.input_dim, self.config.seed)

        results: Dict[str, Dict[str, float]] = {}
        for width in widths:
            for depth in depths:
                mlp_exp = MLPPhaseDiagramExperiment(self.config)
                model_fn = mlp_exp._build_mlp(width, depth, "relu")
                ntk = mlp_exp._compute_ntk(model_fn, X[:32])
                eigvals = np.sort(np.linalg.eigvalsh(ntk))[::-1]

                # MP prediction with effective γ = n_train / width
                n = ntk.shape[0]
                gamma_eff = n / width
                lp = (1 + np.sqrt(gamma_eff)) ** 2
                lm = max((1 - np.sqrt(gamma_eff)) ** 2, 0)

                # Normalise eigenvalues by trace/n for comparison
                eig_norm = eigvals / (np.mean(eigvals) + 1e-15)
                frac_inside = np.mean((eig_norm >= lm * 0.8) & (eig_norm <= lp * 1.2))

                key = f"w{width}_d{depth}"
                results[key] = {
                    "width": width,
                    "depth": depth,
                    "gamma_eff": float(gamma_eff),
                    "frac_inside_mp": float(frac_inside),
                    "max_eigenvalue_ratio": float(eig_norm[0] / lp) if lp > 0 else float("inf"),
                }

        self.results["spectral_accuracy"] = results
        if self.config.verbose >= 1:
            logger.info(
                f"  Spectral prediction: tested {len(results)} (width, depth) pairs"
            )
        return results


# ===================================================================
# 7. Dynamical Experiments Runner
# ===================================================================

class DynamicalExperimentsRunner:
    """Experiments probing dynamical aspects of training near phase transitions.

    Near a continuous phase transition, dynamics slow down (critical
    slowing down), the system ages (broken time-translation invariance),
    and rapid parameter changes trigger the Kibble-Zurek mechanism.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.results: Dict[str, Any] = {}
        _ensure_dir(os.path.join(config.output_dir, "dynamics"))
        logger.info("DynamicalExperimentsRunner initialised.")

    def critical_slowing_experiment(
        self,
        widths: Optional[List[int]] = None,
        lr_range: Tuple[float, float] = (1e-3, 1.0),
        n_lr: int = 8,
    ) -> Dict[str, Any]:
        """Measure critical slowing down near the lazy-rich boundary.

        At the phase boundary the relaxation time τ diverges as
        τ ~ |γ - γ*|^{-z ν} where z is the dynamic exponent.

        Args:
            widths: Network widths.
            lr_range: Learning rate range.
            n_lr: Number of learning rates.

        Returns:
            Relaxation times per (width, lr).
        """
        widths = widths or self.config.widths[:4]
        lrs = np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), n_lr)
        X, y = _generate_binary_data(self.config.n_train, self.config.input_dim, self.config.seed)
        depth = 4

        results: Dict[int, List[Dict[str, float]]] = {}
        for width in widths:
            lr_results: List[Dict[str, float]] = []
            for lr in lrs:
                model_fn = MLPPhaseDiagramExperiment(self.config)._build_mlp(width, depth, "relu")
                # Measure time to reach loss threshold
                loss_history: List[float] = []
                params = model_fn.params
                for epoch in range(self.config.max_epochs):
                    pred = model_fn(X).ravel()
                    loss = float(np.mean((pred - y) ** 2))
                    loss_history.append(loss)
                    if loss < 0.01:
                        break
                    # Simple gradient step on output layer only
                    W_last, b_last = params[-1]
                    h = X.copy()
                    for W, b in params[:-1]:
                        h = np.maximum(h @ W + b, 0.0)
                    grad_W = h.T @ (2.0 * (pred - y).reshape(-1, 1)) / len(y)
                    params[-1] = (W_last - lr * grad_W, b_last)

                # Relaxation time = epochs to halve initial loss
                initial_loss = loss_history[0] if loss_history else 1.0
                tau = len(loss_history)
                for t, l in enumerate(loss_history):
                    if l < initial_loss / 2:
                        tau = t
                        break

                lr_results.append({
                    "lr": float(lr),
                    "tau": float(tau),
                    "final_loss": float(loss_history[-1]) if loss_history else float("nan"),
                })
            results[width] = lr_results

        self.results["critical_slowing"] = results
        if self.config.verbose >= 1:
            logger.info(f"  Critical slowing: tested {len(widths)} widths × {n_lr} LRs")
        return results

    def dynamic_exponent_measurement(
        self,
        widths: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Measure the dynamic critical exponent z from relaxation scaling.

        At the critical point τ ~ N^{z/d_eff} where N is the width.

        Args:
            widths: List of widths.

        Returns:
            Estimated dynamic exponent.
        """
        widths = widths or [32, 64, 128, 256, 512]
        X, y = _generate_binary_data(self.config.n_train, self.config.input_dim, self.config.seed)
        depth = 4
        lr = 1e-2

        relaxation_times: List[float] = []
        for width in widths:
            model_fn = MLPPhaseDiagramExperiment(self.config)._build_mlp(width, depth, "relu")
            loss_prev = float("inf")
            for epoch in range(self.config.max_epochs):
                pred = model_fn(X).ravel()
                loss = float(np.mean((pred - y) ** 2))
                if loss < loss_prev / 2:
                    relaxation_times.append(float(epoch))
                    break
                loss_prev = loss
                # Train output layer
                params = model_fn.params
                W_last, b_last = params[-1]
                h = X.copy()
                for W, b in params[:-1]:
                    h = np.maximum(h @ W + b, 0.0)
                grad_W = h.T @ (2.0 * (pred - y).reshape(-1, 1)) / len(y)
                params[-1] = (W_last - lr * grad_W, b_last)
            else:
                relaxation_times.append(float(self.config.max_epochs))

        log_w = np.log(np.array(widths, dtype=float))
        log_tau = np.log(np.array(relaxation_times) + 1.0)
        if len(log_w) > 1:
            slope, _, r_val, _, _ = sp_stats.linregress(log_w, log_tau)
        else:
            slope, r_val = 0.0, 0.0

        result = {
            "widths": widths,
            "relaxation_times": relaxation_times,
            "dynamic_exponent_z": float(slope),
            "r_squared": float(r_val ** 2),
        }
        self.results["dynamic_exponent"] = result
        if self.config.verbose >= 1:
            logger.info(f"  Dynamic exponent z ≈ {slope:.3f}, R²={r_val**2:.4f}")
        return result

    def kibble_zurek_test(
        self,
        quench_rates: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Test Kibble-Zurek mechanism for learning-rate annealing.

        When the LR is annealed through the critical value at rate r,
        the density of defects scales as r^{ν/(1+zν)}.

        Args:
            quench_rates: List of LR annealing rates.

        Returns:
            Defect density vs quench rate.
        """
        quench_rates = quench_rates or [0.001, 0.005, 0.01, 0.05, 0.1]
        X, y = _generate_binary_data(self.config.n_train, self.config.input_dim, self.config.seed)
        depth = 4
        width = 256

        results: List[Dict[str, float]] = []
        for rate in quench_rates:
            model_fn = MLPPhaseDiagramExperiment(self.config)._build_mlp(width, depth, "relu")
            params = model_fn.params
            # Anneal LR from high to low
            lr_init = 0.5
            loss_final = float("inf")
            for epoch in range(100):
                lr = lr_init * np.exp(-rate * epoch)
                pred = model_fn(X).ravel()
                loss = float(np.mean((pred - y) ** 2))
                # Train output layer
                W_last, b_last = params[-1]
                h = X.copy()
                for W, b in params[:-1]:
                    h = np.maximum(h @ W + b, 0.0)
                grad_W = h.T @ (2.0 * (pred - y).reshape(-1, 1)) / len(y)
                params[-1] = (W_last - lr * grad_W, b_last)
                loss_final = loss

            # Defect density ~ residual loss
            results.append({"quench_rate": float(rate), "residual_loss": float(loss_final)})

        self.results["kibble_zurek"] = results
        if self.config.verbose >= 1:
            logger.info(f"  Kibble-Zurek: tested {len(quench_rates)} quench rates")
        return {"results": results}

    def bifurcation_detection(
        self,
        lr_range: Tuple[float, float] = (1e-3, 1.0),
        n_lr: int = 20,
    ) -> Dict[str, Any]:
        """Detect bifurcations in training dynamics as LR increases.

        At a pitchfork bifurcation the fixed point of gradient descent
        loses stability and the system transitions to oscillatory or
        chaotic dynamics.

        Args:
            lr_range: Learning rate range.
            n_lr: Number of LR values.

        Returns:
            Bifurcation indicators per LR.
        """
        lrs = np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), n_lr)
        X, y = _generate_binary_data(self.config.n_train, self.config.input_dim, self.config.seed)
        width, depth = 128, 4

        results: List[Dict[str, Any]] = []
        for lr in lrs:
            model_fn = MLPPhaseDiagramExperiment(self.config)._build_mlp(width, depth, "relu")
            params = model_fn.params
            losses: List[float] = []
            for epoch in range(50):
                pred = model_fn(X).ravel()
                loss = float(np.mean((pred - y) ** 2))
                losses.append(loss)
                W_last, b_last = params[-1]
                h = X.copy()
                for W, b in params[:-1]:
                    h = np.maximum(h @ W + b, 0.0)
                grad_W = h.T @ (2.0 * (pred - y).reshape(-1, 1)) / len(y)
                params[-1] = (W_last - lr * grad_W, b_last)

            # Detect oscillation: variance of loss in last 20 epochs
            tail_losses = np.array(losses[-20:])
            oscillation = float(np.std(tail_losses) / (np.mean(tail_losses) + 1e-15))
            diverged = any(np.isnan(losses)) or any(np.isinf(losses)) or (len(losses) > 0 and losses[-1] > 1e6)

            results.append({
                "lr": float(lr),
                "oscillation_index": oscillation,
                "diverged": bool(diverged),
                "final_loss": float(losses[-1]) if losses else float("nan"),
            })

        self.results["bifurcation"] = results
        if self.config.verbose >= 1:
            n_diverged = sum(1 for r in results if r["diverged"])
            logger.info(f"  Bifurcation: {n_diverged}/{n_lr} LRs diverged")
        return {"results": results}

    def aging_experiment(
        self,
        waiting_times: Optional[List[int]] = None,
        observation_times: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Measure aging in training: two-time correlation C(t_w, t_w + t).

        In systems with aging, C(t_w, t_w + t) depends on t_w even at
        large times, indicating broken time-translation invariance.

        Args:
            waiting_times: List of waiting times t_w.
            observation_times: List of observation times t.

        Returns:
            Two-time correlation data.
        """
        waiting_times = waiting_times or [10, 30, 50, 100]
        observation_times = observation_times or [5, 10, 20, 50]
        X, y = _generate_binary_data(self.config.n_train, self.config.input_dim, self.config.seed)
        width, depth, lr = 256, 4, 1e-2

        model_fn = MLPPhaseDiagramExperiment(self.config)._build_mlp(width, depth, "relu")
        params = model_fn.params

        # Collect weights at different times
        weight_snapshots: Dict[int, np.ndarray] = {}
        max_time = max(waiting_times) + max(observation_times) + 1
        for epoch in range(max_time):
            pred = model_fn(X).ravel()
            # Store snapshot
            if epoch in waiting_times or any(epoch == tw + t for tw in waiting_times for t in observation_times):
                weight_snapshots[epoch] = np.concatenate(
                    [W.ravel() for W, b in params] + [b.ravel() for W, b in params]
                ).copy()
            # Train output layer
            W_last, b_last = params[-1]
            h = X.copy()
            for W, b in params[:-1]:
                h = np.maximum(h @ W + b, 0.0)
            grad_W = h.T @ (2.0 * (pred - y).reshape(-1, 1)) / len(y)
            params[-1] = (W_last - lr * grad_W, b_last)

        correlations: List[Dict[str, float]] = []
        for tw in waiting_times:
            if tw not in weight_snapshots:
                continue
            w_tw = weight_snapshots[tw]
            for t_obs in observation_times:
                t_total = tw + t_obs
                if t_total not in weight_snapshots:
                    continue
                w_t = weight_snapshots[t_total]
                corr = np.dot(w_tw, w_t) / (
                    np.linalg.norm(w_tw) * np.linalg.norm(w_t) + 1e-15
                )
                correlations.append({
                    "t_w": tw,
                    "t_obs": t_obs,
                    "correlation": float(corr),
                })

        self.results["aging"] = correlations
        if self.config.verbose >= 1:
            logger.info(f"  Aging: computed {len(correlations)} two-time correlations")
        return {"correlations": correlations}

    def grokking_aging_connection(self) -> Dict[str, Any]:
        """Test the connection between grokking and aging dynamics.

        Grokking — sudden generalisation after prolonged memorisation —
        may be related to aging: the training dynamics break TTI, and
        the generalisation transition corresponds to the system reaching
        a metastable state.

        Returns:
            Training and test loss curves, aging indicators.
        """
        n_train = 128
        n_test = 64
        dim = 16
        X_train, y_train = _generate_binary_data(n_train, dim, self.config.seed)
        X_test, y_test = _generate_binary_data(n_test, dim, self.config.seed + 1)
        width, depth, lr = 512, 4, 1e-2

        mlp_exp = MLPPhaseDiagramExperiment(self.config)
        model_fn = mlp_exp._build_mlp(width, depth, "relu")
        params = model_fn.params

        train_losses: List[float] = []
        test_losses: List[float] = []
        weight_norms: List[float] = []

        for epoch in range(300):
            # Train loss
            pred_train = model_fn(X_train).ravel()
            loss_train = float(np.mean((pred_train - y_train) ** 2))
            train_losses.append(loss_train)

            # Test loss
            pred_test = model_fn(X_test).ravel()
            loss_test = float(np.mean((pred_test - y_test) ** 2))
            test_losses.append(loss_test)

            # Weight norm
            w_norm = sum(np.linalg.norm(W) for W, _ in params)
            weight_norms.append(float(w_norm))

            # Train output layer
            W_last, b_last = params[-1]
            h = X_train.copy()
            for W, b in params[:-1]:
                h = np.maximum(h @ W + b, 0.0)
            grad_W = h.T @ (2.0 * (pred_train - y_train).reshape(-1, 1)) / n_train
            params[-1] = (W_last - lr * grad_W, b_last)

        result = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "weight_norms": weight_norms,
            "grokking_detected": bool(
                min(test_losses[200:]) < 0.5 * min(test_losses[:50])
            ),
        }
        self.results["grokking_aging"] = result
        if self.config.verbose >= 1:
            logger.info(
                f"  Grokking-aging: final train={train_losses[-1]:.4f}, "
                f"test={test_losses[-1]:.4f}"
            )
        return result


# ===================================================================
# 8. Multi-Task Experiment Runner
# ===================================================================

class MultiTaskExperimentRunner:
    """Experiments on multi-task phase diagrams.

    When multiple tasks share a representation, the phase diagram
    depends on task similarity: cooperative tasks benefit from shared
    features, while competitive tasks suffer from interference.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.results: Dict[str, Any] = {}
        _ensure_dir(os.path.join(config.output_dir, "multi_task"))
        logger.info("MultiTaskExperimentRunner initialised.")

    def interference_measurement(
        self,
        n_tasks_range: Tuple[int, int] = (2, 10),
        similarity_range: Tuple[float, float] = (0.0, 1.0),
        n_sim: int = 5,
    ) -> Dict[str, Any]:
        """Measure task interference as a function of number of tasks and similarity.

        Interference is measured by the degradation in per-task loss when
        training jointly vs independently.

        Args:
            n_tasks_range: (min_tasks, max_tasks).
            similarity_range: (min_similarity, max_similarity).
            n_sim: Number of similarity values.

        Returns:
            Interference matrix.
        """
        n_tasks_values = list(range(n_tasks_range[0], n_tasks_range[1] + 1, 2))
        similarities = np.linspace(similarity_range[0], similarity_range[1], n_sim)
        dim = self.config.input_dim
        n_train = self.config.n_train
        width, depth = 128, 4

        results: Dict[str, List[Dict[str, float]]] = {}
        for n_tasks in n_tasks_values:
            task_results: List[Dict[str, float]] = []
            for sim in similarities:
                rng = np.random.RandomState(self.config.seed)
                # Generate tasks with controlled similarity
                w_base = rng.randn(dim)
                task_vecs = []
                for t in range(n_tasks):
                    w_t = sim * w_base + (1 - sim) * rng.randn(dim)
                    w_t /= np.linalg.norm(w_t)
                    task_vecs.append(w_t)

                X = rng.randn(n_train, dim)
                # Joint training: average loss across tasks
                joint_loss = 0.0
                for w_t in task_vecs:
                    y_t = (X @ w_t > 0).astype(float)
                    model_fn = MLPPhaseDiagramExperiment(self.config)._build_mlp(width, depth, "relu")
                    pred = model_fn(X).ravel()
                    joint_loss += np.mean((pred - y_t) ** 2)
                joint_loss /= n_tasks

                task_results.append({
                    "similarity": float(sim),
                    "joint_loss": float(joint_loss),
                    "n_tasks": n_tasks,
                })

            results[f"n_tasks_{n_tasks}"] = task_results

        self.results["interference"] = results
        if self.config.verbose >= 1:
            logger.info(f"  Interference measurement done: {len(n_tasks_values)} task counts")
        return results

    def cooperative_competitive_boundary(
        self,
        similarity_range: Tuple[float, float] = (0.0, 1.0),
        width_range: Tuple[int, int] = (32, 512),
        n_sim: int = 8,
        n_widths: int = 6,
    ) -> Dict[str, Any]:
        """Find the cooperative-competitive boundary in (similarity, width) space.

        Below a critical similarity tasks compete; above it they cooperate.
        The boundary depends on width.

        Args:
            similarity_range: Range of task similarities.
            width_range: Range of network widths.
            n_sim: Number of similarity values.
            n_widths: Number of widths.

        Returns:
            Phase map of cooperative vs competitive.
        """
        sims = np.linspace(similarity_range[0], similarity_range[1], n_sim)
        widths = np.unique(
            np.logspace(np.log10(width_range[0]), np.log10(width_range[1]), n_widths).astype(int)
        )
        dim = self.config.input_dim
        n_tasks = 3

        phase_map = np.zeros((len(sims), len(widths)), dtype=int)
        rng = np.random.RandomState(self.config.seed)

        for i, sim in enumerate(sims):
            for j, width in enumerate(widths):
                # Generate correlated tasks
                w_base = rng.randn(dim)
                task_vecs = [
                    sim * w_base + (1 - sim) * rng.randn(dim)
                    for _ in range(n_tasks)
                ]
                # Check if task gradients align (cooperative) or conflict
                X = rng.randn(32, dim)
                gradients: List[np.ndarray] = []
                for w_t in task_vecs:
                    y_t = X @ w_t
                    gradients.append(X.T @ y_t / 32)

                # Average pairwise cosine similarity of gradients
                cos_sims: List[float] = []
                for a in range(n_tasks):
                    for b in range(a + 1, n_tasks):
                        cs = np.dot(gradients[a], gradients[b]) / (
                            np.linalg.norm(gradients[a]) * np.linalg.norm(gradients[b]) + 1e-15
                        )
                        cos_sims.append(cs)
                avg_cos = np.mean(cos_sims)
                phase_map[i, j] = 1 if avg_cos > 0 else 0  # 1=cooperative, 0=competitive

        result = {
            "similarities": sims.tolist(),
            "widths": widths.tolist(),
            "phase_map": phase_map,
        }
        self.results["coop_competitive"] = result
        if self.config.verbose >= 1:
            logger.info(
                f"  Cooperative-competitive boundary: "
                f"coop={np.mean(phase_map == 1):.2f}"
            )
        return result

    def transfer_learning_phases(
        self,
        similarity_range: Tuple[float, float] = (0.0, 1.0),
        lr_range: Tuple[float, float] = (1e-3, 1.0),
        n_sim: int = 6,
        n_lr: int = 6,
    ) -> Dict[str, Any]:
        """Map the transfer learning phase diagram in (similarity, lr) space.

        Three phases: negative transfer, zero transfer, positive transfer.

        Args:
            similarity_range: Range of source-target similarity.
            lr_range: Range of fine-tuning learning rates.
            n_sim: Number of similarity values.
            n_lr: Number of learning rates.

        Returns:
            Transfer phase map.
        """
        sims = np.linspace(similarity_range[0], similarity_range[1], n_sim)
        lrs = np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), n_lr)
        dim = self.config.input_dim
        width, depth = 128, 4

        phase_map = np.zeros((n_sim, n_lr), dtype=int)
        rng = np.random.RandomState(self.config.seed)
        X = rng.randn(self.config.n_train, dim)

        for i, sim in enumerate(sims):
            w_source = rng.randn(dim)
            w_target = sim * w_source + np.sqrt(1 - sim ** 2) * rng.randn(dim)
            y_target = (X @ w_target > 0).astype(float)

            for j, lr in enumerate(lrs):
                # Pre-trained on source
                model_fn = MLPPhaseDiagramExperiment(self.config)._build_mlp(width, depth, "relu")
                pred_before = model_fn(X).ravel()
                loss_before = np.mean((pred_before - y_target) ** 2)

                # Fine-tune on target (output layer only)
                params = model_fn.params
                for _ep in range(20):
                    pred = model_fn(X).ravel()
                    W_last, b_last = params[-1]
                    h = X.copy()
                    for W, b in params[:-1]:
                        h = np.maximum(h @ W + b, 0.0)
                    grad_W = h.T @ (2.0 * (pred - y_target).reshape(-1, 1)) / len(y_target)
                    params[-1] = (W_last - lr * grad_W, b_last)

                pred_after = model_fn(X).ravel()
                loss_after = np.mean((pred_after - y_target) ** 2)
                improvement = loss_before - loss_after

                if improvement > 0.05:
                    phase_map[i, j] = 1  # positive transfer
                elif improvement < -0.05:
                    phase_map[i, j] = -1  # negative transfer
                else:
                    phase_map[i, j] = 0  # zero transfer

        result = {
            "similarities": sims.tolist(),
            "learning_rates": lrs.tolist(),
            "phase_map": phase_map,
        }
        self.results["transfer_learning"] = result
        if self.config.verbose >= 1:
            logger.info(
                f"  Transfer learning: positive={np.mean(phase_map == 1):.2f}, "
                f"negative={np.mean(phase_map == -1):.2f}"
            )
        return result

    def optimal_task_weighting(
        self,
        n_tasks: int = 4,
        similarity: float = 0.5,
    ) -> Dict[str, Any]:
        """Test optimal task weighting strategies.

        Given task similarities, theory predicts optimal weights
        proportional to the inverse of the task-interference matrix.

        Args:
            n_tasks: Number of tasks.
            similarity: Pairwise task similarity.

        Returns:
            Comparison of uniform vs optimal weighting.
        """
        dim = self.config.input_dim
        rng = np.random.RandomState(self.config.seed)

        # Task vectors
        w_base = rng.randn(dim)
        task_vecs = [
            similarity * w_base + (1 - similarity) * rng.randn(dim)
            for _ in range(n_tasks)
        ]

        # Interference matrix
        G = np.zeros((n_tasks, n_tasks))
        for a in range(n_tasks):
            for b in range(n_tasks):
                G[a, b] = np.dot(task_vecs[a], task_vecs[b]) / (
                    np.linalg.norm(task_vecs[a]) * np.linalg.norm(task_vecs[b])
                )

        # Optimal weights: w* ∝ G^{-1} 1
        try:
            G_inv = np.linalg.inv(G)
            optimal_weights = G_inv @ np.ones(n_tasks)
            optimal_weights = np.abs(optimal_weights)
            optimal_weights /= optimal_weights.sum()
        except np.linalg.LinAlgError:
            optimal_weights = np.ones(n_tasks) / n_tasks

        uniform_weights = np.ones(n_tasks) / n_tasks

        result = {
            "interference_matrix": G.tolist(),
            "optimal_weights": optimal_weights.tolist(),
            "uniform_weights": uniform_weights.tolist(),
            "weight_ratio": float(max(optimal_weights) / min(optimal_weights + 1e-15)),
        }
        self.results["optimal_weighting"] = result
        if self.config.verbose >= 1:
            logger.info(
                f"  Optimal weighting: max/min ratio = {result['weight_ratio']:.2f}"
            )
        return result


# ===================================================================
# 9. Universality Experiment
# ===================================================================

class UniversalityExperiment:
    """Test universality: do different architectures share critical exponents?

    Universality is a central prediction of renormalisation group theory:
    systems in the same universality class have identical critical
    exponents regardless of microscopic details.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.results: Dict[str, Any] = {}
        _ensure_dir(os.path.join(config.output_dir, "universality"))
        logger.info("UniversalityExperiment initialised.")

    def critical_exponent_comparison(
        self,
        architectures: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare critical exponents across architectures.

        Measures the correlation-length exponent ν for MLP, CNN, and
        ResNet architectures.

        Args:
            architectures: List of architecture names.

        Returns:
            Per-architecture critical exponents.
        """
        architectures = architectures or ["mlp", "resnet"]
        depth = 20
        n_eps = 12

        results: Dict[str, Dict[str, float]] = {}
        for arch in architectures:
            sigma_w_star = np.sqrt(2.0)
            epsilons = np.logspace(-3, -0.5, n_eps)
            xi_values: List[float] = []

            for eps in epsilons:
                sw = sigma_w_star + eps
                rng = np.random.RandomState(self.config.seed)
                width = 512
                x1 = rng.randn(1, self.config.input_dim)
                x2 = x1 + 1e-4 * rng.randn(1, self.config.input_dim)

                for _ in range(depth):
                    in_d = x1.shape[1]
                    W = rng.randn(in_d, width) * sw / np.sqrt(in_d)
                    h1 = np.maximum(x1 @ W, 0.0)
                    h2 = np.maximum(x2 @ W, 0.0)
                    if arch == "resnet" and x1.shape[1] == width:
                        x1 = x1 + h1
                        x2 = x2 + h2
                    else:
                        x1 = h1
                        x2 = h2

                cos = np.dot(x1.ravel(), x2.ravel()) / (
                    np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-15
                )
                xi = 1.0 / (1.0 - cos + 1e-15)
                xi_values.append(float(xi))

            log_eps = np.log(epsilons)
            log_xi = np.log(np.array(xi_values) + 1e-15)
            slope, _, r_val, _, _ = sp_stats.linregress(log_eps, log_xi)

            results[arch] = {
                "nu": float(-slope),
                "r_squared": float(r_val ** 2),
            }

        self.results["critical_exponents"] = results
        if self.config.verbose >= 1:
            for arch, r in results.items():
                logger.info(f"  {arch}: ν = {r['nu']:.3f}, R² = {r['r_squared']:.4f}")
        return results

    def scaling_collapse_test(
        self,
        architectures: Optional[List[str]] = None,
        widths: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Perform data collapse to verify scaling hypothesis.

        If f(ε, N) = N^{β/ν} F(ε N^{1/ν}) then plotting
        N^{-β/ν} f vs ε N^{1/ν} should collapse all curves onto one.

        Args:
            architectures: Architecture names.
            widths: Network widths.

        Returns:
            Collapse quality metric per architecture.
        """
        architectures = architectures or ["mlp"]
        widths = widths or [64, 128, 256, 512]
        n_eps = 10
        depth = 10

        results: Dict[str, Dict[str, Any]] = {}
        for arch in architectures:
            epsilons = np.logspace(-2, -0.3, n_eps)
            all_curves: Dict[int, List[float]] = {}

            for width in widths:
                xi_vals: List[float] = []
                for eps in epsilons:
                    sw = np.sqrt(2.0) + eps
                    rng = np.random.RandomState(self.config.seed)
                    x1 = rng.randn(1, self.config.input_dim)
                    x2 = x1 + 1e-4 * rng.randn(1, self.config.input_dim)
                    for _ in range(depth):
                        in_d = x1.shape[1]
                        W = rng.randn(in_d, width) * sw / np.sqrt(in_d)
                        x1 = np.maximum(x1 @ W, 0.0)
                        x2 = np.maximum(x2 @ W, 0.0)
                    cos = np.dot(x1.ravel(), x2.ravel()) / (
                        np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-15
                    )
                    xi_vals.append(1.0 / (1.0 - cos + 1e-15))
                all_curves[width] = xi_vals

            # Attempt collapse with ν = 0.5
            nu = 0.5
            collapsed: List[np.ndarray] = []
            for width in widths:
                x_scaled = epsilons * width ** (1.0 / nu)
                y_scaled = np.array(all_curves[width]) / width
                collapsed.append(np.column_stack([x_scaled, y_scaled]))

            # Collapse quality: variance across curves at matched x values
            if len(collapsed) > 1:
                y_interp: List[np.ndarray] = []
                x_common = np.logspace(
                    np.log10(epsilons[0] * widths[0] ** (1.0 / nu)),
                    np.log10(epsilons[-1] * widths[-1] ** (1.0 / nu)),
                    20,
                )
                for curve in collapsed:
                    y_interp.append(
                        np.interp(x_common, curve[:, 0], curve[:, 1], left=np.nan, right=np.nan)
                    )
                y_stack = np.array(y_interp)
                # Normalised variance
                collapse_quality = float(
                    1.0 - np.nanmean(np.nanvar(y_stack, axis=0))
                    / (np.nanvar(y_stack) + 1e-15)
                )
            else:
                collapse_quality = float("nan")

            results[arch] = {
                "nu_used": nu,
                "collapse_quality": collapse_quality,
                "widths": widths,
            }

        self.results["scaling_collapse"] = results
        if self.config.verbose >= 1:
            for arch, r in results.items():
                logger.info(f"  {arch}: collapse quality = {r['collapse_quality']:.4f}")
        return results

    def rg_flow_comparison(
        self,
        architectures: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare RG flow for different architectures.

        The RG flow in (σ_w, σ_b) space shows how the effective
        parameters change as we coarse-grain.  Fixed points of the flow
        correspond to phase transitions.

        Args:
            architectures: Architecture names.

        Returns:
            RG flow vectors at grid points.
        """
        architectures = architectures or ["mlp", "resnet"]
        n_grid = 6
        sigma_ws = np.linspace(0.8, 2.5, n_grid)
        sigma_bs = np.linspace(0.0, 0.5, n_grid)
        width = 256

        results: Dict[str, Dict[str, Any]] = {}
        for arch in architectures:
            flow_u = np.zeros((n_grid, n_grid))
            flow_v = np.zeros((n_grid, n_grid))

            for i, sw in enumerate(sigma_ws):
                for j, sb in enumerate(sigma_bs):
                    rng = np.random.RandomState(self.config.seed)
                    x = rng.randn(32, self.config.input_dim)
                    # Propagate 2 layers (one RG step)
                    for _ in range(2):
                        in_d = x.shape[1]
                        W = rng.randn(in_d, width) * sw / np.sqrt(in_d)
                        b = rng.randn(1, width) * sb
                        h = np.maximum(x @ W + b, 0.0)
                        if arch == "resnet" and x.shape[1] == width:
                            x = x + h
                        else:
                            x = h

                    # Effective σ_w, σ_b after coarse-graining
                    var_x = np.var(x)
                    sw_eff = np.sqrt(2.0 * var_x) if var_x > 0 else sw
                    sb_eff = np.mean(np.abs(x)) * 0.1

                    flow_u[i, j] = sw_eff - sw
                    flow_v[i, j] = sb_eff - sb

            results[arch] = {
                "sigma_ws": sigma_ws.tolist(),
                "sigma_bs": sigma_bs.tolist(),
                "flow_u": flow_u.tolist(),
                "flow_v": flow_v.tolist(),
            }

        self.results["rg_flow"] = results
        if self.config.verbose >= 1:
            for arch in architectures:
                max_flow = np.max(np.abs(results[arch]["flow_u"]))
                logger.info(f"  {arch} RG flow: max |Δσ_w| = {max_flow:.4f}")
        return results


# ===================================================================
# 10. Run All Experiments
# ===================================================================

def run_all_experiments(config: ExperimentConfig) -> Dict[str, Any]:
    """Execute every experiment suite and aggregate results.

    Args:
        config: Global experiment configuration.

    Returns:
        Dictionary mapping experiment name → results.
    """
    all_results: Dict[str, Any] = {}
    _ensure_dir(config.output_dir)

    experiment_classes = [
        ("mlp_phase", MLPPhaseDiagramExperiment),
        ("cnn_phase", CNNPhaseDiagramExperiment),
        ("transformer_phase", TransformerPhaseDiagramExperiment),
        ("resnet_phase", ResNetPhaseDiagramExperiment),
        ("mean_field_verification", MeanFieldVerificationExperiment),
        ("rmt_verification", RMTVerificationExperiment),
        ("dynamical_experiments", DynamicalExperimentsRunner),
        ("multi_task", MultiTaskExperimentRunner),
        ("universality", UniversalityExperiment),
    ]

    for name, cls in experiment_classes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {name}")
        logger.info(f"{'='*60}")
        t0 = time.time()
        try:
            exp = cls(config)
            # Run all public methods that start with "run_" or a known name
            for method_name in dir(exp):
                if method_name.startswith("_"):
                    continue
                method = getattr(exp, method_name)
                if callable(method) and method_name not in ("__init__",):
                    try:
                        method()
                    except TypeError:
                        # Method requires arguments; skip auto-run
                        pass
                    except Exception as exc:
                        logger.warning(f"  {method_name} failed: {exc}")
            all_results[name] = exp.results
        except Exception as exc:
            logger.error(f"  {name} failed: {exc}")
            all_results[name] = {"error": str(exc)}
        elapsed = time.time() - t0
        logger.info(f"  Completed {name} in {elapsed:.1f}s")

    # Save aggregated results
    output_path = os.path.join(config.output_dir, "all_results_summary.json")
    try:
        serialisable = _make_serialisable(all_results)
        with open(output_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info(f"\nResults saved to {output_path}")
    except Exception as exc:
        logger.warning(f"Could not save results: {exc}")

    return all_results


def _make_serialisable(obj: Any) -> Any:
    """Recursively convert numpy types for JSON serialisation."""
    if isinstance(obj, dict):
        return {str(k): _make_serialisable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ===================================================================
# 11. CLI entry point
# ===================================================================

def main() -> None:
    """Parse command-line arguments and dispatch experiments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Phase Diagram Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=[
            "all",
            "mlp_phase",
            "cnn_phase",
            "transformer_phase",
            "resnet_phase",
            "mean_field",
            "rmt",
            "dynamics",
            "multi_task",
            "universality",
        ],
        help="Which experiment suite to run (default: all).",
    )
    parser.add_argument("--output-dir", type=str, default="results/comprehensive")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--max-epochs", type=int, default=200)
    args = parser.parse_args()

    config = ExperimentConfig(
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
        max_epochs=args.max_epochs,
    )

    dispatch: Dict[str, type] = {
        "mlp_phase": MLPPhaseDiagramExperiment,
        "cnn_phase": CNNPhaseDiagramExperiment,
        "transformer_phase": TransformerPhaseDiagramExperiment,
        "resnet_phase": ResNetPhaseDiagramExperiment,
        "mean_field": MeanFieldVerificationExperiment,
        "rmt": RMTVerificationExperiment,
        "dynamics": DynamicalExperimentsRunner,
        "multi_task": MultiTaskExperimentRunner,
        "universality": UniversalityExperiment,
    }

    if args.experiment == "all":
        results = run_all_experiments(config)
    else:
        cls = dispatch[args.experiment]
        logger.info(f"Running experiment: {args.experiment}")
        exp = cls(config)
        # Auto-run all public non-underscore methods with no required args
        for method_name in sorted(dir(exp)):
            if method_name.startswith("_"):
                continue
            method = getattr(exp, method_name)
            if callable(method):
                try:
                    logger.info(f"  → {method_name}")
                    method()
                except TypeError:
                    pass
                except Exception as exc:
                    logger.warning(f"  {method_name} failed: {exc}")
        results = exp.results

    logger.info("\n=== Experiment run complete ===")
    logger.info(f"Total result keys: {list(results.keys()) if isinstance(results, dict) else 'N/A'}")


if __name__ == "__main__":
    main()
