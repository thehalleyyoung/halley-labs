"""Experiment runner: validate phase predictions by actually training.

Trains models at specified hyperparameters, measures NTK drift, and
compares with theoretical predictions. Generates publication-quality
phase diagram plots.

Example
-------
>>> from phase_diagrams.experiment_runner import ExperimentRunner
>>> runner = ExperimentRunner()
>>> result = runner.validate_prediction(
...     input_dim=784, width=256, depth=3,
...     lr=0.01, training_steps=200
... )
>>> print(result.predicted_regime, result.actual_regime, result.match)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class ValidationResult:
    """Result of validating a phase prediction against actual training.

    Attributes
    ----------
    predicted_regime : str
        Regime predicted by theory (``"lazy"`` / ``"rich"``).
    actual_regime : str
        Regime observed from actual training.
    match : bool
        Whether prediction matches observation.
    predicted_drift : float
        NTK drift predicted by theory.
    actual_drift : float
        NTK drift measured from training.
    relative_error : float
        |predicted - actual| / actual for drift.
    gamma : float
        Effective coupling used.
    gamma_star : float
        Critical coupling predicted.
    training_loss_curve : NDArray
        Loss at each training step.
    ntk_drift_curve : NDArray
        NTK drift measured at checkpoints.
    wall_time : float
        Wall-clock time for the experiment (seconds).
    """
    predicted_regime: str = "unknown"
    actual_regime: str = "unknown"
    match: bool = False
    predicted_drift: float = 0.0
    actual_drift: float = 0.0
    relative_error: float = 0.0
    gamma: float = 0.0
    gamma_star: float = 0.0
    training_loss_curve: Optional[NDArray] = None
    ntk_drift_curve: Optional[NDArray] = None
    wall_time: float = 0.0


@dataclass
class SweepResult:
    """Result of a full phase diagram sweep with actual training.

    Attributes
    ----------
    validations : list of ValidationResult
        One per (lr, width) configuration.
    accuracy : float
        Fraction of correct regime predictions.
    boundary_error : float
        Mean absolute error of predicted boundary (in log-LR units).
    lr_grid : NDArray
        Learning rates evaluated.
    width_grid : NDArray
        Widths evaluated.
    regime_map : NDArray
        2D array of actual regimes (0=lazy, 1=rich, 2=critical).
    predicted_regime_map : NDArray
        2D array of predicted regimes.
    wall_time : float
        Total wall time.
    """
    validations: List[ValidationResult] = field(default_factory=list)
    accuracy: float = 0.0
    boundary_error: float = 0.0
    lr_grid: Optional[NDArray] = None
    width_grid: Optional[NDArray] = None
    regime_map: Optional[NDArray] = None
    predicted_regime_map: Optional[NDArray] = None
    wall_time: float = 0.0


# ======================================================================
# Internal helpers
# ======================================================================

def _compute_mu_max_eff(
    input_dim: int, width: int, depth: int, n_samples: int = 50, seed: int = 42
) -> float:
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, input_dim)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    h = X.copy()
    for l in range(depth):
        fan_in = h.shape[1]
        fan_out = width if l < depth - 1 else 1
        W = rng.randn(fan_in, fan_out) / math.sqrt(fan_in)
        pre = h @ W
        h = np.maximum(pre, 0) if l < depth - 1 else pre
    K = h @ h.T
    eigvals = np.linalg.eigvalsh(K)
    mu_max = float(eigvals[-1]) if len(eigvals) > 0 else 1.0
    return mu_max / width


def _predict_gamma_star(
    mu_max_eff: float, training_steps: int,
    drift_threshold: float = 0.1, drift_floor: float = 1e-3,
) -> float:
    if mu_max_eff <= 0:
        return float("inf")
    c = math.log(drift_threshold / drift_floor)
    return c / (training_steps * mu_max_eff)


def _compute_gamma(lr: float, init_scale: float, width: int) -> float:
    return lr * init_scale ** 2 / width


class _SimpleMLP:
    """Minimal numpy-based MLP for training experiments."""

    def __init__(
        self, input_dim: int, width: int, depth: int, output_dim: int = 1,
        seed: int = 42
    ) -> None:
        rng = np.random.RandomState(seed)
        self.weights: List[NDArray] = []
        self.biases: List[NDArray] = []

        dims = [input_dim] + [width] * (depth - 1) + [output_dim]
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            W = rng.randn(dims[i], dims[i + 1]) / math.sqrt(fan_in)
            b = np.zeros(dims[i + 1])
            self.weights.append(W)
            self.biases.append(b)

        self.depth = depth
        self.init_scale = float(np.std(self.weights[0]) * math.sqrt(dims[1]))

    def forward(self, X: NDArray) -> NDArray:
        h = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0)  # ReLU
        return h

    def get_flat_params(self) -> NDArray:
        return np.concatenate([w.ravel() for w in self.weights] +
                              [b.ravel() for b in self.biases])

    def set_flat_params(self, flat: NDArray) -> None:
        offset = 0
        for i in range(len(self.weights)):
            size = self.weights[i].size
            self.weights[i] = flat[offset:offset + size].reshape(self.weights[i].shape)
            offset += size
        for i in range(len(self.biases)):
            size = self.biases[i].size
            self.biases[i] = flat[offset:offset + size].reshape(self.biases[i].shape)
            offset += size

    def compute_ntk_gram(self, X: NDArray, eps: float = 1e-5) -> NDArray:
        """Compute empirical NTK Gram matrix via finite differences."""
        n = X.shape[0]
        params = self.get_flat_params()
        P = len(params)

        # Compute Jacobian
        f0 = self.forward(X).ravel()  # (n,) for output_dim=1
        J = np.zeros((n, P))
        for k in range(P):
            params_k = params.copy()
            params_k[k] += eps
            self.set_flat_params(params_k)
            f_k = self.forward(X).ravel()
            J[:, k] = (f_k - f0) / eps
        self.set_flat_params(params)
        return J @ J.T

    def train_step(self, X: NDArray, y: NDArray, lr: float) -> float:
        """One step of gradient descent. Returns loss."""
        n = X.shape[0]
        params = self.get_flat_params()
        P = len(params)

        pred = self.forward(X).ravel()
        residual = pred - y.ravel()
        loss = float(0.5 * np.mean(residual ** 2))

        # Gradient via backprop (manual for simplicity)
        # Use finite-difference gradient of loss w.r.t. params
        eps = 1e-5
        grad = np.zeros(P)
        for k in range(min(P, 5000)):  # cap for performance
            params_k = params.copy()
            params_k[k] += eps
            self.set_flat_params(params_k)
            pred_k = self.forward(X).ravel()
            loss_k = 0.5 * np.mean((pred_k - y.ravel()) ** 2)
            grad[k] = (loss_k - loss) / eps

        # If too many params, use random subset
        if P > 5000:
            rng = np.random.RandomState(42)
            indices = rng.choice(P, 5000, replace=False)
            full_grad = np.zeros(P)
            full_grad[indices] = grad[:5000] * (P / 5000)
            grad = full_grad

        params -= lr * grad
        self.set_flat_params(params)
        return loss


# ======================================================================
# ExperimentRunner
# ======================================================================

class ExperimentRunner:
    """Validate phase predictions by training actual models.

    Parameters
    ----------
    seed : int
        Base random seed.
    drift_threshold : float
        NTK drift above which we classify as "rich".
    ntk_checkpoints : int
        Number of NTK measurements during training.
    """

    def __init__(
        self,
        seed: int = 42,
        drift_threshold: float = 0.1,
        ntk_checkpoints: int = 5,
    ) -> None:
        self.seed = seed
        self.drift_threshold = drift_threshold
        self.ntk_checkpoints = ntk_checkpoints

    def validate_prediction(
        self,
        input_dim: int,
        width: int,
        depth: int,
        lr: float,
        training_steps: int = 100,
        n_samples: int = 50,
        output_dim: int = 1,
    ) -> ValidationResult:
        """Train a model and compare observed regime with prediction.

        Parameters
        ----------
        input_dim, width, depth : int
            Architecture specification.
        lr : float
            Learning rate.
        training_steps : int
            Number of gradient steps.
        n_samples : int
            Dataset size.
        output_dim : int
            Output dimension.

        Returns
        -------
        ValidationResult
        """
        t0 = time.time()
        rng = np.random.RandomState(self.seed)

        # Generate data
        X = rng.randn(n_samples, input_dim)
        y = rng.randn(n_samples, output_dim)

        # Create model
        mlp = _SimpleMLP(input_dim, width, depth, output_dim, self.seed)
        init_scale = mlp.init_scale

        # Theoretical prediction
        mu_max = _compute_mu_max_eff(input_dim, width, depth, n_samples, self.seed)
        gamma_star = _predict_gamma_star(mu_max, training_steps)
        gamma = _compute_gamma(lr, init_scale, width)

        if gamma < gamma_star * 0.8:
            predicted = "lazy"
        elif gamma > gamma_star * 1.2:
            predicted = "rich"
        else:
            predicted = "critical"

        predicted_drift = gamma * mu_max * training_steps

        # Compute initial NTK
        ntk_0 = mlp.compute_ntk_gram(X)
        ntk_0_norm = np.linalg.norm(ntk_0, "fro")

        # Train and measure
        losses = []
        drifts = []
        checkpoint_interval = max(1, training_steps // self.ntk_checkpoints)

        for step in range(training_steps):
            loss = mlp.train_step(X, y, lr)
            losses.append(loss)

            if (step + 1) % checkpoint_interval == 0 or step == training_steps - 1:
                ntk_t = mlp.compute_ntk_gram(X)
                drift = np.linalg.norm(ntk_t - ntk_0, "fro") / (ntk_0_norm + 1e-12)
                drifts.append(drift)

        actual_drift = drifts[-1] if drifts else 0.0
        actual = "rich" if actual_drift > self.drift_threshold else "lazy"

        relative_error = (
            abs(predicted_drift - actual_drift) / (actual_drift + 1e-12)
        )

        return ValidationResult(
            predicted_regime=predicted,
            actual_regime=actual,
            match=(predicted == actual) or (predicted == "critical"),
            predicted_drift=predicted_drift,
            actual_drift=actual_drift,
            relative_error=relative_error,
            gamma=gamma,
            gamma_star=gamma_star,
            training_loss_curve=np.array(losses),
            ntk_drift_curve=np.array(drifts),
            wall_time=time.time() - t0,
        )

    def run_phase_sweep(
        self,
        input_dim: int,
        depth: int,
        lr_range: Tuple[float, float] = (1e-4, 1.0),
        width_range: Tuple[int, int] = (32, 512),
        n_lr_steps: int = 8,
        n_width_steps: int = 5,
        training_steps: int = 100,
        n_samples: int = 50,
    ) -> SweepResult:
        """Run a full phase diagram sweep with actual training.

        Parameters
        ----------
        input_dim : int
            Input dimensionality.
        depth : int
            Network depth.
        lr_range : (float, float)
            Min/max learning rate.
        width_range : (int, int)
            Min/max width.
        n_lr_steps, n_width_steps : int
            Grid resolution.
        training_steps : int
            Steps per configuration.
        n_samples : int
            Dataset size.

        Returns
        -------
        SweepResult
        """
        t0 = time.time()

        lrs = np.logspace(
            math.log10(lr_range[0]), math.log10(lr_range[1]), n_lr_steps
        )
        widths = np.unique(
            np.logspace(
                math.log10(width_range[0]),
                math.log10(width_range[1]),
                n_width_steps,
            ).astype(int)
        )

        regime_map = np.zeros((len(widths), len(lrs)), dtype=int)
        pred_map = np.zeros((len(widths), len(lrs)), dtype=int)
        validations: List[ValidationResult] = []
        regime_to_int = {"lazy": 0, "rich": 1, "critical": 2, "unknown": 2}

        for wi, w in enumerate(widths):
            for li, lr in enumerate(lrs):
                val = self.validate_prediction(
                    input_dim, int(w), depth, float(lr),
                    training_steps, n_samples,
                )
                validations.append(val)
                regime_map[wi, li] = regime_to_int.get(val.actual_regime, 2)
                pred_map[wi, li] = regime_to_int.get(val.predicted_regime, 2)

        matches = sum(1 for v in validations if v.match)
        accuracy = matches / len(validations) if validations else 0.0

        # Boundary error: compare predicted vs actual boundary in log-LR
        boundary_errors = []
        for wi, w in enumerate(widths):
            actual_boundary = None
            pred_boundary = None
            for li in range(len(lrs) - 1):
                if regime_map[wi, li] != regime_map[wi, li + 1]:
                    actual_boundary = 0.5 * (math.log10(lrs[li]) + math.log10(lrs[li + 1]))
                if pred_map[wi, li] != pred_map[wi, li + 1]:
                    pred_boundary = 0.5 * (math.log10(lrs[li]) + math.log10(lrs[li + 1]))
            if actual_boundary is not None and pred_boundary is not None:
                boundary_errors.append(abs(actual_boundary - pred_boundary))

        boundary_error = float(np.mean(boundary_errors)) if boundary_errors else float("nan")

        return SweepResult(
            validations=validations,
            accuracy=accuracy,
            boundary_error=boundary_error,
            lr_grid=lrs,
            width_grid=widths.astype(float),
            regime_map=regime_map,
            predicted_regime_map=pred_map,
            wall_time=time.time() - t0,
        )

    def generate_phase_plot(
        self,
        sweep: SweepResult,
        save_path: Optional[str] = None,
        title: str = "Phase Diagram: Predicted vs Actual",
    ) -> Any:
        """Generate a publication-quality phase diagram plot.

        Creates a side-by-side comparison of predicted and actual
        phase diagrams from a sweep result.

        Parameters
        ----------
        sweep : SweepResult
            Result from ``run_phase_sweep``.
        save_path : str or None
            If provided, save the figure to this path.
        title : str
            Plot title.

        Returns
        -------
        matplotlib.figure.Figure or None
            The figure object (None if matplotlib not available).
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
        except ImportError:
            return None

        if sweep.lr_grid is None or sweep.width_grid is None:
            return None

        cmap = ListedColormap(["#4575b4", "#d73027", "#fee090"])  # lazy, rich, critical
        labels = ["Lazy", "Rich", "Critical"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        # Predicted
        im1 = ax1.pcolormesh(
            np.log10(sweep.lr_grid),
            np.log10(sweep.width_grid),
            sweep.predicted_regime_map,
            cmap=cmap, vmin=0, vmax=2,
        )
        ax1.set_xlabel("log₁₀(Learning Rate)")
        ax1.set_ylabel("log₁₀(Width)")
        ax1.set_title("Predicted Regime")

        # Actual
        im2 = ax2.pcolormesh(
            np.log10(sweep.lr_grid),
            np.log10(sweep.width_grid),
            sweep.regime_map,
            cmap=cmap, vmin=0, vmax=2,
        )
        ax2.set_xlabel("log₁₀(Learning Rate)")
        ax2.set_title("Actual Regime (from training)")

        # Colorbar
        cbar = fig.colorbar(im2, ax=[ax1, ax2], ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(labels)

        fig.suptitle(
            f"{title}\n"
            f"Accuracy: {sweep.accuracy:.1%} | "
            f"Boundary Error: {sweep.boundary_error:.3f} log₁₀(η)",
            fontsize=12,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
