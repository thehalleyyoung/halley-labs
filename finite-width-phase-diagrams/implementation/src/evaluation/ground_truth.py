"""Ground-truth training harness for evaluating phase diagram predictions.

Trains ensembles of finite-width neural networks and measures kernel
alignment evolution to provide ground-truth regime classifications.

Provides:
  - TrainingConfig: configuration for ground-truth training
  - TrainingMeasurement: single-epoch measurement
  - TrainingRun: full training trajectory for one seed
  - GroundTruthResult: aggregated results across seeds
  - GroundTruthHarness: orchestrates training and measurement
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import linalg as sp_linalg


# ======================================================================
# Configuration
# ======================================================================


@dataclass
class TrainingConfig:
    """Configuration for ground-truth training runs.

    Parameters
    ----------
    width : int
        Hidden layer width of the MLP.
    learning_rate : float
        SGD learning rate.
    init_scale : float
        Initialization scale for weights (std of Gaussian init).
    depth : int
        Number of hidden layers.
    activation : str
        Activation function name ('relu', 'tanh', 'linear').
    num_epochs : int
        Total number of training epochs.
    batch_size : int
        Mini-batch size for SGD.
    dataset_size : int
        Number of training samples.
    input_dim : int
        Dimensionality of input.
    output_dim : int
        Dimensionality of output.
    num_seeds : int
        Number of independent random seeds for the ensemble.
    measure_interval : int
        Measure kernel alignment every this many epochs.
    """

    width: int = 256
    learning_rate: float = 0.01
    init_scale: float = 1.0
    depth: int = 2
    activation: str = "relu"
    num_epochs: int = 200
    batch_size: int = 32
    dataset_size: int = 128
    input_dim: int = 10
    output_dim: int = 1
    num_seeds: int = 5
    measure_interval: int = 10


# ======================================================================
# Measurement data classes
# ======================================================================


@dataclass
class TrainingMeasurement:
    """A single measurement taken during training.

    Parameters
    ----------
    epoch : int
        Epoch at which the measurement was taken.
    loss : float
        Training loss at this epoch.
    gradient_norm : float
        L2 norm of the parameter gradient.
    kernel_matrix : Optional[np.ndarray]
        NTK Gram matrix, if measured at this epoch.
    kernel_alignment : float
        Alignment between current NTK and initial NTK.
    parameter_norm : float
        L2 norm of all parameters.
    """

    epoch: int = 0
    loss: float = 0.0
    gradient_norm: float = 0.0
    kernel_matrix: Optional[np.ndarray] = None
    kernel_alignment: float = 1.0
    parameter_norm: float = 0.0


@dataclass
class TrainingRun:
    """Full training trajectory from a single seed.

    Parameters
    ----------
    config : TrainingConfig
        Configuration used for this run.
    measurements : List[TrainingMeasurement]
        Time series of measurements.
    final_loss : float
        Loss at the end of training.
    regime_label : str
        Regime classification ('lazy' or 'rich').
    seed : int
        Random seed used.
    """

    config: TrainingConfig = field(default_factory=TrainingConfig)
    measurements: List[TrainingMeasurement] = field(default_factory=list)
    final_loss: float = 0.0
    regime_label: str = ""
    seed: int = 0


@dataclass
class GroundTruthResult:
    """Aggregated ground-truth results across an ensemble of seeds.

    Parameters
    ----------
    runs : List[TrainingRun]
        Individual training runs.
    mean_trajectory : np.ndarray
        Mean loss trajectory across seeds.
    std_trajectory : np.ndarray
        Standard deviation of loss trajectory.
    regime_classification : str
        Consensus regime label.
    confidence_intervals : dict
        Confidence intervals for key quantities.
    """

    runs: List[TrainingRun] = field(default_factory=list)
    mean_trajectory: Optional[np.ndarray] = None
    std_trajectory: Optional[np.ndarray] = None
    regime_classification: str = ""
    confidence_intervals: Dict[str, Any] = field(default_factory=dict)


# ======================================================================
# Activation functions
# ======================================================================


def _get_activation(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return an element-wise activation function by name.

    Parameters
    ----------
    name : str
        One of 'relu', 'tanh', 'linear'.

    Returns
    -------
    fn : callable
        Activation function mapping ndarray -> ndarray.
    """
    if name == "relu":
        return lambda x: np.maximum(x, 0.0)
    elif name == "tanh":
        return np.tanh
    elif name == "linear":
        return lambda x: x
    else:
        raise ValueError(f"Unknown activation: {name}")


def _get_activation_derivative(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return the derivative of an activation function by name.

    Parameters
    ----------
    name : str
        One of 'relu', 'tanh', 'linear'.

    Returns
    -------
    fn : callable
        Derivative function mapping ndarray -> ndarray.
    """
    if name == "relu":
        return lambda x: (x > 0.0).astype(np.float64)
    elif name == "tanh":
        return lambda x: 1.0 - np.tanh(x) ** 2
    elif name == "linear":
        return lambda x: np.ones_like(x)
    else:
        raise ValueError(f"Unknown activation derivative: {name}")


# ======================================================================
# Ground-truth harness
# ======================================================================


class GroundTruthHarness:
    """Harness for ground-truth training and measurement.

    Trains simple MLPs with vanilla SGD and measures the NTK evolution
    to classify the training regime as lazy (kernel) or rich (feature).

    Parameters
    ----------
    config : TrainingConfig
        Training configuration.

    Examples
    --------
    >>> cfg = TrainingConfig(width=128, num_seeds=3, num_epochs=50)
    >>> harness = GroundTruthHarness(cfg)
    >>> result = harness.train_ensemble()
    >>> print(result.regime_classification)
    """

    # Threshold for kernel alignment drift to distinguish lazy vs rich.
    _LAZY_ALIGNMENT_THRESHOLD: float = 0.95
    _FINITE_DIFF_EPS: float = 1e-5

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_network(
        self,
        width: int,
        depth: int,
        activation: str,
        init_scale: float,
        seed: int = 0,
    ) -> Tuple[Callable[..., np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
        """Build a simple MLP with given specifications.

        Parameters
        ----------
        width : int
            Hidden layer width.
        depth : int
            Number of hidden layers.
        activation : str
            Activation function name.
        init_scale : float
            Initialization scale.
        seed : int
            Random seed for initialization.

        Returns
        -------
        forward_fn : callable
            Function (params, x) -> output.
        params : list of (W, b) tuples
            Network parameters.
        """
        dims = self._layer_dims(width, depth)
        params = self._init_params(dims, init_scale, seed)
        act = _get_activation(activation)

        def forward_fn(
            params: List[Tuple[np.ndarray, np.ndarray]], x: np.ndarray
        ) -> np.ndarray:
            return self._forward(params, x, activation)

        return forward_fn, params

    def _layer_dims(self, width: int, depth: int) -> List[int]:
        """Compute layer dimensions for the MLP.

        Parameters
        ----------
        width : int
            Hidden layer width.
        depth : int
            Number of hidden layers.

        Returns
        -------
        dims : list of int
            List of layer widths [input_dim, width, ..., width, output_dim].
        """
        dims = [self.config.input_dim]
        for _ in range(depth):
            dims.append(width)
        dims.append(self.config.output_dim)
        return dims

    def _init_params(
        self,
        dims: List[int],
        init_scale: float,
        seed: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Initialize network parameters.

        Uses Gaussian initialization scaled by init_scale / sqrt(fan_in).

        Parameters
        ----------
        dims : list of int
            Layer dimensions.
        init_scale : float
            Overall scale multiplier.
        seed : int
            Random seed.

        Returns
        -------
        params : list of (W, b)
            Weight matrices and bias vectors for each layer.
        """
        rng = np.random.RandomState(seed)
        params: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            fan_out = dims[i + 1]
            scale = init_scale / math.sqrt(fan_in)
            W = rng.randn(fan_in, fan_out).astype(np.float64) * scale
            b = np.zeros(fan_out, dtype=np.float64)
            params.append((W, b))
        return params

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(
        self,
        params: List[Tuple[np.ndarray, np.ndarray]],
        x: np.ndarray,
        activation: str,
    ) -> np.ndarray:
        """Forward pass through the MLP.

        Parameters
        ----------
        params : list of (W, b)
            Network parameters.
        x : ndarray of shape (N, D) or (D,)
            Input data.
        activation : str
            Activation function name.

        Returns
        -------
        output : ndarray of shape (N, C) or (C,)
            Network output.
        """
        act = _get_activation(activation)
        h = np.atleast_2d(x).astype(np.float64)
        for i, (W, b) in enumerate(params):
            h = h @ W + b
            # Apply activation to all but the last layer
            if i < len(params) - 1:
                h = act(h)
        if x.ndim == 1:
            return h.ravel()
        return h

    # ------------------------------------------------------------------
    # Loss and gradients
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        params: List[Tuple[np.ndarray, np.ndarray]],
        x: np.ndarray,
        y: np.ndarray,
        forward_fn: Callable[..., np.ndarray],
    ) -> float:
        """Compute mean squared error loss.

        Parameters
        ----------
        params : list of (W, b)
            Network parameters.
        x : ndarray of shape (N, D)
            Input data.
        y : ndarray of shape (N, C)
            Target data.
        forward_fn : callable
            Forward function (params, x) -> output.

        Returns
        -------
        loss : float
            Mean squared error.
        """
        pred = forward_fn(params, x)
        pred = np.atleast_2d(pred)
        y = np.atleast_2d(y)
        diff = pred - y
        return float(np.mean(diff ** 2))

    def _compute_gradient(
        self,
        params: List[Tuple[np.ndarray, np.ndarray]],
        x: np.ndarray,
        y: np.ndarray,
        forward_fn: Callable[..., np.ndarray],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Compute parameter gradients via finite differences.

        Parameters
        ----------
        params : list of (W, b)
            Network parameters.
        x : ndarray of shape (N, D)
            Input data.
        y : ndarray of shape (N, C)
            Target data.
        forward_fn : callable
            Forward function (params, x) -> output.

        Returns
        -------
        grads : list of (dW, db)
            Gradient of the loss w.r.t. each (W, b).
        """
        eps = self._FINITE_DIFF_EPS
        base_loss = self._compute_loss(params, x, y, forward_fn)
        grads: List[Tuple[np.ndarray, np.ndarray]] = []

        for layer_idx, (W, b) in enumerate(params):
            dW = np.zeros_like(W)
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    params_plus = [
                        (Wl.copy(), bl.copy()) for Wl, bl in params
                    ]
                    params_plus[layer_idx][0][i, j] += eps
                    loss_plus = self._compute_loss(
                        params_plus, x, y, forward_fn
                    )
                    dW[i, j] = (loss_plus - base_loss) / eps

            db = np.zeros_like(b)
            for i in range(b.shape[0]):
                params_plus = [
                    (Wl.copy(), bl.copy()) for Wl, bl in params
                ]
                params_plus[layer_idx][1][i] += eps
                loss_plus = self._compute_loss(
                    params_plus, x, y, forward_fn
                )
                db[i] = (loss_plus - base_loss) / eps

            grads.append((dW, db))

        return grads

    @staticmethod
    def _gradient_norm(
        grads: List[Tuple[np.ndarray, np.ndarray]],
    ) -> float:
        """Compute the L2 norm of the full gradient vector.

        Parameters
        ----------
        grads : list of (dW, db)
            Parameter gradients.

        Returns
        -------
        norm : float
            L2 norm.
        """
        total = 0.0
        for dW, db in grads:
            total += float(np.sum(dW ** 2) + np.sum(db ** 2))
        return math.sqrt(total)

    @staticmethod
    def _param_norm(
        params: List[Tuple[np.ndarray, np.ndarray]],
    ) -> float:
        """Compute the L2 norm of all parameters.

        Parameters
        ----------
        params : list of (W, b)
            Network parameters.

        Returns
        -------
        norm : float
            L2 norm.
        """
        total = 0.0
        for W, b in params:
            total += float(np.sum(W ** 2) + np.sum(b ** 2))
        return math.sqrt(total)

    @staticmethod
    def _flatten_params(
        params: List[Tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Flatten parameter list into a single 1-D vector.

        Parameters
        ----------
        params : list of (W, b)

        Returns
        -------
        flat : 1-D ndarray
        """
        parts: List[np.ndarray] = []
        for W, b in params:
            parts.append(W.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts)

    # ------------------------------------------------------------------
    # NTK and kernel alignment
    # ------------------------------------------------------------------

    def _compute_ntk(
        self,
        params: List[Tuple[np.ndarray, np.ndarray]],
        x: np.ndarray,
        activation: str,
    ) -> np.ndarray:
        """Compute the empirical NTK Gram matrix via finite differences.

        Parameters
        ----------
        params : list of (W, b)
            Network parameters.
        x : ndarray of shape (N, D)
            Input data.
        activation : str
            Activation function name.

        Returns
        -------
        K : ndarray of shape (N, N)
            NTK Gram matrix.
        """
        x = np.atleast_2d(x)
        N = x.shape[0]
        flat_params = self._flatten_params(params)
        P = flat_params.shape[0]

        def flat_forward(p: np.ndarray, xi: np.ndarray) -> np.ndarray:
            unflat = self._unflatten_params(p, params)
            return self._forward(unflat, xi, activation).ravel()

        # Build Jacobian matrix J[i] of shape (C, P) for each sample
        eps = self._FINITE_DIFF_EPS
        C = self.config.output_dim
        jacobians = np.zeros((N, C, P), dtype=np.float64)

        for n in range(N):
            f0 = flat_forward(flat_params, x[n])
            for p_idx in range(P):
                p_plus = flat_params.copy()
                p_plus[p_idx] += eps
                f_plus = flat_forward(p_plus, x[n])
                jacobians[n, :, p_idx] = (f_plus - f0) / eps

        # K[i, j] = sum_c sum_p J[i, c, p] * J[j, c, p]
        J_flat = jacobians.reshape(N, C * P)
        K = J_flat @ J_flat.T
        return K

    def _unflatten_params(
        self,
        flat: np.ndarray,
        template: List[Tuple[np.ndarray, np.ndarray]],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Unflatten a 1-D vector back into parameter list.

        Parameters
        ----------
        flat : 1-D ndarray
            Flattened parameters.
        template : list of (W, b)
            Template for shapes.

        Returns
        -------
        params : list of (W, b)
        """
        params: List[Tuple[np.ndarray, np.ndarray]] = []
        idx = 0
        for W, b in template:
            w_size = W.size
            b_size = b.size
            W_new = flat[idx : idx + w_size].reshape(W.shape)
            idx += w_size
            b_new = flat[idx : idx + b_size].reshape(b.shape)
            idx += b_size
            params.append((W_new, b_new))
        return params

    def _measure_kernel_alignment(
        self,
        params: List[Tuple[np.ndarray, np.ndarray]],
        x: np.ndarray,
        initial_kernel: np.ndarray,
    ) -> float:
        """Compute alignment between current and initial NTK.

        Alignment is defined as the normalized Frobenius inner product:
            A(K, K0) = <K, K0>_F / (||K||_F * ||K0||_F)

        Parameters
        ----------
        params : list of (W, b)
            Current network parameters.
        x : ndarray of shape (N, D)
            Input data used for kernel computation.
        initial_kernel : ndarray of shape (N, N)
            NTK at initialization.

        Returns
        -------
        alignment : float
            Kernel alignment in [0, 1].
        """
        current_kernel = self._compute_ntk(params, x, self.config.activation)
        numer = float(np.sum(current_kernel * initial_kernel))
        denom = float(
            sp_linalg.norm(current_kernel, "fro")
            * sp_linalg.norm(initial_kernel, "fro")
        )
        if denom < 1e-15:
            return 0.0
        return numer / denom

    # ------------------------------------------------------------------
    # Single training run
    # ------------------------------------------------------------------

    def train_single(self, seed: int) -> TrainingRun:
        """Train one network and record measurements.

        Parameters
        ----------
        seed : int
            Random seed for initialization and data generation.

        Returns
        -------
        run : TrainingRun
            Full training trajectory.
        """
        cfg = self.config
        rng = np.random.RandomState(seed)

        # Generate synthetic data
        x_train = rng.randn(cfg.dataset_size, cfg.input_dim).astype(np.float64)
        y_train = rng.randn(cfg.dataset_size, cfg.output_dim).astype(np.float64)

        # Build network
        forward_fn, params = self._build_network(
            cfg.width, cfg.depth, cfg.activation, cfg.init_scale, seed
        )

        # Use a small subset for kernel computation to keep cost manageable
        kernel_subset_size = min(cfg.dataset_size, 32)
        x_kernel = x_train[:kernel_subset_size]
        initial_kernel = self._compute_ntk(params, x_kernel, cfg.activation)

        measurements: List[TrainingMeasurement] = []
        num_batches = max(1, cfg.dataset_size // cfg.batch_size)

        for epoch in range(cfg.num_epochs):
            # Shuffle data
            perm = rng.permutation(cfg.dataset_size)
            x_shuffled = x_train[perm]
            y_shuffled = y_train[perm]

            # SGD over mini-batches
            for b in range(num_batches):
                start = b * cfg.batch_size
                end = min(start + cfg.batch_size, cfg.dataset_size)
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                grads = self._compute_gradient(
                    params, x_batch, y_batch, forward_fn
                )
                # SGD update
                params = [
                    (
                        W - cfg.learning_rate * dW,
                        b_vec - cfg.learning_rate * db,
                    )
                    for (W, b_vec), (dW, db) in zip(params, grads)
                ]

            # Measure at specified intervals (and first/last epoch)
            if epoch % cfg.measure_interval == 0 or epoch == cfg.num_epochs - 1:
                loss = self._compute_loss(params, x_train, y_train, forward_fn)
                full_grads = self._compute_gradient(
                    params, x_train, y_train, forward_fn
                )
                grad_norm = self._gradient_norm(full_grads)
                p_norm = self._param_norm(params)

                alignment = self._measure_kernel_alignment(
                    params, x_kernel, initial_kernel
                )

                kernel_mat = None
                if epoch % (cfg.measure_interval * 5) == 0:
                    kernel_mat = self._compute_ntk(
                        params, x_kernel, cfg.activation
                    )

                measurements.append(
                    TrainingMeasurement(
                        epoch=epoch,
                        loss=loss,
                        gradient_norm=grad_norm,
                        kernel_matrix=kernel_mat,
                        kernel_alignment=alignment,
                        parameter_norm=p_norm,
                    )
                )

        final_loss = self._compute_loss(params, x_train, y_train, forward_fn)
        regime = self.classify_regime_from_measurements(measurements)

        return TrainingRun(
            config=cfg,
            measurements=measurements,
            final_loss=final_loss,
            regime_label=regime,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Ensemble training
    # ------------------------------------------------------------------

    def train_ensemble(self) -> GroundTruthResult:
        """Train an ensemble of networks and aggregate results.

        Returns
        -------
        result : GroundTruthResult
            Aggregated ground-truth result.
        """
        runs: List[TrainingRun] = []
        for s in range(self.config.num_seeds):
            run = self.train_single(seed=s)
            runs.append(run)

        # Aggregate loss trajectories
        max_len = max(len(r.measurements) for r in runs)
        loss_matrix = np.full(
            (len(runs), max_len), np.nan, dtype=np.float64
        )
        for i, run in enumerate(runs):
            for j, m in enumerate(run.measurements):
                loss_matrix[i, j] = m.loss

        mean_traj = np.nanmean(loss_matrix, axis=0)
        std_traj = np.nanstd(loss_matrix, axis=0)

        # Consensus regime classification via majority vote
        labels = [r.regime_label for r in runs]
        lazy_count = sum(1 for l in labels if l == "lazy")
        regime = "lazy" if lazy_count > len(labels) / 2 else "rich"

        ci = self._compute_confidence_intervals(runs)

        return GroundTruthResult(
            runs=runs,
            mean_trajectory=mean_traj,
            std_trajectory=std_traj,
            regime_classification=regime,
            confidence_intervals=ci,
        )

    # ------------------------------------------------------------------
    # Regime classification
    # ------------------------------------------------------------------

    def classify_regime(self, run: TrainingRun) -> str:
        """Classify a training run as 'lazy' or 'rich'.

        Parameters
        ----------
        run : TrainingRun
            Completed training run.

        Returns
        -------
        label : str
            'lazy' if kernel alignment stays high, 'rich' otherwise.
        """
        return self.classify_regime_from_measurements(run.measurements)

    def classify_regime_from_measurements(
        self, measurements: List[TrainingMeasurement]
    ) -> str:
        """Classify regime from a list of measurements.

        Parameters
        ----------
        measurements : list of TrainingMeasurement
            Measurements with kernel alignment values.

        Returns
        -------
        label : str
            'lazy' or 'rich'.
        """
        if not measurements:
            return "lazy"
        alignments = [m.kernel_alignment for m in measurements]
        final_alignment = alignments[-1]
        if final_alignment >= self._LAZY_ALIGNMENT_THRESHOLD:
            return "lazy"
        return "rich"

    # ------------------------------------------------------------------
    # Confidence intervals
    # ------------------------------------------------------------------

    def _compute_confidence_intervals(
        self, runs: List[TrainingRun]
    ) -> Dict[str, Any]:
        """Compute confidence intervals for key quantities.

        Parameters
        ----------
        runs : list of TrainingRun
            Completed training runs.

        Returns
        -------
        ci : dict
            Keys 'loss', 'gradient_norm', 'alignment', each mapping to
            a dict with 'mean', 'lower', 'upper'.
        """
        if not runs:
            return {}

        final_losses = np.array([r.final_loss for r in runs])
        final_grad_norms = np.array(
            [r.measurements[-1].gradient_norm for r in runs if r.measurements]
        )
        final_alignments = np.array(
            [
                r.measurements[-1].kernel_alignment
                for r in runs
                if r.measurements
            ]
        )

        def _ci(values: np.ndarray) -> Dict[str, float]:
            if len(values) == 0:
                return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            n = len(values)
            # 95% CI using t-approximation
            t_val = 1.96 if n > 30 else 2.0
            margin = t_val * std / math.sqrt(max(n, 1))
            return {
                "mean": mean,
                "lower": mean - margin,
                "upper": mean + margin,
            }

        return {
            "loss": _ci(final_losses),
            "gradient_norm": _ci(final_grad_norms),
            "alignment": _ci(final_alignments),
        }

    # ------------------------------------------------------------------
    # Order parameter extraction
    # ------------------------------------------------------------------

    def extract_order_parameter(self, run: TrainingRun) -> np.ndarray:
        """Extract kernel alignment drift rate from a single run.

        Computes the rate of change of kernel alignment over the
        training trajectory, which serves as the order parameter
        distinguishing lazy from rich regimes.

        Parameters
        ----------
        run : TrainingRun
            Completed training run.

        Returns
        -------
        drift_rates : ndarray
            Array of alignment drift rates between successive
            measurement points.
        """
        if len(run.measurements) < 2:
            return np.array([0.0])

        alignments = np.array(
            [m.kernel_alignment for m in run.measurements], dtype=np.float64
        )
        epochs = np.array(
            [m.epoch for m in run.measurements], dtype=np.float64
        )

        dt = np.diff(epochs)
        da = np.diff(alignments)
        # Avoid division by zero
        dt = np.where(dt == 0, 1.0, dt)
        drift_rates = da / dt
        return drift_rates

    # ------------------------------------------------------------------
    # Data generation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_data(
        n: int,
        d_in: int,
        d_out: int,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data.

        Parameters
        ----------
        n : int
            Number of samples.
        d_in : int
            Input dimensionality.
        d_out : int
            Output dimensionality.
        seed : int
            Random seed.

        Returns
        -------
        x : ndarray of shape (n, d_in)
        y : ndarray of shape (n, d_out)
        """
        rng = np.random.RandomState(seed)
        x = rng.randn(n, d_in).astype(np.float64)
        y = rng.randn(n, d_out).astype(np.float64)
        return x, y
