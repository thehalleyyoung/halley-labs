"""Order parameter computation for phase identification.

Provides:
  - OrderParameterType: enum of supported order parameters
  - TrainingTrajectory: time-series data from a training run
  - OrderParameterResult: computed order parameter values and summary
  - OrderParameterComputer: compute, differentiate, and interpolate order
    parameters from training trajectories
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


# ======================================================================
# Enums and data structures
# ======================================================================


class OrderParameterType(enum.Enum):
    """Supported order parameter types."""

    KERNEL_ALIGNMENT_DRIFT = "kernel_alignment_drift"
    NTK_EIGENVALUE_DECAY = "ntk_eigenvalue_decay"
    GRADIENT_NORM_EVOLUTION = "gradient_norm_evolution"
    LOSS_CURVATURE = "loss_curvature"


@dataclass
class TrainingTrajectory:
    """Time-series data from a training run.

    Parameters
    ----------
    timesteps : np.ndarray
        1-D array of time indices (e.g., training steps).
    kernel_matrices : list of np.ndarray
        Kernel (NTK) snapshots at each timestep.  Each entry is a
        2-D array of shape ``(n, n)``.
    loss_values : np.ndarray
        1-D array of loss values at each timestep.
    gradient_norms : np.ndarray
        1-D array of gradient norms at each timestep.
    parameters_history : list of np.ndarray or None
        Optional list of parameter snapshots at each timestep.
    """

    timesteps: np.ndarray = field(default_factory=lambda: np.array([]))
    kernel_matrices: List[np.ndarray] = field(default_factory=list)
    loss_values: np.ndarray = field(default_factory=lambda: np.array([]))
    gradient_norms: np.ndarray = field(default_factory=lambda: np.array([]))
    parameters_history: Optional[List[np.ndarray]] = None


@dataclass
class OrderParameterResult:
    """Computed order parameter values and summary.

    Parameters
    ----------
    values : np.ndarray
        1-D array of order parameter values over time.
    rate : float
        Scalar summary of the order parameter (e.g., drift rate, decay
        exponent).
    gradient_wrt_params : np.ndarray or None
        Gradient of the rate w.r.t. hyperparameters (for boundary
        detection).
    type : OrderParameterType
        Which order parameter was computed.
    """

    values: np.ndarray = field(default_factory=lambda: np.array([]))
    rate: float = 0.0
    gradient_wrt_params: Optional[np.ndarray] = None
    type: OrderParameterType = OrderParameterType.KERNEL_ALIGNMENT_DRIFT


# ======================================================================
# Order parameter computer
# ======================================================================


class OrderParameterComputer:
    """Compute order parameters from training trajectories.

    Parameters
    ----------
    param_type : OrderParameterType
        Which order parameter to compute.
    """

    def __init__(
        self, param_type: OrderParameterType = OrderParameterType.KERNEL_ALIGNMENT_DRIFT,
    ) -> None:
        self.param_type = param_type

        self._dispatch = {
            OrderParameterType.KERNEL_ALIGNMENT_DRIFT: self._kernel_alignment_drift,
            OrderParameterType.NTK_EIGENVALUE_DECAY: self._ntk_eigenvalue_decay,
            OrderParameterType.GRADIENT_NORM_EVOLUTION: self._gradient_norm_evolution,
            OrderParameterType.LOSS_CURVATURE: self._loss_curvature,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, trajectory: TrainingTrajectory) -> OrderParameterResult:
        """Compute the order parameter for the given trajectory.

        Parameters
        ----------
        trajectory : TrainingTrajectory

        Returns
        -------
        OrderParameterResult
        """
        fn = self._dispatch[self.param_type]
        return fn(trajectory)

    def compute_gradient(
        self,
        trajectory: TrainingTrajectory,
        param_index: int,
        eps: float = 1e-5,
    ) -> float:
        """Estimate gradient of the order-parameter rate w.r.t. a hyperparameter.

        Uses central finite differences on the ``parameters_history``
        entries (perturbing entry ``param_index`` of the first snapshot).

        Parameters
        ----------
        trajectory : TrainingTrajectory
        param_index : int
            Index into the parameter vector to perturb.
        eps : float
            Finite-difference step size.

        Returns
        -------
        float
            Approximate gradient value.
        """
        if trajectory.parameters_history is None or len(trajectory.parameters_history) == 0:
            return 0.0

        base_result = self.compute(trajectory)

        # Perturb the first parameter snapshot
        traj_plus = _perturb_trajectory(trajectory, param_index, +eps)
        traj_minus = _perturb_trajectory(trajectory, param_index, -eps)

        rate_plus = self.compute(traj_plus).rate
        rate_minus = self.compute(traj_minus).rate

        return (rate_plus - rate_minus) / (2 * eps)

    def continuous_variant(
        self, trajectory: TrainingTrajectory,
    ) -> OrderParameterResult:
        """Interpolated continuous-time order parameter.

        Computes the discrete order parameter and returns a cubic-spline
        interpolation evaluated at 10× the original resolution.

        Parameters
        ----------
        trajectory : TrainingTrajectory

        Returns
        -------
        OrderParameterResult
        """
        result = self.compute(trajectory)
        t = trajectory.timesteps
        if len(t) < 4:
            return result

        f = interp1d(t, result.values, kind="cubic", fill_value="extrapolate")
        t_fine = np.linspace(t[0], t[-1], len(t) * 10)
        return OrderParameterResult(
            values=f(t_fine),
            rate=result.rate,
            gradient_wrt_params=result.gradient_wrt_params,
            type=result.type,
        )

    def discrete_variant(
        self, trajectory: TrainingTrajectory,
    ) -> OrderParameterResult:
        """Raw discrete-time order parameter values.

        Equivalent to :meth:`compute` but explicitly named for clarity.

        Parameters
        ----------
        trajectory : TrainingTrajectory

        Returns
        -------
        OrderParameterResult
        """
        return self.compute(trajectory)

    # ------------------------------------------------------------------
    # Internal: order parameter implementations
    # ------------------------------------------------------------------

    def _kernel_alignment_drift(
        self, trajectory: TrainingTrajectory,
    ) -> OrderParameterResult:
        """Compute kernel alignment drift over time.

        Measures how much the NTK drifts from its initial value using
        the normalised Frobenius inner product (alignment).

        Parameters
        ----------
        trajectory : TrainingTrajectory

        Returns
        -------
        OrderParameterResult
        """
        kernels = trajectory.kernel_matrices
        if len(kernels) < 2:
            return OrderParameterResult(
                values=np.array([0.0]),
                rate=0.0,
                type=OrderParameterType.KERNEL_ALIGNMENT_DRIFT,
            )

        K0 = kernels[0].astype(np.float64)
        K0_norm = np.linalg.norm(K0, "fro")
        if K0_norm == 0:
            K0_norm = 1.0

        alignments = np.empty(len(kernels), dtype=np.float64)
        for i, K in enumerate(kernels):
            K = K.astype(np.float64)
            K_norm = np.linalg.norm(K, "fro")
            if K_norm == 0:
                alignments[i] = 0.0
            else:
                alignments[i] = np.sum(K0 * K) / (K0_norm * K_norm)

        drift = 1.0 - alignments  # 0 = no drift, 1 = full drift
        rate = self._fit_linear_rate(trajectory.timesteps, drift)

        return OrderParameterResult(
            values=drift,
            rate=rate,
            type=OrderParameterType.KERNEL_ALIGNMENT_DRIFT,
        )

    def _ntk_eigenvalue_decay(
        self, trajectory: TrainingTrajectory,
    ) -> OrderParameterResult:
        """Track top eigenvalue ratio of the NTK over time.

        The ratio ``λ_1(t) / λ_1(0)`` indicates how the spectral
        structure evolves.

        Parameters
        ----------
        trajectory : TrainingTrajectory

        Returns
        -------
        OrderParameterResult
        """
        kernels = trajectory.kernel_matrices
        if len(kernels) < 2:
            return OrderParameterResult(
                values=np.array([1.0]),
                rate=0.0,
                type=OrderParameterType.NTK_EIGENVALUE_DECAY,
            )

        eig0 = _top_eigenvalue(kernels[0])
        if eig0 == 0:
            eig0 = 1.0

        ratios = np.empty(len(kernels), dtype=np.float64)
        for i, K in enumerate(kernels):
            ratios[i] = _top_eigenvalue(K) / eig0

        # Fit exponential decay: ratio ~ exp(-rate * t)
        log_ratios = np.log(np.clip(ratios, 1e-30, None))
        rate = self._fit_linear_rate(trajectory.timesteps, -log_ratios)

        return OrderParameterResult(
            values=ratios,
            rate=rate,
            type=OrderParameterType.NTK_EIGENVALUE_DECAY,
        )

    def _gradient_norm_evolution(
        self, trajectory: TrainingTrajectory,
    ) -> OrderParameterResult:
        """Fit power law to gradient norm trajectory.

        Models ``||g(t)|| ~ t^{-alpha}``; ``alpha`` is the rate.

        Parameters
        ----------
        trajectory : TrainingTrajectory

        Returns
        -------
        OrderParameterResult
        """
        g = trajectory.gradient_norms.astype(np.float64)
        t = trajectory.timesteps.astype(np.float64)
        if len(g) < 2:
            return OrderParameterResult(
                values=g,
                rate=0.0,
                type=OrderParameterType.GRADIENT_NORM_EVOLUTION,
            )

        # Avoid log(0)
        mask = (g > 0) & (t > 0)
        if np.sum(mask) < 2:
            return OrderParameterResult(
                values=g,
                rate=0.0,
                type=OrderParameterType.GRADIENT_NORM_EVOLUTION,
            )

        log_t = np.log(t[mask])
        log_g = np.log(g[mask])
        coeffs = np.polyfit(log_t, log_g, 1)
        alpha = -coeffs[0]

        return OrderParameterResult(
            values=g,
            rate=float(alpha),
            type=OrderParameterType.GRADIENT_NORM_EVOLUTION,
        )

    def _loss_curvature(
        self, trajectory: TrainingTrajectory,
    ) -> OrderParameterResult:
        """Estimate loss Hessian spectral norm over time.

        Uses second-order finite differences on the loss trajectory as a
        proxy for curvature.

        Parameters
        ----------
        trajectory : TrainingTrajectory

        Returns
        -------
        OrderParameterResult
        """
        loss = trajectory.loss_values.astype(np.float64)
        t = trajectory.timesteps.astype(np.float64)
        if len(loss) < 3:
            return OrderParameterResult(
                values=np.zeros(max(len(loss), 1)),
                rate=0.0,
                type=OrderParameterType.LOSS_CURVATURE,
            )

        dt = np.diff(t)
        # Central second derivative at interior points
        curvature = np.zeros(len(loss), dtype=np.float64)
        for i in range(1, len(loss) - 1):
            h1, h2 = dt[i - 1], dt[i]
            curvature[i] = (
                2.0
                * (loss[i + 1] / h2 - loss[i] * (1.0 / h1 + 1.0 / h2) + loss[i - 1] / h1)
                / (h1 + h2)
            )
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]

        abs_curv = np.abs(curvature)
        rate = self._fit_linear_rate(t, abs_curv)

        return OrderParameterResult(
            values=abs_curv,
            rate=rate,
            type=OrderParameterType.LOSS_CURVATURE,
        )

    # ------------------------------------------------------------------
    # Fitting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_linear_rate(t: np.ndarray, y: np.ndarray) -> float:
        """Fit a linear trend ``y = rate * t + b`` and return *rate*.

        Parameters
        ----------
        t : np.ndarray
        y : np.ndarray

        Returns
        -------
        float
        """
        if len(t) < 2 or len(y) < 2:
            return 0.0
        t = t.astype(np.float64)
        y = y.astype(np.float64)
        mask = np.isfinite(t) & np.isfinite(y)
        if np.sum(mask) < 2:
            return 0.0
        coeffs = np.polyfit(t[mask], y[mask], 1)
        return float(coeffs[0])


# ======================================================================
# Module-level helpers
# ======================================================================


def _top_eigenvalue(K: np.ndarray) -> float:
    """Return the largest eigenvalue of a symmetric matrix.

    Parameters
    ----------
    K : np.ndarray
        Symmetric 2-D array.

    Returns
    -------
    float
    """
    eigvals = np.linalg.eigvalsh(K)
    return float(np.max(np.abs(eigvals)))


def _perturb_trajectory(
    trajectory: TrainingTrajectory,
    param_index: int,
    delta: float,
) -> TrainingTrajectory:
    """Create a copy of *trajectory* with one parameter entry perturbed.

    Parameters
    ----------
    trajectory : TrainingTrajectory
    param_index : int
    delta : float

    Returns
    -------
    TrainingTrajectory
    """
    new_params: Optional[List[np.ndarray]] = None
    if trajectory.parameters_history is not None:
        new_params = [p.copy() for p in trajectory.parameters_history]
        for p in new_params:
            if param_index < len(p):
                p[param_index] += delta

    return TrainingTrajectory(
        timesteps=trajectory.timesteps.copy(),
        kernel_matrices=[k.copy() for k in trajectory.kernel_matrices],
        loss_values=trajectory.loss_values.copy(),
        gradient_norms=trajectory.gradient_norms.copy(),
        parameters_history=new_params,
    )
