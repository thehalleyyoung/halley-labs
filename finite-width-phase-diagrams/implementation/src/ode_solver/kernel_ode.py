"""
Kernel Evolution ODE Solver
============================

Numerical integration of kernel matrix ODEs arising in neural network
training dynamics. Supports both fixed-step and adaptive methods,
with energy conservation monitoring for symplectic-like systems.

The central equation is

    dK/dt = F(t, K),

where K ∈ ℝ^{n×n} is a (neural tangent) kernel matrix evolving under
training dynamics.  Depending on the regime the right-hand side F takes
different forms:

* **Lazy / linear regime**:  dK/dt = L K + K Lᵀ   (Lyapunov-type)
* **Rich / nonlinear regime**: dK/dt = F_NTK(t, K) + δF(t, K)
* **Gradient flow**:          dK/dt = -η K (K y − X)(K y − X)ᵀ K

The solver provides Euler, classical RK4, adaptive Dormand–Prince RK4(5),
and implicit midpoint (for stiff problems) integration schemes.
"""

from __future__ import annotations

import enum
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicHermiteSpline, interp1d


# ======================================================================
#  Constants – Dormand–Prince Butcher tableau
# ======================================================================

# Nodes  c_i
_DP_C = np.array([0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0])

# Runge–Kutta matrix  a_{ij}
_DP_A = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0],
        [
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
            0.0,
            0.0,
        ],
        [
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
            0.0,
        ],
    ]
)

# 5th-order weights  b_i  (for the solution)
_DP_B = np.array(
    [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0]
)

# 4th-order weights  b*_i  (for the error estimate)
_DP_B_STAR = np.array(
    [
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339200.0,
        187.0 / 2100.0,
        1.0 / 40.0,
    ]
)

# Error coefficients  e_i = b_i − b*_i
_DP_E = _DP_B - _DP_B_STAR


# ======================================================================
#  Integration Scheme Enum
# ======================================================================


class IntegrationScheme(enum.Enum):
    """Available numerical integration schemes.

    Attributes
    ----------
    EULER : int
        Explicit (forward) Euler method.  First-order, O(dt).
    RK4 : int
        Classical four-stage Runge–Kutta.  Fourth-order, O(dt⁴).
    RK45_ADAPTIVE : int
        Dormand–Prince embedded RK4(5) with adaptive step-size control.
    IMPLICIT_MIDPOINT : int
        Implicit midpoint rule (symplectic, A-stable).  Second-order.
    """

    EULER = 1
    RK4 = 2
    RK45_ADAPTIVE = 3
    IMPLICIT_MIDPOINT = 4


# ======================================================================
#  ODE Trajectory Dataclass
# ======================================================================


@dataclass
class ODETrajectory:
    """Container for the result of an ODE integration.

    Parameters
    ----------
    times : NDArray
        1-D array of time points at which the solution is recorded.
    states : list of NDArray
        Solution vectors at each recorded time.
    energies : NDArray or None
        Optional energy-like functional evaluated along the trajectory.
    step_sizes : NDArray
        Step sizes used during integration.
    num_steps : int
        Total number of integration steps taken.
    final_time : float
        Terminal time of the integration.
    energy_drift : float or None
        Relative energy drift  |E(T) − E(0)| / |E(0)|  if energies
        are available.
    interpolator : callable or None
        If available, a function  t ↦ y(t)  providing dense output.
    """

    times: NDArray
    states: List[NDArray]
    energies: Optional[NDArray] = None
    step_sizes: NDArray = field(default_factory=lambda: np.array([]))
    num_steps: int = 0
    final_time: float = 0.0
    energy_drift: Optional[float] = None
    interpolator: Optional[Callable[[float], NDArray]] = None


# ======================================================================
#  Step-Size Controller (PID)
# ======================================================================


class StepSizeController:
    """PID-based adaptive step-size controller.

    Implements the H211b digital filter controller from
    Söderlind (2003) for robust step-size selection.

    Parameters
    ----------
    atol : float
        Absolute tolerance.
    rtol : float
        Relative tolerance.
    safety : float
        Safety factor  (0 < safety < 1).
    min_factor : float
        Minimum allowed step-size reduction factor.
    max_factor : float
        Maximum allowed step-size growth factor.
    min_step : float
        Hard floor on the step size.
    max_step : float
        Hard ceiling on the step size.
    """

    def __init__(
        self,
        atol: float = 1e-8,
        rtol: float = 1e-6,
        safety: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        min_step: float = 1e-12,
        max_step: float = 1.0,
    ) -> None:
        self.atol = atol
        self.rtol = rtol
        self.safety = safety
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.min_step = min_step
        self.max_step = max_step

        # PID controller state
        self._prev_error_norm: float = 1.0
        self._prev_prev_error_norm: float = 1.0
        self._prev_factor: float = 1.0

        # PID gains  (H211b controller)
        self._beta1: float = 1.0 / 6.0
        self._beta2: float = 1.0 / 6.0

    # ------------------------------------------------------------------

    def propose_step(
        self,
        error_estimate: NDArray,
        current_step: float,
        order: int,
        y: Optional[NDArray] = None,
        y_new: Optional[NDArray] = None,
    ) -> float:
        """Propose a new step size based on the local error estimate.

        Uses a PID controller for smooth step-size transitions:

            h_new = h · S · (ε_{n-1}/ε_n)^β₁ · (1/ε_n)^β₂

        where ε_n is the normalised error norm and S is the safety factor.

        Parameters
        ----------
        error_estimate : NDArray
            Element-wise local truncation error estimate.
        current_step : float
            Current step size  h.
        order : int
            Order of the lower-order embedded method.
        y : NDArray, optional
            Current state (used for relative tolerance scaling).
        y_new : NDArray, optional
            Proposed new state (used for relative tolerance scaling).

        Returns
        -------
        float
            Proposed next step size.
        """
        if y is not None and y_new is not None:
            error_norm = self._error_norm(error_estimate, y, y_new)
        else:
            error_norm = self._error_norm_simple(error_estimate)

        if error_norm == 0.0:
            return min(current_step * self.max_factor, self.max_step)

        # PID step-size controller
        exponent = 1.0 / (order + 1)
        factor_i = (self._prev_error_norm / error_norm) ** self._beta1
        factor_p = (1.0 / error_norm) ** self._beta2
        factor = self.safety * factor_i * factor_p

        factor = max(self.min_factor, min(factor, self.max_factor))

        # Update controller state
        self._prev_prev_error_norm = self._prev_error_norm
        self._prev_error_norm = max(error_norm, 1e-15)
        self._prev_factor = factor

        new_step = current_step * factor
        return float(np.clip(new_step, self.min_step, self.max_step))

    # ------------------------------------------------------------------

    def accept_step(self, error_estimate: NDArray, y: Optional[NDArray] = None,
                    y_new: Optional[NDArray] = None) -> bool:
        """Decide whether to accept the current step.

        The step is accepted when the normalised error norm is ≤ 1.

        Parameters
        ----------
        error_estimate : NDArray
            Element-wise local truncation error estimate.
        y : NDArray, optional
            Current state.
        y_new : NDArray, optional
            New state.

        Returns
        -------
        bool
            True if the step should be accepted.
        """
        if y is not None and y_new is not None:
            return self._error_norm(error_estimate, y, y_new) <= 1.0
        return self._error_norm_simple(error_estimate) <= 1.0

    # ------------------------------------------------------------------

    def _error_norm(
        self,
        error: NDArray,
        y: NDArray,
        y_new: NDArray,
    ) -> float:
        """Mixed absolute / relative error norm.

        Computes the RMS-based error norm

            ε = √( (1/n) Σ_i (e_i / (atol + rtol · max(|y_i|, |ŷ_i|)))² )

        Parameters
        ----------
        error : NDArray
            Error vector.
        y : NDArray
            Current state.
        y_new : NDArray
            Proposed new state.

        Returns
        -------
        float
            Scalar error norm.
        """
        scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y_new))
        return float(np.sqrt(np.mean((error / scale) ** 2)))

    # ------------------------------------------------------------------

    def _error_norm_simple(self, error: NDArray) -> float:
        """Simplified error norm when reference states are unavailable."""
        scale = self.atol + self.rtol * np.abs(error)
        with np.errstate(divide="ignore", invalid="ignore"):
            normed = np.where(scale > 0, error / scale, 0.0)
        return float(np.sqrt(np.mean(normed ** 2)))

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the controller state for a new integration."""
        self._prev_error_norm = 1.0
        self._prev_prev_error_norm = 1.0
        self._prev_factor = 1.0


# ======================================================================
#  Kernel ODE Solver
# ======================================================================


class KernelODESolver:
    """ODE solver specialised for kernel-matrix evolution equations.

    Provides multiple integration schemes and trajectory recording with
    optional energy-conservation monitoring.

    Parameters
    ----------
    scheme : IntegrationScheme
        Integration method to use (default: adaptive Dormand–Prince).
    atol : float
        Absolute tolerance for adaptive methods.
    rtol : float
        Relative tolerance for adaptive methods.
    max_steps : int
        Maximum number of integration steps.
    store_trajectory : bool
        Whether to record the full trajectory.
    """

    def __init__(
        self,
        scheme: IntegrationScheme = IntegrationScheme.RK45_ADAPTIVE,
        atol: float = 1e-8,
        rtol: float = 1e-6,
        max_steps: int = 10_000,
        store_trajectory: bool = True,
    ) -> None:
        self.scheme = scheme
        self.atol = atol
        self.rtol = rtol
        self.max_steps = max_steps
        self.store_trajectory = store_trajectory

        self._controller = StepSizeController(atol=atol, rtol=rtol)

    # ==================================================================
    #  Public API
    # ==================================================================

    def solve(
        self,
        rhs_fn: Callable[[float, NDArray], NDArray],
        y0: NDArray,
        t_span: Tuple[float, float],
        t_eval: Optional[NDArray] = None,
    ) -> ODETrajectory:
        """Solve the initial-value problem  dy/dt = rhs_fn(t, y).

        Parameters
        ----------
        rhs_fn : callable (t, y) → dy/dt
            Right-hand side of the ODE.
        y0 : NDArray
            Initial condition vector.
        t_span : (t0, tf)
            Integration interval.
        t_eval : NDArray, optional
            Times at which to record the solution.  If *None*, the solver
            records at every internal step (adaptive) or at evenly spaced
            points (fixed).

        Returns
        -------
        ODETrajectory
            Solution container.
        """
        y0 = np.asarray(y0, dtype=np.float64)
        t0, tf = float(t_span[0]), float(t_span[1])

        if t_eval is not None:
            t_eval = np.asarray(t_eval, dtype=np.float64)

        if self.scheme == IntegrationScheme.RK45_ADAPTIVE:
            return self._adaptive_integrate(rhs_fn, y0, (t0, tf), t_eval)
        else:
            # Choose a reasonable default fixed step size
            dt = (tf - t0) / max(self.max_steps // 10, 100)
            return self._fixed_step_integrate(rhs_fn, y0, (t0, tf), dt, self.scheme, t_eval)

    # ------------------------------------------------------------------

    def solve_kernel_evolution(
        self,
        K0: NDArray,
        dynamics_fn: Callable[[float, NDArray], NDArray],
        t_span: Tuple[float, float],
        t_eval: Optional[NDArray] = None,
    ) -> ODETrajectory:
        """Solve  dK/dt = dynamics_fn(t, K)  for a kernel matrix K.

        The kernel matrix is flattened to a 1-D vector for integration
        and reshaped back in the recorded trajectory.

        Parameters
        ----------
        K0 : NDArray, shape (n, n)
            Initial kernel matrix.
        dynamics_fn : callable (t, K) → dK/dt
            Right-hand side expressed in matrix form.
        t_span : (t0, tf)
            Integration interval.
        t_eval : NDArray, optional
            Evaluation times.

        Returns
        -------
        ODETrajectory
            Trajectory with states reshaped to (n, n).
        """
        K0 = np.asarray(K0, dtype=np.float64)
        n = K0.shape[0]

        # Wrap to flatten / unflatten
        def _flat_rhs(t: float, y_flat: NDArray) -> NDArray:
            K = y_flat.reshape(n, n)
            dKdt = dynamics_fn(t, K)
            return np.asarray(dKdt, dtype=np.float64).ravel()

        trajectory = self.solve(_flat_rhs, K0.ravel(), t_span, t_eval)

        # Reshape recorded states back to matrices
        trajectory.states = [s.reshape(n, n) for s in trajectory.states]
        return trajectory

    # ------------------------------------------------------------------

    def solve_linearized(
        self,
        K0: NDArray,
        L: NDArray,
        t_span: Tuple[float, float],
        t_eval: Optional[NDArray] = None,
    ) -> ODETrajectory:
        """Solve the continuous Lyapunov-type equation

            dK/dt = L K + K Lᵀ.

        This governs linearised (lazy-regime) NTK evolution.

        Parameters
        ----------
        K0 : NDArray, shape (n, n)
            Initial kernel matrix.
        L : NDArray, shape (n, n)
            Linear operator driving the evolution.
        t_span : (t0, tf)
            Integration interval.
        t_eval : NDArray, optional
            Evaluation times.

        Returns
        -------
        ODETrajectory
        """
        L = np.asarray(L, dtype=np.float64)
        LT = L.T

        def _lyapunov_rhs(_t: float, K: NDArray) -> NDArray:
            return L @ K + K @ LT

        return self.solve_kernel_evolution(K0, _lyapunov_rhs, t_span, t_eval)

    # ==================================================================
    #  Single-Step Methods
    # ==================================================================

    def _euler_step(
        self,
        rhs_fn: Callable[[float, NDArray], NDArray],
        t: float,
        y: NDArray,
        dt: float,
    ) -> NDArray:
        """Explicit (forward) Euler step.

            y_{n+1} = y_n + dt · f(t_n, y_n)

        Parameters
        ----------
        rhs_fn : callable
        t : float
        y : NDArray
        dt : float

        Returns
        -------
        NDArray
            Updated state y_{n+1}.
        """
        return y + dt * rhs_fn(t, y)

    # ------------------------------------------------------------------

    def _rk4_step(
        self,
        rhs_fn: Callable[[float, NDArray], NDArray],
        t: float,
        y: NDArray,
        dt: float,
    ) -> NDArray:
        """Classical fourth-order Runge–Kutta step.

            k₁ = f(tₙ,       yₙ)
            k₂ = f(tₙ + h/2, yₙ + h k₁/2)
            k₃ = f(tₙ + h/2, yₙ + h k₂/2)
            k₄ = f(tₙ + h,   yₙ + h k₃)
            y_{n+1} = yₙ + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)

        Returns
        -------
        NDArray
        """
        k1 = rhs_fn(t, y)
        k2 = rhs_fn(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = rhs_fn(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = rhs_fn(t + dt, y + dt * k3)
        return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ------------------------------------------------------------------

    def _rk45_step(
        self,
        rhs_fn: Callable[[float, NDArray], NDArray],
        t: float,
        y: NDArray,
        dt: float,
    ) -> Tuple[NDArray, NDArray]:
        """Dormand–Prince RK4(5) step with embedded error estimate.

        Computes both the 5th-order solution (used for stepping) and
        the 4th-order solution; the difference gives a local error
        estimate of order O(dt⁵).

        Parameters
        ----------
        rhs_fn : callable
        t : float
        y : NDArray
        dt : float

        Returns
        -------
        y_new : NDArray
            Updated state (5th-order accurate).
        error : NDArray
            Estimated local truncation error.
        """
        k = np.empty((7,) + y.shape, dtype=y.dtype)

        k[0] = rhs_fn(t, y)
        for i in range(1, 7):
            t_i = t + _DP_C[i] * dt
            dy = np.zeros_like(y)
            for j in range(i):
                if _DP_A[i, j] != 0.0:
                    dy += _DP_A[i, j] * k[j]
            k[i] = rhs_fn(t_i, y + dt * dy)

        # 5th-order solution
        y_new = y.copy()
        for i in range(7):
            if _DP_B[i] != 0.0:
                y_new += dt * _DP_B[i] * k[i]

        # Error estimate  =  dt · Σ_i e_i k_i
        error = np.zeros_like(y)
        for i in range(7):
            if _DP_E[i] != 0.0:
                error += dt * _DP_E[i] * k[i]

        return y_new, error

    # ------------------------------------------------------------------

    def _implicit_midpoint_step(
        self,
        rhs_fn: Callable[[float, NDArray], NDArray],
        t: float,
        y: NDArray,
        dt: float,
        tol: float = 1e-10,
        max_iter: int = 50,
    ) -> NDArray:
        """Implicit midpoint rule via fixed-point iteration.

            y_{n+1} = y_n + dt · f(t_n + dt/2, (y_n + y_{n+1})/2)

        The implicit equation is solved by fixed-point (Picard)
        iteration, which is appropriate for moderately stiff problems
        where the Lipschitz constant times dt/2 is < 1.

        Parameters
        ----------
        rhs_fn : callable
        t : float
        y : NDArray
        dt : float
        tol : float
            Convergence tolerance for the fixed-point iteration.
        max_iter : int
            Maximum number of iterations.

        Returns
        -------
        NDArray
            Updated state y_{n+1}.

        Warns
        -----
        RuntimeWarning
            If the iteration does not converge within *max_iter* steps.
        """
        t_mid = t + 0.5 * dt

        # Initial guess: explicit Euler
        y_new = y + dt * rhs_fn(t, y)

        for iteration in range(max_iter):
            y_mid = 0.5 * (y + y_new)
            f_mid = rhs_fn(t_mid, y_mid)
            y_next = y + dt * f_mid

            # Convergence check
            diff = np.linalg.norm(y_next - y_new)
            scale = self.atol + self.rtol * np.linalg.norm(y_next)
            if diff < tol * max(scale, 1.0):
                return y_next

            y_new = y_next

        warnings.warn(
            f"Implicit midpoint iteration did not converge after {max_iter} "
            f"iterations (residual = {diff:.2e}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return y_new

    # ==================================================================
    #  Integration Loops
    # ==================================================================

    def _adaptive_integrate(
        self,
        rhs_fn: Callable[[float, NDArray], NDArray],
        y0: NDArray,
        t_span: Tuple[float, float],
        t_eval: Optional[NDArray],
    ) -> ODETrajectory:
        """Adaptive Dormand–Prince integration with PID step control.

        Parameters
        ----------
        rhs_fn : callable
        y0 : NDArray
        t_span : (t0, tf)
        t_eval : NDArray or None

        Returns
        -------
        ODETrajectory
        """
        t0, tf = t_span
        direction = 1.0 if tf >= t0 else -1.0

        # Initial step-size estimate  (Hairer & Wanner heuristic)
        f0 = rhs_fn(t0, y0)
        d0 = np.linalg.norm(y0) / max(y0.size, 1) ** 0.5
        d1 = np.linalg.norm(f0) / max(y0.size, 1) ** 0.5
        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * (d0 / d1)
        h0 = min(h0, abs(tf - t0))
        dt = direction * h0

        self._controller.reset()

        # Storage
        times_list: List[float] = [t0]
        states_list: List[NDArray] = [y0.copy()]
        step_sizes_list: List[float] = []
        energies_list: List[float] = [self._compute_energy(y0, rhs_fn, t0)]

        t = t0
        y = y0.copy()
        n_steps = 0
        n_rejected = 0

        while direction * (tf - t) > 1e-15 * abs(tf):
            if n_steps >= self.max_steps:
                warnings.warn(
                    f"Maximum number of steps ({self.max_steps}) reached at t = {t:.6e}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            # Clamp step to final time
            if direction * (t + dt - tf) > 0:
                dt = tf - t

            # Attempt a step
            y_new, error = self._rk45_step(rhs_fn, t, y, dt)

            # Evaluate acceptance
            accepted = self._controller.accept_step(error, y, y_new)

            if accepted:
                t_new = t + dt
                step_sizes_list.append(abs(dt))

                if self.store_trajectory:
                    times_list.append(t_new)
                    states_list.append(y_new.copy())
                    energies_list.append(self._compute_energy(y_new, rhs_fn, t_new))

                t = t_new
                y = y_new
                n_steps += 1
            else:
                n_rejected += 1

            # Propose new step size
            dt_new = self._controller.propose_step(error, abs(dt), order=4, y=y, y_new=y_new)
            dt = direction * dt_new

        # Ensure final state is recorded
        if not self.store_trajectory:
            times_list = [t0, t]
            states_list = [y0.copy(), y.copy()]

        times_arr = np.array(times_list)
        energies_arr = np.array(energies_list) if energies_list else None

        # Energy drift
        energy_drift: Optional[float] = None
        if energies_arr is not None and len(energies_arr) >= 2:
            if abs(energies_arr[0]) > 1e-30:
                energy_drift = float(
                    abs(energies_arr[-1] - energies_arr[0]) / abs(energies_arr[0])
                )
            else:
                energy_drift = float(abs(energies_arr[-1] - energies_arr[0]))

        # Build interpolator
        interpolator = self._build_interpolator(times_arr, states_list)

        # If specific evaluation times requested, interpolate
        if t_eval is not None and interpolator is not None:
            eval_states = [interpolator(ti) for ti in t_eval]
            eval_energies = np.array(
                [self._compute_energy(s, rhs_fn, ti) for ti, s in zip(t_eval, eval_states)]
            )
            return ODETrajectory(
                times=t_eval,
                states=eval_states,
                energies=eval_energies,
                step_sizes=np.array(step_sizes_list),
                num_steps=n_steps,
                final_time=float(t),
                energy_drift=energy_drift,
                interpolator=interpolator,
            )

        return ODETrajectory(
            times=times_arr,
            states=states_list,
            energies=energies_arr,
            step_sizes=np.array(step_sizes_list),
            num_steps=n_steps,
            final_time=float(t),
            energy_drift=energy_drift,
            interpolator=interpolator,
        )

    # ------------------------------------------------------------------

    def _fixed_step_integrate(
        self,
        rhs_fn: Callable[[float, NDArray], NDArray],
        y0: NDArray,
        t_span: Tuple[float, float],
        dt: float,
        scheme: IntegrationScheme,
        t_eval: Optional[NDArray] = None,
    ) -> ODETrajectory:
        """Fixed step-size integration loop.

        Parameters
        ----------
        rhs_fn : callable
        y0 : NDArray
        t_span : (t0, tf)
        dt : float
            Step size.
        scheme : IntegrationScheme
        t_eval : NDArray or None

        Returns
        -------
        ODETrajectory
        """
        t0, tf = t_span
        direction = 1.0 if tf >= t0 else -1.0
        dt = direction * abs(dt)

        # Select stepper
        if scheme == IntegrationScheme.EULER:
            stepper = self._euler_step
        elif scheme == IntegrationScheme.RK4:
            stepper = self._rk4_step
        elif scheme == IntegrationScheme.IMPLICIT_MIDPOINT:
            stepper = self._implicit_midpoint_step
        else:
            raise ValueError(f"Use _adaptive_integrate for {scheme}")

        times_list: List[float] = [t0]
        states_list: List[NDArray] = [y0.copy()]
        step_sizes_list: List[float] = []
        energies_list: List[float] = [self._compute_energy(y0, rhs_fn, t0)]

        t = t0
        y = y0.copy()
        n_steps = 0

        while direction * (tf - t) > 1e-15 * abs(tf):
            if n_steps >= self.max_steps:
                warnings.warn(
                    f"Maximum number of steps ({self.max_steps}) reached at t = {t:.6e}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

            # Clamp to final time
            h = dt
            if direction * (t + h - tf) > 0:
                h = tf - t

            y = stepper(rhs_fn, t, y, h)
            t = t + h
            n_steps += 1

            step_sizes_list.append(abs(h))
            if self.store_trajectory:
                times_list.append(t)
                states_list.append(y.copy())
                energies_list.append(self._compute_energy(y, rhs_fn, t))

        if not self.store_trajectory:
            times_list = [t0, t]
            states_list = [y0.copy(), y.copy()]

        times_arr = np.array(times_list)
        energies_arr = np.array(energies_list) if energies_list else None

        energy_drift: Optional[float] = None
        if energies_arr is not None and len(energies_arr) >= 2:
            if abs(energies_arr[0]) > 1e-30:
                energy_drift = float(
                    abs(energies_arr[-1] - energies_arr[0]) / abs(energies_arr[0])
                )
            else:
                energy_drift = float(abs(energies_arr[-1] - energies_arr[0]))

        interpolator = self._build_interpolator(times_arr, states_list)

        # Evaluate at requested times
        if t_eval is not None and interpolator is not None:
            eval_states = [interpolator(ti) for ti in t_eval]
            eval_energies = np.array(
                [self._compute_energy(s, rhs_fn, ti) for ti, s in zip(t_eval, eval_states)]
            )
            return ODETrajectory(
                times=t_eval,
                states=eval_states,
                energies=eval_energies,
                step_sizes=np.array(step_sizes_list),
                num_steps=n_steps,
                final_time=float(t),
                energy_drift=energy_drift,
                interpolator=interpolator,
            )

        return ODETrajectory(
            times=times_arr,
            states=states_list,
            energies=energies_arr,
            step_sizes=np.array(step_sizes_list),
            num_steps=n_steps,
            final_time=float(t),
            energy_drift=energy_drift,
            interpolator=interpolator,
        )

    # ==================================================================
    #  Energy & Conservation
    # ==================================================================

    def _compute_energy(
        self,
        y: NDArray,
        rhs_fn: Callable[[float, NDArray], NDArray],
        t: float,
    ) -> float:
        """Compute an energy-like functional for conservation monitoring.

        Default energy is  E = ½ ||y||² .  Override in subclasses for
        problem-specific Hamiltonians.

        Parameters
        ----------
        y : NDArray
        rhs_fn : callable  (unused in default implementation)
        t : float  (unused in default implementation)

        Returns
        -------
        float
        """
        return 0.5 * float(np.dot(y.ravel(), y.ravel()))

    # ------------------------------------------------------------------

    @staticmethod
    def _check_conservation(
        energies: NDArray,
        tol: float = 1e-6,
    ) -> bool:
        """Check whether an energy-like quantity is conserved.

        Parameters
        ----------
        energies : NDArray
            1-D array of energy values along the trajectory.
        tol : float
            Tolerance on relative drift.

        Returns
        -------
        bool
            True if  max|E(t) − E(0)| / |E(0)| < tol.
        """
        energies = np.asarray(energies, dtype=np.float64)
        if len(energies) < 2:
            return True
        E0 = energies[0]
        if abs(E0) < 1e-30:
            return bool(np.all(np.abs(energies - E0) < tol))
        relative_drift = np.max(np.abs(energies - E0)) / abs(E0)
        return bool(relative_drift < tol)

    # ==================================================================
    #  Interpolation / Dense Output
    # ==================================================================

    def _build_interpolator(
        self,
        times: NDArray,
        states: List[NDArray],
    ) -> Optional[Callable[[float], NDArray]]:
        """Build a dense-output interpolator from recorded trajectory.

        Uses cubic Hermite interpolation when enough points are
        available, falling back to linear interpolation otherwise.

        Parameters
        ----------
        times : NDArray
            Recorded time points.
        states : list of NDArray
            States at each time point.

        Returns
        -------
        callable or None
            Interpolation function  t ↦ y(t), or None if fewer than
            2 points are available.
        """
        if len(times) < 2:
            return None

        # Stack states into (n_times, n_dims) array
        state_matrix = np.array([s.ravel() for s in states])
        n_times, n_dims = state_matrix.shape

        if n_times < 4:
            # Linear interpolation
            interp_fn = interp1d(
                times,
                state_matrix,
                axis=0,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

            def _linear_interp(t: float) -> NDArray:
                return np.asarray(interp_fn(t)).ravel()

            return _linear_interp

        # Cubic interpolation
        interp_fn = interp1d(
            times,
            state_matrix,
            axis=0,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )

        def _cubic_interp(t: float) -> NDArray:
            return np.asarray(interp_fn(t)).ravel()

        return _cubic_interp

    # ------------------------------------------------------------------

    @staticmethod
    def dense_output(
        trajectory: ODETrajectory,
        t: Union[float, NDArray],
    ) -> Union[NDArray, List[NDArray]]:
        """Evaluate a trajectory at arbitrary time(s) via interpolation.

        Parameters
        ----------
        trajectory : ODETrajectory
            Solved trajectory (must have an interpolator).
        t : float or NDArray
            Time(s) at which to evaluate.

        Returns
        -------
        NDArray or list of NDArray
            Interpolated state(s).

        Raises
        ------
        ValueError
            If the trajectory does not have a dense-output interpolator.
        """
        if trajectory.interpolator is None:
            raise ValueError(
                "Trajectory does not have a dense-output interpolator. "
                "Ensure store_trajectory=True and the integration produced "
                "enough points."
            )

        if np.isscalar(t):
            return trajectory.interpolator(float(t))

        t_arr = np.asarray(t, dtype=np.float64)
        return [trajectory.interpolator(float(ti)) for ti in t_arr]


# ======================================================================
#  Kernel Dynamics – Convenience Wrappers
# ======================================================================


class KernelDynamics:
    """Convenience wrappers for common kernel evolution right-hand sides.

    These static methods construct the dynamics functions  F(t, K)
    that appear on the right-hand side of the kernel ODE

        dK/dt = F(t, K).

    They are designed to be passed directly to
    :meth:`KernelODESolver.solve_kernel_evolution`.
    """

    # ------------------------------------------------------------------

    @staticmethod
    def linear_ntk_dynamics(
        K: NDArray,
        y: NDArray,
        X: NDArray,
        learning_rate: float,
    ) -> NDArray:
        """Linearised (lazy-regime) NTK dynamics.

        In the infinite-width limit, the NTK stays constant and the
        training dynamics reduce to

            dK/dt = −η ( K R + R K )

        where  R = (K y − X)(K y − X)ᵀ  is an outer-product residual
        matrix and η is the learning rate.

        For lazy training the kernel itself does not evolve; this
        function returns the *effective* linear change to first order.

        Parameters
        ----------
        K : NDArray, shape (n, n)
            Current kernel matrix.
        y : NDArray, shape (n,) or (n, d)
            Current function values / predictions.
        X : NDArray, shape (n,) or (n, d)
            Target values.
        learning_rate : float
            Learning rate η.

        Returns
        -------
        NDArray, shape (n, n)
            dK/dt.
        """
        residual = K @ y - X  # (n,) or (n, d)
        if residual.ndim == 1:
            R = np.outer(residual, residual)
        else:
            R = residual @ residual.T
        return -learning_rate * (K @ R + R @ K)

    # ------------------------------------------------------------------

    @staticmethod
    def nonlinear_ntk_dynamics(
        K: NDArray,
        y: NDArray,
        X: NDArray,
        learning_rate: float,
        correction_fn: Callable[[NDArray, NDArray], NDArray],
    ) -> NDArray:
        """Nonlinear NTK dynamics with finite-width corrections.

        Combines the linearised dynamics with an additive correction
        term that captures feature learning:

            dK/dt = −η ( K R + R K ) + δF(K, y)

        The correction function  δF  encodes the deviation from the
        lazy / kernel regime.

        Parameters
        ----------
        K : NDArray, shape (n, n)
            Current kernel matrix.
        y : NDArray, shape (n,) or (n, d)
            Current predictions.
        X : NDArray, shape (n,) or (n, d)
            Targets.
        learning_rate : float
            Learning rate η.
        correction_fn : callable (K, y) → δF
            Finite-width correction to the kernel dynamics.

        Returns
        -------
        NDArray, shape (n, n)
            dK/dt.
        """
        linear_part = KernelDynamics.linear_ntk_dynamics(K, y, X, learning_rate)
        correction = correction_fn(K, y)
        return linear_part + correction

    # ------------------------------------------------------------------

    @staticmethod
    def gradient_flow_dynamics(
        K: NDArray,
        y: NDArray,
        X: NDArray,
    ) -> NDArray:
        """Continuous-time gradient flow dynamics for the kernel.

        Under continuous-time gradient descent on the squared loss
        ½||f − X||², the effective kernel dynamics are

            dK/dt = −K (f − X)(f − X)ᵀ K

        where  f = K y  are the current predictions.  This is a
        rank-deficient matrix ODE that preserves positive
        semi-definiteness.

        Parameters
        ----------
        K : NDArray, shape (n, n)
            Current kernel matrix.
        y : NDArray, shape (n,) or (n, d)
            Current function / parameter values.
        X : NDArray, shape (n,) or (n, d)
            Target values.

        Returns
        -------
        NDArray, shape (n, n)
            dK/dt.
        """
        residual = K @ y - X
        if residual.ndim == 1:
            R = np.outer(residual, residual)
        else:
            R = residual @ residual.T
        return -K @ R @ K
