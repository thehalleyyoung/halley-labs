"""
Base classes for time integrators.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from tn_check.tensor.mps import MPS
from tn_check.tensor.mpo import MPO

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TimePoint:
    """A snapshot of the state at a particular time."""
    time: float
    state: MPS
    max_bond_dim: int
    total_probability: float
    truncation_error: float
    integration_error_estimate: float = 0.0
    negativity_violation: float = 0.0
    step_count: int = 0
    wall_time_seconds: float = 0.0


@dataclasses.dataclass
class IntegrationResult:
    """Result of a time integration."""
    final_state: MPS
    final_time: float
    time_points: list[TimePoint]
    total_truncation_error: float
    total_integration_error: float
    total_steps: int
    wall_time_seconds: float
    converged: bool
    method: str
    metadata: dict = dataclasses.field(default_factory=dict)

    @property
    def final_bond_dims(self) -> tuple:
        return self.final_state.bond_dims

    @property
    def max_bond_dim(self) -> int:
        return self.final_state.max_bond_dim

    def state_at_time(self, t: float) -> Optional[MPS]:
        """Get the state closest to time t from stored time points."""
        if not self.time_points:
            return None
        closest = min(self.time_points, key=lambda tp: abs(tp.time - t))
        return closest.state


class IntegratorBase:
    """
    Base class for CME time integrators.

    Provides common functionality for time stepping, error tracking,
    and adaptive step size control.
    """

    def __init__(
        self,
        generator_mpo: MPO,
        max_bond_dim: int = 100,
        truncation_tolerance: float = 1e-10,
        dt: float = 0.01,
        adaptive_dt: bool = True,
        dt_min: float = 1e-8,
        dt_max: float = 1.0,
        dt_safety_factor: float = 0.9,
        conservation_enforcement: bool = True,
        max_steps: int = 100000,
        store_interval: float = 0.0,
    ):
        self.generator_mpo = generator_mpo
        self.max_bond_dim = max_bond_dim
        self.truncation_tolerance = truncation_tolerance
        self.dt = dt
        self.adaptive_dt = adaptive_dt
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_safety_factor = dt_safety_factor
        self.conservation_enforcement = conservation_enforcement
        self.max_steps = max_steps
        self.store_interval = store_interval
        self._total_truncation_error = 0.0
        self._total_integration_error = 0.0
        self._step_count = 0
        self._time_points: list[TimePoint] = []

    def integrate(
        self,
        initial_state: MPS,
        t_final: float,
        t_start: float = 0.0,
    ) -> IntegrationResult:
        """
        Integrate the CME from t_start to t_final.

        Args:
            initial_state: Initial probability MPS.
            t_final: Final time.
            t_start: Start time.

        Returns:
            IntegrationResult.
        """
        start_wall = time.time()

        state = initial_state.copy()
        t = t_start
        dt = min(self.dt, t_final - t_start)
        self._step_count = 0
        self._total_truncation_error = 0.0
        self._total_integration_error = 0.0
        self._time_points = []

        last_store_time = t_start

        # Store initial state
        self._store_time_point(t, state, start_wall)

        while t < t_final - 1e-15:
            dt = min(dt, t_final - t)
            dt = max(dt, self.dt_min)

            # Take a step
            new_state, step_error, new_dt = self._step(state, t, dt)

            # Update tracking
            self._step_count += 1

            # Enforce probability conservation
            if self.conservation_enforcement:
                new_state = self._enforce_conservation(new_state)

            # Store time point if needed
            if self.store_interval > 0 and t + dt - last_store_time >= self.store_interval:
                self._store_time_point(t + dt, new_state, start_wall)
                last_store_time = t + dt

            state = new_state
            t += dt

            # Adaptive step size
            if self.adaptive_dt and new_dt is not None:
                dt = np.clip(new_dt, self.dt_min, self.dt_max)
            else:
                dt = min(self.dt, t_final - t)

            if self._step_count >= self.max_steps:
                logger.warning(
                    f"Maximum steps ({self.max_steps}) reached at t={t:.4f}"
                )
                break

        # Store final state
        self._store_time_point(t, state, start_wall)

        wall_time = time.time() - start_wall

        return IntegrationResult(
            final_state=state,
            final_time=t,
            time_points=self._time_points,
            total_truncation_error=self._total_truncation_error,
            total_integration_error=self._total_integration_error,
            total_steps=self._step_count,
            wall_time_seconds=wall_time,
            converged=abs(t - t_final) < 1e-10,
            method=self.__class__.__name__,
        )

    def _step(
        self,
        state: MPS,
        t: float,
        dt: float,
    ) -> tuple[MPS, float, Optional[float]]:
        """
        Take a single integration step. Must be implemented by subclasses.

        Args:
            state: Current state.
            t: Current time.
            dt: Step size.

        Returns:
            Tuple of (new_state, step_error, suggested_new_dt).
        """
        raise NotImplementedError

    def _enforce_conservation(self, state: MPS) -> MPS:
        """
        Enforce probability conservation (total probability = 1).

        Simple rescaling approach.
        """
        from tn_check.tensor.operations import (
            mps_total_probability,
            mps_normalize_probability,
        )
        total = mps_total_probability(state)
        if abs(total - 1.0) > 1e-10:
            state = mps_normalize_probability(state)
        return state

    def _store_time_point(
        self, t: float, state: MPS, start_wall: float
    ) -> None:
        """Store a time point snapshot."""
        from tn_check.tensor.operations import mps_total_probability
        total_prob = mps_total_probability(state)

        tp = TimePoint(
            time=t,
            state=state.copy(),
            max_bond_dim=state.max_bond_dim,
            total_probability=total_prob,
            truncation_error=self._total_truncation_error,
            integration_error_estimate=self._total_integration_error,
            step_count=self._step_count,
            wall_time_seconds=time.time() - start_wall,
        )
        self._time_points.append(tp)

    def _compress_state(self, state: MPS) -> tuple[MPS, float]:
        """Compress state with current settings."""
        from tn_check.tensor.canonical import svd_compress
        compressed, error = svd_compress(
            state,
            max_bond_dim=self.max_bond_dim,
            tolerance=self.truncation_tolerance,
        )
        self._total_truncation_error += error
        return compressed, error
