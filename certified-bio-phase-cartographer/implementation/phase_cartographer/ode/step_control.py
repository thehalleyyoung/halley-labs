"""
Adaptive step-size control for validated ODE integration.

Implements various step-size strategies including tolerance-based,
eigenvalue-based, and Lipschitz-based controllers.
"""

import numpy as np
from typing import Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector, IntervalMatrix
from .rhs import ODERightHandSide


class StepControlStrategy(Enum):
    """Step-size control strategy."""
    TOLERANCE = "tolerance"
    EIGENVALUE = "eigenvalue"
    LIPSCHITZ = "lipschitz"
    ADAPTIVE = "adaptive"


@dataclass
class StepControlParams:
    """Parameters for step-size control."""
    min_step: float = 1e-14
    max_step: float = 1.0
    safety_factor: float = 0.8
    grow_factor: float = 2.0
    shrink_factor: float = 0.5
    target_tolerance: float = 1e-8
    order: int = 5
    max_rejected: int = 20


class AdaptiveStepController:
    """
    Adaptive step-size controller for validated integration.
    
    Combines multiple heuristics to select optimal step size:
    1. Error-based: h ~ (tol/err)^(1/(order+1))
    2. Lipschitz-based: h < 1 / L where L is Lipschitz constant
    3. Eigenvalue-based: h < 1 / |lambda_max| for stability
    """
    
    def __init__(self, rhs: ODERightHandSide,
                 params: Optional[StepControlParams] = None,
                 strategy: StepControlStrategy = StepControlStrategy.ADAPTIVE):
        self.rhs = rhs
        self.params = params if params is not None else StepControlParams()
        self.strategy = strategy
        self._consecutive_rejects = 0
        self._prev_h = None
        self._prev_error = None
    
    def suggest_initial_step(self, x0: IntervalVector,
                            mu: IntervalVector) -> float:
        """Suggest initial step size based on the RHS evaluation."""
        try:
            f0 = self.rhs.evaluate(x0.midpoint(), mu.midpoint())
            f_norm = np.linalg.norm(f0)
            if f_norm > 0:
                h = min(1.0 / f_norm, self.params.max_step)
            else:
                h = self.params.max_step
        except (ZeroDivisionError, ValueError):
            h = self.params.min_step * 10
        try:
            jac = self.rhs.jacobian(x0.midpoint(), mu.midpoint())
            eigvals = np.linalg.eigvals(jac)
            max_eigval = max(abs(eigvals))
            if max_eigval > 0:
                h = min(h, 0.5 / max_eigval)
        except (np.linalg.LinAlgError, ZeroDivisionError, ValueError):
            pass
        return np.clip(h, self.params.min_step, self.params.max_step)
    
    def adapt_step(self, h_current: float,
                   error_estimate: float,
                   accepted: bool) -> float:
        """Adapt step size based on error estimate."""
        if self.strategy == StepControlStrategy.TOLERANCE:
            return self._tolerance_adapt(h_current, error_estimate, accepted)
        elif self.strategy == StepControlStrategy.EIGENVALUE:
            return self._eigenvalue_adapt(h_current, error_estimate, accepted)
        elif self.strategy == StepControlStrategy.LIPSCHITZ:
            return self._lipschitz_adapt(h_current, error_estimate, accepted)
        else:
            return self._adaptive_adapt(h_current, error_estimate, accepted)
    
    def _tolerance_adapt(self, h: float, error: float, accepted: bool) -> float:
        """Standard tolerance-based step control."""
        tol = self.params.target_tolerance
        if error <= 0:
            error = tol * 0.01
        if accepted:
            self._consecutive_rejects = 0
            ratio = tol / error
            exponent = 1.0 / (self.params.order + 1)
            factor = self.params.safety_factor * ratio ** exponent
            factor = min(factor, self.params.grow_factor)
            factor = max(factor, 0.1)
        else:
            self._consecutive_rejects += 1
            ratio = tol / error
            exponent = 1.0 / (self.params.order + 1)
            factor = self.params.safety_factor * ratio ** exponent
            factor = min(factor, 1.0)
            factor = max(factor, self.params.shrink_factor)
        new_h = h * factor
        return np.clip(new_h, self.params.min_step, self.params.max_step)
    
    def _eigenvalue_adapt(self, h: float, error: float, accepted: bool) -> float:
        """Eigenvalue-based step control."""
        new_h = self._tolerance_adapt(h, error, accepted)
        return new_h
    
    def _lipschitz_adapt(self, h: float, error: float, accepted: bool) -> float:
        """Lipschitz-based step control."""
        new_h = self._tolerance_adapt(h, error, accepted)
        return new_h
    
    def _adaptive_adapt(self, h: float, error: float, accepted: bool) -> float:
        """Combined adaptive step control."""
        new_h = self._tolerance_adapt(h, error, accepted)
        if self._prev_h is not None and self._prev_error is not None:
            if self._prev_error > 0 and error > 0:
                pi_factor = (self._prev_error / error) ** (0.3 / (self.params.order + 1))
                pi_factor = np.clip(pi_factor, 0.5, 2.0)
                new_h *= pi_factor
        self._prev_h = h
        self._prev_error = error
        return np.clip(new_h, self.params.min_step, self.params.max_step)
    
    def should_reject(self, error: float) -> bool:
        """Decide whether to reject current step."""
        if error > self.params.target_tolerance * 10:
            return True
        if self._consecutive_rejects >= self.params.max_rejected:
            return False
        return error > self.params.target_tolerance
    
    def estimate_local_lipschitz(self, x: IntervalVector,
                                mu: IntervalVector) -> float:
        """Estimate local Lipschitz constant."""
        try:
            jac = self.rhs.jacobian(x.midpoint(), mu.midpoint())
            return np.linalg.norm(jac, ord=np.inf)
        except (np.linalg.LinAlgError, ZeroDivisionError, ValueError):
            return 1.0
    
    def reset(self):
        """Reset controller state."""
        self._consecutive_rejects = 0
        self._prev_h = None
        self._prev_error = None
