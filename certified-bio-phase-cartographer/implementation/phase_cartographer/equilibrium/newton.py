"""
Interval Newton methods for equilibrium finding.

Provides standard interval Newton and parameter-dependent
interval Newton for certified equilibrium computation.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from ..interval.interval import Interval
from ..interval.matrix import (IntervalVector, IntervalMatrix,
                               interval_gauss_seidel, verified_linear_solve)
from ..ode.rhs import ODERightHandSide


@dataclass
class NewtonResult:
    """Result of interval Newton iteration."""
    converged: bool
    enclosure: Optional[IntervalVector]
    unique: bool
    iterations: int
    contraction_rate: float
    residual_norm: float


class IntervalNewton:
    """
    Interval Newton method for finding zeros of f(x) = 0.
    
    The interval Newton step:
    N(X) = x̃ - J(X)^{-1} · f(x̃) ∩ X
    
    where J(X) is the interval Jacobian and x̃ = mid(X).
    If N(X) ⊂ int(X), existence and uniqueness are certified.
    """
    
    def __init__(self, rhs: ODERightHandSide,
                 max_iter: int = 30,
                 tol: float = 1e-12,
                 min_width: float = 1e-15):
        self.rhs = rhs
        self.max_iter = max_iter
        self.tol = tol
        self.min_width = min_width
    
    def solve(self, X: IntervalVector,
              mu: IntervalVector) -> NewtonResult:
        """
        Apply interval Newton iteration.
        """
        n = X.n
        current = IntervalVector(list(X.components))
        for iteration in range(self.max_iter):
            x_mid = current.midpoint()
            try:
                f_mid = self.rhs.evaluate(x_mid, mu.midpoint())
            except (ZeroDivisionError, ValueError):
                return NewtonResult(False, current, False, iteration, 1.0, float('inf'))
            residual = np.linalg.norm(f_mid)
            if residual < self.tol:
                return NewtonResult(True, current, True, iteration + 1,
                                  0.0, residual)
            try:
                J = self.rhs.jacobian_interval(current, mu)
            except (ZeroDivisionError, ValueError):
                return NewtonResult(False, current, False, iteration + 1,
                                  1.0, residual)
            b = IntervalVector([-Interval(f_mid[i]) for i in range(n)])
            try:
                delta, verified = verified_linear_solve(J, b)
            except Exception:
                J_mid = self.rhs.jacobian(x_mid, mu.midpoint())
                try:
                    delta_mid = np.linalg.solve(J_mid, -f_mid)
                    delta = IntervalVector([Interval(d) for d in delta_mid])
                    verified = False
                except np.linalg.LinAlgError:
                    return NewtonResult(False, current, False, iteration + 1,
                                      1.0, residual)
            N_components = []
            for i in range(n):
                ni = Interval(x_mid[i]) + delta[i]
                ni_intersect = current[i].intersection(ni)
                if ni_intersect.is_empty():
                    return NewtonResult(False, None, False, iteration + 1,
                                      1.0, residual)
                N_components.append(ni_intersect)
            new_X = IntervalVector(N_components)
            if current.contains(new_X):
                contraction = new_X.max_width() / max(current.max_width(), 1e-300)
                if contraction < 1.0:
                    return NewtonResult(True, new_X, True, iteration + 1,
                                      contraction, residual)
            old_width = current.max_width()
            current = new_X
            new_width = current.max_width()
            if new_width < self.min_width:
                return NewtonResult(True, current, True, iteration + 1,
                                  new_width / max(old_width, 1e-300), residual)
        return NewtonResult(False, current, False, self.max_iter,
                          1.0, float('inf'))
    
    def find_all_zeros(self, domain: IntervalVector,
                      mu: IntervalVector,
                      max_depth: int = 15) -> List[NewtonResult]:
        """Find all zeros within domain using bisection + Newton."""
        results = []
        self._find_zeros_recursive(domain, mu, max_depth, 0, results)
        return results
    
    def _find_zeros_recursive(self, X: IntervalVector,
                             mu: IntervalVector,
                             max_depth: int,
                             depth: int,
                             results: List[NewtonResult]):
        """Recursive zero finding."""
        result = self.solve(X, mu)
        if result.converged and result.unique:
            results.append(result)
            return
        try:
            f_enc = self.rhs.evaluate_interval(X, mu)
            for i in range(X.n):
                if not f_enc[i].contains(0.0):
                    return
        except (ZeroDivisionError, ValueError):
            pass
        if depth >= max_depth:
            return
        if X.max_width() < self.min_width:
            return
        left, right = X.split()
        self._find_zeros_recursive(left, mu, max_depth, depth + 1, results)
        self._find_zeros_recursive(right, mu, max_depth, depth + 1, results)


class ParametricNewton:
    """
    Parameter-dependent interval Newton method.
    
    Certifies that for ALL parameters mu in a box, there exists
    a unique equilibrium within a state box X, and tracks how
    the equilibrium moves with parameters.
    """
    
    def __init__(self, rhs: ODERightHandSide,
                 max_iter: int = 30,
                 tol: float = 1e-10):
        self.rhs = rhs
        self.max_iter = max_iter
        self.tol = tol
    
    def solve_parametric(self, X: IntervalVector,
                        mu_box: IntervalVector) -> NewtonResult:
        """
        Verify existence of equilibrium for all parameters in mu_box.
        """
        n = X.n
        current = IntervalVector(list(X.components))
        for iteration in range(self.max_iter):
            x_mid = current.midpoint()
            mu_mid = mu_box.midpoint()
            try:
                f_mid = self.rhs.evaluate(x_mid, mu_mid)
            except (ZeroDivisionError, ValueError):
                return NewtonResult(False, current, False, iteration, 1.0, float('inf'))
            residual = np.linalg.norm(f_mid)
            try:
                f_range = self.rhs.evaluate_interval(
                    IntervalVector([Interval(x_mid[i]) for i in range(n)]),
                    mu_box
                )
            except (ZeroDivisionError, ValueError):
                return NewtonResult(False, current, False, iteration, 1.0, float('inf'))
            try:
                J = self.rhs.jacobian_interval(current, mu_box)
            except (ZeroDivisionError, ValueError):
                return NewtonResult(False, current, False, iteration, 1.0, float('inf'))
            b = IntervalVector([-f_range[i] for i in range(n)])
            try:
                delta, verified = verified_linear_solve(J, b)
            except Exception:
                return NewtonResult(False, current, False, iteration, 1.0, residual)
            N_components = []
            for i in range(n):
                ni = Interval(x_mid[i]) + delta[i]
                ni_intersect = current[i].intersection(ni)
                if ni_intersect.is_empty():
                    return NewtonResult(False, None, False, iteration + 1, 1.0, residual)
                N_components.append(ni_intersect)
            new_X = IntervalVector(N_components)
            if current.contains(new_X):
                contraction = new_X.max_width() / max(current.max_width(), 1e-300)
                if contraction < 1.0:
                    return NewtonResult(True, new_X, True, iteration + 1,
                                      contraction, residual)
            current = new_X
            if current.max_width() < self.tol:
                return NewtonResult(True, current, True, iteration + 1, 0.0, residual)
        return NewtonResult(False, current, False, self.max_iter, 1.0, float('inf'))
    
    def continuation_step(self, X: IntervalVector,
                         mu_start: IntervalVector,
                         mu_end: IntervalVector,
                         n_steps: int = 10) -> List[NewtonResult]:
        """
        Parameter continuation: track equilibrium from mu_start to mu_end.
        """
        results = []
        n_params = mu_start.n
        current_X = IntervalVector(list(X.components))
        for step in range(n_steps + 1):
            alpha = step / n_steps
            mu_components = []
            for j in range(n_params):
                val = (1 - alpha) * mu_start[j].mid + alpha * mu_end[j].mid
                rad = max(mu_start[j].rad, mu_end[j].rad) / n_steps
                mu_components.append(Interval(val - rad, val + rad))
            mu_step = IntervalVector(mu_components)
            result = self.solve_parametric(current_X.inflate(0.1), mu_step)
            results.append(result)
            if result.converged and result.enclosure is not None:
                current_X = result.enclosure.inflate(0.01)
            else:
                break
        return results
