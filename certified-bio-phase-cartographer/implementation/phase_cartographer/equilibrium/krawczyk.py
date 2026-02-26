"""
Krawczyk operator for rigorous equilibrium certification.

The Krawczyk operator K(X) provides a sufficient condition for
existence and uniqueness of a zero of f within a box X:
if K(X) ⊂ X, then f has a unique zero in X.

K(X) = x̃ - R·f(x̃) + (I - R·f'(X))·(X - x̃)

where x̃ is the midpoint of X and R ≈ (f'(x̃))^(-1).
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector, IntervalMatrix
from ..ode.rhs import ODERightHandSide


class KrawczykStatus(Enum):
    """Status of Krawczyk verification."""
    VERIFIED = "verified"
    INCONCLUSIVE = "inconclusive"
    NO_ZERO = "no_zero"
    SINGULAR = "singular"
    ERROR = "error"


@dataclass
class KrawczykResult:
    """Result of Krawczyk operator application."""
    status: KrawczykStatus
    enclosure: Optional[IntervalVector] = None
    contraction_factor: float = float('inf')
    iterations: int = 0
    error_message: str = ""
    
    @property
    def verified(self) -> bool:
        return self.status == KrawczykStatus.VERIFIED
    
    @property
    def is_contracting(self) -> bool:
        return self.contraction_factor < 1.0


class KrawczykOperator:
    """
    Implementation of the Krawczyk operator for equilibrium verification.
    
    Given f: R^n -> R^n and a box X ⊂ R^n:
    K(X) = x̃ - R·f(x̃) + (I - R·J(X))·(X - x̃)
    
    where:
    - x̃ = mid(X) is the midpoint
    - R ≈ J(x̃)^{-1} is the approximate inverse Jacobian
    - J(X) is the interval Jacobian enclosure over X
    
    If K(X) ⊂ int(X), then f has a unique zero in X.
    """
    
    def __init__(self, rhs: ODERightHandSide,
                 max_iter: int = 20,
                 contraction_tol: float = 0.99,
                 width_tol: float = 1e-12,
                 inflation: float = 1.1):
        self.rhs = rhs
        self.max_iter = max_iter
        self.contraction_tol = contraction_tol
        self.width_tol = width_tol
        self.inflation = inflation
    
    def apply(self, X: IntervalVector,
              mu: IntervalVector,
              R: Optional[np.ndarray] = None) -> KrawczykResult:
        """
        Apply the Krawczyk operator once.
        
        Args:
            X: Box to verify
            mu: Parameter values
            R: Preconditioning matrix (approximate inverse Jacobian)
        
        Returns:
            KrawczykResult with verification status
        """
        n = X.n
        x_mid = X.midpoint()
        try:
            f_mid = self.rhs.evaluate(x_mid, mu.midpoint())
        except (ZeroDivisionError, ValueError, OverflowError) as e:
            return KrawczykResult(KrawczykStatus.ERROR,
                                error_message=str(e))
        if R is None:
            R = self._compute_preconditioner(x_mid, mu.midpoint())
            if R is None:
                return KrawczykResult(KrawczykStatus.SINGULAR,
                                    error_message="Jacobian is singular")
        Rf = R @ f_mid
        try:
            J_X = self.rhs.jacobian_interval(X, mu)
        except (ZeroDivisionError, ValueError) as e:
            return KrawczykResult(KrawczykStatus.ERROR,
                                error_message=str(e))
        I_n = IntervalMatrix.identity(n)
        R_iv = IntervalMatrix.from_numpy(R)
        RJ = R_iv.matmul(J_X)
        C = I_n - RJ
        X_centered = X - IntervalVector(x_mid)
        C_X = C.matmul(X_centered)
        K_components = []
        for i in range(n):
            ki = Interval(x_mid[i]) - Interval(Rf[i]) + C_X[i]
            K_components.append(ki)
        K = IntervalVector(K_components)
        contained = X.contains(K)
        if contained:
            contraction = 0.0
            for i in range(n):
                if X[i].width > 0:
                    ratio = K[i].width / X[i].width
                    contraction = max(contraction, ratio)
            return KrawczykResult(
                status=KrawczykStatus.VERIFIED,
                enclosure=K,
                contraction_factor=contraction
            )
        intersection_result = X.intersection(K)
        if intersection_result is None:
            return KrawczykResult(
                status=KrawczykStatus.NO_ZERO,
                contraction_factor=float('inf')
            )
        return KrawczykResult(
            status=KrawczykStatus.INCONCLUSIVE,
            enclosure=intersection_result,
            contraction_factor=1.0
        )
    
    def verify(self, X: IntervalVector,
               mu: IntervalVector,
               R: Optional[np.ndarray] = None) -> KrawczykResult:
        """
        Iterative Krawczyk verification with contraction.
        Repeatedly applies K until containment is verified or iteration limit reached.
        """
        n = X.n
        if R is None:
            R = self._compute_preconditioner(X.midpoint(), mu.midpoint())
            if R is None:
                return KrawczykResult(KrawczykStatus.SINGULAR,
                                    error_message="Cannot compute preconditioner")
        current_X = IntervalVector(list(X.components))
        for iteration in range(self.max_iter):
            result = self.apply(current_X, mu, R)
            result.iterations = iteration + 1
            if result.status == KrawczykStatus.VERIFIED:
                return result
            if result.status == KrawczykStatus.NO_ZERO:
                return result
            if result.status == KrawczykStatus.ERROR:
                return result
            if result.enclosure is not None:
                new_X = result.enclosure
                width_ratio = new_X.max_width() / max(current_X.max_width(), 1e-300)
                if width_ratio > self.contraction_tol:
                    return KrawczykResult(
                        status=KrawczykStatus.INCONCLUSIVE,
                        enclosure=new_X,
                        contraction_factor=width_ratio,
                        iterations=iteration + 1,
                        error_message="No contraction"
                    )
                current_X = new_X
                if current_X.max_width() < self.width_tol:
                    return KrawczykResult(
                        status=KrawczykStatus.VERIFIED,
                        enclosure=current_X,
                        contraction_factor=width_ratio,
                        iterations=iteration + 1
                    )
            else:
                return KrawczykResult(
                    status=KrawczykStatus.INCONCLUSIVE,
                    iterations=iteration + 1
                )
        return KrawczykResult(
            status=KrawczykStatus.INCONCLUSIVE,
            enclosure=current_X,
            iterations=self.max_iter,
            error_message="Max iterations reached"
        )
    
    def verify_parametric(self, X: IntervalVector,
                         mu_box: IntervalVector,
                         n_subdivisions: int = 1) -> KrawczykResult:
        """
        Verify existence of equilibrium for ALL parameters in mu_box.
        Uses parameter-dependent Krawczyk operator.
        """
        if n_subdivisions <= 1:
            return self.verify(X, mu_box)
        from ..interval.utils import split_interval
        sub_results = []
        mu_subs = self._subdivide_params(mu_box, n_subdivisions)
        for mu_sub in mu_subs:
            result = self.verify(X, mu_sub)
            if not result.verified:
                return KrawczykResult(
                    status=KrawczykStatus.INCONCLUSIVE,
                    error_message=f"Failed for parameter sub-box"
                )
            sub_results.append(result)
        hull_enc = sub_results[0].enclosure
        for r in sub_results[1:]:
            if r.enclosure is not None and hull_enc is not None:
                hull_enc = hull_enc.hull(r.enclosure)
        max_cf = max(r.contraction_factor for r in sub_results)
        return KrawczykResult(
            status=KrawczykStatus.VERIFIED,
            enclosure=hull_enc,
            contraction_factor=max_cf,
            iterations=sum(r.iterations for r in sub_results)
        )
    
    def find_equilibria(self, domain: IntervalVector,
                       mu: IntervalVector,
                       max_depth: int = 10) -> List[KrawczykResult]:
        """
        Find all equilibria within a domain by recursive bisection
        with Krawczyk verification.
        """
        results = []
        self._find_equilibria_recursive(domain, mu, max_depth, 0, results)
        return results
    
    def _find_equilibria_recursive(self, X: IntervalVector,
                                  mu: IntervalVector,
                                  max_depth: int,
                                  depth: int,
                                  results: List[KrawczykResult]):
        """Recursive equilibrium search."""
        result = self.verify(X, mu)
        if result.status == KrawczykStatus.VERIFIED:
            results.append(result)
            return
        if result.status == KrawczykStatus.NO_ZERO:
            return
        if depth >= max_depth:
            return
        if X.max_width() < self.width_tol:
            return
        left, right = X.split()
        self._find_equilibria_recursive(left, mu, max_depth, depth + 1, results)
        self._find_equilibria_recursive(right, mu, max_depth, depth + 1, results)
    
    def _compute_preconditioner(self, x: np.ndarray,
                               mu: np.ndarray) -> Optional[np.ndarray]:
        """Compute preconditioner R ≈ J(x)^{-1}."""
        try:
            J = self.rhs.jacobian(x, mu)
            cond = np.linalg.cond(J)
            if cond > 1e15:
                J_reg = J + 1e-10 * np.eye(len(x))
                return np.linalg.inv(J_reg)
            return np.linalg.inv(J)
        except np.linalg.LinAlgError:
            return None
    
    def _subdivide_params(self, mu_box: IntervalVector,
                         n: int) -> List[IntervalVector]:
        """Subdivide parameter box."""
        from ..interval.utils import split_interval
        widths = mu_box.width()
        widest_dim = int(np.argmax(widths))
        subs = split_interval(mu_box[widest_dim], n)
        result = []
        for sub in subs:
            components = list(mu_box.components)
            components[widest_dim] = sub
            result.append(IntervalVector(components))
        return result
