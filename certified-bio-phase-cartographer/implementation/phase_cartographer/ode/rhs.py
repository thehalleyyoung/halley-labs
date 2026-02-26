"""
ODE right-hand side representations for validated integration.

Supports polynomial, rational, and Hill-function right-hand sides
common in biological ODE models.
"""

import numpy as np
from typing import List, Callable, Optional, Dict, Tuple
from enum import Enum
import sympy as sp

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector, IntervalMatrix
from ..interval.taylor_model import TaylorModel, TaylorModelVector


class RHSType(Enum):
    """Type of right-hand side function."""
    POLYNOMIAL = "polynomial"
    RATIONAL = "rational"
    HILL = "hill"
    GENERAL = "general"


class ODERightHandSide:
    """
    Abstract base for ODE right-hand side f(x, mu).
    
    x: state vector (dimension n)
    mu: parameter vector (dimension p)
    """
    
    def __init__(self, n_states: int, n_params: int, name: str = ""):
        self.n_states = n_states
        self.n_params = n_params
        self.name = name
        self._rhs_type = RHSType.GENERAL
    
    @property
    def rhs_type(self) -> RHSType:
        return self._rhs_type
    
    def evaluate(self, x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Evaluate f(x, mu) at a point (floating-point)."""
        raise NotImplementedError
    
    def evaluate_interval(self, x: IntervalVector,
                         mu: IntervalVector) -> IntervalVector:
        """Evaluate f(x, mu) with interval arithmetic."""
        raise NotImplementedError
    
    def evaluate_taylor(self, x: TaylorModelVector,
                       mu: List[Interval]) -> TaylorModelVector:
        """Evaluate f using Taylor model arithmetic."""
        raise NotImplementedError
    
    def jacobian(self, x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Compute Jacobian df/dx at a point."""
        n = self.n_states
        jac = np.zeros((n, n))
        eps = 1e-8
        f0 = self.evaluate(x, mu)
        for j in range(n):
            x_pert = x.copy()
            x_pert[j] += eps
            f_pert = self.evaluate(x_pert, mu)
            jac[:, j] = (f_pert - f0) / eps
        return jac
    
    def jacobian_interval(self, x: IntervalVector,
                         mu: IntervalVector) -> IntervalMatrix:
        """Compute interval Jacobian enclosure df/dx."""
        raise NotImplementedError
    
    def parameter_jacobian(self, x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Compute parameter Jacobian df/dmu."""
        n = self.n_states
        p = self.n_params
        jac = np.zeros((n, p))
        eps = 1e-8
        f0 = self.evaluate(x, mu)
        for j in range(p):
            mu_pert = mu.copy()
            mu_pert[j] += eps
            f_pert = self.evaluate(x, mu_pert)
            jac[:, j] = (f_pert - f0) / eps
        return jac
    
    def is_positive_invariant(self, x: IntervalVector,
                              mu: IntervalVector) -> bool:
        """Check if the non-negative orthant is forward-invariant."""
        n = self.n_states
        for i in range(n):
            x_boundary = IntervalVector(list(x.components))
            x_boundary[i] = Interval(0.0, 0.0)
            try:
                f = self.evaluate_interval(x_boundary, mu)
                if f[i].lo < 0:
                    return False
            except (ZeroDivisionError, ValueError):
                return False
        return True


class PolynomialRHS(ODERightHandSide):
    """
    Polynomial right-hand side.
    Each component f_i(x, mu) is a multivariate polynomial in x with
    parameter-dependent coefficients.
    """
    
    def __init__(self, n_states: int, n_params: int,
                 terms: Optional[List[List[Tuple]]] = None,
                 name: str = ""):
        super().__init__(n_states, n_params, name)
        self._rhs_type = RHSType.POLYNOMIAL
        self.terms = terms if terms is not None else [[] for _ in range(n_states)]
    
    def add_term(self, component: int, coefficient_fn: Callable,
                 exponents: Tuple[int, ...]):
        """Add a polynomial term to component i."""
        self.terms[component].append((coefficient_fn, exponents))
    
    def evaluate(self, x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        result = np.zeros(self.n_states)
        for i in range(self.n_states):
            for coeff_fn, exponents in self.terms[i]:
                c = coeff_fn(mu) if callable(coeff_fn) else coeff_fn
                term = c
                for j, e in enumerate(exponents):
                    if e > 0:
                        term *= x[j] ** e
                result[i] += term
        return result
    
    def evaluate_interval(self, x: IntervalVector,
                         mu: IntervalVector) -> IntervalVector:
        n = self.n_states
        result = []
        for i in range(n):
            s = Interval(0.0)
            for coeff_fn, exponents in self.terms[i]:
                if callable(coeff_fn):
                    mu_arr = np.array([mu[j].mid for j in range(self.n_params)])
                    c = Interval(coeff_fn(mu_arr))
                else:
                    c = Interval(coeff_fn)
                term = c
                for j, e in enumerate(exponents):
                    if e > 0:
                        term = term * (x[j] ** e)
                s = s + term
            result.append(s)
        return IntervalVector(result)
    
    def evaluate_taylor(self, x: TaylorModelVector,
                       mu: List[Interval]) -> TaylorModelVector:
        n = self.n_states
        dim = x[0].dim
        order = x[0].order
        domain = x[0].domain
        result = []
        for i in range(n):
            s = TaylorModel.constant(dim, order, 0.0, domain)
            for coeff_fn, exponents in self.terms[i]:
                if callable(coeff_fn):
                    mu_arr = np.array([m.mid for m in mu])
                    c = coeff_fn(mu_arr)
                else:
                    c = coeff_fn
                term = TaylorModel.constant(dim, order, c, domain)
                for j, e in enumerate(exponents):
                    if e > 0:
                        term = term * (x[j] ** e)
                s = s + term
            result.append(s)
        return TaylorModelVector(result)
    
    def jacobian_interval(self, x: IntervalVector,
                         mu: IntervalVector) -> IntervalMatrix:
        n = self.n_states
        entries = [[Interval(0.0) for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for coeff_fn, exponents in self.terms[i]:
                if callable(coeff_fn):
                    mu_arr = np.array([mu[j].mid for j in range(self.n_params)])
                    c = Interval(coeff_fn(mu_arr))
                else:
                    c = Interval(coeff_fn)
                for j in range(n):
                    if j < len(exponents) and exponents[j] > 0:
                        d_term = c * Interval(exponents[j])
                        for k, e in enumerate(exponents):
                            if k == j:
                                if e > 1:
                                    d_term = d_term * (x[k] ** (e - 1))
                            elif e > 0:
                                d_term = d_term * (x[k] ** e)
                        entries[i][j] = entries[i][j] + d_term
        return IntervalMatrix(entries)


class RationalRHS(ODERightHandSide):
    """
    Rational right-hand side: f_i = numerator_i / denominator_i.
    Common in biological models with Michaelis-Menten or Hill kinetics.
    """
    
    def __init__(self, n_states: int, n_params: int,
                 name: str = ""):
        super().__init__(n_states, n_params, name)
        self._rhs_type = RHSType.RATIONAL
        self._eval_fn = None
        self._eval_interval_fn = None
        self._eval_taylor_fn = None
        self._jac_interval_fn = None
    
    def set_functions(self, eval_fn: Callable,
                     eval_interval_fn: Optional[Callable] = None,
                     eval_taylor_fn: Optional[Callable] = None,
                     jac_interval_fn: Optional[Callable] = None):
        """Set evaluation functions."""
        self._eval_fn = eval_fn
        self._eval_interval_fn = eval_interval_fn
        self._eval_taylor_fn = eval_taylor_fn
        self._jac_interval_fn = jac_interval_fn
    
    def evaluate(self, x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        if self._eval_fn is None:
            raise NotImplementedError("Evaluation function not set")
        return self._eval_fn(x, mu)
    
    def evaluate_interval(self, x: IntervalVector,
                         mu: IntervalVector) -> IntervalVector:
        if self._eval_interval_fn is None:
            raise NotImplementedError("Interval evaluation not set")
        return self._eval_interval_fn(x, mu)
    
    def evaluate_taylor(self, x: TaylorModelVector,
                       mu: List[Interval]) -> TaylorModelVector:
        if self._eval_taylor_fn is None:
            raise NotImplementedError("Taylor evaluation not set")
        return self._eval_taylor_fn(x, mu)
    
    def jacobian_interval(self, x: IntervalVector,
                         mu: IntervalVector) -> IntervalMatrix:
        if self._jac_interval_fn is not None:
            return self._jac_interval_fn(x, mu)
        n = self.n_states
        eps = 1e-8
        entries = [[Interval(0.0) for _ in range(n)] for _ in range(n)]
        f0 = self.evaluate_interval(x, mu)
        for j in range(n):
            x_pert = IntervalVector(list(x.components))
            x_pert[j] = x[j] + Interval(eps)
            try:
                f_pert = self.evaluate_interval(x_pert, mu)
                for i in range(n):
                    entries[i][j] = (f_pert[i] - f0[i]) / Interval(eps)
            except (ZeroDivisionError, ValueError):
                for i in range(n):
                    entries[i][j] = Interval(-1e10, 1e10)
        return IntervalMatrix(entries)


class HillFunctionRHS(ODERightHandSide):
    """
    Right-hand side using Hill functions: common in gene regulatory networks.
    f_i = production_i * Hill(x_activators, K, n) - degradation_i * x_i
    """
    
    def __init__(self, n_states: int, n_params: int,
                 production_terms: Optional[List] = None,
                 degradation_rates: Optional[List] = None,
                 name: str = ""):
        super().__init__(n_states, n_params, name)
        self._rhs_type = RHSType.HILL
        self.production_terms = production_terms or []
        self.degradation_rates = degradation_rates or []
    
    @staticmethod
    def hill(x: float, K: float, n: int) -> float:
        """Scalar Hill function."""
        xn = x ** n
        return xn / (K ** n + xn)
    
    @staticmethod
    def hill_interval(x: Interval, K: Interval, n: int) -> Interval:
        """Interval Hill function."""
        xn = x ** n
        Kn = K ** n
        denom = Kn + xn
        if denom.contains(0.0):
            raise ValueError("Hill function denominator contains zero")
        return xn / denom
    
    @staticmethod
    def hill_repressive(x: float, K: float, n: int) -> float:
        """Repressive Hill function: K^n / (K^n + x^n)."""
        Kn = K ** n
        return Kn / (Kn + x ** n)
    
    @staticmethod
    def hill_repressive_interval(x: Interval, K: Interval, n: int) -> Interval:
        """Interval repressive Hill function."""
        Kn = K ** n
        xn = x ** n
        denom = Kn + xn
        if denom.contains(0.0):
            raise ValueError("Repressive Hill denominator contains zero")
        return Kn / denom
    
    def evaluate(self, x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclass must implement")
    
    def evaluate_interval(self, x: IntervalVector,
                         mu: IntervalVector) -> IntervalVector:
        raise NotImplementedError("Subclass must implement")


class SymbolicRHS(ODERightHandSide):
    """
    RHS defined symbolically using sympy.
    Automatically generates interval and Taylor model evaluators.
    """
    
    def __init__(self, n_states: int, n_params: int,
                 state_symbols: Optional[List[sp.Symbol]] = None,
                 param_symbols: Optional[List[sp.Symbol]] = None,
                 expressions: Optional[List[sp.Expr]] = None,
                 name: str = ""):
        super().__init__(n_states, n_params, name)
        self._rhs_type = RHSType.GENERAL
        if state_symbols is None:
            self.state_symbols = [sp.Symbol(f'x{i}') for i in range(n_states)]
        else:
            self.state_symbols = list(state_symbols)
        if param_symbols is None:
            self.param_symbols = [sp.Symbol(f'mu{i}') for i in range(n_params)]
        else:
            self.param_symbols = list(param_symbols)
        self.expressions = expressions if expressions is not None else []
        self._compiled_fn = None
        self._jac_expr = None
    
    def set_expressions(self, expressions: List[sp.Expr]):
        """Set the symbolic expressions for each component."""
        self.expressions = list(expressions)
        self._compiled_fn = None
        self._jac_expr = None
    
    def _compile(self):
        """Compile symbolic expressions to numerical functions."""
        if self._compiled_fn is not None:
            return
        all_symbols = self.state_symbols + self.param_symbols
        self._compiled_fn = [
            sp.lambdify(all_symbols, expr, modules=['numpy'])
            for expr in self.expressions
        ]
    
    def _compute_jacobian_expr(self):
        """Compute symbolic Jacobian."""
        if self._jac_expr is not None:
            return
        self._jac_expr = []
        for expr in self.expressions:
            row = [sp.diff(expr, x) for x in self.state_symbols]
            self._jac_expr.append(row)
    
    def evaluate(self, x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        self._compile()
        args = list(x) + list(mu)
        result = np.zeros(self.n_states)
        for i, fn in enumerate(self._compiled_fn):
            result[i] = fn(*args)
        return result
    
    def jacobian(self, x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        self._compute_jacobian_expr()
        all_symbols = self.state_symbols + self.param_symbols
        args = list(x) + list(mu)
        subs = {s: v for s, v in zip(all_symbols, args)}
        n = self.n_states
        jac = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                jac[i, j] = float(self._jac_expr[i][j].subs(subs))
        return jac
    
    def evaluate_interval(self, x: IntervalVector,
                         mu: IntervalVector) -> IntervalVector:
        n = self.n_states
        result = []
        all_symbols = self.state_symbols + self.param_symbols
        all_vals = list(x.components) + list(mu.components)
        subs = dict(zip(all_symbols, all_vals))
        for expr in self.expressions:
            val = self._eval_symbolic_interval(expr, subs)
            result.append(val)
        return IntervalVector(result)
    
    def _eval_symbolic_interval(self, expr, subs: dict) -> Interval:
        """Evaluate a sympy expression with interval substitution."""
        if isinstance(expr, sp.Number):
            return Interval(float(expr))
        if isinstance(expr, sp.Symbol):
            if expr in subs:
                v = subs[expr]
                if isinstance(v, Interval):
                    return v
                return Interval(float(v))
            raise ValueError(f"Unknown symbol: {expr}")
        if isinstance(expr, sp.Add):
            result = Interval(0.0)
            for arg in expr.args:
                result = result + self._eval_symbolic_interval(arg, subs)
            return result
        if isinstance(expr, sp.Mul):
            result = Interval(1.0)
            for arg in expr.args:
                result = result * self._eval_symbolic_interval(arg, subs)
            return result
        if isinstance(expr, sp.Pow):
            base = self._eval_symbolic_interval(expr.args[0], subs)
            exp_val = expr.args[1]
            if isinstance(exp_val, sp.Integer):
                return base ** int(exp_val)
            elif isinstance(exp_val, sp.Rational):
                if exp_val == sp.Rational(1, 2):
                    return base.sqrt()
                return base ** float(exp_val)
            exp_iv = self._eval_symbolic_interval(exp_val, subs)
            return (base.log() * exp_iv).exp()
        if isinstance(expr, sp.exp):
            arg = self._eval_symbolic_interval(expr.args[0], subs)
            return arg.exp()
        if isinstance(expr, sp.log):
            arg = self._eval_symbolic_interval(expr.args[0], subs)
            return arg.log()
        if isinstance(expr, sp.sin):
            arg = self._eval_symbolic_interval(expr.args[0], subs)
            return arg.sin()
        if isinstance(expr, sp.cos):
            arg = self._eval_symbolic_interval(expr.args[0], subs)
            return arg.cos()
        try:
            return Interval(float(expr))
        except (TypeError, ValueError):
            raise NotImplementedError(f"Cannot evaluate {type(expr)}: {expr}")
    
    def jacobian_interval(self, x: IntervalVector,
                         mu: IntervalVector) -> IntervalMatrix:
        self._compute_jacobian_expr()
        n = self.n_states
        all_symbols = self.state_symbols + self.param_symbols
        all_vals = list(x.components) + list(mu.components)
        subs = dict(zip(all_symbols, all_vals))
        entries = []
        for i in range(n):
            row = []
            for j in range(n):
                val = self._eval_symbolic_interval(self._jac_expr[i][j], subs)
                row.append(val)
            entries.append(row)
        return IntervalMatrix(entries)
    
    def evaluate_taylor(self, x: TaylorModelVector,
                       mu: List[Interval]) -> TaylorModelVector:
        """Fallback: evaluate using interval arithmetic on TM bounds."""
        x_iv = x.bound()
        mu_iv = IntervalVector(mu)
        f_iv = self.evaluate_interval(x_iv, mu_iv)
        dim = x[0].dim
        order = x[0].order
        domain = x[0].domain
        result = []
        for i in range(self.n_states):
            tm = TaylorModel.from_interval(dim, order, f_iv[i], domain)
            result.append(tm)
        return TaylorModelVector(result)
