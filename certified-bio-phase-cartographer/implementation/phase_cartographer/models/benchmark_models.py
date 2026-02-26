"""
Benchmark biological model definitions.

Each model provides:
  - Symbolic RHS via SymPy
  - Parameter domain for phase-atlas construction
  - State domain for equilibrium search
  - Expected regimes (for validation, not certification)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import sympy as sp

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector
from ..ode.rhs import SymbolicRHS


@dataclass
class BenchmarkModel:
    """Specification of a benchmark biological model."""
    name: str
    n_states: int
    n_params: int
    rhs: SymbolicRHS
    parameter_domain: List[Tuple[float, float]]
    state_domain: List[Tuple[float, float]]
    rhs_type: str  # "hill", "polynomial", "rational"
    expected_regimes: List[str]
    description: str = ""


def make_toggle_switch() -> BenchmarkModel:
    """Gardner toggle switch (2 states, 4 params)."""
    x1, x2 = sp.symbols('x0 x1')
    a1, a2, n1, n2 = sp.symbols('mu0 mu1 mu2 mu3')

    rhs = SymbolicRHS(
        n_states=2, n_params=4,
        state_symbols=[x1, x2],
        param_symbols=[a1, a2, n1, n2],
        name="toggle_switch",
    )
    rhs.set_expressions([
        a1 / (1 + x2**2) - x1,
        a2 / (1 + x1**2) - x2,
    ])

    return BenchmarkModel(
        name="toggle_switch",
        n_states=2, n_params=4,
        rhs=rhs,
        parameter_domain=[(2.0, 4.0), (2.0, 4.0), (2.0, 2.0), (2.0, 2.0)],
        state_domain=[(0.1, 4.5), (0.1, 4.5)],
        rhs_type="hill",
        expected_regimes=["bistable", "monostable"],
        description="Gardner toggle switch: mutual Hill repression (n1=n2=2 fixed)",
    )


def make_brusselator() -> BenchmarkModel:
    """Brusselator (2 states, 2 params)."""
    x1, x2 = sp.symbols('x0 x1')
    A, B = sp.symbols('mu0 mu1')

    rhs = SymbolicRHS(
        n_states=2, n_params=2,
        state_symbols=[x1, x2],
        param_symbols=[A, B],
        name="brusselator",
    )
    rhs.set_expressions([
        A - (B + 1) * x1 + x1**2 * x2,
        B * x1 - x1**2 * x2,
    ])

    return BenchmarkModel(
        name="brusselator",
        n_states=2, n_params=2,
        rhs=rhs,
        parameter_domain=[(0.8, 2.0), (1.5, 3.5)],
        state_domain=[(0.3, 3.0), (0.5, 5.0)],
        rhs_type="polynomial",
        expected_regimes=["oscillatory", "monostable"],
        description="Brusselator: polynomial autocatalytic model",
    )


def make_selkov() -> BenchmarkModel:
    """Sel'kov glycolysis model (2 states, 2 params)."""
    x1, x2 = sp.symbols('x0 x1')
    a, b = sp.symbols('mu0 mu1')

    rhs = SymbolicRHS(
        n_states=2, n_params=2,
        state_symbols=[x1, x2],
        param_symbols=[a, b],
        name="selkov",
    )
    rhs.set_expressions([
        -x1 + a * x2 + x1**2 * x2,
        b - a * x2 - x1**2 * x2,
    ])

    return BenchmarkModel(
        name="selkov",
        n_states=2, n_params=2,
        rhs=rhs,
        parameter_domain=[(0.05, 0.15), (0.4, 0.8)],
        state_domain=[(0.01, 2.5), (0.01, 3.5)],
        rhs_type="polynomial",
        expected_regimes=["oscillatory", "monostable"],
        description="Sel'kov glycolysis oscillator",
    )


def make_repressilator() -> BenchmarkModel:
    """Repressilator (3 states, 4 params): cyclic repression circuit."""
    x1, x2, x3 = sp.symbols('x0 x1 x2')
    alpha, K, n_h, gamma = sp.symbols('mu0 mu1 mu2 mu3')

    rhs = SymbolicRHS(
        n_states=3, n_params=4,
        state_symbols=[x1, x2, x3],
        param_symbols=[alpha, K, n_h, gamma],
        name="repressilator",
    )
    # Simplified repressilator with Hill coefficient fixed at 2:
    # dx1/dt = alpha/(1 + x3^2) - gamma*x1
    # dx2/dt = alpha/(1 + x1^2) - gamma*x2
    # dx3/dt = alpha/(1 + x2^2) - gamma*x3
    rhs.set_expressions([
        alpha / (1 + x3**2) - gamma * x1,
        alpha / (1 + x1**2) - gamma * x2,
        alpha / (1 + x2**2) - gamma * x3,
    ])

    return BenchmarkModel(
        name="repressilator",
        n_states=3, n_params=4,
        rhs=rhs,
        parameter_domain=[(3.5, 5.5), (1.0, 1.0), (2.0, 2.0), (1.0, 1.0)],
        state_domain=[(0.5, 5.0), (0.5, 5.0), (0.5, 5.0)],
        rhs_type="hill",
        expected_regimes=["oscillatory", "monostable"],
        description="Repressilator: 3-gene cyclic repression oscillator (Hill n=2, gamma=1 fixed)",
    )


def make_goodwin() -> BenchmarkModel:
    """Goodwin oscillator (3 states, 2 varying params).

    dx1/dt = a / (1 + x3^n) - b * x1
    dx2/dt = x1 - b * x2
    dx3/dt = x2 - b * x3

    K = 1 (fixed), n = 4 (fixed Hill coefficient).
    Varying parameters: a ∈ [3, 8], b ∈ [0.8, 1.2].
    """
    x1, x2, x3 = sp.symbols('x0 x1 x2')
    a, b = sp.symbols('mu0 mu1')

    rhs = SymbolicRHS(
        n_states=3, n_params=2,
        state_symbols=[x1, x2, x3],
        param_symbols=[a, b],
        name="goodwin",
    )
    # K=1 fixed, n=4 fixed (keeps bifurcation but tractable interval widths)
    rhs.set_expressions([
        a / (1 + x3**4) - b * x1,
        x1 - b * x2,
        x2 - b * x3,
    ])

    return BenchmarkModel(
        name="goodwin",
        n_states=3, n_params=2,
        rhs=rhs,
        parameter_domain=[(3.0, 8.0), (0.8, 1.2)],
        state_domain=[(0.5, 4.0), (0.5, 4.0), (0.5, 4.0)],
        rhs_type="hill",
        expected_regimes=["oscillatory", "monostable"],
        description="Goodwin oscillator: negative-feedback loop with Hill n=4",
    )


ALL_BENCHMARKS = {
    "toggle_switch": make_toggle_switch,
    "brusselator": make_brusselator,
    "selkov": make_selkov,
    "repressilator": make_repressilator,
    "goodwin": make_goodwin,
}


def get_benchmark(name: str) -> BenchmarkModel:
    if name not in ALL_BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(ALL_BENCHMARKS.keys())}")
    return ALL_BENCHMARKS[name]()


def list_benchmarks() -> List[str]:
    return list(ALL_BENCHMARKS.keys())
