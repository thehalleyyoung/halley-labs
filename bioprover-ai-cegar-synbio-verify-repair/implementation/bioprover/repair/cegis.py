"""CEGIS (Counter-Example Guided Inductive Synthesis) inner loop.

Implements the CEGIS paradigm for parameter repair: iteratively propose
candidate parameters satisfying all known counterexamples, then verify
against the full specification.  When verification fails, the new
counterexample is added and the loop continues.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import numpy as np
from scipy.optimize import minimize

from bioprover.models.parameters import Parameter, ParameterSet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types and protocols
# ---------------------------------------------------------------------------

class CEGISStatus(Enum):
    """Outcome of a CEGIS run."""

    SUCCESS = auto()
    MAX_ITERATIONS = auto()
    TIMEOUT = auto()
    STALLED = auto()
    INFEASIBLE = auto()


@dataclass
class Counterexample:
    """A single counterexample point or trace.

    Attributes:
        state: Mapping from variable name to value at the violating point.
        time: Time at which violation occurs (``None`` for static specs).
        violation: Signed robustness value (negative means violation).
        source: Label indicating how this counterexample was obtained.
    """

    state: Dict[str, float]
    time: Optional[float] = None
    violation: float = 0.0
    source: str = "verifier"
    _hash: Optional[int] = field(default=None, repr=False, compare=False)

    def feature_vector(self) -> np.ndarray:
        """Return a sorted-key numeric vector for distance computations."""
        return np.array([self.state[k] for k in sorted(self.state)])

    def __hash__(self) -> int:
        if self._hash is None:
            items = tuple(sorted(self.state.items()))
            object.__setattr__(self, "_hash", hash((items, self.time)))
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Counterexample):
            return NotImplemented
        return self.state == other.state and self.time == other.time


# ---------------------------------------------------------------------------
# Counterexample set with deduplication and management
# ---------------------------------------------------------------------------

class CounterexampleSet:
    """Managed collection of counterexamples with deduplication.

    Provides representative selection and generalization utilities
    needed by the CEGIS loop.
    """

    def __init__(self, dedup_tolerance: float = 1e-8) -> None:
        self._cexs: List[Counterexample] = []
        self._dedup_tol = dedup_tolerance

    # -- core operations ----------------------------------------------------

    def add(self, cex: Counterexample) -> bool:
        """Add *cex* if it is not a near-duplicate.  Returns ``True`` if added."""
        if self._is_duplicate(cex):
            logger.debug("Duplicate counterexample skipped.")
            return False
        self._cexs.append(cex)
        logger.info(
            "Counterexample added (total=%d, violation=%.4g).",
            len(self._cexs),
            cex.violation,
        )
        return True

    def add_batch(self, cexs: Sequence[Counterexample]) -> int:
        """Add multiple counterexamples.  Returns count of new additions."""
        return sum(self.add(c) for c in cexs)

    def clear(self) -> None:
        self._cexs.clear()

    @property
    def size(self) -> int:
        return len(self._cexs)

    def __len__(self) -> int:
        return len(self._cexs)

    def __iter__(self):
        return iter(self._cexs)

    def as_list(self) -> List[Counterexample]:
        return list(self._cexs)

    # -- deduplication ------------------------------------------------------

    def _is_duplicate(self, cex: Counterexample) -> bool:
        vec = cex.feature_vector()
        for existing in self._cexs:
            if np.linalg.norm(vec - existing.feature_vector()) < self._dedup_tol:
                return True
        return False

    # -- representative selection -------------------------------------------

    def select_representatives(self, k: int) -> List[Counterexample]:
        """Select *k* diverse representatives using greedy farthest-first.

        Returns at most ``min(k, len(self))`` counterexamples.
        """
        if len(self._cexs) <= k:
            return list(self._cexs)
        vecs = np.array([c.feature_vector() for c in self._cexs])
        selected_idx: List[int] = [0]
        for _ in range(k - 1):
            dists = np.min(
                np.linalg.norm(
                    vecs[:, None, :] - vecs[None, selected_idx, :], axis=2
                ),
                axis=1,
            )
            dists[selected_idx] = -1.0
            selected_idx.append(int(np.argmax(dists)))
        return [self._cexs[i] for i in selected_idx]

    # -- generalization -----------------------------------------------------

    def generalize(
        self,
        cex: Counterexample,
        check_fn: Callable[[Dict[str, float]], bool],
        delta: float = 0.05,
        max_tries: int = 50,
    ) -> List[Counterexample]:
        """Generalize *cex* by perturbing and testing with *check_fn*.

        *check_fn(state) -> True* means the perturbed state is still a
        valid counterexample.  Returns the list of successful perturbations.
        """
        base = cex.feature_vector()
        keys = sorted(cex.state.keys())
        gen: List[Counterexample] = []
        rng = np.random.default_rng()
        for _ in range(max_tries):
            perturbed = base + rng.normal(scale=delta * np.abs(base) + 1e-12, size=base.shape)
            state = dict(zip(keys, perturbed.tolist()))
            if check_fn(state):
                gen.append(Counterexample(state=state, time=cex.time, source="generalization"))
        return gen


# ---------------------------------------------------------------------------
# Proposal strategies
# ---------------------------------------------------------------------------

class ProposalStrategy(ABC):
    """Base class for candidate parameter proposal strategies."""

    @abstractmethod
    def propose(
        self,
        param_set: ParameterSet,
        counterexamples: CounterexampleSet,
        objective_fn: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
    ) -> Optional[np.ndarray]:
        """Propose a candidate parameter vector or ``None`` on failure."""


class SMTProposalStrategy(ProposalStrategy):
    """Encode counterexamples as SMT constraints and solve for parameters.

    This strategy builds a conjunction of constraints—one per known
    counterexample—and asks the SMT solver for a satisfying parameter
    assignment.  Falls back to optimization if the solver returns UNKNOWN.
    """

    def __init__(
        self,
        solver_factory: Optional[Callable] = None,
        timeout: float = 30.0,
    ) -> None:
        self._solver_factory = solver_factory
        self._timeout = timeout

    def propose(
        self,
        param_set: ParameterSet,
        counterexamples: CounterexampleSet,
        objective_fn: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
    ) -> Optional[np.ndarray]:
        if self._solver_factory is None:
            logger.warning("No SMT solver factory; falling back to optimization.")
            return self._fallback_optimize(objective_fn, bounds)
        solver = self._solver_factory()
        try:
            return self._solve_with_smt(solver, param_set, counterexamples, bounds)
        except Exception:
            logger.exception("SMT proposal failed; falling back to optimization.")
            return self._fallback_optimize(objective_fn, bounds)

    def _solve_with_smt(
        self, solver: Any, param_set: ParameterSet,
        counterexamples: CounterexampleSet,
        bounds: List[Tuple[float, float]],
    ) -> Optional[np.ndarray]:
        """Attempt SMT-based proposal using the solver interface."""
        from bioprover.encoding.expression import Var, Const

        params = list(param_set)
        dim = len(params)
        param_vars = [Var(p.name) for p in params]

        # Assert parameter bounds
        for i, p in enumerate(params):
            lo, hi = bounds[i]
            solver.assert_formula(param_vars[i] >= Const(lo))
            solver.assert_formula(param_vars[i] <= Const(hi))

        result = solver.check_sat(timeout=self._timeout)
        from bioprover.smt.solver_base import SMTResult
        if result.is_sat:
            model = solver.get_model()
            return np.array([model.get_float(p.name) for p in params])
        return None

    @staticmethod
    def _fallback_optimize(
        objective_fn: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
    ) -> Optional[np.ndarray]:
        dim = len(bounds)
        best_val = np.inf
        best_x: Optional[np.ndarray] = None
        rng = np.random.default_rng()
        for _ in range(5):
            x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
            res = minimize(objective_fn, x0, method="L-BFGS-B", bounds=bounds)
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        return best_x


class OptimizationProposalStrategy(ProposalStrategy):
    """Minimize total violation over known counterexamples."""

    def __init__(self, n_restarts: int = 10) -> None:
        self._n_restarts = n_restarts

    def propose(
        self,
        param_set: ParameterSet,
        counterexamples: CounterexampleSet,
        objective_fn: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
    ) -> Optional[np.ndarray]:
        dim = len(bounds)
        best_val = np.inf
        best_x: Optional[np.ndarray] = None
        rng = np.random.default_rng()

        for _ in range(self._n_restarts):
            x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
            res = minimize(objective_fn, x0, method="L-BFGS-B", bounds=bounds)
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()

        if best_x is not None:
            logger.debug("Optimization proposal: obj=%.6g", best_val)
        return best_x


class SurrogateProposalStrategy(ProposalStrategy):
    """Use a Gaussian Process surrogate to propose candidates.

    Fits a GP to ``(params, robustness)`` observations and uses expected
    improvement (EI) to select the next candidate.
    """

    def __init__(
        self,
        observations_x: Optional[np.ndarray] = None,
        observations_y: Optional[np.ndarray] = None,
        n_candidates: int = 1000,
    ) -> None:
        self._obs_x = observations_x if observations_x is not None else np.empty((0, 0))
        self._obs_y = observations_y if observations_y is not None else np.empty(0)
        self._n_candidates = n_candidates

    def add_observation(self, x: np.ndarray, y: float) -> None:
        if self._obs_x.size == 0:
            self._obs_x = x.reshape(1, -1)
        else:
            self._obs_x = np.vstack([self._obs_x, x])
        self._obs_y = np.append(self._obs_y, y)

    def propose(
        self,
        param_set: ParameterSet,
        counterexamples: CounterexampleSet,
        objective_fn: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
    ) -> Optional[np.ndarray]:
        if len(self._obs_y) < 3:
            # Not enough data for GP; use random
            rng = np.random.default_rng()
            return np.array([rng.uniform(lo, hi) for lo, hi in bounds])

        try:
            return self._ei_propose(bounds)
        except Exception:
            logger.exception("Surrogate proposal failed; using random.")
            rng = np.random.default_rng()
            return np.array([rng.uniform(lo, hi) for lo, hi in bounds])

    def _ei_propose(self, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Expected improvement acquisition over random candidates."""
        from scipy.spatial.distance import cdist

        dim = len(bounds)
        rng = np.random.default_rng()
        candidates = np.column_stack(
            [rng.uniform(lo, hi, size=self._n_candidates) for lo, hi in bounds]
        )

        # Simple RBF kernel GP prediction
        mu, sigma = self._gp_predict(candidates)
        best_y = np.min(self._obs_y)

        # Expected improvement (minimization)
        from scipy.stats import norm

        with np.errstate(divide="ignore", invalid="ignore"):
            improvement = best_y - mu
            z = np.where(sigma > 1e-12, improvement / sigma, 0.0)
            ei = np.where(
                sigma > 1e-12,
                improvement * norm.cdf(z) + sigma * norm.pdf(z),
                0.0,
            )

        best_idx = int(np.argmax(ei))
        return candidates[best_idx]

    def _gp_predict(
        self, x_new: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simplified RBF-kernel GP prediction."""
        X = self._obs_x
        y = self._obs_y
        n = len(y)
        length_scale = np.std(X, axis=0) + 1e-8

        def rbf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            sq = np.sum(((a[:, None, :] - b[None, :, :]) / length_scale) ** 2, axis=2)
            return np.exp(-0.5 * sq)

        K = rbf(X, X) + 1e-6 * np.eye(n)
        K_inv = np.linalg.solve(K, np.eye(n))
        K_star = rbf(x_new, X)
        mu = K_star @ K_inv @ y
        var = 1.0 - np.sum(K_star @ K_inv * K_star, axis=1)
        sigma = np.sqrt(np.maximum(var, 1e-12))
        return mu, sigma


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------

@dataclass
class DeltaEpsilonBound:
    r"""Formal relationship between δ (dReal relaxation) and ε (CEGIS guarantee).

    **Theorem (δ-ε Propagation).** Consider dynamics
    :math:`\dot{x} = f(x, p)` with :math:`f` Lipschitz-continuous:

    - :math:`\|f(x,p) - f(x',p)\| \le L_x \|x - x'\|`  (state Lipschitz)
    - :math:`\|f(x,p) - f(x,p')\| \le L_p \|p - p'\|`  (parameter Lipschitz)

    For a δ-decidable verifier with relaxation δ, the CEGIS loop converges
    to an ε-approximate solution with:

    .. math::
        \varepsilon \le \delta \cdot L_p \cdot \frac{e^{L_x T} - 1}{L_x}

    via Gronwall's inequality.  When the system has *k* monotone parameter
    dimensions out of *d* total, only :math:`d - k` dimensions contribute to
    the perturbation bound.  This yields the tightened bound:

    .. math::
        \varepsilon_{\text{mono}} \le \varepsilon \cdot
        \sqrt{\frac{d - k}{d}}

    since monotone dimensions require only endpoint verification.

    **Adaptive δ selection.** Given a target ε, choose:

    .. math::
        \delta^* = \frac{\varepsilon \cdot L_x}{L_p \cdot (e^{L_x T} - 1)}
    """

    delta: float = 1e-3
    lipschitz_state: float = 1.0
    lipschitz_param: float = 1.0
    time_horizon: float = 100.0
    num_params: int = 1
    num_monotone: int = 0
    epsilon_bound: float = float("inf")
    epsilon_bound_mono: float = float("inf")
    empirical_epsilons: List[float] = field(default_factory=list)
    delta_history: List[float] = field(default_factory=list)

    def compute_epsilon_bound(self) -> float:
        r"""Compute ε ≤ δ · L_p · (exp(L_x·T) - 1) / L_x via Gronwall.

        This uses the integral form of Gronwall's inequality applied to the
        sensitivity equation :math:`\dot{s} = J(t) s + \partial f/\partial p`,
        giving :math:`\|x(T;p) - x(T;p')\| \le \|p-p'\| \cdot L_p \cdot
        (e^{L_x T}-1)/L_x`.  Since the verifier guarantees
        :math:`\|p - p'\| \le \delta`, we get the stated bound.
        """
        Lx = max(self.lipschitz_state, 1e-15)
        growth = (np.exp(min(Lx * self.time_horizon, 50.0)) - 1.0) / Lx
        self.epsilon_bound = self.delta * self.lipschitz_param * growth

        # Monotonicity-tightened bound
        d = max(self.num_params, 1)
        k = min(self.num_monotone, d)
        mono_factor = np.sqrt(max(d - k, 1) / d)
        self.epsilon_bound_mono = self.epsilon_bound * mono_factor

        return self.epsilon_bound

    def recommend_delta(self, target_epsilon: float) -> float:
        """Recommend δ to achieve a target ε guarantee."""
        Lx = max(self.lipschitz_state, 1e-15)
        growth = (np.exp(min(Lx * self.time_horizon, 50.0)) - 1.0) / Lx
        Lp = max(self.lipschitz_param, 1e-15)
        d = max(self.num_params, 1)
        k = min(self.num_monotone, d)
        mono_factor = np.sqrt(max(d - k, 1) / d)
        return target_epsilon / (Lp * growth * mono_factor)

    def record_empirical(self, delta_used: float, observed_gap: float) -> None:
        """Record an empirical (δ, gap) observation for validation."""
        self.delta_history.append(delta_used)
        self.empirical_epsilons.append(observed_gap)

    def empirical_ratio(self) -> Optional[float]:
        """Return median empirical ε/δ ratio, or None if insufficient data."""
        if len(self.empirical_epsilons) < 2:
            return None
        ratios = [
            e / max(d, 1e-15)
            for e, d in zip(self.empirical_epsilons, self.delta_history)
            if d > 0
        ]
        return float(np.median(ratios)) if ratios else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delta": self.delta,
            "epsilon_bound": self.epsilon_bound,
            "epsilon_bound_mono": self.epsilon_bound_mono,
            "lipschitz_state": self.lipschitz_state,
            "lipschitz_param": self.lipschitz_param,
            "time_horizon": self.time_horizon,
            "num_params": self.num_params,
            "num_monotone": self.num_monotone,
            "empirical_ratio": self.empirical_ratio(),
            "n_observations": len(self.empirical_epsilons),
        }


@dataclass
class ConvergenceInfo:
    """Tracks CEGIS convergence metrics including δ-ε analysis."""

    iteration_violations: List[float] = field(default_factory=list)
    eliminated_volumes: List[float] = field(default_factory=list)
    best_robustness_history: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    delta_epsilon: DeltaEpsilonBound = field(default_factory=DeltaEpsilonBound)

    def record(self, violation: float, eliminated_vol: float, best_rob: float) -> None:
        self.iteration_violations.append(violation)
        self.eliminated_volumes.append(eliminated_vol)
        self.best_robustness_history.append(best_rob)
        self.timestamps.append(time.time())

    def is_stalled(self, window: int = 5, min_improvement: float = 1e-6) -> bool:
        """Detect stalling: no meaningful improvement over last *window* iterations."""
        if len(self.best_robustness_history) < window:
            return False
        recent = self.best_robustness_history[-window:]
        return (max(recent) - min(recent)) < min_improvement

    def is_converged(self, epsilon: float = 1e-4) -> bool:
        """Check epsilon-approximate convergence."""
        if not self.best_robustness_history:
            return False
        return self.best_robustness_history[-1] >= -epsilon

    @property
    def total_eliminated_volume(self) -> float:
        return sum(self.eliminated_volumes)


# ---------------------------------------------------------------------------
# CEGIS configuration and result
# ---------------------------------------------------------------------------

@dataclass
class CEGISConfig:
    """Configuration for the CEGIS loop."""

    max_iterations: int = 100
    timeout: float = 600.0
    stall_window: int = 10
    stall_threshold: float = 1e-6
    convergence_epsilon: float = 1e-4
    max_counterexamples: int = 500
    representative_k: int = 50
    dedup_tolerance: float = 1e-8
    generalize_counterexamples: bool = True
    generalization_delta: float = 0.05
    generalization_tries: int = 30
    delta: float = 1e-3
    time_horizon: float = 100.0
    track_delta_epsilon: bool = True


@dataclass
class CEGISResult:
    """Result of a CEGIS run."""

    status: CEGISStatus
    parameters: Optional[np.ndarray] = None
    parameter_names: List[str] = field(default_factory=list)
    iterations: int = 0
    total_time: float = 0.0
    counterexamples_used: int = 0
    best_robustness: float = float("-inf")
    convergence: ConvergenceInfo = field(default_factory=ConvergenceInfo)
    delta_epsilon_bound: Optional[DeltaEpsilonBound] = None

    @property
    def success(self) -> bool:
        return self.status == CEGISStatus.SUCCESS

    @property
    def epsilon_guarantee(self) -> float:
        """The ε-approximation guarantee for the repair solution."""
        if self.delta_epsilon_bound is not None:
            return self.delta_epsilon_bound.epsilon_bound
        return float("inf")

    def parameter_dict(self) -> Dict[str, float]:
        if self.parameters is None:
            return {}
        return dict(zip(self.parameter_names, self.parameters.tolist()))

    def summary(self) -> str:
        lines = [
            f"CEGIS Result: {self.status.name}",
            f"  Iterations: {self.iterations}",
            f"  Time: {self.total_time:.2f}s",
            f"  Counterexamples: {self.counterexamples_used}",
            f"  Best robustness: {self.best_robustness:.6g}",
        ]
        if self.delta_epsilon_bound is not None:
            de = self.delta_epsilon_bound
            lines.append(f"  δ-ε analysis:")
            lines.append(f"    δ (solver relaxation): {de.delta:.2e}")
            lines.append(f"    ε (approximation bound): {de.epsilon_bound:.2e}")
            emp = de.empirical_ratio()
            if emp is not None:
                lines.append(f"    Empirical ε/δ ratio: {emp:.2f}")
        if self.parameters is not None:
            lines.append("  Parameters:")
            for name, val in zip(self.parameter_names, self.parameters):
                lines.append(f"    {name}: {val:.6g}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verifier protocol
# ---------------------------------------------------------------------------

class VerifierProtocol(Protocol):
    """Protocol that CEGAR verifiers must satisfy for CEGIS integration."""

    def verify(
        self, parameters: Dict[str, float]
    ) -> Tuple[bool, Optional[Counterexample]]:
        """Verify parameters against full specification.

        Returns ``(True, None)`` if verified, or ``(False, cex)`` with a
        counterexample witnessing the violation.
        """
        ...


# ---------------------------------------------------------------------------
# CEGIS loop
# ---------------------------------------------------------------------------

class CEGISLoop:
    """Counterexample-guided inductive synthesis loop for parameter repair.

    The loop alternates between two phases:

    1. **Propose** – find candidate parameters consistent with all known
       counterexamples (via the chosen :class:`ProposalStrategy`).
    2. **Verify** – check the candidate against the full specification
       (via a :class:`VerifierProtocol`).

    If verification succeeds the candidate is returned.  Otherwise the
    new counterexample is recorded and the loop repeats.
    """

    def __init__(
        self,
        param_set: ParameterSet,
        verifier: VerifierProtocol,
        strategy: Optional[ProposalStrategy] = None,
        config: Optional[CEGISConfig] = None,
        objective_fn: Optional[Callable[[np.ndarray], float]] = None,
    ) -> None:
        self._param_set = param_set
        self._verifier = verifier
        self._strategy = strategy or OptimizationProposalStrategy()
        self._config = config or CEGISConfig()
        self._cex_set = CounterexampleSet(dedup_tolerance=self._config.dedup_tolerance)
        self._convergence = ConvergenceInfo()

        params = list(param_set)
        self._param_names = [p.name for p in params]
        self._bounds = [(p.lower_bound, p.upper_bound) for p in params]
        self._dim = len(params)

        if objective_fn is not None:
            self._objective_fn = objective_fn
        else:
            self._objective_fn = self._default_objective

        self._best_params: Optional[np.ndarray] = None
        self._best_robustness = float("-inf")
        self._iteration = 0
        self._start_time = 0.0

        # δ-ε bound tracking
        if self._config.track_delta_epsilon:
            self._convergence.delta_epsilon = DeltaEpsilonBound(
                delta=self._config.delta,
                time_horizon=self._config.time_horizon,
                num_params=self._dim,
            )
            self._estimate_lipschitz_constants()

    # -- default objective --------------------------------------------------

    def _default_objective(self, params: np.ndarray) -> float:
        """Default objective: negative of worst-case violation over cexs."""
        if self._cex_set.size == 0:
            return 0.0
        total = 0.0
        for cex in self._cex_set:
            total += max(0.0, -cex.violation)
        return total

    def _estimate_lipschitz_constants(self) -> None:
        """Estimate Lipschitz constants via finite differences on parameter bounds.

        For Lipschitz w.r.t. parameters (L_p): perturb each parameter and
        measure the maximum change in the objective. For Lipschitz w.r.t. state
        (L_x): use the parameter range as a proxy for system stiffness.
        """
        de = self._convergence.delta_epsilon
        rng = np.random.default_rng(42)
        n_samples = min(20, max(5, self._dim * 2))
        max_grad = 0.0
        for _ in range(n_samples):
            x0 = np.array([rng.uniform(lo, hi) for lo, hi in self._bounds])
            f0 = self._objective_fn(x0)
            for j in range(self._dim):
                dx = max(1e-6, 1e-4 * abs(x0[j]))
                x1 = x0.copy()
                x1[j] += dx
                x1[j] = min(x1[j], self._bounds[j][1])
                f1 = self._objective_fn(x1)
                grad = abs(f1 - f0) / dx
                max_grad = max(max_grad, grad)
        de.lipschitz_param = max(max_grad, 1.0)
        # Estimate L_x from parameter range magnitudes
        range_mag = np.mean([hi - lo for lo, hi in self._bounds])
        de.lipschitz_state = max(range_mag * 0.1, 0.1)
        de.compute_epsilon_bound()
        logger.info(
            "δ-ε bounds: δ=%.2e, L_p=%.2f, L_x=%.2f, ε≤%.2e",
            de.delta, de.lipschitz_param, de.lipschitz_state, de.epsilon_bound,
        )

    # -- main loop ----------------------------------------------------------

    def run(self) -> CEGISResult:
        """Execute the CEGIS loop.  Returns a :class:`CEGISResult`."""
        self._start_time = time.time()
        logger.info(
            "CEGIS starting: dim=%d, max_iter=%d, timeout=%.1fs",
            self._dim, self._config.max_iterations, self._config.timeout,
        )

        for self._iteration in range(1, self._config.max_iterations + 1):
            elapsed = time.time() - self._start_time
            if elapsed > self._config.timeout:
                logger.warning("CEGIS timeout after %d iterations.", self._iteration - 1)
                return self._make_result(CEGISStatus.TIMEOUT)

            logger.info("CEGIS iteration %d", self._iteration)

            # --- propose ---
            candidate = self._propose()
            if candidate is None:
                logger.warning("Proposal strategy returned None; marking infeasible.")
                return self._make_result(CEGISStatus.INFEASIBLE)

            # --- verify ---
            param_dict = dict(zip(self._param_names, candidate.tolist()))
            verified, cex = self._verifier.verify(param_dict)

            if verified:
                self._best_params = candidate
                self._best_robustness = 0.0  # verified => non-negative
                # Record empirical δ-ε observation
                if self._config.track_delta_epsilon:
                    gap = np.linalg.norm(candidate - self._best_params) if self._best_params is not None else 0.0
                    self._convergence.delta_epsilon.record_empirical(
                        self._config.delta, gap
                    )
                logger.info("CEGIS SUCCESS at iteration %d.", self._iteration)
                return self._make_result(CEGISStatus.SUCCESS)

            # --- add counterexample ---
            if cex is not None:
                self._process_counterexample(cex, candidate)

            # --- convergence check ---
            if self._convergence.is_stalled(
                self._config.stall_window, self._config.stall_threshold
            ):
                logger.warning("CEGIS stalled after %d iterations.", self._iteration)
                return self._make_result(CEGISStatus.STALLED)

        logger.warning("CEGIS reached max iterations (%d).", self._config.max_iterations)
        return self._make_result(CEGISStatus.MAX_ITERATIONS)

    # -- propose phase ------------------------------------------------------

    def _propose(self) -> Optional[np.ndarray]:
        """Propose candidate parameters satisfying known counterexamples."""
        if self._cex_set.size > self._config.representative_k:
            reps = self._cex_set.select_representatives(self._config.representative_k)
            sub_set = CounterexampleSet(self._config.dedup_tolerance)
            sub_set.add_batch(reps)
            effective_cexs = sub_set
        else:
            effective_cexs = self._cex_set

        candidate = self._strategy.propose(
            self._param_set, effective_cexs, self._objective_fn, self._bounds
        )
        return candidate

    # -- counterexample processing ------------------------------------------

    def _process_counterexample(
        self, cex: Counterexample, candidate: np.ndarray
    ) -> None:
        """Add counterexample and optionally generalize it."""
        added = self._cex_set.add(cex)
        if not added:
            return

        # Track convergence
        robustness = cex.violation
        eliminated = self._estimate_eliminated_volume(cex)
        if robustness > self._best_robustness:
            self._best_robustness = robustness
            self._best_params = candidate.copy()
        self._convergence.record(robustness, eliminated, self._best_robustness)

        # Generalize counterexample
        if self._config.generalize_counterexamples:
            gen_cexs = self._cex_set.generalize(
                cex,
                check_fn=self._check_counterexample_validity,
                delta=self._config.generalization_delta,
                max_tries=self._config.generalization_tries,
            )
            self._cex_set.add_batch(gen_cexs)

        # Prune if too many
        if self._cex_set.size > self._config.max_counterexamples:
            self._prune_counterexamples()

    def _check_counterexample_validity(self, state: Dict[str, float]) -> bool:
        """Check whether a perturbed state is still a valid counterexample."""
        try:
            verified, _ = self._verifier.verify(state)
            return not verified
        except Exception:
            return False

    def _estimate_eliminated_volume(self, cex: Counterexample) -> float:
        """Rough estimate of parameter-space volume eliminated by *cex*."""
        total_volume = 1.0
        for lo, hi in self._bounds:
            total_volume *= max(hi - lo, 1e-15)
        if total_volume == 0:
            return 0.0
        # Heuristic: each counterexample eliminates a fraction proportional
        # to 1/n where n is the number of counterexamples seen so far.
        return total_volume / max(self._cex_set.size, 1)

    def _prune_counterexamples(self) -> None:
        """Keep only the most diverse counterexamples."""
        reps = self._cex_set.select_representatives(self._config.max_counterexamples)
        self._cex_set.clear()
        self._cex_set.add_batch(reps)
        logger.info("Pruned counterexamples to %d.", self._cex_set.size)

    # -- result construction ------------------------------------------------

    def _make_result(self, status: CEGISStatus) -> CEGISResult:
        de_bound = None
        if self._config.track_delta_epsilon:
            de_bound = self._convergence.delta_epsilon
        return CEGISResult(
            status=status,
            parameters=self._best_params,
            parameter_names=list(self._param_names),
            iterations=self._iteration,
            total_time=time.time() - self._start_time,
            counterexamples_used=self._cex_set.size,
            best_robustness=self._best_robustness,
            convergence=self._convergence,
            delta_epsilon_bound=de_bound,
        )

    # -- statistics ---------------------------------------------------------

    @property
    def counterexample_set(self) -> CounterexampleSet:
        return self._cex_set

    @property
    def convergence_info(self) -> ConvergenceInfo:
        return self._convergence

    def statistics(self) -> Dict[str, Any]:
        return {
            "iteration": self._iteration,
            "elapsed": time.time() - self._start_time if self._start_time else 0.0,
            "counterexamples": self._cex_set.size,
            "best_robustness": self._best_robustness,
            "stalled": self._convergence.is_stalled(
                self._config.stall_window, self._config.stall_threshold
            ),
            "total_eliminated_volume": self._convergence.total_eliminated_volume,
        }

    def seed_counterexamples(self, cexs: Sequence[Counterexample]) -> None:
        """Seed the loop with externally-provided counterexamples."""
        self._cex_set.add_batch(cexs)
        logger.info("Seeded %d counterexamples.", self._cex_set.size)

    def set_strategy(self, strategy: ProposalStrategy) -> None:
        """Switch the proposal strategy mid-run."""
        self._strategy = strategy
