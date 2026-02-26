"""STL robustness maximization via CMA-ES and hybrid optimization.

Provides a full CMA-ES implementation (not a wrapper) together with
L-BFGS-B local refinement and a hybrid global/local optimizer for
maximizing Signal Temporal Logic robustness of biological circuits.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize as scipy_minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration and result types
# ---------------------------------------------------------------------------

@dataclass
class CMAESConfig:
    """Configuration for the CMA-ES optimizer."""

    population_size: Optional[int] = None  # None => auto
    sigma0: float = 0.3
    max_generations: int = 500
    max_evaluations: int = 50_000
    tol_fun: float = 1e-11
    tol_x: float = 1e-12
    tol_condition: float = 1e14
    seed: Optional[int] = None


@dataclass
class OptimizationResult:
    """Result of a robustness optimization run."""

    best_params: np.ndarray
    best_robustness: float
    evaluations: int = 0
    generations: int = 0
    wall_time: float = 0.0
    converged: bool = False
    history: List[float] = field(default_factory=list)
    method: str = ""

    def summary(self) -> str:
        return (
            f"OptimizationResult({self.method}): "
            f"robustness={self.best_robustness:.6g}, "
            f"evals={self.evaluations}, gens={self.generations}, "
            f"time={self.wall_time:.2f}s, converged={self.converged}"
        )


# ---------------------------------------------------------------------------
# CMA-ES implementation
# ---------------------------------------------------------------------------

class CMAES:
    """Covariance Matrix Adaptation Evolution Strategy.

    A complete implementation following Hansen & Ostermeier (2001) with
    cumulative step-size adaptation (CSA) and covariance matrix
    adaptation with rank-one and rank-mu updates.
    """

    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], float],
        x0: np.ndarray,
        sigma0: float = 0.3,
        bounds: Optional[List[Tuple[float, float]]] = None,
        config: Optional[CMAESConfig] = None,
    ) -> None:
        self._fn = objective_fn
        self._config = config or CMAESConfig(sigma0=sigma0)
        self._dim = len(x0)
        self._bounds = bounds

        # --- strategy parameters (Hansen 2001 / tutorial) ---
        n = self._dim
        lam = self._config.population_size
        if lam is None:
            lam = 4 + int(3 * np.log(n))
        self._lambda = lam
        self._mu = lam // 2

        # Recombination weights (log-linear)
        raw_w = np.log(self._mu + 0.5) - np.log(np.arange(1, self._mu + 1))
        self._weights = raw_w / raw_w.sum()
        self._mu_eff = 1.0 / np.sum(self._weights ** 2)

        # Step-size adaptation (CSA)
        self._cs = (self._mu_eff + 2.0) / (n + self._mu_eff + 5.0)
        self._ds = 1.0 + 2.0 * max(0.0, np.sqrt((self._mu_eff - 1.0) / (n + 1.0)) - 1.0) + self._cs
        self._chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

        # Covariance adaptation
        self._cc = (4.0 + self._mu_eff / n) / (n + 4.0 + 2.0 * self._mu_eff / n)
        self._c1 = 2.0 / ((n + 1.3) ** 2 + self._mu_eff)
        alpha_mu = 2.0
        self._cmu = min(
            1.0 - self._c1,
            alpha_mu * (self._mu_eff - 2.0 + 1.0 / self._mu_eff)
            / ((n + 2.0) ** 2 + alpha_mu * self._mu_eff / 2.0),
        )

        # State
        self._mean = x0.astype(np.float64).copy()
        self._sigma = float(self._config.sigma0)
        self._C = np.eye(n)
        self._ps = np.zeros(n)  # Evolution path for sigma
        self._pc = np.zeros(n)  # Evolution path for C
        self._B = np.eye(n)
        self._D = np.ones(n)
        self._invsqrtC = np.eye(n)
        self._eigen_update_counter = 0

        self._generation = 0
        self._evaluations = 0
        self._best_x = x0.copy()
        self._best_f = np.inf
        self._history: List[float] = []

        self._rng = np.random.default_rng(self._config.seed)

    # -- eigen decomposition ------------------------------------------------

    def _update_eigensystem(self) -> None:
        """Decompose C = B * diag(D**2) * B^T and cache inverse sqrt."""
        self._C = np.triu(self._C) + np.triu(self._C, 1).T  # enforce symmetry
        eigvals, self._B = np.linalg.eigh(self._C)
        eigvals = np.maximum(eigvals, 1e-20)
        self._D = np.sqrt(eigvals)
        self._invsqrtC = self._B @ np.diag(1.0 / self._D) @ self._B.T

    # -- boundary handling --------------------------------------------------

    def _repair_bounds(self, x: np.ndarray) -> np.ndarray:
        """Reflection-based boundary handling."""
        if self._bounds is None:
            return x
        y = x.copy()
        for i, (lo, hi) in enumerate(self._bounds):
            if lo is None and hi is None:
                continue
            if lo is not None and hi is not None:
                span = hi - lo
                if span <= 0:
                    y[i] = lo
                    continue
                # Reflect into [lo, hi]
                while y[i] < lo or y[i] > hi:
                    if y[i] < lo:
                        y[i] = 2 * lo - y[i]
                    if y[i] > hi:
                        y[i] = 2 * hi - y[i]
                y[i] = np.clip(y[i], lo, hi)
            elif lo is not None:
                y[i] = max(y[i], lo)
            elif hi is not None:
                y[i] = min(y[i], hi)
        return y

    # -- sample population --------------------------------------------------

    def _sample_population(self) -> np.ndarray:
        """Sample lambda offspring from N(mean, sigma^2 * C)."""
        n = self._dim
        pop = np.empty((self._lambda, n))
        for k in range(self._lambda):
            z = self._rng.standard_normal(n)
            y = self._B @ (self._D * z)
            pop[k] = self._repair_bounds(self._mean + self._sigma * y)
        return pop

    # -- single generation --------------------------------------------------

    def step(self) -> float:
        """Run one CMA-ES generation.  Returns best fitness this generation."""
        n = self._dim
        self._generation += 1

        # --- sample and evaluate ---
        population = self._sample_population()
        fitness = np.array([self._fn(ind) for ind in population])
        self._evaluations += self._lambda

        # --- sort by fitness ---
        order = np.argsort(fitness)
        population = population[order]
        fitness = fitness[order]

        # Update best
        if fitness[0] < self._best_f:
            self._best_f = fitness[0]
            self._best_x = population[0].copy()
        self._history.append(self._best_f)

        # --- recombination ---
        old_mean = self._mean.copy()
        self._mean = np.zeros(n)
        for i in range(self._mu):
            self._mean += self._weights[i] * population[i]

        # --- CSA (cumulative step-size adaptation) ---
        mean_diff = (self._mean - old_mean) / self._sigma
        self._ps = (1.0 - self._cs) * self._ps + np.sqrt(
            self._cs * (2.0 - self._cs) * self._mu_eff
        ) * (self._invsqrtC @ mean_diff)

        # --- CMA (covariance matrix adaptation) ---
        h_sig = (
            np.linalg.norm(self._ps)
            / np.sqrt(1.0 - (1.0 - self._cs) ** (2 * self._generation))
            / self._chi_n
            < 1.4 + 2.0 / (n + 1.0)
        )
        self._pc = (1.0 - self._cc) * self._pc + h_sig * np.sqrt(
            self._cc * (2.0 - self._cc) * self._mu_eff
        ) * mean_diff

        # Rank-one update
        artmp = (population[:self._mu] - old_mean) / self._sigma
        self._C = (
            (1.0 - self._c1 - self._cmu + (1.0 - h_sig) * self._c1 * self._cc * (2.0 - self._cc))
            * self._C
            + self._c1 * np.outer(self._pc, self._pc)
            + self._cmu * (artmp.T @ np.diag(self._weights) @ artmp)
        )

        # --- step-size update ---
        self._sigma *= np.exp(
            (self._cs / self._ds)
            * (np.linalg.norm(self._ps) / self._chi_n - 1.0)
        )
        self._sigma = min(self._sigma, 1e10)  # cap

        # --- eigen decomposition (lazy) ---
        self._eigen_update_counter += 1
        if self._eigen_update_counter >= self._lambda / (self._c1 + self._cmu) / n / 10.0:
            self._eigen_update_counter = 0
            self._update_eigensystem()

        return fitness[0]

    # -- termination checks -------------------------------------------------

    def _should_stop(self) -> bool:
        cfg = self._config
        if self._evaluations >= cfg.max_evaluations:
            return True
        if self._generation >= cfg.max_generations:
            return True

        # Function value tolerance
        if len(self._history) > 20:
            recent = self._history[-20:]
            if max(recent) - min(recent) < cfg.tol_fun:
                return True

        # Step size tolerance
        if self._sigma * np.max(self._D) < cfg.tol_x:
            return True

        # Condition number
        if np.max(self._D) / max(np.min(self._D), 1e-20) > cfg.tol_condition:
            return True

        return False

    # -- optimize -----------------------------------------------------------

    def optimize(self) -> OptimizationResult:
        """Run CMA-ES to convergence.  Returns :class:`OptimizationResult`."""
        t0 = time.time()
        self._update_eigensystem()

        while not self._should_stop():
            self.step()

        converged = self._generation < self._config.max_generations
        return OptimizationResult(
            best_params=self._best_x.copy(),
            best_robustness=-self._best_f,  # negate: we minimize negative robustness
            evaluations=self._evaluations,
            generations=self._generation,
            wall_time=time.time() - t0,
            converged=converged,
            history=[-v for v in self._history],
            method="CMA-ES",
        )

    @property
    def best(self) -> Tuple[np.ndarray, float]:
        return self._best_x.copy(), self._best_f

    @property
    def generation(self) -> int:
        return self._generation


# ---------------------------------------------------------------------------
# Robustness optimizer (hybrid CMA-ES + L-BFGS-B)
# ---------------------------------------------------------------------------

class RobustnessOptimizer:
    """Hybrid optimizer for STL robustness maximization.

    Workflow:
    1. Multi-start CMA-ES for global exploration.
    2. L-BFGS-B for local refinement around the best CMA-ES solution.

    The objective function should return the (scalar) STL robustness
    ``rho(phi, simulate(params))``; the optimizer *maximizes* it.
    """

    def __init__(
        self,
        robustness_fn: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        constraint_fn: Optional[Callable[[np.ndarray], bool]] = None,
        cmaes_config: Optional[CMAESConfig] = None,
        n_restarts: int = 3,
        local_budget: int = 500,
        penalty_weight: float = 1e6,
    ) -> None:
        self._robustness_fn = robustness_fn
        self._bounds = bounds
        self._dim = len(bounds)
        self._constraint_fn = constraint_fn
        self._cmaes_config = cmaes_config or CMAESConfig()
        self._n_restarts = n_restarts
        self._local_budget = local_budget
        self._penalty_weight = penalty_weight
        self._history: List[OptimizationResult] = []

    # -- penalised objective ------------------------------------------------

    def _penalised_objective(self, params: np.ndarray) -> float:
        """Objective to *minimize* (negated robustness + constraint penalty)."""
        rob = self._robustness_fn(params)
        penalty = 0.0
        if self._constraint_fn is not None and not self._constraint_fn(params):
            penalty = self._penalty_weight
        return -rob + penalty

    # -- CMA-ES global search ----------------------------------------------

    def _run_cmaes(self, x0: np.ndarray) -> OptimizationResult:
        cma = CMAES(
            objective_fn=self._penalised_objective,
            x0=x0,
            bounds=self._bounds,
            config=self._cmaes_config,
        )
        return cma.optimize()

    # -- L-BFGS-B local refinement ------------------------------------------

    def _run_lbfgsb(self, x0: np.ndarray) -> OptimizationResult:
        t0 = time.time()
        evals = [0]

        def obj(x: np.ndarray) -> float:
            evals[0] += 1
            return self._penalised_objective(x)

        res = scipy_minimize(
            obj, x0, method="L-BFGS-B", bounds=self._bounds,
            options={"maxfun": self._local_budget, "ftol": 1e-12, "gtol": 1e-8},
        )
        return OptimizationResult(
            best_params=res.x.copy(),
            best_robustness=-res.fun,
            evaluations=evals[0],
            generations=0,
            wall_time=time.time() - t0,
            converged=res.success,
            history=[],
            method="L-BFGS-B",
        )

    # -- multi-start --------------------------------------------------------

    def _random_start(self) -> np.ndarray:
        rng = np.random.default_rng()
        return np.array([rng.uniform(lo, hi) for lo, hi in self._bounds])

    # -- main entry ---------------------------------------------------------

    def optimize(
        self,
        x0: Optional[np.ndarray] = None,
        timeout: float = 300.0,
    ) -> OptimizationResult:
        """Run the full hybrid optimization pipeline.

        Parameters
        ----------
        x0 : array, optional
            Initial guess.  If ``None``, a random start is used.
        timeout : float
            Wall-clock budget in seconds.

        Returns
        -------
        OptimizationResult
        """
        t0 = time.time()
        best_result: Optional[OptimizationResult] = None

        starts = []
        if x0 is not None:
            starts.append(x0)
        while len(starts) < self._n_restarts:
            starts.append(self._random_start())

        # Phase 1: multi-start CMA-ES
        for i, start in enumerate(starts):
            if time.time() - t0 > timeout:
                break
            logger.info("CMA-ES restart %d/%d", i + 1, self._n_restarts)
            result = self._run_cmaes(start)
            self._history.append(result)
            if best_result is None or result.best_robustness > best_result.best_robustness:
                best_result = result

        if best_result is None:
            best_result = OptimizationResult(
                best_params=starts[0],
                best_robustness=self._robustness_fn(starts[0]),
                method="none",
            )

        # Phase 2: local refinement
        if time.time() - t0 < timeout:
            logger.info("L-BFGS-B local refinement from CMA-ES best.")
            local_result = self._run_lbfgsb(best_result.best_params)
            self._history.append(local_result)
            if local_result.best_robustness > best_result.best_robustness:
                best_result = local_result

        best_result.wall_time = time.time() - t0
        best_result.method = "hybrid(CMA-ES+L-BFGS-B)"
        logger.info("Optimization complete: %s", best_result.summary())
        return best_result

    # -- convenience --------------------------------------------------------

    def optimize_cmaes_only(self, x0: Optional[np.ndarray] = None) -> OptimizationResult:
        """Run CMA-ES only (no local refinement)."""
        if x0 is None:
            x0 = self._random_start()
        return self._run_cmaes(x0)

    def optimize_lbfgsb_only(self, x0: Optional[np.ndarray] = None) -> OptimizationResult:
        """Run L-BFGS-B only (requires good starting point)."""
        if x0 is None:
            x0 = self._random_start()
        return self._run_lbfgsb(x0)

    @property
    def history(self) -> List[OptimizationResult]:
        return list(self._history)

    def convergence_data(self) -> Dict[str, Any]:
        """Aggregate convergence data across all restarts."""
        all_rob: List[float] = []
        for r in self._history:
            all_rob.extend(r.history)
        return {
            "restarts": len(self._history),
            "total_evaluations": sum(r.evaluations for r in self._history),
            "best_robustness": max((r.best_robustness for r in self._history), default=float("-inf")),
            "robustness_trace": all_rob,
        }
