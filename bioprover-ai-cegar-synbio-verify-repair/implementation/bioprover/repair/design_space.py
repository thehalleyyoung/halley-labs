"""Design space exploration, Pareto frontier, and sensitivity analysis.

Provides adaptive sampling of the parameter space, multi-objective
Pareto frontier computation (NSGA-II style), robustness landscape
mapping, and global sensitivity analysis (Morris, Sobol).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.stats import qmc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Design point
# ---------------------------------------------------------------------------

@dataclass
class DesignPoint:
    """A single evaluated point in the design space."""

    parameters: np.ndarray
    objectives: np.ndarray  # e.g. [robustness, -perturbation]
    feasible: bool = True
    rank: int = 0
    crowding_distance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def dominates(self, other: DesignPoint) -> bool:
        """True if *self* Pareto-dominates *other* (all objectives >=, at least one >)."""
        return bool(
            np.all(self.objectives >= other.objectives)
            and np.any(self.objectives > other.objectives)
        )


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

class ParetoFrontier:
    """Non-dominated sorting and crowding-distance computation (NSGA-II)."""

    def __init__(self) -> None:
        self._points: List[DesignPoint] = []
        self._fronts: List[List[int]] = []

    @property
    def points(self) -> List[DesignPoint]:
        return list(self._points)

    @property
    def front(self) -> List[DesignPoint]:
        """Return the rank-0 (best) Pareto front."""
        return [p for p in self._points if p.rank == 0]

    def add_points(self, points: Sequence[DesignPoint]) -> None:
        self._points.extend(points)
        self._compute()

    def _compute(self) -> None:
        """Run non-dominated sorting and assign crowding distance."""
        n = len(self._points)
        if n == 0:
            return
        self._fronts = self._non_dominated_sort()
        for rank, front_indices in enumerate(self._fronts):
            for idx in front_indices:
                self._points[idx].rank = rank
            self._assign_crowding(front_indices)

    def _non_dominated_sort(self) -> List[List[int]]:
        """Fast non-dominated sort (Deb et al. 2002)."""
        n = len(self._points)
        domination_count = np.zeros(n, dtype=int)
        dominated_set: List[List[int]] = [[] for _ in range(n)]
        fronts: List[List[int]] = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._points[i].dominates(self._points[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif self._points[j].dominates(self._points[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1

        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            next_front: List[int] = []
            for i in fronts[k]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            k += 1
            fronts.append(next_front)

        return [f for f in fronts if f]

    def _assign_crowding(self, front_indices: List[int]) -> None:
        """Assign crowding distance to points in a single front."""
        n = len(front_indices)
        if n <= 2:
            for idx in front_indices:
                self._points[idx].crowding_distance = float("inf")
            return

        n_obj = len(self._points[front_indices[0]].objectives)
        distances = np.zeros(n)

        for m in range(n_obj):
            vals = [(self._points[front_indices[i]].objectives[m], i) for i in range(n)]
            vals.sort(key=lambda x: x[0])
            sorted_local = [v[1] for v in vals]

            distances[sorted_local[0]] = float("inf")
            distances[sorted_local[-1]] = float("inf")

            obj_range = vals[-1][0] - vals[0][0]
            if obj_range < 1e-15:
                continue
            for k in range(1, n - 1):
                distances[sorted_local[k]] += (vals[k + 1][0] - vals[k - 1][0]) / obj_range

        for i in range(n):
            self._points[front_indices[i]].crowding_distance = distances[i]

    def to_array(self) -> np.ndarray:
        """Return objectives of the rank-0 front as an (n, m) array."""
        front = self.front
        if not front:
            return np.empty((0, 0))
        return np.array([p.objectives for p in front])

    def hypervolume(self, reference: np.ndarray) -> float:
        """Compute 2-D hypervolume indicator against *reference* point."""
        front = self.to_array()
        if front.shape[0] == 0 or front.shape[1] != 2:
            return 0.0
        # Sort by first objective descending
        order = np.argsort(-front[:, 0])
        front = front[order]
        hv = 0.0
        prev_y = reference[1]
        for pt in front:
            if pt[0] <= reference[0] or pt[1] <= reference[1]:
                continue
            hv += (pt[0] - reference[0]) * (prev_y - reference[1])
            prev_y = min(prev_y, pt[1])
        return hv


# ---------------------------------------------------------------------------
# Sensitivity analysis results
# ---------------------------------------------------------------------------

@dataclass
class SensitivityResult:
    """Results of a sensitivity analysis."""

    method: str
    parameter_names: List[str]
    indices: Dict[str, np.ndarray]  # e.g. {"mu": ..., "sigma": ..., "S1": ...}
    rankings: List[Tuple[str, float]] = field(default_factory=list)

    def most_sensitive(self, k: int = 5) -> List[Tuple[str, float]]:
        if self.rankings:
            return self.rankings[:k]
        return []

    def summary(self) -> str:
        lines = [f"Sensitivity analysis ({self.method}):"]
        for name, val in self.most_sensitive():
            lines.append(f"  {name}: {val:.4g}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Design space explorer
# ---------------------------------------------------------------------------

class DesignSpace:
    """Parameter space exploration with adaptive sampling and analysis.

    Parameters
    ----------
    bounds : list of (lo, hi)
        Bounds for each parameter dimension.
    names : list of str
        Parameter names.
    """

    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        names: Optional[List[str]] = None,
    ) -> None:
        self._bounds = bounds
        self._dim = len(bounds)
        self._names = names or [f"p{i}" for i in range(self._dim)]
        self._evaluated: List[DesignPoint] = []

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def names(self) -> List[str]:
        return list(self._names)

    @property
    def evaluated_points(self) -> List[DesignPoint]:
        return list(self._evaluated)

    @property
    def volume(self) -> float:
        v = 1.0
        for lo, hi in self._bounds:
            v *= max(hi - lo, 0.0)
        return v

    # -- grid-based exploration ---------------------------------------------

    def grid_sample(self, n_per_dim: int = 5) -> np.ndarray:
        """Generate a regular grid over the design space."""
        axes = [np.linspace(lo, hi, n_per_dim) for lo, hi in self._bounds]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.column_stack([m.ravel() for m in mesh])

    # -- Latin hypercube sampling -------------------------------------------

    def lhs_sample(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """Latin Hypercube Sample of *n* points."""
        sampler = qmc.LatinHypercube(d=self._dim, seed=seed)
        unit = sampler.random(n=n)
        lo = np.array([b[0] for b in self._bounds])
        hi = np.array([b[1] for b in self._bounds])
        return qmc.scale(unit, lo, hi)

    # -- adaptive sampling --------------------------------------------------

    def adaptive_sample(
        self,
        objective_fn: Callable[[np.ndarray], float],
        n_initial: int = 20,
        n_refine: int = 50,
        seed: Optional[int] = None,
    ) -> List[DesignPoint]:
        """Adaptively sample, concentrating on regions near decision boundaries.

        Phase 1: LHS initial samples.
        Phase 2: Iteratively add points near the boundary between
        positive and negative robustness regions.
        """
        rng = np.random.default_rng(seed)

        # Phase 1
        initial = self.lhs_sample(n_initial, seed=seed)
        points: List[DesignPoint] = []
        for x in initial:
            obj = objective_fn(x)
            dp = DesignPoint(parameters=x.copy(), objectives=np.array([obj]))
            points.append(dp)

        # Phase 2: refine near boundary
        for _ in range(n_refine):
            boundary_pts = self._find_boundary_pairs(points)
            if not boundary_pts:
                # Random fallback
                x_new = np.array([rng.uniform(lo, hi) for lo, hi in self._bounds])
            else:
                # Bisect a random boundary pair
                idx = rng.integers(len(boundary_pts))
                p1, p2 = boundary_pts[idx]
                alpha = rng.uniform(0.3, 0.7)
                x_new = alpha * p1.parameters + (1 - alpha) * p2.parameters

            obj = objective_fn(x_new)
            points.append(DesignPoint(parameters=x_new.copy(), objectives=np.array([obj])))

        self._evaluated.extend(points)
        return points

    def _find_boundary_pairs(
        self, points: List[DesignPoint]
    ) -> List[Tuple[DesignPoint, DesignPoint]]:
        """Find pairs of points with different sign of first objective."""
        pairs: List[Tuple[DesignPoint, DesignPoint]] = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if points[i].objectives[0] * points[j].objectives[0] < 0:
                    pairs.append((points[i], points[j]))
        return pairs[:50]  # cap

    # -- robustness landscape mapping ---------------------------------------

    def map_robustness(
        self,
        objective_fn: Callable[[np.ndarray], float],
        n_samples: int = 200,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample and return ``(parameters, robustness)`` arrays."""
        samples = self.lhs_sample(n_samples, seed=seed)
        rob = np.array([objective_fn(x) for x in samples])
        return samples, rob

    def interpolate_robustness(
        self,
        sample_params: np.ndarray,
        sample_rob: np.ndarray,
        query_params: np.ndarray,
    ) -> np.ndarray:
        """Interpolate robustness surface using RBF."""
        interp = RBFInterpolator(sample_params, sample_rob, kernel="thin_plate_spline")
        return interp(query_params)

    def identify_robust_regions(
        self,
        sample_params: np.ndarray,
        sample_rob: np.ndarray,
        threshold: float = 0.0,
    ) -> np.ndarray:
        """Return parameter samples where robustness > *threshold*."""
        mask = sample_rob > threshold
        return sample_params[mask]

    # -- Pareto exploration -------------------------------------------------

    def pareto_explore(
        self,
        objectives: List[Callable[[np.ndarray], float]],
        n_samples: int = 500,
        seed: Optional[int] = None,
    ) -> ParetoFrontier:
        """Evaluate multiple objectives and compute Pareto frontier."""
        samples = self.lhs_sample(n_samples, seed=seed)
        points: List[DesignPoint] = []
        for x in samples:
            objs = np.array([fn(x) for fn in objectives])
            points.append(DesignPoint(parameters=x.copy(), objectives=objs))

        frontier = ParetoFrontier()
        frontier.add_points(points)
        self._evaluated.extend(points)
        return frontier

    # -- sensitivity analysis -----------------------------------------------

    def morris_sensitivity(
        self,
        objective_fn: Callable[[np.ndarray], float],
        n_trajectories: int = 10,
        n_levels: int = 4,
        seed: Optional[int] = None,
    ) -> SensitivityResult:
        """Morris method (elementary effects) for screening parameters."""
        rng = np.random.default_rng(seed)
        lo = np.array([b[0] for b in self._bounds])
        hi = np.array([b[1] for b in self._bounds])
        delta = 1.0 / (n_levels - 1) if n_levels > 1 else 0.5

        effects: Dict[int, List[float]] = {i: [] for i in range(self._dim)}

        for _ in range(n_trajectories):
            # Generate base point on grid
            x_unit = rng.choice(np.linspace(0, 1 - delta, n_levels), size=self._dim)
            x = lo + x_unit * (hi - lo)
            f_base = objective_fn(x)

            # Perturb each dimension
            perm = rng.permutation(self._dim)
            for i in perm:
                x_new = x.copy()
                x_new[i] = x[i] + delta * (hi[i] - lo[i])
                if x_new[i] > hi[i]:
                    x_new[i] = x[i] - delta * (hi[i] - lo[i])
                f_new = objective_fn(x_new)
                ee = (f_new - f_base) / (delta * (hi[i] - lo[i]) + 1e-15)
                effects[i].append(ee)
                x = x_new
                f_base = f_new

        mu = np.array([np.mean(effects[i]) for i in range(self._dim)])
        mu_star = np.array([np.mean(np.abs(effects[i])) for i in range(self._dim)])
        sigma = np.array([np.std(effects[i]) for i in range(self._dim)])

        rankings = sorted(
            zip(self._names, mu_star.tolist()), key=lambda x: x[1], reverse=True
        )
        return SensitivityResult(
            method="Morris",
            parameter_names=list(self._names),
            indices={"mu": mu, "mu_star": mu_star, "sigma": sigma},
            rankings=rankings,
        )

    def sobol_sensitivity(
        self,
        objective_fn: Callable[[np.ndarray], float],
        n_samples: int = 1024,
        seed: Optional[int] = None,
    ) -> SensitivityResult:
        """Sobol sensitivity indices (first-order and total).

        Uses the Saltelli sampling scheme for efficient index estimation.
        """
        lo = np.array([b[0] for b in self._bounds])
        hi = np.array([b[1] for b in self._bounds])

        # Generate two independent sample matrices via Sobol sequence
        sampler = qmc.Sobol(d=self._dim, seed=seed)
        A_unit = sampler.random(n=n_samples)
        sampler2 = qmc.Sobol(d=self._dim, seed=(seed + 1) if seed else None)
        _ = sampler2.random()  # skip first
        B_unit = sampler2.random(n=n_samples)

        A = qmc.scale(A_unit, lo, hi)
        B = qmc.scale(B_unit, lo, hi)

        f_A = np.array([objective_fn(x) for x in A])
        f_B = np.array([objective_fn(x) for x in B])

        S1 = np.zeros(self._dim)
        ST = np.zeros(self._dim)
        var_total = np.var(np.concatenate([f_A, f_B]))
        if var_total < 1e-15:
            return SensitivityResult(
                method="Sobol",
                parameter_names=list(self._names),
                indices={"S1": S1, "ST": ST},
                rankings=list(zip(self._names, [0.0] * self._dim)),
            )

        for i in range(self._dim):
            # AB_i: A with column i from B
            AB_i = A.copy()
            AB_i[:, i] = B[:, i]
            f_AB_i = np.array([objective_fn(x) for x in AB_i])

            # First-order (Jansen 1999)
            S1[i] = np.mean(f_B * (f_AB_i - f_A)) / var_total
            # Total (Jansen 1999)
            ST[i] = 0.5 * np.mean((f_A - f_AB_i) ** 2) / var_total

        S1 = np.clip(S1, 0, 1)
        ST = np.clip(ST, 0, 1)

        rankings = sorted(
            zip(self._names, ST.tolist()), key=lambda x: x[1], reverse=True
        )
        return SensitivityResult(
            method="Sobol",
            parameter_names=list(self._names),
            indices={"S1": S1, "ST": ST},
            rankings=rankings,
        )

    def local_sensitivity(
        self,
        objective_fn: Callable[[np.ndarray], float],
        x0: np.ndarray,
        delta: float = 1e-4,
    ) -> SensitivityResult:
        """Local sensitivity via central finite differences at *x0*."""
        f0 = objective_fn(x0)
        grad = np.zeros(self._dim)
        for i in range(self._dim):
            x_fwd = x0.copy()
            x_bwd = x0.copy()
            h = max(abs(x0[i]) * delta, 1e-10)
            x_fwd[i] += h
            x_bwd[i] -= h
            grad[i] = (objective_fn(x_fwd) - objective_fn(x_bwd)) / (2 * h)

        rankings = sorted(
            zip(self._names, np.abs(grad).tolist()), key=lambda x: x[1], reverse=True
        )
        return SensitivityResult(
            method="local_fd",
            parameter_names=list(self._names),
            indices={"gradient": grad, "abs_gradient": np.abs(grad)},
            rankings=rankings,
        )

    # -- design point ranking -----------------------------------------------

    def rank_points(
        self,
        points: Optional[List[DesignPoint]] = None,
        weights: Optional[np.ndarray] = None,
    ) -> List[DesignPoint]:
        """Rank design points by weighted-sum of objectives (descending)."""
        pts = points or self._evaluated
        if not pts:
            return []
        n_obj = len(pts[0].objectives)
        if weights is None:
            weights = np.ones(n_obj) / n_obj
        scored = [(np.dot(p.objectives, weights), p) for p in pts]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored]
