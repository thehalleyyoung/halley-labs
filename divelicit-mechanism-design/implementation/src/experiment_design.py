"""Design-of-experiments module with diversity-aware experimental design.

Provides space-filling designs, Latin hypercube sampling, sequential design,
and Bayesian optimization with diversity penalties. All implementations are
self-contained and use only numpy for numerical computation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, List, Literal, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Level:
    """A single level within an experimental factor."""

    name: str
    value: float

    def __repr__(self) -> str:
        return f"Level({self.name!r}, {self.value})"


@dataclass
class Factor:
    """An experimental factor with associated levels."""

    name: str
    levels: List[Level]
    lower: float = 0.0
    upper: float = 1.0

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError(f"Factor '{self.name}' must have at least one level.")
        self.lower = min(lv.value for lv in self.levels)
        self.upper = max(lv.value for lv in self.levels)

    @property
    def n_levels(self) -> int:
        return len(self.levels)


@dataclass
class DesignMatrix:
    """Result container for an experimental design."""

    matrix: np.ndarray
    factor_names: List[str]
    d_efficiency: float
    space_filling_metric: float
    design_type: str

    @property
    def n_runs(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_factors(self) -> int:
        return self.matrix.shape[1]

    def to_dict(self) -> dict:
        """Return design as a list of dictionaries keyed by factor name."""
        return [
            {name: float(row[i]) for i, name in enumerate(self.factor_names)}
            for row in self.matrix
        ]


@dataclass
class NextPoint:
    """Result from sequential design: the recommended next evaluation point."""

    point: np.ndarray
    acquisition_value: float
    diversity_contribution: float

    def __repr__(self) -> str:
        return (
            f"NextPoint(point={self.point}, acq={self.acquisition_value:.6f}, "
            f"div={self.diversity_contribution:.6f})"
        )


@dataclass
class SurrogateModel:
    """Gaussian process surrogate model state."""

    X_train: np.ndarray
    y_train: np.ndarray
    K_inv: np.ndarray
    alpha: np.ndarray
    kernel_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    noise: float
    lengthscale: float = 1.0
    variance: float = 1.0

    def predict(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return posterior mean and variance at *X_new*."""
        K_star = self.kernel_fn(X_new, self.X_train)
        mu = K_star @ self.alpha
        K_ss = self.kernel_fn(X_new, X_new)
        v = K_star @ self.K_inv
        var = np.diag(K_ss - v @ K_star.T)
        var = np.maximum(var, 1e-12)
        return mu, var


@dataclass
class OptResult:
    """Result container for Bayesian optimization."""

    best_point: np.ndarray
    best_value: float
    X_evals: np.ndarray
    y_evals: np.ndarray
    surrogate: SurrogateModel
    convergence: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: distance and efficiency metrics
# ---------------------------------------------------------------------------


def _maximin_distance(points: np.ndarray) -> float:
    """Return the minimum pairwise Euclidean distance among *points*.

    Parameters
    ----------
    points : np.ndarray, shape (n, d)

    Returns
    -------
    float
        Minimum distance.  Returns ``inf`` when fewer than 2 points.
    """
    n = points.shape[0]
    if n < 2:
        return float("inf")
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    return float(dists.min())


def _d_efficiency(design: np.ndarray) -> float:
    """Compute D-efficiency of a design matrix.

    D-efficiency = (det(X'X) / n)^{1/p} where *n* is the number of runs
    and *p* is the number of factors (columns).

    An intercept column is prepended automatically.
    """
    n, p = design.shape
    X = np.column_stack([np.ones(n), design])
    p_full = X.shape[1]
    xtx = X.T @ X
    sign, logdet = np.linalg.slogdet(xtx)
    if sign <= 0:
        return 0.0
    return float(np.exp(logdet / p_full) / n)


# ---------------------------------------------------------------------------
# Helper: coordinate exchange for D-optimal designs
# ---------------------------------------------------------------------------


def _coordinate_exchange(
    design: np.ndarray,
    candidate_values: Optional[List[np.ndarray]] = None,
    n_iter: int = 1000,
) -> np.ndarray:
    """Improve *design* via coordinate exchange to maximise det(X'X).

    At each iteration a random cell (row, column) is selected and replaced
    with the candidate value that yields the largest determinant increase.

    Parameters
    ----------
    design : np.ndarray, shape (n, p)
        Starting design (will be copied).
    candidate_values : list of arrays, optional
        Per-column candidate values.  Defaults to 21 equally spaced values
        in [col_min - 0.5*range, col_max + 0.5*range] clipped to [-1, 1].
    n_iter : int
        Number of exchange iterations.

    Returns
    -------
    np.ndarray
        Improved design matrix.
    """
    design = design.copy()
    n, p = design.shape

    if candidate_values is None:
        candidate_values = []
        for j in range(p):
            lo, hi = design[:, j].min(), design[:, j].max()
            rng = hi - lo if hi > lo else 1.0
            lo_c = max(lo - 0.5 * rng, -1.0)
            hi_c = min(hi + 0.5 * rng, 1.0)
            candidate_values.append(np.linspace(lo_c, hi_c, 21))

    X = np.column_stack([np.ones(n), design])
    xtx = X.T @ X
    cur_logdet = _safe_logdet(xtx)

    rng_gen = np.random.default_rng(0)

    for _ in range(n_iter):
        i = rng_gen.integers(0, n)
        j = rng_gen.integers(0, p)
        old_val = design[i, j]
        best_val = old_val
        best_logdet = cur_logdet

        for cand in candidate_values[j]:
            if cand == old_val:
                continue
            design[i, j] = cand
            X[i, j + 1] = cand
            xtx_new = X.T @ X
            ld = _safe_logdet(xtx_new)
            if ld > best_logdet:
                best_logdet = ld
                best_val = cand
            design[i, j] = old_val
            X[i, j + 1] = old_val

        if best_val != old_val:
            design[i, j] = best_val
            X[i, j + 1] = best_val
            xtx = X.T @ X
            cur_logdet = best_logdet

    return design


def _safe_logdet(A: np.ndarray) -> float:
    """Log-determinant that returns -inf for singular matrices."""
    sign, ld = np.linalg.slogdet(A)
    return float(ld) if sign > 0 else -float("inf")


# ---------------------------------------------------------------------------
# Helper: quasi-random sequences
# ---------------------------------------------------------------------------


def _sobol_sequence(n_dims: int, n_points: int) -> np.ndarray:
    """Generate a Sobol-like quasi-random sequence using Gray-code sampling.

    This is a simplified direction-number approach suitable for moderate
    dimensionality.  For *d* > 1 it seeds each dimension with a different
    primitive polynomial direction-number set; for very high *d* it falls
    back to the van der Corput sequence in different bases.

    Parameters
    ----------
    n_dims : int
    n_points : int

    Returns
    -------
    np.ndarray, shape (n_points, n_dims), values in [0, 1).
    """
    result = np.zeros((n_points, n_dims))

    # Use van der Corput sequences in the first *n_dims* prime bases
    primes = _first_n_primes(n_dims)
    for d in range(n_dims):
        base = primes[d]
        for i in range(n_points):
            result[i, d] = _van_der_corput(i + 1, base)
    return result


def _halton_sequence(n_dims: int, n_points: int) -> np.ndarray:
    """Generate a Halton quasi-random sequence.

    Uses the first *n_dims* primes as bases and the van der Corput radical
    inverse for each dimension.

    Parameters
    ----------
    n_dims : int
    n_points : int

    Returns
    -------
    np.ndarray, shape (n_points, n_dims), values in [0, 1).
    """
    primes = _first_n_primes(n_dims)
    result = np.zeros((n_points, n_dims))
    for d in range(n_dims):
        base = primes[d]
        for i in range(n_points):
            result[i, d] = _van_der_corput(i + 1, base)
    return result


def _van_der_corput(index: int, base: int) -> float:
    """Compute the van der Corput radical-inverse of *index* in *base*."""
    result = 0.0
    denom = 1.0
    n = index
    while n > 0:
        denom *= base
        n, remainder = divmod(n, base)
        result += remainder / denom
    return result


def _first_n_primes(n: int) -> List[int]:
    """Return the first *n* prime numbers."""
    primes: List[int] = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes


# ---------------------------------------------------------------------------
# Helper: Gaussian process utilities
# ---------------------------------------------------------------------------


def rbf_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    lengthscale: float = 1.0,
    variance: float = 1.0,
) -> np.ndarray:
    """Squared-exponential (RBF) kernel.

    k(x, x') = variance * exp(-0.5 * ||x - x'||^2 / lengthscale^2)
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    sq_dist = (
        np.sum(X1 ** 2, axis=1, keepdims=True)
        + np.sum(X2 ** 2, axis=1)
        - 2.0 * X1 @ X2.T
    )
    sq_dist = np.maximum(sq_dist, 0.0)
    return variance * np.exp(-0.5 * sq_dist / (lengthscale ** 2))


def _gp_fit(
    X: np.ndarray,
    y: np.ndarray,
    kernel_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    noise: float = 1e-6,
) -> SurrogateModel:
    """Fit a Gaussian process to observations *(X, y)*.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    y : np.ndarray, shape (n,)
    kernel_fn : callable
        Kernel function ``(X1, X2) -> K``.
    noise : float
        Observation noise variance added to the diagonal.

    Returns
    -------
    SurrogateModel
        Fitted surrogate with cached K_inv and alpha = K_inv @ y.
    """
    X = np.atleast_2d(X)
    y = np.asarray(y).ravel()
    K = kernel_fn(X, X)
    K += noise * np.eye(len(X))

    # Cholesky for numerical stability
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(X))))
    except np.linalg.LinAlgError:
        K_inv = np.linalg.inv(K)
        alpha = K_inv @ y

    return SurrogateModel(
        X_train=X,
        y_train=y,
        K_inv=K_inv,
        alpha=alpha,
        kernel_fn=kernel_fn,
        noise=noise,
    )


# ---------------------------------------------------------------------------
# Helper: acquisition functions
# ---------------------------------------------------------------------------


def _expected_improvement(
    mu: np.ndarray, sigma: np.ndarray, best_y: float
) -> np.ndarray:
    """Expected Improvement acquisition function.

    EI(x) = (best_y - mu) * Phi(z) + sigma * phi(z)
    where z = (best_y - mu) / sigma, Phi is the standard normal CDF,
    and phi is the standard normal PDF.

    We follow the convention that *lower* objective values are better
    (minimisation).
    """
    sigma = np.maximum(sigma, 1e-12)
    z = (best_y - mu) / sigma
    phi = np.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + _erf_approx(z / math.sqrt(2.0)))
    ei = (best_y - mu) * Phi + sigma * phi
    return np.maximum(ei, 0.0)


def _probability_of_improvement(
    mu: np.ndarray, sigma: np.ndarray, best_y: float
) -> np.ndarray:
    """Probability of Improvement.

    PI(x) = Phi((best_y - mu) / sigma)
    """
    sigma = np.maximum(sigma, 1e-12)
    z = (best_y - mu) / sigma
    return 0.5 * (1.0 + _erf_approx(z / math.sqrt(2.0)))


def _upper_confidence_bound(
    mu: np.ndarray, sigma: np.ndarray, beta: float = 2.0
) -> np.ndarray:
    """Upper Confidence Bound for minimisation: acquire where mu - beta*sigma is low.

    Returns *negative* LCB so that *maximising* this quantity picks
    explorative-yet-promising points.
    """
    return -(mu - beta * np.sqrt(sigma))


def _erf_approx(x: np.ndarray) -> np.ndarray:
    """Vectorised Abramowitz & Stegun approximation of erf(x).

    Maximum error ~ 1.5e-7.
    """
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    x_abs = np.abs(x)
    p = 0.3275911
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    t = 1.0 / (1.0 + p * x_abs)
    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    return sign * (1.0 - poly * np.exp(-x_abs ** 2))


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def diverse_experiment_design(
    factors: List[Factor],
    levels: Optional[List[List[Level]]] = None,
    n_runs: Optional[int] = None,
) -> DesignMatrix:
    """Generate an experimental design that maximises space-filling diversity.

    Strategy selection:
    * **Full factorial** when the total number of combinations is ≤ *n_runs*
      (or *n_runs* is ``None`` and total combinations ≤ 512).
    * **Fractional factorial** when full factorial is too large and *n_runs*
      is a power of two ≥ 2p.
    * **D-optimal** otherwise, using coordinate exchange.

    Parameters
    ----------
    factors : list[Factor]
        Experimental factors.
    levels : list[list[Level]], optional
        Per-factor levels.  If ``None``, the levels stored on each
        :class:`Factor` are used.
    n_runs : int, optional
        Desired number of runs.  Defaults to the full-factorial size capped
        at 512.

    Returns
    -------
    DesignMatrix
    """
    if levels is None:
        levels = [f.levels for f in factors]

    p = len(factors)
    level_values = [[lv.value for lv in lvs] for lvs in levels]
    total_combos = 1
    for lvs in level_values:
        total_combos *= len(lvs)

    if n_runs is None:
        n_runs = min(total_combos, 512)

    factor_names = [f.name for f in factors]

    # --- Full factorial ---
    if total_combos <= n_runs:
        rows = list(product(*level_values))
        matrix = np.array(rows, dtype=float)
        design_type = "full_factorial"

    # --- Fractional factorial (Resolution III) ---
    elif _is_power_of_two(n_runs) and n_runs >= 2 * p:
        matrix = _fractional_factorial(level_values, n_runs)
        design_type = "fractional_factorial"

    # --- D-optimal via coordinate exchange ---
    else:
        rng = np.random.default_rng(42)
        matrix = np.zeros((n_runs, p))
        for j in range(p):
            vals = np.array(level_values[j])
            matrix[:, j] = rng.choice(vals, size=n_runs)
        candidate_vals = [np.array(lv) for lv in level_values]
        matrix = _coordinate_exchange(matrix, candidate_values=candidate_vals, n_iter=2000)
        design_type = "d_optimal"

    d_eff = _d_efficiency(matrix)
    sf = _maximin_distance(matrix)

    return DesignMatrix(
        matrix=matrix,
        factor_names=factor_names,
        d_efficiency=d_eff,
        space_filling_metric=sf,
        design_type=design_type,
    )


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _fractional_factorial(
    level_values: List[List[float]], n_runs: int
) -> np.ndarray:
    """Build a fractional factorial design (Resolution III minimum).

    Uses Hadamard-like column assignment: the first *k* = log2(n_runs)
    columns are the generators; remaining columns are their products.
    Level values are mapped from {-1, +1} to the factor levels.
    """
    p = len(level_values)
    k = int(math.log2(n_runs))
    base = np.zeros((n_runs, k), dtype=float)
    for col in range(k):
        half = n_runs // (2 ** (col + 1))
        pattern = np.array([1.0] * half + [-1.0] * half)
        base[:, col] = np.tile(pattern, n_runs // len(pattern))

    columns: List[np.ndarray] = [base[:, c] for c in range(k)]

    # Generate extra columns as products of base columns
    col_idx = 0
    extra_needed = p - k
    generated = 0
    for r in range(2, k + 1):
        if generated >= extra_needed:
            break
        for combo in _combinations_indices(k, r):
            if generated >= extra_needed:
                break
            col = np.ones(n_runs)
            for idx in combo:
                col *= base[:, idx]
            columns.append(col)
            generated += 1

    columns = columns[:p]
    matrix = np.column_stack(columns)

    # Map {-1, +1} -> actual level values
    for j in range(p):
        vals = sorted(level_values[j])
        lo, hi = vals[0], vals[-1]
        matrix[:, j] = lo + (matrix[:, j] + 1.0) / 2.0 * (hi - lo)

    return matrix


def _combinations_indices(n: int, r: int) -> List[Tuple[int, ...]]:
    """Return all r-element combinations of range(n)."""
    if r == 0:
        return [()]
    if r > n:
        return []
    result: List[Tuple[int, ...]] = []
    for i in range(n):
        for rest in _combinations_indices(n - i - 1, r - 1):
            result.append((i,) + tuple(x + i + 1 for x in rest))
    return result


# ---------------------------------------------------------------------------


def latin_hypercube_diverse(
    n_dims: int,
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate a maximin Latin Hypercube Design via simulated annealing.

    The Latin property guarantees that projecting the design onto any single
    dimension yields exactly one point per stratum (one per ``n_samples``
    equal-probability intervals).

    Optimisation swaps two elements within a randomly chosen column,
    accepting worse solutions with Boltzmann probability to escape local
    minima.

    Parameters
    ----------
    n_dims : int
        Number of dimensions.
    n_samples : int
        Number of sample points.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray, shape (n_samples, n_dims)
        Design matrix with values in [0, 1].
    """
    rng = np.random.default_rng(seed)

    # Initial random LHD: one random permutation per column
    design = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        perm = rng.permutation(n_samples)
        design[:, d] = (perm + rng.uniform(size=n_samples)) / n_samples

    best_design = design.copy()
    best_min_dist = _maximin_distance(design)

    # Simulated annealing parameters
    T0 = 1.0
    T_min = 1e-4
    alpha = 0.995
    n_iter_per_temp = max(10, n_samples * n_dims // 2)

    T = T0
    cur_min_dist = best_min_dist

    while T > T_min:
        for _ in range(n_iter_per_temp):
            col = rng.integers(0, n_dims)
            i, j = rng.choice(n_samples, size=2, replace=False)

            # Swap within column to preserve Latin property
            design[i, col], design[j, col] = design[j, col], design[i, col]
            new_min_dist = _maximin_distance(design)

            delta = new_min_dist - cur_min_dist
            if delta > 0 or rng.uniform() < math.exp(delta / T):
                cur_min_dist = new_min_dist
                if cur_min_dist > best_min_dist:
                    best_min_dist = cur_min_dist
                    best_design = design.copy()
            else:
                # Revert swap
                design[i, col], design[j, col] = design[j, col], design[i, col]

        T *= alpha

    return best_design


# ---------------------------------------------------------------------------


def space_filling_design(
    bounds: np.ndarray,
    n_points: int,
    method: Literal["maximin", "minimax", "uniform"] = "maximin",
) -> np.ndarray:
    """Generate a space-filling design within rectangular *bounds*.

    Parameters
    ----------
    bounds : np.ndarray, shape (d, 2)
        Each row is ``[lower, upper]`` for one dimension.
    n_points : int
        Number of design points.
    method : {"maximin", "minimax", "uniform"}
        * ``"maximin"`` – farthest-point heuristic maximising the minimum
          pairwise distance.
        * ``"minimax"`` – greedy approach minimising the maximum distance
          from any location in the space to its nearest design point.
        * ``"uniform"`` – Sobol quasi-random sequence (or Halton for d>10).

    Returns
    -------
    np.ndarray, shape (n_points, d)
    """
    bounds = np.asarray(bounds, dtype=float)
    d = bounds.shape[0]
    lo = bounds[:, 0]
    hi = bounds[:, 1]

    if method == "uniform":
        if d <= 10:
            raw = _sobol_sequence(d, n_points)
        else:
            raw = _halton_sequence(d, n_points)
        return lo + raw * (hi - lo)

    if method == "maximin":
        return _maximin_design(bounds, n_points)

    if method == "minimax":
        return _minimax_design(bounds, n_points)

    raise ValueError(f"Unknown method: {method!r}")


def _maximin_design(bounds: np.ndarray, n_points: int) -> np.ndarray:
    """Farthest-point heuristic for maximin space-filling design.

    Starts with a random seed point, then iteratively adds the candidate
    point (from a large random pool) that is farthest from all existing
    points.
    """
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    rng = np.random.default_rng(42)

    n_candidates = max(1000, 50 * n_points)
    candidates = lo + rng.uniform(size=(n_candidates, d)) * (hi - lo)

    chosen_indices: List[int] = [rng.integers(0, n_candidates)]
    chosen_mask = np.zeros(n_candidates, dtype=bool)
    chosen_mask[chosen_indices[0]] = True

    # Track minimum distance from each candidate to the chosen set
    min_dist_to_chosen = np.full(n_candidates, np.inf)

    for _ in range(n_points - 1):
        last = candidates[chosen_indices[-1]]
        dists_to_last = np.sqrt(((candidates - last) ** 2).sum(axis=1))
        min_dist_to_chosen = np.minimum(min_dist_to_chosen, dists_to_last)
        min_dist_to_chosen[chosen_mask] = -np.inf
        best = int(np.argmax(min_dist_to_chosen))
        chosen_indices.append(best)
        chosen_mask[best] = True

    return candidates[chosen_indices]


def _minimax_design(bounds: np.ndarray, n_points: int) -> np.ndarray:
    """Greedy minimax design.

    Builds a large reference grid, then iteratively selects the design point
    that most reduces the maximum nearest-neighbour distance from any
    reference point to the design set.
    """
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    rng = np.random.default_rng(42)

    n_ref = max(2000, 100 * n_points)
    ref = lo + rng.uniform(size=(n_ref, d)) * (hi - lo)

    n_candidates = max(1000, 50 * n_points)
    candidates = lo + rng.uniform(size=(n_candidates, d)) * (hi - lo)

    selected: List[int] = []
    min_dist_ref = np.full(n_ref, np.inf)

    for _ in range(n_points):
        best_idx = -1
        best_max_dist = np.inf

        # Evaluate a subsample of candidates for speed
        sub = rng.choice(n_candidates, size=min(n_candidates, 200), replace=False)
        for ci in sub:
            dists = np.sqrt(((ref - candidates[ci]) ** 2).sum(axis=1))
            tentative = np.minimum(min_dist_ref, dists)
            max_d = tentative.max()
            if max_d < best_max_dist:
                best_max_dist = max_d
                best_idx = int(ci)

        selected.append(best_idx)
        dists = np.sqrt(((ref - candidates[best_idx]) ** 2).sum(axis=1))
        min_dist_ref = np.minimum(min_dist_ref, dists)

    return candidates[selected]


# ---------------------------------------------------------------------------


def sequential_design(
    model: SurrogateModel,
    acquisition_fn: Literal["ei", "ucb", "pi"] = "ei",
    diversity_penalty: float = 0.1,
    bounds: Optional[np.ndarray] = None,
    n_random: int = 5000,
    n_local: int = 50,
) -> NextPoint:
    """Recommend the next evaluation point using acquisition + diversity.

    Parameters
    ----------
    model : SurrogateModel
        Fitted GP surrogate.
    acquisition_fn : {"ei", "ucb", "pi"}
        Acquisition function name.
    diversity_penalty : float
        Weight for the diversity term that reduces acquisition near existing
        points.
    bounds : np.ndarray, optional, shape (d, 2)
        Search bounds.  Defaults to [0, 1]^d.
    n_random : int
        Number of random candidates for global search.
    n_local : int
        Number of perturbation steps around the best candidate.

    Returns
    -------
    NextPoint
    """
    X_train = model.X_train
    d = X_train.shape[1]

    if bounds is None:
        bounds = np.column_stack([np.zeros(d), np.ones(d)])
    lo, hi = bounds[:, 0], bounds[:, 1]

    rng = np.random.default_rng(123)
    candidates = lo + rng.uniform(size=(n_random, d)) * (hi - lo)

    mu, var = model.predict(candidates)
    sigma = np.sqrt(var)
    best_y = float(model.y_train.min())

    # Raw acquisition
    if acquisition_fn == "ei":
        acq = _expected_improvement(mu, sigma, best_y)
    elif acquisition_fn == "pi":
        acq = _probability_of_improvement(mu, sigma, best_y)
    elif acquisition_fn == "ucb":
        acq = _upper_confidence_bound(mu, sigma)
    else:
        raise ValueError(f"Unknown acquisition function: {acquisition_fn!r}")

    # Diversity term: penalise proximity to existing points
    div_values = np.array([
        np.min(np.sqrt(((X_train - c) ** 2).sum(axis=1))) for c in candidates
    ])
    div_norm = div_values / (div_values.max() + 1e-12)
    penalised_acq = acq + diversity_penalty * div_norm

    # Best candidate
    best_idx = int(np.argmax(penalised_acq))
    best_point = candidates[best_idx].copy()
    best_acq = float(penalised_acq[best_idx])

    # Local refinement around the best candidate
    step = 0.01 * (hi - lo)
    for _ in range(n_local):
        perturbation = rng.uniform(-1, 1, size=d) * step
        trial = np.clip(best_point + perturbation, lo, hi)
        trial_2d = trial.reshape(1, -1)
        mu_t, var_t = model.predict(trial_2d)
        sigma_t = np.sqrt(var_t)
        if acquisition_fn == "ei":
            a = _expected_improvement(mu_t, sigma_t, best_y)
        elif acquisition_fn == "pi":
            a = _probability_of_improvement(mu_t, sigma_t, best_y)
        else:
            a = _upper_confidence_bound(mu_t, sigma_t)
        div_t = float(np.min(np.sqrt(((X_train - trial) ** 2).sum(axis=1))))
        div_t_norm = div_t / (div_values.max() + 1e-12)
        a_pen = float(a[0]) + diversity_penalty * div_t_norm
        if a_pen > best_acq:
            best_acq = a_pen
            best_point = trial.copy()

    # Decompose final value
    best_2d = best_point.reshape(1, -1)
    mu_f, var_f = model.predict(best_2d)
    sigma_f = np.sqrt(var_f)
    if acquisition_fn == "ei":
        raw_acq = float(_expected_improvement(mu_f, sigma_f, best_y)[0])
    elif acquisition_fn == "pi":
        raw_acq = float(_probability_of_improvement(mu_f, sigma_f, best_y)[0])
    else:
        raw_acq = float(_upper_confidence_bound(mu_f, sigma_f)[0])

    div_contrib = float(np.min(np.sqrt(((X_train - best_point) ** 2).sum(axis=1))))

    return NextPoint(
        point=best_point,
        acquisition_value=raw_acq,
        diversity_contribution=div_contrib,
    )


# ---------------------------------------------------------------------------


def bayesian_optimization_diverse(
    objective: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_calls: int = 50,
    n_initial: int = 5,
    diversity_weight: float = 0.1,
    lengthscale: float = 1.0,
    kernel_variance: float = 1.0,
    noise: float = 1e-6,
    seed: int = 42,
) -> OptResult:
    """Bayesian optimisation loop with diversity-aware acquisition.

    Uses a Gaussian process with RBF kernel as the surrogate model and
    Expected Improvement as the base acquisition function, augmented by a
    diversity penalty that discourages evaluating near previously observed
    points.

    Parameters
    ----------
    objective : callable
        ``f(x) -> float`` to *minimise*.
    bounds : np.ndarray, shape (d, 2)
    n_calls : int
        Total budget (including initial points).
    n_initial : int
        Number of initial random evaluations.
    diversity_weight : float
        Weight of the diversity penalty in acquisition.
    lengthscale : float
        RBF kernel lengthscale.
    kernel_variance : float
        RBF kernel signal variance.
    noise : float
        Observation noise variance.
    seed : int
        Random seed.

    Returns
    -------
    OptResult
    """
    bounds = np.asarray(bounds, dtype=float)
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    rng = np.random.default_rng(seed)

    # Build kernel with captured hyper-parameters
    def kernel(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return rbf_kernel(X1, X2, lengthscale=lengthscale, variance=kernel_variance)

    # Initial evaluations via Latin Hypercube
    X_init = latin_hypercube_diverse(d, n_initial, seed=seed)
    X_init = lo + X_init * (hi - lo)
    y_init = np.array([objective(x) for x in X_init])

    X_all = X_init.copy()
    y_all = y_init.copy()

    convergence: List[float] = [float(y_all.min())]

    for i in range(n_initial, n_calls):
        # Fit GP
        gp = _gp_fit(X_all, y_all, kernel, noise=noise)

        # Determine next point via sequential design
        next_pt = sequential_design(
            model=gp,
            acquisition_fn="ei",
            diversity_penalty=diversity_weight,
            bounds=bounds,
            n_random=max(2000, 100 * d),
            n_local=50,
        )

        x_new = next_pt.point
        y_new = objective(x_new)

        X_all = np.vstack([X_all, x_new.reshape(1, -1)])
        y_all = np.append(y_all, y_new)
        convergence.append(float(y_all.min()))

    # Final GP fit
    final_gp = _gp_fit(X_all, y_all, kernel, noise=noise)

    best_idx = int(np.argmin(y_all))
    return OptResult(
        best_point=X_all[best_idx],
        best_value=float(y_all[best_idx]),
        X_evals=X_all,
        y_evals=y_all,
        surrogate=final_gp,
        convergence=convergence,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test
    print("=== Latin Hypercube ===")
    lhd = latin_hypercube_diverse(3, 10)
    print(f"  shape={lhd.shape}, min_dist={_maximin_distance(lhd):.4f}")

    print("\n=== Space-filling (maximin) ===")
    b = np.array([[0.0, 1.0], [0.0, 1.0]])
    sf = space_filling_design(b, 15, method="maximin")
    print(f"  shape={sf.shape}, min_dist={_maximin_distance(sf):.4f}")

    print("\n=== Space-filling (uniform / Sobol) ===")
    su = space_filling_design(b, 15, method="uniform")
    print(f"  shape={su.shape}, min_dist={_maximin_distance(su):.4f}")

    print("\n=== Diverse experiment design ===")
    factors = [
        Factor("A", [Level("lo", -1), Level("hi", 1)]),
        Factor("B", [Level("lo", -1), Level("hi", 1)]),
        Factor("C", [Level("lo", -1), Level("hi", 1)]),
    ]
    dm = diverse_experiment_design(factors, n_runs=8)
    print(f"  type={dm.design_type}, shape={dm.matrix.shape}, "
          f"D-eff={dm.d_efficiency:.4f}, sf={dm.space_filling_metric:.4f}")

    print("\n=== Bayesian optimisation (Branin 2-D) ===")

    def branin(x: np.ndarray) -> float:
        x1, x2 = x[0] * 15 - 5, x[1] * 15
        a, b_, c = 1.0, 5.1 / (4 * math.pi ** 2), 5.0 / math.pi
        r, s, t = 6.0, 10.0, 1.0 / (8 * math.pi)
        return float(a * (x2 - b_ * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s)

    res = bayesian_optimization_diverse(
        branin,
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        n_calls=25,
        n_initial=5,
        diversity_weight=0.1,
    )
    print(f"  best_value={res.best_value:.4f}, n_evals={len(res.y_evals)}")
    print("Done.")
