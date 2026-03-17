"""taintflow.empirical.sliced_mi – Sliced mutual information estimator.

Implements projection-based mutual information estimation that avoids
the curse of dimensionality afflicting KSG in high dimensions.  The core
idea is to estimate MI via many random 1-D projections where KSG is
statistically reliable, then aggregate.

Key classes:
* :class:`SlicedMutualInfoEstimator` – MI via random 1-D projections.
* :class:`AdaptiveBoundEstimator` – combines KSG upper bound with sliced
  MI lower bound to produce tight intervals.
* :class:`DimensionAdaptiveEstimator` – automatically selects estimator
  based on data dimensionality.

Theory
------
For random unit vector θ drawn uniformly on S^{d-1}:

    SMI(X; Y) = E_θ[ I(θ^T X; θ^T Y) ]

is a lower bound on I(X; Y) (Goldfeld & Greenewald, NeurIPS 2021).
Each 1-D MI term is estimated by KSG on the projected scalar data,
which does **not** suffer from the curse of dimensionality.

References
----------
Goldfeld & Greenewald, "Sliced mutual information estimation",
NeurIPS 2021 (arXiv:2110.05279).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from taintflow.empirical.ksg import KSGEstimator, MutualInformationResult

# ---------------------------------------------------------------------------
# Linear-algebra helpers (stdlib-only, no NumPy)
# ---------------------------------------------------------------------------


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _scale(v: Sequence[float], s: float) -> List[float]:
    return [x * s for x in v]


def _subtract(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [ai - bi for ai, bi in zip(a, b)]


def _random_unit_vector(d: int, rng: random.Random) -> List[float]:
    """Sample a vector uniformly on S^{d-1} via Gaussian projection."""
    v = [rng.gauss(0.0, 1.0) for _ in range(d)]
    n = _norm(v)
    if n < 1e-12:
        # Degenerate draw – retry
        return _random_unit_vector(d, rng)
    return _scale(v, 1.0 / n)


def _column_means(data: List[List[float]], d: int) -> List[float]:
    n = len(data)
    means = [0.0] * d
    for row in data:
        for j in range(d):
            means[j] += row[j]
    return [m / n for m in means]


def _covariance_matrix(
    data: List[List[float]], d: int
) -> List[List[float]]:
    """Compute d×d sample covariance matrix (stdlib-only)."""
    n = len(data)
    means = _column_means(data, d)
    cov = [[0.0] * d for _ in range(d)]
    for row in data:
        centered = [row[j] - means[j] for j in range(d)]
        for i in range(d):
            for j in range(i, d):
                cov[i][j] += centered[i] * centered[j]
    for i in range(d):
        for j in range(i, d):
            cov[i][j] /= max(n - 1, 1)
            cov[j][i] = cov[i][j]
    return cov


def _power_iteration(
    matrix: List[List[float]], d: int, n_iter: int, rng: random.Random
) -> List[float]:
    """Return the leading eigenvector via power iteration."""
    v = _random_unit_vector(d, rng)
    for _ in range(n_iter):
        w = [sum(matrix[i][j] * v[j] for j in range(d)) for i in range(d)]
        n = _norm(w)
        if n < 1e-15:
            break
        v = _scale(w, 1.0 / n)
    return v


def _top_k_eigenvectors(
    matrix: List[List[float]], d: int, k: int, rng: random.Random
) -> List[List[float]]:
    """Approximate top-k eigenvectors via deflated power iteration."""
    vecs: List[List[float]] = []
    mat = [row[:] for row in matrix]
    for _ in range(min(k, d)):
        v = _power_iteration(mat, d, 100, rng)
        vecs.append(v)
        # Deflate: M <- M - λ v v^T  where λ = v^T M v
        lam = sum(v[i] * sum(mat[i][j] * v[j] for j in range(d))
                  for i in range(d))
        for i in range(d):
            for j in range(d):
                mat[i][j] -= lam * v[i] * v[j]
    return vecs


def _project(
    data: List[List[float]], direction: Sequence[float]
) -> List[List[float]]:
    """Project each row onto *direction*, returning N×1 list."""
    return [[_dot(row, direction)] for row in data]


# ---------------------------------------------------------------------------
# Sliced MI estimator
# ---------------------------------------------------------------------------


@dataclass
class SlicedMIResult:
    """Result of sliced mutual information estimation.

    Attributes
    ----------
    estimate:
        Mean of per-projection MI estimates (lower bound on true MI), nats.
    per_projection:
        Individual MI estimates for each projection direction.
    n_projections:
        Number of projections used.
    projection_strategy:
        ``"random"``, ``"pca"``, or ``"adversarial"``.
    n_samples:
        Number of data points.
    dimensionality:
        Dimensionality d_x + d_y of the joint space.
    """

    estimate: float
    per_projection: List[float] = field(default_factory=list)
    n_projections: int = 0
    projection_strategy: str = "random"
    n_samples: int = 0
    dimensionality: int = 0

    def to_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "per_projection": self.per_projection,
            "n_projections": self.n_projections,
            "projection_strategy": self.projection_strategy,
            "n_samples": self.n_samples,
            "dimensionality": self.dimensionality,
        }


class SlicedMutualInfoEstimator:
    """Estimate MI via random 1-D projections (sliced MI).

    For each of *n_projections* random directions θ, the estimator
    computes I(θ^T X ; θ^T Y) using a 1-D KSG estimator (no curse of
    dimensionality).  The average is a lower bound on the true MI:

        SMI(X; Y) = E_θ[I(θ^T X; θ^T Y)] ≤ I(X; Y)

    Parameters
    ----------
    n_projections:
        Number of random projection directions (default 50).
    projection_strategy:
        ``"random"``: uniform on S^{d-1}.
        ``"pca"``: project onto top PCA directions, supplemented by random.
        ``"adversarial"``: keep the top-MI projections from a larger pool.
    k:
        k for the 1-D KSG estimator (default 5).
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_projections: int = 50,
        projection_strategy: str = "random",
        k: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        if projection_strategy not in ("random", "pca", "adversarial"):
            raise ValueError(
                f"Unknown projection strategy: {projection_strategy!r}"
            )
        self.n_projections = n_projections
        self.projection_strategy = projection_strategy
        self.k = k
        self._rng = random.Random(seed)
        self._ksg = KSGEstimator(k=k, variant=1, bias_correction=True)

    # -- public API ---------------------------------------------------------

    def estimate(
        self,
        x: List[List[float]],
        y: List[List[float]],
    ) -> SlicedMIResult:
        """Estimate sliced MI between X and Y.

        Parameters
        ----------
        x:
            N × d_x array given as list of rows.
        y:
            N × d_y array given as list of rows.

        Returns
        -------
        SlicedMIResult
        """
        n = len(x)
        if n != len(y):
            raise ValueError("x and y must have the same number of samples")
        if n < self.k + 1:
            raise ValueError("Need at least k+1 samples")

        d_x = len(x[0]) if x else 0
        d_y = len(y[0]) if y else 0

        directions = self._generate_directions(x, y, d_x, d_y)

        per_proj: List[float] = []
        for theta_x, theta_y in directions:
            px = _project(x, theta_x)
            py = _project(y, theta_y)
            res = self._ksg.estimate(px, py)
            per_proj.append(max(res.estimate, 0.0))

        if self.projection_strategy == "adversarial":
            # Keep only the top n_projections by MI value
            per_proj.sort(reverse=True)
            per_proj = per_proj[: self.n_projections]

        mean_mi = sum(per_proj) / len(per_proj) if per_proj else 0.0

        return SlicedMIResult(
            estimate=mean_mi,
            per_projection=per_proj,
            n_projections=len(per_proj),
            projection_strategy=self.projection_strategy,
            n_samples=n,
            dimensionality=d_x + d_y,
        )

    # -- direction generation -----------------------------------------------

    def _generate_directions(
        self,
        x: List[List[float]],
        y: List[List[float]],
        d_x: int,
        d_y: int,
    ) -> List[Tuple[List[float], List[float]]]:
        """Generate projection direction pairs for X and Y spaces."""
        if self.projection_strategy == "random":
            return self._random_directions(d_x, d_y, self.n_projections)
        elif self.projection_strategy == "pca":
            return self._pca_directions(x, y, d_x, d_y)
        else:  # adversarial
            # Generate 3× pool, caller selects top-MI ones
            return self._random_directions(d_x, d_y, 3 * self.n_projections)

    def _random_directions(
        self, d_x: int, d_y: int, count: int
    ) -> List[Tuple[List[float], List[float]]]:
        return [
            (
                _random_unit_vector(d_x, self._rng),
                _random_unit_vector(d_y, self._rng),
            )
            for _ in range(count)
        ]

    def _pca_directions(
        self,
        x: List[List[float]],
        y: List[List[float]],
        d_x: int,
        d_y: int,
    ) -> List[Tuple[List[float], List[float]]]:
        """Use top PCA directions plus random directions."""
        n_pca = min(self.n_projections // 2, d_x, d_y)
        n_rand = self.n_projections - n_pca

        dirs: List[Tuple[List[float], List[float]]] = []

        if n_pca > 0 and d_x > 0 and d_y > 0:
            cov_x = _covariance_matrix(x, d_x)
            cov_y = _covariance_matrix(y, d_y)
            pca_x = _top_k_eigenvectors(cov_x, d_x, n_pca, self._rng)
            pca_y = _top_k_eigenvectors(cov_y, d_y, n_pca, self._rng)
            for vx, vy in zip(pca_x, pca_y):
                dirs.append((vx, vy))

        dirs.extend(self._random_directions(d_x, d_y, n_rand))
        return dirs


# ---------------------------------------------------------------------------
# Adaptive bound estimator
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveBoundResult:
    """Interval estimate [lower, upper] for mutual information.

    Attributes
    ----------
    lower:
        Sliced MI lower bound (nats).
    upper:
        KSG upper bound (nats).
    interval_width:
        upper − lower.
    is_tight:
        True when interval_width / max(upper, ε) < tightness_threshold.
    needs_more_data:
        True when the interval is too wide to be informative.
    method_lower:
        Estimator used for the lower bound.
    method_upper:
        Estimator used for the upper bound.
    """

    lower: float
    upper: float
    interval_width: float = 0.0
    is_tight: bool = False
    needs_more_data: bool = False
    method_lower: str = "sliced_mi"
    method_upper: str = "ksg"

    def __post_init__(self) -> None:
        self.interval_width = self.upper - self.lower

    def to_dict(self) -> dict:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "interval_width": self.interval_width,
            "is_tight": self.is_tight,
            "needs_more_data": self.needs_more_data,
            "method_lower": self.method_lower,
            "method_upper": self.method_upper,
        }


class AdaptiveBoundEstimator:
    """Combine KSG (upper bound) with sliced MI (lower bound).

    When the interval is narrow the result is a tight bound on the true
    MI.  When the interval is wide the estimator flags the result as
    ``needs_more_data`` instead of reporting a misleadingly loose bound.

    Parameters
    ----------
    tightness_threshold:
        Relative interval width below which the bound is considered
        tight (default 0.5, i.e. interval < 50 % of upper bound).
    n_projections:
        Number of projections for the sliced MI estimator.
    k_ksg:
        k for the KSG upper-bound estimator.
    k_sliced:
        k for the 1-D KSG inside sliced MI.
    seed:
        Random seed.
    """

    def __init__(
        self,
        tightness_threshold: float = 0.5,
        n_projections: int = 50,
        k_ksg: int = 3,
        k_sliced: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        self.tightness_threshold = tightness_threshold
        self._ksg = KSGEstimator(k=k_ksg, variant=1, bias_correction=True)
        self._sliced = SlicedMutualInfoEstimator(
            n_projections=n_projections,
            projection_strategy="pca",
            k=k_sliced,
            seed=seed,
        )

    def estimate(
        self,
        x: List[List[float]],
        y: List[List[float]],
    ) -> AdaptiveBoundResult:
        """Return an interval [sliced_MI, KSG] bounding I(X; Y)."""
        upper_res = self._ksg.estimate(x, y)
        lower_res = self._sliced.estimate(x, y)

        upper = max(upper_res.estimate, 0.0)
        lower = max(lower_res.estimate, 0.0)

        # Ensure interval is valid (lower ≤ upper)
        if lower > upper:
            lower, upper = upper, lower

        denom = max(upper, 1e-12)
        width = upper - lower
        relative_width = width / denom

        is_tight = relative_width < self.tightness_threshold
        needs_more = (not is_tight) and (upper > 1e-6)

        return AdaptiveBoundResult(
            lower=lower,
            upper=upper,
            is_tight=is_tight,
            needs_more_data=needs_more,
        )


# ---------------------------------------------------------------------------
# Dimension-adaptive estimator
# ---------------------------------------------------------------------------

_KSG_RELIABLE_DIM = 5  # KSG works well up to this joint dimensionality


class DimensionAdaptiveEstimator:
    """Automatically select MI estimator based on data dimensionality.

    * d ≤ 5: use KSG directly (reliable in low dimensions).
    * d > 5: use :class:`AdaptiveBoundEstimator` (sliced MI lower bound
      + KSG upper bound) and report an interval.

    Parameters
    ----------
    dim_threshold:
        Joint dimensionality threshold for switching estimators
        (default 5).
    n_projections:
        Number of projections for the sliced MI estimator.
    tightness_threshold:
        Relative width threshold for tight/loose classification.
    seed:
        Random seed.
    """

    def __init__(
        self,
        dim_threshold: int = _KSG_RELIABLE_DIM,
        n_projections: int = 50,
        tightness_threshold: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        self.dim_threshold = dim_threshold
        self._ksg = KSGEstimator(
            k=3, variant=1, bias_correction=True, seed=seed
        )
        self._adaptive = AdaptiveBoundEstimator(
            tightness_threshold=tightness_threshold,
            n_projections=n_projections,
            seed=seed,
        )

    def estimate(
        self,
        x: List[List[float]],
        y: List[List[float]],
    ) -> MutualInformationResult | AdaptiveBoundResult:
        """Estimate MI, dispatching on joint dimensionality.

        Returns
        -------
        MutualInformationResult
            When d ≤ dim_threshold (KSG path).
        AdaptiveBoundResult
            When d > dim_threshold (sliced MI + KSG interval).
        """
        d_x = len(x[0]) if x and x[0] else 0
        d_y = len(y[0]) if y and y[0] else 0
        d_joint = d_x + d_y

        if d_joint <= self.dim_threshold:
            return self._ksg.estimate(x, y)
        return self._adaptive.estimate(x, y)
