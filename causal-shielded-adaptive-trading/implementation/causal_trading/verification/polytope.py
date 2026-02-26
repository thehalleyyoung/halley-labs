"""
Polytope vertex enumeration for HMM posterior credible sets.

Provides H-representation / V-representation conversion, facet enumeration,
Chebyshev center computation, and highest-posterior-density credible set
construction over the space of Markov-chain transition matrices.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, spatial
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _simplex_vertices(dim: int) -> NDArray:
    """Return the dim standard-simplex vertices in R^dim."""
    return np.eye(dim, dtype=np.float64)


def _is_on_simplex(point: NDArray, tol: float = 1e-9) -> bool:
    """Check whether *point* lies on the probability simplex."""
    return bool(np.all(point >= -tol) and abs(point.sum() - 1.0) < tol)


def _project_to_simplex(v: NDArray) -> NDArray:
    """Project *v* onto the probability simplex (Duchi et al. 2008)."""
    n = len(v)
    u = np.sort(v)[::-1]
    cumsum = np.cumsum(u)
    rho = np.where(u > (cumsum - 1.0) / np.arange(1, n + 1))[0]
    if len(rho) == 0:
        return np.ones(n) / n
    rho_max = rho[-1]
    theta = (cumsum[rho_max] - 1.0) / (rho_max + 1)
    return np.maximum(v - theta, 0.0)


# ---------------------------------------------------------------------------
# Data classes for representations
# ---------------------------------------------------------------------------

@dataclass
class HRepresentation:
    """Half-space representation  Ax <= b  of a polytope."""
    A: NDArray  # (m, d)
    b: NDArray  # (m,)

    @property
    def n_constraints(self) -> int:
        return self.A.shape[0]

    @property
    def dim(self) -> int:
        return self.A.shape[1]


@dataclass
class VRepresentation:
    """Vertex representation of a polytope."""
    vertices: NDArray  # (n_vertices, d)

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def dim(self) -> int:
        return self.vertices.shape[1]


@dataclass
class Facet:
    """A single facet of a polytope, defined by a normal and offset."""
    normal: NDArray   # (d,)
    offset: float     # normal . x <= offset
    vertex_indices: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dirichlet utilities
# ---------------------------------------------------------------------------

def dirichlet_log_pdf(x: NDArray, alpha: NDArray) -> float:
    """Compute log-pdf of the Dirichlet distribution at *x*."""
    x = np.clip(x, 1e-300, None)
    log_beta = gammaln(alpha).sum() - gammaln(alpha.sum())
    return float(((alpha - 1.0) * np.log(x)).sum() - log_beta)


def dirichlet_mode(alpha: NDArray) -> NDArray:
    """Mode of Dirichlet(alpha) for alpha_i > 1."""
    alpha = np.asarray(alpha, dtype=np.float64)
    if np.any(alpha <= 1.0):
        return alpha / alpha.sum()
    return (alpha - 1.0) / (alpha.sum() - len(alpha))


# ---------------------------------------------------------------------------
# CredibleSetPolytope
# ---------------------------------------------------------------------------

class CredibleSetPolytope:
    """
    Credible-set polytope over the space of K×K Markov transition matrices.

    Each row of the transition matrix is modelled with an independent
    Dirichlet posterior.  The credible set at level *confidence* is the
    intersection of per-row HPD credible intervals, forming a polytope
    in the flattened transition-matrix space R^{K^2}.

    Parameters
    ----------
    n_regimes : int
        Number of latent regimes (K).
    prior_counts : NDArray, shape (K, K)
        Dirichlet prior pseudo-counts for each row.
    posterior_counts : NDArray, shape (K, K), optional
        If given, used directly; otherwise equals *prior_counts*.
    confidence : float
        HPD credible level (e.g. 0.95).
    """

    def __init__(
        self,
        n_regimes: int,
        prior_counts: NDArray,
        posterior_counts: Optional[NDArray] = None,
        confidence: float = 0.95,
    ) -> None:
        self.K = n_regimes
        self.prior_counts = np.array(prior_counts, dtype=np.float64)
        self.posterior_counts = (
            np.array(posterior_counts, dtype=np.float64)
            if posterior_counts is not None
            else self.prior_counts.copy()
        )
        assert self.prior_counts.shape == (self.K, self.K)
        assert self.posterior_counts.shape == (self.K, self.K)
        self.confidence = confidence

        # Caches
        self._vertices: Optional[NDArray] = None
        self._h_rep: Optional[HRepresentation] = None
        self._facets: Optional[List[Facet]] = None
        self._row_bounds: Optional[List[Tuple[NDArray, NDArray]]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_counts(self, observation_counts: NDArray) -> None:
        """Bayesian update with new transition counts."""
        observation_counts = np.asarray(observation_counts, dtype=np.float64)
        assert observation_counts.shape == (self.K, self.K)
        self.posterior_counts = self.posterior_counts + observation_counts
        self._invalidate_cache()

    def get_hpd_set(self, confidence: Optional[float] = None) -> "CredibleSetPolytope":
        """Return a new polytope at the requested HPD confidence level."""
        conf = confidence if confidence is not None else self.confidence
        clone = CredibleSetPolytope(
            self.K, self.prior_counts, self.posterior_counts.copy(), conf
        )
        return clone

    def compute_row_bounds(self) -> List[Tuple[NDArray, NDArray]]:
        """
        For each row i of the transition matrix, compute element-wise
        lower/upper bounds from the Dirichlet HPD credible region.

        Returns list of (lower, upper) pairs, one per row.
        """
        if self._row_bounds is not None:
            return self._row_bounds

        bounds: List[Tuple[NDArray, NDArray]] = []
        for i in range(self.K):
            alpha = self.posterior_counts[i]
            lo, hi = self._dirichlet_hpd_bounds(alpha, self.confidence)
            bounds.append((lo, hi))
        self._row_bounds = bounds
        return bounds

    def compute_vertices(self) -> NDArray:
        """
        Enumerate vertices of the credible polytope in R^{K^2}.

        For each row of the transition matrix the feasible set is a
        sub-polytope of the simplex.  The full polytope is the Cartesian
        product, so its vertices are all combinations of per-row vertices.

        Returns
        -------
        vertices : NDArray, shape (n_vertices, K*K)
        """
        if self._vertices is not None:
            return self._vertices

        row_vertex_lists: List[NDArray] = []
        for i in range(self.K):
            alpha = self.posterior_counts[i]
            rv = self._enumerate_row_vertices(alpha, self.confidence)
            row_vertex_lists.append(rv)

        # Cartesian product of per-row vertex sets
        index_lists = [range(len(rv)) for rv in row_vertex_lists]
        combined: List[NDArray] = []
        for combo in itertools.product(*index_lists):
            flat = np.concatenate(
                [row_vertex_lists[r][idx] for r, idx in enumerate(combo)]
            )
            combined.append(flat)
        self._vertices = np.array(combined, dtype=np.float64)
        return self._vertices

    def enumerate_facets(self) -> List[Facet]:
        """
        Enumerate facets of the credible polytope.

        Uses the H-representation (compute on demand) and identifies
        each active constraint as a facet, recording which vertices
        lie on it.
        """
        if self._facets is not None:
            return self._facets

        h = self.get_h_representation()
        verts = self.compute_vertices()

        facets: List[Facet] = []
        for j in range(h.n_constraints):
            a_j = h.A[j]
            b_j = h.b[j]
            residuals = verts @ a_j - b_j
            on_facet = np.where(np.abs(residuals) < 1e-8)[0].tolist()
            if len(on_facet) >= 1:
                facets.append(Facet(normal=a_j.copy(), offset=b_j, vertex_indices=on_facet))
        self._facets = facets
        return facets

    def get_h_representation(self) -> HRepresentation:
        """
        Build the H-representation  Ax <= b  of the credible polytope.

        Constraints come from:
        * per-row element-wise lower and upper bounds,
        * simplex equality constraints encoded as pairs of inequalities.
        """
        if self._h_rep is not None:
            return self._h_rep

        bounds = self.compute_row_bounds()
        d = self.K * self.K
        rows_A: List[NDArray] = []
        rows_b: List[float] = []

        for i in range(self.K):
            lo, hi = bounds[i]
            for j in range(self.K):
                idx = i * self.K + j
                # x_{ij} >= lo_j  →  -e_{idx} . x <= -lo_j
                e = np.zeros(d)
                e[idx] = -1.0
                rows_A.append(e)
                rows_b.append(-lo[j])
                # x_{ij} <= hi_j
                e2 = np.zeros(d)
                e2[idx] = 1.0
                rows_A.append(e2)
                rows_b.append(hi[j])

            # sum_j x_{ij} = 1  →  two inequalities
            eq = np.zeros(d)
            eq[i * self.K: (i + 1) * self.K] = 1.0
            rows_A.append(eq.copy())
            rows_b.append(1.0)
            rows_A.append(-eq)
            rows_b.append(-1.0)

        self._h_rep = HRepresentation(
            A=np.array(rows_A, dtype=np.float64),
            b=np.array(rows_b, dtype=np.float64),
        )
        return self._h_rep

    def get_v_representation(self) -> VRepresentation:
        """Return the V-representation of the polytope."""
        return VRepresentation(vertices=self.compute_vertices())

    def contains(self, point: NDArray, tol: float = 1e-8) -> bool:
        """Check whether *point* (flattened K×K) lies inside the polytope."""
        point = np.asarray(point, dtype=np.float64).ravel()
        assert point.shape == (self.K * self.K,)
        h = self.get_h_representation()
        return bool(np.all(h.A @ point - h.b <= tol))

    def chebyshev_center(self) -> NDArray:
        """
        Compute the Chebyshev center – the point inside the polytope
        that maximises the inscribed ball radius.

        Solves:  max r  s.t.  A x + r ||a_i|| <= b  for every row i.
        """
        h = self.get_h_representation()
        d = h.dim
        norms = np.linalg.norm(h.A, axis=1, keepdims=True)
        # Decision variables: [x (d), r (1)]
        A_lp = np.hstack([h.A, norms])  # (m, d+1)
        b_lp = h.b
        c = np.zeros(d + 1)
        c[-1] = -1.0  # maximise r

        bounds_lp = [(None, None)] * d + [(0, None)]
        result = optimize.linprog(c, A_ub=A_lp, b_ub=b_lp, bounds=bounds_lp, method="highs")
        if not result.success:
            # Fallback: centroid of vertices
            verts = self.compute_vertices()
            return verts.mean(axis=0)
        return result.x[:d]

    def centroid(self) -> NDArray:
        """Arithmetic mean of the vertices."""
        return self.compute_vertices().mean(axis=0)

    def volume_estimate(self, n_samples: int = 10000, rng: Optional[np.random.Generator] = None) -> float:
        """
        Monte-Carlo estimate of the polytope volume inside the
        bounding box defined by the row bounds.
        """
        rng = rng or np.random.default_rng(42)
        bounds = self.compute_row_bounds()
        lo_all = np.concatenate([lo for lo, _ in bounds])
        hi_all = np.concatenate([hi for _, hi in bounds])
        box_vol = float(np.prod(hi_all - lo_all))
        if box_vol <= 0:
            return 0.0

        samples = rng.uniform(lo_all, hi_all, size=(n_samples, self.K * self.K))
        inside = 0
        h = self.get_h_representation()
        for s in samples:
            if np.all(h.A @ s <= h.b + 1e-12):
                inside += 1
        return box_vol * inside / n_samples

    # ------------------------------------------------------------------
    # Set operations
    # ------------------------------------------------------------------

    def intersect(self, other: "CredibleSetPolytope") -> HRepresentation:
        """
        Intersection of two polytopes (returned as H-representation).

        Simply stacks the constraints.
        """
        h1 = self.get_h_representation()
        h2 = other.get_h_representation()
        A = np.vstack([h1.A, h2.A])
        b = np.concatenate([h1.b, h2.b])
        return HRepresentation(A=A, b=b)

    def union_vertices(self, other: "CredibleSetPolytope") -> VRepresentation:
        """
        Compute the convex hull of the union of two polytopes'
        vertex sets (convex-hull union, not set union).
        """
        v1 = self.compute_vertices()
        v2 = other.compute_vertices()
        all_v = np.vstack([v1, v2])
        if all_v.shape[1] <= 1:
            return VRepresentation(vertices=all_v)
        try:
            hull = spatial.ConvexHull(all_v)
            return VRepresentation(vertices=all_v[hull.vertices])
        except spatial.QhullError:
            # Degenerate – return unique rows
            unique = np.unique(all_v, axis=0)
            return VRepresentation(vertices=unique)

    def minkowski_sum_vertex(self, other: "CredibleSetPolytope") -> VRepresentation:
        """
        Minkowski sum via pairwise vertex addition.

        Result has at most |V1|*|V2| vertices before pruning.
        """
        v1 = self.compute_vertices()
        v2 = other.compute_vertices()
        sums = (v1[:, None, :] + v2[None, :, :]).reshape(-1, v1.shape[1])
        try:
            hull = spatial.ConvexHull(sums)
            return VRepresentation(vertices=sums[hull.vertices])
        except spatial.QhullError:
            return VRepresentation(vertices=np.unique(sums, axis=0))

    # ------------------------------------------------------------------
    # Projection / slicing
    # ------------------------------------------------------------------

    def project_to_row(self, row_index: int) -> NDArray:
        """
        Project the polytope vertices onto the coordinates
        corresponding to row *row_index* of the transition matrix.

        Returns array of shape (n_vertices, K).
        """
        verts = self.compute_vertices()
        start = row_index * self.K
        end = start + self.K
        return verts[:, start:end]

    def slice_at_row(self, row_index: int, row_value: NDArray) -> HRepresentation:
        """
        Fix row *row_index* to *row_value* and return the
        H-representation of the residual polytope over other rows.
        """
        row_value = np.asarray(row_value, dtype=np.float64)
        assert row_value.shape == (self.K,)
        h = self.get_h_representation()
        start = row_index * self.K
        end = start + self.K
        # Substitute: A x = A_other x_other + A_row row_value
        keep_cols = list(range(0, start)) + list(range(end, h.dim))
        A_new = h.A[:, keep_cols]
        b_new = h.b - h.A[:, start:end] @ row_value
        return HRepresentation(A=A_new, b=b_new)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invalidate_cache(self) -> None:
        self._vertices = None
        self._h_rep = None
        self._facets = None
        self._row_bounds = None

    @staticmethod
    def _dirichlet_hpd_bounds(
        alpha: NDArray, confidence: float
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute element-wise HPD bounds for Dirichlet(alpha).

        Uses marginal Beta distributions for each component.
        A component x_j ~ Beta(alpha_j, sum(alpha) - alpha_j).
        We find the shortest interval containing *confidence* mass.
        """
        K = len(alpha)
        alpha_sum = alpha.sum()
        lo = np.zeros(K, dtype=np.float64)
        hi = np.ones(K, dtype=np.float64)

        from scipy import stats

        for j in range(K):
            a_j = alpha[j]
            b_j = alpha_sum - a_j
            if a_j <= 0 or b_j <= 0:
                continue
            beta_dist = stats.beta(a_j, b_j)
            lo[j], hi[j] = _shortest_credible_interval(beta_dist, confidence)

        # Enforce simplex feasibility
        lo = np.maximum(lo, 0.0)
        hi = np.minimum(hi, 1.0)
        # Tighten: sum(lo) <= 1 and sum(hi) >= 1
        if lo.sum() > 1.0:
            lo = _project_to_simplex(lo)
        if hi.sum() < 1.0:
            hi = hi * (1.0 / hi.sum())
        return lo, hi

    def _enumerate_row_vertices(self, alpha: NDArray, confidence: float) -> NDArray:
        """
        Enumerate vertices of the credible sub-polytope for a single
        row Dirichlet(alpha) within the simplex.

        The feasible region is the intersection of the simplex with
        the axis-aligned box [lo, hi].  For K dimensions the vertices
        are all tight points where K-1 constraints are active.
        """
        K = len(alpha)
        lo, hi = self._dirichlet_hpd_bounds(alpha, confidence)

        if K == 1:
            return np.array([[1.0]])

        if K == 2:
            # Intersection of [lo, hi] box with simplex line x1+x2=1
            verts = []
            for x0 in [lo[0], hi[0]]:
                x1 = 1.0 - x0
                if lo[1] - 1e-12 <= x1 <= hi[1] + 1e-12:
                    verts.append(np.array([x0, x1]))
            if len(verts) == 0:
                mode = dirichlet_mode(alpha)
                verts.append(mode)
            return np.array(verts)

        # General K >= 3: solve vertex enumeration via LP at each
        # combination of active bound constraints.
        vertices: List[NDArray] = []
        seen: set = set()

        # Each vertex is defined by fixing K-1 of the 2K box constraints
        # to equality, subject to sum = 1.
        choices = list(range(K))

        # Strategy: try fixing each variable to its lower or upper bound
        # and solve for the rest on the simplex.
        for n_fixed in range(1, K):
            for fixed_indices in itertools.combinations(choices, n_fixed):
                free_indices = [j for j in choices if j not in fixed_indices]
                # Try all 2^n_fixed combinations of lo/hi
                if n_fixed > 12:
                    continue  # skip combinatorial explosion
                for bits in itertools.product([0, 1], repeat=n_fixed):
                    fixed_vals = np.array(
                        [lo[j] if bits[k] == 0 else hi[j] for k, j in enumerate(fixed_indices)]
                    )
                    remainder = 1.0 - fixed_vals.sum()
                    if remainder < -1e-10:
                        continue
                    if len(free_indices) == 0:
                        if abs(remainder) < 1e-10:
                            v = np.zeros(K)
                            for k, j in enumerate(fixed_indices):
                                v[j] = fixed_vals[k]
                            vertices.append(v)
                        continue

                    # Distribute remainder among free indices respecting bounds
                    v = self._solve_free_vars(
                        free_indices, fixed_indices, fixed_vals, lo, hi, remainder, K
                    )
                    if v is not None:
                        key = tuple(np.round(v, 10))
                        if key not in seen:
                            seen.add(key)
                            vertices.append(v)

        if len(vertices) == 0:
            vertices.append(dirichlet_mode(alpha))

        return np.array(vertices, dtype=np.float64)

    @staticmethod
    def _solve_free_vars(
        free_indices: List[int],
        fixed_indices: tuple,
        fixed_vals: NDArray,
        lo: NDArray,
        hi: NDArray,
        remainder: float,
        K: int,
    ) -> Optional[NDArray]:
        """Try to find a feasible assignment for free variables."""
        n_free = len(free_indices)
        lo_free = lo[free_indices]
        hi_free = hi[free_indices]

        if lo_free.sum() > remainder + 1e-10:
            return None
        if hi_free.sum() < remainder - 1e-10:
            return None

        # Greedy: set each free var to its lower bound, then distribute surplus
        vals = lo_free.copy()
        surplus = remainder - vals.sum()
        if surplus < -1e-10:
            return None
        for idx in range(n_free):
            add = min(surplus, hi_free[idx] - vals[idx])
            vals[idx] += add
            surplus -= add
        if abs(surplus) > 1e-8:
            return None

        v = np.zeros(K)
        for k, j in enumerate(fixed_indices):
            v[j] = fixed_vals[k]
        for k, j in enumerate(free_indices):
            v[j] = vals[k]

        if np.any(v < -1e-10) or abs(v.sum() - 1.0) > 1e-8:
            return None
        v = np.maximum(v, 0.0)
        v /= v.sum()
        return v

    def __repr__(self) -> str:
        n_v = self._vertices.shape[0] if self._vertices is not None else "?"
        return (
            f"CredibleSetPolytope(K={self.K}, confidence={self.confidence}, "
            f"vertices={n_v})"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _shortest_credible_interval(
    dist, confidence: float, n_grid: int = 500
) -> Tuple[float, float]:
    """
    Find the shortest interval [lo, hi] of a univariate distribution
    *dist* (scipy.stats frozen) that contains at least *confidence* mass.
    """
    alpha = 1.0 - confidence
    best_width = np.inf
    best_lo = dist.ppf(0.0)
    best_hi = dist.ppf(1.0)

    lo_candidates = np.linspace(0.0, alpha, n_grid)
    for a_lo in lo_candidates:
        a_hi = a_lo + confidence
        if a_hi > 1.0:
            break
        lo_val = dist.ppf(a_lo)
        hi_val = dist.ppf(a_hi)
        width = hi_val - lo_val
        if width < best_width:
            best_width = width
            best_lo = lo_val
            best_hi = hi_val
    return float(best_lo), float(best_hi)


def h_to_v(h_rep: HRepresentation) -> VRepresentation:
    """
    Convert an H-representation to a V-representation by solving
    a series of linear programmes to find extreme points.

    Uses the *double description* approach: for each subset of d
    active constraints, solve for the intersection point and keep
    it if feasible.
    """
    A, b = h_rep.A, h_rep.b
    m, d = A.shape
    vertices: List[NDArray] = []
    seen: set = set()

    if d == 0:
        return VRepresentation(vertices=np.empty((0, 0)))

    # LP-based vertex enumeration: push towards each constraint normal
    for j in range(m):
        c = A[j]
        res = optimize.linprog(-c, A_ub=A, b_ub=b, method="highs")
        if res.success:
            key = tuple(np.round(res.x, 10))
            if key not in seen:
                seen.add(key)
                vertices.append(res.x.copy())
        res2 = optimize.linprog(c, A_ub=A, b_ub=b, method="highs")
        if res2.success:
            key = tuple(np.round(res2.x, 10))
            if key not in seen:
                seen.add(key)
                vertices.append(res2.x.copy())

    # Also push along coordinate directions
    for i in range(d):
        c = np.zeros(d)
        c[i] = 1.0
        for sign in [1.0, -1.0]:
            res = optimize.linprog(sign * c, A_ub=A, b_ub=b, method="highs")
            if res.success:
                key = tuple(np.round(res.x, 10))
                if key not in seen:
                    seen.add(key)
                    vertices.append(res.x.copy())

    if len(vertices) == 0:
        return VRepresentation(vertices=np.empty((0, d)))
    return VRepresentation(vertices=np.array(vertices, dtype=np.float64))


def v_to_h(v_rep: VRepresentation) -> HRepresentation:
    """
    Convert a V-representation to an H-representation via convex hull.

    Falls back to bounding-box representation if the points are
    degenerate.
    """
    verts = v_rep.vertices
    if verts.shape[0] <= 1:
        d = verts.shape[1]
        if verts.shape[0] == 0:
            return HRepresentation(A=np.empty((0, d)), b=np.empty(0))
        # Single point: equality constraints
        A_rows = []
        b_rows = []
        for i in range(d):
            e = np.zeros(d)
            e[i] = 1.0
            A_rows.append(e.copy())
            b_rows.append(verts[0, i])
            A_rows.append(-e)
            b_rows.append(-verts[0, i])
        return HRepresentation(A=np.array(A_rows), b=np.array(b_rows))

    try:
        hull = spatial.ConvexHull(verts)
        A = hull.equations[:, :-1]
        b = -hull.equations[:, -1]
        return HRepresentation(A=A, b=b)
    except spatial.QhullError:
        d = verts.shape[1]
        lo = verts.min(axis=0)
        hi = verts.max(axis=0)
        A_rows = []
        b_rows = []
        for i in range(d):
            e = np.zeros(d)
            e[i] = 1.0
            A_rows.append(e.copy())
            b_rows.append(hi[i])
            A_rows.append(-e)
            b_rows.append(-lo[i])
        return HRepresentation(A=np.array(A_rows), b=np.array(b_rows))


def polytope_contains(h_rep: HRepresentation, point: NDArray, tol: float = 1e-9) -> bool:
    """Check whether *point* satisfies all constraints of *h_rep*."""
    return bool(np.all(h_rep.A @ point - h_rep.b <= tol))


def chebyshev_center(h_rep: HRepresentation) -> Tuple[NDArray, float]:
    """
    Chebyshev center and inscribed-ball radius.

    Returns (center, radius).
    """
    A, b = h_rep.A, h_rep.b
    d = A.shape[1]
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1e-15, norms)
    A_lp = np.hstack([A, norms])
    c = np.zeros(d + 1)
    c[-1] = -1.0
    bounds = [(None, None)] * d + [(0.0, None)]
    res = optimize.linprog(c, A_ub=A_lp, b_ub=b, bounds=bounds, method="highs")
    if res.success:
        return res.x[:d], float(res.x[-1])
    return np.zeros(d), 0.0


def random_interior_point(
    h_rep: HRepresentation, rng: Optional[np.random.Generator] = None
) -> Optional[NDArray]:
    """Sample a random interior point via hit-and-run from the Chebyshev center."""
    rng = rng or np.random.default_rng()
    center, radius = chebyshev_center(h_rep)
    if radius <= 0:
        return center

    point = center.copy()
    A, b = h_rep.A, h_rep.b
    d = A.shape[1]

    for _ in range(200):
        direction = rng.standard_normal(d)
        direction /= np.linalg.norm(direction) + 1e-15
        Ad = A @ direction
        slack = b - A @ point
        t_lo = -np.inf
        t_hi = np.inf
        for i in range(len(Ad)):
            if Ad[i] > 1e-15:
                t_hi = min(t_hi, slack[i] / Ad[i])
            elif Ad[i] < -1e-15:
                t_lo = max(t_lo, slack[i] / Ad[i])
        if t_lo >= t_hi:
            continue
        t = rng.uniform(max(t_lo, -1e6), min(t_hi, 1e6))
        point = point + t * direction
    return point
