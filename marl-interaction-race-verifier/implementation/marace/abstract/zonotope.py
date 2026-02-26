"""
Zonotope abstract domain for MARACE.

A zonotope Z = {c + G ε | ε ∈ [-1,1]^p} is represented by a center vector c ∈ ℝ^n
and a generator matrix G ∈ ℝ^{n×p}. Zonotopes are a sub-polyhedric domain that
supports efficient affine transforms and reasonable join operations, making them
well-suited for neural network verification.

This implementation follows the DeepZono approach with extensions for
HB-constrained multi-agent verification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solve_lp(
    c: np.ndarray,
    A_ub: Optional[np.ndarray] = None,
    b_ub: Optional[np.ndarray] = None,
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    method: str = "highs",
) -> Optional[np.ndarray]:
    """Thin wrapper around scipy linprog with sensible defaults."""
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
    if res.success:
        return res.x
    return None


def _orth_reduce(G: np.ndarray, max_gens: int) -> np.ndarray:
    """Girard's PCA-based generator reduction.

    Keep the *max_gens* generators with the largest contribution to the
    zonotope's volume and over-approximate the rest with an axis-aligned
    bounding box (interval hull of the removed generators added as diagonal
    generators).
    """
    n, p = G.shape
    if p <= max_gens:
        return G.copy()

    norms = np.linalg.norm(G, axis=0)
    order = np.argsort(norms)  # ascending

    num_remove = p - max_gens + n  # we will add up to n axis-aligned gens
    if num_remove > p:
        num_remove = p
    keep_idx = order[num_remove:]
    remove_idx = order[:num_remove]

    G_keep = G[:, keep_idx]

    # Interval hull of removed generators -> diagonal generators
    removed = G[:, remove_idx]
    box_half = np.sum(np.abs(removed), axis=1)
    G_box = np.diag(box_half)

    # Remove zero-width box generators
    nonzero_mask = box_half > 0
    G_box = G_box[:, nonzero_mask]

    G_new = np.hstack([G_keep, G_box]) if G_box.size > 0 else G_keep

    # Final trim if we overshot
    if G_new.shape[1] > max_gens:
        norms2 = np.linalg.norm(G_new, axis=0)
        top_idx = np.argsort(norms2)[-max_gens:]
        G_new = G_new[:, top_idx]

    return G_new


# ---------------------------------------------------------------------------
# Zonotope
# ---------------------------------------------------------------------------

@dataclass
class Zonotope:
    """Zonotope abstract domain element.

    Parameters
    ----------
    center : np.ndarray
        Center vector of shape (n,).
    generators : np.ndarray
        Generator matrix of shape (n, p) where p is the number of generators.
    """

    center: np.ndarray
    generators: np.ndarray

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64).ravel()
        self.generators = np.atleast_2d(
            np.asarray(self.generators, dtype=np.float64)
        )
        if self.generators.ndim == 1:
            self.generators = self.generators.reshape(-1, 1)
        n = self.center.shape[0]
        # Auto-transpose: if rows don't match center dim but columns do,
        # assume user passed generators in row-major (p, n) form.
        if self.generators.shape[0] != n and self.generators.shape[1] == n:
            self.generators = self.generators.T
        if self.generators.size > 0 and self.generators.shape[0] != n:
            raise ValueError(
                f"Generator rows ({self.generators.shape[0]}) must match "
                f"center dimension ({self.center.shape[0]})"
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Dimension of the ambient space."""
        return self.center.shape[0]

    @property
    def num_generators(self) -> int:
        """Number of generators (noise symbols)."""
        return self.generators.shape[1]

    def bounding_box(self) -> np.ndarray:
        """Axis-aligned bounding box as array of shape (dim, 2).

        Returns ``bbox`` where ``bbox[i, 0]`` is the lower bound and
        ``bbox[i, 1]`` is the upper bound along dimension *i*.
        """
        half = np.sum(np.abs(self.generators), axis=1)
        lo = self.center - half
        hi = self.center + half
        return np.column_stack([lo, hi])

    @property
    def volume_bound(self) -> float:
        """Upper bound on the zonotope volume via the bounding-box volume.

        For exact volume computation one would need to enumerate zone-otope
        vertices, which is exponential; the interval hull gives a tractable
        sound over-approximation.
        """
        bbox = self.bounding_box()
        widths = bbox[:, 1] - bbox[:, 0]
        widths = np.maximum(widths, 0.0)
        return float(np.prod(widths))

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @staticmethod
    def from_interval(lower: np.ndarray, upper: np.ndarray) -> "Zonotope":
        """Create a zonotope from an axis-aligned box [lower, upper]."""
        lower = np.asarray(lower, dtype=np.float64).ravel()
        upper = np.asarray(upper, dtype=np.float64).ravel()
        if lower.shape != upper.shape:
            raise ValueError("lower and upper must have the same shape")
        center = (lower + upper) / 2.0
        half = (upper - lower) / 2.0
        generators = np.diag(half)
        return Zonotope(center=center, generators=generators)

    @staticmethod
    def from_point(point: np.ndarray) -> "Zonotope":
        """Degenerate zonotope containing a single point."""
        point = np.asarray(point, dtype=np.float64).ravel()
        n = point.shape[0]
        return Zonotope(center=point.copy(), generators=np.zeros((n, 0)))

    @staticmethod
    def unit_ball(n: int, num_generators: Optional[int] = None) -> "Zonotope":
        """Create a zonotope inscribed in the unit ℓ∞ ball in ℝ^n."""
        if num_generators is None:
            num_generators = n
        center = np.zeros(n)
        generators = np.eye(n, num_generators)
        return Zonotope(center=center, generators=generators)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def affine_transform(self, W: np.ndarray, b: Optional[np.ndarray] = None) -> "Zonotope":
        """Apply affine mapping z' = W z + b.

        This is exact: the image of a zonotope under an affine map is again a
        zonotope.

        Parameters
        ----------
        W : np.ndarray
            Weight matrix of shape (m, n).
        b : np.ndarray, optional
            Bias vector of shape (m,).
        """
        W = np.atleast_2d(np.asarray(W, dtype=np.float64))
        new_center = W @ self.center
        if b is not None:
            new_center = new_center + np.asarray(b, dtype=np.float64).ravel()
        new_generators = W @ self.generators
        return Zonotope(center=new_center, generators=new_generators)

    def join(self, other: "Zonotope") -> "Zonotope":
        """Sound over-approximation of the convex hull of two zonotopes.

        Uses the generator-merging method: the result has center equal to the
        midpoint of the two centers, with generators from both zonotopes plus
        the half-difference of centers.

        This is sound: conv(Z₁ ∪ Z₂) ⊆ Z_join.
        """
        if self.dimension != other.dimension:
            raise ValueError("Cannot join zonotopes of different dimensions")

        mid = (self.center + other.center) / 2.0
        delta = (self.center - other.center) / 2.0

        # Union generators: [G1, G2, delta_col]
        gen_parts = [self.generators, other.generators]
        if np.linalg.norm(delta) > 1e-15:
            gen_parts.append(delta.reshape(-1, 1))
        new_generators = np.hstack(gen_parts)

        return Zonotope(center=mid, generators=new_generators)

    def meet_halfspace(self, a: np.ndarray, b: float) -> "Zonotope":
        """Intersect with halfspace {x : a^T x <= b}.

        Uses the constrained-zonotope tightening approach:
        1. Compute the range [lo, hi] of a^T x over the zonotope.
        2. If hi <= b, the zonotope is already inside -> return self.
        3. If lo > b, the intersection is empty -> return empty.
        4. Otherwise, shrink the zonotope along direction a so that the
           maximum of a^T x equals b. We use a sound over-approximation
           via a generator-adjustment technique.
        """
        a = np.asarray(a, dtype=np.float64).ravel()
        if a.shape[0] != self.dimension:
            raise ValueError("Halfspace normal must match zonotope dimension")

        a_center = float(a @ self.center)
        a_gens = a @ self.generators  # shape (p,)
        a_spread = float(np.sum(np.abs(a_gens)))

        lo = a_center - a_spread
        hi = a_center + a_spread

        if hi <= b + 1e-12:
            return Zonotope(center=self.center.copy(), generators=self.generators.copy())

        if lo > b + 1e-12:
            return Zonotope(center=self.center.copy(), generators=np.zeros((self.dimension, 0)))

        # Tightening: adjust center and generators.
        # We want a^T x <= b. Currently max is hi = a_center + a_spread.
        # Strategy: shift center and scale generators along a by factor
        # lambda = (b - a_center) / a_spread.  This ensures the new max
        # of a^T x equals b while remaining sound.
        lam = (b - a_center) / a_spread if a_spread > 1e-15 else 0.0
        lam = max(0.0, min(1.0, lam))

        # New center: shift toward satisfying the constraint
        shift = ((1.0 - lam) / 2.0) * a_spread
        new_center = self.center - shift * (a / (np.dot(a, a) + 1e-30))

        # Scale generators that contribute to a^T g
        new_gens = self.generators.copy()
        for j in range(self.num_generators):
            contrib = abs(a_gens[j])
            if contrib > 1e-15:
                # Shrink this generator proportionally
                scale = lam + (1.0 - lam) * (1.0 - contrib / a_spread)
                scale = max(0.0, min(1.0, scale))
                new_gens[:, j] *= scale

        return Zonotope(center=new_center, generators=new_gens)

    def widening(self, other: "Zonotope", threshold: float = 1.05) -> "Zonotope":
        """Widening operator for fixpoint iteration.

        Given *self* (old iterate) and *other* (new iterate), produce a
        widened zonotope that is guaranteed to contain both and forces
        convergence.

        Strategy: for each dimension, if the new iterate is wider, expand
        by a multiplicative *threshold* factor.
        """
        if self.dimension != other.dimension:
            raise ValueError("Cannot widen zonotopes of different dimensions")

        lo_self, hi_self = self.bounding_box()[:, 0], self.bounding_box()[:, 1]
        lo_other, hi_other = other.bounding_box()[:, 0], other.bounding_box()[:, 1]

        new_lo = np.minimum(lo_self, lo_other)
        new_hi = np.maximum(hi_self, hi_other)

        # Apply widening: where new bounds exceed old, extrapolate
        for i in range(self.dimension):
            if new_lo[i] < lo_self[i] - 1e-12:
                width = hi_self[i] - lo_self[i]
                new_lo[i] = lo_self[i] - threshold * max(width, abs(lo_self[i] - new_lo[i]))
            if new_hi[i] > hi_self[i] + 1e-12:
                width = hi_self[i] - lo_self[i]
                new_hi[i] = hi_self[i] + threshold * max(width, abs(new_hi[i] - hi_self[i]))

        return Zonotope.from_interval(new_lo, new_hi)

    def contains_point(self, x: np.ndarray) -> bool:
        """Check if point x is contained in the zonotope via LP.

        Solves: find ε ∈ [-1,1]^p s.t. c + G ε = x
        which is feasible iff x ∈ Z.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.shape[0] != self.dimension:
            raise ValueError("Point dimension must match zonotope dimension")

        n = self.dimension
        p = self.num_generators

        if p == 0:
            return bool(np.allclose(x, self.center, atol=1e-10))

        # Formulate as LP: min 0 s.t. G ε = x - c, -1 <= ε <= 1
        # Use slack-variable formulation: ε = ε⁺ - ε⁻, ε⁺,ε⁻ >= 0,
        # ε⁺ + ε⁻ <= 1 (component-wise)
        target = x - self.center

        # Variables: [ε⁺(p), ε⁻(p)]
        c_obj = np.zeros(2 * p)
        # Equality: G (ε⁺ - ε⁻) = target
        A_eq = np.hstack([self.generators, -self.generators])
        b_eq = target

        # Inequality: ε⁺ + ε⁻ <= 1 (per component)
        A_ub = np.hstack([np.eye(p), np.eye(p)])
        b_ub = np.ones(p)

        bounds = [(0.0, 1.0)] * (2 * p)

        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        return res.success and res.status == 0

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample *n* points from the zonotope via generator coefficients.

        Uses the generator representation: x = c + G ε where each εᵢ is
        drawn uniformly from [-1,1].

        Note: this does NOT produce a uniform distribution over the
        zonotope when num_generators > dimension.  The resulting
        distribution has higher density near the center (by the CLT).
        For membership testing and coverage checks this is adequate.
        """
        if rng is None:
            rng = np.random.default_rng()
        p = self.num_generators
        if p == 0:
            return np.tile(self.center, (n, 1))
        eps = rng.uniform(-1.0, 1.0, size=(p, n))
        points = self.center[:, None] + self.generators @ eps
        return points.T  # (n, dim)

    # ------------------------------------------------------------------
    # Generator management
    # ------------------------------------------------------------------

    def reduce_generators(self, max_gens: int) -> "Zonotope":
        """Reduce the number of generators using Girard's PCA method.

        Returns a new zonotope with at most *max_gens* generators that
        soundly over-approximates self.
        """
        if self.num_generators <= max_gens:
            return Zonotope(center=self.center.copy(),
                            generators=self.generators.copy())
        new_G = _orth_reduce(self.generators, max_gens)
        return Zonotope(center=self.center.copy(), generators=new_G)

    def merge_generators(self, indices: Sequence[int]) -> "Zonotope":
        """Merge selected generators into a single generator.

        The merged generator's direction is the sum of the selected
        generators and its magnitude is the sum of their norms (sound
        over-approximation).
        """
        indices = list(indices)
        if not indices:
            return Zonotope(center=self.center.copy(),
                            generators=self.generators.copy())

        selected = self.generators[:, indices]
        direction = np.sum(selected, axis=1)
        dir_norm = np.linalg.norm(direction)
        mag = np.sum(np.linalg.norm(selected, axis=0))

        if dir_norm < 1e-15:
            # Degenerate: merge into interval hull
            merged = np.sum(np.abs(selected), axis=1).reshape(-1, 1)
        else:
            merged = (direction / dir_norm * mag).reshape(-1, 1)

        keep_mask = np.ones(self.num_generators, dtype=bool)
        keep_mask[indices] = False
        remaining = self.generators[:, keep_mask]

        new_G = np.hstack([remaining, merged]) if remaining.size > 0 else merged
        return Zonotope(center=self.center.copy(), generators=new_G)

    def remove_zero_generators(self, tol: float = 1e-12) -> "Zonotope":
        """Remove generators with norm below *tol*."""
        norms = np.linalg.norm(self.generators, axis=0)
        keep = norms > tol
        if np.all(keep):
            return Zonotope(center=self.center.copy(),
                            generators=self.generators.copy())
        kept = self.generators[:, keep]
        if kept.size == 0:
            kept = np.zeros((self.dimension, 0))
        return Zonotope(center=self.center.copy(), generators=kept)

    # ------------------------------------------------------------------
    # Geometric operations
    # ------------------------------------------------------------------

    def project(self, dims: Sequence[int]) -> "Zonotope":
        """Project onto a subset of dimensions."""
        dims = list(dims)
        return Zonotope(
            center=self.center[dims],
            generators=self.generators[dims, :],
        )

    def minkowski_sum(self, other: "Zonotope") -> "Zonotope":
        """Minkowski sum Z₁ ⊕ Z₂ = {x+y | x∈Z₁, y∈Z₂}.

        Exact: center sums, generators concatenate.
        """
        if self.dimension != other.dimension:
            raise ValueError("Dimensions must match for Minkowski sum")
        new_center = self.center + other.center
        new_gens = np.hstack([self.generators, other.generators])
        return Zonotope(center=new_center, generators=new_gens)

    def interval_hull(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the interval hull (bounding box) as (lower, upper)."""
        bbox = self.bounding_box()
        return bbox[:, 0], bbox[:, 1]

    def vertices_2d(self, dims: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Compute the 2D projection vertices for visualization.

        Parameters
        ----------
        dims : tuple of two ints, optional
            Dimensions to project onto. Defaults to (0, 1).

        Returns
        -------
        np.ndarray of shape (k, 2)
            Vertices of the 2D zonotope in counter-clockwise order.
        """
        if dims is None:
            dims = (0, 1)
        z2d = self.project(list(dims))
        z2d = z2d.remove_zero_generators()

        if z2d.num_generators == 0:
            return z2d.center.reshape(1, 2)

        # For 2D zonotopes: sort generators by angle, then sweep
        gens = z2d.generators  # (2, p)
        angles = np.arctan2(gens[1, :], gens[0, :])

        # Make generators point in upper half-plane (flip those pointing down)
        for j in range(gens.shape[1]):
            if angles[j] < -1e-12:
                gens[:, j] = -gens[:, j]
                angles[j] += np.pi

        order = np.argsort(angles)
        sorted_gens = gens[:, order]

        # Build vertices by tracing the boundary
        p = sorted_gens.shape[1]
        vertex = z2d.center - np.sum(sorted_gens, axis=1)

        vertices = [vertex.copy()]
        for j in range(p):
            vertex = vertex + 2.0 * sorted_gens[:, j]
            vertices.append(vertex.copy())
        for j in range(p):
            vertex = vertex - 2.0 * sorted_gens[:, j]
            vertices.append(vertex.copy())

        verts = np.array(vertices)
        # Remove near-duplicate closing vertex
        if np.allclose(verts[0], verts[-1], atol=1e-10):
            verts = verts[:-1]

        return verts

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: Union["Zonotope", np.ndarray, float]) -> "Zonotope":
        if isinstance(other, Zonotope):
            return self.minkowski_sum(other)
        offset = np.asarray(other, dtype=np.float64).ravel()
        if offset.shape[0] == 1:
            offset = np.full(self.dimension, offset[0])
        return Zonotope(center=self.center + offset,
                        generators=self.generators.copy())

    def __radd__(self, other: Union[np.ndarray, float]) -> "Zonotope":
        return self.__add__(other)

    def __sub__(self, other: Union["Zonotope", np.ndarray, float]) -> "Zonotope":
        if isinstance(other, Zonotope):
            return Zonotope(
                center=self.center - other.center,
                generators=np.hstack([self.generators, other.generators]),
            )
        offset = np.asarray(other, dtype=np.float64).ravel()
        if offset.shape[0] == 1:
            offset = np.full(self.dimension, offset[0])
        return Zonotope(center=self.center - offset,
                        generators=self.generators.copy())

    def __mul__(self, scalar: float) -> "Zonotope":
        return self.scale(scalar)

    def __rmul__(self, scalar: float) -> "Zonotope":
        return self.scale(scalar)

    def scale(self, scalar: float) -> "Zonotope":
        """Scale zonotope by a scalar factor."""
        return Zonotope(
            center=self.center * scalar,
            generators=self.generators * scalar,
        )

    def __neg__(self) -> "Zonotope":
        return self.scale(-1.0)

    # ------------------------------------------------------------------
    # Comparison / metrics
    # ------------------------------------------------------------------

    def hausdorff_upper_bound(self, other: "Zonotope") -> float:
        """Upper bound on the Hausdorff distance between two zonotopes.

        Uses the interval-hull metric: max_i max(|lo_i - lo'_i|, |hi_i - hi'_i|).
        """
        lo1, hi1 = self.bounding_box()[:, 0], self.bounding_box()[:, 1]
        lo2, hi2 = other.bounding_box()[:, 0], other.bounding_box()[:, 1]
        return float(np.max(np.maximum(np.abs(lo1 - lo2), np.abs(hi1 - hi2))))

    def is_subset_of(self, other: "Zonotope", num_samples: int = 200) -> bool:
        """Probabilistic subset check via sampling.

        Sound in one direction: if any sample is outside *other*, then
        self ⊄ other. If all samples are inside, returns True but this is
        not a proof of inclusion.
        """
        pts = self.sample(num_samples)
        for pt in pts:
            if not other.contains_point(pt):
                return False
        return True

    def is_empty(self) -> bool:
        """A zonotope with generators is never empty; only the degenerate
        zero-generator case with no feasible center."""
        return self.num_generators == 0 and self.generators.shape[1] == 0

    # ------------------------------------------------------------------
    # Optimization over the zonotope
    # ------------------------------------------------------------------

    def maximize(self, direction: np.ndarray) -> Tuple[float, np.ndarray]:
        """Maximize direction^T x over the zonotope.

        The optimal ε* has εᵢ = sign(aᵢ) where a = G^T direction.
        """
        direction = np.asarray(direction, dtype=np.float64).ravel()
        a = self.generators.T @ direction  # shape (p,)
        eps_star = np.sign(a)
        x_star = self.center + self.generators @ eps_star
        return float(direction @ x_star), x_star

    def minimize(self, direction: np.ndarray) -> Tuple[float, np.ndarray]:
        """Minimize direction^T x over the zonotope."""
        val, x = self.maximize(-np.asarray(direction))
        return -val, x

    def support_function(self, direction: np.ndarray) -> float:
        """Evaluate the support function h_Z(d) = max { d^T x : x ∈ Z }."""
        val, _ = self.maximize(direction)
        return val

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Serialize zonotope to a JSON-compatible dictionary."""
        return {
            "center": self.center.tolist(),
            "generators": self.generators.tolist(),
        }

    @staticmethod
    def from_dict(d: Dict) -> "Zonotope":
        """Deserialize zonotope from a dictionary."""
        return Zonotope(
            center=np.array(d["center"], dtype=np.float64),
            generators=np.array(d["generators"], dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Zonotope(dim={self.dimension}, gens={self.num_generators}, "
            f"bbox_vol={self.volume_bound:.4g})"
        )

    def summary(self) -> str:
        """Human-readable summary."""
        bbox = self.bounding_box()
        lo, hi = bbox[:, 0], bbox[:, 1]
        lines = [
            f"Zonotope in ℝ^{self.dimension} with {self.num_generators} generators",
            f"  Center: {self.center}",
            f"  Bounding box volume: {self.volume_bound:.6g}",
        ]
        for i in range(min(self.dimension, 8)):
            lines.append(f"  dim {i}: [{lo[i]:.6g}, {hi[i]:.6g}]")
        if self.dimension > 8:
            lines.append(f"  ... ({self.dimension - 8} more dimensions)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Copying
    # ------------------------------------------------------------------

    def copy(self) -> "Zonotope":
        """Return a deep copy."""
        return Zonotope(
            center=self.center.copy(),
            generators=self.generators.copy(),
        )

    # ------------------------------------------------------------------
    # Static analysis helpers
    # ------------------------------------------------------------------

    def component_range(self, dim: int) -> Tuple[float, float]:
        """Range of the zonotope along a single dimension."""
        half = float(np.sum(np.abs(self.generators[dim, :])))
        c = float(self.center[dim])
        return c - half, c + half

    def split_halfspace(
        self, a: np.ndarray, b: float
    ) -> Tuple["Zonotope", "Zonotope"]:
        """Split into (Z ∩ {a^T x <= b}, Z ∩ {a^T x >= b}).

        Both halves are sound over-approximations.
        """
        positive = self.meet_halfspace(a, b)
        negative = self.meet_halfspace(-np.asarray(a), -b)
        return positive, negative

    def element_wise_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per-dimension lower and upper bounds (same as bounding_box)."""
        bbox = self.bounding_box()
        return bbox[:, 0], bbox[:, 1]

    # ------------------------------------------------------------------
    # Batch operations for neural network layers
    # ------------------------------------------------------------------

    def batch_affine(
        self, weights: List[np.ndarray], biases: List[Optional[np.ndarray]]
    ) -> List["Zonotope"]:
        """Apply a sequence of affine transforms, returning all intermediates."""
        results: List[Zonotope] = []
        current = self
        for W, b in zip(weights, biases):
            current = current.affine_transform(W, b)
            results.append(current)
        return results
