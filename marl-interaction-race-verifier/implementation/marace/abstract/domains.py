"""
Additional abstract domains for MARACE.

Provides interval, hybrid, constrained-zonotope, and product domains
alongside conversion utilities and a factory for selecting the appropriate
domain based on analysis requirements.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
from scipy.optimize import linprog

from marace.abstract.zonotope import Zonotope

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interval domain
# ---------------------------------------------------------------------------


@dataclass
class IntervalDomain:
    """Simple interval (box) abstract domain for baseline comparison.

    Represents the set {x : lower <= x <= upper} component-wise.
    """

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=np.float64).ravel()
        self.upper = np.asarray(self.upper, dtype=np.float64).ravel()
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape")

    @property
    def dimension(self) -> int:
        return self.lower.shape[0]

    @property
    def widths(self) -> np.ndarray:
        return self.upper - self.lower

    @property
    def center(self) -> np.ndarray:
        return (self.lower + self.upper) / 2.0

    @property
    def volume(self) -> float:
        w = np.maximum(self.widths, 0.0)
        return float(np.prod(w))

    @property
    def is_empty(self) -> bool:
        return bool(np.any(self.lower > self.upper + 1e-12))

    # --- constructors ---

    @staticmethod
    def from_point(x: np.ndarray) -> "IntervalDomain":
        x = np.asarray(x, dtype=np.float64).ravel()
        return IntervalDomain(lower=x.copy(), upper=x.copy())

    @staticmethod
    def full_space(n: int, bound: float = 1e6) -> "IntervalDomain":
        return IntervalDomain(
            lower=np.full(n, -bound),
            upper=np.full(n, bound),
        )

    # --- operations ---

    def join(self, other: "IntervalDomain") -> "IntervalDomain":
        """Join (union over-approximation) of two intervals."""
        return IntervalDomain(
            lower=np.minimum(self.lower, other.lower),
            upper=np.maximum(self.upper, other.upper),
        )

    def meet(self, other: "IntervalDomain") -> "IntervalDomain":
        """Meet (intersection) of two intervals."""
        return IntervalDomain(
            lower=np.maximum(self.lower, other.lower),
            upper=np.minimum(self.upper, other.upper),
        )

    def affine_transform(
        self, W: np.ndarray, b: Optional[np.ndarray] = None
    ) -> "IntervalDomain":
        """Apply affine transform y = Wx + b using interval arithmetic.

        For each output y_i = Σ_j W_{ij} x_j + b_i:
        - Positive W_{ij}: contributes W_{ij} * [l_j, u_j]
        - Negative W_{ij}: contributes W_{ij} * [u_j, l_j] (reversed)
        """
        W = np.atleast_2d(np.asarray(W, dtype=np.float64))
        m, n = W.shape

        W_pos = np.maximum(W, 0.0)
        W_neg = np.minimum(W, 0.0)

        new_lower = W_pos @ self.lower + W_neg @ self.upper
        new_upper = W_pos @ self.upper + W_neg @ self.lower

        if b is not None:
            b = np.asarray(b, dtype=np.float64).ravel()
            new_lower += b
            new_upper += b

        return IntervalDomain(lower=new_lower, upper=new_upper)

    def relu(self) -> "IntervalDomain":
        """Apply ReLU element-wise."""
        return IntervalDomain(
            lower=np.maximum(self.lower, 0.0),
            upper=np.maximum(self.upper, 0.0),
        )

    def tanh(self) -> "IntervalDomain":
        """Apply tanh element-wise (monotone)."""
        return IntervalDomain(
            lower=np.tanh(self.lower),
            upper=np.tanh(self.upper),
        )

    def contains_point(self, x: np.ndarray) -> bool:
        x = np.asarray(x, dtype=np.float64).ravel()
        return bool(np.all(x >= self.lower - 1e-12) and np.all(x <= self.upper + 1e-12))

    def meet_halfspace(self, a: np.ndarray, b: float) -> "IntervalDomain":
        """Intersect with halfspace a^T x <= b.

        For intervals, we tighten each upper bound where the constraint
        allows.  This is a sound but potentially imprecise operation.
        """
        a = np.asarray(a, dtype=np.float64).ravel()

        new_lower = self.lower.copy()
        new_upper = self.upper.copy()

        # For each dimension i where a[i] > 0, we can upper-bound x[i]:
        #   x[i] <= (b - Σ_{j≠i} a[j]*x[j]_min_contrib) / a[i]
        # using interval arithmetic for the rest.
        for i in range(self.dimension):
            if abs(a[i]) < 1e-15:
                continue

            # Compute the range of Σ_{j≠i} a[j]*x[j]
            rest_min = 0.0
            for j in range(self.dimension):
                if j == i:
                    continue
                if a[j] >= 0:
                    rest_min += a[j] * self.lower[j]
                else:
                    rest_min += a[j] * self.upper[j]

            if a[i] > 0:
                bound_i = (b - rest_min) / a[i]
                new_upper[i] = min(new_upper[i], bound_i)
            else:
                bound_i = (b - rest_min) / a[i]
                new_lower[i] = max(new_lower[i], bound_i)

        return IntervalDomain(lower=new_lower, upper=new_upper)

    def widening(self, other: "IntervalDomain", threshold: float = 1.05) -> "IntervalDomain":
        """Widening operator for fixpoint iteration."""
        new_lo = self.lower.copy()
        new_hi = self.upper.copy()

        for i in range(self.dimension):
            if other.lower[i] < self.lower[i] - 1e-12:
                width = self.upper[i] - self.lower[i]
                new_lo[i] = self.lower[i] - threshold * max(
                    width, abs(self.lower[i] - other.lower[i])
                )
            if other.upper[i] > self.upper[i] + 1e-12:
                width = self.upper[i] - self.lower[i]
                new_hi[i] = self.upper[i] + threshold * max(
                    width, abs(other.upper[i] - self.upper[i])
                )

        return IntervalDomain(lower=new_lo, upper=new_hi)

    def project(self, dims: Sequence[int]) -> "IntervalDomain":
        dims = list(dims)
        return IntervalDomain(lower=self.lower[dims], upper=self.upper[dims])

    def hausdorff_distance(self, other: "IntervalDomain") -> float:
        return float(np.max(np.maximum(
            np.abs(self.lower - other.lower),
            np.abs(self.upper - other.upper),
        )))

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.lower, self.upper, size=(n, self.dimension))

    # --- serialization ---

    def to_dict(self) -> Dict[str, Any]:
        return {"lower": self.lower.tolist(), "upper": self.upper.tolist()}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "IntervalDomain":
        return IntervalDomain(
            lower=np.array(d["lower"]),
            upper=np.array(d["upper"]),
        )

    def __repr__(self) -> str:
        return f"IntervalDomain(dim={self.dimension}, vol={self.volume:.4g})"


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------


def interval_to_zonotope(interval: IntervalDomain) -> Zonotope:
    """Convert an interval domain element to a zonotope."""
    return Zonotope.from_interval(interval.lower, interval.upper)


def zonotope_to_interval(z: Zonotope) -> IntervalDomain:
    """Convert a zonotope to its interval hull."""
    lo, hi = z.bounding_box()[:, 0], z.bounding_box()[:, 1]
    return IntervalDomain(lower=lo, upper=hi)


# ---------------------------------------------------------------------------
# Hybrid domain
# ---------------------------------------------------------------------------


class HybridDomain:
    """Combine zonotope (for relational dims) with interval (for others).

    Some state dimensions benefit from the relational precision of
    zonotopes (e.g., shared state), while others are adequately tracked
    with intervals (e.g., local counters). This domain applies zonotopes
    to the *relational_dims* and intervals to the rest.
    """

    def __init__(
        self,
        zonotope_part: Zonotope,
        interval_part: IntervalDomain,
        zonotope_dims: List[int],
        interval_dims: List[int],
        total_dim: int,
    ) -> None:
        self.zonotope_part = zonotope_part
        self.interval_part = interval_part
        self.zonotope_dims = zonotope_dims
        self.interval_dims = interval_dims
        self.total_dim = total_dim

    @staticmethod
    def from_interval(
        interval: IntervalDomain,
        zonotope_dims: List[int],
    ) -> "HybridDomain":
        """Create a hybrid domain from an interval, promoting selected
        dimensions to zonotope representation."""
        interval_dims = [i for i in range(interval.dimension) if i not in zonotope_dims]
        z = Zonotope.from_interval(
            interval.lower[zonotope_dims],
            interval.upper[zonotope_dims],
        )
        iv = IntervalDomain(
            lower=interval.lower[interval_dims],
            upper=interval.upper[interval_dims],
        )
        return HybridDomain(
            zonotope_part=z,
            interval_part=iv,
            zonotope_dims=zonotope_dims,
            interval_dims=interval_dims,
            total_dim=interval.dimension,
        )

    def to_zonotope(self) -> Zonotope:
        """Convert to a full zonotope (embedding interval part as diagonal generators)."""
        center = np.zeros(self.total_dim)
        gen_blocks: List[np.ndarray] = []

        # Zonotope part
        for i, d in enumerate(self.zonotope_dims):
            center[d] = self.zonotope_part.center[i]
        z_embed = np.zeros((self.total_dim, self.zonotope_part.num_generators))
        for i, d in enumerate(self.zonotope_dims):
            z_embed[d, :] = self.zonotope_part.generators[i, :]
        gen_blocks.append(z_embed)

        # Interval part as diagonal generators
        for i, d in enumerate(self.interval_dims):
            center[d] = self.interval_part.center[i]
            half = self.interval_part.widths[i] / 2.0
            if half > 1e-15:
                col = np.zeros((self.total_dim, 1))
                col[d, 0] = half
                gen_blocks.append(col)

        if gen_blocks:
            generators = np.hstack(gen_blocks)
        else:
            generators = np.zeros((self.total_dim, 0))

        return Zonotope(center=center, generators=generators)

    def to_interval(self) -> IntervalDomain:
        """Convert to full interval domain."""
        lower = np.zeros(self.total_dim)
        upper = np.zeros(self.total_dim)

        z_bbox = self.zonotope_part.bounding_box()
        for i, d in enumerate(self.zonotope_dims):
            lower[d] = z_bbox[i, 0]
            upper[d] = z_bbox[i, 1]

        for i, d in enumerate(self.interval_dims):
            lower[d] = self.interval_part.lower[i]
            upper[d] = self.interval_part.upper[i]

        return IntervalDomain(lower=lower, upper=upper)

    @property
    def volume_bound(self) -> float:
        return self.zonotope_part.volume_bound * self.interval_part.volume

    def __repr__(self) -> str:
        return (
            f"HybridDomain(z_dims={len(self.zonotope_dims)}, "
            f"i_dims={len(self.interval_dims)}, "
            f"vol={self.volume_bound:.4g})"
        )


# ---------------------------------------------------------------------------
# Constrained zonotope
# ---------------------------------------------------------------------------


class ConstrainedZonotope:
    """Zonotope with explicit linear constraints on the noise symbols.

    A constrained zonotope is:
        CZ = { c + G ε | A ε <= b, ε ∈ [-1,1]^p }

    This provides tighter HB-awareness than intersecting the zonotope
    with halfspaces, because constraints on ε directly restrict the
    noise-symbol space rather than the concrete state space.
    """

    def __init__(
        self,
        center: np.ndarray,
        generators: np.ndarray,
        A_cons: Optional[np.ndarray] = None,
        b_cons: Optional[np.ndarray] = None,
    ) -> None:
        self.center = np.asarray(center, dtype=np.float64).ravel()
        self.generators = np.atleast_2d(
            np.asarray(generators, dtype=np.float64)
        )
        if self.generators.ndim == 1:
            self.generators = self.generators.reshape(-1, 1)

        p = self.generators.shape[1]
        if A_cons is not None:
            self.A_cons = np.atleast_2d(np.asarray(A_cons, dtype=np.float64))
            self.b_cons = np.asarray(b_cons, dtype=np.float64).ravel()
        else:
            self.A_cons = np.zeros((0, p))
            self.b_cons = np.zeros(0)

    @property
    def dimension(self) -> int:
        return self.center.shape[0]

    @property
    def num_generators(self) -> int:
        return self.generators.shape[1]

    @property
    def num_constraints(self) -> int:
        return self.A_cons.shape[0]

    @staticmethod
    def from_zonotope(z: Zonotope) -> "ConstrainedZonotope":
        """Lift a plain zonotope to a constrained zonotope (no extra constraints)."""
        return ConstrainedZonotope(
            center=z.center.copy(),
            generators=z.generators.copy(),
        )

    def to_zonotope(self) -> Zonotope:
        """Drop constraints (sound over-approximation)."""
        return Zonotope(center=self.center.copy(), generators=self.generators.copy())

    def add_state_constraint(self, a: np.ndarray, b: float) -> "ConstrainedZonotope":
        """Add constraint a^T x <= b by converting to noise-symbol space.

        a^T (c + G ε) <= b  =>  (a^T G) ε <= b - a^T c
        """
        a = np.asarray(a, dtype=np.float64).ravel()
        a_eps = (a @ self.generators).reshape(1, -1)
        b_eps = np.array([b - float(a @ self.center)])

        new_A = np.vstack([self.A_cons, a_eps]) if self.A_cons.size > 0 else a_eps
        new_b = np.concatenate([self.b_cons, b_eps])

        return ConstrainedZonotope(
            center=self.center.copy(),
            generators=self.generators.copy(),
            A_cons=new_A,
            b_cons=new_b,
        )

    def contains_point(self, x: np.ndarray) -> bool:
        """Check if point x is in the constrained zonotope via LP."""
        x = np.asarray(x, dtype=np.float64).ravel()
        p = self.num_generators
        if p == 0:
            return bool(np.allclose(x, self.center, atol=1e-10))

        target = x - self.center

        c_obj = np.zeros(2 * p)
        A_eq = np.hstack([self.generators, -self.generators])
        b_eq = target

        # Box constraints on ε: ε⁺ + ε⁻ <= 1
        A_ub_box = np.hstack([np.eye(p), np.eye(p)])
        b_ub_box = np.ones(p)

        # User constraints: A (ε⁺ - ε⁻) <= b
        if self.A_cons.size > 0:
            A_ub_user = np.hstack([self.A_cons, -self.A_cons])
            b_ub_user = self.b_cons
            A_ub = np.vstack([A_ub_box, A_ub_user])
            b_ub = np.concatenate([b_ub_box, b_ub_user])
        else:
            A_ub = A_ub_box
            b_ub = b_ub_box

        bounds = [(0.0, 1.0)] * (2 * p)

        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        return res.success and res.status == 0

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute tight bounding box via LP (respecting constraints)."""
        n = self.dimension
        p = self.num_generators

        if p == 0:
            return self.center.copy(), self.center.copy()

        lo = np.zeros(n)
        hi = np.zeros(n)

        bounds_eps = [(-1.0, 1.0)] * p

        # User constraints
        A_ub = self.A_cons if self.A_cons.size > 0 else None
        b_ub = self.b_cons if self.b_cons.size > 0 else None

        for i in range(n):
            g_i = self.generators[i, :]

            # Maximize g_i^T ε
            res_max = linprog(-g_i, A_ub=A_ub, b_ub=b_ub,
                              bounds=bounds_eps, method="highs")
            if res_max.success:
                hi[i] = self.center[i] + float(-res_max.fun)
            else:
                hi[i] = self.center[i] + float(np.sum(np.abs(g_i)))

            # Minimize g_i^T ε
            res_min = linprog(g_i, A_ub=A_ub, b_ub=b_ub,
                              bounds=bounds_eps, method="highs")
            if res_min.success:
                lo[i] = self.center[i] + float(res_min.fun)
            else:
                lo[i] = self.center[i] - float(np.sum(np.abs(g_i)))

        return lo, hi

    @property
    def volume_bound(self) -> float:
        lo, hi = self.bounding_box()
        w = np.maximum(hi - lo, 0.0)
        return float(np.prod(w))

    def __repr__(self) -> str:
        return (
            f"ConstrainedZonotope(dim={self.dimension}, "
            f"gens={self.num_generators}, "
            f"constraints={self.num_constraints})"
        )


# ---------------------------------------------------------------------------
# Product domain
# ---------------------------------------------------------------------------


class ProductDomain:
    """Cartesian product of domains for multi-agent states.

    Each agent's state is tracked in its own domain (zonotope or interval),
    and the joint state is the Cartesian product.
    """

    def __init__(
        self,
        components: Dict[str, Union[Zonotope, IntervalDomain]],
        component_dims: Dict[str, List[int]],
        total_dim: int,
    ) -> None:
        self.components = components
        self.component_dims = component_dims
        self.total_dim = total_dim

    @property
    def num_components(self) -> int:
        return len(self.components)

    def get_component(self, key: str) -> Union[Zonotope, IntervalDomain]:
        return self.components[key]

    def to_zonotope(self) -> Zonotope:
        """Combine all components into a single joint zonotope."""
        center = np.zeros(self.total_dim)
        gen_blocks: List[np.ndarray] = []

        for key, comp in self.components.items():
            dims = self.component_dims[key]

            if isinstance(comp, Zonotope):
                for i, d in enumerate(dims):
                    center[d] = comp.center[i]
                block = np.zeros((self.total_dim, comp.num_generators))
                for i, d in enumerate(dims):
                    block[d, :] = comp.generators[i, :]
                gen_blocks.append(block)

            elif isinstance(comp, IntervalDomain):
                for i, d in enumerate(dims):
                    center[d] = comp.center[i]
                    half = comp.widths[i] / 2.0
                    if half > 1e-15:
                        col = np.zeros((self.total_dim, 1))
                        col[d, 0] = half
                        gen_blocks.append(col)

        if gen_blocks:
            generators = np.hstack(gen_blocks)
        else:
            generators = np.zeros((self.total_dim, 0))

        return Zonotope(center=center, generators=generators)

    def to_interval(self) -> IntervalDomain:
        """Combine all components into a joint interval domain."""
        lower = np.zeros(self.total_dim)
        upper = np.zeros(self.total_dim)

        for key, comp in self.components.items():
            dims = self.component_dims[key]

            if isinstance(comp, Zonotope):
                lo, hi = comp.bounding_box()[:, 0], comp.bounding_box()[:, 1]
            else:
                lo, hi = comp.lower, comp.upper

            for i, d in enumerate(dims):
                lower[d] = lo[i]
                upper[d] = hi[i]

        return IntervalDomain(lower=lower, upper=upper)

    def join(self, other: "ProductDomain") -> "ProductDomain":
        """Component-wise join."""
        new_components: Dict[str, Union[Zonotope, IntervalDomain]] = {}
        for key in self.components:
            c1 = self.components[key]
            c2 = other.components[key]
            if isinstance(c1, Zonotope) and isinstance(c2, Zonotope):
                new_components[key] = c1.join(c2)
            elif isinstance(c1, IntervalDomain) and isinstance(c2, IntervalDomain):
                new_components[key] = c1.join(c2)
            else:
                # Mixed: convert both to zonotope and join
                z1 = c1 if isinstance(c1, Zonotope) else interval_to_zonotope(c1)
                z2 = c2 if isinstance(c2, Zonotope) else interval_to_zonotope(c2)
                new_components[key] = z1.join(z2)
        return ProductDomain(
            components=new_components,
            component_dims=self.component_dims,
            total_dim=self.total_dim,
        )

    def __repr__(self) -> str:
        parts = []
        for key, comp in self.components.items():
            parts.append(f"{key}:{type(comp).__name__}")
        return f"ProductDomain({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Domain factory
# ---------------------------------------------------------------------------


class DomainType(Enum):
    INTERVAL = auto()
    ZONOTOPE = auto()
    CONSTRAINED_ZONOTOPE = auto()
    HYBRID = auto()
    PRODUCT = auto()


class AbstractDomainFactory:
    """Create appropriate domain based on analysis requirements.

    Selection heuristics:
    - Small state dim (<= 10): zonotope is efficient
    - Large state dim with few shared dims: hybrid
    - Multiple independent agents: product
    - Need tight HB constraints: constrained zonotope
    - Baseline/fast: interval
    """

    @staticmethod
    def create_initial(
        lower: np.ndarray,
        upper: np.ndarray,
        domain_type: DomainType = DomainType.ZONOTOPE,
        zonotope_dims: Optional[List[int]] = None,
        agent_decomposition: Optional[Dict[str, List[int]]] = None,
    ) -> Union[Zonotope, IntervalDomain, ConstrainedZonotope, HybridDomain, ProductDomain]:
        """Create an initial abstract element of the requested type."""
        lower = np.asarray(lower, dtype=np.float64).ravel()
        upper = np.asarray(upper, dtype=np.float64).ravel()

        if domain_type == DomainType.INTERVAL:
            return IntervalDomain(lower=lower, upper=upper)

        if domain_type == DomainType.ZONOTOPE:
            return Zonotope.from_interval(lower, upper)

        if domain_type == DomainType.CONSTRAINED_ZONOTOPE:
            z = Zonotope.from_interval(lower, upper)
            return ConstrainedZonotope.from_zonotope(z)

        if domain_type == DomainType.HYBRID:
            if zonotope_dims is None:
                zonotope_dims = list(range(min(len(lower), 10)))
            iv = IntervalDomain(lower=lower, upper=upper)
            return HybridDomain.from_interval(iv, zonotope_dims)

        if domain_type == DomainType.PRODUCT:
            if agent_decomposition is None:
                raise ValueError(
                    "agent_decomposition required for PRODUCT domain"
                )
            components: Dict[str, Union[Zonotope, IntervalDomain]] = {}
            for agent_id, dims in agent_decomposition.items():
                components[agent_id] = Zonotope.from_interval(
                    lower[dims], upper[dims]
                )
            return ProductDomain(
                components=components,
                component_dims=agent_decomposition,
                total_dim=len(lower),
            )

        raise ValueError(f"Unknown domain type: {domain_type}")

    @staticmethod
    def recommend_domain(
        state_dim: int,
        num_agents: int = 1,
        num_shared_dims: int = 0,
        need_hb_constraints: bool = False,
    ) -> DomainType:
        """Recommend the best domain type given analysis characteristics."""
        if need_hb_constraints and state_dim <= 50:
            return DomainType.CONSTRAINED_ZONOTOPE

        if num_agents > 1 and num_shared_dims < state_dim // 2:
            if state_dim > 30:
                return DomainType.PRODUCT
            return DomainType.HYBRID

        if state_dim <= 100:
            return DomainType.ZONOTOPE

        return DomainType.INTERVAL
