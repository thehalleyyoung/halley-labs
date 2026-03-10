"""
usability_oracle.core.types — Fundamental geometric, cost, and trajectory types.

Every value type is an immutable dataclass with full comparison, hashing,
serialisation (``to_dict`` / ``from_dict``), and arithmetic where appropriate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# NewType aliases — lightweight nominal types for static checking
# ---------------------------------------------------------------------------

AccessibilityNodeId = NewType("AccessibilityNodeId", str)
"""Globally unique identifier for a node in an accessibility tree."""

TaskId = NewType("TaskId", str)
"""Unique identifier for a user task specification."""

StateId = NewType("StateId", str)
"""Identifier for a state in the usability MDP."""

ActionId = NewType("ActionId", str)
"""Identifier for an action (transition) in the usability MDP."""


# ═══════════════════════════════════════════════════════════════════════════
# Point2D
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Point2D:
    """Two-dimensional point in screen coordinates (pixels)."""

    x: float
    y: float

    # --- geometric helpers -------------------------------------------------

    def distance(self, other: Point2D) -> float:
        """Euclidean distance to *other*."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def midpoint(self, other: Point2D) -> Point2D:
        """Midpoint between *self* and *other*."""
        return Point2D((self.x + other.x) / 2.0, (self.y + other.y) / 2.0)

    def translate(self, dx: float, dy: float) -> Point2D:
        """Return a new point shifted by *(dx, dy)*."""
        return Point2D(self.x + dx, self.y + dy)

    def scale(self, factor: float) -> Point2D:
        """Scale both coordinates by *factor* (about the origin)."""
        return Point2D(self.x * factor, self.y * factor)

    def manhattan_distance(self, other: Point2D) -> float:
        """L1 (Manhattan) distance to *other*."""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def angle_to(self, other: Point2D) -> float:
        """Angle in radians from self to other (atan2)."""
        return math.atan2(other.y - self.y, other.x - self.x)

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Point2D:
        return cls(x=float(d["x"]), y=float(d["y"]))

    @classmethod
    def origin(cls) -> Point2D:
        return cls(0.0, 0.0)

    def __add__(self, other: Any) -> Point2D:
        if isinstance(other, Point2D):
            return Point2D(self.x + other.x, self.y + other.y)
        return NotImplemented

    def __sub__(self, other: Any) -> Point2D:
        if isinstance(other, Point2D):
            return Point2D(self.x - other.x, self.y - other.y)
        return NotImplemented

    def __repr__(self) -> str:
        return f"Point2D({self.x:.2f}, {self.y:.2f})"


# ═══════════════════════════════════════════════════════════════════════════
# BoundingBox
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding box (screen coordinates, pixels)."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError(
                f"BoundingBox dimensions must be non-negative, "
                f"got width={self.width}, height={self.height}"
            )

    # --- derived properties ------------------------------------------------

    @property
    def center(self) -> Point2D:
        """Centre point of the bounding box."""
        return Point2D(self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def perimeter(self) -> float:
        return 2.0 * (self.width + self.height)

    @property
    def top_left(self) -> Point2D:
        return Point2D(self.x, self.y)

    @property
    def top_right(self) -> Point2D:
        return Point2D(self.x + self.width, self.y)

    @property
    def bottom_left(self) -> Point2D:
        return Point2D(self.x, self.y + self.height)

    @property
    def bottom_right(self) -> Point2D:
        return Point2D(self.x + self.width, self.y + self.height)

    @property
    def diagonal(self) -> float:
        """Length of the diagonal."""
        return math.hypot(self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """Width / height (inf if height is zero)."""
        if self.height == 0:
            return float("inf") if self.width > 0 else 1.0
        return self.width / self.height

    @property
    def min_dimension(self) -> float:
        return min(self.width, self.height)

    @property
    def max_dimension(self) -> float:
        return max(self.width, self.height)

    # --- spatial predicates ------------------------------------------------

    def contains(self, point: Point2D) -> bool:
        """Return *True* if *point* lies inside (inclusive)."""
        return (
            self.x <= point.x <= self.x + self.width
            and self.y <= point.y <= self.y + self.height
        )

    def contains_box(self, other: BoundingBox) -> bool:
        """Return *True* if *other* is entirely within *self*."""
        return (
            self.x <= other.x
            and self.y <= other.y
            and other.x + other.width <= self.x + self.width
            and other.y + other.height <= self.y + self.height
        )

    def overlaps(self, other: BoundingBox) -> bool:
        """Return *True* if the two boxes overlap at all."""
        if self.x + self.width <= other.x or other.x + other.width <= self.x:
            return False
        if self.y + self.height <= other.y or other.y + other.height <= self.y:
            return False
        return True

    def intersection(self, other: BoundingBox) -> Optional[BoundingBox]:
        """Return the intersection box or *None* if disjoint."""
        ix = max(self.x, other.x)
        iy = max(self.y, other.y)
        ix2 = min(self.x + self.width, other.x + other.width)
        iy2 = min(self.y + self.height, other.y + other.height)
        if ix2 <= ix or iy2 <= iy:
            return None
        return BoundingBox(ix, iy, ix2 - ix, iy2 - iy)

    def union(self, other: BoundingBox) -> BoundingBox:
        """Smallest bounding box enclosing both."""
        ux = min(self.x, other.x)
        uy = min(self.y, other.y)
        ux2 = max(self.x + self.width, other.x + other.width)
        uy2 = max(self.y + self.height, other.y + other.height)
        return BoundingBox(ux, uy, ux2 - ux, uy2 - uy)

    def distance_to(self, other: BoundingBox) -> float:
        """Minimum Euclidean distance between the surfaces of the two boxes."""
        dx = max(0.0, max(self.x - (other.x + other.width),
                          other.x - (self.x + self.width)))
        dy = max(0.0, max(self.y - (other.y + other.height),
                          other.y - (self.y + self.height)))
        return math.hypot(dx, dy)

    def iou(self, other: BoundingBox) -> float:
        """Intersection-over-union (Jaccard index) of the two boxes."""
        inter = self.intersection(other)
        if inter is None:
            return 0.0
        inter_area = inter.area
        union_area = self.area + other.area - inter_area
        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area

    def pad(self, padding: float) -> BoundingBox:
        """Expand the box by *padding* on all sides."""
        return BoundingBox(
            self.x - padding,
            self.y - padding,
            self.width + 2 * padding,
            self.height + 2 * padding,
        )

    def scale(self, factor: float) -> BoundingBox:
        """Scale around the centre."""
        cx, cy = self.center.x, self.center.y
        nw, nh = self.width * factor, self.height * factor
        return BoundingBox(cx - nw / 2, cy - nh / 2, nw, nh)

    # --- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BoundingBox:
        return cls(
            x=float(d["x"]),
            y=float(d["y"]),
            width=float(d["width"]),
            height=float(d["height"]),
        )

    def __repr__(self) -> str:
        return (
            f"BoundingBox(x={self.x:.1f}, y={self.y:.1f}, "
            f"w={self.width:.1f}, h={self.height:.1f})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Interval  —  closed real interval with arithmetic
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Interval:
    """Closed real interval [low, high] with interval-arithmetic operations.

    Used to represent uncertainty ranges for cognitive parameters (e.g.
    Fitts' *a* and *b* coefficients).  Arithmetic follows the standard
    interval-extension rules.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(
                f"Interval requires low <= high, got [{self.low}, {self.high}]"
            )

    # --- basic properties --------------------------------------------------

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2.0

    @property
    def radius(self) -> float:
        return self.width / 2.0

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def contains_interval(self, other: Interval) -> bool:
        return self.low <= other.low and other.high <= self.high

    def overlaps(self, other: Interval) -> bool:
        return self.low <= other.high and other.low <= self.high

    def clamp(self, value: float) -> float:
        """Clamp *value* to lie within the interval."""
        return max(self.low, min(self.high, value))

    # --- set operations ----------------------------------------------------

    def union(self, other: Interval) -> Interval:
        """Smallest interval covering both."""
        return Interval(min(self.low, other.low), max(self.high, other.high))

    def intersection(self, other: Interval) -> Optional[Interval]:
        """Intersection or *None* if disjoint."""
        lo = max(self.low, other.low)
        hi = min(self.high, other.high)
        if lo > hi:
            return None
        return Interval(lo, hi)

    # --- interval arithmetic -----------------------------------------------

    def __add__(self, other: Any) -> Interval:
        if isinstance(other, Interval):
            return Interval(self.low + other.low, self.high + other.high)
        if isinstance(other, (int, float)):
            return Interval(self.low + other, self.high + other)
        return NotImplemented

    def __radd__(self, other: Any) -> Interval:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Interval:
        if isinstance(other, Interval):
            return Interval(self.low - other.high, self.high - other.low)
        if isinstance(other, (int, float)):
            return Interval(self.low - other, self.high - other)
        return NotImplemented

    def __rsub__(self, other: Any) -> Interval:
        if isinstance(other, (int, float)):
            return Interval(other - self.high, other - self.low)
        return NotImplemented

    def __mul__(self, other: Any) -> Interval:
        if isinstance(other, Interval):
            products = (
                self.low * other.low,
                self.low * other.high,
                self.high * other.low,
                self.high * other.high,
            )
            return Interval(min(products), max(products))
        if isinstance(other, (int, float)):
            a, b = self.low * other, self.high * other
            return Interval(min(a, b), max(a, b))
        return NotImplemented

    def __rmul__(self, other: Any) -> Interval:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Interval:
        if isinstance(other, Interval):
            if other.low <= 0 <= other.high:
                raise ZeroDivisionError("Division by interval containing zero")
            inv = Interval(1.0 / other.high, 1.0 / other.low)
            return self * inv
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return self * (1.0 / other)
        return NotImplemented

    def __neg__(self) -> Interval:
        return Interval(-self.high, -self.low)

    def __abs__(self) -> Interval:
        if self.low >= 0:
            return Interval(self.low, self.high)
        if self.high <= 0:
            return Interval(-self.high, -self.low)
        return Interval(0.0, max(-self.low, self.high))

    def __pow__(self, n: int) -> Interval:
        """Integer power (handles even/odd correctly)."""
        if not isinstance(n, int) or n < 0:
            return NotImplemented
        if n == 0:
            return Interval(1.0, 1.0)
        if n == 1:
            return Interval(self.low, self.high)
        vals = (self.low ** n, self.high ** n)
        if n % 2 == 0 and self.low < 0 < self.high:
            return Interval(0.0, max(vals))
        return Interval(min(vals), max(vals))

    # --- comparison (partial order) ----------------------------------------

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Interval):
            return self.high < other.low
        if isinstance(other, (int, float)):
            return self.high < other
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        if isinstance(other, Interval):
            return self.high <= other.low
        if isinstance(other, (int, float)):
            return self.high <= other
        return NotImplemented

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, Interval):
            return self.low > other.high
        if isinstance(other, (int, float)):
            return self.low > other
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, Interval):
            return self.low >= other.high
        if isinstance(other, (int, float)):
            return self.low >= other
        return NotImplemented

    # --- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, float]:
        return {"low": self.low, "high": self.high}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Interval:
        return cls(low=float(d["low"]), high=float(d["high"]))

    @classmethod
    def point(cls, value: float) -> Interval:
        """Degenerate interval [v, v]."""
        return cls(value, value)

    def sample_uniform(self, rng: Optional[np.random.Generator] = None) -> float:
        """Draw a uniform random sample from the interval."""
        gen = rng or np.random.default_rng()
        return float(gen.uniform(self.low, self.high))

    def linspace(self, n: int) -> np.ndarray:
        """Return *n* evenly spaced values spanning the interval."""
        return np.linspace(self.low, self.high, n)

    def __repr__(self) -> str:
        return f"[{self.low:.4g}, {self.high:.4g}]"


# ═══════════════════════════════════════════════════════════════════════════
# CostTuple  —  (μ, σ², κ, λ) cognitive cost element
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class CostTuple:
    """Four-component cognitive cost element.

    * **mu** (μ) — expected cost (seconds or bits, context-dependent)
    * **sigma_sq** (σ²) — variance of the cost
    * **kappa** (κ) — skewness-related shape parameter (≥ 0)
    * **lambda_** (λ) — cognitive load weighting factor (≥ 0)

    The algebra supports *sequential composition* (series),
    *parallel composition* (independent channels), and
    *context application* (scaling by a rationality coefficient).
    """

    mu: float
    sigma_sq: float = 0.0
    kappa: float = 0.0
    lambda_: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma_sq < 0:
            raise ValueError(f"sigma_sq must be >= 0, got {self.sigma_sq}")
        if self.kappa < 0:
            raise ValueError(f"kappa must be >= 0, got {self.kappa}")
        if self.lambda_ < 0:
            raise ValueError(f"lambda_ must be >= 0, got {self.lambda_}")

    # --- algebra -----------------------------------------------------------

    def compose_sequential(self, other: CostTuple) -> CostTuple:
        """Sequential composition: costs add, variances add (independence)."""
        return CostTuple(
            mu=self.mu + other.mu,
            sigma_sq=self.sigma_sq + other.sigma_sq,
            kappa=self.kappa + other.kappa,
            lambda_=max(self.lambda_, other.lambda_),
        )

    def compose_parallel(self, other: CostTuple) -> CostTuple:
        """Parallel (multi-channel) composition: max of means, added variance."""
        return CostTuple(
            mu=max(self.mu, other.mu),
            sigma_sq=self.sigma_sq + other.sigma_sq,
            kappa=max(self.kappa, other.kappa),
            lambda_=self.lambda_ + other.lambda_,
        )

    def apply_context(self, beta: float) -> CostTuple:
        """Scale the cost by rationality parameter beta.

        Under the free-energy formulation the effective cost seen by the
        bounded-rational agent is ``(1/beta) * cost``.  A higher *beta* means
        the agent is more rational (lower perceived cost).
        """
        if beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta}")
        inv_beta = 1.0 / beta
        return CostTuple(
            mu=self.mu * inv_beta,
            sigma_sq=self.sigma_sq * (inv_beta ** 2),
            kappa=self.kappa * inv_beta,
            lambda_=self.lambda_,
        )

    @staticmethod
    def zero() -> CostTuple:
        """Additive identity."""
        return CostTuple(0.0, 0.0, 0.0, 0.0)

    # --- comparison helpers ------------------------------------------------

    @property
    def std(self) -> float:
        """Standard deviation."""
        return math.sqrt(self.sigma_sq)

    @property
    def coefficient_of_variation(self) -> float:
        """CV = sigma / mu  (undefined when mu = 0)."""
        if self.mu == 0:
            return float("inf") if self.sigma_sq > 0 else 0.0
        return self.std / abs(self.mu)

    @property
    def total_weighted_cost(self) -> float:
        """Convenience: mu + lambda * sqrt(sigma_sq) as a single scalar."""
        return self.mu + self.lambda_ * self.std

    def dominates(self, other: CostTuple) -> bool:
        """Pareto dominance: self <= other on every component, strict on >= 1."""
        le = (
            self.mu <= other.mu
            and self.sigma_sq <= other.sigma_sq
            and self.kappa <= other.kappa
            and self.lambda_ <= other.lambda_
        )
        strict = (
            self.mu < other.mu
            or self.sigma_sq < other.sigma_sq
            or self.kappa < other.kappa
            or self.lambda_ < other.lambda_
        )
        return le and strict

    # --- arithmetic --------------------------------------------------------

    def __add__(self, other: Any) -> CostTuple:
        if isinstance(other, CostTuple):
            return self.compose_sequential(other)
        return NotImplemented

    def __mul__(self, scalar: Any) -> CostTuple:
        if isinstance(scalar, (int, float)):
            return CostTuple(
                mu=self.mu * scalar,
                sigma_sq=self.sigma_sq * (scalar ** 2),
                kappa=self.kappa * abs(scalar),
                lambda_=self.lambda_,
            )
        return NotImplemented

    def __rmul__(self, scalar: Any) -> CostTuple:
        return self.__mul__(scalar)

    # --- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, float]:
        return {
            "mu": self.mu,
            "sigma_sq": self.sigma_sq,
            "kappa": self.kappa,
            "lambda": self.lambda_,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CostTuple:
        return cls(
            mu=float(d["mu"]),
            sigma_sq=float(d.get("sigma_sq", 0.0)),
            kappa=float(d.get("kappa", 0.0)),
            lambda_=float(d.get("lambda", d.get("lambda_", 0.0))),
        )

    def to_numpy(self) -> np.ndarray:
        return np.array([self.mu, self.sigma_sq, self.kappa, self.lambda_])

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> CostTuple:
        return cls(mu=float(arr[0]), sigma_sq=float(arr[1]),
                   kappa=float(arr[2]), lambda_=float(arr[3]))

    def __repr__(self) -> str:
        return (
            f"CostTuple(mu={self.mu:.4f}, sigma_sq={self.sigma_sq:.4f}, "
            f"kappa={self.kappa:.4f}, lambda_={self.lambda_:.4f})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TrajectoryStep  /  Trajectory
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """Single step in a task-completion trajectory through the UI MDP."""

    state_id: StateId
    action_id: ActionId
    cost: CostTuple
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "action_id": self.action_id,
            "cost": self.cost.to_dict(),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TrajectoryStep:
        return cls(
            state_id=StateId(d["state_id"]),
            action_id=ActionId(d["action_id"]),
            cost=CostTuple.from_dict(d["cost"]),
            timestamp=float(d.get("timestamp", 0.0)),
        )

    def __repr__(self) -> str:
        return (
            f"TrajectoryStep({self.state_id} --{self.action_id}--> "
            f"cost={self.cost.mu:.3f}s)"
        )


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Ordered sequence of trajectory steps with aggregate statistics."""

    steps: Tuple[TrajectoryStep, ...]
    total_cost: CostTuple
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_steps(
        cls,
        steps: Sequence[TrajectoryStep],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Trajectory:
        """Construct a trajectory, computing *total_cost* automatically."""
        total = CostTuple.zero()
        for s in steps:
            total = total.compose_sequential(s.cost)
        return cls(
            steps=tuple(steps),
            total_cost=total,
            metadata=metadata or {},
        )

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def duration(self) -> float:
        """Wall-clock duration (last timestamp - first timestamp)."""
        if len(self.steps) < 2:
            return 0.0
        return self.steps[-1].timestamp - self.steps[0].timestamp

    @property
    def state_ids(self) -> Tuple[StateId, ...]:
        return tuple(s.state_id for s in self.steps)

    @property
    def action_ids(self) -> Tuple[ActionId, ...]:
        return tuple(s.action_id for s in self.steps)

    @property
    def costs(self) -> Tuple[CostTuple, ...]:
        return tuple(s.cost for s in self.steps)

    @property
    def mean_step_cost(self) -> CostTuple:
        if not self.steps:
            return CostTuple.zero()
        n = len(self.steps)
        return CostTuple(
            mu=self.total_cost.mu / n,
            sigma_sq=self.total_cost.sigma_sq / (n ** 2),
            kappa=self.total_cost.kappa / n,
            lambda_=self.total_cost.lambda_,
        )

    @property
    def cost_variance(self) -> float:
        """Variance of step costs (mu component)."""
        if len(self.steps) < 2:
            return 0.0
        mus = np.array([s.cost.mu for s in self.steps])
        return float(np.var(mus, ddof=1))

    def slice(self, start: int, end: Optional[int] = None) -> Trajectory:
        """Return a sub-trajectory."""
        sliced = self.steps[start:end]
        return Trajectory.from_steps(sliced, metadata=dict(self.metadata))

    def append(self, step: TrajectoryStep) -> Trajectory:
        """Return a new trajectory with *step* appended."""
        new_steps = self.steps + (step,)
        return Trajectory.from_steps(new_steps, metadata=dict(self.metadata))

    # --- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "total_cost": self.total_cost.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Trajectory:
        steps = tuple(TrajectoryStep.from_dict(s) for s in d["steps"])
        return cls(
            steps=steps,
            total_cost=CostTuple.from_dict(d["total_cost"]),
            metadata=d.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return (
            f"Trajectory(steps={len(self.steps)}, "
            f"total_mu={self.total_cost.mu:.3f}s)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PolicyDistribution
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PolicyDistribution:
    """Mapping from state ids to action probability distributions.

    ``mapping[state_id]`` is a dict ``{action_id: probability}``.
    """

    mapping: Dict[StateId, Dict[ActionId, float]] = field(default_factory=dict)

    def action_probs(self, state: StateId) -> Dict[ActionId, float]:
        """Return action probabilities for *state* (empty dict if unknown)."""
        return self.mapping.get(state, {})

    def sample_action(
        self, state: StateId, rng: Optional[np.random.Generator] = None
    ) -> ActionId:
        """Sample an action from the policy at *state*."""
        probs = self.action_probs(state)
        if not probs:
            raise KeyError(f"No policy defined for state {state}")
        actions = list(probs.keys())
        probabilities = np.array(list(probs.values()))
        probabilities = probabilities / probabilities.sum()
        gen = rng or np.random.default_rng()
        idx = gen.choice(len(actions), p=probabilities)
        return actions[idx]

    def greedy_action(self, state: StateId) -> ActionId:
        """Return the most probable action at *state*."""
        probs = self.action_probs(state)
        if not probs:
            raise KeyError(f"No policy defined for state {state}")
        return max(probs, key=probs.get)  # type: ignore[arg-type]

    def entropy(self, state: StateId) -> float:
        """Shannon entropy of the action distribution at *state* (nats)."""
        probs = self.action_probs(state)
        if not probs:
            return 0.0
        ps = np.array(list(probs.values()))
        ps = ps[ps > 0]
        return float(-np.sum(ps * np.log(ps)))

    def max_entropy(self, state: StateId) -> float:
        """Maximum possible entropy at *state* (uniform distribution)."""
        n = len(self.action_probs(state))
        return math.log(n) if n > 0 else 0.0

    def kl_divergence(self, other: PolicyDistribution, state: StateId) -> float:
        """KL(self || other) at *state*.  Returns inf if support mismatch."""
        p = self.action_probs(state)
        q = other.action_probs(state)
        if not p:
            return 0.0
        kl = 0.0
        for action, p_val in p.items():
            if p_val <= 0:
                continue
            q_val = q.get(action, 0.0)
            if q_val <= 0:
                return float("inf")
            kl += p_val * math.log(p_val / q_val)
        return kl

    def expected_kl(
        self,
        other: PolicyDistribution,
        state_weights: Optional[Dict[StateId, float]] = None,
    ) -> float:
        """Weighted average KL across all states."""
        states = set(self.mapping.keys())
        if state_weights is None:
            state_weights = {s: 1.0 / max(len(states), 1) for s in states}
        total = 0.0
        for s, w in state_weights.items():
            total += w * self.kl_divergence(other, s)
        return total

    def mean_entropy(self) -> float:
        """Average entropy across all states."""
        if not self.mapping:
            return 0.0
        return sum(self.entropy(s) for s in self.mapping) / len(self.mapping)

    @property
    def num_states(self) -> int:
        return len(self.mapping)

    @property
    def num_actions(self) -> int:
        """Total number of unique actions across all states."""
        actions: set = set()
        for ap in self.mapping.values():
            actions.update(ap.keys())
        return len(actions)

    def set_action_probs(self, state: StateId, probs: Dict[ActionId, float]) -> None:
        """Set (overwrite) the action distribution for *state*."""
        self.mapping[state] = dict(probs)

    def normalize(self, state: StateId) -> None:
        """Renormalize action probabilities at *state* to sum to 1."""
        probs = self.mapping.get(state)
        if probs:
            total = sum(probs.values())
            if total > 0:
                self.mapping[state] = {a: p / total for a, p in probs.items()}

    # --- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mapping": {
                str(s): {str(a): p for a, p in ap.items()}
                for s, ap in self.mapping.items()
            }
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PolicyDistribution:
        mapping: Dict[StateId, Dict[ActionId, float]] = {}
        raw = d.get("mapping", d)
        for s_str, ap in raw.items():
            mapping[StateId(s_str)] = {
                ActionId(a): float(p) for a, p in ap.items()
            }
        return cls(mapping=mapping)

    def __repr__(self) -> str:
        return f"PolicyDistribution(states={self.num_states})"


__all__ = [
    "AccessibilityNodeId",
    "TaskId",
    "StateId",
    "ActionId",
    "Point2D",
    "BoundingBox",
    "Interval",
    "CostTuple",
    "TrajectoryStep",
    "Trajectory",
    "PolicyDistribution",
]
