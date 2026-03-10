"""
usability_oracle.fragility.models — Data structures for cognitive fragility analysis.

Defines :class:`FragilityResult`, :class:`CliffLocation`, and
:class:`SensitivityResult` — the core data structures for describing
how sensitive usability cost is to changes in the rationality parameter β.

A *fragile* interface is one where small changes in β produce large
cost changes — meaning the usability is unstable w.r.t. the user's
cognitive capacity.  A *robust* interface maintains similar usability
across a wide range of β values.

References
----------
- Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a theory of
  decision-making with information-processing costs. *Proc. R. Soc. A*, 469.
- Tversky, A. & Kahneman, D. (1974). Judgment under uncertainty:
  Heuristics and biases. *Science*, 185(4157).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Interval type (lightweight, for robustness intervals)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Interval:
    """A closed real interval [lo, hi].

    Used to represent the β range over which a verdict is stable
    (the *robustness interval*).

    Attributes
    ----------
    lo : float
        Lower bound.
    hi : float
        Upper bound.
    """

    lo: float
    hi: float

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def mid(self) -> float:
        return (self.lo + self.hi) / 2.0

    def contains(self, x: float) -> bool:
        return self.lo <= x <= self.hi

    def __repr__(self) -> str:
        return f"[{self.lo:.4f}, {self.hi:.4f}]"


# ---------------------------------------------------------------------------
# CliffLocation
# ---------------------------------------------------------------------------

@dataclass
class CliffLocation:
    """A location where usability cost changes dramatically.

    A *cliff* in the cost-vs-β curve indicates a phase transition in
    the user's optimal strategy — typically caused by a policy switch
    where the bounded-rational agent flips from one action to another.

    Attributes
    ----------
    beta_star : float
        β value at the center of the cliff.
    cost_before : float
        Cost just below β* (at β* − ε).
    cost_after : float
        Cost just above β* (at β* + ε).
    affected_states : list[str]
        State IDs where the policy changes at this cliff.
    gradient : float
        Rate of change |dC/dβ| at the cliff (higher = steeper).
    cliff_type : str
        Classification: ``"policy_switch"``, ``"state_collapse"``,
        or ``"information_cliff"``.
    severity : float
        Normalized severity score in [0, 1].
    """

    beta_star: float = 0.0
    cost_before: float = 0.0
    cost_after: float = 0.0
    affected_states: list[str] = field(default_factory=list)
    gradient: float = 0.0
    cliff_type: str = "policy_switch"
    severity: float = 0.0

    @property
    def cost_jump(self) -> float:
        """Absolute cost change across the cliff."""
        return abs(self.cost_after - self.cost_before)

    @property
    def relative_jump(self) -> float:
        """Relative cost change (fraction of the larger cost)."""
        denom = max(abs(self.cost_before), abs(self.cost_after), 1e-12)
        return self.cost_jump / denom


# ---------------------------------------------------------------------------
# SensitivityResult
# ---------------------------------------------------------------------------

@dataclass
class SensitivityResult:
    """Sensitivity of usability cost to a single model parameter.

    Attributes
    ----------
    parameter_name : str
        Name of the parameter (e.g., ``"beta"``, ``"fitts_b"``).
    sensitivity : float
        Sensitivity index (higher = more sensitive).  For OAT analysis
        this is |ΔC/Δp|; for Sobol analysis this is the first-order
        Sobol index S_i.
    direction : str
        Whether increasing the parameter increases (``"positive"``) or
        decreases (``"negative"``) the cost, or has mixed effects
        (``"mixed"``).
    confidence_interval : tuple[float, float]
        95% CI for the sensitivity estimate.
    method : str
        Analysis method used (``"oat"``, ``"sobol"``, ``"morris"``).
    """

    parameter_name: str = ""
    sensitivity: float = 0.0
    direction: str = "mixed"
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    method: str = "oat"


# ---------------------------------------------------------------------------
# FragilityResult
# ---------------------------------------------------------------------------

@dataclass
class FragilityResult:
    """Complete result of a cognitive fragility analysis.

    Attributes
    ----------
    fragility_score : float
        Overall fragility score in [0, 1], where 0 = perfectly robust
        and 1 = maximally fragile.  Computed as the normalized maximum
        gradient of the cost-vs-β curve.
    cliff_locations : list[CliffLocation]
        Detected cliff locations where cost changes dramatically.
    beta_sensitivity : dict[str, float]
        Sensitivity of each cost component to β.
    robustness_interval : Interval
        The β range over which the verdict is stable.
    population_impact : dict[str, float]
        Cost impact at different population percentiles (5th, 25th, 50th,
        75th, 95th), representing different user cognitive capacities.
    cost_curve : list[tuple[float, float]]
        The cost-vs-β curve as ``(β, cost)`` pairs.
    metadata : dict
        Additional analysis metadata.
    """

    fragility_score: float = 0.0
    cliff_locations: list[CliffLocation] = field(default_factory=list)
    beta_sensitivity: dict[str, float] = field(default_factory=dict)
    robustness_interval: Interval = field(
        default_factory=lambda: Interval(0.0, float("inf"))
    )
    population_impact: dict[str, float] = field(default_factory=dict)
    cost_curve: list[tuple[float, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_fragile(self) -> bool:
        """True if the fragility score exceeds a moderate threshold."""
        return self.fragility_score > 0.5

    @property
    def n_cliffs(self) -> int:
        return len(self.cliff_locations)

    @property
    def worst_cliff(self) -> Optional[CliffLocation]:
        """Return the cliff with the highest severity."""
        if not self.cliff_locations:
            return None
        return max(self.cliff_locations, key=lambda c: c.severity)


# ---------------------------------------------------------------------------
# InclusiveDesignResult
# ---------------------------------------------------------------------------

@dataclass
class InclusiveDesignResult:
    """Result of an inclusive design analysis.

    Attributes
    ----------
    per_profile_costs : dict[str, float]
        Mean task cost for each population profile.
    equity_gap : float
        Maximum cost gap between any two profiles.
    most_affected_profile : str
        Profile with the highest cost.
    least_affected_profile : str
        Profile with the lowest cost.
    recommendations : list[str]
        Actionable recommendations for more inclusive design.
    population_coverage : float
        Fraction of the population that can complete the task within
        a reasonable cost threshold.
    exclusive_elements : dict[str, list[str]]
        Per-profile list of UI elements that are exclusionary.
    """

    per_profile_costs: dict[str, float] = field(default_factory=dict)
    equity_gap: float = 0.0
    most_affected_profile: str = ""
    least_affected_profile: str = ""
    recommendations: list[str] = field(default_factory=list)
    population_coverage: float = 1.0
    exclusive_elements: dict[str, list[str]] = field(default_factory=dict)
