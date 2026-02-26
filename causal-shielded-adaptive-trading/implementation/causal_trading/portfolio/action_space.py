"""
Discrete action space management for causal-shielded trading.

Defines the mapping between discrete position levels and continuous
position sizes, and provides utilities for computing shield-restricted
action subsets.

The default action space uses 7 levels:

    -3  -2  -1   0  +1  +2  +3
    ──────────────────────────────
    max   partial   neutral   partial   max
    short  short   (flat)     long    long

Each level maps linearly to a weight in [-1, +1].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ActionInfo:
    """Metadata for a single discrete action."""
    level: int
    weight: float
    label: str
    max_position_pct: float  # fraction of capital


class ActionSpace:
    """Discrete position-level action space with shield integration.

    Parameters
    ----------
    levels : sequence of int
        Sorted list of position levels (e.g. ``[-3,-2,-1,0,1,2,3]``).
    max_position : float
        Maximum absolute portfolio weight (maps to the most extreme level).
    position_sizing : str
        ``"linear"`` or ``"quadratic"`` mapping from level to weight.
    min_holding_period : int
        Minimum number of steps before a position change is allowed.
    max_step_size : int
        Maximum level change in a single step.
    """

    DEFAULT_LEVELS = [-3, -2, -1, 0, 1, 2, 3]

    def __init__(
        self,
        levels: Optional[Sequence[int]] = None,
        max_position: float = 1.0,
        position_sizing: str = "linear",
        min_holding_period: int = 1,
        max_step_size: int = 7,
    ) -> None:
        self.levels: List[int] = sorted(levels or self.DEFAULT_LEVELS)
        self.max_position = max_position
        self.position_sizing = position_sizing
        self.min_holding_period = min_holding_period
        self.max_step_size = max_step_size

        if len(self.levels) < 2:
            raise ValueError("Need at least 2 action levels.")

        self._max_abs = max(abs(l) for l in self.levels)
        self._action_map: Dict[int, ActionInfo] = self._build_map()
        self._steps_since_change: int = 0
        self._current_level: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
        return len(self.levels)

    @property
    def actions(self) -> List[int]:
        return list(self.levels)

    def level_to_weight(self, level: int) -> float:
        """Map a discrete level to a continuous portfolio weight."""
        if level not in self._action_map:
            raise ValueError(f"Unknown action level {level}")
        return self._action_map[level].weight

    def weight_to_nearest_level(self, weight: float) -> int:
        """Snap a continuous weight to the nearest discrete level."""
        best = self.levels[0]
        best_dist = abs(self.level_to_weight(best) - weight)
        for lvl in self.levels[1:]:
            d = abs(self.level_to_weight(lvl) - weight)
            if d < best_dist:
                best, best_dist = lvl, d
        return best

    def get_permitted_actions(
        self,
        current_level: int,
        shield_mask: Optional[NDArray] = None,
    ) -> List[int]:
        """Return actions available from *current_level*.

        Applies:
        1. Maximum step-size constraint.
        2. Minimum holding-period constraint.
        3. Shield mask (boolean array aligned with ``self.levels``).

        Parameters
        ----------
        current_level : int
            Agent's current position level.
        shield_mask : (n_actions,) bool array or None
            ``True`` for actions the shield permits.

        Returns
        -------
        List of permitted action levels.
        """
        candidates: List[int] = []

        for lvl in self.levels:
            step = abs(lvl - current_level)
            if step > self.max_step_size:
                continue
            candidates.append(lvl)

        if self._steps_since_change < self.min_holding_period and current_level in candidates:
            candidates = [current_level]

        if shield_mask is not None:
            mask = np.asarray(shield_mask, dtype=bool)
            if mask.shape[0] != len(self.levels):
                raise ValueError("shield_mask length must match n_actions")
            level_set = set(self.levels[i] for i, m in enumerate(mask) if m)
            candidates = [c for c in candidates if c in level_set]

        if not candidates:
            candidates = [current_level]

        return candidates

    def apply_action(self, new_level: int) -> float:
        """Record a transition and return the new portfolio weight.

        Updates internal holding-period tracking.
        """
        if new_level != self._current_level:
            self._steps_since_change = 0
        else:
            self._steps_since_change += 1
        self._current_level = new_level
        return self.level_to_weight(new_level)

    def get_transition_cost_matrix(
        self, linear_cost: float = 0.001, quadratic_cost: float = 0.0005
    ) -> NDArray:
        """Build (n_actions × n_actions) transition cost matrix.

        ``C[i, j]`` is the cost of moving from level ``levels[i]`` to
        ``levels[j]``.
        """
        n = self.n_actions
        C = np.zeros((n, n), dtype=np.float64)
        for i, li in enumerate(self.levels):
            for j, lj in enumerate(self.levels):
                delta = abs(self.level_to_weight(lj) - self.level_to_weight(li))
                C[i, j] = linear_cost * delta + quadratic_cost * delta ** 2
        return C

    def get_action_info(self, level: int) -> ActionInfo:
        """Return metadata for a single action level."""
        return self._action_map[level]

    def reset(self) -> None:
        """Reset internal state."""
        self._current_level = 0
        self._steps_since_change = 0

    def describe(self) -> str:
        """Human-readable description of the action space."""
        lines = [f"ActionSpace with {self.n_actions} levels:"]
        for lvl in self.levels:
            info = self._action_map[lvl]
            lines.append(
                f"  {info.label:>12s}  level={lvl:+d}  weight={info.weight:+.3f}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_map(self) -> Dict[int, ActionInfo]:
        action_map: Dict[int, ActionInfo] = {}
        for lvl in self.levels:
            w = self._compute_weight(lvl)
            label = self._level_label(lvl)
            action_map[lvl] = ActionInfo(
                level=lvl,
                weight=w,
                label=label,
                max_position_pct=abs(w) * 100.0,
            )
        return action_map

    def _compute_weight(self, level: int) -> float:
        if self._max_abs == 0:
            return 0.0
        normalised = level / self._max_abs
        if self.position_sizing == "quadratic":
            return float(np.sign(normalised) * normalised ** 2) * self.max_position
        return normalised * self.max_position

    @staticmethod
    def _level_label(level: int) -> str:
        if level == 0:
            return "flat"
        direction = "long" if level > 0 else "short"
        magnitude = abs(level)
        if magnitude == 1:
            size = "small"
        elif magnitude == 2:
            size = "medium"
        else:
            size = "large"
        return f"{size}_{direction}"
