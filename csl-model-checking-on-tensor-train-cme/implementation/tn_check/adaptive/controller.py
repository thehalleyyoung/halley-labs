"""
Adaptive rank controller for TT bond dimensions.

Greedy-doubling heuristic with per-bond singular-value monitoring:
- Start with initial bond dimension χ₀
- Monitor truncation error and singular value spectra
- If error > 2ε_target: double bond dimensions, retry
- If smallest kept singular value < threshold: consider shrinking
- Geometric retry guarantees eventual convergence
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tn_check.config import AdaptiveConfig

logger = logging.getLogger(__name__)


@dataclass
class RankDecision:
    """Decision about bond dimensions for next step."""
    bond_dims: list[int]
    action: str  # "keep", "double", "shrink"
    reason: str
    error_estimate: float = 0.0


class AdaptiveRankController:
    """
    Adaptive rank controller using greedy-doubling with convergence monitoring.

    Strategy:
    1. Start with initial_bond_dim at all bonds
    2. After each integration step, check truncation error
    3. If error > 2 * target: double all bond dims, re-integrate
    4. If all singular value ratios < threshold: try shrinking
    5. Per-bond monitoring: only grow bonds that need it

    Convergence guarantee: geometric doubling means at most
    O(log(χ_opt / χ_0)) retries before reaching sufficient rank.
    """

    def __init__(self, config: Optional[AdaptiveConfig] = None):
        self.config = config or AdaptiveConfig()
        self.current_dims: list[int] = []
        self.history: list[RankDecision] = []
        self.retry_count: int = 0

    def initialize(self, num_bonds: int) -> list[int]:
        """Initialize bond dimensions."""
        self.current_dims = [self.config.initial_bond_dim] * num_bonds
        return self.current_dims.copy()

    def decide(
        self,
        truncation_error: float,
        target_error: float,
        singular_value_spectra: Optional[list[np.ndarray]] = None,
    ) -> RankDecision:
        """
        Decide whether to adjust bond dimensions.

        Args:
            truncation_error: Current step truncation error.
            target_error: Target error tolerance.
            singular_value_spectra: Per-bond singular value arrays.

        Returns:
            RankDecision with new bond dimensions and action.
        """
        if truncation_error > 2 * target_error:
            # Error too high: double bond dimensions
            new_dims = [
                min(d * 2, self.config.max_bond_dim)
                for d in self.current_dims
            ]
            self.retry_count += 1
            decision = RankDecision(
                bond_dims=new_dims,
                action="double",
                reason=f"Error {truncation_error:.2e} > 2*target {target_error:.2e}",
                error_estimate=truncation_error,
            )
        elif (
            singular_value_spectra is not None
            and all(
                spectrum[-1] / spectrum[0] < self.config.min_singular_value_ratio
                if len(spectrum) > 1 and spectrum[0] > 1e-300
                else True
                for spectrum in singular_value_spectra
            )
        ):
            # All bonds have negligible smallest singular values: try shrinking
            new_dims = [
                max(int(d * self.config.shrink_factor), 1)
                for d in self.current_dims
            ]
            decision = RankDecision(
                bond_dims=new_dims,
                action="shrink",
                reason="All bonds have negligible trailing singular values",
                error_estimate=truncation_error,
            )
        else:
            # Keep current dimensions
            decision = RankDecision(
                bond_dims=self.current_dims.copy(),
                action="keep",
                reason="Error within tolerance",
                error_estimate=truncation_error,
            )

        self.current_dims = decision.bond_dims.copy()
        self.history.append(decision)
        return decision

    def per_bond_adjust(
        self,
        bond_idx: int,
        spectrum: np.ndarray,
        target_error: float,
    ) -> int:
        """
        Adjust a single bond dimension based on its singular value spectrum.

        Args:
            bond_idx: Bond index.
            spectrum: Singular values at this bond.
            target_error: Target truncation error per bond.

        Returns:
            New bond dimension for this bond.
        """
        if len(spectrum) == 0:
            return self.current_dims[bond_idx]

        # Find minimal rank that captures target error
        total_sq = np.sum(spectrum ** 2)
        cumsum = np.cumsum(spectrum[::-1] ** 2)[::-1]

        for k in range(1, len(spectrum) + 1):
            remaining = cumsum[k] if k < len(cumsum) else 0.0
            if np.sqrt(remaining / max(total_sq, 1e-300)) < target_error:
                new_dim = min(k, self.config.max_bond_dim)
                self.current_dims[bond_idx] = new_dim
                return new_dim

        new_dim = min(len(spectrum), self.config.max_bond_dim)
        self.current_dims[bond_idx] = new_dim
        return new_dim
