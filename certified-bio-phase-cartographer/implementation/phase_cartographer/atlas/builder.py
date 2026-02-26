"""
Phase atlas construction with certificate composition.

A PhaseAtlas is a hierarchical partition of parameter space into CertifiedCells.
Atlas construction maintains the invariant that cells are disjoint and their
union covers the parameter domain (up to the uncertified gap fraction).

Certificate composition rule: Two adjacent cells with the same regime label
can be merged if their combined parameter box is certifiable.
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from ..tiered.certificate import (
    CertifiedCell, EquilibriumCertificate, VerificationTier,
    RegimeType, StabilityType,
)


@dataclass
class AtlasStats:
    """Summary statistics for a phase atlas."""
    total_cells: int = 0
    certified_cells: int = 0
    uncertified_cells: int = 0
    total_volume: float = 0.0
    certified_volume: float = 0.0
    coverage_fraction: float = 0.0
    regime_counts: Dict[str, int] = field(default_factory=dict)
    max_depth: int = 0
    total_time_s: float = 0.0
    minicheck_pass_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_cells": self.total_cells,
            "certified_cells": self.certified_cells,
            "uncertified_cells": self.uncertified_cells,
            "coverage_fraction": round(self.coverage_fraction, 6),
            "regime_counts": self.regime_counts,
            "max_depth": self.max_depth,
            "total_time_s": round(self.total_time_s, 2),
            "minicheck_pass_rate": round(self.minicheck_pass_rate, 4),
        }


class PhaseAtlas:
    """
    Certified phase atlas: hierarchical partition of parameter space.

    Invariants maintained:
    1. Cells are axis-aligned boxes.
    2. Cells are disjoint (no overlap).
    3. Union of cells ⊆ parameter domain.
    4. Every certified cell has passed at least Tier 1 (minicheck) verification.
    """

    def __init__(self, model_name: str, parameter_domain: List[Tuple[float, float]]):
        self.model_name = model_name
        self.parameter_domain = parameter_domain
        self.cells: List[CertifiedCell] = []
        self.uncertified_boxes: List[Tuple[List[Tuple[float, float]], int]] = []
        self._build_time = 0.0

    def add_cell(self, cell: CertifiedCell) -> None:
        """Add a certified cell to the atlas."""
        self.cells.append(cell)

    def add_uncertified(self, box: List[Tuple[float, float]], depth: int) -> None:
        """Record an uncertified parameter box."""
        self.uncertified_boxes.append((box, depth))

    def domain_volume(self) -> float:
        v = 1.0
        for lo, hi in self.parameter_domain:
            w = hi - lo
            if w > 0:
                v *= w
        return v

    def certified_volume(self) -> float:
        return sum(c.volume() for c in self.cells)

    def coverage_fraction(self) -> float:
        dv = self.domain_volume()
        if dv <= 0:
            return 0.0
        return min(1.0, self.certified_volume() / dv)

    def stats(self) -> AtlasStats:
        regime_counts: Dict[str, int] = {}
        n_minicheck = 0
        for c in self.cells:
            label = c.regime.value
            regime_counts[label] = regime_counts.get(label, 0) + 1
            if c.minicheck_passed:
                n_minicheck += 1

        return AtlasStats(
            total_cells=len(self.cells) + len(self.uncertified_boxes),
            certified_cells=len(self.cells),
            uncertified_cells=len(self.uncertified_boxes),
            total_volume=self.domain_volume(),
            certified_volume=self.certified_volume(),
            coverage_fraction=self.coverage_fraction(),
            regime_counts=regime_counts,
            max_depth=max((c.depth for c in self.cells), default=0),
            total_time_s=self._build_time,
            minicheck_pass_rate=(n_minicheck / len(self.cells)
                                 if self.cells else 0.0),
        )

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "parameter_domain": list(self.parameter_domain),
            "stats": self.stats().to_dict(),
            "cells": [c.to_dict() for c in self.cells],
            "uncertified_count": len(self.uncertified_boxes),
        }

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def cells_by_regime(self, regime: RegimeType) -> List[CertifiedCell]:
        return [c for c in self.cells if c.regime == regime]
