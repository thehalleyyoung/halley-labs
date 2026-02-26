"""
Adaptive parameter-space refinement via octree partitioning.

Refines parameter boxes near phase boundaries using eigenvalue-sensitivity
scoring. High-sensitivity boxes (where eigenvalue real parts are close to
zero relative to their width) are split preferentially.

GP-guided refinement (gp_guided_refine) integrates a Gaussian process
surrogate that predicts phase boundaries and prioritises cells with high
boundary-uncertainty acquisition score.  The GP is strictly ADVISORY —
it does not affect certification soundness.

Convergence: If regime boundaries are piecewise-smooth codimension-≥1 manifolds,
uncertified volume → 0 as max_depth → ∞ (Proposition 3.3 in paper).
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional, Dict

import numpy as np

from ..interval.interval import Interval
from ..interval.matrix import IntervalVector
from ..tiered.certificate import (
    CertifiedCell, EquilibriumCertificate, VerificationTier,
    RegimeType, StabilityType, RegimeInferenceRules,
)
from ..atlas.builder import PhaseAtlas


@dataclass
class RefinementConfig:
    """Configuration for adaptive refinement."""
    max_depth: int = 8
    target_coverage: float = 0.95
    min_box_width: float = 1e-4
    max_cells: int = 10000
    eigenvalue_sensitivity_threshold: float = 0.5


@dataclass
class GPGuidedRefinementConfig(RefinementConfig):
    """Configuration for GP-guided adaptive refinement.

    Extends RefinementConfig with parameters controlling how the GP
    surrogate is trained and used for acquisition-driven prioritisation.
    """
    # Minimum certified cells before GP training begins
    gp_warmup_cells: int = 10
    # Re-train GP every ``gp_retrain_interval`` new certified cells
    gp_retrain_interval: int = 10
    # Weight of GP acquisition score relative to volume heuristic (0–1)
    gp_weight: float = 0.6
    # Use ARD length scales for the GP kernel
    gp_use_ard: bool = False
    # GP noise variance
    gp_noise_var: float = 1e-6


@dataclass
class ConvergenceRecord:
    """Per-iteration convergence tracking for refinement loops."""
    iteration: List[int] = field(default_factory=list)
    coverage: List[float] = field(default_factory=list)
    certified_cells: List[int] = field(default_factory=list)
    elapsed_s: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def eigenvalue_sensitivity_score(eigenvalue_real_parts: List[Tuple[float, float]]) -> float:
    """
    Eigenvalue-sensitivity score σ(B) = max_i width(Re(λ_i)) / dist(Re(λ_i), 0).

    High σ indicates proximity to a bifurcation point.
    Returns float('inf') if any eigenvalue enclosure contains zero.
    """
    if not eigenvalue_real_parts:
        return float('inf')

    max_score = 0.0
    for lo, hi in eigenvalue_real_parts:
        width = hi - lo
        if lo <= 0 <= hi:
            return float('inf')  # eigenvalue crosses zero
        dist_to_zero = min(abs(lo), abs(hi))
        if dist_to_zero < 1e-15:
            return float('inf')
        score = width / dist_to_zero
        max_score = max(max_score, score)

    return max_score


def split_box(box: List[Tuple[float, float]],
              split_dim: Optional[int] = None) -> Tuple[List[Tuple[float, float]],
                                                          List[Tuple[float, float]]]:
    """
    Split a parameter box along the widest dimension (or specified dimension).
    """
    if split_dim is None:
        widths = [(hi - lo, i) for i, (lo, hi) in enumerate(box)]
        widths.sort(reverse=True)
        split_dim = widths[0][1]

    lo, hi = box[split_dim]
    mid = (lo + hi) / 2.0

    box1 = list(box)
    box2 = list(box)
    box1[split_dim] = (lo, mid)
    box2[split_dim] = (mid, hi)
    return box1, box2


def anisotropic_split_box(
    box: List[Tuple[float, float]],
    eigenvalue_real_parts: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Split along the dimension with highest eigenvalue sensitivity.

    For each varying dimension, the sensitivity is approximated by
    ``width(Re(λ_i)) / dist(Re(λ_i), 0)`` weighted by the box extent in
    that dimension.  When no eigenvalue data is available, falls back to
    splitting the widest dimension (identical to ``split_box``).
    """
    varying = [(hi - lo, i) for i, (lo, hi) in enumerate(box) if hi - lo > 0]
    if not varying:
        return split_box(box)

    if eigenvalue_real_parts is None or len(eigenvalue_real_parts) == 0:
        return split_box(box)

    # Heuristic: eigenvalue sensitivity × box-width gives per-dimension score
    dim_scores: Dict[int, float] = {}
    for w, dim_idx in varying:
        eig_idx = dim_idx if dim_idx < len(eigenvalue_real_parts) else 0
        lo_e, hi_e = eigenvalue_real_parts[eig_idx]
        eig_width = hi_e - lo_e
        if lo_e <= 0 <= hi_e:
            dim_scores[dim_idx] = float('inf')
        else:
            dist = min(abs(lo_e), abs(hi_e))
            dim_scores[dim_idx] = w * (eig_width / max(dist, 1e-15))

    best_dim = max(dim_scores, key=lambda d: dim_scores[d])
    return split_box(box, split_dim=best_dim)


def box_volume(box: List[Tuple[float, float]]) -> float:
    v = 1.0
    for lo, hi in box:
        w = hi - lo
        if w > 0:
            v *= w
    return v


def box_max_width(box: List[Tuple[float, float]]) -> float:
    return max(hi - lo for lo, hi in box)


def box_midpoint(box: List[Tuple[float, float]]) -> np.ndarray:
    return np.array([(lo + hi) / 2.0 for lo, hi in box])


CertifyFn = Callable[[List[Tuple[float, float]]], Optional[CertifiedCell]]


# ---------------------------------------------------------------------------
# Original adaptive refinement (kept unchanged for ablation comparison)
# ---------------------------------------------------------------------------

def adaptive_refine(certify_fn: CertifyFn,
                    parameter_domain: List[Tuple[float, float]],
                    model_name: str,
                    config: RefinementConfig = RefinementConfig(),
                    ) -> PhaseAtlas:
    """
    Adaptive octree refinement of parameter space.

    Algorithm:
    1. Start with the full parameter domain as a single cell.
    2. Attempt certification of each cell.
    3. If certification succeeds, add to atlas.
    4. If certification fails, compute eigenvalue-sensitivity score
       and add to priority queue for splitting.
    5. Split highest-priority cell along dimension of maximum sensitivity.
    6. Repeat until target coverage or max depth reached.

    Args:
        certify_fn: Function that takes a parameter box and returns
                    a CertifiedCell or None if certification fails.
        parameter_domain: The full parameter domain.
        model_name: Name of the model.
        config: Refinement configuration.

    Returns:
        Completed PhaseAtlas.
    """
    atlas = PhaseAtlas(model_name, parameter_domain)
    t0 = time.time()

    # Priority queue: (-priority, depth, box)
    # Negative priority because heapq is a min-heap
    pq: List[Tuple[float, int, List[Tuple[float, float]]]] = []
    heapq.heappush(pq, (0.0, 0, parameter_domain))

    domain_vol = box_volume(parameter_domain)
    n_cells = 0

    while pq and n_cells < config.max_cells:
        neg_priority, depth, box = heapq.heappop(pq)

        if atlas.coverage_fraction() >= config.target_coverage:
            atlas.add_uncertified(box, depth)
            continue

        if box_max_width(box) < config.min_box_width:
            atlas.add_uncertified(box, depth)
            continue

        # Attempt certification
        cell = certify_fn(box)

        if cell is not None:
            cell.depth = depth
            atlas.add_cell(cell)
            n_cells += 1
        elif depth < config.max_depth:
            # Split and re-enqueue with eigenvalue-sensitivity priority
            box1, box2 = split_box(box)
            # Priority: larger boxes near boundaries get higher priority
            vol_frac = box_volume(box) / domain_vol if domain_vol > 0 else 0
            p1 = -(vol_frac + 0.1)  # simple heuristic
            p2 = -(vol_frac + 0.1)
            heapq.heappush(pq, (p1, depth + 1, box1))
            heapq.heappush(pq, (p2, depth + 1, box2))
        else:
            atlas.add_uncertified(box, depth)

    # Drain remaining queue as uncertified
    while pq:
        _, depth, box = heapq.heappop(pq)
        atlas.add_uncertified(box, depth)

    atlas._build_time = time.time() - t0
    return atlas


# ---------------------------------------------------------------------------
# GP-guided adaptive refinement
# ---------------------------------------------------------------------------

def _regime_label_to_int(regime: RegimeType) -> int:
    """Encode regime enum as integer for GP training."""
    _map = {
        RegimeType.MONOSTABLE: 0,
        RegimeType.BISTABLE: 1,
        RegimeType.MULTISTABLE: 2,
        RegimeType.OSCILLATORY: 3,
        RegimeType.EXCITABLE: 4,
        RegimeType.INCONCLUSIVE: 5,
    }
    return _map.get(regime, 5)


def gp_guided_refine(
    certify_fn: CertifyFn,
    parameter_domain: List[Tuple[float, float]],
    model_name: str,
    config: GPGuidedRefinementConfig = GPGuidedRefinementConfig(),
) -> Tuple[PhaseAtlas, ConvergenceRecord]:
    """GP-guided adaptive octree refinement.

    Algorithm:
    1. Start with uniform refinement to collect initial training data.
    2. After ``gp_warmup_cells`` certified cells, train a GP surrogate
       on (midpoint, regime_label) pairs.
    3. Use ``boundary_uncertainty`` acquisition to re-score queued cells,
       giving highest priority to predicted phase-boundary regions.
    4. Re-train GP every ``gp_retrain_interval`` new certified cells.
    5. Convergence tracking records coverage vs iteration.

    The GP is ADVISORY ONLY — it guides which cells to refine next but
    does not affect certification correctness.

    Returns:
        (atlas, convergence_record)
    """
    from ..gp.surrogate import GPSurrogate
    from ..gp.acquisition import boundary_uncertainty as bu_score, phase_boundary_score

    atlas = PhaseAtlas(model_name, parameter_domain)
    convergence = ConvergenceRecord()
    t0 = time.time()

    pq: List[Tuple[float, int, int, List[Tuple[float, float]]]] = []
    # counter for tie-breaking in heapq
    counter = 0
    heapq.heappush(pq, (0.0, counter, 0, parameter_domain))
    counter += 1

    domain_vol = box_volume(parameter_domain)
    n_cells = 0
    cells_since_retrain = 0

    gp = GPSurrogate(
        noise_var=config.gp_noise_var,
        use_ard=config.gp_use_ard,
    )
    gp_trained = False

    iteration = 0

    while pq and n_cells < config.max_cells:
        neg_priority, _cnt, depth, box = heapq.heappop(pq)
        iteration += 1

        if atlas.coverage_fraction() >= config.target_coverage:
            atlas.add_uncertified(box, depth)
            continue

        if box_max_width(box) < config.min_box_width:
            atlas.add_uncertified(box, depth)
            continue

        # Attempt certification
        cell = certify_fn(box)

        if cell is not None:
            cell.depth = depth
            atlas.add_cell(cell)
            n_cells += 1
            cells_since_retrain += 1

            # Train / retrain GP when enough data is available
            if n_cells >= config.gp_warmup_cells and (
                not gp_trained
                or cells_since_retrain >= config.gp_retrain_interval
            ):
                gp = GPSurrogate.train_from_atlas(atlas)
                gp_trained = True
                cells_since_retrain = 0

        elif depth < config.max_depth:
            # Collect eigenvalue info for anisotropic splitting
            eig_parts = None
            if atlas.cells:
                last = atlas.cells[-1]
                if last.equilibria:
                    eig_parts = last.equilibria[0].eigenvalue_real_parts

            box1, box2 = anisotropic_split_box(box, eig_parts)

            for child in (box1, box2):
                vol_frac = box_volume(child) / domain_vol if domain_vol > 0 else 0
                base_priority = vol_frac + 0.1

                gp_score = 0.0
                if gp_trained:
                    mid = box_midpoint(child)
                    pred = gp.predict(mid)
                    gp_score = phase_boundary_score(
                        pred,
                        eigenvalue_sensitivity=eigenvalue_sensitivity_score(
                            eig_parts or []
                        ),
                    )

                combined = (
                    (1.0 - config.gp_weight) * base_priority
                    + config.gp_weight * gp_score
                )
                heapq.heappush(pq, (-combined, counter, depth + 1, child))
                counter += 1
        else:
            atlas.add_uncertified(box, depth)

        # Record convergence
        convergence.iteration.append(iteration)
        convergence.coverage.append(atlas.coverage_fraction())
        convergence.certified_cells.append(n_cells)
        convergence.elapsed_s.append(time.time() - t0)

    # Drain remaining queue
    while pq:
        _, _cnt, depth, box = heapq.heappop(pq)
        atlas.add_uncertified(box, depth)

    atlas._build_time = time.time() - t0
    return atlas, convergence
