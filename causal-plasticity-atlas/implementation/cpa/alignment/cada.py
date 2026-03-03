"""
Context-Aware DAG Alignment (CADA) — ALG1 from Theory.

Implements the full 6-phase CADA alignment algorithm for comparing
causal DAGs across observational contexts. This is the core engine
of the Causal-Plasticity Atlas.

Phases:
    1. Anchor Propagation — initialize and extend known correspondences
    2. Markov Blanket Candidate Generation — filter by MB Jaccard >= 0.3
    3. Statistical Compatibility Scoring — CI-fingerprint (0.6) + shape (0.4)
    4. Optimal Matching via Hungarian Algorithm — scipy linear_sum_assignment
    5. Edge Classification — SHARED / MODIFIED / CONTEXT_SPECIFIC_A / CONTEXT_SPECIFIC_B
    6. Score & Divergence Computation — alignment quality + weighted SHD

Complexity:
    O(n^2 d^2 + u^3) where n = |V|, d = max in-degree, u = |unanchored vars|
    (Theorem T7b: fixed-parameter tractable)

Classes:
    - CADAAligner: main 6-phase alignment engine
    - AlignmentCache: LRU cache for pairwise alignments
    - BatchAligner: align all K*(K-1)/2 context pairs

References:
    ALG1: Context-Aware DAG Alignment algorithm
    T7a: NP-hardness of general DAG alignment
    T7b: FPT result under bounded degree
    D6: DAG alignment distance
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import logging
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EPSILON = 1e-12
_DEFAULT_MB_OVERLAP_THRESHOLD = 0.3
_DEFAULT_QUALITY_THRESHOLD = 0.5
_DEFAULT_CI_WEIGHT = 0.6
_DEFAULT_SHAPE_WEIGHT = 0.4
_MAX_UNANCHORED_WARN = 50
_MAX_UNANCHORED_ERROR = 200

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------
try:
    from cpa.core.types import MechanismParams, VariableID
except ImportError:

    @dataclass
    class MechanismParams:
        """Parameters of a linear-Gaussian conditional."""

        coeffs: NDArray[np.floating]
        intercept: float = 0.0
        noise_var: float = 1.0

    VariableID = Union[int, str]

from cpa.alignment.scoring import (
    AnchorValidator,
    CIFingerprintScorer,
    DistributionShapeSimilarity,
    MarkovBlanketOverlap,
    ScoreMatrix,
)
from cpa.alignment.hungarian import (
    MatchResult,
    PaddedHungarianSolver,
    QualityFilter,
)
from cpa.core.mechanism_distance import (
    MechanismDistanceComputer,
    _mccm_context_ids,
    _mccm_context_pairs,
    _mccm_n_contexts,
    _mccm_n_vars,
    _scm_adjacency,
    _scm_implied_covariance,
    _scm_markov_blanket,
    _scm_mechanism_params,
    _scm_n_vars,
)


# ===================================================================
#  Edge classification enum
# ===================================================================
class EdgeType(Enum):
    """Classification of edges in aligned DAGs (Phase 5).

    SHARED:
        Edge present in both DAGs at aligned positions.
    MODIFIED:
        Edge present in both but with reversed direction (i->j vs j->i).
    CONTEXT_SPECIFIC_A:
        Edge present only in DAG A.
    CONTEXT_SPECIFIC_B:
        Edge present only in DAG B.
    """

    SHARED = "shared"
    MODIFIED = "modified"
    CONTEXT_SPECIFIC_A = "context_specific_a"
    CONTEXT_SPECIFIC_B = "context_specific_b"


# ===================================================================
#  Alignment result data classes
# ===================================================================
@dataclass
class EdgeClassification:
    """Classification of a single edge.

    Attributes
    ----------
    source_a : int or None
        Source variable index in DAG A (None if context-specific B).
    target_a : int or None
        Target variable index in DAG A.
    source_b : int or None
        Source variable index in DAG B.
    target_b : int or None
        Target variable index in DAG B.
    edge_type : EdgeType
        Classification type.
    weight : float
        Divergence weight for this edge type.
    """

    source_a: Optional[int]
    target_a: Optional[int]
    source_b: Optional[int]
    target_b: Optional[int]
    edge_type: EdgeType
    weight: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "source_a": self.source_a,
            "target_a": self.target_a,
            "source_b": self.source_b,
            "target_b": self.target_b,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
        }


@dataclass
class AlignmentResult:
    """Complete result of CADA alignment between two DAGs.

    Attributes
    ----------
    alignment : dict mapping int -> Optional[int]
        Variable alignment from A to B. None values mean unmatched.
    inverse_alignment : dict mapping int -> Optional[int]
        Reverse mapping from B to A.
    match_scores : dict mapping int -> float
        Match score for each aligned pair.
    edge_classifications : list of EdgeClassification
        Per-edge classification results.
    alignment_quality : float
        Overall alignment quality score.
    structural_divergence : float
        Weighted SHD between aligned DAGs.
    normalized_divergence : float
        Structural divergence normalized by n*(n-1)/2.
    n_shared : int
        Number of shared edges.
    n_modified : int
        Number of modified (reversed) edges.
    n_context_specific_a : int
        Number of context-specific edges in A.
    n_context_specific_b : int
        Number of context-specific edges in B.
    n_anchored : int
        Number of anchored variables.
    n_matched : int
        Number of total matched variables (anchored + Hungarian).
    n_unmatched_a : int
        Number of unmatched variables in A.
    n_unmatched_b : int
        Number of unmatched variables in B.
    context_a : str
        Context identifier for DAG A.
    context_b : str
        Context identifier for DAG B.
    computation_time : float
        Wall-clock time for alignment computation (seconds).
    phase_times : dict
        Time spent in each phase.
    """

    alignment: Dict[int, Optional[int]]
    inverse_alignment: Dict[int, Optional[int]]
    match_scores: Dict[int, float]
    edge_classifications: List[EdgeClassification]
    alignment_quality: float
    structural_divergence: float
    normalized_divergence: float
    n_shared: int
    n_modified: int
    n_context_specific_a: int
    n_context_specific_b: int
    n_anchored: int
    n_matched: int
    n_unmatched_a: int
    n_unmatched_b: int
    context_a: str
    context_b: str
    computation_time: float = 0.0
    phase_times: Dict[str, float] = field(default_factory=dict)

    @property
    def edge_jaccard(self) -> float:
        """Edge Jaccard index: |shared| / |E_a ∪ aligned(E_b)|."""
        total = self.n_shared + self.n_modified + self.n_context_specific_a + self.n_context_specific_b
        return self.n_shared / total if total > 0 else 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "alignment": {str(k): v for k, v in self.alignment.items()},
            "inverse_alignment": {str(k): v for k, v in self.inverse_alignment.items()},
            "match_scores": {str(k): v for k, v in self.match_scores.items()},
            "edge_classifications": [e.to_dict() for e in self.edge_classifications],
            "alignment_quality": self.alignment_quality,
            "structural_divergence": self.structural_divergence,
            "normalized_divergence": self.normalized_divergence,
            "n_shared": self.n_shared,
            "n_modified": self.n_modified,
            "n_context_specific_a": self.n_context_specific_a,
            "n_context_specific_b": self.n_context_specific_b,
            "n_anchored": self.n_anchored,
            "n_matched": self.n_matched,
            "n_unmatched_a": self.n_unmatched_a,
            "n_unmatched_b": self.n_unmatched_b,
            "context_a": self.context_a,
            "context_b": self.context_b,
            "edge_jaccard": self.edge_jaccard,
            "computation_time": self.computation_time,
            "phase_times": self.phase_times,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"CADA Alignment: {self.context_a} <-> {self.context_b}",
            f"  Matched: {self.n_matched} / {self.n_matched + self.n_unmatched_a + self.n_unmatched_b}",
            f"  Anchored: {self.n_anchored}",
            f"  Alignment quality: {self.alignment_quality:.4f}",
            f"  Edge Jaccard: {self.edge_jaccard:.4f}",
            f"  Structural divergence: {self.structural_divergence:.4f}",
            f"  Edges: {self.n_shared} shared, {self.n_modified} modified, "
            f"{self.n_context_specific_a} ctx-A, {self.n_context_specific_b} ctx-B",
            f"  Time: {self.computation_time:.3f}s",
        ]
        return "\n".join(lines)


# ===================================================================
#  Custom exceptions
# ===================================================================
class TooManyUnanchoredError(ValueError):
    """Raised when the number of unanchored variables exceeds the limit.

    With u unanchored variables, the Hungarian algorithm has O(u^3) complexity.
    Beyond _MAX_UNANCHORED_ERROR, this becomes impractical.
    """

    def __init__(self, n_unanchored: int, limit: int) -> None:
        self.n_unanchored = n_unanchored
        self.limit = limit
        super().__init__(
            f"Too many unanchored variables: {n_unanchored} > {limit}. "
            f"Provide more anchors or increase the limit."
        )


class AnchorConflictError(ValueError):
    """Raised when anchors are inconsistent (non-bijective or conflicting)."""

    def __init__(self, conflicts: List[str]) -> None:
        self.conflicts = conflicts
        msg = "Anchor conflicts detected:\n" + "\n".join(f"  - {c}" for c in conflicts)
        super().__init__(msg)


# ===================================================================
#  CADAAligner — Main 6-phase alignment engine
# ===================================================================
class CADAAligner:
    """Context-Aware DAG Alignment (ALG1).

    Aligns two DAGs across observational contexts using a 6-phase algorithm:

    1. **Anchor Propagation**: Initialize from known correspondences, propagate
       through Markov blanket overlap.
    2. **MB Candidate Generation**: For each unanchored variable, compute
       Markov blanket overlap with all candidates (filter by Jaccard >= 0.3).
    3. **Statistical Scoring**: Score candidates by CI-fingerprint similarity
       (weight 0.6) and distribution shape similarity (weight 0.4).
    4. **Hungarian Matching**: Optimal bipartite matching on the score matrix.
    5. **Edge Classification**: Classify each edge as shared, modified, or
       context-specific.
    6. **Score & Divergence**: Compute alignment quality and weighted SHD.

    Parameters
    ----------
    mb_overlap_threshold : float
        Minimum Markov blanket Jaccard overlap for candidate generation (Phase 2).
        Default 0.3.
    ci_weight : float
        Weight for CI-fingerprint similarity in scoring (Phase 3). Default 0.6.
    shape_weight : float
        Weight for distribution shape similarity (Phase 3). Default 0.4.
    quality_threshold : float
        Minimum match quality for Hungarian post-filtering (Phase 4). Default 0.5.
    anchor_propagation_rounds : int
        Maximum rounds of anchor propagation (Phase 1). Default 3.
    propagation_threshold : float
        Minimum MB overlap to propagate an anchor. Default 0.5.
    w_reversal : float
        Divergence weight for edge reversals (Phase 5). Default 0.5.
    w_addition : float
        Divergence weight for edge additions (Phase 5). Default 1.0.
    w_deletion : float
        Divergence weight for edge deletions (Phase 5). Default 1.0.
    w_context_specific : float
        Divergence weight for context-specific edges. Default 0.8.
    max_unanchored : int
        Maximum number of unanchored variables before raising error. Default 200.
    validate_anchors : bool
        Whether to validate anchor consistency. Default True.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        mb_overlap_threshold: float = _DEFAULT_MB_OVERLAP_THRESHOLD,
        ci_weight: float = _DEFAULT_CI_WEIGHT,
        shape_weight: float = _DEFAULT_SHAPE_WEIGHT,
        quality_threshold: float = _DEFAULT_QUALITY_THRESHOLD,
        anchor_propagation_rounds: int = 3,
        propagation_threshold: float = 0.5,
        w_reversal: float = 0.5,
        w_addition: float = 1.0,
        w_deletion: float = 1.0,
        w_context_specific: float = 0.8,
        max_unanchored: int = _MAX_UNANCHORED_ERROR,
        validate_anchors: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.mb_overlap_threshold = mb_overlap_threshold
        self.ci_weight = ci_weight
        self.shape_weight = shape_weight
        self.quality_threshold = quality_threshold
        self.anchor_propagation_rounds = anchor_propagation_rounds
        self.propagation_threshold = propagation_threshold
        self.w_reversal = w_reversal
        self.w_addition = w_addition
        self.w_deletion = w_deletion
        self.w_context_specific = w_context_specific
        self.max_unanchored = max_unanchored
        self.validate_anchors = validate_anchors
        self.seed = seed

        # Sub-components
        self._mb_scorer = MarkovBlanketOverlap(overlap_threshold=mb_overlap_threshold)
        self._ci_scorer = CIFingerprintScorer(method="cosine")
        self._shape_scorer = DistributionShapeSimilarity(method="kl_symmetric")
        self._score_builder = ScoreMatrix(ci_weight=ci_weight, shape_weight=shape_weight)
        self._anchor_validator = AnchorValidator()
        self._hungarian = PaddedHungarianSolver(
            quality_filter=QualityFilter(min_score=quality_threshold),
            maximize=True,
        )
        self._distance_computer = MechanismDistanceComputer(seed=seed)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def align(
        self,
        scm_a: Any,
        scm_b: Any,
        anchors: Optional[Dict[int, int]] = None,
        context_a: str = "A",
        context_b: str = "B",
    ) -> AlignmentResult:
        """Align two SCMs using the full 6-phase CADA algorithm.

        Parameters
        ----------
        scm_a, scm_b : StructuralCausalModel (or compatible)
            The two SCMs to align.
        anchors : dict or None
            Known variable correspondences: maps indices in A to indices in B.
            If None, starts with empty anchors (full alignment from scratch).
        context_a, context_b : str
            Context identifiers.

        Returns
        -------
        AlignmentResult
            Complete alignment result.

        Raises
        ------
        TooManyUnanchoredError
            If the number of unanchored variables exceeds max_unanchored.
        AnchorConflictError
            If anchors are inconsistent.
        """
        t_start = time.monotonic()
        phase_times: Dict[str, float] = {}

        if anchors is None:
            anchors = {}

        # === Phase 1: Anchor Propagation ===
        t_phase = time.monotonic()
        anchors = self._phase1_anchor_propagation(scm_a, scm_b, anchors)
        phase_times["phase1_anchor_propagation"] = time.monotonic() - t_phase
        logger.info(
            "Phase 1 complete: %d anchors after propagation", len(anchors)
        )

        # Determine unanchored variables
        all_a = set(range(_scm_n_vars(scm_a)))
        all_b = set(range(_scm_n_vars(scm_b)))
        anchored_a = set(anchors.keys())
        anchored_b = set(anchors.values())
        unanchored_a = sorted(all_a - anchored_a)
        unanchored_b = sorted(all_b - anchored_b)

        n_unanchored = max(len(unanchored_a), len(unanchored_b))
        if n_unanchored > self.max_unanchored:
            raise TooManyUnanchoredError(n_unanchored, self.max_unanchored)
        if n_unanchored > _MAX_UNANCHORED_WARN:
            warnings.warn(
                f"{n_unanchored} unanchored variables; alignment may be slow (O(u^3))",
                stacklevel=2,
            )

        # === Phase 2: MB Candidate Generation ===
        t_phase = time.monotonic()
        candidates, candidate_mask = self._phase2_mb_candidates(
            scm_a, scm_b, unanchored_a, unanchored_b, anchors
        )
        phase_times["phase2_mb_candidates"] = time.monotonic() - t_phase
        logger.info("Phase 2 complete: %d candidate pairs", len(candidates))

        # === Phase 3: Statistical Scoring ===
        t_phase = time.monotonic()
        score_matrix = self._phase3_statistical_scoring(
            scm_a, scm_b, unanchored_a, unanchored_b, anchors, candidate_mask,
            context_a, context_b,
        )
        phase_times["phase3_statistical_scoring"] = time.monotonic() - t_phase
        logger.info("Phase 3 complete: score matrix shape %s", score_matrix.shape)

        # === Phase 4: Hungarian Matching ===
        t_phase = time.monotonic()
        match_result = self._phase4_hungarian_matching(
            score_matrix, unanchored_a, unanchored_b
        )
        phase_times["phase4_hungarian_matching"] = time.monotonic() - t_phase

        # Merge anchors and Hungarian matches into full alignment
        full_alignment: Dict[int, Optional[int]] = {}
        match_scores: Dict[int, float] = {}

        for va, vb in anchors.items():
            full_alignment[va] = vb
            match_scores[va] = 1.0  # anchors get perfect score

        for r, c, s in zip(match_result.row_indices, match_result.col_indices, match_result.scores):
            va = unanchored_a[r]
            vb = unanchored_b[c]
            full_alignment[va] = vb
            match_scores[va] = s

        # Mark unmatched
        for r in match_result.unmatched_rows:
            if r < len(unanchored_a):
                va = unanchored_a[r]
                full_alignment[va] = None
                match_scores[va] = 0.0

        # Build inverse alignment
        inverse_alignment: Dict[int, Optional[int]] = {}
        matched_b = set()
        for va, vb in full_alignment.items():
            if vb is not None:
                inverse_alignment[vb] = va
                matched_b.add(vb)
        for vb in all_b - matched_b:
            inverse_alignment[vb] = None

        n_matched = sum(1 for v in full_alignment.values() if v is not None)
        logger.info(
            "Phase 4 complete: %d matched (%d anchored + %d Hungarian), "
            "%d unmatched A, %d unmatched B",
            n_matched,
            len(anchors),
            match_result.n_matched,
            sum(1 for v in full_alignment.values() if v is None),
            sum(1 for v in inverse_alignment.values() if v is None),
        )

        # === Phase 5: Edge Classification ===
        t_phase = time.monotonic()
        edge_classifications = self._phase5_edge_classification(
            _scm_adjacency(scm_a), _scm_adjacency(scm_b), full_alignment
        )
        phase_times["phase5_edge_classification"] = time.monotonic() - t_phase

        n_shared = sum(1 for e in edge_classifications if e.edge_type == EdgeType.SHARED)
        n_modified = sum(1 for e in edge_classifications if e.edge_type == EdgeType.MODIFIED)
        n_ctx_a = sum(1 for e in edge_classifications if e.edge_type == EdgeType.CONTEXT_SPECIFIC_A)
        n_ctx_b = sum(1 for e in edge_classifications if e.edge_type == EdgeType.CONTEXT_SPECIFIC_B)

        logger.info(
            "Phase 5 complete: %d shared, %d modified, %d ctx-A, %d ctx-B edges",
            n_shared, n_modified, n_ctx_a, n_ctx_b,
        )

        # === Phase 6: Score & Divergence ===
        t_phase = time.monotonic()
        quality, struct_div, norm_div = self._phase6_score_divergence(
            scm_a, scm_b, full_alignment, edge_classifications
        )
        phase_times["phase6_score_divergence"] = time.monotonic() - t_phase

        computation_time = time.monotonic() - t_start

        n_unmatched_a = sum(1 for v in full_alignment.values() if v is None)
        n_unmatched_b = sum(1 for v in inverse_alignment.values() if v is None)

        return AlignmentResult(
            alignment=full_alignment,
            inverse_alignment=inverse_alignment,
            match_scores=match_scores,
            edge_classifications=edge_classifications,
            alignment_quality=quality,
            structural_divergence=struct_div,
            normalized_divergence=norm_div,
            n_shared=n_shared,
            n_modified=n_modified,
            n_context_specific_a=n_ctx_a,
            n_context_specific_b=n_ctx_b,
            n_anchored=len(anchors),
            n_matched=n_matched,
            n_unmatched_a=n_unmatched_a,
            n_unmatched_b=n_unmatched_b,
            context_a=context_a,
            context_b=context_b,
            computation_time=computation_time,
            phase_times=phase_times,
        )

    # ------------------------------------------------------------------
    # Phase 1: Anchor Propagation
    # ------------------------------------------------------------------
    def _phase1_anchor_propagation(
        self,
        scm_a: Any,
        scm_b: Any,
        initial_anchors: Dict[int, int],
    ) -> Dict[int, int]:
        """Phase 1: Anchor Propagation.

        Initialize alignment from known anchor correspondences and propagate
        through Markov blanket overlap. New anchors are added when a unique
        high-confidence correspondence is found.

        Parameters
        ----------
        scm_a, scm_b : StructuralCausalModel (or compatible)
            The two SCMs.
        initial_anchors : dict
            Known variable correspondences.

        Returns
        -------
        dict
            Extended anchor mapping.

        Raises
        ------
        AnchorConflictError
            If initial anchors are inconsistent.
        """
        anchors = dict(initial_anchors)

        # Validate anchors
        if self.validate_anchors and anchors:
            validator = self._anchor_validator
            bij_ok, bij_errors = validator.check_bijectivity(anchors)
            if not bij_ok:
                raise AnchorConflictError(bij_errors)

        # Iterative propagation
        for round_idx in range(self.anchor_propagation_rounds):
            new_anchors = self._propagate_anchors_one_round(scm_a, scm_b, anchors)

            if not new_anchors:
                logger.debug("Anchor propagation converged at round %d", round_idx)
                break

            # Validate new anchors don't conflict
            for va, vb in new_anchors.items():
                if va in anchors:
                    continue  # already anchored
                if vb in anchors.values():
                    logger.debug(
                        "Skipping propagated anchor %d->%d: target already anchored",
                        va, vb,
                    )
                    continue
                anchors[va] = vb

            logger.debug(
                "Propagation round %d: added %d anchors, total %d",
                round_idx, len(new_anchors), len(anchors),
            )

        return anchors

    def _propagate_anchors_one_round(
        self,
        scm_a: Any,
        scm_b: Any,
        current_anchors: Dict[int, int],
    ) -> Dict[int, int]:
        """Single round of anchor propagation.

        For each anchored variable, examine its Markov blanket neighbors.
        If a neighbor in A has a unique high-overlap match in B, add it.

        Parameters
        ----------
        scm_a, scm_b : StructuralCausalModel (or compatible)
            The two SCMs.
        current_anchors : dict
            Current anchor mapping.

        Returns
        -------
        dict
            Newly discovered anchors (not yet in current_anchors).
        """
        new_anchors: Dict[int, int] = {}
        anchored_a = set(current_anchors.keys())
        anchored_b = set(current_anchors.values())

        for va_anchor, vb_anchor in current_anchors.items():
            # Get MB neighbors in both contexts
            mb_a = _scm_markov_blanket(scm_a, va_anchor)
            mb_b = _scm_markov_blanket(scm_b, vb_anchor)

            # Find unanchored neighbors
            unanchored_neighbors_a = mb_a - anchored_a - set(new_anchors.keys())
            unanchored_neighbors_b = mb_b - anchored_b - set(new_anchors.values())

            if not unanchored_neighbors_a or not unanchored_neighbors_b:
                continue

            # Try to match unanchored neighbors
            for va_neighbor in unanchored_neighbors_a:
                mb_va = _scm_markov_blanket(scm_a, va_neighbor)
                best_match = None
                best_overlap = 0.0
                second_best = 0.0

                for vb_candidate in unanchored_neighbors_b:
                    mb_vb = _scm_markov_blanket(scm_b, vb_candidate)
                    overlap = self._mb_scorer.anchored_jaccard(
                        mb_va, mb_vb, current_anchors
                    )

                    if overlap > best_overlap:
                        second_best = best_overlap
                        best_overlap = overlap
                        best_match = vb_candidate
                    elif overlap > second_best:
                        second_best = overlap

                # Add only if unique and confident match
                if (
                    best_match is not None
                    and best_overlap >= self.propagation_threshold
                    and best_overlap > second_best + 0.15  # gap criterion
                    and best_match not in anchored_b
                    and best_match not in new_anchors.values()
                ):
                    new_anchors[va_neighbor] = best_match

        return new_anchors

    # ------------------------------------------------------------------
    # Phase 2: Markov Blanket Candidate Generation
    # ------------------------------------------------------------------
    def _phase2_mb_candidates(
        self,
        scm_a: Any,
        scm_b: Any,
        unanchored_a: List[int],
        unanchored_b: List[int],
        anchors: Dict[int, int],
    ) -> Tuple[List[Tuple[int, int, float]], NDArray[np.bool_]]:
        """Phase 2: Generate candidate pairs via Markov blanket overlap.

        For each unanchored variable in A, compute MB overlap with each
        unanchored variable in B, using anchored variables as reference.
        Filter pairs with overlap < threshold.

        Parameters
        ----------
        scm_a, scm_b : StructuralCausalModel (or compatible)
            The two SCMs.
        unanchored_a, unanchored_b : list of int
            Unanchored variable indices.
        anchors : dict
            Current anchor mapping.

        Returns
        -------
        (candidates, candidate_mask) tuple:
            candidates: list of (var_a, var_b, overlap) tuples
            candidate_mask: boolean matrix (len(unanchored_a), len(unanchored_b))
        """
        n_a = len(unanchored_a)
        n_b = len(unanchored_b)

        if n_a == 0 or n_b == 0:
            return [], np.zeros((n_a, n_b), dtype=bool)

        candidates = self._mb_scorer.generate_candidates(
            _scm_adjacency(scm_a), _scm_adjacency(scm_b),
            unanchored_a, unanchored_b,
            anchors,
        )

        # Build candidate mask matrix
        a_idx_map = {v: i for i, v in enumerate(unanchored_a)}
        b_idx_map = {v: i for i, v in enumerate(unanchored_b)}
        mask = np.zeros((n_a, n_b), dtype=bool)

        for va, vb, overlap in candidates:
            i = a_idx_map.get(va)
            j = b_idx_map.get(vb)
            if i is not None and j is not None:
                mask[i, j] = True

        # If no candidates pass the threshold, fall back to all pairs
        if not candidates:
            logger.info(
                "No candidates passed MB overlap threshold %.2f; "
                "falling back to all-pairs scoring",
                self.mb_overlap_threshold,
            )
            mask[:] = True
            for i, va in enumerate(unanchored_a):
                for j, vb in enumerate(unanchored_b):
                    candidates.append((va, vb, 0.0))

        return candidates, mask

    # ------------------------------------------------------------------
    # Phase 3: Statistical Compatibility Scoring
    # ------------------------------------------------------------------
    def _phase3_statistical_scoring(
        self,
        scm_a: Any,
        scm_b: Any,
        unanchored_a: List[int],
        unanchored_b: List[int],
        anchors: Dict[int, int],
        candidate_mask: NDArray[np.bool_],
        context_a: str = "A",
        context_b: str = "B",
    ) -> NDArray[np.floating]:
        """Phase 3: Score candidate pairs using CI-fingerprint + distribution shape.

        Combines:
            - CI-fingerprint similarity (weight 0.6): partial correlation vectors
            - Distribution shape similarity (weight 0.4): KL-based comparison

        Parameters
        ----------
        scm_a, scm_b : StructuralCausalModel (or compatible)
            The two SCMs.
        unanchored_a, unanchored_b : list of int
            Unanchored variable indices.
        anchors : dict
            Anchor mapping.
        candidate_mask : NDArray of bool
            Which pairs to score.
        context_a, context_b : str
            Context identifiers.

        Returns
        -------
        NDArray, shape (len(unanchored_a), len(unanchored_b))
            Combined score matrix.
        """
        n_a = len(unanchored_a)
        n_b = len(unanchored_b)

        if n_a == 0 or n_b == 0:
            return np.zeros((n_a, n_b), dtype=np.float64)

        # Compute covariance matrices (cached)
        cov_a = _scm_implied_covariance(scm_a)
        cov_b = _scm_implied_covariance(scm_b)

        ci_scores = np.zeros((n_a, n_b), dtype=np.float64)
        shape_scores = np.zeros((n_a, n_b), dtype=np.float64)

        for i, va in enumerate(unanchored_a):
            for j, vb in enumerate(unanchored_b):
                if not candidate_mask[i, j]:
                    continue

                # CI-fingerprint similarity
                try:
                    ci_score = self._ci_scorer.score_pair(
                        va, vb, cov_a, cov_b,
                        alignment=anchors,
                        ctx_a=context_a,
                        ctx_b=context_b,
                    )
                except Exception as e:
                    logger.warning(
                        "CI scoring failed for (%d, %d): %s", va, vb, e
                    )
                    ci_score = 0.0

                ci_scores[i, j] = ci_score

                # Distribution shape similarity
                try:
                    params_a = _scm_mechanism_params(scm_a, va)
                    params_b = _scm_mechanism_params(scm_b, vb)
                    shape_score = self._shape_scorer.score(params_a, params_b)
                except Exception as e:
                    logger.warning(
                        "Shape scoring failed for (%d, %d): %s", va, vb, e
                    )
                    shape_score = 0.0

                shape_scores[i, j] = shape_score

        # Combine scores
        combined = self._score_builder.build(ci_scores, shape_scores, candidate_mask=candidate_mask)

        return combined

    # ------------------------------------------------------------------
    # Phase 4: Hungarian Matching
    # ------------------------------------------------------------------
    def _phase4_hungarian_matching(
        self,
        score_matrix: NDArray[np.floating],
        unanchored_a: List[int],
        unanchored_b: List[int],
    ) -> MatchResult:
        """Phase 4: Optimal matching via Hungarian algorithm.

        Pads score matrix to square, runs linear_sum_assignment, and filters
        matches by quality threshold.

        Parameters
        ----------
        score_matrix : NDArray, shape (n_a, n_b)
            Score matrix from Phase 3.
        unanchored_a, unanchored_b : list of int
            Unanchored variable indices.

        Returns
        -------
        MatchResult
            Matching result with quality-filtered pairs.
        """
        if score_matrix.size == 0:
            return MatchResult(
                row_indices=[],
                col_indices=[],
                scores=[],
                unmatched_rows=list(range(len(unanchored_a))),
                unmatched_cols=list(range(len(unanchored_b))),
                total_score=0.0,
                quality=0.0,
                n_original_rows=len(unanchored_a),
                n_original_cols=len(unanchored_b),
            )

        return self._hungarian.solve(
            score_matrix,
            row_labels=[str(v) for v in unanchored_a],
            col_labels=[str(v) for v in unanchored_b],
        )

    # ------------------------------------------------------------------
    # Phase 5: Edge Classification
    # ------------------------------------------------------------------
    def _phase5_edge_classification(
        self,
        adj_a: NDArray[np.floating],
        adj_b: NDArray[np.floating],
        alignment: Dict[int, Optional[int]],
    ) -> List[EdgeClassification]:
        """Phase 5: Classify each edge as shared, modified, or context-specific.

        Edge types and their divergence weights:
            - SHARED: edge i->j in both (weight 0)
            - MODIFIED: edge i->j in A but j->i in B (weight w_reversal=0.5)
            - CONTEXT_SPECIFIC_A: edge only in A (weight w_deletion=1.0)
            - CONTEXT_SPECIFIC_B: edge only in B (weight w_addition=1.0)

        Parameters
        ----------
        adj_a, adj_b : NDArray
            Adjacency matrices.
        alignment : dict
            Full alignment mapping A -> B (with None for unmatched).

        Returns
        -------
        list of EdgeClassification
        """
        n_a = adj_a.shape[0]
        n_b = adj_b.shape[0]

        # Build reverse alignment
        inv_alignment: Dict[int, Optional[int]] = {}
        for va, vb in alignment.items():
            if vb is not None:
                inv_alignment[vb] = va

        classifications: List[EdgeClassification] = []
        processed_b_edges: Set[Tuple[int, int]] = set()

        # Process edges in A
        for i_a in range(n_a):
            for j_a in range(n_a):
                if i_a == j_a or adj_a[i_a, j_a] == 0:
                    continue

                i_b = alignment.get(i_a)
                j_b = alignment.get(j_a)

                if i_b is None or j_b is None:
                    # At least one endpoint unmatched => context-specific A
                    classifications.append(EdgeClassification(
                        source_a=i_a,
                        target_a=j_a,
                        source_b=None,
                        target_b=None,
                        edge_type=EdgeType.CONTEXT_SPECIFIC_A,
                        weight=self.w_context_specific,
                    ))
                    continue

                if i_b >= n_b or j_b >= n_b:
                    classifications.append(EdgeClassification(
                        source_a=i_a,
                        target_a=j_a,
                        source_b=i_b,
                        target_b=j_b,
                        edge_type=EdgeType.CONTEXT_SPECIFIC_A,
                        weight=self.w_context_specific,
                    ))
                    continue

                # Check if edge exists in B at aligned positions
                if adj_b[i_b, j_b] != 0:
                    # Shared edge
                    classifications.append(EdgeClassification(
                        source_a=i_a,
                        target_a=j_a,
                        source_b=i_b,
                        target_b=j_b,
                        edge_type=EdgeType.SHARED,
                        weight=0.0,
                    ))
                    processed_b_edges.add((i_b, j_b))
                elif adj_b[j_b, i_b] != 0:
                    # Reversed edge (modified)
                    classifications.append(EdgeClassification(
                        source_a=i_a,
                        target_a=j_a,
                        source_b=j_b,
                        target_b=i_b,
                        edge_type=EdgeType.MODIFIED,
                        weight=self.w_reversal,
                    ))
                    processed_b_edges.add((j_b, i_b))
                else:
                    # Edge in A, not in B at all => context-specific A
                    classifications.append(EdgeClassification(
                        source_a=i_a,
                        target_a=j_a,
                        source_b=None,
                        target_b=None,
                        edge_type=EdgeType.CONTEXT_SPECIFIC_A,
                        weight=self.w_deletion,
                    ))

        # Process remaining edges in B (not yet matched)
        for i_b in range(n_b):
            for j_b in range(n_b):
                if i_b == j_b or adj_b[i_b, j_b] == 0:
                    continue
                if (i_b, j_b) in processed_b_edges:
                    continue

                i_a = inv_alignment.get(i_b)
                j_a = inv_alignment.get(j_b)

                classifications.append(EdgeClassification(
                    source_a=i_a,
                    target_a=j_a,
                    source_b=i_b,
                    target_b=j_b,
                    edge_type=EdgeType.CONTEXT_SPECIFIC_B,
                    weight=self.w_addition if (i_a is not None and j_a is not None) else self.w_context_specific,
                ))

        return classifications

    # ------------------------------------------------------------------
    # Phase 6: Score & Divergence Computation
    # ------------------------------------------------------------------
    def _phase6_score_divergence(
        self,
        scm_a: Any,
        scm_b: Any,
        alignment: Dict[int, Optional[int]],
        edge_classifications: List[EdgeClassification],
    ) -> Tuple[float, float, float]:
        """Phase 6: Compute alignment quality and structural divergence.

        Alignment quality (edge Jaccard):
            Q = |S_shared| / |E_a ∪ aligned(E_b)|

        Structural divergence (weighted SHD):
            D = sum of edge weights from classification

        Normalized divergence:
            D_norm = D / (n * (n-1) / 2)

        Parameters
        ----------
        scm_a, scm_b : StructuralCausalModel (or compatible)
            The two SCMs.
        alignment : dict
            Full alignment.
        edge_classifications : list of EdgeClassification
            Edge classification results.

        Returns
        -------
        (quality, structural_divergence, normalized_divergence) tuple.
        """
        n_shared = sum(1 for e in edge_classifications if e.edge_type == EdgeType.SHARED)
        n_total = len(edge_classifications)

        # Alignment quality = |shared| / |total edges|
        quality = n_shared / n_total if n_total > 0 else 1.0

        # Structural divergence = sum of weights
        struct_div = sum(e.weight for e in edge_classifications)

        # Normalize by n*(n-1)/2
        n = max(_scm_n_vars(scm_a), _scm_n_vars(scm_b))
        max_edges = n * (n - 1) / 2.0
        norm_div = struct_div / max_edges if max_edges > 0 else 0.0

        return quality, struct_div, norm_div

    # ------------------------------------------------------------------
    # Convenience: align from MCCM
    # ------------------------------------------------------------------
    def align_contexts(
        self,
        mccm: Any,
        ctx_a: str,
        ctx_b: str,
        anchors: Optional[Dict[int, int]] = None,
    ) -> AlignmentResult:
        """Align two contexts from a multi-context causal model.

        Parameters
        ----------
        mccm : MultiContextCausalModel (or compatible)
            Multi-context model.
        ctx_a, ctx_b : str
            Context identifiers.
        anchors : dict or None
            Variable anchors.

        Returns
        -------
        AlignmentResult
        """
        scm_a = mccm.get_scm(ctx_a)
        scm_b = mccm.get_scm(ctx_b)
        return self.align(scm_a, scm_b, anchors, context_a=ctx_a, context_b=ctx_b)


# ===================================================================
#  AlignmentCache
# ===================================================================
class AlignmentCache:
    """LRU cache for pairwise alignments.

    Caches O(K^2) alignment results with memory-bounded LRU eviction.

    Parameters
    ----------
    max_size : int
        Maximum number of cached alignments. Default 1000.
    """

    def __init__(self, max_size: int = 1000) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self.max_size = max_size
        self._cache: OrderedDict[str, AlignmentResult] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(ctx_a: str, ctx_b: str) -> str:
        """Create a canonical cache key (order-independent)."""
        a, b = sorted([ctx_a, ctx_b])
        return f"{a}||{b}"

    def get(
        self,
        ctx_a: str,
        ctx_b: str,
    ) -> Optional[AlignmentResult]:
        """Get cached alignment result.

        Parameters
        ----------
        ctx_a, ctx_b : str
            Context identifiers.

        Returns
        -------
        AlignmentResult or None
        """
        key = self._make_key(ctx_a, ctx_b)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                self._cache.move_to_end(key)
                return self._cache[key]
            self._misses += 1
            return None

    def put(
        self,
        ctx_a: str,
        ctx_b: str,
        result: AlignmentResult,
    ) -> None:
        """Cache an alignment result.

        Parameters
        ----------
        ctx_a, ctx_b : str
            Context identifiers.
        result : AlignmentResult
            Alignment result to cache.
        """
        key = self._make_key(ctx_a, ctx_b)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = result
            else:
                if len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)  # LRU eviction
                self._cache[key] = result

    def invalidate(self, ctx: str) -> int:
        """Invalidate all cached alignments involving context *ctx*.

        Parameters
        ----------
        ctx : str
            Context identifier.

        Returns
        -------
        int
            Number of evicted entries.
        """
        with self._lock:
            keys_to_remove = [
                k for k in self._cache if ctx in k.split("||")
            ]
            for k in keys_to_remove:
                del self._cache[k]
            return len(keys_to_remove)

    def invalidate_all(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize cache (without alignment data, just stats and keys)."""
        return {
            "max_size": self.max_size,
            "entries": list(self._cache.keys()),
            "stats": self.stats(),
        }

    def get_all_pairs(self) -> List[Tuple[str, str]]:
        """Return all cached context pairs."""
        pairs = []
        for key in self._cache:
            a, b = key.split("||")
            pairs.append((a, b))
        return pairs


# ===================================================================
#  BatchAligner
# ===================================================================
class BatchAligner:
    """Align all K*(K-1)/2 context pairs.

    Provides batch alignment with caching, progress tracking,
    and consensus computation.

    Parameters
    ----------
    aligner : CADAAligner
        The alignment engine.
    cache : AlignmentCache or None
        Alignment cache. If None, creates a new one.
    n_jobs : int
        Number of parallel jobs. Default 1 (sequential).
    """

    def __init__(
        self,
        aligner: Optional[CADAAligner] = None,
        cache: Optional[AlignmentCache] = None,
        n_jobs: int = 1,
    ) -> None:
        self.aligner = aligner if aligner is not None else CADAAligner()
        self.cache = cache if cache is not None else AlignmentCache()
        self.n_jobs = n_jobs

    def align_all_pairs(
        self,
        mccm: Any,
        anchors: Optional[Dict[int, int]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[Tuple[str, str], AlignmentResult]:
        """Align all K*(K-1)/2 context pairs.

        Parameters
        ----------
        mccm : MultiContextCausalModel (or compatible)
            Multi-context model.
        anchors : dict or None
            Variable anchors (shared across all pairs).
        progress_callback : callable or None
            Called as progress_callback(i, total, pair_str) after each alignment.

        Returns
        -------
        dict mapping (ctx_a, ctx_b) -> AlignmentResult
        """
        pairs = _mccm_context_pairs(mccm)
        total = len(pairs)
        results: Dict[Tuple[str, str], AlignmentResult] = {}

        logger.info("Aligning %d context pairs", total)

        for idx, (ctx_a, ctx_b) in enumerate(pairs):
            # Check cache
            cached = self.cache.get(ctx_a, ctx_b)
            if cached is not None:
                results[(ctx_a, ctx_b)] = cached
                if progress_callback:
                    progress_callback(idx + 1, total, f"{ctx_a}<->{ctx_b} (cached)")
                continue

            # Compute alignment
            try:
                result = self.aligner.align_contexts(mccm, ctx_a, ctx_b, anchors)
                results[(ctx_a, ctx_b)] = result
                self.cache.put(ctx_a, ctx_b, result)
            except TooManyUnanchoredError:
                logger.warning("Skipping pair %s<->%s: too many unanchored vars", ctx_a, ctx_b)
                continue
            except Exception as e:
                logger.error("Alignment failed for %s<->%s: %s", ctx_a, ctx_b, e)
                continue

            if progress_callback:
                progress_callback(idx + 1, total, f"{ctx_a}<->{ctx_b}")

        return results

    def alignment_quality_matrix(
        self,
        results: Dict[Tuple[str, str], AlignmentResult],
        context_ids: List[str],
    ) -> NDArray[np.floating]:
        """Build K x K alignment quality matrix.

        Parameters
        ----------
        results : dict
            Alignment results from align_all_pairs.
        context_ids : list of str
            Context identifiers.

        Returns
        -------
        NDArray, shape (K, K)
            Symmetric matrix of alignment qualities.
        """
        K = len(context_ids)
        mat = np.ones((K, K), dtype=np.float64)  # diagonal = 1

        ctx_idx = {c: i for i, c in enumerate(context_ids)}
        for (ca, cb), result in results.items():
            i = ctx_idx.get(ca)
            j = ctx_idx.get(cb)
            if i is not None and j is not None:
                mat[i, j] = result.alignment_quality
                mat[j, i] = result.alignment_quality

        return mat

    def divergence_matrix(
        self,
        results: Dict[Tuple[str, str], AlignmentResult],
        context_ids: List[str],
        normalized: bool = True,
    ) -> NDArray[np.floating]:
        """Build K x K structural divergence matrix.

        Parameters
        ----------
        results : dict
            Alignment results.
        context_ids : list of str
            Context identifiers.
        normalized : bool
            If True, use normalized divergence. Default True.

        Returns
        -------
        NDArray, shape (K, K)
            Symmetric divergence matrix.
        """
        K = len(context_ids)
        mat = np.zeros((K, K), dtype=np.float64)

        ctx_idx = {c: i for i, c in enumerate(context_ids)}
        for (ca, cb), result in results.items():
            i = ctx_idx.get(ca)
            j = ctx_idx.get(cb)
            if i is not None and j is not None:
                d = result.normalized_divergence if normalized else result.structural_divergence
                mat[i, j] = d
                mat[j, i] = d

        return mat

    def consensus_alignment(
        self,
        results: Dict[Tuple[str, str], AlignmentResult],
        context_ids: List[str],
        reference_context: Optional[str] = None,
    ) -> Dict[str, Dict[int, Optional[int]]]:
        """Compute consensus alignment across all context pairs.

        Uses a reference context (or the most central one) and aggregates
        all pairwise alignments to produce a consensus variable mapping.

        Parameters
        ----------
        results : dict
            Alignment results.
        context_ids : list of str
            Context identifiers.
        reference_context : str or None
            Reference context. If None, uses the most connected one.

        Returns
        -------
        dict mapping context_id -> {var_idx -> aligned_var_idx_in_reference}
        """
        if reference_context is None:
            # Pick context with highest mean alignment quality
            quality_sums: Dict[str, float] = {c: 0.0 for c in context_ids}
            quality_counts: Dict[str, int] = {c: 0 for c in context_ids}
            for (ca, cb), result in results.items():
                quality_sums[ca] = quality_sums.get(ca, 0.0) + result.alignment_quality
                quality_sums[cb] = quality_sums.get(cb, 0.0) + result.alignment_quality
                quality_counts[ca] = quality_counts.get(ca, 0) + 1
                quality_counts[cb] = quality_counts.get(cb, 0) + 1

            best_ctx = max(
                context_ids,
                key=lambda c: quality_sums[c] / max(quality_counts[c], 1),
            )
            reference_context = best_ctx

        logger.info("Computing consensus alignment with reference: %s", reference_context)

        consensus: Dict[str, Dict[int, Optional[int]]] = {}
        consensus[reference_context] = {}  # identity mapping

        for ctx in context_ids:
            if ctx == reference_context:
                continue

            # Find direct alignment
            pair_key = None
            for (ca, cb), result in results.items():
                if ca == reference_context and cb == ctx:
                    consensus[ctx] = dict(result.alignment)
                    pair_key = (ca, cb)
                    break
                if cb == reference_context and ca == ctx:
                    consensus[ctx] = dict(result.inverse_alignment)
                    pair_key = (ca, cb)
                    break

            if pair_key is None:
                logger.warning("No alignment found for %s <-> %s", reference_context, ctx)
                consensus[ctx] = {}

        return consensus

    def alignment_statistics(
        self,
        results: Dict[Tuple[str, str], AlignmentResult],
    ) -> Dict[str, Any]:
        """Compute summary statistics over all alignments.

        Parameters
        ----------
        results : dict
            Alignment results.

        Returns
        -------
        dict with summary statistics.
        """
        if not results:
            return {
                "n_pairs": 0,
                "mean_quality": 0.0,
                "mean_divergence": 0.0,
                "mean_matched_ratio": 0.0,
            }

        qualities = [r.alignment_quality for r in results.values()]
        divergences = [r.normalized_divergence for r in results.values()]
        match_ratios = [
            r.n_matched / max(r.n_matched + r.n_unmatched_a + r.n_unmatched_b, 1)
            for r in results.values()
        ]
        times = [r.computation_time for r in results.values()]

        edge_jaccards = [r.edge_jaccard for r in results.values()]

        return {
            "n_pairs": len(results),
            "mean_quality": float(np.mean(qualities)),
            "std_quality": float(np.std(qualities)),
            "min_quality": float(np.min(qualities)),
            "max_quality": float(np.max(qualities)),
            "mean_divergence": float(np.mean(divergences)),
            "std_divergence": float(np.std(divergences)),
            "min_divergence": float(np.min(divergences)),
            "max_divergence": float(np.max(divergences)),
            "mean_edge_jaccard": float(np.mean(edge_jaccards)),
            "mean_matched_ratio": float(np.mean(match_ratios)),
            "total_time": float(np.sum(times)),
            "mean_time": float(np.mean(times)),
            "cache_stats": self.cache.stats(),
        }

    def find_most_similar_pair(
        self,
        results: Dict[Tuple[str, str], AlignmentResult],
    ) -> Optional[Tuple[str, str, AlignmentResult]]:
        """Find the most similar context pair (highest alignment quality).

        Parameters
        ----------
        results : dict
            Alignment results.

        Returns
        -------
        (ctx_a, ctx_b, result) or None.
        """
        if not results:
            return None

        best_pair = max(results.items(), key=lambda x: x[1].alignment_quality)
        return best_pair[0][0], best_pair[0][1], best_pair[1]

    def find_most_divergent_pair(
        self,
        results: Dict[Tuple[str, str], AlignmentResult],
    ) -> Optional[Tuple[str, str, AlignmentResult]]:
        """Find the most divergent context pair.

        Parameters
        ----------
        results : dict
            Alignment results.

        Returns
        -------
        (ctx_a, ctx_b, result) or None.
        """
        if not results:
            return None

        worst_pair = max(results.items(), key=lambda x: x[1].normalized_divergence)
        return worst_pair[0][0], worst_pair[0][1], worst_pair[1]

    def edge_type_summary(
        self,
        results: Dict[Tuple[str, str], AlignmentResult],
    ) -> Dict[str, int]:
        """Aggregate edge type counts across all pairs.

        Parameters
        ----------
        results : dict
            Alignment results.

        Returns
        -------
        dict mapping EdgeType.value -> total count.
        """
        totals: Dict[str, int] = {
            EdgeType.SHARED.value: 0,
            EdgeType.MODIFIED.value: 0,
            EdgeType.CONTEXT_SPECIFIC_A.value: 0,
            EdgeType.CONTEXT_SPECIFIC_B.value: 0,
        }
        for result in results.values():
            for ec in result.edge_classifications:
                totals[ec.edge_type.value] = totals.get(ec.edge_type.value, 0) + 1

        return totals
