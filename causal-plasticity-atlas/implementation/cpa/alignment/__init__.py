"""
Context-Aware DAG Alignment package.

Provides the CADA alignment engine (ALG1) and supporting utilities
for aligning causal DAGs across observational contexts.

Main classes:
    - CADAAligner: full 6-phase alignment engine
    - AlignmentCache: LRU cache for pairwise alignments
    - BatchAligner: batch alignment for all context pairs
    - AlignmentResult: structured alignment results

Scoring utilities:
    - CIFingerprintScorer: CI-fingerprint comparison
    - MarkovBlanketOverlap: Jaccard-based MB overlap
    - DistributionShapeSimilarity: KL-based shape comparison
    - ScoreMatrix: combined score matrix construction
    - AnchorValidator: anchor consistency validation

Hungarian algorithm:
    - PaddedHungarianSolver: non-square matrix support
    - QualityFilter: post-matching quality filtering
    - MatchResult: structured match result
"""

from cpa.alignment.cada import (
    AlignmentResult,
    AnchorConflictError,
    BatchAligner,
    AlignmentCache,
    CADAAligner,
    EdgeClassification,
    EdgeType,
    TooManyUnanchoredError,
)
from cpa.alignment.hungarian import (
    MatchResult,
    PaddedHungarianSolver,
    QualityFilter,
    solve_assignment,
)
from cpa.alignment.scoring import (
    AnchorValidator,
    CIFingerprintScorer,
    DistributionShapeSimilarity,
    MarkovBlanketOverlap,
    ScoreMatrix,
)

__all__ = [
    # cada.py
    "AlignmentResult",
    "AnchorConflictError",
    "BatchAligner",
    "AlignmentCache",
    "CADAAligner",
    "EdgeClassification",
    "EdgeType",
    "TooManyUnanchoredError",
    # hungarian.py
    "MatchResult",
    "PaddedHungarianSolver",
    "QualityFilter",
    "solve_assignment",
    # scoring.py
    "AnchorValidator",
    "CIFingerprintScorer",
    "DistributionShapeSimilarity",
    "MarkovBlanketOverlap",
    "ScoreMatrix",
]
