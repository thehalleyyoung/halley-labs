"""
Classifiers module — behavioral atom classifiers for LLM outputs.

Provides:
    RefusalClassifier            — Refusal detection (pattern + statistical)
    SafetyClassifier             — Taxonomy-based safety classification
    BehavioralClassifier         — Opinion, sycophancy, instruction-following detection
    SemanticEmbeddingClassifier  — Embedding-based behavioral atom classifier
    EmbeddingProvider            — OpenAI embedding API wrapper with caching
"""

from caber.classifiers.refusal import RefusalClassifier
from caber.classifiers.safety import SafetyClassifier
from caber.classifiers.behavioral import BehavioralClassifier
from caber.classifiers.embedding import (
    SemanticEmbeddingClassifier,
    EmbeddingProvider,
    EmbeddingProfile,
    EmbeddingAtom,
    LOPOResult,
    CalibrationResult,
    compute_temporal_pattern,
    bisimulation_distance,
)

__all__ = [
    "RefusalClassifier",
    "SafetyClassifier",
    "BehavioralClassifier",
    "SemanticEmbeddingClassifier",
    "EmbeddingProvider",
    "EmbeddingProfile",
    "EmbeddingAtom",
    "LOPOResult",
    "CalibrationResult",
    "compute_temporal_pattern",
    "bisimulation_distance",
]
