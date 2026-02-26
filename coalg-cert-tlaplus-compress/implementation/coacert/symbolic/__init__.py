"""
Symbolic analysis module for CoaCert-TLA.

Provides symbolic state-space analysis techniques that complement
the explicit-state explorer, including:
- Bloom filter false-positive rate analysis
- Symbolic state counting via variable domain analysis
- State-space diameter estimation
"""

from .bloom_analysis import (
    BloomFilterAnalysis,
    BloomFilterConfig,
    BloomFilterSoundnessResult,
    optimal_bloom_parameters,
    false_positive_bound,
)
from .state_space_bounds import (
    StateSpaceBounds,
    DomainAnalyzer,
    VariableDomain,
    compute_upper_bound,
)

# Re-export the formal soundness analyzer from evaluation for convenience
from ..evaluation.bloom_soundness import (
    BloomSoundnessAnalyzer,
    SoundnessBound,
    AdaptiveBloomConfig,
    annotate_certificate as annotate_certificate_bloom,
)

__all__ = [
    "BloomFilterAnalysis",
    "BloomFilterConfig",
    "BloomFilterSoundnessResult",
    "optimal_bloom_parameters",
    "false_positive_bound",
    "StateSpaceBounds",
    "DomainAnalyzer",
    "VariableDomain",
    "compute_upper_bound",
    "BloomSoundnessAnalyzer",
    "SoundnessBound",
    "AdaptiveBloomConfig",
    "annotate_certificate_bloom",
]
