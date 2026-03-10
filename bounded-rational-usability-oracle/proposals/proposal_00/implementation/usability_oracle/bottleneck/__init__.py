"""
usability_oracle.bottleneck — Cognitive bottleneck classification.

Classifies usability problems into five types using information-theoretic
signatures derived from the bounded-rational MDP analysis:

  1. **Perceptual overload** — visual entropy exceeds channel capacity.
  2. **Choice paralysis** — too many equally-attractive options (high H(π)).
  3. **Motor difficulty** — Fitts' law index of difficulty is high.
  4. **Memory decay** — working-memory load exceeds Miller's 4±1 threshold.
  5. **Cross-channel interference** — concurrent demands on same resource
     (Wickens' Multiple Resource Theory).

Each bottleneck type is detected by a specialised detector and characterised
by an information-theoretic signature (entropy, mutual information, channel
capacity utilisation).

Re-exports
----------
>>> from usability_oracle.bottleneck import (
...     BottleneckResult, BottleneckSignature, BottleneckReport,
...     BottleneckClassifier, SignatureComputer, RepairMapper,
... )
"""

from __future__ import annotations

from usability_oracle.bottleneck.models import (
    BottleneckResult,
    BottleneckSignature,
    BottleneckReport,
)
from usability_oracle.bottleneck.classifier import BottleneckClassifier
from usability_oracle.bottleneck.signatures import SignatureComputer
from usability_oracle.bottleneck.repair_map import RepairMapper
from usability_oracle.bottleneck.perceptual import PerceptualOverloadDetector
from usability_oracle.bottleneck.choice import ChoiceParalysisDetector
from usability_oracle.bottleneck.motor import MotorDifficultyDetector
from usability_oracle.bottleneck.memory import MemoryDecayDetector
from usability_oracle.bottleneck.interference import CrossChannelInterferenceDetector

__all__ = [
    "BottleneckResult",
    "BottleneckSignature",
    "BottleneckReport",
    "BottleneckClassifier",
    "SignatureComputer",
    "RepairMapper",
    "PerceptualOverloadDetector",
    "ChoiceParalysisDetector",
    "MotorDifficultyDetector",
    "MemoryDecayDetector",
    "CrossChannelInterferenceDetector",
]
