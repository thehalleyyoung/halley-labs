"""CPA streaming subpackage.

Online and streaming algorithms for incremental plasticity updates,
online DAG alignment, windowed tipping-point detection, and
efficient stream buffering.

Modules
-------
incremental_plasticity
    IPU (Incremental Plasticity Update) algorithm.
online_alignment
    Online DAG alignment with warm starts.
windowed_detection
    Windowed tipping-point detection.
stream_buffer
    Efficient stream buffering with sliding windows.
"""

from cpa.streaming.incremental_plasticity import (
    IncrementalPlasticityUpdater,
    PlasticityDelta,
    SufficientStatistics,
)
from cpa.streaming.online_alignment import (
    OnlineAligner,
    WarmStartState,
    AlignmentCache,
)
from cpa.streaming.windowed_detection import (
    WindowedDetector,
    DetectionWindow,
    OnlineCUSUM,
)
from cpa.streaming.stream_buffer import (
    StreamBuffer,
    SlidingWindow,
    CircularBuffer,
    TimestampedBuffer,
)

__all__ = [
    # incremental_plasticity.py
    "IncrementalPlasticityUpdater",
    "PlasticityDelta",
    "SufficientStatistics",
    # online_alignment.py
    "OnlineAligner",
    "WarmStartState",
    "AlignmentCache",
    # windowed_detection.py
    "WindowedDetector",
    "DetectionWindow",
    "OnlineCUSUM",
    # stream_buffer.py
    "StreamBuffer",
    "SlidingWindow",
    "CircularBuffer",
    "TimestampedBuffer",
]
