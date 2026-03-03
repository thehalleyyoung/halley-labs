"""CPA utilities subpackage.

Provides validation, caching, parallel computation, and logging utilities
used throughout the Causal-Plasticity Atlas engine.
"""

from cpa.utils.validation import (
    validate_adjacency_matrix,
    validate_probability,
    validate_positive,
    validate_dag,
    validate_array_shape,
    validate_dtype,
    validate_square_matrix,
    validate_sample_size,
    validated,
)
from cpa.utils.caching import (
    LRUCache,
    DiskCache,
    memoize,
    array_cache_key,
)
from cpa.utils.parallel import (
    parallel_map,
    batch_process,
    ThreadSafeAccumulator,
)
from cpa.utils.logging import (
    get_logger,
    TimingContext,
    MemoryTracker,
    ProgressReporter,
)

__all__ = [
    "validate_adjacency_matrix",
    "validate_probability",
    "validate_positive",
    "validate_dag",
    "validate_array_shape",
    "validate_dtype",
    "validate_square_matrix",
    "validate_sample_size",
    "validated",
    "LRUCache",
    "DiskCache",
    "memoize",
    "array_cache_key",
    "parallel_map",
    "batch_process",
    "ThreadSafeAccumulator",
    "get_logger",
    "TimingContext",
    "MemoryTracker",
    "ProgressReporter",
]
