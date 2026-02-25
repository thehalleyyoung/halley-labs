"""Utility modules for the finite-width phase diagram system.

Provides configuration management, structured logging, I/O helpers,
numerical stability utilities, and parallel computation support.
"""

from .config import PhaseDiagramConfig, ConfigProfile, load_config, save_config
from .logging import get_logger, timer, ProgressTracker, MemoryMonitor
from .io import (
    save_phase_diagram,
    load_phase_diagram,
    save_kernel_matrix,
    load_kernel_matrix,
    save_calibration,
    load_calibration,
    CheckpointManager,
)
from .numerical import (
    stable_log_sum_exp,
    stable_softmax,
    check_condition_number,
    enforce_psd,
    sorted_eigenvalues,
    regularize_gram,
    numerical_gradient_check,
)
from .parallel import (
    parallel_grid_sweep,
    parallel_ntk_widths,
    parallel_training,
    ParallelConfig,
)
