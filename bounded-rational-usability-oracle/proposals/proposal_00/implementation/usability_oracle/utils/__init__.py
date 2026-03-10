"""
usability_oracle.utils — Shared utilities for the usability oracle.

Provides mathematical functions, graph algorithms, information-theoretic
computations, sampling methods, timing helpers, logging setup, and
serialisation utilities.
"""

from __future__ import annotations

from usability_oracle.utils.math import (
    entropy,
    jensen_shannon_divergence,
    kl_divergence,
    log2_safe,
    log_sum_exp,
    mutual_information,
    normalize_distribution,
    softmax,
    total_variation_distance,
    wasserstein_distance,
)
from usability_oracle.utils.timing import Timer, TimingReport, format_timing, timed
from usability_oracle.utils.logging import LogContext, get_logger, setup_logging
from usability_oracle.utils.serialization import (
    DataclassEncoder,
    NumpyEncoder,
    deserialize_from_json,
    load_from_file,
    save_to_file,
    serialize_to_json,
)

__all__ = [
    # math
    "entropy",
    "jensen_shannon_divergence",
    "kl_divergence",
    "log2_safe",
    "log_sum_exp",
    "mutual_information",
    "normalize_distribution",
    "softmax",
    "total_variation_distance",
    "wasserstein_distance",
    # timing
    "Timer",
    "TimingReport",
    "format_timing",
    "timed",
    # logging
    "LogContext",
    "get_logger",
    "setup_logging",
    # serialization
    "DataclassEncoder",
    "NumpyEncoder",
    "deserialize_from_json",
    "load_from_file",
    "save_to_file",
    "serialize_to_json",
]
