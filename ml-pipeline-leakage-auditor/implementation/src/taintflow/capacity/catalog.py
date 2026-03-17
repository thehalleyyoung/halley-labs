"""
taintflow.capacity.catalog – Capacity catalog for sklearn / pandas operations.

Maps each recognised ``OpType`` to an information-theoretic transfer
function that returns a :class:`ChannelCapacityBound` parameterised by
(ρ, n, d) – test fraction, sample size, dimensionality.

The catalog distinguishes **tight** bounds (κ close to 1) from
**sound-but-generic** bounds (large κ).  Unknown operations fall back
to the Gaussian channel model as a conservative upper bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from taintflow.core.types import OpType, Origin
from taintflow.capacity.channels import (
    ChannelCapacityBound,
    ChannelKind,
    GaussianChannel,
    DiscreteChannel,
    BinaryChannel,
    DataProcessingInequality,
    _DEFAULT_B_MAX,
    _EPSILON,
)


# ===================================================================
#  Type aliases
# ===================================================================

CapacityFunction = Callable[[float, int, int], ChannelCapacityBound]
"""Signature: (rho, n, d) -> ChannelCapacityBound."""


# ===================================================================
#  Tightness classification
# ===================================================================


class TightnessClass(Enum):
    """Qualitative tightness of a capacity bound."""

    EXACT = auto()        # κ = 1, provably tight
    NEAR_TIGHT = auto()   # κ = O(1), constant gap
    LOGARITHMIC = auto()  # κ = O(log n) or O(log d)
    POLYNOMIAL = auto()   # κ = O(poly(n, d))
    CONSERVATIVE = auto() # sound but potentially very loose
    UNKNOWN = auto()       # no tightness guarantee


# ===================================================================
#  Catalog entry
# ===================================================================


@dataclass(frozen=True)
class CatalogEntry:
    """Entry in the capacity catalog for a single operation type.

    Attributes:
        op_name: Human-readable operation name (e.g. ``"StandardScaler"``).
        op_type: The :class:`OpType` enum member.
        capacity_fn: Function ``(rho, n, d) -> ChannelCapacityBound``.
        tightness_class: Qualitative tightness classification.
        tightness_factor_fn: Optional callable returning κ(n, d).
        assumptions: Description of assumptions for the bound to hold.
        sklearn_class_name: Fully-qualified sklearn class name for registry
            lookup (e.g. ``"sklearn.preprocessing.StandardScaler"``).
        description: Human-readable description of the bound.
    """

    op_name: str
    op_type: OpType
    capacity_fn: CapacityFunction
    tightness_class: TightnessClass = TightnessClass.UNKNOWN
    tightness_factor_fn: Optional[Callable[[int, int], float]] = None
    assumptions: str = ""
    sklearn_class_name: str = ""
    description: str = ""

    def compute(self, rho: float, n: int, d: int) -> ChannelCapacityBound:
        """Evaluate the capacity bound for the given parameters."""
        return self.capacity_fn(rho, n, d)

    def tightness_factor(self, n: int, d: int) -> float:
        """Return the tightness factor κ for the given problem size."""
        if self.tightness_factor_fn is not None:
            return self.tightness_factor_fn(n, d)
        return 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_name": self.op_name,
            "op_type": self.op_type.value,
            "tightness_class": self.tightness_class.name,
            "assumptions": self.assumptions,
            "sklearn_class_name": self.sklearn_class_name,
            "description": self.description,
        }

    def __repr__(self) -> str:
        return (
            f"CatalogEntry({self.op_name!r}, {self.op_type.name}, "
            f"tightness={self.tightness_class.name})"
        )


# ===================================================================
#  Capacity functions for specific operations
# ===================================================================


def _cap_zero(_rho: float, _n: int, _d: int) -> ChannelCapacityBound:
    """Zero capacity – operation does not leak any information through fit."""
    return ChannelCapacityBound(
        bits=0.0,
        tightness_factor=1.0,
        is_tight=True,
        confidence=1.0,
        channel_kind=ChannelKind.DETERMINISTIC,
        description="C_fit = 0: no leakage through fit",
    )


def _cap_identity(_rho: float, _n: int, d: int) -> ChannelCapacityBound:
    """Identity channel: output equals input, capacity = ∞ (capped at b_max)."""
    return ChannelCapacityBound(
        bits=d * _DEFAULT_B_MAX,
        tightness_factor=1.0,
        is_tight=True,
        confidence=1.0,
        channel_kind=ChannelKind.DETERMINISTIC,
        description=f"identity: no information loss, C = d·B_max = {d * _DEFAULT_B_MAX}",
    )


# -- Linear aggregates (mean, std, var, sum, count) -------------------


def _cap_mean(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """C_mean ≤ 0.5 · log₂(1 + n_te/(n − n_te)) per feature, κ = O(1)."""
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    n_te = rho * n
    n_tr = n - n_te
    if n_tr <= 0:
        return ChannelCapacityBound.infinite()
    snr = n_te / n_tr
    cap_per_feature = 0.5 * math.log2(1.0 + snr)
    total = d * cap_per_feature
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=1.0,
        is_tight=True,
        confidence=1.0,
        channel_kind=ChannelKind.GAUSSIAN,
        description=f"mean: C = d·0.5·log₂(1+ρ/(1−ρ)), ρ={rho:.3f}, n={n}, d={d}",
    )


def _cap_std(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """Standard deviation: same channel as mean for Gaussian data.

    std = sqrt(var), a monotone transformation.  By the data-processing
    inequality, I(D_te; std(X)) ≤ I(D_te; var(X)) = I(D_te; mean(X²−x̄²)).
    For Gaussian data the bound is tight at κ = 1.
    """
    return ChannelCapacityBound(
        bits=_cap_mean(rho, n, d).bits,
        tightness_factor=1.0,
        is_tight=True,
        confidence=1.0,
        channel_kind=ChannelKind.GAUSSIAN,
        description=f"std: same as mean (monotone of var), ρ={rho:.3f}, n={n}, d={d}",
    )


def _cap_var(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """Variance: same channel as mean for Gaussian data, κ = O(1)."""
    return ChannelCapacityBound(
        bits=_cap_mean(rho, n, d).bits,
        tightness_factor=1.0,
        is_tight=True,
        confidence=1.0,
        channel_kind=ChannelKind.GAUSSIAN,
        description=f"var: C = d·0.5·log₂(1+ρ/(1−ρ)), ρ={rho:.3f}, n={n}, d={d}",
    )


def _cap_sum(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """Sum: sum(X) = n · mean(X), deterministic scaling → same capacity."""
    return ChannelCapacityBound(
        bits=_cap_mean(rho, n, d).bits,
        tightness_factor=1.0,
        is_tight=True,
        confidence=1.0,
        channel_kind=ChannelKind.GAUSSIAN,
        description=f"sum: = n·mean, same capacity, ρ={rho:.3f}, n={n}, d={d}",
    )


def _cap_count(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """Count: scalar statistic, capacity bounded as for mean with d=1."""
    return ChannelCapacityBound(
        bits=_cap_mean(rho, n, 1).bits,
        tightness_factor=1.0,
        is_tight=True,
        confidence=1.0,
        channel_kind=ChannelKind.GAUSSIAN,
        description=f"count: scalar, C = 0.5·log₂(1+ρ/(1−ρ)), ρ={rho:.3f}, n={n}",
    )


# -- Rank statistics (median, quantiles) ------------------------------


def _kappa_log_n(n: int, _d: int) -> float:
    """Tightness factor κ = O(log n)."""
    return max(1.0, math.log2(max(2, n)))


def _cap_median(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """Median: rank statistic, κ = O(log n).

    C_median ≤ d · O(log n) · 0.5 · log₂(1 + ρ/(1−ρ)).
    """
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    n_te = rho * n
    n_tr = n - n_te
    if n_tr <= 0:
        return ChannelCapacityBound.infinite()
    snr = n_te / n_tr
    cap_per_feature = 0.5 * math.log2(1.0 + snr)
    kappa = max(1.0, math.log2(max(2, n)))
    total = d * kappa * cap_per_feature
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=kappa,
        is_tight=False,
        confidence=0.95,
        channel_kind=ChannelKind.GAUSSIAN,
        description=f"median: C ≤ d·κ·C_gauss, κ={kappa:.2f}, ρ={rho:.3f}, n={n}, d={d}",
    )


def _cap_quantile(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """General quantile: same bound as median, κ = O(log n)."""
    return _cap_median(rho, n, d)


# -- Covariance-based operations (PCA, SVD, corr, cov) ---------------


def _kappa_log_d(_n: int, d: int) -> float:
    """Tightness factor κ = O(log d)."""
    return max(1.0, math.log2(max(2, d)))


def _cap_pca(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """PCA / SVD: C_pca ≤ d² · C_cov(n_te, n), κ = O(log d).

    PCA fits a d×d covariance matrix on the full dataset.  Each of the
    ~d² entries of the covariance matrix is a linear aggregate with
    Gaussian channel capacity C_cov = 0.5 · log₂(1 + ρ/(1−ρ)).
    """
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    n_te = rho * n
    n_tr = n - n_te
    if n_tr <= 0:
        return ChannelCapacityBound.infinite()
    snr = n_te / n_tr
    c_cov = 0.5 * math.log2(1.0 + snr)
    total = d * d * c_cov
    kappa = max(1.0, math.log2(max(2, d)))
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=kappa,
        is_tight=False,
        confidence=0.95,
        channel_kind=ChannelKind.GAUSSIAN,
        description=f"PCA: C ≤ d²·C_cov, κ=O(log d)={kappa:.2f}, ρ={rho:.3f}, n={n}, d={d}",
    )


def _cap_corr(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """Correlation matrix: same as PCA (covariance after normalisation)."""
    return _cap_pca(rho, n, d)


def _cap_cov(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """Covariance matrix: d(d+1)/2 free entries, each with Gaussian capacity."""
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    n_te = rho * n
    n_tr = n - n_te
    if n_tr <= 0:
        return ChannelCapacityBound.infinite()
    snr = n_te / n_tr
    c_per_entry = 0.5 * math.log2(1.0 + snr)
    n_entries = d * (d + 1) // 2
    total = n_entries * c_per_entry
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=1.0,
        is_tight=True,
        confidence=1.0,
        channel_kind=ChannelKind.GAUSSIAN,
        description=f"cov: C = {n_entries}·C_gauss, ρ={rho:.3f}, n={n}, d={d}",
    )


# -- Group aggregates -------------------------------------------------


def _cap_groupby_transform_mean(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """GroupBy.transform('mean'): ≤ H(group_key) bits per group, κ = O(1).

    Without knowing the actual number of groups, we conservatively assume
    the worst case: each row is its own group, giving H(key) = log₂(n).
    When n_groups is known, use the specialized function in channels.py.
    """
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    h_key_worst = math.log2(max(1, n))
    total = d * h_key_worst
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=1.0,
        is_tight=True,
        confidence=0.9,
        channel_kind=ChannelKind.DISCRETE,
        description=(
            f"groupby_mean: C ≤ d·H(key) ≤ d·log₂(n), "
            f"H_worst={h_key_worst:.2f}, d={d}, ρ={rho:.3f}"
        ),
    )


def _cap_target_encoding(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """Target encoding: ≤ H(Y|group) bits per level, κ = O(1).

    Conservative: use H(Y) ≤ log₂(n) as upper bound on conditional
    entropy when the true target distribution is unknown.
    """
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    h_y_worst = math.log2(max(1, n))
    total = d * h_y_worst
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=1.0,
        is_tight=True,
        confidence=0.9,
        channel_kind=ChannelKind.DISCRETE,
        description=f"target_encoding: C ≤ d·H(Y|group), H_worst={h_y_worst:.2f}",
    )


# -- sklearn Scalers --------------------------------------------------


def _cap_standard_scaler(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """StandardScaler: fit learns mean and std per feature.

    Each feature has two statistics (mean, std), each with Gaussian
    channel capacity.  Total: C ≤ 2d · 0.5 · log₂(1 + ρ/(1−ρ)).
    Tight for Gaussian data, κ = O(1).
    """
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    n_te = rho * n
    n_tr = n - n_te
    if n_tr <= 0:
        return ChannelCapacityBound.infinite()
    snr = n_te / n_tr
    cap_per_stat = 0.5 * math.log2(1.0 + snr)
    total = 2 * d * cap_per_stat
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=1.0,
        is_tight=True,
        confidence=1.0,
        channel_kind=ChannelKind.GAUSSIAN,
        description=f"StandardScaler: C = 2d·C_gauss (mean+std), ρ={rho:.3f}, n={n}, d={d}",
    )


def _cap_minmax_scaler(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """MinMaxScaler: fit learns min and max per feature.

    Min and max are order statistics (rank 1 and rank n).  Their capacity
    is bounded similarly to the median with κ = O(log n):

        C ≤ 2d · O(log n) · 0.5 · log₂(1 + ρ/(1−ρ))
    """
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    n_te = rho * n
    n_tr = n - n_te
    if n_tr <= 0:
        return ChannelCapacityBound.infinite()
    snr = n_te / n_tr
    cap_per_stat = 0.5 * math.log2(1.0 + snr)
    kappa = max(1.0, math.log2(max(2, n)))
    total = 2 * d * kappa * cap_per_stat
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=kappa,
        is_tight=False,
        confidence=0.95,
        channel_kind=ChannelKind.GAUSSIAN,
        description=(
            f"MinMaxScaler: C ≤ 2d·κ·C_gauss (min+max), "
            f"κ={kappa:.2f}, ρ={rho:.3f}, n={n}, d={d}"
        ),
    )


def _cap_maxabs_scaler(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """MaxAbsScaler: fit learns max(|x|) per feature.

    Single order statistic per feature, κ = O(log n).
    """
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    n_te = rho * n
    n_tr = n - n_te
    if n_tr <= 0:
        return ChannelCapacityBound.infinite()
    snr = n_te / n_tr
    cap_per_stat = 0.5 * math.log2(1.0 + snr)
    kappa = max(1.0, math.log2(max(2, n)))
    total = d * kappa * cap_per_stat
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=kappa,
        is_tight=False,
        confidence=0.95,
        channel_kind=ChannelKind.GAUSSIAN,
        description=(
            f"MaxAbsScaler: C ≤ d·κ·C_gauss, "
            f"κ={kappa:.2f}, ρ={rho:.3f}, n={n}, d={d}"
        ),
    )


def _cap_robust_scaler(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """RobustScaler: fit learns median and IQR per feature.

    Both are rank statistics, so κ = O(log n):
        C ≤ 2d · O(log n) · 0.5 · log₂(1 + ρ/(1−ρ))
    """
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    n_te = rho * n
    n_tr = n - n_te
    if n_tr <= 0:
        return ChannelCapacityBound.infinite()
    snr = n_te / n_tr
    cap_per_stat = 0.5 * math.log2(1.0 + snr)
    kappa = max(1.0, math.log2(max(2, n)))
    total = 2 * d * kappa * cap_per_stat
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=kappa,
        is_tight=False,
        confidence=0.95,
        channel_kind=ChannelKind.GAUSSIAN,
        description=(
            f"RobustScaler: C ≤ 2d·κ·C_gauss (median+IQR), "
            f"κ={kappa:.2f}, ρ={rho:.3f}, n={n}, d={d}"
        ),
    )


# -- sklearn Encoders -------------------------------------------------


def _cap_onehot_encoder(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """OneHotEncoder: fit learns the set of unique categories per feature.

    The categories are a set-valued statistic.  The capacity is bounded by
    the log of the number of possible category sets:

        C ≤ d · log₂(n + 1)

    (each feature can have at most n distinct categories).
    Tight with κ = O(1) when categories are uniformly distributed.
    """
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    cap_per_feature = math.log2(max(1, n) + 1)
    n_te = rho * n
    n_tr = n - n_te
    if n_tr <= 0:
        return ChannelCapacityBound.infinite()
    snr = n_te / n_tr
    gauss_cap = 0.5 * math.log2(1.0 + snr)
    cap_per_feature = min(cap_per_feature, gauss_cap * math.log2(max(2, n)))
    total = d * cap_per_feature
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=1.0,
        is_tight=True,
        confidence=0.95,
        channel_kind=ChannelKind.DISCRETE,
        description=f"OneHotEncoder: C ≤ d·log₂(n+1), ρ={rho:.3f}, n={n}, d={d}",
    )


def _cap_ordinal_encoder(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """OrdinalEncoder: same information as OneHotEncoder (bijection)."""
    bound = _cap_onehot_encoder(rho, n, d)
    return ChannelCapacityBound(
        bits=bound.bits,
        tightness_factor=bound.tightness_factor,
        is_tight=bound.is_tight,
        confidence=bound.confidence,
        channel_kind=ChannelKind.DISCRETE,
        description=f"OrdinalEncoder: same as OneHot (bijection), ρ={rho:.3f}",
    )


def _cap_label_encoder(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """LabelEncoder: single-column version of OrdinalEncoder."""
    return _cap_ordinal_encoder(rho, n, max(1, d))


# -- Imputers ---------------------------------------------------------


def _cap_simple_imputer(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """SimpleImputer: fit learns a fill value (mean, median, mode, constant).

    For strategy='mean': same as _cap_mean.
    For strategy='median': same as _cap_median.
    For strategy='most_frequent': bounded by log₂(n) per feature.
    For strategy='constant': C_fit = 0.

    Conservative bound: use median capacity (κ = O(log n)).
    """
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    n_te = rho * n
    n_tr = n - n_te
    if n_tr <= 0:
        return ChannelCapacityBound.infinite()
    snr = n_te / n_tr
    cap_per_feature = 0.5 * math.log2(1.0 + snr)
    kappa = max(1.0, math.log2(max(2, n)))
    total = d * kappa * cap_per_feature
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=kappa,
        is_tight=False,
        confidence=0.9,
        channel_kind=ChannelKind.GAUSSIAN,
        description=(
            f"SimpleImputer: C ≤ d·κ·C_gauss (conservative), "
            f"κ={kappa:.2f}, ρ={rho:.3f}, n={n}, d={d}"
        ),
    )


def _cap_knn_imputer(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """KNNImputer: conservative ∞ bound (capped at B_max).

    KNN imputation uses the full dataset as a lookup table, creating a
    potentially high-capacity channel.  Without additional assumptions
    on the data distribution and the number of neighbors, we use the
    conservative ∞ bound.
    """
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    total = d * _DEFAULT_B_MAX
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=float("inf"),
        is_tight=False,
        confidence=1.0,
        channel_kind=ChannelKind.UNKNOWN,
        description=f"KNNImputer: conservative ∞ bound, C = d·B_max = {total:.1f}",
    )


# -- Other sklearn operations -----------------------------------------


def _cap_polynomial_features(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """PolynomialFeatures: C_fit = 0, no leakage through fit.

    PolynomialFeatures.fit() only learns the number of input features
    and the polynomial degree – both are deterministic functions of the
    pipeline configuration, not of the data.  The transform() is a
    deterministic function of the input data.
    """
    return ChannelCapacityBound(
        bits=0.0,
        tightness_factor=1.0,
        is_tight=True,
        confidence=1.0,
        channel_kind=ChannelKind.DETERMINISTIC,
        description="PolynomialFeatures: C_fit = 0, no data-dependent fit",
    )


def _cap_normalizer(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """Normalizer: C_fit = 0 (stateless transform).

    Normalizer does not learn any parameters from the data during fit().
    Each row is independently normalised to unit norm.
    """
    return ChannelCapacityBound(
        bits=0.0,
        tightness_factor=1.0,
        is_tight=True,
        confidence=1.0,
        channel_kind=ChannelKind.DETERMINISTIC,
        description="Normalizer: C_fit = 0, stateless per-row transform",
    )


def _cap_binarizer(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """Binarizer: C_fit = 0 (threshold is a hyperparameter, not learned).

    Binarizer.fit() is a no-op; the threshold is set at construction time.
    """
    return ChannelCapacityBound(
        bits=0.0,
        tightness_factor=1.0,
        is_tight=True,
        confidence=1.0,
        channel_kind=ChannelKind.DETERMINISTIC,
        description="Binarizer: C_fit = 0, threshold is a hyperparameter",
    )


# -- Generic / fallback -----------------------------------------------


def _cap_gaussian_fallback(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """Sound-but-generic Gaussian channel bound for unknown operations.

    When the operation is not in the catalog, we use the Gaussian channel
    as a conservative upper bound.  This is sound because the Gaussian
    channel maximises entropy for a given variance (maximum entropy
    principle), so the true capacity is at most the Gaussian capacity.

    The tightness factor is large (κ = n) because we know nothing about
    the operation's structure.
    """
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    n_te = rho * n
    n_tr = n - n_te
    if n_tr <= 0:
        return ChannelCapacityBound.infinite()
    snr = n_te / n_tr
    cap_per_feature = 0.5 * math.log2(1.0 + snr)
    kappa = max(1.0, float(n))
    total = d * kappa * cap_per_feature
    total = min(total, d * _DEFAULT_B_MAX)
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=kappa,
        is_tight=False,
        confidence=0.5,
        channel_kind=ChannelKind.GAUSSIAN,
        description=(
            f"FALLBACK Gaussian: C ≤ d·n·C_gauss (very conservative), "
            f"κ={kappa:.0f}, ρ={rho:.3f}, n={n}, d={d}"
        ),
    )


# -- Pandas aggregation wrappers that delegate to the core functions --


def _cap_describe(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """describe(): computes count, mean, std, min, 25%, 50%, 75%, max.

    8 statistics per feature.  Mixed tightness: mean/std are κ=O(1),
    min/max/quantiles are κ=O(log n).  Use the worst case.
    """
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    n_te = rho * n
    n_tr = n - n_te
    if n_tr <= 0:
        return ChannelCapacityBound.infinite()
    snr = n_te / n_tr
    cap_per_stat = 0.5 * math.log2(1.0 + snr)
    kappa = max(1.0, math.log2(max(2, n)))
    n_stats = 8
    total = n_stats * d * kappa * cap_per_stat
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=kappa,
        is_tight=False,
        confidence=0.95,
        channel_kind=ChannelKind.GAUSSIAN,
        description=f"describe: 8 stats/feature, κ={kappa:.2f}, ρ={rho:.3f}, n={n}, d={d}",
    )


def _cap_value_counts(rho: float, n: int, d: int) -> ChannelCapacityBound:
    """value_counts: histogram of each feature.  Worst case: n bins per feature."""
    if rho <= 0.0 or n <= 0 or d <= 0:
        return ChannelCapacityBound.zero()
    cap_per_feature = math.log2(max(1, n) + 1)
    total = d * cap_per_feature
    return ChannelCapacityBound(
        bits=total,
        tightness_factor=1.0,
        is_tight=True,
        confidence=0.9,
        channel_kind=ChannelKind.DISCRETE,
        description=f"value_counts: C ≤ d·log₂(n+1), ρ={rho:.3f}, n={n}, d={d}",
    )


# ===================================================================
#  Capacity catalog
# ===================================================================


class CapacityCatalog:
    """Registry mapping ``OpType`` to capacity transfer functions.

    Usage::

        catalog = CapacityCatalog.default()
        entry = catalog.lookup(OpType.STANDARD_SCALER)
        bound = entry.compute(rho=0.2, n=1000, d=10)
    """

    def __init__(self) -> None:
        self._entries: Dict[OpType, CatalogEntry] = {}
        self._sklearn_index: Dict[str, CatalogEntry] = {}

    def register(self, entry: CatalogEntry) -> None:
        """Register a catalog entry for an operation type."""
        self._entries[entry.op_type] = entry
        if entry.sklearn_class_name:
            self._sklearn_index[entry.sklearn_class_name] = entry

    def lookup(self, op_type: OpType) -> CatalogEntry:
        """Look up the catalog entry for *op_type*.

        Returns the fallback Gaussian entry if the operation is not
        registered.
        """
        if op_type in self._entries:
            return self._entries[op_type]
        return CatalogEntry(
            op_name=f"unknown({op_type.value})",
            op_type=op_type,
            capacity_fn=_cap_gaussian_fallback,
            tightness_class=TightnessClass.CONSERVATIVE,
            assumptions="No operation-specific bound; using Gaussian fallback.",
            description="Fallback Gaussian channel bound",
        )

    def lookup_by_sklearn_class(self, class_name: str) -> Optional[CatalogEntry]:
        """Look up by fully-qualified sklearn class name."""
        return self._sklearn_index.get(class_name)

    def compute_bound(
        self, op_type: OpType, rho: float, n: int, d: int
    ) -> ChannelCapacityBound:
        """Convenience: look up and evaluate in one call."""
        entry = self.lookup(op_type)
        return entry.compute(rho, n, d)

    @property
    def registered_ops(self) -> list[OpType]:
        """All registered operation types."""
        return list(self._entries.keys())

    @property
    def registered_sklearn_classes(self) -> list[str]:
        """All registered sklearn class names."""
        return list(self._sklearn_index.keys())

    def entries(self) -> list[CatalogEntry]:
        """All registered catalog entries."""
        return list(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, op_type: OpType) -> bool:
        return op_type in self._entries

    def __repr__(self) -> str:
        return f"CapacityCatalog({len(self._entries)} entries)"

    # -----------------------------------------------------------------
    #  Default catalog factory
    # -----------------------------------------------------------------

    @classmethod
    def default(cls) -> "CapacityCatalog":
        """Create the default catalog with all known operation bounds."""
        cat = cls()
        _register_all_defaults(cat)
        return cat


def _register_all_defaults(cat: CapacityCatalog) -> None:
    """Populate *cat* with all built-in capacity bounds."""

    # -- Linear aggregates (mean, std, var, sum, count) ----------------

    _kappa_one = lambda n, d: 1.0

    cat.register(CatalogEntry(
        op_name="mean",
        op_type=OpType.NP_MEAN,
        capacity_fn=_cap_mean,
        tightness_class=TightnessClass.EXACT,
        tightness_factor_fn=_kappa_one,
        assumptions="Gaussian data or sub-Gaussian tails.",
        description="C_mean ≤ 0.5·log₂(1 + ρ/(1−ρ)) per feature",
    ))

    cat.register(CatalogEntry(
        op_name="std",
        op_type=OpType.NP_STD,
        capacity_fn=_cap_std,
        tightness_class=TightnessClass.EXACT,
        tightness_factor_fn=_kappa_one,
        assumptions="Gaussian data.",
        description="Same Gaussian capacity as mean",
    ))

    cat.register(CatalogEntry(
        op_name="var",
        op_type=OpType.NP_VAR,
        capacity_fn=_cap_var,
        tightness_class=TightnessClass.EXACT,
        tightness_factor_fn=_kappa_one,
        assumptions="Gaussian data.",
        description="C_var = C_mean for Gaussian data",
    ))

    cat.register(CatalogEntry(
        op_name="sum",
        op_type=OpType.NP_SUM,
        capacity_fn=_cap_sum,
        tightness_class=TightnessClass.EXACT,
        tightness_factor_fn=_kappa_one,
        assumptions="Deterministic scaling of mean.",
        description="sum = n·mean, same capacity",
    ))

    cat.register(CatalogEntry(
        op_name="median",
        op_type=OpType.NP_MEDIAN,
        capacity_fn=_cap_median,
        tightness_class=TightnessClass.LOGARITHMIC,
        tightness_factor_fn=_kappa_log_n,
        assumptions="Continuous distribution with bounded density at median.",
        description="C_median ≤ O(log n)·C_gauss per feature",
    ))

    # -- Pandas aggregation ops ----------------------------------------

    cat.register(CatalogEntry(
        op_name="agg",
        op_type=OpType.AGG,
        capacity_fn=_cap_mean,
        tightness_class=TightnessClass.NEAR_TIGHT,
        tightness_factor_fn=_kappa_one,
        assumptions="Linear aggregation (mean-like). Conservative for non-linear aggs.",
        description="Aggregation: defaults to mean capacity",
    ))

    cat.register(CatalogEntry(
        op_name="aggregate",
        op_type=OpType.AGGREGATE,
        capacity_fn=_cap_mean,
        tightness_class=TightnessClass.NEAR_TIGHT,
        tightness_factor_fn=_kappa_one,
        assumptions="Linear aggregation.",
        description="Aggregation: defaults to mean capacity",
    ))

    cat.register(CatalogEntry(
        op_name="groupby",
        op_type=OpType.GROUPBY,
        capacity_fn=_cap_groupby_transform_mean,
        tightness_class=TightnessClass.EXACT,
        tightness_factor_fn=_kappa_one,
        assumptions="Uniform group distribution (worst case).",
        description="GroupBy: C ≤ d·H(key)",
    ))

    cat.register(CatalogEntry(
        op_name="transform",
        op_type=OpType.TRANSFORM,
        capacity_fn=_cap_groupby_transform_mean,
        tightness_class=TightnessClass.EXACT,
        tightness_factor_fn=_kappa_one,
        assumptions="GroupBy.transform('mean') pattern.",
        description="GroupBy.transform: C ≤ d·H(key)",
    ))

    cat.register(CatalogEntry(
        op_name="describe",
        op_type=OpType.DESCRIBE,
        capacity_fn=_cap_describe,
        tightness_class=TightnessClass.LOGARITHMIC,
        tightness_factor_fn=_kappa_log_n,
        assumptions="8 statistics per feature, mixed tightness.",
        description="describe: 8 stats/feature",
    ))

    cat.register(CatalogEntry(
        op_name="corr",
        op_type=OpType.CORR,
        capacity_fn=_cap_corr,
        tightness_class=TightnessClass.LOGARITHMIC,
        tightness_factor_fn=_kappa_log_d,
        assumptions="Gaussian data.",
        description="Correlation: same as PCA/covariance",
    ))

    cat.register(CatalogEntry(
        op_name="cov",
        op_type=OpType.COV,
        capacity_fn=_cap_cov,
        tightness_class=TightnessClass.EXACT,
        tightness_factor_fn=_kappa_one,
        assumptions="Gaussian data.",
        description="Covariance matrix: d(d+1)/2 entries",
    ))

    cat.register(CatalogEntry(
        op_name="value_counts",
        op_type=OpType.VALUE_COUNTS,
        capacity_fn=_cap_value_counts,
        tightness_class=TightnessClass.NEAR_TIGHT,
        tightness_factor_fn=_kappa_one,
        assumptions="Worst case: n distinct values per feature.",
        description="value_counts: C ≤ d·log₂(n+1)",
    ))

    # -- sklearn preprocessing: Scalers --------------------------------

    cat.register(CatalogEntry(
        op_name="StandardScaler",
        op_type=OpType.STANDARD_SCALER,
        capacity_fn=_cap_standard_scaler,
        tightness_class=TightnessClass.EXACT,
        tightness_factor_fn=_kappa_one,
        sklearn_class_name="sklearn.preprocessing.StandardScaler",
        assumptions="Gaussian data.",
        description="StandardScaler: 2 stats/feature (mean, std)",
    ))

    cat.register(CatalogEntry(
        op_name="MinMaxScaler",
        op_type=OpType.MINMAX_SCALER,
        capacity_fn=_cap_minmax_scaler,
        tightness_class=TightnessClass.LOGARITHMIC,
        tightness_factor_fn=_kappa_log_n,
        sklearn_class_name="sklearn.preprocessing.MinMaxScaler",
        assumptions="Bounded continuous distribution.",
        description="MinMaxScaler: 2 order stats/feature (min, max)",
    ))

    cat.register(CatalogEntry(
        op_name="RobustScaler",
        op_type=OpType.ROBUST_SCALER,
        capacity_fn=_cap_robust_scaler,
        tightness_class=TightnessClass.LOGARITHMIC,
        tightness_factor_fn=_kappa_log_n,
        sklearn_class_name="sklearn.preprocessing.RobustScaler",
        assumptions="Continuous distribution with bounded density at quartiles.",
        description="RobustScaler: 2 rank stats/feature (median, IQR)",
    ))

    cat.register(CatalogEntry(
        op_name="Normalizer",
        op_type=OpType.NORMALIZER,
        capacity_fn=_cap_normalizer,
        tightness_class=TightnessClass.EXACT,
        tightness_factor_fn=_kappa_one,
        sklearn_class_name="sklearn.preprocessing.Normalizer",
        assumptions="Stateless transform.",
        description="Normalizer: C_fit = 0 (no fit parameters)",
    ))

    # -- sklearn preprocessing: Encoders -------------------------------

    cat.register(CatalogEntry(
        op_name="OneHotEncoder",
        op_type=OpType.ONEHOT_ENCODER,
        capacity_fn=_cap_onehot_encoder,
        tightness_class=TightnessClass.NEAR_TIGHT,
        tightness_factor_fn=_kappa_one,
        sklearn_class_name="sklearn.preprocessing.OneHotEncoder",
        assumptions="Categories observed in data.",
        description="OneHotEncoder: C ≤ d·log₂(n+1)",
    ))

    cat.register(CatalogEntry(
        op_name="OrdinalEncoder",
        op_type=OpType.ORDINAL_ENCODER,
        capacity_fn=_cap_ordinal_encoder,
        tightness_class=TightnessClass.NEAR_TIGHT,
        tightness_factor_fn=_kappa_one,
        sklearn_class_name="sklearn.preprocessing.OrdinalEncoder",
        assumptions="Same as OneHotEncoder.",
        description="OrdinalEncoder: same as OneHot (bijection)",
    ))

    cat.register(CatalogEntry(
        op_name="LabelEncoder",
        op_type=OpType.LABEL_ENCODER,
        capacity_fn=_cap_label_encoder,
        tightness_class=TightnessClass.NEAR_TIGHT,
        tightness_factor_fn=_kappa_one,
        sklearn_class_name="sklearn.preprocessing.LabelEncoder",
        assumptions="Single-column encoding.",
        description="LabelEncoder: single-column OrdinalEncoder",
    ))

    cat.register(CatalogEntry(
        op_name="TargetEncoder",
        op_type=OpType.TARGET_ENCODER,
        capacity_fn=_cap_target_encoding,
        tightness_class=TightnessClass.EXACT,
        tightness_factor_fn=_kappa_one,
        sklearn_class_name="sklearn.preprocessing.TargetEncoder",
        assumptions="Bounded target entropy.",
        description="TargetEncoder: C ≤ d·H(Y|group)",
    ))

    # -- sklearn preprocessing: Imputers -------------------------------

    cat.register(CatalogEntry(
        op_name="SimpleImputer",
        op_type=OpType.IMPUTER,
        capacity_fn=_cap_simple_imputer,
        tightness_class=TightnessClass.LOGARITHMIC,
        tightness_factor_fn=_kappa_log_n,
        sklearn_class_name="sklearn.impute.SimpleImputer",
        assumptions="Conservative (median-like) for all strategies.",
        description="SimpleImputer: C ≤ d·κ·C_gauss",
    ))

    cat.register(CatalogEntry(
        op_name="KNNImputer",
        op_type=OpType.KNN_IMPUTER,
        capacity_fn=_cap_knn_imputer,
        tightness_class=TightnessClass.CONSERVATIVE,
        tightness_factor_fn=lambda n, d: float("inf"),
        sklearn_class_name="sklearn.impute.KNNImputer",
        assumptions="No structural assumption; full data lookup.",
        description="KNNImputer: conservative ∞ bound",
    ))

    # -- sklearn preprocessing: Other ----------------------------------

    cat.register(CatalogEntry(
        op_name="PolynomialFeatures",
        op_type=OpType.POLYNOMIAL_FEATURES,
        capacity_fn=_cap_polynomial_features,
        tightness_class=TightnessClass.EXACT,
        tightness_factor_fn=_kappa_one,
        sklearn_class_name="sklearn.preprocessing.PolynomialFeatures",
        assumptions="Fit is data-independent.",
        description="PolynomialFeatures: C_fit = 0",
    ))

    cat.register(CatalogEntry(
        op_name="Binarizer",
        op_type=OpType.BINARIZER,
        capacity_fn=_cap_binarizer,
        tightness_class=TightnessClass.EXACT,
        tightness_factor_fn=_kappa_one,
        sklearn_class_name="sklearn.preprocessing.Binarizer",
        assumptions="Threshold is a hyperparameter.",
        description="Binarizer: C_fit = 0",
    ))

    # -- PCA / SVD -----------------------------------------------------

    cat.register(CatalogEntry(
        op_name="PCA",
        op_type=OpType.FIT_TRANSFORM,
        capacity_fn=_cap_pca,
        tightness_class=TightnessClass.LOGARITHMIC,
        tightness_factor_fn=_kappa_log_d,
        sklearn_class_name="sklearn.decomposition.PCA",
        assumptions="Gaussian data, top-k eigenvector extraction.",
        description="PCA: C ≤ d²·C_cov, κ = O(log d)",
    ))

    # -- MaxAbsScaler (not a dedicated OpType, use TRANSFORM_SK) -------

    cat.register(CatalogEntry(
        op_name="MaxAbsScaler",
        op_type=OpType.TRANSFORM_SK,
        capacity_fn=_cap_maxabs_scaler,
        tightness_class=TightnessClass.LOGARITHMIC,
        tightness_factor_fn=_kappa_log_n,
        sklearn_class_name="sklearn.preprocessing.MaxAbsScaler",
        assumptions="Single order statistic per feature.",
        description="MaxAbsScaler: C ≤ d·κ·C_gauss",
    ))

    # -- Operations with zero fit capacity (pass-through) --------------

    for op in (OpType.COPY, OpType.DEEPCOPY, OpType.IDENTITY):
        cat.register(CatalogEntry(
            op_name=op.value,
            op_type=op,
            capacity_fn=_cap_zero,
            tightness_class=TightnessClass.EXACT,
            tightness_factor_fn=_kappa_one,
            assumptions="Deterministic copy, no data-dependent parameters.",
            description=f"{op.value}: C_fit = 0",
        ))

    # -- Pandas selection ops (no fit, pass-through) -------------------

    passthrough_ops = [
        OpType.GETITEM, OpType.LOC, OpType.ILOC, OpType.AT, OpType.IAT,
        OpType.HEAD, OpType.TAIL, OpType.DROP, OpType.RENAME,
        OpType.SET_INDEX, OpType.RESET_INDEX, OpType.SORT_VALUES,
        OpType.SORT_INDEX, OpType.REINDEX, OpType.ASTYPE,
        OpType.ISNA, OpType.NOTNA, OpType.CLIP,
    ]
    for op in passthrough_ops:
        cat.register(CatalogEntry(
            op_name=op.value,
            op_type=op,
            capacity_fn=_cap_zero,
            tightness_class=TightnessClass.EXACT,
            tightness_factor_fn=_kappa_one,
            assumptions="Selection / reshaping, no data-dependent fit.",
            description=f"{op.value}: C_fit = 0 (selection/reshape)",
        ))

    # -- Rolling / expanding / EWM (window aggregates) -----------------

    for op in (OpType.ROLLING, OpType.EXPANDING, OpType.EWM):
        cat.register(CatalogEntry(
            op_name=op.value,
            op_type=op,
            capacity_fn=_cap_mean,
            tightness_class=TightnessClass.NEAR_TIGHT,
            tightness_factor_fn=_kappa_one,
            assumptions="Window aggregate ≈ local mean.",
            description=f"{op.value}: bounded as mean aggregate",
        ))

    # -- Cumulative ops ------------------------------------------------

    for op in (OpType.CUMSUM, OpType.CUMPROD, OpType.CUMMAX, OpType.CUMMIN):
        cat.register(CatalogEntry(
            op_name=op.value,
            op_type=op,
            capacity_fn=_cap_mean,
            tightness_class=TightnessClass.NEAR_TIGHT,
            tightness_factor_fn=_kappa_one,
            assumptions="Cumulative op ≈ running aggregate.",
            description=f"{op.value}: bounded as running aggregate",
        ))

    # -- Fill / interpolate (data-dependent) ---------------------------

    cat.register(CatalogEntry(
        op_name="fillna",
        op_type=OpType.FILLNA,
        capacity_fn=_cap_simple_imputer,
        tightness_class=TightnessClass.LOGARITHMIC,
        tightness_factor_fn=_kappa_log_n,
        assumptions="Fill value may be data-dependent (e.g. mean fill).",
        description="fillna: bounded as SimpleImputer",
    ))

    cat.register(CatalogEntry(
        op_name="interpolate",
        op_type=OpType.INTERPOLATE,
        capacity_fn=_cap_knn_imputer,
        tightness_class=TightnessClass.CONSERVATIVE,
        tightness_factor_fn=lambda n, d: float("inf"),
        assumptions="Interpolation uses neighboring values; conservative bound.",
        description="interpolate: conservative ∞ bound",
    ))
