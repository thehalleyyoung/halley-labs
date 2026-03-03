"""
Subsampling protocol execution for DP-Forge.

Defines the :class:`SubsamplingProtocol` that encapsulates a complete
subsampled mechanism: a base mechanism combined with a subsampling step.
The protocol handles:

    - **Poisson inclusion**: Each record is included independently with
      probability q.
    - **Stratified sampling**: Variant that guarantees exactly ⌊qN⌋ or
      ⌈qN⌉ records per batch for reduced variance.
    - **Sensitivity adjustment**: Query sensitivity scales by 1/q when
      applied to subsamples (for sum queries) or remains unchanged
      (for per-record queries).
    - **Error estimation**: Expected error of the subsampled mechanism,
      accounting for both mechanism noise and subsampling variance.

Classes:
    - :class:`SubsamplingProtocol` — Complete subsampled mechanism protocol.
    - :class:`SubsamplingMode` — Enum of subsampling strategies.
    - :class:`ExecutionResult` — Result of protocol execution on a dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from dp_forge.exceptions import ConfigurationError
from dp_forge.types import PrivacyBudget

from dp_forge.subsampling.amplification import AmplificationResult


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SubsamplingMode(Enum):
    """Strategy for selecting the subsample.

    ``POISSON``
        Each record included independently with probability q.
        Subsample size is random (Binomial(N, q)).

    ``WITHOUT_REPLACEMENT``
        Draw exactly m = ⌊qN⌋ records uniformly without replacement.
        Subsample size is deterministic.

    ``STRATIFIED``
        Partition dataset into ⌈1/q⌉ strata and sample one record
        per stratum.  Reduces variance compared to Poisson while
        maintaining the same privacy guarantee.
    """

    POISSON = auto()
    WITHOUT_REPLACEMENT = auto()
    STRATIFIED = auto()

    def __repr__(self) -> str:
        return f"SubsamplingMode.{self.name}"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Result of executing a subsampling protocol on a dataset.

    Attributes:
        output: The mechanism's output (noisy answer).
        subsample_size: Number of records in the subsample.
        inclusion_mask: Boolean mask indicating which records were included.
        dataset_size: Total size of the input dataset.
        mode: Subsampling strategy used.
    """

    output: Any
    subsample_size: int
    inclusion_mask: npt.NDArray[np.bool_]
    dataset_size: int
    mode: SubsamplingMode

    @property
    def inclusion_rate(self) -> float:
        """Actual fraction of records included."""
        if self.dataset_size == 0:
            return 0.0
        return self.subsample_size / self.dataset_size

    def __repr__(self) -> str:
        return (
            f"ExecutionResult(output={self.output}, "
            f"subsample={self.subsample_size}/{self.dataset_size}, "
            f"mode={self.mode.name})"
        )


# ---------------------------------------------------------------------------
# SubsamplingProtocol
# ---------------------------------------------------------------------------


@dataclass
class SubsamplingProtocol:
    """Complete subsampled mechanism protocol.

    Encapsulates a base mechanism together with its subsampling parameters
    and amplified privacy guarantee.  Provides methods for executing the
    protocol on a dataset (generating the subsample and applying the
    mechanism) and for estimating expected error.

    Attributes:
        q_rate: Subsampling rate q ∈ (0, 1].
        base_mechanism: The n × k probability table of the base mechanism.
        base_eps: Base mechanism ε₀.
        base_delta: Base mechanism δ₀.
        amplified: The AmplificationResult with amplified privacy.
        y_grid: Output discretization grid of the base mechanism.
        mode: Subsampling strategy.
        seed: Random seed for reproducibility.
    """

    q_rate: float
    base_mechanism: npt.NDArray[np.float64]
    base_eps: float
    base_delta: float
    amplified: AmplificationResult
    y_grid: Optional[npt.NDArray[np.float64]] = None
    mode: SubsamplingMode = SubsamplingMode.POISSON
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.base_mechanism = np.asarray(self.base_mechanism, dtype=np.float64)
        if self.base_mechanism.ndim != 2:
            raise ValueError(
                f"base_mechanism must be 2-D, got shape {self.base_mechanism.shape}"
            )
        if not (0.0 < self.q_rate <= 1.0):
            raise ConfigurationError(
                f"q_rate must be in (0, 1], got {self.q_rate}",
                parameter="q_rate",
                value=self.q_rate,
                constraint="(0, 1]",
            )
        if self.y_grid is not None:
            self.y_grid = np.asarray(self.y_grid, dtype=np.float64)

    @property
    def n(self) -> int:
        """Number of database inputs in the base mechanism."""
        return self.base_mechanism.shape[0]

    @property
    def k(self) -> int:
        """Number of output bins in the base mechanism."""
        return self.base_mechanism.shape[1]

    @property
    def amplified_budget(self) -> PrivacyBudget:
        """Return the amplified privacy guarantee as a PrivacyBudget."""
        return self.amplified.budget

    def sample_inclusion_mask(
        self,
        dataset_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> npt.NDArray[np.bool_]:
        """Generate a random inclusion mask for the given dataset size.

        Args:
            dataset_size: Number of records in the dataset.
            rng: Random number generator. If None, creates one from seed.

        Returns:
            Boolean array of shape (dataset_size,) where True means
            the record is included in the subsample.
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)

        if self.mode == SubsamplingMode.POISSON:
            return _poisson_mask(dataset_size, self.q_rate, rng)
        elif self.mode == SubsamplingMode.WITHOUT_REPLACEMENT:
            return _wor_mask(dataset_size, self.q_rate, rng)
        elif self.mode == SubsamplingMode.STRATIFIED:
            return _stratified_mask(dataset_size, self.q_rate, rng)
        else:
            return _poisson_mask(dataset_size, self.q_rate, rng)

    def execute(
        self,
        dataset: npt.NDArray[np.float64],
        query_fn: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> ExecutionResult:
        """Execute the subsampled mechanism on a dataset.

        Steps:
            1. Generate inclusion mask according to the subsampling mode.
            2. Select the subsample from the dataset.
            3. Apply the base mechanism to the subsample.

        If ``query_fn`` is provided, it is applied to the subsample first
        to compute the query value, which is then used as the input index
        for the base mechanism.  If ``query_fn`` is None, the dataset is
        assumed to contain integer indices directly.

        Args:
            dataset: Input dataset as a numpy array.
            query_fn: Optional callable that maps a subsample to a query
                value (integer index into the base mechanism).
            rng: Random number generator.

        Returns:
            ExecutionResult with the mechanism output and metadata.
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)

        dataset = np.asarray(dataset, dtype=np.float64)
        n_records = len(dataset)

        mask = self.sample_inclusion_mask(n_records, rng)
        subsample = dataset[mask]
        subsample_size = len(subsample)

        # Compute query value on the subsample
        if query_fn is not None:
            query_val = int(query_fn(subsample))
        else:
            # Default: use the sum of the subsample as query value,
            # clamped to valid mechanism input range
            query_val = int(np.clip(np.sum(subsample), 0, self.n - 1))

        query_val = max(0, min(query_val, self.n - 1))

        # Sample from the base mechanism at the query value
        row_probs = self.base_mechanism[query_val]
        output_idx = rng.choice(self.k, p=row_probs)

        # Map to output value if grid is available
        if self.y_grid is not None and output_idx < len(self.y_grid):
            output = float(self.y_grid[output_idx])
        else:
            output = float(output_idx)

        return ExecutionResult(
            output=output,
            subsample_size=subsample_size,
            inclusion_mask=mask,
            dataset_size=n_records,
            mode=self.mode,
        )

    def expected_error(
        self,
        query_values: npt.NDArray[np.float64],
        loss_fn: Optional[Any] = None,
    ) -> float:
        """Estimate the expected error of the subsampled mechanism.

        The total error has two components:
            1. **Mechanism noise**: Error from the base mechanism's
               randomized output.
            2. **Subsampling variance**: Error from the random subsample
               composition, which scales as ~1/q for sum queries.

        For the base mechanism error, we compute the expected loss over
        the output distribution.  For subsampling variance, we use the
        fact that Poisson subsampling of a sum query scales variance by
        approximately 1/q.

        Args:
            query_values: Array of true query values f(x_i).
            loss_fn: Loss function (true, noisy) → loss. Defaults to L2.

        Returns:
            Estimated expected error.
        """
        query_values = np.asarray(query_values, dtype=np.float64)

        if loss_fn is None:
            loss_fn = lambda t, n: (t - n) ** 2

        # Mechanism noise: E[loss] over output distribution
        total_loss = 0.0
        n = min(len(query_values), self.n)

        for i in range(n):
            true_val = query_values[i]
            row_probs = self.base_mechanism[i]
            for j in range(self.k):
                if row_probs[j] < 1e-15:
                    continue
                if self.y_grid is not None and j < len(self.y_grid):
                    noisy_val = self.y_grid[j]
                else:
                    noisy_val = float(j)
                total_loss += row_probs[j] * loss_fn(true_val, noisy_val)

        mechanism_error = total_loss / n if n > 0 else 0.0

        # Subsampling variance contribution
        # For Poisson subsampling of sum queries, variance scales as σ²/q
        # where σ² is the variance of individual records
        subsampling_factor = 1.0 / self.q_rate if self.q_rate > 0 else float("inf")

        return mechanism_error * subsampling_factor

    def __repr__(self) -> str:
        return (
            f"SubsamplingProtocol(q={self.q_rate:.4f}, "
            f"base_ε={self.base_eps:.4f}, "
            f"amplified_ε={self.amplified.eps:.6f}, "
            f"mode={self.mode.name})"
        )


# ---------------------------------------------------------------------------
# Mask generation helpers
# ---------------------------------------------------------------------------


def _poisson_mask(
    n: int,
    q: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.bool_]:
    """Generate Poisson subsampling mask.

    Each record is included independently with probability q.

    Args:
        n: Dataset size.
        q: Inclusion probability.
        rng: Random number generator.

    Returns:
        Boolean mask of shape (n,).
    """
    return rng.random(n) < q


def _wor_mask(
    n: int,
    q: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.bool_]:
    """Generate without-replacement subsampling mask.

    Selects exactly m = ⌊q·n⌋ records uniformly without replacement.

    Args:
        n: Dataset size.
        q: Subsampling rate.
        rng: Random number generator.

    Returns:
        Boolean mask of shape (n,).
    """
    m = max(1, int(math.floor(q * n)))
    m = min(m, n)
    indices = rng.choice(n, size=m, replace=False)
    mask = np.zeros(n, dtype=np.bool_)
    mask[indices] = True
    return mask


def _stratified_mask(
    n: int,
    q: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.bool_]:
    """Generate stratified subsampling mask.

    Partitions the dataset into ⌈1/q⌉ strata of roughly equal size
    and selects one record per stratum.  This gives exactly ⌈n·q⌉
    records with reduced variance compared to Poisson.

    The privacy guarantee is the same as Poisson subsampling because
    the inclusion probability of each record is still q (or close to it).

    Args:
        n: Dataset size.
        q: Subsampling rate.
        rng: Random number generator.

    Returns:
        Boolean mask of shape (n,).
    """
    if q >= 1.0:
        return np.ones(n, dtype=np.bool_)

    stride = int(math.ceil(1.0 / q))
    if stride < 1:
        stride = 1

    mask = np.zeros(n, dtype=np.bool_)

    # For each stratum, pick one record uniformly at random
    for start in range(0, n, stride):
        end = min(start + stride, n)
        stratum_size = end - start
        chosen = rng.integers(0, stratum_size)
        mask[start + chosen] = True

    return mask
